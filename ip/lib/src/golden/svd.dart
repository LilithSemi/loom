// SVD-LLM low-rank weight compression for the study: factor a linear weight
// W [rows, cols] into A [rows, rank] . B [rank, cols] using a truncation-AWARE
// SVD (SVD-LLM, ICLR 2025). Plain SVD minimizes ||W - AB||_F, which is not the
// output error. SVD-LLM whitens W by the input covariance (the same H = sum(x xT)
// the GPTQ path collects) so truncation minimizes ||(W - AB) x|| on real
// activations. Emits two ordinary matmuls -> runs on the existing int4 datapath,
// and stacks with quantization (quantize A and B afterwards).
import 'dart:math' as math;
import 'dart:typed_data';

import 'gptq.dart'; // CalibStats

/// A [rows, rank] and B [rank, cols] such that A.B approximates the original
/// weight. Both are row-major fp64.
class SvdFactors {
  final Float64List a; // rows * rank
  final Float64List b; // rank * cols
  final int rows;
  final int cols;
  final int rank;
  const SvdFactors({
    required this.a,
    required this.b,
    required this.rows,
    required this.cols,
    required this.rank,
  });

  /// Parameter count of the factored form, vs rows*cols dense.
  int get paramCount => rank * (rows + cols);

  /// Reconstruct the dense rows*cols approximation A.B (for error checks).
  Float64List reconstruct() {
    final out = Float64List(rows * cols);
    for (var i = 0; i < rows; i++) {
      for (var k = 0; k < rank; k++) {
        final aik = a[i * rank + k];
        if (aik == 0) continue;
        final bOff = k * cols;
        final oOff = i * cols;
        for (var j = 0; j < cols; j++) {
          out[oOff + j] += aik * b[bOff + j];
        }
      }
    }
    return out;
  }
}

/// Jacobi eigenvalue decomposition of a symmetric n*n matrix [aIn] (row-major).
/// Returns (eigenvalues, eigenvectors) where eigenvectors[j] is the column
/// eigenvector for eigenvalues[j]. Unsorted.
(Float64List, List<Float64List>) jacobiEigenSym(Float64List aIn, int n) {
  final a = Float64List.fromList(aIn); // mutated toward diagonal
  final v = Float64List(n * n); // eigenvectors accumulate here (identity start)
  for (var i = 0; i < n; i++) {
    v[i * n + i] = 1.0;
  }
  for (var sweep = 0; sweep < 100; sweep++) {
    var off = 0.0;
    for (var p = 0; p < n; p++) {
      for (var q = p + 1; q < n; q++) {
        off += a[p * n + q] * a[p * n + q];
      }
    }
    if (off < 1e-28) break;
    for (var p = 0; p < n; p++) {
      for (var q = p + 1; q < n; q++) {
        final apq = a[p * n + q];
        if (apq.abs() < 1e-300) continue;
        final app = a[p * n + p];
        final aqq = a[q * n + q];
        final phi = 0.5 * math.atan2(2 * apq, aqq - app);
        final c = math.cos(phi);
        final s = math.sin(phi);
        // A = R^T A R over rows/cols p,q.
        for (var i = 0; i < n; i++) {
          final aip = a[i * n + p];
          final aiq = a[i * n + q];
          a[i * n + p] = c * aip - s * aiq;
          a[i * n + q] = s * aip + c * aiq;
        }
        for (var i = 0; i < n; i++) {
          final api = a[p * n + i];
          final aqi = a[q * n + i];
          a[p * n + i] = c * api - s * aqi;
          a[q * n + i] = s * api + c * aqi;
        }
        for (var i = 0; i < n; i++) {
          final vip = v[i * n + p];
          final viq = v[i * n + q];
          v[i * n + p] = c * vip - s * viq;
          v[i * n + q] = s * vip + c * viq;
        }
      }
    }
  }
  final eig = Float64List(n);
  for (var i = 0; i < n; i++) {
    eig[i] = a[i * n + i];
  }
  final vecs = List<Float64List>.generate(n, (j) {
    final col = Float64List(n);
    for (var i = 0; i < n; i++) {
      col[i] = v[i * n + j];
    }
    return col;
  });
  return (eig, vecs);
}

/// Lower Cholesky L (L.LT = h), h n*n row-major SPD (caller damps). Mirrors the
/// GPTQ path. A non-positive pivot is floored to keep going.
Float64List _cholesky(Float64List h, int n) {
  final l = Float64List(n * n);
  for (var i = 0; i < n; i++) {
    for (var j = 0; j <= i; j++) {
      var sum = h[i * n + j];
      for (var k = 0; k < j; k++) {
        sum -= l[i * n + k] * l[j * n + k];
      }
      if (i == j) {
        l[i * n + i] = sum > 1e-12 ? math.sqrt(sum) : 1e-6;
      } else {
        l[i * n + j] = sum / l[j * n + j];
      }
    }
  }
  return l;
}

/// Low-rank factor W [rows, cols] to A.B keeping [rank] singular directions.
/// When [stats] is given, uses SVD-LLM truncation-aware whitening (W' = W L,
/// L = chol of the damped input covariance), so truncation minimizes activation
/// error. Otherwise plain SVD on W.
SvdFactors svdCompress(
  Float64List w,
  int rows,
  int cols, {
  required int rank,
  CalibStats? stats,
}) {
  final int r = rank.clamp(1, math.min(rows, cols)).toInt();

  // Whitening L (lower) and its inverse-transpose action, or identity.
  Float64List? l;
  if (stats != null) {
    final n = cols;
    final h = Float64List(n * n);
    var dsum = 0.0;
    for (var i = 0; i < n; i++) {
      dsum += stats.hessian[i * n + i];
    }
    final lambda = 0.01 * (dsum / n > 0 ? dsum / n : 1.0);
    for (var i = 0; i < n * n; i++) {
      h[i] = stats.hessian[i];
    }
    for (var i = 0; i < n; i++) {
      h[i * n + i] += lambda;
    }
    l = _cholesky(h, n);
  }

  // W' = W L (rows x cols); with no whitening W' = W.
  final wp = Float64List(rows * cols);
  if (l == null) {
    wp.setAll(0, w);
  } else {
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        var s = 0.0;
        // (W L)[i,j] = sum_k W[i,k] L[k,j]; L lower so k >= j.
        for (var k = j; k < cols; k++) {
          s += w[i * cols + k] * l[k * cols + j];
        }
        wp[i * cols + j] = s;
      }
    }
  }

  // SVD of W' via the smaller Gram matrix. cols <= rows is the common case for
  // these weights (or equal). Use G = W'^T W' [cols x cols].
  final n = cols;
  final g = Float64List(n * n);
  for (var p = 0; p < n; p++) {
    for (var q = p; q < n; q++) {
      var s = 0.0;
      for (var i = 0; i < rows; i++) {
        s += wp[i * cols + p] * wp[i * cols + q];
      }
      g[p * n + q] = s;
      g[q * n + p] = s;
    }
  }
  final (eig, vecs) = jacobiEigenSym(g, n);
  // Top-r by eigenvalue (= sigma^2).
  final order = List<int>.generate(n, (i) => i)
    ..sort((x, y) => eig[y].compareTo(eig[x]));

  // Un-whitening acts on V: B row i = sigma_i^{1/2} * (v_i^T L^{-1}); solve
  // L^T z = v_i (L^T upper) so z = L^{-T} v_i, then v_i^T L^{-1} = z^T.
  final a = Float64List(rows * r);
  final b = Float64List(r * cols);
  for (var ri = 0; ri < r; ri++) {
    final idx = order[ri];
    final sigma = math.sqrt(eig[idx] > 0 ? eig[idx] : 0.0);
    final vi = vecs[idx]; // length cols
    // u_i = W' v_i / sigma (length rows).
    final sqrtSigma = math.sqrt(sigma > 0 ? sigma : 0.0);
    if (sigma > 1e-20) {
      for (var i = 0; i < rows; i++) {
        var s = 0.0;
        for (var k = 0; k < cols; k++) {
          s += wp[i * cols + k] * vi[k];
        }
        a[i * r + ri] = (s / sigma) * sqrtSigma; // u_i * sigma^{1/2}
      }
    }
    // z = L^{-T} v_i (or v_i if no whitening).
    Float64List z;
    if (l == null) {
      z = vi;
    } else {
      z = Float64List(cols);
      // Back-substitution for L^T z = v_i (L^T is upper triangular).
      for (var i = cols - 1; i >= 0; i--) {
        var s = vi[i];
        for (var k = i + 1; k < cols; k++) {
          s -= l[k * cols + i] * z[k]; // (L^T)[i,k] = L[k,i]
        }
        z[i] = s / l[i * cols + i];
      }
    }
    for (var j = 0; j < cols; j++) {
      b[ri * cols + j] = sqrtSigma * z[j]; // sigma^{1/2} * (v_i^T L^{-1})
    }
  }
  return SvdFactors(a: a, b: b, rows: rows, cols: cols, rank: r);
}
