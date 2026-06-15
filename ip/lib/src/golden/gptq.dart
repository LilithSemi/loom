// Calibration-based weight quantization for the study: collects per-linear input
// statistics (GPTQ Hessian + AWQ per-channel activation scale) via the golden
// runner's linearImpl hook, and quantizes with GPTQ / AWQ. Emits the same
// GroupQuantizedMatrix the RTN path (quant.dart) and the hardware use.
import 'dart:math' as math;
import 'dart:typed_data';

import '../ir/model_graph.dart';
import '../weights/binding.dart';
import 'quant.dart';
import 'runner.dart';

/// Per-linear calibration stats keyed by the weight tensor identity.
class CalibStats {
  final Float64List hessian; // cols*cols, sum over positions of x.xT
  final Float64List channelSumAbs; // cols, sum over positions of |x|
  int count; // positions accumulated
  final int cols;
  CalibStats(this.cols)
    : hessian = Float64List(cols * cols),
      channelSumAbs = Float64List(cols),
      count = 0;
  Float64List get channelMeanAbs {
    final m = Float64List(cols);
    final n = count == 0 ? 1 : count;
    for (var i = 0; i < cols; i++) m[i] = channelSumAbs[i] / n;
    return m;
  }
}

/// One fp forward over [sample]. Records per-weight input stats. The recording
/// linearImpl computes the exact fp linear (so downstream activations are
/// correct) and accumulates H += x.xT and sum|x|.
Map<Float64List, CalibStats> collectCalibStats(
  ModelGraph graph,
  BoundModel model,
  List<int> sample,
) {
  final out = <Float64List, CalibStats>{};
  Float64List rec(Float64List w, int rows, int cols, Float64List x) {
    final st = out.putIfAbsent(w, () => CalibStats(cols));
    st.count++;
    for (var a = 0; a < cols; a++) {
      final xa = x[a];
      st.channelSumAbs[a] += xa.abs();
      final rowOff = a * cols;
      for (var b = 0; b < cols; b++) st.hessian[rowOff + b] += xa * x[b];
    }
    // exact fp linear so the calibration forward is faithful
    final y = Float64List(rows);
    for (var r = 0; r < rows; r++) {
      var s = 0.0;
      final o = r * cols;
      for (var c = 0; c < cols; c++) s += w[o + c] * x[c];
      y[r] = s;
    }
    return y;
  }

  // forwardAll (not forward) so lm_head is exercised at EVERY position, giving it
  // T calibration samples instead of just 1 (the last position).
  GoldenRunner(graph, model, linearImpl: rec).forwardAll(sample);
  return out;
}

int _gptqQmax(int bits) => (1 << (bits - 1)) - 1;

/// Lower Cholesky L (L.LT = h). h is n*n row-major, symmetric positive definite
/// (caller damps). Guards a non-positive pivot to keep going.
Float64List _cholesky(Float64List h, int n) {
  final l = Float64List(n * n);
  for (var i = 0; i < n; i++) {
    for (var j = 0; j <= i; j++) {
      var sum = h[i * n + j];
      for (var k = 0; k < j; k++) sum -= l[i * n + k] * l[j * n + k];
      if (i == j) {
        l[i * n + i] = sum > 1e-12 ? math.sqrt(sum) : 1e-6;
      } else {
        l[i * n + j] = sum / l[j * n + j];
      }
    }
  }
  return l;
}

/// Inverse of h from its lower Cholesky [l] (h = l.lT), solving h.X = I per column
/// by forward then back substitution. Returns hinv (n*n).
Float64List _choleskyInverse(Float64List l, int n) {
  final inv = Float64List(n * n);
  final y = Float64List(n);
  for (var col = 0; col < n; col++) {
    for (var i = 0; i < n; i++) {
      var s = (i == col) ? 1.0 : 0.0;
      for (var k = 0; k < i; k++) s -= l[i * n + k] * y[k];
      y[i] = s / l[i * n + i];
    }
    for (var i = n - 1; i >= 0; i--) {
      var s = y[i];
      for (var k = i + 1; k < n; k++) s -= l[k * n + i] * inv[k * n + col];
      inv[i * n + col] = s / l[i * n + i];
    }
  }
  return inv;
}

/// The expensive, scheme-INDEPENDENT part of GPTQ: the damped inverse-Hessian
/// Cholesky factor. It depends only on the calibration Hessian (not on bits or
/// groupSize or the weight values), so it is computed ONCE per weight and reused
/// across every {bits, groupSize} scheme. `lc` is the lower Cholesky of H^-1;
/// the upper factor U (with U^T U = H^-1) is uAt(i,j) = lc[j*n+i].
class GptqFactor {
  final Float64List lc; // n*n, lower Cholesky of H^-1
  final int n;
  const GptqFactor(this.lc, this.n);
  double uAt(int i, int j) => lc[j * n + i]; // upper entry (i<=j)
}

/// Factor the damped inverse Hessian: chol(H) -> l, invert -> hinv, chol(hinv) -> lc.
/// This is the O(n^3) cost. Cache the result per weight across schemes.
GptqFactor gptqFactor(CalibStats stats) {
  final n = stats.cols;
  final h = Float64List(n * n);
  var dsum = 0.0;
  for (var i = 0; i < n; i++) dsum += stats.hessian[i * n + i];
  final dmean = dsum / n;
  final lambda = 0.01 * (dmean > 0 ? dmean : 1.0);
  for (var i = 0; i < n * n; i++) h[i] = stats.hessian[i];
  for (var i = 0; i < n; i++) {
    if (h[i * n + i] == 0.0) h[i * n + i] = 1.0; // dead column
    h[i * n + i] += lambda;
  }
  final l = _cholesky(h, n);
  final hinv = _choleskyInverse(l, n);
  final lc = _cholesky(hinv, n);
  return GptqFactor(lc, n);
}

/// Full GPTQ (OBQ) quantization to symmetric int-N with per-group scales, using
/// the calibration Hessian to propagate each column's rounding error into the
/// not-yet-quantized columns. Emits a GroupQuantizedMatrix (HW-compatible).
/// Convenience wrapper: factors the Hessian then quantizes. Prefer
/// [quantizeGptqWithFactor] when quantizing the same weight at multiple schemes
/// so the O(n^3) factor is shared.
GroupQuantizedMatrix quantizeGptq(
  Float64List w,
  int rows,
  int cols, {
  required int bits,
  required int groupSize,
  required CalibStats stats,
}) {
  return quantizeGptqWithFactor(
    w,
    rows,
    cols,
    bits: bits,
    groupSize: groupSize,
    factor: gptqFactor(stats),
  );
}

/// GPTQ column sweep given a precomputed [GptqFactor] (the cheap, per-scheme part).
GroupQuantizedMatrix quantizeGptqWithFactor(
  Float64List w,
  int rows,
  int cols, {
  required int bits,
  required int groupSize,
  required GptqFactor factor,
}) {
  final qmax = _gptqQmax(bits);
  final gpr = (cols + groupSize - 1) ~/ groupSize;
  final wq = Float64List.fromList(w); // mutated by error propagation
  assert(factor.n == cols, 'factor n (${factor.n}) != cols ($cols)');
  double uAt(int i, int j) => factor.uAt(i, j);

  final values = Int8List(rows * cols);
  final scales = Float64List(rows * gpr);
  for (var i = 0; i < cols; i++) {
    final g = i ~/ groupSize;
    if (i % groupSize == 0) {
      // Per-group per-row scale from the CURRENT (error-updated) weights.
      final gEnd = (i + groupSize <= cols) ? i + groupSize : cols;
      for (var r = 0; r < rows; r++) {
        var maxAbs = 0.0;
        for (var c = i; c < gEnd; c++) {
          final a = wq[r * cols + c].abs();
          if (a > maxAbs) maxAbs = a;
        }
        scales[r * gpr + g] = maxAbs == 0.0 ? 1.0 : maxAbs / qmax;
      }
    }
    final d = uAt(i, i);
    for (var r = 0; r < rows; r++) {
      final sc = scales[r * gpr + g];
      var q = (wq[r * cols + i] / sc).round();
      if (q < -qmax)
        q = -qmax;
      else if (q > qmax)
        q = qmax;
      values[r * cols + i] = q;
      final err = (wq[r * cols + i] - q * sc) / d;
      for (var j = i + 1; j < cols; j++) {
        wq[r * cols + j] -= err * uAt(i, j);
      }
    }
  }
  return GroupQuantizedMatrix(
    values: values,
    scales: scales,
    rows: rows,
    cols: cols,
    bits: bits,
    groupSize: groupSize,
  );
}

/// AWQ result: the quantized (scaled) weights plus the per-input-channel scale
/// `channelScale`. At inference the backend feeds x[j]/channelScale[j].
class AwqResult {
  final GroupQuantizedMatrix q;
  final Float64List channelScale;
  const AwqResult({required this.q, required this.channelScale});
}

/// AWQ: search a per-input-channel scale s_j = channelMeanAbs_j^alpha (alpha in
/// [0,1], normalized to mean 1), scale weight columns by s, RTN-quantize the
/// scaled weights per-group, keep the alpha with the lowest activation-weighted
/// reconstruction error of the ORIGINAL weights (dequant(q)/s vs w).
AwqResult quantizeAwq(
  Float64List w,
  int rows,
  int cols, {
  required int bits,
  required int groupSize,
  required CalibStats stats,
}) {
  final chMean = stats.channelMeanAbs; // per-channel importance
  const nAlpha = 20;
  var bestErr = double.infinity;
  Float64List bestS = Float64List(cols)..fillRange(0, cols, 1.0);
  GroupQuantizedMatrix? bestQ;
  final ws = Float64List(rows * cols);
  for (var ai = 0; ai <= nAlpha; ai++) {
    final alpha = ai / nAlpha;
    final s = Float64List(cols);
    var ssum = 0.0;
    for (var j = 0; j < cols; j++) {
      final v = chMean[j] <= 0 ? 1e-6 : chMean[j];
      s[j] = math.pow(v, alpha).toDouble();
      ssum += s[j];
    }
    final smean = ssum / cols;
    for (var j = 0; j < cols; j++)
      s[j] = smean == 0 ? 1.0 : s[j] / smean; // mean-1 normalize
    for (var r = 0; r < rows; r++) {
      final o = r * cols;
      for (var j = 0; j < cols; j++) ws[o + j] = w[o + j] * s[j];
    }
    final qm = quantizeGroupwise(
      ws,
      rows,
      cols,
      bits: bits,
      groupSize: groupSize,
    );
    final gpr = qm.groupsPerRow;
    var err = 0.0;
    for (var r = 0; r < rows; r++) {
      for (var j = 0; j < cols; j++) {
        final deq =
            qm.values[r * cols + j] *
            qm.scales[r * gpr + j ~/ groupSize] /
            s[j];
        final d = w[r * cols + j] - deq;
        err += d * d * chMean[j];
      }
    }
    if (err < bestErr) {
      bestErr = err;
      bestS = Float64List.fromList(s);
      bestQ = qm;
    }
  }
  return AwqResult(q: bestQ!, channelScale: bestS);
}
