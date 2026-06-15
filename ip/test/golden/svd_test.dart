import 'dart:math' as math;
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

double _maxAbsDiff(Float64List a, Float64List b) {
  var m = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = (a[i] - b[i]).abs();
    if (d > m) m = d;
  }
  return m;
}

void main() {
  test('jacobiEigenSym finds known eigenvalues of a 2x2', () {
    // [[2,1],[1,2]] has eigenvalues 1 and 3.
    final (eig, _) = jacobiEigenSym(Float64List.fromList([2, 1, 1, 2]), 2);
    final vals = [eig[0], eig[1]]..sort();
    expect(vals[0], closeTo(1.0, 1e-9));
    expect(vals[1], closeTo(3.0, 1e-9));
  });

  test('full-rank SVD reconstructs the weight (plain, no whitening)', () {
    final rnd = math.Random(1);
    const rows = 5, cols = 4;
    final w = Float64List(rows * cols);
    for (var i = 0; i < w.length; i++) {
      w[i] = rnd.nextDouble() * 2 - 1;
    }
    final f = svdCompress(w, rows, cols, rank: cols); // full rank
    expect(_maxAbsDiff(f.reconstruct(), w), lessThan(1e-9));
  });

  test('full-rank whitened SVD also reconstructs the weight', () {
    final rnd = math.Random(2);
    const rows = 5, cols = 4;
    final w = Float64List(rows * cols);
    for (var i = 0; i < w.length; i++) {
      w[i] = rnd.nextDouble() * 2 - 1;
    }
    // A non-trivial SPD Hessian from random samples.
    final stats = CalibStats(cols);
    for (var k = 0; k < 20; k++) {
      final x = Float64List(cols);
      for (var j = 0; j < cols; j++) {
        x[j] = rnd.nextDouble() * 2 - 1 + (j > 0 ? 0.4 * x[j - 1] : 0);
      }
      stats.count++;
      for (var p = 0; p < cols; p++) {
        for (var q = 0; q < cols; q++) {
          stats.hessian[p * cols + q] += x[p] * x[q];
        }
      }
    }
    final f = svdCompress(w, rows, cols, rank: cols, stats: stats);
    expect(_maxAbsDiff(f.reconstruct(), w), lessThan(1e-6));
  });

  test('truncation error is monotone non-increasing in rank', () {
    final rnd = math.Random(3);
    const rows = 8, cols = 6;
    final w = Float64List(rows * cols);
    for (var i = 0; i < w.length; i++) {
      w[i] = rnd.nextDouble() * 2 - 1;
    }
    double err(int r) =>
        _frob(svdCompress(w, rows, cols, rank: r).reconstruct(), w);
    var prev = double.infinity;
    for (var r = 1; r <= cols; r++) {
      final e = err(r);
      expect(e, lessThanOrEqualTo(prev + 1e-9));
      prev = e;
    }
  });

  test('whitening lowers ACTIVATION error vs plain SVD at a truncated rank', () {
    final rnd = math.Random(4);
    const rows = 10, cols = 8, r = 3;
    final w = Float64List(rows * cols);
    for (var i = 0; i < w.length; i++) {
      w[i] = rnd.nextDouble() * 2 - 1;
    }
    // Correlated activations so the input covariance is non-isotropic (only then
    // does whitening matter).
    final xs = <Float64List>[];
    final stats = CalibStats(cols);
    for (var k = 0; k < 64; k++) {
      final x = Float64List(cols);
      for (var j = 0; j < cols; j++) {
        x[j] = (rnd.nextDouble() * 2 - 1) + (j > 0 ? 0.8 * x[j - 1] : 0);
      }
      xs.add(x);
      stats.count++;
      for (var p = 0; p < cols; p++) {
        for (var q = 0; q < cols; q++) {
          stats.hessian[p * cols + q] += x[p] * x[q];
        }
      }
    }
    double actErr(Float64List recon) {
      var e = 0.0;
      for (final x in xs) {
        for (var i = 0; i < rows; i++) {
          var yw = 0.0, yr = 0.0;
          for (var j = 0; j < cols; j++) {
            yw += w[i * cols + j] * x[j];
            yr += recon[i * cols + j] * x[j];
          }
          final d = yw - yr;
          e += d * d;
        }
      }
      return e;
    }

    final plain = actErr(svdCompress(w, rows, cols, rank: r).reconstruct());
    final white = actErr(
      svdCompress(w, rows, cols, rank: r, stats: stats).reconstruct(),
    );
    expect(
      white,
      lessThan(plain),
      reason: 'whitened act-err $white should beat plain $plain',
    );
  });
}

double _frob(Float64List a, Float64List b) {
  var s = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = a[i] - b[i];
    s += d * d;
  }
  return s;
}
