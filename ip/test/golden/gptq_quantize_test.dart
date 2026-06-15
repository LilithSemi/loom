import 'dart:math' as math;
import 'dart:typed_data';
import 'package:loom/loom.dart';
import 'package:test/test.dart';

// Build a CalibStats with a given diagonal-ish Hessian for a cols-wide input.
CalibStats _statsWithHessian(Float64List hDiag) {
  final n = hDiag.length;
  final s = CalibStats(n);
  s.count = 1;
  for (var i = 0; i < n; i++) {
    s.hessian[i * n + i] = hDiag[i];
    s.channelSumAbs[i] = 1.0;
  }
  return s;
}

void main() {
  test(
    'identity Hessian degenerates toward RTN (same per-group scale, near-equal values)',
    () {
      final w = Float64List.fromList([1.0, -2.0, 0.5, 0.25, 0.9, -0.7]); // 1x6
      final stats = _statsWithHessian(Float64List.fromList([1, 1, 1, 1, 1, 1]));
      final g = quantizeGptq(w, 1, 6, bits: 4, groupSize: 6, stats: stats);
      final r = quantizeRowwiseInt4(w, 1, 6);
      // With a scaled-identity Hessian, error propagation coefficients off-diagonal are ~0,
      // so GPTQ reduces to per-column RTN: scale matches and values match.
      expect(g.scales[0], closeTo(r.rowScales[0], 1e-9));
      for (var c = 0; c < 6; c++) expect(g.values[c], r.values[c]);
    },
  );

  test('with a non-trivial Hessian, GPTQ Hessian-quadratic loss <= RTN', () {
    // 4x8 weights, a correlated Hessian from a small random calibration.
    final rnd = math.Random(7);
    final rows = 4, cols = 8;
    final w = Float64List(rows * cols);
    for (var i = 0; i < rows * cols; i++) w[i] = rnd.nextDouble() * 2 - 1;
    // Fabricate H from 32 random activation samples (full-rank-ish), plus channelSumAbs.
    final s = CalibStats(cols);
    for (var k = 0; k < 32; k++) {
      final x = Float64List(cols);
      for (var j = 0; j < cols; j++) x[j] = rnd.nextDouble() * 2 - 1;
      s.count++;
      for (var a = 0; a < cols; a++) {
        s.channelSumAbs[a] += x[a].abs();
        for (var b = 0; b < cols; b++) s.hessian[a * cols + b] += x[a] * x[b];
      }
    }
    // GPTQ minimizes the TRUE Hessian quadratic loss sum_r err_r^T H err_r (with
    // cross-channel terms), NOT a diagonal per-channel proxy. Test against that
    // objective: GPTQ's error propagation provably makes it <= RTN here.
    double hessianErr(GroupQuantizedMatrix q) {
      final gpr = q.groupsPerRow;
      var e = 0.0;
      for (var r = 0; r < rows; r++) {
        final err = Float64List(cols);
        for (var c = 0; c < cols; c++) {
          final deq =
              q.values[r * cols + c] * q.scales[r * gpr + c ~/ q.groupSize];
          err[c] = w[r * cols + c] - deq;
        }
        for (var a = 0; a < cols; a++) {
          final ea = err[a];
          if (ea == 0) continue;
          final ho = a * cols;
          for (var b = 0; b < cols; b++) e += ea * s.hessian[ho + b] * err[b];
        }
      }
      return e;
    }

    final gptq = quantizeGptq(w, rows, cols, bits: 3, groupSize: 8, stats: s);
    final rtn = quantizeGroupwise(w, rows, cols, bits: 3, groupSize: 8);
    expect(hessianErr(gptq), lessThanOrEqualTo(hessianErr(rtn) + 1e-6));
  });
}
