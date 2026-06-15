import 'dart:math' as math;
import 'dart:typed_data';
import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  test('AWQ activation-weighted error <= RTN, and channelScale is positive', () {
    final rnd = math.Random(3);
    final rows = 4, cols = 8;
    final w = Float64List(rows * cols);
    for (var i = 0; i < rows * cols; i++) w[i] = rnd.nextDouble() * 2 - 1;
    // Skewed activations: one channel is much larger (salient).
    final s = CalibStats(cols);
    for (var k = 0; k < 32; k++) {
      final x = Float64List(cols);
      for (var j = 0; j < cols; j++)
        x[j] = (rnd.nextDouble() * 2 - 1) * (j == 2 ? 8.0 : 1.0);
      s.count++;
      for (var a = 0; a < cols; a++) {
        s.channelSumAbs[a] += x[a].abs();
        for (var b = 0; b < cols; b++) s.hessian[a * cols + b] += x[a] * x[b];
      }
    }
    final awq = quantizeAwq(w, rows, cols, bits: 3, groupSize: 8, stats: s);
    for (final v in awq.channelScale) expect(v, greaterThan(0.0));
    // Reconstruction of ORIGINAL w = dequant(q)/s, activation-weighted, <= RTN.
    final mean = s.channelMeanAbs;
    double err(GroupQuantizedMatrix q, Float64List sc) {
      final gpr = q.groupsPerRow;
      var e = 0.0;
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          final deq =
              q.values[r * cols + c] *
              q.scales[r * gpr + c ~/ q.groupSize] /
              sc[c];
          final d = w[r * cols + c] - deq;
          e += d * d * mean[c];
        }
      }
      return e;
    }

    final rtn = quantizeGroupwise(w, rows, cols, bits: 3, groupSize: 8);
    final ones = Float64List(cols)..fillRange(0, cols, 1.0);
    expect(
      err(awq.q, awq.channelScale),
      lessThanOrEqualTo(err(rtn, ones) + 1e-9),
    );
  });
}
