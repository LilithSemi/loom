import 'dart:typed_data';
import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  test('groupSize>=cols with bits=4 equals quantizeRowwiseInt4', () {
    final w = Float64List.fromList([0.1, -0.5, 0.9, -0.2, 0.3, 0.7]); // 2x3
    final g = quantizeGroupwise(w, 2, 3, bits: 4, groupSize: 3);
    final r = quantizeRowwiseInt4(w, 2, 3);
    expect(g.groupsPerRow, 1);
    expect(g.values, r.values);
    for (var i = 0; i < 2; i++)
      expect(g.scales[i], closeTo(r.rowScales[i], 1e-12));
  });

  test('smaller groups + more bits reduce reconstruction error', () {
    final rnd = [for (var i = 0; i < 128; i++) ((i * 37 % 101) - 50) / 50.0];
    final w = Float64List.fromList(rnd); // 1x128
    double err(int bits, int gs) {
      final q = quantizeGroupwise(w, 1, 128, bits: bits, groupSize: gs);
      var e = 0.0;
      for (var c = 0; c < 128; c++) {
        final g = c ~/ gs;
        e += (w[c] - q.values[c] * q.scales[g]).abs();
      }
      return e;
    }

    expect(err(4, 32), lessThan(err(4, 128))); // finer group -> less error
    expect(err(4, 128), lessThan(err(2, 128))); // more bits -> less error
  });

  test('quantizedLinearGroupwise matches manual int path', () {
    final w = Float64List.fromList([1.0, -2.0, 0.5, 0.25]); // 1x4
    final x = Float64List.fromList([0.5, -1.0, 2.0, 0.1]);
    final y = quantizedLinearGroupwise(w, 1, 4, x, bits: 4, groupSize: 2);
    final qm = quantizeGroupwise(w, 1, 4, bits: 4, groupSize: 2);
    final qv = quantizePerTensorInt8(x);
    final man = dequantGroupwise(matmulIntGroupwise(qm, qv), qm, qv);
    expect(y[0], closeTo(man[0], 1e-12));
  });
}
