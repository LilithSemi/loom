import 'dart:typed_data';
import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  test('quantizeTernaryAbsmean: per-tensor absmean, values in {-1,0,+1}', () {
    final w = Float64List.fromList([0.9, -0.1, 0.5, -0.6]); // mean|w| = 0.525
    final qm = quantizeTernaryAbsmean(w, 1, 4);
    expect(qm.rowScales[0], closeTo(0.525, 1e-9));
    expect(qm.values, orderedEquals([1, 0, 1, -1]));
    for (final v in qm.values) {
      expect(v >= -1 && v <= 1, isTrue);
    }
  });

  test('quantizeTernaryAbsmean: beta replicated across every row', () {
    final w = Float64List.fromList([
      1.0,
      -1.0,
      0.5,
      0.5,
      -2.0,
      0.0,
    ]); // 2 rows x 3
    final qm = quantizeTernaryAbsmean(w, 2, 3);
    final beta = (1 + 1 + 0.5 + 0.5 + 2 + 0) / 6;
    expect(qm.rowScales[0], closeTo(beta, 1e-9));
    expect(qm.rowScales[1], closeTo(beta, 1e-9));
  });

  test('quantizeTernaryAbsmean: all-zero matrix -> beta 1.0, all zeros', () {
    final qm = quantizeTernaryAbsmean(Float64List(4), 1, 4);
    expect(qm.rowScales[0], 1.0);
    expect(qm.values, orderedEquals([0, 0, 0, 0]));
  });

  test('quantizedLinearTernary: hand-computed W1.58A8 case', () {
    // Independent of the impl: beta = mean|w| = 1, ternary w = [+1x4, -1x4].
    // x maxabs = 2 -> gamma = 2/127, x_q = [127, 0, 0, 0].
    // row0 acc = 127, row1 acc = -127; y = acc * beta * gamma = +2, -2.
    final w = Float64List.fromList([
      1.0,
      1.0,
      1.0,
      1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0,
    ]); // 2x4
    final x = Float64List.fromList([2.0, 0.0, 0.0, 0.0]);
    final y = quantizedLinearTernary(w, 2, 4, x);
    expect(y.length, 2);
    expect(y[0], closeTo(2.0, 1e-9));
    expect(y[1], closeTo(-2.0, 1e-9));
  });
}
