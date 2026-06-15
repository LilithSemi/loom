// W4A8 golden: int4 row-wise weight quantization (symmetric [-7,7]) + int4
// pack/unpack for the DDR/flash weight image. int4 halves weight storage so
// SmolLM2-135M (~67MB) fits the OrangeCrab's 128MB DDR3.

import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  group('quantizeRowwiseInt4', () {
    test('values land in the symmetric int4 range [-7, 7]', () {
      final w = Float64List.fromList([
        1.0, -2.0, 3.0, -4.0, // row 0
        0.5, -0.5, 0.25, -0.1, // row 1
      ]);
      final q = quantizeRowwiseInt4(w, 2, 4);
      for (final v in q.values) {
        expect(v, inInclusiveRange(-7, 7));
      }
      // Row 0 maxAbs is 4.0 (index 3) -> maps to -7; 3.0 -> round(3*7/4)=5.
      expect(q.values[3], equals(-7));
      expect(q.values[2], equals(5));
    });

    test('per-row scale = maxAbs/7 and dequant recovers the row max', () {
      final w = Float64List.fromList([2.0, -8.0, 4.0, 1.0]);
      final q = quantizeRowwiseInt4(w, 1, 4);
      expect(q.rowScales[0], closeTo(8.0 / 7.0, 1e-12));
      expect(q.values[1], equals(-7)); // -8 is the max-abs element
    });

    test('W4A8 matmul approximates the fp linear', () {
      final w = Float64List.fromList([
        for (var i = 0; i < 16; i++) (i % 5) - 2.0,
      ]);
      final x = Float64List.fromList([1.0, -2.0, 0.5, 3.0]);
      final approx = quantizedLinearW4A8(w, 4, 4, x);
      // Reference fp result.
      final exact = [
        for (var r = 0; r < 4; r++)
          [
            for (var c = 0; c < 4; c++) w[r * 4 + c] * x[c],
          ].reduce((a, b) => a + b),
      ];
      for (var r = 0; r < 4; r++) {
        // int4 is coarse. Allow a generous relative tolerance.
        expect((approx[r] - exact[r]).abs(), lessThan(2.0), reason: 'row $r');
      }
    });
  });

  group('int4 pack/unpack (memory image, 2 nibbles per byte)', () {
    test('round-trips signed nibbles including negatives', () {
      final vals = Int8List.fromList([0, 1, -1, 7, -7, -8, 3, -4]);
      final packed = packInt4(vals);
      expect(packed.length, equals(vals.length ~/ 2));
      final back = unpackInt4(packed, vals.length);
      expect(back, equals(vals));
    });

    test('low nibble is the even index, high nibble the odd index', () {
      final vals = Int8List.fromList([0x3, -1]); // -1 -> 0xF
      final packed = packInt4(vals);
      expect(packed[0], equals(0xF3));
    });

    test('odd count packs the last nibble into a final byte', () {
      final vals = Int8List.fromList([5, -2, 1]);
      final packed = packInt4(vals);
      expect(packed.length, equals(2));
      expect(unpackInt4(packed, 3), equals(vals));
    });
  });
}
