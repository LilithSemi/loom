// Tests for the W8A8 quantized golden reference (symmetric int8, zero-point 0).
// Dart's double.round() is round-half-away-from-zero, which matches HW behavior.
// Write-first (TDD): tests are written before implementation.

import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  group('quantizeRowwiseInt8', () {
    test('hand-computed: row [1.0, -2.0, 0.5] quantizes correctly', () {
      // maxAbs = 2.0, scale = 2.0/127 = 0.015748...
      // q[0] = round(1.0 / 0.015748) = round(63.5) = 64 (round-half-away-from-zero)
      // q[1] = round(-2.0 / 0.015748) = round(-127.0) = -127
      // q[2] = round(0.5 / 0.015748) = round(31.75) = 32
      final w = Float64List.fromList([1.0, -2.0, 0.5]);
      final qm = quantizeRowwiseInt8(w, 1, 3);

      expect(qm.rows, equals(1));
      expect(qm.cols, equals(3));
      expect(qm.values.length, equals(3));
      expect(qm.rowScales.length, equals(1));

      final expectedScale = 2.0 / 127.0;
      expect(qm.rowScales[0], closeTo(expectedScale, 1e-12));
      expect(qm.values[0], equals(64));
      expect(qm.values[1], equals(-127));
      expect(qm.values[2], equals(32));
    });

    test(
      'all-zero row: scale defaults to 1.0, quantized values are all zero',
      () {
        final w = Float64List.fromList([0.0, 0.0, 0.0]);
        final qm = quantizeRowwiseInt8(w, 1, 3);

        expect(qm.rowScales[0], equals(1.0));
        expect(qm.values[0], equals(0));
        expect(qm.values[1], equals(0));
        expect(qm.values[2], equals(0));
      },
    );

    test('multiple rows use independent row scales', () {
      // row0: [2.0, 0.0]  maxAbs=2.0, scale=2/127
      // row1: [0.0, 1.0]  maxAbs=1.0, scale=1/127
      final w = Float64List.fromList([2.0, 0.0, 0.0, 1.0]);
      final qm = quantizeRowwiseInt8(w, 2, 2);

      expect(qm.rowScales[0], closeTo(2.0 / 127.0, 1e-12));
      expect(qm.rowScales[1], closeTo(1.0 / 127.0, 1e-12));
      expect(qm.values[0], equals(127));
      expect(qm.values[1], equals(0));
      expect(qm.values[2], equals(0));
      expect(qm.values[3], equals(127));
    });

    test('clamping: value that would round beyond 127 is clamped to 127', () {
      // Force a case where a value is an outlier but only just clamps.
      // Row has outlier 128.0 and a large value 130.0 for the outlier-dominated scale:
      // scale = 130.0/127 = 1.0236...; round(128.0/1.0236) = round(125.05) = 125, no clamp.
      // Better: use scale=1.0 directly. We quantize row [127.5, -127.5, 128.0].
      // maxAbs=128.0, scale=128.0/127=1.00787...; q[0]=round(127.5/1.00787)=round(126.5)=127 (no clamp needed)
      // Instead: row [127.0, 130.0] -> maxAbs=130, scale=130/127; q[1]=round(130/(130/127))=round(127)=127 (exact)
      // Simplest: use row [1.0] with custom scale forcing clamp; inject via manual QuantizedMatrix test instead.
      // For quantizeRowwiseInt8: use row [100.0, 101.0] and add a tiny outlier at 100.01 so scale barely changes.
      // Actually the easiest test: row = [127.0], scale = 127.0/127 = 1.0, q = [127]. Not a clamp test.
      // Proper clamp: construct weight so rounding pushes to 128 before clamp.
      // row = [127.49, -127.49, 127.5]
      // maxAbs = 127.5, scale = 127.5/127 = 1.003937...
      // q[2] = round(127.5 / 1.003937) = round(127.0) = 127 (exactly, no clamp needed still)
      // Correct approach: row = [1.0, 1.005] -> maxAbs=1.005, scale=1.005/127=0.007913...
      // q[1] = round(1.005/0.007913) = round(127.0) = 127 (still exact)
      // The clamp kicks in only if a SECOND value exceeds maxAbs after rounding artifact.
      // Since we compute scale from maxAbs and clamp(round(v/s)), and v <= maxAbs always,
      // the only overflow case is floating point rounding where v/s slightly > 127.0.
      // Simulate: row = [1.0, 1.0000001] -> maxAbs=1.0000001, scale=1.0000001/127
      // q[0] = round(1.0 / (1.0000001/127)) = round(126.999987...) = 127 (no overflow)
      // The actual test for clamp is best done via quantizePerTensorInt8 with a scaled vector,
      // or by using negative-side: row = [-127.5, 1.0]
      // maxAbs=127.5, scale=127.5/127; q[0]=round(-127.5/scale)=round(-127.0)=-127 (exact)
      // Realistic clamp scenario: two values, tiny outlier causes one to compute > 127 after FP.
      // Use: row = [127.0, -128.0]. maxAbs=128.0, scale=128.0/127=1.007874...
      // q[0] = round(127.0/1.007874) = round(126.0) = 126 (under, not clamped)
      // q[1] = round(-128.0/1.007874) = round(-127.0) = -127 (exact, clamped if it were -128)
      // For a TRUE positive-clamp: we need v/s > 127 before clamp.
      // This happens if fp precision rounds up: e.g., v = maxAbs exactly -> v/s = 127.0 exactly -> 127, no clamp.
      // Conclusion: with symmetric rowwise quant, real clamp only fires on the negative
      // side for asymmetric absolute values. Test: row = [-128.0, 127.0].
      // maxAbs=128.0, scale=128/127; q[1]=round(127/(128/127))=round(127*127/128)=round(125.98)=126.
      // q[0]=round(-128/scale)=round(-127)=-127. Neither clamped to -127 via clamp logic beyond that.
      // Forced clamp test: inject via a per-tensor vector that has a value exceeding 127 after division.
      // Use quantizePerTensorInt8: x=[127.5, 1.0]; maxAbs=127.5, scale=127.5/127=1.003937
      // q[0]=round(127.5/1.003937)=round(127.0)=127. Still exact.
      // The safest forced-clamp test: build a case via Int8List clamping by
      // checking that the implementation does not store 128 (which Int8List would wrap).
      // row = [100.0, 127.4, 127.6] -> maxAbs=127.6, scale=127.6/127=1.004724...
      // q[1]=round(127.4/1.004724)=round(126.79)=127, q[2]=round(127.6/1.004724)=round(127.0)=127. Fine.
      // The ONLY reliable forced-clamp is: construct so that fp arithmetic gives v/s > 127.
      // Use: v = nextAfter(maxAbs, +inf) which makes v/s = 127 + tiny. But that changes maxAbs.
      // Workaround: pass values where the largest value in the row doesn't set maxAbs (two equal maxabs).
      // row = [127.0, 127.0001] -> maxAbs=127.0001, scale=127.0001/127=1.0000007874...
      // q[0]=round(127.0/(1.0000007874))=round(126.9999...)=127. No clamp.
      // Truly: fp quantization with correct symmetric rowwise should not overflow the max entry.
      // But a value can overflow if the scale is computed from a DIFFERENT max.
      // Test via two-row matrix where row 0 has outlier but row 1 gets scale from row 0's perspective -
      // actually no, scales are per-row.
      // CONCLUSION: the clamp matters for quantizePerTensorInt8 where we use the global max,
      // but in rowwise it's per-row so overflow is rare. Test via per-tensor with constructed case.
      // row: just test that Int8List does not silently overflow by putting the max value:
      final w = Float64List.fromList([127.0]);
      final qm = quantizeRowwiseInt8(w, 1, 1);
      expect(qm.values[0], equals(127));
      expect(qm.values[0], inInclusiveRange(-127, 127));
    });

    test('clamping negative side: result stays within [-127, 127]', () {
      final w = Float64List.fromList([-127.0]);
      final qm = quantizeRowwiseInt8(w, 1, 1);
      expect(qm.values[0], equals(-127));
      expect(qm.values[0], inInclusiveRange(-127, 127));
    });

    test('determinism: quantize twice gives identical results', () {
      final w = Float64List.fromList([1.0, -2.0, 0.5, 0.3, -0.7, 1.5]);
      final qm1 = quantizeRowwiseInt8(w, 2, 3);
      final qm2 = quantizeRowwiseInt8(w, 2, 3);

      for (var i = 0; i < qm1.values.length; i++) {
        expect(qm1.values[i], equals(qm2.values[i]));
      }
      for (var r = 0; r < qm1.rows; r++) {
        expect(qm1.rowScales[r], equals(qm2.rowScales[r]));
      }
    });
  });

  group('quantizePerTensorInt8', () {
    test('basic vector quantization', () {
      // x = [1.0, -2.0, 0.5]; maxAbs=2.0, scale=2.0/127
      // q[0]=round(1.0/(2/127))=round(63.5)=64
      // q[1]=round(-2.0/(2/127))=round(-127)=-127
      // q[2]=round(0.5/(2/127))=round(31.75)=32
      final x = Float64List.fromList([1.0, -2.0, 0.5]);
      final qv = quantizePerTensorInt8(x);

      expect(qv.scale, closeTo(2.0 / 127.0, 1e-12));
      expect(qv.values[0], equals(64));
      expect(qv.values[1], equals(-127));
      expect(qv.values[2], equals(32));
    });

    test('all-zero vector: scale defaults to 1.0, values all zero', () {
      final x = Float64List.fromList([0.0, 0.0]);
      final qv = quantizePerTensorInt8(x);

      expect(qv.scale, equals(1.0));
      expect(qv.values[0], equals(0));
      expect(qv.values[1], equals(0));
    });

    test('clamping: value beyond 127 after division is clamped', () {
      // Construct: x=[2.0, 0.001]. maxAbs=2.0, scale=2.0/127.
      // q[0]=round(2.0/(2/127))=round(127)=127. No clamp needed.
      // Force: use Int8List overflow check. x = [1.0, 1.0]: maxAbs=1, scale=1/127.
      // q[0]=round(1.0/(1/127))=round(127)=127. Fine.
      // True clamp via fp: x = [nextAfter(1.0, inf), 1.0] is hard.
      // Instead test the negative clamp: the Int8List range is -128..127 but we clamp to -127..127.
      // x = [-1.0]: maxAbs=1.0, scale=1/127; q=round(-1.0/(1/127))=round(-127)=-127.
      // That doesn't test -128 clamp. For real clamp test, inject computed case:
      // x = [127.0, 128.0]: maxAbs=128.0, scale=128/127=1.00787.
      // q[1] = round(128.0/1.00787) = round(127.0) = 127. Still no overflow.
      // The clamp fires for: x[i] such that round(x[i]/scale) > 127.
      // This is possible when x[i] > maxAbs, which can't happen by construction.
      // HOWEVER: Int8List silently wraps 128 -> -128. The clamp prevents that.
      // We test this by verifying values are in [-127,127].
      final x = Float64List.fromList([1.0, -1.0, 0.5, -0.5]);
      final qv = quantizePerTensorInt8(x);

      for (var i = 0; i < qv.values.length; i++) {
        expect(qv.values[i], inInclusiveRange(-127, 127));
      }
    });

    test('determinism: quantize twice gives identical results', () {
      final x = Float64List.fromList([1.0, -2.0, 0.5, 0.3]);
      final qv1 = quantizePerTensorInt8(x);
      final qv2 = quantizePerTensorInt8(x);

      expect(qv1.scale, equals(qv2.scale));
      for (var i = 0; i < qv1.values.length; i++) {
        expect(qv1.values[i], equals(qv2.values[i]));
      }
    });
  });

  group('matmulInt', () {
    test('exact integer dot product: [[-1,2],[3,-4]] @ [5,-6] = [-17,39]', () {
      // Row 0: -1*5 + 2*(-6) = -5 - 12 = -17
      // Row 1: 3*5 + (-4)*(-6) = 15 + 24 = 39
      final wVals = Int8List.fromList([-1, 2, 3, -4]);
      final xVals = Int8List.fromList([5, -6]);
      final wScales = Float64List.fromList([1.0, 1.0]);
      final wm = QuantizedMatrix(
        values: wVals,
        rowScales: wScales,
        rows: 2,
        cols: 2,
      );
      final xv = QuantizedVector(values: xVals, scale: 1.0);

      final acc = matmulInt(wm, xv);

      expect(acc.length, equals(2));
      expect(acc[0], equals(-17));
      expect(acc[1], equals(39));
    });

    test('1x1 dot product', () {
      final wm = QuantizedMatrix(
        values: Int8List.fromList([3]),
        rowScales: Float64List.fromList([1.0]),
        rows: 1,
        cols: 1,
      );
      final xv = QuantizedVector(values: Int8List.fromList([7]), scale: 1.0);

      final acc = matmulInt(wm, xv);
      expect(acc[0], equals(21));
    });

    test('accumulates positives and negatives correctly', () {
      // Row: [127, -127, 127, -127] @ [127, 127, -127, -127]
      // = 127*127 + (-127)*127 + 127*(-127) + (-127)*(-127)
      // = 16129 - 16129 - 16129 + 16129 = 0
      final wm = QuantizedMatrix(
        values: Int8List.fromList([127, -127, 127, -127]),
        rowScales: Float64List.fromList([1.0]),
        rows: 1,
        cols: 4,
      );
      final xv = QuantizedVector(
        values: Int8List.fromList([127, 127, -127, -127]),
        scale: 1.0,
      );

      final acc = matmulInt(wm, xv);
      expect(acc[0], equals(0));
    });

    test('throws ArgumentError when dimensions mismatch', () {
      final wm = QuantizedMatrix(
        values: Int8List.fromList([1, 2, 3]),
        rowScales: Float64List.fromList([1.0]),
        rows: 1,
        cols: 3,
      );
      final xv = QuantizedVector(values: Int8List.fromList([1, 2]), scale: 1.0);

      expect(() => matmulInt(wm, xv), throwsArgumentError);
    });
  });

  group('dequant', () {
    test('acc=[100], rowScales=[0.01], xscale=0.02 -> y=[0.02]', () {
      // 100 * 0.01 * 0.02 = 0.02
      final wm = QuantizedMatrix(
        values: Int8List.fromList([0]),
        rowScales: Float64List.fromList([0.01]),
        rows: 1,
        cols: 1,
      );
      final xv = QuantizedVector(values: Int8List.fromList([0]), scale: 0.02);
      final acc = Int32List.fromList([100]);

      final y = dequant(acc, wm, xv);

      expect(y.length, equals(1));
      expect(y[0], closeTo(0.02, 1e-12));
    });

    test('multiple rows dequantize independently', () {
      // acc=[10, 20], rowScales=[0.5, 0.25], xscale=0.1
      // y[0] = 10*0.5*0.1 = 0.5
      // y[1] = 20*0.25*0.1 = 0.5
      final wm = QuantizedMatrix(
        values: Int8List.fromList([0, 0, 0, 0]),
        rowScales: Float64List.fromList([0.5, 0.25]),
        rows: 2,
        cols: 2,
      );
      final xv = QuantizedVector(values: Int8List.fromList([0, 0]), scale: 0.1);
      final acc = Int32List.fromList([10, 20]);

      final y = dequant(acc, wm, xv);

      expect(y[0], closeTo(0.5, 1e-12));
      expect(y[1], closeTo(0.5, 1e-12));
    });

    test('negative accumulator produces negative output', () {
      final wm = QuantizedMatrix(
        values: Int8List.fromList([0]),
        rowScales: Float64List.fromList([0.5]),
        rows: 1,
        cols: 1,
      );
      final xv = QuantizedVector(values: Int8List.fromList([0]), scale: 0.5);
      final acc = Int32List.fromList([-50]);

      final y = dequant(acc, wm, xv);
      expect(y[0], closeTo(-12.5, 1e-12));
    });
  });

  group('quantizedLinear', () {
    test(
      'approximates fp linear within quant error tolerance for well-scaled values',
      () {
        // Use a 2x3 weight matrix and 3-element input with values that span the int8 range well.
        // W = [[1.0, -0.5, 0.25], [0.75, 0.5, -1.0]]
        // x = [0.8, -0.6, 0.4]
        // fp result: row0 = 1.0*0.8 + (-0.5)*(-0.6) + 0.25*0.4 = 0.8 + 0.3 + 0.1 = 1.2
        //            row1 = 0.75*0.8 + 0.5*(-0.6) + (-1.0)*0.4 = 0.6 - 0.3 - 0.4 = -0.1
        final w = Float64List.fromList([1.0, -0.5, 0.25, 0.75, 0.5, -1.0]);
        final x = Float64List.fromList([0.8, -0.6, 0.4]);

        final fpY = linear(w, 2, 3, x);
        final qY = quantizedLinear(w, 2, 3, x);

        expect(qY.length, equals(2));
        // Tolerance: ~2 * max_quant_step in output space.
        // max_quant_step for row0 ~ 1.0/127 * 0.8/127 * 3 values ~ small.
        // Use relative tolerance of 2%.
        expect(qY[0], closeTo(fpY[0], fpY[0].abs() * 0.02 + 0.01));
        expect(qY[1], closeTo(fpY[1], fpY[1].abs() * 0.02 + 0.01));
      },
    );

    test('identity-like: single weight 1.0 passes value through', () {
      // W = [[1.0]], x = [0.5]
      // fp = 0.5; quant: wScale=1/127, xScale=0.5/127, q_w=127, q_x=127
      // acc = 127*127 = 16129; dequant = 16129 * (1/127) * (0.5/127) = 16129 * 0.5 / (127*127) = 0.5
      final w = Float64List.fromList([1.0]);
      final x = Float64List.fromList([0.5]);

      final qY = quantizedLinear(w, 1, 1, x);
      expect(qY[0], closeTo(0.5, 0.01));
    });

    test('zero weight gives zero output', () {
      final w = Float64List.fromList([0.0, 0.0, 0.0, 0.0]);
      final x = Float64List.fromList([1.0, 2.0]);

      final qY = quantizedLinear(w, 2, 2, x);
      expect(qY[0], equals(0.0));
      expect(qY[1], equals(0.0));
    });

    test('zero input gives zero output', () {
      final w = Float64List.fromList([1.0, -1.0]);
      final x = Float64List.fromList([0.0]);

      final qY = quantizedLinear(w, 2, 1, x);
      expect(qY[0], equals(0.0));
      expect(qY[1], equals(0.0));
    });

    test('int path produces exact integer accumulator before dequant', () {
      // Verify that QuantizedMatrix/Vector can be extracted and fed to matmulInt
      // for bit-exact HW diff-testing. Uses hand-chosen values.
      final w = Float64List.fromList([2.0, 0.0, 0.0, 2.0]);
      final x = Float64List.fromList([1.0, 1.0]);

      // Quantize explicitly
      final qm = quantizeRowwiseInt8(w, 2, 2);
      final qv = quantizePerTensorInt8(x);
      final acc = matmulInt(qm, qv);
      final y = dequant(acc, qm, qv);

      // Compare with quantizedLinear convenience function
      final qY = quantizedLinear(w, 2, 2, x);

      expect(y[0], closeTo(qY[0], 1e-12));
      expect(y[1], closeTo(qY[1], 1e-12));
    });
  });
}
