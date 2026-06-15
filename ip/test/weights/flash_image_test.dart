// The flash weight-image generator lays every linear weight tile-major (int4)
// plus a per-row fp16 scale table, with a self-describing manifest of
// per-matrix offsets. This manifest is the weight-base table the on-chip
// sequencer loads into LoomFpLinear's CSRs.
//
// Correctness bar: reconstructing W4A8 from the flash image + manifest must
// reproduce the golden W4A8 linear (quantizedLinearW4A8) bit-for-bit.

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

TensorView _randF32(String name, List<int> shape, math.Random rng) {
  final n = shape.reduce((a, b) => a * b);
  final data = Float32List(n);
  for (var i = 0; i < n; i++) {
    data[i] = rng.nextDouble() * 2 - 1;
  }
  return TensorView(
    name: name,
    shape: shape,
    dtype: TensorDType.f32,
    bytes: ByteData.sublistView(data),
  );
}

BoundModel _syntheticModel(math.Random rng) {
  const h = 8; // hidden
  const nH = 4, hd = 2; // 4 heads x head_dim 2 => qDim 8
  const nKV = 2; // kvDim 4
  const iSize = 12; // ffn
  const v = 16; // vocab
  BoundLayer layer(int i) => BoundLayer(
    inputNorm: _randF32('l$i.in', [h], rng),
    qProj: _randF32('l$i.q', [nH * hd, h], rng),
    kProj: _randF32('l$i.k', [nKV * hd, h], rng),
    vProj: _randF32('l$i.v', [nKV * hd, h], rng),
    oProj: _randF32('l$i.o', [h, nH * hd], rng),
    postAttnNorm: _randF32('l$i.pn', [h], rng),
    gate: _randF32('l$i.g', [iSize, h], rng),
    up: _randF32('l$i.u', [iSize, h], rng),
    down: _randF32('l$i.d', [h, iSize], rng),
  );
  final embed = _randF32('embed', [v, h], rng);
  return BoundModel(
    embedTokens: embed,
    layers: [layer(0), layer(1)],
    finalNorm: _randF32('fn', [h], rng),
    lmHead: embed,
    lmHeadTied: true,
  );
}

void main() {
  test('flash image + manifest reproduces golden W4A8 for every linear', () {
    final rng = math.Random(1234);
    final model = _syntheticModel(rng);

    final image = flashImageFor(model);

    // One manifest entry per linear: 7 per layer * 2 layers + lm_head = 15.
    expect(image.manifest.length, 15);

    final xRng = math.Random(99);
    for (final e in image.manifest) {
      // Reconstruct the int4 weights from the tile-major slice.
      final slice = Uint8List.sublistView(
        image.weights,
        e.weightOffset,
        e.weightOffset + e.weightLength,
      );
      final values = unpackTileMajorInt4(slice, e.rows, e.cols);

      // Reconstruct per-row fp16 scales from the scale table.
      final rowScales = Float64List(e.rows);
      for (var r = 0; r < e.rows; r++) {
        final lo = image.scales[(e.scaleOffset + r) * 2];
        final hi = image.scales[(e.scaleOffset + r) * 2 + 1];
        rowScales[r] = Half.toDouble(lo | (hi << 8));
      }

      // Device-faithful reference: the SAME int4 quantization the device does,
      // with scales rounded to fp16 (which is what the hardware stores and what
      // the flash image must encode). This is exact, no tolerance fudge.
      final w = _weightFor(model, e.name).toFloat64List();
      final ref = quantizeRowwiseInt4(w, e.rows, e.cols);
      final refScales = Float64List(e.rows);
      for (var r = 0; r < e.rows; r++) {
        refScales[r] = Half.toDouble(Half.fromDouble(ref.rowScales[r]));
      }

      // (1) int4 weight values encode bit-exactly.
      expect(
        values,
        orderedEquals(ref.values),
        reason: 'int4 values differ for ${e.name}',
      );
      // (2) fp16 scales encode bit-exactly.
      expect(
        rowScales,
        orderedEquals(refScales),
        reason: 'fp16 scales differ for ${e.name}',
      );

      // (3) End-to-end W4A8 from the flash image == device-faithful reference.
      final qmFlash = QuantizedMatrix(
        values: values,
        rowScales: rowScales,
        rows: e.rows,
        cols: e.cols,
      );
      final qmRef = QuantizedMatrix(
        values: ref.values,
        rowScales: refScales,
        rows: e.rows,
        cols: e.cols,
      );
      final x = Float64List(e.cols);
      for (var c = 0; c < e.cols; c++) {
        x[c] = xRng.nextDouble() * 2 - 1;
      }
      final qv = quantizePerTensorInt8(x);
      expect(
        dequant(matmulInt(qmFlash, qv), qmFlash, qv),
        orderedEquals(dequant(matmulInt(qmRef, qv), qmRef, qv)),
        reason: 'flash W4A8 != device-faithful W4A8 for ${e.name}',
      );
    }
  });

  _perGroupTests();
  _ternaryTests();
}

// BitNet ternary emit: flashImageFor(ternary: true) must reproduce the golden
// quantizeTernaryAbsmean bit-for-bit (values {-1,0,+1} packed as int4, one
// per-tensor beta replicated per row, groups == 1), so the existing int4
// sim/runtime datapath runs the ternary model unchanged.
void _ternaryTests() {
  test('flash image (ternary: true) reproduces golden ternary per matrix', () {
    final rng = math.Random(2718);
    final model = _syntheticModel(rng);
    final image = flashImageFor(
      model,
      ternary: true,
    ); // maxCols 0: single block

    final xRng = math.Random(31);
    for (final e in image.manifest) {
      expect(
        e.groups,
        1,
        reason: 'ternary is per-tensor (groups==1) for ${e.name}',
      );
      final slice = Uint8List.sublistView(
        image.weights,
        e.weightOffset,
        e.weightOffset + e.weightLength,
      );
      final values = unpackTileMajorInt4(slice, e.rows, e.cols);
      final rowScales = Float64List(e.rows);
      for (var r = 0; r < e.rows; r++) {
        final lo = image.scales[(e.scaleOffset + r) * 2];
        final hi = image.scales[(e.scaleOffset + r) * 2 + 1];
        rowScales[r] = Half.toDouble(lo | (hi << 8));
      }

      final w = _weightFor(model, e.name).toFloat64List();
      final ref = quantizeTernaryAbsmean(w, e.rows, e.cols);
      final refScales = Float64List(e.rows);
      for (var r = 0; r < e.rows; r++) {
        refScales[r] = Half.toDouble(Half.fromDouble(ref.rowScales[r]));
      }

      // (1) ternary int4 values encode bit-exactly and are all in {-1,0,+1}.
      expect(
        values,
        orderedEquals(ref.values),
        reason: 'ternary values ${e.name}',
      );
      for (final v in values) {
        expect(v >= -1 && v <= 1, isTrue);
      }
      // (2) every stored scale is the one per-tensor beta.
      for (var r = 0; r < e.rows; r++) {
        expect(
          rowScales[r],
          refScales[0],
          reason: 'scale[$r] != beta for ${e.name}',
        );
      }
      // (3) full W1.58A8 matmul from the flash image == golden ternary.
      final qmFlash = QuantizedMatrix(
        values: values,
        rowScales: rowScales,
        rows: e.rows,
        cols: e.cols,
      );
      final qmRef = QuantizedMatrix(
        values: ref.values,
        rowScales: refScales,
        rows: e.rows,
        cols: e.cols,
      );
      final x = Float64List(e.cols);
      for (var c = 0; c < e.cols; c++) {
        x[c] = xRng.nextDouble() * 2 - 1;
      }
      final qv = quantizePerTensorInt8(x);
      expect(
        dequant(matmulInt(qmFlash, qv), qmFlash, qv),
        orderedEquals(dequant(matmulInt(qmRef, qv), qmRef, qv)),
        reason: 'flash ternary W1.58A8 != golden for ${e.name}',
      );
    }
  });
}

/// Unpacks a col-block-contiguous int4 image (each block an independent
/// tile-major [rows x bc]) back to a dense [rows x cols] int8, the inverse of
/// _packColBlocksInt4.
Int8List _unpackColBlocksInt4(
  Uint8List img,
  int rows,
  int cols,
  int blockCols,
) {
  final out = Int8List(rows * cols);
  var off = 0;
  var c = 0;
  while (c < cols) {
    final bc = (c + blockCols <= cols) ? blockCols : cols - c;
    final bb = packTileMajorInt4(Int8List(rows * bc), rows, bc).length;
    final block = unpackTileMajorInt4(
      Uint8List.sublistView(img, off, off + bb),
      rows,
      bc,
    );
    for (var r = 0; r < rows; r++) {
      for (var k = 0; k < bc; k++) {
        out[r * cols + c + k] = block[r * bc + k];
      }
    }
    off += bb;
    c += blockCols;
  }
  return out;
}

// Per-group emit: with maxCols small every matrix col-tiles, so the flash image
// must switch to per-group W4A8 (group == col-block) with a group-major scale
// table. Reconstructing from the image + manifest must reproduce the golden
// quantizeGroupwise (values, scales, and the full groupwise matmul) bit-for-bit.
void _perGroupTests() {
  test(
    'flash image emits per-group W4A8 == quantizeGroupwise (group == col-block)',
    () {
      final rng = math.Random(4321);
      final model = _syntheticModel(rng);
      const maxCols =
          4; // every matrix (cols 8/12/16) col-tiles into groups of 4

      final image = flashImageFor(model, maxCols: maxCols);
      final xRng = math.Random(7);
      var sawGrouped = false;
      for (final e in image.manifest) {
        final blockCols = e.cols > maxCols ? maxCols : e.cols;
        final gpr = (e.cols + blockCols - 1) ~/ blockCols;
        expect(e.groups, gpr, reason: 'groups mismatch for ${e.name}');
        if (gpr <= 1) continue;
        sawGrouped = true;

        // Reconstruct int4 values (col-block layout) + group-major scales.
        final slice = Uint8List.sublistView(
          image.weights,
          e.weightOffset,
          e.weightOffset + e.weightLength,
        );
        final values = _unpackColBlocksInt4(slice, e.rows, e.cols, blockCols);
        final scales = Float64List(
          gpr * e.rows,
        ); // [group0 rows][group1 rows]...
        for (var i = 0; i < gpr * e.rows; i++) {
          final lo = image.scales[(e.scaleOffset + i) * 2];
          final hi = image.scales[(e.scaleOffset + i) * 2 + 1];
          scales[i] = Half.toDouble(lo | (hi << 8));
        }

        // Golden groupwise, with scales fp16-rounded like the flash image stores.
        final w = _weightFor(model, e.name).toFloat64List();
        final ref = quantizeGroupwise(
          w,
          e.rows,
          e.cols,
          bits: 4,
          groupSize: blockCols,
        );
        expect(
          values,
          orderedEquals(ref.values),
          reason: 'int4 values differ for ${e.name}',
        );
        // ref.scales is row-major [r*gpr + g]. Flash is group-major [g*rows + r].
        for (var g = 0; g < gpr; g++) {
          for (var r = 0; r < e.rows; r++) {
            expect(
              scales[g * e.rows + r],
              Half.toDouble(Half.fromDouble(ref.scales[r * gpr + g])),
              reason: 'scale[$g,$r] differs for ${e.name}',
            );
          }
        }

        // Full groupwise matmul from the flash image == golden (fp16-rounded scales).
        final refFp16 = GroupQuantizedMatrix(
          values: ref.values,
          scales: Float64List.fromList([
            for (final s in ref.scales) Half.toDouble(Half.fromDouble(s)),
          ]),
          rows: e.rows,
          cols: e.cols,
          bits: 4,
          groupSize: blockCols,
        );
        // Rebuild a GroupQuantizedMatrix from the flash image's group-major scales.
        final flashScalesRowMajor = Float64List(e.rows * gpr);
        for (var g = 0; g < gpr; g++) {
          for (var r = 0; r < e.rows; r++) {
            flashScalesRowMajor[r * gpr + g] = scales[g * e.rows + r];
          }
        }
        final flashQm = GroupQuantizedMatrix(
          values: values,
          scales: flashScalesRowMajor,
          rows: e.rows,
          cols: e.cols,
          bits: 4,
          groupSize: blockCols,
        );
        final x = Float64List(e.cols);
        for (var c = 0; c < e.cols; c++) {
          x[c] = xRng.nextDouble() * 2 - 1;
        }
        final qv = quantizePerTensorInt8(x);
        expect(
          dequantGroupwise(matmulIntGroupwise(flashQm, qv), flashQm, qv),
          orderedEquals(
            dequantGroupwise(matmulIntGroupwise(refFp16, qv), refFp16, qv),
          ),
          reason: 'flash groupwise W4A8 != golden for ${e.name}',
        );
      }
      expect(
        sawGrouped,
        isTrue,
        reason: 'no matrix col-tiled; test is vacuous',
      );
    },
  );
}

TensorView _weightFor(BoundModel m, String name) {
  if (name == 'lm_head') return m.lmHead;
  final parts = name.split('.'); // layers.<i>.<which>
  final i = int.parse(parts[1]);
  final l = m.layers[i];
  return switch (parts[2]) {
    'q_proj' => l.qProj,
    'k_proj' => l.kProj,
    'v_proj' => l.vProj,
    'o_proj' => l.oProj,
    'gate_proj' => l.gate!,
    'up_proj' => l.up!,
    'down_proj' => l.down!,
    _ => throw ArgumentError(name),
  };
}
