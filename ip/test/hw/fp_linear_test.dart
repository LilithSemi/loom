// LoomFpLinear end-to-end: fp16 activations + fp16 row scales in, int4 weights
// served from a sim Wishbone memory, fp16 results out, matching the golden fp64
// linear within W4A8 tolerance. This is the reusable compute brick the
// transformer sequencer drives for every projection / MLP matrix.

import 'dart:async';
import 'dart:typed_data';

import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/fp_linear.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

/// Quantize [x] to symmetric int8 on a CALLER-PROVIDED [scale] (not the
/// vector's own max-abs), mirroring sim.zig's `quantWithScale`: a shared/
/// forced act-scale must quantize on that exact grid, not derive its own.
QuantizedVector _quantizeWithScale(Float64List x, double scale) {
  final values = Int8List(x.length);
  for (var i = 0; i < x.length; i++) {
    final v = (x[i] / scale).round();
    values[i] = v < -127 ? -127 : (v > 127 ? 127 : v);
  }
  return QuantizedVector(values: values, scale: scale);
}

/// Result of one [runFpLinear] simulation: the streamed-out fp16 rows plus the
/// fp16 act-scale bit pattern the design actually used for the dequant
/// (captured off the internal LoomActQuant submodule's `scale_out`, so a
/// forced `hostActScale` run can be checked against the auto-computed one
/// bit-exactly).
class FpLinearRun {
  final List<double> rows;
  final int actScaleUsed;
  FpLinearRun(this.rows, this.actScaleUsed);
}

/// Build + drive a LoomFpLinear over one matmul (weights [w], activation [x]),
/// optionally forcing the act scale via the constructor's `hostActScale`
/// signal (fp16 bits). When [hostActScale] is null, no override signal is
/// passed to the constructor, so LoomFpLinear creates no `host_act_scale`
/// port and uses the on-chip aqScale path. Fresh Simulator per call so this
/// can be invoked more than once within a single test.
Future<FpLinearRun> runFpLinear({
  required Float64List w,
  required int rows,
  required int cols,
  required Float64List x,
  int? hostActScale,
}) async {
  await Simulator.reset();
  const peR = 2, peC = 2;
  final rowBlocks = (rows + peR - 1) ~/ peR;
  final colTiles = (cols + peC - 1) ~/ peC;
  final wordsPerRow = (colTiles + 1) ~/ 2;

  // int4 weights with per-row fp16 scales.
  final qm = quantizeRowwiseInt4(w, rows, cols);

  // Weight memory: tile-major int4, 2 tiles per 32-bit word.
  const weightBase = 0x000;
  final mem = List<int>.filled(0x400, 0);
  int wq(int gr, int gc) =>
      (gr < rows && gc < cols) ? (qm.values[gr * cols + gc] & 0xF) : 0;
  for (var rb = 0; rb < rowBlocks; rb++) {
    for (var ct = 0; ct < colTiles; ct++) {
      final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
      final tileByte = wordByte + (ct.isOdd ? 2 : 0);
      mem[weightBase + tileByte] =
          wq(rb * peR, ct * peC) | (wq(rb * peR, ct * peC + 1) << 4);
      mem[weightBase + tileByte + 1] =
          wq(rb * peR + 1, ct * peC) | (wq(rb * peR + 1, ct * peC + 1) << 4);
    }
  }
  int memWord(int a) => (a < 0 || a + 3 >= mem.length)
      ? 0
      : mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);

  final clk = SimpleClockGenerator(10).clk;
  final reset = Logic(name: 'reset');
  final start = Logic(name: 'start');
  final colTilesL = Logic(name: 'ct', width: 16)..inject(colTiles);
  final rowBlocksL = Logic(name: 'rb', width: 16)..inject(rowBlocks);
  final wBase = Logic(name: 'wb', width: 32)..inject(weightBase);
  final xEn = Logic(name: 'x_en');
  final xInL = Logic(name: 'x_in', width: 16);
  final rsEn = Logic(name: 'rs_en');
  final rsInL = Logic(name: 'rs_in', width: 16);
  // Only built (and only passed to the constructor) when a forced value is
  // requested, so LoomFpLinear adds no `host_act_scale` port when this stays
  // null and existing callers are unaffected.
  final hostActScaleL = hostActScale == null
      ? null
      : (Logic(name: 'host_act_scale', width: 16)..inject(hostActScale));
  final ack = Logic(name: 'ack')..inject(0);
  final datMiso = Logic(name: 'miso', width: 32)..inject(0);

  final dut = LoomFpLinear(
    maxColTiles: 4,
    maxRowBlocks: 4,
    hostActScale: hostActScaleL,
  );
  for (final (p, s) in [
    ('clk', clk),
    ('reset', reset),
    ('start', start),
    ('col_tiles', colTilesL),
    ('row_blocks', rowBlocksL),
    ('weight_base', wBase),
    ('x_en', xEn),
    ('x_in', xInL),
    ('rs_en', rsEn),
    ('rs_in', rsInL),
    ('mem_ACK', ack),
    ('mem_DAT_MISO', datMiso),
  ]) {
    dut.input(p).srcConnection! <= s;
  }
  await dut.build();
  Logic o(String n) => dut.output(n);

  // Tap the internal act-quant submodule's scale_out so an "auto" run can
  // report the fp16 bits it actually used (for a bit-exact forced re-run).
  final aqSub = dut.subModules.firstWhere((m) => m.name == 'loom_act_quant');
  final aqScaleOut = aqSub.output('scale_out');

  final fp = FloatingPoint16();
  int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
  final outFp = FloatingPoint16();

  for (final l in [start, xEn, rsEn]) {
    l.inject(0);
  }
  xInL.inject(0);
  rsInL.inject(0);
  reset.inject(1);
  Simulator.setMaxSimTime(5000000);
  unawaited(Simulator.run());
  await clk.nextPosedge;
  await clk.nextPosedge;
  reset.inject(0);
  await clk.nextPosedge;

  // LOAD: stream fp16 activations (col_tiles*2 values, real cols here).
  for (var c = 0; c < cols; c++) {
    xEn.inject(1);
    xInL.inject(e(x[c]));
    await clk.nextPosedge;
  }
  xEn.inject(0);
  // LOAD: stream fp16 per-row weight scales (row_blocks*2 = rows here).
  for (var r = 0; r < rows; r++) {
    rsEn.inject(1);
    rsInL.inject(e(qm.rowScales[r]));
    await clk.nextPosedge;
  }
  rsEn.inject(0);

  start.inject(1);
  await clk.nextPosedge;
  start.inject(0);

  // Run: service weight reads, collect results.
  final got = List<double>.filled(rows, double.nan);
  var ny = 0;
  var guard = 0;
  var actScaleUsed = 0;
  while (guard++ < 20000) {
    await clk.nextNegedge;
    if (o('mem_STB').value.toBool() && o('mem_CYC').value.toBool()) {
      ack.inject(1);
      datMiso.inject(memWord(o('mem_ADR').value.toInt()));
    } else {
      ack.inject(0);
    }
    if (o('y_valid').value.toBool() && ny < rows) {
      outFp.put(o('y').value);
      got[ny++] = outFp.floatingPointValue.toDouble();
    }
    if (aqScaleOut.value.isValid) {
      actScaleUsed = aqScaleOut.value.toInt();
    }
    await clk.nextPosedge;
    if (o('done').value.toBool()) break;
  }
  await Simulator.endSimulation();

  return FpLinearRun(got, actScaleUsed);
}

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('fp16 W4A8 linear matches golden fp64 linear', () async {
    const rows = 4, cols = 6;

    // Weights / activation (deterministic).
    final w = Float64List(rows * cols);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        w[r * cols + c] = ((r * 7 + c * 3) % 11 - 5) * 0.21;
      }
    }
    final x = Float64List.fromList([
      for (var c = 0; c < cols; c++) ((c * 5 + 2) % 9 - 4) * 0.37,
    ]);
    final goldenFp = linear(w, rows, cols, x);

    final run = await runFpLinear(w: w, rows: rows, cols: cols, x: x);

    expect(run.rows.every((v) => !v.isNaN), isTrue, reason: 'emitted all rows');
    for (var r = 0; r < rows; r++) {
      expect(
        run.rows[r],
        closeTo(goldenFp[r], 0.15 + goldenFp[r].abs() * 0.2),
        reason: 'row $r: hw=${run.rows[r]} fp=${goldenFp[r]}',
      );
    }
  });

  test(
    'LoomFpLinear host_act_scale override drives the dequant with a '
    'FORCED, DIFFERENT scale (not the tautological same-value check)',
    () async {
      const rows = 4, cols = 6;
      final w = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          w[r * cols + c] = ((r * 7 + c * 3) % 11 - 5) * 0.21;
        }
      }
      final x = Float64List.fromList([
        for (var c = 0; c < cols; c++) ((c * 5 + 2) % 9 - 4) * 0.37,
      ]);

      // Learn the on-chip scale with NO override wired at all (hostActScale:
      // null -> no host_act_scale port on the DUT, the true default path).
      final auto = await runFpLinear(w: w, rows: rows, cols: cols, x: x);
      final onchipScale = auto.actScaleUsed; // fp16 bits the design computed
      expect(
        onchipScale,
        isNot(equals(0)),
        reason: 'design must have computed a scale',
      );

      final fp = FloatingPoint16();
      double decodeFp16(int bits) {
        fp.put(LogicValue.ofInt(bits, 16));
        return fp.floatingPointValue.toDouble();
      }

      int encodeFp16(double d) => fp.valuePopulator().ofDouble(d).value.toInt();

      final onchipScaleDouble = decodeFp16(onchipScale);
      // Force a scale different from what the design would compute on its own
      // (2x), so a broken/ignored override (mux swapped, or falling back to
      // aqScale) produces the auto result and fails this check.
      final forcedScaleDouble = onchipScaleDouble * 2.0;
      final forcedScaleBits = encodeFp16(forcedScaleDouble);
      expect(
        forcedScaleBits,
        isNot(equals(onchipScale)),
        reason: 'forced scale must differ from the on-chip one',
      );

      final forced = await runFpLinear(
        w: w,
        rows: rows,
        cols: cols,
        x: x,
        hostActScale: forcedScaleBits,
      );

      // The override drives LoomActQuant end to end: both the int8 activation
      // quantization and the dequant multiplier use the forced scale, since a
      // shared act-scale must quantize on the same grid everywhere it's used,
      // not just override the dequant. scale_out reports the forced value
      // itself, not the on-chip one.
      expect(forced.actScaleUsed, equals(forcedScaleBits));

      // Independent golden: int4 weights as usual, but int8 activations
      // quantized with the FORCED scale directly (mirrors sim.zig's
      // quantWithScale: q = round(x/scale), clamp [-127,127]) and dequantized
      // with that same forced scale (decoded back through fp16 to match
      // hardware precision).
      final qm = quantizeRowwiseInt4(w, rows, cols);
      final qv = _quantizeWithScale(x, forcedScaleDouble);
      final acc = matmulInt(qm, qv);
      final golden = [
        for (var r = 0; r < rows; r++)
          acc[r] * qm.rowScales[r] * forcedScaleDouble,
      ];

      expect(
        forced.rows.every((v) => !v.isNaN),
        isTrue,
        reason: 'emitted all rows',
      );
      for (var r = 0; r < rows; r++) {
        expect(
          forced.rows[r],
          closeTo(golden[r], 0.15 + golden[r].abs() * 0.2),
          reason: 'row $r: hw=${forced.rows[r]} golden(forced)=${golden[r]}',
        );
      }
    },
  );
}
