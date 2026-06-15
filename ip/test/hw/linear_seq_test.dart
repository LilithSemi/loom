// LoomLinearSeq: on-chip FSM buffers an activation vector + per-row weight
// scales, then streams them into an internal LoomFpLinear. Checks FSM +
// Wishbone-master orchestration against the golden W4A8 linear.

import 'dart:async';
import 'dart:typed_data';

import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/linear_seq.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('sequencer FSM drives LoomFpLinear to the golden W4A8 result', () async {
    const rows = 4, cols = 6;
    const peR = 2, peC = 2;
    final rowBlocks = (rows + peR - 1) ~/ peR;
    final colTiles = (cols + peC - 1) ~/ peC;
    final wordsPerRow = (colTiles + 1) ~/ 2;

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

    final qm = quantizeRowwiseInt4(w, rows, cols);

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
    final ack = Logic(name: 'ack')..inject(0);
    final datMiso = Logic(name: 'miso', width: 32)..inject(0);

    final dut = LoomLinearSeq(maxColTiles: 4, maxRowBlocks: 4);
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

    // LOAD: stream activations, then per-row weight scales.
    for (var c = 0; c < cols; c++) {
      xEn.inject(1);
      xInL.inject(e(x[c]));
      await clk.nextPosedge;
    }
    xEn.inject(0);
    for (var r = 0; r < rows; r++) {
      rsEn.inject(1);
      rsInL.inject(e(qm.rowScales[r]));
      await clk.nextPosedge;
    }
    rsEn.inject(0);

    // Kick the sequencer.
    start.inject(1);
    await clk.nextPosedge;
    start.inject(0);

    // Run: service weight reads, collect results, until done.
    final got = List<double>.filled(rows, double.nan);
    var ny = 0;
    var guard = 0;
    while (guard++ < 40000) {
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
      await clk.nextPosedge;
      if (o('done').value.toBool()) break;
    }
    await Simulator.endSimulation();

    expect(ny, equals(rows), reason: 'emitted all rows');
    for (var r = 0; r < rows; r++) {
      expect(
        got[r],
        closeTo(goldenFp[r], 0.15 + goldenFp[r].abs() * 0.2),
        reason: 'row $r: hw=${got[r]} fp=${goldenFp[r]}',
      );
    }
  });
}
