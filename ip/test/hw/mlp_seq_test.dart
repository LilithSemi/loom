// LoomMlpSeq: the SwiGLU MLP sequencer brick, proving ONE reused LoomFpLinear
// time-shared across three matmuls (gate/up/down) with an elementwise
// SiLU(gate)*up between them: y = W_down @ (SiLU(W_gate @ x) * (W_up @ x)).
// Compared against the W4A8-quantized golden chain within fp16 tolerance.

import 'dart:async';
import 'dart:typed_data';

import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/mlp_seq.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test(
    'MLP sequencer computes SwiGLU via one reused engine to golden',
    () async {
      const hidden = 6, iSize = 8;
      const peR = 2, peC = 2;

      Float64List mat(int rows, int cols, int seed) {
        final m = Float64List(rows * cols);
        for (var r = 0; r < rows; r++) {
          for (var c = 0; c < cols; c++) {
            m[r * cols + c] = ((r * 5 + c * 3 + seed) % 11 - 5) * 0.19;
          }
        }
        return m;
      }

      final wGate = mat(iSize, hidden, 1);
      final wUp = mat(iSize, hidden, 4);
      final wDown = mat(hidden, iSize, 7);
      final x = Float64List.fromList([
        for (var c = 0; c < hidden; c++) ((c * 5 + 2) % 9 - 4) * 0.33,
      ]);

      // W4A8-quantized golden chain (matches the hardware datapath closely).
      final gate = quantizedLinearW4A8(wGate, iSize, hidden, x);
      final up = quantizedLinearW4A8(wUp, iSize, hidden, x);
      final h = Float64List(iSize);
      for (var i = 0; i < iSize; i++) {
        h[i] = silu(Float64List.fromList([gate[i]]))[0] * up[i];
      }
      final golden = quantizedLinearW4A8(wDown, hidden, iSize, h);

      // Weight memory: gate @0x000, up @0x040, down @0x080.
      const wbGate = 0x000, wbUp = 0x040, wbDown = 0x080;
      final mem = List<int>.filled(0x400, 0);
      void pack(int base, Float64List w, int rows, int cols) {
        final qm = quantizeRowwiseInt4(w, rows, cols);
        final rowBlocks = (rows + peR - 1) ~/ peR;
        final colTiles = (cols + peC - 1) ~/ peC;
        final wordsPerRow = (colTiles + 1) ~/ 2;
        int wq(int gr, int gc) =>
            (gr < rows && gc < cols) ? (qm.values[gr * cols + gc] & 0xF) : 0;
        for (var rb = 0; rb < rowBlocks; rb++) {
          for (var ct = 0; ct < colTiles; ct++) {
            final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
            final tileByte = base + wordByte + (ct.isOdd ? 2 : 0);
            mem[tileByte] =
                wq(rb * peR, ct * peC) | (wq(rb * peR, ct * peC + 1) << 4);
            mem[tileByte + 1] =
                wq(rb * peR + 1, ct * peC) |
                (wq(rb * peR + 1, ct * peC + 1) << 4);
          }
        }
      }

      pack(wbGate, wGate, iSize, hidden);
      pack(wbUp, wUp, iSize, hidden);
      pack(wbDown, wDown, hidden, iSize);
      int memWord(int a) => (a < 0 || a + 3 >= mem.length)
          ? 0
          : mem[a] |
                (mem[a + 1] << 8) |
                (mem[a + 2] << 16) |
                (mem[a + 3] << 24);

      final qmGate = quantizeRowwiseInt4(wGate, iSize, hidden);
      final qmUp = quantizeRowwiseInt4(wUp, iSize, hidden);
      final qmDown = quantizeRowwiseInt4(wDown, hidden, iSize);

      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      final start = Logic(name: 'start');
      final wbg = Logic(name: 'wbg', width: 32)..inject(wbGate);
      final wbu = Logic(name: 'wbu', width: 32)..inject(wbUp);
      final wbd = Logic(name: 'wbd', width: 32)..inject(wbDown);
      final xEn = Logic(name: 'x_en');
      final xInL = Logic(name: 'x_in', width: 16);
      final sgEn = Logic(name: 'sg_en');
      final sgInL = Logic(name: 'sg_in', width: 16);
      final suEn = Logic(name: 'su_en');
      final suInL = Logic(name: 'su_in', width: 16);
      final sdEn = Logic(name: 'sd_en');
      final sdInL = Logic(name: 'sd_in', width: 16);
      final ack = Logic(name: 'ack')..inject(0);
      final datMiso = Logic(name: 'miso', width: 32)..inject(0);

      final dut = LoomMlpSeq(hidden: hidden, iSize: iSize);
      for (final (p, s) in [
        ('clk', clk),
        ('reset', reset),
        ('start', start),
        ('wb_gate', wbg),
        ('wb_up', wbu),
        ('wb_down', wbd),
        ('x_en', xEn),
        ('x_in', xInL),
        ('sg_en', sgEn),
        ('sg_in', sgInL),
        ('su_en', suEn),
        ('su_in', suInL),
        ('sd_en', sdEn),
        ('sd_in', sdInL),
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

      for (final l in [start, xEn, sgEn, suEn, sdEn]) {
        l.inject(0);
      }
      xInL.inject(0);
      sgInL.inject(0);
      suInL.inject(0);
      sdInL.inject(0);
      reset.inject(1);
      Simulator.setMaxSimTime(10000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;

      // LOAD: x, then the three scale tables.
      for (var i = 0; i < hidden; i++) {
        xEn.inject(1);
        xInL.inject(e(x[i]));
        await clk.nextPosedge;
      }
      xEn.inject(0);
      for (var i = 0; i < iSize; i++) {
        sgEn.inject(1);
        sgInL.inject(e(qmGate.rowScales[i]));
        suEn.inject(1);
        suInL.inject(e(qmUp.rowScales[i]));
        await clk.nextPosedge;
      }
      sgEn.inject(0);
      suEn.inject(0);
      for (var i = 0; i < hidden; i++) {
        sdEn.inject(1);
        sdInL.inject(e(qmDown.rowScales[i]));
        await clk.nextPosedge;
      }
      sdEn.inject(0);

      start.inject(1);
      await clk.nextPosedge;
      start.inject(0);

      final got = <double>[];
      var guard = 0;
      while (guard++ < 80000) {
        await clk.nextNegedge;
        if (o('mem_STB').value.toBool() && o('mem_CYC').value.toBool()) {
          ack.inject(1);
          datMiso.inject(memWord(o('mem_ADR').value.toInt()));
        } else {
          ack.inject(0);
        }
        if (o('o_valid').value.toBool() && got.length < hidden) {
          outFp.put(o('o').value);
          got.add(outFp.floatingPointValue.toDouble());
        }
        await clk.nextPosedge;
        if (o('done').value.toBool()) break;
      }
      await Simulator.endSimulation();

      expect(got.length, equals(hidden), reason: 'emitted all rows');
      for (var r = 0; r < hidden; r++) {
        expect(
          got[r],
          closeTo(golden[r], 0.15 + golden[r].abs() * 0.25),
          reason: 'row $r: hw=${got[r]} golden=${golden[r]}',
        );
      }
    },
  );
}
