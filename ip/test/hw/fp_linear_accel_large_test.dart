// 8x8 W4A8 linear through the maxColTiles/maxRowBlocks=64 accelerator config,
// exercising a larger tile count than the size=4 bring-up config.

import 'dart:async';
import 'dart:typed_data';

import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/fp_linear_accelerator.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test(
    '8x8 W4A8 through the maxColTiles=64 accelerator matches golden',
    () async {
      const rows = 8, cols = 8, peR = 2, peC = 2;
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
      final goldenFp = quantizedLinearW4A8(w, rows, cols, x);
      final qm = quantizeRowwiseInt4(w, rows, cols);

      const weightBase = 0x000;
      final mem = List<int>.filled(0x4000, 0);
      int wq(int gr, int gc) =>
          (gr < rows && gc < cols) ? (qm.values[gr * cols + gc] & 0xF) : 0;
      for (var rb = 0; rb < rowBlocks; rb++) {
        for (var ct = 0; ct < colTiles; ct++) {
          final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
          final tileByte = wordByte + (ct.isOdd ? 2 : 0);
          mem[weightBase + tileByte] =
              wq(rb * peR, ct * peC) | (wq(rb * peR, ct * peC + 1) << 4);
          mem[weightBase + tileByte + 1] =
              wq(rb * peR + 1, ct * peC) |
              (wq(rb * peR + 1, ct * peC + 1) << 4);
        }
      }
      int memWord(int a) => (a < 0 || a + 3 >= mem.length)
          ? 0
          : mem[a] |
                (mem[a + 1] << 8) |
                (mem[a + 2] << 16) |
                (mem[a + 3] << 24);

      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      final cCyc = Logic(name: 'c_cyc');
      final cStb = Logic(name: 'c_stb');
      final cWe = Logic(name: 'c_we');
      final cAdr = Logic(name: 'c_adr', width: 32);
      final cDat = Logic(name: 'c_dat', width: 32);
      final cSel = Logic(name: 'c_sel', width: 4);
      final memAck = Logic(name: 'mem_ack');
      final memMiso = Logic(name: 'mem_miso', width: 32);

      final dut = LoomFpLinearAccelerator(maxColTiles: 64, maxRowBlocks: 64);
      dut.input('clk').srcConnection! <= clk;
      dut.input('reset').srcConnection! <= reset;
      dut.input('bus_CYC').srcConnection! <= cCyc;
      dut.input('bus_STB').srcConnection! <= cStb;
      dut.input('bus_WE').srcConnection! <= cWe;
      dut.input('bus_ADR').srcConnection! <= cAdr;
      dut.input('bus_DAT_MOSI').srcConnection! <= cDat;
      dut.input('bus_SEL').srcConnection! <= cSel;
      dut.input('mem_ACK').srcConnection! <= memAck;
      dut.input('mem_DAT_MISO').srcConnection! <= memMiso;
      await dut.build();
      Logic o(String n) => dut.output(n);

      final fp = FloatingPoint16();
      int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
      final outFp = FloatingPoint16();

      for (final s in [cCyc, cStb, cWe, memAck]) {
        s.inject(0);
      }
      cAdr.inject(0);
      cDat.inject(0);
      cSel.inject(0xF);
      memMiso.inject(0);
      reset.inject(1);
      Simulator.setMaxSimTime(10000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;

      Future<void> tick() async {
        await clk.nextNegedge;
        if (o('mem_STB').value.toBool() && o('mem_CYC').value.toBool()) {
          memAck.inject(1);
          memMiso.inject(memWord(o('mem_ADR').value.toInt()));
        } else {
          memAck.inject(0);
        }
        await clk.nextPosedge;
      }

      Future<void> csrWrite(int addr, int val) async {
        cCyc.inject(1);
        cStb.inject(1);
        cWe.inject(1);
        cAdr.inject(addr);
        cDat.inject(val);
        var g = 0;
        do {
          await tick();
        } while (!o('bus_ACK').value.toBool() && g++ < 100);
        cCyc.inject(0);
        cStb.inject(0);
        cWe.inject(0);
        await tick();
      }

      Future<int> csrRead(int addr) async {
        cCyc.inject(1);
        cStb.inject(1);
        cWe.inject(0);
        cAdr.inject(addr);
        var g = 0;
        do {
          await tick();
        } while (!o('bus_ACK').value.toBool() && g++ < 100);
        final v = o('bus_DAT_MISO').value.toInt();
        cCyc.inject(0);
        cStb.inject(0);
        await tick();
        return v;
      }

      await csrWrite(0x004, colTiles);
      await csrWrite(0x008, rowBlocks);
      await csrWrite(0x00C, weightBase);
      for (var c = 0; c < cols; c++) {
        await csrWrite(0x018, e(x[c]));
      }
      for (var r = 0; r < rows; r++) {
        await csrWrite(0x01C, e(qm.rowScales[r]));
      }
      await csrWrite(0x010, 0x1);

      var g = 0;
      while (g++ < 20000) {
        if ((await csrRead(0x014) & 0x2) != 0) break;
      }

      // Results are packed two fp16 rows per word: row 2w in low16, row 2w+1
      // in high16.
      final got = <double>[];
      for (var w = 0; w < (rows + 1) ~/ 2; w++) {
        final word = await csrRead(0x100 + w * 4);
        outFp.put(LogicValue.ofInt(word & 0xFFFF, 16));
        got.add(outFp.floatingPointValue.toDouble());
        if (2 * w + 1 < rows) {
          outFp.put(LogicValue.ofInt((word >> 16) & 0xFFFF, 16));
          got.add(outFp.floatingPointValue.toDouble());
        }
      }
      await Simulator.endSimulation();

      for (var r = 0; r < rows; r++) {
        expect(
          got[r],
          closeTo(goldenFp[r], 0.04 + goldenFp[r].abs() * 0.06),
          reason: 'row $r: hw=${got[r]} golden=${goldenFp[r]}  all=$got',
        );
      }
    },
  );
}
