// Integration: LoomStreamMatmul (Wishbone master) reads its weights from a
// REAL HarborSram (Harbor's on-chip scratchpad peripheral, registered ack)
// instead of an idealized testbench memory. The test preloads the SRAM through
// a 2-source bus mux (testbench master during preload, the matmul master during
// the run), then checks the result is bit-exact to the golden.

import 'dart:async';

import 'package:harbor/harbor.dart';
import 'package:loom/src/hw/stream_matmul.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

int _roundShift(int prod, int shift) {
  if (shift == 0) return prod;
  final bias = 1 << (shift - 1);
  return prod >= 0 ? (prod + bias) >> shift : -((-prod + bias) >> shift);
}

int _requant(int acc, int mult, int shift) {
  final r = _roundShift(acc * mult, shift);
  if (r > 127) return 127;
  if (r < -127) return -127;
  return r;
}

int _toI8(int b) => b >= 0x80 ? b - 256 : b;

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test(
    'matmul reads weights from a real HarborSram, bit-exact to golden',
    () async {
      const rows = 6, cols = 16, peR = 2, peC = 2;
      final w = [
        for (var r = 0; r < rows; r++)
          for (var c = 0; c < cols; c++) ((r * 7 + c * 3) % 11) - 5,
      ];
      final x = [for (var c = 0; c < cols; c++) ((c * 5) % 9) - 4];
      final mults = [for (var r = 0; r < rows; r++) 8 + r];
      const shift = 5;

      final rowBlocks = (rows + peR - 1) ~/ peR;
      final colTiles = (cols + peC - 1) ~/ peC;
      final paddedCols = colTiles * peC;
      final paddedRows = rowBlocks * peR;

      const weightBase = 0x000;
      const actBase = 0x400;
      const multBase = 0x800;

      // Build the word image the SRAM must hold.
      final words = <int, int>{}; // byte addr -> 32-bit word
      void putByte(int addr, int v) {
        final wa = addr & ~3;
        final sh = (addr & 3) * 8;
        words[wa] = (words[wa] ?? 0) & ~(0xFF << sh) | ((v & 0xFF) << sh);
      }

      var wp = weightBase;
      for (var rb = 0; rb < rowBlocks; rb++) {
        for (var ct = 0; ct < colTiles; ct++) {
          for (var lr = 0; lr < peR; lr++) {
            for (var lc = 0; lc < peC; lc++) {
              final gr = rb * peR + lr, gc = ct * peC + lc;
              final v = (gr < rows && gc < cols) ? w[gr * cols + gc] : 0;
              putByte(wp++, v);
            }
          }
        }
      }
      for (var c = 0; c < paddedCols; c++) {
        putByte(actBase + c, c < cols ? x[c] : 0);
      }
      for (var r = 0; r < paddedRows; r++) {
        final m = r < rows ? mults[r] : 0;
        putByte(multBase + r * 2, m & 0xFF);
        putByte(multBase + r * 2 + 1, (m >> 8) & 0xFF);
      }

      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');

      final dut = LoomStreamMatmul(config: const LoomStreamMatmulConfig());
      final sram = HarborSram(baseAddress: 0, size: 4096, busAddressWidth: 32);

      // Test-side master signals (used to preload the SRAM).
      final mode = Logic(name: 'mode'); // 0 = testbench, 1 = matmul
      final tbCyc = Logic(name: 'tb_cyc');
      final tbStb = Logic(name: 'tb_stb');
      final tbWe = Logic(name: 'tb_we');
      final tbAdr = Logic(name: 'tb_adr', width: 32);
      final tbDat = Logic(name: 'tb_dat', width: 32);
      final tbSel = Logic(name: 'tb_sel', width: 4);

      // Matmul control.
      final start = Logic(name: 'start');
      final rowBlocksL = Logic(name: 'rbL', width: 16)..inject(rowBlocks);
      final colTilesL = Logic(name: 'ctL', width: 16)..inject(colTiles);
      final shiftL = Logic(name: 'shL', width: 6)..inject(shift);
      final wBaseL = Logic(name: 'wbL', width: 32)..inject(weightBase);
      final aBaseL = Logic(name: 'abL', width: 32)..inject(actBase);
      final mBaseL = Logic(name: 'mbL', width: 32)..inject(multBase);

      dut.input('clk').srcConnection! <= clk;
      dut.input('reset').srcConnection! <= reset;
      dut.input('start').srcConnection! <= start;
      dut.input('row_blocks').srcConnection! <= rowBlocksL;
      dut.input('col_tiles').srcConnection! <= colTilesL;
      dut.input('shift').srcConnection! <= shiftL;
      dut.input('weight_base').srcConnection! <= wBaseL;
      dut.input('act_base').srcConnection! <= aBaseL;
      dut.input('mult_base').srcConnection! <= mBaseL;

      sram.input('clk').srcConnection! <= clk;
      sram.input('reset').srcConnection! <= reset;

      // 2-source mux onto the SRAM slave inputs (matmul master vs testbench).
      sram.input('bus_CYC').srcConnection! <=
          mux(mode, dut.output('bus_CYC'), tbCyc);
      sram.input('bus_STB').srcConnection! <=
          mux(mode, dut.output('bus_STB'), tbStb);
      sram.input('bus_WE').srcConnection! <=
          mux(mode, dut.output('bus_WE'), tbWe);
      sram.input('bus_ADR').srcConnection! <=
          mux(mode, dut.output('bus_ADR'), tbAdr);
      sram.input('bus_DAT_MOSI').srcConnection! <=
          mux(mode, dut.output('bus_DAT_MOSI'), tbDat);
      sram.input('bus_SEL').srcConnection! <=
          mux(mode, dut.output('bus_SEL'), tbSel);

      dut.input('bus_ACK').srcConnection! <= sram.output('bus_ACK');
      dut.input('bus_DAT_MISO').srcConnection! <= sram.output('bus_DAT_MISO');

      await dut.build();

      for (final s in [start, tbCyc, tbStb, tbWe, mode]) {
        s.inject(0);
      }
      tbAdr.inject(0);
      tbDat.inject(0);
      tbSel.inject(0xF);
      reset.inject(1);
      Simulator.setMaxSimTime(5000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;

      mode.inject(0);
      Future<void> wbWrite(int addr, int word) async {
        tbCyc.inject(1);
        tbStb.inject(1);
        tbWe.inject(1);
        tbAdr.inject(addr);
        tbDat.inject(word);
        var guard = 0;
        do {
          await clk.nextPosedge;
        } while (!sram.output('bus_ACK').value.toBool() && guard++ < 50);
        tbCyc.inject(0);
        tbStb.inject(0);
        tbWe.inject(0);
        await clk.nextPosedge;
      }

      for (final entry in words.entries) {
        await wbWrite(entry.key, entry.value);
      }

      mode.inject(1);
      await clk.nextPosedge;
      start.inject(1);
      await clk.nextPosedge;
      start.inject(0);

      final outputs = List<int>.filled(paddedRows, 0);
      var guard = 0;
      while (guard++ < 20000) {
        await clk.nextPosedge;
        if (dut.output('result_valid').value.toBool()) {
          final blk = dut.output('result_block').value.toInt();
          final word = dut.output('result').value.toInt();
          outputs[blk * peR] = _toI8(word & 0xFF);
          outputs[blk * peR + 1] = _toI8((word >> 8) & 0xFF);
        }
        if (dut.output('done').value.toBool()) break;
      }
      await Simulator.endSimulation();

      final golden = [
        for (var r = 0; r < rows; r++)
          _requant(
            [
              for (var c = 0; c < cols; c++) w[r * cols + c] * x[c],
            ].fold(0, (a, b) => a + b),
            mults[r],
            shift,
          ),
      ];
      expect(outputs.sublist(0, rows), equals(golden));
    },
  );
}
