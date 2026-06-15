// LoomStreamMatmul diff-test: weights streamed from a (testbench) Wishbone
// memory, results compared bit-exact to the Dart quant reference. Demonstrates
// the inner/outer dims scaling past a small fixed tile cap.

import 'dart:async';

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

  group('LoomStreamMatmul', () {
    // Runs one matmul on the DUT with weights/acts/mults in a behavioral
    // Wishbone memory, returns the `rows` int8 outputs.
    Future<List<int>> runMatmul({
      required int rows,
      required int cols,
      required List<int> w, // dense row-major rows*cols, int8
      required List<int> x, // dense cols, int8
      required List<int> mults, // rows, uint16
      required int shift,
    }) async {
      const peR = 2, peC = 2;
      final rowBlocks = (rows + peR - 1) ~/ peR;
      final colTiles = (cols + peC - 1) ~/ peC;
      final paddedRows = rowBlocks * peR;
      final paddedCols = colTiles * peC;

      const weightBase = 0x000;
      const actBase = 0x400;
      const multBase = 0x800;
      final mem = List<int>.filled(0x1000, 0);

      // weights, tile-major. Zero-padded.
      var wp = weightBase;
      for (var rb = 0; rb < rowBlocks; rb++) {
        for (var ct = 0; ct < colTiles; ct++) {
          for (var lr = 0; lr < peR; lr++) {
            for (var lc = 0; lc < peC; lc++) {
              final gr = rb * peR + lr;
              final gc = ct * peC + lc;
              final v = (gr < rows && gc < cols) ? w[gr * cols + gc] : 0;
              mem[wp++] = v & 0xFF;
            }
          }
        }
      }
      // activations, packed. Zero-padded to paddedCols.
      for (var c = 0; c < paddedCols; c++) {
        mem[actBase + c] = (c < cols ? x[c] : 0) & 0xFF;
      }
      // mults, uint16 packed. Zero-padded to paddedRows.
      for (var r = 0; r < paddedRows; r++) {
        final m = r < rows ? mults[r] : 0;
        mem[multBase + r * 2] = m & 0xFF;
        mem[multBase + r * 2 + 1] = (m >> 8) & 0xFF;
      }

      int memWord(int addr) {
        if (addr < 0 || addr + 3 >= mem.length) return 0;
        return mem[addr] |
            (mem[addr + 1] << 8) |
            (mem[addr + 2] << 16) |
            (mem[addr + 3] << 24);
      }

      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      final start = Logic(name: 'start');
      final rowBlocksL = Logic(name: 'rb', width: 16);
      final colTilesL = Logic(name: 'ct', width: 16);
      final shiftL = Logic(name: 'sh', width: 6);
      final wBase = Logic(name: 'wb', width: 32);
      final aBase = Logic(name: 'ab', width: 32);
      final mBase = Logic(name: 'mb', width: 32);
      final ack = Logic(name: 'ack');
      final datMiso = Logic(name: 'miso', width: 32);

      final dut = LoomStreamMatmul(config: const LoomStreamMatmulConfig());
      dut.input('clk').srcConnection! <= clk;
      dut.input('reset').srcConnection! <= reset;
      dut.input('start').srcConnection! <= start;
      dut.input('row_blocks').srcConnection! <= rowBlocksL;
      dut.input('col_tiles').srcConnection! <= colTilesL;
      dut.input('shift').srcConnection! <= shiftL;
      dut.input('weight_base').srcConnection! <= wBase;
      dut.input('act_base').srcConnection! <= aBase;
      dut.input('mult_base').srcConnection! <= mBase;
      dut.input('bus_ACK').srcConnection! <= ack;
      dut.input('bus_DAT_MISO').srcConnection! <= datMiso;

      await dut.build();

      Logic o(String n) => dut.output(n);

      for (final s in [start, ack]) {
        s.inject(0);
      }
      rowBlocksL.inject(rowBlocks);
      colTilesL.inject(colTiles);
      shiftL.inject(shift);
      wBase.inject(weightBase);
      aBase.inject(actBase);
      mBase.inject(multBase);
      datMiso.inject(0);
      reset.inject(1);
      Simulator.setMaxSimTime(2000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;

      start.inject(1);
      await clk.nextPosedge;
      start.inject(0);

      final outputs = List<int>.filled(paddedRows, 0);
      var guard = 0;
      while (guard++ < 5000) {
        await clk.nextNegedge;
        if (o('bus_STB').value.toBool() && o('bus_CYC').value.toBool()) {
          ack.inject(1);
          datMiso.inject(memWord(o('bus_ADR').value.toInt()));
        } else {
          ack.inject(0);
        }
        await clk.nextPosedge;
        if (o('result_valid').value.toBool()) {
          final blk = o('result_block').value.toInt();
          final word = o('result').value.toInt();
          outputs[blk * peR] = _toI8(word & 0xFF);
          outputs[blk * peR + 1] = _toI8((word >> 8) & 0xFF);
        }
        if (o('done').value.toBool()) break;
      }
      await Simulator.endSimulation();
      return outputs.sublist(0, rows);
    }

    List<int> golden(
      int rows,
      int cols,
      List<int> w,
      List<int> x,
      List<int> mults,
      int shift,
    ) {
      return [
        for (var r = 0; r < rows; r++)
          _requant(
            [
              for (var c = 0; c < cols; c++) w[r * cols + c] * x[c],
            ].fold(0, (a, b) => a + b),
            mults[r],
            shift,
          ),
      ];
    }

    test('4x6 matmul streamed from memory == golden', () async {
      final w = [
        3, -2, 1, -1, 2, 0, //
        -4, 3, -2, 2, 1, -1, //
        1, 1, 1, 1, 1, 1, //
        -1, 2, -3, 0, 2, -2, //
      ];
      final x = [5, -3, 2, -1, 4, 2];
      final mults = [16, 16, 8, 32];
      const shift = 4;
      final hw = await runMatmul(
        rows: 4,
        cols: 6,
        w: w,
        x: x,
        mults: mults,
        shift: shift,
      );
      expect(hw, equals(golden(4, 6, w, x, mults, shift)));
    });

    test('6x16 matmul (cols far past the old 8x8 cap) == golden', () async {
      final w = [
        for (var r = 0; r < 6; r++)
          for (var c = 0; c < 16; c++) ((r * 7 + c * 3) % 11) - 5,
      ];
      final x = [for (var c = 0; c < 16; c++) ((c * 5) % 9) - 4];
      final mults = [for (var r = 0; r < 6; r++) 8 + r];
      const shift = 5;
      final hw = await runMatmul(
        rows: 6,
        cols: 16,
        w: w,
        x: x,
        mults: mults,
        shift: shift,
      );
      expect(hw, equals(golden(6, 16, w, x, mults, shift)));
    });

    test('8x64 matmul (large inner dim) == golden', () async {
      const rows = 8, cols = 64;
      final w = [
        for (var r = 0; r < rows; r++)
          for (var c = 0; c < cols; c++) ((r * 13 + c * 7) % 15) - 7,
      ];
      final x = [for (var c = 0; c < cols; c++) ((c * 11) % 13) - 6];
      final mults = [for (var r = 0; r < rows; r++) 32 + r];
      const shift = 8;
      final hw = await runMatmul(
        rows: rows,
        cols: cols,
        w: w,
        x: x,
        mults: mults,
        shift: shift,
      );
      expect(hw, equals(golden(rows, cols, w, x, mults, shift)));
    });
  });
}
