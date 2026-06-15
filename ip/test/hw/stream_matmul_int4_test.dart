// W4A8 diff-test: signed int4 weights packed 2-per-byte (8/word = 2 tiles/word)
// streamed from memory, sign-extended to int8 on chip, matmul'd with int8
// activations. Bit-exact to the golden. int4 halves weight storage so SmolLM2
// fits the 128MB DDR3.

import 'dart:async';

import 'package:loom/src/hw/stream_matmul.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

int _roundShift(int p, int s) {
  if (s == 0) return p;
  final b = 1 << (s - 1);
  return p >= 0 ? (p + b) >> s : -((-p + b) >> s);
}

int _requant(int acc, int mult, int shift) {
  final r = _roundShift(acc * mult, shift);
  return r > 127 ? 127 : (r < -127 ? -127 : r);
}

int _toI8(int b) => b >= 0x80 ? b - 256 : b;

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  group('LoomStreamMatmul int4 weights (W4A8)', () {
    Future<List<int>> runMatmul({
      required int rows,
      required int cols,
      required List<int> w, // dense row-major, each in [-7,7]
      required List<int> x, // dense cols, int8
      required List<int> mults,
      required int shift,
    }) async {
      const peR = 2, peC = 2;
      final rowBlocks = (rows + peR - 1) ~/ peR;
      final colTiles = (cols + peC - 1) ~/ peC;
      final paddedCols = colTiles * peC;
      final paddedRows = rowBlocks * peR;
      final wordsPerRow = (colTiles + 1) ~/ 2; // 2 int4 tiles per 32-bit word

      const weightBase = 0x000, actBase = 0x400, multBase = 0x800;
      final mem = List<int>.filled(0x1000, 0);

      int wq(int gr, int gc) =>
          (gr < rows && gc < cols) ? (w[gr * cols + gc] & 0xF) : 0;

      // Weights: int4 tiles, 2 tiles per word (low tile = bytes 0-1, high =
      // bytes 2-3 of the word). Tile (rb,ct) nibbles: [w00,w01,w10,w11].
      for (var rb = 0; rb < rowBlocks; rb++) {
        for (var ct = 0; ct < colTiles; ct++) {
          final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
          final tileByte = wordByte + (ct.isOdd ? 2 : 0);
          final w00 = wq(rb * peR, ct * peC);
          final w01 = wq(rb * peR, ct * peC + 1);
          final w10 = wq(rb * peR + 1, ct * peC);
          final w11 = wq(rb * peR + 1, ct * peC + 1);
          mem[weightBase + tileByte] = w00 | (w01 << 4);
          mem[weightBase + tileByte + 1] = w10 | (w11 << 4);
        }
      }
      for (var c = 0; c < paddedCols; c++) {
        mem[actBase + c] = (c < cols ? x[c] : 0) & 0xFF;
      }
      for (var r = 0; r < paddedRows; r++) {
        final m = r < rows ? mults[r] : 0;
        mem[multBase + r * 2] = m & 0xFF;
        mem[multBase + r * 2 + 1] = (m >> 8) & 0xFF;
      }
      int memWord(int a) => (a < 0 || a + 3 >= mem.length)
          ? 0
          : mem[a] |
                (mem[a + 1] << 8) |
                (mem[a + 2] << 16) |
                (mem[a + 3] << 24);

      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      final start = Logic(name: 'start');
      final rowBlocksL = Logic(name: 'rb', width: 16)..inject(rowBlocks);
      final colTilesL = Logic(name: 'ct', width: 16)..inject(colTiles);
      final shiftL = Logic(name: 'sh', width: 6)..inject(shift);
      final wBase = Logic(name: 'wb', width: 32)..inject(weightBase);
      final aBase = Logic(name: 'ab', width: 32)..inject(actBase);
      final mBase = Logic(name: 'mb', width: 32)..inject(multBase);
      final ack = Logic(name: 'ack')..inject(0);
      final datMiso = Logic(name: 'miso', width: 32)..inject(0);

      final dut = LoomStreamMatmul(
        config: const LoomStreamMatmulConfig(int4Weights: true),
      );
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

      start.inject(0);
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
      while (guard++ < 8000) {
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

    test(
      '4x6 (odd colTiles, exercises the unused high nibble) == golden',
      () async {
        final w = [
          3, -2, 1, -1, 2, 0, //
          -4, 3, -2, 2, 1, -1, //
          7, -7, 1, 1, -3, 4, //
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
      },
    );

    test('6x8 (even colTiles) == golden', () async {
      final w = [
        for (var r = 0; r < 6; r++)
          for (var c = 0; c < 8; c++) ((r * 3 + c * 5) % 15) - 7,
      ];
      final x = [for (var c = 0; c < 8; c++) ((c * 5) % 9) - 4];
      final mults = [for (var r = 0; r < 6; r++) 8 + r];
      const shift = 5;
      final hw = await runMatmul(
        rows: 6,
        cols: 8,
        w: w,
        x: x,
        mults: mults,
        shift: shift,
      );
      expect(hw, equals(golden(6, 8, w, x, mults, shift)));
    });
  });
}
