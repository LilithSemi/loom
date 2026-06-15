// LoomStreamMatmul emitAcc: the raw int32 accumulators (result_acc) match the
// exact integer matmul, bit-exact. This is the fp16 W4A8 path's int32 source
// (dequantized by LoomDequant instead of requantized to int8).

import 'dart:async';

import 'package:loom/src/hw/stream_matmul.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

int _toI32(int v) => v >= 0x80000000 ? v - 0x100000000 : v;

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  Future<List<int>> runAcc({
    required int rows,
    required int cols,
    required List<int> w, // dense row-major, each in [-7,7]
    required List<int> x, // dense cols, int8
    bool actsExternal = false,
  }) async {
    const peR = 2, peC = 2;
    const maxColTiles = 4; // xWordCap = 2 -> acts_ext fits 64 bits
    final rowBlocks = (rows + peR - 1) ~/ peR;
    final colTiles = (cols + peC - 1) ~/ peC;
    final paddedCols = colTiles * peC;
    final paddedRows = rowBlocks * peR;
    final wordsPerRow = (colTiles + 1) ~/ 2;
    final xWordCap = (maxColTiles + 1) ~/ 2;

    const weightBase = 0x000, actBase = 0x400, multBase = 0x800;
    final mem = List<int>.filled(0x1000, 0);

    int wq(int gr, int gc) =>
        (gr < rows && gc < cols) ? (w[gr * cols + gc] & 0xF) : 0;

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
    for (var c = 0; c < paddedCols; c++) {
      mem[actBase + c] = (c < cols ? x[c] : 0) & 0xFF;
    }
    // mults unused for acc, leave zero.
    int memWord(int a) => (a < 0 || a + 3 >= mem.length)
        ? 0
        : mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final start = Logic(name: 'start');
    final rowBlocksL = Logic(name: 'rb', width: 16)..inject(rowBlocks);
    final colTilesL = Logic(name: 'ct', width: 16)..inject(colTiles);
    final shiftL = Logic(name: 'sh', width: 6)..inject(0);
    final wBase = Logic(name: 'wb', width: 32)..inject(weightBase);
    final aBase = Logic(name: 'ab', width: 32)..inject(actBase);
    final mBase = Logic(name: 'mb', width: 32)..inject(multBase);
    final ack = Logic(name: 'ack')..inject(0);
    final datMiso = Logic(name: 'miso', width: 32)..inject(0);

    final dut = LoomStreamMatmul(
      config: LoomStreamMatmulConfig(
        int4Weights: true,
        emitAcc: true,
        requantLatency: 0,
        actsExternal: actsExternal,
        maxColTiles: maxColTiles,
      ),
    );
    for (final (p, s) in [
      ('clk', clk),
      ('reset', reset),
      ('start', start),
      ('row_blocks', rowBlocksL),
      ('col_tiles', colTilesL),
      ('shift', shiftL),
      ('weight_base', wBase),
      ('act_base', aBase),
      ('mult_base', mBase),
      ('bus_ACK', ack),
      ('bus_DAT_MISO', datMiso),
    ]) {
      dut.input(p).srcConnection! <= s;
    }
    if (actsExternal) {
      // Pack int8 acts 4/word into the acts_ext bus (xWordCap*32 bits).
      var packed = BigInt.zero;
      for (var c = 0; c < paddedCols; c++) {
        final b = (c < cols ? x[c] : 0) & 0xFF;
        packed |= BigInt.from(b) << (8 * c);
      }
      final actsExt = Logic(name: 'acts_ext', width: xWordCap * 32)
        ..inject(LogicValue.ofBigInt(packed, xWordCap * 32));
      dut.input('acts_ext').srcConnection! <= actsExt;
    }
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

    final accs = List<int>.filled(paddedRows, 0);
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
        final accWord = o('result_acc').value.toInt();
        accs[blk * peR] = _toI32(accWord & 0xFFFFFFFF);
        accs[blk * peR + 1] = _toI32((accWord >> 32) & 0xFFFFFFFF);
      }
      if (o('done').value.toBool()) break;
    }
    await Simulator.endSimulation();
    return accs.sublist(0, rows);
  }

  List<int> goldenAcc(int rows, int cols, List<int> w, List<int> x) => [
    for (var r = 0; r < rows; r++)
      [
        for (var c = 0; c < cols; c++) w[r * cols + c] * x[c],
      ].fold(0, (a, b) => a + b),
  ];

  test('result_acc matches exact int32 matmul (4x6 odd colTiles)', () async {
    final w = [
      3, -2, 1, -1, 2, 0, //
      -4, 3, -2, 2, 1, -1, //
      7, -7, 1, 1, -3, 4, //
      -1, 2, -3, 0, 2, -2, //
    ];
    final x = [5, -3, 2, -1, 4, 2];
    final hw = await runAcc(rows: 4, cols: 6, w: w, x: x);
    expect(hw, equals(goldenAcc(4, 6, w, x)));
  });

  test('result_acc matches exact int32 matmul (6x8 even colTiles)', () async {
    final w = [
      for (var r = 0; r < 6; r++)
        for (var c = 0; c < 8; c++) ((r * 3 + c * 5) % 15) - 7,
    ];
    final x = [for (var c = 0; c < 8; c++) ((c * 5) % 9) - 4];
    final hw = await runAcc(rows: 6, cols: 8, w: w, x: x);
    expect(hw, equals(goldenAcc(6, 8, w, x)));
  });

  test('actsExternal: acts from registers, weights from mem (4x6)', () async {
    final w = [
      3, -2, 1, -1, 2, 0, //
      -4, 3, -2, 2, 1, -1, //
      7, -7, 1, 1, -3, 4, //
      -1, 2, -3, 0, 2, -2, //
    ];
    final x = [5, -3, 2, -1, 4, 2];
    final hw = await runAcc(rows: 4, cols: 6, w: w, x: x, actsExternal: true);
    expect(hw, equals(goldenAcc(4, 6, w, x)));
  });

  test('actsExternal: 6x8 even colTiles', () async {
    final w = [
      for (var r = 0; r < 6; r++)
        for (var c = 0; c < 8; c++) ((r * 3 + c * 5) % 15) - 7,
    ];
    final x = [for (var c = 0; c < 8; c++) ((c * 5) % 9) - 4];
    final hw = await runAcc(rows: 6, cols: 8, w: w, x: x, actsExternal: true);
    expect(hw, equals(goldenAcc(6, 8, w, x)));
  });
}
