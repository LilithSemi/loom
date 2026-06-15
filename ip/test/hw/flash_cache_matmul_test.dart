// Flash read-ahead cache efficiency measurement for the Loom W4A8 matmul path.
//
// Wires the real int4 weight-streaming matmul master (LoomStreamMatmul) to the
// real memory-mapped SPI flash controller (HarborSpiFlashController) in
// STANDARD mode, backed by a procedural single-bit SPI flash model. Standard
// mode reuses the proven MISO/MOSI bit model from harbor's
// spi_flash_read_test.dart (no quad inout co-sim).
//
// Trusted only when the matmul result is bit-exact to the Dart golden: a wrong
// result means the flash model is not faithful and any CS count is meaningless.

import 'dart:async';

import 'package:harbor/harbor.dart';
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

List<int> _golden(
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

/// Result of one measured matmul run.
class _RunResult {
  final List<int> outputs;
  final int totalCs; // all flash commands (CS falling edges)
  final int
  weightCs; // flash commands whose decoded addr is in the weight region
  final int actMultCs; // flash commands in the act/mult regions
  final int weightWords; // wordsPerRow * rowBlocks
  _RunResult(
    this.outputs,
    this.totalCs,
    this.weightCs,
    this.actMultCs,
    this.weightWords,
  );
}

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  group('flash read-ahead coalescing in the W4A8 matmul path', () {
    Future<_RunResult> runMatmul({
      required int rows,
      required int cols,
      required List<int> w,
      required List<int> x,
      required List<int> mults,
      required int shift,
      required int readAheadWords,
      bool ternary = false,
    }) async {
      // Fresh simulator per run (this helper is called twice inside one test to
      // compare readAhead=8 vs readAhead=1).
      await Simulator.reset();
      const peR = 2, peC = 2;
      final rowBlocks = (rows + peR - 1) ~/ peR;
      final colTiles = (cols + peC - 1) ~/ peC;
      final paddedCols = colTiles * peC;
      final paddedRows = rowBlocks * peR;
      final wordsPerRow = (colTiles + 1) ~/ 2; // 2 int4 tiles per 32-bit word
      final weightWords = wordsPerRow * rowBlocks;

      const weightBase = 0x000, actBase = 0x400, multBase = 0x800;
      final mem = List<int>.filled(0x1000, 0);

      int wq(int gr, int gc) =>
          (gr < rows && gc < cols) ? (w[gr * cols + gc] & 0xF) : 0;

      // Weights: int4 tiles, 2 tiles/word (low tile bytes 0-1, high bytes 2-3).
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

      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      final start = Logic(name: 'start');
      final rowBlocksL = Logic(name: 'rb', width: 16)..inject(rowBlocks);
      final colTilesL = Logic(name: 'ct', width: 16)..inject(colTiles);
      final shiftL = Logic(name: 'sh', width: 6)..inject(shift);
      final wBase = Logic(name: 'wb', width: 32)..inject(weightBase);
      final aBase = Logic(name: 'ab', width: 32)..inject(actBase);
      final mBase = Logic(name: 'mb', width: 32)..inject(multBase);
      final miso = Logic(name: 'miso');

      final dut = LoomStreamMatmul(
        config: LoomStreamMatmulConfig(
          int4Weights: true,
          ternaryWeights: ternary,
        ),
      );
      final flash = HarborSpiFlashController(
        config: HarborSpiFlashConfig(
          size: 1024 * 1024,
          mode: HarborSpiFlashMode.standard,
          readCommand: 0x03,
          addressBytes: 3,
          dummyCycles: 0,
          readAheadWords: readAheadWords,
        ),
        baseAddress: 0x0,
        busAddressWidth: 32,
        busDataWidth: 32,
      );

      // Matmul scalar/config inputs.
      dut.input('clk').srcConnection! <= clk;
      dut.input('reset').srcConnection! <= reset;
      dut.input('start').srcConnection! <= start;
      dut.input('row_blocks').srcConnection! <= rowBlocksL;
      dut.input('col_tiles').srcConnection! <= colTilesL;
      dut.input('shift').srcConnection! <= shiftL;
      dut.input('weight_base').srcConnection! <= wBase;
      dut.input('act_base').srcConnection! <= aBase;
      dut.input('mult_base').srcConnection! <= mBase;

      // Bus: matmul MASTER outputs drive flash SLAVE inputs, and vice versa.
      flash.input('clk').srcConnection! <= clk;
      flash.input('reset').srcConnection! <= reset;
      flash.input('bus_CYC').srcConnection! <= dut.output('bus_CYC');
      flash.input('bus_STB').srcConnection! <= dut.output('bus_STB');
      flash.input('bus_WE').srcConnection! <= dut.output('bus_WE');
      flash.input('bus_ADR').srcConnection! <= dut.output('bus_ADR');
      flash.input('bus_DAT_MOSI').srcConnection! <= dut.output('bus_DAT_MOSI');
      flash.input('bus_SEL').srcConnection! <= dut.output('bus_SEL');
      dut.input('bus_ACK').srcConnection! <= flash.output('bus_ACK');
      dut.input('bus_DAT_MISO').srcConnection! <= flash.output('bus_DAT_MISO');

      // Standard-mode SPI flash pins.
      flash.input('spi_miso').srcConnection! <= miso;
      flash.input('wr_req').srcConnection! <= Const(0);
      flash.input('wr_op').srcConnection! <= Const(0);
      flash.input('wr_addr').srcConnection! <= Const(0, width: 24);
      flash.input('wr_len').srcConnection! <= Const(0, width: 9);
      flash.input('wr_data').srcConnection! <= Const(0, width: 8);

      await dut.build();
      await flash.build();

      Logic o(String n) => dut.output(n);
      final spiClk = flash.output('spi_clk');
      final csN = flash.output('spi_cs_n');
      final mosi = flash.output('spi_mosi');

      const cmdClocks = 8;
      const addrClocks = 24;
      const dataStart = cmdClocks + addrClocks; // 32

      start.inject(0);
      reset.inject(1);
      miso.inject(0);
      Simulator.setMaxSimTime(20000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;
      start.inject(1);
      await clk.nextPosedge;
      start.inject(0);

      // Procedural standard-SPI flash model + CS-assertion counters.
      var prevClk = 0;
      var prevCs = 1;
      var riseCount = 0;
      var decodedAddr = 0; // aligned line base decoded from the address phase
      var totalCs = 0, weightCs = 0, actMultCs = 0;

      final outputs = List<int>.filled(paddedRows, 0);
      var guard = 0;
      while (guard++ < 400000) {
        await clk.nextNegedge;
        final cs = csN.value.toInt();
        final sc = spiClk.value.toInt();
        if (cs == 1) {
          riseCount = 0;
        } else {
          if (prevCs == 1) {
            // CS just asserted: a fresh flash command.
            riseCount = 0;
            decodedAddr = 0;
            totalCs++;
          }
          if (sc == 1 && prevClk == 0) {
            // Rising edge: sample MOSI (cmd/addr phase) or drive MISO (data).
            if (riseCount >= cmdClocks && riseCount < dataStart) {
              // Address bit, MSB-first: bit (riseCount - cmdClocks) of the
              // 24-bit address, from the flash controller's spi_mosi output.
              decodedAddr = (decodedAddr << 1) | (mosi.value.toInt() & 1);
              if (riseCount == dataStart - 1) {
                // Address fully decoded: classify this flash command by region.
                if (decodedAddr < actBase) {
                  weightCs++;
                } else {
                  actMultCs++;
                }
              }
            } else if (riseCount >= dataStart) {
              // Data phase: serve mem[decodedAddr + byteOffset] MSB-first. The
              // controller streams the whole read-ahead line (readAheadWords
              // words) from the aligned base, so serve sequentially.
              final idx = riseCount - dataStart;
              final byteOff = idx >> 3;
              final bit = 7 - (idx & 7);
              final a = decodedAddr + byteOff;
              final byteVal = (a >= 0 && a < mem.length) ? mem[a] : 0;
              miso.inject((byteVal >> bit) & 1);
            }
            riseCount++;
          }
        }
        prevClk = sc;
        prevCs = cs;

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
      return _RunResult(
        outputs.sublist(0, rows),
        totalCs,
        weightCs,
        actMultCs,
        weightWords,
      );
    }

    // Deterministic pseudo-random-ish weights/acts/mults spanning several lines.
    ({List<int> w, List<int> x, List<int> mults}) makeCase(int rows, int cols) {
      final w = [
        for (var r = 0; r < rows; r++)
          for (var c = 0; c < cols; c++) ((r * 7 + c * 3 + r * c) % 15) - 7,
      ];
      final x = [for (var c = 0; c < cols; c++) ((c * 11 + 5) % 19) - 9];
      final mults = [for (var r = 0; r < rows; r++) 8 + (r % 24)];
      return (w: w, x: x, mults: mults);
    }

    test(
      'TERNARY multiply-free PE + flash: bit-exact to golden',
      timeout: const Timeout(Duration(minutes: 4)),
      () async {
        const rows = 16, cols = 32, shift = 5;
        // Ternary weights {-1,0,+1} (int4 nibbles 0xF/0x0/0x1) in a deterministic
        // pattern that exercises all three. X/mults nonzero so a scramble shows.
        final w = [for (var i = 0; i < rows * cols; i++) (i * 7 + 1) % 3 - 1];
        final x = [for (var c = 0; c < cols; c++) (c * 11 + 3) % 61 - 30];
        final mults = [for (var r = 0; r < rows; r++) (r % 4) + 1];
        final gold = _golden(rows, cols, w, x, mults, shift);
        final r = await runMatmul(
          rows: rows,
          cols: cols,
          w: w,
          x: x,
          mults: mults,
          shift: shift,
          readAheadWords: 8,
          ternary: true,
        );
        expect(
          r.outputs,
          equals(gold),
          reason: 'multiply-free ternary PE + flash must equal the golden',
        );
      },
    );

    test(
      '16x32 (wordsPerRow=8 = one line/block): golden + coalescing report',
      timeout: const Timeout(Duration(minutes: 4)),
      () async {
        const rows = 16, cols = 32, shift = 5;
        final c = makeCase(rows, cols);
        final gold = _golden(rows, cols, c.w, c.x, c.mults, shift);

        final r8 = await runMatmul(
          rows: rows,
          cols: cols,
          w: c.w,
          x: c.x,
          mults: c.mults,
          shift: shift,
          readAheadWords: 8,
        );
        final r1 = await runMatmul(
          rows: rows,
          cols: cols,
          w: c.w,
          x: c.x,
          mults: c.mults,
          shift: shift,
          readAheadWords: 1,
        );

        expect(
          r8.outputs,
          equals(gold),
          reason: 'flash model must be faithful',
        );
        expect(
          r1.outputs,
          equals(gold),
          reason: 'flash model must be faithful',
        );
      },
    );

    test(
      '16x64 (wordsPerRow=16 = two lines/block): golden + coalescing report',
      timeout: const Timeout(Duration(minutes: 4)),
      () async {
        const rows = 16, cols = 64, shift = 6;
        final c = makeCase(rows, cols);
        final gold = _golden(rows, cols, c.w, c.x, c.mults, shift);

        final r8 = await runMatmul(
          rows: rows,
          cols: cols,
          w: c.w,
          x: c.x,
          mults: c.mults,
          shift: shift,
          readAheadWords: 8,
        );

        expect(
          r8.outputs,
          equals(gold),
          reason: 'flash model must be faithful',
        );
      },
    );
  });
}
