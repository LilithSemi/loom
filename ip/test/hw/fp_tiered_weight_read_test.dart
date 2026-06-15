// Tiered weight-read correctness: LoomFpLinearAccelerator reads two W4A8
// matmuls whose weights + resident scales live in two distinct memory
// regions (BRAM-range 0x30000000, flash-range 0x20200000), selected per
// matrix via WEIGHT_BASE / SCALE_BASE. Proves addressing + read correctness
// across regions; speed is a board-only concern.

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

  test('accelerator reads weights split across BRAM-range + flash-range', () async {
    const rows = 4, cols = 6, peR = 2, peC = 2;
    final rowBlocks = (rows + peR - 1) ~/ peR;
    final colTiles = (cols + peC - 1) ~/ peC;
    final wordsPerRow = (colTiles + 1) ~/ 2;

    // Two independent weight matrices over the same activation vector: A is the
    // "hot" matrix (served from the BRAM range), B is the "cold" matrix (served
    // from the flash range).
    final wA = Float64List(rows * cols);
    final wB = Float64List(rows * cols);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        wA[r * cols + c] = ((r * 7 + c * 3) % 11 - 5) * 0.21;
        wB[r * cols + c] = ((r * 2 + c * 5) % 9 - 4) * 0.17;
      }
    }
    final x = Float64List.fromList([
      for (var c = 0; c < cols; c++) ((c * 5 + 2) % 9 - 4) * 0.37,
    ]);
    final goldA = linear(wA, rows, cols, x);
    final goldB = linear(wB, rows, cols, x);
    final qmA = quantizeRowwiseInt4(wA, rows, cols);
    final qmB = quantizeRowwiseInt4(wB, rows, cols);

    final fp = FloatingPoint16();
    int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();

    // Two backing stores, each addressed from 0. Within a store: weights at
    // offset 0, resident per-row scales (one fp16 per 32-bit word) at 0x200.
    const weightOff = 0x000;
    const scaleOff = 0x200;
    const storeBytes = 0x400;

    Uint8List buildStore(QuantizedMatrix qm) {
      final mem = Uint8List(storeBytes);
      int wq(int gr, int gc) =>
          (gr < rows && gc < cols) ? (qm.values[gr * cols + gc] & 0xF) : 0;
      for (var rb = 0; rb < rowBlocks; rb++) {
        for (var ct = 0; ct < colTiles; ct++) {
          final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
          final tileByte = weightOff + wordByte + (ct.isOdd ? 2 : 0);
          mem[tileByte] =
              wq(rb * peR, ct * peC) | (wq(rb * peR, ct * peC + 1) << 4);
          mem[tileByte + 1] =
              wq(rb * peR + 1, ct * peC) |
              (wq(rb * peR + 1, ct * peC + 1) << 4);
        }
      }
      for (var r = 0; r < rows; r++) {
        final s = e(qm.rowScales[r]);
        mem[scaleOff + r * 4] = s & 0xFF;
        mem[scaleOff + r * 4 + 1] = (s >> 8) & 0xFF;
      }
      return mem;
    }

    // The two distinct SoC memory-map ranges. Weight/scale bases point per
    // matrix into the region that backs it, exactly as the fabric routes them.
    const bramBase = 0x30000000; // hot BRAM weight store
    const flashBase = 0x20200000; // cold flash weight store (flash_weight_base)
    final bramMem = buildStore(qmA);
    final flashMem = buildStore(qmB);

    int memWord(int a) {
      Uint8List? region;
      var off = 0;
      if (a >= bramBase && a + 3 < bramBase + storeBytes) {
        region = bramMem;
        off = a - bramBase;
      } else if (a >= flashBase && a + 3 < flashBase + storeBytes) {
        region = flashMem;
        off = a - flashBase;
      }
      if (region == null) return 0; // unmapped -> read as 0
      return region[off] |
          (region[off + 1] << 8) |
          (region[off + 2] << 16) |
          (region[off + 3] << 24);
    }

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

    final dut = LoomFpLinearAccelerator(maxColTiles: 4, maxRowBlocks: 4);
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
    final outFp = FloatingPoint16();

    for (final s in [cCyc, cStb, cWe, memAck]) {
      s.inject(0);
    }
    cAdr.inject(0);
    cDat.inject(0);
    cSel.inject(0xF);
    memMiso.inject(0);
    reset.inject(1);
    Simulator.setMaxSimTime(16000000);
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

    Future<List<double>> runMatmul(int weightBase, int scaleBase) async {
      await csrWrite(0x004, colTiles);
      await csrWrite(0x008, rowBlocks);
      await csrWrite(0x00C, weightBase); // WEIGHT_BASE -> this store's region
      await csrWrite(
        0x020,
        scaleBase,
      ); // SCALE_BASE  -> resident scales, same region
      for (var c = 0; c < cols; c++) {
        await csrWrite(0x018, e(x[c]));
      }
      await csrWrite(0x010, 0x1); // CONTROL.start
      var g = 0;
      while (g++ < 8000) {
        final status = await csrRead(0x014);
        if (status & 0x2 != 0) break; // done
      }
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
      return got;
    }

    // Matrix A: hot -> read entirely from the BRAM range.
    final gotA = await runMatmul(bramBase + weightOff, bramBase + scaleOff);
    // Matrix B: cold -> read entirely from the flash range.
    final gotB = await runMatmul(flashBase + weightOff, flashBase + scaleOff);
    await Simulator.endSimulation();

    for (var r = 0; r < rows; r++) {
      expect(
        gotA[r],
        closeTo(goldA[r], 0.15 + goldA[r].abs() * 0.2),
        reason: 'BRAM-range row $r: hw=${gotA[r]} fp=${goldA[r]}',
      );
      expect(
        gotB[r],
        closeTo(goldB[r], 0.15 + goldB[r].abs() * 0.2),
        reason: 'flash-range row $r: hw=${gotB[r]} fp=${goldB[r]}',
      );
    }
  });
}
