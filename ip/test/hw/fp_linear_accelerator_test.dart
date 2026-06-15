// LoomFpLinearAccelerator: the host (CSR slave) configures dims/weight_base,
// pushes fp16 activations + per-row fp16 weight scales, strobes start, polls
// done, and reads back fp16 result rows. Weights are served on the 'mem'
// master. End result matches the golden fp64 linear within W4A8 tolerance.
//
// This is the demo path: a model-agnostic host runs a real W4A8 linear on the
// device over a transport.

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:harbor/harbor.dart';
import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/fp_linear_accelerator.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

/// Round [d] to fp32 precision (models one FloatingPointAdderSinglePath /
/// FloatingPointMultiplierSimple rounding step in the hardware pipeline).
double _toF32(double d) => (Float32List(1)..[0] = d)[0];

/// Quantize [x] to symmetric int8 on a CALLER-PROVIDED [scale] (not the
/// block's own max-abs), mirroring the sim's `quantWithScale` (sim.zig): every
/// col-block of a column-tiled matmul must quantize on the SAME shared grid
/// before their partial sums are added, so this uses [scale] directly rather
/// than `quantizePerTensorInt8`'s own per-block max-abs.
QuantizedVector _quantWithSharedScale(Float64List x, double scale) {
  final values = Int8List(x.length);
  for (var i = 0; i < x.length; i++) {
    final v = (x[i] / scale).round();
    values[i] = v < -127 ? -127 : (v > 127 ? 127 : v);
  }
  return QuantizedVector(values: values, scale: scale);
}

/// Per-tensor col-tile golden. Mirrors the accelerator: the shared act-scale
/// (REG_ACT_SCALE / host override) drives BOTH the int8 activation
/// quantization AND the dequant multiplier for every col-block, so every
/// block's activations land on the same int8 grid before their partial sums
/// are added (see sim.zig's `quantWithScale`). Per block: quantize x with the
/// SHARED [sharedActScale] (not the block's own local max-abs), quantize w
/// locally (own int4 row scales), int matmul, fp32 dequant using
/// [sharedActScale], fp32-accumulate across blocks (each stage rounded to
/// fp32, like the real adder/multiplier pipeline), fp16-narrow only once at
/// the very end (LAST).
Float64List goldenColTiled(
  Float64List w,
  int rows,
  int cols,
  int blockCols,
  Float64List x,
  double sharedActScale,
) {
  final numBlocks = (cols + blockCols - 1) ~/ blockCols;
  final acc = Float64List(rows);
  for (var b = 0; b < numBlocks; b++) {
    final bStart = b * blockCols;
    final bLen = (bStart + blockCols <= cols) ? blockCols : cols - bStart;
    final wBlock = Float64List(rows * bLen);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < bLen; c++) {
        wBlock[r * bLen + c] = w[r * cols + bStart + c];
      }
    }
    final xBlock = Float64List.fromList(x.sublist(bStart, bStart + bLen));
    final qm = quantizeRowwiseInt4(wBlock, rows, bLen);
    final qv = _quantWithSharedScale(xBlock, sharedActScale);
    final intAcc = matmulInt(qm, qv);
    for (var r = 0; r < rows; r++) {
      // acc * rowScale (mult1, fp32-rounded), then * sharedActScale (mult2,
      // fp32-rounded) - the two clocked multiplies inside LoomDequant.
      final p1 = _toF32(intAcc[r].toDouble() * qm.rowScales[r]);
      final blockY = _toF32(p1 * sharedActScale);
      acc[r] = b == 0 ? blockY : _toF32(acc[r] + blockY);
    }
  }
  final fp16 = FloatingPoint16();
  return Float64List.fromList([
    for (var r = 0; r < rows; r++)
      fp16.valuePopulator().ofDouble(acc[r]).toDouble(),
  ]);
}

/// fp16-precision-aware closeness check: the recip-approximated on-chip act
/// scale (LoomFpRecip) means the accelerator's int8 activations differ
/// slightly from the golden's exact-division quantization, same tolerance
/// philosophy as every other test in this file (closeTo(.., 0.15 + 0.2*abs)).
bool fp16Close(double got, double golden) =>
    (got - golden).abs() <= 0.15 + golden.abs() * 0.2;

/// Per-GROUP col-tile golden: mirrors `quantizedLinearGroupwise`
/// (`quantizeGroupwise`/`matmulIntGroupwise`/`dequantGroupwise`) but
/// column-tiled and fp32-accumulated across blocks, like `goldenColTiled`
/// above. One group == one col-block (groupSize == blockCols): block/group g
/// dequants on ITS OWN per-row scale (`qm.scales[r*gpr+g]`), not a scale
/// shared across every block. The act-scale IS still shared across every
/// block (same reasoning as `goldenColTiled`: every block must quantize its
/// activation slice on the SAME grid). Per block: quantize x with the shared
/// [sharedActScale], take group g's own weight scale, int matmul, fp32
/// dequant, fp32-accumulate across blocks (each stage fp32-rounded, like the
/// real adder/multiplier pipeline), fp16-narrow only once at the very end
/// (LAST).
Float64List goldenGroupwiseColTiled(
  Float64List w,
  int rows,
  int cols,
  int blockCols,
  Float64List x,
) {
  final sharedActScale = quantizePerTensorInt8(x).scale;
  final qm = quantizeGroupwise(w, rows, cols, bits: 4, groupSize: blockCols);
  final gpr = qm.groupsPerRow;
  final acc = Float64List(rows);
  for (var g = 0; g < gpr; g++) {
    final gStart = g * blockCols;
    final gLen = (gStart + blockCols <= cols) ? blockCols : cols - gStart;
    final xBlock = Float64List.fromList(x.sublist(gStart, gStart + gLen));
    final qv = _quantWithSharedScale(xBlock, sharedActScale);
    for (var r = 0; r < rows; r++) {
      var sum = 0;
      for (var c = 0; c < gLen; c++) {
        sum += qm.values[r * cols + gStart + c] * qv.values[c];
      }
      final rowScale = qm.scales[r * gpr + g];
      // acc * rowScale (mult1, fp32-rounded), then * sharedActScale (mult2,
      // fp32-rounded) - the two clocked multiplies inside LoomDequant.
      final p1 = _toF32(sum.toDouble() * rowScale);
      final blockY = _toF32(p1 * sharedActScale);
      acc[r] = g == 0 ? blockY : _toF32(acc[r] + blockY);
    }
  }
  final fp16 = FloatingPoint16();
  return Float64List.fromList([
    for (var r = 0; r < rows; r++)
      fp16.valuePopulator().ofDouble(acc[r]).toDouble(),
  ]);
}

/// Drive the accelerator through a col-tiled matmul: [cols] split into
/// [blockCols]-wide blocks, each provisioned + run as a SEPARATE device op
/// (block 0 = COLTILE_FIRST, the last block = COLTILE_LAST, any blocks in
/// between = COLTILE_MID), sharing ONE REG_ACT_SCALE write across all of
/// them. One DUT/Simulator for the whole run (the fp32 accumulator BRAM must
/// persist across blocks). Returns the host-read fp16 result rows after LAST.
Future<List<double>> runColTiled({
  required int rows,
  required int cols,
  required int blockCols,
  required Float64List w,
  required Float64List x,
  required int sharedActScaleBits,
}) async {
  await Simulator.reset();
  const peR = 2, peC = 2;
  final rowBlocks = (rows + peR - 1) ~/ peR;
  final blockColTiles = (blockCols + peC - 1) ~/ peC;
  final wordsPerRow = (blockColTiles + 1) ~/ 2;
  final numBlocks = (cols + blockCols - 1) ~/ blockCols;

  final mem = List<int>.filled(0x2000, 0);
  final blockBase = [for (var b = 0; b < numBlocks; b++) b * 0x200];
  final qms = <QuantizedMatrix>[];
  for (var b = 0; b < numBlocks; b++) {
    final bStart = b * blockCols;
    final bLen = (bStart + blockCols <= cols) ? blockCols : cols - bStart;
    final wBlock = Float64List(rows * bLen);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < bLen; c++) {
        wBlock[r * bLen + c] = w[r * cols + bStart + c];
      }
    }
    final qm = quantizeRowwiseInt4(wBlock, rows, bLen);
    qms.add(qm);
    int wq(int gr, int gc) =>
        (gr < rows && gc < bLen) ? (qm.values[gr * bLen + gc] & 0xF) : 0;
    for (var rb = 0; rb < rowBlocks; rb++) {
      for (var ct = 0; ct < blockColTiles; ct++) {
        final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
        final tileByte = wordByte + (ct.isOdd ? 2 : 0);
        mem[blockBase[b] + tileByte] =
            wq(rb * peR, ct * peC) | (wq(rb * peR, ct * peC + 1) << 4);
        mem[blockBase[b] + tileByte + 1] =
            wq(rb * peR + 1, ct * peC) | (wq(rb * peR + 1, ct * peC + 1) << 4);
      }
    }
  }
  int memWord(int a) => (a < 0 || a + 3 >= mem.length)
      ? 0
      : mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);

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

  final dut = LoomFpLinearAccelerator(maxColTiles: 32, maxRowBlocks: 4);
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
  Simulator.setMaxSimTime(8000000);
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

  // Shared act-scale: written ONCE, persists (no CONTROL.start reset) across
  // every block's run below.
  await csrWrite(0x02C, sharedActScaleBits);

  Future<void> runBlock(int mode, int b) async {
    final bStart = b * blockCols;
    final bLen = (bStart + blockCols <= cols) ? blockCols : cols - bStart;
    await csrWrite(0x004, blockColTiles);
    await csrWrite(0x008, rowBlocks);
    await csrWrite(0x00C, blockBase[b]);
    await csrWrite(0x028, mode); // MODE
    for (var c = 0; c < bLen; c++) {
      await csrWrite(0x018, e(x[bStart + c]));
    }
    for (var r = 0; r < rows; r++) {
      await csrWrite(0x01C, e(qms[b].rowScales[r]));
    }
    await csrWrite(0x010, 0x1); // CONTROL.start
    var g = 0;
    while (g++ < 8000) {
      final status = await csrRead(0x014);
      if (status & 0x2 != 0) break; // done
    }
  }

  for (var b = 0; b < numBlocks; b++) {
    final mode = b == 0 ? 3 : (b == numBlocks - 1 ? 5 : 4); // FIRST/MID/LAST
    await runBlock(mode, b);
  }

  final got = <double>[];
  for (var wi = 0; wi < (rows + 1) ~/ 2; wi++) {
    final word = await csrRead(0x100 + wi * 4);
    outFp.put(LogicValue.ofInt(word & 0xFFFF, 16));
    got.add(outFp.floatingPointValue.toDouble());
    if (2 * wi + 1 < rows) {
      outFp.put(LogicValue.ofInt((word >> 16) & 0xFFFF, 16));
      got.add(outFp.floatingPointValue.toDouble());
    }
  }
  await Simulator.endSimulation();
  return got;
}

/// Drive the accelerator through a PER-GROUP col-tiled matmul: same
/// FIRST/MID/LAST device-op split as [runColTiled], but the per-row weight
/// scales are RESIDENT in flash (SCALE_BASE, one write) laid out GROUP-MAJOR
/// (group g's `rows` fp16 scales at `scaleBase + g*rows*4`), and each block
/// writes its own REG_SCALE_GROUP_OFF (0x030 = b*rows*4) right before it
/// starts, so the scale-fetch FSM loads GROUP b's own scales (scaleBaseReg +
/// groupOffReg) instead of the host pushing them per block. Weights are
/// quantized ONCE as a whole via `quantizeGroupwise` (groupSize == blockCols),
/// so group g's values/scale are exactly the group this block's weight tile
/// is packed from. Returns the host-read fp16 result rows after LAST.
Future<List<double>> runColTiledGrouped({
  required int rows,
  required int cols,
  required int blockCols,
  required Float64List w,
  required Float64List x,
}) async {
  await Simulator.reset();
  const peR = 2, peC = 2;
  final rowBlocks = (rows + peR - 1) ~/ peR;
  final blockColTiles = (blockCols + peC - 1) ~/ peC;
  final wordsPerRow = (blockColTiles + 1) ~/ 2;
  final numBlocks = (cols + blockCols - 1) ~/ blockCols;

  final qm = quantizeGroupwise(w, rows, cols, bits: 4, groupSize: blockCols);
  final gpr = qm.groupsPerRow;

  final blockBase = [for (var b = 0; b < numBlocks; b++) b * 0x200];
  final scaleBase = numBlocks * 0x200;
  final mem = List<int>.filled(scaleBase + gpr * rows * 4 + 0x100, 0);

  int wq(int b, int gr, int gc) {
    final bStart = b * blockCols;
    final bLen = (bStart + blockCols <= cols) ? blockCols : cols - bStart;
    if (gr >= rows || gc >= bLen) return 0;
    return qm.values[gr * cols + bStart + gc] & 0xF;
  }

  for (var b = 0; b < numBlocks; b++) {
    for (var rb = 0; rb < rowBlocks; rb++) {
      for (var ct = 0; ct < blockColTiles; ct++) {
        final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
        final tileByte = wordByte + (ct.isOdd ? 2 : 0);
        mem[blockBase[b] + tileByte] =
            wq(b, rb * peR, ct * peC) | (wq(b, rb * peR, ct * peC + 1) << 4);
        mem[blockBase[b] + tileByte + 1] =
            wq(b, rb * peR + 1, ct * peC) |
            (wq(b, rb * peR + 1, ct * peC + 1) << 4);
      }
    }
  }

  final fp = FloatingPoint16();
  int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
  // Lay each group's per-row fp16 scales into flash, GROUP-MAJOR: group g's
  // rows*4 bytes start at scaleBase + g*rows*4 (low16 of each 32-bit word).
  for (var g = 0; g < gpr; g++) {
    for (var r = 0; r < rows; r++) {
      final s = e(qm.scales[r * gpr + g]);
      final addr = scaleBase + g * rows * 4 + r * 4;
      mem[addr] = s & 0xFF;
      mem[addr + 1] = (s >> 8) & 0xFF;
    }
  }
  int memWord(int a) => (a < 0 || a + 3 >= mem.length)
      ? 0
      : mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);

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

  final dut = LoomFpLinearAccelerator(maxColTiles: 32, maxRowBlocks: 4);
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
  Simulator.setMaxSimTime(8000000);
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

  // Shared act-scale + resident SCALE_BASE: each written ONCE, persisting (no
  // CONTROL.start reset) across every block's run below.
  final sharedActScaleExact = quantizePerTensorInt8(x).scale;
  await csrWrite(0x02C, e(sharedActScaleExact));
  await csrWrite(0x020, scaleBase);

  Future<void> runBlock(int mode, int b) async {
    final bStart = b * blockCols;
    final bLen = (bStart + blockCols <= cols) ? blockCols : cols - bStart;
    await csrWrite(0x004, blockColTiles);
    await csrWrite(0x008, rowBlocks);
    await csrWrite(0x00C, blockBase[b]);
    await csrWrite(0x028, mode); // MODE
    // This block's own group: scaleAddr = SCALE_BASE + SCALE_GROUP_OFF.
    await csrWrite(0x030, b * rows * 4);
    for (var c = 0; c < bLen; c++) {
      await csrWrite(0x018, e(x[bStart + c]));
    }
    // NO scale pushes: the resident scale-fetch FSM loads this block's group.
    await csrWrite(0x010, 0x1); // CONTROL.start
    var g = 0;
    while (g++ < 8000) {
      final status = await csrRead(0x014);
      if (status & 0x2 != 0) break; // done
    }
  }

  for (var b = 0; b < numBlocks; b++) {
    final mode = b == 0 ? 3 : (b == numBlocks - 1 ? 5 : 4); // FIRST/MID/LAST
    await runBlock(mode, b);
  }

  final got = <double>[];
  for (var wi = 0; wi < (rows + 1) ~/ 2; wi++) {
    final word = await csrRead(0x100 + wi * 4);
    outFp.put(LogicValue.ofInt(word & 0xFFFF, 16));
    got.add(outFp.floatingPointValue.toDouble());
    if (2 * wi + 1 < rows) {
      outFp.put(LogicValue.ofInt((word >> 16) & 0xFFFF, 16));
      got.add(outFp.floatingPointValue.toDouble());
    }
  }
  await Simulator.endSimulation();
  return got;
}

/// Same as [runColTiledGrouped] but weights AND resident group scales come from
/// the REAL HarborSpiFlashController (read-ahead 8), the board config. Isolates a
/// col-tile + resident-scale + flash-read-ahead interaction from a 1-cycle mem.
Future<List<double>> runColTiledGroupedFlash({
  required int rows,
  required int cols,
  required int blockCols,
  required Float64List w,
  required Float64List x,
}) async {
  await Simulator.reset();
  const peR = 2, peC = 2;
  final rowBlocks = (rows + peR - 1) ~/ peR;
  final blockColTiles = (blockCols + peC - 1) ~/ peC;
  final wordsPerRow = (blockColTiles + 1) ~/ 2;
  final numBlocks = (cols + blockCols - 1) ~/ blockCols;

  final qm = quantizeGroupwise(w, rows, cols, bits: 4, groupSize: blockCols);
  final gpr = qm.groupsPerRow;
  final blockBase = [for (var b = 0; b < numBlocks; b++) b * 0x200];
  final scaleBase = numBlocks * 0x200;
  final mem = List<int>.filled(scaleBase + gpr * rows * 4 + 0x100, 0);

  int wq(int b, int gr, int gc) {
    final bStart = b * blockCols;
    final bLen = (bStart + blockCols <= cols) ? blockCols : cols - bStart;
    if (gr >= rows || gc >= bLen) return 0;
    return qm.values[gr * cols + bStart + gc] & 0xF;
  }

  for (var b = 0; b < numBlocks; b++) {
    for (var rb = 0; rb < rowBlocks; rb++) {
      for (var ct = 0; ct < blockColTiles; ct++) {
        final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
        final tileByte = wordByte + (ct.isOdd ? 2 : 0);
        mem[blockBase[b] + tileByte] =
            wq(b, rb * peR, ct * peC) | (wq(b, rb * peR, ct * peC + 1) << 4);
        mem[blockBase[b] + tileByte + 1] =
            wq(b, rb * peR + 1, ct * peC) |
            (wq(b, rb * peR + 1, ct * peC + 1) << 4);
      }
    }
  }
  final fp = FloatingPoint16();
  int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
  for (var g = 0; g < gpr; g++) {
    for (var r = 0; r < rows; r++) {
      final s = e(qm.scales[r * gpr + g]);
      final addr = scaleBase + g * rows * 4 + r * 4;
      mem[addr] = s & 0xFF;
      mem[addr + 1] = (s >> 8) & 0xFF;
    }
  }

  final clk = SimpleClockGenerator(10).clk;
  final reset = Logic(name: 'reset');
  final cCyc = Logic(name: 'c_cyc');
  final cStb = Logic(name: 'c_stb');
  final cWe = Logic(name: 'c_we');
  final cAdr = Logic(name: 'c_adr', width: 32);
  final cDat = Logic(name: 'c_dat', width: 32);
  final cSel = Logic(name: 'c_sel', width: 4);
  final miso = Logic(name: 'miso');

  final dut = LoomFpLinearAccelerator(maxColTiles: 32, maxRowBlocks: 4);
  final flash = HarborSpiFlashController(
    config: HarborSpiFlashConfig(
      size: 1024 * 1024,
      mode: HarborSpiFlashMode.standard,
      readCommand: 0x03,
      addressBytes: 3,
      dummyCycles: 0,
      readAheadWords: 8,
    ),
    baseAddress: 0x0,
    busAddressWidth: 32,
    busDataWidth: 32,
  );
  dut.input('clk').srcConnection! <= clk;
  dut.input('reset').srcConnection! <= reset;
  dut.input('bus_CYC').srcConnection! <= cCyc;
  dut.input('bus_STB').srcConnection! <= cStb;
  dut.input('bus_WE').srcConnection! <= cWe;
  dut.input('bus_ADR').srcConnection! <= cAdr;
  dut.input('bus_DAT_MOSI').srcConnection! <= cDat;
  dut.input('bus_SEL').srcConnection! <= cSel;
  flash.input('bus_CYC').srcConnection! <= dut.output('mem_CYC');
  flash.input('bus_STB').srcConnection! <= dut.output('mem_STB');
  flash.input('bus_WE').srcConnection! <= dut.output('mem_WE');
  flash.input('bus_ADR').srcConnection! <= dut.output('mem_ADR');
  flash.input('bus_DAT_MOSI').srcConnection! <= dut.output('mem_DAT_MOSI');
  flash.input('bus_SEL').srcConnection! <= dut.output('mem_SEL');
  dut.input('mem_ACK').srcConnection! <= flash.output('bus_ACK');
  dut.input('mem_DAT_MISO').srcConnection! <= flash.output('bus_DAT_MISO');
  flash.input('clk').srcConnection! <= clk;
  flash.input('reset').srcConnection! <= reset;
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
  final outFp = FloatingPoint16();

  const cmdClocks = 8, dataStart = 32;
  var prevClk = 0, prevCs = 1, riseCount = 0, decodedAddr = 0;
  void spiModel() {
    final cs = csN.value.toInt();
    final sc = spiClk.value.toInt();
    if (cs == 1) {
      riseCount = 0;
    } else {
      if (prevCs == 1) {
        riseCount = 0;
        decodedAddr = 0;
      }
      if (sc == 1 && prevClk == 0) {
        if (riseCount >= cmdClocks && riseCount < dataStart) {
          decodedAddr = (decodedAddr << 1) | (mosi.value.toInt() & 1);
        } else if (riseCount >= dataStart) {
          final idx = riseCount - dataStart;
          final byteOff = idx >> 3;
          final bit = 7 - (idx & 7);
          final a = decodedAddr + byteOff;
          miso.inject(((a >= 0 && a < mem.length ? mem[a] : 0) >> bit) & 1);
        }
        riseCount++;
      }
    }
    prevClk = sc;
    prevCs = cs;
  }

  for (final s in [cCyc, cStb, cWe]) {
    s.inject(0);
  }
  cAdr.inject(0);
  cDat.inject(0);
  cSel.inject(0xF);
  miso.inject(0);
  reset.inject(1);
  Simulator.setMaxSimTime(400000000);
  unawaited(Simulator.run());
  await clk.nextPosedge;
  await clk.nextPosedge;
  reset.inject(0);
  await clk.nextPosedge;

  Future<void> tick() async {
    await clk.nextNegedge;
    spiModel();
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
    } while (!o('bus_ACK').value.toBool() && g++ < 5000);
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
    } while (!o('bus_ACK').value.toBool() && g++ < 5000);
    final v = o('bus_DAT_MISO').value.toInt();
    cCyc.inject(0);
    cStb.inject(0);
    await tick();
    return v;
  }

  final sharedActScaleExact = quantizePerTensorInt8(x).scale;
  await csrWrite(0x02C, e(sharedActScaleExact));
  await csrWrite(0x020, scaleBase);

  Future<void> runBlock(int mode, int b) async {
    final bStart = b * blockCols;
    final bLen = (bStart + blockCols <= cols) ? blockCols : cols - bStart;
    await csrWrite(0x004, blockColTiles);
    await csrWrite(0x008, rowBlocks);
    await csrWrite(0x00C, blockBase[b]);
    await csrWrite(0x028, mode);
    await csrWrite(0x030, b * rows * 4);
    for (var c = 0; c < bLen; c++) {
      await csrWrite(0x018, e(x[bStart + c]));
    }
    await csrWrite(0x010, 0x1);
    var g = 0;
    while (g++ < 500000) {
      if (await csrRead(0x014) & 0x2 != 0) break;
    }
  }

  for (var b = 0; b < numBlocks; b++) {
    final mode = b == 0 ? 3 : (b == numBlocks - 1 ? 5 : 4);
    await runBlock(mode, b);
  }

  final got = <double>[];
  for (var wi = 0; wi < (rows + 1) ~/ 2; wi++) {
    final word = await csrRead(0x100 + wi * 4);
    outFp.put(LogicValue.ofInt(word & 0xFFFF, 16));
    got.add(outFp.floatingPointValue.toDouble());
    if (2 * wi + 1 < rows) {
      outFp.put(LogicValue.ofInt((word >> 16) & 0xFFFF, 16));
      got.add(outFp.floatingPointValue.toDouble());
    }
  }
  await Simulator.endSimulation();
  return got;
}

/// [runColTiledGroupedFlash] PLUS ROW-TILING: maxRowBlocks is small so rows >
/// maxRows, and the host drives row-tiles OUTER / col-blocks INNER exactly like
/// the runtime (linear.zig linearColTiled), REUSING the maxRows accumulator bank
/// per row-tile. This is the board's real shape (q_proj = 8 col-blocks x 8 row-
/// tiles) which no other ROHD test exercises. maxRb = row-blocks per device call.
Future<List<double>> runColTiledGroupedRowTiledFlash({
  required int rows,
  required int cols,
  required int blockCols,
  required int maxRowBlocks,
  required Float64List w,
  required Float64List x,
}) async {
  await Simulator.reset();
  const peR = 2, peC = 2;
  final totalRb = (rows + peR - 1) ~/ peR;
  final blockColTiles = (blockCols + peC - 1) ~/ peC;
  final wordsPerRow = (blockColTiles + 1) ~/ 2;
  final numBlocks = (cols + blockCols - 1) ~/ blockCols;

  final qm = quantizeGroupwise(w, rows, cols, bits: 4, groupSize: blockCols);
  final gpr = qm.groupsPerRow;
  // Each block's weights span ALL row-blocks, tile-major. Blocks are consecutive.
  final blockStride = totalRb * wordsPerRow * 4;
  final blockBase = [for (var b = 0; b < numBlocks; b++) b * blockStride];
  final scaleBase = numBlocks * blockStride + 0x40;
  final mem = List<int>.filled(scaleBase + gpr * rows * 4 + 0x100, 0);

  int wq(int b, int gr, int gc) {
    final bStart = b * blockCols;
    final bLen = (bStart + blockCols <= cols) ? blockCols : cols - bStart;
    if (gr >= rows || gc >= bLen) return 0;
    return qm.values[gr * cols + bStart + gc] & 0xF;
  }

  for (var b = 0; b < numBlocks; b++) {
    for (var rb = 0; rb < totalRb; rb++) {
      for (var ct = 0; ct < blockColTiles; ct++) {
        final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
        final tileByte = wordByte + (ct.isOdd ? 2 : 0);
        mem[blockBase[b] + tileByte] =
            wq(b, rb * peR, ct * peC) | (wq(b, rb * peR, ct * peC + 1) << 4);
        mem[blockBase[b] + tileByte + 1] =
            wq(b, rb * peR + 1, ct * peC) |
            (wq(b, rb * peR + 1, ct * peC + 1) << 4);
      }
    }
  }
  final fp = FloatingPoint16();
  int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
  for (var g = 0; g < gpr; g++) {
    for (var r = 0; r < rows; r++) {
      final s = e(qm.scales[r * gpr + g]);
      final addr = scaleBase + g * rows * 4 + r * 4;
      mem[addr] = s & 0xFF;
      mem[addr + 1] = (s >> 8) & 0xFF;
    }
  }

  final clk = SimpleClockGenerator(10).clk;
  final reset = Logic(name: 'reset');
  final cCyc = Logic(name: 'c_cyc');
  final cStb = Logic(name: 'c_stb');
  final cWe = Logic(name: 'c_we');
  final cAdr = Logic(name: 'c_adr', width: 32);
  final cDat = Logic(name: 'c_dat', width: 32);
  final cSel = Logic(name: 'c_sel', width: 4);
  final miso = Logic(name: 'miso');

  final dut = LoomFpLinearAccelerator(
    maxColTiles: 32,
    maxRowBlocks: maxRowBlocks,
  );
  final flash = HarborSpiFlashController(
    config: HarborSpiFlashConfig(
      size: 1024 * 1024,
      mode: HarborSpiFlashMode.standard,
      readCommand: 0x03,
      addressBytes: 3,
      dummyCycles: 0,
      readAheadWords: 8,
    ),
    baseAddress: 0x0,
    busAddressWidth: 32,
    busDataWidth: 32,
  );
  dut.input('clk').srcConnection! <= clk;
  dut.input('reset').srcConnection! <= reset;
  dut.input('bus_CYC').srcConnection! <= cCyc;
  dut.input('bus_STB').srcConnection! <= cStb;
  dut.input('bus_WE').srcConnection! <= cWe;
  dut.input('bus_ADR').srcConnection! <= cAdr;
  dut.input('bus_DAT_MOSI').srcConnection! <= cDat;
  dut.input('bus_SEL').srcConnection! <= cSel;
  flash.input('bus_CYC').srcConnection! <= dut.output('mem_CYC');
  flash.input('bus_STB').srcConnection! <= dut.output('mem_STB');
  flash.input('bus_WE').srcConnection! <= dut.output('mem_WE');
  flash.input('bus_ADR').srcConnection! <= dut.output('mem_ADR');
  flash.input('bus_DAT_MOSI').srcConnection! <= dut.output('mem_DAT_MOSI');
  flash.input('bus_SEL').srcConnection! <= dut.output('mem_SEL');
  dut.input('mem_ACK').srcConnection! <= flash.output('bus_ACK');
  dut.input('mem_DAT_MISO').srcConnection! <= flash.output('bus_DAT_MISO');
  flash.input('clk').srcConnection! <= clk;
  flash.input('reset').srcConnection! <= reset;
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
  final outFp = FloatingPoint16();

  const cmdClocks = 8, dataStart = 32;
  var prevClk = 0, prevCs = 1, riseCount = 0, decodedAddr = 0;
  void spiModel() {
    final cs = csN.value.toInt();
    final sc = spiClk.value.toInt();
    if (cs == 1) {
      riseCount = 0;
    } else {
      if (prevCs == 1) {
        riseCount = 0;
        decodedAddr = 0;
      }
      if (sc == 1 && prevClk == 0) {
        if (riseCount >= cmdClocks && riseCount < dataStart) {
          decodedAddr = (decodedAddr << 1) | (mosi.value.toInt() & 1);
        } else if (riseCount >= dataStart) {
          final idx = riseCount - dataStart;
          final byteOff = idx >> 3;
          final bit = 7 - (idx & 7);
          final a = decodedAddr + byteOff;
          miso.inject(((a >= 0 && a < mem.length ? mem[a] : 0) >> bit) & 1);
        }
        riseCount++;
      }
    }
    prevClk = sc;
    prevCs = cs;
  }

  for (final s in [cCyc, cStb, cWe]) {
    s.inject(0);
  }
  cAdr.inject(0);
  cDat.inject(0);
  cSel.inject(0xF);
  miso.inject(0);
  reset.inject(1);
  Simulator.setMaxSimTime(800000000);
  unawaited(Simulator.run());
  await clk.nextPosedge;
  await clk.nextPosedge;
  reset.inject(0);
  await clk.nextPosedge;

  Future<void> tick() async {
    await clk.nextNegedge;
    spiModel();
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
    } while (!o('bus_ACK').value.toBool() && g++ < 5000);
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
    } while (!o('bus_ACK').value.toBool() && g++ < 5000);
    final v = o('bus_DAT_MISO').value.toInt();
    cCyc.inject(0);
    cStb.inject(0);
    await tick();
    return v;
  }

  final sharedActScaleExact = quantizePerTensorInt8(x).scale;
  await csrWrite(0x02C, e(sharedActScaleExact));
  await csrWrite(0x020, scaleBase);

  final got = List<double>.filled(rows, 0);
  // ROW-TILES OUTER, COL-BLOCKS INNER (mirror linear.zig linearColTiled).
  for (var rbStart = 0; rbStart < totalRb; rbStart += maxRowBlocks) {
    final chunkRb = (maxRowBlocks < totalRb - rbStart)
        ? maxRowBlocks
        : totalRb - rbStart;
    final rowStart = rbStart * peR;
    for (var b = 0; b < numBlocks; b++) {
      final mode = b == 0 ? 3 : (b == numBlocks - 1 ? 5 : 4);
      final bStart = b * blockCols;
      final bLen = (bStart + blockCols <= cols) ? blockCols : cols - bStart;
      await csrWrite(0x004, blockColTiles);
      await csrWrite(0x008, chunkRb);
      await csrWrite(0x00C, blockBase[b] + rbStart * wordsPerRow * 4);
      await csrWrite(0x028, mode);
      // group b's scales for THIS row-tile: scaleBase + b*rows*4 + rowStart*4.
      await csrWrite(0x030, b * rows * 4 + rowStart * 4);
      for (var c = 0; c < bLen; c++) {
        await csrWrite(0x018, e(x[bStart + c]));
      }
      await csrWrite(0x010, 0x1);
      var g = 0;
      while (g++ < 500000) {
        if (await csrRead(0x014) & 0x2 != 0) break;
      }
    }
    // Read this row-tile's results (packed) after its LAST block.
    final chunkRows = chunkRb * peR;
    for (var wi = 0; wi < (chunkRows + 1) ~/ 2; wi++) {
      final word = await csrRead(0x100 + wi * 4);
      final r0 = rowStart + 2 * wi;
      final r1 = rowStart + 2 * wi + 1;
      outFp.put(LogicValue.ofInt(word & 0xFFFF, 16));
      if (r0 < rows) got[r0] = outFp.floatingPointValue.toDouble();
      outFp.put(LogicValue.ofInt((word >> 16) & 0xFFFF, 16));
      if (r1 < rows) got[r1] = outFp.floatingPointValue.toDouble();
    }
  }
  await Simulator.endSimulation();
  return got;
}

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  // The full fp accelerator reads BOTH int4 weights AND resident per-row
  // scales from the REAL HarborSpiFlashController (read-ahead 8), the board
  // config. rows/cols chosen so weights (32 words) and scales (16 words) each
  // span multiple 8-word read-ahead lines.
  test(
    'fp accelerator + REAL flash (resident scales, read-ahead): golden',
    timeout: const Timeout(Duration(minutes: 8)),
    () async {
      const rows = 16, cols = 16, peR = 2, peC = 2;
      final rowBlocks = (rows + peR - 1) ~/ peR; // 8
      final colTiles = (cols + peC - 1) ~/ peC; // 8
      final wordsPerRow = (colTiles + 1) ~/ 2; // 4
      final weightWords = wordsPerRow * rowBlocks; // 32

      final w = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          w[r * cols + c] = ((r * 5 + c * 3 + r * c) % 11 - 5) * 0.17;
        }
      }
      final x = Float64List.fromList([
        for (var c = 0; c < cols; c++) ((c * 7 + 2) % 13 - 6) * 0.23,
      ]);
      final goldenFp = linear(w, rows, cols, x);
      final qm = quantizeRowwiseInt4(w, rows, cols);

      const weightBase = 0x000;
      final scaleBase =
          weightWords * 4; // scales right after the weights in flash
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
              wq(rb * peR + 1, ct * peC) |
              (wq(rb * peR + 1, ct * peC + 1) << 4);
        }
      }
      final fp = FloatingPoint16();
      int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
      for (var r = 0; r < rows; r++) {
        final s = e(qm.rowScales[r]);
        mem[scaleBase + r * 4] = s & 0xFF;
        mem[scaleBase + r * 4 + 1] = (s >> 8) & 0xFF;
      }

      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      final cCyc = Logic(name: 'c_cyc');
      final cStb = Logic(name: 'c_stb');
      final cWe = Logic(name: 'c_we');
      final cAdr = Logic(name: 'c_adr', width: 32);
      final cDat = Logic(name: 'c_dat', width: 32);
      final cSel = Logic(name: 'c_sel', width: 4);
      final miso = Logic(name: 'miso');

      final dut = LoomFpLinearAccelerator(maxColTiles: 8, maxRowBlocks: 8);
      final flash = HarborSpiFlashController(
        config: HarborSpiFlashConfig(
          size: 1024 * 1024,
          mode: HarborSpiFlashMode.standard,
          readCommand: 0x03,
          addressBytes: 3,
          dummyCycles: 0,
          readAheadWords: 8,
        ),
        baseAddress: 0x0,
        busAddressWidth: 32,
        busDataWidth: 32,
      );

      dut.input('clk').srcConnection! <= clk;
      dut.input('reset').srcConnection! <= reset;
      dut.input('bus_CYC').srcConnection! <= cCyc;
      dut.input('bus_STB').srcConnection! <= cStb;
      dut.input('bus_WE').srcConnection! <= cWe;
      dut.input('bus_ADR').srcConnection! <= cAdr;
      dut.input('bus_DAT_MOSI').srcConnection! <= cDat;
      dut.input('bus_SEL').srcConnection! <= cSel;
      // accelerator 'mem' MASTER -> flash SLAVE, and back.
      flash.input('bus_CYC').srcConnection! <= dut.output('mem_CYC');
      flash.input('bus_STB').srcConnection! <= dut.output('mem_STB');
      flash.input('bus_WE').srcConnection! <= dut.output('mem_WE');
      flash.input('bus_ADR').srcConnection! <= dut.output('mem_ADR');
      flash.input('bus_DAT_MOSI').srcConnection! <= dut.output('mem_DAT_MOSI');
      flash.input('bus_SEL').srcConnection! <= dut.output('mem_SEL');
      dut.input('mem_ACK').srcConnection! <= flash.output('bus_ACK');
      dut.input('mem_DAT_MISO').srcConnection! <= flash.output('bus_DAT_MISO');
      flash.input('clk').srcConnection! <= clk;
      flash.input('reset').srcConnection! <= reset;
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
      final outFp = FloatingPoint16();

      const cmdClocks = 8, dataStart = 32;
      var prevClk = 0, prevCs = 1, riseCount = 0, decodedAddr = 0;
      void spiModel() {
        final cs = csN.value.toInt();
        final sc = spiClk.value.toInt();
        if (cs == 1) {
          riseCount = 0;
        } else {
          if (prevCs == 1) {
            riseCount = 0;
            decodedAddr = 0;
          }
          if (sc == 1 && prevClk == 0) {
            if (riseCount >= cmdClocks && riseCount < dataStart) {
              decodedAddr = (decodedAddr << 1) | (mosi.value.toInt() & 1);
            } else if (riseCount >= dataStart) {
              final idx = riseCount - dataStart;
              final byteOff = idx >> 3;
              final bit = 7 - (idx & 7);
              final a = decodedAddr + byteOff;
              miso.inject(((a >= 0 && a < mem.length ? mem[a] : 0) >> bit) & 1);
            }
            riseCount++;
          }
        }
        prevClk = sc;
        prevCs = cs;
      }

      for (final s in [cCyc, cStb, cWe]) {
        s.inject(0);
      }
      cAdr.inject(0);
      cDat.inject(0);
      cSel.inject(0xF);
      miso.inject(0);
      reset.inject(1);
      Simulator.setMaxSimTime(200000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;

      Future<void> tick() async {
        await clk.nextNegedge;
        spiModel();
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
        } while (!o('bus_ACK').value.toBool() && g++ < 5000);
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
        } while (!o('bus_ACK').value.toBool() && g++ < 5000);
        final v = o('bus_DAT_MISO').value.toInt();
        cCyc.inject(0);
        cStb.inject(0);
        await tick();
        return v;
      }

      expect(await csrRead(0x000), equals(0x4C4F4F4D)); // VERSION
      await csrWrite(0x004, colTiles);
      await csrWrite(0x008, rowBlocks);
      await csrWrite(0x00C, weightBase);
      await csrWrite(0x020, scaleBase); // resident scales in flash
      for (var c = 0; c < cols; c++) {
        await csrWrite(0x018, e(x[c]));
      }
      await csrWrite(0x010, 0x1); // start
      var g = 0;
      while (g++ < 500000) {
        if (await csrRead(0x014) & 0x2 != 0) break; // done
      }
      final got = <double>[];
      for (var wI = 0; wI < (rows + 1) ~/ 2; wI++) {
        final word = await csrRead(0x100 + wI * 4);
        outFp.put(LogicValue.ofInt(word & 0xFFFF, 16));
        got.add(outFp.floatingPointValue.toDouble());
        if (2 * wI + 1 < rows) {
          outFp.put(LogicValue.ofInt((word >> 16) & 0xFFFF, 16));
          got.add(outFp.floatingPointValue.toDouble());
        }
      }
      await Simulator.endSimulation();
      for (var r = 0; r < rows; r++) {}
      for (var r = 0; r < rows; r++) {
        expect(
          fp16Close(got[r], goldenFp[r]),
          isTrue,
          reason: 'row $r: hw=${got[r]} golden=${goldenFp[r]}',
        );
      }
    },
  );

  test(
    'host drives a memory-backed fp16 W4A8 linear over the CSR bus',
    () async {
      const rows = 4, cols = 6, peR = 2, peC = 2;
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
      Simulator.setMaxSimTime(8000000);
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

      expect(await csrRead(0x000), equals(0x4C4F4F4D)); // VERSION

      await csrWrite(0x004, colTiles);
      await csrWrite(0x008, rowBlocks);
      await csrWrite(0x00C, weightBase);
      // Push fp16 activations, then fp16 per-row weight scales.
      for (var c = 0; c < cols; c++) {
        await csrWrite(0x018, e(x[c]));
      }
      for (var r = 0; r < rows; r++) {
        await csrWrite(0x01C, e(qm.rowScales[r]));
      }
      await csrWrite(0x010, 0x1); // CONTROL.start

      var g = 0;
      while (g++ < 8000) {
        final status = await csrRead(0x014);
        if (status & 0x2 != 0) break; // done
      }

      final got = <double>[];
      // Packed results: word w holds {row 2w+1 high16, row 2w low16}.
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
          closeTo(goldenFp[r], 0.15 + goldenFp[r].abs() * 0.2),
          reason: 'row $r: hw=${got[r]} fp=${goldenFp[r]}',
        );
      }
    },
  );

  test(
    'resident scales: accelerator fetches per-row scales from flash itself',
    () async {
      const rows = 4, cols = 6, peR = 2, peC = 2;
      final rowBlocks = (rows + peR - 1) ~/ peR;
      final colTiles = (cols + peC - 1) ~/ peC;
      final wordsPerRow = (colTiles + 1) ~/ 2;

      final w = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          w[r * cols + c] = ((r * 5 + c * 2) % 11 - 5) * 0.19;
        }
      }
      final x = Float64List.fromList([
        for (var c = 0; c < cols; c++) ((c * 3 + 1) % 9 - 4) * 0.31,
      ]);
      final goldenFp = linear(w, rows, cols, x);
      final qm = quantizeRowwiseInt4(w, rows, cols);

      const weightBase = 0x000;
      const scaleBase = 0x200; // scales live here in 'mem', one fp16 per word
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
              wq(rb * peR + 1, ct * peC) |
              (wq(rb * peR + 1, ct * peC + 1) << 4);
        }
      }
      final fp = FloatingPoint16();
      int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
      // Lay the per-row fp16 scales into flash (low16 of each 32-bit word).
      for (var r = 0; r < rows; r++) {
        final s = e(qm.rowScales[r]);
        mem[scaleBase + r * 4] = s & 0xFF;
        mem[scaleBase + r * 4 + 1] = (s >> 8) & 0xFF;
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
      Simulator.setMaxSimTime(8000000);
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
      await csrWrite(0x020, scaleBase); // SCALE_BASE: resident scales
      // Push only activations. NO scale pushes (the accelerator fetches them).
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
      // Packed results: word w holds {row 2w+1 high16, row 2w low16}.
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
          closeTo(goldenFp[r], 0.15 + goldenFp[r].abs() * 0.2),
          reason: 'row $r: hw=${got[r]} fp=${goldenFp[r]}',
        );
      }
    },
  );

  test('packed acts (ACT_PUSH2): two fp16 activations per write', () async {
    const rows = 4, cols = 6, peR = 2, peC = 2;
    final rowBlocks = (rows + peR - 1) ~/ peR;
    final colTiles = (cols + peC - 1) ~/ peC;
    final wordsPerRow = (colTiles + 1) ~/ 2;

    final w = Float64List(rows * cols);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        w[r * cols + c] = ((r * 3 + c * 5) % 11 - 5) * 0.17;
      }
    }
    final x = Float64List.fromList([
      for (var c = 0; c < cols; c++) ((c * 7 + 3) % 9 - 4) * 0.29,
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
    Simulator.setMaxSimTime(8000000);
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
    // Push scales the legacy way (isolating the packed-act path under test).
    for (var r = 0; r < rows; r++) {
      await csrWrite(0x01C, e(qm.rowScales[r]));
    }
    // Push activations two-at-a-time: {low16 = x[2i], high16 = x[2i+1]}.
    for (var c = 0; c < cols; c += 2) {
      final lo = e(x[c]);
      final hi = c + 1 < cols ? e(x[c + 1]) : 0;
      await csrWrite(0x024, lo | (hi << 16));
    }
    await csrWrite(0x010, 0x1); // CONTROL.start

    var g = 0;
    while (g++ < 8000) {
      final status = await csrRead(0x014);
      if (status & 0x2 != 0) break; // done
    }

    final got = <double>[];
    // Packed results: word w holds {row 2w+1 high16, row 2w low16}.
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
        closeTo(goldenFp[r], 0.15 + goldenFp[r].abs() * 0.2),
        reason: 'row $r: hw=${got[r]} fp=${goldenFp[r]}',
      );
    }
  });

  test(
    'SwiGLU fusion: capture_gate then fuse_up reads back silu(gate)*up',
    () async {
      const rows = 4, cols = 6, peR = 2, peC = 2;
      final rowBlocks = (rows + peR - 1) ~/ peR;
      final colTiles = (cols + peC - 1) ~/ peC;
      final wordsPerRow = (colTiles + 1) ~/ 2;

      // Two projections over the same activation vector x: gate and up.
      final wg = Float64List(rows * cols);
      final wu = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          wg[r * cols + c] = ((r * 7 + c * 3) % 11 - 5) * 0.19;
          wu[r * cols + c] = ((r * 2 + c * 5) % 9 - 4) * 0.23;
        }
      }
      final x = Float64List.fromList([
        for (var c = 0; c < cols; c++) ((c * 5 + 2) % 9 - 4) * 0.33,
      ]);
      final goldGate = linear(wg, rows, cols, x);
      final goldUp = linear(wu, rows, cols, x);
      double silu(double v) => v / (1.0 + math.exp(-v));
      final goldFused = [
        for (var r = 0; r < rows; r++) silu(goldGate[r]) * goldUp[r],
      ];

      final qmG = quantizeRowwiseInt4(wg, rows, cols);
      final qmU = quantizeRowwiseInt4(wu, rows, cols);

      // Lay both weight matrices into 'mem' at distinct bases.
      const gateBase = 0x000;
      const upBase = 0x040;
      final mem = List<int>.filled(0x400, 0);
      void packWeights(QuantizedMatrix qm, int base) {
        int wq(int gr, int gc) =>
            (gr < rows && gc < cols) ? (qm.values[gr * cols + gc] & 0xF) : 0;
        for (var rb = 0; rb < rowBlocks; rb++) {
          for (var ct = 0; ct < colTiles; ct++) {
            final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
            final tileByte = wordByte + (ct.isOdd ? 2 : 0);
            mem[base + tileByte] =
                wq(rb * peR, ct * peC) | (wq(rb * peR, ct * peC + 1) << 4);
            mem[base + tileByte + 1] =
                wq(rb * peR + 1, ct * peC) |
                (wq(rb * peR + 1, ct * peC + 1) << 4);
          }
        }
      }

      packWeights(qmG, gateBase);
      packWeights(qmU, upBase);
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
      Simulator.setMaxSimTime(8000000);
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

      Future<void> runMatmul(
        int mode,
        int weightBase,
        QuantizedMatrix qm,
      ) async {
        await csrWrite(0x004, colTiles);
        await csrWrite(0x008, rowBlocks);
        await csrWrite(0x00C, weightBase);
        await csrWrite(0x028, mode); // MODE
        for (var c = 0; c < cols; c++) {
          await csrWrite(0x018, e(x[c]));
        }
        for (var r = 0; r < rows; r++) {
          await csrWrite(0x01C, e(qm.rowScales[r]));
        }
        await csrWrite(0x010, 0x1); // CONTROL.start
        var g = 0;
        while (g++ < 8000) {
          final status = await csrRead(0x014);
          if (status & 0x2 != 0) break; // done
        }
      }

      // Pass A: capture_gate -> gate lands in gateBuf (we never read RESULT here).
      await runMatmul(1, gateBase, qmG);
      // Pass B: fuse_up -> up matmul, then on-chip silu(gate)*up into resultBuf.
      await runMatmul(2, upBase, qmU);

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
          closeTo(goldFused[r], 0.25 + goldFused[r].abs() * 0.35),
          reason:
              'row $r: hw=${got[r]} fused=${goldFused[r]} '
              '(gate=${goldGate[r]} up=${goldUp[r]})',
        );
      }
    },
  );

  test(
    'accelerator decodes 3-bit MODE + REG_ACT_SCALE + REG_SCALE_GROUP_OFF',
    () async {
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

      for (final s in [cCyc, cStb, cWe, memAck]) {
        s.inject(0);
      }
      cAdr.inject(0);
      cDat.inject(0);
      cSel.inject(0xF);
      memMiso.inject(0);
      reset.inject(1);
      Simulator.setMaxSimTime(8000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;

      Future<void> tick() async {
        await clk.nextNegedge;
        memAck.inject(0);
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

      // MODE is 3-bit width, accepts 0-5.
      await csrWrite(0x028, 5);
      expect(await csrRead(0x028), equals(5));
      // REG_ACT_SCALE (host act-scale override, 16-bit).
      await csrWrite(0x02C, 0x1234);
      expect(await csrRead(0x02C), equals(0x1234));
      // REG_SCALE_GROUP_OFF (address-width group offset).
      await csrWrite(0x030, 0x40);
      expect(await csrRead(0x030), equals(0x40));

      await Simulator.endSimulation();
    },
  );

  test(
    'accelerator col-tile (per-tensor): fp32 accumulate across blocks == golden',
    () async {
      // rows=4, cols=128, block_cols=64 -> 2 col-blocks. Shared act-scale
      // (REG_ACT_SCALE), written once, drives BOTH blocks' dequant.
      const rows = 4, cols = 128, blockCols = 64;

      final w = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          w[r * cols + c] = ((r * 7 + c * 3) % 11 - 5) * 0.083;
        }
      }
      final x = Float64List.fromList([
        for (var c = 0; c < cols; c++) ((c * 5 + 2) % 9 - 4) * 0.091,
      ]);
      // Outlier CONFINED to col-block 0 (index < blockCols). Catches per-block
      // local act-scaling: block 0's local max-abs would equal the shared one
      // (the outlier IS the global max), but block 1's local scale would be
      // far SMALLER than shared, so block 1's int8 acts would dequantize with
      // the wrong (much larger) shared multiplier and fail the check.
      x[10] = 6.0;

      final fp = FloatingPoint16();
      int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
      double toF16(double d) => fp.valuePopulator().ofDouble(d).toDouble();

      // ONE shared per-tensor act-scale (computed over the FULL activation
      // vector, spanning both blocks), fp16-rounded since REG_ACT_SCALE is a
      // 16-bit register - the golden and the hardware must see the IDENTICAL
      // value.
      final sharedActScaleExact = quantizePerTensorInt8(x).scale;
      final sharedActScaleBits = e(sharedActScaleExact);
      final sharedActScale = toF16(sharedActScaleExact);

      final y = await runColTiled(
        rows: rows,
        cols: cols,
        blockCols: blockCols,
        w: w,
        x: x,
        sharedActScaleBits: sharedActScaleBits,
      );
      final g = goldenColTiled(w, rows, cols, blockCols, x, sharedActScale);

      for (var r = 0; r < rows; r++) {
        expect(
          fp16Close(y[r], g[r]),
          isTrue,
          reason: 'row $r: hw ${y[r]} golden ${g[r]}',
        );
      }
    },
  );

  test(
    'accelerator col-tile (per-tensor): FIRST -> MID -> LAST, 3 blocks',
    () async {
      // rows=4, cols=192, block_cols=64 -> 3 col-blocks (FIRST, MID, LAST).
      // The 2-block test above never exercises MID's accumulate path (it only
      // ever does FIRST then LAST). This one forces a real MID block in between.
      const rows = 4, cols = 192, blockCols = 64;

      final w = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          w[r * cols + c] = ((r * 5 + c * 4) % 11 - 5) * 0.071;
        }
      }
      final x = Float64List.fromList([
        for (var c = 0; c < cols; c++) ((c * 3 + 4) % 9 - 4) * 0.087,
      ]);
      // Outlier CONFINED to the MID block (indices in [blockCols, 2*blockCols)).
      // Catches per-block local act-scaling: MID's local scale would match
      // shared, but FIRST's and LAST's would not, so a wrong per-block scale
      // corrupts only MID's contribution.
      x[blockCols + 5] = 6.5;

      final fp = FloatingPoint16();
      int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
      double toF16(double d) => fp.valuePopulator().ofDouble(d).toDouble();

      final sharedActScaleExact = quantizePerTensorInt8(x).scale;
      final sharedActScaleBits = e(sharedActScaleExact);
      final sharedActScale = toF16(sharedActScaleExact);

      final y = await runColTiled(
        rows: rows,
        cols: cols,
        blockCols: blockCols,
        w: w,
        x: x,
        sharedActScaleBits: sharedActScaleBits,
      );
      final g = goldenColTiled(w, rows, cols, blockCols, x, sharedActScale);

      for (var r = 0; r < rows; r++) {
        expect(
          fp16Close(y[r], g[r]),
          isTrue,
          reason: 'row $r: hw ${y[r]} golden ${g[r]}',
        );
      }
    },
  );

  test('accelerator col-tile (per-group): each block dequants on its group '
      'scale == golden', () async {
    // rows=4, cols=128, block_cols=64 -> 2 col-blocks == 2 groups. Group b's
    // per-row scales are RESIDENT in flash at scaleBase + b*rows*4
    // (group-major). Each block writes REG_SCALE_GROUP_OFF (0x030) =
    // b*rows*4 before it runs, so the scale-fetch FSM loads GROUP b's own
    // scales (scaleBaseReg + groupOffReg), not always group 0's.
    const rows = 4, cols = 128, blockCols = 64;

    final w = Float64List(rows * cols);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        w[r * cols + c] = ((r * 7 + c * 3) % 11 - 5) * 0.083;
      }
    }
    final x = Float64List.fromList([
      for (var c = 0; c < cols; c++) ((c * 5 + 2) % 9 - 4) * 0.091,
    ]);
    // Outlier CONFINED to col-block/group 0's weights (index < blockCols).
    // Catches the group offset being ignored: block 1 would wrongly dequant
    // on group 0's (outlier-inflated) row 0 scale instead of group 1's, and
    // diverge from the groupwise golden.
    w[10] = 6.0;

    final y = await runColTiledGrouped(
      rows: rows,
      cols: cols,
      blockCols: blockCols,
      w: w,
      x: x,
    );
    final g = goldenGroupwiseColTiled(w, rows, cols, blockCols, x);

    for (var r = 0; r < rows; r++) {
      expect(
        fp16Close(y[r], g[r]),
        isTrue,
        reason: 'row $r: hw ${y[r]} golden ${g[r]}',
      );
    }
  });

  // Full board shape: col-tile + resident scales + ROW-TILING (rows>maxRows,
  // accumulator bank reused per row-tile) + REAL flash. maxRowBlocks=2
  // (maxRows=4) with rows=8 forces 2 row-tiles, 2 col-blocks each.
  test(
    'fp accel + REAL flash, COL-TILE + ROW-TILING + resident scales: golden',
    timeout: const Timeout(Duration(minutes: 25)),
    () async {
      const rows = 32, cols = 32, blockCols = 16, maxRowBlocks = 2;
      final w = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          w[r * cols + c] = ((r * 5 + c * 3 + r * c) % 11 - 5) * 0.13;
        }
      }
      final x = Float64List.fromList([
        for (var c = 0; c < cols; c++) ((c * 7 + 2) % 13 - 6) * 0.19,
      ]);
      final y = await runColTiledGroupedRowTiledFlash(
        rows: rows,
        cols: cols,
        blockCols: blockCols,
        maxRowBlocks: maxRowBlocks,
        w: w,
        x: x,
      );
      final g = goldenGroupwiseColTiled(w, rows, cols, blockCols, x);
      for (var r = 0; r < rows; r++) {}
      for (var r = 0; r < rows; r++) {
        expect(
          fp16Close(y[r], g[r]),
          isTrue,
          reason: 'row $r: hw ${y[r]} golden ${g[r]}',
        );
      }
    },
  );

  // Board shape: col-tile (FIRST/MID/LAST accumulate) + resident group
  // scales, both read from the REAL flash controller per block. Weights span
  // 2 read-ahead lines per block, and the per-block scale read interleaves
  // between them, thrashing the read-ahead line buffer.
  test(
    'fp accel + REAL flash, COL-TILE + resident scales: golden',
    timeout: const Timeout(Duration(minutes: 10)),
    () async {
      const rows = 8, cols = 32, blockCols = 16;
      final w = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          w[r * cols + c] = ((r * 5 + c * 3 + r * c) % 11 - 5) * 0.13;
        }
      }
      final x = Float64List.fromList([
        for (var c = 0; c < cols; c++) ((c * 7 + 2) % 13 - 6) * 0.19,
      ]);
      final y = await runColTiledGroupedFlash(
        rows: rows,
        cols: cols,
        blockCols: blockCols,
        w: w,
        x: x,
      );
      final g = goldenGroupwiseColTiled(w, rows, cols, blockCols, x);
      for (var r = 0; r < rows; r++) {}
      for (var r = 0; r < rows; r++) {
        expect(
          fp16Close(y[r], g[r]),
          isTrue,
          reason: 'row $r: hw ${y[r]} golden ${g[r]}',
        );
      }
    },
  );
}
