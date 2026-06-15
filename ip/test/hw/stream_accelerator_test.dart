// LoomStreamAccelerator: host-drivable memory-backed matmul. The testbench
// plays the HOST on the CSR slave (writes config, strobes start, polls status,
// reads results) AND the MEMORY on the 'mem' master (answers weight reads),
// concurrently. Result must be bit-exact to the golden.

import 'dart:async';

import 'package:loom/src/hw/stream_accelerator.dart';
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

  test('host drives a memory-backed matmul end-to-end, bit-exact', () async {
    const rows = 4, cols = 6, peR = 2, peC = 2;
    final w = [
      3, -2, 1, -1, 2, 0, //
      -4, 3, -2, 2, 1, -1, //
      1, 1, 1, 1, 1, 1, //
      -1, 2, -3, 0, 2, -2, //
    ];
    final x = [5, -3, 2, -1, 4, 2];
    final mults = [16, 16, 8, 32];
    const shift = 4;

    final rowBlocks = (rows + peR - 1) ~/ peR;
    final colTiles = (cols + peC - 1) ~/ peC;
    final paddedCols = colTiles * peC, paddedRows = rowBlocks * peR;
    const weightBase = 0x000, actBase = 0x400, multBase = 0x800;

    final mem = List<int>.filled(0x1000, 0);
    var wp = weightBase;
    for (var rb = 0; rb < rowBlocks; rb++) {
      for (var ct = 0; ct < colTiles; ct++) {
        for (var lr = 0; lr < peR; lr++) {
          for (var lc = 0; lc < peC; lc++) {
            final gr = rb * peR + lr, gc = ct * peC + lc;
            mem[wp++] =
                ((gr < rows && gc < cols) ? w[gr * cols + gc] : 0) & 0xFF;
          }
        }
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
        : mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    // CSR (host) signals.
    final cCyc = Logic(name: 'c_cyc');
    final cStb = Logic(name: 'c_stb');
    final cWe = Logic(name: 'c_we');
    final cAdr = Logic(name: 'c_adr', width: 32);
    final cDat = Logic(name: 'c_dat', width: 32);
    final cSel = Logic(name: 'c_sel', width: 4);
    // mem responder signals.
    final memAck = Logic(name: 'mem_ack');
    final memMiso = Logic(name: 'mem_miso', width: 32);

    final dut = LoomStreamAccelerator();
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
    Simulator.setMaxSimTime(5000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    // One clock step that ALSO services the weight-memory master.
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

    await csrWrite(0x004, rowBlocks);
    await csrWrite(0x008, colTiles);
    await csrWrite(0x00C, shift);
    await csrWrite(0x018, weightBase);
    await csrWrite(0x01C, actBase);
    await csrWrite(0x020, multBase);
    await csrWrite(0x010, 0x1); // CONTROL.start

    var g = 0;
    while (g++ < 5000) {
      final status = await csrRead(0x014);
      if (status & 0x2 != 0) break; // done
    }

    final outputs = <int>[];
    for (var b = 0; b < rowBlocks; b++) {
      final word = await csrRead(0x100 + b * 4);
      outputs.add(_toI8(word & 0xFF));
      outputs.add(_toI8((word >> 8) & 0xFF));
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
  });
}
