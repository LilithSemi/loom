// LoomLmHead: final RMSNorm -> logits = Wcls @ f -> streaming fp16 argmax -> token id.
// Compared against golden's final-norm + W4A8 lm_head by argmax index, robust to small fp error.

import 'dart:async';
import 'dart:typed_data';

import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/lm_head.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('lm_head emits argmax token matching golden', () async {
    const H = 8, vocab = 32;
    const peR = 2, peC = 2;
    const eps = 0.01;

    Float64List mat(int rows, int cols, int seed) {
      final m = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          m[r * cols + c] = ((r * 5 + c * 3 + seed) % 13 - 6) * 0.18;
        }
      }
      return m;
    }

    final wCls = mat(vocab, H, 5);
    final gamma = Float64List.fromList([
      for (var c = 0; c < H; c++) 0.5 + (c % 3) * 0.4,
    ]);
    final hidden = Float64List.fromList([
      for (var c = 0; c < H; c++) ((c * 5 + 2) % 9 - 4) * 0.33,
    ]);

    // Golden: final norm then W4A8 lm_head, argmax over logits.
    final f = rmsNorm(hidden, gamma, eps);
    final logits = quantizedLinearW4A8(wCls, vocab, H, f);
    var goldenTok = 0;
    for (var i = 1; i < vocab; i++) {
      if (logits[i] > logits[goldenTok]) goldenTok = i;
    }

    // Weight image (int4 tile-major).
    const wbCls = 0x000;
    final mem = List<int>.filled(0x800, 0);
    void pack(int base, Float64List w, int rows, int cols) {
      final qm = quantizeRowwiseInt4(w, rows, cols);
      final rb = (rows + peR - 1) ~/ peR;
      final cts = (cols + peC - 1) ~/ peC;
      final wpr = (cts + 1) ~/ 2;
      int wqf(int gr, int gc) =>
          (gr < rows && gc < cols) ? (qm.values[gr * cols + gc] & 0xF) : 0;
      for (var rbi = 0; rbi < rb; rbi++) {
        for (var ct = 0; ct < cts; ct++) {
          final tb = base + (rbi * wpr + (ct >> 1)) * 4 + (ct.isOdd ? 2 : 0);
          mem[tb] = wqf(rbi * 2, ct * 2) | (wqf(rbi * 2, ct * 2 + 1) << 4);
          mem[tb + 1] =
              wqf(rbi * 2 + 1, ct * 2) | (wqf(rbi * 2 + 1, ct * 2 + 1) << 4);
        }
      }
    }

    pack(wbCls, wCls, vocab, H);
    int memWord(int a) => (a < 0 || a + 3 >= mem.length)
        ? 0
        : mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);

    final qmCls = quantizeRowwiseInt4(wCls, vocab, H);

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final start = Logic(name: 'start');
    final invN = Logic(name: 'inv_n', width: 16);
    final epsL = Logic(name: 'eps', width: 16);
    final wbcls = Logic(name: 'wbcls', width: 32)..inject(wbCls);
    final xEn = Logic(name: 'x_en');
    final xInL = Logic(name: 'x_in', width: 16);
    final gInL = Logic(name: 'gamma_in', width: 16);
    final scEn = Logic(name: 'sc_en');
    final scInL = Logic(name: 'sc_in', width: 16);
    final ack = Logic(name: 'ack')..inject(0);
    final datMiso = Logic(name: 'miso', width: 32)..inject(0);

    final dut = LoomLmHead(hidden: H, vocab: vocab);
    for (final (p, s) in [
      ('clk', clk),
      ('reset', reset),
      ('start', start),
      ('inv_n', invN),
      ('eps', epsL),
      ('wb_cls', wbcls),
      ('x_en', xEn),
      ('x_in', xInL),
      ('gamma_in', gInL),
      ('sc_en', scEn),
      ('sc_in', scInL),
      ('mem_ACK', ack),
      ('mem_DAT_MISO', datMiso),
    ]) {
      dut.input(p).srcConnection! <= s;
    }
    await dut.build();
    Logic o(String n) => dut.output(n);

    final fp = FloatingPoint16();
    int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();

    for (final l in [start, xEn, scEn]) {
      l.inject(0);
    }
    xInL.inject(0);
    gInL.inject(0);
    scInL.inject(0);
    invN.inject(e(1.0 / H));
    epsL.inject(e(eps));
    reset.inject(1);
    Simulator.setMaxSimTime(20000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    // LOAD: hidden + gamma, then Wcls row scales.
    for (var i = 0; i < H; i++) {
      xEn.inject(1);
      xInL.inject(e(hidden[i]));
      gInL.inject(e(gamma[i]));
      await clk.nextPosedge;
    }
    xEn.inject(0);
    for (var i = 0; i < vocab; i++) {
      scEn.inject(1);
      scInL.inject(e(qmCls.rowScales[i]));
      await clk.nextPosedge;
    }
    scEn.inject(0);

    start.inject(1);
    await clk.nextPosedge;
    start.inject(0);

    var token = -1;
    var guard = 0;
    while (guard++ < 200000) {
      await clk.nextNegedge;
      if (o('mem_STB').value.toBool() && o('mem_CYC').value.toBool()) {
        ack.inject(1);
        datMiso.inject(memWord(o('mem_ADR').value.toInt()));
      } else {
        ack.inject(0);
      }
      await clk.nextPosedge;
      if (o('done').value.toBool()) {
        token = o('token').value.toInt();
        break;
      }
    }
    await Simulator.endSimulation();

    expect(
      token,
      equals(goldenTok),
      reason: 'hw token=$token golden=$goldenTok logits=$logits',
    );
  });
}
