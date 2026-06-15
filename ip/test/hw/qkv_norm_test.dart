// LoomQkvNorm: one RMSNorm feeds Q/K/V through a single reused LoomFpLinear
// engine. Checked against the W4A8 golden chain: q/k/v = W_{q,k,v} @ rmsNorm(x, gamma).

import 'dart:async';
import 'dart:typed_data';

import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/qkv_norm.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('norm feeds Q/K/V through one reused engine to golden', () async {
    const hidden = 6, qDim = 8, kvDim = 4;
    const peR = 2, peC = 2;
    const eps = 0.01;

    Float64List mat(int rows, int cols, int seed) {
      final m = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          m[r * cols + c] = ((r * 5 + c * 3 + seed) % 11 - 5) * 0.2;
        }
      }
      return m;
    }

    final wq = mat(qDim, hidden, 1);
    final wk = mat(kvDim, hidden, 4);
    final wv = mat(kvDim, hidden, 7);
    final x = Float64List.fromList([
      for (var c = 0; c < hidden; c++) ((c * 5 + 2) % 9 - 4) * 0.33,
    ]);
    final gamma = Float64List.fromList([
      for (var c = 0; c < hidden; c++) 0.5 + (c % 3) * 0.4,
    ]);
    final normed = rmsNorm(x, gamma, eps);
    final gQ = quantizedLinearW4A8(wq, qDim, hidden, normed);
    final gK = quantizedLinearW4A8(wk, kvDim, hidden, normed);
    final gV = quantizedLinearW4A8(wv, kvDim, hidden, normed);

    const wbQ = 0x000, wbK = 0x040, wbV = 0x080;
    final mem = List<int>.filled(0x400, 0);
    void pack(int base, Float64List w, int rows, int cols) {
      final qm = quantizeRowwiseInt4(w, rows, cols);
      final rb = (rows + peR - 1) ~/ peR;
      final cts = (cols + peC - 1) ~/ peC;
      final wpr = (cts + 1) ~/ 2;
      int wq(int gr, int gc) =>
          (gr < rows && gc < cols) ? (qm.values[gr * cols + gc] & 0xF) : 0;
      for (var rbi = 0; rbi < rb; rbi++) {
        for (var ct = 0; ct < cts; ct++) {
          final tb = base + (rbi * wpr + (ct >> 1)) * 4 + (ct.isOdd ? 2 : 0);
          mem[tb] = wq(rbi * 2, ct * 2) | (wq(rbi * 2, ct * 2 + 1) << 4);
          mem[tb + 1] =
              wq(rbi * 2 + 1, ct * 2) | (wq(rbi * 2 + 1, ct * 2 + 1) << 4);
        }
      }
    }

    pack(wbQ, wq, qDim, hidden);
    pack(wbK, wk, kvDim, hidden);
    pack(wbV, wv, kvDim, hidden);
    int memWord(int a) => (a < 0 || a + 3 >= mem.length)
        ? 0
        : mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final start = Logic(name: 'start');
    final wbg = Logic(name: 'wbg', width: 32)..inject(wbQ);
    final wbkL = Logic(name: 'wbk', width: 32)..inject(wbK);
    final wbvL = Logic(name: 'wbv', width: 32)..inject(wbV);
    final invN = Logic(name: 'inv_n', width: 16);
    final epsL = Logic(name: 'eps', width: 16);
    final xEn = Logic(name: 'x_en');
    final xInL = Logic(name: 'x_in', width: 16);
    final gInL = Logic(name: 'gamma_in', width: 16);
    final sqEn = Logic(name: 'sq_en');
    final sqInL = Logic(name: 'sq_in', width: 16);
    final skEn = Logic(name: 'sk_en');
    final skInL = Logic(name: 'sk_in', width: 16);
    final svEn = Logic(name: 'sv_en');
    final svInL = Logic(name: 'sv_in', width: 16);
    final ack = Logic(name: 'ack')..inject(0);
    final datMiso = Logic(name: 'miso', width: 32)..inject(0);

    final dut = LoomQkvNorm(hidden: hidden, qDim: qDim, kvDim: kvDim);
    for (final (p, s) in [
      ('clk', clk),
      ('reset', reset),
      ('start', start),
      ('wb_q', wbg),
      ('wb_k', wbkL),
      ('wb_v', wbvL),
      ('inv_n', invN),
      ('eps', epsL),
      ('x_en', xEn),
      ('x_in', xInL),
      ('gamma_in', gInL),
      ('sq_en', sqEn),
      ('sq_in', sqInL),
      ('sk_en', skEn),
      ('sk_in', skInL),
      ('sv_en', svEn),
      ('sv_in', svInL),
      ('mem_ACK', ack),
      ('mem_DAT_MISO', datMiso),
    ]) {
      dut.input(p).srcConnection! <= s;
    }
    await dut.build();
    Logic o(String n) => dut.output(n);

    final fp = FloatingPoint16();
    int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
    final outFp = FloatingPoint16();

    for (final l in [start, xEn, sqEn, skEn, svEn]) {
      l.inject(0);
    }
    for (final l in [xInL, gInL, sqInL, skInL, svInL]) {
      l.inject(0);
    }
    invN.inject(e(1.0 / hidden));
    epsL.inject(e(eps));
    reset.inject(1);
    Simulator.setMaxSimTime(8000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    final qmQ = quantizeRowwiseInt4(wq, qDim, hidden);
    final qmK = quantizeRowwiseInt4(wk, kvDim, hidden);
    final qmV = quantizeRowwiseInt4(wv, kvDim, hidden);

    // LOAD: x+gamma, then Q/K/V scales.
    for (var i = 0; i < hidden; i++) {
      xEn.inject(1);
      xInL.inject(e(x[i]));
      gInL.inject(e(gamma[i]));
      await clk.nextPosedge;
    }
    xEn.inject(0);
    for (var i = 0; i < qDim; i++) {
      sqEn.inject(1);
      sqInL.inject(e(qmQ.rowScales[i]));
      await clk.nextPosedge;
    }
    sqEn.inject(0);
    for (var i = 0; i < kvDim; i++) {
      skEn.inject(1);
      skInL.inject(e(qmK.rowScales[i]));
      svEn.inject(1);
      svInL.inject(e(qmV.rowScales[i]));
      await clk.nextPosedge;
    }
    skEn.inject(0);
    svEn.inject(0);

    start.inject(1);
    await clk.nextPosedge;
    start.inject(0);

    final got = [<double>[], <double>[], <double>[]];
    var guard = 0;
    while (guard++ < 40000) {
      await clk.nextNegedge;
      if (o('mem_STB').value.toBool() && o('mem_CYC').value.toBool()) {
        ack.inject(1);
        datMiso.inject(memWord(o('mem_ADR').value.toInt()));
      } else {
        ack.inject(0);
      }
      if (o('y_valid').value.toBool()) {
        outFp.put(o('y').value);
        got[o('y_phase').value.toInt()].add(
          outFp.floatingPointValue.toDouble(),
        );
      }
      await clk.nextPosedge;
      if (o('done').value.toBool()) break;
    }
    await Simulator.endSimulation();

    void check(String tag, List<double> hw, Float64List g) {
      expect(hw.length, equals(g.length), reason: '$tag count');
      for (var r = 0; r < g.length; r++) {
        expect(
          hw[r],
          closeTo(g[r], 0.2 + g[r].abs() * 0.25),
          reason: '$tag[$r] hw=${hw[r]} g=${g[r]}',
        );
      }
    }

    check('Q', got[0], gQ);
    check('K', got[1], gK);
    check('V', got[2], gV);
  });
}
