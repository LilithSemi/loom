// LoomAttnBlock: attention half of one on-chip decoder layer (FSM chains
// qkv_norm -> rope_vec -> attn_seq -> linear_seq -> fp_residual). Verified
// against the golden runner at t=0 and t=1 with a growing on-chip KV cache.

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:loom/src/golden/attention.dart';
import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/attn_block.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('attention block matches golden runner at t=0 then t=1', () async {
    const H = 8, nH = 2, nKV = 1, hd = 4, maxSeq = 4;
    const qDim = nH * hd, kvDim = nKV * hd, half = hd ~/ 2;
    const peR = 2, peC = 2;
    const eps = 0.01, theta = 10000.0;
    final invSqrtHd = 1.0 / math.sqrt(hd.toDouble());

    Float64List mat(int rows, int cols, int seed) {
      final m = Float64List(rows * cols);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          m[r * cols + c] = ((r * 5 + c * 3 + seed) % 11 - 5) * 0.2;
        }
      }
      return m;
    }

    final wq = mat(qDim, H, 1);
    final wk = mat(kvDim, H, 4);
    final wv = mat(kvDim, H, 7);
    final wo = mat(H, qDim, 2);
    final gamma = Float64List.fromList([
      for (var c = 0; c < H; c++) 0.5 + (c % 3) * 0.4,
    ]);

    final hiddens = [
      Float64List.fromList([
        for (var c = 0; c < H; c++) ((c * 5 + 2) % 9 - 4) * 0.33,
      ]),
      Float64List.fromList([
        for (var c = 0; c < H; c++) ((c * 7 + 1) % 9 - 4) * 0.29,
      ]),
    ];

    // rows accumulate across positions just like the on-chip cache.
    final qRows = <Float64List>[];
    final kRows = <Float64List>[];
    final vRows = <Float64List>[];
    Float64List goldenStep(Float64List hidden, int t) {
      final n = rmsNorm(hidden, gamma, eps);
      final qFull = quantizedLinearW4A8(wq, qDim, H, n);
      final kFull = quantizedLinearW4A8(wk, kvDim, H, n);
      final vFull = quantizedLinearW4A8(wv, kvDim, H, n);
      for (var h = 0; h < nH; h++) {
        final s = Float64List.fromList(qFull.sublist(h * hd, h * hd + hd));
        applyRopeHead(s, t, theta);
        qFull.setRange(h * hd, h * hd + hd, s);
      }
      for (var h = 0; h < nKV; h++) {
        final s = Float64List.fromList(kFull.sublist(h * hd, h * hd + hd));
        applyRopeHead(s, t, theta);
        kFull.setRange(h * hd, h * hd + hd, s);
      }
      qRows.add(qFull);
      kRows.add(kFull);
      vRows.add(vFull);
      final attn = causalGqaAttention(qRows, kRows, vRows, nH, nKV, hd)[t];
      final o = quantizedLinearW4A8(wo, H, qDim, attn);
      final hp = Float64List.fromList(hidden);
      addInPlace(hp, o);
      return hp;
    }

    final golden0 = goldenStep(hiddens[0], 0);
    final golden1 = goldenStep(hiddens[1], 1);

    const wbQ = 0x000, wbK = 0x080, wbV = 0x100, wbO = 0x180;
    final mem = List<int>.filled(0x800, 0);
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

    pack(wbQ, wq, qDim, H);
    pack(wbK, wk, kvDim, H);
    pack(wbV, wv, kvDim, H);
    pack(wbO, wo, H, qDim);
    int memWord(int a) => (a < 0 || a + 3 >= mem.length)
        ? 0
        : mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);

    final qmQ = quantizeRowwiseInt4(wq, qDim, H);
    final qmK = quantizeRowwiseInt4(wk, kvDim, H);
    final qmV = quantizeRowwiseInt4(wv, kvDim, H);
    final qmO = quantizeRowwiseInt4(wo, H, qDim);

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final start = Logic(name: 'start');
    final pos = Logic(name: 'pos', width: maxSeq.bitLength);
    final invN = Logic(name: 'inv_n', width: 16);
    final epsL = Logic(name: 'eps', width: 16);
    final wbq = Logic(name: 'wbq', width: 32)..inject(wbQ);
    final wbk = Logic(name: 'wbk', width: 32)..inject(wbK);
    final wbv = Logic(name: 'wbv', width: 32)..inject(wbV);
    final wbo = Logic(name: 'wbo', width: 32)..inject(wbO);
    final xEn = Logic(name: 'x_en');
    final xInL = Logic(name: 'x_in', width: 16);
    final gInL = Logic(name: 'gamma_in', width: 16);
    final sqEn = Logic(name: 'sq_en');
    final sqInL = Logic(name: 'sq_in', width: 16);
    final skEn = Logic(name: 'sk_en');
    final skInL = Logic(name: 'sk_in', width: 16);
    final svEn = Logic(name: 'sv_en');
    final svInL = Logic(name: 'sv_in', width: 16);
    final soEn = Logic(name: 'so_en');
    final soInL = Logic(name: 'so_in', width: 16);
    final cosQ = [
      for (var jj = 0; jj < half; jj++) Logic(name: 'cq$jj', width: 16),
    ];
    final sinQ = [
      for (var jj = 0; jj < half; jj++) Logic(name: 'sq$jj', width: 16),
    ];
    final cosK = [
      for (var jj = 0; jj < half; jj++) Logic(name: 'ck$jj', width: 16),
    ];
    final sinK = [
      for (var jj = 0; jj < half; jj++) Logic(name: 'sk$jj', width: 16),
    ];
    final ack = Logic(name: 'ack')..inject(0);
    final datMiso = Logic(name: 'miso', width: 32)..inject(0);

    final dut = LoomAttnBlock(
      hidden: H,
      numHeads: nH,
      numKvHeads: nKV,
      headDim: hd,
      maxSeq: maxSeq,
    );
    for (final (p, s) in [
      ('clk', clk),
      ('reset', reset),
      ('start', start),
      ('pos', pos),
      ('inv_n', invN),
      ('eps', epsL),
      ('wb_q', wbq),
      ('wb_k', wbk),
      ('wb_v', wbv),
      ('wb_o', wbo),
      ('x_en', xEn),
      ('x_in', xInL),
      ('gamma_in', gInL),
      ('sq_en', sqEn),
      ('sq_in', sqInL),
      ('sk_en', skEn),
      ('sk_in', skInL),
      ('sv_en', svEn),
      ('sv_in', svInL),
      ('so_en', soEn),
      ('so_in', soInL),
      ('mem_ACK', ack),
      ('mem_DAT_MISO', datMiso),
    ]) {
      dut.input(p).srcConnection! <= s;
    }
    for (var jj = 0; jj < half; jj++) {
      dut.input('cos_q$jj').srcConnection! <= cosQ[jj];
      dut.input('sin_q$jj').srcConnection! <= sinQ[jj];
      dut.input('cos_k$jj').srcConnection! <= cosK[jj];
      dut.input('sin_k$jj').srcConnection! <= sinK[jj];
    }
    await dut.build();
    Logic o(String n) => dut.output(n);

    final fp = FloatingPoint16();
    int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
    final outFp = FloatingPoint16();

    void setRope(int t) {
      for (var jj = 0; jj < half; jj++) {
        final invFreq = 1.0 / math.pow(theta, (2 * jj) / hd);
        final angle = t * invFreq;
        final c = math.cos(angle), s = math.sin(angle);
        cosK[jj].inject(e(c));
        sinK[jj].inject(e(s));
        cosQ[jj].inject(e(c * invSqrtHd));
        sinQ[jj].inject(e(s * invSqrtHd));
      }
    }

    for (final l in [start, xEn, sqEn, skEn, svEn, soEn]) {
      l.inject(0);
    }
    for (final l in [xInL, gInL, sqInL, skInL, svInL, soInL]) {
      l.inject(0);
    }
    pos.inject(0);
    invN.inject(e(1.0 / H));
    epsL.inject(e(eps));
    setRope(0);
    reset.inject(1);
    Simulator.setMaxSimTime(20000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    Future<List<double>> runBlock(Float64List hidden, int t) async {
      // Wait for the block to be idle in LOAD (busy low) before streaming. After
      // a prior position it spends one cycle in FIN before returning to LOAD.
      while (o('busy').value.toBool()) {
        await clk.nextPosedge;
      }
      setRope(t);
      pos.inject(t);
      // LOAD: hidden+gamma, then Q/K/V scales, then o_proj row scales.
      for (var i = 0; i < H; i++) {
        xEn.inject(1);
        xInL.inject(e(hidden[i]));
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
      for (var i = 0; i < H; i++) {
        soEn.inject(1);
        soInL.inject(e(qmO.rowScales[i]));
        await clk.nextPosedge;
      }
      soEn.inject(0);

      start.inject(1);
      await clk.nextPosedge;
      start.inject(0);

      final got = <double>[];
      var guard = 0;
      while (guard++ < 200000) {
        await clk.nextNegedge;
        if (o('mem_STB').value.toBool() && o('mem_CYC').value.toBool()) {
          ack.inject(1);
          datMiso.inject(memWord(o('mem_ADR').value.toInt()));
        } else {
          ack.inject(0);
        }
        if (o('h_valid').value.toBool()) {
          outFp.put(o('h_out').value);
          got.add(outFp.floatingPointValue.toDouble());
        }
        await clk.nextPosedge;
        if (o('done').value.toBool()) break;
      }
      return got;
    }

    final got0 = await runBlock(hiddens[0], 0);
    final got1 = await runBlock(hiddens[1], 1);
    await Simulator.endSimulation();

    void check(String tag, List<double> hw, Float64List g) {
      expect(hw.length, equals(g.length), reason: '$tag count');
      for (var i = 0; i < g.length; i++) {
        expect(
          hw[i],
          closeTo(g[i], 0.06 + g[i].abs() * 0.2),
          reason: '$tag[$i] hw=${hw[i]} g=${g[i]}',
        );
      }
    }

    check('t0', got0, golden0);
    check('t1', got1, golden1);
  });
}
