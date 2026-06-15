// LoomDecoderLayer: one full on-chip decoder layer (attention half via
// LoomAttnBlock, then SwiGLU MLP half via LoomRmsNorm + LoomMlpSeq +
// LoomFpResidual). Verified against the golden runner's per-token whole-layer
// output in W4A8+fp16 at t=0 and t=1 (growing KV cache across positions).

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:loom/src/golden/attention.dart';
import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/decoder_layer.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('decoder layer matches golden runner at t=0 then t=1', () async {
    const H = 8, nH = 2, nKV = 1, hd = 4, iSize = 16, maxSeq = 4;
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
    final wGate = mat(iSize, H, 3);
    final wUp = mat(iSize, H, 6);
    final wDown = mat(H, iSize, 9);
    final iGamma = Float64List.fromList([
      for (var c = 0; c < H; c++) 0.5 + (c % 3) * 0.4,
    ]);
    final pGamma = Float64List.fromList([
      for (var c = 0; c < H; c++) 0.6 + (c % 2) * 0.5,
    ]);

    final hiddens = [
      Float64List.fromList([
        for (var c = 0; c < H; c++) ((c * 5 + 2) % 9 - 4) * 0.33,
      ]),
      Float64List.fromList([
        for (var c = 0; c < H; c++) ((c * 7 + 1) % 9 - 4) * 0.29,
      ]),
    ];

    // KV rows accumulate across positions, mirroring the cache.
    final qRows = <Float64List>[];
    final kRows = <Float64List>[];
    final vRows = <Float64List>[];
    Float64List goldenLayer(Float64List hidden, int t) {
      // Attention sub-block.
      final n = rmsNorm(hidden, iGamma, eps);
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
      final hidden1 = Float64List.fromList(hidden);
      addInPlace(hidden1, o);

      // MLP sub-block (gated SwiGLU).
      final n2 = rmsNorm(hidden1, pGamma, eps);
      final g = quantizedLinearW4A8(wGate, iSize, H, n2);
      final u = quantizedLinearW4A8(wUp, iSize, H, n2);
      final act = silu(g);
      final inter = mul(act, u);
      final d = quantizedLinearW4A8(wDown, H, iSize, inter);
      final hidden2 = Float64List.fromList(hidden1);
      addInPlace(hidden2, d);
      return hidden2;
    }

    final golden0 = goldenLayer(hiddens[0], 0);
    final golden1 = goldenLayer(hiddens[1], 1);

    const wbQ = 0x000,
        wbK = 0x100,
        wbV = 0x200,
        wbO = 0x300,
        wbGate = 0x400,
        wbUp = 0x500,
        wbDown = 0x600;
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
    pack(wbGate, wGate, iSize, H);
    pack(wbUp, wUp, iSize, H);
    pack(wbDown, wDown, H, iSize);
    int memWord(int a) => (a < 0 || a + 3 >= mem.length)
        ? 0
        : mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);

    final qmQ = quantizeRowwiseInt4(wq, qDim, H);
    final qmK = quantizeRowwiseInt4(wk, kvDim, H);
    final qmV = quantizeRowwiseInt4(wv, kvDim, H);
    final qmO = quantizeRowwiseInt4(wo, H, qDim);
    final qmG = quantizeRowwiseInt4(wGate, iSize, H);
    final qmU = quantizeRowwiseInt4(wUp, iSize, H);
    final qmD = quantizeRowwiseInt4(wDown, H, iSize);

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
    final wbg = Logic(name: 'wbg', width: 32)..inject(wbGate);
    final wbu = Logic(name: 'wbu', width: 32)..inject(wbUp);
    final wbd = Logic(name: 'wbd', width: 32)..inject(wbDown);
    final xEn = Logic(name: 'x_en');
    final xInL = Logic(name: 'x_in', width: 16);
    final igaInL = Logic(name: 'iga_in', width: 16);
    final pgaInL = Logic(name: 'pga_in', width: 16);
    final sqEn = Logic(name: 'sq_en');
    final sqInL = Logic(name: 'sq_in', width: 16);
    final skEn = Logic(name: 'sk_en');
    final skInL = Logic(name: 'sk_in', width: 16);
    final svEn = Logic(name: 'sv_en');
    final svInL = Logic(name: 'sv_in', width: 16);
    final soEn = Logic(name: 'so_en');
    final soInL = Logic(name: 'so_in', width: 16);
    final sgEn = Logic(name: 'sg_en');
    final sgInL = Logic(name: 'sg_in', width: 16);
    final suEn = Logic(name: 'su_en');
    final suInL = Logic(name: 'su_in', width: 16);
    final sdEn = Logic(name: 'sd_en');
    final sdInL = Logic(name: 'sd_in', width: 16);
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

    final dut = LoomDecoderLayer(
      hidden: H,
      numHeads: nH,
      numKvHeads: nKV,
      headDim: hd,
      intermediateSize: iSize,
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
      ('wb_gate', wbg),
      ('wb_up', wbu),
      ('wb_down', wbd),
      ('x_en', xEn),
      ('x_in', xInL),
      ('iga_in', igaInL),
      ('pga_in', pgaInL),
      ('sq_en', sqEn),
      ('sq_in', sqInL),
      ('sk_en', skEn),
      ('sk_in', skInL),
      ('sv_en', svEn),
      ('sv_in', svInL),
      ('so_en', soEn),
      ('so_in', soInL),
      ('sg_en', sgEn),
      ('sg_in', sgInL),
      ('su_en', suEn),
      ('su_in', suInL),
      ('sd_en', sdEn),
      ('sd_in', sdInL),
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

    for (final l in [start, xEn, sqEn, skEn, svEn, soEn, sgEn, suEn, sdEn]) {
      l.inject(0);
    }
    for (final l in [
      xInL,
      igaInL,
      pgaInL,
      sqInL,
      skInL,
      svInL,
      soInL,
      sgInL,
      suInL,
      sdInL,
    ]) {
      l.inject(0);
    }
    pos.inject(0);
    invN.inject(e(1.0 / H));
    epsL.inject(e(eps));
    setRope(0);
    reset.inject(1);
    Simulator.setMaxSimTime(40000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    Future<List<double>> runLayer(Float64List hidden, int t) async {
      // Wait for idle (busy low) before streaming. After a prior position the
      // FSM spends one cycle in FIN before returning to LOAD.
      while (o('busy').value.toBool()) {
        await clk.nextPosedge;
      }
      setRope(t);
      pos.inject(t);
      // LOAD: hidden + input gamma + post-attn gamma.
      for (var i = 0; i < H; i++) {
        xEn.inject(1);
        xInL.inject(e(hidden[i]));
        igaInL.inject(e(iGamma[i]));
        pgaInL.inject(e(pGamma[i]));
        await clk.nextPosedge;
      }
      xEn.inject(0);
      // Attention Q/K/V/O row scales.
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
      // MLP gate/up (iSize) then down (H) row scales.
      for (var i = 0; i < iSize; i++) {
        sgEn.inject(1);
        sgInL.inject(e(qmG.rowScales[i]));
        suEn.inject(1);
        suInL.inject(e(qmU.rowScales[i]));
        await clk.nextPosedge;
      }
      sgEn.inject(0);
      suEn.inject(0);
      for (var i = 0; i < H; i++) {
        sdEn.inject(1);
        sdInL.inject(e(qmD.rowScales[i]));
        await clk.nextPosedge;
      }
      sdEn.inject(0);

      start.inject(1);
      await clk.nextPosedge;
      start.inject(0);

      final got = <double>[];
      var guard = 0;
      while (guard++ < 400000) {
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

    final got0 = await runLayer(hiddens[0], 0);
    final got1 = await runLayer(hiddens[1], 1);
    await Simulator.endSimulation();

    void check(String tag, List<double> hw, Float64List g) {
      expect(hw.length, equals(g.length), reason: '$tag count');
      for (var i = 0; i < g.length; i++) {
        expect(
          hw[i],
          closeTo(g[i], 0.08 + g[i].abs() * 0.22),
          reason: '$tag[$i] hw=${hw[i]} g=${g[i]}',
        );
      }
    }

    check('t0', got0, golden0);
    check('t1', got1, golden1);
  });
}
