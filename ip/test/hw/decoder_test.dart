// LoomDecoder: multi-layer decoder FSM, reusing ONE LoomDecoderLayer over a
// resident hidden[H] across numLayers, with per-layer KV caches (layer-indexed
// in LoomAttnBlock). Checked against golden at t=0/t=1 for per-layer KV isolation.

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:loom/src/golden/attention.dart';
import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/decoder.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('2-layer decoder matches golden forward at t=0 then t=1', () async {
    const H = 8, nH = 2, nKV = 1, hd = 4, iSize = 16, maxSeq = 4;
    const numLayers = 2;
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

    // Per-layer weights + gammas (distinct per layer via seed offset).
    final wq = <Float64List>[], wk = <Float64List>[], wv = <Float64List>[];
    final wo = <Float64List>[], wGate = <Float64List>[], wUp = <Float64List>[];
    final wDown = <Float64List>[];
    final iGamma = <Float64List>[], pGamma = <Float64List>[];
    for (var l = 0; l < numLayers; l++) {
      final s = l * 13;
      wq.add(mat(qDim, H, 1 + s));
      wk.add(mat(kvDim, H, 4 + s));
      wv.add(mat(kvDim, H, 7 + s));
      wo.add(mat(H, qDim, 2 + s));
      wGate.add(mat(iSize, H, 3 + s));
      wUp.add(mat(iSize, H, 6 + s));
      wDown.add(mat(H, iSize, 9 + s));
      iGamma.add(
        Float64List.fromList([
          for (var c = 0; c < H; c++) 0.5 + ((c + l) % 3) * 0.4,
        ]),
      );
      pGamma.add(
        Float64List.fromList([
          for (var c = 0; c < H; c++) 0.6 + ((c + l) % 2) * 0.5,
        ]),
      );
    }

    final hiddens = [
      Float64List.fromList([
        for (var c = 0; c < H; c++) ((c * 5 + 2) % 9 - 4) * 0.33,
      ]),
      Float64List.fromList([
        for (var c = 0; c < H; c++) ((c * 7 + 1) % 9 - 4) * 0.29,
      ]),
    ];

    // Per-layer KV row lists: each layer keeps its own K/V history across positions.
    final kRows = [for (var l = 0; l < numLayers; l++) <Float64List>[]];
    final vRows = [for (var l = 0; l < numLayers; l++) <Float64List>[]];
    final qRows = [for (var l = 0; l < numLayers; l++) <Float64List>[]];
    Float64List goldenForward(Float64List hidden0, int t) {
      final hidden = Float64List.fromList(hidden0);
      for (var l = 0; l < numLayers; l++) {
        // Attention.
        final n = rmsNorm(hidden, iGamma[l], eps);
        final qFull = quantizedLinearW4A8(wq[l], qDim, H, n);
        final kFull = quantizedLinearW4A8(wk[l], kvDim, H, n);
        final vFull = quantizedLinearW4A8(wv[l], kvDim, H, n);
        for (var h = 0; h < nH; h++) {
          final sl = Float64List.fromList(qFull.sublist(h * hd, h * hd + hd));
          applyRopeHead(sl, t, theta);
          qFull.setRange(h * hd, h * hd + hd, sl);
        }
        for (var h = 0; h < nKV; h++) {
          final sl = Float64List.fromList(kFull.sublist(h * hd, h * hd + hd));
          applyRopeHead(sl, t, theta);
          kFull.setRange(h * hd, h * hd + hd, sl);
        }
        qRows[l].add(qFull);
        kRows[l].add(kFull);
        vRows[l].add(vFull);
        final attn = causalGqaAttention(
          qRows[l],
          kRows[l],
          vRows[l],
          nH,
          nKV,
          hd,
        )[t];
        final o = quantizedLinearW4A8(wo[l], H, qDim, attn);
        addInPlace(hidden, o);
        // MLP.
        final n2 = rmsNorm(hidden, pGamma[l], eps);
        final g = quantizedLinearW4A8(wGate[l], iSize, H, n2);
        final u = quantizedLinearW4A8(wUp[l], iSize, H, n2);
        final inter = mul(silu(g), u);
        final d = quantizedLinearW4A8(wDown[l], H, iSize, inter);
        addInPlace(hidden, d);
      }
      return hidden;
    }

    final golden0 = goldenForward(hiddens[0], 0);
    final golden1 = goldenForward(hiddens[1], 1);

    final mem = List<int>.filled(0x1000, 0);
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

    // Per-layer weight bases (0x800 stride per layer, 0x100 per matrix).
    int wbQ(int l) => l * 0x800 + 0x000;
    int wbK(int l) => l * 0x800 + 0x100;
    int wbV(int l) => l * 0x800 + 0x200;
    int wbO(int l) => l * 0x800 + 0x300;
    int wbG(int l) => l * 0x800 + 0x400;
    int wbU(int l) => l * 0x800 + 0x500;
    int wbD(int l) => l * 0x800 + 0x600;
    for (var l = 0; l < numLayers; l++) {
      pack(wbQ(l), wq[l], qDim, H);
      pack(wbK(l), wk[l], kvDim, H);
      pack(wbV(l), wv[l], kvDim, H);
      pack(wbO(l), wo[l], H, qDim);
      pack(wbG(l), wGate[l], iSize, H);
      pack(wbU(l), wUp[l], iSize, H);
      pack(wbD(l), wDown[l], H, iSize);
    }
    int memWord(int a) => (a < 0 || a + 3 >= mem.length)
        ? 0
        : mem[a] | (mem[a + 1] << 8) | (mem[a + 2] << 16) | (mem[a + 3] << 24);

    // Per-layer quantized matrices (for the row scales).
    final qmQ = [
      for (var l = 0; l < numLayers; l++) quantizeRowwiseInt4(wq[l], qDim, H),
    ];
    final qmK = [
      for (var l = 0; l < numLayers; l++) quantizeRowwiseInt4(wk[l], kvDim, H),
    ];
    final qmV = [
      for (var l = 0; l < numLayers; l++) quantizeRowwiseInt4(wv[l], kvDim, H),
    ];
    final qmO = [
      for (var l = 0; l < numLayers; l++) quantizeRowwiseInt4(wo[l], H, qDim),
    ];
    final qmG = [
      for (var l = 0; l < numLayers; l++)
        quantizeRowwiseInt4(wGate[l], iSize, H),
    ];
    final qmU = [
      for (var l = 0; l < numLayers; l++) quantizeRowwiseInt4(wUp[l], iSize, H),
    ];
    final qmD = [
      for (var l = 0; l < numLayers; l++)
        quantizeRowwiseInt4(wDown[l], H, iSize),
    ];

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final start = Logic(name: 'start');
    final pos = Logic(name: 'pos', width: maxSeq.bitLength);
    final invN = Logic(name: 'inv_n', width: 16);
    final epsL = Logic(name: 'eps', width: 16);
    final wbq = Logic(name: 'wbq', width: 32);
    final wbk = Logic(name: 'wbk', width: 32);
    final wbv = Logic(name: 'wbv', width: 32);
    final wbo = Logic(name: 'wbo', width: 32);
    final wbg = Logic(name: 'wbg', width: 32);
    final wbu = Logic(name: 'wbu', width: 32);
    final wbd = Logic(name: 'wbd', width: 32);
    final xEn = Logic(name: 'x_en');
    final xInL = Logic(name: 'x_in', width: 16);
    final gEn = Logic(name: 'g_en');
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

    final dut = LoomDecoder(
      hidden: H,
      numHeads: nH,
      numKvHeads: nKV,
      headDim: hd,
      intermediateSize: iSize,
      maxSeq: maxSeq,
      numLayers: numLayers,
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
      ('g_en', gEn),
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

    void setWb(int l) {
      wbq.inject(wbQ(l));
      wbk.inject(wbK(l));
      wbv.inject(wbV(l));
      wbo.inject(wbO(l));
      wbg.inject(wbG(l));
      wbu.inject(wbU(l));
      wbd.inject(wbD(l));
    }

    for (final l in [
      start,
      xEn,
      gEn,
      sqEn,
      skEn,
      svEn,
      soEn,
      sgEn,
      suEn,
      sdEn,
    ]) {
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
    setWb(0);
    setRope(0);
    reset.inject(1);
    Simulator.setMaxSimTime(80000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    Future<void> loadGammas(int l) async {
      for (var i = 0; i < H; i++) {
        gEn.inject(1);
        igaInL.inject(e(iGamma[l][i]));
        pgaInL.inject(e(pGamma[l][i]));
        await clk.nextPosedge;
      }
      gEn.inject(0);
    }

    Future<void> loadScales(int l) async {
      for (var i = 0; i < qDim; i++) {
        sqEn.inject(1);
        sqInL.inject(e(qmQ[l].rowScales[i]));
        await clk.nextPosedge;
      }
      sqEn.inject(0);
      for (var i = 0; i < kvDim; i++) {
        skEn.inject(1);
        skInL.inject(e(qmK[l].rowScales[i]));
        svEn.inject(1);
        svInL.inject(e(qmV[l].rowScales[i]));
        await clk.nextPosedge;
      }
      skEn.inject(0);
      svEn.inject(0);
      for (var i = 0; i < H; i++) {
        soEn.inject(1);
        soInL.inject(e(qmO[l].rowScales[i]));
        await clk.nextPosedge;
      }
      soEn.inject(0);
      for (var i = 0; i < iSize; i++) {
        sgEn.inject(1);
        sgInL.inject(e(qmG[l].rowScales[i]));
        suEn.inject(1);
        suInL.inject(e(qmU[l].rowScales[i]));
        await clk.nextPosedge;
      }
      sgEn.inject(0);
      suEn.inject(0);
      for (var i = 0; i < H; i++) {
        sdEn.inject(1);
        sdInL.inject(e(qmD[l].rowScales[i]));
        await clk.nextPosedge;
      }
      sdEn.inject(0);
    }

    Future<List<double>> runDecoder(Float64List hidden, int t) async {
      while (o('busy').value.toBool()) {
        await clk.nextPosedge;
      }
      setRope(t);
      pos.inject(t);
      // Initial hidden + layer 0 params.
      setWb(0);
      for (var i = 0; i < H; i++) {
        xEn.inject(1);
        xInL.inject(e(hidden[i]));
        await clk.nextPosedge;
      }
      xEn.inject(0);
      await loadGammas(0);
      await loadScales(0);
      start.inject(1);
      await clk.nextPosedge;
      start.inject(0);

      final got = <double>[];
      for (var l = 0; l < numLayers; l++) {
        final isLast = l == numLayers - 1;
        var complete = false;
        var guard = 0;
        while (!complete && guard++ < 400000) {
          await clk.nextNegedge;
          if (o('mem_STB').value.toBool() && o('mem_CYC').value.toBool()) {
            ack.inject(1);
            datMiso.inject(memWord(o('mem_ADR').value.toInt()));
          } else {
            ack.inject(0);
          }
          if (isLast && o('h_valid').value.toBool()) {
            outFp.put(o('h_out').value);
            got.add(outFp.floatingPointValue.toDouble());
          }
          await clk.nextPosedge;
          if (isLast) {
            if (o('done').value.toBool()) complete = true;
          } else {
            // Non-final layer done -> decoder returns to LOAD (busy low).
            if (!o('busy').value.toBool()) complete = true;
          }
        }
        if (!isLast) {
          // Decoder idle in LOAD: stream next layer's params, kick it.
          setWb(l + 1);
          await loadGammas(l + 1);
          await loadScales(l + 1);
          start.inject(1);
          await clk.nextPosedge;
          start.inject(0);
        }
      }
      return got;
    }

    final got0 = await runDecoder(hiddens[0], 0);
    final got1 = await runDecoder(hiddens[1], 1);
    await Simulator.endSimulation();

    void check(String tag, List<double> hw, Float64List g) {
      expect(hw.length, equals(g.length), reason: '$tag count');
      for (var i = 0; i < g.length; i++) {
        expect(
          hw[i],
          closeTo(g[i], 0.1 + g[i].abs() * 0.25),
          reason: '$tag[$i] hw=${hw[i]} g=${g[i]}',
        );
      }
    }

    check('t0', got0, golden0);
    check('t1', got1, golden1);
  });
}
