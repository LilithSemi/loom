// LoomSeq: whole-model forward through one shared LoomFpLinear engine, with model
// params (gammas, weight/scale offsets) baked in as constructor constants and
// per-row scales fetched from flash on the fly. Checked against golden's full
// forward (embed -> layers -> final norm -> lm_head -> argmax) by emitted token id, at t=0 and t=1.

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:loom/src/golden/attention.dart';
import 'package:loom/src/golden/ops.dart';
import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/loom_seq.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test(
    'baked-param LoomSeq emits next token matching golden at t=0 then t=1',
    () async {
      const H = 8, nH = 2, nKV = 1, hd = 4, iSize = 16, maxSeq = 4;
      const numLayers = 2, vocab = 32;
      const qDim = nH * hd, kvDim = nKV * hd, half = hd ~/ 2;
      const peR = 2, peC = 2;
      const eps = 0.01, theta = 10000.0;
      final invSqrtHd = 1.0 / math.sqrt(hd.toDouble());

      Float64List mat(int rows, int cols, int seed) {
        final m = Float64List(rows * cols);
        for (var r = 0; r < rows; r++) {
          for (var c = 0; c < cols; c++) {
            m[r * cols + c] = ((r * 5 + c * 3 + seed) % 13 - 6) * 0.17;
          }
        }
        return m;
      }

      final fp = FloatingPoint16();
      int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
      final rFp = FloatingPoint16();
      double r16(double d) {
        rFp.put(fp.valuePopulator().ofDouble(d).value);
        return rFp.floatingPointValue.toDouble();
      }

      final embed = mat(vocab, H, 11);
      Float64List embRow(int tok) => Float64List.fromList([
        for (var i = 0; i < H; i++) r16(embed[tok * H + i]),
      ]);

      final wq = <Float64List>[], wk = <Float64List>[], wv = <Float64List>[];
      final wo = <Float64List>[], wG = <Float64List>[], wU = <Float64List>[];
      final wD = <Float64List>[];
      final iGamma = <Float64List>[], pGamma = <Float64List>[];
      for (var l = 0; l < numLayers; l++) {
        final s = l * 13;
        wq.add(mat(qDim, H, 1 + s));
        wk.add(mat(kvDim, H, 4 + s));
        wv.add(mat(kvDim, H, 7 + s));
        wo.add(mat(H, qDim, 2 + s));
        wG.add(mat(iSize, H, 3 + s));
        wU.add(mat(iSize, H, 6 + s));
        wD.add(mat(H, iSize, 9 + s));
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
      final fGamma = Float64List.fromList([
        for (var c = 0; c < H; c++) 0.7 + (c % 3) * 0.3,
      ]);
      final wCls = mat(vocab, H, 21);

      final kRows = [for (var l = 0; l < numLayers; l++) <Float64List>[]];
      final vRows = [for (var l = 0; l < numLayers; l++) <Float64List>[]];
      final qRows = [for (var l = 0; l < numLayers; l++) <Float64List>[]];
      int goldenForward(int token, int t) {
        final hidden = embRow(token);
        for (var l = 0; l < numLayers; l++) {
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
          final n2 = rmsNorm(hidden, pGamma[l], eps);
          final g = quantizedLinearW4A8(wG[l], iSize, H, n2);
          final u = quantizedLinearW4A8(wU[l], iSize, H, n2);
          final inter = mul(silu(g), u);
          final d = quantizedLinearW4A8(wD[l], H, iSize, inter);
          addInPlace(hidden, d);
        }
        final f = rmsNorm(hidden, fGamma, eps);
        final logits = quantizedLinearW4A8(wCls, vocab, H, f);
        var best = 0;
        for (var i = 1; i < vocab; i++) {
          if (logits[i] > logits[best]) best = i;
        }
        return best;
      }

      const tok0 = 3, tok1 = 10;
      final gold0 = goldenForward(tok0, 0);
      final gold1 = goldenForward(tok1, 1);

      // Scale tables: one fp16 per 32-bit word, low16 bits used.
      const wbEmbed = 0x0000, wbCls = 0x4000;
      final mem = List<int>.filled(0x8000, 0);
      for (var tok = 0; tok < vocab; tok++) {
        for (var i = 0; i < H; i++) {
          final a = wbEmbed + (tok * H + i) * 4;
          final v = e(embed[tok * H + i]);
          mem[a] = v & 0xFF;
          mem[a + 1] = (v >> 8) & 0xFF;
        }
      }
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

      void packScales(int base, Float64List scales) {
        for (var i = 0; i < scales.length; i++) {
          final a = base + i * 4;
          final v = e(scales[i]);
          mem[a] = v & 0xFF;
          mem[a + 1] = (v >> 8) & 0xFF;
        }
      }

      int lbase(int l) => 0x1000 + l * 0x800;
      int wbQ(int l) => lbase(l) + 0x000;
      int wbK(int l) => lbase(l) + 0x100;
      int wbV(int l) => lbase(l) + 0x200;
      int wbO(int l) => lbase(l) + 0x300;
      int wbG(int l) => lbase(l) + 0x400;
      int wbU(int l) => lbase(l) + 0x500;
      int wbD(int l) => lbase(l) + 0x600;
      // Scale-table flash offsets: 0x100 per table, [q,k,v,o,g,u,d] per layer.
      int sb(int l, int k) => 0x5000 + (l * 7 + k) * 0x100;
      final sbCls = 0x5000 + numLayers * 7 * 0x100;

      final qmQ = [
        for (var l = 0; l < numLayers; l++) quantizeRowwiseInt4(wq[l], qDim, H),
      ];
      final qmK = [
        for (var l = 0; l < numLayers; l++)
          quantizeRowwiseInt4(wk[l], kvDim, H),
      ];
      final qmV = [
        for (var l = 0; l < numLayers; l++)
          quantizeRowwiseInt4(wv[l], kvDim, H),
      ];
      final qmO = [
        for (var l = 0; l < numLayers; l++) quantizeRowwiseInt4(wo[l], H, qDim),
      ];
      final qmG = [
        for (var l = 0; l < numLayers; l++)
          quantizeRowwiseInt4(wG[l], iSize, H),
      ];
      final qmU = [
        for (var l = 0; l < numLayers; l++)
          quantizeRowwiseInt4(wU[l], iSize, H),
      ];
      final qmD = [
        for (var l = 0; l < numLayers; l++)
          quantizeRowwiseInt4(wD[l], H, iSize),
      ];
      final qmCls = quantizeRowwiseInt4(wCls, vocab, H);

      for (var l = 0; l < numLayers; l++) {
        pack(wbQ(l), wq[l], qDim, H);
        pack(wbK(l), wk[l], kvDim, H);
        pack(wbV(l), wv[l], kvDim, H);
        pack(wbO(l), wo[l], H, qDim);
        pack(wbG(l), wG[l], iSize, H);
        pack(wbU(l), wU[l], iSize, H);
        pack(wbD(l), wD[l], H, iSize);
        packScales(sb(l, 0), qmQ[l].rowScales);
        packScales(sb(l, 1), qmK[l].rowScales);
        packScales(sb(l, 2), qmV[l].rowScales);
        packScales(sb(l, 3), qmO[l].rowScales);
        packScales(sb(l, 4), qmG[l].rowScales);
        packScales(sb(l, 5), qmU[l].rowScales);
        packScales(sb(l, 6), qmD[l].rowScales);
      }
      pack(wbCls, wCls, vocab, H);
      packScales(sbCls, qmCls.rowScales);
      int memWord(int a) => (a < 0 || a + 3 >= mem.length)
          ? 0
          : mem[a] |
                (mem[a + 1] << 8) |
                (mem[a + 2] << 16) |
                (mem[a + 3] << 24);

      final inputGamma = [
        for (var l = 0; l < numLayers; l++)
          for (var i = 0; i < H; i++) e(iGamma[l][i]),
      ];
      final postGamma = [
        for (var l = 0; l < numLayers; l++)
          for (var i = 0; i < H; i++) e(pGamma[l][i]),
      ];
      final finalGamma = [for (var i = 0; i < H; i++) e(fGamma[i])];
      final weightBase = [
        for (var l = 0; l < numLayers; l++) ...[
          wbQ(l),
          wbK(l),
          wbV(l),
          wbO(l),
          wbG(l),
          wbU(l),
          wbD(l),
        ],
      ];
      final scaleBase = [
        for (var l = 0; l < numLayers; l++)
          for (var k = 0; k < 7; k++) sb(l, k),
      ];

      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      final start = Logic(name: 'start');
      final pos = Logic(name: 'pos', width: maxSeq.bitLength);
      final invN = Logic(name: 'inv_n', width: 16);
      final epsL = Logic(name: 'eps', width: 16);
      final inTok = Logic(name: 'in_token', width: vocab.bitLength);
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

      final dut = LoomSeq(
        hidden: H,
        numHeads: nH,
        numKvHeads: nKV,
        headDim: hd,
        intermediateSize: iSize,
        maxSeq: maxSeq,
        numLayers: numLayers,
        vocab: vocab,
        inputGamma: inputGamma,
        postGamma: postGamma,
        finalGamma: finalGamma,
        weightBase: weightBase,
        clsWeightBase: wbCls,
        embedBase: wbEmbed,
        scaleBase: scaleBase,
        clsScaleBase: sbCls,
      );
      for (final (p, s) in [
        ('clk', clk),
        ('reset', reset),
        ('start', start),
        ('pos', pos),
        ('inv_n', invN),
        ('eps', epsL),
        ('in_token', inTok),
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

      start.inject(0);
      pos.inject(0);
      inTok.inject(0);
      invN.inject(e(1.0 / H));
      epsL.inject(e(eps));
      setRope(0);
      reset.inject(1);
      Simulator.setMaxSimTime(120000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;

      Future<int> runForward(int token, int t) async {
        while (o('busy').value.toBool()) {
          await clk.nextPosedge;
        }
        setRope(t);
        pos.inject(t);
        inTok.inject(token);
        start.inject(1);
        await clk.nextPosedge;
        start.inject(0);

        var tok = -1;
        var guard = 0;
        while (guard++ < 1000000) {
          await clk.nextNegedge;
          if (o('mem_STB').value.toBool() && o('mem_CYC').value.toBool()) {
            ack.inject(1);
            datMiso.inject(memWord(o('mem_ADR').value.toInt()));
          } else {
            ack.inject(0);
          }
          await clk.nextPosedge;
          if (o('done').value.toBool()) {
            tok = o('token').value.toInt();
            break;
          }
        }
        return tok;
      }

      final got0 = await runForward(tok0, 0);
      final got1 = await runForward(tok1, 1);
      await Simulator.endSimulation();

      expect(got0, equals(gold0), reason: 't0 hw=$got0 golden=$gold0');
      expect(got1, equals(gold1), reason: 't1 hw=$got1 golden=$gold1');
    },
  );
}
