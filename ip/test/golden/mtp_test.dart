import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

Float64List _f(List<double> v) => Float64List.fromList(v);

void main() {
  test(
    'mtpModuleForward fuses prev-hidden and embed via eh_proj (residuals pass through)',
    () {
      // hidden=2, 1 head, headDim 2, intermediate 2. With vProj=0 the attention
      // output is 0 (h1 = combined) and with gate=0 the FFN output is 0
      // (h2 = h1). So the module reduces to combined = eh_proj @ concat(hn, en).
      // enorm=hnorm=ones and eps=0 make hn=en=[1,1] for inputs [1,1], so
      // combined = row-sums of eh_proj = [1+2+3+4, 5+6+7+8] = [10, 26].
      const h = 2;
      final w = MtpModuleWeights(
        enorm: _f([1, 1]),
        hnorm: _f([1, 1]),
        ehProj: _f([1, 2, 3, 4, 5, 6, 7, 8]), // [2 x 4]
        inputNorm: _f([1, 1]),
        qProj: _f([0.3, -0.2, 0.1, 0.5]),
        kProj: _f([0.1, 0.2, -0.3, 0.4]),
        vProj: _f([0, 0, 0, 0]), // -> attention output 0
        oProj: _f([0.7, -0.1, 0.2, 0.9]),
        postNorm: _f([1, 1]),
        gate: _f([0, 0, 0, 0]), // -> silu(0)*u = 0 -> FFN output 0
        up: _f([0.4, 0.5, 0.6, 0.7]),
        down: _f([0.2, 0.1, 0.3, 0.8]),
        numHeads: 1,
        numKvHeads: 1,
        headDim: 2,
        intermediate: 2,
      );
      final out = mtpModuleForward(_f([1, 1]), _f([1, 1]), w, h, 0.0);
      expect(out[0], closeTo(10.0, 1e-12));
      expect(out[1], closeTo(26.0, 1e-12));
    },
  );

  test(
    'mtpModuleForward attention is GQA-grouped (value broadcast across query heads)',
    () {
      // 2 query heads share 1 kv head (group=2). vProj=identity-ish, gate=0, and
      // inputNorm=ones with eps=0. combined for input [1,1] with eh_proj selecting
      // hnorm(prev) only is [1,1]; n1 = rmsNorm([1,1]) = [1,1]; v = vProj @ [1,1].
      // Both query heads must read the same kv head's value, then o_proj maps back.
      const h = 2;
      final w = MtpModuleWeights(
        enorm: _f([1, 1]),
        hnorm: _f([1, 1]),
        // eh_proj = [I | 0] so combined = hnorm(prev) = [1,1] for prev [1,1].
        ehProj: _f([1, 0, 0, 0, 0, 1, 0, 0]),
        inputNorm: _f([1, 1]),
        qProj: _f([0, 0, 0, 0]),
        kProj: _f([0, 0, 0, 0]),
        // vProj is [numKvHeads*headDim x hidden] = [2 x 2]; v = [1,1] @ this rows.
        vProj: _f([1, 0, 0, 1]), // v = [1, 1]
        // oProj is [hidden x numHeads*headDim] = [2 x 4]; attn = [v; v] = [1,1,1,1].
        // Pick oProj rows so o = [sum row0, sum row1].
        oProj: _f([1, 1, 1, 1, 2, 2, 2, 2]), // o = [4, 8]
        postNorm: _f([1, 1]),
        gate: _f([0, 0, 0, 0]),
        up: _f([1, 1, 1, 1]),
        down: _f([1, 1, 1, 1]),
        numHeads: 2,
        numKvHeads: 1,
        headDim: 2,
        intermediate: 2,
      );
      // h1 = combined + o = [1,1] + [4,8] = [5, 9]; FFN is 0 so h2 = h1.
      final out = mtpModuleForward(_f([1, 1]), _f([1, 1]), w, h, 0.0);
      expect(out[0], closeTo(5.0, 1e-12));
      expect(out[1], closeTo(9.0, 1e-12));
    },
  );
}
