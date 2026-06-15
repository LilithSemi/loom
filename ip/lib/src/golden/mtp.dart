// Multi-Token Prediction (DeepSeek-V3 style) for the golden reference. Each MTP
// module predicts one token further than the last: it fuses the previous hidden
// state with the embedding of the last drafted token, runs a transformer block,
// and reads out through the shared final-norm + lm_head. Chained, the modules
// draft several tokens per forward, which drives speculative decoding.
//
// The module's attention is single-position: the query attends only to itself,
// so softmax over one score is 1 and the head output is just its value
// projection. This keeps the block a well-defined transformer step without a
// per-module KV cache. It does not affect decode correctness, since speculative
// decoding always falls back to the verified token, so any draft mechanism
// yields exactly-greedy output.
import 'dart:typed_data';

import 'ops.dart';

/// Linear backend: `y = W @ x`, W row-major `[outDim, inDim]`. Matches
/// GoldenRunner.LinearImpl so an MTP forward can route matmuls to the device.
typedef MtpLinear =
    Float64List Function(Float64List w, int outDim, int inDim, Float64List x);

/// Decoded weights for one MTP module: the two input norms, the fusion
/// projection `eh_proj` (`[hidden, 2*hidden]`), and a dense transformer block.
class MtpModuleWeights {
  final Float64List enorm; // [hidden]
  final Float64List hnorm; // [hidden]
  final Float64List ehProj; // [hidden, 2*hidden]

  final Float64List inputNorm;
  final Float64List qProj;
  final Float64List kProj;
  final Float64List vProj;
  final Float64List oProj;
  final Float64List postNorm;
  final Float64List gate;
  final Float64List up;
  final Float64List down;

  final int numHeads;
  final int numKvHeads;
  final int headDim;
  final int intermediate;

  const MtpModuleWeights({
    required this.enorm,
    required this.hnorm,
    required this.ehProj,
    required this.inputNorm,
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
    required this.postNorm,
    required this.gate,
    required this.up,
    required this.down,
    required this.numHeads,
    required this.numKvHeads,
    required this.headDim,
    required this.intermediate,
  });
}

/// Runs one MTP module. [prevHidden] is the previous depth's hidden state at the
/// draft position (`hidden`-length), [embedRow] is the embedding of the last
/// drafted token. Returns this module's hidden state. Feed it through the shared
/// final-norm + lm_head for the drafted-token logits, and into the next module.
Float64List mtpModuleForward(
  Float64List prevHidden,
  Float64List embedRow,
  MtpModuleWeights w,
  int hidden,
  double eps, {
  MtpLinear lin = linear,
}) {
  // Fuse: combined = eh_proj @ concat(hnorm(prevHidden), enorm(embed)).
  final hn = rmsNorm(prevHidden, w.hnorm, eps);
  final en = rmsNorm(embedRow, w.enorm, eps);
  final cat = Float64List(2 * hidden)
    ..setRange(0, hidden, hn)
    ..setRange(hidden, 2 * hidden, en);
  final combined = lin(w.ehProj, hidden, 2 * hidden, cat);

  // Transformer block (single position). Attention degenerates to the value
  // projection (softmax over one score = 1), grouped per GQA head.
  final n1 = rmsNorm(combined, w.inputNorm, eps);
  final v = lin(w.vProj, w.numKvHeads * w.headDim, hidden, n1);
  final attn = Float64List(w.numHeads * w.headDim);
  final group = w.numHeads ~/ w.numKvHeads;
  for (var h = 0; h < w.numHeads; h++) {
    final kv = h ~/ group;
    for (var d = 0; d < w.headDim; d++) {
      attn[h * w.headDim + d] = v[kv * w.headDim + d];
    }
  }
  final o = lin(w.oProj, hidden, w.numHeads * w.headDim, attn);
  final h1 = Float64List(hidden);
  for (var i = 0; i < hidden; i++) {
    h1[i] = combined[i] + o[i];
  }

  // Gated-SwiGLU FFN + residual.
  final n2 = rmsNorm(h1, w.postNorm, eps);
  final g = lin(w.gate, w.intermediate, hidden, n2);
  final u = lin(w.up, w.intermediate, hidden, n2);
  final act = mul(silu(g), u);
  final d = lin(w.down, hidden, w.intermediate, act);
  final h2 = Float64List(hidden);
  for (var i = 0; i < hidden; i++) {
    h2[i] = h1[i] + d[i];
  }
  return h2;
}
