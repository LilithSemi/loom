// Mixture-of-Experts FFN for the golden reference. Each expert is an ordinary
// gated SwiGLU FFN (the same matmul datapath Loom already runs). A router picks
// the top-k experts per token and their outputs are weight-summed. Hardware-free:
// experts map to weight matrices, routing + top-k is host-side.
import 'dart:math' as math;
import 'dart:typed_data';

/// One expert's gated-FFN weights. `gate`/`up` are [moeInter x hidden] row-major,
/// `down` is [hidden x moeInter].
class MoeExpert {
  final Float64List gate;
  final Float64List up;
  final Float64List down;
  final int moeInter;
  const MoeExpert({
    required this.gate,
    required this.up,
    required this.down,
    required this.moeInter,
  });
}

Float64List _matvec(Float64List m, int rows, int cols, Float64List x) {
  final out = Float64List(rows);
  for (var r = 0; r < rows; r++) {
    var acc = 0.0;
    final o = r * cols;
    for (var c = 0; c < cols; c++) {
      acc += m[o + c] * x[c];
    }
    out[r] = acc;
  }
  return out;
}

/// One expert's gated SwiGLU: `silu(gate.x) * (up.x)` then `down`.
Float64List _expertFfn(MoeExpert e, int hidden, Float64List x) {
  final g = _matvec(e.gate, e.moeInter, hidden, x);
  final u = _matvec(e.up, e.moeInter, hidden, x);
  final inter = Float64List(e.moeInter);
  for (var i = 0; i < e.moeInter; i++) {
    final gi = g[i];
    inter[i] = (gi / (1.0 + math.exp(-gi))) * u[i]; // silu(g)*u
  }
  return _matvec(e.down, hidden, e.moeInter, inter);
}

/// MoE FFN over one token `x` (length [hidden]). `router` is [numExperts x hidden];
/// its softmax picks the [topK] experts, whose (optionally renormalized) weights
/// combine their expert-FFN outputs. Returns the [hidden]-length result.
Float64List moeMlp(
  Float64List x,
  int hidden,
  Float64List router,
  int numExperts,
  List<MoeExpert> experts,
  int topK, {
  bool normTopK = true,
}) {
  assert(experts.length == numExperts);
  final k = topK.clamp(1, numExperts);

  // Router softmax over all experts (numerically stable).
  final logits = _matvec(router, numExperts, hidden, x);
  var maxL = logits[0];
  for (final v in logits) {
    if (v > maxL) maxL = v;
  }
  final probs = Float64List(numExperts);
  var sum = 0.0;
  for (var i = 0; i < numExperts; i++) {
    probs[i] = math.exp(logits[i] - maxL);
    sum += probs[i];
  }
  for (var i = 0; i < numExperts; i++) {
    probs[i] /= sum;
  }

  // Top-k expert indices by probability.
  final order = List<int>.generate(numExperts, (i) => i)
    ..sort((a, b) => probs[b].compareTo(probs[a]));
  final chosen = order.sublist(0, k);

  // Combine weights: the top-k probs, renormalized to sum 1 when normTopK.
  final weights = Float64List(k);
  var wsum = 0.0;
  for (var j = 0; j < k; j++) {
    weights[j] = probs[chosen[j]];
    wsum += weights[j];
  }
  if (normTopK && wsum > 0) {
    for (var j = 0; j < k; j++) {
      weights[j] /= wsum;
    }
  }

  final y = Float64List(hidden);
  for (var j = 0; j < k; j++) {
    final ye = _expertFfn(experts[chosen[j]], hidden, x);
    final w = weights[j];
    for (var h = 0; h < hidden; h++) {
      y[h] += w * ye[h];
    }
  }
  return y;
}
