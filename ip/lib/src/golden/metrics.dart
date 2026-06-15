// Quality metrics for the quant study: KL divergence between fp and quant logits.
import 'dart:math' as math;

/// KL(softmax(pLogits) || softmax(qLogits)) in nats. p is the fp reference
/// distribution, q the quantized one. 0.0 iff the distributions match. Softmax
/// makes it invariant to a constant logit offset. Numerically stable (max-shift).
double klDivergence(List<double> pLogits, List<double> qLogits) {
  final n = pLogits.length;
  double maxOf(List<double> v) {
    var m = v[0];
    for (final x in v) if (x > m) m = x;
    return m;
  }

  final pm = maxOf(pLogits), qm = maxOf(qLogits);
  var pSum = 0.0, qSum = 0.0;
  final pe = List<double>.filled(n, 0.0), qe = List<double>.filled(n, 0.0);
  for (var i = 0; i < n; i++) {
    pe[i] = math.exp(pLogits[i] - pm);
    pSum += pe[i];
    qe[i] = math.exp(qLogits[i] - qm);
    qSum += qe[i];
  }
  var kl = 0.0;
  for (var i = 0; i < n; i++) {
    final p = pe[i] / pSum;
    if (p > 0) kl += p * math.log(p / (qe[i] / qSum));
  }
  return kl;
}
