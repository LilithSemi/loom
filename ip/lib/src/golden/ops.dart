import 'dart:math' as math;
import 'dart:typed_data';

double _sigmoid(double v) => 1.0 / (1.0 + math.exp(-v));

Float64List linear(
  Float64List weight,
  int outDim,
  int inDim,
  Float64List x, {
  Float64List? bias,
}) {
  if (x.length != inDim) {
    throw ArgumentError('x.length (${x.length}) != inDim ($inDim)');
  }
  if (weight.length != outDim * inDim) {
    throw ArgumentError(
      'weight.length (${weight.length}) != outDim*inDim (${outDim * inDim})',
    );
  }
  if (bias != null && bias.length != outDim) {
    throw ArgumentError('bias.length (${bias.length}) != outDim ($outDim)');
  }
  final y = Float64List(outDim);
  for (var o = 0; o < outDim; o++) {
    var sum = 0.0;
    final offset = o * inDim;
    for (var i = 0; i < inDim; i++) {
      sum += weight[offset + i] * x[i];
    }
    y[o] = bias != null ? sum + bias[o] : sum;
  }
  return y;
}

Float64List rmsNorm(Float64List x, Float64List gamma, double eps) {
  if (gamma.length != x.length) {
    throw ArgumentError(
      'gamma.length (${gamma.length}) != x.length (${x.length})',
    );
  }
  var ms = 0.0;
  for (final v in x) {
    ms += v * v;
  }
  ms /= x.length;
  final inv = 1.0 / math.sqrt(ms + eps);
  final y = Float64List(x.length);
  for (var i = 0; i < x.length; i++) {
    y[i] = x[i] * inv * gamma[i];
  }
  return y;
}

Float64List silu(Float64List x) {
  final y = Float64List(x.length);
  for (var i = 0; i < x.length; i++) {
    y[i] = x[i] * _sigmoid(x[i]);
  }
  return y;
}

/// LayerNorm: normalize [x] to zero mean / unit variance (biased, /N) then scale
/// by [gamma] and shift by [beta]. Used by Vision Transformer towers (which use
/// LayerNorm rather than the LLM's RMSNorm).
Float64List layerNorm(
  Float64List x,
  Float64List gamma,
  Float64List beta,
  double eps,
) {
  if (gamma.length != x.length || beta.length != x.length) {
    throw ArgumentError('gamma/beta length must equal x.length (${x.length})');
  }
  var mean = 0.0;
  for (final v in x) {
    mean += v;
  }
  mean /= x.length;
  var variance = 0.0;
  for (final v in x) {
    final d = v - mean;
    variance += d * d;
  }
  variance /= x.length;
  final inv = 1.0 / math.sqrt(variance + eps);
  final y = Float64List(x.length);
  for (var i = 0; i < x.length; i++) {
    y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
  }
  return y;
}

/// GELU (tanh approximation, matching `gelu_pytorch_tanh` used by SigLIP and
/// most ViTs): `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
Float64List gelu(Float64List x) {
  const c = 0.7978845608028654; // sqrt(2/pi)
  final y = Float64List(x.length);
  for (var i = 0; i < x.length; i++) {
    final v = x[i];
    final inner = c * (v + 0.044715 * v * v * v);
    y[i] = 0.5 * v * (1.0 + _tanh(inner));
  }
  return y;
}

double _tanh(double x) {
  // math.tanh is not in dart:math. Derive from exp for a stable form.
  if (x > 20) return 1.0;
  if (x < -20) return -1.0;
  final e2 = math.exp(2 * x);
  return (e2 - 1) / (e2 + 1);
}

Float64List softmax(Float64List x) {
  var maxVal = x[0];
  for (final v in x) {
    if (v > maxVal) maxVal = v;
  }
  final y = Float64List(x.length);
  var sum = 0.0;
  for (var i = 0; i < x.length; i++) {
    y[i] = math.exp(x[i] - maxVal);
    sum += y[i];
  }
  for (var i = 0; i < y.length; i++) {
    y[i] /= sum;
  }
  return y;
}

void addInPlace(Float64List a, Float64List b) {
  if (a.length != b.length) {
    throw ArgumentError('a.length (${a.length}) != b.length (${b.length})');
  }
  for (var i = 0; i < a.length; i++) {
    a[i] += b[i];
  }
}

Float64List mul(Float64List a, Float64List b) {
  if (a.length != b.length) {
    throw ArgumentError('a.length (${a.length}) != b.length (${b.length})');
  }
  final y = Float64List(a.length);
  for (var i = 0; i < a.length; i++) {
    y[i] = a[i] * b[i];
  }
  return y;
}
