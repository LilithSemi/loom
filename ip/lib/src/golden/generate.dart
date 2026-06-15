/// Computes logits for the given context tokens.
typedef ForwardFn = List<double> Function(List<int> tokens);

/// Greedy (argmax) autoregressive decoding decoupled from any particular model.
///
/// [forward] maps a context to logits for the next token. Generation appends
/// the argmax token, feeds it back, and repeats until [maxNewTokens] are
/// produced or a token in [stopTokens] is emitted (the stop token is included
/// in the result). Each new token is reported through [onToken] as it is
/// produced, enabling streaming output.
List<int> greedyGenerate(
  ForwardFn forward,
  List<int> prompt, {
  required int maxNewTokens,
  Set<int> stopTokens = const {},
  void Function(int token)? onToken,
}) {
  final context = List<int>.from(prompt);
  final generated = <int>[];
  for (var i = 0; i < maxNewTokens; i++) {
    final logits = forward(context);
    final next = _argmax(logits);
    context.add(next);
    generated.add(next);
    onToken?.call(next);
    if (stopTokens.contains(next)) break;
  }
  return generated;
}

int _argmax(List<double> values) {
  var best = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}
