import 'dart:typed_data';

/// Loom Nano: a tiny char/bit-level language model designed to run ENTIRELY on
/// the silicon accelerator. v0 is an order-N binary Markov model.
///
/// The model predicts the next bit from the previous [contextBits] bits. The
/// context (a window of the last [contextBits] bits) indexes one of
/// `2^contextBits` states. The prediction is `argmax(W . onehot(state))` where
/// `W` is an int8 weight matrix of shape `[vocab=2][nStates]`.
///
/// Because the activation is one-hot, `W . onehot(state)` selects column
/// `state` of `W`. This is exactly the int8 matrix-vector product the
/// [LoomAccelerator] already computes (rows = vocab = 2, cols = nStates), so the
/// whole forward pass is a single on-chip matmul + argmax.
class NanoConfig {
  /// Number of previous bits used as context.
  final int contextBits;

  const NanoConfig({this.contextBits = 3});

  /// Vocabulary size: binary.
  int get vocab => 2;

  /// Number of distinct context states (= 2^contextBits).
  int get nStates => 1 << contextBits;

  /// Context bit-mask.
  int get contextMask => nStates - 1;
}

/// A trained Loom Nano model.
class NanoModel {
  final NanoConfig config;

  /// Weight matrix, row-major `[vocab][nStates]`: `w[r * nStates + c]` is the
  /// logit for output bit `r` given context state `c`. int8 range.
  final Int8List weights;

  NanoModel(this.config, this.weights)
    : assert(weights.length == config.vocab * config.nStates);

  /// High/low logit values used by the trainer. Kept well inside int8 range.
  static const int _hi = 100;
  static const int _lo = -100;

  /// Trains a model to reproduce [target] (a list of 0/1 bits) under greedy
  /// argmax decoding. For each context state we pick the most frequent next bit
  /// observed in [target] and bias the weights toward it.
  factory NanoModel.train(NanoConfig config, List<int> target) {
    if (target.length <= config.contextBits) {
      throw ArgumentError(
        'target must be longer than contextBits (${config.contextBits})',
      );
    }
    // counts[state][bit]
    final counts = List.generate(config.nStates, (_) => <int>[0, 0]);
    var ctx = 0;
    for (var i = 0; i < config.contextBits; i++) {
      ctx = ((ctx << 1) | (target[i] & 1)) & config.contextMask;
    }
    for (var i = config.contextBits; i < target.length; i++) {
      final bit = target[i] & 1;
      counts[ctx][bit]++;
      ctx = ((ctx << 1) | bit) & config.contextMask;
    }

    final w = Int8List(config.vocab * config.nStates);
    for (var s = 0; s < config.nStates; s++) {
      // Pick the majority next bit for this state (ties -> 0).
      final pick = counts[s][1] > counts[s][0] ? 1 : 0;
      w[1 * config.nStates + s] = pick == 1 ? _hi : _lo;
      w[0 * config.nStates + s] = pick == 0 ? _hi : _lo;
    }
    return NanoModel(config, w);
  }

  /// Logits for a given context state: column [state] of the weight matrix.
  List<int> logits(int state) {
    final s = state & config.contextMask;
    return [
      for (var r = 0; r < config.vocab; r++) weights[r * config.nStates + s],
    ];
  }

  /// Greedy next bit for a context state.
  int next(int state) {
    final l = logits(state);
    return l[1] > l[0] ? 1 : 0;
  }

  /// Greedily generates [n] bits starting from context [startState], feeding
  /// each produced bit back into the context window.
  List<int> generate(int startState, int n) {
    var ctx = startState & config.contextMask;
    final out = <int>[];
    for (var i = 0; i < n; i++) {
      final bit = next(ctx);
      out.add(bit);
      ctx = ((ctx << 1) | bit) & config.contextMask;
    }
    return out;
  }
}
