import 'dart:math' as math;
import 'dart:typed_data';

import 'ops.dart';

/// Applies Rotary Position Embedding (RoPE) to a single attention head in-place.
///
/// Uses the HuggingFace Llama half-split convention: the two paired coordinates
/// are head[j] and head[j+half], NOT the interleaved (j, j+1) pairing used by
/// llama.cpp/GGUF.
///
/// [head] must have even length (the head dimension). Throws [ArgumentError]
/// otherwise.
void applyRopeHead(Float64List head, int pos, double theta) {
  final hd = head.length;
  if (hd % 2 != 0) {
    throw ArgumentError('head.length must be even, got $hd');
  }
  final half = hd ~/ 2;
  for (var j = 0; j < half; j++) {
    final invFreq = 1.0 / math.pow(theta, (2 * j) / hd);
    final angle = pos * invFreq;
    final c = math.cos(angle);
    final s = math.sin(angle);
    final x1 = head[j];
    final x2 = head[j + half];
    head[j] = x1 * c - x2 * s;
    head[j + half] = x2 * c + x1 * s;
  }
}

/// Causal grouped-query attention (GQA).
///
/// [q] is a list of T rows, each of length numHeads*headDim (RoPE already
/// applied per head by the caller).
/// [k] and [v] are lists of T rows, each of length numKvHeads*headDim.
///
/// Returns a list of T rows, each of length numHeads*headDim, containing the
/// per-head attention outputs concatenated (the input to o_proj).
///
/// GQA mapping: query head h uses kv head h ~/ (numHeads ~/ numKvHeads).
///
/// Throws [ArgumentError] if:
///   - numHeads is not divisible by numKvHeads
///   - q/k/v lists differ in length T
///   - any q row has length != numHeads*headDim
///   - any k or v row has length != numKvHeads*headDim
List<Float64List> causalGqaAttention(
  List<Float64List> q,
  List<Float64List> k,
  List<Float64List> v,
  int numHeads,
  int numKvHeads,
  int headDim,
) {
  if (numHeads % numKvHeads != 0) {
    throw ArgumentError(
      'numHeads ($numHeads) must be divisible by numKvHeads ($numKvHeads)',
    );
  }
  final t = q.length;
  if (k.length != t || v.length != t) {
    throw ArgumentError(
      'q, k, v must all have the same length T. '
      'Got q=${q.length}, k=${k.length}, v=${v.length}',
    );
  }
  final qRowLen = numHeads * headDim;
  final kvRowLen = numKvHeads * headDim;
  for (var i = 0; i < t; i++) {
    if (q[i].length != qRowLen) {
      throw ArgumentError(
        'q[$i].length (${q[i].length}) != numHeads*headDim ($qRowLen)',
      );
    }
    if (k[i].length != kvRowLen) {
      throw ArgumentError(
        'k[$i].length (${k[i].length}) != numKvHeads*headDim ($kvRowLen)',
      );
    }
    if (v[i].length != kvRowLen) {
      throw ArgumentError(
        'v[$i].length (${v[i].length}) != numKvHeads*headDim ($kvRowLen)',
      );
    }
  }

  final group = numHeads ~/ numKvHeads;
  final scale = 1.0 / math.sqrt(headDim.toDouble());
  final result = List<Float64List>.generate(t, (_) => Float64List(qRowLen));

  for (var qPos = 0; qPos < t; qPos++) {
    for (var h = 0; h < numHeads; h++) {
      final kvHead = h ~/ group;
      final qOffset = h * headDim;
      final kvOffset = kvHead * headDim;

      // Compute causal attention scores for head h at query position qPos.
      final scores = Float64List(qPos + 1);
      for (var s = 0; s <= qPos; s++) {
        var dot = 0.0;
        for (var d = 0; d < headDim; d++) {
          dot += q[qPos][qOffset + d] * k[s][kvOffset + d];
        }
        scores[s] = dot * scale;
      }

      final weights = softmax(scores);

      // Weighted sum over values.
      final outOffset = h * headDim;
      for (var s = 0; s <= qPos; s++) {
        final w = weights[s];
        for (var d = 0; d < headDim; d++) {
          result[qPos][outOffset + d] += w * v[s][kvOffset + d];
        }
      }
    }
  }

  return result;
}
