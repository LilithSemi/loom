library;

import 'dart:typed_data';

import '../frontend/hf_config.dart';
import '../ir/model_graph.dart';
import '../ir/tensor.dart';
import 'weight_store.dart';

/// Ingests the original llama2.c checkpoint format (Karpathy's `tinyllamas`,
/// e.g. `stories260K.bin`) into Loom's IR: a [ModelGraph] plus a [WeightStore]
/// that exposes the weights under the SAME HuggingFace tensor names
/// [bindWeights] expects, so the rest of the pipeline is format-agnostic.
///
/// The "legacy" (version 0) format is a 7 x int32 little-endian header
/// `{dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len}`
/// followed by fp32 weights in a fixed order. A positive `vocab_size` signals a
/// classifier tied to the token-embedding table (no separate `wcls`).
///
/// RoPE convention: llama2.c rotates INTERLEAVED coordinate pairs `(2p, 2p+1)`,
/// whereas Loom's golden uses the HuggingFace HALF-SPLIT pairing `(j, j+half)`.
/// The two are reconciled by the standard HF row permutation applied to
/// `q_proj`/`k_proj` here, so the (unchanged) half-split golden reproduces
/// llama2.c's output exactly.
class Llama2cCheckpoint {
  final ModelGraph graph;
  final WeightStore store;

  /// The model's trained context length (llama2.c header `seq_len`). The runtime
  /// uses it as the default generation cap (positions beyond it are untrained).
  final int maxSeq;

  const Llama2cCheckpoint(this.graph, this.store, {this.maxSeq = 0});

  factory Llama2cCheckpoint.parse(Uint8List bytes, {String? name}) {
    final bd = ByteData.sublistView(bytes);
    var p = 0;
    int i32() {
      final v = bd.getInt32(p, Endian.little);
      p += 4;
      return v;
    }

    final dim = i32();
    final hiddenDim = i32();
    final nLayers = i32();
    final nHeads = i32();
    final nKvHeads = i32();
    final rawVocab = i32();
    final seqLen = i32();

    // Negative vocab => separate (untied) classifier weights present.
    final tied = rawVocab > 0;
    final vocab = rawVocab.abs();
    final headSize = dim ~/ nHeads;

    // Decode the fp32 payload (everything after the 28-byte header).
    final nFloats = (bytes.length - 28) ~/ 4;
    final floats = Float32List(nFloats);
    for (var k = 0; k < nFloats; k++) {
      floats[k] = bd.getFloat32(28 + k * 4, Endian.little);
    }

    var cur = 0;
    Float32List take(int n) {
      final out = Float32List.fromList(
        Float32List.sublistView(floats, cur, cur + n),
      );
      cur += n;
      return out;
    }

    // Fixed llama2.c weight order.
    final tokenEmb = take(vocab * dim); // [vocab, dim]
    final rmsAtt = take(nLayers * dim); // per layer [dim]
    final wq = take(nLayers * (nHeads * headSize) * dim); // [nH*hs, dim]
    final wk = take(nLayers * (nKvHeads * headSize) * dim); // [nKV*hs, dim]
    final wv = take(nLayers * (nKvHeads * headSize) * dim);
    final wo = take(nLayers * dim * (nHeads * headSize)); // [dim, nH*hs]
    final rmsFfn = take(nLayers * dim);
    final w1 = take(nLayers * hiddenDim * dim); // gate [hidden, dim]
    final w2 = take(nLayers * dim * hiddenDim); // down [dim, hidden]
    final w3 = take(nLayers * hiddenDim * dim); // up   [hidden, dim]
    final rmsFinal = take(dim); // [dim]
    // Remaining floats are freq_cis_real/imag (+ optional wcls). Unused here
    // because RoPE is recomputed from theta and the classifier is tied.

    final qDim = nHeads * headSize;
    final kvDim = nKvHeads * headSize;

    final tensors = <String, TensorView>{};
    void put(String n, List<int> shape, Float32List data) {
      tensors[n] = TensorView(
        name: n,
        shape: shape,
        dtype: TensorDType.f32,
        bytes: ByteData.sublistView(data),
      );
    }

    Float32List slice(Float32List src, int rowStart, int rows, int cols) =>
        Float32List.fromList(
          Float32List.sublistView(
            src,
            rowStart * cols,
            (rowStart + rows) * cols,
          ),
        );

    put('model.embed_tokens.weight', [vocab, dim], tokenEmb);
    put('model.norm.weight', [dim], rmsFinal);

    for (var l = 0; l < nLayers; l++) {
      put('model.layers.$l.input_layernorm.weight', [
        dim,
      ], slice(rmsAtt, l, 1, dim));
      put(
        'model.layers.$l.self_attn.q_proj.weight',
        [qDim, dim],
        _permuteHeads(slice(wq, l * qDim, qDim, dim), nHeads, headSize, dim),
      );
      put(
        'model.layers.$l.self_attn.k_proj.weight',
        [kvDim, dim],
        _permuteHeads(
          slice(wk, l * kvDim, kvDim, dim),
          nKvHeads,
          headSize,
          dim,
        ),
      );
      put('model.layers.$l.self_attn.v_proj.weight', [
        kvDim,
        dim,
      ], slice(wv, l * kvDim, kvDim, dim));
      put('model.layers.$l.self_attn.o_proj.weight', [
        dim,
        qDim,
      ], slice(wo, l * dim, dim, qDim));
      put('model.layers.$l.post_attention_layernorm.weight', [
        dim,
      ], slice(rmsFfn, l, 1, dim));
      put('model.layers.$l.mlp.gate_proj.weight', [
        hiddenDim,
        dim,
      ], slice(w1, l * hiddenDim, hiddenDim, dim));
      put('model.layers.$l.mlp.up_proj.weight', [
        hiddenDim,
        dim,
      ], slice(w3, l * hiddenDim, hiddenDim, dim));
      put('model.layers.$l.mlp.down_proj.weight', [
        dim,
        hiddenDim,
      ], slice(w2, l * dim, dim, hiddenDim));
    }

    final graph = parseHfConfig(<String, dynamic>{
      'model_type': 'llama',
      'hidden_size': dim,
      'num_hidden_layers': nLayers,
      'num_attention_heads': nHeads,
      'num_key_value_heads': nKvHeads,
      'intermediate_size': hiddenDim,
      'vocab_size': vocab,
      'max_position_embeddings': seqLen,
      'head_dim': headSize,
      'rms_norm_eps': 1e-5,
      'rope_theta': 10000.0,
      'hidden_act': 'silu',
      'tie_word_embeddings': tied,
    }, name: name);

    return Llama2cCheckpoint(graph, MapWeightStore(tensors), maxSeq: seqLen);
  }
}

/// Reorders the rows of a `[nHeads*headDim, inDim]` projection from llama2.c's
/// interleaved RoPE pairing to HuggingFace half-split, per HF's `permute`:
/// output row `(s*half + p)` within a head is taken from input row `(p*2 + s)`.
Float32List _permuteHeads(Float32List w, int nHeads, int headDim, int inDim) {
  final out = Float32List(w.length);
  final half = headDim ~/ 2;
  for (var h = 0; h < nHeads; h++) {
    for (var nr = 0; nr < headDim; nr++) {
      final s = nr ~/ half;
      final pdx = nr % half;
      final srcRow = h * headDim + (pdx * 2 + s);
      final dstRow = h * headDim + nr;
      out.setRange(
        dstRow * inDim,
        (dstRow + 1) * inDim,
        w.sublist(srcRow * inDim, (srcRow + 1) * inDim),
      );
    }
  }
  return out;
}
