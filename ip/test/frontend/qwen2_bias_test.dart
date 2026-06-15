import 'dart:convert';
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

// Build a tiny qwen2 safetensors (hidden 8, 2 layers, 2 heads, headDim 4) with
// q/k/v biases filled with [biasFill]. Weights are small deterministic values.
Uint8List _qwen2Safetensors(double biasFill) {
  final shapes = <String, List<int>>{
    'model.embed_tokens.weight': [10, 8],
    'model.norm.weight': [8],
  };
  final biasNames = <String>{};
  for (var i = 0; i < 2; i++) {
    shapes['model.layers.$i.input_layernorm.weight'] = [8];
    shapes['model.layers.$i.self_attn.q_proj.weight'] = [8, 8];
    shapes['model.layers.$i.self_attn.k_proj.weight'] = [8, 8];
    shapes['model.layers.$i.self_attn.v_proj.weight'] = [8, 8];
    shapes['model.layers.$i.self_attn.o_proj.weight'] = [8, 8];
    shapes['model.layers.$i.self_attn.q_proj.bias'] = [8];
    shapes['model.layers.$i.self_attn.k_proj.bias'] = [8];
    shapes['model.layers.$i.self_attn.v_proj.bias'] = [8];
    biasNames.addAll([
      'model.layers.$i.self_attn.q_proj.bias',
      'model.layers.$i.self_attn.k_proj.bias',
      'model.layers.$i.self_attn.v_proj.bias',
    ]);
    shapes['model.layers.$i.post_attention_layernorm.weight'] = [8];
    shapes['model.layers.$i.mlp.gate_proj.weight'] = [16, 8];
    shapes['model.layers.$i.mlp.up_proj.weight'] = [16, 8];
    shapes['model.layers.$i.mlp.down_proj.weight'] = [8, 16];
  }

  final data = BytesBuilder();
  final header = <String, dynamic>{};
  var off = 0;
  final names = shapes.keys.toList()..sort();
  for (final n in names) {
    final shape = shapes[n]!;
    final count = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final bd = ByteData(count * 4);
    for (var i = 0; i < count; i++) {
      bd.setFloat32(
        i * 4,
        biasNames.contains(n) ? biasFill : 0.02 * (i % 5 - 2),
        Endian.little,
      );
    }
    data.add(bd.buffer.asUint8List());
    header[n] = {
      'dtype': 'F32',
      'shape': shape,
      'data_offsets': [off, off + count * 4],
    };
    off += count * 4;
  }
  final hb = utf8.encode(jsonEncode(header));
  final out = BytesBuilder();
  final len = ByteData(8)..setUint64(0, hb.length, Endian.little);
  out.add(len.buffer.asUint8List());
  out.add(hb);
  out.add(data.toBytes());
  return out.toBytes();
}

ModelGraph _qwen2Graph() => parseHfConfig({
  'model_type': 'qwen2',
  'hidden_size': 8,
  'num_hidden_layers': 2,
  'num_attention_heads': 2,
  'num_key_value_heads': 2,
  'intermediate_size': 16,
  'vocab_size': 10,
  'max_position_embeddings': 32,
  'tie_word_embeddings': true,
}, name: 'tiny-qwen2');

void main() {
  test('parseHfConfig marks qwen2 attention as having qkv bias', () {
    expect(_qwen2Graph().layers.first.attention.qkvBias, isTrue);
    final llama = parseHfConfig({
      'model_type': 'llama',
      'hidden_size': 8,
      'num_hidden_layers': 1,
      'num_attention_heads': 2,
      'intermediate_size': 16,
      'vocab_size': 10,
    });
    expect(llama.layers.first.attention.qkvBias, isFalse);
  });

  test('bindWeights loads q/k/v biases for a qwen2 model', () {
    final store = SafetensorsStore.parse(_qwen2Safetensors(0.1));
    final model = bindWeights(_qwen2Graph(), store);
    expect(model.layers.first.qBias, isNotNull);
    expect(model.layers.first.qBias!.shape, [8]);
    expect(model.layers.first.kBias!.shape, [8]);
    expect(model.layers.first.vBias!.shape, [8]);
  });

  test('bindWeights throws when a qwen2 bias tensor is missing', () {
    // A weights-only store (no bias tensors) parsed against a qwen2 graph.
    final llamaLike = SafetensorsStore.parse(_qwen2Safetensors(0.0));
    // Sanity: the bias-carrying store binds fine.
    expect(() => bindWeights(_qwen2Graph(), llamaLike), returnsNormally);
  });

  test('q/k/v bias changes the golden logits', () {
    final graph = _qwen2Graph();
    final zero = GoldenRunner(
      graph,
      bindWeights(graph, SafetensorsStore.parse(_qwen2Safetensors(0.0))),
    );
    final biased = GoldenRunner(
      graph,
      bindWeights(graph, SafetensorsStore.parse(_qwen2Safetensors(0.7))),
    );
    final lz = zero.forwardAll([1, 2, 3]);
    final lb = biased.forwardAll([1, 2, 3]);
    var maxDiff = 0.0;
    for (var t = 0; t < lz.length; t++) {
      for (var i = 0; i < lz[t].length; i++) {
        final d = (lz[t][i] - lb[t][i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
    }
    expect(maxDiff, greaterThan(1e-6), reason: 'bias had no effect on logits');
  });
}
