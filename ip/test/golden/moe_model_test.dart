import 'dart:convert';
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

// A tiny Qwen3-MoE-style model: hidden 8, 2 layers, 2 heads (headDim 4),
// 4 experts, top-2, moe_intermediate 16. Router (mlp.gate) + per-expert
// gate/up/down_proj use the Qwen-MoE naming that bindWeights understands.
// [expertSeed] offsets the expert weight fill so two stores can differ only
// in the routed experts.
Uint8List _moeSafetensors({double expertSeed = 0.0}) {
  const hidden = 8, experts = 4, moeInter = 16, layers = 2;
  final shapes = <String, List<int>>{
    'model.embed_tokens.weight': [10, hidden],
    'model.norm.weight': [hidden],
  };
  final expertNames = <String>{};
  for (var i = 0; i < layers; i++) {
    shapes['model.layers.$i.input_layernorm.weight'] = [hidden];
    shapes['model.layers.$i.self_attn.q_proj.weight'] = [hidden, hidden];
    shapes['model.layers.$i.self_attn.k_proj.weight'] = [hidden, hidden];
    shapes['model.layers.$i.self_attn.v_proj.weight'] = [hidden, hidden];
    shapes['model.layers.$i.self_attn.o_proj.weight'] = [hidden, hidden];
    shapes['model.layers.$i.post_attention_layernorm.weight'] = [hidden];
    shapes['model.layers.$i.mlp.gate.weight'] = [experts, hidden]; // router
    for (var e = 0; e < experts; e++) {
      shapes['model.layers.$i.mlp.experts.$e.gate_proj.weight'] = [
        moeInter,
        hidden,
      ];
      shapes['model.layers.$i.mlp.experts.$e.up_proj.weight'] = [
        moeInter,
        hidden,
      ];
      shapes['model.layers.$i.mlp.experts.$e.down_proj.weight'] = [
        hidden,
        moeInter,
      ];
      expertNames.addAll([
        'model.layers.$i.mlp.experts.$e.gate_proj.weight',
        'model.layers.$i.mlp.experts.$e.up_proj.weight',
        'model.layers.$i.mlp.experts.$e.down_proj.weight',
      ]);
    }
  }

  final data = BytesBuilder();
  final header = <String, dynamic>{};
  var off = 0;
  final names = shapes.keys.toList()..sort();
  for (final n in names) {
    final shape = shapes[n]!;
    final count = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final bd = ByteData(count * 4);
    final bias = expertNames.contains(n) ? expertSeed : 0.0;
    for (var i = 0; i < count; i++) {
      bd.setFloat32(i * 4, 0.02 * (i % 5 - 2) + bias, Endian.little);
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

ModelGraph _moeGraph() => parseHfConfig({
  'model_type': 'qwen3_moe',
  'hidden_size': 8,
  'num_hidden_layers': 2,
  'num_attention_heads': 2,
  'num_key_value_heads': 2,
  'intermediate_size': 16,
  'moe_intermediate_size': 16,
  'num_local_experts': 4,
  'num_experts_per_tok': 2,
  'norm_topk_prob': true,
  'vocab_size': 10,
  'max_position_embeddings': 32,
  'tie_word_embeddings': true,
}, name: 'tiny-qwen3-moe');

void main() {
  test('parseHfConfig builds a MoeSpec from the MoE config keys', () {
    final moe = _moeGraph().layers.first.mlp.moe;
    expect(moe, isNotNull);
    expect(moe!.numExperts, 4);
    expect(moe.topK, 2);
    expect(moe.moeIntermediate, 16);
    expect(moe.normTopK, isTrue);
  });

  test('bindWeights resolves the router and every expert', () {
    final model = bindWeights(
      _moeGraph(),
      SafetensorsStore.parse(_moeSafetensors()),
    );
    final layer = model.layers.first;
    expect(layer.gate, isNull, reason: 'MoE layer has no dense gate');
    expect(layer.moe, isNotNull);
    expect(layer.moe!.router.shape, [4, 8]);
    expect(layer.moe!.experts.length, 4);
    expect(layer.moe!.experts.first.gate.shape, [16, 8]);
    expect(layer.moe!.experts.first.down.shape, [8, 16]);
  });

  test('MoE golden forward produces finite logits', () {
    final runner = GoldenRunner(
      _moeGraph(),
      bindWeights(_moeGraph(), SafetensorsStore.parse(_moeSafetensors())),
    );
    final logits = runner.forward([1, 2, 3]);
    expect(logits.length, 10);
    for (final v in logits) {
      expect(v.isFinite, isTrue);
    }
  });

  test(
    'the experts genuinely participate: changing expert weights moves the logits',
    () {
      final graph = _moeGraph();
      final a = GoldenRunner(
        graph,
        bindWeights(graph, SafetensorsStore.parse(_moeSafetensors())),
      );
      final b = GoldenRunner(
        graph,
        bindWeights(
          graph,
          SafetensorsStore.parse(_moeSafetensors(expertSeed: 0.3)),
        ),
      );
      final la = a.forward([1, 2, 3]);
      final lb = b.forward([1, 2, 3]);
      var maxDiff = 0.0;
      for (var i = 0; i < la.length; i++) {
        final d = (la[i] - lb[i]).abs();
        if (d > maxDiff) maxDiff = d;
      }
      expect(
        maxDiff,
        greaterThan(1e-6),
        reason: 'experts had no effect on the output',
      );
    },
  );
}
