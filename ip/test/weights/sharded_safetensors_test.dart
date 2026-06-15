import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

// Minimal safetensors writer: 8-byte LE header length, JSON header, data block.
Uint8List buildSafetensors(Map<String, List<int>> shapes) {
  final data = BytesBuilder();
  final header = <String, dynamic>{};
  var off = 0;
  final names = shapes.keys.toList()..sort();
  for (final n in names) {
    final shape = shapes[n]!;
    final count = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final bd = ByteData(count * 4);
    for (var i = 0; i < count; i++) {
      bd.setFloat32(i * 4, 0.01 * (i + 1), Endian.little);
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

void main() {
  test('ShardedSafetensorsStore routes tensors to the right shard', () {
    final dir = Directory.systemTemp.createTempSync('loom_shard_');
    try {
      File('${dir.path}/model-00001-of-00002.safetensors').writeAsBytesSync(
        buildSafetensors({
          'a.weight': [2, 2],
          'b.weight': [2],
        }),
      );
      File('${dir.path}/model-00002-of-00002.safetensors').writeAsBytesSync(
        buildSafetensors({
          'c.weight': [3],
        }),
      );
      File('${dir.path}/model.safetensors.index.json').writeAsStringSync(
        jsonEncode({
          'metadata': {'total_size': 0},
          'weight_map': {
            'a.weight': 'model-00001-of-00002.safetensors',
            'b.weight': 'model-00001-of-00002.safetensors',
            'c.weight': 'model-00002-of-00002.safetensors',
          },
        }),
      );
      final s = ShardedSafetensorsStore.fromIndexFile(
        '${dir.path}/model.safetensors.index.json',
      );
      expect(s.contains('a.weight'), isTrue);
      expect(s.contains('c.weight'), isTrue);
      expect(s.contains('missing'), isFalse);
      expect(s.get('a.weight').shape, [2, 2]);
      expect(s.get('c.weight').shape, [3]);
      expect(s.names.toSet(), {'a.weight', 'b.weight', 'c.weight'});
      expect(() => s.get('missing'), throwsArgumentError);
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test('fromIndexFile throws when a shard file is missing', () {
    final dir = Directory.systemTemp.createTempSync('loom_shard_bad_');
    try {
      File('${dir.path}/model.safetensors.index.json').writeAsStringSync(
        jsonEncode({
          'weight_map': {'a.weight': 'model-00001-of-00001.safetensors'},
        }),
      );
      expect(
        () => ShardedSafetensorsStore.fromIndexFile(
          '${dir.path}/model.safetensors.index.json',
        ),
        throwsArgumentError,
      );
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test('loadModelSource loads a sharded HF model directory', () {
    final dir = Directory.systemTemp.createTempSync('loom_shard_hf_');
    try {
      // Tiny 2-layer tied llama, tensors split across two shards.
      File('${dir.path}/config.json').writeAsStringSync(
        jsonEncode({
          'model_type': 'llama',
          'hidden_size': 8,
          'num_hidden_layers': 2,
          'num_attention_heads': 2,
          'num_key_value_heads': 2,
          'intermediate_size': 16,
          'vocab_size': 10,
          'max_position_embeddings': 32,
          'tie_word_embeddings': true,
        }),
      );
      Map<String, List<int>> layer(int i) => {
        'model.layers.$i.input_layernorm.weight': [8],
        'model.layers.$i.self_attn.q_proj.weight': [8, 8],
        'model.layers.$i.self_attn.k_proj.weight': [8, 8],
        'model.layers.$i.self_attn.v_proj.weight': [8, 8],
        'model.layers.$i.self_attn.o_proj.weight': [8, 8],
        'model.layers.$i.post_attention_layernorm.weight': [8],
        'model.layers.$i.mlp.gate_proj.weight': [16, 8],
        'model.layers.$i.mlp.up_proj.weight': [16, 8],
        'model.layers.$i.mlp.down_proj.weight': [8, 16],
      };
      final shard1 = {
        'model.embed_tokens.weight': [10, 8],
        'model.norm.weight': [8],
        ...layer(0),
      };
      final shard2 = layer(1);
      File(
        '${dir.path}/model-00001-of-00002.safetensors',
      ).writeAsBytesSync(buildSafetensors(shard1));
      File(
        '${dir.path}/model-00002-of-00002.safetensors',
      ).writeAsBytesSync(buildSafetensors(shard2));
      final weightMap = <String, String>{
        for (final n in shard1.keys) n: 'model-00001-of-00002.safetensors',
        for (final n in shard2.keys) n: 'model-00002-of-00002.safetensors',
      };
      File(
        '${dir.path}/model.safetensors.index.json',
      ).writeAsStringSync(jsonEncode({'weight_map': weightMap}));

      final loaded = loadModelSource(dir.path);
      expect(loaded.graph.layers.length, 2);
      // layer 0 weight from shard 1, layer 1 weight from shard 2.
      expect(loaded.model.layers[0].qProj.shape, [8, 8]);
      expect(loaded.model.layers[1].down!.shape, [8, 16]);
    } finally {
      dir.deleteSync(recursive: true);
    }
  });
}
