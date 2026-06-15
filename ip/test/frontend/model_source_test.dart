import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:loom/loom.dart';
import 'package:test/test.dart';

// Minimal safetensors writer: 8-byte little-endian header length, JSON header,
// then the raw data block. Mirrors test/weights/safetensors_test.dart.
Uint8List buildSafetensors(String headerJson, Uint8List dataBlock) {
  final headerBytes = utf8.encode(headerJson);
  final out = BytesBuilder();
  final len = ByteData(8)..setUint64(0, headerBytes.length, Endian.little);
  out.add(len.buffer.asUint8List());
  out.add(headerBytes);
  out.add(dataBlock);
  return out.toBytes();
}

// A tiny 2-layer tied-embedding llama config + matching F32 safetensors so the
// full parseHfConfig -> SafetensorsStore -> bindWeights chain runs hermetically.
// hidden=8, heads=2, kvHeads=2, headDim=4, intermediate=16, vocab=10, 2 layers.
String _tinyConfigJson() => jsonEncode({
  'model_type': 'llama',
  'hidden_size': 8,
  'num_hidden_layers': 2,
  'num_attention_heads': 2,
  'num_key_value_heads': 2,
  'intermediate_size': 16,
  'vocab_size': 10,
  'max_position_embeddings': 32,
  'rms_norm_eps': 1e-5,
  'rope_theta': 10000.0,
  'tie_word_embeddings': true,
  'hidden_act': 'silu',
});

// Every tensor bindWeights expects must be present with the right shape.
// Values are arbitrary.
Uint8List _tinySafetensors() {
  final tensors = <String, List<int>>{
    'model.embed_tokens.weight': [10, 8],
  };
  for (var i = 0; i < 2; i++) {
    tensors['model.layers.$i.input_layernorm.weight'] = [8];
    tensors['model.layers.$i.self_attn.q_proj.weight'] = [8, 8];
    tensors['model.layers.$i.self_attn.k_proj.weight'] = [8, 8];
    tensors['model.layers.$i.self_attn.v_proj.weight'] = [8, 8];
    tensors['model.layers.$i.self_attn.o_proj.weight'] = [8, 8];
    tensors['model.layers.$i.post_attention_layernorm.weight'] = [8];
    tensors['model.layers.$i.mlp.gate_proj.weight'] = [16, 8];
    tensors['model.layers.$i.mlp.up_proj.weight'] = [16, 8];
    tensors['model.layers.$i.mlp.down_proj.weight'] = [8, 16];
  }
  tensors['model.norm.weight'] = [8];

  final data = BytesBuilder();
  final header = <String, dynamic>{};
  var offset = 0;
  // Deterministic key order so offsets are stable.
  final names = tensors.keys.toList()..sort();
  for (final name in names) {
    final shape = tensors[name]!;
    final count = shape.reduce((a, b) => a * b);
    final bytes = ByteData(count * 4);
    for (var k = 0; k < count; k++) {
      bytes.setFloat32(k * 4, 0.01 * (k + 1), Endian.little);
    }
    data.add(bytes.buffer.asUint8List());
    header[name] = {
      'dtype': 'F32',
      'shape': shape,
      'data_offsets': [offset, offset + count * 4],
    };
    offset += count * 4;
  }
  return buildSafetensors(jsonEncode(header), data.toBytes());
}

Directory _writeTinyHfDir() {
  final dir = Directory.systemTemp.createTempSync('loom_hf_');
  File('${dir.path}/config.json').writeAsStringSync(_tinyConfigJson());
  File('${dir.path}/model.safetensors').writeAsBytesSync(_tinySafetensors());
  File(
    '${dir.path}/tokenizer.json',
  ).writeAsStringSync('{"model":{"vocab":{},"merges":[]}}');
  return dir;
}

void main() {
  test('loadModelSource loads a HF directory into a bound model', () {
    final dir = _writeTinyHfDir();
    try {
      final loaded = loadModelSource(dir.path);
      expect(loaded.graph.hiddenSize, 8);
      expect(loaded.graph.layers.length, 2);
      expect(loaded.graph.vocabSize, 10);
      expect(loaded.maxSeq, 32);
      expect(loaded.model.layers.first.qProj.shape, [8, 8]);
      expect(loaded.model.layers.first.down!.shape, [8, 16]);
      expect(loaded.tokenizerPath, endsWith('tokenizer.json'));
    } finally {
      dir.deleteSync(recursive: true);
    }
  });

  test('loadModelSource routes a .gguf path to the GGUF loader', () {
    // A malformed .gguf fails in the GGUF parser, not with UnsupportedError.
    final f = File(
      '${Directory.systemTemp.createTempSync('loom_gguf_').path}/m.gguf',
    )..writeAsBytesSync(Uint8List(4));
    expect(
      () => loadModelSource(f.path),
      throwsA(isNot(isA<UnsupportedError>())),
    );
  });

  test('loadModelSource errors clearly on a HF dir missing config.json', () {
    final dir = Directory.systemTemp.createTempSync('loom_bad_');
    File('${dir.path}/model.safetensors').writeAsBytesSync(Uint8List(8));
    expect(
      () => loadModelSource(dir.path),
      throwsA(predicate((e) => e.toString().contains('config.json'))),
    );
    dir.deleteSync(recursive: true);
  });
}
