import 'dart:convert';
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

const _H = 8, _INTER = 16, _LAYERS = 2, _VOCAB = 10;

// A tiny llama text model (no vision weights), enough to exercise the fusion:
// forwardWithVision splices a projected vision embedding at the placeholder.
Uint8List _llamaSafetensors() {
  final tensors = <String, List<int>>{
    'model.embed_tokens.weight': [_VOCAB, _H],
    'model.norm.weight': [_H],
  };
  for (var i = 0; i < _LAYERS; i++) {
    tensors['model.layers.$i.input_layernorm.weight'] = [_H];
    tensors['model.layers.$i.self_attn.q_proj.weight'] = [_H, _H];
    tensors['model.layers.$i.self_attn.k_proj.weight'] = [_H, _H];
    tensors['model.layers.$i.self_attn.v_proj.weight'] = [_H, _H];
    tensors['model.layers.$i.self_attn.o_proj.weight'] = [_H, _H];
    tensors['model.layers.$i.post_attention_layernorm.weight'] = [_H];
    tensors['model.layers.$i.mlp.gate_proj.weight'] = [_INTER, _H];
    tensors['model.layers.$i.mlp.up_proj.weight'] = [_INTER, _H];
    tensors['model.layers.$i.mlp.down_proj.weight'] = [_H, _INTER];
  }

  final data = BytesBuilder();
  final header = <String, dynamic>{};
  var off = 0;
  final names = tensors.keys.toList()..sort();
  for (final n in names) {
    final shape = tensors[n]!;
    final count = shape.reduce((a, b) => a * b);
    final bd = ByteData(count * 4);
    for (var i = 0; i < count; i++) {
      bd.setFloat32(i * 4, 0.02 * (i % 7 - 3), Endian.little);
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

ModelGraph _graph() => parseHfConfig({
  'model_type': 'llama',
  'hidden_size': _H,
  'num_hidden_layers': _LAYERS,
  'num_attention_heads': 2,
  'num_key_value_heads': 2,
  'intermediate_size': _INTER,
  'vocab_size': _VOCAB,
  'max_position_embeddings': 32,
  'tie_word_embeddings': true,
}, name: 'tiny-vlm-text');

Float64List _rand(int seed) {
  var s = seed;
  double next() {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff * 2 - 1;
  }

  return Float64List.fromList([for (var i = 0; i < _H; i++) next()]);
}

void main() {
  late GoldenRunner runner;
  late BoundModel model;
  setUp(() {
    final g = _graph();
    model = bindWeights(g, SafetensorsStore.parse(_llamaSafetensors()));
    runner = GoldenRunner(g, model);
  });

  const imageTok = 9;

  test(
    'forwardWithVision splicing the token-k embedding equals forward([..k..])',
    () {
      // The projected vision embed set to token 4's embedding row must reproduce a
      // plain forward with token 4 at the placeholder position.
      final embed = model.embedTokens.toFloat64List();
      final rowK = Float64List.fromList(embed.sublist(4 * _H, 4 * _H + _H));
      final fused = runner.forwardWithVision(
        [1, imageTok, 2],
        [rowK],
        imageTok,
      );
      final plain = runner.forward([1, 4, 2]);
      expect(fused, orderedEquals(plain));
    },
  );

  test('the vision embedding changes the output', () {
    final a = runner.forwardWithVision([1, imageTok, 2], [_rand(1)], imageTok);
    final b = runner.forwardWithVision(
      [1, imageTok, 2],
      [_rand(999)],
      imageTok,
    );
    var maxDiff = 0.0;
    for (var i = 0; i < a.length; i++) {
      final d = (a[i] - b[i]).abs();
      if (d > maxDiff) maxDiff = d;
    }
    expect(maxDiff, greaterThan(1e-6));
  });

  test('multiple placeholders consume successive vision embeds in order', () {
    final embed = model.embedTokens.toFloat64List();
    Float64List row(int k) =>
        Float64List.fromList(embed.sublist(k * _H, k * _H + _H));
    // Two placeholders <- tokens 3 and 5.
    final fused = runner.forwardWithVision(
      [imageTok, imageTok, 2],
      [row(3), row(5)],
      imageTok,
    );
    final plain = runner.forward([3, 5, 2]);
    expect(fused, orderedEquals(plain));
  });

  test('placeholder count must equal the number of vision embeds', () {
    expect(
      () => runner.forwardWithVision(
        [1, imageTok, imageTok],
        [_rand(1)],
        imageTok,
      ),
      throwsArgumentError,
    );
    expect(
      () => runner.forwardWithVision([1, 2, 3], [_rand(1)], imageTok),
      throwsArgumentError,
    );
  });
}
