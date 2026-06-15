import 'dart:convert';
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

const _H = 8, _C = 3, _P = 2, _IMG = 4, _LAYERS = 1, _INTER = 16;

// A tiny CLIP-style vision tower safetensors. [pe] offsets the patch-embedding
// fill so two stores differ only there. seqLen = (4/2)^2 + 1 class = 5.
Uint8List _visionSafetensors({double pe = 0.0}) {
  const pfx = 'vision_model';
  const seqLen = (_IMG ~/ _P) * (_IMG ~/ _P) + 1;
  final shapes = <String, List<int>>{
    '$pfx.embeddings.patch_embedding.weight': [_H, _C, _P, _P],
    '$pfx.embeddings.patch_embedding.bias': [_H],
    '$pfx.embeddings.class_embedding': [_H],
    '$pfx.embeddings.position_embedding.weight': [seqLen, _H],
    '$pfx.pre_layrnorm.weight': [_H],
    '$pfx.pre_layrnorm.bias': [_H],
    '$pfx.post_layernorm.weight': [_H],
    '$pfx.post_layernorm.bias': [_H],
  };
  for (var i = 0; i < _LAYERS; i++) {
    final l = '$pfx.encoder.layers.$i';
    shapes['$l.layer_norm1.weight'] = [_H];
    shapes['$l.layer_norm1.bias'] = [_H];
    shapes['$l.self_attn.q_proj.weight'] = [_H, _H];
    shapes['$l.self_attn.q_proj.bias'] = [_H];
    shapes['$l.self_attn.k_proj.weight'] = [_H, _H];
    shapes['$l.self_attn.k_proj.bias'] = [_H];
    shapes['$l.self_attn.v_proj.weight'] = [_H, _H];
    shapes['$l.self_attn.v_proj.bias'] = [_H];
    shapes['$l.self_attn.out_proj.weight'] = [_H, _H];
    shapes['$l.self_attn.out_proj.bias'] = [_H];
    shapes['$l.layer_norm2.weight'] = [_H];
    shapes['$l.layer_norm2.bias'] = [_H];
    shapes['$l.mlp.fc1.weight'] = [_INTER, _H];
    shapes['$l.mlp.fc1.bias'] = [_INTER];
    shapes['$l.mlp.fc2.weight'] = [_H, _INTER];
    shapes['$l.mlp.fc2.bias'] = [_H];
  }
  final peName = '$pfx.embeddings.patch_embedding.weight';

  final data = BytesBuilder();
  final header = <String, dynamic>{};
  var off = 0;
  final names = shapes.keys.toList()..sort();
  for (final n in names) {
    final shape = shapes[n]!;
    final count = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final bd = ByteData(count * 4);
    for (var i = 0; i < count; i++) {
      // A per-output-dim-varying patch-embedding perturbation. The modulus must
      // be coprime with the patch length (12) or the pattern repeats identically
      // across output rows and the LayerNorm mean-centering wipes it out.
      final bias = n == peName ? pe * (i % 5) : 0.0;
      bd.setFloat32(i * 4, 0.02 * (i % 7 - 3) + bias, Endian.little);
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

VisionTowerSpec _spec() => parseVisionConfig({
  'model_type': 'clip_vision_model',
  'hidden_size': _H,
  'image_size': _IMG,
  'patch_size': _P,
  'num_channels': _C,
  'num_hidden_layers': _LAYERS,
  'num_attention_heads': 2,
  'intermediate_size': _INTER,
  'hidden_act': 'gelu',
});

Float64List _image({double scale = 0.1}) => Float64List.fromList([
  for (var i = 0; i < _C * _IMG * _IMG; i++) (i % 9 - 4) * scale,
]);

void main() {
  test(
    'bindVision + encodeImage produces seqLen x hidden finite embeddings',
    () {
      final w = bindVision(
        _spec(),
        SafetensorsStore.parse(_visionSafetensors()),
      );
      expect(w.blocks.length, _LAYERS);
      expect(w.classToken, isNotNull);
      expect(w.preLnGamma, isNotNull);

      final emb = encodeImage(_image(), w);
      expect(emb.length, 5); // 4 patches + class token
      for (final row in emb) {
        expect(row.length, _H);
        for (final x in row) {
          expect(x.isFinite, isTrue);
        }
      }
    },
  );

  test('encodeImage is deterministic and sensitive to the input pixels', () {
    final w = bindVision(_spec(), SafetensorsStore.parse(_visionSafetensors()));
    final a = encodeImage(_image(scale: 0.1), w);
    final b = encodeImage(_image(scale: 0.1), w);
    for (var i = 0; i < a.length; i++) {
      expect(a[i], orderedEquals(b[i]));
    }
    // A different image gives different embeddings.
    final c = encodeImage(_image(scale: 0.5), w);
    var maxDiff = 0.0;
    for (var i = 0; i < a.length; i++) {
      for (var d = 0; d < _H; d++) {
        final diff = (a[i][d] - c[i][d]).abs();
        if (diff > maxDiff) maxDiff = diff;
      }
    }
    expect(maxDiff, greaterThan(1e-6));
  });

  test('the patch-embedding weights participate in the encoding', () {
    final w0 = bindVision(
      _spec(),
      SafetensorsStore.parse(_visionSafetensors()),
    );
    final w1 = bindVision(
      _spec(),
      SafetensorsStore.parse(_visionSafetensors(pe: 0.3)),
    );
    final a = encodeImage(_image(), w0);
    final b = encodeImage(_image(), w1);
    var maxDiff = 0.0;
    for (var i = 0; i < a.length; i++) {
      for (var d = 0; d < _H; d++) {
        final diff = (a[i][d] - b[i][d]).abs();
        if (diff > maxDiff) maxDiff = diff;
      }
    }
    expect(maxDiff, greaterThan(1e-6), reason: 'patch_embedding had no effect');
  });
}
