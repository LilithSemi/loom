import 'package:loom/loom.dart';
import 'package:test/test.dart';

Map<String, dynamic> _clipVc() => {
  'model_type': 'clip_vision_model',
  'hidden_size': 32,
  'image_size': 8,
  'patch_size': 2,
  'num_channels': 3,
  'num_hidden_layers': 2,
  'num_attention_heads': 4,
  'intermediate_size': 64,
  'layer_norm_eps': 1e-5,
  'hidden_act': 'quick_gelu',
};

void main() {
  test('parseVisionConfig reads CLIP dims and gives it a class token', () {
    final v = parseVisionConfig(_clipVc());
    expect(v.hiddenSize, 32);
    expect(v.imageSize, 8);
    expect(v.patchSize, 2);
    expect(v.numChannels, 3);
    expect(v.numLayers, 2);
    expect(v.numHeads, 4);
    expect(v.headDim, 8); // 32 / 4
    expect(v.intermediateSize, 64);
    expect(v.hasClassToken, isTrue);
    // 8/2 = 4 per side -> 16 patches, +1 class token = 17.
    expect(v.numPatches, 16);
    expect(v.seqLen, 17);
  });

  test('parseVisionConfig: SigLIP has no class token', () {
    final vc = _clipVc()
      ..['model_type'] = 'siglip_vision_model'
      ..['hidden_act'] = 'gelu_pytorch_tanh';
    final v = parseVisionConfig(vc);
    expect(v.hasClassToken, isFalse);
    expect(v.seqLen, 16); // patches only
  });

  test(
    'parseHfConfig attaches a vision tower when vision_config is present',
    () {
      final graph = parseHfConfig({
        'model_type': 'llama',
        'hidden_size': 16,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'intermediate_size': 32,
        'vocab_size': 100,
        'vision_config': _clipVc(),
      });
      expect(graph.vision, isNotNull);
      expect(graph.vision!.hiddenSize, 32);
      // A text-only model has none.
      // (projector / image token asserted in the next test)
      final textOnly = parseHfConfig({
        'model_type': 'llama',
        'hidden_size': 16,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'intermediate_size': 32,
        'vocab_size': 100,
      });
      expect(textOnly.vision, isNull);
      expect(textOnly.projector, isNull);
    },
  );

  test('parseHfConfig builds the projector + image token for a VLM', () {
    Map<String, dynamic> vlm(String? projType) => {
      'model_type': 'llama',
      'hidden_size': 16,
      'num_hidden_layers': 1,
      'num_attention_heads': 2,
      'intermediate_size': 32,
      'vocab_size': 100,
      'vision_config': _clipVc(),
      'image_token_index': 42,
      if (projType != null) 'mm_projector_type': projType,
    };
    final mlp = parseHfConfig(vlm('mlp2x_gelu'));
    expect(mlp.projector, isNotNull);
    expect(mlp.projector!.numLayers, 2);
    expect(mlp.imageTokenIndex, 42);
    // A linear projector is 1 layer.
    expect(parseHfConfig(vlm('linear')).projector!.numLayers, 1);
    // Default (no type) is the 2-layer MLP.
    expect(parseHfConfig(vlm(null)).projector!.numLayers, 2);
  });
}
