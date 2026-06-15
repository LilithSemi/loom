@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  Map<String, dynamic> fixture() =>
      jsonDecode(File('test/fixtures/smollm2_config.json').readAsStringSync())
          as Map<String, dynamic>;

  test('parses SmolLM2 config into a valid llama ModelGraph', () {
    final g = parseHfConfig(fixture(), name: 'smollm2-135m');
    expect(g.arch, LlmArch.llama);
    expect(g.hiddenSize, 576);
    expect(g.numLayers, 30);
    expect(g.vocabSize, 49152);
    expect(g.tieEmbeddings, isTrue);
    expect(g.layers.first.attention.numHeads, 9);
    expect(g.layers.first.attention.numKvHeads, 3);
    expect(g.layers.first.attention.headDim, 64);
    expect(g.layers.first.mlp.gated, isTrue);
    expect(g.validate, returnsNormally);
  });

  test('num_key_value_heads defaults to num_attention_heads when absent', () {
    final m = fixture()..remove('num_key_value_heads');
    final g = parseHfConfig(m);
    expect(g.layers.first.attention.numKvHeads, 9);
  });

  test('unknown model_type throws', () {
    final m = fixture()..['model_type'] = 'mamba';
    expect(() => parseHfConfig(m), throwsArgumentError);
  });

  test('missing required field hidden_size throws ArgumentError', () {
    final m = fixture()..remove('hidden_size');
    expect(() => parseHfConfig(m), throwsArgumentError);
  });

  test('parseHfConfig accepts a mistral config as a llama-shaped graph', () {
    final g = parseHfConfig({
      'model_type': 'mistral',
      'hidden_size': 32,
      'num_hidden_layers': 2,
      'num_attention_heads': 4,
      'num_key_value_heads': 2,
      'intermediate_size': 64,
      'vocab_size': 100,
      'max_position_embeddings': 128,
      'sliding_window': 64,
      'rms_norm_eps': 1e-5,
      'rope_theta': 10000.0,
    }, name: 'tiny-mistral');
    expect(g.layers.length, 2);
    expect(g.layers.first.normKind, NormKind.rmsNorm);
    expect(g.layers.first.attention.posEncoding, PosEncoding.rope);
    expect(g.layers.first.mlp.gated, isTrue);
    expect(g.layers.first.attention.numKvHeads, 2);
  });
}
