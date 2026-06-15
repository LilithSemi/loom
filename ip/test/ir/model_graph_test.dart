@TestOn('vm')
library;

import 'package:loom/loom.dart';
import 'package:test/test.dart';

ModelGraph smolLm2Like() => ModelGraph(
  name: 'smollm2-135m',
  arch: LlmArch.llama,
  hiddenSize: 576,
  vocabSize: 49152,
  maxSeqLen: 8192,
  tieEmbeddings: true,
  layers: List.generate(
    30,
    (i) => LayerSpec(
      index: i,
      normKind: NormKind.rmsNorm,
      normEps: 1e-5,
      attention: const AttentionSpec(
        numHeads: 9,
        numKvHeads: 3,
        headDim: 64,
        posEncoding: PosEncoding.rope,
        ropeTheta: 10000.0,
      ),
      mlp: const MlpSpec(
        intermediateSize: 1536,
        activation: ActivationKind.silu,
        gated: true,
      ),
    ),
  ),
);

void main() {
  test('well-formed graph validates', () {
    expect(smolLm2Like().validate, returnsNormally);
  });

  test('numLayers derives from layers', () {
    expect(smolLm2Like().numLayers, 30);
  });
}
