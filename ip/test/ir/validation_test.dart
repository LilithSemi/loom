@TestOn('vm')
library;

import 'package:loom/loom.dart';
import 'package:test/test.dart';

LayerSpec layer({
  int numHeads = 9,
  int numKvHeads = 3,
  int headDim = 64,
  double normEps = 1e-5,
  int intermediateSize = 1536,
}) => LayerSpec(
  index: 0,
  normKind: NormKind.rmsNorm,
  normEps: normEps,
  attention: AttentionSpec(
    numHeads: numHeads,
    numKvHeads: numKvHeads,
    headDim: headDim,
    posEncoding: PosEncoding.rope,
  ),
  mlp: MlpSpec(
    intermediateSize: intermediateSize,
    activation: ActivationKind.silu,
    gated: true,
  ),
);

ModelGraph graph({
  int hiddenSize = 576,
  int vocabSize = 49152,
  int maxSeqLen = 8192,
  List<LayerSpec>? layers,
}) => ModelGraph(
  name: 'm',
  arch: LlmArch.llama,
  hiddenSize: hiddenSize,
  vocabSize: vocabSize,
  maxSeqLen: maxSeqLen,
  tieEmbeddings: true,
  layers: layers ?? [layer()],
);

void main() {
  test('headDim * numHeads must equal hiddenSize', () {
    expect(graph(hiddenSize: 512).validate, throwsArgumentError);
  });

  test('numHeads must be divisible by numKvHeads', () {
    expect(
      graph(layers: [layer(numHeads: 9, numKvHeads: 2)]).validate,
      throwsArgumentError,
    );
  });

  test('empty layer list is rejected', () {
    expect(graph(layers: []).validate, throwsArgumentError);
  });

  test('well-formed graph passes', () {
    expect(graph().validate, returnsNormally);
  });

  test('maxSeqLen <= 0 is rejected', () {
    expect(graph(maxSeqLen: 0).validate, throwsArgumentError);
  });

  test('hiddenSize <= 0 is rejected', () {
    expect(graph(hiddenSize: 0).validate, throwsArgumentError);
  });

  test('vocabSize <= 0 is rejected', () {
    expect(graph(vocabSize: 0).validate, throwsArgumentError);
  });

  test('numHeads <= 0 per-layer is rejected', () {
    expect(
      graph(layers: [layer(numHeads: 0, numKvHeads: 1)]).validate,
      throwsArgumentError,
    );
  });

  test('normEps <= 0 per-layer is rejected', () {
    expect(graph(layers: [layer(normEps: 0)]).validate, throwsArgumentError);
  });

  test('intermediateSize <= 0 per-layer is rejected', () {
    expect(
      graph(layers: [layer(intermediateSize: 0)]).validate,
      throwsArgumentError,
    );
  });
}
