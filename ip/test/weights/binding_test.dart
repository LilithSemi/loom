import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:loom/loom.dart';

// Helper: build a zero-filled f32 TensorView for the given name and shape.
TensorView makeTensor(String name, List<int> shape) {
  final count = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
  final bytes = ByteData(count * 4);
  return TensorView(
    name: name,
    shape: shape,
    dtype: TensorDType.f32,
    bytes: bytes,
  );
}

// Tiny model dims: H=4, V=5, nH=2, nKV=1, hd=2, I=8, 1 layer.
const int H = 4;
const int V = 5;
const int nH = 2;
const int nKV = 1;
const int hd = 2;
const int I = 8;

ModelGraph tinyGraph({bool tieEmbeddings = true, int numLayers = 1}) {
  final layers = List<LayerSpec>.generate(
    numLayers,
    (i) => LayerSpec(
      index: i,
      normKind: NormKind.rmsNorm,
      normEps: 1e-5,
      attention: const AttentionSpec(
        numHeads: nH,
        numKvHeads: nKV,
        headDim: hd,
        posEncoding: PosEncoding.rope,
      ),
      mlp: const MlpSpec(
        intermediateSize: I,
        activation: ActivationKind.silu,
        gated: true,
      ),
    ),
  );
  return ModelGraph(
    name: 'tiny',
    arch: LlmArch.llama,
    hiddenSize: H,
    vocabSize: V,
    maxSeqLen: 128,
    tieEmbeddings: tieEmbeddings,
    layers: layers,
  );
}

/// Build a MapWeightStore with all tensors for [numLayers] layers.
/// If [omitTensor] is non-null, that tensor is excluded.
/// If [wrongShapeFor] is non-null, that tensor gets a wrong shape [1].
MapWeightStore makeStore({
  int numLayers = 1,
  bool includeLmHead = false,
  String? omitTensor,
  String? wrongShapeFor,
}) {
  final tensors = <String, TensorView>{};

  void add(String name, List<int> shape) {
    if (name == omitTensor) return;
    final actualShape = (name == wrongShapeFor) ? [1] : shape;
    tensors[name] = makeTensor(name, actualShape);
  }

  add('model.embed_tokens.weight', [V, H]);
  for (var i = 0; i < numLayers; i++) {
    add('model.layers.$i.input_layernorm.weight', [H]);
    add('model.layers.$i.self_attn.q_proj.weight', [nH * hd, H]);
    add('model.layers.$i.self_attn.k_proj.weight', [nKV * hd, H]);
    add('model.layers.$i.self_attn.v_proj.weight', [nKV * hd, H]);
    add('model.layers.$i.self_attn.o_proj.weight', [H, nH * hd]);
    add('model.layers.$i.post_attention_layernorm.weight', [H]);
    add('model.layers.$i.mlp.gate_proj.weight', [I, H]);
    add('model.layers.$i.mlp.up_proj.weight', [I, H]);
    add('model.layers.$i.mlp.down_proj.weight', [H, I]);
  }
  add('model.norm.weight', [H]);
  if (includeLmHead) {
    add('lm_head.weight', [V, H]);
  }

  return MapWeightStore(tensors);
}

void main() {
  group('bindWeights', () {
    test('happy path tied: returns BoundModel with correct structure', () {
      final graph = tinyGraph(tieEmbeddings: true);
      final store = makeStore(numLayers: 1);

      final bound = bindWeights(graph, store);

      expect(bound.lmHeadTied, isTrue);
      expect(bound.lmHead, same(bound.embedTokens));
      expect(bound.layers.length, 1);

      final layer = bound.layers[0];
      // qProj: [nH*hd, H] = [4, 4]
      expect(layer.qProj.shape, [nH * hd, H]);
      // kProj: [nKV*hd, H] = [2, 4]
      expect(layer.kProj.shape, [nKV * hd, H]);
      // vProj: [nKV*hd, H] = [2, 4]
      expect(layer.vProj.shape, [nKV * hd, H]);
      // oProj: [H, nH*hd] = [4, 4]
      expect(layer.oProj.shape, [H, nH * hd]);
      // inputNorm: [H]
      expect(layer.inputNorm.shape, [H]);
      // postAttnNorm: [H]
      expect(layer.postAttnNorm.shape, [H]);
      // gate: [I, H]
      expect(layer.gate!.shape, [I, H]);
      // up: [I, H]
      expect(layer.up!.shape, [I, H]);
      // down: [H, I]
      expect(layer.down!.shape, [H, I]);

      // embedTokens and finalNorm
      expect(bound.embedTokens.shape, [V, H]);
      expect(bound.finalNorm.shape, [H]);
    });

    test('untied path: lm_head.weight present, tieEmbeddings false', () {
      final graph = tinyGraph(tieEmbeddings: false);
      final store = makeStore(numLayers: 1, includeLmHead: true);

      final bound = bindWeights(graph, store);

      expect(bound.lmHeadTied, isFalse);
      expect(bound.lmHead, isNot(same(bound.embedTokens)));
      expect(bound.lmHead.name, 'lm_head.weight');
    });

    test('tieEmbeddings false but lm_head absent: falls back to embed', () {
      final graph = tinyGraph(tieEmbeddings: false);
      // No lm_head.weight in store
      final store = makeStore(numLayers: 1, includeLmHead: false);

      final bound = bindWeights(graph, store);

      expect(bound.lmHeadTied, isTrue);
      expect(bound.lmHead, same(bound.embedTokens));
    });

    test('missing required tensor throws ArgumentError naming it', () {
      final graph = tinyGraph();
      final store = makeStore(
        numLayers: 1,
        omitTensor: 'model.layers.0.self_attn.q_proj.weight',
      );

      expect(
        () => bindWeights(graph, store),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('model.layers.0.self_attn.q_proj.weight'),
          ),
        ),
      );
    });

    test('shape mismatch throws ArgumentError naming tensor and shapes', () {
      final graph = tinyGraph();
      final store = makeStore(
        numLayers: 1,
        wrongShapeFor: 'model.layers.0.mlp.down_proj.weight',
      );

      expect(
        () => bindWeights(graph, store),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            allOf(
              contains('model.layers.0.mlp.down_proj.weight'),
              contains('shape mismatch'),
            ),
          ),
        ),
      );
    });

    test('gpt2 arch throws ArgumentError', () {
      final graph = ModelGraph(
        name: 'gpt2',
        arch: LlmArch.gpt2,
        hiddenSize: H,
        vocabSize: V,
        maxSeqLen: 128,
        tieEmbeddings: true,
        layers: [
          LayerSpec(
            index: 0,
            normKind: NormKind.layerNorm,
            normEps: 1e-5,
            attention: const AttentionSpec(
              numHeads: nH,
              numKvHeads: nH,
              headDim: hd,
              posEncoding: PosEncoding.learned,
            ),
            mlp: const MlpSpec(
              intermediateSize: I,
              activation: ActivationKind.gelu,
              gated: false,
            ),
          ),
        ],
      );
      final store = makeStore(numLayers: 1);

      expect(
        () => bindWeights(graph, store),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            contains('gpt2'),
          ),
        ),
      );
    });

    test('2-layer happy path: indices format correctly', () {
      final graph = tinyGraph(tieEmbeddings: true, numLayers: 2);
      final store = makeStore(numLayers: 2);

      final bound = bindWeights(graph, store);

      expect(bound.layers.length, 2);
      expect(
        bound.layers[1].qProj.name,
        'model.layers.1.self_attn.q_proj.weight',
      );
      expect(bound.layers[1].down!.name, 'model.layers.1.mlp.down_proj.weight');
    });
  });
}
