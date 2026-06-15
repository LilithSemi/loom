import 'dart:math' as math;
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:loom/loom.dart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a TensorView from a flat double list + shape.  dtype=f32, LE.
TensorView tvFromList(String name, List<double> values, List<int> shape) {
  final count = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
  if (count != values.length) {
    throw ArgumentError(
      'tvFromList: shape $shape => $count elements but values.length=${values.length}',
    );
  }
  final buf = ByteData(count * 4);
  for (var i = 0; i < count; i++) {
    buf.setFloat32(i * 4, values[i].toDouble(), Endian.little);
  }
  return TensorView(
    name: name,
    shape: shape,
    dtype: TensorDType.f32,
    bytes: buf,
  );
}

/// Zero-filled TensorView.
TensorView tvZeros(String name, List<int> shape) {
  final count = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
  return tvFromList(name, List<double>.filled(count, 0.0), shape);
}

// ---------------------------------------------------------------------------
// Tiny model constants  H=2, V=3, nH=1, nKV=1, hd=2, I=2, 1 layer
// ---------------------------------------------------------------------------
const int H = 2;
const int V = 3;
const int nH = 1;
const int nKV = 1;
const int hd = 2;
const int I = 2;
const double eps = 1e-5;

ModelGraph tinyGraph() {
  return ModelGraph(
    name: 'tiny',
    arch: LlmArch.llama,
    hiddenSize: H,
    vocabSize: V,
    maxSeqLen: 32,
    tieEmbeddings: true,
    layers: [
      LayerSpec(
        index: 0,
        normKind: NormKind.rmsNorm,
        normEps: eps,
        attention: const AttentionSpec(
          numHeads: nH,
          numKvHeads: nKV,
          headDim: hd,
          posEncoding: PosEncoding.rope,
        ),
        mlp: MlpSpec(
          intermediateSize: I,
          activation: ActivationKind.silu,
          gated: true,
        ),
      ),
    ],
  );
}

/// Build a BoundModel where attention and MLP projections are all zeroed,
/// so the layer passes hidden state through unchanged (residual passthrough).
/// embed = [[3,4],[1,0],[0,1]], finalNorm gamma = [1,1].
BoundModel passthruModel() {
  // embed_tokens: shape [V, H] = [3, 2]
  final embed = tvFromList(
    'model.embed_tokens.weight',
    [3.0, 4.0, 1.0, 0.0, 0.0, 1.0],
    [V, H],
  );

  // All layer projections zero.
  final inputNorm = tvFromList(
    'model.layers.0.input_layernorm.weight',
    [1.0, 1.0],
    [H],
  );
  final qProj = tvZeros('model.layers.0.self_attn.q_proj.weight', [nH * hd, H]);
  final kProj = tvZeros('model.layers.0.self_attn.k_proj.weight', [
    nKV * hd,
    H,
  ]);
  final vProj = tvZeros('model.layers.0.self_attn.v_proj.weight', [
    nKV * hd,
    H,
  ]);
  final oProj = tvZeros('model.layers.0.self_attn.o_proj.weight', [H, nH * hd]);
  final postAttnNorm = tvFromList(
    'model.layers.0.post_attention_layernorm.weight',
    [1.0, 1.0],
    [H],
  );
  final gate = tvZeros('model.layers.0.mlp.gate_proj.weight', [I, H]);
  final up = tvZeros('model.layers.0.mlp.up_proj.weight', [I, H]);
  final down = tvZeros('model.layers.0.mlp.down_proj.weight', [H, I]);

  final finalNorm = tvFromList('model.norm.weight', [1.0, 1.0], [H]);

  final layer = BoundLayer(
    inputNorm: inputNorm,
    qProj: qProj,
    kProj: kProj,
    vProj: vProj,
    oProj: oProj,
    postAttnNorm: postAttnNorm,
    gate: gate,
    up: up,
    down: down,
  );

  return BoundModel(
    embedTokens: embed,
    layers: [layer],
    finalNorm: finalNorm,
    lmHead: embed,
    lmHeadTied: true,
  );
}

/// Build a BoundModel where embed = [[1,0],[0,1],[3,4]] so that token 2 wins.
BoundModel passthruModelToken2Wins() {
  final embed = tvFromList(
    'model.embed_tokens.weight',
    [1.0, 0.0, 0.0, 1.0, 3.0, 4.0],
    [V, H],
  );

  final inputNorm = tvFromList(
    'model.layers.0.input_layernorm.weight',
    [1.0, 1.0],
    [H],
  );
  final qProj = tvZeros('model.layers.0.self_attn.q_proj.weight', [nH * hd, H]);
  final kProj = tvZeros('model.layers.0.self_attn.k_proj.weight', [
    nKV * hd,
    H,
  ]);
  final vProj = tvZeros('model.layers.0.self_attn.v_proj.weight', [
    nKV * hd,
    H,
  ]);
  final oProj = tvZeros('model.layers.0.self_attn.o_proj.weight', [H, nH * hd]);
  final postAttnNorm = tvFromList(
    'model.layers.0.post_attention_layernorm.weight',
    [1.0, 1.0],
    [H],
  );
  final gate = tvZeros('model.layers.0.mlp.gate_proj.weight', [I, H]);
  final up = tvZeros('model.layers.0.mlp.up_proj.weight', [I, H]);
  final down = tvZeros('model.layers.0.mlp.down_proj.weight', [H, I]);

  final finalNorm = tvFromList('model.norm.weight', [1.0, 1.0], [H]);

  final layer = BoundLayer(
    inputNorm: inputNorm,
    qProj: qProj,
    kProj: kProj,
    vProj: vProj,
    oProj: oProj,
    postAttnNorm: postAttnNorm,
    gate: gate,
    up: up,
    down: down,
  );

  return BoundModel(
    embedTokens: embed,
    layers: [layer],
    finalNorm: finalNorm,
    lmHead: embed,
    lmHeadTied: true,
  );
}

void main() {
  // -------------------------------------------------------------------------
  // Test 1: residual passthrough + final norm + lm_head (hand-computable)
  // -------------------------------------------------------------------------
  // embed[0] = [3, 4]
  // After zeroed attention and MLP: h[0] = [3, 4] (residual passthrough)
  // finalNorm gamma = [1, 1], eps = 1e-5
  // rmsNorm([3, 4], [1, 1], 1e-5):
  //   ms = (9 + 16) / 2 = 12.5
  //   inv = 1 / sqrt(12.5 + 1e-5) ~= 0.28284271...
  //   normed = [3 * inv, 4 * inv] = [0.84852813..., 1.13137084...]
  // logits[r] = dot(embed[r], normed):
  //   logits[0] = 3 * 0.84852813 + 4 * 1.13137084 = 2.54558441 + 4.52548340 = 7.07106781
  //   logits[1] = 1 * 0.84852813 + 0 = 0.84852813
  //   logits[2] = 0 + 1 * 1.13137084 = 1.13137084
  group('GoldenRunner.forward', () {
    test('residual passthrough: logits match hand-computed values', () {
      final graph = tinyGraph();
      final model = passthruModel();
      final runner = GoldenRunner(graph, model);

      final logits = runner.forward([0]);

      expect(logits.length, V);

      // Hand-computed: see above.
      final inv = 1.0 / math.sqrt(12.5 + eps);
      final n0 = 3.0 * inv;
      final n1 = 4.0 * inv;
      // logits[r] = dot(embed_row_r, normed)
      final expected0 = 3.0 * n0 + 4.0 * n1; // ~7.071
      final expected1 = 1.0 * n0 + 0.0 * n1; // ~0.848
      final expected2 = 0.0 * n0 + 1.0 * n1; // ~1.131

      expect(logits[0], closeTo(expected0, 1e-4));
      expect(logits[1], closeTo(expected1, 1e-4));
      expect(logits[2], closeTo(expected2, 1e-4));

      // Token 0 should have max logit.
      expect(logits[0], greaterThan(logits[1]));
      expect(logits[0], greaterThan(logits[2]));
    });

    // -------------------------------------------------------------------------
    // Test 2: 2-token causal - forward([a, b]) returns logits for LAST token
    // With zeroed attention output, h[1] stays embed[b]; result equals forward([b]).
    // -------------------------------------------------------------------------
    test(
      '2-token causal: forward([a,b]) equals forward([b]) for zeroed attention',
      () {
        final graph = tinyGraph();
        final model = passthruModel();
        final runner = GoldenRunner(graph, model);

        // forward with just token 1
        final logitsSingle = runner.forward([1]);
        // forward with tokens [0, 1] - last token is 1
        final logitsCausal = runner.forward([0, 1]);

        expect(logitsCausal.length, V);
        for (var i = 0; i < V; i++) {
          expect(logitsCausal[i], closeTo(logitsSingle[i], 1e-6));
        }
      },
    );

    // -------------------------------------------------------------------------
    // Test 3: MLP contributes a known delta
    // Set down weight so MLP adds a computable non-zero vector to h.
    // gate and up are identity-like so we can compute the mlp output.
    // -------------------------------------------------------------------------
    test('MLP delta: non-zero down proj shifts logits as expected', () {
      // We set up MLP so that for h=[1,0]:
      //   postAttnNorm gamma=[1,1], so n2=rmsNorm([1,0],[1,1],eps)=[1/sqrt(1/2),0]*inv = [sqrt(2), 0]
      //   Actually rmsNorm([1,0]): ms=(1+0)/2=0.5, inv=1/sqrt(0.5+eps)~=sqrt(2)
      //   n2 = [1*sqrt(2), 0*sqrt(2)] = [sqrt(2), 0]
      //   gate (shape [2,2]): row0=[1,0], row1=[0,0] => g=[sqrt(2), 0]
      //   up (shape [2,2]): row0=[1,0], row1=[0,0] => u=[sqrt(2), 0]
      //   silu(g) = [sqrt(2)*sigmoid(sqrt(2)), 0] ~ [sqrt(2)*0.80706, 0] = [1.14159, 0]
      //   inter = silu(g) * u = [1.14159*sqrt(2), 0] = [1.61463, 0]
      //   down (shape [2,2]): row0=[1,0], row1=[0,0] => d=[1.61463, 0]
      //   So MLP adds [1.61463, 0] to h=[1,0] => h after MLP = [2.61463, 0]
      // finalNorm([2.61463, 0], [1,1], eps): ms=2.61463^2/2=3.41818, inv=1/sqrt(3.41828)=0.54105
      //   normed=[2.61463*0.54105, 0]=[1.41421, 0]  (~sqrt(2))
      // logits: dot(embed[r], normed)
      //   embed=[[3,4],[1,0],[0,1]]
      //   logits[0]=3*1.41421=4.24264, logits[1]=1*1.41421=1.41421, logits[2]=0

      final embed = tvFromList(
        'model.embed_tokens.weight',
        [3.0, 4.0, 1.0, 0.0, 0.0, 1.0],
        [V, H],
      );
      final inputNorm = tvZeros('model.layers.0.input_layernorm.weight', [H]);
      final qProj = tvZeros('model.layers.0.self_attn.q_proj.weight', [
        nH * hd,
        H,
      ]);
      final kProj = tvZeros('model.layers.0.self_attn.k_proj.weight', [
        nKV * hd,
        H,
      ]);
      final vProj = tvZeros('model.layers.0.self_attn.v_proj.weight', [
        nKV * hd,
        H,
      ]);
      final oProj = tvZeros('model.layers.0.self_attn.o_proj.weight', [
        H,
        nH * hd,
      ]);
      final postAttnNorm = tvFromList(
        'model.layers.0.post_attention_layernorm.weight',
        [1.0, 1.0],
        [H],
      );
      // gate row0=[1,0] row1=[0,0]  => shape [I=2, H=2]
      final gate = tvFromList(
        'model.layers.0.mlp.gate_proj.weight',
        [1.0, 0.0, 0.0, 0.0],
        [I, H],
      );
      // up same as gate
      final up = tvFromList(
        'model.layers.0.mlp.up_proj.weight',
        [1.0, 0.0, 0.0, 0.0],
        [I, H],
      );
      // down row0=[1,0] row1=[0,0] => shape [H=2, I=2]
      final down = tvFromList(
        'model.layers.0.mlp.down_proj.weight',
        [1.0, 0.0, 0.0, 0.0],
        [H, I],
      );
      final finalNorm = tvFromList('model.norm.weight', [1.0, 1.0], [H]);

      final layer = BoundLayer(
        inputNorm: inputNorm,
        qProj: qProj,
        kProj: kProj,
        vProj: vProj,
        oProj: oProj,
        postAttnNorm: postAttnNorm,
        gate: gate,
        up: up,
        down: down,
      );
      final model = BoundModel(
        embedTokens: embed,
        layers: [layer],
        finalNorm: finalNorm,
        lmHead: embed,
        lmHeadTied: true,
      );

      final graph = tinyGraph();
      final runner = GoldenRunner(graph, model);

      // Use token 1: embed[1] = [1, 0]
      final logits = runner.forward([1]);

      // Hand-compute:
      // h = [1, 0]
      // inputNorm gamma=[0,0], so attn_in=rmsNorm([1,0],[0,0],eps)=[0,0], all projs -> 0, attn out -> 0, oProj -> 0
      // After attn residual: h=[1,0]
      // postAttnNorm([1,0],[1,1],eps): ms=0.5, inv=1/sqrt(0.5+eps)~=sqrt(2)
      //   n2=[sqrt(2), 0]
      // g = linear(gate,[sqrt(2),0])=[sqrt(2)*1+0*0, 0] = [sqrt(2), 0]
      // u = linear(up,[sqrt(2),0])   = [sqrt(2), 0]
      // silu(g): silu(x)=x*sigmoid(x), sigmoid(sqrt2)~=0.80706
      //   act=[sqrt(2)*0.80706, 0]=[1.14159, 0]
      // inter = act*u = [1.14159*sqrt(2), 0]
      final sqrt2 = math.sqrt(2.0);
      final siluSqrt2 = sqrt2 / (1.0 + math.exp(-sqrt2));
      final interVal = siluSqrt2 * sqrt2;
      // d = linear(down, inter) = [interVal*1+0*0, 0] = [interVal, 0]
      // h after MLP = [1 + interVal, 0]
      final hAfter0 = 1.0 + interVal;
      // finalNorm([hAfter0, 0], [1,1], eps):
      //   ms = hAfter0^2 / 2
      //   inv = 1 / sqrt(ms + eps)
      final msFinal = hAfter0 * hAfter0 / 2.0;
      final invFinal = 1.0 / math.sqrt(msFinal + eps);
      final normed0 = hAfter0 * invFinal;
      // logits[r] = dot(embed[r], [normed0, 0])
      //   logits[0] = 3 * normed0
      //   logits[1] = 1 * normed0
      //   logits[2] = 0
      expect(logits[0], closeTo(3.0 * normed0, 1e-4));
      expect(logits[1], closeTo(1.0 * normed0, 1e-4));
      expect(logits[2], closeTo(0.0, 1e-4));
    });

    // -------------------------------------------------------------------------
    // Test 4: empty token list throws ArgumentError
    // -------------------------------------------------------------------------
    test('forward with empty token list throws ArgumentError', () {
      final graph = tinyGraph();
      final model = passthruModel();
      final runner = GoldenRunner(graph, model);

      expect(() => runner.forward([]), throwsA(isA<ArgumentError>()));
    });
  });

  // -------------------------------------------------------------------------
  // Test group: GoldenRunner.generate
  // -------------------------------------------------------------------------
  group('GoldenRunner.generate', () {
    // With passthruModel embed=[[3,4],[1,0],[0,1]], forward([0]) => argmax=0
    test(
      'generate(maxNewTokens:1) returns token 0 when token 0 has max logit',
      () {
        final graph = tinyGraph();
        final model = passthruModel();
        final runner = GoldenRunner(graph, model);

        final newTokens = runner.generate([0], maxNewTokens: 1);

        expect(newTokens, hasLength(1));
        expect(newTokens[0], 0);
      },
    );

    // With embed=[[1,0],[0,1],[3,4]], forward([0]) => argmax should be 2
    test('generate picks token 2 when its embedding gives max logit', () {
      final graph = tinyGraph();
      final model = passthruModelToken2Wins();
      final runner = GoldenRunner(graph, model);

      final newTokens = runner.generate([0], maxNewTokens: 1);

      expect(newTokens[0], 2);
    });

    test('generate returns exactly maxNewTokens tokens', () {
      final graph = tinyGraph();
      final model = passthruModel();
      final runner = GoldenRunner(graph, model);

      final newTokens = runner.generate([0], maxNewTokens: 3);

      expect(newTokens, hasLength(3));
    });

    test('generate only returns new tokens, not the prompt', () {
      final graph = tinyGraph();
      final model = passthruModel();
      final runner = GoldenRunner(graph, model);

      final newTokens = runner.generate([1, 2], maxNewTokens: 2);

      // Should return exactly 2 new tokens, not prompt tokens.
      expect(newTokens, hasLength(2));
    });
  });
}
