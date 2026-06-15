library;

import '../golden/projector.dart';
import '../golden/vision.dart';
import '../ir/model_graph.dart';
import '../ir/tensor.dart';
import 'weight_store.dart';

/// One MoE expert's gated-FFN tensors (same shape as a dense FFN).
class BoundExpert {
  final TensorView gate;
  final TensorView up;
  final TensorView down;
  const BoundExpert({required this.gate, required this.up, required this.down});
}

/// A layer's Mixture-of-Experts FFN: a router (`[numExperts x hidden]`) plus the
/// per-expert gated FFNs.
class BoundMoe {
  final TensorView router;
  final List<BoundExpert> experts;
  const BoundMoe({required this.router, required this.experts});
}

/// One Multi-Token Prediction module: a transformer [block] plus the two input
/// norms and the projection that fuses the previous hidden state with the next
/// token's embedding (`eh_proj @ concat(hnorm(h), enorm(embed))`, `[H, 2H]`).
class BoundMtpModule {
  final TensorView enorm;
  final TensorView hnorm;
  final TensorView ehProj;
  final BoundLayer block;
  const BoundMtpModule({
    required this.enorm,
    required this.hnorm,
    required this.ehProj,
    required this.block,
  });
}

/// A model's Multi-Token Prediction heads. The token embedding, final norm, and
/// lm_head are shared with the main model (see [BoundModel]).
class BoundMtp {
  final List<BoundMtpModule> modules;
  const BoundMtp({required this.modules});
}

/// The resolved, shape-validated tensors for one transformer layer.
class BoundLayer {
  final TensorView inputNorm;
  final TensorView qProj;
  final TensorView kProj;
  final TensorView vProj;
  final TensorView oProj;
  final TensorView postAttnNorm;

  /// Dense gated-FFN tensors. Null for MoE layers (use [moe] instead).
  final TensorView? gate;
  final TensorView? up;
  final TensorView? down;

  /// Non-null for a Mixture-of-Experts layer (then [gate]/[up]/[down] are null).
  final BoundMoe? moe;

  /// q/k/v projection biases, present only when the arch uses them (Qwen2).
  final TensorView? qBias;
  final TensorView? kBias;
  final TensorView? vBias;

  const BoundLayer({
    required this.inputNorm,
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
    required this.postAttnNorm,
    this.gate,
    this.up,
    this.down,
    this.moe,
    this.qBias,
    this.kBias,
    this.vBias,
  });
}

/// All resolved, shape-validated tensors for a model.
class BoundModel {
  final TensorView embedTokens;
  final List<BoundLayer> layers;
  final TensorView finalNorm;

  /// The LM head projection tensor. Equal to [embedTokens] when [lmHeadTied].
  final TensorView lmHead;

  /// True when [lmHead] is reused from [embedTokens] (tied or absent).
  final bool lmHeadTied;

  /// Multi-Token Prediction heads, null for models without them.
  final BoundMtp? mtp;

  /// Decoded vision tower + projector, non-null for vision-language models.
  final VisionWeights? vision;
  final ProjectorWeights? projector;

  const BoundModel({
    required this.embedTokens,
    required this.layers,
    required this.finalNorm,
    required this.lmHead,
    required this.lmHeadTied,
    this.mtp,
    this.vision,
    this.projector,
  });
}

/// Fetches [name] from [store], checking presence and validating [expected] shape.
TensorView _fetch(WeightStore store, String name, List<int> expected) {
  if (!store.contains(name)) {
    throw ArgumentError('missing tensor: $name');
  }
  final tv = store.get(name);
  if (!_shapeEqual(tv.shape, expected)) {
    throw ArgumentError(
      'shape mismatch for $name: expected $expected, got ${tv.shape}',
    );
  }
  return tv;
}

bool _shapeEqual(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// Resolves one transformer block's tensors (attention + FFN + norms) at layer
/// index [i]. Shared by the main layers and the MTP module blocks.
BoundLayer _bindLayer(
  WeightStore store,
  int i,
  LayerSpec layer,
  int H, {
  String prefix = 'model',
}) {
  final attn = layer.attention;
  final nH = attn.numHeads;
  final nKV = attn.numKvHeads;
  final hd = attn.headDim;
  final iSize = layer.mlp.intermediateSize;

  final inputNorm = _fetch(store, '$prefix.layers.$i.input_layernorm.weight', [
    H,
  ]);
  final qProj = _fetch(store, '$prefix.layers.$i.self_attn.q_proj.weight', [
    nH * hd,
    H,
  ]);
  final kProj = _fetch(store, '$prefix.layers.$i.self_attn.k_proj.weight', [
    nKV * hd,
    H,
  ]);
  final vProj = _fetch(store, '$prefix.layers.$i.self_attn.v_proj.weight', [
    nKV * hd,
    H,
  ]);
  final oProj = _fetch(store, '$prefix.layers.$i.self_attn.o_proj.weight', [
    H,
    nH * hd,
  ]);
  final postAttnNorm = _fetch(
    store,
    '$prefix.layers.$i.post_attention_layernorm.weight',
    [H],
  );

  // MLP: a dense gated FFN, or a Mixture-of-Experts (router + N experts). MoE
  // uses the Qwen-MoE naming (mlp.gate router, mlp.experts.E.*_proj). Mixtral's
  // block_sparse_moe / w1-w2-w3 naming is a follow-on.
  TensorView? gate, up, down;
  BoundMoe? moe;
  final moeSpec = layer.mlp.moe;
  if (moeSpec == null) {
    gate = _fetch(store, '$prefix.layers.$i.mlp.gate_proj.weight', [iSize, H]);
    up = _fetch(store, '$prefix.layers.$i.mlp.up_proj.weight', [iSize, H]);
    down = _fetch(store, '$prefix.layers.$i.mlp.down_proj.weight', [H, iSize]);
  } else {
    final mi = moeSpec.moeIntermediate;
    final router = _fetch(store, '$prefix.layers.$i.mlp.gate.weight', [
      moeSpec.numExperts,
      H,
    ]);
    final experts = <BoundExpert>[
      for (var e = 0; e < moeSpec.numExperts; e++)
        BoundExpert(
          gate: _fetch(
            store,
            '$prefix.layers.$i.mlp.experts.$e.gate_proj.weight',
            [mi, H],
          ),
          up: _fetch(store, '$prefix.layers.$i.mlp.experts.$e.up_proj.weight', [
            mi,
            H,
          ]),
          down: _fetch(
            store,
            '$prefix.layers.$i.mlp.experts.$e.down_proj.weight',
            [H, mi],
          ),
        ),
    ];
    moe = BoundMoe(router: router, experts: experts);
  }

  // Qwen2-style q/k/v biases (o_proj has none). Loaded only when the arch uses
  // them, so llama models without bias tensors still bind cleanly.
  TensorView? qBias, kBias, vBias;
  if (attn.qkvBias) {
    qBias = _fetch(store, '$prefix.layers.$i.self_attn.q_proj.bias', [nH * hd]);
    kBias = _fetch(store, '$prefix.layers.$i.self_attn.k_proj.bias', [
      nKV * hd,
    ]);
    vBias = _fetch(store, '$prefix.layers.$i.self_attn.v_proj.bias', [
      nKV * hd,
    ]);
  }

  return BoundLayer(
    inputNorm: inputNorm,
    qProj: qProj,
    kProj: kProj,
    vProj: vProj,
    oProj: oProj,
    postAttnNorm: postAttnNorm,
    gate: gate,
    up: up,
    down: down,
    moe: moe,
    qBias: qBias,
    kBias: kBias,
    vBias: vBias,
  );
}

/// Resolves and shape-validates all tensors in [store] against [graph].
/// Only supports llama/qwen2 HuggingFace naming conventions.
/// Throws [ArgumentError] for unsupported architectures, missing tensors,
/// or shape mismatches.
BoundModel bindWeights(ModelGraph graph, WeightStore store) {
  if (graph.arch != LlmArch.llama && graph.arch != LlmArch.qwen2) {
    throw ArgumentError(
      'bindWeights: only llama/qwen2 naming supported (got ${graph.arch.name})',
    );
  }

  final H = graph.hiddenSize;
  final V = graph.vocabSize;
  final tp = graph.textPrefix; // 'model', or 'model.text_model' for Idefics3

  final embedTokens = _fetch(store, '$tp.embed_tokens.weight', [V, H]);

  final boundLayers = [
    for (final layer in graph.layers)
      _bindLayer(store, layer.index, layer, H, prefix: tp),
  ];

  // Multi-Token Prediction modules (DeepSeek-V3): each is a transformer block at
  // layer index numLayers+m, plus enorm/hnorm/eh_proj. embed_tokens, final norm,
  // and lm_head are shared with the main model (not re-bound here).
  BoundMtp? mtp;
  if (graph.mtp != null) {
    final base = graph.layers.first;
    final modules = <BoundMtpModule>[
      for (var m = 0; m < graph.mtp!.numModules; m++)
        () {
          final l = graph.layers.length + m;
          final blockSpec = LayerSpec(
            index: l,
            normKind: base.normKind,
            normEps: base.normEps,
            attention: base.attention,
            mlp: base.mlp,
          );
          return BoundMtpModule(
            enorm: _fetch(store, 'model.layers.$l.enorm.weight', [H]),
            hnorm: _fetch(store, 'model.layers.$l.hnorm.weight', [H]),
            ehProj: _fetch(store, 'model.layers.$l.eh_proj.weight', [H, 2 * H]),
            block: _bindLayer(store, l, blockSpec, H),
          );
        }(),
    ];
    mtp = BoundMtp(modules: modules);
  }

  final finalNorm = _fetch(store, '$tp.norm.weight', [H]);

  // Resolve lm_head: use embed if tieEmbeddings is true OR lm_head.weight absent.
  final bool lmHeadTied;
  final TensorView lmHead;
  if (!graph.tieEmbeddings && store.contains('lm_head.weight')) {
    lmHead = _fetch(store, 'lm_head.weight', [V, H]);
    lmHeadTied = false;
  } else {
    lmHead = embedTokens;
    lmHeadTied = true;
  }

  // Vision-language: decode the vision tower + projector so genip can emit them
  // and a VLM forward can consume them (reuses the golden binders).
  final vision = graph.vision == null
      ? null
      : bindVision(graph.vision!, store, prefix: graph.visionPrefix);
  final projector = graph.projector == null
      ? null
      : bindProjector(
          graph.projector!,
          store,
          graph.vision!.hiddenSize,
          graph.hiddenSize,
        );

  return BoundModel(
    embedTokens: embedTokens,
    layers: boundLayers,
    finalNorm: finalNorm,
    lmHead: lmHead,
    lmHeadTied: lmHeadTied,
    mtp: mtp,
    vision: vision,
    projector: projector,
  );
}
