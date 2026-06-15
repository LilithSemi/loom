library;

import '../ir/model_graph.dart';

T _require<T>(Map<String, dynamic> json, String key) {
  if (!json.containsKey(key) || json[key] == null) {
    throw ArgumentError.value(null, key, 'required field is missing');
  }
  return json[key] as T;
}

/// Maps a HuggingFace `model_type` string to a Loom [LlmArch].
// Mistral is treated as llama: sliding_window is a no-op at the small context
// lengths this compiler targets, a dedicated sliding-window mask is a later
// slice if a long-context target appears.
LlmArch _archOf(String modelType) => switch (modelType) {
  // llama-shaped attention (RMSNorm + RoPE + GQA). Mixtral / Qwen3-MoE add a
  // Mixture-of-Experts FFN (see MoeSpec) but keep the llama attention block.
  'llama' || 'mistral' || 'mixtral' || 'qwen3_moe' => LlmArch.llama,
  // qwen2-shaped: adds q/k/v bias. qwen2_moe is that plus a MoE FFN.
  'qwen2' || 'qwen2_moe' => LlmArch.qwen2,
  'gpt2' => LlmArch.gpt2,
  _ => throw ArgumentError.value(
    modelType,
    'model_type',
    'unsupported architecture',
  ),
};

/// Builds a [VisionTowerSpec] from a HuggingFace `vision_config` sub-dict
/// (CLIP / SigLIP style). Class-token presence follows the sub-model type
/// (SigLIP has none) unless `vision_use_class_token` overrides it.
VisionTowerSpec parseVisionConfig(Map<String, dynamic> vc) {
  final hidden = _require<num>(vc, 'hidden_size').toInt();
  final numHeads = _require<num>(vc, 'num_attention_heads').toInt();
  final act = switch (vc['hidden_act'] as String? ?? 'gelu') {
    'gelu' ||
    'gelu_new' ||
    'gelu_pytorch_tanh' ||
    'quick_gelu' => ActivationKind.gelu,
    'silu' || 'swish' => ActivationKind.silu,
    'relu' => ActivationKind.relu,
    final other => throw ArgumentError.value(other, 'hidden_act'),
  };
  final modelType = vc['model_type'] as String? ?? 'clip_vision_model';
  // Only CLIP prepends a class token. SigLIP and Idefics3/SmolVLM towers do not.
  final isClip = modelType == 'clip_vision_model';
  final hasClassToken = (vc['vision_use_class_token'] as bool?) ?? isClip;
  // Image normalization (usually in the HF preprocessor config. Default by tower
  // type: CLIP uses its own constants, SigLIP/Idefics3 normalize to 0.5).
  List<double>? readVec(String key) =>
      (vc[key] as List?)?.map((e) => (e as num).toDouble()).toList();
  final imageMean =
      readVec('image_mean') ??
      (isClip ? [0.48145466, 0.4578275, 0.40821073] : [0.5, 0.5, 0.5]);
  final imageStd =
      readVec('image_std') ??
      (isClip ? [0.26862954, 0.26130258, 0.27577711] : [0.5, 0.5, 0.5]);
  return VisionTowerSpec(
    imageSize: _require<num>(vc, 'image_size').toInt(),
    patchSize: _require<num>(vc, 'patch_size').toInt(),
    numChannels: (vc['num_channels'] as num?)?.toInt() ?? 3,
    hiddenSize: hidden,
    numLayers: _require<num>(vc, 'num_hidden_layers').toInt(),
    numHeads: numHeads,
    headDim: (vc['head_dim'] as num?)?.toInt() ?? hidden ~/ numHeads,
    intermediateSize: _require<num>(vc, 'intermediate_size').toInt(),
    activation: act,
    layerNormEps: (vc['layer_norm_eps'] as num?)?.toDouble() ?? 1e-5,
    hasClassToken: hasClassToken,
    imageMean: imageMean,
    imageStd: imageStd,
  );
}

/// Builds a [ModelGraph] from a parsed HuggingFace `config.json` map.
ModelGraph parseHfConfig(Map<String, dynamic> json, {String? name}) {
  // Idefics3 / SmolVLM wrap a text LLM (`text_config`, weights under
  // `model.text_model`) + a SigLIP vision tower (`vision_config`, under
  // `model.vision_model`) + a pixel-shuffle connector.
  final mt = json['model_type'] as String?;
  if ((mt == 'idefics3' || mt == 'smolvlm') && json['text_config'] is Map) {
    return _parseIdefics3(json, name);
  }
  final arch = _archOf(_require<String>(json, 'model_type'));
  final hiddenSize = _require<num>(json, 'hidden_size').toInt();
  final numLayers = _require<num>(json, 'num_hidden_layers').toInt();
  final numHeads = _require<num>(json, 'num_attention_heads').toInt();
  final numKvHeads = (json['num_key_value_heads'] as num?)?.toInt() ?? numHeads;
  final intermediate = _require<num>(json, 'intermediate_size').toInt();
  final vocabSize = _require<num>(json, 'vocab_size').toInt();
  final maxSeqLen = (json['max_position_embeddings'] as num?)?.toInt() ?? 2048;
  final normEps = (json['rms_norm_eps'] as num?)?.toDouble() ?? 1e-5;
  final ropeTheta = (json['rope_theta'] as num?)?.toDouble() ?? 10000.0;
  final tie = (json['tie_word_embeddings'] as bool?) ?? false;
  final act = switch (json['hidden_act'] as String? ?? 'silu') {
    'silu' || 'swish' => ActivationKind.silu,
    'gelu' || 'gelu_new' => ActivationKind.gelu,
    'relu' => ActivationKind.relu,
    final other => throw ArgumentError.value(other, 'hidden_act'),
  };

  if (numHeads == 0) {
    throw ArgumentError.value(numHeads, 'num_attention_heads', 'must be > 0');
  }
  final headDim = (json['head_dim'] as num?)?.toInt() ?? hiddenSize ~/ numHeads;

  // Qwen2 gives q/k/v projections a bias. For llama-family models it is opt-in
  // via `attention_bias` (default off, as SmolLM2 sets it).
  final qkvBias = (json['attention_bias'] as bool?) ?? (arch == LlmArch.qwen2);

  // Mixture-of-Experts FFN: Mixtral uses `num_local_experts`, Qwen-MoE uses
  // `num_experts`. Shared experts (Qwen2-MoE/DeepSeek) are a follow-on (numShared 0).
  final numExperts =
      (json['num_local_experts'] as num?)?.toInt() ??
      (json['num_experts'] as num?)?.toInt();
  final moe = numExperts == null
      ? null
      : MoeSpec(
          numExperts: numExperts,
          topK: (json['num_experts_per_tok'] as num).toInt(),
          moeIntermediate:
              (json['moe_intermediate_size'] as num?)?.toInt() ?? intermediate,
          normTopK: (json['norm_topk_prob'] as bool?) ?? true,
        );

  // Modern decoder-only LLMs (llama/qwen2) use RMSNorm + RoPE + gated SiLU MLP.
  // gpt2 differs (LayerNorm/learned-pos/non-gated) and is handled when that
  // path is exercised. For now build the common decoder block.
  final isModern = arch == LlmArch.llama || arch == LlmArch.qwen2;

  final layers = List.generate(
    numLayers,
    (i) => LayerSpec(
      index: i,
      normKind: isModern ? NormKind.rmsNorm : NormKind.layerNorm,
      normEps: normEps,
      attention: AttentionSpec(
        numHeads: numHeads,
        numKvHeads: numKvHeads,
        headDim: headDim,
        posEncoding: isModern ? PosEncoding.rope : PosEncoding.learned,
        ropeTheta: ropeTheta,
        qkvBias: qkvBias,
      ),
      mlp: MlpSpec(
        intermediateSize: intermediate,
        activation: act,
        gated: isModern,
        moe: moe,
      ),
    ),
  );

  // Multi-Token Prediction heads (DeepSeek-V3: `num_nextn_predict_layers`).
  // Absent or 0 -> no MTP.
  final numMtp = (json['num_nextn_predict_layers'] as num?)?.toInt() ?? 0;
  final mtp = numMtp > 0 ? MtpSpec(numModules: numMtp) : null;

  // Vision tower for a VLM: a `vision_config` sub-dict (CLIP/SigLIP). Attached
  // when present. A text-only model leaves it null.
  final vc = json['vision_config'];
  final vision = vc is Map<String, dynamic> ? parseVisionConfig(vc) : null;

  // Multimodal projector + image placeholder token (VLMs). Default to a 2-layer
  // GELU MLP (LLaVA `mlp2x_gelu` / modern HF), a single linear when the config
  // says `mm_projector_type: linear`.
  final projType = json['mm_projector_type'] as String?;
  final projector = vision == null
      ? null
      : ProjectorSpec(
          numLayers: (projType != null && !projType.contains('mlp')) ? 1 : 2,
          activation: ActivationKind.gelu,
        );
  final imageTokenIndex = (json['image_token_index'] as num?)?.toInt();

  final graph = ModelGraph(
    name: name ?? json['model_type'] as String,
    arch: arch,
    hiddenSize: hiddenSize,
    vocabSize: vocabSize,
    maxSeqLen: maxSeqLen,
    tieEmbeddings: tie,
    // BitNet-b1.58 marker: emit every linear as ternary {-1,0,+1} + per-tensor scale.
    ternary: json['quant'] == 'bitnet_ternary',
    layers: layers,
    mtp: mtp,
    vision: vision,
    projector: projector,
    imageTokenIndex: imageTokenIndex,
  );
  graph.validate();
  return graph;
}

/// Builds an Idefics3 / SmolVLM graph: the text LLM from `text_config` (weights
/// under `model.text_model`), the SigLIP vision tower from `vision_config`
/// (`model.vision_model`), and a pixel-shuffle connector (`scale_factor`).
ModelGraph _parseIdefics3(Map<String, dynamic> json, String? name) {
  final tc = Map<String, dynamic>.from(json['text_config'] as Map);
  tc['model_type'] ??= 'llama';
  final text = parseHfConfig(tc); // flat llama parse for the text tower

  final vision = parseVisionConfig(
    Map<String, dynamic>.from(json['vision_config'] as Map),
  );
  final scale = (json['scale_factor'] as num?)?.toInt() ?? 1;
  final connector = ProjectorSpec(
    numLayers: 1,
    activation: ActivationKind.gelu,
    kind: ProjectorKind.pixelShuffle,
    scaleFactor: scale,
  );
  // Idefics3 keys the placeholder as `image_token_id`.
  final imgTok =
      (json['image_token_id'] as num?)?.toInt() ??
      (json['image_token_index'] as num?)?.toInt();
  final tie = (json['tie_word_embeddings'] as bool?) ?? text.tieEmbeddings;

  final graph = ModelGraph(
    name: name ?? (json['model_type'] as String? ?? 'idefics3'),
    arch: text.arch,
    hiddenSize: text.hiddenSize,
    vocabSize: text.vocabSize,
    maxSeqLen: text.maxSeqLen,
    tieEmbeddings: tie,
    layers: text.layers,
    vision: vision,
    projector: connector,
    imageTokenIndex: imgTok,
    textPrefix: 'model.text_model',
    visionPrefix: 'model.vision_model',
  );
  graph.validate();
  return graph;
}
