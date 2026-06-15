library;

/// Recognized LLM architectures (drives op selection downstream).
enum LlmArch { llama, qwen2, gpt2 }

/// Normalization flavor.
enum NormKind { rmsNorm, layerNorm }

/// Positional encoding scheme.
enum PosEncoding { rope, learned, none }

/// MLP activation.
enum ActivationKind { silu, gelu, relu }

/// Self-attention block shape for one layer.
class AttentionSpec {
  final int numHeads;
  final int numKvHeads;
  final int headDim;
  final PosEncoding posEncoding;
  final double ropeTheta;

  /// The q/k/v projections carry a bias vector (Qwen2). o_proj stays bias-free.
  final bool qkvBias;

  const AttentionSpec({
    required this.numHeads,
    required this.numKvHeads,
    required this.headDim,
    required this.posEncoding,
    this.ropeTheta = 10000.0,
    this.qkvBias = false,
  });
}

/// Mixture-of-Experts shape for one MLP block. Each expert is an ordinary gated
/// FFN (same matmul datapath). A router picks [topK] of [numExperts] per token.
class MoeSpec {
  final int numExperts;
  final int topK;

  /// Per-expert intermediate size (may differ from a dense layer's).
  final int moeIntermediate;

  /// Renormalize the top-k router weights to sum to 1 (Qwen-MoE, Mixtral).
  final bool normTopK;

  /// Always-on shared experts added to the routed output (Qwen2-MoE, DeepSeek);
  /// 0 = none. Each shared expert is a gated FFN over `numShared*moeIntermediate`.
  final int numShared;

  const MoeSpec({
    required this.numExperts,
    required this.topK,
    required this.moeIntermediate,
    this.normTopK = true,
    this.numShared = 0,
  });
}

/// Multi-Token Prediction (DeepSeek-V3 style): [numModules] extra "next-token"
/// prediction heads. Each is a transformer block (same shape as a main layer)
/// fed a projection of concat(norm(prev_hidden), norm(embed(next_token))), then
/// the shared final-norm + lm_head. Drives speculative decoding: the heads draft
/// k tokens per forward, a single verify pass keeps the longest correct prefix.
/// Hardware-free (extra matmuls + host draft/verify), a SPEED lever.
class MtpSpec {
  /// Number of MTP modules (DeepSeek `num_nextn_predict_layers`). Each predicts
  /// one further token (module m predicts token t+2+m).
  final int numModules;

  const MtpSpec({required this.numModules});
}

/// A Vision Transformer encoder tower (CLIP / SigLIP style) for a VLM. The ViT
/// is the same matmul datapath as the LLM (attention + MLP), differing in: a
/// patch-embedding conv (lowered to im2col + matmul), LayerNorm instead of
/// RMSNorm, GELU instead of SiLU, learned position embeddings, bidirectional
/// (non-causal) attention, and an optional leading class token.
class VisionTowerSpec {
  final int imageSize;
  final int patchSize;
  final int numChannels;
  final int hiddenSize;
  final int numLayers;
  final int numHeads;
  final int headDim;
  final int intermediateSize;
  final ActivationKind activation;
  final double layerNormEps;

  /// CLIP prepends a learned class token (so seqLen = numPatches + 1); SigLIP
  /// does not.
  final bool hasClassToken;

  /// Per-channel image normalization (from the HF image processor). CLIP uses
  /// its own mean/std. SigLIP uses 0.5. Used by the host preprocessing that
  /// turns a decoded image into the tower's input tensor.
  final List<double> imageMean;
  final List<double> imageStd;

  const VisionTowerSpec({
    required this.imageSize,
    required this.patchSize,
    required this.numChannels,
    required this.hiddenSize,
    required this.numLayers,
    required this.numHeads,
    required this.headDim,
    required this.intermediateSize,
    required this.activation,
    required this.layerNormEps,
    required this.hasClassToken,
    required this.imageMean,
    required this.imageStd,
  });

  /// Patches per side and total (a square image tiled by patchSize).
  int get patchesPerSide => imageSize ~/ patchSize;
  int get numPatches => patchesPerSide * patchesPerSide;

  /// Token count fed to the transformer (patches plus the optional class token).
  int get seqLen => numPatches + (hasClassToken ? 1 : 0);
}

/// How the multimodal projector maps vision embeddings into the LLM token space.
/// [linear] is a single matmul; [mlpGelu] is LLaVA's `mlp2x_gelu` (linear, GELU,
/// linear); [pixelShuffle] is Idefics3/SmolVLM (space-to-depth merge of
/// `scaleFactor^2` neighbouring patches, then a single linear).
enum ProjectorKind { linear, mlpGelu, pixelShuffle }

/// The multimodal projector spec. [numLayers] is 1 or 2 (LLaVA); [scaleFactor]
/// is the pixel-shuffle merge factor (Idefics3; 1 for the LLaVA kinds).
class ProjectorSpec {
  final int numLayers;
  final ActivationKind activation;
  final ProjectorKind kind;
  final int scaleFactor;
  const ProjectorSpec({
    required this.numLayers,
    required this.activation,
    this.kind = ProjectorKind.mlpGelu,
    this.scaleFactor = 1,
  });
}

class MlpSpec {
  final int intermediateSize;
  final ActivationKind activation;

  /// `true` for gated MLPs (SwiGLU): two projections multiplied before down.
  final bool gated;

  /// Non-null when this layer's FFN is a Mixture of Experts.
  final MoeSpec? moe;

  const MlpSpec({
    required this.intermediateSize,
    required this.activation,
    required this.gated,
    this.moe,
  });
}

/// One transformer block.
class LayerSpec {
  final int index;
  final NormKind normKind;
  final double normEps;
  final AttentionSpec attention;
  final MlpSpec mlp;

  const LayerSpec({
    required this.index,
    required this.normKind,
    required this.normEps,
    required this.attention,
    required this.mlp,
  });
}

/// Architecture-level model description: shape and structure only, no tensor
/// data.
class ModelGraph {
  final String name;
  final LlmArch arch;
  final int hiddenSize;
  final int vocabSize;
  final int maxSeqLen;
  final bool tieEmbeddings;

  /// BitNet-b1.58 ternary weights (config `"quant": "bitnet_ternary"`): every
  /// linear is quantized per-tensor to {-1,0,+1}. genip emits these as int4-
  /// packed ternary with a per-tensor scale (the int4 datapath runs it as-is).
  final bool ternary;

  final List<LayerSpec> layers;

  /// Multi-Token Prediction heads, null for models without them.
  final MtpSpec? mtp;

  /// Vision encoder tower, non-null for vision-language models.
  final VisionTowerSpec? vision;

  /// The vision->text projector (VLMs only. Paired with [vision]).
  final ProjectorSpec? projector;

  /// The token id that marks where image embeddings splice into the sequence
  /// (HF `image_token_index`). Null for text-only models.
  final int? imageTokenIndex;

  /// Weight-name prefix for the text model (`model` for plain llama/qwen2;
  /// `model.text_model` for wrapped VLMs like Idefics3).
  final String textPrefix;

  /// Weight-name prefix for the vision tower (`vision_model` standalone;
  /// `model.vision_model` for Idefics3).
  final String visionPrefix;

  const ModelGraph({
    required this.name,
    required this.arch,
    required this.hiddenSize,
    required this.vocabSize,
    required this.maxSeqLen,
    required this.tieEmbeddings,
    required this.layers,
    this.ternary = false,
    this.mtp,
    this.vision,
    this.projector,
    this.imageTokenIndex,
    this.textPrefix = 'model',
    this.visionPrefix = 'vision_model',
  });

  int get numLayers => layers.length;

  /// Validates structural invariants; see [validateModelGraph].
  void validate() => validateModelGraph(this);
}

/// Validates structural invariants of a [ModelGraph]. Throws [ArgumentError].
void validateModelGraph(ModelGraph g) {
  if (g.hiddenSize <= 0) {
    throw ArgumentError.value(g.hiddenSize, 'hiddenSize', 'must be > 0');
  }
  if (g.maxSeqLen <= 0) {
    throw ArgumentError.value(g.maxSeqLen, 'maxSeqLen', 'must be > 0');
  }
  if (g.vocabSize <= 0) {
    throw ArgumentError.value(g.vocabSize, 'vocabSize', 'must be > 0');
  }
  if (g.layers.isEmpty) {
    throw ArgumentError.value(g.layers, 'layers', 'must be non-empty');
  }
  for (final l in g.layers) {
    final a = l.attention;
    if (a.numHeads <= 0 || a.numKvHeads <= 0) {
      throw ArgumentError('layer ${l.index}: head counts must be > 0');
    }
    if (a.numHeads % a.numKvHeads != 0) {
      throw ArgumentError(
        'layer ${l.index}: numHeads (${a.numHeads}) must be divisible by '
        'numKvHeads (${a.numKvHeads})',
      );
    }
    if (a.headDim * a.numHeads != g.hiddenSize) {
      throw ArgumentError(
        'layer ${l.index}: headDim*numHeads (${a.headDim * a.numHeads}) must '
        'equal hiddenSize (${g.hiddenSize})',
      );
    }
    if (l.normEps <= 0) {
      throw ArgumentError('layer ${l.index}: normEps must be > 0');
    }
    if (l.mlp.intermediateSize <= 0) {
      throw ArgumentError('layer ${l.index}: intermediateSize must be > 0');
    }
  }
}
