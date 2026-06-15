// Vision Transformer encoder golden (CLIP / SigLIP style), pure fp64 reference.
// Pipeline: patch-embed (a conv lowered to im2col + matmul) + optional class
// token + learned position embeddings, then N pre-norm ViT blocks (LayerNorm +
// bidirectional multi-head attention + GELU MLP), then a final LayerNorm. Same
// matmul datapath as the LLM. The differences are LayerNorm (not RMSNorm), GELU
// (not SiLU), learned positions, non-causal attention, and the patch conv.
import 'dart:math' as math;
import 'dart:typed_data';

import '../ir/model_graph.dart';
import '../weights/weight_store.dart';
import 'ops.dart';

/// One ViT block's weights: pre-attention LayerNorm, q/k/v/o projections (with
/// biases), pre-MLP LayerNorm, and the two MLP projections (with biases).
class VisionBlockWeights {
  final Float64List ln1Gamma;
  final Float64List ln1Beta;
  final Float64List qProj;
  final Float64List qBias;
  final Float64List kProj;
  final Float64List kBias;
  final Float64List vProj;
  final Float64List vBias;
  final Float64List oProj;
  final Float64List oBias;
  final Float64List ln2Gamma;
  final Float64List ln2Beta;
  final Float64List fc1;
  final Float64List fc1Bias;
  final Float64List fc2;
  final Float64List fc2Bias;

  const VisionBlockWeights({
    required this.ln1Gamma,
    required this.ln1Beta,
    required this.qProj,
    required this.qBias,
    required this.kProj,
    required this.kBias,
    required this.vProj,
    required this.vBias,
    required this.oProj,
    required this.oBias,
    required this.ln2Gamma,
    required this.ln2Beta,
    required this.fc1,
    required this.fc1Bias,
    required this.fc2,
    required this.fc2Bias,
  });
}

/// A full vision tower's decoded weights plus dimensions.
class VisionWeights {
  /// Patch-embedding conv, flattened to `[hidden x (numChannels*patch*patch)]`
  /// (out-channel major, then in-channel, then kernel row/col), and its bias.
  final Float64List patchEmbed;
  final Float64List patchEmbedBias;

  /// Learned class token `[hidden]`, prepended when [hasClassToken].
  final Float64List? classToken;

  /// Learned position embeddings `[seqLen x hidden]`.
  final Float64List posEmbed;

  /// Optional pre-encoder LayerNorm (CLIP's `pre_layrnorm`. SigLIP has none).
  final Float64List? preLnGamma;
  final Float64List? preLnBeta;

  final List<VisionBlockWeights> blocks;

  /// Final post-encoder LayerNorm.
  final Float64List postLnGamma;
  final Float64List postLnBeta;

  final int hidden;
  final int numHeads;
  final int headDim;
  final int intermediate;
  final int patchSize;
  final int numChannels;
  final int imageSize;
  final double lnEps;
  final bool hasClassToken;

  const VisionWeights({
    required this.patchEmbed,
    required this.patchEmbedBias,
    required this.classToken,
    required this.posEmbed,
    required this.preLnGamma,
    required this.preLnBeta,
    required this.blocks,
    required this.postLnGamma,
    required this.postLnBeta,
    required this.hidden,
    required this.numHeads,
    required this.headDim,
    required this.intermediate,
    required this.patchSize,
    required this.numChannels,
    required this.imageSize,
    required this.lnEps,
    required this.hasClassToken,
  });

  int get patchesPerSide => imageSize ~/ patchSize;
  int get numPatches => patchesPerSide * patchesPerSide;
  int get seqLen => numPatches + (hasClassToken ? 1 : 0);
}

/// Resolves and decodes a vision tower's weights from [store] using the
/// HuggingFace CLIP/SigLIP naming under [prefix] (default `vision_model`). The
/// patch-embedding conv weight (`[hidden, C, P, P]`) flattens to the
/// `[hidden, C*P*P]` matrix [patchify] expects. Class token and pre-LayerNorm
/// are loaded only when present (CLIP has both. SigLIP has neither).
VisionWeights bindVision(
  VisionTowerSpec s,
  WeightStore store, {
  String prefix = 'vision_model',
}) {
  Float64List get(String name) {
    if (!store.contains(name)) {
      throw ArgumentError('missing vision tensor: $name');
    }
    return store.get(name).toFloat64List();
  }

  Float64List? getOpt(String name) =>
      store.contains(name) ? store.get(name).toFloat64List() : null;

  final blocks = <VisionBlockWeights>[
    for (var i = 0; i < s.numLayers; i++)
      VisionBlockWeights(
        ln1Gamma: get('$prefix.encoder.layers.$i.layer_norm1.weight'),
        ln1Beta: get('$prefix.encoder.layers.$i.layer_norm1.bias'),
        qProj: get('$prefix.encoder.layers.$i.self_attn.q_proj.weight'),
        qBias: get('$prefix.encoder.layers.$i.self_attn.q_proj.bias'),
        kProj: get('$prefix.encoder.layers.$i.self_attn.k_proj.weight'),
        kBias: get('$prefix.encoder.layers.$i.self_attn.k_proj.bias'),
        vProj: get('$prefix.encoder.layers.$i.self_attn.v_proj.weight'),
        vBias: get('$prefix.encoder.layers.$i.self_attn.v_proj.bias'),
        oProj: get('$prefix.encoder.layers.$i.self_attn.out_proj.weight'),
        oBias: get('$prefix.encoder.layers.$i.self_attn.out_proj.bias'),
        ln2Gamma: get('$prefix.encoder.layers.$i.layer_norm2.weight'),
        ln2Beta: get('$prefix.encoder.layers.$i.layer_norm2.bias'),
        fc1: get('$prefix.encoder.layers.$i.mlp.fc1.weight'),
        fc1Bias: get('$prefix.encoder.layers.$i.mlp.fc1.bias'),
        fc2: get('$prefix.encoder.layers.$i.mlp.fc2.weight'),
        fc2Bias: get('$prefix.encoder.layers.$i.mlp.fc2.bias'),
      ),
  ];

  return VisionWeights(
    patchEmbed: get('$prefix.embeddings.patch_embedding.weight'),
    patchEmbedBias:
        getOpt('$prefix.embeddings.patch_embedding.bias') ??
        Float64List(s.hiddenSize),
    classToken: s.hasClassToken
        ? get('$prefix.embeddings.class_embedding')
        : null,
    posEmbed: get('$prefix.embeddings.position_embedding.weight'),
    preLnGamma: getOpt('$prefix.pre_layrnorm.weight'),
    preLnBeta: getOpt('$prefix.pre_layrnorm.bias'),
    blocks: blocks,
    postLnGamma: get('$prefix.post_layernorm.weight'),
    postLnBeta: get('$prefix.post_layernorm.bias'),
    hidden: s.hiddenSize,
    numHeads: s.numHeads,
    headDim: s.headDim,
    intermediate: s.intermediateSize,
    patchSize: s.patchSize,
    numChannels: s.numChannels,
    imageSize: s.imageSize,
    lnEps: s.layerNormEps,
    hasClassToken: s.hasClassToken,
  );
}

/// Splits a channels-first image `[numChannels x imageSize x imageSize]` into
/// non-overlapping `patchSize` patches, each flattened to length
/// `numChannels*patch*patch` in (channel, row, col) order to match the conv
/// weight layout. Returns `numPatches` patch vectors in row-major patch order.
List<Float64List> patchify(Float64List pixels, VisionWeights w) {
  final c = w.numChannels, s = w.imageSize, p = w.patchSize;
  final pps = s ~/ p;
  final expected = c * s * s;
  if (pixels.length != expected) {
    throw ArgumentError('pixels.length ${pixels.length} != $expected (C*S*S)');
  }
  final patchLen = c * p * p;
  final out = <Float64List>[];
  for (var py = 0; py < pps; py++) {
    for (var px = 0; px < pps; px++) {
      final vec = Float64List(patchLen);
      var idx = 0;
      for (var ch = 0; ch < c; ch++) {
        for (var dy = 0; dy < p; dy++) {
          for (var dx = 0; dx < p; dx++) {
            vec[idx++] = pixels[ch * s * s + (py * p + dy) * s + (px * p + dx)];
          }
        }
      }
      out.add(vec);
    }
  }
  return out;
}

/// Patch-embed + class token + position embeddings. Returns `seqLen` rows.
List<Float64List> visionEmbed(Float64List pixels, VisionWeights w) {
  final patchLen = w.numChannels * w.patchSize * w.patchSize;
  final patches = patchify(pixels, w);
  final rows = <Float64List>[];
  if (w.hasClassToken) {
    rows.add(Float64List.fromList(w.classToken!));
  }
  for (final patch in patches) {
    rows.add(
      linear(w.patchEmbed, w.hidden, patchLen, patch, bias: w.patchEmbedBias),
    );
  }
  // Add learned position embeddings (row i gets posEmbed[i]).
  for (var i = 0; i < rows.length; i++) {
    for (var d = 0; d < w.hidden; d++) {
      rows[i][d] += w.posEmbed[i * w.hidden + d];
    }
  }
  return rows;
}

/// Bidirectional (non-causal) multi-head attention: every position attends to
/// every position. `q`/`k`/`v` are `seqLen` rows of `numHeads*headDim`.
List<Float64List> bidirectionalAttention(
  List<Float64List> q,
  List<Float64List> k,
  List<Float64List> v,
  int numHeads,
  int headDim,
) {
  final t = q.length;
  final dim = numHeads * headDim;
  final scale = 1.0 / math.sqrt(headDim.toDouble());
  final out = List<Float64List>.generate(t, (_) => Float64List(dim));
  for (var h = 0; h < numHeads; h++) {
    final off = h * headDim;
    for (var i = 0; i < t; i++) {
      final scores = Float64List(t);
      for (var j = 0; j < t; j++) {
        var dot = 0.0;
        for (var d = 0; d < headDim; d++) {
          dot += q[i][off + d] * k[j][off + d];
        }
        scores[j] = dot * scale;
      }
      final sm = softmax(scores);
      for (var j = 0; j < t; j++) {
        final weight = sm[j];
        for (var d = 0; d < headDim; d++) {
          out[i][off + d] += weight * v[j][off + d];
        }
      }
    }
  }
  return out;
}

/// One pre-norm ViT block: `x + attn(LN1(x))` then `h + mlp(LN2(h))`.
List<Float64List> visionBlock(
  List<Float64List> x,
  VisionBlockWeights bw,
  VisionWeights w,
) {
  final t = x.length;
  final q = <Float64List>[];
  final k = <Float64List>[];
  final v = <Float64List>[];
  for (var i = 0; i < t; i++) {
    final n1 = layerNorm(x[i], bw.ln1Gamma, bw.ln1Beta, w.lnEps);
    q.add(linear(bw.qProj, w.hidden, w.hidden, n1, bias: bw.qBias));
    k.add(linear(bw.kProj, w.hidden, w.hidden, n1, bias: bw.kBias));
    v.add(linear(bw.vProj, w.hidden, w.hidden, n1, bias: bw.vBias));
  }
  final attn = bidirectionalAttention(q, k, v, w.numHeads, w.headDim);

  final h1 = <Float64List>[];
  for (var i = 0; i < t; i++) {
    final o = linear(bw.oProj, w.hidden, w.hidden, attn[i], bias: bw.oBias);
    final r = Float64List(w.hidden);
    for (var d = 0; d < w.hidden; d++) {
      r[d] = x[i][d] + o[d];
    }
    h1.add(r);
  }

  final out = <Float64List>[];
  for (var i = 0; i < t; i++) {
    final n2 = layerNorm(h1[i], bw.ln2Gamma, bw.ln2Beta, w.lnEps);
    final f1 = linear(bw.fc1, w.intermediate, w.hidden, n2, bias: bw.fc1Bias);
    final act = gelu(f1);
    final f2 = linear(bw.fc2, w.hidden, w.intermediate, act, bias: bw.fc2Bias);
    final r = Float64List(w.hidden);
    for (var d = 0; d < w.hidden; d++) {
      r[d] = h1[i][d] + f2[d];
    }
    out.add(r);
  }
  return out;
}

/// Encodes a preprocessed image (`[numChannels x imageSize x imageSize]`,
/// already resized + normalized) into `seqLen` patch embeddings of `hidden`.
List<Float64List> encodeImage(Float64List pixels, VisionWeights w) {
  var x = visionEmbed(pixels, w);
  if (w.preLnGamma != null) {
    x = [
      for (final row in x) layerNorm(row, w.preLnGamma!, w.preLnBeta!, w.lnEps),
    ];
  }
  for (final bw in w.blocks) {
    x = visionBlock(x, bw, w);
  }
  return [
    for (final row in x) layerNorm(row, w.postLnGamma, w.postLnBeta, w.lnEps),
  ];
}
