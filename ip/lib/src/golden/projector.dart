// Multimodal projector for a VLM: maps vision-tower patch embeddings into the
// LLM's token-embedding space so they can be spliced in as "image tokens". Two
// shapes, matching HF LLaVA: a single linear ("linear"), or a two-layer GELU
// MLP ("mlp2x_gelu": linear_1 -> GELU -> linear_2). Pure matmuls + GELU.
import 'dart:math' as math;
import 'dart:typed_data';

import '../ir/model_graph.dart';
import '../weights/weight_store.dart';
import 'ops.dart';

/// Decoded projector weights. For the 2-layer MLP, [linear1] is
/// `[hiddenDim x inputDim]` and [linear2] is `[outputDim x hiddenDim]`. For the
/// single-linear form, [linear1] is `[outputDim x inputDim]` and [linear2] is
/// null.
class ProjectorWeights {
  final Float64List linear1;
  final Float64List? bias1;
  final Float64List? linear2;
  final Float64List? bias2;
  final int inputDim; // projector input dim (vision hidden, or vision*scale^2)
  final int hiddenDim; // 2-layer inner dim (== outputDim for LLaVA)
  final int outputDim; // text hidden

  /// Pixel-shuffle merge factor (Idefics3); 1 for the LLaVA kinds.
  final int scaleFactor;

  const ProjectorWeights({
    required this.linear1,
    required this.bias1,
    required this.linear2,
    required this.bias2,
    required this.inputDim,
    required this.hiddenDim,
    required this.outputDim,
    this.scaleFactor = 1,
  });

  bool get isTwoLayer => linear2 != null;
  bool get isPixelShuffle => scaleFactor > 1;
}

/// Resolves and decodes the multimodal projector from [store] using the HF
/// LLaVA naming under [prefix] (default `multi_modal_projector`): `linear_1`
/// (and `linear_2` for the 2-layer form). [visionHidden] and [textHidden] are
/// the input/output dims. The 2-layer inner dim is [textHidden] (LLaVA).
ProjectorWeights bindProjector(
  ProjectorSpec spec,
  WeightStore store,
  int visionHidden,
  int textHidden, {
  String prefix = 'multi_modal_projector',
}) {
  Float64List get(String name) {
    if (!store.contains(name)) {
      throw ArgumentError('missing projector tensor: $name');
    }
    return store.get(name).toFloat64List();
  }

  Float64List? getOpt(String name) =>
      store.contains(name) ? store.get(name).toFloat64List() : null;

  // Idefics3 / SmolVLM: pixel-shuffle merges scale^2 neighbouring patches, then a
  // single (bias-free) linear `model.connector.modality_projection.proj` maps
  // vision_hidden*scale^2 -> text_hidden.
  if (spec.kind == ProjectorKind.pixelShuffle) {
    final inDim = visionHidden * spec.scaleFactor * spec.scaleFactor;
    return ProjectorWeights(
      linear1: get('model.connector.modality_projection.proj.weight'),
      bias1: getOpt('model.connector.modality_projection.proj.bias'),
      linear2: null,
      bias2: null,
      inputDim: inDim,
      hiddenDim: textHidden,
      outputDim: textHidden,
      scaleFactor: spec.scaleFactor,
    );
  }

  if (spec.numLayers == 1) {
    return ProjectorWeights(
      linear1: get('$prefix.linear_1.weight'),
      bias1: getOpt('$prefix.linear_1.bias'),
      linear2: null,
      bias2: null,
      inputDim: visionHidden,
      hiddenDim: textHidden,
      outputDim: textHidden,
    );
  }
  return ProjectorWeights(
    linear1: get('$prefix.linear_1.weight'),
    bias1: getOpt('$prefix.linear_1.bias'),
    linear2: get('$prefix.linear_2.weight'),
    bias2: getOpt('$prefix.linear_2.bias'),
    inputDim: visionHidden,
    hiddenDim: textHidden,
    outputDim: textHidden,
  );
}

/// Projects one vision embedding (`inputDim`) into the text embedding space
/// (`outputDim`).
Float64List projectOne(Float64List v, ProjectorWeights p) {
  if (!p.isTwoLayer) {
    return linear(p.linear1, p.outputDim, p.inputDim, v, bias: p.bias1);
  }
  final h1 = linear(p.linear1, p.hiddenDim, p.inputDim, v, bias: p.bias1);
  final a = gelu(h1);
  return linear(p.linear2!, p.outputDim, p.hiddenDim, a, bias: p.bias2);
}

/// Projects every vision embedding into the text embedding space.
List<Float64List> projectVision(List<Float64List> embeds, ProjectorWeights p) =>
    [for (final e in embeds) projectOne(e, p)];

/// Idefics3 pixel shuffle (space-to-depth): the `seq = side*side` vision
/// embeddings (no class token) are merged in `scale x scale` spatial blocks, so
/// `seq/scale^2` output tokens each carry `embed*scale^2` channels. Exactly
/// replicates HF Idefics3Connector.pixel_shuffle's view/permute sequence.
List<Float64List> pixelShuffle(List<Float64List> embeds, int scale) {
  final seq = embeds.length;
  final e = embeds[0].length;
  final side = math.sqrt(seq).round();
  if (side * side != seq) {
    throw ArgumentError('pixelShuffle: seq $seq is not a perfect square');
  }
  final os = side ~/ scale; // output grid side
  final outSeq = os * os;
  final outE = e * scale * scale;
  final es = e * scale;
  return [
    for (var o = 0; o < outSeq; o++)
      () {
        final h2 = o ~/ os, w2 = o % os;
        final row = Float64List(outE);
        for (var c = 0; c < outE; c++) {
          final hh = c ~/ es; // 0..scale-1
          final rem = c % es;
          final ww = rem ~/ e; // 0..scale-1
          final ec = rem % e; // 0..e-1
          final inTok = (h2 * scale + hh) * side + (w2 * scale + ww);
          row[c] = embeds[inTok][ec];
        }
        return row;
      }(),
  ];
}

/// Runs the connector: for Idefics3, pixel-shuffle then project. Otherwise the
/// per-token LLaVA projection. Returns text-space image embeddings.
List<Float64List> connectorProject(
  List<Float64List> visionEmbeds,
  ProjectorWeights p,
) {
  final tokens = p.isPixelShuffle
      ? pixelShuffle(visionEmbeds, p.scaleFactor)
      : visionEmbeds;
  return [for (final t in tokens) projectOne(t, p)];
}
