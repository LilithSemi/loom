library;

import 'dart:typed_data';

import '../golden/quant.dart';
import '../runtime/half.dart';
import '../runtime/loom_device.dart' show packTileMajorInt4;
import 'binding.dart';

/// One linear weight matrix's placement in a [FlashWeightImage].
///
/// [weightOffset] is a byte offset into [FlashWeightImage.weights] (the base the
/// sequencer loads into `LoomFpLinear`'s `weight_base` CSR); [scaleOffset] is an
/// fp16-element offset into [FlashWeightImage.scales]. Together with [colTiles]/
/// [wordsPerRow] this is the full weight-base table the on-chip sequencer needs.
class LinearWeightEntry {
  /// Logical name, e.g. `layers.3.q_proj` or `lm_head`.
  final String name;
  final int weightOffset;
  final int weightLength;
  final int scaleOffset;
  final int rows;
  final int cols;
  final int colTiles;
  final int wordsPerRow;

  /// Number of per-row scale groups along the input dim. `1` = one scale per row
  /// (per-tensor-row W4A8). When the matrix col-tiles with per-group W4A8, this
  /// is the col-block count and the scale table is group-major (`groups*rows`
  /// fp16, block b's `rows` scales at scaleOffset + b*rows).
  final int groups;

  const LinearWeightEntry({
    required this.name,
    required this.weightOffset,
    required this.weightLength,
    required this.scaleOffset,
    required this.rows,
    required this.cols,
    required this.colTiles,
    required this.wordsPerRow,
    this.groups = 1,
  });
}

/// The flash-resident weight artifact: every linear packed int4 tile-major, a
/// parallel per-row fp16 scale table, and a manifest describing where each
/// matrix lives. Backend-agnostic: the same bytes go to SPI flash, BRAM, or DDR
/// behind the [WeightStore] interface. The manifest is what the generated
/// sequencer consumes.
class FlashWeightImage {
  /// Concatenated tile-major int4 weight images, each 4-byte aligned.
  final Uint8List weights;

  /// Concatenated per-row fp16 scales (2 bytes each, little-endian).
  final Uint8List scales;

  final List<LinearWeightEntry> manifest;

  const FlashWeightImage({
    required this.weights,
    required this.scales,
    required this.manifest,
  });
}

/// Packs a rows x cols int4 matrix as consecutive col-blocks of width [blockCols]
/// (each an independent tile-major image), the layout the runtime col-tiler
/// streams. Falls back to one block when [blockCols] >= cols.
Uint8List _packColBlocksInt4(
  Int8List values,
  int rows,
  int cols,
  int blockCols,
) {
  if (blockCols >= cols) return packTileMajorInt4(values, rows, cols);
  final out = BytesBuilder();
  var c = 0;
  while (c < cols) {
    final bc = (c + blockCols <= cols) ? blockCols : cols - c;
    final block = Int8List(rows * bc);
    for (var r = 0; r < rows; r++) {
      for (var k = 0; k < bc; k++) {
        block[r * bc + k] = values[r * cols + c + k];
      }
    }
    out.add(packTileMajorInt4(block, rows, bc));
    c += blockCols;
  }
  return out.toBytes();
}

/// Builds the [FlashWeightImage] for [model]: every projection of every layer
/// (q/k/v/o, gate/up/down) then the (possibly tied) lm_head, in the order the
/// sequencer consumes them. Reuses the exact int4 row-wise quantization and
/// tile-major packing the device already implements, so a matrix reconstructed
/// from the image reproduces the golden W4A8 linear bit-for-bit.
///
/// The token-embedding table, norm gammas, and RoPE tables are NOT included:
/// those are gathers/vectors with a different on-chip access pattern, provided
/// separately. [maxCols] > 0 lays matrices wider than it out as consecutive
/// col-blocks (each a tile-major image of [rows x blockCols]) so the runtime can
/// column-tile them on a small accelerator (see linearColTiled). 0 disables
/// col-tiling (single block).
FlashWeightImage flashImageFor(
  BoundModel model, {
  int maxCols = 0,
  bool ternary = false,
}) {
  final weights = BytesBuilder();
  final scales = BytesBuilder();
  final manifest = <LinearWeightEntry>[];
  var wCursor = 0;
  var sCursor = 0; // in fp16 elements

  void add(String name, List<int> shape, Float64List w) {
    final rows = shape[0];
    final cols = shape[1];
    final blockCols = (maxCols > 0 && cols > maxCols) ? maxCols : cols;
    // A col-tiled matrix uses per-group W4A8 with group == col-block, so each
    // block dequants on its own scale (recovers most of the fp accuracy the
    // per-row cliff drops). Non-tiled matrices stay per-row (one group).
    // Ternary never groups: it is per-tensor (one beta), which the runtime reads
    // as groups==1 (shared scale across any col-blocks).
    final grouped = !ternary && blockCols < cols;

    final int gpr;
    final Int8List values;
    // Group-major scale table: [group0's rows][group1's rows]... so the runtime
    // reads block b's scales contiguously at scaleOffset + b*rows.
    final Float64List scaleTable;
    if (ternary) {
      // BitNet-b1.58 ternary: per-tensor absmean beta, replicated into every
      // rowScales[r]. Values {-1,0,+1} ARE valid int4, so the int4 col-block
      // packing and the whole int4 sim/runtime/RTL datapath run this unchanged
      // (groups==1 = shared scale). This is 4-bit-packed ternary (correctness +
      // fits flash). The 2-bit byte-win packing is a separate follow-on.
      final qm = quantizeTernaryAbsmean(w, rows, cols);
      values = qm.values;
      scaleTable = qm.rowScales; // every entry == beta
      gpr = 1;
    } else if (grouped) {
      gpr = (cols + blockCols - 1) ~/ blockCols;
      final gqm = quantizeGroupwise(
        w,
        rows,
        cols,
        bits: 4,
        groupSize: blockCols,
      );
      values = gqm.values;
      scaleTable = Float64List(gpr * rows);
      for (var g = 0; g < gpr; g++) {
        for (var r = 0; r < rows; r++) {
          scaleTable[g * rows + r] = gqm.scales[r * gpr + g];
        }
      }
    } else {
      gpr = 1;
      final qm = quantizeRowwiseInt4(w, rows, cols);
      values = qm.values;
      scaleTable = qm.rowScales;
    }
    final img = _packColBlocksInt4(values, rows, cols, blockCols);
    final colTiles = (cols + 1) ~/ 2;
    final wordsPerRow = (colTiles + 1) ~/ 2;

    manifest.add(
      LinearWeightEntry(
        name: name,
        weightOffset: wCursor,
        weightLength: img.length,
        scaleOffset: sCursor,
        rows: rows,
        cols: cols,
        colTiles: colTiles,
        wordsPerRow: wordsPerRow,
        groups: gpr,
      ),
    );

    weights.add(img);
    wCursor += img.length;
    while (wCursor % 4 != 0) {
      weights.addByte(0);
      wCursor++;
    }

    for (var i = 0; i < scaleTable.length; i++) {
      final h = Half.fromDouble(scaleTable[i]);
      scales.addByte(h & 0xFF);
      scales.addByte((h >> 8) & 0xFF);
    }
    sCursor += scaleTable.length;
  }

  for (var i = 0; i < model.layers.length; i++) {
    final l = model.layers[i];
    add('layers.$i.q_proj', l.qProj.shape, l.qProj.toFloat64List());
    add('layers.$i.k_proj', l.kProj.shape, l.kProj.toFloat64List());
    add('layers.$i.v_proj', l.vProj.shape, l.vProj.toFloat64List());
    add('layers.$i.o_proj', l.oProj.shape, l.oProj.toFloat64List());
    if (l.moe != null) {
      // MoE: every expert is a gated FFN riding the same int4 datapath. The
      // router is tiny + precision-sensitive, so it ships as fp16 glue (host
      // matmul), NOT here. Experts are named layers.$i.experts.$e.{gate,up,down}
      // _proj so the runtime can address each selected expert by name.
      final experts = l.moe!.experts;
      for (var e = 0; e < experts.length; e++) {
        add(
          'layers.$i.experts.$e.gate_proj',
          experts[e].gate.shape,
          experts[e].gate.toFloat64List(),
        );
        add(
          'layers.$i.experts.$e.up_proj',
          experts[e].up.shape,
          experts[e].up.toFloat64List(),
        );
        add(
          'layers.$i.experts.$e.down_proj',
          experts[e].down.shape,
          experts[e].down.toFloat64List(),
        );
      }
    } else {
      add('layers.$i.gate_proj', l.gate!.shape, l.gate!.toFloat64List());
      add('layers.$i.up_proj', l.up!.shape, l.up!.toFloat64List());
      add('layers.$i.down_proj', l.down!.shape, l.down!.toFloat64List());
    }
  }
  add('lm_head', model.lmHead.shape, model.lmHead.toFloat64List());

  // Multi-Token Prediction modules: the fusion projection + each module's dense
  // transformer block, named mtp.$m.* so the runtime addresses them by name.
  // enorm/hnorm and the block norms are fp16 glue (emitted separately).
  final mtp = model.mtp;
  if (mtp != null) {
    for (var m = 0; m < mtp.modules.length; m++) {
      final mod = mtp.modules[m];
      final b = mod.block;
      add('mtp.$m.eh_proj', mod.ehProj.shape, mod.ehProj.toFloat64List());
      add('mtp.$m.q_proj', b.qProj.shape, b.qProj.toFloat64List());
      add('mtp.$m.k_proj', b.kProj.shape, b.kProj.toFloat64List());
      add('mtp.$m.v_proj', b.vProj.shape, b.vProj.toFloat64List());
      add('mtp.$m.o_proj', b.oProj.shape, b.oProj.toFloat64List());
      add('mtp.$m.gate_proj', b.gate!.shape, b.gate!.toFloat64List());
      add('mtp.$m.up_proj', b.up!.shape, b.up!.toFloat64List());
      add('mtp.$m.down_proj', b.down!.shape, b.down!.toFloat64List());
    }
  }

  // Vision tower: patch-embed conv (flattened) + per-block attention/MLP
  // matrices, named vision.* so the runtime addresses them by name. LayerNorm
  // gammas/betas, biases, class token, and position embeddings are fp16 glue.
  final vision = model.vision;
  if (vision != null) {
    final h = vision.hidden;
    final patchLen = vision.numChannels * vision.patchSize * vision.patchSize;
    add('vision.patch_embed', [h, patchLen], vision.patchEmbed);
    for (var i = 0; i < vision.blocks.length; i++) {
      final b = vision.blocks[i];
      add('vision.layers.$i.q_proj', [h, h], b.qProj);
      add('vision.layers.$i.k_proj', [h, h], b.kProj);
      add('vision.layers.$i.v_proj', [h, h], b.vProj);
      add('vision.layers.$i.out_proj', [h, h], b.oProj);
      add('vision.layers.$i.fc1', [vision.intermediate, h], b.fc1);
      add('vision.layers.$i.fc2', [h, vision.intermediate], b.fc2);
    }
  }

  // Multimodal projector matrices (vision->text).
  final projector = model.projector;
  if (projector != null) {
    if (projector.isTwoLayer) {
      add('projector.linear_1', [
        projector.hiddenDim,
        projector.inputDim,
      ], projector.linear1);
      add('projector.linear_2', [
        projector.outputDim,
        projector.hiddenDim,
      ], projector.linear2!);
    } else {
      add('projector.linear_1', [
        projector.outputDim,
        projector.inputDim,
      ], projector.linear1);
    }
  }

  return FlashWeightImage(
    weights: weights.toBytes(),
    scales: scales.toBytes(),
    manifest: manifest,
  );
}
