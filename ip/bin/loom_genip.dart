/// Generates the Loom SoC RTL + device tree / SVD + ECP5 build scripts.
///
/// Composes a [LoomUsbBridge] (USB-device-to-Wishbone MASTER, the SoC's only
/// bus master) with a [LoomAccelerator] (Wishbone SLAVE peripheral) onto a
/// harbor [HarborSoC] Wishbone fabric, then emits synthesizable SystemVerilog
/// (plus DTS, SVD, ECP5 .lpf, synth.tcl and a Makefile) targeting the
/// OrangeCrab ECP5.
///
/// Net effect: bytes a host streams over USB (DFU DNLOAD) become Wishbone
/// writes into the accelerator's CSRs / buffers. The bridge is the bus master;
/// the accelerator is the only slave. Harbor's [WishboneDecoder] sits between.
///
/// Usage:
///   dart run bin/loom_genip.dart -o out
///
/// Clock-domain note
/// [LoomUsbBridge] natively has TWO clock domains (usb_clk at 48 MHz for the
/// USB engine, bus_clk for the Wishbone master). harbor's
/// [HarborSoC.addMaster] / [HarborSoC.addPeripheral] wire a SINGLE
/// `clk` / `reset` pair onto each submodule, so a split-clock master cannot be
/// added directly. We reconcile by running the WHOLE SoC in one 48 MHz domain:
/// the bridge is built with `singleClock: true`, which ties its USB and bus
/// domains to one `clk`/`reset` pair. This is electrically sound because the
/// RamSink's dual-clock CDC FIFO degenerates to a same-clock FIFO when
/// usb_clk == bus_clk. The accelerator already exposes `clk`/`reset`. So the
/// generated SoC is single-clock at the USB full-speed rate (48 MHz).
///
/// Bus-width note
/// harbor's [WishboneDecoder] gives every slave interface the SoC busConfig
/// address width, and [connectInterfaces] requires matching widths. The
/// accelerator's Wishbone slave is 12 bits wide (its 4 KiB window), so the SoC
/// bus is built 12 bits wide and the bridge master is built 12 bits wide to
/// match. The accelerator therefore sits at base 0x000.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:args/args.dart';
import 'package:harbor/harbor.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';

import 'package:loom/loom.dart';

/// SoC bus address width. Matched to the accelerator's 12-bit slave window so
/// the harbor decoder's slave interface width lines up with the peripheral
/// (connectInterfaces requires equal widths).
///
/// The harbor Wishbone decoder gates each slave with
/// `adr >= start && adr < Const(start+size, width: addressWidth)`. The
/// EXCLUSIVE end (start+size) must be representable in `addressWidth` bits, or
/// it wraps and collapses the decode window to empty. With a 12-bit fabric the
/// accelerator therefore advertises a 2 KiB (0x800) window (end 0x800 fits in
/// 12 bits) rather than the full 4 KiB (end 0x1000 wraps to 0). 0x800 still
/// covers every real register (the highest used region ends at 0x4FF).
const int _busAddressWidth = 12;

/// Accelerator base address inside the SoC map. The peripheral lives at 0x000;
/// with a 12-bit fabric its window [0x000, 0x800) decodes cleanly.
const int _acceleratorBase = 0x000;

/// OrangeCrab r0.2 ECP5 (LFE5U-25F-8MG285C, CSFBGA285) AUTHORITATIVE pin map.
///
/// These are the real OrangeCrab r0.2 board assignments (all LVCMOS33). This
/// map constrains the bare [HarborSoC] top (`LoomSoC`) for reference / fit
/// checks. The FLASHABLE bitstream is built from the board-top wrapper
/// `LoomTop` (see [_boardTopSv] / [_boardTopLpf]) which wraps the split USB
/// driver ports into true bidirectional pads on D+ (N1) / D- (M2) and adds an
/// LED blink + active-low reset invert. The board-top LPF below is the one used
/// for the real build.
///
/// The SoC exposes SPLIT usb_dp(in)/usb_dp_out/usb_oe pads, so this
/// reference map cannot use one physical pad for both the in and the out of D+.
/// The split-out pads here are therefore parked on spare board pins so the bare
/// SoC still constrains cleanly. The bidirectional reality lives in LoomTop.
const Map<String, String> _orangeCrabPinMap = {
  // 48 MHz reference oscillator -> SoC clk.
  'clk': 'A9 LVCMOS33',
  // USB D+ / D- bidirectional pads (real board pins).
  'usb_dp': 'N1 LVCMOS33',
  'usb_dm': 'M2 LVCMOS33',
  // Split driver pads of the bare SoC parked on spare pins (the board-top
  // folds usb_dp_out/usb_dm_out/usb_oe back onto N1/M2 as tristate drivers).
  'usb_dp_out': 'C1 LVCMOS33',
  'usb_dm_out': 'D1 LVCMOS33',
  'usb_oe': 'E1 LVCMOS33',
  // 1k5 FS pull-up enable on D+ (real board pin).
  'usb_pullup': 'N2 LVCMOS33',
};

/// Hand-written OrangeCrab r0.2 board-top SystemVerilog shim (`LoomTop`).
///
/// This is a deliberately minimal BOARD SHIM, not generated RTL. ROHD models
/// 2-state logic and does not emit top-level `inout` tristate pads cleanly, so
/// the bidirectional USB D+/D- pads live here in hand-written SV instead. It:
///
///   1. Folds the SoC's SPLIT usb_dp(in)/usb_dp_out/usb_oe (and the D- set)
///      back onto ONE bidirectional pad each: `assign usb_dp = usb_oe ?
///      usb_dp_out : 1'bz;` and reads the pad back into the SoC input. D- shares
///      usb_oe. This is the true inout the OrangeCrab's single D+/D- balls need.
///   2. Inverts the active-low reset button (rst_n) into the SoC's active-high
///      reset, gated by a small power-on counter so the SoC is held in reset for
///      ~the first 2^16 clocks after config even if the button is never pressed.
///   3. Drives one RGB LED channel (green, M3, active-low) from a free-running
///      counter so liveness is visible on the bench WITHOUT a USB host: the LED
///      blinks at ~48e6/2^24 ~= 2.9 Hz.
///
/// Port names here match `_boardTopLpf` exactly (the real build's constraints).
const String _boardTopSv = '''
// OrangeCrab r0.2 board-top shim for the Loom SoC.
//
// HAND-WRITTEN board shim (not generated). Wraps the generated LoomSoC: real
// OrangeCrab pinout, bidirectional USB D+/D- tristate pads, active-low reset
// invert with a power-on hold, and a free-running LED blink for liveness.
//
// Target: LFE5U-25F-8MG285C (CSFBGA285), all LVCMOS33.
module LoomTop (
    input  logic clk48,    // A9  - 48 MHz oscillator
    input  logic rst_n,    // V17 - reset button, ACTIVE-LOW
    inout  logic usb_dp,   // N1  - USB D+ (bidirectional)
    inout  logic usb_dm,   // M2  - USB D- (bidirectional)
    output logic usb_pullup,// N2 - 1k5 FS pull-up enable on D+
    output logic led        // M3 - RGB green channel, ACTIVE-LOW (lit = 0)
);
  logic soc_usb_dp_in, soc_usb_dm_in;   // pad value read back into the SoC
  logic soc_usb_dp_out, soc_usb_dm_out; // SoC-driven values
  logic soc_usb_oe;                      // shared output-enable for D+/D-
  logic soc_usb_pullup;

  // Hold the SoC in reset for the first 2^16 clocks after config, then release
  // unless the user holds the active-low button (rst_n == 0 => reset asserted).
  logic [15:0] por_cnt = 16'd0;
  logic        por_done = 1'b0;
  always_ff @(posedge clk48) begin
    if (!por_done) begin
      por_cnt  <= por_cnt + 16'd1;
      por_done <= &por_cnt;  // done when counter saturates
    end
  end
  // SoC reset is ACTIVE-HIGH: assert during POR or while the button is pressed.
  logic soc_reset;
  assign soc_reset = (!por_done) | (~rst_n);

  logic [23:0] blink = 24'd0;
  always_ff @(posedge clk48) blink <= blink + 24'd1;
  // RGB LED is active-low (common anode): drive 0 to light. Toggle the MSB.
  assign led = ~blink[23];

  // Drive the pad when the SoC asserts output-enable, else high-Z (the host /
  // pull resistors own the line). Read the pad value back into the SoC input.
  assign usb_dp = soc_usb_oe ? soc_usb_dp_out : 1'bz;
  assign usb_dm = soc_usb_oe ? soc_usb_dm_out : 1'bz;
  assign soc_usb_dp_in = usb_dp;
  assign soc_usb_dm_in = usb_dm;
  assign usb_pullup = soc_usb_pullup;

  LoomSoC u_soc (
      .clk        (clk48),
      .reset      (soc_reset),
      .usb_dp     (soc_usb_dp_in),
      .usb_dm     (soc_usb_dm_in),
      .usb_dp_out (soc_usb_dp_out),
      .usb_dm_out (soc_usb_dm_out),
      .usb_oe     (soc_usb_oe),
      .usb_pullup (soc_usb_pullup)
  );
endmodule
''';

/// OrangeCrab r0.2 LPF for the board-top [_boardTopSv]. Constrains the REAL
/// board pins for `LoomTop`'s ports (every IO constrained, LVCMOS33, 48 MHz on
/// the oscillator pin). This is the constraint file the flashable build uses.
const String _boardTopLpf = '''
# OrangeCrab r0.2 (LFE5U-25F-8MG285C, CSFBGA285) - LoomTop board-shim LPF.
# All LVCMOS33. Authoritative OrangeCrab r0.2 pinout.

# 48 MHz oscillator -> clk48
LOCATE COMP "clk48" SITE "A9";
IOBUF PORT "clk48" IO_TYPE=LVCMOS33;
FREQUENCY PORT "clk48" 48.0 MHz;

# Reset button (active-low)
LOCATE COMP "rst_n" SITE "V17";
IOBUF PORT "rst_n" IO_TYPE=LVCMOS33;

# USB D+/D- bidirectional pads + FS pull-up
LOCATE COMP "usb_dp" SITE "N1";
IOBUF PORT "usb_dp" IO_TYPE=LVCMOS33;
LOCATE COMP "usb_dm" SITE "M2";
IOBUF PORT "usb_dm" IO_TYPE=LVCMOS33;
LOCATE COMP "usb_pullup" SITE "N2";
IOBUF PORT "usb_pullup" IO_TYPE=LVCMOS33;

# RGB LED green channel (active-low)
LOCATE COMP "led" SITE "M3";
IOBUF PORT "led" IO_TYPE=LVCMOS33;
''';

/// OrangeCrab r0.2 pin map for the UART-transport SoC (`LoomUartSoC`).
///
/// The UART build is far simpler than the USB one: just a serial TX/RX pair on
/// the feather GPIO plus the 48 MHz oscillator. The flashable bitstream is the
/// board-top wrapper `LoomUartTop` (see [_uartBoardTopSv] / [_uartBoardTopLpf]),
/// which adds the active-low reset invert + LED blink. This bare-SoC map just
/// lets the un-wrapped SoC constrain cleanly for fit checks.
const Map<String, String> _orangeCrabUartPinMap = {
  // 48 MHz reference oscillator -> SoC clk.
  'clk': 'A9 LVCMOS33',
  // UART on the feather GPIO header (PROVEN orientation: TX=N17, RX=M18).
  'uart_tx': 'N17 LVCMOS33',
  'uart_rx': 'M18 LVCMOS33',
};

/// OrangeCrab r0.2 config SPI flash (GD25Q128, 16MB) used as the RESIDENT weight
/// store. The clock goes through the ECP5 USRMCLK macro (no GPIO pad). The data
/// pads mirror River's proven OrangeCrab wiring (quad IO0..IO3 + CS#).

/// Flash weight-store pads exposed to the SoC top (clock is via USRMCLK). The
/// quad IO is split out/oe/in at the SoC boundary; a Verilog top folds them into
/// the bidirectional spi_io pad, the same way the USB dp/dm pads are handled.
const _flashPads = ['spi_cs_n', 'spi_io_out', 'spi_io_oe', 'spi_io_in'];

/// Flash byte offset the weight image is programmed to (clear of the bitstream).
const _flashWeightOffset = 0x200000;

/// Total W25Q128 config-flash size (16MB). The resident scale image is placed
/// dynamically ABOVE the full flash weight image (see `_emitModelArtifacts`);
/// scales are stored one fp16 per 32-bit word (low16) so the accelerator reads
/// them itself instead of the host pushing them.
const _configFlashBytes = 0x1000000;

/// Emits the model's flash weight artifacts alongside the SoC RTL: `weights.bin`
/// (int4 tile-major, `ecpprog -o 0x200000`), `scales.bin` (per-row fp16 the host
/// pushes), and `loom.json` (model dims + per-matrix manifest + CSR/flash
/// addresses) - the config the runtime consumes instead of hardcoding. Loads a
/// llama2.c `.bin` checkpoint (stories260K and friends).
/// Repacks a SUBSET of [entries] (each a slice of the global [srcWeights] /
/// [srcScales]) into a standalone tile-major int4 weight image + a parallel
/// resident per-row scale image (one fp16 per 32-bit word, low16, rest zero) so
/// the accelerator reads a matrix's scales straight from memory (SCALE_BASE).
/// Returns fresh per-matrix byte offsets into each image, tagged with [store].
/// This is the per-store packing for the tiered (BRAM|flash) weight split: each
/// store's images start at offset 0, so the runtime sets WEIGHT_BASE/SCALE_BASE
/// to that store's absolute base plus the recorded offset.
({Uint8List weights, Uint8List scales, List<Map<String, Object>> entries})
_packStore(
  List<LinearWeightEntry> entries,
  Uint8List srcWeights,
  Uint8List srcScales,
  String store,
) {
  final wb = BytesBuilder();
  final sb = BytesBuilder();
  final out = <Map<String, Object>>[];
  var wCur = 0; // byte cursor into this store's weight image
  var sCur = 0; // byte cursor into this store's resident scale image
  for (final e in entries) {
    final wOff = wCur;
    final slice = srcWeights.sublist(
      e.weightOffset,
      e.weightOffset + e.weightLength,
    );
    wb.add(slice);
    wCur += slice.length;
    while (wCur % 4 != 0) {
      wb.addByte(0);
      wCur++;
    }

    final sOff = sCur;
    // Group-major scale table: groups*rows fp16 (groups == 1 for per-row). Each
    // resident scale is one fp16 padded to a 32-bit word.
    final nScales = e.groups * e.rows;
    for (var i = 0; i < nScales; i++) {
      sb
        ..addByte(srcScales[(e.scaleOffset + i) * 2])
        ..addByte(srcScales[(e.scaleOffset + i) * 2 + 1])
        ..addByte(0)
        ..addByte(0);
      sCur += 4;
    }

    out.add({
      'name': e.name,
      'store': store,
      'weight_offset': wOff,
      'scale_flash_offset': sOff,
      'rows': e.rows,
      'cols': e.cols,
      'col_tiles': e.colTiles,
      'groups': e.groups,
    });
  }
  return (weights: wb.toBytes(), scales: sb.toBytes(), entries: out);
}

/// The shared MoE config for [g], or null if no layer is Mixture-of-Experts.
/// MoE config (expert count, top-k, ...) is uniform across a model's MoE layers.
MoeSpec? _moeSpecOf(ModelGraph g) {
  for (final l in g.layers) {
    final m = l.mlp.moe;
    if (m != null) return m;
  }
  return null;
}

Map<String, Object> _moeManifest(MoeSpec m) => {
  'num_experts': m.numExperts,
  'top_k': m.topK,
  'norm_topk': m.normTopK,
  'moe_intermediate': m.moeIntermediate,
  'num_shared': m.numShared,
};

/// Returns the id of an added/special token by content, or null.
int? _findAddedTokenId(String tokenizerJson, String content) {
  final j = jsonDecode(tokenizerJson) as Map<String, dynamic>;
  final added = (j['added_tokens'] as List?) ?? const [];
  for (final a in added) {
    final m = a as Map<String, dynamic>;
    if (m['content'] == content) return m['id'] as int;
  }
  return null;
}

void _emitModelArtifacts(
  String output,
  LoadedModel loaded, {
  int bramCacheKb = 0,
  int maxCols = 0,
}) {
  final model = loaded.model;
  final ternary = loaded.graph.ternary;
  final img = flashImageFor(model, maxCols: maxCols, ternary: ternary);
  if (img.weights.length > _ecp5UsableFlashBytes) {
    stderr.writeln(
      'WARNING: weights ${img.weights.length}B exceed '
      '~${_ecp5UsableFlashBytes ~/ (1024 * 1024)}MB usable flash on the ecp5 '
      '25f. It will not fit config flash, target a larger store (DDR) or a '
      'smaller/more-quantized model.',
    );
  }
  final g = loaded.graph;
  final a = g.layers.first.attention;

  File('$output/scales.bin').writeAsBytesSync(img.scales);

  // Resident-scale flash image: one fp16 per 32-bit word (low16, rest zero), in
  // the same order as scales.bin, so the accelerator can read a matrix's per-row
  // scales straight from flash (SCALE_BASE) instead of the host pushing them.
  // (all-flash path only. The tiered split repacks this per store below.)
  final scalesFlash = BytesBuilder();
  for (var i = 0; i + 1 < img.scales.length; i += 2) {
    scalesFlash
      ..addByte(img.scales[i])
      ..addByte(img.scales[i + 1])
      ..addByte(0)
      ..addByte(0);
  }
  if (bramCacheKb <= 0) {
    File('$output/weights.bin').writeAsBytesSync(img.weights);
    File('$output/scales_flash.bin').writeAsBytesSync(scalesFlash.toBytes());
  }

  // Glue weights the HOST uses (not on the device matmul path): the embedding
  // table + per-layer + final RMSNorm gammas, all fp16, packed into glue.bin
  // with fp16-element offsets recorded in loom.json.
  final glue = BytesBuilder();
  var glueCursor = 0; // fp16 elements
  int appendVec(List<double> v) {
    final at = glueCursor;
    for (final x in v) {
      final h = Half.fromDouble(x);
      glue.addByte(h & 0xFF);
      glue.addByte((h >> 8) & 0xFF);
    }
    glueCursor += v.length;
    return at;
  }

  final embedOffset = appendVec(model.embedTokens.toFloat64List());
  final layerGlue = [
    for (final bl in model.layers)
      {
        'input_norm': appendVec(bl.inputNorm.toFloat64List()),
        'post_norm': appendVec(bl.postAttnNorm.toFloat64List()),
        // Qwen2 q/k/v biases (fp16 glue, host adds them after the matmul before
        // RoPE). Only emitted when the arch has them, so llama manifests are
        // unchanged.
        if (bl.qBias != null) 'q_bias': appendVec(bl.qBias!.toFloat64List()),
        if (bl.kBias != null) 'k_bias': appendVec(bl.kBias!.toFloat64List()),
        if (bl.vBias != null) 'v_bias': appendVec(bl.vBias!.toFloat64List()),
        // MoE router (fp16 glue): a [num_experts x hidden] linear the host runs
        // to pick top-k experts. Its presence marks this layer as MoE.
        if (bl.moe != null) 'router': appendVec(bl.moe!.router.toFloat64List()),
      },
  ];
  final finalNormOffset = appendVec(model.finalNorm.toFloat64List());

  // MTP module norms (enorm/hnorm + the block's input/post norms) as fp16 glue.
  // Each module's eh_proj + transformer matrices live in the int4 image (named
  // mtp.$m.*). embed/final-norm/lm_head are shared with the main model.
  final mtpModuleGlue = model.mtp == null
      ? null
      : [
          for (final mod in model.mtp!.modules)
            {
              'enorm': appendVec(mod.enorm.toFloat64List()),
              'hnorm': appendVec(mod.hnorm.toFloat64List()),
              'input_norm': appendVec(mod.block.inputNorm.toFloat64List()),
              'post_norm': appendVec(mod.block.postAttnNorm.toFloat64List()),
            },
        ];
  // Vision tower glue: per-block LayerNorm gammas/betas + attention/MLP biases,
  // plus patch-embed bias, class token, position embeddings, and the pre/post
  // LayerNorms. The vision matmul matrices live in the int4 image (vision.*).
  Map<String, Object>? visionManifest;
  final vw = model.vision;
  if (vw != null) {
    final layerEntries = <Map<String, Object>>[
      for (final b in vw.blocks)
        {
          'ln1_gamma': appendVec(b.ln1Gamma),
          'ln1_beta': appendVec(b.ln1Beta),
          'q_bias': appendVec(b.qBias),
          'k_bias': appendVec(b.kBias),
          'v_bias': appendVec(b.vBias),
          'o_bias': appendVec(b.oBias),
          'ln2_gamma': appendVec(b.ln2Gamma),
          'ln2_beta': appendVec(b.ln2Beta),
          'fc1_bias': appendVec(b.fc1Bias),
          'fc2_bias': appendVec(b.fc2Bias),
        },
    ];
    visionManifest = {
      'image_size': vw.imageSize,
      'patch_size': vw.patchSize,
      'num_channels': vw.numChannels,
      'hidden': vw.hidden,
      'num_layers': vw.blocks.length,
      'num_heads': vw.numHeads,
      'head_dim': vw.headDim,
      'intermediate': vw.intermediate,
      'layer_norm_eps': vw.lnEps,
      'has_class_token': vw.hasClassToken,
      'seq_len': vw.seqLen,
      // Image normalization for host preprocessing (jpeg -> tensor).
      'image_mean': g.vision!.imageMean,
      'image_std': g.vision!.imageStd,
      'patch_embed_bias': appendVec(vw.patchEmbedBias),
      if (vw.hasClassToken) 'class_token': appendVec(vw.classToken!),
      'pos_embed': appendVec(vw.posEmbed),
      if (vw.preLnGamma != null) 'pre_ln_gamma': appendVec(vw.preLnGamma!),
      if (vw.preLnBeta != null) 'pre_ln_beta': appendVec(vw.preLnBeta!),
      'post_ln_gamma': appendVec(vw.postLnGamma),
      'post_ln_beta': appendVec(vw.postLnBeta),
      'layers': layerEntries,
    };
  }

  Map<String, Object>? projectorManifest;
  final pw = model.projector;
  if (pw != null) {
    projectorManifest = {
      'num_layers': pw.isTwoLayer ? 2 : 1,
      'input_dim': pw.inputDim,
      'hidden_dim': pw.hiddenDim,
      'output_dim': pw.outputDim,
      // Idefics3 pixel-shuffle merge factor (1 = LLaVA, no shuffle).
      'scale_factor': pw.scaleFactor,
      if (pw.bias1 != null) 'bias1': appendVec(pw.bias1!),
      if (pw.isTwoLayer && pw.bias2 != null) 'bias2': appendVec(pw.bias2!),
    };
  }

  File('$output/glue.bin').writeAsBytesSync(glue.toBytes());

  // Weight-store split. bramCacheKb<=0 keeps EVERY matrix in flash (unchanged).
  // N>0 sorts matrices by weight-byte size DESCENDING and greedily fills the
  // N-KiB BRAM budget with the biggest (each weight is read once/token, so
  // flash-time saved is proportional to bytes cached -> biggest first). Each
  // store gets its own repacked images (offsets from 0). Matrices are tagged
  // store=bram|flash + their offset in that store so the runtime points
  // WEIGHT_BASE/SCALE_BASE at the right absolute address.
  int weightBudgetBytes(LinearWeightEntry e) => ((e.weightLength + 3) ~/ 4) * 4;
  int scaleBudgetBytes(LinearWeightEntry e) =>
      e.groups * e.rows * 4; // resident: 4B/scale, groups*rows for per-group

  final hotNames = <String>{};
  if (bramCacheKb > 0) {
    final budget = bramCacheKb * 1024;
    final bySize = [...img.manifest]
      ..sort((x, y) => weightBudgetBytes(y).compareTo(weightBudgetBytes(x)));
    var used = 0;
    for (final e in bySize) {
      final need = weightBudgetBytes(e) + scaleBudgetBytes(e);
      if (used + need <= budget) {
        hotNames.add(e.name);
        used += need;
      }
    }
  }
  // Partition in the model's (sequencer) order.
  final hot = [
    for (final e in img.manifest)
      if (hotNames.contains(e.name)) e,
  ];
  final cold = [
    for (final e in img.manifest)
      if (!hotNames.contains(e.name)) e,
  ];

  final bramStore = _packStore(hot, img.weights, img.scales, 'bram');
  final flashStore = _packStore(cold, img.weights, img.scales, 'flash');

  final matrices = <Map<String, Object>>[];
  if (bramCacheKb > 0) {
    // Tiered: cold weights keep the existing filename/flash base. Hot weights go
    // into the BRAM images. Emit both stores' images.
    File('$output/weights.bin').writeAsBytesSync(flashStore.weights);
    File('$output/scales_flash.bin').writeAsBytesSync(flashStore.scales);
    File('$output/weights_bram.bin').writeAsBytesSync(bramStore.weights);
    File('$output/scales_bram.bin').writeAsBytesSync(bramStore.scales);
    // Re-emit in model order so the runtime consumes matrices in sequence.
    final byName = {
      for (final m in [...bramStore.entries, ...flashStore.entries])
        m['name'] as String: m,
    };
    for (final e in img.manifest) {
      matrices.add(byName[e.name]!);
    }
  } else {
    // All-flash: emit the single-store manifest shape (no store split).
    for (final e in img.manifest) {
      matrices.add({
        'name': e.name,
        'store': 'flash',
        'weight_offset': e.weightOffset,
        'scale_offset': e.scaleOffset,
        'scale_flash_offset': e.scaleOffset * 4,
        'rows': e.rows,
        'cols': e.cols,
        'col_tiles': e.colTiles,
        'groups': e.groups,
      });
    }
  }

  // Place the resident scale image ABOVE the full flash weight image. A fixed
  // gap only worked for tiny models (stories260K). A real multi-MB weight image
  // overruns it, so the scales land inside the weights. 64KB-aligned so the
  // flash write offset stays clean. Assert weights+scales fit the 16MB flash.
  final flashWeightBytes = bramCacheKb > 0
      ? flashStore.weights.length
      : img.weights.length;
  final flashScaleBytes = bramCacheKb > 0
      ? flashStore.scales.length
      : img.scales.length * 2;
  final flashScaleOffset =
      _flashWeightOffset + ((flashWeightBytes + 0xFFFF) & ~0xFFFF);
  if (flashScaleOffset + flashScaleBytes > _configFlashBytes) {
    throw StateError(
      'flash weights (${flashWeightBytes}B @ 0x${_flashWeightOffset.toRadixString(16)}) '
      '+ scales (${flashScaleBytes}B @ 0x${flashScaleOffset.toRadixString(16)}) '
      'overrun the 16MB config flash by '
      '${flashScaleOffset + flashScaleBytes - _configFlashBytes}B',
    );
  }

  // snake_case keys so the Zig runtime's config struct fields match directly.
  final manifest = {
    'name': g.name,
    'hidden': g.hiddenSize,
    'vocab': g.vocabSize,
    'layers': g.layers.length,
    'num_heads': a.numHeads,
    'num_kv_heads': a.numKvHeads,
    'head_dim': a.headDim,
    'intermediate': g.layers.first.mlp.intermediateSize,
    // Mixture-of-Experts config (shared across all MoE layers). Present only for
    // MoE models. A layer is MoE iff its layer_glue carries a 'router' offset.
    // Experts are matrices named layers.$i.experts.$e.{gate,up,down}_proj.
    if (_moeSpecOf(g) != null) 'moe': _moeManifest(_moeSpecOf(g)!),
    // Multi-Token Prediction heads. num_modules + per-module glue offsets. The
    // eh_proj/block matrices are in `matrices` named mtp.$m.*.
    if (mtpModuleGlue != null)
      'mtp': {'num_modules': mtpModuleGlue.length, 'modules': mtpModuleGlue},
    // Vision-language: the ViT tower + projector + image placeholder token. The
    // vision/projector matmul matrices are in `matrices` (vision.*/projector.*).
    if (visionManifest != null) 'vision': visionManifest,
    if (projectorManifest != null) 'projector': projectorManifest,
    if (g.imageTokenIndex != null) 'image_token_index': g.imageTokenIndex,
    'max_seq': loaded.maxSeq,
    // Accelerator column capacity. Matrices wider than this are stored col-block
    // contiguous and the runtime col-tiles them (0 = no tiling, single block).
    'max_cols': maxCols,
    'rope_theta': a.ropeTheta,
    'norm_eps': g.layers.first.normEps,
    'tie_embeddings': g.tieEmbeddings,
    // Provenance: weights are int4-packed BitNet ternary (per-tensor scale). The
    // int4 runtime/sim path runs them as-is. Recorded for the future 2-bit path.
    if (g.ternary) 'quant': 'bitnet_ternary',
    'csr_base': _streamCsrBase,
    'flash_weight_base': _streamDdrBase + _flashWeightOffset,
    'scale_flash_base': _streamDdrBase + flashScaleOffset,
    // Tiered BRAM store: hot weights at [_bramWeightBase], their resident scales
    // packed immediately after (both in the one on-chip HarborSram).
    'bram_cache_kb': bramCacheKb,
    'bram_weight_base': _bramWeightBase,
    'bram_scale_base': _bramWeightBase + bramStore.weights.length,
    'embed_offset': embedOffset,
    'final_norm_offset': finalNormOffset,
    'layer_glue': layerGlue,
    'matrices': matrices,
  };
  File(
    '$output/loom.json',
  ).writeAsStringSync(const JsonEncoder.withIndent('  ').convert(manifest));

  if (bramCacheKb > 0) {
    final bramBytes = bramStore.weights.length + bramStore.scales.length;
    final flashBytes = flashStore.weights.length + flashStore.scales.length;
    stdout.writeln(
      'Weight split (BRAM cache ${bramCacheKb}KB): '
      '${hot.length} hot / ${cold.length} cold matrices',
    );
    stdout.writeln(
      '  BRAM : weights ${bramStore.weights.length}B + scales '
      '${bramStore.scales.length}B = ${(bramBytes / 1024).toStringAsFixed(1)}KB '
      'of ${bramCacheKb}KB budget',
    );
    stdout.writeln(
      '  flash: weights ${flashStore.weights.length}B + scales '
      '${flashStore.scales.length}B = ${(flashBytes / 1024).toStringAsFixed(1)}KB',
    );
    for (final e in hot) {
      stdout.writeln('    [bram]  ${e.name}  ${weightBudgetBytes(e)}B');
    }
    for (final e in cold) {
      stdout.writeln('    [flash] ${e.name}  ${weightBudgetBytes(e)}B');
    }
    stdout.writeln(
      'Emitted weights_bram.bin + scales_bram.bin + weights.bin + '
      'scales_flash.bin + loom.json for ${g.name}',
    );
  } else {
    stdout.writeln(
      'Emitted weights.bin (${img.weights.length}B) + scales.bin + '
      'loom.json (${img.manifest.length} matrices) for ${g.name}',
    );
  }
}

/// OrangeCrab r0.2 board-top SystemVerilog shim for the UART SoC
/// (`LoomUartTop`).
///
/// Wraps the generated `LoomUartSoC`: real OrangeCrab pinout (UART TX=N17,
/// RX=M18 - the PROVEN crossover orientation, clk48=A9, rst_n=V17), an
/// active-low reset invert with a power-on
/// hold, and a free-running LED blink for liveness. The UART pins are plain
/// LVCMOS33 in/out (no tristate), so unlike the USB top there are no inout pads.
const String _uartBoardTopSv = '''
// OrangeCrab r0.2 board-top shim for the Loom UART SoC.
//
// HAND-WRITTEN board shim (not generated). Wraps the generated LoomUartSoC:
// real OrangeCrab UART pinout (PROVEN crossover: TX=N17, RX=M18), active-low
// reset invert with a power-on hold, and a free-running LED blink for liveness.
//
// Target: LFE5U-25F-8MG285C (CSFBGA285), all LVCMOS33.
module LoomUartTop (
    input  logic clk48,    // A9  - 48 MHz oscillator
    input  logic rst_n,    // V17 - reset button, ACTIVE-LOW
    input  logic uart_rx,  // M18 - serial in  (from DirtyJTAG TX)
    output logic uart_tx,  // N17 - serial out (to   DirtyJTAG RX)
    output logic led       // M3 - RGB green channel, ACTIVE-LOW (lit = 0)
);
  // Hold the SoC in reset for the first 2^16 clocks after config, then release
  // unless the user holds the active-low button (rst_n == 0 => reset asserted).
  logic [15:0] por_cnt = 16'd0;
  logic        por_done = 1'b0;
  always_ff @(posedge clk48) begin
    if (!por_done) begin
      por_cnt  <= por_cnt + 16'd1;
      por_done <= &por_cnt;  // done when counter saturates
    end
  end
  // SoC reset is ACTIVE-HIGH: assert during POR or while the button is pressed.
  logic soc_reset;
  assign soc_reset = (!por_done) | (~rst_n);

  logic [23:0] blink = 24'd0;
  always_ff @(posedge clk48) blink <= blink + 24'd1;
  // RGB LED is active-low (common anode): drive 0 to light. Toggle the MSB.
  assign led = ~blink[23];

  LoomUartSoC u_soc (
      .clk     (clk48),
      .reset   (soc_reset),
      .uart_rx (uart_rx),
      .uart_tx (uart_tx)
  );
endmodule
''';

/// OrangeCrab r0.2 LPF for [_uartBoardTopSv]. Constrains every `LoomUartTop`
/// port to its real board pin (all LVCMOS33), with the 48 MHz timing
/// constraint on the oscillator. This is the constraint file the flashable
/// UART build uses.
const String _uartBoardTopLpf = '''
# OrangeCrab r0.2 (LFE5U-25F-8MG285C, CSFBGA285) - LoomUartTop board-shim LPF.
# All LVCMOS33. Authoritative OrangeCrab r0.2 pinout.

# 48 MHz oscillator -> clk48
LOCATE COMP "clk48" SITE "A9";
IOBUF PORT "clk48" IO_TYPE=LVCMOS33;
FREQUENCY PORT "clk48" 48.0 MHz;

# Reset button (active-low)
LOCATE COMP "rst_n" SITE "V17";
IOBUF PORT "rst_n" IO_TYPE=LVCMOS33;

# UART on the feather GPIO header (PROVEN crossover orientation: TX=N17, RX=M18)
LOCATE COMP "uart_tx" SITE "N17";
IOBUF PORT "uart_tx" IO_TYPE=LVCMOS33;
LOCATE COMP "uart_rx" SITE "M18";
IOBUF PORT "uart_rx" IO_TYPE=LVCMOS33;

# RGB LED green channel (active-low)
LOCATE COMP "led" SITE "M3";
IOBUF PORT "led" IO_TYPE=LVCMOS33;
''';

/// Composes the streaming, memory-backed-matmul SoC as a real [HarborSoC]:
/// a [LoomUartBridge] host master + a [LoomStreamAccelerator] (DUAL ROLE: CSR
/// slave peripheral AND weight-read master) + a [HarborSram] scratchpad, wired
/// by [HarborSoC.buildFabric] (which auto-inserts a WishboneArbiter for the two
/// masters). Fully Harbor-generated and target-parameterized (no hand-written
/// SV, no hardcoded vendor): pass an ECP5/Spartan-7/Sky130 [target].
///
/// A primary 48 MHz clock domain makes FPGA targets use Harbor's internal
/// power-on reset, so the generated SoC is flashable from just clk + UART pins
/// (no external reset, no board shim).
/// Parses a "vendor:device:package" target spec into a HarborFpgaTarget with
/// the given pin map (River-style declarative targeting. No hardcoded device).
HarborFpgaTarget _parseFpgaTarget(String spec, Map<String, String> pins) {
  final parts = spec.split(':');
  if (parts.length != 3) {
    throw FormatException(
      'Target format is vendor:device:package (e.g. ecp5:25f:CSFBGA285), '
      'got: $spec',
    );
  }
  final (vendor, device, package) = (parts[0], parts[1], parts[2]);
  return switch (vendor) {
    'ecp5' => HarborFpgaTarget.ecp5(
      device: device,
      package: package,
      frequency: 48000000,
      pinMap: pins,
    ),
    'ice40' => HarborFpgaTarget.ice40(
      device: device,
      package: package,
      frequency: 48000000,
      pinMap: pins,
    ),
    _ => throw UnsupportedError('Unknown FPGA vendor: $vendor'),
  };
}

/// Resolves the build target from `--board` (Harbor's [HarborBoard] catalog) or
/// the `--target` escape hatch, constraining `exposedPins` (the board-catalog
/// signals THIS SoC drives) plus any `--pin name=site` overrides. The returned
/// target's `frequency` is the board oscillator, so callers read the source clock
/// off `target.frequency` (no second hardcoded osc constant).
HarborFpgaTarget _resolveTarget({
  required String? board,
  required String? targetSpec,
  required List<String> pinSpecs,
  required Iterable<String> exposedPins,
  Map<String, String> extraPinMap = const {},
}) {
  final extraPins = <String, String>{...extraPinMap};
  for (final spec in pinSpecs) {
    final eq = spec.indexOf('=');
    if (eq < 0) throw FormatException('--pin format is name=site, got: $spec');
    extraPins[spec.substring(0, eq)] = '${spec.substring(eq + 1)} LVCMOS33';
  }
  // The --target escape hatch (board not in the catalog) wins when set: its pins
  // come entirely from --pin (defaulting to the OrangeCrab UART pinout).
  if (targetSpec != null) {
    if (extraPins.isEmpty) extraPins.addAll(_orangeCrabUartPinMap);
    return _parseFpgaTarget(targetSpec, extraPins);
  }
  final b = HarborBoard.get(board!);
  // Constrain only the exposed signals the board actually catalogs. A signal the
  // board lacks (e.g. sdram_* on a board whose DDR sites are not entered yet) is
  // supplied via --pin instead of failing the whole build.
  final selected = exposedPins.where(b.pins.containsKey);
  return b.fpgaTarget(pins: selected, extraPins: extraPins);
}

/// SoC memory map (32-bit).
const _streamSramBase = 0x00000000; // on-chip scratchpad (acts/results)
const _streamCsrBase = 0x00010000; // accelerator CSRs
const _streamDdrBase = 0x20000000; // DDR3 / flash weight store
const _bramWeightBase =
    0x30000000; // tiered BRAM hot-weight store (fp-bram-cache)

// W25Q128 config flash is 16MB, the bitstream lives at 0x200000, leaving ~14MB
// for weights on the OrangeCrab ecp5 25f. A bigger store (DDR) may be targeted,
// so overflow is a loud WARNING, not an error.
const _ecp5UsableFlashBytes = 14 * 1024 * 1024;

/// Host transport that drives the SoC's Wishbone fabric as the bus master.
enum LoomTransport { uart, usb }

/// A Loom datapath: the accelerator plus its on-chip memories / weight store.
/// [buildLoomSoc] switches on the concrete subtype to attach the right
/// peripherals, so the SoC is ONE board/target-driven builder with the datapath
/// as a plug-in (River-style) instead of a per-variant function.
sealed class LoomDatapath {
  const LoomDatapath();
}

/// The original 8x8 overlay accelerator ([LoomAccelerator]) as a bus SLAVE, with
/// no on-chip memories - a 12-bit bus run directly off the board oscillator.
final class OverlayDatapath extends LoomDatapath {
  final String modelName;
  const OverlayDatapath({this.modelName = 'SmolLM2-135M'});
}

/// The int8 streaming matmul accelerator + on-chip SRAM scratchpad, with an
/// optional DDR3 weight store.
final class StreamDatapath extends LoomDatapath {
  final int sramSize;

  /// The DDR board that supplies the part configuration and the pad sites, or
  /// null to leave the weights out of DDR. See [LoomDdrBoard].
  final LoomDdrBoard? ddr;

  /// The static DDR3 read tap (`--ddr-read-tap`).
  final int ddrReadTap;

  bool get weightsInDdr => ddr != null;

  const StreamDatapath({this.sramSize = 16384, this.ddr, this.ddrReadTap = 40});
}

/// The fp16 W4A8 linear accelerator + SRAM scratchpad, with an optional on-chip
/// BRAM hot-weight cache, a DDR3 store, and/or a resident SPI-flash weight store.
final class FpDatapath extends LoomDatapath {
  final int maxColTiles;
  final int maxRowBlocks;
  final int sramSize;

  /// The DDR board that supplies the part configuration and the pad sites, or
  /// null to leave the weights out of DDR. See [LoomDdrBoard].
  final LoomDdrBoard? ddr;

  /// The static DDR3 read tap (`--ddr-read-tap`).
  final int ddrReadTap;
  final bool weightsInFlash;
  final int flashReadAhead;
  final int bramCacheKb;
  final bool ternary;

  bool get weightsInDdr => ddr != null;

  const FpDatapath({
    this.maxColTiles = 32,
    this.maxRowBlocks = 32,
    this.sramSize = 16384,
    this.ddr,
    this.ddrReadTap = 40,
    this.weightsInFlash = false,
    this.flashReadAhead = 8,
    this.bramCacheKb = 0,
    this.ternary = false,
  });
}

/// Attaches [board]'s DDR3 as the weight store at [_streamDdrBase] and exposes
/// its pads. The part configuration comes from the board entry, so a retarget
/// changes the geometry with it. [target] selects the PHY: Harbor builds the
/// ECP5 PHY for a Lattice target and the Xilinx PHY for a 7-series target.
void _attachDdr(
  HarborSoC soc,
  LoomDdrBoard board, {
  required int sysHz,
  required int readTap,
  required HarborDeviceTarget? target,
}) {
  final ddr = HarborDdrController(
    config: board.config,
    baseAddress: _streamDdrBase,
    clockHz: sysHz,
    busAddressWidth: 32,
    readTaps: readTap,
    target: target,
  );
  soc.addPeripheral(ddr);
  // The controller builds its pad set from the PHY and the clock: the ECP5 PHY
  // above ~60 MHz makes a true-differential DQS pad and creates no
  // `sdram_dqs_n` port. Expose the pads that this controller has.
  for (final pad in LoomDdrBoard.padPorts) {
    if (ddr.tryInput(pad) == null &&
        ddr.tryOutput(pad) == null &&
        ddr.tryInOut(pad) == null) {
      continue;
    }
    soc.exposePin(ddr, pad, externalName: pad);
  }
}

/// Attaches [datapath]'s accelerator + memories/weight-store to [soc]. The stream
/// and fp accelerators are also weight-read MASTERs ('mem'). The overlay one is a
/// plain slave. Weight-store pads are exposed here so the ordering matches the
/// historical per-variant builders (DDR/flash before the transport pins).
void _attachDatapath(
  HarborSoC soc,
  LoomDatapath datapath, {
  required int sysHz,
  required HarborDeviceTarget? target,
}) {
  switch (datapath) {
    case OverlayDatapath(:final modelName):
      final accel = LoomAccelerator(
        config: LoomAcceleratorConfig(
          baseAddress: _acceleratorBase,
          modelName: modelName,
        ),
      );
      soc.addPeripheral(accel);
    case StreamDatapath(:final sramSize, :final ddr, :final ddrReadTap):
      final accel = LoomStreamAccelerator(baseAddress: _streamCsrBase);
      final sram = HarborSram(
        baseAddress: _streamSramBase,
        size: sramSize,
        busAddressWidth: 32,
        target: target,
      );
      soc.addMaster(accel, busInterfaceName: 'mem');
      soc.addPeripheral(accel);
      soc.addPeripheral(sram);
      // Weight store: the board's DDR3, through Harbor's controller. The part
      // configuration comes from the board entry, so the geometry follows the
      // board and not one hardcoded part. CK = system clock (1:1).
      if (ddr != null) {
        _attachDdr(soc, ddr, sysHz: sysHz, readTap: ddrReadTap, target: target);
      }
    case FpDatapath(
      :final maxColTiles,
      :final maxRowBlocks,
      :final sramSize,
      :final ddr,
      :final ddrReadTap,
      :final weightsInFlash,
      :final flashReadAhead,
      :final bramCacheKb,
      :final ternary,
    ):
      final accel = LoomFpLinearAccelerator(
        baseAddress: _streamCsrBase,
        maxColTiles: maxColTiles,
        maxRowBlocks: maxRowBlocks,
        addressWidth: 32,
        ternaryWeights: ternary,
      );
      final sram = HarborSram(
        baseAddress: _streamSramBase,
        size: sramSize,
        busAddressWidth: 32,
        target: target,
      );
      soc.addMaster(accel, busInterfaceName: 'mem');
      soc.addPeripheral(accel);
      soc.addPeripheral(sram);
      // Tiered weight store: an on-chip BRAM cache holding the HOT (biggest)
      // weight matrices at [_bramWeightBase]. The host writes it at startup, the
      // accelerator reads it. Cold matrices stay resident in SPI flash.
      if (bramCacheKb > 0) {
        soc.addPeripheral(
          HarborSram(
            baseAddress: _bramWeightBase,
            size: bramCacheKb * 1024,
            busAddressWidth: 32,
            target: target,
            name: 'weightbram',
          ),
        );
      }
      if (ddr != null) {
        _attachDdr(soc, ddr, sysHz: sysHz, readTap: ddrReadTap, target: target);
      }
      // Resident weight store in the config SPI flash: the accelerator reads int4
      // weights straight from flash (memory-mapped at [_streamDdrBase]), so the
      // host provisions nothing. Reads are slow (bit-serial SPI) but resident.
      if (weightsInFlash) {
        final flash = HarborSpiFlashController(
          config: HarborSpiFlashConfig.w25q128(readAheadWords: flashReadAhead),
          baseAddress: _streamDdrBase,
          busAddressWidth: 32,
          busDataWidth: 32,
          useUsrmclk: true,
        );
        soc.addPeripheral(flash);
        for (final pad in _flashPads) {
          soc.exposePin(flash, pad, externalName: pad);
        }
      }
  }
}

/// Assembles a Loom SoC: a [transport] host master + a [datapath] (accelerator +
/// memories) on a Harbor Wishbone fabric, targeting [target]. ONE builder for
/// every variant - the transport and datapath are plug-ins and the board / PLL /
/// pins / constraints all come from [target] + [oscHz], so retargeting or
/// swapping a datapath is a parameter change, not a new function. [fabricHz] is
/// the requested system clock: the stream/fp datapaths derive it from [oscHz] via
/// a PLL, the overlay runs directly off the oscillator. A null [target] keeps the
/// legacy hardcoded OrangeCrab pinout for the overlay bring-up SoCs.
HarborSoC buildLoomSoc({
  required String name,
  required LoomTransport transport,
  required LoomDatapath datapath,
  HarborDeviceTarget? target,
  int oscHz = 48000000,
  int fabricHz = 48000000,
  int baudRate = 115200,
}) {
  // The overlay datapath keeps its historical 12-bit bus + direct-oscillator
  // clock. The memory-backed datapaths use a 32-bit bus + a PLL-derived clock.
  final overlay = datapath is OverlayDatapath;
  final busWidth = overlay ? _busAddressWidth : 32;
  final cfg = WishboneConfig(addressWidth: busWidth, dataWidth: 32);

  // Clock policy per datapath.
  final List<HarborClockConfig> clocks;
  var sysHz = fabricHz;
  switch (datapath) {
    case OverlayDatapath():
      clocks = const []; // direct oscillator, no PLL
      sysHz = oscHz;
    case StreamDatapath(:final weightsInDdr):
      // DDR3 is qualified 1:1 at the oscillator: the PHY makes its own capture
      // PLL, so the system clock stays the primary direct oscillator. Without DDR
      // a slower PLL-derived clock is allowed.
      if (weightsInDdr) sysHz = oscHz;
      final primary = sysHz == oscHz;
      clocks = [
        HarborClockConfig.fixed(
          name: 'sys',
          frequency: sysHz,
          sourceFrequency: primary ? null : oscHz,
          isPrimary: primary,
        ),
      ];
    case FpDatapath():
      // The fp16 combinational cores cap ~22MHz, so run a PLL-derived slow clock.
      clocks = [
        HarborClockConfig.fixed(
          name: 'sys',
          frequency: sysHz,
          sourceFrequency: oscHz,
          isPrimary: false,
        ),
      ];
  }

  // Overlay bring-up SoCs keep their legacy hardcoded OrangeCrab target (the pad
  // map differs by transport) when the caller supplies none. The memory-backed
  // datapaths pass a null target straight through (target-agnostic elaboration).
  final resolvedTarget =
      target ??
      (overlay
          ? HarborFpgaTarget.ecp5(
              device: '25f',
              package: 'CSFBGA285',
              frequency: oscHz,
              pinMap: transport == LoomTransport.usb
                  ? _orangeCrabPinMap
                  : _orangeCrabUartPinMap,
            )
          : null);

  final soc = HarborSoC(
    name: name,
    compatible: 'midstall,loom',
    busConfig: cfg,
    target: resolvedTarget,
    clocks: clocks,
  );

  switch (transport) {
    case LoomTransport.uart:
      final bridge = LoomUartBridge(
        config: LoomUartConfig(
          busAddressWidth: busWidth,
          busDataWidth: 32,
          clockFrequency: sysHz, // baud divisor derived from the system clock
          baudRate: baudRate,
        ),
      );
      soc.addMaster(bridge, busInterfaceName: 'bus');
      _attachDatapath(soc, datapath, sysHz: sysHz, target: resolvedTarget);
      soc.exposePin(bridge, 'rx', externalName: 'uart_rx');
      soc.exposePin(bridge, 'tx', externalName: 'uart_tx');
    case LoomTransport.usb:
      // The custom vendor-class device (class 0xFF, 0x1209:0x10C0, two bulk
      // endpoints): its EP1 bulk command engine is a Wishbone MASTER. tbCmdPorts
      // false removes the testbench byte-stream ports so the only path in is the
      // real EP1 bulk endpoints through the dp/dm PHY.
      final device = LoomUsbDevice(
        config: LoomUsbDeviceConfig(
          busAddressWidth: busWidth,
          busDataWidth: 32,
          idVendor: 0x1209,
          idProduct: 0x10C0,
        ),
        tbCmdPorts: false,
      );
      soc.addMaster(device, busInterfaceName: 'bus');
      _attachDatapath(soc, datapath, sysHz: sysHz, target: resolvedTarget);
      // Port name -> external pad name. LoomUsbDevice exposes the SPLIT
      // dp(in)/dp_out/oe shape the board-top folds back onto the real
      // bidirectional balls. The pullup port is already usb-prefixed.
      const usbPins = {
        'dp': 'usb_dp',
        'dm': 'usb_dm',
        'dp_out': 'usb_dp_out',
        'dm_out': 'usb_dm_out',
        'oe': 'usb_oe',
        'usb_pullup': 'usb_pullup',
      };
      usbPins.forEach(
        (port, external) => soc.exposePin(device, port, externalName: external),
      );
  }

  soc.buildFabric();
  return soc;
}

// uartprobe transport: a TX-ONLY continuous serial probe.
//
// This is a LINK-DIAGNOSTIC build, NOT a command bridge. It instantiates ONLY
// the [LoomUart] TX engine (RX tied idle-high, unused) and continuously streams
// the ASCII char 'U' (0x55) out the OrangeCrab UART TX pin (M18) at 115200 8N1,
// forever, with a small inter-byte gap. 0x55 is 0b01010101 (alternating bits),
// so any baud-rate error shows up as a recognizable mangling rather than a
// plausible-looking byte. This isolates the device->host return path (FPGA M18
// -> DirtyJTAG RX -> CDC -> /dev/ttyACM0) independent of whether the device can
// RECEIVE commands.
//
// The pacing logic lives in a small hand-written board-top SV shim
// ([_uartProbeTopSv]), exactly mirroring the existing LoomUartTop pattern: a
// free-running counter raises tx_valid for one cycle each time the TX engine is
// idle (tx_ready high) and a gap timer has elapsed, so a fresh 'U' frame goes
// out continuously. clk48=A9, uart_tx=M18, rst_n=V17, led=M3.

/// Generates just the [LoomUart] SystemVerilog (the TX/RX line engine) for the
/// probe build. The probe top hand-instantiates this `loom_uart` module.
Future<void> _generateUartProbeRtl(Directory outDir) async {
  final uart = LoomUart(
    config: const LoomUartConfig(
      busAddressWidth: 12,
      busDataWidth: 32,
      clockFrequency: 48000000,
      baudRate: 115200,
    ),
    name: 'loom_uart',
  );
  await uart.build();

  final rtlDir = Directory('${outDir.path}/rtl')..createSync(recursive: true);
  final filelist = StringBuffer();
  final synthBuilder = SynthBuilder(uart, SystemVerilogSynthesizer());
  for (final fileContents in synthBuilder.getSynthFileContents()) {
    final fileName = '${fileContents.name}.sv';
    File('${rtlDir.path}/$fileName').writeAsStringSync(fileContents.contents);
    filelist.writeln('./rtl/$fileName');
  }
  File('${outDir.path}/filelist.f').writeAsStringSync(filelist.toString());
}

/// Hand-written OrangeCrab r0.2 board-top for the TX-only UART probe
/// (`LoomUartProbeTop`). Instantiates the generated `LoomUart` (definition name
/// emitted by ROHD) and continuously feeds it 0x55 ('U').
const String _uartProbeTopSv = '''
// OrangeCrab r0.2 board-top for the Loom TX-ONLY UART probe.
//
// LINK DIAGNOSTIC: continuously transmits 0x55 ('U') on uart_tx (M18) at
// 115200 8N1, forever, regardless of RX. Isolates the device->host return path.
//
// Target: LFE5U-25F-8MG285C (CSFBGA285), all LVCMOS33.
module LoomUartProbeTop (
    input  logic clk48,    // A9  - 48 MHz oscillator
    input  logic rst_n,    // V17 - reset button, ACTIVE-LOW
    output logic uart_tx,  // M18 - serial out (to DirtyJTAG RX)
    output logic led       // M3  - RGB green channel, ACTIVE-LOW (lit = 0)
);
  logic [15:0] por_cnt = 16'd0;
  logic        por_done = 1'b0;
  always_ff @(posedge clk48) begin
    if (!por_done) begin
      por_cnt  <= por_cnt + 16'd1;
      por_done <= &por_cnt;
    end
  end
  logic soc_reset;
  assign soc_reset = (!por_done) | (~rst_n);

  logic [23:0] blink = 24'd0;
  always_ff @(posedge clk48) blink <= blink + 24'd1;
  assign led = ~blink[23];

  logic       tx_ready;
  logic       tx_accept;
  logic       tx_valid;
  logic [7:0] tx_data;
  logic [7:0] rx_data_unused;
  logic       rx_valid_unused;

  assign tx_data = 8'h55;  // 'U' = 0b01010101

  // Gap timer: after a byte is accepted, wait ~1024 clocks (well under a bit
  // period * 10) before offering the next, giving a small, clean inter-byte
  // gap. Offer a byte whenever idle and the gap has elapsed.
  logic [9:0] gap = 10'd0;
  always_ff @(posedge clk48) begin
    if (soc_reset) begin
      gap <= 10'd0;
    end else begin
      if (tx_accept) gap <= 10'd0;        // restart gap after each accepted byte
      else if (gap != 10'd1023) gap <= gap + 10'd1;
    end
  end
  // Offer a new 'U' once the engine is idle (tx_ready) and the gap has elapsed.
  assign tx_valid = tx_ready & (gap == 10'd1023);

  LoomUart u_uart (
      .clk       (clk48),
      .reset     (soc_reset),
      .rx        (1'b1),          // idle-high, unused
      .tx        (uart_tx),
      .tx_data   (tx_data),
      .tx_valid  (tx_valid),
      .tx_ready  (tx_ready),
      .tx_accept (tx_accept),
      .rx_data   (rx_data_unused),
      .rx_valid  (rx_valid_unused)
  );
endmodule
''';

/// OrangeCrab r0.2 LPF for [_uartProbeTopSv].
const String _uartProbeTopLpf = '''
# OrangeCrab r0.2 (LFE5U-25F-8MG285C, CSFBGA285) - LoomUartProbeTop LPF.
# All LVCMOS33. Authoritative OrangeCrab r0.2 pinout.

# 48 MHz oscillator -> clk48
LOCATE COMP "clk48" SITE "A9";
IOBUF PORT "clk48" IO_TYPE=LVCMOS33;
FREQUENCY PORT "clk48" 48.0 MHz;

# Reset button (active-low)
LOCATE COMP "rst_n" SITE "V17";
IOBUF PORT "rst_n" IO_TYPE=LVCMOS33;

# UART TX on the feather GPIO header
LOCATE COMP "uart_tx" SITE "M18";
IOBUF PORT "uart_tx" IO_TYPE=LVCMOS33;

# RGB LED green channel (active-low)
LOCATE COMP "led" SITE "M3";
IOBUF PORT "led" IO_TYPE=LVCMOS33;
''';

// uartecho transport: a full-duplex UART loopback (RX -> TX).
//
// This is a LINK-DIAGNOSTIC build, NOT a command bridge. It instantiates ONE
// full [LoomUart] (both TX and RX engines) and wires its RECEIVE stream
// straight back to its TRANSMIT stream: every byte the OrangeCrab receives on
// uart_rx (N17) it immediately re-sends on uart_tx (M18). This exercises BOTH
// link legs at once (host->device on N17, device->host on M18) plus the
// DirtyJTAG CDC, independent of the command parser. If a host write to
// /dev/ttyACM0 comes back byte-for-byte, the LINK is proven and any
// missing-data symptom is isolated to the command-engine glue, not the wires.
//
// Echo glue: a one-deep RX holding register. rx_valid is a one-cycle pulse, so
// the received byte is latched into rx_hold/rx_hold_full. The byte is offered
// to TX (tx_valid = rx_hold_full & tx_ready) and the holding register is
// cleared on the UART's one-cycle tx_accept pulse. At 115200 with single-byte
// echoes the host can't overrun, but a second byte arriving while TX is busy
// simply overwrites the (already-consumed-or-in-flight) holding reg, which is
// acceptable for a link probe. clk48=A9, uart_rx=N17, uart_tx=M18, rst_n=V17,
// led=M3.

/// Generates just the [LoomUart] SystemVerilog (the TX/RX line engine) for the
/// echo build. The echo top hand-instantiates this `loom_uart` module. Shares
/// the exact same RTL emission as the probe build.
Future<void> _generateUartEchoRtl(Directory outDir) async {
  final uart = LoomUart(
    config: const LoomUartConfig(
      busAddressWidth: 12,
      busDataWidth: 32,
      clockFrequency: 48000000,
      baudRate: 115200,
    ),
    name: 'loom_uart',
  );
  await uart.build();

  final rtlDir = Directory('${outDir.path}/rtl')..createSync(recursive: true);
  final filelist = StringBuffer();
  final synthBuilder = SynthBuilder(uart, SystemVerilogSynthesizer());
  for (final fileContents in synthBuilder.getSynthFileContents()) {
    final fileName = '${fileContents.name}.sv';
    File('${rtlDir.path}/$fileName').writeAsStringSync(fileContents.contents);
    filelist.writeln('./rtl/$fileName');
  }
  File('${outDir.path}/filelist.f').writeAsStringSync(filelist.toString());
}

/// Hand-written OrangeCrab r0.2 board-top for the UART ECHO loopback
/// (`LoomUartEchoTop`). Instantiates the generated `LoomUart` (both TX and RX)
/// and feeds every received byte straight back out the transmitter.
const String _uartEchoTopSv = '''
// OrangeCrab r0.2 board-top for the Loom full-duplex UART ECHO loopback.
//
// LINK DIAGNOSTIC: every byte received on uart_rx (N17) is re-transmitted on
// uart_tx (M18) at 115200 8N1. Exercises BOTH link legs + the DirtyJTAG CDC at
// once, independent of the command parser. A byte-for-byte round trip proves
// the link is real.
//
// Target: LFE5U-25F-8MG285C (CSFBGA285), all LVCMOS33.
module LoomUartEchoTop (
    input  logic clk48,    // A9  - 48 MHz oscillator
    input  logic rst_n,    // V17 - reset button, ACTIVE-LOW
    input  logic uart_rx,  // N17 - serial in  (from DirtyJTAG TX)
    output logic uart_tx,  // M18 - serial out (to   DirtyJTAG RX)
    output logic led       // M3  - RGB green channel, ACTIVE-LOW (lit = 0)
);
  logic [15:0] por_cnt = 16'd0;
  logic        por_done = 1'b0;
  always_ff @(posedge clk48) begin
    if (!por_done) begin
      por_cnt  <= por_cnt + 16'd1;
      por_done <= &por_cnt;
    end
  end
  logic soc_reset;
  assign soc_reset = (!por_done) | (~rst_n);

  logic [23:0] blink = 24'd0;
  always_ff @(posedge clk48) blink <= blink + 24'd1;
  assign led = ~blink[23];

  logic       tx_ready;
  logic       tx_accept;
  logic       tx_valid;
  logic [7:0] tx_data;
  logic [7:0] rx_data;
  logic       rx_valid;

  // rx_valid is a one-cycle pulse. Latch the byte, offer it to TX while TX is
  // free, and clear it on the one-cycle tx_accept pulse.
  logic [7:0] rx_hold = 8'h00;
  logic       rx_hold_full = 1'b0;
  always_ff @(posedge clk48) begin
    if (soc_reset) begin
      rx_hold      <= 8'h00;
      rx_hold_full <= 1'b0;
    end else begin
      if (rx_valid) begin
        // Capture the freshly received byte (overwrite is fine for a probe).
        rx_hold      <= rx_data;
        rx_hold_full <= 1'b1;
      end else if (tx_accept) begin
        // The held byte has been handed to the transmitter.
        rx_hold_full <= 1'b0;
      end
    end
  end

  assign tx_data  = rx_hold;
  // Offer the held byte to TX whenever the transmitter is free.
  assign tx_valid = rx_hold_full & tx_ready;

  LoomUart u_uart (
      .clk       (clk48),
      .reset     (soc_reset),
      .rx        (uart_rx),
      .tx        (uart_tx),
      .tx_data   (tx_data),
      .tx_valid  (tx_valid),
      .tx_ready  (tx_ready),
      .tx_accept (tx_accept),
      .rx_data   (rx_data),
      .rx_valid  (rx_valid)
  );
endmodule
''';

/// OrangeCrab r0.2 LPF for [_uartEchoTopSv]. Constrains every `LoomUartEchoTop`
/// port to its real board pin (all LVCMOS33), with the 48 MHz timing
/// constraint on the oscillator. Mirrors the UART board LPF (TX=M18, RX=N17).
const String _uartEchoTopLpf = '''
# OrangeCrab r0.2 (LFE5U-25F-8MG285C, CSFBGA285) - LoomUartEchoTop LPF.
# All LVCMOS33. Authoritative OrangeCrab r0.2 pinout.

# 48 MHz oscillator -> clk48
LOCATE COMP "clk48" SITE "A9";
IOBUF PORT "clk48" IO_TYPE=LVCMOS33;
FREQUENCY PORT "clk48" 48.0 MHz;

# Reset button (active-low)
LOCATE COMP "rst_n" SITE "V17";
IOBUF PORT "rst_n" IO_TYPE=LVCMOS33;

# UART on the feather GPIO header (full duplex: TX out, RX in)
LOCATE COMP "uart_tx" SITE "M18";
IOBUF PORT "uart_tx" IO_TYPE=LVCMOS33;
LOCATE COMP "uart_rx" SITE "N17";
IOBUF PORT "uart_rx" IO_TYPE=LVCMOS33;

# RGB LED green channel (active-low)
LOCATE COMP "led" SITE "M3";
IOBUF PORT "led" IO_TYPE=LVCMOS33;
''';

// uartechoswap transport: the UART ECHO loopback with TX/RX pins SWAPPED.
//
// Identical echo glue to LoomUartEchoTop, but the serial pins are CROSSED OVER
// relative to it: uart_tx is on N17 (GPIO:0) and uart_rx is on M18 (GPIO:1),
// the OPPOSITE of LoomUartEchoTop (tx=M18, rx=N17). This tests the hypothesis
// that the original echo returns 0 bytes because the host's jumpers are
// straight-through relative to our tx=M18/rx=N17 assignment (so a real TX/RX
// crossover mismatch kills comms both ways). If THIS swapped echo round-trips,
// the fix is to set the bridge/transport pins to tx=N17 / rx=M18.
//
// Same echo glue (rx_data -> tx_data via a one-deep holding reg, tx_valid
// pulsed while TX is free, cleared on tx_accept), same LoomUart (115200,
// divisor 417), clk48=A9, rst_n=V17, led=M3. ONLY the M18/N17 LOCATE lines and
// the tx/tx port wiring are swapped.

/// Generates just the [LoomUart] SystemVerilog for the swapped-echo build.
/// Shares the exact same RTL emission as the probe / echo builds.
Future<void> _generateUartEchoSwapRtl(Directory outDir) async {
  final uart = LoomUart(
    config: const LoomUartConfig(
      busAddressWidth: 12,
      busDataWidth: 32,
      clockFrequency: 48000000,
      baudRate: 115200,
    ),
    name: 'loom_uart',
  );
  await uart.build();

  final rtlDir = Directory('${outDir.path}/rtl')..createSync(recursive: true);
  final filelist = StringBuffer();
  final synthBuilder = SynthBuilder(uart, SystemVerilogSynthesizer());
  for (final fileContents in synthBuilder.getSynthFileContents()) {
    final fileName = '${fileContents.name}.sv';
    File('${rtlDir.path}/$fileName').writeAsStringSync(fileContents.contents);
    filelist.writeln('./rtl/$fileName');
  }
  File('${outDir.path}/filelist.f').writeAsStringSync(filelist.toString());
}

/// Hand-written OrangeCrab r0.2 board-top for the SWAPPED UART ECHO loopback
/// (`LoomUartEchoSwapTop`). Identical to [_uartEchoTopSv] except the serial
/// pins are crossed over: uart_tx=N17, uart_rx=M18 (the opposite of the
/// straight echo). Tests a host-side TX/RX crossover mismatch.
const String _uartEchoSwapTopSv = '''
// OrangeCrab r0.2 board-top for the Loom full-duplex UART ECHO loopback,
// TX/RX SWAPPED.
//
// LINK DIAGNOSTIC: every byte received on uart_rx (M18) is re-transmitted on
// uart_tx (N17) at 115200 8N1. This is the CROSSOVER of LoomUartEchoTop
// (which used rx=N17, tx=M18). If the straight echo returns 0 bytes but THIS
// swapped echo round-trips, the host wiring was straight-through and the fix
// is to set the transport pins to tx=N17 / rx=M18.
//
// Target: LFE5U-25F-8MG285C (CSFBGA285), all LVCMOS33.
module LoomUartEchoSwapTop (
    input  logic clk48,    // A9  - 48 MHz oscillator
    input  logic rst_n,    // V17 - reset button, ACTIVE-LOW
    input  logic uart_rx,  // M18 - serial in  (SWAPPED: was N17)
    output logic uart_tx,  // N17 - serial out (SWAPPED: was M18)
    output logic led       // M3  - RGB green channel, ACTIVE-LOW (lit = 0)
);
  logic [15:0] por_cnt = 16'd0;
  logic        por_done = 1'b0;
  always_ff @(posedge clk48) begin
    if (!por_done) begin
      por_cnt  <= por_cnt + 16'd1;
      por_done <= &por_cnt;
    end
  end
  logic soc_reset;
  assign soc_reset = (!por_done) | (~rst_n);

  logic [23:0] blink = 24'd0;
  always_ff @(posedge clk48) blink <= blink + 24'd1;
  assign led = ~blink[23];

  logic       tx_ready;
  logic       tx_accept;
  logic       tx_valid;
  logic [7:0] tx_data;
  logic [7:0] rx_data;
  logic       rx_valid;

  logic [7:0] rx_hold = 8'h00;
  logic       rx_hold_full = 1'b0;
  always_ff @(posedge clk48) begin
    if (soc_reset) begin
      rx_hold      <= 8'h00;
      rx_hold_full <= 1'b0;
    end else begin
      if (rx_valid) begin
        rx_hold      <= rx_data;
        rx_hold_full <= 1'b1;
      end else if (tx_accept) begin
        rx_hold_full <= 1'b0;
      end
    end
  end

  assign tx_data  = rx_hold;
  assign tx_valid = rx_hold_full & tx_ready;

  LoomUart u_uart (
      .clk       (clk48),
      .reset     (soc_reset),
      .rx        (uart_rx),
      .tx        (uart_tx),
      .tx_data   (tx_data),
      .tx_valid  (tx_valid),
      .tx_ready  (tx_ready),
      .tx_accept (tx_accept),
      .rx_data   (rx_data),
      .rx_valid  (rx_valid)
  );
endmodule
''';

/// OrangeCrab r0.2 LPF for [_uartEchoSwapTopSv]. Identical to [_uartEchoTopLpf]
/// except the M18/N17 LOCATE assignments are SWAPPED: uart_tx -> N17,
/// uart_rx -> M18.
const String _uartEchoSwapTopLpf = '''
# OrangeCrab r0.2 (LFE5U-25F-8MG285C, CSFBGA285) - LoomUartEchoSwapTop LPF.
# All LVCMOS33. Authoritative OrangeCrab r0.2 pinout. TX/RX SWAPPED.

# 48 MHz oscillator -> clk48
LOCATE COMP "clk48" SITE "A9";
IOBUF PORT "clk48" IO_TYPE=LVCMOS33;
FREQUENCY PORT "clk48" 48.0 MHz;

# Reset button (active-low)
LOCATE COMP "rst_n" SITE "V17";
IOBUF PORT "rst_n" IO_TYPE=LVCMOS33;

# UART on the feather GPIO header, SWAPPED relative to the straight echo:
# tx now on N17, rx now on M18 (full duplex: TX out, RX in)
LOCATE COMP "uart_tx" SITE "N17";
IOBUF PORT "uart_tx" IO_TYPE=LVCMOS33;
LOCATE COMP "uart_rx" SITE "M18";
IOBUF PORT "uart_rx" IO_TYPE=LVCMOS33;

# RGB LED green channel (active-low)
LOCATE COMP "led" SITE "M3";
IOBUF PORT "led" IO_TYPE=LVCMOS33;
''';

Future<void> main(List<String> argv) async {
  final parser = ArgParser()
    ..addOption(
      'output',
      abbr: 'o',
      defaultsTo: 'out',
      help: 'Output directory.',
    )
    ..addOption('name', help: 'SoC name (defaults per transport).')
    ..addOption(
      'transport',
      defaultsTo: 'usb',
      allowed: ['usb', 'uart', 'uartprobe', 'uartecho', 'uartechoswap'],
      help:
          'Command transport: bit-banged USB device, 8N1 UART (115200), '
          'uartprobe (TX-only continuous 0x55 link diagnostic), uartecho '
          '(full-duplex RX->TX loopback link diagnostic), or uartechoswap '
          '(same loopback with TX/RX pins crossed: tx=N17, rx=M18).',
    )
    ..addOption(
      'soc',
      defaultsTo: 'overlay',
      allowed: ['overlay', 'stream', 'fp'],
      help:
          'SoC variant: overlay (the small 8x8 accelerator per --transport), '
          'stream (memory-backed int8 matmul: UART host + LoomStreamAccelerator '
          '+ SRAM), or fp (memory-backed fp16 W4A8 LINEAR: UART host + '
          'LoomFpLinearAccelerator + SRAM/DDR - the hybrid-demo datapath).',
    )
    ..addOption(
      'board',
      defaultsTo: 'orangecrab-25f',
      help:
          'Target board from Harbor\'s HarborBoard catalog (orangecrab-25f, '
          'arty-s7-50, ...). Supplies the vendor/device/package, the oscillator '
          'frequency, the pin sites, and the program command - so retargeting is '
          'one flag, no code change. --pin adds/overrides individual sites; '
          '--target overrides the whole thing for a board not in the catalog.',
    )
    ..addOption(
      'target',
      help:
          'Escape hatch for a board NOT in the catalog: "vendor:device:package" '
          '(e.g. ecp5:25f:CSFBGA285, ice40:up5k:sg48) with pin sites from --pin. '
          'Overrides --board when set.',
    )
    ..addMultiOption(
      'pin',
      abbr: 'p',
      help:
          'Pin assignment "name=site" (e.g. clk=A9, uart_rx=M18, uart_tx=N17). '
          'Repeatable. Defaults to the OrangeCrab r0.2 pinout if none given.',
    )
    ..addFlag(
      'ddr',
      negatable: false,
      help:
          'With --soc stream: add the DDR3 controller as the weight store '
          '(128MB), exposing the sdram_* pads. Synth needs the DDR pinout via '
          '--pin.',
    )
    ..addFlag(
      'fp-flash',
      negatable: false,
      help:
          '--soc fp: use the OrangeCrab config SPI flash as a RESIDENT weight '
          'store (accelerator reads int4 weights straight from flash; host '
          'provisions nothing). Weights are ecpprog\'d above the bitstream.',
    )
    ..addOption(
      'fp-col-tiles',
      defaultsTo: '32',
      help:
          '--soc fp: max inner-dim tiles (maxCols = 2*this). Flop buffers, '
          'so keep modest until the BRAM redesign.',
    )
    ..addOption(
      'fp-row-blocks',
      defaultsTo: '32',
      help: '--soc fp: max output-row blocks (maxRows = 2*this).',
    )
    ..addOption(
      'fp-mhz',
      defaultsTo: '30',
      help:
          '--soc fp: fabric clock MHz (PLL-derived from the 48MHz osc). '
          'The placed design closes ~37MHz; 30 runs ~19% under that (verified '
          'correct on silicon incl. flash reads) and makes 1500000 baud an EXACT '
          'divisor (20). Drop to 24 for more timing margin if a board is flaky.',
    )
    ..addOption(
      'fp-read-ahead',
      defaultsTo: '8',
      help:
          '--soc fp --fp-flash: SPI flash read-ahead line length (words per '
          'flash command). 8 amortizes the per-word cmd+addr+dummy overhead; '
          '1 disables it (one command per word), for A/B measuring the cache.',
    )
    ..addOption(
      'fp-bram-cache-kb',
      defaultsTo: '0',
      help:
          '--soc fp: size (KiB) of the on-chip BRAM HOT-weight cache. 0 '
          '(default) keeps all weights in flash (unchanged). N>0 adds a '
          'HarborSram weight store at 0x30000000 and splits the model: the '
          'biggest matrices (by byte size) fill the N KiB BRAM budget, the '
          'rest stay in flash. Emits weights_bram.bin/scales_bram.bin + '
          'weights.bin/scales_flash.bin, each matrix tagged store=bram|flash '
          'in loom.json. ~90 fills the 25F BRAM (stories260K ~130KB total).',
    )
    ..addOption(
      'fp-baud',
      defaultsTo: '1500000',
      help:
          '--soc fp: UART baud. The transport is bandwidth-bound, so this is '
          'the main speed lever. 1500000 is the DirtyJTAG CDC bridge\'s reliable '
          'ceiling on this board (2000000 corrupts); divisor derives from the '
          'fabric clock, so keep --fp-mhz such that it divides cleanly.',
    )
    ..addOption(
      'model',
      help:
          '--soc fp: path to a model (llama2.c .bin) to also emit as flash '
          'weight artifacts (weights.bin + scales.bin + loom.json manifest) '
          'the runtime consumes. The generator emits everything for the config.',
    )
    ..addOption(
      'tokenizer',
      help:
          '--soc fp --model: path to the tokenizer (llama2.c tok*.bin) to '
          'copy into the output as tokenizer.bin for the runtime.',
    )
    ..addOption(
      'ddr-read-tap',
      defaultsTo: '40',
      help:
          '--soc fp --ddr: static DDR3 read tap (DELAYG DEL_VALUE, 0..127). '
          'Sweep on the board to centre the DLL-off read eye.',
    )
    ..addFlag('help', abbr: 'h', negatable: false, help: 'Show usage.');

  final args = parser.parse(argv);
  if (args.flag('help')) {
    stdout.writeln(
      'loom_genip - generate Loom SoC RTL + DTS/SVD + ECP5 flow\n',
    );
    stdout.writeln(parser.usage);
    return;
  }

  final output = args.option('output')!;
  final transport = args.option('transport')!;

  // --ddr: the DRAM part configuration AND the pad sites come from the board
  // entry ([LoomDdrBoard.byName]), in the way River's DdrBoard.byName works. A
  // board with no entry is an error, so a build cannot take the pad sites of a
  // different board. --target names no board, so it cannot supply either.
  final ddrReadTap = int.parse(args.option('ddr-read-tap')!);
  LoomDdrBoard? ddrBoard;
  var ddrPinMap = const <String, String>{};
  if (args.flag('ddr')) {
    if (args.option('target') != null) {
      throw ArgumentError(
        '--ddr needs --board: the DRAM part configuration and the pad sites '
        'come from the board entry. Known: '
        '${LoomDdrBoard.byName.keys.join(', ')}.',
      );
    }
    final board = args.option('board')!;
    ddrBoard = LoomDdrBoard.require(board);
    ddrPinMap = LoomDdrBoard.pinsFor(board);
  }

  // The streaming, memory-backed-matmul SoC: fully Harbor-generated and
  // target-parameterized. Orthogonal to --transport (it always uses the UART
  // host bridge for now). Selected via --soc stream.
  if (args.option('soc') == 'stream') {
    // Declarative target + pins (River-style): nothing about the device,
    // package, or pinout is hardcoded - it all comes from the CLI, so the same
    // SoC retargets by changing flags.
    final target = _resolveTarget(
      board: args.option('board'),
      targetSpec: args.option('target'),
      pinSpecs: args.multiOption('pin'),
      exposedPins: const ['clk', 'uart_tx', 'uart_rx'],
      extraPinMap: ddrPinMap,
    );
    final soc = buildLoomSoc(
      name: args.option('name') ?? 'LoomStreamSoC',
      transport: LoomTransport.uart,
      target: target,
      oscHz: target.frequency,
      // --ddr drops the closing freq below 48. Default the stream fabric to the
      // --fp-mhz knob (32) when DDR is on, else keep the direct osc.
      fabricHz: args.flag('ddr')
          ? int.parse(args.option('fp-mhz')!) * 1000000
          : target.frequency,
      datapath: StreamDatapath(ddr: ddrBoard, ddrReadTap: ddrReadTap),
    );
    await soc.generateAll(Directory(output));
    stdout.writeln(
      'Generated Loom streaming SoC (${args.option('board') ?? args.option('target')}) into $output',
    );
    return;
  }

  // The fp16 W4A8 LINEAR SoC: the hybrid-demo datapath (device does the matmuls
  // in floating point, host orchestrates the nonlinear glue).
  if (args.option('soc') == 'fp') {
    // The signals this SoC drives, constrained from the board catalog: the osc,
    // the UART, plus the quad-SPI flash pads when --fp-flash resides weights in
    // flash. DDR sdram_* sites (SSTL135, not in the board catalog yet) come in as
    // an extra pin map when --ddr is set.
    final exposed = <String>[
      'clk',
      'uart_tx',
      'uart_rx',
      if (args.flag('fp-flash')) ...[
        'spi_cs_n',
        'spi_io[0]',
        'spi_io[1]',
        'spi_io[2]',
        'spi_io[3]',
      ],
    ];
    final target = _resolveTarget(
      board: args.option('board'),
      targetSpec: args.option('target'),
      pinSpecs: args.multiOption('pin'),
      exposedPins: exposed,
      extraPinMap: ddrPinMap,
    );
    // Load the model BEFORE building the SoC: a BitNet ternary model makes the
    // fp accelerator's PE multiply-free (ternaryWeights), which is a build-time
    // RTL choice, so the graph.ternary flag must be known before generateAll.
    final modelPath = args.option('model');
    final loaded = modelPath != null ? loadModelSource(modelPath) : null;
    final soc = buildLoomSoc(
      name: args.option('name') ?? 'LoomFpSoC',
      transport: LoomTransport.uart,
      target: target,
      oscHz: target.frequency, // board oscillator -> PLL source
      fabricHz: int.parse(args.option('fp-mhz')!) * 1000000,
      baudRate: int.parse(args.option('fp-baud')!),
      datapath: FpDatapath(
        maxColTiles: int.parse(args.option('fp-col-tiles')!),
        maxRowBlocks: int.parse(args.option('fp-row-blocks')!),
        ddr: ddrBoard,
        ddrReadTap: ddrReadTap,
        weightsInFlash: args.flag('fp-flash'),
        flashReadAhead: int.parse(args.option('fp-read-ahead')!),
        bramCacheKb: int.parse(args.option('fp-bram-cache-kb')!),
        ternary: loaded?.graph.ternary ?? false,
      ),
    );
    await soc.generateAll(Directory(output));
    stdout.writeln(
      'Generated Loom fp16 W4A8 linear SoC (${args.option('board') ?? args.option('target')}) into $output',
    );
    if (loaded != null) {
      _emitModelArtifacts(
        output,
        loaded,
        bramCacheKb: int.parse(args.option('fp-bram-cache-kb')!),
        // Accelerator maxCols = maxColTiles * peCols(2); wider matrices get the
        // col-block layout the runtime col-tiler consumes.
        maxCols: int.parse(args.option('fp-col-tiles')!) * 2,
      );
      // Tokenizer emit. SentencePiece tok*.bin is copied as-is. An HF BPE
      // tokenizer.json is compiled to the runtime's LTB1 tokenizer.bin plus a
      // small encode fixture for the runtime unit gate.
      final tokPath = args.option('tokenizer') ?? loaded.tokenizerPath;
      String? bpeJsonText;
      if (tokPath != null && tokPath.toLowerCase().endsWith('.json')) {
        bpeJsonText = File(tokPath).readAsStringSync();
      } else if (tokPath != null) {
        // SentencePiece tok*.bin: copy unchanged.
        File(
          '$output/tokenizer.bin',
        ).writeAsBytesSync(File(tokPath).readAsBytesSync());
        stdout.writeln('Copied tokenizer -> $output/tokenizer.bin');
      } else if (loaded.tokenizerJson != null) {
        bpeJsonText = loaded.tokenizerJson;
      }
      if (bpeJsonText != null) {
        final tok = BpeTokenizer.fromJson(
          jsonDecode(bpeJsonText) as Map<String, dynamic>,
        );
        // These stories/SmolLM2 tokenizers use <|endoftext|> as bos/eos and do
        // not prepend BOS to prompts (fixture prompt_ids carry no leading BOS).
        final eot = _findAddedTokenId(bpeJsonText, '<|endoftext|>');
        final bytes = tok.toLtb1(
          bosId: eot ?? -1,
          eosId: eot ?? -1,
          addBos: false,
        );
        File('$output/tokenizer.bin').writeAsBytesSync(bytes);
        // Encode fixture: a few prompts the runtime must reproduce exactly.
        final cases = <Map<String, dynamic>>[];
        for (final s in const [
          'Once upon a time',
          'The little',
          'She said, "Hello!"',
        ]) {
          cases.add({'text': s, 'ids': tok.encode(s)});
        }
        File(
          '$output/tokenizer_fixture.json',
        ).writeAsStringSync(jsonEncode({'cases': cases}));
        stdout.writeln(
          'Compiled BPE tokenizer -> $output/tokenizer.bin '
          '(${bytes.length}B) + tokenizer_fixture.json',
        );
      }
    }
    return;
  }

  final boardDir = Directory('$output/board')..createSync(recursive: true);

  if (transport == 'uartprobe') {
    // TX-only continuous probe: emit just the LoomUart RTL + the hand-written
    // probe board-top and its LPF. No SoC, no bus, no command engine.
    await _generateUartProbeRtl(Directory(output));
    File(
      '${boardDir.path}/LoomUartProbeTop.sv',
    ).writeAsStringSync(_uartProbeTopSv);
    File(
      '${boardDir.path}/LoomUartProbeTop.lpf',
    ).writeAsStringSync(_uartProbeTopLpf);
  } else if (transport == 'uartecho') {
    // Full-duplex loopback: emit just the LoomUart RTL + the hand-written echo
    // board-top and its LPF. No SoC, no bus, no command engine.
    await _generateUartEchoRtl(Directory(output));
    File(
      '${boardDir.path}/LoomUartEchoTop.sv',
    ).writeAsStringSync(_uartEchoTopSv);
    File(
      '${boardDir.path}/LoomUartEchoTop.lpf',
    ).writeAsStringSync(_uartEchoTopLpf);
  } else if (transport == 'uartechoswap') {
    // Full-duplex loopback with TX/RX pins SWAPPED (tx=N17, rx=M18): emit just
    // the LoomUart RTL + the hand-written swapped-echo board-top and its LPF.
    await _generateUartEchoSwapRtl(Directory(output));
    File(
      '${boardDir.path}/LoomUartEchoSwapTop.sv',
    ).writeAsStringSync(_uartEchoSwapTopSv);
    File(
      '${boardDir.path}/LoomUartEchoSwapTop.lpf',
    ).writeAsStringSync(_uartEchoSwapTopLpf);
  } else if (transport == 'uart') {
    final name = args.option('name') ?? 'LoomUartSoC';
    final soc = buildLoomSoc(
      name: name,
      transport: LoomTransport.uart,
      datapath: const OverlayDatapath(),
    );
    await soc.generateAll(Directory(output));
    // The flashable UART artifacts: board-top wrapper + real-pin LPF.
    File('${boardDir.path}/LoomUartTop.sv').writeAsStringSync(_uartBoardTopSv);
    File(
      '${boardDir.path}/LoomUartTop.lpf',
    ).writeAsStringSync(_uartBoardTopLpf);
  } else {
    final name = args.option('name') ?? 'LoomSoC';
    final soc = buildLoomSoc(
      name: name,
      transport: LoomTransport.usb,
      datapath: const OverlayDatapath(),
    );
    await soc.generateAll(Directory(output));

    // Emit the hand-written OrangeCrab board-top shim + its real-pin LPF. These
    // are the artifacts the flashable bitstream is built from: LoomTop wraps the
    // generated LoomSoC with bidirectional USB pads, reset invert, and LED
    // blink.
    File('${boardDir.path}/LoomTop.sv').writeAsStringSync(_boardTopSv);
    File('${boardDir.path}/LoomTop.lpf').writeAsStringSync(_boardTopLpf);
  }

  stdout.writeln('Generated Loom SoC into $output:');
  final dir = Directory(output);
  final entries = dir.listSync(recursive: true)
    ..sort((a, b) => a.path.compareTo(b.path));
  for (final entry in entries) {
    if (entry is File) {
      stdout.writeln('  ${entry.path}');
    }
  }
}
