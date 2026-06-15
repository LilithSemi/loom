// Loom Linear Layer: int8 linear layer combining LoomMatmul + LoomRequantize.
//
// Computes y[r] = requantize(acc[r], rowMult[r], shift) where
//   acc[r] = sum_c W[r,c] * x[c]   (int32, via LoomMatmul)
//   y[r]   = saturated int8 output  (via LoomRequantize per row)
//
// Microarchitecture:
//   - One LoomMatmul(peRows, peCols, inWidth, accWidth) handles the
//     streaming tile-accumulating matrix-vector multiply.
//   - peRows LoomRequantize(accWidth, multWidth, shiftWidth, outWidth) units,
//     one per output row, perform the combinational requantization.
//   - requant[r].acc <= matmul.acc[r*accWidth +: accWidth]
//   - requant[r].mult <= rowMult[r*multWidth +: multWidth]
//   - requant[r].shift <= shift (shared)
//   - out[r*outWidth +: outWidth] <= requant[r].out
//   - outValid = matmul.resultValid
//
// Since requant is combinational, out is valid the exact cycle resultValid
// pulses: one cycle after the last tile's posedge (matmul timing).
//
// Port layout:
//   wTile   : peRows*peCols*inWidth   W[r,c] at (r*peCols+c)*inWidth (signed)
//   xTile   : peCols*inWidth          x[c] at c*inWidth (signed)
//   valid   : 1                       tile-valid strobe
//   first   : 1                       first tile of a block
//   last    : 1                       last tile of a block
//   rowMult : peRows*multWidth        mult[r] at r*multWidth (unsigned)
//   shift   : shiftWidth              shared shift amount (unsigned)
//   out     : peRows*outWidth         y[r] at r*outWidth (signed int8)
//   outValid: 1                       one-cycle pulse matching matmul.resultValid

import 'package:rohd/rohd.dart';

import 'matmul.dart';
import 'requantize.dart';

// Configuration

/// Immutable configuration for [LoomLinear].
class LoomLinearConfig {
  /// Number of output rows (height of the weight tile block).
  final int peRows;

  /// Number of columns per tile (inner dimension tile size).
  final int peCols;

  /// Bit width of each signed input operand (W and x elements).
  final int inWidth;

  /// Bit width of each signed 32-bit accumulator element.
  final int accWidth;

  /// Bit width of the unsigned per-row requant multiplier.
  final int multWidth;

  /// Bit width of the shared requant shift amount.
  final int shiftWidth;

  /// Bit width of the saturated signed output per row.
  final int outWidth;

  /// Pipeline latency (clock cycles) of the inner PE array (see LoomMatmul).
  final int peLatency;

  /// Pipeline latency (clock cycles, 0..2) of each LoomRequantize unit. Stage 1
  /// registers the wide product. Stage 2 registers the rounded value. outValid
  /// is delayed by this amount so it still pulses exactly when `out` is valid.
  final int requantLatency;

  /// Multiply-free BitNet path: ternary {-1,0,+1} weights, forwarded to the
  /// inner [LoomMatmul]/[LoomPeArray] which use select/negate instead of a DSP.
  final bool ternaryWeights;

  const LoomLinearConfig({
    required this.peRows,
    required this.peCols,
    this.inWidth = 8,
    this.accWidth = 32,
    this.multWidth = 16,
    this.shiftWidth = 6,
    this.outWidth = 8,
    this.peLatency = 0,
    this.requantLatency = 0,
    this.ternaryWeights = false,
  });

  /// Validate all sub-config constraints plus the top-level ones.
  void validate() {
    if (peRows <= 0) {
      throw ArgumentError('LoomLinearConfig.peRows must be > 0, got $peRows');
    }
    LoomMatmulConfig(
      peRows: peRows,
      peCols: peCols,
      inWidth: inWidth,
      accWidth: accWidth,
      peLatency: peLatency,
      ternaryWeights: ternaryWeights,
    ).validate();
    LoomRequantizeConfig(
      accWidth: accWidth,
      multWidth: multWidth,
      shiftWidth: shiftWidth,
      outWidth: outWidth,
      latency: requantLatency,
    ).validate();
  }
}

// Module

/// Int8 linear layer: streaming tile matmul + per-row combinational requant.
///
/// Instantiates one [LoomMatmul] and [peRows] [LoomRequantize] units.
/// See the file-level comment for the timing contract.
///
/// See [LoomLinearConfig] for parameter constraints and [validate()].
class LoomLinear extends Module {
  /// Clock input.
  Logic get clk => input('clk');

  /// Synchronous active-high reset.
  Logic get reset => input('reset');

  /// Weight tile bus.
  Logic get wTile => input('wTile');

  /// Activation tile bus.
  Logic get xTile => input('xTile');

  /// Tile-valid strobe.
  Logic get valid => input('valid');

  /// First-tile-of-block strobe.
  Logic get first => input('first');

  /// Last-tile-of-block strobe.
  Logic get last => input('last');

  /// Per-row unsigned requant multipliers: mult[r] at r*multWidth.
  Logic get rowMult => input('rowMult');

  /// Shared unsigned requant shift amount.
  Logic get shift => input('shift');

  /// Requantized signed int8 outputs: y[r] at r*outWidth.
  Logic get out => output('out');

  /// High for one cycle when out is valid (mirrors matmul.resultValid).
  Logic get outValid => output('outValid');

  /// Raw signed int32 accumulators, acc[r] at r*accWidth. Valid the same cycle
  /// as [outValid] (the requantize path is a parallel tap, not in series). The
  /// fp16 W4A8 path dequantizes this directly instead of using [out].
  Logic get acc => output('acc');

  /// Construct [LoomLinear].
  ///
  /// All Logic args are external signals to connect. Config params are
  /// forwarded to [LoomLinearConfig] which calls [validate()] at construction.
  LoomLinear({
    required Logic clk,
    required Logic reset,
    required Logic wTile,
    required Logic xTile,
    required Logic valid,
    required Logic first,
    required Logic last,
    required Logic rowMult,
    required Logic shift,
    required int peRows,
    required int peCols,
    int inWidth = 8,
    int accWidth = 32,
    int multWidth = 16,
    int shiftWidth = 6,
    int outWidth = 8,
    int peLatency = 0,
    int requantLatency = 0,
    bool ternaryWeights = false,
  }) : super(name: 'LoomLinear', definitionName: 'LoomLinear') {
    LoomLinearConfig(
      peRows: peRows,
      peCols: peCols,
      inWidth: inWidth,
      accWidth: accWidth,
      multWidth: multWidth,
      shiftWidth: shiftWidth,
      outWidth: outWidth,
      peLatency: peLatency,
      ternaryWeights: ternaryWeights,
      requantLatency: requantLatency,
    ).validate();

    // Register ports.
    final clkPort = addInput('clk', clk);
    final resetPort = addInput('reset', reset);
    final wTilePort = addInput(
      'wTile',
      wTile,
      width: peRows * peCols * inWidth,
    );
    final xTilePort = addInput('xTile', xTile, width: peCols * inWidth);
    final validPort = addInput('valid', valid);
    final firstPort = addInput('first', first);
    final lastPort = addInput('last', last);
    final rowMultPort = addInput('rowMult', rowMult, width: peRows * multWidth);
    final shiftPort = addInput('shift', shift, width: shiftWidth);
    addOutput('out', width: peRows * outWidth);
    addOutput('outValid');
    addOutput('acc', width: peRows * accWidth);

    // Feed the module-internal port copies (result of addInput) to it.
    final mmWTile = Logic(name: 'mm_wTile', width: peRows * peCols * inWidth);
    final mmXTile = Logic(name: 'mm_xTile', width: peCols * inWidth);
    final mmValid = Logic(name: 'mm_valid');
    final mmFirst = Logic(name: 'mm_first');
    final mmLast = Logic(name: 'mm_last');
    mmWTile <= wTilePort;
    mmXTile <= xTilePort;
    mmValid <= validPort;
    mmFirst <= firstPort;
    mmLast <= lastPort;

    final matmul = LoomMatmul(
      clk: clkPort,
      reset: resetPort,
      wTile: mmWTile,
      xTile: mmXTile,
      valid: mmValid,
      first: mmFirst,
      last: mmLast,
      peRows: peRows,
      peCols: peCols,
      inWidth: inWidth,
      accWidth: accWidth,
      peLatency: peLatency,
      ternaryWeights: ternaryWeights,
    );

    // Instantiate peRows LoomRequantize units, one per output row.
    // requant[r].acc  <- matmul.acc[r*accWidth +: accWidth]
    // requant[r].mult <- rowMult[r*multWidth +: multWidth]
    // requant[r].shift <- shift (shared)
    // out[r*outWidth +: outWidth] <- requant[r].out
    final outSlices = <Logic>[];

    for (var r = 0; r < peRows; r++) {
      // Slice acc for row r (signed accWidth bits).
      final accSlice = matmul.acc
          .slice((r + 1) * accWidth - 1, r * accWidth)
          .named('acc_r$r');

      // Slice rowMult for row r (unsigned multWidth bits).
      final multSlice = rowMultPort
          .slice((r + 1) * multWidth - 1, r * multWidth)
          .named('mult_r$r');

      final rqAcc = Logic(name: 'rq_acc_r$r', width: accWidth);
      final rqMult = Logic(name: 'rq_mult_r$r', width: multWidth);
      final rqShift = Logic(name: 'rq_shift_r$r', width: shiftWidth);
      rqAcc <= accSlice;
      rqMult <= multSlice;
      rqShift <= shiftPort;

      final rq = LoomRequantize(
        acc: rqAcc,
        mult: rqMult,
        shift: rqShift,
        accWidth: accWidth,
        multWidth: multWidth,
        shiftWidth: shiftWidth,
        outWidth: outWidth,
        latency: requantLatency,
        clk: requantLatency >= 1 ? clkPort : null,
        reset: requantLatency >= 1 ? resetPort : null,
      );

      outSlices.add(rq.out.named('out_r$r'));
    }

    // Assemble output bus: element 0 in low bits.
    // swizzle puts the last element in the MSB. Reverse so row 0 is low.
    out <= outSlices.reversed.toList().swizzle();

    // Expose the raw int32 accumulators (parallel tap of the matmul) for the
    // fp16 W4A8 dequant path.
    output('acc') <= matmul.acc;

    // outValid mirrors matmul.resultValid, delayed by requantLatency so it
    // pulses exactly when the requantized `out` is valid. At requantLatency=0
    // this reduces to a direct combinational mirror.
    Logic ov = matmul.resultValid;
    for (var s = 0; s < requantLatency; s++) {
      final r = Logic(name: 'ov_d$s');
      r <= flop(clkPort, ov, reset: resetPort, resetValue: 0);
      ov = r;
    }
    outValid <= ov;
  }
}
