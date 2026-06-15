// Loom Matrix-Vector Engine: sequential streaming tile-accumulating matmul.
//
// Computes a full peRows-row block's dot products over an arbitrarily long
// inner dimension K by streaming peCols-wide column tiles through a LoomPeArray
// across multiple clock cycles and accumulating.
//
// Microarchitecture:
//   - Instantiates one LoomPeArray(rows=peRows, cols=peCols, latency=peLatency).
//     The PE array is PIPELINED: its output pe.y for a tile presented at
//     posedge N appears peLatency cycles later (peLatency=0 => combinational).
//   - Maintains a register bank: accReg[r] (peRows registers, each accWidth
//     bits, signed two's complement).
//   - The tile control signals (valid, first, last) are delayed by peLatency
//     cycles through a shift register so they line up with the matching
//     (delayed) pe.y. The accumulate then uses these DELAYED control signals:
//       dvalid/dfirst/dlast = control of the tile whose product is on pe.y now.
//   - On each cycle where dvalid=1:
//       p[r] = pe.y[r] (the product of the tile that entered peLatency cycles
//              ago, which is exactly the tile dfirst/dlast describe)
//       accReg[r] <= (dfirst ? 0 : accReg[r]) + p[r]  (registered at posedge)
//   - acc output = accReg (registered).
//   - resultValid asserts the cycle AFTER dvalid&dlast (one accumulate bubble),
//     i.e. peLatency+1 cycles after the last tile is presented. With the old
//     combinational PE (peLatency=0) this reduces to the previous "cycle after
//     valid&last" contract exactly.
//   - resultValid itself is a 1-bit register: set when (dvalid & dlast), cleared
//     otherwise (default) or by reset.
//
// Overflow assumption:
//   The caller is responsible for choosing accWidth large enough that no
//   signed 32-bit overflow occurs across K/peCols tiles.  For int8 inputs
//   accumulating K=4096 elements: max |sum| <= 127*127*4096 ~ 66M, which
//   fits comfortably in 32 bits.  validate() enforces the per-tile minimum
//   (same as LoomPeArrayConfig).
//
// Timing summary (posedge-aligned), for PE latency L = peLatency:
//   Cycle 0:   valid=1, first=1, wTile/xTile = tile0  (enters the PE pipeline)
//   ...
//   Cycle N:   valid=1, last=1,  wTile/xTile = tileN  (enters the PE pipeline)
//   Cycle L:   pe.y holds tile0's product. Dfirst=1 -> accReg gets tile0 result
//   ...
//   Cycle N+L: pe.y holds tileN's product. Dlast=1  -> accReg gets full sum
//   Cycle N+L+1: resultValid=1, acc=accReg (full sum readable)
// With L=0 this is exactly the legacy contract (resultValid at N+1).
//
// Port layout (little-element-endian, element 0 in low bits):
//   wTile : peRows*peCols*inWidth  W[r,c] at (r*peCols+c)*inWidth
//   xTile : peCols*inWidth         x[c] at c*inWidth
//   acc   : peRows*accWidth        acc[r] at r*accWidth (signed)

import 'dart:math' as math;

import 'package:rohd/rohd.dart';

import 'pe_array.dart';

// Configuration

/// Immutable configuration for [LoomMatmul].
class LoomMatmulConfig {
  /// Number of output rows processed per block (PE array height).
  final int peRows;

  /// Number of columns per tile (PE array width).
  final int peCols;

  /// Bit width of each signed input operand (W and x elements).
  final int inWidth;

  /// Bit width of each signed accumulator element.
  final int accWidth;

  /// Pipeline latency (clock cycles) of the inner [LoomPeArray] from operands
  /// to product. 0 => combinational PE (legacy). >= 1 => pipelined PE. The
  /// accumulate and resultValid timing are shifted by this many cycles.
  final int peLatency;

  /// Multiply-free BitNet path: weights are ternary {-1,0,+1}, so the inner
  /// [LoomPeArray] uses select/negate instead of a NativeMultiplier (no DSP).
  final bool ternaryWeights;

  const LoomMatmulConfig({
    required this.peRows,
    required this.peCols,
    this.inWidth = 8,
    this.accWidth = 32,
    this.peLatency = 0,
    this.ternaryWeights = false,
  });

  /// Minimum accWidth to hold a single-tile partial sum without overflow.
  /// Same formula as LoomPeArrayConfig: 2*inWidth + ceil(log2(peCols)).
  int get minAccWidth {
    if (peCols <= 1) return 2 * inWidth;
    final extraBits = (math.log(peCols) / math.ln2).ceil();
    return 2 * inWidth + extraBits;
  }

  /// Validate configuration. Throws [ArgumentError] on violation.
  void validate() {
    if (peRows <= 0) {
      throw ArgumentError('LoomMatmulConfig.peRows must be > 0, got $peRows');
    }
    if (peCols <= 0) {
      throw ArgumentError('LoomMatmulConfig.peCols must be > 0, got $peCols');
    }
    if (inWidth < 2) {
      throw ArgumentError(
        'LoomMatmulConfig.inWidth must be >= 2 (signed), got $inWidth',
      );
    }
    if (accWidth < minAccWidth) {
      throw ArgumentError(
        'LoomMatmulConfig.accWidth ($accWidth) is too small for '
        'peCols=$peCols inWidth=$inWidth; minimum is $minAccWidth',
      );
    }
    if (peLatency < 0) {
      throw ArgumentError(
        'LoomMatmulConfig.peLatency must be >= 0, got $peLatency',
      );
    }
  }
}

// Module

/// Sequential streaming tile-accumulating matrix-vector multiply engine.
///
/// Tiles a [LoomPeArray] over an arbitrarily long inner dimension K,
/// accumulating partial sums into a register bank. See [LoomMatmulConfig]
/// and the file-level comment for the timing contract.
class LoomMatmul extends Module {
  /// Clock input.
  Logic get clk => input('clk');

  /// Synchronous active-high reset.
  Logic get reset => input('reset');

  /// Weight tile bus: peRows x peCols int[inWidth], row-major.
  Logic get wTile => input('wTile');

  /// Activation tile bus: peCols int[inWidth] elements.
  Logic get xTile => input('xTile');

  /// Asserted when wTile/xTile present a valid tile this cycle.
  Logic get valid => input('valid');

  /// Asserted with valid to start a fresh accumulation (clear accReg).
  Logic get first => input('first');

  /// Asserted with valid to mark the last tile of a block.
  Logic get last => input('last');

  /// Accumulated result bus: peRows int[accWidth], signed, element 0 low.
  /// Valid when resultValid is high.
  Logic get acc => output('acc');

  /// High the cycle after valid&last: the full accumulated sum is on acc.
  Logic get resultValid => output('resultValid');

  /// Construct [LoomMatmul]. [clk], [reset], [wTile], [xTile], [valid],
  /// [first], [last] are the external signals to connect. The rest configure
  /// the module. Calls [LoomMatmulConfig.validate()] at construction.
  LoomMatmul({
    required Logic clk,
    required Logic reset,
    required Logic wTile,
    required Logic xTile,
    required Logic valid,
    required Logic first,
    required Logic last,
    required int peRows,
    required int peCols,
    int inWidth = 8,
    int accWidth = 32,
    int peLatency = 0,
    bool ternaryWeights = false,
  }) : super(name: 'LoomMatmul', definitionName: 'LoomMatmul') {
    LoomMatmulConfig(
      peRows: peRows,
      peCols: peCols,
      inWidth: inWidth,
      accWidth: accWidth,
      peLatency: peLatency,
      ternaryWeights: ternaryWeights,
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
    addOutput('acc', width: peRows * accWidth);
    addOutput('resultValid');

    // peWire/peX buffer the module input ports feeding the PE array.
    final peWire = Logic(name: 'pe_w', width: peRows * peCols * inWidth);
    final peX = Logic(name: 'pe_x', width: peCols * inWidth);
    peWire <= wTilePort;
    peX <= xTilePort;

    final pe = LoomPeArray(
      w: peWire,
      x: peX,
      rows: peRows,
      cols: peCols,
      inWidth: inWidth,
      accWidth: accWidth,
      latency: peLatency,
      ternaryWeights: ternaryWeights,
      clk: peLatency >= 1 ? clkPort : null,
      reset: peLatency >= 1 ? resetPort : null,
    );

    // pe.y is the partial-product vector: peRows*accWidth bits, delayed by
    // peLatency cycles relative to wTile/xTile.
    // Extract each row's partial product as a signed accWidth-bit value.

    // Delay the tile control signals (valid, first, last) by peLatency cycles
    // so they line up with the corresponding (delayed) pe.y. With peLatency=0
    // the delayed signals ARE the live ports (legacy behavior).
    Logic dValid = validPort;
    Logic dFirst = firstPort;
    Logic dLast = lastPort;
    for (var s = 0; s < peLatency; s++) {
      final vReg = Logic(name: 'dvalid_$s');
      final fReg = Logic(name: 'dfirst_$s');
      final lReg = Logic(name: 'dlast_$s');
      vReg <= flop(clkPort, dValid, reset: resetPort, resetValue: 0);
      fReg <= flop(clkPort, dFirst, reset: resetPort, resetValue: 0);
      lReg <= flop(clkPort, dLast, reset: resetPort, resetValue: 0);
      dValid = vReg;
      dFirst = fReg;
      dLast = lReg;
    }

    // Register bank: accReg[r] holds the running sum for row r.
    // All registers share the same Sequential block and reset together.
    final accRegs = [
      for (var r = 0; r < peRows; r++)
        Logic(name: 'accReg_$r', width: accWidth),
    ];

    // resultValid register: set when valid&last, cleared when valid&first or
    // reset. Cleared also when next valid block starts (first=1).
    final resultValidReg = Logic(name: 'resultValid_reg');

    final seqActions = <Conditional>[
      // Reset arm: zero everything.
      If(
        resetPort,
        then: [
          for (var r = 0; r < peRows; r++)
            accRegs[r] < Const(0, width: accWidth),
          resultValidReg < Const(0),
        ],
        orElse: [
          // Default: clear resultValidReg each cycle unless we just finished.
          resultValidReg < Const(0),

          If(
            dValid,
            then: [
              // On a delayed-valid cycle: update accReg for each row using the
              // pe.y product that lines up with this tile. p[r] below is
              // already accWidth bits, sign-extended inside LoomPeArray.
              for (var r = 0; r < peRows; r++)
                () {
                  final pSlice = pe.y
                      .slice((r + 1) * accWidth - 1, r * accWidth)
                      .named('p_r$r');

                  // Sign-extend both sides to accWidth+1 to avoid carry loss,
                  // then take the low accWidth bits of the result.
                  final prevAcc = accRegs[r].signExtend(accWidth + 1);
                  final partial = pSlice.signExtend(accWidth + 1);

                  final sumFull = Mux(dFirst, partial, prevAcc + partial).out;
                  return accRegs[r] < sumFull.slice(accWidth - 1, 0);
                }(),

              // resultValidReg: set on the delayed-last tile, clear implicitly
              // on the next cycle (default above). For a single-tile block
              // (first&last same cycle) the delayed signals also coincide, so
              // resultValidReg is set correctly.
              resultValidReg < dLast,
            ],
          ),
        ],
      ),
    ];

    Sequential(clkPort, seqActions);

    // Wire outputs.
    // acc output: concatenate accRegs, element 0 in low bits.
    // swizzle puts the last list element in the MSB. Reverse for row 0 low.
    acc <= accRegs.reversed.toList().swizzle();
    resultValid <= resultValidReg;
  }
}
