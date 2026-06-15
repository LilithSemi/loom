// Loom PE Array: pipelined signed-integer matrix-vector multiply engine.
//
// Computes y[r] = sum_c W[r,c] * x[c] in two's-complement signed arithmetic,
// where W is inWidth-bit signed and x is inWidth-bit signed, and y is
// accWidth-bit signed.  This mirrors the golden linear() semantics exactly:
// W row-major [out=rows, in=cols].
//
// Microarchitecture: PIPELINED.  Each row gets a bank of cols multipliers
// (NativeMultiplier, both operands signed) whose products are summed with a
// sign-extended adder tree.  The combinational multiply -> full adder tree was
// the SoC critical path (~40 ns => ~22 MHz on the OrangeCrab ECP5), because the
// whole multiply carry chain AND every adder-tree level sat between two flops.
//
// To close 48 MHz we insert pipeline registers:
//   - Stage 1 registers the (sign-extended) multiplier OUTPUTS.  This cuts the
//     longest combinational segment to {operands -> multiply}, breaking the
//     multiply away from the adder tree.
//   - Stages 2..N register successive ADDER-TREE levels (one register per tree
//     level until we run out of levels).  Each remaining segment is then a
//     single add, which is short.
// The total number of pipeline register stages from inputs to `y` is the
// [latency] parameter (in clock cycles).  Because the datapath is fully
// pipelined and feed-forward (no feedback), the value on `y` at cycle T+latency
// equals the combinational result of the operands presented at cycle T.
//
// latency semantics:
//   latency == 0 : purely combinational (no clk needed).
//   latency >= 1 : stage 1 registers the products; the remaining (latency-1)
//                  stages register the first (latency-1) adder-tree levels.
//                  Extra requested stages beyond the available tree depth are
//                  realized as plain pass-through pipeline registers so the
//                  declared latency always holds (and Fmax does not regress).
//
// When latency >= 1 a clk (and reset) input is required.
//
// Port layout (little-element-endian, element 0 in low bits):
//   w  : width = rows*cols*inWidth   W[r,c] at bit (r*cols+c)*inWidth
//   x  : width = cols*inWidth        x[c] at bit c*inWidth
//   y  : width = rows*accWidth       y[r] at bit r*accWidth
//   clk, reset : present only when latency >= 1
//
// The product of two inWidth-bit signed values is 2*inWidth bits wide.
// Summing cols such products without overflow requires
//   accWidth >= 2*inWidth + ceil(log2(cols)).
// The config validates this lower bound.

import 'dart:math' as math;

import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

// Configuration

/// Immutable configuration for [LoomPeArray].
class LoomPeArrayConfig {
  /// Number of output rows (= output dimension of the linear layer).
  final int rows;

  /// Number of input columns (= input dimension of the linear layer).
  final int cols;

  /// Bit width of each signed integer operand (W and x elements).
  final int inWidth;

  /// Bit width of the signed accumulator output per row.
  final int accWidth;

  /// Number of pipeline register stages from inputs to `y` (clock cycles of
  /// latency). 0 => combinational. See file header for the staging rules.
  final int latency;

  /// Multiply-free BitNet path: each W element is ternary {-1,0,+1}, so the
  /// per-element product is a select/negate of x (no NativeMultiplier, freeing
  /// the ECP5's scarce DSPs). Numerically identical to the multiply path for
  /// ternary weights. A wider weight silently truncates, so only set this when
  /// the weights are genuinely ternary (genip `quant: bitnet_ternary`).
  final bool ternaryWeights;

  const LoomPeArrayConfig({
    required this.rows,
    required this.cols,
    this.inWidth = 8,
    this.accWidth = 32,
    this.latency = 0,
    this.ternaryWeights = false,
  });

  /// Number of balanced adder-tree levels needed to reduce [cols] products.
  /// 0 when cols <= 1 (no addition needed).
  int get treeLevels {
    if (cols <= 1) return 0;
    return (math.log(cols) / math.ln2).ceil();
  }

  /// Minimum accumulator bits to hold the worst-case signed sum without
  /// overflow: one product is 2*inWidth bits signed. Summing [cols] of them
  /// needs ceil(log2(cols)) more bits.
  int get minAccWidth {
    if (cols <= 1) return 2 * inWidth;
    return 2 * inWidth + treeLevels;
  }

  /// Validate configuration at construction time.
  void validate() {
    if (rows <= 0) {
      throw ArgumentError('LoomPeArrayConfig.rows must be > 0, got $rows');
    }
    if (cols <= 0) {
      throw ArgumentError('LoomPeArrayConfig.cols must be > 0, got $cols');
    }
    if (inWidth < 2) {
      throw ArgumentError(
        'LoomPeArrayConfig.inWidth must be >= 2 (signed), got $inWidth',
      );
    }
    if (accWidth < minAccWidth) {
      throw ArgumentError(
        'LoomPeArrayConfig.accWidth ($accWidth) is too small for rows=$rows '
        'cols=$cols inWidth=$inWidth; minimum is $minAccWidth',
      );
    }
    if (latency < 0) {
      throw ArgumentError(
        'LoomPeArrayConfig.latency must be >= 0, got $latency',
      );
    }
  }
}

// Module

/// Pipelined signed integer matrix-vector multiply PE array.
///
/// Computes y = W * x where W is [rows]x[cols] and x is length [cols],
/// all in two's-complement signed [inWidth]-bit arithmetic.  Each output
/// y[r] is a signed [accWidth]-bit value, available [latency] clock cycles
/// after the operands are presented (or combinationally when latency == 0).
///
/// See [LoomPeArrayConfig] for parameter constraints and [validate()].
class LoomPeArray extends Module {
  /// The weight matrix bus (W row-major, element 0 in low bits).
  Logic get w => input('w');

  /// The input vector bus (element 0 in low bits).
  Logic get x => input('x');

  /// The output vector bus: y[r] at bit offset r*accWidth (signed).
  Logic get y => output('y');

  final int _cols;
  final int _inWidth;
  final int _accWidth;
  final int _latency;
  final bool _ternaryWeights;

  /// Pipeline latency in clock cycles from inputs to `y` (0 => combinational).
  int get latency => _latency;

  Logic? _clk;
  Logic? _reset;

  /// Construct [LoomPeArray] from named inputs [w] and [x] plus dimensions.
  ///
  /// [rows] and [cols] define the weight matrix shape.
  /// [inWidth] is the signed bit-width of each W and x element (default 8).
  /// [accWidth] is the signed bit-width of each y element (default 32).
  /// [latency] is the number of pipeline register stages (default 0 =
  /// combinational). When [latency] >= 1, [clk] (and [reset]) must be provided.
  ///
  /// Calls [LoomPeArrayConfig.validate()] at construction. Throws
  /// [ArgumentError] if parameters are invalid.
  LoomPeArray({
    required Logic w,
    required Logic x,
    required int rows,
    required int cols,
    int inWidth = 8,
    int accWidth = 32,
    int latency = 0,
    bool ternaryWeights = false,
    Logic? clk,
    Logic? reset,
  }) : _cols = cols,
       _inWidth = inWidth,
       _accWidth = accWidth,
       _latency = latency,
       _ternaryWeights = ternaryWeights,
       super(name: 'LoomPeArray', definitionName: 'LoomPeArray') {
    // Validate before touching ROHD infrastructure.
    LoomPeArrayConfig(
      rows: rows,
      cols: cols,
      inWidth: inWidth,
      accWidth: accWidth,
      latency: latency,
      ternaryWeights: ternaryWeights,
    ).validate();

    if (latency >= 1) {
      if (clk == null) {
        throw ArgumentError(
          'LoomPeArray requires a clk when latency >= 1 (got latency=$latency)',
        );
      }
      _clk = addInput('clk', clk);
      // reset is optional. Pipeline registers do not strictly need a reset
      // (feed-forward data path), but we honor one if given for clean sim.
      if (reset != null) {
        _reset = addInput('reset', reset);
      }
    }

    // Register ports.
    final wPort = addInput('w', w, width: rows * cols * inWidth);
    final xPort = addInput('x', x, width: cols * inWidth);
    addOutput('y', width: rows * accWidth);

    // Build datapath: one row at a time.
    final rowResults = <Logic>[];
    for (var r = 0; r < rows; r++) {
      rowResults.add(_buildRow(r, wPort, xPort));
    }

    // Concatenate row results into the output bus: row 0 in the low bits.
    // We swizzle: [rowResults.reversed] puts row 0 in the LSB position.
    y <= rowResults.reversed.toList().swizzle();
  }

  /// Register [sig] through one pipeline stage (FlipFlop on the PE clk).
  /// Only valid when latency >= 1. Returns the registered signal.
  Logic _pipe(Logic sig, String name) {
    final reg = Logic(name: name, width: sig.width);
    if (_reset != null) {
      reg <= flop(_clk!, sig, reset: _reset, resetValue: 0);
    } else {
      reg <= flop(_clk!, sig);
    }
    return reg;
  }

  /// Build the dot-product datapath for row [r].
  ///
  /// Extracts W[r,c] and x[c] for each c, sign-extends the products to
  /// [_accWidth] bits, then reduces with an adder tree.  Pipeline registers are
  /// inserted per the [_latency] policy (see file header).
  Logic _buildRow(int r, Logic wPort, Logic xPort) {
    final products = <Logic>[];
    for (var c = 0; c < _cols; c++) {
      // Extract the inWidth-bit slice for W[r,c] and x[c].
      final wOffset = (r * _cols + c) * _inWidth;
      final xOffset = c * _inWidth;
      final wElem = wPort
          .slice(wOffset + _inWidth - 1, wOffset)
          .named('w_r${r}_c$c');
      final xElem = xPort.slice(xOffset + _inWidth - 1, xOffset).named('x_c$c');

      if (_ternaryWeights) {
        // Multiply-free: w in {-1,0,+1} (int8 0x01/0x00/0xFF), so prod = +x when
        // w==+1, -x when w==-1, 0 when w==0. Just a select + two's-complement
        // negate, no DSP. Widened to 2*inWidth to match the multiply product so
        // the downstream sign-extend + adder tree are byte-identical.
        final pw = 2 * _inWidth;
        final xExt = xElem.signExtend(pw);
        // Two's-complement negate at pw bits. ROHD '+' does NOT widen: the sum
        // keeps the operand width and the carry is a separate, dropped output
        // (the adder tree below signExtends by 1 before adding for that reason).
        // pw = 2*inWidth has full headroom to negate an inWidth-bit value (incl.
        // -2^(inWidth-1)), so the product is exactly 2*inWidth like the
        // NativeMultiplier product, with no slice needed.
        final negX = (~xExt) + Const(1, width: pw);
        final wNeg =
            wElem[_inWidth - 1]; // sign bit: 1 for -1, 0 for 0/+1 (any inWidth)
        final wZero = wElem.eq(Const(0, width: _inWidth));
        final signedX = mux(wNeg, negX, xExt);
        final prod = mux(wZero, Const(0, width: pw), signedX);
        products.add(prod.named('prod_r${r}_c$c'));
      } else {
        // NativeMultiplier with both signed: internally sign-extends each
        // operand to 2*inWidth bits and multiplies.
        final mul = NativeMultiplier(
          wElem,
          xElem,
          signedMultiplicand: true,
          signedMultiplier: true,
          name: 'mul_r${r}_c$c',
        );
        // product is productWidth (2*inWidth) bits.
        products.add(mul.product.named('prod_r${r}_c$c'));
      }
    }

    // Sign-extend each product to accWidth before adding so the tree width
    // stays consistent and carries propagate correctly.
    var extended = [
      for (var i = 0; i < products.length; i++)
        products[i].signExtend(_accWidth).named('ext_r${r}_c$i'),
    ];

    // Track how many pipeline register stages remain to insert.
    var stagesLeft = _latency;

    // Stage 1: register the (sign-extended) products. This is the key cut: it
    // separates the multiply carry chains from the adder tree.
    if (stagesLeft >= 1) {
      extended = [
        for (var i = 0; i < extended.length; i++)
          _pipe(extended[i], 'preg_r${r}_c$i'),
      ];
      stagesLeft -= 1;
    }

    Logic sum;
    if (extended.length == 1) {
      // No adder tree. Burn any remaining requested stages as pass-through
      // pipeline registers so the declared latency holds exactly.
      sum = extended[0];
      for (var s = 0; s < stagesLeft; s++) {
        sum = _pipe(sum, 'pass_r${r}_s$s');
      }
    } else {
      // Build a balanced adder tree, registering the first `stagesLeft` levels.
      sum = _adderTree(extended, r, stagesLeft);
    }

    // Slice down to accWidth bits (drop carry-out if tree gave wider result).
    return sum.slice(_accWidth - 1, 0).named('y_r$r');
  }

  /// Reduce a list of same-width [Logic] values with a balanced adder tree.
  ///
  /// The first [regLevels] tree levels are followed by a pipeline register. The
  /// remaining levels are combinational. After the tree is exhausted, any
  /// still-unspent register stages are inserted as pass-through registers so the
  /// overall latency matches the request exactly.
  Logic _adderTree(List<Logic> inputs, int rowIdx, int regLevels) {
    var level = inputs;
    var nodeIdx = 0;
    var levelIdx = 0;
    var regLeft = regLevels;
    while (level.length > 1) {
      final next = <Logic>[];
      for (var i = 0; i < level.length; i += 2) {
        if (i + 1 < level.length) {
          // Sign-extend both operands by 1 bit before adding to preserve
          // the sign of the sum.
          final a = level[i].signExtend(level[i].width + 1);
          final b = level[i + 1].signExtend(level[i + 1].width + 1);
          final s = (a + b).named('tree_r${rowIdx}_n$nodeIdx');
          nodeIdx++;
          next.add(s);
        } else {
          next.add(level[i]);
        }
      }
      // Register this level if we still have register stages to spend.
      if (regLeft > 0) {
        level = [
          for (var i = 0; i < next.length; i++)
            _pipe(next[i], 'treg_r${rowIdx}_l${levelIdx}_$i'),
        ];
        regLeft -= 1;
      } else {
        level = next;
      }
      levelIdx++;
    }

    // Spend any leftover register stages as pass-through registers so the
    // declared latency is honored even when the tree was shallower than
    // latency-1.
    var result = level[0];
    for (var s = 0; s < regLeft; s++) {
      result = _pipe(result, 'tpass_r${rowIdx}_s$s');
    }
    return result;
  }
}
