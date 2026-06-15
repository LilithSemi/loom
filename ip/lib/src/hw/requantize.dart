// Loom Requantize: integer-only PE-array accumulator -> saturated int8.
//
// Microarchitecture (combinational):
//
//   1. SIGNED PRODUCT
//      accExt = signExtend(acc, prodWidth)
//      multExt = zeroExtend(mult, prodWidth)
//      prod = (accExt * multExt)[prodWidth-1:0]   (low prodW bits)
//      prodWidth = accWidth + multWidth
//      ROHD * on two prodW-bit values returns prodW bits (lower half is
//      exact for signed*unsigned when acc is sign-extended).
//
//   2. SIGN-AWARE ROUND-HALF-AWAY-FROM-ZERO RIGHT SHIFT
//      sign    = prod[prodWidth-1]
//      mag     = sign==1 ? (-prod) : prod      (at magW = prodW+2 bits)
//                zero-extend prod to magW before negation; handles -2^(N-1)
//      bias    = (1 << shift) >> 1             = 0 if shift==0, 1<<(shift-1) else
//                computed as: left-shift 1 by shift, then logical-right-shift by 1
//      biased  = mag + bias                    (magW bits; +1 from ROHD add)
//      shifted = biased >>> shift              (logical, at biased.width bits)
//      magOut  = shifted[prodW-1:0]            (truncate to prodW)
//      negMag  = (-magOut) [prodW-1:0]
//      rounded = sign==1 ? negMag : magOut
//
//   3. SATURATION (symmetric, sign-bit-steered unsigned comparisons)
//      sign_r = rounded[prodW-1]               (sign of rounded result)
//      magR   = sign_r==1 ? (-rounded)[prodW-1:0] : rounded    (magnitude)
//      hi = 2^(outWidth-1)-1  e.g. 127
//
//      gtHi = (sign_r==0) AND (rounded > hi)   i.e. positive AND > hi
//           = ~sign_r AND rounded.gt(Const(hi))
//      ltLo = (sign_r==1) AND (magR > hi)      i.e. negative AND mag > hi
//           = sign_r AND magR.gt(Const(hi))
//
//      out = gtHi ? hi : (ltLo ? lo : rounded[outWidth-1:0])
//
// All signals named for readable SV.  Combinational only.

import 'package:rohd/rohd.dart';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Immutable configuration for [LoomRequantize].
class LoomRequantizeConfig {
  /// Bit width of the signed accumulator input (from PE array).
  final int accWidth;

  /// Bit width of the unsigned fixed-point multiplier.
  final int multWidth;

  /// Bit width of the unsigned shift amount.
  final int shiftWidth;

  /// Bit width of the signed saturated output.
  final int outWidth;

  /// Pipeline latency in clock cycles. When >= 1 a clk must be supplied.
  ///   0 => fully combinational.
  ///   1 => register the signed product, splitting the wide multiply away from
  ///        the round/saturate chain.
  ///   2 => additionally register the rounded value, splitting the round/shift
  ///        carry chain away from the saturation compares (the second-stage
  ///        path that limited Fmax after stage 1).
  final int latency;

  const LoomRequantizeConfig({
    this.accWidth = 32,
    this.multWidth = 16,
    this.shiftWidth = 6,
    this.outWidth = 8,
    this.latency = 0,
  });

  /// Width of the signed product bus.
  int get prodWidth => accWidth + multWidth;

  /// Symmetric saturation upper bound: 2^(outWidth-1) - 1 (e.g. 127).
  int get saturateHi => (1 << (outWidth - 1)) - 1;

  /// Symmetric saturation lower bound: -(2^(outWidth-1) - 1) (e.g. -127).
  int get saturateLo => -(1 << (outWidth - 1)) + 1;

  /// Validate at construction: widths must be positive and self-consistent.
  void validate() {
    if (accWidth <= 0) {
      throw ArgumentError(
        'LoomRequantizeConfig.accWidth must be > 0, got $accWidth',
      );
    }
    if (multWidth <= 0) {
      throw ArgumentError(
        'LoomRequantizeConfig.multWidth must be > 0, got $multWidth',
      );
    }
    if (shiftWidth <= 0) {
      throw ArgumentError(
        'LoomRequantizeConfig.shiftWidth must be > 0, got $shiftWidth',
      );
    }
    if (outWidth < 2) {
      throw ArgumentError(
        'LoomRequantizeConfig.outWidth must be >= 2 (signed), got $outWidth',
      );
    }
    if (multWidth > accWidth) {
      throw ArgumentError(
        'LoomRequantizeConfig.multWidth ($multWidth) must be <= accWidth '
        '($accWidth) to keep the multiplier scale reasonable',
      );
    }
    if (latency < 0 || latency > 2) {
      throw ArgumentError(
        'LoomRequantizeConfig.latency must be 0, 1, or 2, got $latency',
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Combinational integer requantization: signed int32 accumulator ->
/// saturated signed int8 output.
///
/// Operation:
///   prod    = acc * mult              (prodWidth = accWidth+multWidth, signed)
///   rounded = roundHalfAwayFromZero(prod, shift)
///   out     = saturate(rounded, [saturateLo, saturateHi])
///
/// See [LoomRequantizeConfig] for parameter definitions and [validate()].
class LoomRequantize extends Module {
  /// Signed accumulator input.
  Logic get acc => input('acc');

  /// Unsigned multiplier input.
  Logic get mult => input('mult');

  /// Unsigned shift amount.
  Logic get shift => input('shift');

  /// Saturated signed output.
  Logic get out => output('out');

  final LoomRequantizeConfig _cfg;

  Logic? _clk;
  Logic? _reset;

  /// Pipeline latency in clock cycles (0 => combinational).
  int get latency => _cfg.latency;

  LoomRequantize({
    required Logic acc,
    required Logic mult,
    required Logic shift,
    int accWidth = 32,
    int multWidth = 16,
    int shiftWidth = 6,
    int outWidth = 8,
    int latency = 0,
    Logic? clk,
    Logic? reset,
  }) : _cfg = LoomRequantizeConfig(
         accWidth: accWidth,
         multWidth: multWidth,
         shiftWidth: shiftWidth,
         outWidth: outWidth,
         latency: latency,
       ),
       super(name: 'LoomRequantize', definitionName: 'LoomRequantize') {
    _cfg.validate();

    if (latency >= 1) {
      if (clk == null) {
        throw ArgumentError(
          'LoomRequantize requires a clk when latency >= 1 (got $latency)',
        );
      }
      _clk = addInput('clk', clk);
      if (reset != null) {
        _reset = addInput('reset', reset);
      }
    }

    final accPort = addInput('acc', acc, width: accWidth);
    final multPort = addInput('mult', mult, width: multWidth);
    final shiftPort = addInput('shift', shift, width: shiftWidth);
    addOutput('out', width: outWidth);

    _buildDatapath(accPort, multPort, shiftPort);
  }

  // Helper: two's-complement negation of a Logic at the given width [w].
  //
  // To correctly negate a signed value, sign-extend it to [w] bits so that
  // the MSB is replicated and the NOT + 1 gives the proper magnitude.
  // For x.width < w: signExtend(w) replicates the sign bit upward, then
  //   ~x_ext + 1 gives the correct negation at [w] bits.
  // For x.width == w: same operation, no change.
  //
  // Returns a signal of width [w].
  static Logic _neg(Logic x, int w, String tag) {
    // Ensure x is represented at w bits with sign extension.
    final xe = x.width < w ? x.signExtend(w) : x;
    final xn = xe.named('${tag}_sx');
    // ~xn at w bits + 1: ROHD add gives w+1 bits; slice to w.
    return (~xn + Const(1, width: w)).slice(w - 1, 0).named('${tag}_neg');
  }

  /// Register [sig] through one pipeline stage on the requant clk.
  Logic _reg(Logic sig, String name) {
    final r = Logic(name: name, width: sig.width);
    if (_reset != null) {
      r <= flop(_clk!, sig, reset: _reset, resetValue: 0);
    } else {
      r <= flop(_clk!, sig);
    }
    return r;
  }

  void _buildDatapath(Logic accPort, Logic multPort, Logic shiftPort) {
    final cfg = _cfg;
    final prodW = cfg.prodWidth;

    // magW: extra bits for negation and bias addition without overflow.
    // +1 so negation of -2^(prodW-1) fits, +1 for bias add carry.
    final magW = prodW + 2;

    // ------------------------------------------------------------------
    // Step 1: signed product
    //
    // Sign-extend acc and zero-extend mult both to prodW bits.
    // ROHD Multiply keeps width = in0.width = prodW; the low prodW bits of
    // a full 2*prodW multiply equal the two's-complement product for
    // signed*unsigned when acc is sign-extended.
    // ------------------------------------------------------------------
    final accExt = accPort.signExtend(prodW).named('acc_ext');
    final multExt = multPort.zeroExtend(prodW).named('mult_ext');
    var prod = (accExt * multExt).named('prod');

    // Optional pipeline cut: register the wide signed product (and the shift,
    // so it stays aligned). This splits the expensive multiply away from the
    // round/saturate carry chains, which together were the SoC critical path.
    var shiftStage = shiftPort;
    if (cfg.latency >= 1) {
      prod = _reg(prod, 'prod_reg');
      shiftStage = _reg(shiftPort, 'shift_reg');
    }

    // ------------------------------------------------------------------
    // Step 2: sign-aware round-half-away-from-zero shift
    //
    // Work on magnitude, add bias, shift, reapply sign.
    // ------------------------------------------------------------------
    final signBit = prod[prodW - 1].named('sign');

    // Magnitude of prod at magW bits.
    // negProd = (-prod) at magW via _neg.
    final negProd = _neg(prod, magW, 'neg_prod');
    final prodZext = prod.zeroExtend(magW).named('prod_zext');
    final mag = Mux(signBit, negProd, prodZext).out.named('mag');

    // Bias = (1 << shift) >>> 1
    // = 0 for shift==0, 1<<(shift-1) otherwise.
    // Compute: left-shift 1 by shift, then logical-right-shift by 1.
    // Constant '1' needs enough bits to hold 1 << maxShift.
    final one = Const(1, width: magW);
    final shiftExt = shiftStage.zeroExtend(magW).named('shift_ext');
    final oneShifted = (one << shiftExt).named('one_shifted');
    // Logical-right-shift by 1: drop bit 0, prepend a 0.
    // oneShifted is magW bits; bias = {1'b0, oneShifted[magW-1:1]}.
    final bias = [
      Const(0, width: 1),
      oneShifted.slice(magW - 1, 1),
    ].swizzle().named('bias');

    // biasedMag is magW+1 bits (ROHD add adds 1 bit).
    final biasedMag = (mag + bias).named('biased_mag');
    final biasedW = biasedMag.width; // magW + 1

    // Logical right shift of biasedMag by shift.
    final shiftForB = shiftStage.zeroExtend(biasedW).named('shift_for_biased');
    final shiftedMag = (biasedMag >>> shiftForB).named('shifted_mag');

    // Truncate shifted magnitude to prodW bits.
    final magOut = shiftedMag.slice(prodW - 1, 0).named('mag_out');

    // Negate magOut for the sign-reapply (keeps prodW bits).
    final negMagOut = _neg(magOut, prodW, 'neg_mag_out');

    // Reapply sign: sign==1 -> -magnitude, sign==0 -> +magnitude.
    var rounded = Mux(signBit, negMagOut, magOut).out.named('rounded');

    // Optional stage-2 pipeline cut: register the rounded value, separating the
    // round/shift carry chain from the saturation compares.
    if (cfg.latency >= 2) {
      rounded = _reg(rounded, 'rounded_reg');
    }

    // ------------------------------------------------------------------
    // Step 3: saturation to [saturateLo, saturateHi]
    //
    // ROHD comparison is unsigned.  We steer using the sign bit of rounded:
    //
    //   sign_r = rounded[prodW-1]
    //   magR   = unsigned magnitude of rounded (prodW bits)
    //          = sign_r==1 ? (-rounded) : rounded
    //
    //   gtHi = ~sign_r AND rounded > hi        (both nonneg -> unsigned ok)
    //   ltLo = sign_r  AND magR > hi            (hi == -lo for symmetric)
    //
    //   lo is -(hi), so |lo| == hi for symmetric quant.
    // ------------------------------------------------------------------
    final hi = cfg.saturateHi;
    // lo == -hi for symmetric (saturateLo = -(2^(outWidth-1)-1) = -hi).
    // Assertion: cfg.saturateLo == -hi always holds for symmetric quant.

    final signR = rounded[prodW - 1].named('sign_r');
    final notSignR = (~signR).named('not_sign_r');

    // Magnitude of rounded.
    final negRounded = _neg(rounded, prodW, 'neg_rounded');
    final magR = Mux(signR, negRounded, rounded).out.named('mag_r');

    // hiConst at prodW bits (unsigned, positive, fits easily).
    final hiConst = Const(hi, width: prodW);

    // gtHi: rounded is positive AND exceeds hi.
    final posGtHi = rounded.gt(hiConst).named('pos_gt_hi');
    final gtHi = (notSignR & posGtHi).named('gt_hi');

    // ltLo: rounded is negative AND its magnitude exceeds hi (==-lo).
    final magGtHi = magR.gt(hiConst).named('mag_gt_hi');
    final ltLo = (signR & magGtHi).named('lt_lo');

    // Output constants at outWidth bits.
    // lo in two's-complement at outWidth bits.
    final loMask = (BigInt.one << cfg.outWidth) - BigInt.one;
    final loTwos = BigInt.from(cfg.saturateLo) & loMask;
    final hiOut = Const(hi, width: cfg.outWidth);
    final loOut = Const(loTwos, width: cfg.outWidth);
    final passOut = rounded.slice(cfg.outWidth - 1, 0).named('pass_out');

    // Clamp: gtHi -> hi, ltLo -> lo, else passthrough (already prodW->outWidth).
    final clamped = Mux(
      gtHi,
      hiOut,
      Mux(ltLo, loOut, passOut).out,
    ).out.named('clamped');

    out <= clamped;
  }
}
