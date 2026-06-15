// Loom Vector Unit: combinational elementwise signed-integer ALU.
//
// Computes per-lane: ADD, SUB, or MUL on signed inWidth-bit inputs,
// producing signed accWidth-bit outputs. No saturation -- accWidth is
// required to be >= 2*inWidth so a full signed product always fits.
// Downstream requantize handles narrowing.
//
// Intended uses:
//   ADD (op=0): residual connection (h + sublayer_out)
//   SUB (op=1): elementwise subtract
//   MUL (op=2): SwiGLU gate fuse (silu(gate) * up -- after upstream silu)
//   op=3: reserved; behaves as ADD (documented, not a hardware guarantee)
//
// Port layout (little-element-endian, element 0 in low bits):
//   a  : width = lanes*inWidth    a[i] at bit i*inWidth (signed)
//   b  : width = lanes*inWidth    b[i] at bit i*inWidth (signed)
//   op : width = 2                operation select (0=ADD,1=SUB,2=MUL)
//   y  : width = lanes*accWidth   y[i] at bit i*accWidth (signed)
//
// Microarchitecture:
//   Per lane i:
//     aExt = signExtend(a[i], accWidth)
//     bExt = signExtend(b[i], accWidth)
//     addResult = (aExt + bExt)[accWidth-1:0]
//     subResult = (aExt - bExt)[accWidth-1:0]
//     NativeMultiplier(a[i], b[i], signed, signed) -> product (2*inWidth bits)
//     mulResult = signExtend(product, accWidth)[accWidth-1:0]
//     y[i] = Mux(op[1], mulResult, Mux(op[0], subResult, addResult))
//
// op-select note: op is a 2-bit value. We use a two-level Mux:
//   outer Mux on op[1]: selects mul (op[1]=1) vs add/sub (op[1]=0)
//   inner Mux on op[0]: selects sub (op[0]=1) vs add (op[0]=0)
// This covers op 0,1,2 correctly. op=3 (op[1]=1, op[0]=1) falls into the
// outer mul arm so it returns mul; that is acceptable for reserved.
//
// All signals are named for readable SV emission. Combinational only.

import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

// ---------------------------------------------------------------------------
// Op constants
// ---------------------------------------------------------------------------

/// Elementwise operation: add a+b.
const int vecOpAdd = 0;

/// Elementwise operation: subtract a-b.
const int vecOpSub = 1;

/// Elementwise operation: multiply a*b (SwiGLU style).
const int vecOpMul = 2;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Immutable configuration for [LoomVectorUnit].
class LoomVectorUnitConfig {
  /// Number of parallel lanes (elements processed simultaneously).
  final int lanes;

  /// Bit width of each signed input element (a[i] and b[i]).
  final int inWidth;

  /// Bit width of each signed output element (y[i]).
  /// Must be >= 2*inWidth so a full signed product fits without overflow.
  final int accWidth;

  const LoomVectorUnitConfig({
    required this.lanes,
    this.inWidth = 16,
    this.accWidth = 32,
  });

  /// Validate configuration. Throws [ArgumentError] on any violation.
  void validate() {
    if (lanes <= 0) {
      throw ArgumentError('LoomVectorUnitConfig.lanes must be > 0, got $lanes');
    }
    if (inWidth < 2) {
      throw ArgumentError(
        'LoomVectorUnitConfig.inWidth must be >= 2 (signed), got $inWidth',
      );
    }
    if (accWidth < 2 * inWidth) {
      throw ArgumentError(
        'LoomVectorUnitConfig.accWidth ($accWidth) must be >= 2*inWidth '
        '(${2 * inWidth}) to hold a full signed product without overflow',
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Combinational elementwise signed-integer vector ALU.
///
/// Computes per-lane ADD, SUB, or MUL on [lanes] signed [inWidth]-bit inputs,
/// producing [lanes] signed [accWidth]-bit outputs.
///
/// See [LoomVectorUnitConfig] for parameter constraints.
class LoomVectorUnit extends Module {
  /// Signed input bus A: element i at bit i*inWidth.
  Logic get a => input('a');

  /// Signed input bus B: element i at bit i*inWidth.
  Logic get b => input('b');

  /// Operation select: 0=ADD, 1=SUB, 2=MUL, 3=reserved (acts as MUL).
  Logic get op => input('op');

  /// Signed output bus: element i at bit i*accWidth.
  Logic get y => output('y');

  final int _inWidth;
  final int _accWidth;

  /// Construct [LoomVectorUnit].
  ///
  /// [a] and [b] are the input element buses (lanes*inWidth bits each).
  /// [op] is a 2-bit operation select.
  /// [inWidth] defaults to 16, [accWidth] defaults to 32.
  ///
  /// Calls [LoomVectorUnitConfig.validate()] at construction.
  LoomVectorUnit({
    required Logic a,
    required Logic b,
    required Logic op,
    required int lanes,
    int inWidth = 16,
    int accWidth = 32,
  }) : _inWidth = inWidth,
       _accWidth = accWidth,
       super(name: 'LoomVectorUnit', definitionName: 'LoomVectorUnit') {
    // Validate before touching ROHD infrastructure.
    LoomVectorUnitConfig(
      lanes: lanes,
      inWidth: inWidth,
      accWidth: accWidth,
    ).validate();

    // Register ports.
    final aPort = addInput('a', a, width: lanes * inWidth);
    final bPort = addInput('b', b, width: lanes * inWidth);
    final opPort = addInput('op', op, width: 2);
    addOutput('y', width: lanes * accWidth);

    // Build one result per lane then concatenate (element 0 in low bits).
    final laneResults = <Logic>[];
    for (var i = 0; i < lanes; i++) {
      laneResults.add(_buildLane(i, aPort, bPort, opPort));
    }

    // swizzle puts the last element in the MSB, so reverse to put lane 0
    // in the low bits.
    y <= laneResults.reversed.toList().swizzle();
  }

  /// Build the combinational datapath for lane [i].
  Logic _buildLane(int i, Logic aPort, Logic bPort, Logic opPort) {
    final lo = i * _inWidth;
    final hi = lo + _inWidth - 1;

    // Extract signed inWidth-bit elements.
    final aElem = aPort.slice(hi, lo).named('a_$i');
    final bElem = bPort.slice(hi, lo).named('b_$i');

    // Sign-extend operands to accWidth for add/sub.
    final aExt = aElem.signExtend(_accWidth).named('a_ext_$i');
    final bExt = bElem.signExtend(_accWidth).named('b_ext_$i');

    // ADD: (aExt + bExt), keep low accWidth bits.
    // ROHD add returns width+1 bits; slice to accWidth.
    final addFull = (aExt + bExt).named('add_full_$i');
    final addResult = addFull.slice(_accWidth - 1, 0).named('add_$i');

    // SUB: (aExt - bExt), keep low accWidth bits.
    // ROHD subtraction: aExt - bExt. Both are accWidth bits sign-extended
    // from inWidth, so the difference fits in accWidth (inWidth >= 2 implies
    // accWidth >= 4; worst case for sub is -2^(inWidth-1) - (2^(inWidth-1)-1)
    // which fits in (inWidth+1) bits, well within accWidth >= 2*inWidth).
    final subFull = (aExt - bExt).named('sub_full_$i');
    final subResult = subFull.slice(_accWidth - 1, 0).named('sub_$i');

    // MUL: signed * signed using NativeMultiplier.
    // Product is 2*inWidth bits wide (both operands signed).
    // Sign-extend to accWidth (accWidth >= 2*inWidth by validate()).
    final mul = NativeMultiplier(
      aElem,
      bElem,
      signedMultiplicand: true,
      signedMultiplier: true,
      name: 'mul_$i',
    );
    final mulProduct = mul.product.named('mul_prod_$i');
    // mulProduct is 2*inWidth bits; sign-extend to accWidth, then slice.
    final mulExt = mulProduct.signExtend(_accWidth).named('mul_ext_$i');
    final mulResult = mulExt.slice(_accWidth - 1, 0).named('mul_$i');

    // op-select: two-level Mux.
    //   op[1]=0 -> add or sub (op[0] picks sub=1, add=0)
    //   op[1]=1 -> mul
    final op1 = opPort[1].named('op1_$i');
    final op0 = opPort[0].named('op0_$i');

    final addSubSel = Mux(op0, subResult, addResult).out.named('addsub_$i');
    final result = Mux(op1, mulResult, addSubSel).out.named('result_$i');

    return result;
  }
}
