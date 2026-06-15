import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

/// LoomRope: rotary position embedding (RoPE) on a single half-split coordinate
/// pair, matching the golden `applyRopeHead` (HuggingFace Llama convention):
///
///   y1 = x1 * c - x2 * s
///   y2 = x2 * c + x1 * s
///
/// with `c = cos(angle)`, `s = sin(angle)`, `angle = pos / theta^(2j/headDim)`.
///
/// The cos/sin cache is precomputed once (per position, per dim pair) and fed in
/// as `cos_in`/`sin_in`, mirroring how HF and llama.cpp build a rotary cache.
/// That keeps angle range-reduction (pos*invFreq reaches thousands, far past any
/// LUT window) out of the datapath, so this unit is just the rotation: four fp16
/// multiplies and two adds. The subtraction reuses the adder by flipping the
/// sign bit of the `x2*s` product. Combinational.
///
/// Ports: in x1[16], x2[16], cos_in[16], sin_in[16]. Out y1[16], y2[16].
/// All are packed IEEE fp16 (1 sign, 5 exp, 10 mantissa).
class LoomRope extends BridgeModule {
  LoomRope({String? name}) : super('LoomRope', name: name ?? 'loom_rope') {
    createPort('x1', PortDirection.input, width: 16);
    createPort('x2', PortDirection.input, width: 16);
    createPort('cos_in', PortDirection.input, width: 16);
    createPort('sin_in', PortDirection.input, width: 16);
    final y1P = addOutput('y1', width: 16);
    final y2P = addOutput('y2', width: 16);

    final x1 = input('x1');
    final x2 = input('x2');
    final cos = input('cos_in');
    final sin = input('sin_in');

    Logic mul(Logic a, Logic b) => FloatingPointMultiplierSimple(
      FloatingPoint16()..gets(a),
      FloatingPoint16()..gets(b),
    ).product.packed;

    Logic add(Logic a, Logic b) => FloatingPointAdderSinglePath(
      FloatingPoint16()..gets(a),
      FloatingPoint16()..gets(b),
    ).sum.packed;

    // Flip the fp16 sign bit (negation) to turn `+` into `-`.
    Logic neg(Logic v) => [~v[15], v.getRange(0, 15)].swizzle();

    final x1c = mul(x1, cos);
    final x2s = mul(x2, sin);
    final x2c = mul(x2, cos);
    final x1s = mul(x1, sin);

    y1P <= add(x1c, neg(x2s)); // x1*c - x2*s
    y2P <= add(x2c, x1s); // x2*c + x1*s
  }
}
