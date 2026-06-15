import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

/// LoomFpResidual: elementwise fp16 add of two activations, `out = a + b`.
///
/// Implements the transformer's residual connections (hidden += attn_out,
/// hidden += mlp_out). Plain packed-fp16 (16-bit) ports for SoC integration,
/// with a rohd_hcl FloatingPoint core inside. Combinational.
///
/// Ports: in a[16], b[16]. Out sum[16]. All are packed IEEE fp16
/// (1 sign, 5 exp, 10 mantissa).
class LoomFpResidual extends BridgeModule {
  LoomFpResidual({String? name})
    : super('LoomFpResidual', name: name ?? 'loom_fp_residual') {
    createPort('a', PortDirection.input, width: 16);
    createPort('b', PortDirection.input, width: 16);
    final sum = addOutput('sum', width: 16);

    final fa = FloatingPoint16()..gets(input('a'));
    final fb = FloatingPoint16()..gets(input('b'));
    final adder = FloatingPointAdderSinglePath(fa, fb);
    sum <= adder.sum.packed;
  }
}
