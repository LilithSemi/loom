import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

/// LoomFpMac: streaming fp16 multiply-accumulate, `acc += a * b`.
///
/// Used for attention dot products (scores = q . k), the weighted-V sum
/// (out += weight * v), and other fp16 activation reductions the sequencer
/// needs (the W4A8 matmuls themselves stay integer, in LoomStreamMatmul).
///
/// One fp16 multiplier + one accumulating adder. Host/sequencer-paced: pulse
/// `clear` to zero the accumulator, then pulse `en` with each (a, b) pair to
/// fold it in. The running sum is on `acc` every cycle. Combinational core,
/// registered accumulator.
///
/// Ports: in clk, reset, clear, en, a[16], b[16]. Out acc[16] (packed fp16).
class LoomFpMac extends BridgeModule {
  LoomFpMac({String? name}) : super('LoomFpMac', name: name ?? 'loom_fp_mac') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('clear', PortDirection.input);
    createPort('en', PortDirection.input);
    createPort('a', PortDirection.input, width: 16);
    createPort('b', PortDirection.input, width: 16);
    final accP = addOutput('acc', width: 16);

    final clk = input('clk');
    final acc = Logic(name: 'acc_reg', width: 16);

    final prod = FloatingPointMultiplierSimple(
      FloatingPoint16()..gets(input('a')),
      FloatingPoint16()..gets(input('b')),
    ).product.packed;
    final sum = FloatingPointAdderSinglePath(
      FloatingPoint16()..gets(acc),
      FloatingPoint16()..gets(prod),
    ).sum.packed;

    Sequential(clk, [
      If(
        input('reset') | input('clear'),
        then: [acc < Const(0, width: 16)],
        orElse: [
          If(input('en'), then: [acc < sum]),
        ],
      ),
    ]);

    accP <= acc;
  }
}
