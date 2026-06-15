import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

/// LoomDequant: int32 accumulator -> fp16 activation, the output boundary of
/// every W4A8 linear. Matches the golden `dequant`:
///
///   y[r] = acc[r] * rowScale[r] * actScale
///
/// where `rowScale` is the per-row int4 weight scale (precomputed at
/// provisioning) and `actScale` is the per-tensor activation scale from
/// LoomActQuant.
///
/// A real matmul accumulator (sum of int4*int8 over hundreds-thousands of cols)
/// reaches ~10^6, far past fp16's 65504 ceiling, so the int->float conversion
/// and both scale multiplies run in fp32 and only the final result narrows to
/// fp16 (the product is small once scaled).
///
/// PIPELINED: the fp32 int-to-float + two multiplies + narrow form a ~100 ns
/// combinational chain (it was THE critical path, capping the SoC at ~9 MHz), so
/// it is registered into [latency] stages: a flop after the conversions, after
/// each multiply, and after the narrow. Latency is 4 cycles; `valid_in` is
/// pipelined to `valid_out`. The consumer (LoomFpLinear) streams rows in and
/// collects them [latency] cycles later, in order.
///
/// Ports: in clk, reset, valid_in, acc[32] (signed int32), row_scale[16] (fp16),
/// act_scale[16] (fp16). Out y[16] (fp16), y_acc[32] (fp32, the pre-narrow
/// product aligned to the same valid_out cycle as y), valid_out.
class LoomDequant extends BridgeModule {
  /// Pipeline depth (cycles from valid_in to valid_out). With the multiplies
  /// clocked (coreLatency=1 each), the data path is: stage0 reg, mult1 internal
  /// flop, p1 reg, mult2 internal flop, p2 reg, y reg = 6 flops.
  static const int latency = 6;

  LoomDequant({int coreLatency = 1, String? name})
    : super('LoomDequant', name: name ?? 'loom_dequant') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('valid_in', PortDirection.input);
    createPort('acc', PortDirection.input, width: 32);
    createPort('row_scale', PortDirection.input, width: 16);
    createPort('act_scale', PortDirection.input, width: 16);
    final yP = addOutput('y', width: 16);
    final yAccP = addOutput('y_acc', width: 32);
    final validOutP = addOutput('valid_out');

    final clk = input('clk');
    final reset = input('reset');

    Logic reg(String name, Logic d, int w) {
      final r = Logic(name: name, width: w);
      r <= flop(clk, d, reset: reset, resetValue: 0);
      return r;
    }

    // FixedToFloat needs a nonzero fraction width, so use a 1-bit fraction tied
    // to zero with the 32-bit acc in the integer field (value unchanged).
    final accFx = FixedPoint(integerWidth: 31, fractionWidth: 1, signed: true)
      ..gets([input('acc'), Const(0, width: 1)].swizzle());
    final accF = FloatingPoint32();
    FixedToFloat(accFx, accF);
    final rs32 = FloatingPoint32();
    FloatingPointConverter(FloatingPoint16()..gets(input('row_scale')), rs32);
    final as32 = FloatingPoint32();
    FloatingPointConverter(FloatingPoint16()..gets(input('act_scale')), as32);

    final accFr = reg('acc_f32', accF.packed, 32);
    final rsR = reg('rs_f32', rs32.packed, 32);
    final asR = reg('as_f32', as32.packed, 32);

    // FloatingPointMultiplierSimple is clocked, so its product lags the
    // inputs by coreLatency. Register the product (p1r).
    final p1 = FloatingPointMultiplierSimple(
      FloatingPoint32()..gets(accFr),
      FloatingPoint32()..gets(rsR),
      clk: clk,
    ).product;
    final p1r = reg('p1_f32', p1.packed, 32);
    // Carry actScale to align with p1r: stage0 reg (asR) + coreLatency (mult1
    // internal flop) + p1r reg = 1 + coreLatency more delays after asR.
    var asAligned = asR;
    for (var i = 0; i < coreLatency + 1; i++) {
      asAligned = reg('as_f32_d$i', asAligned, 32);
    }

    final p2 = FloatingPointMultiplierSimple(
      FloatingPoint32()..gets(p1r),
      FloatingPoint32()..gets(asAligned),
      clk: clk,
    ).product;
    final p2r = reg('p2_f32', p2.packed, 32);

    final y16 = FloatingPoint16();
    FloatingPointConverter(FloatingPoint32()..gets(p2r), y16);
    yP <= reg('y_f16', y16.packed, 16);
    // y_acc: the fp32 pre-narrow product (p2r), delayed by one reg to align
    // with y (which is p2r narrowed, then registered). Same cycle as y at
    // valid_out, higher precision, for bit-exact cross-col-block accumulation.
    yAccP <= reg('y_acc_f32', p2r, 32);

    // Valid pipeline: match the data latency = stage0(1) + mult1(coreLatency) +
    // p1r(1) + mult2(coreLatency) + p2r(1) + y(1) = 4 + 2*coreLatency.
    final lat = 4 + 2 * coreLatency;
    var v = input('valid_in');
    for (var i = 0; i < lat; i++) {
      v = flop(clk, v, reset: reset, resetValue: 0);
    }
    validOutP <= v;
  }
}
