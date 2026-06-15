import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

import 'fp_recip.dart';

/// LoomActQuant: dynamic per-tensor symmetric int8 quantization of an fp16
/// activation vector, the input boundary of every W4A8 linear.
///
/// Golden (quant.dart quantizePerTensorInt8): `scale = maxAbs/127`,
/// `q[i] = clamp(round(x[i]/scale), -127, 127)`, dequant `y = q * scale`.
///
/// The hardware has no FP divider, so it multiplies by a reciprocal. A naive
/// `invScale = recip(maxAbs)*127` leaves the recip's few-percent error in the
/// quant scale, which would compound across layers. Instead the dequant scale
/// is the LITERAL reciprocal of the quant multiplier: `q = round(x*invScale)`,
/// `scale_out = recip(invScale)`. The recip error then cancels in the
/// round-trip (`q*scale_out ~= x`) regardless of recip accuracy, so `invScale`
/// only has to be roughly `127/maxAbs` to fill the int8 range. Two (shared)
/// reciprocal passes.
///
/// Host-paced: MAX pass streams `x_en`+`x_in`. Pulse `compute`. The unit runs
/// two recips. `ready` asserts with `scale_out` (the fp16 dequant scale, feed
/// it to LoomDequant alongside the weight row scale). Then the QUANT pass
/// re-streams `x_en`+`x_in` and emits `q_out` (int8) with `q_valid`.
///
/// Ports: in clk, reset, x_en, x_in[16], compute. Out q_out[8], q_valid,
/// scale_out[16], ready.
class LoomActQuant extends BridgeModule {
  static const int _max = 0;
  static const int _r1start = 1;
  static const int _r1wait = 2;
  static const int _r1mul = 6; // wait for the clocked invScale multiply
  static const int _r2start = 3;
  static const int _r2wait = 4;
  static const int _ready = 5;

  LoomActQuant({
    int recipIterations = 4,
    int coreLatency = 1,
    Logic? scaleOverride,
    String? name,
  }) : super('LoomActQuant', name: name ?? 'loom_act_quant') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('x_en', PortDirection.input);
    createPort('x_in', PortDirection.input, width: 16);
    createPort('compute', PortDirection.input);
    // Optional shared-scale override, mirrors the LoomFpLinear hostActScale
    // pattern: a real `scale_override` port is only created when a caller
    // passes a signal (col-tile linears sharing one act-scale across blocks).
    // Callers that pass nothing get no port, and the override branch below is
    // Dart-level dead code for them.
    Logic? scaleOverrideIn;
    if (scaleOverride != null) {
      createPort('scale_override', PortDirection.input, width: 16);
      input('scale_override').srcConnection! <= scaleOverride;
      scaleOverrideIn = input('scale_override');
    }
    final hasOverride = scaleOverrideIn != null;
    final qOutP = addOutput('q_out', width: 8);
    final qValidP = addOutput('q_valid');
    final scaleOutP = addOutput('scale_out', width: 16);
    final readyP = addOutput('ready');

    final clk = input('clk');
    final reset = input('reset');
    final xIn = input('x_in');

    final state = Logic(name: 'state', width: 3);
    final maxAbs = Logic(name: 'max_abs', width: 16);
    final invScale = Logic(name: 'inv_scale', width: 16);
    final deqScale = Logic(name: 'deq_scale', width: 16);
    final iwW = (coreLatency + 1).bitLength.clamp(1, 8);
    final iwait = Logic(name: 'iwait', width: iwW); // settle the clocked mults
    // Only built when hasOverride. Every use below is Dart-guarded by
    // `hasOverride`, so a caller that never passes scaleOverride gets none of
    // this constructed at all.
    final ovrActive = hasOverride ? Logic(name: 'ovr_active') : null;
    final ovrScale = hasOverride ? Logic(name: 'ovr_scale', width: 16) : null;

    Logic st(int s) => state.eq(Const(s, width: 3));

    // |x| for fp16 = clear the sign bit. Positive fp16 is monotonic in its bit
    // pattern, so the running max is a plain unsigned compare.
    final absX = xIn & Const(0x7FFF, width: 16);
    final absBigger = absX.gt(maxAbs);

    // Shared reciprocal: pass 1 = recip(maxAbs), pass 2 = recip(invScale).
    final recip = LoomFpRecip(iterations: recipIterations);
    recip.input('clk').srcConnection! <= clk;
    recip.input('reset').srcConnection! <= reset;
    // Pass-1 source: the shared override (when active) instead of the local
    // max-abs, so a shared act-scale quantizes on the same grid as everyone
    // else using it. When hasOverride is false this is just `maxAbs`.
    final r1Source = hasOverride ? mux(ovrActive!, ovrScale!, maxAbs) : maxAbs;
    recip.input('d').srcConnection! <=
        mux(st(_r1wait) | st(_r1start), r1Source, invScale);
    recip.input('start').srcConnection! <= st(_r1start) | st(_r2start);
    final recipOut = recip.output('out');
    final recipDone = recip.output('done');

    // invScale ~= recip(maxAbs) * 127.
    final c127 = Const(
      FloatingPoint16().valuePopulator().ofDouble(127.0).value.toInt(),
      width: 16,
    );
    final invScaleNext = FloatingPointMultiplierSimple(
      FloatingPoint16()..gets(recipOut),
      FloatingPoint16()..gets(c127),
      clk: clk,
    ).product.packed;

    // Quantize: q = clamp(round(x * invScale), -127, 127). The multiply is
    // clocked (mantissa-mult flopped), so `scaled` (and thus q8) lag x_in by
    // coreLatency. Q_valid is delayed to match and the consumer packs on it.
    final scaled = FloatingPointMultiplierSimple(
      FloatingPoint16()..gets(xIn),
      FloatingPoint16()..gets(invScale),
      clk: clk,
    ).product.packed;
    // 10-bit signed fixed (1 sign + 9 int), then clamp.
    final sval = FloatToFixed(
      FloatingPoint16()..gets(scaled),
      integerWidth: 9,
      fractionWidth: 0,
    ).fixed.packed;
    // Clamp in the unsigned domain (flip the sign bit -> monotonic).
    final u = sval ^ Const(0x200, width: 10);
    final hi = Const(127 + 512, width: 10);
    final lo = Const(-127 + 512, width: 10);
    final uClamped = mux(u.gt(hi), hi, mux(u.lt(lo), lo, u));
    final q8 = (uClamped ^ Const(0x200, width: 10)).getRange(0, 8);

    Sequential(clk, [
      If(
        reset,
        then: [
          state < Const(_max, width: 3),
          maxAbs < Const(0, width: 16),
          invScale < Const(0, width: 16),
          deqScale < Const(0, width: 16),
          iwait < Const(0, width: iwW),
          if (hasOverride) ovrActive! < Const(0),
          if (hasOverride) ovrScale! < Const(0, width: 16),
        ],
        orElse: [
          Case(state, [
            CaseItem(Const(_max, width: 3), [
              If(input('x_en') & absBigger, then: [maxAbs < absX]),
              If(
                input('compute'),
                then: [
                  state < Const(_r1start, width: 3),
                  if (hasOverride)
                    ovrActive! < scaleOverrideIn!.neq(Const(0, width: 16)),
                  if (hasOverride) ovrScale! < scaleOverrideIn!,
                ],
              ),
            ]),
            CaseItem(Const(_r1start, width: 3), [
              state < Const(_r1wait, width: 3),
            ]),
            CaseItem(Const(_r1wait, width: 3), [
              // recipOut is final at done and holds. Wait for the clocked
              // invScale multiply to settle, then capture. Override path: skip
              // the *127 multiply and the second recip pass entirely - recipOut
              // here is already 1/ovrScale, and the dequant scale is the
              // override value itself (exact), not a recip round-trip of it.
              If(
                recipDone,
                then: hasOverride
                    ? [
                        If(
                          ovrActive!,
                          then: [
                            invScale < recipOut,
                            deqScale < ovrScale!,
                            state < Const(_ready, width: 3),
                          ],
                          orElse: [
                            iwait < Const(coreLatency, width: iwW),
                            state < Const(_r1mul, width: 3),
                          ],
                        ),
                      ]
                    : [
                        iwait < Const(coreLatency, width: iwW),
                        state < Const(_r1mul, width: 3),
                      ],
              ),
            ]),
            CaseItem(Const(_r1mul, width: 3), [
              If(
                iwait.gt(Const(0, width: iwW)),
                then: [iwait < (iwait - Const(1, width: iwW))],
                orElse: [
                  invScale < invScaleNext,
                  state < Const(_r2start, width: 3),
                ],
              ),
            ]),
            CaseItem(Const(_r2start, width: 3), [
              state < Const(_r2wait, width: 3),
            ]),
            CaseItem(Const(_r2wait, width: 3), [
              If(
                recipDone,
                then: [deqScale < recipOut, state < Const(_ready, width: 3)],
              ),
            ]),
            CaseItem(Const(_ready, width: 3), []),
          ]),
        ],
      ),
    ]);

    qOutP <= q8;
    // q8 lags x_in by coreLatency (clocked multiply), so delay the valid to
    // match: the consumer packs q_out on q_valid.
    var qv = st(_ready) & input('x_en');
    for (var i = 0; i < coreLatency; i++) {
      qv = flop(clk, qv, reset: reset, resetValue: 0);
    }
    qValidP <= qv;
    scaleOutP <= deqScale;
    readyP <= st(_ready);
  }
}
