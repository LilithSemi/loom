import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

import 'fp_recip.dart';

/// LoomRmsNorm: RMSNorm over a vector in fp16, matching the golden
///   inv = 1 / sqrt(sum(x^2)/N + eps);  y_i = x_i * inv * gamma_i.
///
/// Two host-paced passes (the sequencer re-reads x from memory. No big buffer):
///   1. ACC: drive acc_en + x_in for each of the N elements -> sum(x^2).
///   2. pulse `compute`: meansq = sumsq * inv_n; mse = meansq + eps;
///      rms = sqrt(mse); rinv = 1/rms (the multi-cycle LoomFpRecip). `ready`
///      asserts when rinv is latched.
///   3. NORM: drive norm_en + x_in + gamma_in per element -> y on `y`/`y_valid`.
///
/// One multiplier + one adder are SHARED across the phases (muxed by state),
/// plus one sqrt and one LoomFpRecip, keeping the FP-core count low (cascading
/// many FP cores is untenable - see LoomFpRecip). All ports are packed fp16.
class LoomRmsNorm extends BridgeModule {
  static const int _acc = 0;
  static const int _calc = 1;
  static const int _wait = 2;
  static const int _ready = 3;

  LoomRmsNorm({int recipIterations = 4, String? name})
    : super('LoomRmsNorm', name: name ?? 'loom_rmsnorm') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('acc_en', PortDirection.input);
    createPort('compute', PortDirection.input);
    createPort('norm_en', PortDirection.input);
    createPort('x_in', PortDirection.input, width: 16);
    createPort('gamma_in', PortDirection.input, width: 16);
    createPort('eps', PortDirection.input, width: 16);
    createPort('inv_n', PortDirection.input, width: 16);
    final yP = addOutput('y', width: 16);
    final yValidP = addOutput('y_valid');
    final readyP = addOutput('ready');

    final clk = input('clk');
    final reset = input('reset');
    final xIn = input('x_in');

    final sumsq = Logic(name: 'sumsq', width: 16);
    final rinv = Logic(name: 'rinv', width: 16);
    final state = Logic(name: 'state', width: 2);

    final isAcc = state.eq(Const(_acc, width: 2));
    final isCalc =
        state.eq(Const(_calc, width: 2)) | state.eq(Const(_wait, width: 2));
    final isReady = state.eq(Const(_ready, width: 2));

    // Shared multiplier (mul1): ACC -> x*x. CALC/WAIT -> sumsq*invN. READY -> x*rinv.
    final mul1A = FloatingPoint16()..gets(mux(isCalc, sumsq, xIn));
    final mul1B = FloatingPoint16()
      ..gets(mux(isAcc, xIn, mux(isReady, rinv, input('inv_n'))));
    final prod1 = FloatingPointMultiplierSimple(mul1A, mul1B).product.packed;

    // Shared adder: ACC -> sumsq + x^2. CALC -> meansq + eps.
    final addA = FloatingPoint16()..gets(mux(isCalc, prod1, sumsq));
    final addB = FloatingPoint16()..gets(mux(isCalc, input('eps'), prod1));
    final addS = FloatingPointAdderSinglePath(addA, addB).sum.packed;

    // sqrt(mse) (CALC). mse = addS while isCalc.
    final mse = FloatingPoint16()..gets(addS);
    final rms = FloatingPointSqrtSimple(mse).sqrt.packed;

    // 1/rms via the multi-cycle reciprocal.
    final recip = LoomFpRecip(iterations: recipIterations);
    recip.input('clk').srcConnection! <= clk;
    recip.input('reset').srcConnection! <= reset;
    recip.input('d').srcConnection! <= rms;
    recip.input('start').srcConnection! <= state.eq(Const(_calc, width: 2));

    // NORM second multiply: y = (x*rinv) * gamma.
    final tFp = FloatingPoint16()..gets(prod1); // x*rinv in READY
    final gFp = FloatingPoint16()..gets(input('gamma_in'));
    final yProd = FloatingPointMultiplierSimple(tFp, gFp).product.packed;

    Sequential(clk, [
      If(
        reset,
        then: [
          sumsq < Const(0, width: 16),
          rinv < Const(0, width: 16),
          state < Const(_acc, width: 2),
        ],
        orElse: [
          Case(state, [
            CaseItem(Const(_acc, width: 2), [
              If(input('acc_en'), then: [sumsq < addS]),
              If(input('compute'), then: [state < Const(_calc, width: 2)]),
            ]),
            CaseItem(Const(_calc, width: 2), [
              state < Const(_wait, width: 2), // recip.start pulsed this cycle
            ]),
            CaseItem(Const(_wait, width: 2), [
              If(
                recip.output('done'),
                then: [
                  rinv < recip.output('out'),
                  state < Const(_ready, width: 2),
                ],
              ),
            ]),
            CaseItem(Const(_ready, width: 2), []),
          ]),
        ],
      ),
    ]);

    yP <= yProd;
    yValidP <= isReady & input('norm_en');
    readyP <= isReady;
  }
}
