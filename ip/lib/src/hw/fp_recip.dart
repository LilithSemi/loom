import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

/// LoomFpRecip: multi-cycle fp16 reciprocal `out = 1/d`, the division primitive
/// RMSNorm (x/rms) and softmax (.../sum) need (rohd_hcl has no FP divider).
///
/// Bit-trick seed (`reinterpret(MAGIC - bits(d))` ~ 1/d) refined by Newton-
/// Raphson `y = y*(2 - d*y)`. Crucially ITERATIVE, not combinational: it
/// instantiates ONE multiplier + ONE adder and sequences them over 3 micro-
/// steps per iteration, so area stays one FP mult + one adder regardless of
/// iteration count (the combinational unroll cascaded ~12 FP modules and was
/// untenable).
///
/// PIPELINED: the shared mult/adder are clocked ([coreLatency]-cycle internal
/// latency, splitting the mantissa op from the normalize so the fabric isn't
/// capped by a ~45ns combinational FP op). Each microstep waits coreLatency
/// cycles for the piped core before capturing, so latency = 3*(1+coreLatency)*
/// iterations cycles after `start`. The recip runs once per linear, so the
/// extra cycles are free.
///
/// Ports: in clk, reset, start (pulse), d[16]. Out out[16], done, busy.
/// Assumes d > 0 and normal (RMSNorm's rms always is).
class LoomFpRecip extends BridgeModule {
  static const int recipMagic = 0x7800; // ~ (2*bias)<<mantissaBits = 30<<10
  static const int _sMul1 = 0; // y' = d * y
  static const int _sSub = 1; //  s = 2 - dy
  static const int _sMul2 = 2; // y  = y * s

  LoomFpRecip({int iterations = 4, int coreLatency = 1, String? name})
    : super('LoomFpRecip', name: name ?? 'loom_fp_recip') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('start', PortDirection.input);
    createPort('d', PortDirection.input, width: 16);
    final outP = addOutput('out', width: 16);
    final doneP = addOutput('done');
    final busyP = addOutput('busy');

    final clk = input('clk');
    final reset = input('reset');

    // State.
    final yReg = Logic(name: 'y', width: 16);
    final dReg = Logic(name: 'd_reg', width: 16);
    final dyReg = Logic(name: 'dy', width: 16);
    final sReg = Logic(name: 's', width: 16);
    final state = Logic(name: 'state', width: 2);
    final iter = Logic(name: 'iter', width: 8);
    final busy = Logic(name: 'busy_r');
    final done = Logic(name: 'done_r');
    // Wait counter: cycles remaining for the clocked FP core to settle.
    final wcW = (coreLatency + 1).bitLength.clamp(1, 8);
    final wc = Logic(name: 'wc', width: wcW);

    final two = FloatingPoint16()..gets(Const(0x4000, width: 16)); // 2.0

    // Shared multiplier (clocked): a = y; b = (state==MUL2 ? s : d).
    final mulA = FloatingPoint16()..gets(yReg);
    final mulB = FloatingPoint16()
      ..gets(mux(state.eq(Const(_sMul2, width: 2)), sReg, dReg));
    final product = FloatingPointMultiplierSimple(
      mulA,
      mulB,
      clk: clk,
    ).product.packed;

    // Shared adder (clocked): 2 + (-dy)  =  2 - dy.
    final negDy = FloatingPoint16()
      ..gets([~dyReg[15], dyReg.getRange(0, 15)].swizzle());
    final sub = FloatingPointAdderSinglePath(two, negDy, clk: clk).sum.packed;

    final lastIter = iter.eq(Const(iterations - 1, width: 8));
    final waiting = wc.gt(Const(0, width: wcW));
    final wcInit = Const(coreLatency, width: wcW);

    Sequential(clk, [
      If(
        reset,
        then: [
          yReg < Const(0, width: 16),
          dReg < Const(0, width: 16),
          dyReg < Const(0, width: 16),
          sReg < Const(0, width: 16),
          state < Const(0, width: 2),
          iter < Const(0, width: 8),
          busy < Const(0),
          done < Const(0),
          wc < Const(0, width: wcW),
        ],
        orElse: [
          done < Const(0),
          If(
            ~busy,
            then: [
              If(
                input('start'),
                then: [
                  dReg < input('d'),
                  yReg < (Const(recipMagic, width: 16) - input('d')), // seed
                  iter < Const(0, width: 8),
                  state < Const(_sMul1, width: 2),
                  busy < Const(1),
                  wc < wcInit, // let the piped mult settle before first capture
                ],
              ),
            ],
            orElse: [
              // Wait for the clocked FP core, then capture + advance.
              If(
                waiting,
                then: [wc < (wc - Const(1, width: wcW))],
                orElse: [
                  Case(state, [
                    CaseItem(Const(_sMul1, width: 2), [
                      dyReg < product, // d*y
                      state < Const(_sSub, width: 2),
                      wc < wcInit,
                    ]),
                    CaseItem(Const(_sSub, width: 2), [
                      sReg < sub, // 2 - dy
                      state < Const(_sMul2, width: 2),
                      wc < wcInit,
                    ]),
                    CaseItem(Const(_sMul2, width: 2), [
                      yReg < product, // y*s
                      If(
                        lastIter,
                        then: [busy < Const(0), done < Const(1)],
                        orElse: [
                          iter < (iter + Const(1, width: 8)),
                          state < Const(_sMul1, width: 2),
                          wc < wcInit,
                        ],
                      ),
                    ]),
                  ]),
                ],
              ),
            ],
          ),
        ],
      ),
    ]);

    outP <= yReg;
    doneP <= done;
    busyP <= busy;
  }
}
