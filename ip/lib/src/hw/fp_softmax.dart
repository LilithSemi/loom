import 'dart:math' as math;

import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

import 'fp_recip.dart';

Logic _muxTree(Logic index, List<Logic> entries) {
  if (entries.length == 1) return entries[0];
  final next = <Logic>[];
  for (var i = 0; i < entries.length; i += 2) {
    next.add(
      i + 1 < entries.length
          ? mux(index[0], entries[i + 1], entries[i])
          : entries[i],
    );
  }
  return _muxTree(index.getRange(1, index.width), next);
}

/// LoomSoftmax: numerically-stable softmax over a vector in fp16, matching the
/// golden `e_i = exp(x_i - max); y_i = e_i / sum(e)`.
///
/// Three host-paced passes (sequencer re-reads x. No buffer):
///   1. MAX: max_en + x_in per element -> running max (float-key unsigned
///      compare, so no FP comparator needed).
///   2. SUM: sum_en + x_in per element -> sum of exp(x_in - max) (exp via the
///      LUT pattern: FloatToFixed Q4.4 index -> baked table).
///   3. pulse `compute` -> rinv = 1/sum (LoomFpRecip), `ready` asserts.
///   4. NORM: norm_en + x_in -> y = exp(x_in - max) * rinv.
/// The MAX->SUM transition is implicit on the first sum_en. fp16 ports.
class LoomSoftmax extends BridgeModule {
  static const int _intW = 4;
  static const int _fracW = 4;
  static const int _max = 0;
  static const int _sum = 1;
  static const int _calc = 2;
  static const int _wait = 3;
  static const int _ready = 4;

  LoomSoftmax({int recipIterations = 4, String? name})
    : super('LoomSoftmax', name: name ?? 'loom_softmax') {
    createPort('clk', PortDirection.input);
    createPort('reset', PortDirection.input);
    createPort('max_en', PortDirection.input);
    createPort('sum_en', PortDirection.input);
    createPort('compute', PortDirection.input);
    createPort('norm_en', PortDirection.input);
    createPort('x_in', PortDirection.input, width: 16);
    final yP = addOutput('y', width: 16);
    final yValidP = addOutput('y_valid');
    final readyP = addOutput('ready');

    final clk = input('clk');
    final reset = input('reset');
    final xIn = input('x_in');

    final maxReg = Logic(name: 'max', width: 16);
    final sumReg = Logic(name: 'sum', width: 16);
    final rinv = Logic(name: 'rinv', width: 16);
    final state = Logic(name: 'state', width: 3);

    // Monotonic unsigned key for fp16 compare: negatives invert, positives set
    // the MSB. key(x) > key(max) iff x > max.
    Logic key(Logic b) => mux(b[15], ~b, b | Const(0x8000, width: 16));
    final xBigger = key(xIn).gt(key(maxReg));

    // t = x - max  (max is final once we leave the MAX phase).
    final negMax = FloatingPoint16()
      ..gets([~maxReg[15], maxReg.getRange(0, 15)].swizzle());
    final t = FloatingPointAdderSinglePath(
      FloatingPoint16()..gets(xIn),
      negMax,
    ).sum.packed;

    // exp(t) via the LUT pattern (t <= 0 here -> exp in (0,1]).
    final expIdx = FloatToFixed(
      FloatingPoint16()..gets(t),
      integerWidth: _intW,
      fractionWidth: _fracW,
    ).fixed.packed;
    final n = 1 << (1 + _intW + _fracW);
    final enc = FloatingPoint16();
    final expTable = <Logic>[
      for (var b = 0; b < n; b++)
        Const(
          enc
              .valuePopulator()
              .ofDouble(math.exp(_indexToValue(b, n)))
              .value
              .toInt(),
          width: 16,
        ),
    ];
    final expVal = _muxTree(expIdx, expTable);

    // sum += exp (SUM phase). y = exp * rinv (NORM phase).
    final sumNext = FloatingPointAdderSinglePath(
      FloatingPoint16()..gets(sumReg),
      FloatingPoint16()..gets(expVal),
    ).sum.packed;
    final yVal = FloatingPointMultiplierSimple(
      FloatingPoint16()..gets(expVal),
      FloatingPoint16()..gets(rinv),
    ).product.packed;

    final recip = LoomFpRecip(iterations: recipIterations);
    recip.input('clk').srcConnection! <= clk;
    recip.input('reset').srcConnection! <= reset;
    recip.input('d').srcConnection! <= sumReg;
    recip.input('start').srcConnection! <= state.eq(Const(_calc, width: 3));

    Sequential(clk, [
      If(
        reset,
        then: [
          maxReg < Const(0xFC00, width: 16), // -inf
          sumReg < Const(0, width: 16),
          rinv < Const(0, width: 16),
          state < Const(_max, width: 3),
        ],
        orElse: [
          Case(state, [
            CaseItem(Const(_max, width: 3), [
              If(input('max_en') & xBigger, then: [maxReg < xIn]),
              // First sum_en ends MAX and processes that element in SUM.
              If(
                input('sum_en'),
                then: [sumReg < sumNext, state < Const(_sum, width: 3)],
              ),
            ]),
            CaseItem(Const(_sum, width: 3), [
              If(input('sum_en'), then: [sumReg < sumNext]),
              If(input('compute'), then: [state < Const(_calc, width: 3)]),
            ]),
            CaseItem(Const(_calc, width: 3), [state < Const(_wait, width: 3)]),
            CaseItem(Const(_wait, width: 3), [
              If(
                recip.output('done'),
                then: [
                  rinv < recip.output('out'),
                  state < Const(_ready, width: 3),
                ],
              ),
            ]),
            CaseItem(Const(_ready, width: 3), []),
          ]),
        ],
      ),
    ]);

    yP <= yVal;
    yValidP <= state.eq(Const(_ready, width: 3)) & input('norm_en');
    readyP <= state.eq(Const(_ready, width: 3));
  }

  static double _indexToValue(int b, int n) =>
      (b >= n ~/ 2 ? b - n : b) / (1 << _fracW);
}
