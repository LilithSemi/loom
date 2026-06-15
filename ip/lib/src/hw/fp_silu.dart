import 'dart:math' as math;

import 'package:rohd/rohd.dart';
import 'package:rohd_bridge/rohd_bridge.dart';
import 'package:rohd_hcl/rohd_hcl.dart';

/// Selects `entries[index]` with a balanced binary mux tree (depth ~log2 N).
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

/// LoomSiLU: elementwise fp16 SiLU `y = x * sigmoid(x)` via a baked BRAM LUT.
///
/// Establishes the transcendental-table pattern stage 4 reuses (softmax's exp,
/// RoPE's cos/sin): convert the fp16 input to a fixed-point index with rohd_hcl
/// FloatToFixed (Q4.4 -> a 9-bit signed index over [-16, 16) step 1/16 = 512
/// entries), look up a precomputed fp16 SiLU value. Combinational. SiLU
/// saturates, so the +-16 table window + (TODO) clamp covers the real range. The
/// test stays within +-8.
class LoomSiLU extends BridgeModule {
  /// Q4.4 fixed-point index: 1 sign + 4 integer + 4 fraction bits.
  static const int _intW = 4;
  static const int _fracW = 4;

  LoomSiLU({String? name}) : super('LoomSiLU', name: name ?? 'loom_silu') {
    createPort('x', PortDirection.input, width: 16);
    final yP = addOutput('y', width: 16);

    final xf = FloatingPoint16()..gets(input('x'));
    final ftf = FloatToFixed(xf, integerWidth: _intW, fractionWidth: _fracW);
    final idx = ftf.fixed.packed; // 1+intW+fracW = 9 bits, two's complement

    // Bake the SiLU table: entry b (unsigned) <-> value = toSigned(b)/2^fracW.
    final n = 1 << (1 + _intW + _fracW); // 512
    final enc = FloatingPoint16();
    final table = <Logic>[
      for (var b = 0; b < n; b++)
        Const(
          enc
              .valuePopulator()
              .ofDouble(_silu(_indexToValue(b, n)))
              .value
              .toInt(),
          width: 16,
        ),
    ];

    yP <= _muxTree(idx, table);
  }

  static double _indexToValue(int b, int n) =>
      (b >= n ~/ 2 ? b - n : b) / (1 << _fracW);

  static double _silu(double x) => x / (1.0 + math.exp(-x));
}
