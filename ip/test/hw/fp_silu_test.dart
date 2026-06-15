// LoomSiLU diff-test: fp16 x*sigmoid(x) via LUT vs golden, within tolerance.

import 'dart:math' as math;

import 'package:loom/src/hw/fp_silu.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

double _silu(double x) => x / (1.0 + math.exp(-x));

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('silu(x) via LUT matches golden within tolerance', () async {
    final x = Logic(name: 'x', width: 16);
    final dut = LoomSiLU();
    dut.input('x').srcConnection! <= x;
    await dut.build();

    final enc = FloatingPoint16();
    final outFp = FloatingPoint16();

    for (final v in [
      0.0,
      0.5,
      1.0,
      2.0,
      4.0,
      -0.5,
      -1.0,
      -2.0,
      -4.0,
      3.0,
      -3.0,
      1.5,
      -6.0,
    ]) {
      x.put(enc.valuePopulator().ofDouble(v).value);
      outFp.put(dut.output('y').value);
      final got = outFp.floatingPointValue.toDouble();
      // LUT step 1/16 -> allow absolute + small relative slack.
      expect(
        got,
        closeTo(_silu(v), 0.1 + _silu(v).abs() * 0.05),
        reason: 'silu($v)=$got vs ${_silu(v)}',
      );
    }
  });
}
