// LoomRope diff-test: fp16 half-split rotary embedding vs the golden
// applyRopeHead, within fp16 tolerance.

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:loom/src/golden/attention.dart';
import 'package:loom/src/hw/fp_rope.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  test('rope rotation matches golden applyRopeHead within tolerance', () async {
    final x1 = Logic(name: 'x1', width: 16);
    final x2 = Logic(name: 'x2', width: 16);
    final cosIn = Logic(name: 'cos_in', width: 16);
    final sinIn = Logic(name: 'sin_in', width: 16);

    final dut = LoomRope();
    dut.input('x1').srcConnection! <= x1;
    dut.input('x2').srcConnection! <= x2;
    dut.input('cos_in').srcConnection! <= cosIn;
    dut.input('sin_in').srcConnection! <= sinIn;
    await dut.build();

    final enc = FloatingPoint16();
    int e(double d) => enc.valuePopulator().ofDouble(d).value.toInt();
    final outY1 = FloatingPoint16();
    final outY2 = FloatingPoint16();

    const headDim = 8;
    const theta = 10000.0;
    final half = headDim ~/ 2;

    // Sweep a few positions and dim pairs against the golden, which rotates a
    // whole head in place. We mirror its per-pair math through the DUT.
    for (final pos in [0, 1, 5, 17, 64]) {
      // Build a head with known values, run the golden in place.
      final head = Float64List(headDim);
      for (var i = 0; i < headDim; i++) {
        head[i] = math.sin(i * 0.7 + 1.0) * 2.0; // arbitrary but deterministic
      }
      final golden = Float64List.fromList(head);
      applyRopeHead(golden, pos, theta);

      for (var j = 0; j < half; j++) {
        final invFreq = 1.0 / math.pow(theta, (2 * j) / headDim);
        final angle = pos * invFreq;
        final c = math.cos(angle);
        final s = math.sin(angle);

        x1.put(e(head[j]));
        x2.put(e(head[j + half]));
        cosIn.put(e(c));
        sinIn.put(e(s));

        outY1.put(dut.output('y1').value);
        outY2.put(dut.output('y2').value);
        final gotY1 = outY1.floatingPointValue.toDouble();
        final gotY2 = outY2.floatingPointValue.toDouble();

        expect(
          gotY1,
          closeTo(golden[j], 0.03 + golden[j].abs() * 0.05),
          reason: 'pos=$pos j=$j y1=$gotY1 vs ${golden[j]}',
        );
        expect(
          gotY2,
          closeTo(golden[j + half], 0.03 + golden[j + half].abs() * 0.05),
          reason: 'pos=$pos j=$j y2=$gotY2 vs ${golden[j + half]}',
        );
      }
    }
  });
}
