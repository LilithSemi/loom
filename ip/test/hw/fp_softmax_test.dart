// LoomSoftmax diff-test: stable softmax over a vector in fp16 vs golden.

import 'dart:async';
import 'dart:math' as math;

import 'package:loom/src/hw/fp_softmax.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('softmax matches golden within tolerance', () async {
    final x = [1.0, 2.0, -1.0, 0.5, -2.0, 1.5];
    final n = x.length;
    // Golden.
    final mx = x.reduce(math.max);
    final ex = [for (final v in x) math.exp(v - mx)];
    final sum = ex.reduce((a, b) => a + b);
    final golden = [for (final e in ex) e / sum];

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final maxEn = Logic(name: 'max_en');
    final sumEn = Logic(name: 'sum_en');
    final compute = Logic(name: 'compute');
    final normEn = Logic(name: 'norm_en');
    final xIn = Logic(name: 'x_in', width: 16);

    final dut = LoomSoftmax();
    for (final (p, s) in [
      ('clk', clk),
      ('reset', reset),
      ('max_en', maxEn),
      ('sum_en', sumEn),
      ('compute', compute),
      ('norm_en', normEn),
      ('x_in', xIn),
    ]) {
      dut.input(p).srcConnection! <= s;
    }
    await dut.build();

    final fp = FloatingPoint16();
    int enc(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
    final outFp = FloatingPoint16();

    for (final l in [maxEn, sumEn, compute, normEn]) {
      l.inject(0);
    }
    xIn.inject(0);
    reset.inject(1);
    Simulator.setMaxSimTime(2000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    // MAX pass.
    for (var i = 0; i < n; i++) {
      maxEn.inject(1);
      xIn.inject(enc(x[i]));
      await clk.nextPosedge;
    }
    maxEn.inject(0);

    // SUM pass (first sum_en also ends MAX).
    for (var i = 0; i < n; i++) {
      sumEn.inject(1);
      xIn.inject(enc(x[i]));
      await clk.nextPosedge;
    }
    sumEn.inject(0);

    // compute reciprocal of sum.
    compute.inject(1);
    await clk.nextPosedge;
    compute.inject(0);
    var g = 0;
    while (!dut.output('ready').value.toBool() && g++ < 100) {
      await clk.nextPosedge;
    }
    expect(dut.output('ready').value.toBool(), isTrue);

    // NORM pass.
    final got = <double>[];
    for (var i = 0; i < n; i++) {
      normEn.inject(1);
      xIn.inject(enc(x[i]));
      await clk.nextNegedge;
      outFp.put(dut.output('y').value);
      got.add(outFp.floatingPointValue.toDouble());
      await clk.nextPosedge;
    }
    await Simulator.endSimulation();

    for (var i = 0; i < n; i++) {
      expect(
        got[i],
        closeTo(golden[i], 0.03 + golden[i] * 0.1),
        reason: 'y[$i]=${got[i]} vs ${golden[i]}',
      );
    }
    expect(
      got.reduce((a, b) => a + b),
      closeTo(1.0, 0.1),
      reason: 'softmax sums to ~1',
    );
  });
}
