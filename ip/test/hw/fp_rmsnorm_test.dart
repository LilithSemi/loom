// LoomRmsNorm diff-test: RMSNorm over a vector in fp16 vs the golden formula,
// within fp16 tolerance.

import 'dart:async';
import 'dart:math' as math;

import 'package:loom/src/hw/fp_rmsnorm.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('rmsnorm vector matches golden within fp16 tolerance', () async {
    final x = [1.5, -2.0, 0.5, -1.0, 2.0, -0.5, 1.0, -1.5];
    final gamma = [1.0, 0.5, 1.5, 1.0, 0.75, 1.25, 1.0, 0.5];
    const eps = 0.01;
    final n = x.length;

    // Golden.
    var ms = 0.0;
    for (final v in x) {
      ms += v * v;
    }
    ms /= n;
    final inv = 1.0 / math.sqrt(ms + eps);
    final golden = [for (var i = 0; i < n; i++) x[i] * inv * gamma[i]];

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final accEn = Logic(name: 'acc_en');
    final compute = Logic(name: 'compute');
    final normEn = Logic(name: 'norm_en');
    final xIn = Logic(name: 'x_in', width: 16);
    final gIn = Logic(name: 'g_in', width: 16);
    final epsL = Logic(name: 'eps', width: 16);
    final invN = Logic(name: 'inv_n', width: 16);

    final dut = LoomRmsNorm();
    for (final (p, s) in [
      ('clk', clk),
      ('reset', reset),
      ('acc_en', accEn),
      ('compute', compute),
      ('norm_en', normEn),
      ('x_in', xIn),
      ('gamma_in', gIn),
      ('eps', epsL),
      ('inv_n', invN),
    ]) {
      dut.input(p).srcConnection! <= s;
    }
    await dut.build();

    final fp = FloatingPoint16();
    int enc(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
    final outFp = FloatingPoint16();

    for (final l in [accEn, compute, normEn]) {
      l.inject(0);
    }
    xIn.inject(0);
    gIn.inject(0);
    epsL.inject(enc(eps));
    invN.inject(enc(1.0 / n));
    reset.inject(1);
    Simulator.setMaxSimTime(2000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    // Pass 1: accumulate sum(x^2).
    for (var i = 0; i < n; i++) {
      accEn.inject(1);
      xIn.inject(enc(x[i]));
      await clk.nextPosedge;
    }
    accEn.inject(0);

    // Compute rinv.
    compute.inject(1);
    await clk.nextPosedge;
    compute.inject(0);
    var g = 0;
    while (!dut.output('ready').value.toBool() && g++ < 100) {
      await clk.nextPosedge;
    }
    expect(dut.output('ready').value.toBool(), isTrue, reason: 'rinv ready');

    // Pass 2: normalize.
    final got = <double>[];
    for (var i = 0; i < n; i++) {
      normEn.inject(1);
      xIn.inject(enc(x[i]));
      gIn.inject(enc(gamma[i]));
      await clk.nextNegedge; // combinational settle
      outFp.put(dut.output('y').value);
      got.add(outFp.floatingPointValue.toDouble());
      await clk.nextPosedge;
    }
    await Simulator.endSimulation();

    for (var i = 0; i < n; i++) {
      expect(
        got[i],
        closeTo(golden[i], golden[i].abs() * 0.1 + 0.05),
        reason: 'y[$i]=${got[i]} vs ${golden[i]}',
      );
    }
  });
}
