// LoomActQuant test: dynamic int8 activation quant round-trips within the
// quant step (q * scale_out ~= x), and q stays in [-127, 127].

import 'dart:async';

import 'package:loom/src/hw/act_quant.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('int8 activation quant round-trips within the quant step', () async {
    final x = [0.5, -1.0, 2.0, 1.0, -2.5, 0.25];
    final maxAbs = x.map((v) => v.abs()).reduce((a, b) => a > b ? a : b);
    final step = maxAbs / 127.0;

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final xEn = Logic(name: 'x_en');
    final xInL = Logic(name: 'x_in', width: 16);
    final compute = Logic(name: 'compute');

    final dut = LoomActQuant();
    for (final (p, s) in [
      ('clk', clk),
      ('reset', reset),
      ('x_en', xEn),
      ('x_in', xInL),
      ('compute', compute),
    ]) {
      dut.input(p).srcConnection! <= s;
    }
    await dut.build();

    final fp = FloatingPoint16();
    int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
    final scaleFp = FloatingPoint16();

    for (final l in [xEn, compute]) {
      l.inject(0);
    }
    xInL.inject(0);
    reset.inject(1);
    Simulator.setMaxSimTime(2000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    // MAX pass.
    for (final v in x) {
      xEn.inject(1);
      xInL.inject(e(v));
      await clk.nextPosedge;
    }
    xEn.inject(0);

    // compute -> two recips.
    compute.inject(1);
    await clk.nextPosedge;
    compute.inject(0);
    var guard = 0;
    while (!dut.output('ready').value.toBool() && guard++ < 200) {
      await clk.nextPosedge;
    }
    expect(dut.output('ready').value.toBool(), isTrue);

    scaleFp.put(dut.output('scale_out').value);
    final scale = scaleFp.floatingPointValue.toDouble();

    // QUANT pass: feed x. Collect q_out on q_valid (delayed by the clocked
    // multiply's latency), in order.
    final q = <int>[];
    final collect = () async {
      while (q.length < x.length) {
        await clk.nextNegedge;
        if (dut.output('q_valid').value.toBool()) {
          var raw = dut.output('q_out').value.toInt();
          if (raw >= 128) raw -= 256;
          q.add(raw);
        }
      }
    }();
    for (final v in x) {
      xEn.inject(1);
      xInL.inject(e(v));
      await clk.nextPosedge;
    }
    xEn.inject(0);
    await collect;
    await Simulator.endSimulation();

    for (final qi in q) {
      expect(qi, inInclusiveRange(-127, 127));
    }
    for (var i = 0; i < x.length; i++) {
      final y = q[i] * scale;
      expect(
        y,
        closeTo(x[i], step * 1.5 + 0.02),
        reason: 'x[$i]=${x[i]} q=${q[i]} scale=$scale y=$y',
      );
    }
    // The largest-magnitude element should saturate near +-127.
    expect(
      q.map((v) => v.abs()).reduce((a, b) => a > b ? a : b),
      greaterThanOrEqualTo(120),
    );
  });
}
