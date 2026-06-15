// LoomFpRecip (multi-cycle) diff-test: fp16 1/d vs golden within tolerance.

import 'dart:async';

import 'package:loom/src/hw/fp_recip.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('out = 1/d in fp16 within tolerance over a positive range', () async {
    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final start = Logic(name: 'start');
    final d = Logic(name: 'd', width: 16);
    final dut = LoomFpRecip();
    dut.input('clk').srcConnection! <= clk;
    dut.input('reset').srcConnection! <= reset;
    dut.input('start').srcConnection! <= start;
    dut.input('d').srcConnection! <= d;
    await dut.build();

    final enc = FloatingPoint16();
    final outFp = FloatingPoint16();

    start.inject(0);
    reset.inject(1);
    Simulator.setMaxSimTime(2000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    Future<double> recip(double v) async {
      d.inject(enc.valuePopulator().ofDouble(v).value);
      start.inject(1);
      await clk.nextPosedge;
      start.inject(0);
      var g = 0;
      while (!dut.output('done').value.toBool() && g++ < 100) {
        await clk.nextPosedge;
      }
      outFp.put(dut.output('out').value);
      await clk.nextPosedge; // settle before next op
      return outFp.floatingPointValue.toDouble();
    }

    for (final v in [1.0, 2.0, 4.0, 0.5, 3.0, 8.0, 0.25, 6.0, 1.5, 12.0, 0.1]) {
      final got = await recip(v);
      expect(
        got,
        closeTo(1.0 / v, (1.0 / v).abs() * 0.05),
        reason: '1/$v=$got',
      );
    }
    await Simulator.endSimulation();
  });
}
