// LoomFpMac diff-test: streaming fp16 multiply-accumulate vs a golden dot
// product, within fp16 tolerance.

import 'dart:async';

import 'package:loom/src/hw/fp_mac.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('mac accumulates a dot product within tolerance', () async {
    final a = [1.0, 2.0, -0.5, 3.0, 0.25, -1.5];
    final b = [0.5, -1.0, 2.0, 1.0, 4.0, -2.0];
    var golden = 0.0;
    for (var i = 0; i < a.length; i++) {
      golden += a[i] * b[i];
    }

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final clear = Logic(name: 'clear');
    final en = Logic(name: 'en');
    final aIn = Logic(name: 'a', width: 16);
    final bIn = Logic(name: 'b', width: 16);

    final dut = LoomFpMac();
    for (final (p, s) in [
      ('clk', clk),
      ('reset', reset),
      ('clear', clear),
      ('en', en),
      ('a', aIn),
      ('b', bIn),
    ]) {
      dut.input(p).srcConnection! <= s;
    }
    await dut.build();

    final fp = FloatingPoint16();
    int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
    final outFp = FloatingPoint16();

    for (final l in [clear, en]) {
      l.inject(0);
    }
    aIn.inject(0);
    bIn.inject(0);
    reset.inject(1);
    Simulator.setMaxSimTime(1000000);
    unawaited(Simulator.run());
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);

    // Clear the accumulator.
    clear.inject(1);
    await clk.nextPosedge;
    clear.inject(0);

    // Stream the pairs.
    for (var i = 0; i < a.length; i++) {
      en.inject(1);
      aIn.inject(e(a[i]));
      bIn.inject(e(b[i]));
      await clk.nextPosedge;
    }
    en.inject(0);
    await clk.nextNegedge;

    outFp.put(dut.output('acc').value);
    final got = outFp.floatingPointValue.toDouble();
    await Simulator.endSimulation();

    expect(
      got,
      closeTo(golden, 0.05 + golden.abs() * 0.03),
      reason: 'mac=$got vs $golden',
    );
  });
}
