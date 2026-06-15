// Probe: how many cycles of latency does passing `clk` add to the rohd_hcl FP
// multiplier / adder (it pipelines the mantissa op)? Needed to thread the right
// wait counts through LoomFpRecip / LoomActQuant when pipelining.

import 'dart:async';

import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  Future<int> latency(String which) async {
    final clk = SimpleClockGenerator(10).clk;
    final a = FloatingPoint16();
    final b = FloatingPoint16();
    final Logic out;
    if (which == 'mul') {
      out = FloatingPointMultiplierSimple(a, b, clk: clk).product.packed;
    } else {
      out = FloatingPointAdderSinglePath(a, b, clk: clk).sum.packed;
    }
    final probe = FloatingPoint16();
    final expected = which == 'mul' ? 6.0 : 5.0;

    a.put(a.valuePopulator().ofDouble(2.0).value);
    b.put(b.valuePopulator().ofDouble(3.0).value);
    Simulator.setMaxSimTime(100000);
    unawaited(Simulator.run());
    var cycles = 0;
    for (var i = 0; i < 12; i++) {
      await clk.nextNegedge;
      probe.put(out.value);
      if (probe.floatingPointValue.toDouble() == expected) {
        cycles = i;
        break;
      }
    }
    await Simulator.endSimulation();
    return cycles;
  }

  test('print pipelined FP core latencies', () async {
    final lm = await latency('mul');
    await Simulator.reset();
    final la = await latency('add');
    expect(lm, greaterThanOrEqualTo(0));
    expect(la, greaterThanOrEqualTo(0));
  });
}
