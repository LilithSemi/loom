// LoomRopeVec: vector-level RoPE (HF half-split) built from per-pair LoomRope,
// matching the golden applyRopeHead within fp16 tolerance. cos/sin are supplied
// precomputed (from the model's baked freq_cis table). Combinational.

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:loom/src/golden/attention.dart';
import 'package:loom/src/hw/rope_vec.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('vector RoPE matches golden applyRopeHead', () async {
    const headDim = 8;
    const pos = 5;
    const theta = 10000.0;
    final half = headDim ~/ 2;

    final x = [0.5, -1.0, 2.0, 1.0, -0.5, 1.5, -2.0, 0.25];
    final golden = Float64List.fromList(x);
    applyRopeHead(golden, pos, theta);

    final clk = SimpleClockGenerator(10).clk;
    final dut = LoomRopeVec(headDim: headDim);

    final fp = FloatingPoint16();
    int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
    final outFp = FloatingPoint16();

    final xs = [
      for (var i = 0; i < headDim; i++)
        Logic(name: 'x$i', width: 16)..inject(e(x[i])),
    ];
    final coss = <Logic>[];
    final sins = <Logic>[];
    for (var j = 0; j < half; j++) {
      final invFreq = 1.0 / math.pow(theta, (2 * j) / headDim);
      final angle = pos * invFreq;
      coss.add(Logic(name: 'cos$j', width: 16)..inject(e(math.cos(angle))));
      sins.add(Logic(name: 'sin$j', width: 16)..inject(e(math.sin(angle))));
    }

    for (var i = 0; i < headDim; i++) {
      dut.input('x$i').srcConnection! <= xs[i];
    }
    for (var j = 0; j < half; j++) {
      dut.input('cos$j').srcConnection! <= coss[j];
      dut.input('sin$j').srcConnection! <= sins[j];
    }
    await dut.build();

    Simulator.setMaxSimTime(200000);
    unawaited(Simulator.run());
    await clk.nextNegedge;

    final got = <double>[];
    for (var i = 0; i < headDim; i++) {
      outFp.put(dut.output('y$i').value);
      got.add(outFp.floatingPointValue.toDouble());
    }
    await Simulator.endSimulation();

    for (var i = 0; i < headDim; i++) {
      expect(
        got[i],
        closeTo(golden[i], 0.02 + golden[i].abs() * 0.05),
        reason: 'y[$i]=${got[i]} vs ${golden[i]}',
      );
    }
  });
}
