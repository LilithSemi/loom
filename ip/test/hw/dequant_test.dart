// LoomDequant test: pipelined int32 acc -> fp16 (acc * rowScale * actScale),
// including a large accumulator that would overflow an fp16 intermediate.

import 'dart:async';

import 'package:loom/src/hw/dequant.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test(
    'dequant computes acc * rowScale * actScale in fp16 (pipelined)',
    () async {
      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      final validIn = Logic(name: 'valid_in');
      final acc = Logic(name: 'acc', width: 32);
      final rowScale = Logic(name: 'row_scale', width: 16);
      final actScale = Logic(name: 'act_scale', width: 16);

      final dut = LoomDequant();
      for (final (p, s) in [
        ('clk', clk),
        ('reset', reset),
        ('valid_in', validIn),
        ('acc', acc),
        ('row_scale', rowScale),
        ('act_scale', actScale),
      ]) {
        dut.input(p).srcConnection! <= s;
      }
      await dut.build();

      final fp = FloatingPoint16();
      int e(double d) => fp.valuePopulator().ofDouble(d).value.toInt();
      final outFp = FloatingPoint16();

      final cases = <(int, double, double)>[
        (1000, 0.01, 0.02),
        (-2048, 0.005, 0.03),
        (500000, 0.001, 0.0005), // ~10^5 acc, overflows fp16 if not widened
        (-1200000, 0.0008, 0.0006),
        (0, 0.01, 0.02),
      ];

      validIn.inject(0);
      acc.inject(0);
      rowScale.inject(0);
      actScale.inject(0);
      reset.inject(1);
      Simulator.setMaxSimTime(1000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;

      // Stream all cases in, one per cycle. Collect valid_out results in order.
      final got = <double>[];
      final fut = () async {
        var seen = 0;
        while (seen < cases.length) {
          await clk.nextNegedge;
          if (dut.output('valid_out').value.toBool()) {
            outFp.put(dut.output('y').value);
            got.add(outFp.floatingPointValue.toDouble());
            seen++;
          }
        }
      }();

      for (final (a, rs, as_) in cases) {
        validIn.inject(1);
        acc.inject(a & 0xFFFFFFFF);
        rowScale.inject(e(rs));
        actScale.inject(e(as_));
        await clk.nextPosedge;
      }
      validIn.inject(0);
      await fut;
      await Simulator.endSimulation();

      expect(got.length, equals(cases.length));
      for (var i = 0; i < cases.length; i++) {
        final (a, rs, as_) = cases[i];
        final golden = a * rs * as_;
        expect(
          got[i],
          closeTo(golden, 0.02 + golden.abs() * 0.05),
          reason: 'acc=$a rs=$rs as=$as_ y=${got[i]} vs $golden',
        );
      }
    },
  );

  test(
    'LoomDequant exposes an fp32 pre-narrow product that narrows to y',
    () async {
      // acc=100, row_scale=fp16(0.1), act_scale=fp16(0.02) -> ~0.2
      final got = await runDequant(acc: 100, rowScaleF: 0.1, actScaleF: 0.02);
      expect(got.validOut, isTrue);
      // got.yAcc is the fp32 bits. Narrowing it to fp16 must equal got.y (fp16 bits).
      expect(
        fp32BitsToFp16Bits(got.yAcc),
        equals(got.y),
        reason: 'y_acc must be the same value as y at higher precision',
      );
    },
  );
}

/// Drives a single (acc, rowScale, actScale) sample through a fresh
/// [LoomDequant] instance and returns the fp16 `y`, the fp32 `y_acc`, and
/// whether `valid_out` fired on the cycle both were sampled.
Future<({int y, int yAcc, bool validOut})> runDequant({
  required int acc,
  required double rowScaleF,
  required double actScaleF,
}) async {
  final clk = SimpleClockGenerator(10).clk;
  final reset = Logic(name: 'reset');
  final validIn = Logic(name: 'valid_in');
  final accL = Logic(name: 'acc', width: 32);
  final rowScale = Logic(name: 'row_scale', width: 16);
  final actScale = Logic(name: 'act_scale', width: 16);

  final dut = LoomDequant();
  for (final (p, s) in [
    ('clk', clk),
    ('reset', reset),
    ('valid_in', validIn),
    ('acc', accL),
    ('row_scale', rowScale),
    ('act_scale', actScale),
  ]) {
    dut.input(p).srcConnection! <= s;
  }
  await dut.build();

  final fp16 = FloatingPoint16();
  int e(double d) => fp16.valuePopulator().ofDouble(d).value.toInt();

  validIn.inject(0);
  accL.inject(0);
  rowScale.inject(0);
  actScale.inject(0);
  reset.inject(1);
  Simulator.setMaxSimTime(100000);
  unawaited(Simulator.run());
  await clk.nextPosedge;
  await clk.nextPosedge;
  reset.inject(0);
  await clk.nextPosedge;

  validIn.inject(1);
  accL.inject(acc & 0xFFFFFFFF);
  rowScale.inject(e(rowScaleF));
  actScale.inject(e(actScaleF));
  await clk.nextPosedge;
  validIn.inject(0);

  int? y;
  int? yAcc;
  var validOut = false;
  for (var i = 0; i < LoomDequant.latency + 2; i++) {
    await clk.nextNegedge;
    if (dut.output('valid_out').value.toBool()) {
      y = dut.output('y').value.toInt();
      yAcc = dut.output('y_acc').value.toInt();
      validOut = true;
      break;
    }
  }
  await Simulator.endSimulation();

  return (y: y ?? 0, yAcc: yAcc ?? 0, validOut: validOut);
}

/// Software oracle for narrowing an fp32 bit pattern to fp16, using the same
/// rohd_hcl [FloatingPointConverter] hardware the DUT uses internally, so
/// rounding is bit-identical.
int fp32BitsToFp16Bits(int fp32Bits) {
  final source = FloatingPoint32()..put(LogicValue.ofInt(fp32Bits, 32));
  final dest = FloatingPoint16();
  FloatingPointConverter(source, dest);
  return dest.packed.value.toInt();
}
