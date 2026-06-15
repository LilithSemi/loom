// LoomFpResidual diff-test: fp16 a+b vs the golden, within fp16 tolerance.

import 'package:loom/src/hw/fp_residual.dart';
import 'package:rohd/rohd.dart';
import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  test('out = a + b in fp16, within tolerance', () async {
    final a = Logic(name: 'a', width: 16);
    final b = Logic(name: 'b', width: 16);
    final dut = LoomFpResidual();
    dut.input('a').srcConnection! <= a;
    dut.input('b').srcConnection! <= b;
    await dut.build();

    final enc = FloatingPoint16();
    final outFp = FloatingPoint16();

    for (final (x, y) in [
      (1.5, 2.25),
      (-3.0, 0.5),
      (10.0, -4.0),
      (0.125, 0.125),
      (-1.0, -2.0),
    ]) {
      a.put(enc.valuePopulator().ofDouble(x).value);
      b.put(enc.valuePopulator().ofDouble(y).value);
      outFp.put(dut.output('sum').value);
      final got = outFp.floatingPointValue.toDouble();
      expect(
        got,
        closeTo(x + y, (x + y).abs() * 0.02 + 0.01),
        reason: '$x + $y',
      );
    }
  });
}
