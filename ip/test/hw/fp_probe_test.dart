// Confirms rohd_hcl's floating-point cores (building blocks for RMSNorm,
// softmax, SiLU, residual) work in Loom's build, fp16 sqrt + add, value-exact
// within fp16 tolerance. No FP divider exists in rohd_hcl, so division uses
// Newton-Raphson reciprocal or fixed-point instead.

import 'package:rohd_hcl/rohd_hcl.dart';
import 'package:test/test.dart';

void main() {
  test('fp16 sqrt via FloatingPointSqrtSimple', () async {
    final a = FloatingPoint16();
    final dut = FloatingPointSqrtSimple(a);
    await dut.build();
    a.put(a.valuePopulator().ofDouble(4.0));
    expect(dut.sqrt.floatingPointValue.toDouble(), closeTo(2.0, 0.05));
  });

  test('fp16 add via FloatingPointAdderSinglePath', () async {
    final a = FloatingPoint16();
    final b = FloatingPoint16();
    final dut = FloatingPointAdderSinglePath(a, b);
    await dut.build();
    a.put(a.valuePopulator().ofDouble(1.5));
    b.put(b.valuePopulator().ofDouble(2.25));
    expect(dut.sum.floatingPointValue.toDouble(), closeTo(3.75, 0.05));
  });

  test('fp16 multiply via FloatingPointMultiplierSimple', () async {
    final a = FloatingPoint16();
    final b = FloatingPoint16();
    final dut = FloatingPointMultiplierSimple(a, b);
    await dut.build();
    a.put(a.valuePopulator().ofDouble(3.0));
    b.put(b.valuePopulator().ofDouble(-2.5));
    expect(dut.product.floatingPointValue.toDouble(), closeTo(-7.5, 0.1));
  });
}
