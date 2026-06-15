@TestOn('vm')
library;

import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  test('valid overlay config validates', () {
    final c = LoomConfig(
      strategy: LoomStrategy.overlay,
      numerics: {LoomNumeric.int8, LoomNumeric.fp16},
      target: LoomTarget.sim,
    );
    expect(c.validate, returnsNormally);
  });

  test('empty numerics is rejected', () {
    final c = LoomConfig(
      strategy: LoomStrategy.overlay,
      numerics: {},
      target: LoomTarget.sim,
    );
    expect(c.validate, throwsArgumentError);
  });

  test('baking other than auto requires spatial strategy', () {
    final c = LoomConfig(
      strategy: LoomStrategy.overlay,
      numerics: {LoomNumeric.int8},
      target: LoomTarget.sim,
      baking: LoomBaking.romBaked,
    );
    expect(c.validate, throwsArgumentError);
  });

  test('spatial config may set a concrete baking', () {
    final c = LoomConfig(
      strategy: LoomStrategy.spatial,
      numerics: {LoomNumeric.int4},
      target: LoomTarget.fpgaIcesugar,
      baking: LoomBaking.folded,
    );
    expect(c.validate, returnsNormally);
  });
}
