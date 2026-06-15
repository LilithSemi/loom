// TDD tests for LoomVectorUnit -- written BEFORE the implementation.
//
// Tests elementwise ADD, SUB, and MUL (SwiGLU gate fuse style) in signed
// two's-complement integer arithmetic. Reference is pure Dart; hardware
// must match bit-exact.

import 'package:loom/src/hw/vector_unit.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

// ---------------------------------------------------------------------------
// Plain-Dart signed integer reference (independent of module internals)
// ---------------------------------------------------------------------------

const int vecOpAdd = 0;
const int vecOpSub = 1;
const int vecOpMul = 2;

/// Elementwise reference for ADD/SUB/MUL in plain Dart ints.
List<int> vecRef(List<int> a, List<int> b, int op) {
  if (a.length != b.length) throw ArgumentError('length mismatch');
  return [
    for (var i = 0; i < a.length; i++)
      switch (op) {
        vecOpAdd => a[i] + b[i],
        vecOpSub => a[i] - b[i],
        vecOpMul => a[i] * b[i],
        _ => a[i] + b[i], // op 3 reserved; treat as ADD (documented)
      },
  ];
}

// ---------------------------------------------------------------------------
// Pack/unpack helpers (independent of module internals)
// ---------------------------------------------------------------------------

/// Pack a list of signed [width]-bit integers into a BigInt (element 0 in
/// low bits). Values are masked to [width] bits (two's complement).
BigInt packElements(List<int> vals, int width) {
  final mask = (BigInt.one << width) - BigInt.one;
  var result = BigInt.zero;
  for (var i = vals.length - 1; i >= 0; i--) {
    result = (result << width) | (BigInt.from(vals[i]) & mask);
  }
  return result;
}

/// Unpack [count] signed [width]-bit values from a LogicValue.
/// Element 0 is in the low bits. Interprets each chunk as two's complement.
List<int> unpackElements(LogicValue val, int count, int width) {
  final mask = (BigInt.one << width) - BigInt.one;
  final threshold = BigInt.one << (width - 1);
  final modulus = BigInt.one << width;
  final raw = val.toBigInt();
  return [
    for (var i = 0; i < count; i++)
      () {
        final chunk = (raw >> (i * width)) & mask;
        final signed = chunk >= threshold ? chunk - modulus : chunk;
        return signed.toInt();
      }(),
  ];
}

// ---------------------------------------------------------------------------
// Simulation helper
// ---------------------------------------------------------------------------

Future<List<int>> runSim(
  List<int> aVec,
  List<int> bVec,
  int op, {
  int inWidth = 16,
  int accWidth = 32,
}) async {
  final lanes = aVec.length;
  assert(bVec.length == lanes);

  final aIn = Logic(name: 'a', width: lanes * inWidth);
  final bIn = Logic(name: 'b', width: lanes * inWidth);
  final opIn = Logic(name: 'op', width: 2);

  final mod = LoomVectorUnit(
    a: aIn,
    b: bIn,
    op: opIn,
    lanes: lanes,
    inWidth: inWidth,
    accWidth: accWidth,
  );
  await mod.build();

  aIn.put(packElements(aVec, inWidth));
  bIn.put(packElements(bVec, inWidth));
  opIn.put(op);

  return unpackElements(mod.y.value, lanes, accWidth);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  // --- ADD ------------------------------------------------------------------
  test('ADD lanes=3: [10,-5,100] + [3,-7,50] = [13,-12,150]', () async {
    final a = [10, -5, 100];
    final b = [3, -7, 50];
    final expected = vecRef(a, b, vecOpAdd);
    expect(expected, equals([13, -12, 150]));
    final got = await runSim(a, b, vecOpAdd);
    expect(got, equals(expected));
  });

  // --- SUB ------------------------------------------------------------------
  test('SUB lanes=2: [10,-5] - [3,-20] = [7,15]', () async {
    final a = [10, -5];
    final b = [3, -20];
    final expected = vecRef(a, b, vecOpSub);
    expect(expected, equals([7, 15]));
    final got = await runSim(a, b, vecOpSub);
    expect(got, equals(expected));
  });

  // --- MUL (signed, including max int8 magnitudes) -------------------------
  // inWidth=16 so 8-bit-magnitude values fit comfortably as signed int16.
  test('MUL lanes=4 signed: [7,-8,-128,127] * [-3,9,-128,127]', () async {
    final a = [7, -8, -128, 127];
    final b = [-3, 9, -128, 127];
    final expected = vecRef(a, b, vecOpMul);
    // Expected: [-21, -72, 16384, 16129]
    expect(expected, equals([-21, -72, 16384, 16129]));
    final got = await runSim(a, b, vecOpMul);
    expect(got, equals(expected));
  });

  // --- MUL negative result sign correctness --------------------------------
  test('MUL negative result decodes as negative, not large unsigned', () async {
    final a = [7];
    final b = [-3];
    final got = await runSim(a, b, vecOpMul);
    // -21 in a 32-bit output must decode as -21, not 2^32 - 21
    expect(got, equals([-21]));
  });

  // --- MUL large inWidth=16 near signed range ------------------------------
  test(
    'MUL inWidth=16: [30000] * [30000] = 900000000 fits in accWidth=32',
    () async {
      final a = [30000];
      final b = [30000];
      final expected = vecRef(a, b, vecOpMul);
      expect(expected, equals([900000000]));
      final got = await runSim(a, b, vecOpMul, inWidth: 16, accWidth: 32);
      expect(got, equals(expected));
    },
  );

  // --- 1-lane sanity for each op --------------------------------------------
  test('1-lane ADD: 42 + (-10) = 32', () async {
    final got = await runSim([42], [-10], vecOpAdd);
    expect(got, equals([32]));
  });

  test('1-lane SUB: 0 - 1 = -1', () async {
    final got = await runSim([0], [1], vecOpSub);
    expect(got, equals([-1]));
  });

  test('1-lane MUL: -1 * -1 = 1', () async {
    final got = await runSim([-1], [-1], vecOpMul);
    expect(got, equals([1]));
  });

  // --- validate() rejects bad params ----------------------------------------
  group('LoomVectorUnitConfig.validate()', () {
    test('valid config passes', () {
      expect(() => LoomVectorUnitConfig(lanes: 4).validate(), returnsNormally);
    });

    test('rejects lanes=0', () {
      expect(
        () => LoomVectorUnitConfig(lanes: 0).validate(),
        throwsArgumentError,
      );
    });

    test('rejects inWidth < 2', () {
      expect(
        () => LoomVectorUnitConfig(lanes: 1, inWidth: 1).validate(),
        throwsArgumentError,
      );
    });

    test('rejects accWidth < 2*inWidth', () {
      expect(
        () =>
            LoomVectorUnitConfig(lanes: 1, inWidth: 8, accWidth: 15).validate(),
        throwsArgumentError,
      );
    });

    test('rejects accWidth < 2*inWidth (boundary: accWidth=2*inWidth-1)', () {
      // inWidth=16 -> need accWidth >= 32; 31 must fail
      expect(
        () => LoomVectorUnitConfig(
          lanes: 2,
          inWidth: 16,
          accWidth: 31,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('accepts accWidth == 2*inWidth exactly', () {
      expect(
        () => LoomVectorUnitConfig(
          lanes: 2,
          inWidth: 16,
          accWidth: 32,
        ).validate(),
        returnsNormally,
      );
    });
  });

  // --- SV emission ----------------------------------------------------------
  test('SV emission: non-empty and contains module name', () async {
    final aIn = Logic(name: 'a_sv', width: 2 * 16);
    final bIn = Logic(name: 'b_sv', width: 2 * 16);
    final opIn = Logic(name: 'op_sv', width: 2);
    final mod = LoomVectorUnit(a: aIn, b: bIn, op: opIn, lanes: 2);
    await mod.build();
    final sv = mod.generateSynth();
    expect(sv, isNotEmpty);
    expect(sv, contains('LoomVectorUnit'));
  });
}
