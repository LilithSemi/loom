// Tests for LoomRequantize. The requantization operation is:
//   1. prod = acc * mult  (signed result, acc signed, mult unsigned nonneg)
//   2. rounded = signAwareRoundHalfAwayFromZero(prod, shift)
//   3. out = saturate(rounded, [-(2^(outWidth-1)-1), 2^(outWidth-1)-1])
//
// Symmetric int8 range: [-127, 127].  Zero-point = 0.
// Round-half-away-from-zero: 1.5->2, -1.5->-2, 0.5->1, -0.5->-1.

import 'dart:async';

import 'package:loom/src/hw/requantize.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

// Plain-Dart reference implementation. Must match HW bit-for-bit.

/// Sign-aware round-half-away-from-zero right shift.
///
/// shift == 0: returns prod unchanged.
/// prod >= 0:  (prod + (1 << (shift-1))) >> shift   (arithmetic = logical here)
/// prod <  0:  -(((-prod) + (1 << (shift-1))) >> shift)
int _roundShift(int prod, int shift) {
  if (shift == 0) return prod;
  final bias = 1 << (shift - 1);
  if (prod >= 0) {
    return (prod + bias) >> shift;
  } else {
    // Work on magnitude, reapply sign.
    return -((-prod + bias) >> shift);
  }
}

/// Dart reference: acc * mult then shift-round then saturate.
///
/// [acc] is treated as signed, [mult] as unsigned nonneg.
int requantRef(int acc, int mult, int shift, {int outWidth = 8}) {
  final prod = acc * mult;
  final rounded = _roundShift(prod, shift);
  final lo = -(1 << (outWidth - 1)) + 1; // -127 for 8-bit
  final hi = (1 << (outWidth - 1)) - 1; // 127 for 8-bit
  if (rounded < lo) return lo;
  if (rounded > hi) return hi;
  return rounded;
}

// Simulation helper

/// Interpret a [width]-bit LogicValue as a signed two's-complement integer.
int signedFromLogic(LogicValue v, int width) {
  final raw = v.toBigInt();
  final threshold = BigInt.one << (width - 1);
  final modulus = BigInt.one << width;
  if (raw >= threshold) return (raw - modulus).toInt();
  return raw.toInt();
}

/// Drive the module inputs and return the signed output.
Future<int> runSim(
  int acc,
  int mult,
  int shift, {
  int accWidth = 32,
  int multWidth = 16,
  int shiftWidth = 6,
  int outWidth = 8,
}) async {
  final accIn = Logic(name: 'acc', width: accWidth);
  final multIn = Logic(name: 'mult', width: multWidth);
  final shiftIn = Logic(name: 'shift', width: shiftWidth);

  final mod = LoomRequantize(
    acc: accIn,
    mult: multIn,
    shift: shiftIn,
    accWidth: accWidth,
    multWidth: multWidth,
    shiftWidth: shiftWidth,
    outWidth: outWidth,
  );
  await mod.build();

  // Mask inputs to their declared widths (two's complement for acc).
  final accMask = BigInt.from((1 << accWidth) - 1);
  final multMask = BigInt.from((1 << multWidth) - 1);
  final shiftMask = BigInt.from((1 << shiftWidth) - 1);

  accIn.put(BigInt.from(acc) & accMask);
  multIn.put(BigInt.from(mult) & multMask);
  shiftIn.put(BigInt.from(shift) & shiftMask);

  return signedFromLogic(mod.out.value, outWidth);
}

// Tests

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  // shift=0 passthrough with saturation
  group('shift=0 passthrough', () {
    test('acc=50 mult=1 shift=0 -> 50', () async {
      expect(requantRef(50, 1, 0), equals(50));
      expect(await runSim(50, 1, 0), equals(50));
    });

    test('acc=200 mult=1 shift=0 -> saturate to 127', () async {
      expect(requantRef(200, 1, 0), equals(127));
      expect(await runSim(200, 1, 0), equals(127));
    });

    test('acc=-200 mult=1 shift=0 -> saturate to -127', () async {
      expect(requantRef(-200, 1, 0), equals(-127));
      expect(await runSim(-200, 1, 0), equals(-127));
    });
  });

  // Basic scaling
  group('basic scaling', () {
    test('acc=1000 mult=1 shift=3 -> round(125.0)=125', () async {
      expect(requantRef(1000, 1, 3), equals(125));
      expect(await runSim(1000, 1, 3), equals(125));
    });

    test('acc=1000 mult=1 shift=4 -> round(62.5)=63 (half-away)', () async {
      // 1000/16 = 62.5 -> round-half-away-from-zero = 63
      expect(requantRef(1000, 1, 4), equals(63));
      expect(await runSim(1000, 1, 4), equals(63));
    });

    test('acc=-1000 mult=1 shift=4 -> -63 (half-away negative)', () async {
      // -1000/16 = -62.5 -> round-half-away-from-zero = -63
      expect(requantRef(-1000, 1, 4), equals(-63));
      expect(await runSim(-1000, 1, 4), equals(-63));
    });
  });

  // Rounding ties
  group('rounding ties (half-away-from-zero)', () {
    test('acc=3 mult=1 shift=1 -> round(1.5)=2', () async {
      // prod=3, shift=1: bias=1, (3+1)>>1=2
      expect(requantRef(3, 1, 1), equals(2));
      expect(await runSim(3, 1, 1), equals(2));
    });

    test('acc=-3 mult=1 shift=1 -> round(-1.5)=-2 (half-away)', () async {
      // prod=-3, shift=1: magnitude=3, (3+1)>>1=2, reapply sign => -2
      expect(requantRef(-3, 1, 1), equals(-2));
      expect(await runSim(-3, 1, 1), equals(-2));
    });

    test('acc=1 mult=1 shift=1 -> round(0.5)=1 (half-away)', () async {
      // prod=1, shift=1: bias=1, (1+1)>>1=1
      expect(requantRef(1, 1, 1), equals(1));
      expect(await runSim(1, 1, 1), equals(1));
    });

    test('acc=-1 mult=1 shift=1 -> round(-0.5)=-1 (half-away)', () async {
      // prod=-1, shift=1: magnitude=1, (1+1)>>1=1, reapply sign => -1
      expect(requantRef(-1, 1, 1), equals(-1));
      expect(await runSim(-1, 1, 1), equals(-1));
    });

    test('acc=5 mult=1 shift=1 -> round(2.5)=3 (half-away)', () async {
      // prod=5, bias=1, (5+1)>>1=3
      expect(requantRef(5, 1, 1), equals(3));
      expect(await runSim(5, 1, 1), equals(3));
    });

    test('acc=-5 mult=1 shift=1 -> -3 (half-away)', () async {
      expect(requantRef(-5, 1, 1), equals(-3));
      expect(await runSim(-5, 1, 1), equals(-3));
    });
  });

  // Real multiplier
  group('real multiplier', () {
    test(
      'acc=100 mult=300 shift=5 -> round(937.5)=938 then saturate to 127',
      () async {
        // prod=30000, shift=5: bias=16, (30000+16)>>5 = 30016>>5 = 938
        // 938 > 127 so saturated result is 127
        final ref = requantRef(100, 300, 5);
        expect(ref, equals(127)); // saturated
        expect(await runSim(100, 300, 5), equals(127));
      },
    );

    test('acc=100 mult=300 shift=8 -> round(117.1875)=117', () async {
      // prod=30000, shift=8: bias=128, (30000+128)>>8=30128>>8=117
      expect(requantRef(100, 300, 8), equals(117));
      expect(await runSim(100, 300, 8), equals(117));
    });

    test('acc=-100 mult=300 shift=8 -> -117', () async {
      expect(requantRef(-100, 300, 8), equals(-117));
      expect(await runSim(-100, 300, 8), equals(-117));
    });
  });

  // Saturation after scaling
  group('saturation after scaling', () {
    test('large positive -> clamp to 127', () async {
      // acc=10000, mult=1000, shift=10: prod=10M, rounded=10M/1024~9765 -> 127
      expect(requantRef(10000, 1000, 10), equals(127));
      expect(await runSim(10000, 1000, 10), equals(127));
    });

    test('large negative -> clamp to -127', () async {
      expect(requantRef(-10000, 1000, 10), equals(-127));
      expect(await runSim(-10000, 1000, 10), equals(-127));
    });

    test('exactly 127 does not saturate', () async {
      // acc=127, mult=1, shift=0 -> 127 (no clamp)
      expect(requantRef(127, 1, 0), equals(127));
      expect(await runSim(127, 1, 0), equals(127));
    });

    test('exactly -127 does not saturate', () async {
      expect(requantRef(-127, 1, 0), equals(-127));
      expect(await runSim(-127, 1, 0), equals(-127));
    });

    test('128 saturates to 127', () async {
      expect(requantRef(128, 1, 0), equals(127));
      expect(await runSim(128, 1, 0), equals(127));
    });

    test('-128 saturates to -127 (asymmetric clamp)', () async {
      expect(requantRef(-128, 1, 0), equals(-127));
      expect(await runSim(-128, 1, 0), equals(-127));
    });
  });

  // Spread of fixed literals
  group('spread of fixed literals vs reference', () {
    // Each entry: (acc, mult, shift).
    final cases = [
      (0, 1, 0),
      (1, 1, 0),
      (-1, 1, 0),
      (64, 2, 1),
      (-64, 2, 1),
      (100, 1, 1),
      (-100, 1, 1),
      (255, 1, 2),
      (-255, 1, 2),
      (1000, 8, 6),
      (-1000, 8, 6),
      (32767, 256, 15),
      (-32767, 256, 15),
      (127 * 127, 1, 7),
      (-(127 * 127), 1, 7),
    ];

    for (final (acc, mult, shift) in cases) {
      test('acc=$acc mult=$mult shift=$shift matches reference', () async {
        final expected = requantRef(acc, mult, shift);
        expect(
          await runSim(acc, mult, shift),
          equals(expected),
          reason: 'acc=$acc mult=$mult shift=$shift ref=$expected',
        );
      });
    }
  });

  // Config validate()
  group('LoomRequantizeConfig.validate()', () {
    test('valid config passes', () {
      expect(() => LoomRequantizeConfig().validate(), returnsNormally);
    });

    test('rejects accWidth=0', () {
      expect(
        () => LoomRequantizeConfig(accWidth: 0).validate(),
        throwsArgumentError,
      );
    });

    test('rejects multWidth=0', () {
      expect(
        () => LoomRequantizeConfig(multWidth: 0).validate(),
        throwsArgumentError,
      );
    });

    test('rejects shiftWidth=0', () {
      expect(
        () => LoomRequantizeConfig(shiftWidth: 0).validate(),
        throwsArgumentError,
      );
    });

    test('rejects outWidth < 2', () {
      expect(
        () => LoomRequantizeConfig(outWidth: 1).validate(),
        throwsArgumentError,
      );
    });

    test('rejects multWidth > accWidth (unreasonably large multiplier)', () {
      expect(
        () => LoomRequantizeConfig(accWidth: 8, multWidth: 32).validate(),
        throwsArgumentError,
      );
    });
  });

  // Pipelined requantize: bit-exact vs combinational, just delayed by latency.
  Future<int> runSimPipelined(
    int acc,
    int mult,
    int shift,
    int latency, {
    int accWidth = 32,
    int multWidth = 16,
    int shiftWidth = 6,
    int outWidth = 8,
  }) async {
    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final accIn = Logic(name: 'acc', width: accWidth);
    final multIn = Logic(name: 'mult', width: multWidth);
    final shiftIn = Logic(name: 'shift', width: shiftWidth);

    final mod = LoomRequantize(
      acc: accIn,
      mult: multIn,
      shift: shiftIn,
      accWidth: accWidth,
      multWidth: multWidth,
      shiftWidth: shiftWidth,
      outWidth: outWidth,
      latency: latency,
      clk: clk,
      reset: reset,
    );
    await mod.build();
    expect(mod.latency, equals(latency));

    Simulator.setMaxSimTime(100000);
    unawaited(Simulator.run());

    final accMask = BigInt.from((1 << accWidth) - 1);
    final multMask = BigInt.from((1 << multWidth) - 1);
    final shiftMask = BigInt.from((1 << shiftWidth) - 1);

    reset.inject(1);
    accIn.inject(0);
    multIn.inject(0);
    shiftIn.inject(0);
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);

    accIn.inject(BigInt.from(acc) & accMask);
    multIn.inject(BigInt.from(mult) & multMask);
    shiftIn.inject(BigInt.from(shift) & shiftMask);
    for (var i = 0; i < latency; i++) {
      await clk.nextPosedge;
    }
    final got = signedFromLogic(mod.out.value, outWidth);
    await Simulator.endSimulation();
    return got;
  }

  group('pipelined latency bit-exact', () {
    final cases = <(int, int, int)>[
      (50, 1, 0),
      (200, 1, 0),
      (-200, 1, 0),
      (1000, 7, 3),
      (-1000, 7, 3),
      (66000, 13, 6),
      (-66000, 13, 6),
      (3, 5, 1),
      (-3, 5, 1),
    ];
    for (final latency in [1, 2]) {
      for (final (acc, mult, shift) in cases) {
        test('latency=$latency acc=$acc mult=$mult shift=$shift', () async {
          final expected = requantRef(acc, mult, shift);
          final got = await runSimPipelined(acc, mult, shift, latency);
          expect(got, equals(expected));
        });
      }
    }
  });

  test('rejects latency > 2', () {
    expect(
      () => LoomRequantizeConfig(latency: 3).validate(),
      throwsArgumentError,
    );
  });

  test('throws if latency>=1 without clk', () {
    final accIn = Logic(name: 'a', width: 32);
    final multIn = Logic(name: 'm', width: 16);
    final shiftIn = Logic(name: 's', width: 6);
    expect(
      () =>
          LoomRequantize(acc: accIn, mult: multIn, shift: shiftIn, latency: 1),
      throwsArgumentError,
    );
  });

  // SV emission
  test('SV emission: non-empty and contains module name', () async {
    final accIn = Logic(name: 'acc_sv', width: 32);
    final multIn = Logic(name: 'mult_sv', width: 16);
    final shiftIn = Logic(name: 'shift_sv', width: 6);

    final mod = LoomRequantize(acc: accIn, mult: multIn, shift: shiftIn);
    await mod.build();
    final sv = mod.generateSynth();
    expect(sv, isNotEmpty);
    expect(sv, contains('LoomRequantize'));
  });
}
