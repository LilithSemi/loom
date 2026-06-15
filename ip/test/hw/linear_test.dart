// TDD tests for LoomLinear -- written BEFORE the implementation.
//
// LoomLinear composes LoomMatmul (clocked, streaming tile accumulator) with
// peRows LoomRequantize units (combinational, one per output row).
//
// Timing contract:
//   - Stream weight/activation tiles through the matmul the same way as
//     matmul_test.dart: assert first on tile 0, last on the last tile.
//   - When matmul.resultValid pulses (one cycle), outValid pulses and out holds
//     the requantized int8 result for all peRows rows.
//   - out[r] at r*outWidth bits, signed two's-complement int8.
//
// References:
//   - dartMatVec: plain signed integer matrix-vector multiply (acc).
//   - requantRef: round-half-away-from-zero shift then saturate (copied from
//     requantize_test.dart for independence).

import 'dart:async';

import 'package:loom/src/hw/linear.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

// ---------------------------------------------------------------------------
// Plain-Dart reference implementations (independent of hardware)
// ---------------------------------------------------------------------------

/// Compute y[r] = sum_c W[r,c] * x[c] in plain Dart integers.
/// W is peRows x K (row-major list-of-rows), x is K-element.
List<int> dartMatVec(List<List<int>> w, List<int> x) {
  final rows = w.length;
  final cols = x.length;
  return [
    for (var r = 0; r < rows; r++)
      [for (var c = 0; c < cols; c++) w[r][c] * x[c]].fold(0, (a, b) => a + b),
  ];
}

/// Sign-aware round-half-away-from-zero right shift.
int _roundShift(int prod, int shift) {
  if (shift == 0) return prod;
  final bias = 1 << (shift - 1);
  if (prod >= 0) {
    return (prod + bias) >> shift;
  } else {
    return -((-prod + bias) >> shift);
  }
}

/// Reference requantize: acc * mult then shift-round then saturate to int8.
/// Matches the hardware LoomRequantize exactly.
int requantRef(int acc, int mult, int shift, {int outWidth = 8}) {
  final prod = acc * mult;
  final rounded = _roundShift(prod, shift);
  final lo = -(1 << (outWidth - 1)) + 1; // -127 for 8-bit
  final hi = (1 << (outWidth - 1)) - 1; // 127 for 8-bit
  if (rounded < lo) return lo;
  if (rounded > hi) return hi;
  return rounded;
}

// ---------------------------------------------------------------------------
// Pack helpers
// ---------------------------------------------------------------------------

/// Pack a flat list of signed [width]-bit values into a BigInt, element 0 low.
BigInt packSignedFlat(List<int> vals, int width) {
  final mask = BigInt.from((1 << width) - 1);
  var result = BigInt.zero;
  for (var i = vals.length - 1; i >= 0; i--) {
    result = (result << width) | (BigInt.from(vals[i]) & mask);
  }
  return result;
}

/// Pack a peRows x peCols weight tile (row-major) into the wTile bus.
BigInt packWTile(List<List<int>> wTile, int inWidth) {
  final flat = [for (final row in wTile) ...row];
  return packSignedFlat(flat, inWidth);
}

/// Pack a peCols-element activation sub-tile into the xTile bus.
BigInt packXTile(List<int> xTile, int inWidth) =>
    packSignedFlat(xTile, inWidth);

/// Pack peRows unsigned mult values (each multWidth bits) into the rowMult bus.
/// mult[r] at r*multWidth, element 0 in low bits.
BigInt packRowMult(List<int> mults, int multWidth) {
  final mask = BigInt.from((1 << multWidth) - 1);
  var result = BigInt.zero;
  for (var i = mults.length - 1; i >= 0; i--) {
    result = (result << multWidth) | (BigInt.from(mults[i]) & mask);
  }
  return result;
}

/// Unpack peRows signed outWidth-bit values from the out bus. Element 0 low.
List<int> unpackOut(LogicValue outVal, int rows, int outWidth) {
  final mask = BigInt.from((1 << outWidth) - 1);
  final threshold = BigInt.one << (outWidth - 1);
  final modulus = BigInt.one << outWidth;
  final raw = outVal.toBigInt();
  return [
    for (var r = 0; r < rows; r++)
      () {
        final chunk = (raw >> (r * outWidth)) & mask;
        final signed = chunk >= threshold ? chunk - modulus : chunk;
        return signed.toInt();
      }(),
  ];
}

// ---------------------------------------------------------------------------
// Sim harness
// ---------------------------------------------------------------------------

/// Run one complete linear-layer block through LoomLinear.
///
/// [wMat] is peRows x K (row-major). [x] is K elements. K must be a multiple
/// of peCols. [mults] has peRows elements (per-row mult). [shift] is shared.
/// Streams K/peCols tiles, asserts first on tile 0, last on the last tile,
/// then awaits outValid and returns the out contents as signed int8 values.
Future<List<int>> runLinear(
  List<List<int>> wMat,
  List<int> x,
  int peCols,
  List<int> mults,
  int shift, {
  int inWidth = 8,
  int accWidth = 32,
  int multWidth = 16,
  int shiftWidth = 6,
  int outWidth = 8,
}) async {
  final peRows = wMat.length;
  final k = x.length;
  assert(k % peCols == 0, 'K must be a multiple of peCols');
  assert(mults.length == peRows, 'mults.length must equal peRows');
  final numTiles = k ~/ peCols;

  final clk = SimpleClockGenerator(10).clk;
  final reset = Logic(name: 'reset');
  final wTilePort = Logic(name: 'wTile', width: peRows * peCols * inWidth);
  final xTilePort = Logic(name: 'xTile', width: peCols * inWidth);
  final validPort = Logic(name: 'valid');
  final firstPort = Logic(name: 'first');
  final lastPort = Logic(name: 'last');
  final rowMultPort = Logic(name: 'rowMult', width: peRows * multWidth);
  final shiftPort = Logic(name: 'shift', width: shiftWidth);

  final mod = LoomLinear(
    clk: clk,
    reset: reset,
    wTile: wTilePort,
    xTile: xTilePort,
    valid: validPort,
    first: firstPort,
    last: lastPort,
    rowMult: rowMultPort,
    shift: shiftPort,
    peRows: peRows,
    peCols: peCols,
    inWidth: inWidth,
    accWidth: accWidth,
    multWidth: multWidth,
    shiftWidth: shiftWidth,
    outWidth: outWidth,
  );
  await mod.build();

  Simulator.setMaxSimTime(500000);
  unawaited(Simulator.run());

  // Idle defaults.
  reset.inject(0);
  wTilePort.inject(0);
  xTilePort.inject(0);
  validPort.inject(0);
  firstPort.inject(0);
  lastPort.inject(0);
  rowMultPort.inject(packRowMult(mults, multWidth));
  shiftPort.inject(BigInt.from(shift));

  // Brief reset pulse.
  reset.inject(1);
  await clk.nextPosedge;
  await clk.nextPosedge;
  reset.inject(0);
  await clk.nextPosedge;

  // Stream tiles.
  for (var t = 0; t < numTiles; t++) {
    final isFirst = t == 0;
    final isLast = t == numTiles - 1;

    final wTileRows = [
      for (var r = 0; r < peRows; r++)
        [for (var c = 0; c < peCols; c++) wMat[r][t * peCols + c]],
    ];
    final xTileVals = [for (var c = 0; c < peCols; c++) x[t * peCols + c]];

    wTilePort.inject(packWTile(wTileRows, inWidth));
    xTilePort.inject(packXTile(xTileVals, inWidth));
    validPort.inject(1);
    firstPort.inject(isFirst ? 1 : 0);
    lastPort.inject(isLast ? 1 : 0);
    await clk.nextPosedge;
  }

  // De-assert tile inputs.
  validPort.inject(0);
  firstPort.inject(0);
  lastPort.inject(0);

  // Wait for outValid (one-cycle pulse, should be up right after last tile).
  var guard = 0;
  while (!mod.outValid.value.toBool() && guard++ < (numTiles + 4)) {
    await clk.nextPosedge;
  }

  if (!mod.outValid.value.toBool()) {
    await Simulator.endSimulation();
    throw StateError('outValid never asserted (guard exhausted)');
  }

  final result = unpackOut(mod.out.value, peRows, outWidth);
  await Simulator.endSimulation();
  return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  // -------------------------------------------------------------------------
  // Case 1a: peRows=2, peCols=2, K=4, mult=[1,1], shift=0 -> passthrough
  // W=[[1,2,3,4],[5,6,7,8]], x=[1,1,1,1] -> acc=[10,26] -> out=[10,26]
  // -------------------------------------------------------------------------
  test('2x2 PE K=4 mult=[1,1] shift=0: out=[10,26] (passthrough)', () async {
    final w = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ];
    final x = [1, 1, 1, 1];
    final mults = [1, 1];
    const shift = 0;

    final acc = dartMatVec(w, x);
    expect(acc, equals([10, 26]));

    final expected = [
      requantRef(acc[0], mults[0], shift),
      requantRef(acc[1], mults[1], shift),
    ];
    expect(expected, equals([10, 26]));

    final got = await runLinear(w, x, 2, mults, shift);
    expect(got, equals(expected));
  });

  // -------------------------------------------------------------------------
  // Case 1b: same acc, shift=2 -> out=[3,7] (2.5->3, 6.5->7 half-away)
  // -------------------------------------------------------------------------
  test(
    '2x2 PE K=4 mult=[1,1] shift=2: requantRef half-away rounding',
    () async {
      final w = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ];
      final x = [1, 1, 1, 1];
      final mults = [1, 1];
      const shift = 2;

      // 10/4=2.5 -> round-half-away=3, 26/4=6.5 -> 7
      final acc = dartMatVec(w, x);
      final expected = [
        requantRef(acc[0], mults[0], shift),
        requantRef(acc[1], mults[1], shift),
      ];
      expect(expected, equals([3, 7]));

      final got = await runLinear(w, x, 2, mults, shift);
      expect(got, equals(expected));
    },
  );

  // -------------------------------------------------------------------------
  // Case 2: per-row DIFFERENT multipliers: mult=[2,1], shift=1
  // acc=[10,26]: out[0]=requantRef(10,2,1), out[1]=requantRef(26,1,1)
  // Confirms each row uses its own mult.
  // -------------------------------------------------------------------------
  test(
    '2x2 PE K=4 per-row diff mults [2,1] shift=1: row-independent requant',
    () async {
      final w = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ];
      final x = [1, 1, 1, 1];
      final mults = [2, 1];
      const shift = 1;

      final acc = dartMatVec(w, x); // [10, 26]
      // out[0] = requantRef(10, 2, 1) = round(20/2) = 10
      // out[1] = requantRef(26, 1, 1) = round(26/2) = 13
      final expected = [
        requantRef(acc[0], mults[0], shift),
        requantRef(acc[1], mults[1], shift),
      ];

      final got = await runLinear(w, x, 2, mults, shift);
      expect(got, equals(expected), reason: 'each row must use its own mult');
    },
  );

  // -------------------------------------------------------------------------
  // Case 3: Negative accumulator values
  // W=[[-1,-2],[-3,-4]], x=[1,2] -> acc=[-5,-11]
  // mult=[1,1] shift=0 -> out=[-5,-11]
  // -------------------------------------------------------------------------
  test('negative acc: bit-exact vs requantRef', () async {
    final w = [
      [-1, -2, -3, -4],
      [-5, -6, -7, -8],
    ];
    final x = [1, 1, 1, 1];
    final mults = [1, 1];
    const shift = 0;

    final acc = dartMatVec(w, x); // [-10, -26]
    final expected = [
      requantRef(acc[0], mults[0], shift),
      requantRef(acc[1], mults[1], shift),
    ];
    expect(expected, equals([-10, -26]));

    final got = await runLinear(w, x, 2, mults, shift);
    expect(got, equals(expected));
  });

  // -------------------------------------------------------------------------
  // Case 4: Negative acc with scaling and rounding
  // -------------------------------------------------------------------------
  test('negative acc with shift rounding: bit-exact', () async {
    final w = [
      [-3, 5, -7, 9],
      [2, -4, 6, -8],
    ];
    final x = [1, -1, 2, -2];
    final mults = [3, 2];
    const shift = 3;

    final acc = dartMatVec(w, x);
    final expected = [
      requantRef(acc[0], mults[0], shift),
      requantRef(acc[1], mults[1], shift),
    ];

    final got = await runLinear(w, x, 2, mults, shift);
    expect(got, equals(expected));
  });

  // -------------------------------------------------------------------------
  // Case 5: Saturation - positive clamp to 127
  // acc[0] will be large, mult large, low shift -> exceeds 127
  // -------------------------------------------------------------------------
  test('saturation: positive overflow clamps to 127', () async {
    // acc[0] = 127*127*4 = 64516, mult=1000, shift=10:
    // prod=64516000, round(62027.34...)=62027 > 127 -> clamp to 127
    final w = [
      [127, 127, 127, 127],
      [1, 1, 1, 1],
    ];
    final x = [127, 127, 127, 127];
    final mults = [1000, 1];
    const shift = 10;

    final acc = dartMatVec(w, x);
    final expected = [
      requantRef(acc[0], mults[0], shift),
      requantRef(acc[1], mults[1], shift),
    ];
    // acc[0]=127*127*4=64516, requantRef(64516,1000,10) must be 127
    expect(expected[0], equals(127));

    final got = await runLinear(w, x, 2, mults, shift);
    expect(got, equals(expected));
  });

  // -------------------------------------------------------------------------
  // Case 6: Saturation - negative clamp to -127
  // -------------------------------------------------------------------------
  test('saturation: negative overflow clamps to -127', () async {
    final w = [
      [-127, -127, -127, -127],
      [1, 1, 1, 1],
    ];
    final x = [127, 127, 127, 127];
    final mults = [1000, 1];
    const shift = 10;

    final acc = dartMatVec(w, x);
    final expected = [
      requantRef(acc[0], mults[0], shift),
      requantRef(acc[1], mults[1], shift),
    ];
    expect(expected[0], equals(-127));

    final got = await runLinear(w, x, 2, mults, shift);
    expect(got, equals(expected));
  });

  // -------------------------------------------------------------------------
  // Case 7: peRows=4, peCols=4, K=16, mixed signs + max int8, per-row mults
  // Bit-exact vs dartMatVec + requantRef on every row.
  // -------------------------------------------------------------------------
  test('4x4 PE K=16 mixed signs max int8 per-row mults: bit-exact', () async {
    final w = [
      [127, -128, 100, -50, 30, -20, 10, -5, 127, -128, 64, -64, 1, -1, 0, 7],
      [
        -128,
        127,
        -64,
        64,
        -10,
        20,
        -30,
        50,
        -100,
        127,
        -127,
        100,
        -1,
        1,
        2,
        -3,
      ],
      [10, 20, -30, 40, -50, 60, -70, 80, -90, 100, -110, 120, -128, 127, 0, 1],
      [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8],
    ];
    final x = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8];
    final mults = [3, 7, 2, 10];
    const shift = 8;

    final acc = dartMatVec(w, x);
    final expected = [
      for (var r = 0; r < 4; r++) requantRef(acc[r], mults[r], shift),
    ];

    final got = await runLinear(w, x, 4, mults, shift);
    expect(got, equals(expected), reason: 'bit-exact for all 4 rows');
  });

  // -------------------------------------------------------------------------
  // Case 8: outValid timing aligns with acc
  // Build the module, run the tiles, confirm outValid high exactly when we
  // read out (not before, not indefinitely after the one-cycle pulse).
  // -------------------------------------------------------------------------
  test('outValid timing: high exactly when reading out', () async {
    final w = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ];
    final x = [1, 1, 1, 1];
    final mults = [1, 1];
    const shift = 0;
    const peRows = 2;
    const peCols = 2;
    const inWidth = 8;
    const accWidth = 32;
    const multWidth = 16;
    const shiftWidth = 6;
    const outWidth = 8;
    final k = x.length;
    final numTiles = k ~/ peCols;

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset_tv');
    final wTilePort = Logic(name: 'wTile_tv', width: peRows * peCols * inWidth);
    final xTilePort = Logic(name: 'xTile_tv', width: peCols * inWidth);
    final validPort = Logic(name: 'valid_tv');
    final firstPort = Logic(name: 'first_tv');
    final lastPort = Logic(name: 'last_tv');
    final rowMultPort = Logic(name: 'rowMult_tv', width: peRows * multWidth);
    final shiftPortSig = Logic(name: 'shift_tv', width: shiftWidth);

    final mod = LoomLinear(
      clk: clk,
      reset: reset,
      wTile: wTilePort,
      xTile: xTilePort,
      valid: validPort,
      first: firstPort,
      last: lastPort,
      rowMult: rowMultPort,
      shift: shiftPortSig,
      peRows: peRows,
      peCols: peCols,
      inWidth: inWidth,
      accWidth: accWidth,
      multWidth: multWidth,
      shiftWidth: shiftWidth,
      outWidth: outWidth,
    );
    await mod.build();

    Simulator.setMaxSimTime(500000);
    unawaited(Simulator.run());

    reset.inject(0);
    wTilePort.inject(0);
    xTilePort.inject(0);
    validPort.inject(0);
    firstPort.inject(0);
    lastPort.inject(0);
    rowMultPort.inject(packRowMult(mults, multWidth));
    shiftPortSig.inject(BigInt.from(shift));

    reset.inject(1);
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);
    await clk.nextPosedge;

    // Confirm outValid is not high after reset, before any tiles are sent.
    expect(
      mod.outValid.value.toBool(),
      isFalse,
      reason: 'outValid must not be high after reset before any tiles',
    );

    for (var t = 0; t < numTiles; t++) {
      final wTileRows = [
        for (var r = 0; r < peRows; r++)
          [for (var c = 0; c < peCols; c++) w[r][t * peCols + c]],
      ];
      final xTileVals = [for (var c = 0; c < peCols; c++) x[t * peCols + c]];
      wTilePort.inject(packWTile(wTileRows, inWidth));
      xTilePort.inject(packXTile(xTileVals, inWidth));
      validPort.inject(1);
      firstPort.inject(t == 0 ? 1 : 0);
      lastPort.inject(t == numTiles - 1 ? 1 : 0);
      await clk.nextPosedge;
    }

    validPort.inject(0);
    firstPort.inject(0);
    lastPort.inject(0);

    // outValid should be high now (one cycle after last tile's posedge).
    var guard = 0;
    while (!mod.outValid.value.toBool() && guard++ < (numTiles + 4)) {
      await clk.nextPosedge;
    }
    expect(
      mod.outValid.value.toBool(),
      isTrue,
      reason: 'outValid must be high after last tile',
    );

    // Read out and verify.
    final acc = dartMatVec(w, x);
    final expected = [
      requantRef(acc[0], mults[0], shift),
      requantRef(acc[1], mults[1], shift),
    ];
    final got = unpackOut(mod.out.value, peRows, outWidth);
    expect(got, equals(expected));

    // One cycle later: outValid must be low (it is a one-cycle pulse).
    await clk.nextPosedge;
    expect(
      mod.outValid.value.toBool(),
      isFalse,
      reason: 'outValid must de-assert after one cycle',
    );

    await Simulator.endSimulation();
  });

  // -------------------------------------------------------------------------
  // Config validate() rejects bad params
  // -------------------------------------------------------------------------
  group('LoomLinearConfig.validate()', () {
    test('valid config passes', () {
      expect(
        () => LoomLinearConfig(peRows: 2, peCols: 2).validate(),
        returnsNormally,
      );
    });

    test('rejects peRows=0', () {
      expect(
        () => LoomLinearConfig(peRows: 0, peCols: 2).validate(),
        throwsArgumentError,
      );
    });

    test('rejects peCols=0', () {
      expect(
        () => LoomLinearConfig(peRows: 2, peCols: 0).validate(),
        throwsArgumentError,
      );
    });

    test('rejects inWidth < 2', () {
      expect(
        () => LoomLinearConfig(peRows: 2, peCols: 2, inWidth: 1).validate(),
        throwsArgumentError,
      );
    });

    test('rejects outWidth < 2', () {
      expect(
        () => LoomLinearConfig(peRows: 2, peCols: 2, outWidth: 1).validate(),
        throwsArgumentError,
      );
    });

    test('rejects multWidth=0', () {
      expect(
        () => LoomLinearConfig(peRows: 2, peCols: 2, multWidth: 0).validate(),
        throwsArgumentError,
      );
    });

    test('rejects shiftWidth=0', () {
      expect(
        () => LoomLinearConfig(peRows: 2, peCols: 2, shiftWidth: 0).validate(),
        throwsArgumentError,
      );
    });
  });

  // -------------------------------------------------------------------------
  // SV emission: non-empty, contains module name, shows submodules
  // -------------------------------------------------------------------------
  test(
    'SV emission: contains LoomLinear, LoomMatmul, LoomRequantize',
    () async {
      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset_sv');
      final wTile = Logic(name: 'wTile_sv', width: 2 * 2 * 8);
      final xTile = Logic(name: 'xTile_sv', width: 2 * 8);
      final valid = Logic(name: 'valid_sv');
      final first = Logic(name: 'first_sv');
      final last = Logic(name: 'last_sv');
      final rowMult = Logic(name: 'rowMult_sv', width: 2 * 16);
      final shiftSig = Logic(name: 'shift_sv', width: 6);

      final mod = LoomLinear(
        clk: clk,
        reset: reset,
        wTile: wTile,
        xTile: xTile,
        valid: valid,
        first: first,
        last: last,
        rowMult: rowMult,
        shift: shiftSig,
        peRows: 2,
        peCols: 2,
      );
      await mod.build();
      final sv = mod.generateSynth();
      expect(sv, isNotEmpty);
      expect(sv, contains('LoomLinear'));
      expect(sv, contains('LoomMatmul'));
      expect(sv, contains('LoomRequantize'));
    },
  );
}
