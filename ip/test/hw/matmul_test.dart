// Tests for LoomMatmul, a clocked ROHD module that tiles a PE array over a long
// inner dimension K, accumulating partial products across cycles.
//
// Timing contract (as implemented):
//   - On cycle N where valid=1, last=1: the PE computes the final tile and
//     accReg is updated at the NEXT rising edge (cycle N+1).
//   - resultValid goes high at cycle N+1 (the cycle accReg holds the full sum).
//   - The test reads `acc` WHEN resultValid is high.
//
// All tests use Simulator.reset() in tearDown (never between build and run).
// Signals are driven with inject(); clk comes from SimpleClockGenerator.

import 'dart:async';
import 'dart:typed_data';

import 'package:loom/src/golden/quant.dart';
import 'package:loom/src/hw/matmul.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

// Pack / unpack helpers (independent of module internals)

/// Pack a flat list of signed [width]-bit values into a BigInt, element 0 in
/// the low bits. Uses two's complement masking.
BigInt packSignedFlat(List<int> vals, int width) {
  final mask = BigInt.from((1 << width) - 1);
  var result = BigInt.zero;
  for (var i = vals.length - 1; i >= 0; i--) {
    result = (result << width) | (BigInt.from(vals[i]) & mask);
  }
  return result;
}

/// Pack a peRows x peCols weight tile (row-major) into the wTile bus.
/// W[r,c] is at element index r*peCols+c in the flat list.
BigInt packWTile(List<List<int>> wTile, int inWidth) {
  final flat = [for (final row in wTile) ...row];
  return packSignedFlat(flat, inWidth);
}

/// Pack a peCols-element activation sub-tile into the xTile bus.
BigInt packXTile(List<int> xTile, int inWidth) =>
    packSignedFlat(xTile, inWidth);

/// Unpack [rows] signed [accWidth]-bit values from the acc output bus.
/// Element 0 is in the low bits. Interprets as two's complement.
List<int> unpackAcc(LogicValue accVal, int rows, int accWidth) {
  final mask = BigInt.from((1 << accWidth) - 1);
  final threshold = BigInt.one << (accWidth - 1);
  final modulus = BigInt.one << accWidth;
  final raw = accVal.toBigInt();
  return [
    for (var r = 0; r < rows; r++)
      () {
        final chunk = (raw >> (r * accWidth)) & mask;
        final signed = chunk >= threshold ? chunk - modulus : chunk;
        return signed.toInt();
      }(),
  ];
}

// Dart reference: signed integer matmul (independent of hardware)

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

// Sim harness

/// Run one complete accumulation block through LoomMatmul.
///
/// [wMat] is peRows x K (row-major). [x] is K elements. K must be a multiple
/// of peCols. Streams K/peCols tiles, asserts first on tile 0, last on the
/// last tile, then awaits resultValid and returns the acc contents.
Future<List<int>> runMatmul(
  List<List<int>> wMat,
  List<int> x,
  int peCols, {
  int inWidth = 8,
  int accWidth = 32,
  int peLatency = 0,
}) async {
  final peRows = wMat.length;
  final k = x.length;
  assert(k % peCols == 0, 'K must be a multiple of peCols');
  final numTiles = k ~/ peCols;

  // Build ports.
  final clk = SimpleClockGenerator(10).clk;
  final reset = Logic(name: 'reset');
  final wTile = Logic(name: 'wTile', width: peRows * peCols * inWidth);
  final xTile = Logic(name: 'xTile', width: peCols * inWidth);
  final valid = Logic(name: 'valid');
  final first = Logic(name: 'first');
  final last = Logic(name: 'last');

  final mod = LoomMatmul(
    clk: clk,
    reset: reset,
    wTile: wTile,
    xTile: xTile,
    valid: valid,
    first: first,
    last: last,
    peRows: peRows,
    peCols: peCols,
    inWidth: inWidth,
    accWidth: accWidth,
    peLatency: peLatency,
  );
  await mod.build();

  Simulator.setMaxSimTime(500000);
  unawaited(Simulator.run());

  // Idle defaults (after sim is running).
  reset.inject(0);
  wTile.inject(0);
  xTile.inject(0);
  valid.inject(0);
  first.inject(0);
  last.inject(0);

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

    // Build the column-tile slice for each row.
    final wTileRows = [
      for (var r = 0; r < peRows; r++)
        [for (var c = 0; c < peCols; c++) wMat[r][t * peCols + c]],
    ];
    final xTileVals = [for (var c = 0; c < peCols; c++) x[t * peCols + c]];

    wTile.inject(packWTile(wTileRows, inWidth));
    xTile.inject(packXTile(xTileVals, inWidth));
    valid.inject(1);
    first.inject(isFirst ? 1 : 0);
    last.inject(isLast ? 1 : 0);
    await clk.nextPosedge;
  }

  // De-assert inputs.
  valid.inject(0);
  first.inject(0);
  last.inject(0);

  // Wait for resultValid. The register update happens at the posedge following
  // the last valid cycle, so resultValid should be high already at this point.
  // Guard: wait up to numTiles+4 extra cycles.
  var guard = 0;
  while (!mod.resultValid.value.toBool() &&
      guard++ < (numTiles + peLatency + 4)) {
    await clk.nextPosedge;
  }

  if (!mod.resultValid.value.toBool()) {
    await Simulator.endSimulation();
    throw StateError('resultValid never asserted (guard exhausted)');
  }

  final result = unpackAcc(mod.acc.value, peRows, accWidth);
  await Simulator.endSimulation();
  return result;
}

// Tests

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  // Case 1: peRows=2, peCols=2, K=4 (2 tiles), all positive
  // W=[[1,2,3,4],[5,6,7,8]], x=[1,1,1,1] -> y=[10,26]
  test('2x2 PE, K=4, all-positive: y=[10,26]', () async {
    final w = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ];
    final x = [1, 1, 1, 1];

    final expected = dartMatVec(w, x);
    expect(expected, equals([10, 26]));

    final got = await runMatmul(w, x, 2);
    expect(got, equals(expected));
  });

  // Case 2: Negative values in W and x
  test('2x2 PE, K=4, with negatives', () async {
    final w = [
      [-1, 2, -3, 4],
      [5, -6, 7, -8],
    ];
    final x = [1, -1, 2, -2];

    final expected = dartMatVec(w, x);
    final got = await runMatmul(w, x, 2);
    expect(got, equals(expected));
  });

  // Case 3: peRows=4, peCols=4, K=16 (4 tiles), mixed signs, max int8
  // Cross-check against matmulInt from quant.dart
  test('4x4 PE, K=16, mixed signs + max int8, cross-check matmulInt', () async {
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

    final expected = dartMatVec(w, x);

    // Cross-check against matmulInt from quant.dart.
    final flatW = Int8List(4 * 16);
    for (var r = 0; r < 4; r++) {
      for (var c = 0; c < 16; c++) {
        flatW[r * 16 + c] = w[r][c];
      }
    }
    final xInt8 = Int8List.fromList(x);
    final qm = QuantizedMatrix(
      values: flatW,
      rowScales: Float64List.fromList([1.0, 1.0, 1.0, 1.0]),
      rows: 4,
      cols: 16,
    );
    final qv = QuantizedVector(values: xInt8, scale: 1.0);
    final golden = matmulInt(qm, qv);
    expect(List<int>.from(expected), equals(List<int>.from(golden)));

    final got = await runMatmul(w, x, 4);
    expect(got, equals(expected));
  });

  // Case 4: Single-tile (K=peCols, first&last same cycle)
  test('single tile (K=peCols=2): first&last same cycle', () async {
    final w = [
      [3, -5],
      [-2, 7],
    ];
    final x = [4, -3];

    final expected = dartMatVec(w, x);
    // y[0] = 3*4 + (-5)*(-3) = 12+15=27
    // y[1] = (-2)*4 + 7*(-3) = -8-21=-29
    expect(expected, equals([27, -29]));

    final got = await runMatmul(w, x, 2);
    expect(got, equals(expected));
  });

  // Case 5: Back-to-back blocks. Second block must be independent of first.
  test(
    'back-to-back blocks: second accumulation independent of first',
    () async {
      final peRows = 2;
      final peCols = 2;
      final k = 4;
      final inWidth = 8;
      final accWidth = 32;

      final clk = SimpleClockGenerator(10).clk;
      final reset = Logic(name: 'reset');
      final wTile = Logic(name: 'wTile', width: peRows * peCols * inWidth);
      final xTile = Logic(name: 'xTile', width: peCols * inWidth);
      final valid = Logic(name: 'valid');
      final first = Logic(name: 'first');
      final last = Logic(name: 'last');

      final mod = LoomMatmul(
        clk: clk,
        reset: reset,
        wTile: wTile,
        xTile: xTile,
        valid: valid,
        first: first,
        last: last,
        peRows: peRows,
        peCols: peCols,
        inWidth: inWidth,
        accWidth: accWidth,
      );
      await mod.build();

      Simulator.setMaxSimTime(1000000);
      unawaited(Simulator.run());

      reset.inject(0);
      wTile.inject(0);
      xTile.inject(0);
      valid.inject(0);
      first.inject(0);
      last.inject(0);

      reset.inject(1);
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;

      // Block 1
      final w1 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ];
      final x1 = [1, 1, 1, 1];
      final expected1 = dartMatVec(w1, x1); // [10, 26]

      final numTiles = k ~/ peCols;
      for (var t = 0; t < numTiles; t++) {
        final wTileRows = [
          for (var r = 0; r < peRows; r++)
            [for (var c = 0; c < peCols; c++) w1[r][t * peCols + c]],
        ];
        final xTileVals = [for (var c = 0; c < peCols; c++) x1[t * peCols + c]];
        wTile.inject(packWTile(wTileRows, inWidth));
        xTile.inject(packXTile(xTileVals, inWidth));
        valid.inject(1);
        first.inject(t == 0 ? 1 : 0);
        last.inject(t == numTiles - 1 ? 1 : 0);
        await clk.nextPosedge;
      }
      valid.inject(0);
      first.inject(0);
      last.inject(0);

      // Wait for resultValid for block 1.
      var guard = 0;
      while (!mod.resultValid.value.toBool() && guard++ < 10) {
        await clk.nextPosedge;
      }
      expect(
        mod.resultValid.value.toBool(),
        isTrue,
        reason: 'block1 resultValid',
      );
      final got1 = unpackAcc(mod.acc.value, peRows, accWidth);
      expect(got1, equals(expected1), reason: 'block 1 result');

      // Let resultValid settle (one idle cycle).
      await clk.nextPosedge;

      // Block 2: different values, must not be contaminated by block 1
      final w2 = [
        [-1, -2, -3, -4],
        [10, 20, -10, -20],
      ];
      final x2 = [2, 2, 2, 2];
      final expected2 = dartMatVec(w2, x2); // [-20, 0]

      for (var t = 0; t < numTiles; t++) {
        final wTileRows = [
          for (var r = 0; r < peRows; r++)
            [for (var c = 0; c < peCols; c++) w2[r][t * peCols + c]],
        ];
        final xTileVals = [for (var c = 0; c < peCols; c++) x2[t * peCols + c]];
        wTile.inject(packWTile(wTileRows, inWidth));
        xTile.inject(packXTile(xTileVals, inWidth));
        valid.inject(1);
        first.inject(t == 0 ? 1 : 0);
        last.inject(t == numTiles - 1 ? 1 : 0);
        await clk.nextPosedge;
      }
      valid.inject(0);
      first.inject(0);
      last.inject(0);

      guard = 0;
      while (!mod.resultValid.value.toBool() && guard++ < 10) {
        await clk.nextPosedge;
      }
      expect(
        mod.resultValid.value.toBool(),
        isTrue,
        reason: 'block2 resultValid',
      );
      final got2 = unpackAcc(mod.acc.value, peRows, accWidth);
      expect(got2, equals(expected2), reason: 'block 2 result independent');
      await Simulator.endSimulation();
    },
  );

  // Pipelined PE: bit-exact across peLatency values, only the timing changes.
  for (final peLatency in [1, 2, 3]) {
    test('peLatency=$peLatency: 2x2 PE K=4 bit-exact', () async {
      final w = [
        [-1, 2, -3, 4],
        [5, -6, 7, -8],
      ];
      final x = [1, -1, 2, -2];
      final expected = dartMatVec(w, x);
      final got = await runMatmul(w, x, 2, peLatency: peLatency);
      expect(got, equals(expected));
    });

    test('peLatency=$peLatency: 4x4 PE K=16 max int8 bit-exact', () async {
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
        [
          10,
          20,
          -30,
          40,
          -50,
          60,
          -70,
          80,
          -90,
          100,
          -110,
          120,
          -128,
          127,
          0,
          1,
        ],
        [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8],
      ];
      final x = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8];
      final expected = dartMatVec(w, x);
      final got = await runMatmul(w, x, 4, peLatency: peLatency);
      expect(got, equals(expected));
    });

    test(
      'peLatency=$peLatency: single tile (first&last same) bit-exact',
      () async {
        final w = [
          [3, -5],
          [-2, 7],
        ];
        final x = [4, -3];
        final expected = dartMatVec(w, x);
        final got = await runMatmul(w, x, 2, peLatency: peLatency);
        expect(got, equals(expected));
      },
    );
  }

  // Config validation
  group('LoomMatmulConfig.validate()', () {
    test('valid config passes', () {
      expect(
        () => LoomMatmulConfig(peRows: 4, peCols: 4).validate(),
        returnsNormally,
      );
    });

    test('rejects peRows=0', () {
      expect(
        () => LoomMatmulConfig(peRows: 0, peCols: 4).validate(),
        throwsArgumentError,
      );
    });

    test('rejects peCols=0', () {
      expect(
        () => LoomMatmulConfig(peRows: 4, peCols: 0).validate(),
        throwsArgumentError,
      );
    });

    test('rejects inWidth < 2', () {
      expect(
        () => LoomMatmulConfig(peRows: 4, peCols: 4, inWidth: 1).validate(),
        throwsArgumentError,
      );
    });

    test('rejects accWidth too small', () {
      // For peCols=4, inWidth=8: minAccWidth = 2*8 + ceil(log2(4)) = 18.
      expect(
        () => LoomMatmulConfig(
          peRows: 4,
          peCols: 4,
          inWidth: 8,
          accWidth: 15,
        ).validate(),
        throwsArgumentError,
      );
    });
  });

  // SV emission
  test('SV emission: non-empty and contains module name', () async {
    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset_sv');
    final wTile = Logic(name: 'wTile_sv', width: 2 * 2 * 8);
    final xTile = Logic(name: 'xTile_sv', width: 2 * 8);
    final valid = Logic(name: 'valid_sv');
    final first = Logic(name: 'first_sv');
    final last = Logic(name: 'last_sv');

    final mod = LoomMatmul(
      clk: clk,
      reset: reset,
      wTile: wTile,
      xTile: xTile,
      valid: valid,
      first: first,
      last: last,
      peRows: 2,
      peCols: 2,
    );
    await mod.build();
    final sv = mod.generateSynth();
    expect(sv, isNotEmpty);
    expect(sv, contains('LoomMatmul'));
  });
}
