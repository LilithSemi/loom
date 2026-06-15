// Tests for LoomPeArray against golden linear() (ip/lib/src/golden/ops.dart):
// weight row-major [out=rows, in=cols], y[r] = sum_c W[r,c]*x[c], checked bit-exact.

import 'dart:async';
import 'dart:math' as math;

import 'package:loom/src/hw/pe_array.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

// Plain-Dart signed integer reference

/// Compute y[r] = sum_c W[r,c] * x[c] in plain Dart (no overflow assumed).
/// W is row-major: W[r][c] = w[r * cols + c].
List<int> refMatVec(List<List<int>> w, List<int> x) {
  final rows = w.length;
  final cols = x.length;
  return [
    for (var r = 0; r < rows; r++)
      [for (var c = 0; c < cols; c++) w[r][c] * x[c]].fold(0, (a, b) => a + b),
  ];
}

// Pack/unpack helpers

/// Pack a list of signed [width]-bit integers into a single BigInt value,
/// element 0 in the low bits. Values are masked to [width] bits (two's
/// complement representation).
BigInt packSigned(List<int> vals, int width) {
  final mask = BigInt.from((1 << width) - 1);
  var result = BigInt.zero;
  for (var i = vals.length - 1; i >= 0; i--) {
    result = (result << width) | (BigInt.from(vals[i]) & mask);
  }
  return result;
}

/// Pack W (row-major list-of-rows) into the flat w bus.
BigInt packW(List<List<int>> w, int inWidth) {
  final flat = [for (final row in w) ...row];
  return packSigned(flat, inWidth);
}

/// Pack x vector into the x bus.
BigInt packX(List<int> x, int inWidth) => packSigned(x, inWidth);

/// Unpack [rows] signed [accWidth]-bit values from the y output bus.
/// Element 0 is in the low bits. Sign-interprets each accWidth-bit chunk.
List<int> unpackY(LogicValue yVal, int rows, int accWidth) {
  final mask = BigInt.from((1 << accWidth) - 1);
  final threshold = BigInt.from(1) << (accWidth - 1);
  final modulus = BigInt.from(1) << accWidth;
  final raw = yVal.toBigInt();
  return [
    for (var r = 0; r < rows; r++)
      () {
        final chunk = (raw >> (r * accWidth)) & mask;
        final signed = chunk >= threshold ? chunk - modulus : chunk;
        return signed.toInt();
      }(),
  ];
}

// Tests

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  // Helper: build the module with given W and x, return sim output.
  Future<List<int>> runSim(
    List<List<int>> wMat,
    List<int> xVec, {
    int inWidth = 8,
    int accWidth = 32,
    bool ternary = false,
  }) async {
    final rows = wMat.length;
    final cols = xVec.length;

    final wIn = Logic(name: 'w', width: rows * cols * inWidth);
    final xIn = Logic(name: 'x', width: cols * inWidth);

    final mod = LoomPeArray(
      w: wIn,
      x: xIn,
      rows: rows,
      cols: cols,
      inWidth: inWidth,
      accWidth: accWidth,
      ternaryWeights: ternary,
    );
    await mod.build();

    wIn.put(packW(wMat, inWidth));
    xIn.put(packX(xVec, inWidth));

    return unpackY(mod.y.value, rows, accWidth);
  }

  // Ternary select/negate must match a plain integer matmul bit-exact (and the
  // NativeMultiplier path). Uses power-of-2 cols since the accelerator tiles the
  // PE array at power-of-2 widths.
  test('ternary PE: each sign case +1/-1/0 selects/negates/zeros x', () async {
    final w = [
      [1, -1, 0, 1], // +x0 -x1 +0 +x3
      [0, 1, -1, 0], // +0 +x1 -x2 +0
    ];
    final x = [40, -30, 7, 5];
    final got = await runSim(w, x, ternary: true);
    expect(got, equals(refMatVec(w, x)));
    expect(got, equals([75, -37])); // 40+30+0+5 ; 0-30-7+0
  });

  test(
    'ternary PE == multiply PE for random ternary weights (bit-exact)',
    () async {
      final rng = math.Random(9091);
      for (var trial = 0; trial < 6; trial++) {
        const rows = 4, cols = 8;
        final w = [
          for (var r = 0; r < rows; r++)
            [for (var c = 0; c < cols; c++) rng.nextInt(3) - 1], // {-1,0,1}
        ];
        final x = [for (var c = 0; c < cols; c++) rng.nextInt(255) - 127];
        final tern = await runSim(w, x, ternary: true);
        final mult = await runSim(w, x); // NativeMultiplier path
        expect(tern, equals(mult), reason: 'ternary != multiply, trial $trial');
        expect(
          tern,
          equals(refMatVec(w, x)),
          reason: 'ternary != golden, trial $trial',
        );
      }
    },
  );

  test('ternary PE is width-generic: inWidth=4 (int4 acts) == golden', () async {
    // The select/negate is written against _inWidth (sign bit at inWidth-1), not
    // hardcoded to int8, so it must hold for other widths too. int4 acts: [-7,7].
    final rng = math.Random(404);
    for (var trial = 0; trial < 4; trial++) {
      const rows = 2, cols = 4;
      final w = [
        for (var r = 0; r < rows; r++)
          [for (var c = 0; c < cols; c++) rng.nextInt(3) - 1], // {-1,0,1}
      ];
      final x = [for (var c = 0; c < cols; c++) rng.nextInt(15) - 7]; // int4
      final tern = await runSim(w, x, inWidth: 4, ternary: true);
      final mult = await runSim(w, x, inWidth: 4);
      expect(
        tern,
        equals(mult),
        reason: 'inWidth=4 ternary != multiply, trial $trial',
      );
      expect(
        tern,
        equals(refMatVec(w, x)),
        reason: 'inWidth=4 ternary != golden, trial $trial',
      );
    }
  });

  test('ternary PE: extreme x (-127, 127) with all sign cases', () async {
    final w = [
      [1, -1, 1, -1],
    ];
    final x = [127, 127, -127, -127];
    final got = await runSim(w, x, ternary: true);
    // 127 - 127 + (-127) - (-127) = 0
    expect(got, equals(refMatVec(w, x)));
    expect(got, equals([0]));
  });

  test('2x2 all-positive: [[1,2],[3,4]] * [1,1] = [3,7]', () async {
    final got = await runSim(
      [
        [1, 2],
        [3, 4],
      ],
      [1, 1],
    );
    expect(
      got,
      equals(
        refMatVec(
          [
            [1, 2],
            [3, 4],
          ],
          [1, 1],
        ),
      ),
    );
    expect(got, equals([3, 7]));
  });

  test('2x2 negative: [[-1,2],[3,-4]] * [5,-6] = [-17,39]', () async {
    final w = [
      [-1, 2],
      [3, -4],
    ];
    final x = [5, -6];
    final expected = refMatVec(w, x);
    expect(expected, equals([-17, 39]));
    final got = await runSim(w, x);
    expect(got, equals(expected));
  });

  test(
    '2x2 max int8 magnitudes: [[-128,127],[127,-128]] * [-128,127]',
    () async {
      final w = [
        [-128, 127],
        [127, -128],
      ];
      final x = [-128, 127];
      final expected = refMatVec(w, x);
      final got = await runSim(w, x);
      expect(got, equals(expected));
    },
  );

  test('1x1: [[5]] * [-3] = -15', () async {
    final w = [
      [5],
    ];
    final x = [-3];
    final expected = refMatVec(w, x);
    expect(expected, equals([-15]));
    final got = await runSim(w, x);
    expect(got, equals(expected));
  });

  test('1x1: [[-7]] * [-9] = 63', () async {
    final w = [
      [-7],
    ];
    final x = [-9];
    final expected = refMatVec(w, x);
    expect(expected, equals([63]));
    final got = await runSim(w, x);
    expect(got, equals(expected));
  });

  test('4x8 mixed signs fixed literals', () async {
    final w = [
      [1, -2, 3, -4, 5, -6, 7, -8],
      [-1, 2, -3, 4, -5, 6, -7, 8],
      [10, 20, -30, 40, -50, 60, -70, 80],
      [-128, 127, -128, 127, -128, 127, -128, 127],
    ];
    final x = [1, -1, 2, -2, 3, -3, 4, -4];
    final expected = refMatVec(w, x);
    final got = await runSim(w, x, accWidth: 32);
    expect(got, equals(expected));
  });

  group('LoomPeArrayConfig.validate()', () {
    test('valid config passes', () {
      expect(
        () => LoomPeArrayConfig(rows: 2, cols: 2).validate(),
        returnsNormally,
      );
    });

    test('rejects rows=0', () {
      expect(
        () => LoomPeArrayConfig(rows: 0, cols: 2).validate(),
        throwsArgumentError,
      );
    });

    test('rejects cols=0', () {
      expect(
        () => LoomPeArrayConfig(rows: 2, cols: 0).validate(),
        throwsArgumentError,
      );
    });

    test('rejects inWidth < 2', () {
      expect(
        () => LoomPeArrayConfig(rows: 2, cols: 2, inWidth: 1).validate(),
        throwsArgumentError,
      );
    });

    test('rejects accWidth too small for worst-case sum', () {
      // cols=4, inWidth=8: max product per term = 127*127 = 16129.
      // Sum of 4 terms = 64516. That fits in 17 bits but not 15.
      // accWidth must be >= 2*inWidth + ceil(log2(cols)).
      // With inWidth=8, cols=4: product is 16 bits, log2(4)=2, so need 18.
      expect(
        () => LoomPeArrayConfig(
          rows: 2,
          cols: 4,
          inWidth: 8,
          accWidth: 15,
        ).validate(),
        throwsArgumentError,
      );
    });
  });

  // Drive the same operands through a latency-L pipelined PE array, wait L
  // posedges, and confirm y matches the combinational reference exactly.
  Future<List<int>> runSimPipelined(
    List<List<int>> wMat,
    List<int> xVec,
    int latency, {
    int inWidth = 8,
    int accWidth = 32,
  }) async {
    final rows = wMat.length;
    final cols = xVec.length;

    final clk = SimpleClockGenerator(10).clk;
    final reset = Logic(name: 'reset');
    final wIn = Logic(name: 'w', width: rows * cols * inWidth);
    final xIn = Logic(name: 'x', width: cols * inWidth);

    final mod = LoomPeArray(
      w: wIn,
      x: xIn,
      rows: rows,
      cols: cols,
      inWidth: inWidth,
      accWidth: accWidth,
      latency: latency,
      clk: clk,
      reset: reset,
    );
    await mod.build();
    expect(mod.latency, equals(latency));

    Simulator.setMaxSimTime(100000);
    unawaited(Simulator.run());

    reset.inject(1);
    wIn.inject(0);
    xIn.inject(0);
    await clk.nextPosedge;
    await clk.nextPosedge;
    reset.inject(0);

    // Present operands and hold them. Sample y `latency` posedges later.
    wIn.inject(packW(wMat, inWidth));
    xIn.inject(packX(xVec, inWidth));
    for (var i = 0; i < latency; i++) {
      await clk.nextPosedge;
    }
    // y now reflects the operands presented `latency` cycles ago.
    final got = unpackY(mod.y.value, rows, accWidth);
    await Simulator.endSimulation();
    return got;
  }

  for (final latency in [1, 2, 3, 4]) {
    test(
      'pipelined latency=$latency: 4x8 bit-exact vs combinational',
      () async {
        final w = [
          [1, -2, 3, -4, 5, -6, 7, -8],
          [-1, 2, -3, 4, -5, 6, -7, 8],
          [10, 20, -30, 40, -50, 60, -70, 80],
          [-128, 127, -128, 127, -128, 127, -128, 127],
        ];
        final x = [1, -1, 2, -2, 3, -3, 4, -4];
        final expected = refMatVec(w, x);
        final got = await runSimPipelined(w, x, latency);
        expect(got, equals(expected));
      },
    );
  }

  test('pipelined latency=2: max int8 magnitudes bit-exact', () async {
    final w = [
      [-128, 127],
      [127, -128],
    ];
    final x = [-128, 127];
    final expected = refMatVec(w, x);
    final got = await runSimPipelined(w, x, 2);
    expect(got, equals(expected));
  });

  test('LoomPeArray throws if latency>=1 without clk', () {
    final wIn = Logic(name: 'w_noclk', width: 2 * 2 * 8);
    final xIn = Logic(name: 'x_noclk', width: 2 * 8);
    expect(
      () => LoomPeArray(w: wIn, x: xIn, rows: 2, cols: 2, latency: 1),
      throwsArgumentError,
    );
  });

  test('SV emission: non-empty and contains module name', () async {
    final wIn = Logic(name: 'w_sv', width: 2 * 2 * 8);
    final xIn = Logic(name: 'x_sv', width: 2 * 8);
    final mod = LoomPeArray(w: wIn, x: xIn, rows: 2, cols: 2);
    await mod.build();
    final sv = mod.generateSynth();
    expect(sv, isNotEmpty);
    expect(sv, contains('LoomPeArray'));
  });

  test(
    'ternary PE emits NO multiply operator (multiply-free / DSP-free)',
    () async {
      Future<String> emit({required bool ternary}) async {
        final wIn = Logic(name: 'w_sv', width: 2 * 2 * 8);
        final xIn = Logic(name: 'x_sv', width: 2 * 8);
        final mod = LoomPeArray(
          w: wIn,
          x: xIn,
          rows: 2,
          cols: 2,
          ternaryWeights: ternary,
        );
        await mod.build();
        return mod.generateSynth();
      }

      // Strip ROHD's block comments (the header banner and the /* 31:0 */ bit-range
      // annotations contain '*' but are not arithmetic) before checking for a real
      // multiply operator.
      String body(String sv) => sv.replaceAll(RegExp(r'/\*[\s\S]*?\*/'), '');

      // The multiply PE emits a Verilog '*' (yosys infers a MULT18/DSP for it).
      // The ternary PE must emit NONE, that is the whole DSP-freeing point.
      final multBody = body(await emit(ternary: false));
      final ternBody = body(await emit(ternary: true));
      expect(
        multBody,
        contains('*'),
        reason: 'multiply PE should have a multiply',
      );
      expect(
        ternBody,
        isNot(contains('*')),
        reason: 'ternary PE must be multiply-free (no DSP inferred)',
      );
    },
  );
}
