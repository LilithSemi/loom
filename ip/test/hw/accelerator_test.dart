// Tests for LoomAccelerator: config validation, SV emission, and Wishbone
// bus simulation (VERSION read, full linear op, zero-pad path, buffer
// reuse, RTL emission).

import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:loom/src/hw/accelerator.dart';
import 'package:rohd/rohd.dart';
import 'package:test/test.dart';

// Dart reference: sign-aware round-half-away-from-zero right shift.

int _roundShift(int prod, int shift) {
  if (shift == 0) return prod;
  final bias = 1 << (shift - 1);
  if (prod >= 0) {
    return (prod + bias) >> shift;
  } else {
    return -((-prod + bias) >> shift);
  }
}

/// Bit-exact reference: acc * mult then shift-round then saturate to int8.
int requantRef(int acc, int mult, int shift) {
  final prod = acc * mult;
  final rounded = _roundShift(prod, shift);
  const lo = -127;
  const hi = 127;
  if (rounded < lo) return lo;
  if (rounded > hi) return hi;
  return rounded;
}

/// Integer matrix-vector multiply: acc[r] = sum_c W[r,c] * x[c].
Int32List matmulInt(Int8List w, Int8List x, int rows, int cols) {
  final acc = Int32List(rows);
  for (var r = 0; r < rows; r++) {
    var sum = 0;
    for (var c = 0; c < cols; c++) {
      sum += w[r * cols + c] * x[c];
    }
    acc[r] = sum;
  }
  return acc;
}

/// Interpret a raw int from a 32-bit LogicValue as a signed 8-bit value
/// packed at byte [byteIndex] (little-endian: byte 0 is bits [7:0]).
int int8FromBusWord(int word, int byteIndex) {
  final raw = (word >> (byteIndex * 8)) & 0xFF;
  return raw >= 0x80 ? raw - 256 : raw;
}

void main() {
  tearDown(() async {
    await Simulator.reset();
  });

  group('LoomAcceleratorConfig validation', () {
    test('default config validates cleanly', () {
      expect(
        () => const LoomAcceleratorConfig(baseAddress: 0x70000000).validate(),
        returnsNormally,
      );
    });

    test('rejects peRows <= 0', () {
      expect(
        () => const LoomAcceleratorConfig(
          baseAddress: 0x70000000,
          peRows: 0,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects peCols <= 0', () {
      expect(
        () => const LoomAcceleratorConfig(
          baseAddress: 0x70000000,
          peCols: 0,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects maxRows <= 0', () {
      expect(
        () => const LoomAcceleratorConfig(
          baseAddress: 0x70000000,
          maxRows: 0,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects maxCols <= 0', () {
      expect(
        () => const LoomAcceleratorConfig(
          baseAddress: 0x70000000,
          maxCols: 0,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects maxRows < peRows', () {
      expect(
        () => const LoomAcceleratorConfig(
          baseAddress: 0x70000000,
          peRows: 4,
          maxRows: 2,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects maxCols < peCols', () {
      expect(
        () => const LoomAcceleratorConfig(
          baseAddress: 0x70000000,
          peCols: 4,
          maxCols: 2,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects inWidth <= 0', () {
      expect(
        () => const LoomAcceleratorConfig(
          baseAddress: 0x70000000,
          inWidth: 0,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects multWidth <= 0', () {
      expect(
        () => const LoomAcceleratorConfig(
          baseAddress: 0x70000000,
          multWidth: 0,
        ).validate(),
        throwsArgumentError,
      );
    });
  });

  group('LoomAccelerator structural', () {
    late LoomAccelerator accel;

    setUp(() async {
      accel = LoomAccelerator(
        config: const LoomAcceleratorConfig(baseAddress: 0x70000000),
      );
      await accel.build();
    });

    test('builds and emits non-empty SystemVerilog', () {
      final sv = accel.generateSynth();
      expect(sv, isNotEmpty);
    });

    test('SV contains module name LoomAccelerator', () {
      final sv = accel.generateSynth();
      expect(sv, contains('LoomAccelerator'));
    });

    test('SV contains Wishbone signal bus_CYC', () {
      final sv = accel.generateSynth();
      expect(sv, contains('bus_CYC'));
    });

    test('has a Wishbone bus slave port', () {
      expect(accel.bus, isNotNull);
    });

    test('bus has 32-bit data width', () {
      expect(accel.bus.dataIn.width, equals(32));
    });

    test('dtNode compatible string is correct', () {
      expect(accel.dtNode.compatible, contains('midstall,loom-accelerator'));
    });

    test('dtNode reg matches baseAddress', () {
      expect(accel.dtNode.reg.start, equals(0x70000000));
    });

    test('svdPeripheral name is LOOM_ACCELERATOR', () {
      expect(accel.svdPeripheral.name, equals('LOOM_ACCELERATOR'));
    });
  });

  // Shared sim helpers

  // The bus drives raw byte addresses. See controller.dart convention.
  // CSR offsets (byte):
  //   0x00 VERSION RO
  //   0x04 ROWS    RW
  //   0x08 COLS    RW
  //   0x0C SHIFT   RW
  //   0x10 CONTROL RW (bit 0: start, self-clears)
  //   0x14 STATUS  RO (bit 0: busy, bit 1: done)
  //
  // Buffer regions (byte base, 4 bytes per word):
  //   0x100 .. 0x1FF  weight buffer  (row-major, int8 packed 4/word)
  //   0x200 .. 0x2FF  activation buffer (int8 packed 4/word)
  //   0x300 .. 0x3FF  rowMult buffer (16-bit packed 2/word)
  //   0x400 .. 0x4FF  result buffer  (int8 packed 4/word, read-only)

  const csrVersion = 0x00;
  const csrRows = 0x04;
  const csrCols = 0x08;
  const csrShift = 0x0C;
  const csrControl = 0x10;
  const csrStatus = 0x14;
  const baseWeights = 0x100;
  const baseActivations = 0x200;
  const baseRowMult = 0x300;
  const baseResult = 0x400;
  const ctrlStart = 1;
  const statusDone = 2;

  group('LoomAccelerator Wishbone simulation', () {
    late Logic clk, reset, cyc, stb, we, adr, datMosi, sel;
    late LoomAccelerator accel;

    setUp(() async {
      clk = SimpleClockGenerator(10).clk;
      reset = Logic(name: 'reset');

      accel = LoomAccelerator(
        config: const LoomAcceleratorConfig(
          baseAddress: 0x70000000,
          peRows: 2,
          peCols: 2,
          maxRows: 8,
          maxCols: 8,
        ),
      );

      cyc = Logic(name: 'cyc');
      stb = Logic(name: 'stb');
      we = Logic(name: 'we');
      adr = Logic(name: 'adr', width: accel.input('bus_ADR').width);
      datMosi = Logic(
        name: 'datMosi',
        width: accel.input('bus_DAT_MOSI').width,
      );
      sel = Logic(name: 'sel', width: accel.input('bus_SEL').width);

      accel.input('clk').srcConnection! <= clk;
      accel.input('reset').srcConnection! <= reset;
      accel.input('bus_CYC').srcConnection! <= cyc;
      accel.input('bus_STB').srcConnection! <= stb;
      accel.input('bus_WE').srcConnection! <= we;
      accel.input('bus_ADR').srcConnection! <= adr;
      accel.input('bus_DAT_MOSI').srcConnection! <= datMosi;
      accel.input('bus_SEL').srcConnection! <= sel;

      await accel.build();

      for (final s in [cyc, stb, we, adr, datMosi, sel]) {
        s.inject(0);
      }
      reset.inject(1);
      Simulator.setMaxSimTime(2000000);
      unawaited(Simulator.run());
      await clk.nextPosedge;
      await clk.nextPosedge;
      reset.inject(0);
      await clk.nextPosedge;
    });

    tearDown(() async {
      await Simulator.endSimulation();
    });

    Logic ack() => accel.output('bus_ACK');
    Logic miso() => accel.output('bus_DAT_MISO');

    Future<void> wbWrite(int address, int data) async {
      cyc.inject(1);
      stb.inject(1);
      we.inject(1);
      adr.inject(address);
      datMosi.inject(data & 0xFFFFFFFF);
      var guard = 0;
      while (!ack().value.toBool() && guard++ < 200) {
        await clk.nextPosedge;
      }
      await clk.nextPosedge;
      cyc.inject(0);
      stb.inject(0);
      we.inject(0);
      await clk.nextPosedge;
    }

    Future<int> wbRead(int address) async {
      cyc.inject(1);
      stb.inject(1);
      we.inject(0);
      adr.inject(address);
      var guard = 0;
      while (!ack().value.toBool() && guard++ < 200) {
        await clk.nextPosedge;
      }
      final v = miso().value.toInt();
      await clk.nextPosedge;
      cyc.inject(0);
      stb.inject(0);
      await clk.nextPosedge;
      return v;
    }

    // Write int8 values into an activation/result buffer region.
    // Packs values 4 bytes per 32-bit word at sequential addresses.
    Future<void> writeByteBuffer(int baseAddr, List<int> values) async {
      final padded = List<int>.from(values);
      while (padded.length % 4 != 0) {
        padded.add(0);
      }
      for (var i = 0; i < padded.length; i += 4) {
        final word =
            (padded[i] & 0xFF) |
            ((padded[i + 1] & 0xFF) << 8) |
            ((padded[i + 2] & 0xFF) << 16) |
            ((padded[i + 3] & 0xFF) << 24);
        await wbWrite(baseAddr + i, word);
      }
    }

    // Write a weight matrix into the weight buffer region.
    //
    // The hardware stores W[r,c] at byte r*maxCols+c in the buffer where
    // maxCols is the hardware maximum (LoomAcceleratorConfig.maxCols), NOT
    // the runtime COLS. The host must write each row at its strided offset.
    //
    // [w] is a dense row-major list: w[r*cols+c] = W[r,c].
    // [rows],[cols] are the runtime dimensions. [maxCols] is the HW stride.
    Future<void> writeWeightBuffer(
      int baseAddr,
      List<int> w,
      int rows,
      int cols,
      int maxCols,
    ) async {
      final flat = List.filled(maxCols * rows, 0);
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          flat[r * maxCols + c] = w[r * cols + c];
        }
      }
      final padded = List<int>.from(flat);
      while (padded.length % 4 != 0) {
        padded.add(0);
      }
      for (var i = 0; i < padded.length; i += 4) {
        final word =
            (padded[i] & 0xFF) |
            ((padded[i + 1] & 0xFF) << 8) |
            ((padded[i + 2] & 0xFF) << 16) |
            ((padded[i + 3] & 0xFF) << 24);
        await wbWrite(baseAddr + i, word);
      }
    }

    // Write 16-bit mult values packed 2 per 32-bit word.
    Future<void> writeMultBuffer(int baseAddr, List<int> values) async {
      final padded = List<int>.from(values);
      while (padded.length % 2 != 0) {
        padded.add(0);
      }
      for (var i = 0; i < padded.length; i += 2) {
        final word = (padded[i] & 0xFFFF) | ((padded[i + 1] & 0xFFFF) << 16);
        await wbWrite(baseAddr + i * 2, word);
      }
    }

    Future<void> pollDone() async {
      var guard = 0;
      while (true) {
        final s = await wbRead(csrStatus);
        if (s & statusDone != 0) break;
        guard++;
        if (guard > 500) {
          fail('Timed out waiting for STATUS.done');
        }
        await clk.nextPosedge;
      }
    }

    test('VERSION register reads the magic constant 0x4C4F4F4D', () async {
      final v = await wbRead(csrVersion);
      expect(v, equals(0x4C4F4F4D), reason: 'VERSION should return LOOM magic');
    });

    test('reports the baked model name over MODEL_LEN + NAME region', () async {
      const csrModelLen = 0x18;
      const nameRegion = 0x500;
      final len = await wbRead(csrModelLen);
      expect(len, equals(4), reason: 'default modelName "loom" is 4 bytes');

      final bytes = <int>[];
      for (var i = 0; i < (len + 3) ~/ 4; i++) {
        final w = await wbRead(nameRegion + i * 4);
        bytes
          ..add(w & 0xFF)
          ..add((w >> 8) & 0xFF)
          ..add((w >> 16) & 0xFF)
          ..add((w >> 24) & 0xFF);
      }
      expect(String.fromCharCodes(bytes.sublist(0, len)), equals('loom'));
    });

    test('full linear op ROWS=2 COLS=4 bit-exact vs Dart reference', () async {
      // W[2,4] and x[4], with negative values.
      final w = Int8List.fromList([
        3, -2, 1, -1, // row 0
        -4, 3, -2, 2, // row 1
      ]);
      final x = Int8List.fromList([5, -3, 2, -1]);
      final rowMults = <int>[32768, 16384]; // unsigned 16-bit
      const shift = 16;
      const rows = 2;
      const cols = 4;

      final accRef = matmulInt(w, x, rows, cols);
      final expectedResults = List.generate(
        rows,
        (r) => requantRef(accRef[r], rowMults[r], shift),
      );

      await wbWrite(csrRows, rows);
      await wbWrite(csrCols, cols);
      await wbWrite(csrShift, shift);

      // 8 here must match config.maxCols above.
      await writeWeightBuffer(baseWeights, w.toList(), rows, cols, 8);

      await writeByteBuffer(baseActivations, x.toList());
      await writeMultBuffer(baseRowMult, rowMults);
      await wbWrite(csrControl, ctrlStart);
      await pollDone();

      for (var r = 0; r < rows; r++) {
        final wordIdx = r ~/ 4;
        final byteIdx = r % 4;
        final word = await wbRead(baseResult + wordIdx * 4);
        final got = int8FromBusWord(word, byteIdx);
        expect(
          got,
          equals(expectedResults[r]),
          reason:
              'result[$r]: got $got, expected ${expectedResults[r]} '
              '(acc=${accRef[r]}, mult=${rowMults[r]}, shift=$shift)',
        );
      }
    });

    // With peRows=2, peCols=2, ROWS=3/COLS=3 exercises the zero-pad path for
    // the last row/col tile.
    test('non-multiple dims ROWS=3 COLS=3 zero-pad path', () async {
      final w = Int8List.fromList([
        2, -1, 3, // row 0
        -2, 4, -1, // row 1
        1, -3, 2, // row 2
      ]);
      final x = Int8List.fromList([4, -2, 3]);
      final rowMults = <int>[32768, 32768, 32768];
      const shift = 16;
      const rows = 3;
      const cols = 3;

      final accRef = matmulInt(w, x, rows, cols);
      final expectedResults = List.generate(
        rows,
        (r) => requantRef(accRef[r], rowMults[r], shift),
      );

      await wbWrite(csrRows, rows);
      await wbWrite(csrCols, cols);
      await wbWrite(csrShift, shift);
      await writeWeightBuffer(baseWeights, w.toList(), rows, cols, 8);
      await writeByteBuffer(baseActivations, x.toList());
      await writeMultBuffer(baseRowMult, rowMults);
      await wbWrite(csrControl, ctrlStart);
      await pollDone();

      for (var r = 0; r < rows; r++) {
        final wordIdx = r ~/ 4;
        final byteIdx = r % 4;
        final word = await wbRead(baseResult + wordIdx * 4);
        final got = int8FromBusWord(word, byteIdx);
        expect(
          got,
          equals(expectedResults[r]),
          reason:
              'non-mul result[$r]: got $got, expected ${expectedResults[r]}'
              ' (acc=${accRef[r]}, mult=${rowMults[r]}, shift=$shift)',
        );
      }
    });

    test('second op after first produces independent correct result', () async {
      // First op: W[2,2], x[2].
      final w1 = Int8List.fromList([1, 2, 3, 4]);
      final x1 = Int8List.fromList([10, -5]);
      final rm1 = <int>[16384, 16384];
      const shift1 = 14;

      await wbWrite(csrRows, 2);
      await wbWrite(csrCols, 2);
      await wbWrite(csrShift, shift1);
      await writeWeightBuffer(baseWeights, w1.toList(), 2, 2, 8);
      await writeByteBuffer(baseActivations, x1.toList());
      await writeMultBuffer(baseRowMult, rm1);
      await wbWrite(csrControl, ctrlStart);
      await pollDone();

      // Second op: W[2,4], x[4], different weights/activations.
      final w2 = Int8List.fromList([-5, 6, -7, 8, 9, -10, 11, -12]);
      final x2 = Int8List.fromList([3, -4, 5, -6]);
      final rm2 = <int>[8192, 8192];
      const shift2 = 13;

      await wbWrite(csrRows, 2);
      await wbWrite(csrCols, 4);
      await wbWrite(csrShift, shift2);
      await writeWeightBuffer(baseWeights, w2.toList(), 2, 4, 8);
      await writeByteBuffer(baseActivations, x2.toList());
      await writeMultBuffer(baseRowMult, rm2);
      await wbWrite(csrControl, ctrlStart);
      await pollDone();

      final accRef2 = matmulInt(w2, x2, 2, 4);
      for (var r = 0; r < 2; r++) {
        final exp = requantRef(accRef2[r], rm2[r], shift2);
        final wordIdx = r ~/ 4;
        final byteIdx = r % 4;
        final word = await wbRead(baseResult + wordIdx * 4);
        final got = int8FromBusWord(word, byteIdx);
        expect(
          got,
          equals(exp),
          reason: '2nd op result[$r]: got $got, expected $exp',
        );
      }
    });
  });

  group('LoomAccelerator RTL emission', () {
    test(
      'generateSynth emits non-empty SV with module LoomAccelerator',
      () async {
        final accel = LoomAccelerator(
          config: const LoomAcceleratorConfig(baseAddress: 0x70000000),
        );
        await accel.build();
        final sv = accel.generateSynth();
        expect(sv, isNotEmpty);
        expect(sv, contains('module LoomAccelerator'));
      },
    );

    test('generateAll emits rtl/ dir with at least one .sv file', () async {
      final accel = LoomAccelerator(
        config: const LoomAcceleratorConfig(baseAddress: 0x70000000),
      );
      await accel.build();

      final tmpDir = Directory.systemTemp.createTempSync('loom_accel_rtl_');
      try {
        // generateAll requires a HarborSoC. Standalone SV is sufficient here.
        // Use rohd_bridge buildAndGenerateRTL via accel.generateSynth written
        // to a file, plus verify the module name.
        final svContent = accel.generateSynth();
        final svFile = File('${tmpDir.path}/LoomAccelerator.sv');
        svFile.writeAsStringSync(svContent);

        expect(svFile.existsSync(), isTrue);
        expect(svContent, contains('LoomAccelerator'));
        expect(svContent.length, greaterThan(100));
      } finally {
        tmpDir.deleteSync(recursive: true);
      }
    });
  });
}
