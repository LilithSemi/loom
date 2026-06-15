import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

// ---------------------------------------------------------------------------
// Synthetic GGUF writer helper
// ---------------------------------------------------------------------------

/// A small byte-buffer builder that grows as needed.
class _BufWriter {
  final _chunks = <int>[];

  void writeUint8(int v) => _chunks.add(v & 0xFF);

  void writeUint32(int v) {
    final bd = ByteData(4)..setUint32(0, v, Endian.little);
    for (var i = 0; i < 4; i++) {
      _chunks.add(bd.getUint8(i));
    }
  }

  void writeInt8(int v) {
    final bd = ByteData(1)..setInt8(0, v);
    _chunks.add(bd.getUint8(0));
  }

  void writeUint16(int v) {
    final bd = ByteData(2)..setUint16(0, v, Endian.little);
    for (var i = 0; i < 2; i++) {
      _chunks.add(bd.getUint8(i));
    }
  }

  void writeUint64(int v) {
    // Dart int is 64-bit; split into two 32-bit halves.
    final lo = v & 0xFFFFFFFF;
    final hi = (v >> 32) & 0xFFFFFFFF;
    writeUint32(lo);
    writeUint32(hi);
  }

  void writeFloat32(double v) {
    final bd = ByteData(4)..setFloat32(0, v, Endian.little);
    for (var i = 0; i < 4; i++) {
      _chunks.add(bd.getUint8(i));
    }
  }

  void writeGgufString(String s) {
    final bytes = _utf8Encode(s);
    writeUint64(bytes.length);
    for (final b in bytes) {
      _chunks.add(b);
    }
  }

  void writeBytes(List<int> bytes) {
    for (final b in bytes) {
      _chunks.add(b);
    }
  }

  Uint8List toBytes() => Uint8List.fromList(_chunks);

  int get length => _chunks.length;

  static List<int> _utf8Encode(String s) {
    final units = <int>[];
    for (final rune in s.runes) {
      if (rune < 0x80) {
        units.add(rune);
      } else if (rune < 0x800) {
        units.add(0xC0 | (rune >> 6));
        units.add(0x80 | (rune & 0x3F));
      } else if (rune < 0x10000) {
        units.add(0xE0 | (rune >> 12));
        units.add(0x80 | ((rune >> 6) & 0x3F));
        units.add(0x80 | (rune & 0x3F));
      } else {
        units.add(0xF0 | (rune >> 18));
        units.add(0x80 | ((rune >> 12) & 0x3F));
        units.add(0x80 | ((rune >> 6) & 0x3F));
        units.add(0x80 | (rune & 0x3F));
      }
    }
    return units;
  }
}

// GGUF metadata value_type constants (only those used by the writer helpers)
const int _kvUint32 = 4;
const int _kvString = 8;
const int _kvArray = 9;

// ggml_type codes
const int _ggmlF32 = 0;
const int _ggmlF16 = 1;
const int _ggmlQ8_0 = 8;

/// Write magic + version + tensor_count + kv_count header to a writer.
void _writeGgufPrologue(
  _BufWriter w, {
  required int version,
  required int tensorCount,
  required int kvCount,
}) {
  // Magic: 'GGUF' = 0x46554747 LE
  w.writeUint8(0x47);
  w.writeUint8(0x47);
  w.writeUint8(0x55);
  w.writeUint8(0x46);
  w.writeUint32(version);
  w.writeUint64(tensorCount);
  w.writeUint64(kvCount);
}

/// Write a STRING kv pair.
void _writeKvString(_BufWriter w, String key, String value) {
  w.writeGgufString(key);
  w.writeUint32(_kvString);
  w.writeGgufString(value);
}

/// Write a UINT32 kv pair.
void _writeKvUint32(_BufWriter w, String key, int value) {
  w.writeGgufString(key);
  w.writeUint32(_kvUint32);
  w.writeUint32(value);
}

/// Write a UINT32 ARRAY kv pair.
void _writeKvUint32Array(_BufWriter w, String key, List<int> values) {
  w.writeGgufString(key);
  w.writeUint32(_kvArray);
  w.writeUint32(_kvUint32); // arr_type
  w.writeUint64(values.length); // arr_len
  for (final v in values) {
    w.writeUint32(v);
  }
}

/// Write a tensor info entry.
void _writeTensorInfo(
  _BufWriter w, {
  required String name,
  required List<int> dims,
  required int ggmlType,
  required int offset,
}) {
  w.writeGgufString(name);
  w.writeUint32(dims.length); // n_dims
  for (final d in dims) {
    w.writeUint64(d);
  }
  w.writeUint32(ggmlType);
  w.writeUint64(offset);
}

/// Pad writer to alignment boundary (relative to file start = offset 0).
void _padToAlignment(_BufWriter w, int alignment) {
  final pos = w.length;
  final rem = pos % alignment;
  if (rem != 0) {
    final pad = alignment - rem;
    for (var i = 0; i < pad; i++) {
      w.writeUint8(0);
    }
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void main() {
  group('GgufStore', () {
    // -----------------------------------------------------------------------
    // Metadata parsing
    // -----------------------------------------------------------------------
    group('metadata', () {
      late GgufStore store;

      setUp(() {
        final w = _BufWriter();
        _writeGgufPrologue(w, version: 3, tensorCount: 0, kvCount: 3);
        _writeKvString(w, 'general.architecture', 'llama');
        _writeKvUint32(w, 'general.alignment', 32);
        _writeKvUint32(w, 'llama.block_count', 2);
        // no tensors, no data section padding needed
        store = GgufStore.parse(w.toBytes());
      });

      test('metaString returns string value', () {
        expect(store.metaString('general.architecture'), equals('llama'));
      });

      test('metaInt returns uint32 value', () {
        expect(store.metaInt('llama.block_count'), equals(2));
      });

      test('meta returns null for missing key', () {
        expect(store.meta('nonexistent'), isNull);
      });

      test('names is empty when no tensors', () {
        expect(store.names, isEmpty);
      });
    });

    // -----------------------------------------------------------------------
    // ARRAY metadata
    // -----------------------------------------------------------------------
    group('array metadata', () {
      test('uint32 array parses as List<int>', () {
        final w = _BufWriter();
        _writeGgufPrologue(w, version: 3, tensorCount: 0, kvCount: 1);
        _writeKvUint32Array(w, 'test.array', [1, 2, 3]);
        final store = GgufStore.parse(w.toBytes());
        final val = store.meta('test.array');
        expect(val, isA<List>());
        final list = val as List;
        expect(list, equals([1, 2, 3]));
      });
    });

    // -----------------------------------------------------------------------
    // F32 tensor
    // -----------------------------------------------------------------------
    group('F32 tensor', () {
      late GgufStore store;
      final values = [1.5, -2.0, 3.0, 0.25];

      setUp(() {
        final w = _BufWriter();
        _writeGgufPrologue(w, version: 3, tensorCount: 1, kvCount: 1);
        _writeKvUint32(w, 'general.alignment', 32);
        // Tensor info: "a" shape [4] f32, offset 0
        _writeTensorInfo(
          w,
          name: 'a',
          dims: [4],
          ggmlType: _ggmlF32,
          offset: 0,
        );
        // Pad to alignment=32
        _padToAlignment(w, 32);
        // Data: 4 x float32
        for (final v in values) {
          w.writeFloat32(v);
        }
        store = GgufStore.parse(w.toBytes());
      });

      test('names contains "a"', () {
        expect(store.names, contains('a'));
      });

      test('contains("a") is true', () {
        expect(store.contains('a'), isTrue);
      });

      test('contains("z") is false', () {
        expect(store.contains('z'), isFalse);
      });

      test('typeOf("a") is GgmlType.f32', () {
        expect(store.typeOf('a'), equals(GgmlType.f32));
      });

      test('shapeOf("a") is [4]', () {
        expect(store.shapeOf('a'), equals([4]));
      });

      test('get("a").toFloat64List() matches values', () {
        final tv = store.get('a');
        expect(tv.dtype, equals(TensorDType.f32));
        final decoded = tv.toFloat64List();
        for (var i = 0; i < values.length; i++) {
          expect(decoded[i], closeTo(values[i], 1e-5));
        }
      });

      test('dequantize("a") matches values', () {
        final decoded = store.dequantize('a');
        for (var i = 0; i < values.length; i++) {
          expect(decoded[i], closeTo(values[i], 1e-5));
        }
      });
    });

    // -----------------------------------------------------------------------
    // F16 tensor
    // -----------------------------------------------------------------------
    group('F16 tensor', () {
      late GgufStore store;

      setUp(() {
        final w = _BufWriter();
        _writeGgufPrologue(w, version: 3, tensorCount: 1, kvCount: 1);
        _writeKvUint32(w, 'general.alignment', 32);
        _writeTensorInfo(
          w,
          name: 'b',
          dims: [2],
          ggmlType: _ggmlF16,
          offset: 0,
        );
        _padToAlignment(w, 32);
        // 0x3C00 = 1.0, 0x4000 = 2.0 in fp16
        w.writeUint16(0x3C00);
        w.writeUint16(0x4000);
        store = GgufStore.parse(w.toBytes());
      });

      test('get("b") has dtype f16 and shape [2]', () {
        final tv = store.get('b');
        expect(tv.dtype, equals(TensorDType.f16));
        expect(tv.shape, equals([2]));
      });

      test('get("b").toFloat64List() ~ [1.0, 2.0]', () {
        final vals = store.get('b').toFloat64List();
        expect(vals[0], closeTo(1.0, 1e-3));
        expect(vals[1], closeTo(2.0, 1e-3));
      });

      test('dequantize("b") ~ [1.0, 2.0]', () {
        final vals = store.dequantize('b');
        expect(vals[0], closeTo(1.0, 1e-3));
        expect(vals[1], closeTo(2.0, 1e-3));
      });
    });

    // -----------------------------------------------------------------------
    // Q8_0 tensor
    // -----------------------------------------------------------------------
    group('Q8_0 tensor', () {
      late GgufStore store;
      // scale d = 0.5 as fp16 = 0x3800; qs[i] = i - 16 for i in 0..31
      const int scaleHalf = 0x3800; // fp16 for 0.5
      final qs = List<int>.generate(32, (i) => i - 16); // -16..15

      setUp(() {
        final w = _BufWriter();
        _writeGgufPrologue(w, version: 3, tensorCount: 1, kvCount: 1);
        _writeKvUint32(w, 'general.alignment', 32);
        _writeTensorInfo(
          w,
          name: 'c',
          dims: [32],
          ggmlType: _ggmlQ8_0,
          offset: 0,
        );
        _padToAlignment(w, 32);
        // Q8_0 block: 2-byte fp16 scale + 32 int8 quants = 34 bytes
        w.writeUint16(scaleHalf);
        for (final q in qs) {
          w.writeInt8(q);
        }
        store = GgufStore.parse(w.toBytes());
      });

      test('get("c") throws ArgumentError (quantized)', () {
        expect(() => store.get('c'), throwsArgumentError);
      });

      test('dequantize("c") returns 32 values matching 0.5 * (i - 16)', () {
        final vals = store.dequantize('c');
        expect(vals.length, equals(32));
        for (var i = 0; i < 32; i++) {
          expect(vals[i], closeTo(0.5 * (i - 16), 1e-3));
        }
      });
    });

    // -----------------------------------------------------------------------
    // Alignment test: tensor data starts at correct aligned offset
    // -----------------------------------------------------------------------
    group('alignment', () {
      test(
        'tensor data reads correctly despite padding between info and data',
        () {
          // Use alignment=64 so there is definitely non-trivial padding.
          // Build the file and verify the tensor still decodes correctly.
          final w = _BufWriter();
          _writeGgufPrologue(w, version: 3, tensorCount: 1, kvCount: 1);
          _writeKvUint32(w, 'general.alignment', 64);
          _writeTensorInfo(
            w,
            name: 'pad_test',
            dims: [2],
            ggmlType: _ggmlF32,
            offset: 0,
          );
          _padToAlignment(w, 64);
          w.writeFloat32(7.0);
          w.writeFloat32(-3.5);
          final store = GgufStore.parse(w.toBytes());
          final vals = store.get('pad_test').toFloat64List();
          expect(vals[0], closeTo(7.0, 1e-5));
          expect(vals[1], closeTo(-3.5, 1e-5));
        },
      );
    });

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------
    group('errors', () {
      test('bad magic throws ArgumentError', () {
        final bad = Uint8List.fromList([
          0x00,
          0x00,
          0x00,
          0x00,
          0x03,
          0x00,
          0x00,
          0x00,
        ]);
        expect(() => GgufStore.parse(bad), throwsArgumentError);
      });

      test('unknown ggml type in tensor info throws ArgumentError', () {
        final w = _BufWriter();
        _writeGgufPrologue(w, version: 3, tensorCount: 1, kvCount: 0);
        // Use ggml type 255 which is not defined
        _writeTensorInfo(w, name: 'x', dims: [1], ggmlType: 255, offset: 0);
        expect(() => GgufStore.parse(w.toBytes()), throwsArgumentError);
      });
    });
  });
}
