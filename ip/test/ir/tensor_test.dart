import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:loom/loom.dart';

void main() {
  group('TensorDType.bytesPerElement', () {
    test('f32 is 4 bytes', () => expect(TensorDType.f32.bytesPerElement, 4));
    test('f16 is 2 bytes', () => expect(TensorDType.f16.bytesPerElement, 2));
    test('bf16 is 2 bytes', () => expect(TensorDType.bf16.bytesPerElement, 2));
    test('i8 is 1 byte', () => expect(TensorDType.i8.bytesPerElement, 1));
    test('i16 is 2 bytes', () => expect(TensorDType.i16.bytesPerElement, 2));
    test('i32 is 4 bytes', () => expect(TensorDType.i32.bytesPerElement, 4));
  });

  group('TensorDType.isFloat', () {
    test('f32 isFloat true', () => expect(TensorDType.f32.isFloat, isTrue));
    test('f16 isFloat true', () => expect(TensorDType.f16.isFloat, isTrue));
    test('bf16 isFloat true', () => expect(TensorDType.bf16.isFloat, isTrue));
    test('i8 isFloat false', () => expect(TensorDType.i8.isFloat, isFalse));
    test('i16 isFloat false', () => expect(TensorDType.i16.isFloat, isFalse));
    test('i32 isFloat false', () => expect(TensorDType.i32.isFloat, isFalse));
  });

  group('TensorView.elementCount and byteLength', () {
    test('shape [2,3] f16 => elementCount 6, byteLength 12', () {
      final bytes = ByteData(12);
      final tv = TensorView(
        name: 'x',
        shape: [2, 3],
        dtype: TensorDType.f16,
        bytes: bytes,
      );
      expect(tv.elementCount, 6);
      expect(tv.byteLength, 12);
    });

    test('empty shape => elementCount 1', () {
      final bytes = ByteData(4);
      final tv = TensorView(
        name: 'x',
        shape: [],
        dtype: TensorDType.f32,
        bytes: bytes,
      );
      expect(tv.elementCount, 1);
    });
  });

  group('TensorView constructor validation', () {
    test(
      'shape [2,2] f32 (needs 16 bytes) given 8 bytes throws ArgumentError',
      () {
        final bytes = ByteData(8);
        expect(
          () => TensorView(
            name: 'x',
            shape: [2, 2],
            dtype: TensorDType.f32,
            bytes: bytes,
          ),
          throwsArgumentError,
        );
      },
    );
  });

  group('TensorView f32 elementAt', () {
    ByteData makeF32(List<double> values) {
      final bd = ByteData(values.length * 4);
      for (var i = 0; i < values.length; i++) {
        bd.setFloat32(i * 4, values[i], Endian.little);
      }
      return bd;
    }

    test('1.5 decodes correctly', () {
      final bd = makeF32([1.5]);
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.f32,
        bytes: bd,
      );
      expect(tv.elementAt(0), closeTo(1.5, 1e-6));
    });

    test('-2.0 decodes correctly', () {
      final bd = makeF32([-2.0]);
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.f32,
        bytes: bd,
      );
      expect(tv.elementAt(0), closeTo(-2.0, 1e-6));
    });
  });

  group('TensorView f16 elementAt', () {
    ByteData makeF16(int uint16le) {
      final bd = ByteData(2);
      bd.setUint16(0, uint16le, Endian.little);
      return bd;
    }

    test('0x3C00 => 1.0', () {
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.f16,
        bytes: makeF16(0x3C00),
      );
      expect(tv.elementAt(0), closeTo(1.0, 1e-6));
    });

    test('0x4000 => 2.0', () {
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.f16,
        bytes: makeF16(0x4000),
      );
      expect(tv.elementAt(0), closeTo(2.0, 1e-6));
    });

    test('0xC000 => -2.0', () {
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.f16,
        bytes: makeF16(0xC000),
      );
      expect(tv.elementAt(0), closeTo(-2.0, 1e-6));
    });

    test('0x3800 => 0.5', () {
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.f16,
        bytes: makeF16(0x3800),
      );
      expect(tv.elementAt(0), closeTo(0.5, 1e-6));
    });

    test('0x0000 => 0.0', () {
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.f16,
        bytes: makeF16(0x0000),
      );
      expect(tv.elementAt(0), closeTo(0.0, 1e-6));
    });
  });

  group('TensorView bf16 elementAt', () {
    ByteData makeBf16(int uint16le) {
      final bd = ByteData(2);
      bd.setUint16(0, uint16le, Endian.little);
      return bd;
    }

    test('0x3F80 => 1.0', () {
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.bf16,
        bytes: makeBf16(0x3F80),
      );
      expect(tv.elementAt(0), closeTo(1.0, 1e-6));
    });

    test('0x4000 => 2.0', () {
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.bf16,
        bytes: makeBf16(0x4000),
      );
      expect(tv.elementAt(0), closeTo(2.0, 1e-6));
    });

    test('0xC000 => -2.0', () {
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.bf16,
        bytes: makeBf16(0xC000),
      );
      expect(tv.elementAt(0), closeTo(-2.0, 1e-6));
    });

    test('0x3F00 => 0.5', () {
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.bf16,
        bytes: makeBf16(0x3F00),
      );
      expect(tv.elementAt(0), closeTo(0.5, 1e-6));
    });
  });

  group('TensorView i8/i16/i32 elementAt', () {
    test('i8 0xFF => -1.0', () {
      final bd = ByteData(1)..setUint8(0, 0xFF);
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.i8,
        bytes: bd,
      );
      expect(tv.elementAt(0), closeTo(-1.0, 1e-9));
    });

    test('i8 0x7F => 127.0', () {
      final bd = ByteData(1)..setUint8(0, 0x7F);
      final tv = TensorView(
        name: 'x',
        shape: [1],
        dtype: TensorDType.i8,
        bytes: bd,
      );
      expect(tv.elementAt(0), closeTo(127.0, 1e-9));
    });
  });

  group('TensorView toFloat64List', () {
    test('3-element f32 tensor decodes all values', () {
      final bd = ByteData(12);
      bd.setFloat32(0, 1.0, Endian.little);
      bd.setFloat32(4, -3.5, Endian.little);
      bd.setFloat32(8, 42.0, Endian.little);
      final tv = TensorView(
        name: 'x',
        shape: [3],
        dtype: TensorDType.f32,
        bytes: bd,
      );
      final out = tv.toFloat64List();
      expect(out.length, 3);
      expect(out[0], closeTo(1.0, 1e-6));
      expect(out[1], closeTo(-3.5, 1e-6));
      expect(out[2], closeTo(42.0, 1e-6));
    });
  });
}
