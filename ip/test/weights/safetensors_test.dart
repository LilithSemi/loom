import 'dart:convert';
import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

/// Build a synthetic safetensors buffer.
/// [headerJson] is the JSON string for the header.
/// [dataBlock] is the raw tensor data bytes.
Uint8List buildSafetensors(String headerJson, Uint8List dataBlock) {
  final headerBytes = utf8.encode(headerJson);
  final headerLen = headerBytes.length;
  final buf = ByteData(8 + headerLen + dataBlock.length);
  // Write header length as uint64 LE (use two uint32s for safety).
  buf.setUint32(0, headerLen & 0xFFFFFFFF, Endian.little);
  buf.setUint32(4, (headerLen >> 32) & 0xFFFFFFFF, Endian.little);
  // Write header bytes.
  for (var i = 0; i < headerLen; i++) {
    buf.setUint8(8 + i, headerBytes[i]);
  }
  // Write data block.
  for (var i = 0; i < dataBlock.length; i++) {
    buf.setUint8(8 + headerLen + i, dataBlock[i]);
  }
  return buf.buffer.asUint8List();
}

void main() {
  group('SafetensorsStore', () {
    late Uint8List twoTensorFile;

    setUp(() {
      // Build data block: tensor "a" F32 [2] at [0,8]; tensor "b" F16 [2] at [8,12].
      final data = ByteData(12);
      // "a": 1.5 and -2.0 as F32 LE.
      data.setFloat32(0, 1.5, Endian.little);
      data.setFloat32(4, -2.0, Endian.little);
      // "b": 1.0 and 2.0 as F16 LE (0x3C00 = 1.0, 0x4000 = 2.0).
      data.setUint16(8, 0x3C00, Endian.little);
      data.setUint16(10, 0x4000, Endian.little);

      const header =
          '{'
          '"__metadata__": {"author": "test"},'
          '"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},'
          '"b": {"dtype": "F16", "shape": [2], "data_offsets": [8, 12]}'
          '}';

      twoTensorFile = buildSafetensors(header, data.buffer.asUint8List());
    });

    test('names contains a and b but not __metadata__', () {
      final store = SafetensorsStore.parse(twoTensorFile);
      final names = store.names.toList();
      expect(names, contains('a'));
      expect(names, contains('b'));
      expect(names, isNot(contains('__metadata__')));
    });

    test('contains returns true for known tensors and false for unknown', () {
      final store = SafetensorsStore.parse(twoTensorFile);
      expect(store.contains('a'), isTrue);
      expect(store.contains('b'), isTrue);
      expect(store.contains('zzz'), isFalse);
    });

    test('get("a") returns F32 [2] tensor with values [1.5, -2.0]', () {
      final store = SafetensorsStore.parse(twoTensorFile);
      final a = store.get('a');
      expect(a.dtype, equals(TensorDType.f32));
      expect(a.shape, equals([2]));
      final values = a.toFloat64List();
      expect(values[0], closeTo(1.5, 1e-6));
      expect(values[1], closeTo(-2.0, 1e-6));
    });

    test('get("b") returns F16 [2] tensor with values [1.0, 2.0]', () {
      final store = SafetensorsStore.parse(twoTensorFile);
      final b = store.get('b');
      expect(b.dtype, equals(TensorDType.f16));
      expect(b.shape, equals([2]));
      final values = b.toFloat64List();
      expect(values[0], closeTo(1.0, 1e-4));
      expect(values[1], closeTo(2.0, 1e-4));
    });

    test('get returns a view (not a copy) - byte identity check', () {
      final store = SafetensorsStore.parse(twoTensorFile);
      final a1 = store.get('a');
      final a2 = store.get('a');
      // Both views should point to the same underlying buffer.
      expect(a1.bytes.buffer == a2.bytes.buffer, isTrue);
    });

    test('__metadata__ key in header is ignored', () {
      final store = SafetensorsStore.parse(twoTensorFile);
      expect(store.contains('__metadata__'), isFalse);
      expect(store.names.toList(), isNot(contains('__metadata__')));
    });

    test('get missing tensor throws ArgumentError', () {
      final store = SafetensorsStore.parse(twoTensorFile);
      expect(() => store.get('missing'), throwsArgumentError);
    });

    test('unsupported dtype throws ArgumentError', () {
      final data = ByteData(8);
      const header =
          '{"x": {"dtype": "F64", "shape": [1], "data_offsets": [0, 8]}}';
      final file = buildSafetensors(header, data.buffer.asUint8List());
      // Either at parse time or at get time, ArgumentError must be thrown.
      expect(() {
        final store = SafetensorsStore.parse(file);
        store.get('x');
      }, throwsArgumentError);
    });

    test('data_offsets size mismatch throws ArgumentError', () {
      // F32 [2] expects 8 bytes, but data_offsets says [0, 4].
      final data = ByteData(4);
      const header =
          '{"bad": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}';
      final file = buildSafetensors(header, data.buffer.asUint8List());
      expect(() {
        final store = SafetensorsStore.parse(file);
        store.get('bad');
      }, throwsArgumentError);
    });
  });
}
