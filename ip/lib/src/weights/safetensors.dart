library;

import 'dart:convert';
import 'dart:typed_data';

import '../ir/tensor.dart';
import 'weight_store.dart';

/// Internal descriptor parsed from the safetensors header for one tensor.
class _TensorDescriptor {
  final TensorDType dtype;
  final List<int> shape;
  final int begin;
  final int end;

  const _TensorDescriptor({
    required this.dtype,
    required this.shape,
    required this.begin,
    required this.end,
  });
}

/// Parse a safetensors dtype string to [TensorDType].
/// Throws [ArgumentError] for unsupported dtypes.
TensorDType _parseDtype(String s) {
  switch (s) {
    case 'F32':
      return TensorDType.f32;
    case 'F16':
      return TensorDType.f16;
    case 'BF16':
      return TensorDType.bf16;
    case 'I8':
      return TensorDType.i8;
    case 'I16':
      return TensorDType.i16;
    case 'I32':
      return TensorDType.i32;
    default:
      throw ArgumentError('Unsupported safetensors dtype: $s');
  }
}

/// A [WeightStore] that reads the safetensors binary format.
///
/// Format:
/// - 8 bytes: uint64 LE = header length N.
/// - N bytes: JSON header mapping tensor names to metadata.
/// - Remaining bytes: data block; data_offsets are relative to its start.
class SafetensorsStore implements WeightStore {
  final Map<String, _TensorDescriptor> _descriptors;
  final ByteData _data;
  final int _dataOffset;

  SafetensorsStore._(this._descriptors, this._data, this._dataOffset);

  /// Parse a safetensors file from [bytes].
  factory SafetensorsStore.parse(Uint8List bytes) {
    final bd = ByteData.sublistView(bytes);

    // Read the 8-byte header length (uint64 LE via two uint32s).
    final lenLo = bd.getUint32(0, Endian.little);
    final lenHi = bd.getUint32(4, Endian.little);
    final headerLen = lenLo + (lenHi * 0x100000000);

    // Decode the JSON header.
    final headerBytes = bytes.sublist(8, 8 + headerLen);
    final headerJson = utf8.decode(headerBytes);
    final headerMap = jsonDecode(headerJson) as Map<String, dynamic>;

    final dataOffset = 8 + headerLen;
    final descriptors = <String, _TensorDescriptor>{};

    for (final entry in headerMap.entries) {
      final name = entry.key;
      if (name == '__metadata__') continue;

      final meta = entry.value as Map<String, dynamic>;
      final dtypeStr = meta['dtype'] as String;
      final dtype = _parseDtype(dtypeStr);

      final shapeRaw = meta['shape'] as List<dynamic>;
      final shape = shapeRaw.cast<int>();

      final offsets = meta['data_offsets'] as List<dynamic>;
      final begin = offsets[0] as int;
      final end = offsets[1] as int;

      descriptors[name] = _TensorDescriptor(
        dtype: dtype,
        shape: shape,
        begin: begin,
        end: end,
      );
    }

    return SafetensorsStore._(descriptors, bd, dataOffset);
  }

  @override
  Iterable<String> get names => _descriptors.keys;

  @override
  bool contains(String name) => _descriptors.containsKey(name);

  @override
  TensorView get(String name) {
    final desc = _descriptors[name];
    if (desc == null) {
      throw ArgumentError('Tensor not found in store: $name');
    }

    // Validate that the byte range matches the expected size.
    final rangeBytes = desc.end - desc.begin;
    final elementCount = desc.shape.isEmpty
        ? 1
        : desc.shape.reduce((a, b) => a * b);
    final expectedBytes = elementCount * desc.dtype.bytesPerElement;

    if (rangeBytes != expectedBytes) {
      throw ArgumentError(
        'data_offsets byte range ($rangeBytes) does not match expected '
        'byteLength ($expectedBytes) for tensor "$name" '
        'with shape ${desc.shape} and dtype ${desc.dtype}',
      );
    }

    // Build a VIEW (not a copy) over the data block.
    final absoluteBegin = _dataOffset + desc.begin;
    final tensorBytes = ByteData.sublistView(
      _data.buffer.asUint8List(),
      absoluteBegin,
      absoluteBegin + rangeBytes,
    );

    return TensorView(
      name: name,
      shape: desc.shape,
      dtype: desc.dtype,
      bytes: tensorBytes,
    );
  }
}
