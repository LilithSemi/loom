library;

import 'dart:typed_data';

import '../ir/tensor.dart';
import 'weight_store.dart';

/// All recognized ggml tensor types.
enum GgmlType {
  f32,
  f16,
  q4_0,
  q4_1,
  q5_0,
  q5_1,
  q8_0,
  q8_1,
  q2k,
  q3k,
  q4k,
  q5k,
  q6k,
  q8k;

  /// Construct from the integer code stored in a GGUF tensor-info entry.
  /// Throws [ArgumentError] for unrecognized codes.
  static GgmlType fromCode(int code) {
    switch (code) {
      case 0:
        return GgmlType.f32;
      case 1:
        return GgmlType.f16;
      case 2:
        return GgmlType.q4_0;
      case 3:
        return GgmlType.q4_1;
      case 6:
        return GgmlType.q5_0;
      case 7:
        return GgmlType.q5_1;
      case 8:
        return GgmlType.q8_0;
      case 9:
        return GgmlType.q8_1;
      case 10:
        return GgmlType.q2k;
      case 11:
        return GgmlType.q3k;
      case 12:
        return GgmlType.q4k;
      case 13:
        return GgmlType.q5k;
      case 14:
        return GgmlType.q6k;
      case 15:
        return GgmlType.q8k;
      default:
        throw ArgumentError('Unknown ggml type code: $code');
    }
  }
}

/// Internal tensor descriptor parsed from a GGUF tensor-info block.
class _TensorInfo {
  final String name;
  final List<int> dims;
  final GgmlType ggmlType;
  final int offset; // relative to start of data section

  const _TensorInfo({
    required this.name,
    required this.dims,
    required this.ggmlType,
    required this.offset,
  });
}

// ---------------------------------------------------------------------------
// GGUF metadata value-type codes
// ---------------------------------------------------------------------------
const int _kvUint8 = 0;
const int _kvInt8 = 1;
const int _kvUint16 = 2;
const int _kvInt16 = 3;
const int _kvUint32 = 4;
const int _kvInt32 = 5;
const int _kvFloat32 = 6;
const int _kvBool = 7;
const int _kvString = 8;
const int _kvArray = 9;
const int _kvUint64 = 10;
const int _kvInt64 = 11;
const int _kvFloat64 = 12;

const int _ggufMagic = 0x46554747; // 'GGUF'
const int _defaultAlignment = 32;

/// A [WeightStore] that parses the GGUF v2/v3 binary format (little-endian).
///
/// Parse a GGUF file buffer with [GgufStore.parse].
class GgufStore implements WeightStore {
  final Map<String, Object?> _metadata;
  final Map<String, _TensorInfo> _tensors;
  final ByteData _data;
  final int _dataSectionStart;

  GgufStore._(
    this._metadata,
    this._tensors,
    this._data,
    this._dataSectionStart,
  );

  // --------------------------------------------------------------------------
  // Factory / parse
  // --------------------------------------------------------------------------

  /// Parse a GGUF buffer from [data].
  ///
  /// Throws [ArgumentError] if the magic bytes are wrong or an unknown ggml
  /// type code is encountered.
  factory GgufStore.parse(Uint8List data) {
    final bd = ByteData.sublistView(data);
    var pos = 0;

    // 1. Magic
    final magic = bd.getUint32(pos, Endian.little);
    if (magic != _ggufMagic) {
      throw ArgumentError(
        'Not a GGUF file: expected magic 0x${_ggufMagic.toRadixString(16)}, '
        'got 0x${magic.toRadixString(16)}',
      );
    }
    pos += 4;

    // 2. Version (uint32)
    // final version = bd.getUint32(pos, Endian.little); // recorded but not validated further
    pos += 4;

    // 3. tensor_count (uint64)
    final tensorCount = _readUint64(bd, pos);
    pos += 8;

    // 4. metadata_kv_count (uint64)
    final kvCount = _readUint64(bd, pos);
    pos += 8;

    // 5. Metadata KV pairs
    final metadata = <String, Object?>{};
    for (var i = 0; i < kvCount; i++) {
      final keyResult = _readGgufString(bd, pos);
      pos = keyResult.$1;
      final key = keyResult.$2;

      final valueType = bd.getUint32(pos, Endian.little);
      pos += 4;

      final valResult = _readValue(bd, pos, valueType);
      pos = valResult.$1;
      metadata[key] = valResult.$2;
    }

    // 6. Tensor infos
    final tensors = <String, _TensorInfo>{};
    for (var i = 0; i < tensorCount; i++) {
      final nameResult = _readGgufString(bd, pos);
      pos = nameResult.$1;
      final name = nameResult.$2;

      final nDims = bd.getUint32(pos, Endian.little);
      pos += 4;

      final dims = <int>[];
      for (var d = 0; d < nDims; d++) {
        dims.add(_readUint64(bd, pos));
        pos += 8;
      }

      final ggmlCode = bd.getUint32(pos, Endian.little);
      pos += 4;
      final ggmlType = GgmlType.fromCode(ggmlCode); // throws on unknown

      final offset = _readUint64(bd, pos);
      pos += 8;

      tensors[name] = _TensorInfo(
        name: name,
        dims: dims,
        ggmlType: ggmlType,
        offset: offset,
      );
    }

    // 7. Alignment padding
    final alignment = () {
      final a = metadata['general.alignment'];
      if (a is int) return a;
      return _defaultAlignment;
    }();

    // Advance pos to next multiple of alignment.
    final rem = pos % alignment;
    if (rem != 0) {
      pos += alignment - rem;
    }

    final dataSectionStart = pos;

    return GgufStore._(metadata, tensors, bd, dataSectionStart);
  }

  // --------------------------------------------------------------------------
  // WeightStore interface
  // --------------------------------------------------------------------------

  @override
  Iterable<String> get names => _tensors.keys;

  @override
  bool contains(String name) => _tensors.containsKey(name);

  @override
  TensorView get(String name) {
    final info = _tensors[name];
    if (info == null) throw ArgumentError('Tensor not found: $name');

    switch (info.ggmlType) {
      case GgmlType.f32:
        return _viewAs(info, TensorDType.f32);
      case GgmlType.f16:
        return _viewAs(info, TensorDType.f16);
      default:
        throw ArgumentError(
          'use dequantize() for quantized tensor ${info.name} (${info.ggmlType})',
        );
    }
  }

  // --------------------------------------------------------------------------
  // Extra accessors
  // --------------------------------------------------------------------------

  /// Returns the raw metadata value for [key], or null if absent.
  Object? meta(String key) => _metadata[key];

  /// Returns metadata value for [key] as an [int], or null.
  int? metaInt(String key) {
    final v = _metadata[key];
    if (v is int) return v;
    return null;
  }

  /// Returns metadata value for [key] as a [String], or null.
  String? metaString(String key) {
    final v = _metadata[key];
    if (v is String) return v;
    return null;
  }

  /// Returns the [GgmlType] for tensor [name].
  GgmlType typeOf(String name) {
    final info = _tensors[name];
    if (info == null) throw ArgumentError('Tensor not found: $name');
    return info.ggmlType;
  }

  /// Returns the dimension list for tensor [name] (as stored in the file).
  List<int> shapeOf(String name) {
    final info = _tensors[name];
    if (info == null) throw ArgumentError('Tensor not found: $name');
    return info.dims;
  }

  /// Dequantize tensor [name] to a [Float64List].
  ///
  /// Supports f32, f16, and q8_0. Throws [ArgumentError] for other types.
  Float64List dequantize(String name) {
    final info = _tensors[name];
    if (info == null) throw ArgumentError('Tensor not found: $name');

    switch (info.ggmlType) {
      case GgmlType.f32:
        return _viewAs(info, TensorDType.f32).toFloat64List();

      case GgmlType.f16:
        return _viewAs(info, TensorDType.f16).toFloat64List();

      case GgmlType.q8_0:
        return _dequantizeQ8_0(info);

      default:
        throw ArgumentError(
          'dequantize not yet supported for ${info.ggmlType} (tensor ${info.name})',
        );
    }
  }

  // --------------------------------------------------------------------------
  // Internal helpers
  // --------------------------------------------------------------------------

  TensorView _viewAs(_TensorInfo info, TensorDType dtype) {
    final elementCount = info.dims.isEmpty
        ? 1
        : info.dims.reduce((a, b) => a * b);
    final byteLen = elementCount * dtype.bytesPerElement;
    final start = _dataSectionStart + info.offset;
    final bytes = ByteData.sublistView(
      _data.buffer.asUint8List(),
      start,
      start + byteLen,
    );
    return TensorView(
      name: info.name,
      shape: info.dims,
      dtype: dtype,
      bytes: bytes,
    );
  }

  Float64List _dequantizeQ8_0(_TensorInfo info) {
    final elementCount = info.dims.isEmpty
        ? 1
        : info.dims.reduce((a, b) => a * b);
    const blockSize = 32;
    const blockBytes = 34; // 2 (fp16 scale) + 32 (int8 quants)
    final blockCount = elementCount ~/ blockSize;
    final result = Float64List(elementCount);
    var byteOff = _dataSectionStart + info.offset;

    for (var b = 0; b < blockCount; b++) {
      final dRaw = _data.getUint16(byteOff, Endian.little);
      final d = _decodeF16(dRaw);
      byteOff += 2;
      for (var i = 0; i < blockSize; i++) {
        final q = _data.getInt8(byteOff + i);
        result[b * blockSize + i] = d * q;
      }
      byteOff += blockSize;
    }
    // Sanity: byteOff == dataSectionStart + info.offset + blockCount * blockBytes
    assert(
      byteOff == _dataSectionStart + info.offset + blockCount * blockBytes,
    );
    return result;
  }

  // --------------------------------------------------------------------------
  // Static parsing helpers
  // --------------------------------------------------------------------------

  /// Read a uint64 from [bd] at [pos] (little-endian, via two uint32s).
  static int _readUint64(ByteData bd, int pos) {
    final lo = bd.getUint32(pos, Endian.little);
    final hi = bd.getUint32(pos + 4, Endian.little);
    return lo + hi * 0x100000000;
  }

  /// Read a gguf-string (uint64 len + UTF-8 bytes).
  /// Returns (newPos, string).
  static (int, String) _readGgufString(ByteData bd, int pos) {
    final len = _readUint64(bd, pos);
    pos += 8;
    final bytes = bd.buffer.asUint8List(bd.offsetInBytes + pos, len);
    pos += len;
    return (pos, _decodeUtf8(bytes));
  }

  /// Read a single scalar metadata value of [valueType].
  /// Returns (newPos, value).
  static (int, Object?) _readValue(ByteData bd, int pos, int valueType) {
    switch (valueType) {
      case _kvUint8:
        return (pos + 1, bd.getUint8(pos));
      case _kvInt8:
        return (pos + 1, bd.getInt8(pos));
      case _kvUint16:
        return (pos + 2, bd.getUint16(pos, Endian.little));
      case _kvInt16:
        return (pos + 2, bd.getInt16(pos, Endian.little));
      case _kvUint32:
        return (pos + 4, bd.getUint32(pos, Endian.little));
      case _kvInt32:
        return (pos + 4, bd.getInt32(pos, Endian.little));
      case _kvFloat32:
        return (pos + 4, bd.getFloat32(pos, Endian.little));
      case _kvBool:
        return (pos + 1, bd.getUint8(pos) != 0);
      case _kvString:
        final r = _readGgufString(bd, pos);
        return (r.$1, r.$2);
      case _kvUint64:
        return (pos + 8, _readUint64(bd, pos));
      case _kvInt64:
        final lo = bd.getUint32(pos, Endian.little);
        final hi = bd.getInt32(pos + 4, Endian.little);
        return (pos + 8, lo + hi * 0x100000000);
      case _kvFloat64:
        return (pos + 8, bd.getFloat64(pos, Endian.little));
      case _kvArray:
        return _readArray(bd, pos);
      default:
        throw ArgumentError('Unknown GGUF metadata value_type: $valueType');
    }
  }

  /// Read an ARRAY metadata value.
  static (int, List<Object?>) _readArray(ByteData bd, int pos) {
    final arrType = bd.getUint32(pos, Endian.little);
    pos += 4;
    final arrLen = _readUint64(bd, pos);
    pos += 8;
    final list = <Object?>[];
    for (var i = 0; i < arrLen; i++) {
      final r = _readValue(bd, pos, arrType);
      pos = r.$1;
      list.add(r.$2);
    }
    return (pos, list);
  }

  /// Decode IEEE-754 half-precision float from uint16.
  static double _decodeF16(int raw) {
    final sign = (raw >> 15) & 1;
    final exp = (raw >> 10) & 0x1F;
    final mant = raw & 0x3FF;
    final signFactor = sign == 0 ? 1.0 : -1.0;
    if (exp == 0) {
      return signFactor * _pow2(-14) * (mant / 1024.0);
    } else if (exp == 0x1F) {
      if (mant == 0) {
        return sign == 0 ? double.infinity : double.negativeInfinity;
      }
      return double.nan;
    } else {
      return signFactor * _pow2(exp - 15) * (1.0 + mant / 1024.0);
    }
  }

  static double _pow2(int exp) {
    if (exp >= 0) return (1 << exp).toDouble();
    return 1.0 / (1 << -exp).toDouble();
  }

  /// Minimal UTF-8 decoder (avoids dart:convert dependency in core lib).
  static String _decodeUtf8(Uint8List bytes) {
    // Use dart:core String.fromCharCodes - handles ASCII fast, and Dart's
    // String.fromCharCodes handles Latin-1 but not full UTF-8 multi-byte.
    // Use a proper decode via codeUnits reconstruction.
    final buf = StringBuffer();
    var i = 0;
    while (i < bytes.length) {
      final b = bytes[i];
      if (b < 0x80) {
        buf.writeCharCode(b);
        i++;
      } else if (b < 0xE0) {
        final cp = ((b & 0x1F) << 6) | (bytes[i + 1] & 0x3F);
        buf.writeCharCode(cp);
        i += 2;
      } else if (b < 0xF0) {
        final cp =
            ((b & 0x0F) << 12) |
            ((bytes[i + 1] & 0x3F) << 6) |
            (bytes[i + 2] & 0x3F);
        buf.writeCharCode(cp);
        i += 3;
      } else {
        final cp =
            ((b & 0x07) << 18) |
            ((bytes[i + 1] & 0x3F) << 12) |
            ((bytes[i + 2] & 0x3F) << 6) |
            (bytes[i + 3] & 0x3F);
        buf.writeCharCode(cp);
        i += 4;
      }
    }
    return buf.toString();
  }
}
