import 'dart:typed_data';

/// Describes the element data type of a tensor.
enum TensorDType {
  f32(4, true),
  f16(2, true),
  bf16(2, true),
  i8(1, false),
  i16(2, false),
  i32(4, false);

  const TensorDType(this.bytesPerElement, this.isFloat);

  /// Number of bytes per element for this dtype.
  final int bytesPerElement;

  /// True for floating-point dtypes (f32, f16, bf16).
  final bool isFloat;
}

/// A view over a raw little-endian byte buffer for a single named tensor.
class TensorView {
  final String name;
  final List<int> shape;
  final TensorDType dtype;
  final ByteData bytes;

  /// Memoized fp64 decode. Weights are read-only for every consumer (the golden
  /// runner reads them, quantizers copy-then-mutate), so decoding once and
  /// sharing the array is safe and gives a STABLE object identity across
  /// GoldenRunner instances (needed to key calibration stats by weight).
  Float64List? _f64cache;

  /// Product of [shape] entries; 1 for an empty (scalar) shape.
  int get elementCount {
    if (shape.isEmpty) return 1;
    return shape.reduce((a, b) => a * b);
  }

  /// Total byte length: elementCount * dtype.bytesPerElement.
  int get byteLength => elementCount * dtype.bytesPerElement;

  TensorView({
    required this.name,
    required this.shape,
    required this.dtype,
    required this.bytes,
  }) {
    if (bytes.lengthInBytes != byteLength) {
      throw ArgumentError(
        'bytes.lengthInBytes (${bytes.lengthInBytes}) != '
        'expected byteLength ($byteLength) for shape $shape and dtype $dtype',
      );
    }
  }

  /// Decode the i-th element as a double.
  double elementAt(int i) {
    switch (dtype) {
      case TensorDType.f32:
        return bytes.getFloat32(i * 4, Endian.little).toDouble();

      case TensorDType.f16:
        return _decodeF16(bytes.getUint16(i * 2, Endian.little));

      case TensorDType.bf16:
        return _decodeBf16(bytes.getUint16(i * 2, Endian.little));

      case TensorDType.i8:
        return bytes.getInt8(i).toDouble();

      case TensorDType.i16:
        return bytes.getInt16(i * 2, Endian.little).toDouble();

      case TensorDType.i32:
        return bytes.getInt32(i * 4, Endian.little).toDouble();
    }
  }

  /// Decode all elements into a [Float64List] (memoized. See [_f64cache]).
  Float64List toFloat64List() {
    if (_f64cache != null) return _f64cache!;
    final result = Float64List(elementCount);
    for (var i = 0; i < elementCount; i++) {
      result[i] = elementAt(i);
    }
    _f64cache = result;
    return result;
  }

  /// Decode an IEEE-754 half-precision float from a raw uint16.
  static double _decodeF16(int raw) {
    final sign = (raw >> 15) & 1;
    final exp = (raw >> 10) & 0x1F;
    final mant = raw & 0x3FF;
    final signFactor = sign == 0 ? 1.0 : -1.0;

    if (exp == 0) {
      // Subnormal (or zero when mant==0 too)
      return signFactor * _pow2(-14) * (mant / 1024.0);
    } else if (exp == 0x1F) {
      // Inf or NaN
      if (mant == 0) {
        return sign == 0 ? double.infinity : double.negativeInfinity;
      }
      return double.nan;
    } else {
      // Normal
      return signFactor * _pow2(exp - 15) * (1.0 + mant / 1024.0);
    }
  }

  /// Decode a bfloat16 value from a raw uint16.
  /// bf16 is the top 16 bits of an IEEE-754 float32.
  static double _decodeBf16(int raw) {
    final buf = ByteData(4);
    // Place the 16-bit value in the upper two bytes of a float32.
    buf.setUint32(0, raw << 16, Endian.big);
    return buf.getFloat32(0, Endian.big).toDouble();
  }

  /// Fast integer power of 2 (positive or negative exponent).
  static double _pow2(int exp) {
    // Using bit manipulation for speed on whole-number exponents.
    if (exp >= 0) {
      return (1 << exp).toDouble();
    } else {
      return 1.0 / (1 << -exp).toDouble();
    }
  }
}
