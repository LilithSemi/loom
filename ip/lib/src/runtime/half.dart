import 'dart:typed_data';

/// IEEE-754 binary16 (fp16) <-> double codec, matching the device's fp16
/// activation / scale / result format. Round-to-nearest-even on encode;
/// flushes subnormals to zero and saturates to +-inf on overflow.
class Half {
  Half._();

  static final Float32List _f32 = Float32List(1);
  static final ByteData _view = ByteData.view(_f32.buffer);

  static int _f32bits(double d) {
    _f32[0] = d;
    return _view.getUint32(0, Endian.little);
  }

  static double _bitsToF32(int bits) {
    _view.setUint32(0, bits & 0xFFFFFFFF, Endian.little);
    return _f32[0];
  }

  /// Encode a double to fp16 bits (0..0xFFFF).
  static int fromDouble(double d) {
    final b = _f32bits(d);
    final sign = (b >> 16) & 0x8000;
    var exp = (b >> 23) & 0xFF;
    var mant = b & 0x7FFFFF;

    if (exp == 0xFF) {
      // Inf / NaN.
      return sign | 0x7C00 | (mant != 0 ? 0x200 : 0);
    }

    // Unbias f32 (127) -> rebias f16 (15).
    var e = exp - 127 + 15;
    if (e >= 0x1F) {
      return sign | 0x7C00; // overflow -> inf
    }
    if (e <= 0) {
      // Subnormal or underflow -> flush to signed zero.
      return sign;
    }
    // Round mantissa to 10 bits, round-to-nearest-even.
    final roundBit = (mant >> 12) & 1;
    final sticky = (mant & 0xFFF) != 0 ? 1 : 0;
    var m10 = mant >> 13;
    if (roundBit == 1 && (sticky == 1 || (m10 & 1) == 1)) {
      m10 += 1;
      if (m10 == 0x400) {
        m10 = 0;
        e += 1;
        if (e >= 0x1F) return sign | 0x7C00;
      }
    }
    return sign | (e << 10) | m10;
  }

  /// Decode fp16 bits to a double.
  static double toDouble(int h) {
    final sign = (h & 0x8000) << 16;
    final exp = (h >> 10) & 0x1F;
    final mant = h & 0x3FF;
    int f32;
    if (exp == 0) {
      if (mant == 0) {
        f32 = sign; // signed zero
      } else {
        // Subnormal fp16 -> normalize into f32.
        var e = -1;
        var m = mant;
        do {
          e += 1;
          m <<= 1;
        } while ((m & 0x400) == 0);
        m &= 0x3FF;
        final f32exp = (127 - 15 - e) << 23;
        f32 = sign | f32exp | (m << 13);
      }
    } else if (exp == 0x1F) {
      f32 = sign | 0x7F800000 | (mant << 13); // inf / nan
    } else {
      final f32exp = (exp - 15 + 127) << 23;
      f32 = sign | f32exp | (mant << 13);
    }
    return _bitsToF32(f32);
  }
}
