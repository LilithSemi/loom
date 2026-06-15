// W8A8 quantized golden reference for the Loom accelerator.
// Symmetric int8 quantization: range [-127, 127], zero-point 0.
//
// Rounding: uses Dart's double.round() which is round-half-away-from-zero
// (verified: 63.5.round()==64, (-63.5).round()==-64). This matches the
// hardware PE array behavior, so HW diff-tests can use these values directly.

import 'dart:typed_data';

/// Row-major quantized weight matrix (int8, symmetric, per-row scales).
class QuantizedMatrix {
  /// Row-major int8 values, length rows*cols.
  final Int8List values;

  /// Per-row scales, length rows. Dequant: w_fp[r,c] = values[r*cols+c] * rowScales[r].
  final Float64List rowScales;

  final int rows;
  final int cols;

  const QuantizedMatrix({
    required this.values,
    required this.rowScales,
    required this.rows,
    required this.cols,
  });
}

/// Per-tensor quantized activation vector (int8, symmetric, single scale).
class QuantizedVector {
  /// Int8 values, length matches the weight cols dimension.
  final Int8List values;

  /// Single scale for the whole vector. Dequant: x_fp[i] = values[i] * scale.
  final double scale;

  const QuantizedVector({required this.values, required this.scale});
}

/// Clamp [v] to [-127, 127].
int _clampInt8(int v) => v < -127 ? -127 : (v > 127 ? 127 : v);

/// Max absolute value of a sub-range of [data] from [start] to [start+len].
double _maxAbs(Float64List data, int start, int len) {
  var m = 0.0;
  for (var i = start; i < start + len; i++) {
    final a = data[i].abs();
    if (a > m) m = a;
  }
  return m;
}

/// Quantize a row-major weight matrix [w] (shape [rows, cols]) to symmetric
/// int8 with per-row scales.
///
/// For each row r:
///   scale = maxAbs(row) / 127.0  (1.0 if all-zero to avoid division by zero)
///   q[r,c] = clamp(round(w[r,c] / scale), -127, 127)
///
/// Dart's double.round() is round-half-away-from-zero, matching HW behavior.
QuantizedMatrix quantizeRowwiseInt8(Float64List w, int rows, int cols) {
  final values = Int8List(rows * cols);
  final rowScales = Float64List(rows);

  for (var r = 0; r < rows; r++) {
    final offset = r * cols;
    final maxAbsVal = _maxAbs(w, offset, cols);
    final scale = maxAbsVal == 0.0 ? 1.0 : maxAbsVal / 127.0;
    rowScales[r] = scale;
    for (var c = 0; c < cols; c++) {
      values[offset + c] = _clampInt8((w[offset + c] / scale).round());
    }
  }

  return QuantizedMatrix(
    values: values,
    rowScales: rowScales,
    rows: rows,
    cols: cols,
  );
}

/// Clamp [v] to the symmetric int4 range [-7, 7].
int _clampInt4(int v) => v < -7 ? -7 : (v > 7 ? 7 : v);

/// Quantize a row-major weight matrix [w] to symmetric int4 with per-row
/// scales (range [-7, 7], scale = maxAbs/7). int4 halves weight storage so a
/// 135M-param model fits the OrangeCrab's 128MB DDR3. Values are stored as
/// int8 in [-7, 7]; the integer matmul ([matmulInt]) and [dequant] are reused
/// directly (W4A8 = int4 weights, int8 activations).
QuantizedMatrix quantizeRowwiseInt4(Float64List w, int rows, int cols) {
  final values = Int8List(rows * cols);
  final rowScales = Float64List(rows);
  for (var r = 0; r < rows; r++) {
    final offset = r * cols;
    final maxAbsVal = _maxAbs(w, offset, cols);
    final scale = maxAbsVal == 0.0 ? 1.0 : maxAbsVal / 7.0;
    rowScales[r] = scale;
    for (var c = 0; c < cols; c++) {
      values[offset + c] = _clampInt4((w[offset + c] / scale).round());
    }
  }
  return QuantizedMatrix(
    values: values,
    rowScales: rowScales,
    rows: rows,
    cols: cols,
  );
}

/// Row-major weights quantized to BitNet-b1.58 ternary {-1,0,+1} with a single
/// per-tensor absmean scale beta = mean(|W|). Beta is replicated into every
/// rowScales[r] so the existing [matmulInt] + [dequant] per-row-scale path
/// reproduces per-tensor dequant, matching BitLinear inference quantization.
QuantizedMatrix quantizeTernaryAbsmean(Float64List w, int rows, int cols) {
  var sumAbs = 0.0;
  for (var i = 0; i < w.length; i++) {
    sumAbs += w[i] < 0 ? -w[i] : w[i];
  }
  final beta = w.isEmpty || sumAbs == 0.0 ? 1.0 : sumAbs / w.length;
  final values = Int8List(rows * cols);
  for (var i = 0; i < w.length; i++) {
    final q = (w[i] / beta).round();
    values[i] = q < -1 ? -1 : (q > 1 ? 1 : q);
  }
  final rowScales = Float64List(rows);
  for (var r = 0; r < rows; r++) {
    rowScales[r] = beta;
  }
  return QuantizedMatrix(
    values: values,
    rowScales: rowScales,
    rows: rows,
    cols: cols,
  );
}

/// Row-major weights quantized to symmetric int-N with PER-GROUP scales: one
/// scale per contiguous [groupSize] weights within a row (per-row = the special
/// case groupSize >= cols). Values are int-N held in int8 in [-qmax, qmax] where
/// qmax = 2^(bits-1)-1; the flash byte cost uses [bits]. Mirrors the eventual
/// hardware (per-group dequant), so study numbers are faithful to a real impl.
class GroupQuantizedMatrix {
  final Int8List values; // rows*cols, int-N in int8
  final Float64List scales; // rows*groupsPerRow, fp scale per group
  final int rows, cols, bits, groupSize;
  const GroupQuantizedMatrix({
    required this.values,
    required this.scales,
    required this.rows,
    required this.cols,
    required this.bits,
    required this.groupSize,
  });
  int get groupsPerRow => (cols + groupSize - 1) ~/ groupSize;
}

int _qmaxForBits(int bits) => (1 << (bits - 1)) - 1; // 4->7, 3->3, 2->1

GroupQuantizedMatrix quantizeGroupwise(
  Float64List w,
  int rows,
  int cols, {
  required int bits,
  required int groupSize,
}) {
  final qmax = _qmaxForBits(bits);
  final gpr = (cols + groupSize - 1) ~/ groupSize;
  final values = Int8List(rows * cols);
  final scales = Float64List(rows * gpr);
  for (var r = 0; r < rows; r++) {
    final rowOff = r * cols;
    for (var g = 0; g < gpr; g++) {
      final gStart = g * groupSize;
      final gLen = (gStart + groupSize <= cols) ? groupSize : cols - gStart;
      final maxAbsVal = _maxAbs(w, rowOff + gStart, gLen);
      final scale = maxAbsVal == 0.0 ? 1.0 : maxAbsVal / qmax;
      scales[r * gpr + g] = scale;
      for (var c = gStart; c < gStart + gLen; c++) {
        final q = (w[rowOff + c] / scale).round();
        values[rowOff + c] = q < -qmax ? -qmax : (q > qmax ? qmax : q);
      }
    }
  }
  return GroupQuantizedMatrix(
    values: values,
    scales: scales,
    rows: rows,
    cols: cols,
    bits: bits,
    groupSize: groupSize,
  );
}

/// Per-group partial accumulators: acc[r*gpr + g] = sum_{c in group g} w*x.
Int32List matmulIntGroupwise(GroupQuantizedMatrix w, QuantizedVector x) {
  if (w.cols != x.values.length) {
    throw ArgumentError(
      'w.cols (${w.cols}) != x.values.length (${x.values.length})',
    );
  }
  final gpr = w.groupsPerRow;
  final acc = Int32List(w.rows * gpr);
  for (var r = 0; r < w.rows; r++) {
    final rowOff = r * w.cols;
    for (var g = 0; g < gpr; g++) {
      final gStart = g * w.groupSize;
      final gEnd = (gStart + w.groupSize <= w.cols)
          ? gStart + w.groupSize
          : w.cols;
      var sum = 0;
      for (var c = gStart; c < gEnd; c++)
        sum += w.values[rowOff + c] * x.values[c];
      acc[r * gpr + g] = sum;
    }
  }
  return acc;
}

/// Dequant: y[r] = x.scale * sum_g groupAcc[r,g] * scale[r,g].
Float64List dequantGroupwise(
  Int32List groupAcc,
  GroupQuantizedMatrix w,
  QuantizedVector x,
) {
  final gpr = w.groupsPerRow;
  final y = Float64List(w.rows);
  for (var r = 0; r < w.rows; r++) {
    var s = 0.0;
    for (var g = 0; g < gpr; g++)
      s += groupAcc[r * gpr + g] * w.scales[r * gpr + g];
    y[r] = s * x.scale;
  }
  return y;
}

/// Groupwise W(N)A8 linear backend (drop-in for GoldenRunner.linearImpl once bits
/// and groupSize are bound via a closure).
Float64List quantizedLinearGroupwise(
  Float64List w,
  int rows,
  int cols,
  Float64List x, {
  required int bits,
  required int groupSize,
}) {
  final qm = quantizeGroupwise(w, rows, cols, bits: bits, groupSize: groupSize);
  final qv = quantizePerTensorInt8(x);
  return dequantGroupwise(matmulIntGroupwise(qm, qv), qm, qv);
}

/// W4A8 quantized approximation of linear(w, rows, cols, x): int4 row-wise
/// weights, int8 per-tensor activations, int32 accumulate, dequantize.
Float64List quantizedLinearW4A8(
  Float64List w,
  int rows,
  int cols,
  Float64List x,
) {
  final qm = quantizeRowwiseInt4(w, rows, cols);
  final qv = quantizePerTensorInt8(x);
  return dequant(matmulInt(qm, qv), qm, qv);
}

/// BitNet-b1.58 W1.58A8 linear: per-tensor ternary weights, per-tensor int8
/// activations. Drop-in [LinearImpl] for GoldenRunner (matches BitLinear
/// inference, where per-token == per-tensor at autoregressive decode).
Float64List quantizedLinearTernary(
  Float64List w,
  int rows,
  int cols,
  Float64List x,
) {
  final qm = quantizeTernaryAbsmean(w, rows, cols);
  final qv = quantizePerTensorInt8(x);
  return dequant(matmulInt(qm, qv), qm, qv);
}

/// Packs signed int4 [nibbles] (each in [-8, 7]) two-per-byte for the weight
/// image: even index -> low nibble, odd index -> high nibble (two's complement).
/// An odd count leaves the final byte's high nibble zero.
Uint8List packInt4(Int8List nibbles) {
  final out = Uint8List((nibbles.length + 1) ~/ 2);
  for (var i = 0; i < nibbles.length; i++) {
    final n = nibbles[i] & 0xF;
    if (i.isEven) {
      out[i ~/ 2] = (out[i ~/ 2] & 0xF0) | n;
    } else {
      out[i ~/ 2] = (out[i ~/ 2] & 0x0F) | (n << 4);
    }
  }
  return out;
}

/// Unpacks [count] signed int4 values from [bytes] (sign-extended to int8),
/// the inverse of [packInt4].
Int8List unpackInt4(Uint8List bytes, int count) {
  final out = Int8List(count);
  for (var i = 0; i < count; i++) {
    final raw = i.isEven ? (bytes[i ~/ 2] & 0xF) : ((bytes[i ~/ 2] >> 4) & 0xF);
    out[i] = raw >= 8 ? raw - 16 : raw; // sign-extend 4-bit
  }
  return out;
}

/// Quantize a flat activation vector [x] to symmetric int8 with a single
/// per-tensor scale.
///
///   scale = maxAbs(x) / 127.0  (1.0 if all-zero)
///   q[i] = clamp(round(x[i] / scale), -127, 127)
QuantizedVector quantizePerTensorInt8(Float64List x) {
  final maxAbsVal = _maxAbs(x, 0, x.length);
  final scale = maxAbsVal == 0.0 ? 1.0 : maxAbsVal / 127.0;
  final values = Int8List(x.length);
  for (var i = 0; i < x.length; i++) {
    values[i] = _clampInt8((x[i] / scale).round());
  }
  return QuantizedVector(values: values, scale: scale);
}

/// Exact integer matrix-vector multiply. Returns Int32List of length w.rows.
///
///   acc[r] = sum_c w.values[r*cols+c] * x.values[c]
///
/// Dart ints are 64-bit so no overflow for int8 inputs (max abs sum =
/// 127*127*cols. For cols=4096 that is ~66M, well within int64).
/// Int32List is used to match the 32-bit accumulator width of the hardware PE.
///
/// Throws [ArgumentError] if w.cols != x.values.length.
Int32List matmulInt(QuantizedMatrix w, QuantizedVector x) {
  if (w.cols != x.values.length) {
    throw ArgumentError(
      'w.cols (${w.cols}) != x.values.length (${x.values.length})',
    );
  }
  final acc = Int32List(w.rows);
  for (var r = 0; r < w.rows; r++) {
    var sum = 0;
    final offset = r * w.cols;
    for (var c = 0; c < w.cols; c++) {
      sum += w.values[offset + c] * x.values[c];
    }
    acc[r] = sum;
  }
  return acc;
}

/// Dequantize integer accumulator back to fp64.
///
///   y[r] = acc[r] * w.rowScales[r] * x.scale
Float64List dequant(Int32List acc, QuantizedMatrix w, QuantizedVector x) {
  final y = Float64List(w.rows);
  for (var r = 0; r < w.rows; r++) {
    y[r] = acc[r] * w.rowScales[r] * x.scale;
  }
  return y;
}

/// Convenience: quantize w (rowwise) and x (per-tensor), run integer matmul,
/// dequantize. Returns approximate fp64 output.
///
/// This is the W8A8 quantized approximation of linear(w, rows, cols, x).
/// The int path (quantizeRowwiseInt8, quantizePerTensorInt8, matmulInt) is
/// exposed separately so HW tests can feed the same Int8 tensors into the PE
/// array and compare accumulators bit-exactly before dequant.
Float64List quantizedLinear(Float64List w, int rows, int cols, Float64List x) {
  final qm = quantizeRowwiseInt8(w, rows, cols);
  final qv = quantizePerTensorInt8(x);
  final acc = matmulInt(qm, qv);
  return dequant(acc, qm, qv);
}
