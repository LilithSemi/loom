import 'dart:typed_data';

import '../golden/quant.dart';
import 'half.dart';

/// A raw link to the Loom device: little-endian Wishbone-style byte writes and
/// reads at absolute addresses, exactly the WRITE (0x01) / READ (0x02) frames
/// the LoomUartBridge / LoomUsbDevice command engine speaks. A real
/// implementation wraps a serial port; [EmulatedLoomDevice] implements it in
/// process for sim + tests.
abstract class LoomLink {
  /// Write [bytes] (little-endian) starting at [addr].
  void write(int addr, List<int> bytes);

  /// Read [len] bytes (little-endian) starting at [addr].
  List<int> read(int addr, int len);

  /// Convenience: write one 32-bit word.
  void writeWord(int addr, int value) => write(addr, [
    value & 0xFF,
    (value >> 8) & 0xFF,
    (value >> 16) & 0xFF,
    (value >> 24) & 0xFF,
  ]);

  /// Convenience: read one 32-bit word.
  int readWord(int addr) {
    final b = read(addr, 4);
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
  }
}

/// Pack an int4 weight matrix (values in [-7,7], row-major [rows,cols]) into the
/// device's TILE-MAJOR byte image: peR=peC=2, two 2x2 tiles per 32-bit word,
/// nibbles [w00,w01,w10,w11] - the layout LoomStreamMatmul reads.
Uint8List packTileMajorInt4(Int8List values, int rows, int cols) {
  const peR = 2, peC = 2;
  final rowBlocks = (rows + peR - 1) ~/ peR;
  final colTiles = (cols + peC - 1) ~/ peC;
  final wordsPerRow = (colTiles + 1) ~/ 2;
  final out = Uint8List(rowBlocks * wordsPerRow * 4);
  int v(int gr, int gc) =>
      (gr < rows && gc < cols) ? (values[gr * cols + gc] & 0xF) : 0;
  for (var rb = 0; rb < rowBlocks; rb++) {
    for (var ct = 0; ct < colTiles; ct++) {
      final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
      final tileByte = wordByte + (ct.isOdd ? 2 : 0);
      out[tileByte] = v(rb * peR, ct * peC) | (v(rb * peR, ct * peC + 1) << 4);
      out[tileByte + 1] =
          v(rb * peR + 1, ct * peC) | (v(rb * peR + 1, ct * peC + 1) << 4);
    }
  }
  return out;
}

int _sext4(int n) => (n & 0x8) != 0 ? (n & 0xF) - 16 : (n & 0xF);

/// Inverse of [packTileMajorInt4]: read a [rows]x[cols] int4 matrix back from a
/// tile-major byte image (sign-extended to int8).
Int8List unpackTileMajorInt4(Uint8List bytes, int rows, int cols) {
  const peR = 2, peC = 2;
  final rowBlocks = (rows + peR - 1) ~/ peR;
  final colTiles = (cols + peC - 1) ~/ peC;
  final wordsPerRow = (colTiles + 1) ~/ 2;
  final out = Int8List(rows * cols);
  void put(int gr, int gc, int val) {
    if (gr < rows && gc < cols) out[gr * cols + gc] = val;
  }

  for (var rb = 0; rb < rowBlocks; rb++) {
    for (var ct = 0; ct < colTiles; ct++) {
      final wordByte = (rb * wordsPerRow + (ct >> 1)) * 4;
      final tileByte = wordByte + (ct.isOdd ? 2 : 0);
      final b0 = bytes[tileByte], b1 = bytes[tileByte + 1];
      put(rb * peR, ct * peC, _sext4(b0 & 0xF));
      put(rb * peR, ct * peC + 1, _sext4((b0 >> 4) & 0xF));
      put(rb * peR + 1, ct * peC, _sext4(b1 & 0xF));
      put(rb * peR + 1, ct * peC + 1, _sext4((b1 >> 4) & 0xF));
    }
  }
  return out;
}

/// In-process emulator of LoomFpLinearAccelerator + its weight memory. Speaks
/// the exact CSR protocol and computes the W4A8 linear with the same quant
/// scheme (quant.dart) the hardware uses, so host orchestration code runs
/// unmodified against either this or a real serial [LoomLink].
class EmulatedLoomDevice extends LoomLink {
  final int csrBase;
  static const int versionMagic = 0x4C4F4F4D;

  // CSR offsets (relative to csrBase) - mirror LoomFpLinearAccelerator.
  static const int _version = 0x000;
  static const int _colTiles = 0x004;
  static const int _rowBlocks = 0x008;
  static const int _weightBase = 0x00C;
  static const int _control = 0x010;
  static const int _status = 0x014;
  static const int _actPush = 0x018;
  static const int _scalePush = 0x01C;
  static const int _resultBase = 0x100;

  final int weightBase;
  final Uint8List _mem; // weight store (addr - weightBase)
  int _ct = 0, _rb = 0, _wbase = 0;
  bool _done = false;
  final List<int> _acts = []; // fp16 bits, in push order
  final List<int> _scales = []; // fp16 bits
  List<int> _results = []; // fp16 bits per row

  EmulatedLoomDevice({
    this.csrBase = 0x00010000,
    this.weightBase = 0x20000000,
    int weightSize = 1 << 20,
  }) : _mem = Uint8List(weightSize);

  bool _inCsr(int addr) => addr >= csrBase && addr < csrBase + 0x800;

  @override
  void write(int addr, List<int> bytes) {
    if (_inCsr(addr)) {
      final off = addr - csrBase;
      final val =
          bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
      switch (off) {
        case _colTiles:
          _ct = val & 0xFFFF;
        case _rowBlocks:
          _rb = val & 0xFFFF;
        case _weightBase:
          _wbase = val;
        case _actPush:
          _acts.add(val & 0xFFFF);
        case _scalePush:
          _scales.add(val & 0xFFFF);
        case _control:
          if (val & 0x1 != 0) _run();
      }
      return;
    }
    // Weight memory region.
    final off = addr - weightBase;
    for (var i = 0; i < bytes.length; i++) {
      _mem[off + i] = bytes[i] & 0xFF;
    }
  }

  @override
  List<int> read(int addr, int len) {
    if (_inCsr(addr)) {
      final off = addr - csrBase;
      int word;
      if (off == _version) {
        word = versionMagic;
      } else if (off == _status) {
        word = _done ? 0x2 : 0x1; // bit1 done, bit0 busy
      } else if (off >= _resultBase && off < _resultBase + 0x400) {
        final r = (off - _resultBase) >> 2;
        word = (r < _results.length) ? _results[r] : 0;
      } else {
        word = 0;
      }
      return [
        word & 0xFF,
        (word >> 8) & 0xFF,
        (word >> 16) & 0xFF,
        (word >> 24) & 0xFF,
      ];
    }
    final off = addr - weightBase;
    return [for (var i = 0; i < len; i++) _mem[off + i]];
  }

  void _run() {
    final cols = _ct * 2;
    final rows = _rb * 2;

    final x = Float64List(cols);
    for (var c = 0; c < cols; c++) {
      x[c] = c < _acts.length ? Half.toDouble(_acts[c]) : 0.0;
    }
    final rowScales = Float64List(rows);
    for (var r = 0; r < rows; r++) {
      rowScales[r] = r < _scales.length ? Half.toDouble(_scales[r]) : 0.0;
    }

    // Read int4 weights for this (tiled) row block from memory.
    final colTiles = (cols + 1) ~/ 2;
    final wordsPerRow = (colTiles + 1) ~/ 2;
    final nbytes = (rows ~/ 2) * wordsPerRow * 4;
    final off = _wbase - weightBase;
    final wbytes = Uint8List.sublistView(_mem, off, off + nbytes);
    final w = unpackTileMajorInt4(wbytes, rows, cols);

    // W4A8: per-tensor int8 activations, int4 weights, dequant with the pushed
    // per-row scales (same path as LoomActQuant + LoomStreamMatmul + LoomDequant).
    final qv = quantizePerTensorInt8(x);
    _results = List<int>.filled(rows, 0);
    for (var r = 0; r < rows; r++) {
      var acc = 0;
      for (var c = 0; c < cols; c++) {
        acc += w[r * cols + c] * qv.values[c];
      }
      _results[r] = Half.fromDouble(acc * rowScales[r] * qv.scale);
    }

    _acts.clear();
    _scales.clear();
    _done = true;
  }
}
