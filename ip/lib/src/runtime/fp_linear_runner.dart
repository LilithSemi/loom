import 'dart:typed_data';

import '../golden/quant.dart';
import '../golden/runner.dart' show LinearImpl;
import 'half.dart';
import 'loom_device.dart';

/// CSR offsets of LoomFpLinearAccelerator (relative to its base).
class _Csr {
  static const int colTiles = 0x004;
  static const int rowBlocks = 0x008;
  static const int weightBase = 0x00C;
  static const int control = 0x010;
  static const int status = 0x014;
  static const int actPush = 0x018;
  static const int scalePush = 0x01C;
  static const int resultBase = 0x100;
}

class _Provisioned {
  final int base; // byte address of the tile-major int4 image in device memory
  final List<int> rowScalesFp16; // length == rows, fp16 bits
  final int rows;
  final int cols;
  _Provisioned(this.base, this.rowScalesFp16, this.rows, this.cols);
}

/// Drives a [LoomFpLinearAccelerator] over a [LoomLink], and builds a
/// [LinearImpl] so a full forward pass runs its matmuls on the device (W4A8),
/// with the host doing only the nonlinear glue.
///
/// Weights are provisioned once per matrix (int4, tile-major) and cached.
/// Each linear call pushes the fp16 activation vector + per-row fp16 scales,
/// strobes start, and reads back fp16 results, tiling output rows into
/// [maxRowsPerCall] chunks so matrices larger than the device's row buffer
/// (e.g. lm_head) still run. The same code drives [EmulatedLoomDevice] and a
/// real serial [LoomLink].
class FpLinearRunner {
  final LoomLink link;
  final int csrBase;
  final int maxRowsPerCall; // must be even (row-block = 2 rows)
  int _weightCursor;

  final Map<Float64List, _Provisioned> _cache = {};

  /// Number of device linear invocations performed (for diagnostics).
  int calls = 0;

  FpLinearRunner(
    this.link, {
    this.csrBase = 0x00010000,
    int weightRegionBase = 0x20000000,
    this.maxRowsPerCall = 256,
  }) : _weightCursor = weightRegionBase {
    // The result CSR region is 0x100..0x500 (256 words), so a single device
    // call can return at most 256 rows. Larger matrices are row-tiled.
    if (maxRowsPerCall <= 0 || maxRowsPerCall.isOdd || maxRowsPerCall > 256) {
      throw ArgumentError.value(
        maxRowsPerCall,
        'maxRowsPerCall',
        'must be a positive even number <= 256 (result CSR window)',
      );
    }
  }

  int _readWord(int off) => link.readWord(csrBase + off);
  void _writeWord(int off, int v) => link.writeWord(csrBase + off, v);

  _Provisioned _provision(Float64List w, int rows, int cols) {
    final qm = quantizeRowwiseInt4(w, rows, cols);
    final image = packTileMajorInt4(qm.values, rows, cols);
    final base = _weightCursor;
    link.write(base, image);
    _weightCursor = base + ((image.length + 3) & ~3); // 4-byte align
    final scales = [
      for (var r = 0; r < rows; r++) Half.fromDouble(qm.rowScales[r]),
    ];
    return _Provisioned(base, scales, rows, cols);
  }

  /// Run one linear `y = W @ x` on the device. W is row-major [outDim, inDim].
  Float64List linear(Float64List w, int outDim, int inDim, Float64List x) {
    final prov = _cache.putIfAbsent(w, () => _provision(w, outDim, inDim));
    final colTiles = (inDim + 1) ~/ 2;
    final wordsPerRow = (colTiles + 1) ~/ 2;
    final actsFp16 = [for (var c = 0; c < inDim; c++) Half.fromDouble(x[c])];

    final y = Float64List(outDim);
    final maxRb = maxRowsPerCall ~/ 2;
    final totalRb = (outDim + 1) ~/ 2;

    for (var rbStart = 0; rbStart < totalRb; rbStart += maxRb) {
      final chunkRb = (rbStart + maxRb <= totalRb) ? maxRb : totalRb - rbStart;
      final chunkRows = chunkRb * 2;
      final rowStart = rbStart * 2;

      _writeWord(_Csr.colTiles, colTiles);
      _writeWord(_Csr.rowBlocks, chunkRb);
      _writeWord(_Csr.weightBase, prov.base + rbStart * wordsPerRow * 4);

      for (final a in actsFp16) {
        _writeWord(_Csr.actPush, a);
      }
      for (var r = 0; r < chunkRows; r++) {
        final gr = rowStart + r;
        _writeWord(_Csr.scalePush, gr < outDim ? prov.rowScalesFp16[gr] : 0);
      }

      _writeWord(_Csr.control, 0x1); // start
      var guard = 0;
      while ((_readWord(_Csr.status) & 0x2) == 0 && guard++ < 1000000) {}

      for (var r = 0; r < chunkRows; r++) {
        final gr = rowStart + r;
        if (gr >= outDim) break;
        final bits = _readWord(_Csr.resultBase + r * 4) & 0xFFFF;
        y[gr] = Half.toDouble(bits);
      }
      calls++;
    }
    return y;
  }

  /// A [LinearImpl] bound to this runner, for `GoldenRunner(..., linearImpl:)`.
  LinearImpl get asLinearImpl => linear;
}
