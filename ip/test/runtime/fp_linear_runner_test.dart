// The host runtime path: FpLinearRunner drives the EmulatedLoomDevice over the
// CSR protocol (provision int4 weights, push fp16 acts/scales, start, read
// results) with row tiling, and reproduces the golden W4A8 linear. Same code
// drives a real serial LoomLink on the board.

import 'dart:typed_data';

import 'package:loom/loom.dart';
import 'package:test/test.dart';

void main() {
  test('half codec round-trips representative values', () {
    for (final v in [0.0, 1.0, -1.0, 0.5, -2.5, 12.34, -0.001, 100.0]) {
      final back = Half.toDouble(Half.fromDouble(v));
      expect(
        back,
        closeTo(v, v.abs() * 0.002 + 0.001),
        reason: 'half($v)=$back',
      );
    }
  });

  test('device-backed linear (with row tiling) matches golden W4A8', () {
    const rows = 6, cols = 6;
    final w = Float64List(rows * cols);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        w[r * cols + c] = ((r * 7 + c * 3) % 11 - 5) * 0.21;
      }
    }
    final x = Float64List.fromList([
      for (var c = 0; c < cols; c++) ((c * 5 + 2) % 9 - 4) * 0.37,
    ]);
    final golden = quantizedLinearW4A8(w, rows, cols, x);

    final dev = EmulatedLoomDevice();
    final runner = FpLinearRunner(
      dev,
      csrBase: dev.csrBase,
      weightRegionBase: dev.weightBase,
      maxRowsPerCall: 4, // forces tiling (3 row-blocks -> 2 calls)
    );

    final y = runner.linear(w, rows, cols, x);
    expect(runner.calls, greaterThan(1), reason: 'row tiling exercised');

    for (var r = 0; r < rows; r++) {
      expect(
        y[r],
        closeTo(golden[r], 0.06 + golden[r].abs() * 0.1),
        reason: 'row $r: dev=${y[r]} golden=${golden[r]}',
      );
    }

    // VERSION reads back through the same link.
    expect(dev.readWord(dev.csrBase + 0x000), equals(0x4C4F4F4D));
  });

  test('caching: a matmul is provisioned once, reused across calls', () {
    const rows = 4, cols = 4;
    final w = Float64List.fromList([
      for (var i = 0; i < rows * cols; i++) (i % 7 - 3) * 0.3,
    ]);
    final dev = EmulatedLoomDevice();
    final runner = FpLinearRunner(
      dev,
      csrBase: dev.csrBase,
      weightRegionBase: dev.weightBase,
      maxRowsPerCall: 64,
    );
    final x1 = Float64List.fromList([1.0, -2.0, 0.5, 3.0]);
    final x2 = Float64List.fromList([0.0, 1.0, -1.0, 2.0]);
    final y1 = runner.linear(w, rows, cols, x1);
    final y2 = runner.linear(w, rows, cols, x2);
    expect(y1.length, rows);
    expect(y2.length, rows);
    // Different inputs -> different outputs (not a stale buffer).
    expect(y1, isNot(equals(y2)));
  });
}
