//! Golden int8 quantized matmul reference, mirroring
//! ip/lib/src/golden/quant.dart (matmulInt + requantize) and the Python
//! reference in tools/loom_uart_matmul.py. Pure, no IO.

const std = @import("std");

/// Symmetric round-half-away-from-zero arithmetic right shift.
pub fn roundHalfAway(prod: i64, shift: u6) i64 {
    if (shift == 0) return prod;
    const bias: i64 = @as(i64, 1) << (shift - 1);
    if (prod >= 0) {
        return (prod + bias) >> shift;
    }
    return -((-prod + bias) >> shift);
}

/// Requantize an int32 accumulator: multiply, round-shift, saturate to the
/// symmetric int8 range [-127, 127].
pub fn requant(acc: i64, mult: u32, shift: u6) i8 {
    const rounded = roundHalfAway(acc * @as(i64, mult), shift);
    const hi: i64 = 127;
    const lo: i64 = -127;
    if (rounded > hi) return 127;
    if (rounded < lo) return -127;
    return @intCast(rounded);
}

/// Reinterpret a raw byte as a signed int8.
pub fn toI8(b: u8) i8 {
    return @bitCast(b);
}

/// Row-major int8 matrix (rows x cols) times an int8 vector (cols), then
/// per-row requantize. `weights` is rows*cols, `acts` is cols, `mults` is rows.
/// Writes `rows` int8 results into `out`.
pub fn matmul(
    weights: []const i8,
    acts: []const i8,
    mults: []const u32,
    shift: u6,
    rows: usize,
    cols: usize,
    out: []i8,
) void {
    for (0..rows) |r| {
        var acc: i64 = 0;
        for (0..cols) |c| {
            acc += @as(i64, weights[r * cols + c]) * @as(i64, acts[c]);
        }
        out[r] = requant(acc, mults[r], shift);
    }
}

test "roundHalfAway rounds halves away from zero" {
    try std.testing.expectEqual(@as(i64, 2), roundHalfAway(3, 1)); // (3+1)>>1
    try std.testing.expectEqual(@as(i64, -2), roundHalfAway(-3, 1));
    try std.testing.expectEqual(@as(i64, 3), roundHalfAway(5, 1)); // (5+1)>>1
    try std.testing.expectEqual(@as(i64, -3), roundHalfAway(-5, 1));
    try std.testing.expectEqual(@as(i64, 5), roundHalfAway(5, 0)); // shift 0 is identity
}

test "requant saturates to symmetric int8" {
    try std.testing.expectEqual(@as(i8, 127), requant(1000, 16, 0));
    try std.testing.expectEqual(@as(i8, -127), requant(-1000, 16, 0));
    try std.testing.expectEqual(@as(i8, 16), requant(16, 16, 4)); // 256, (256+8)>>4
}

test "toI8 reinterprets high bytes as negative" {
    try std.testing.expectEqual(@as(i8, -1), toI8(0xFF));
    try std.testing.expectEqual(@as(i8, -128), toI8(0x80));
    try std.testing.expectEqual(@as(i8, 5), toI8(0x05));
}

test "matmul matches the loom_uart_matmul demo case" {
    // weights 2x4, acts 4, mults [16,8], shift 4.
    const weights = [_]i8{ 1, -2, 3, -4, 5, 6, -7, 8 };
    const acts = [_]i8{ 10, -3, 2, 1 };
    const mults = [_]u32{ 16, 8 };
    var out: [2]i8 = undefined;
    matmul(&weights, &acts, &mults, 4, 2, 4, &out);
    // row0 acc = 10+6+6-4 = 18; *16=288; >>4 = 18.
    // row1 acc = 50-18-14+8 = 26; *8=208; >>4(bias8)= (208+8)>>4=13.
    try std.testing.expectEqual(@as(i8, 18), out[0]);
    try std.testing.expectEqual(@as(i8, 13), out[1]);
}
