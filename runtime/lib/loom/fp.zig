//! Loom fp16 W4A8 linear accelerator: register map and fp16 helpers (pure, no IO).
//!
//! The device computes `y = W @ x` with int4 weights read from memory/flash at
//! WEIGHT_BASE and fp16 activations/scales pushed over the CSR, returning fp16
//! results. Host sequence per matmul: set COL_TILES / ROW_BLOCKS / WEIGHT_BASE,
//! push each activation to ACT_PUSH and each per-row scale to SCALE_PUSH, strobe
//! CONTROL, poll STATUS for done, read RESULT rows.

const std = @import("std");

// CSR offsets relative to the accelerator base (SoC address 0x10000). Distinct
// from the old int8 accel map in protocol.zig; this is the fp W4A8 datapath.
pub const REG_VERSION: u32 = 0x000;
pub const REG_COL_TILES: u32 = 0x004;
pub const REG_ROW_BLOCKS: u32 = 0x008;
pub const REG_WEIGHT_BASE: u32 = 0x00C;
pub const REG_CONTROL: u32 = 0x010;
pub const REG_STATUS: u32 = 0x014;
pub const REG_ACT_PUSH: u32 = 0x018;
/// Write = two packed fp16 activations (low16 then high16); halves act bytes.
pub const REG_ACT_PUSH2: u32 = 0x024;
pub const REG_SCALE_PUSH: u32 = 0x01C;
/// Flash byte addr of this matrix's resident per-row scales (one fp16 per word).
/// Nonzero => the accelerator reads them from flash itself; the host pushes none.
pub const REG_SCALE_BASE: u32 = 0x020;
/// Fusion mode for the next matmul (SwiGLU on-device fold). Persists until
/// rewritten; default 0. See MODE_* below.
pub const REG_MODE: u32 = 0x028;
/// Host-provided activation scale as f32 bits. 0 = the accelerator computes the
/// per-tensor max-abs scale itself. Column-tiled matmuls set this to the scale of
/// the WHOLE activation vector so every block quantizes on the same grid, making
/// a col-tiled result numerically identical to a single-shot matmul.
pub const REG_ACT_SCALE: u32 = 0x02C;
/// Per-group W4A8: extra BYTE offset added to the scale READ address only (not
/// the col-tile accumulate row). A col-block that is its own scale group sets
/// this to group*rows*4 so it dequants on its group's scales while still
/// accumulating into the same global result row. 0 = per-row (shared scale).
pub const REG_SCALE_GROUP_OFF: u32 = 0x030;
pub const REG_RESULT: u32 = 0x100;

/// MODE_NORMAL: result -> resultBuf, host reads it (default, every plain matmul).
pub const MODE_NORMAL: u32 = 0;
/// MODE_CAPTURE_GATE: result -> gateBuf, held on-device (host does NOT read).
pub const MODE_CAPTURE_GATE: u32 = 1;
/// MODE_FUSE_UP: after the matmul, fold silu(gateBuf)*result into resultBuf; the
/// host reads back the SwiGLU product, halving the MLP result-read traffic.
pub const MODE_FUSE_UP: u32 = 2;
/// Column-tiling accumulate modes: the accelerator keeps a running fp32 result
/// accumulator across a matrix's col-blocks and fp16-rounds only after the LAST
/// block, so a col-tiled matmul is numerically equal to a single-shot one (no
/// per-block fp16 rounding). FIRST clears the accumulator, MID adds, LAST adds
/// then writes resultBuf for the host to read.
pub const MODE_COLTILE_FIRST: u32 = 3;
pub const MODE_COLTILE_MID: u32 = 4;
pub const MODE_COLTILE_LAST: u32 = 5;

pub const VERSION_MAGIC: u32 = 0x4C4F4F4D; // 'LOOM'

/// CONTROL bit0 strobes a matmul start.
pub const CONTROL_START: u32 = 0x1;
/// STATUS bit0 = busy, bit1 = done.
pub const STATUS_BUSY: u32 = 0x1;
pub const STATUS_DONE: u32 = 0x2;

/// The RESULT CSR window holds this many output rows; taller matrices are
/// row-tiled by the host into calls of at most this many rows. Must equal the
/// accelerator's maxRows (2*maxRowBlocks). Bigger = fewer invocations = fewer
/// per-call cold-flash + pipeline-fill penalties (the dominant on-device cost),
/// the measured 2.2x lever. 256 (--fp-row-blocks 128) is the flop-accumulator
/// fit ceiling; going higher needs a DP16KD read-during-write bypass first.
pub const MAX_ROWS_PER_CALL: usize = 64; // SPEED SWEEP: match board fp-row-blocks*2

/// PE tile is 2x2, so inner-dim col-tiles and output row-blocks both ceil-divide
/// by 2.
pub const PE: usize = 2;

/// Packs an f32 into its IEEE fp16 bit pattern (native f16, round-to-nearest).
pub fn toBits(x: f32) u16 {
    return @bitCast(@as(f16, @floatCast(x)));
}

/// Decodes an IEEE fp16 bit pattern back to f32.
pub fn fromBits(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// Int4 col-tiles for a matmul with inner dimension `cols`.
pub fn colTiles(cols: usize) usize {
    // PE is a nonzero compile-time constant, so divCeil cannot fault here.
    return std.math.divCeil(usize, cols, PE) catch unreachable;
}

/// Output row-blocks for `rows` output rows.
pub fn rowBlocks(rows: usize) usize {
    return std.math.divCeil(usize, rows, PE) catch unreachable;
}

test "version magic spells LOOM little-endian on the wire" {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, VERSION_MAGIC, .little);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x4D, 0x4F, 0x4F, 0x4C }, &b);
}

test "fp16 round-trips exactly-representable values" {
    try std.testing.expectEqual(@as(f32, 1.5), fromBits(toBits(1.5)));
    try std.testing.expectEqual(@as(f32, -0.25), fromBits(toBits(-0.25)));
    try std.testing.expectEqual(@as(f32, 0.0), fromBits(toBits(0.0)));
}

test "fp16 rounds a non-representable value to the nearest half step" {
    // 1.0 + 2^-11 is below the fp16 step at 1.0 (2^-10), so it rounds to 1.0.
    try std.testing.expectEqual(@as(f32, 1.0), fromBits(toBits(1.0 + 0.0004)));
}

test "colTiles/rowBlocks ceil-divide by the 2x2 PE" {
    try std.testing.expectEqual(@as(usize, 3), colTiles(6));
    try std.testing.expectEqual(@as(usize, 32), colTiles(64));
    try std.testing.expectEqual(@as(usize, 32), colTiles(63)); // odd -> rounds up
    try std.testing.expectEqual(@as(usize, 86), colTiles(172)); // down_proj cols
    try std.testing.expectEqual(@as(usize, 2), rowBlocks(4));
    try std.testing.expectEqual(@as(usize, 86), rowBlocks(172)); // gate/up rows
}
