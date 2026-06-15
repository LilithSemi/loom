//! The device W4A8 linear: `y = W @ x`. The accelerator reads the int4 weights
//! from flash itself (host provisions none); the host sets WEIGHT_BASE to the
//! matrix's flash address, pushes fp16 activations + per-row fp16 scales,
//! strobes start, polls, and reads fp16 results. Output rows beyond
//! MAX_ROWS_PER_CALL are row-tiled. Generic over any `device` exposing
//! `regWrite(addr,value)!void` and `regRead(addr)!u32` (the real transport.Device
//! or the sim).

const std = @import("std");
const Io = std.Io;
const fp = @import("fp.zig");
const model = @import("model.zig");
const Device = @import("device.zig").Device;

pub const Error = error{DeviceTimeout};

/// Max packed activation words per matmul (`act_words` buffer): two fp16
/// activations per word, so it bounds an input of up to `2*MAX_WRITES` elements.
/// Sized for the widest matmul in the roadmap models, including SmolVLM's ViT MLP
/// (3072-wide) and its pixel-shuffle connector (12288-wide input). On real
/// hardware inputs beyond the activation SRAM depth are column-tiled instead.
const MAX_WRITES: usize = 8192;

/// Computes `out = mat @ x` on the device. `x` holds `mat.cols` activations,
/// `scales` is the model's fp16 scale table (indexed from `mat.scale_offset`),
/// and `out` receives `mat.rows` results. `csr_base`/`flash_weight_base` come
/// from the model config.
pub fn linear(
    io: Io,
    device: Device,
    csr_base: u32,
    stores: model.Stores,
    poll_timeout: usize,
    mat: model.Matrix,
    x: []const f32,
    out: []f32,
) !void {
    return linearScaled(io, device, csr_base, stores, poll_timeout, mat, x, out, 0, fp.MODE_NORMAL, 0);
}

/// Like [linear], with a caller-provided fp16 activation scale (0 = derive from
/// max-abs) and `mode` (MODE_NORMAL, or a MODE_COLTILE_* accumulate step). A null
/// `out` (COLTILE_FIRST/MID) means the accelerator holds the running result and
/// the host does not read it yet.
fn linearScaled(
    io: Io,
    device: Device,
    csr_base: u32,
    stores: model.Stores,
    poll_timeout: usize,
    mat: model.Matrix,
    x: []const f32,
    out: ?[]f32,
    act_scale: u32,
    mode: u32,
    group_off: u32,
) !void {
    std.debug.assert(x.len == mat.cols); // caller contract
    if (out) |o| std.debug.assert(o.len == mat.rows);

    const total_rb = fp.rowBlocks(mat.rows);
    const max_rb = fp.MAX_ROWS_PER_CALL / fp.PE;

    var rb_start: usize = 0;
    while (rb_start < total_rb) : (rb_start += max_rb) {
        const chunk_rb = @min(max_rb, total_rb - rb_start);
        try runChunk(io, device, csr_base, stores, poll_timeout, mat, x, rb_start, chunk_rb, mode, out, act_scale, group_off);
    }
}

/// Column-tiled matmul for matrices WIDER than the accelerator's column capacity
/// (`block_cols` = maxCols). The weight is stored as consecutive col-blocks, each
/// a tile-major image of `[rows x block_cols]`; run each block on the small
/// accelerator and SUM the partial row-outputs, since
/// `y = W x = sum_b W[:, block_b] . x[block_b]`. Scales: with `mat.groups == 1`
/// one per-row scale is shared across blocks; with `mat.groups > 1` (per-group
/// W4A8) each col-block is its own group, reading its `rows` scales at
/// `scale_flash_offset + b*rows*4`. `partial` is caller scratch (>= rows).
/// This is what lets a small (25f-fitting) accelerator run any-width matmul via
/// more invocations, instead of sizing maxCols to the widest matrix.
pub fn linearColTiled(
    io: Io,
    device: Device,
    csr_base: u32,
    stores: model.Stores,
    poll_timeout: usize,
    mat: model.Matrix,
    block_cols: usize,
    x: []const f32,
    out: []f32,
    partial: []f32,
) !void {
    std.debug.assert(x.len == mat.cols);
    std.debug.assert(out.len == mat.rows and partial.len >= mat.rows);
    // (partial is unused now: the accelerator accumulates on-chip.)
    // Two fixes make a col-tiled matmul equal to a single-shot one: (1) quantize
    // the WHOLE activation on one shared scale (over the fp16 values the device
    // sees), and (2) accumulate the partials on-chip in fp32 (MODE_COLTILE_*),
    // fp16-rounding only after the last block instead of once per block.
    var max_abs: f32 = 0;
    for (x) |v| max_abs = @max(max_abs, @abs(fp.fromBits(fp.toBits(v))));
    const gscale: f32 = if (max_abs == 0) 1.0 else max_abs / 127.0;
    const act_scale: u32 = fp.toBits(gscale); // fp16 bits: REG_ACT_SCALE is 16-bit in the RTL

    const n_blocks = std.math.divCeil(usize, mat.cols, block_cols) catch unreachable;
    const total_rb = fp.rowBlocks(mat.rows);
    const max_rb = fp.MAX_ROWS_PER_CALL / fp.PE;
    // Per-group scales: when `groups > 1` each col-block is its own scale group in
    // the group-major scale table, so block b dequants on its group's scales (byte
    // offset b*rows*4) via its own CSR (group_off); groups == 1 shares one per-row
    // scale across blocks (group_off == 0).
    const grouped = mat.groups > 1;

    // ROW-TILES OUTER, COL-BLOCKS INNER. The accelerator's on-chip accumulator is a
    // maxRows bank REUSED per row-tile, so a row-tile must finish all its col-blocks
    // (FIRST..LAST) before the next one starts. Col-blocks outer would interleave
    // row-tiles and clobber the bank (which holds only maxRows rows), corrupting any
    // matrix with rows > maxRows AND cols > block_cols (the common case).
    var rb_start: usize = 0;
    while (rb_start < total_rb) : (rb_start += max_rb) {
        const chunk_rb = @min(max_rb, total_rb - rb_start);
        var c: usize = 0;
        var b: usize = 0;
        var woff: u32 = mat.weight_offset; // block-0 base (block bases are row-tile independent)
        while (c < mat.cols) : (c += block_cols) {
            const bc = @min(block_cols, mat.cols - c);
            const bct = fp.colTiles(bc);
            const last = b == n_blocks - 1;
            const mode: u32 = if (b == 0) fp.MODE_COLTILE_FIRST else if (last) fp.MODE_COLTILE_LAST else fp.MODE_COLTILE_MID;
            const group_off: u32 = if (grouped) @as(u32, @intCast(b)) * mat.rows * 4 else 0;
            const sub: model.Matrix = .{
                .name = mat.name,
                .weight_offset = woff,
                .scale_offset = mat.scale_offset,
                .scale_flash_offset = mat.scale_flash_offset,
                .rows = mat.rows,
                .cols = @intCast(bc),
                .col_tiles = @intCast(bct),
                .store = mat.store,
            };
            // runChunk adds this row-tile's tile offset (rb_start*words_per_row_b*4)
            // to weight_offset and reads out[row_start..] only on the LAST block.
            try runChunk(io, device, csr_base, stores, poll_timeout, sub, x[c .. c + bc], rb_start, chunk_rb, mode, if (last) out else null, act_scale, group_off);
            woff += @intCast(total_rb * (std.math.divCeil(usize, bct, 2) catch unreachable) * 4);
            b += 1;
        }
    }
}

/// Fused SwiGLU MLP inner: `out = silu(gate @ x) * (up @ x)`, all on-device.
/// Per row-tile it runs the gate matmul in CAPTURE_GATE mode (result held in the
/// accelerator's gate buffer, not read back), then the up matmul in FUSE_UP mode
/// (the accelerator folds `silu(gate)*up` and the host reads only that product).
/// Halves the MLP result-read traffic vs reading gate and up separately, and
/// moves the SiLU + elementwise multiply off the host. `gate` and `up` must have
/// the same shape (intermediate x hidden) and share the model input `x`.
pub fn linearSwiGlu(
    io: Io,
    device: Device,
    csr_base: u32,
    stores: model.Stores,
    poll_timeout: usize,
    gate: model.Matrix,
    up: model.Matrix,
    x: []const f32,
    out: []f32,
) !void {
    std.debug.assert(x.len == gate.cols and x.len == up.cols);
    std.debug.assert(gate.rows == up.rows and out.len == gate.rows);

    const total_rb = fp.rowBlocks(gate.rows);
    const max_rb = fp.MAX_ROWS_PER_CALL / fp.PE;

    var rb_start: usize = 0;
    while (rb_start < total_rb) : (rb_start += max_rb) {
        const chunk_rb = @min(max_rb, total_rb - rb_start);
        // gate -> on-chip gate buffer (captured, not read back).
        try runChunk(io, device, csr_base, stores, poll_timeout, gate, x, rb_start, chunk_rb, fp.MODE_CAPTURE_GATE, null, 0, 0);
        // up -> matmul, then fold silu(gate)*up; read only the fused product.
        try runChunk(io, device, csr_base, stores, poll_timeout, up, x, rb_start, chunk_rb, fp.MODE_FUSE_UP, out, 0, 0);
    }
}

/// Runs one row-tile chunk: coalesced CSR setup (with `mode`), packed activation
/// push, start, poll, and, when `out` is non-null, reads the packed fp16 results
/// into `out[row_start..]`. A null `out` is the CAPTURE_GATE case (nothing to
/// read: the result stays on-device). `mode` is one of fp.MODE_*.
fn runChunk(
    io: Io,
    device: Device,
    csr_base: u32,
    stores: model.Stores,
    poll_timeout: usize,
    mat: model.Matrix,
    x: []const f32,
    rb_start: usize,
    chunk_rb: usize,
    mode: u32,
    out: ?[]f32,
    act_scale: u32,
    group_off: u32,
) !void {
    const words_per_row = std.math.divCeil(usize, mat.col_tiles, 2) catch unreachable;
    const chunk_rows = chunk_rb * fp.PE;
    const row_start = rb_start * fp.PE;
    const tile_off: u32 = @intCast(rb_start * words_per_row * 4);
    // Pick this matrix's store (flash vs on-chip BRAM cache) per its `store` tag.
    const weight_base = stores.weightStore(mat) + mat.weight_offset + tile_off;
    std.debug.assert((x.len + 1) / 2 <= MAX_WRITES and chunk_rows <= fp.MAX_ROWS_PER_CALL);
    // This chunk's per-row scales are resident in the same store; point the
    // accelerator at them (4 bytes per fp16 scale) so it fetches them itself. The
    // host pushes only the activations.
    const scale_base = stores.scaleStore(mat) + mat.scale_flash_offset + @as(u32, @intCast(row_start * 4));

    const tw = Io.Clock.Timestamp.now(io, .awake);
    try device.writeRegs(&[_][2]u32{
        .{ csr_base + fp.REG_COL_TILES, mat.col_tiles },
        .{ csr_base + fp.REG_ROW_BLOCKS, @intCast(chunk_rb) },
        .{ csr_base + fp.REG_WEIGHT_BASE, weight_base },
        .{ csr_base + fp.REG_SCALE_BASE, scale_base },
        .{ csr_base + fp.REG_ACT_SCALE, act_scale },
        .{ csr_base + fp.REG_SCALE_GROUP_OFF, group_off },
        .{ csr_base + fp.REG_MODE, mode },
    });

    // Pack two fp16 activations per 32-bit word (low16 then high16) and stream to
    // ACT_PUSH2, halving the activation bytes on the wire.
    var act_words: [MAX_WRITES]u32 = undefined;
    const n_words = (x.len + 1) / 2;
    for (0..n_words) |i| {
        const lo: u32 = fp.toBits(x[2 * i]);
        const hi: u32 = if (2 * i + 1 < x.len) fp.toBits(x[2 * i + 1]) else 0;
        act_words[i] = lo | (hi << 16);
    }
    try device.writeStream(csr_base + fp.REG_ACT_PUSH2, act_words[0..n_words]);

    try device.regWrite(csr_base + fp.REG_CONTROL, fp.CONTROL_START);
    const tp = Io.Clock.Timestamp.now(io, .awake);
    t_write_ns += @intCast(tw.durationTo(tp).raw.nanoseconds);

    try pollDone(io, device, csr_base, poll_timeout);
    const tr = Io.Clock.Timestamp.now(io, .awake);
    t_poll_ns += @intCast(tp.durationTo(tr).raw.nanoseconds);

    // CAPTURE_GATE holds the result on-device; nothing to read.
    if (out) |o| {
        // Results come back PACKED, two fp16 per 32-bit word.
        var rwords: [fp.MAX_ROWS_PER_CALL / 2]u32 = undefined;
        const n_rwords = chunk_rows / 2;
        result_reads += 1;
        try device.readRegs(csr_base + fp.REG_RESULT, rwords[0..n_rwords]);
        for (0..n_rwords) |w| {
            const gr0 = row_start + 2 * w;
            const gr1 = row_start + 2 * w + 1;
            if (gr0 < mat.rows) o[gr0] = fp.fromBits(@truncate(rwords[w]));
            if (gr1 < mat.rows) o[gr1] = fp.fromBits(@truncate(rwords[w] >> 16));
        }
    }
    t_read_ns += @intCast(tr.durationTo(Io.Clock.Timestamp.now(io, .awake)).raw.nanoseconds);
}

/// Instrumentation: round-trip counters + wall-time nanoseconds (reset by the
/// caller). Poll reads spin on STATUS; result reads are one burst per chunk.
pub var poll_reads: usize = 0;
pub var result_reads: usize = 0;
pub var t_write_ns: u64 = 0;
pub var t_poll_ns: u64 = 0;
pub var t_read_ns: u64 = 0;

/// Default cap on STATUS reads before pollDone gives up on a matmul.
pub const DEFAULT_TIMEOUT: usize = 1_000_000;

/// Optional sleep between STATUS reads (microseconds). 0 = tight spin. A nonzero
/// value backs off so polling stops fighting the accelerator's flash mem-master
/// for the shared Wishbone bus; also lets us measure true device compute time
/// (poll count at a known delay reveals it).
pub var poll_delay_us: u64 = 0;

fn pollDone(io: Io, device: Device, csr_base: u32, timeout: usize) Error!void {
    var i: usize = 0;
    while (i < timeout) : (i += 1) {
        poll_reads += 1;
        const st = device.regRead(csr_base + fp.REG_STATUS) catch return Error.DeviceTimeout;
        if (st & fp.STATUS_DONE != 0) return;
        if (poll_delay_us != 0)
            io.sleep(Io.Duration.fromMicroseconds(@intCast(poll_delay_us)), .awake) catch {};
    }
    return Error.DeviceTimeout;
}

test "linear drives the sim to the W4A8 result, single tile" {
    const sim = @import("sim.zig");
    const gpa = std.testing.allocator;

    const rows = 4;
    const cols = 6;
    // A known int4 weight matrix and its tile-major flash image.
    var w: [rows * cols]i8 = undefined;
    for (0..rows) |r| for (0..cols) |c| {
        w[r * cols + c] = @intCast(@as(i32, @intCast((r * 7 + c * 3) % 11)) - 5); // in [-5,5]
    };
    const nbytes = fp.rowBlocks(rows) * (std.math.divCeil(usize, fp.colTiles(cols), 2) catch unreachable) * 4;
    var image: [64]u8 = undefined;
    sim.packTileMajor(&w, rows, cols, image[0..nbytes]);

    const csr_base: u32 = 0x10000;
    const flash_base: u32 = 0x20200000;
    const scale_flash_base: u32 = 0x20280000;
    var s = sim.Sim.init(gpa, csr_base, flash_base, image[0..nbytes]);
    defer s.deinit();

    // Activations and per-row fp16 scales, the latter resident in flash (one
    // fp16 per 32-bit word, low16).
    var x: [cols]f32 = .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.5 };
    var scales: [rows]u16 = undefined;
    var scales_flash: [rows * 4]u8 = [_]u8{0} ** (rows * 4);
    for (0..rows) |r| {
        scales[r] = fp.toBits(0.1 + @as(f32, @floatFromInt(r)) * 0.05);
        std.mem.writeInt(u16, scales_flash[r * 4 ..][0..2], scales[r], .little);
    }
    s.scale_flash_base = scale_flash_base;
    s.scales_flash = &scales_flash;

    const mat: model.Matrix = .{
        .name = "test",
        .weight_offset = 0,
        .scale_offset = 0,
        .scale_flash_offset = 0,
        .rows = rows,
        .cols = cols,
        .col_tiles = @intCast(fp.colTiles(cols)),
    };

    var out: [rows]f32 = undefined;
    try linear(std.testing.io, s.device(), csr_base, .{ .flash_weight_base = flash_base, .scale_flash_base = scale_flash_base }, DEFAULT_TIMEOUT, mat, &x, &out);

    // Independent W4A8 oracle: int8-quant x, int MAC with the int4 weights,
    // dequant with the fp16 scales - exactly what the accelerator does.
    var qx: [cols]i8 = undefined;
    const act_scale = sim.quantPerTensorI8(&x, &qx);
    for (0..rows) |r| {
        var acc: i64 = 0;
        for (0..cols) |c| acc += @as(i64, w[r * cols + c]) * @as(i64, qx[c]);
        const y = @as(f32, @floatFromInt(acc)) * fp.fromBits(scales[r]) * act_scale;
        try std.testing.expectEqual(fp.fromBits(fp.toBits(y)), out[r]);
    }
}

test "linearColTiled sums col-block partials to the full W4A8 matmul" {
    const sim = @import("sim.zig");
    const gpa = std.testing.allocator;

    const rows = 4;
    const cols = 8;
    const block_cols = 4; // accelerator caps at 4 cols, so an 8-wide matmul = 2 blocks
    var w: [rows * cols]i8 = undefined;
    for (0..rows) |r| for (0..cols) |c| {
        w[r * cols + c] = @intCast(@as(i32, @intCast((r * 5 + c * 3) % 11)) - 5);
    };

    // Col-tiled weight image: consecutive tile-major images, one per col-block.
    var image: [256]u8 = [_]u8{0} ** 256;
    var off: usize = 0;
    var c0: usize = 0;
    while (c0 < cols) : (c0 += block_cols) {
        const bc: usize = @min(@as(usize, block_cols), cols - c0);
        var wb: [rows * block_cols]i8 = undefined;
        var n: usize = 0;
        for (0..rows) |r| {
            for (0..bc) |k| {
                wb[n] = w[r * @as(usize, cols) + c0 + k];
                n += 1;
            }
        }
        const bb: usize = fp.rowBlocks(rows) * (std.math.divCeil(usize, fp.colTiles(bc), 2) catch unreachable) * 4;
        sim.packTileMajor(wb[0..n], rows, bc, image[off .. off + bb]);
        off += bb;
    }

    const csr_base: u32 = 0x10000;
    const flash_base: u32 = 0x20200000;
    const scale_flash_base: u32 = 0x20280000;
    var s = sim.Sim.init(gpa, csr_base, flash_base, image[0..off]);
    defer s.deinit();

    var x: [cols]f32 = .{ 0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.4, 0.9 };
    var scales: [rows]u16 = undefined;
    var scales_flash: [rows * 4]u8 = [_]u8{0} ** (rows * 4);
    for (0..rows) |r| {
        scales[r] = fp.toBits(0.1 + @as(f32, @floatFromInt(r)) * 0.05);
        std.mem.writeInt(u16, scales_flash[r * 4 ..][0..2], scales[r], .little);
    }
    s.scale_flash_base = scale_flash_base;
    s.scales_flash = &scales_flash;

    const mat: model.Matrix = .{
        .name = "test",
        .weight_offset = 0,
        .scale_offset = 0,
        .scale_flash_offset = 0,
        .rows = rows,
        .cols = cols,
        .col_tiles = @intCast(fp.colTiles(cols)),
    };

    var out: [rows]f32 = undefined;
    var partial: [rows]f32 = undefined;
    try linearColTiled(std.testing.io, s.device(), csr_base, .{ .flash_weight_base = flash_base, .scale_flash_base = scale_flash_base }, DEFAULT_TIMEOUT, mat, block_cols, &x, &out, &partial);

    // Oracle: col-tiling shares ONE activation scale (over the whole fp16
    // vector), int MAC per block, dequant to fp16, sum the fp16 partials.
    var xf: [cols]f32 = undefined;
    for (0..cols) |c| xf[c] = fp.fromBits(fp.toBits(x[c]));
    var gmax: f32 = 0;
    for (0..cols) |c| gmax = @max(gmax, @abs(xf[c]));
    const gscale: f32 = if (gmax == 0) 1.0 else gmax / 127.0;
    var qxf: [cols]i8 = undefined;
    _ = sim.quantWithScale(&xf, gscale, &qxf);
    // On-chip accumulate -> col-tiling equals a single-shot matmul (one fp16 round).
    var expected: [rows]f32 = [_]f32{0} ** rows;
    for (0..rows) |r| {
        var acc: i64 = 0;
        for (0..cols) |c| acc += @as(i64, w[r * cols + c]) * @as(i64, qxf[c]);
        expected[r] = fp.fromBits(fp.toBits(@as(f32, @floatFromInt(acc)) * fp.fromBits(scales[r]) * gscale));
    }
    for (0..rows) |r| {
        try std.testing.expectEqual(expected[r], out[r]);
    }
}

test "col-tiled (shared scale) vs single-shot: magnitude of fp16-partial drift" {
    const sim = @import("sim.zig");
    const gpa = std.testing.allocator;
    const rows: usize = 64;
    const cols: usize = 576;
    const block_cols: usize = 64; // 9 blocks (the SmolVLM case)
    const w = try gpa.alloc(i8, rows * cols);
    defer gpa.free(w);
    for (0..rows) |r| for (0..cols) |c| {
        w[r * cols + c] = @intCast(@as(i32, @intCast((r * 7 + c * 13) % 11)) - 5);
    };
    // Single-block image (for plain linear).
    const wpr_full = std.math.divCeil(usize, fp.colTiles(cols), 2) catch unreachable;
    const single_img = try gpa.alloc(u8, fp.rowBlocks(rows) * wpr_full * 4);
    defer gpa.free(single_img);
    sim.packTileMajor(w, rows, cols, single_img);
    // Col-block image (for linearColTiled).
    const col_img = try gpa.alloc(u8, rows * cols);
    defer gpa.free(col_img);
    var off: usize = 0;
    var c0: usize = 0;
    while (c0 < cols) : (c0 += block_cols) {
        const bc = @min(block_cols, cols - c0);
        const wb = try gpa.alloc(i8, rows * bc);
        defer gpa.free(wb);
        for (0..rows) |r| for (0..bc) |k| {
            wb[r * bc + k] = w[r * cols + c0 + k];
        };
        const bb = fp.rowBlocks(rows) * (std.math.divCeil(usize, fp.colTiles(bc), 2) catch unreachable) * 4;
        sim.packTileMajor(wb, rows, bc, col_img[off .. off + bb]);
        off += bb;
    }
    const flash_base: u32 = 0x20200000;
    const scale_flash_base: u32 = 0x20280000;
    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (0..cols) |c| x[c] = @sin(@as(f32, @floatFromInt(c)) * 0.11);
    const scales_flash = try gpa.alloc(u8, rows * 4);
    defer gpa.free(scales_flash);
    @memset(scales_flash, 0);
    for (0..rows) |r| std.mem.writeInt(u16, scales_flash[r * 4 ..][0..2], fp.toBits(0.1), .little);

    var s1 = sim.Sim.init(gpa, 0x10000, flash_base, single_img);
    defer s1.deinit();
    s1.scale_flash_base = scale_flash_base;
    s1.scales_flash = scales_flash;
    const mat1: model.Matrix = .{ .name = "t", .weight_offset = 0, .scale_offset = 0, .scale_flash_offset = 0, .rows = @intCast(rows), .cols = @intCast(cols), .col_tiles = @intCast(fp.colTiles(cols)) };
    const out1 = try gpa.alloc(f32, rows);
    defer gpa.free(out1);
    try linear(std.testing.io, s1.device(), 0x10000, .{ .flash_weight_base = flash_base, .scale_flash_base = scale_flash_base }, DEFAULT_TIMEOUT, mat1, x, out1);

    var s2 = sim.Sim.init(gpa, 0x10000, flash_base, col_img[0..off]);
    defer s2.deinit();
    s2.scale_flash_base = scale_flash_base;
    s2.scales_flash = scales_flash;
    const mat2 = mat1;
    const out2 = try gpa.alloc(f32, rows);
    defer gpa.free(out2);
    const partial = try gpa.alloc(f32, rows);
    defer gpa.free(partial);
    try linearColTiled(std.testing.io, s2.device(), 0x10000, .{ .flash_weight_base = flash_base, .scale_flash_base = scale_flash_base }, DEFAULT_TIMEOUT, mat2, block_cols, x, out2, partial);

    // On-chip fp32 accumulation makes col-tiling bit-identical to single-shot.
    for (0..rows) |r| {
        try std.testing.expectEqual(out1[r], out2[r]);
    }
}

test "plain linear row-tiling (rows>172) with a NARROW matrix (cols=64)" {
    const sim = @import("sim.zig");
    const gpa = std.testing.allocator;
    const rows: usize = 200;
    const cols: usize = 64;
    const w = try gpa.alloc(i8, rows * cols);
    defer gpa.free(w);
    for (0..rows) |r| for (0..cols) |c| {
        w[r * cols + c] = @intCast(@as(i32, @intCast((r * 7 + c * 3) % 11)) - 5);
    };
    const wpr = std.math.divCeil(usize, fp.colTiles(cols), 2) catch unreachable;
    const nbytes = fp.rowBlocks(rows) * wpr * 4;
    const image = try gpa.alloc(u8, nbytes);
    defer gpa.free(image);
    sim.packTileMajor(w, rows, cols, image);

    const csr_base: u32 = 0x10000;
    const flash_base: u32 = 0x20200000;
    const scale_flash_base: u32 = 0x20280000;
    var s = sim.Sim.init(gpa, csr_base, flash_base, image);
    defer s.deinit();
    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (0..cols) |c| x[c] = @sin(@as(f32, @floatFromInt(c)) * 0.3);
    const scales = try gpa.alloc(u16, rows);
    defer gpa.free(scales);
    const scales_flash = try gpa.alloc(u8, rows * 4);
    defer gpa.free(scales_flash);
    @memset(scales_flash, 0);
    for (0..rows) |r| {
        scales[r] = fp.toBits(0.1 + @as(f32, @floatFromInt(r)) * 0.001);
        std.mem.writeInt(u16, scales_flash[r * 4 ..][0..2], scales[r], .little);
    }
    s.scale_flash_base = scale_flash_base;
    s.scales_flash = scales_flash;
    const mat: model.Matrix = .{ .name = "t", .weight_offset = 0, .scale_offset = 0, .scale_flash_offset = 0, .rows = @intCast(rows), .cols = @intCast(cols), .col_tiles = @intCast(fp.colTiles(cols)) };
    const out = try gpa.alloc(f32, rows);
    defer gpa.free(out);
    try linear(std.testing.io, s.device(), csr_base, .{ .flash_weight_base = flash_base, .scale_flash_base = scale_flash_base }, DEFAULT_TIMEOUT, mat, x, out);
    // Device receives fp16 activations; the oracle must quantize the SAME
    // fp16-rounded values (not the full-f32 x).
    const xf = try gpa.alloc(f32, cols);
    defer gpa.free(xf);
    for (0..cols) |c| xf[c] = fp.fromBits(fp.toBits(x[c]));
    const qx = try gpa.alloc(i8, cols);
    defer gpa.free(qx);
    const act_scale = sim.quantPerTensorI8(xf, qx);
    for (0..rows) |r| {
        var acc: i64 = 0;
        for (0..cols) |c| acc += @as(i64, w[r * cols + c]) * @as(i64, qx[c]);
        const y = @as(f32, @floatFromInt(acc)) * fp.fromBits(scales[r]) * act_scale;
        try std.testing.expectEqual(fp.fromBits(fp.toBits(y)), out[r]);
    }
}

// (debug) placeholder anchor
test "linearColTiled with row-tiling (rows > MAX_ROWS_PER_CALL) inside col-blocks" {
    // The SmolVLM case: many rows (row-tiled) AND wide (col-tiled). Exercises the
    // interaction the small test above misses.
    const sim = @import("sim.zig");
    const gpa = std.testing.allocator;

    const rows: usize = 200; // > MAX_ROWS_PER_CALL (172) -> row-tiled
    const cols: usize = 192;
    const block_cols: usize = 64; // 3 col-blocks
    const w = try gpa.alloc(i8, rows * cols);
    defer gpa.free(w);
    for (0..rows) |r| for (0..cols) |c| {
        w[r * cols + c] = @intCast(@as(i32, @intCast((r * 7 + c * 3) % 11)) - 5);
    };

    // Col-block-contiguous tile-major image.
    const image = try gpa.alloc(u8, rows * cols); // generous
    defer gpa.free(image);
    var off: usize = 0;
    var c0: usize = 0;
    while (c0 < cols) : (c0 += block_cols) {
        const bc = @min(block_cols, cols - c0);
        const wb = try gpa.alloc(i8, rows * bc);
        defer gpa.free(wb);
        for (0..rows) |r| for (0..bc) |k| {
            wb[r * bc + k] = w[r * cols + c0 + k];
        };
        const bb = fp.rowBlocks(rows) * (std.math.divCeil(usize, fp.colTiles(bc), 2) catch unreachable) * 4;
        sim.packTileMajor(wb, rows, bc, image[off .. off + bb]);
        off += bb;
    }

    const csr_base: u32 = 0x10000;
    const flash_base: u32 = 0x20200000;
    const scale_flash_base: u32 = 0x20280000;
    var s = sim.Sim.init(gpa, csr_base, flash_base, image[0..off]);
    defer s.deinit();

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (0..cols) |c| x[c] = @sin(@as(f32, @floatFromInt(c)) * 0.3);
    const scales = try gpa.alloc(u16, rows);
    defer gpa.free(scales);
    const scales_flash = try gpa.alloc(u8, rows * 4);
    defer gpa.free(scales_flash);
    @memset(scales_flash, 0);
    for (0..rows) |r| {
        scales[r] = fp.toBits(0.1 + @as(f32, @floatFromInt(r)) * 0.001);
        std.mem.writeInt(u16, scales_flash[r * 4 ..][0..2], scales[r], .little);
    }
    s.scale_flash_base = scale_flash_base;
    s.scales_flash = scales_flash;

    const mat: model.Matrix = .{
        .name = "test",
        .weight_offset = 0,
        .scale_offset = 0,
        .scale_flash_offset = 0,
        .rows = @intCast(rows),
        .cols = @intCast(cols),
        .col_tiles = @intCast(fp.colTiles(cols)),
    };

    const out = try gpa.alloc(f32, rows);
    defer gpa.free(out);
    const partial = try gpa.alloc(f32, rows);
    defer gpa.free(partial);
    try linearColTiled(std.testing.io, s.device(), csr_base, .{ .flash_weight_base = flash_base, .scale_flash_base = scale_flash_base }, DEFAULT_TIMEOUT, mat, block_cols, x, out, partial);

    const expected = try gpa.alloc(f32, rows);
    defer gpa.free(expected);
    @memset(expected, 0);
    // Oracle: shared activation scale over the WHOLE fp16 vector (what
    // linearColTiled now provides), so col-tiling == single-shot.
    const xf = try gpa.alloc(f32, cols);
    defer gpa.free(xf);
    for (0..cols) |c| xf[c] = fp.fromBits(fp.toBits(x[c]));
    var gmax: f32 = 0;
    for (0..cols) |c| gmax = @max(gmax, @abs(xf[c]));
    const gscale: f32 = if (gmax == 0) 1.0 else gmax / 127.0;
    const qxf = try gpa.alloc(i8, cols);
    defer gpa.free(qxf);
    _ = sim.quantWithScale(xf, gscale, qxf);
    // Mirror the accelerator's on-chip fp32 accumulate: sum each block's f32
    // partial, fp16-round only at the end.
    for (0..rows) |r| {
        var facc: f32 = 0;
        var cc: usize = 0;
        while (cc < cols) : (cc += block_cols) {
            const bc = @min(block_cols, cols - cc);
            var acc: i64 = 0;
            for (0..bc) |k| acc += @as(i64, w[r * cols + cc + k]) * @as(i64, qxf[cc + k]);
            facc += @as(f32, @floatFromInt(acc)) * fp.fromBits(scales[r]) * gscale;
        }
        expected[r] = fp.fromBits(fp.toBits(facc));
    }
    for (0..rows) |r| {
        try std.testing.expectEqual(expected[r], out[r]);
    }
}

test "linearColTiled per-group scales (groups>1): each col-block dequants on its own scale" {
    // Per-group W4A8: with group == col-block, block b reads group b's per-row
    // scales (group-major table) but still accumulates into the same result row.
    const sim = @import("sim.zig");
    const gpa = std.testing.allocator;

    const rows: usize = 96;
    const cols: usize = 192;
    const block_cols: usize = 64; // 3 col-blocks == 3 groups
    const n_groups = cols / block_cols;
    const w = try gpa.alloc(i8, rows * cols);
    defer gpa.free(w);
    for (0..rows) |r| for (0..cols) |c| {
        w[r * cols + c] = @intCast(@as(i32, @intCast((r * 5 + c * 3) % 15)) - 7);
    };

    // Col-block-contiguous tile-major image.
    const image = try gpa.alloc(u8, rows * cols);
    defer gpa.free(image);
    var off: usize = 0;
    var c0: usize = 0;
    while (c0 < cols) : (c0 += block_cols) {
        const bc = @min(block_cols, cols - c0);
        const wb = try gpa.alloc(i8, rows * bc);
        defer gpa.free(wb);
        for (0..rows) |r| for (0..bc) |k| {
            wb[r * bc + k] = w[r * cols + c0 + k];
        };
        const bb = fp.rowBlocks(rows) * (std.math.divCeil(usize, fp.colTiles(bc), 2) catch unreachable) * 4;
        sim.packTileMajor(wb, rows, bc, image[off .. off + bb]);
        off += bb;
    }

    const csr_base: u32 = 0x10000;
    const flash_base: u32 = 0x20200000;
    const scale_flash_base: u32 = 0x20280000;
    var s = sim.Sim.init(gpa, csr_base, flash_base, image[0..off]);
    defer s.deinit();

    const x = try gpa.alloc(f32, cols);
    defer gpa.free(x);
    for (0..cols) |c| x[c] = @sin(@as(f32, @floatFromInt(c)) * 0.23);

    // Group-major scale table: [group0's rows][group1's rows]..., each group with
    // a DISTINCT per-row scale so a wrong group->block mapping shows up.
    const gscales = try gpa.alloc(f32, n_groups * rows);
    defer gpa.free(gscales);
    const scales_flash = try gpa.alloc(u8, n_groups * rows * 4);
    defer gpa.free(scales_flash);
    @memset(scales_flash, 0);
    for (0..n_groups) |g| for (0..rows) |r| {
        const v = 0.05 + @as(f32, @floatFromInt(g)) * 0.03 + @as(f32, @floatFromInt(r)) * 0.0007;
        gscales[g * rows + r] = fp.fromBits(fp.toBits(v));
        std.mem.writeInt(u16, scales_flash[(g * rows + r) * 4 ..][0..2], fp.toBits(v), .little);
    };
    s.scale_flash_base = scale_flash_base;
    s.scales_flash = scales_flash;

    const mat: model.Matrix = .{
        .name = "test",
        .weight_offset = 0,
        .scale_offset = 0,
        .scale_flash_offset = 0,
        .rows = @intCast(rows),
        .cols = @intCast(cols),
        .col_tiles = @intCast(fp.colTiles(cols)),
        .groups = @intCast(n_groups),
    };

    const out = try gpa.alloc(f32, rows);
    defer gpa.free(out);
    const partial = try gpa.alloc(f32, rows);
    defer gpa.free(partial);
    try linearColTiled(std.testing.io, s.device(), csr_base, .{ .flash_weight_base = flash_base, .scale_flash_base = scale_flash_base }, DEFAULT_TIMEOUT, mat, block_cols, x, out, partial);

    // Oracle: shared activation scale over the whole fp16 vector, on-chip fp32
    // accumulate, and group b's scale for block b (dequantGroupwise).
    const xf = try gpa.alloc(f32, cols);
    defer gpa.free(xf);
    for (0..cols) |c| xf[c] = fp.fromBits(fp.toBits(x[c]));
    var gmax: f32 = 0;
    for (0..cols) |c| gmax = @max(gmax, @abs(xf[c]));
    const gscale: f32 = if (gmax == 0) 1.0 else gmax / 127.0;
    const qxf = try gpa.alloc(i8, cols);
    defer gpa.free(qxf);
    _ = sim.quantWithScale(xf, gscale, qxf);
    for (0..rows) |r| {
        var facc: f32 = 0;
        var g: usize = 0;
        var cc: usize = 0;
        while (cc < cols) : (cc += block_cols) {
            const bc = @min(block_cols, cols - cc);
            var acc: i64 = 0;
            for (0..bc) |k| acc += @as(i64, w[r * cols + cc + k]) * @as(i64, qxf[cc + k]);
            facc += @as(f32, @floatFromInt(acc)) * gscales[g * rows + r] * gscale;
            g += 1;
        }
        try std.testing.expectEqual(fp.fromBits(fp.toBits(facc)), out[r]);
    }
}
