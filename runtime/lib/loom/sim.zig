//! In-process fp W4A8 accelerator emulator, the `.sim` transport backend. It
//! parses the exact command frames the host writes, maintains the accelerator's
//! CSR + fp16 FIFO state, and on a CONTROL start computes the W4A8 matmul
//! bit-for-bit against the golden quant path (int4 weights read from a preloaded
//! flash image, per-tensor int8 activations, per-row fp16 dequant). This is the
//! frontend-first test double for `linear.zig`/`forward.zig` - no board needed.

const std = @import("std");
const fp = @import("fp.zig");
const proto = @import("protocol.zig");
const Iface = @import("device.zig").Device;

/// Sign-extends a 4-bit two's-complement nibble to i8.
fn sext4(n: u8) i8 {
    return if (n & 0x8 != 0) @as(i8, @intCast(n & 0xF)) - 16 else @intCast(n & 0xF);
}

/// Unpacks a `rows`x`cols` int4 matrix from the device's tile-major byte image
/// (2x2 PE tiles, two tiles per 32-bit word), the inverse of the dart
/// `packTileMajorInt4`. `out` must be `rows*cols` long.
pub fn unpackTileMajor(bytes: []const u8, rows: usize, cols: usize, out: []i8) void {
    const row_blocks = fp.rowBlocks(rows);
    const col_tiles = fp.colTiles(cols);
    const words_per_row = std.math.divCeil(usize, col_tiles, 2) catch unreachable;
    @memset(out, 0);
    for (0..row_blocks) |rb| {
        for (0..col_tiles) |ct| {
            const word_byte = (rb * words_per_row + (ct >> 1)) * 4;
            const tile_byte = word_byte + (if (ct & 1 == 1) @as(usize, 2) else 0);
            const b0 = bytes[tile_byte];
            const b1 = bytes[tile_byte + 1];
            put(out, rows, cols, rb * 2, ct * 2, sext4(b0 & 0xF));
            put(out, rows, cols, rb * 2, ct * 2 + 1, sext4(b0 >> 4));
            put(out, rows, cols, rb * 2 + 1, ct * 2, sext4(b1 & 0xF));
            put(out, rows, cols, rb * 2 + 1, ct * 2 + 1, sext4(b1 >> 4));
        }
    }
}

fn put(out: []i8, rows: usize, cols: usize, r: usize, c: usize, v: i8) void {
    if (r < rows and c < cols) out[r * cols + c] = v;
}

/// Packs a `rows`x`cols` int4 matrix into the tile-major byte image the
/// accelerator reads (inverse of `unpackTileMajor`). `bytes` must be
/// `rowBlocks(rows)*wordsPerRow*4` long. Test/generation helper.
pub fn packTileMajor(vals: []const i8, rows: usize, cols: usize, bytes: []u8) void {
    const row_blocks = fp.rowBlocks(rows);
    const col_tiles = fp.colTiles(cols);
    const words_per_row = std.math.divCeil(usize, col_tiles, 2) catch unreachable;
    @memset(bytes, 0);
    for (0..row_blocks) |rb| {
        for (0..col_tiles) |ct| {
            const word_byte = (rb * words_per_row + (ct >> 1)) * 4;
            const tile_byte = word_byte + (if (ct & 1 == 1) @as(usize, 2) else 0);
            bytes[tile_byte] = nib(vals, rows, cols, rb * 2, ct * 2) |
                (nib(vals, rows, cols, rb * 2, ct * 2 + 1) << 4);
            bytes[tile_byte + 1] = nib(vals, rows, cols, rb * 2 + 1, ct * 2) |
                (nib(vals, rows, cols, rb * 2 + 1, ct * 2 + 1) << 4);
        }
    }
}

fn nib(vals: []const i8, rows: usize, cols: usize, r: usize, c: usize) u8 {
    if (r >= rows or c >= cols) return 0;
    return @as(u8, @bitCast(vals[r * cols + c])) & 0xF;
}

/// Symmetric per-tensor int8 quantization (round-half-away, saturate to
/// [-127,127]). Returns the scale; writes int8 values into `q`.
pub fn quantPerTensorI8(x: []const f32, q: []i8) f32 {
    var max_abs: f32 = 0;
    for (x) |v| max_abs = @max(max_abs, @abs(v));
    const scale = if (max_abs == 0) 1.0 else max_abs / 127.0;
    for (x, 0..) |v, i| {
        const r = @round(v / scale);
        q[i] = @intFromFloat(std.math.clamp(r, -127.0, 127.0));
    }
    return scale;
}

/// Quantizes `x` to int8 with a caller-provided `scale` (rather than deriving it
/// from the block's max-abs). Used for column-tiled matmuls, where every block
/// must share the whole vector's scale. Returns `scale` unchanged.
pub fn quantWithScale(x: []const f32, scale: f32, q: []i8) f32 {
    for (x, 0..) |v, i| {
        const r = @round(v / scale);
        q[i] = @intFromFloat(std.math.clamp(r, -127.0, 127.0));
    }
    return scale;
}

pub const Sim = struct {
    csr_base: u32,
    flash_base: u32,
    flash: []const u8, // borrowed int4 weight image
    gpa: std.mem.Allocator,

    // Resident scales: set these to emulate the accelerator reading per-row
    // scales from flash (one fp16 per 32-bit word) instead of SCALE_PUSH.
    scale_flash_base: u32 = 0,
    scales_flash: []const u8 = &.{},

    col_tiles: u32 = 0,
    row_blocks: u32 = 0,
    weight_base: u32 = 0,
    scale_base: u32 = 0,
    /// Host-provided activation scale as f32 bits (0 = derive from max-abs).
    act_scale_override: u32 = 0,
    /// Per-group W4A8: byte offset added to the scale read address only (set via
    /// REG_SCALE_GROUP_OFF), so a col-block reads its group's scales without
    /// moving the col-tile accumulate row. Cleared after each run.
    scale_group_off: u32 = 0,
    acts: std.ArrayList(u16) = .empty,
    scales: std.ArrayList(u16) = .empty,
    results: std.ArrayList(u16) = .empty,
    /// Persistent fp32 result accumulator for column-tiling (MODE_COLTILE_*), so
    /// partials sum without per-block fp16 rounding. Indexed by global row.
    accum: std.ArrayList(f32) = .empty,
    resp: std.ArrayList(u8) = .empty,
    done: bool = false,
    // SwiGLU fusion: MODE selects normal / capture_gate / fuse_up; gate_buf holds
    // a captured gate result to fold silu(gate)*up on the next (fuse) matmul.
    mode: u32 = 0,
    gate_buf: std.ArrayList(u16) = .empty,

    pub fn init(gpa: std.mem.Allocator, csr_base: u32, flash_base: u32, flash: []const u8) Sim {
        return .{ .csr_base = csr_base, .flash_base = flash_base, .flash = flash, .gpa = gpa };
    }

    pub fn deinit(self: *Sim) void {
        self.acts.deinit(self.gpa);
        self.scales.deinit(self.gpa);
        self.results.deinit(self.gpa);
        self.accum.deinit(self.gpa);
        self.resp.deinit(self.gpa);
        self.gate_buf.deinit(self.gpa);
    }

    /// Applies a complete command frame (WRITE mutates state; READ stages the
    /// response bytes for the next `readResponse`).
    pub fn writeFrame(self: *Sim, bytes: []const u8) !void {
        std.debug.assert(bytes.len >= proto.HEADER_LEN); // our own framing
        const op = bytes[0];
        const addr = std.mem.readInt(u32, bytes[1..5], .little);
        const len = std.mem.readInt(u16, bytes[5..7], .little);
        if (op == proto.OP_WRITE) {
            const data = bytes[proto.HEADER_LEN..][0..len];
            var i: usize = 0;
            while (i < len) : (i += 4) {
                try self.applyWrite(addr + @as(u32, @intCast(i)), std.mem.readInt(u32, data[i..][0..4], .little));
            }
        } else if (op == proto.OP_READ) {
            self.resp.clearRetainingCapacity();
            var i: usize = 0;
            while (i < len) : (i += 4) {
                var wb: [4]u8 = undefined;
                std.mem.writeInt(u32, &wb, self.readReg(addr + @as(u32, @intCast(i))), .little);
                const n = @min(@as(usize, 4), len - i);
                try self.resp.appendSlice(self.gpa, wb[0..n]);
            }
        }
    }

    /// Copies the staged READ response into `buf`.
    pub fn readResponse(self: *Sim, buf: []u8) !void {
        if (buf.len != self.resp.items.len) return error.SimResponseLenMismatch;
        @memcpy(buf, self.resp.items);
    }

    /// Reg-level access matching transport.Device, so a device-generic consumer
    /// (linear.zig) drives the real device or this sim interchangeably.
    pub fn regWrite(self: *Sim, addr: u32, value: u32) !void {
        try self.applyWrite(addr, value);
    }

    pub fn regRead(self: *Sim, addr: u32) !u32 {
        return self.readReg(addr);
    }

    /// Matches the transport's coalesced-write API; the sim has no transport cost,
    /// so it just applies each write in order.
    pub fn writeRegs(self: *Sim, pairs: []const [2]u32) !void {
        for (pairs) |p| try self.regWrite(p[0], p[1]);
    }

    /// Matches the transport's burst-read API: reads consecutive words.
    pub fn readRegs(self: *Sim, addr: u32, words: []u32) !void {
        for (words, 0..) |*wd, i| wd.* = try self.regRead(addr + @as(u32, @intCast(i * 4)));
    }

    /// Matches the transport's streaming-write API: each value is written to the
    /// fixed `addr` (a FIFO push), exactly like the device holds the address.
    pub fn writeStream(self: *Sim, addr: u32, values: []const u32) !void {
        for (values) |v| try self.regWrite(addr, v);
    }

    /// No-op: the emulator serves all weights from its single flash image, so the
    /// tiered BRAM cache is a hardware-only path. Sim tests use bram_cache_kb=0
    /// (every matrix `store == .flash`), so this is never actually called.
    pub fn writeMem(self: *Sim, addr: u32, bytes: []const u8) !void {
        _ = self;
        _ = addr;
        _ = bytes;
    }

    /// Erases this emulator into the runner-facing `Device` interface, so the
    /// same forward pass drives the sim or real silicon.
    pub fn device(self: *Sim) Iface {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn cast(ptr: *anyopaque) *Sim {
        return @ptrCast(@alignCast(ptr));
    }
    const vtable = Iface.VTable{
        .reg_write = struct {
            fn f(p: *anyopaque, a: u32, v: u32) anyerror!void {
                return cast(p).regWrite(a, v);
            }
        }.f,
        .reg_read = struct {
            fn f(p: *anyopaque, a: u32) anyerror!u32 {
                return cast(p).regRead(a);
            }
        }.f,
        .write_regs = struct {
            fn f(p: *anyopaque, pairs: []const [2]u32) anyerror!void {
                return cast(p).writeRegs(pairs);
            }
        }.f,
        .read_regs = struct {
            fn f(p: *anyopaque, a: u32, w: []u32) anyerror!void {
                return cast(p).readRegs(a, w);
            }
        }.f,
        .write_stream = struct {
            fn f(p: *anyopaque, a: u32, vals: []const u32) anyerror!void {
                return cast(p).writeStream(a, vals);
            }
        }.f,
        .write_mem = struct {
            fn f(p: *anyopaque, a: u32, bytes: []const u8) anyerror!void {
                return cast(p).writeMem(a, bytes);
            }
        }.f,
    };

    fn applyWrite(self: *Sim, addr: u32, word: u32) !void {
        switch (addr -% self.csr_base) {
            fp.REG_COL_TILES => self.col_tiles = word,
            fp.REG_ROW_BLOCKS => self.row_blocks = word,
            fp.REG_WEIGHT_BASE => self.weight_base = word,
            fp.REG_ACT_PUSH => try self.acts.append(self.gpa, @truncate(word)),
            fp.REG_ACT_PUSH2 => { // two packed fp16 acts: low16 then high16
                try self.acts.append(self.gpa, @truncate(word));
                try self.acts.append(self.gpa, @truncate(word >> 16));
            },
            fp.REG_SCALE_PUSH => try self.scales.append(self.gpa, @truncate(word)),
            fp.REG_SCALE_BASE => self.scale_base = word,
            fp.REG_ACT_SCALE => self.act_scale_override = word,
            fp.REG_SCALE_GROUP_OFF => self.scale_group_off = word,
            fp.REG_MODE => self.mode = word,
            fp.REG_CONTROL => if (word & fp.CONTROL_START != 0) try self.run(),
            else => {}, // weight-region writes don't happen in the flash flow
        }
    }

    fn readReg(self: Sim, addr: u32) u32 {
        const off = addr -% self.csr_base;
        if (off == fp.REG_VERSION) return fp.VERSION_MAGIC;
        if (off == fp.REG_STATUS) return if (self.done) fp.STATUS_DONE else fp.STATUS_BUSY;
        if (off >= fp.REG_RESULT and off < fp.REG_RESULT + (fp.MAX_ROWS_PER_CALL / 2) * 4) {
            // Packed: word w = {result 2w+1 high16, result 2w low16}.
            const w = (off - fp.REG_RESULT) / 4;
            const lo: u32 = if (2 * w < self.results.items.len) self.results.items[2 * w] else 0;
            const hi: u32 = if (2 * w + 1 < self.results.items.len) self.results.items[2 * w + 1] else 0;
            return lo | (hi << 16);
        }
        return 0;
    }

    /// The W4A8 matmul, mirroring EmulatedLoomDevice._run bit-for-bit.
    fn run(self: *Sim) !void {
        const cols: usize = self.col_tiles * 2;
        const rows: usize = self.row_blocks * 2;

        const x = try self.gpa.alloc(f32, cols);
        defer self.gpa.free(x);
        for (0..cols) |c| x[c] = if (c < self.acts.items.len) fp.fromBits(self.acts.items[c]) else 0;

        const qx = try self.gpa.alloc(i8, cols);
        defer self.gpa.free(qx);
        // A host-provided activation scale (col-tiling) quantizes on that fixed
        // grid; otherwise the accelerator derives it from this block's max-abs.
        const act_scale = if (self.act_scale_override != 0)
            quantWithScale(x, fp.fromBits(@as(u16, @truncate(self.act_scale_override))), qx)
        else
            quantPerTensorI8(x, qx);

        const words_per_row = std.math.divCeil(usize, fp.colTiles(cols), 2) catch unreachable;
        const nbytes = (rows / 2) * words_per_row * 4;
        const off = self.weight_base - self.flash_base;
        const w = try self.gpa.alloc(i8, rows * cols);
        defer self.gpa.free(w);
        unpackTileMajor(self.flash[off .. off + nbytes], rows, cols, w);

        // Column-tiling accumulate: the RTL's on-chip accumulator is a maxRows bank
        // indexed by LOCAL row (reset on COLTILE_FIRST) and REUSED per row-tile. The
        // runtime must drive col-blocks INNER / row-tiles OUTER so a row-tile finishes
        // all its blocks (FIRST..LAST) before the next reuses the bank. We model that
        // faithfully with a LOCAL accumulator (index r), so a col-outer driver that
        // interleaved row-tiles would corrupt here exactly as it would on silicon.
        // The scale READ below stays global (via scale_base + r*4), only the accum
        // index is local. Partials sum in fp32; the fp16 round happens only at LAST.
        const coltile = self.mode >= fp.MODE_COLTILE_FIRST and self.mode <= fp.MODE_COLTILE_LAST;
        if (coltile) {
            while (self.accum.items.len < rows) try self.accum.append(self.gpa, 0);
        }

        self.results.clearRetainingCapacity();
        try self.results.ensureTotalCapacity(self.gpa, rows);
        for (0..rows) |r| {
            var acc: i64 = 0;
            for (0..cols) |c| acc += @as(i64, w[r * cols + c]) * @as(i64, qx[c]);
            // Resident scales: read row r's fp16 from flash (word r, low16). Else
            // the pushed scale buffer.
            const row_scale = blk: {
                if (self.scale_base != 0) {
                    // group_off shifts only the scale READ (per-group W4A8), not the
                    // local accum index, so a col-block dequants on its group's per-row
                    // scales while accumulating into the same (local) result row.
                    const soff = self.scale_base - self.scale_flash_base + self.scale_group_off + r * 4;
                    if (soff + 2 <= self.scales_flash.len)
                        break :blk fp.fromBits(std.mem.readInt(u16, self.scales_flash[soff..][0..2], .little));
                    break :blk 0;
                }
                break :blk if (r < self.scales.items.len) fp.fromBits(self.scales.items[r]) else 0;
            };
            const y: f32 = @as(f32, @floatFromInt(acc)) * row_scale * act_scale;
            if (coltile) {
                const gr = r; // local accum index (RTL reuses the maxRows bank per row-tile)
                if (self.mode == fp.MODE_COLTILE_FIRST) self.accum.items[gr] = y else self.accum.items[gr] += y;
                self.results.appendAssumeCapacity(if (self.mode == fp.MODE_COLTILE_LAST) fp.toBits(self.accum.items[gr]) else 0);
            } else {
                self.results.appendAssumeCapacity(fp.toBits(y));
            }
        }

        // SwiGLU fusion (mirrors the accelerator's on-chip fold): CAPTURE_GATE
        // stashes this matmul's result as `gate`; FUSE_UP folds silu(gate)*result
        // in place so the host reads the SwiGLU product.
        if (self.mode == fp.MODE_CAPTURE_GATE) {
            self.gate_buf.clearRetainingCapacity();
            try self.gate_buf.appendSlice(self.gpa, self.results.items);
        } else if (self.mode == fp.MODE_FUSE_UP) {
            for (self.results.items, 0..) |*rbits, i| {
                const g: f32 = if (i < self.gate_buf.items.len) fp.fromBits(self.gate_buf.items[i]) else 0;
                const u: f32 = fp.fromBits(rbits.*);
                const silu = g / (1.0 + std.math.exp(-g));
                rbits.* = fp.toBits(silu * u);
            }
        }

        self.acts.clearRetainingCapacity();
        self.scales.clearRetainingCapacity();
        self.act_scale_override = 0;
        self.scale_group_off = 0;
        self.done = true;
    }
};

test "unpackTileMajor round-trips a small int4 matrix" {
    // 2x2 matrix packed as one tile in one word: nibbles w00,w01 / w10,w11.
    const bytes = [_]u8{ 0x21, 0x43, 0, 0 }; // b0=0x21 -> w00=1,w01=2; b1=0x43 -> w10=3,w11=4
    var out: [4]i8 = undefined;
    unpackTileMajor(&bytes, 2, 2, &out);
    try std.testing.expectEqualSlices(i8, &[_]i8{ 1, 2, 3, 4 }, &out);
}

test "quantPerTensorI8 scales to the max-abs and rounds half away" {
    var q: [3]i8 = undefined;
    const s = quantPerTensorI8(&[_]f32{ 1.27, -0.635, 0 }, &q);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), s, 1e-6);
    try std.testing.expectEqual(@as(i8, 127), q[0]);
    try std.testing.expectEqual(@as(i8, -64), q[1]); // -63.5 -> -64 (away from zero)
    try std.testing.expectEqual(@as(i8, 0), q[2]);
}
