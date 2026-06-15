//! Host-side image preprocessing for the vision tower: turn a decoded RGB image
//! into the normalized, channels-first tensor `vision.encodeImage` expects.
//! Pure Zig, no dependencies (Loom builds its own front door). The JPEG decoder
//! (below) plus `preprocess` form the "image file -> ViT input" path.
const std = @import("std");

/// Bilinear resize of an `in_h x in_w x channels` (row-major, HWC, u8) image to
/// `out_h x out_w`, writing rounded u8 HWC into `dst`. Uses PIL's pixel-center
/// convention (`center = (o+0.5)*scale - 0.5`) with edge clamping, so it matches
/// PIL's BILINEAR exactly on upscales (downscales differ: PIL antialiases, this
/// does plain bilinear - close enough for ViT features, not bit-exact).
pub fn resizeBilinear(src: []const u8, in_w: usize, in_h: usize, channels: usize, dst: []u8, out_w: usize, out_h: usize) void {
    std.debug.assert(src.len == in_w * in_h * channels);
    std.debug.assert(dst.len == out_w * out_h * channels);
    const sx = @as(f32, @floatFromInt(in_w)) / @as(f32, @floatFromInt(out_w));
    const sy = @as(f32, @floatFromInt(in_h)) / @as(f32, @floatFromInt(out_h));
    const iw: i64 = @intCast(in_w);
    const ih: i64 = @intCast(in_h);

    for (0..out_h) |oy| {
        const cy = (@as(f32, @floatFromInt(oy)) + 0.5) * sy - 0.5;
        const y0f = @floor(cy);
        const fy = cy - y0f;
        const y0: i64 = @intFromFloat(y0f);
        const y0c: usize = @intCast(std.math.clamp(y0, 0, ih - 1));
        const y1c: usize = @intCast(std.math.clamp(y0 + 1, 0, ih - 1));
        for (0..out_w) |ox| {
            const cx = (@as(f32, @floatFromInt(ox)) + 0.5) * sx - 0.5;
            const x0f = @floor(cx);
            const fx = cx - x0f;
            const x0: i64 = @intFromFloat(x0f);
            const x0c: usize = @intCast(std.math.clamp(x0, 0, iw - 1));
            const x1c: usize = @intCast(std.math.clamp(x0 + 1, 0, iw - 1));
            for (0..channels) |c| {
                const v00: f32 = @floatFromInt(src[(y0c * in_w + x0c) * channels + c]);
                const v01: f32 = @floatFromInt(src[(y0c * in_w + x1c) * channels + c]);
                const v10: f32 = @floatFromInt(src[(y1c * in_w + x0c) * channels + c]);
                const v11: f32 = @floatFromInt(src[(y1c * in_w + x1c) * channels + c]);
                const top = v00 * (1 - fx) + v01 * fx;
                const bot = v10 * (1 - fx) + v11 * fx;
                const v = top * (1 - fy) + bot * fy;
                const r = std.math.clamp(@round(v), 0.0, 255.0);
                dst[(oy * out_w + ox) * channels + c] = @intFromFloat(r);
            }
        }
    }
}

/// Full preprocess: resize a `src` (`in_h x in_w x 3` HWC u8 RGB) to
/// `size x size`, rescale to [0,1], normalize per channel with `mean`/`std`, and
/// write the result channels-first (`3 x size x size`) into `out`. Matches the
/// HF CLIP/SigLIP image processor (resize+rescale+normalize). Allocates a small
/// resize scratch. `out.len` must be `3*size*size`.
pub fn preprocess(gpa: std.mem.Allocator, src: []const u8, in_w: usize, in_h: usize, size: usize, mean: [3]f32, std_: [3]f32, out: []f32) !void {
    std.debug.assert(out.len == 3 * size * size);
    const resized = try gpa.alloc(u8, size * size * 3);
    defer gpa.free(resized);
    resizeBilinear(src, in_w, in_h, 3, resized, size, size);
    const plane = size * size;
    for (0..plane) |i| {
        for (0..3) |c| {
            const p: f32 = @floatFromInt(resized[i * 3 + c]);
            out[c * plane + i] = (p / 255.0 - mean[c]) / std_[c];
        }
    }
}

// ---------------------------------------------------------------------------
// Baseline (SOF0) JPEG decoder. Clean-room, no dependencies. Supports grayscale
// and YCbCr, Huffman entropy coding, restart markers, and arbitrary chroma
// sampling (nearest-neighbour upsampling). Progressive JPEG is not supported.
// A float separable IDCT is used, so output differs from libjpeg's integer IDCT
// by at most ~1-2 per channel.
// ---------------------------------------------------------------------------

/// A decoded image (`width*height*3` RGB, HWC, owned by the caller).
pub const Decoded = struct {
    rgb: []u8,
    width: usize,
    height: usize,
    pub fn deinit(self: *Decoded, gpa: std.mem.Allocator) void {
        gpa.free(self.rgb);
    }
};

pub const JpegError = error{
    NotJpeg,
    Truncated,
    UnsupportedMode, // progressive / arithmetic / non-8-bit
    BadHuffman,
    BadComponent,
};

/// JPEG zigzag scan order -> natural 8x8 index.
const ZIGZAG = [64]u8{
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
};

const HuffTable = struct {
    mincode: [17]i32 = [_]i32{0} ** 17,
    maxcode: [17]i32 = [_]i32{-1} ** 17,
    valptr: [17]u16 = [_]u16{0} ** 17,
    symbols: [256]u8 = undefined,
    defined: bool = false,

    fn build(self: *HuffTable, counts: [16]u8) void {
        var code: i32 = 0;
        var k: u16 = 0;
        for (1..17) |len| {
            self.valptr[len] = k;
            self.mincode[len] = code;
            const c = counts[len - 1];
            if (c > 0) {
                code += c;
                self.maxcode[len] = code - 1;
                k += c;
            } else {
                self.maxcode[len] = -1;
            }
            code <<= 1;
        }
        self.defined = true;
    }
};

const Component = struct {
    id: u8,
    h: u8,
    v: u8,
    quant_id: u8,
    dc_pred: i32 = 0,
    dc_table: u8 = 0,
    ac_table: u8 = 0,
};

const BitReader = struct {
    data: []const u8,
    pos: usize,
    bits: u8 = 0, // current byte
    nbits: u4 = 0, // valid bits remaining in `bits` (MSB-first)
    marker: u8 = 0, // non-zero once a marker (restart/EOI) is reached

    /// Loads the next entropy byte, handling 0xFF byte-stuffing. On a real
    /// marker it stops (leaves `pos` at the 0xFF) and feeds zeros thereafter.
    fn fill(self: *BitReader) void {
        if (self.marker != 0 or self.pos >= self.data.len) {
            self.bits = 0;
            self.nbits = 8;
            if (self.marker == 0) self.marker = 0xFF; // ran off the end
            return;
        }
        var b = self.data[self.pos];
        self.pos += 1;
        if (b == 0xFF) {
            const nxt = if (self.pos < self.data.len) self.data[self.pos] else 0;
            if (nxt == 0x00) {
                self.pos += 1; // stuffed literal 0xFF
            } else {
                self.pos -= 1; // rewind to the 0xFF; it is a marker
                self.marker = nxt;
                b = 0;
            }
        }
        self.bits = b;
        self.nbits = 8;
    }

    fn getBit(self: *BitReader) u1 {
        if (self.nbits == 0) self.fill();
        self.nbits -= 1;
        return @intCast((self.bits >> @intCast(self.nbits)) & 1);
    }

    fn getBits(self: *BitReader, n: u5) u32 {
        var v: u32 = 0;
        var i: u5 = 0;
        while (i < n) : (i += 1) v = (v << 1) | self.getBit();
        return v;
    }

    fn decode(self: *BitReader, t: *const HuffTable) !u8 {
        var code: i32 = 0;
        for (1..17) |len| {
            code = (code << 1) | self.getBit();
            if (t.maxcode[len] >= 0 and code <= t.maxcode[len]) {
                const idx = t.valptr[len] + @as(u16, @intCast(code - t.mincode[len]));
                return t.symbols[idx];
            }
        }
        return JpegError.BadHuffman;
    }

    /// Discard partial bits at a restart marker (the caller realigns `pos`).
    fn restart(self: *BitReader) void {
        self.nbits = 0;
        self.marker = 0;
    }
};

/// Sign-extend an `s`-bit magnitude read from the stream (JPEG "receive+extend").
fn extend(v: u32, s: u5) i32 {
    const vv: i32 = @intCast(v);
    if (s == 0) return 0;
    const half: i32 = @as(i32, 1) << (s - 1);
    return if (vv < half) vv - (@as(i32, 1) << s) + 1 else vv;
}

/// Separable float 8x8 inverse DCT of `block` (natural order) into `out`
/// (spatial, level-shifted +128, clamped 0..255).
fn idct8x8(block: *const [64]f32, out: *[64]f32) void {
    const C = struct {
        var table: [8][8]f32 = undefined;
        var init = false;
    };
    if (!C.init) {
        for (0..8) |x| {
            for (0..8) |u| {
                const cu: f32 = if (u == 0) 0.70710678118654752 else 1.0;
                C.table[x][u] = cu * @cos((2.0 * @as(f32, @floatFromInt(x)) + 1.0) * @as(f32, @floatFromInt(u)) * std.math.pi / 16.0);
            }
        }
        C.init = true;
    }
    var tmp: [64]f32 = undefined;
    // Rows: for each row y, IDCT over u.
    for (0..8) |y| {
        for (0..8) |x| {
            var sum: f32 = 0;
            for (0..8) |u| sum += C.table[x][u] * block[y * 8 + u];
            tmp[y * 8 + x] = sum * 0.5;
        }
    }
    // Columns: for each column x, IDCT over v.
    for (0..8) |x| {
        for (0..8) |y| {
            var sum: f32 = 0;
            for (0..8) |v| sum += C.table[y][v] * tmp[v * 8 + x];
            const val = sum * 0.5 + 128.0;
            out[y * 8 + x] = std.math.clamp(val, 0.0, 255.0);
        }
    }
}

fn rd16(d: []const u8, i: usize) u16 {
    return (@as(u16, d[i]) << 8) | d[i + 1];
}

/// Decodes a baseline JPEG into RGB. Caller owns `result.rgb`.
pub fn decodeJpeg(gpa: std.mem.Allocator, data: []const u8) !Decoded {
    if (data.len < 2 or data[0] != 0xFF or data[1] != 0xD8) return JpegError.NotJpeg;

    var quant: [4][64]u16 = undefined; // zigzag order
    var huff_dc = [_]HuffTable{.{}} ** 4;
    var huff_ac = [_]HuffTable{.{}} ** 4;
    var comps: [3]Component = undefined;
    var ncomp: usize = 0;
    var width: usize = 0;
    var height: usize = 0;
    var restart_interval: usize = 0;

    var p: usize = 2;
    while (p + 4 <= data.len) {
        if (data[p] != 0xFF) return JpegError.Truncated;
        const marker = data[p + 1];
        p += 2;
        if (marker == 0xD9) break; // EOI
        const seg_len = rd16(data, p);
        const seg = data[p + 2 .. p + seg_len];
        const seg_end = p + seg_len;

        switch (marker) {
            0xDB => { // DQT
                var q: usize = 0;
                while (q < seg.len) {
                    const pq = seg[q] >> 4; // precision: 0=8bit,1=16bit
                    const tq = seg[q] & 0x0F;
                    q += 1;
                    for (0..64) |i| {
                        if (pq == 0) {
                            quant[tq][i] = seg[q];
                            q += 1;
                        } else {
                            quant[tq][i] = rd16(seg, q);
                            q += 2;
                        }
                    }
                }
            },
            0xC0 => { // SOF0 (baseline)
                if (seg[0] != 8) return JpegError.UnsupportedMode;
                height = rd16(seg, 1);
                width = rd16(seg, 3);
                ncomp = seg[5];
                if (ncomp != 1 and ncomp != 3) return JpegError.BadComponent;
                var c: usize = 0;
                while (c < ncomp) : (c += 1) {
                    const o = 6 + c * 3;
                    comps[c] = .{ .id = seg[o], .h = seg[o + 1] >> 4, .v = seg[o + 1] & 0x0F, .quant_id = seg[o + 2] };
                }
            },
            0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB => return JpegError.UnsupportedMode,
            0xC4 => { // DHT
                var q: usize = 0;
                while (q < seg.len) {
                    const tc = seg[q] >> 4; // 0=DC,1=AC
                    const th = seg[q] & 0x0F;
                    q += 1;
                    var counts: [16]u8 = undefined;
                    var total: usize = 0;
                    for (0..16) |i| {
                        counts[i] = seg[q + i];
                        total += counts[i];
                    }
                    q += 16;
                    var t = if (tc == 0) &huff_dc[th] else &huff_ac[th];
                    t.build(counts);
                    for (0..total) |i| t.symbols[i] = seg[q + i];
                    q += total;
                }
            },
            0xDD => { // DRI
                restart_interval = rd16(seg, 0);
            },
            0xDA => { // SOS -> entropy data follows the segment
                const ns = seg[0];
                var c: usize = 0;
                while (c < ns) : (c += 1) {
                    const cid = seg[1 + c * 2];
                    const tables = seg[2 + c * 2];
                    for (0..ncomp) |ci| {
                        if (comps[ci].id == cid) {
                            comps[ci].dc_table = tables >> 4;
                            comps[ci].ac_table = tables & 0x0F;
                        }
                    }
                }
                return try decodeScan(gpa, data, seg_end, width, height, ncomp, &comps, &quant, &huff_dc, &huff_ac, restart_interval);
            },
            else => {}, // APPn, COM, etc: skip
        }
        p = seg_end;
    }
    return JpegError.Truncated;
}

fn decodeScan(gpa: std.mem.Allocator, data: []const u8, scan_start: usize, width: usize, height: usize, ncomp: usize, comps: *[3]Component, quant: *[4][64]u16, huff_dc: *[4]HuffTable, huff_ac: *[4]HuffTable, restart_interval: usize) !Decoded {
    var hmax: usize = 1;
    var vmax: usize = 1;
    for (0..ncomp) |c| {
        hmax = @max(hmax, comps[c].h);
        vmax = @max(vmax, comps[c].v);
    }
    const mcu_w = 8 * hmax;
    const mcu_h = 8 * vmax;
    const mcus_x = (width + mcu_w - 1) / mcu_w;
    const mcus_y = (height + mcu_h - 1) / mcu_h;

    // Full-resolution sample plane per component.
    var planes: [3][]u8 = undefined;
    var nplanes: usize = 0;
    defer for (0..nplanes) |i| gpa.free(planes[i]);
    for (0..ncomp) |c| {
        planes[c] = try gpa.alloc(u8, mcus_x * mcu_w * mcus_y * mcu_h);
        nplanes += 1;
    }
    const plane_w = mcus_x * mcu_w;

    var br = BitReader{ .data = data, .pos = scan_start };
    for (0..ncomp) |c| comps[c].dc_pred = 0;

    var mcu_index: usize = 0;
    for (0..mcus_y) |my| {
        for (0..mcus_x) |mx| {
            if (restart_interval != 0 and mcu_index != 0 and mcu_index % restart_interval == 0) {
                // Consume the restart marker and realign.
                br.restart();
                // Skip to just past the RSTn marker in the stream.
                while (br.pos + 1 < data.len) {
                    if (data[br.pos] == 0xFF and data[br.pos + 1] >= 0xD0 and data[br.pos + 1] <= 0xD7) {
                        br.pos += 2;
                        break;
                    }
                    br.pos += 1;
                }
                for (0..ncomp) |c| comps[c].dc_pred = 0;
            }
            for (0..ncomp) |c| {
                const comp = &comps[c];
                for (0..comp.v) |by| {
                    for (0..comp.h) |bx| {
                        var block: [64]f32 = [_]f32{0} ** 64;
                        try decodeBlock(&br, comp, quant[comp.quant_id], &huff_dc[comp.dc_table], &huff_ac[comp.ac_table], &block);
                        var spatial: [64]f32 = undefined;
                        idct8x8(&block, &spatial);
                        // Place the 8x8 block into the component plane, scaled up
                        // by (hmax/h, vmax/v) via nearest replication.
                        const sx = hmax / comp.h;
                        const sy = vmax / comp.v;
                        const base_x = mx * mcu_w + bx * 8 * sx;
                        const base_y = my * mcu_h + by * 8 * sy;
                        for (0..8) |yy| {
                            for (0..8) |xx| {
                                const val: u8 = @intFromFloat(spatial[yy * 8 + xx]);
                                for (0..sy) |ry| {
                                    for (0..sx) |rx| {
                                        const px = base_x + xx * sx + rx;
                                        const py2 = base_y + yy * sy + ry;
                                        planes[c][py2 * plane_w + px] = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            mcu_index += 1;
        }
    }

    // Compose RGB.
    const rgb = try gpa.alloc(u8, width * height * 3);
    errdefer gpa.free(rgb);
    for (0..height) |y| {
        for (0..width) |x| {
            const yy: f32 = @floatFromInt(planes[0][y * plane_w + x]);
            var r = yy;
            var g = yy;
            var b = yy;
            if (ncomp == 3) {
                const cb: f32 = @as(f32, @floatFromInt(planes[1][y * plane_w + x])) - 128.0;
                const cr: f32 = @as(f32, @floatFromInt(planes[2][y * plane_w + x])) - 128.0;
                r = yy + 1.402 * cr;
                g = yy - 0.344136 * cb - 0.714136 * cr;
                b = yy + 1.772 * cb;
            }
            const o = (y * width + x) * 3;
            rgb[o] = @intFromFloat(std.math.clamp(@round(r), 0.0, 255.0));
            rgb[o + 1] = @intFromFloat(std.math.clamp(@round(g), 0.0, 255.0));
            rgb[o + 2] = @intFromFloat(std.math.clamp(@round(b), 0.0, 255.0));
        }
    }
    return .{ .rgb = rgb, .width = width, .height = height };
}

fn decodeBlock(br: *BitReader, comp: *Component, qt: [64]u16, dc: *const HuffTable, ac: *const HuffTable, block: *[64]f32) !void {
    // DC coefficient (differential).
    const s = try br.decode(dc);
    const diff = extend(br.getBits(@intCast(s)), @intCast(s));
    comp.dc_pred += diff;
    block[0] = @as(f32, @floatFromInt(comp.dc_pred)) * @as(f32, @floatFromInt(qt[0]));
    // AC coefficients.
    var k: usize = 1;
    while (k < 64) {
        const rs = try br.decode(ac);
        const r = rs >> 4;
        const sz: u5 = @intCast(rs & 0x0F);
        if (sz == 0) {
            if (r == 15) {
                k += 16; // ZRL: 16 zeros
                continue;
            }
            break; // EOB
        }
        k += r;
        if (k >= 64) break;
        const val = extend(br.getBits(sz), sz);
        block[ZIGZAG[k]] = @as(f32, @floatFromInt(val)) * @as(f32, @floatFromInt(qt[k]));
        k += 1;
    }
}

test "resizeBilinear upscales a 2x2 corner-exactly (PIL convention)" {
    // A 2x2 grayscale image upscaled to 4x4; corners equal the source corners.
    const src = [_]u8{ 0, 30, 90, 255 }; // 1 channel
    var dst: [16]u8 = undefined;
    resizeBilinear(&src, 2, 2, 1, &dst, 4, 4);
    try std.testing.expectEqual(@as(u8, 0), dst[0]); // top-left corner
    try std.testing.expectEqual(@as(u8, 30), dst[3]); // top-right
    try std.testing.expectEqual(@as(u8, 90), dst[12]); // bottom-left
    try std.testing.expectEqual(@as(u8, 255), dst[15]); // bottom-right
}

test "preprocess matches the PIL CLIP reference (2x2 -> 4x4)" {
    const src = [_]u8{ 10, 200, 30, 240, 20, 60, 50, 90, 220, 130, 160, 15 };
    const mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
    const std_ = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };
    var out: [48]f32 = undefined;
    try preprocess(std.testing.allocator, &src, 2, 2, 4, mean, std_, &out);
    const expected = [48]f32{
        -1.646278, -0.799570, 0.879250,  1.711360,  -1.500294, -0.784971,
        0.616478,  1.317202,  -1.208325, -0.770373, 0.076336,  0.514289,
        -1.062341, -0.770373, -0.186436, 0.105533,  1.249457,  0.574107,
        -0.776592, -1.451942, 0.844247,  0.394014,  -0.476437, -0.926670,
        0.018820,  0.048835,  0.108866,  0.123874,  -0.401398, -0.131258,
        0.394014,  0.649146,  -1.053618, -0.939857, -0.726556, -0.627016,
        -0.371055, -0.470595, -0.683896, -0.783437, 0.979852,  0.453709,
        -0.584356, -1.110498, 1.648195,  0.922971,  -0.541695, -1.266919,
    };
    for (out, expected) |got, want| {
        // ~2e-2 tolerance: PIL resizes with 8-bit fixed-point coefficients while
        // this uses f32 bilinear, so rounded pixels can differ by +-1 LSB.
        try std.testing.expectApproxEqAbs(want, got, 2e-2);
    }
}

test "decodeJpeg decodes a solid-color 4:4:4 baseline JPEG (~PIL)" {
    // 16x16 solid teal, JPEG quality 95, subsampling 0. Only DC survives, so the
    // decode should be near-uniform and match PIL's decode (21,160,141) closely.
    const SOLID_JPEG = [_]u8{ 255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 255, 219, 0, 67, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 4, 3, 2, 2, 2, 2, 5, 4, 4, 3, 4, 6, 5, 6, 6, 6, 5, 6, 6, 6, 7, 9, 8, 6, 7, 9, 7, 6, 6, 8, 11, 8, 9, 10, 10, 10, 10, 10, 6, 8, 11, 12, 11, 10, 12, 9, 10, 10, 10, 255, 219, 0, 67, 1, 2, 2, 2, 2, 2, 2, 5, 3, 3, 5, 10, 7, 6, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 255, 192, 0, 17, 8, 0, 16, 0, 16, 3, 1, 17, 0, 2, 17, 1, 3, 17, 1, 255, 196, 0, 31, 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 16, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125, 1, 2, 3, 0, 4, 17, 5, 18, 33, 49, 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129, 145, 161, 8, 35, 66, 177, 193, 21, 82, 209, 240, 36, 51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 196, 0, 31, 1, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 17, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119, 0, 1, 2, 3, 17, 4, 5, 33, 49, 6, 18, 65, 81, 7, 97, 113, 19, 34, 50, 129, 8, 20, 66, 145, 161, 177, 193, 9, 35, 51, 82, 240, 21, 98, 114, 209, 10, 22, 36, 52, 225, 37, 241, 23, 24, 25, 26, 38, 39, 40, 41, 42, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 130, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 226, 227, 228, 229, 230, 231, 232, 233, 234, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0, 227, 235, 238, 15, 243, 188, 40, 0, 160, 2, 128, 63, 255, 217 };
    var img = try decodeJpeg(std.testing.allocator, &SOLID_JPEG);
    defer img.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 16), img.width);
    try std.testing.expectEqual(@as(usize, 16), img.height);
    // Every pixel is ~ (21,160,141); a float IDCT differs from libjpeg by <= ~2.
    for (0..img.width * img.height) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 21), @floatFromInt(img.rgb[i * 3]), 3);
        try std.testing.expectApproxEqAbs(@as(f32, 160), @floatFromInt(img.rgb[i * 3 + 1]), 3);
        try std.testing.expectApproxEqAbs(@as(f32, 141), @floatFromInt(img.rgb[i * 3 + 2]), 3);
    }
}

test "decodeJpeg decodes a gradient (AC coefficients) close to PIL" {
    const GRAD_JPEG = [_]u8{255,216,255,224,0,16,74,70,73,70,0,1,1,0,0,1,0,1,0,0,255,219,0,67,0,3,2,2,2,2,2,3,2,2,2,3,3,3,3,4,6,4,4,4,4,4,8,6,6,5,6,9,8,10,10,9,8,9,9,10,12,15,12,10,11,14,11,9,9,13,17,13,14,15,16,16,17,16,10,12,18,19,18,16,19,15,16,16,16,255,219,0,67,1,3,3,3,4,3,4,8,4,4,8,16,11,9,11,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,255,192,0,17,8,0,16,0,16,3,1,17,0,2,17,1,3,17,1,255,196,0,31,0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,255,196,0,181,16,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125,1,2,3,0,4,17,5,18,33,49,65,6,19,81,97,7,34,113,20,50,129,145,161,8,35,66,177,193,21,82,209,240,36,51,98,114,130,9,10,22,23,24,25,26,37,38,39,40,41,42,52,53,54,55,56,57,58,67,68,69,70,71,72,73,74,83,84,85,86,87,88,89,90,99,100,101,102,103,104,105,106,115,116,117,118,119,120,121,122,131,132,133,134,135,136,137,138,146,147,148,149,150,151,152,153,154,162,163,164,165,166,167,168,169,170,178,179,180,181,182,183,184,185,186,194,195,196,197,198,199,200,201,202,210,211,212,213,214,215,216,217,218,225,226,227,228,229,230,231,232,233,234,241,242,243,244,245,246,247,248,249,250,255,196,0,31,1,0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,255,196,0,181,17,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,119,0,1,2,3,17,4,5,33,49,6,18,65,81,7,97,113,19,34,50,129,8,20,66,145,161,177,193,9,35,51,82,240,21,98,114,209,10,22,36,52,225,37,241,23,24,25,26,38,39,40,41,42,53,54,55,56,57,58,67,68,69,70,71,72,73,74,83,84,85,86,87,88,89,90,99,100,101,102,103,104,105,106,115,116,117,118,119,120,121,122,130,131,132,133,134,135,136,137,138,146,147,148,149,150,151,152,153,154,162,163,164,165,166,167,168,169,170,178,179,180,181,182,183,184,185,186,194,195,196,197,198,199,200,201,202,210,211,212,213,214,215,216,217,218,226,227,228,229,230,231,232,233,234,242,243,244,245,246,247,248,249,250,255,218,0,12,3,1,0,2,17,3,17,0,63,0,248,255,0,193,127,8,127,213,255,0,162,250,118,163,15,136,14,18,226,223,135,222,61,231,193,127,8,127,213,255,0,162,255,0,227,181,238,97,241,7,244,239,9,113,111,195,239,30,199,224,191,132,63,234,255,0,209,127,241,218,248,124,62,32,255,0,49,120,75,139,126,31,120,247,159,5,252,33,255,0,87,254,139,233,218,189,204,62,32,254,157,225,46,45,248,125,227,255,217};
    const GRAD_RGB = [_]u8{0,0,0,14,1,8,33,1,16,49,0,22,64,1,32,78,0,40,97,0,48,113,0,54,128,0,63,141,0,69,161,0,78,178,0,88,190,0,94,206,0,103,226,0,112,239,0,119,1,14,7,16,16,16,34,14,23,48,14,31,64,16,40,80,15,49,99,14,56,113,15,64,129,15,74,144,15,80,163,15,89,178,15,96,192,15,105,209,15,112,227,16,123,242,16,128,0,31,16,14,34,25,33,33,33,48,32,42,64,33,49,78,32,58,97,32,66,111,33,73,128,31,82,141,32,87,161,32,96,177,33,105,190,33,114,206,32,121,226,32,129,239,32,136,0,48,26,15,49,32,33,49,39,48,48,48,62,49,58,79,48,64,96,48,72,112,49,80,126,48,88,141,48,95,160,48,104,176,47,111,189,47,119,205,48,129,225,48,138,238,47,142,0,63,32,16,64,40,33,64,48,50,63,54,64,64,64,79,63,73,97,63,80,113,64,86,129,64,98,142,64,103,162,64,111,179,64,121,192,63,128,208,64,136,227,64,145,242,64,152,1,79,39,15,80,46,34,80,54,48,79,63,64,80,70,79,79,79,98,78,87,113,81,96,129,79,104,143,80,111,162,80,120,177,79,126,191,79,135,208,79,144,227,79,153,240,79,157,0,97,48,13,98,56,32,97,63,48,96,72,64,98,81,78,98,89,96,96,96,111,98,105,127,96,112,140,97,117,160,97,128,177,97,136,189,96,143,205,96,151,225,96,160,238,97,166,0,112,58,15,113,64,34,112,72,48,111,80,63,112,90,79,111,96,98,111,104,112,112,112,128,112,122,143,112,128,162,112,137,177,112,146,190,112,152,208,111,162,226,112,171,240,111,176,0,128,63,14,128,69,32,129,78,50,128,88,63,128,94,78,128,103,97,128,112,112,128,118,128,128,128,142,129,136,161,129,144,177,128,150,192,129,160,206,128,168,225,127,176,241,128,182,2,143,74,15,144,80,35,144,89,51,144,97,63,143,104,80,143,112,100,143,123,113,144,128,129,142,135,144,144,144,162,142,151,176,142,159,192,144,168,208,143,177,226,141,183,241,143,192,0,160,82,13,161,87,32,161,96,49,161,105,63,161,114,78,160,120,97,160,129,111,161,136,127,159,144,142,162,153,160,160,160,175,159,169,192,161,177,206,160,186,225,160,194,239,161,201,0,176,88,13,176,95,32,176,104,47,176,111,61,176,119,78,176,129,98,176,137,111,176,142,126,175,153,143,177,160,160,176,166,175,175,175,190,177,186,207,176,192,224,176,200,240,177,208,1,192,97,15,192,102,35,192,111,51,193,121,64,193,129,80,192,136,99,192,145,114,192,152,128,191,160,144,192,168,161,192,176,178,191,182,192,192,192,207,191,201,225,191,208,241,192,214,1,208,104,14,208,111,34,208,119,50,207,126,63,207,135,79,208,144,99,208,153,112,209,158,129,207,167,143,208,174,162,208,182,176,207,191,192,208,198,207,207,207,226,206,215,240,208,223,0,224,112,13,224,117,31,225,128,48,225,135,61,224,143,77,225,151,96,225,160,111,225,166,127,225,176,141,226,184,160,225,191,176,224,200,191,225,208,205,225,216,224,224,224,239,226,233,1,240,121,14,240,128,34,241,137,49,240,145,62,240,152,79,240,162,99,240,171,112,241,177,126,239,185,143,241,192,161,239,199,176,239,208,191,240,218,207,239,224,226,239,232,240,240,240};
    var img = try decodeJpeg(std.testing.allocator, &GRAD_JPEG);
    defer img.deinit(std.testing.allocator);
    var total: u64 = 0;
    var worst: u32 = 0;
    for (img.rgb, GRAD_RGB) |got, want| {
        const d = @abs(@as(i32, got) - @as(i32, want));
        total += @intCast(d);
        worst = @max(worst, @as(u32, @intCast(d)));
    }
    const mean = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(img.rgb.len));
    try std.testing.expect(mean < 2.0); // float IDCT vs libjpeg integer IDCT
    try std.testing.expect(worst <= 8);
}
