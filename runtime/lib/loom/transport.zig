//! Transport-agnostic Loom device access. Both UART and USB expose the same
//! byte-pipe primitives (writeFrame / readResponse); the register and buffer
//! protocol logic lives here once.

const std = @import("std");
const Io = std.Io;
const proto = @import("protocol.zig");
const uart = @import("uart.zig");
const usb = @import("usb.zig");
const Iface = @import("device.zig").Device;

pub const Kind = enum { uart, usb };

/// Max 32-bit words per READ response frame. Bounded by the DirtyJTAG CDC
/// bridge's read ceiling, NOT the accelerator: a single response above ~255
/// bytes intermittently corrupts (verified - 252B clean+deterministic, 344B
/// diverges run-to-run), so 63 words = 252 bytes is the safe max. readRegs
/// chunks wider row-tiles (172-row gate/up = 86 words) across frames. Also
/// sizes the on-stack response buffer.
const fp_max_read_words = 32;

const Link = union(Kind) {
    uart: uart.Port,
    usb: usb.Device,
};

pub const Device = struct {
    io: Io,
    link: Link,

    pub fn openUart(io: Io, path: []const u8, baud: u32) !Device {
        return .{ .io = io, .link = .{ .uart = try uart.Port.open(io, path, baud) } };
    }

    pub fn openUsb(io: Io) !Device {
        return .{ .io = io, .link = .{ .usb = try usb.Device.open(io) } };
    }

    pub fn close(self: Device) void {
        switch (self.link) {
            inline else => |t| t.close(self.io),
        }
    }

    fn writeFrame(self: Device, bytes: []const u8) !void {
        switch (self.link) {
            inline else => |t| try t.writeFrame(bytes),
        }
    }

    fn readResponse(self: Device, buf: []u8) !void {
        switch (self.link) {
            inline else => |t| try t.readResponse(buf),
        }
    }

    pub fn regWrite(self: Device, addr: u32, value: u32) !void {
        var frame: [proto.HEADER_LEN + 4]u8 = undefined;
        _ = proto.encodeWriteHeader(frame[0..proto.HEADER_LEN], addr, 4);
        std.mem.writeInt(u32, frame[proto.HEADER_LEN..][0..4], value, .little);
        try self.writeFrame(&frame);
    }

    /// Frames per burst. 92 * 11 = 1012 bytes, safely under the ~2KB CDC-bridge
    /// buffer whose overflow is the >2KB write wedge.
    const BURST_FRAMES = 92;

    /// Coalesces many single-word writes `{addr, value}` into few USB
    /// transactions. Individually each `regWrite` is its own USB round-trip, so a
    /// 64-wide matmul costs ~130 of them; packing the frames into ~1KB bursts and
    /// syncing with one barrier read between bursts collapses that to a handful.
    /// The device sees the identical byte stream either way. Same-address writes
    /// (FIFO pushes like ACT_PUSH) are preserved in order.
    pub fn writeRegs(self: Device, pairs: []const [2]u32) !void {
        const FRAME = proto.HEADER_LEN + 4;
        var buf: [BURST_FRAMES * FRAME]u8 = undefined;
        var i: usize = 0;
        while (i < pairs.len) {
            const n = @min(BURST_FRAMES, pairs.len - i);
            var off: usize = 0;
            for (pairs[i .. i + n]) |p| {
                _ = proto.encodeWriteHeader(buf[off..][0..proto.HEADER_LEN], p[0], 4);
                std.mem.writeInt(u32, buf[off + proto.HEADER_LEN ..][0..4], p[1], .little);
                off += FRAME;
            }
            try self.writeFrame(buf[0..off]);
            i += n;
            // Barrier between bursts: a pure read only returns once the bridge has
            // drained everything sent so far, so the next burst can't overflow it.
            // The last burst needs none (the caller's status poll drains it).
            if (i < pairs.len) _ = try self.regRead(proto.REG_VERSION);
        }
    }

    pub fn regRead(self: Device, addr: u32) !u32 {
        var hdr: [proto.HEADER_LEN]u8 = undefined;
        _ = proto.encodeRead(&hdr, addr, 4);
        try self.writeFrame(&hdr);
        var resp: [4]u8 = undefined;
        try self.readResponse(&resp);
        return std.mem.readInt(u32, &resp, .little);
    }

    /// Words per streaming burst. 240 * 4 = 960 bytes, under the ~2KB CDC bridge
    /// buffer.
    const STREAM_WORDS: usize = 240;

    /// Streams `values` as FIFO pushes to the fixed `addr` (opcode 0x03): one
    /// 7-byte header then the values as 32-bit LE words, so N pushes cost one
    /// header instead of N. Chunked into ~1KB bursts with a barrier read between
    /// (the device keeps pushing across chunks; the FIFO accumulates them).
    pub fn writeStream(self: Device, addr: u32, values: []const u32) !void {
        var buf: [proto.HEADER_LEN + STREAM_WORDS * 4]u8 = undefined;
        var i: usize = 0;
        while (i < values.len) {
            const n: usize = @min(STREAM_WORDS, values.len - i);
            const nbytes: usize = n * 4; // compute in usize, then narrow to the u16 len
            _ = proto.encodeWriteStreamHeader(buf[0..proto.HEADER_LEN], addr, @intCast(nbytes));
            var off: usize = proto.HEADER_LEN;
            for (values[i .. i + n]) |v| {
                std.mem.writeInt(u32, buf[off..][0..4], v, .little);
                off += 4;
            }
            try self.writeFrame(buf[0..off]);
            i += n;
            if (i < values.len) _ = try self.regRead(proto.REG_VERSION); // barrier
        }
    }

    /// Burst-writes `bytes` to INCREMENTING addresses from `addr` (opcode 0x01:
    /// the command engine advances the address per word, unlike the fixed-FIFO
    /// stream op). Chunked to ~1KB with a barrier read between chunks - a >2KB
    /// write desyncs the CDC bridge (the write wedge). Loads the on-chip BRAM
    /// weight cache at startup. `bytes.len` must be a multiple of 4.
    pub fn writeMem(self: Device, addr: u32, bytes: []const u8) !void {
        std.debug.assert(bytes.len % 4 == 0);
        var buf: [proto.HEADER_LEN + STREAM_WORDS * 4]u8 = undefined;
        var off: usize = 0;
        while (off < bytes.len) {
            const chunk = @min(STREAM_WORDS * 4, bytes.len - off);
            _ = proto.encodeWriteHeader(buf[0..proto.HEADER_LEN], addr + @as(u32, @intCast(off)), @intCast(chunk));
            @memcpy(buf[proto.HEADER_LEN..][0..chunk], bytes[off..][0..chunk]);
            try self.writeFrame(buf[0 .. proto.HEADER_LEN + chunk]);
            off += chunk;
            if (off < bytes.len) _ = try self.regRead(proto.REG_VERSION); // barrier
        }
    }

    /// Reads `words.len` consecutive 32-bit words from `addr`, in as few READ
    /// round-trips as the frame size allows (up to `fp_max_read_words` per frame,
    /// the command engine auto-increments the address within a frame). The per-row
    /// result read is the dominant cost of a matmul, so this is where the time
    /// goes. Row-tiles wider than one read frame (e.g. lm_head, gate/up at
    /// --fp-row-blocks 86 = 86 result words) span multiple frames.
    pub fn readRegs(self: Device, addr: u32, words: []u32) !void {
        var off: usize = 0;
        while (off < words.len) {
            const n: usize = @min(@as(usize, fp_max_read_words), words.len - off);
            const rd_addr: u32 = addr + @as(u32, @intCast(off)) * 4;
            const rd_len: u16 = @as(u16, @intCast(n)) * 4;
            var hdr: [proto.HEADER_LEN]u8 = undefined;
            _ = proto.encodeRead(&hdr, rd_addr, rd_len);
            try self.writeFrame(&hdr);
            var buf: [fp_max_read_words * 4]u8 = undefined;
            try self.readResponse(buf[0 .. n * 4]);
            for (0..n) |i| words[off + i] = std.mem.readInt(u32, buf[i * 4 ..][0..4], .little);
            off += n;
        }
    }

    /// Writes a byte buffer one 32-bit word at a time (the bus is
    /// word-addressed; the accelerator decodes the word index from addr[7:2]).
    pub fn bufWrite(self: Device, addr: u32, data: []const u8) !void {
        var i: usize = 0;
        while (i < data.len) : (i += 4) {
            var word: [4]u8 = .{ 0, 0, 0, 0 };
            const n = @min(4, data.len - i);
            @memcpy(word[0..n], data[i .. i + n]);
            try self.regWrite(addr + @as(u32, @intCast(i)), std.mem.readInt(u32, &word, .little));
        }
    }

    /// Reads the baked model name the silicon reports (MODEL_LEN + NAME region)
    /// into [out], returning the populated slice. The runtime stays model
    /// agnostic: the device declares what it is.
    pub fn readModelName(self: Device, out: []u8) ![]const u8 {
        const len = try self.regRead(proto.REG_MODEL_LEN);
        const n = @min(len, out.len);
        if (n == 0) return out[0..0];
        try self.bufRead(proto.BUF_NAME, out[0..n]);
        return out[0..n];
    }

    /// Reads `out.len` bytes from a buffer region, one word at a time.
    pub fn bufRead(self: Device, addr: u32, out: []u8) !void {
        var i: usize = 0;
        while (i < out.len) : (i += 4) {
            const word = try self.regRead(addr + @as(u32, @intCast(i)));
            var wb: [4]u8 = undefined;
            std.mem.writeInt(u32, &wb, word, .little);
            const n = @min(4, out.len - i);
            @memcpy(out[i .. i + n], wb[0..n]);
        }
    }

    /// Erases this concrete transport into the runner-facing `Device` interface.
    pub fn device(self: *Device) Iface {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn cast(ptr: *anyopaque) *Device {
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
};
