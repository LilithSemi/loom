//! Loom command protocol framing (pure, no IO).
//!
//! Wire format (little-endian):
//!   header = { opcode:u8, addr:u32, len:u16 }   (7 bytes)
//!   WRITE (0x01): header then `len` data bytes
//!   READ  (0x02): header only; device returns `len` raw bytes (LSB-first)

const std = @import("std");

pub const OP_WRITE: u8 = 0x01;
pub const OP_READ: u8 = 0x02;
/// Like WRITE but the device holds the address fixed: every word is a push to
/// the same FIFO register (streams N activations/scales under one header).
pub const OP_WRITE_STREAM: u8 = 0x03;

pub const HEADER_LEN: usize = 7;

// CSR offsets.
pub const REG_VERSION: u32 = 0x000;
pub const REG_ROWS: u32 = 0x004;
pub const REG_COLS: u32 = 0x008;
pub const REG_SHIFT: u32 = 0x00C;
pub const REG_CONTROL: u32 = 0x010;
pub const REG_STATUS: u32 = 0x014;
pub const REG_MODEL_LEN: u32 = 0x018;

// NAME region: baked ASCII model name, 4 chars/word.
pub const BUF_NAME: u32 = 0x500;

// Buffer offsets.
pub const BUF_WEIGHT: u32 = 0x100;
pub const BUF_ACT: u32 = 0x200;
pub const BUF_MULT: u32 = 0x300;
pub const BUF_RESULT: u32 = 0x400;

pub const VERSION_MAGIC: u32 = 0x4C4F4F4D; // 'LOOM'
pub const MAX_ROWS: usize = 8;
pub const MAX_COLS: usize = 8;

/// Encodes a WRITE command header into `buf` (just the 7-byte header). The
/// caller appends the payload. Returns the header length.
pub fn encodeWriteHeader(buf: []u8, addr: u32, len: u16) usize {
    buf[0] = OP_WRITE;
    std.mem.writeInt(u32, buf[1..5], addr, .little);
    std.mem.writeInt(u16, buf[5..7], len, .little);
    return HEADER_LEN;
}

/// Encodes a streaming-WRITE header (opcode 0x03) into `buf`. The caller appends
/// `len` payload bytes, all written to the fixed `addr` (a FIFO push burst).
pub fn encodeWriteStreamHeader(buf: []u8, addr: u32, len: u16) usize {
    buf[0] = OP_WRITE_STREAM;
    std.mem.writeInt(u32, buf[1..5], addr, .little);
    std.mem.writeInt(u16, buf[5..7], len, .little);
    return HEADER_LEN;
}

/// Encodes a READ command (header only) into `buf`. Returns the length written.
pub fn encodeRead(buf: []u8, addr: u32, len: u16) usize {
    buf[0] = OP_READ;
    std.mem.writeInt(u32, buf[1..5], addr, .little);
    std.mem.writeInt(u16, buf[5..7], len, .little);
    return HEADER_LEN;
}

test "encodeWriteHeader lays out opcode, LE addr, LE len" {
    var buf: [HEADER_LEN]u8 = undefined;
    const n = encodeWriteHeader(&buf, 0x0100, 4);
    try std.testing.expectEqual(HEADER_LEN, n);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x01, 0x00, 0x01, 0x00, 0x00, 0x04, 0x00 }, &buf);
}

test "encodeRead lays out READ opcode for a CSR" {
    var buf: [HEADER_LEN]u8 = undefined;
    const n = encodeRead(&buf, REG_VERSION, 4);
    try std.testing.expectEqual(HEADER_LEN, n);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x02, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00 }, &buf);
}

test "version magic spells LOOM little-endian" {
    // bytes on the wire LSB-first: 4d 4f 4f 4c
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, VERSION_MAGIC, .little);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x4D, 0x4F, 0x4F, 0x4C }, &b);
}
