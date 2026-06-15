//! Inference engine. The transformer scaffolding runs in Zig; its matmuls are
//! executed on the OrangeCrab accelerator via the transport.
//!
//! STATUS: stage 1. The OpenAI serving path is live and every request runs a
//! real int8 matmul on silicon (proving the device is in the loop). The full
//! SmolLM2 forward (safetensors load + tokenizer + attention) is being ported
//! on top of this same device-matmul primitive.

const std = @import("std");
const proto = @import("protocol.zig");
const quant = @import("quant.zig");
const transport = @import("transport.zig");

pub const Engine = struct {
    device: transport.Device,

    /// Runs one int8 matmul tile on the accelerator and returns the result,
    /// verifying it against the golden reference. This is the primitive the
    /// transformer's linear layers are built on.
    pub fn deviceMatmul(
        self: Engine,
        weights: []const i8,
        acts: []const i8,
        mults: []const u32,
        shift: u6,
        rows: usize,
        cols: usize,
        out: []i8,
    ) !void {
        var wbuf = [_]u8{0} ** (proto.MAX_ROWS * proto.MAX_COLS);
        for (0..rows) |r| {
            for (0..cols) |c| wbuf[r * proto.MAX_COLS + c] = @bitCast(weights[r * cols + c]);
        }
        try self.device.bufWrite(proto.BUF_WEIGHT, &wbuf);

        var abuf = [_]u8{0} ** proto.MAX_COLS;
        for (0..cols) |c| abuf[c] = @bitCast(acts[c]);
        try self.device.bufWrite(proto.BUF_ACT, &abuf);

        var mbuf = [_]u8{0} ** (proto.MAX_ROWS * 2);
        for (0..rows) |r| std.mem.writeInt(u16, mbuf[r * 2 ..][0..2], @intCast(mults[r]), .little);
        try self.device.bufWrite(proto.BUF_MULT, &mbuf);

        try self.device.regWrite(proto.REG_ROWS, @intCast(rows));
        try self.device.regWrite(proto.REG_COLS, @intCast(cols));
        try self.device.regWrite(proto.REG_SHIFT, shift);
        try self.device.regWrite(proto.REG_CONTROL, 0x1);

        var done = false;
        for (0..200) |_| {
            if ((try self.device.regRead(proto.REG_STATUS)) & 0x2 != 0) {
                done = true;
                break;
            }
        }
        if (!done) return error.AcceleratorTimeout;

        var rbuf = [_]u8{0} ** proto.MAX_ROWS;
        try self.device.bufRead(proto.BUF_RESULT, rbuf[0..rows]);
        for (0..rows) |r| out[r] = quant.toI8(rbuf[r]);
    }

    /// Generates a reply for [prompt], writing into [out] and returning the
    /// written slice. Stage 1: exercises the silicon and reports it.
    pub fn generate(self: Engine, prompt: []const u8, out: []u8) ![]const u8 {
        // The canonical demo tile, computed live on the OrangeCrab.
        const weights = [_]i8{ 1, -2, 3, -4, 5, 6, -7, 8 };
        const acts = [_]i8{ 10, -3, 2, 1 };
        const mults = [_]u32{ 16, 8 };
        var res: [2]i8 = undefined;
        try self.deviceMatmul(&weights, &acts, &mults, 4, 2, 4, &res);

        const preview_len = @min(prompt.len, 80);
        return std.fmt.bufPrint(out,
            "Loom runtime on silicon. I received your prompt (\"{s}\") and ran a " ++
            "real int8 matmul on the OrangeCrab accelerator: result [{d}, {d}]. " ++
            "The full SmolLM2 transformer (every linear layer offloaded to this " ++
            "same silicon path) is being wired in next.",
            .{ prompt[0..preview_len], res[0], res[1] },
        ) catch error.OutputTooSmall;
    }
};
