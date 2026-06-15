//! Serial transport for the Loom command protocol over a CDC-ACM UART
//! (115200 8N1, the DirtyJTAG bridge that fronts the OrangeCrab).
//!
//! The node is opened with std.Io; termios config and the framed, timeout-
//! bounded reads use the raw fd (std.Io models neither).

const std = @import("std");
const posix = std.posix;
const linux = std.os.linux;
const Io = std.Io;
const dev = @import("dev.zig");

pub const Error = error{Timeout};

pub const Port = struct {
    file: Io.File,

    /// Maps a plain integer baud to the termios speed enum. Only the rates the
    /// device is built for are supported.
    fn speedOf(baud: u32) !std.os.linux.speed_t {
        return switch (baud) {
            115200 => .B115200,
            230400 => .B230400,
            460800 => .B460800,
            500000 => .B500000,
            921600 => .B921600,
            1000000 => .B1000000,
            1152000 => .B1152000,
            1500000 => .B1500000,
            2000000 => .B2000000,
            else => error.UnsupportedBaud,
        };
    }

    /// Opens and configures a serial port for raw 8N1 access at `baud`.
    pub fn open(io: Io, path: []const u8, baud: u32) !Port {
        const file = try Io.Dir.openFileAbsolute(io, path, .{
            .mode = .read_write,
            .allow_ctty = false,
        });
        errdefer file.close(io);

        const fd = file.handle;
        var tio = try posix.tcgetattr(fd);

        // Raw input.
        tio.iflag.IGNBRK = false;
        tio.iflag.BRKINT = false;
        tio.iflag.PARMRK = false;
        tio.iflag.ISTRIP = false;
        tio.iflag.INLCR = false;
        tio.iflag.IGNCR = false;
        tio.iflag.ICRNL = false;
        tio.iflag.IXON = false;
        // Raw output.
        tio.oflag.OPOST = false;
        // Raw local modes.
        tio.lflag.ECHO = false;
        tio.lflag.ECHONL = false;
        tio.lflag.ICANON = false;
        tio.lflag.ISIG = false;
        tio.lflag.IEXTEN = false;
        // 8 data bits, no parity, one stop bit, ignore modem lines, enable rx.
        tio.cflag.PARENB = false;
        tio.cflag.CSTOPB = false;
        tio.cflag.CSIZE = .CS8;
        tio.cflag.CLOCAL = true;
        tio.cflag.CREAD = true;
        // 115200 on both directions.
        const speed = try speedOf(baud);
        // Linux keeps the line speed in c_cflag's CBAUD bits. Zig's cflag struct
        // models those as reserved fields, and TCSETS reads CBAUD (not ispeed/
        // ospeed), so without setting them the baud silently stays at the tty
        // default. Clear CBAUD and OR in the speed's bit pattern directly.
        const CBAUD: u32 = 0o010017; // bits 0-3 + CBAUDEX (0x1000)
        var cflag: u32 = @bitCast(tio.cflag);
        cflag = (cflag & ~CBAUD) | @as(u32, @intFromEnum(speed));
        tio.cflag = @bitCast(cflag);
        tio.ispeed = speed;
        tio.ospeed = speed;
        // Return whatever is available; 0.2s inter-byte timeout (short so the
        // drain below is quick; responses arrive in ms, so this never delays a
        // real read).
        tio.cc[@intFromEnum(linux.V.MIN)] = 0;
        tio.cc[@intFromEnum(linux.V.TIME)] = 2;

        try posix.tcsetattr(fd, .NOW, tio);

        // TCIFLUSH clears the tty buffer, but the DirtyJTAG CDC delivers a
        // spurious byte on open that can arrive AFTER the flush; a stale byte
        // byte-misaligns every framed read. Flush, then actively drain until a
        // read comes back empty.
        _ = dev.ioctl(fd, 0x540B, 0) catch {}; // TCFLSH, TCIFLUSH
        var junk: [64]u8 = undefined;
        for (0..8) |_| {
            if ((dev.read(fd, &junk) catch 0) == 0) break;
        }

        return .{ .file = file };
    }

    pub fn close(self: Port, io: Io) void {
        self.file.close(io);
    }

    /// Writes a whole command frame to the serial line.
    pub fn writeFrame(self: Port, bytes: []const u8) !void {
        try dev.writeAll(self.file.handle, bytes);
    }

    /// Reads exactly `buf.len` response bytes or returns error.Timeout.
    pub fn readResponse(self: Port, buf: []u8) !void {
        var off: usize = 0;
        var idle: u32 = 0;
        while (off < buf.len) {
            const n = try dev.read(self.file.handle, buf[off..]);
            if (n == 0) {
                idle += 1;
                if (idle >= 300) return Error.Timeout; // ~60s: a big flash-weight matmul (bit-serial SPI) can hold the bus many seconds; 10s was too short
                continue;
            }
            idle = 0;
            off += n;
        }
    }
};
