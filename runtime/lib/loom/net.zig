//! Minimal blocking TCP server on raw Linux syscalls (std.net moved/churned in
//! 0.16; raw sockets are stable and self-contained).

const std = @import("std");
const linux = std.os.linux;
const errno = std.posix.errno;
const dev = @import("dev.zig");

pub const fd_t = std.posix.fd_t;
pub const Error = error{ Socket, SetSockOpt, Bind, Listen, Accept };

/// Opens a listening TCP socket bound to 127.0.0.1:port.
pub fn listen(port: u16) Error!fd_t {
    const sock = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.CLOEXEC, 0);
    if (errno(sock) != .SUCCESS) return Error.Socket;
    const fd: fd_t = @intCast(sock);

    const one: u32 = 1;
    const so = linux.setsockopt(fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, std.mem.asBytes(&one), 4);
    if (errno(so) != .SUCCESS) return Error.SetSockOpt;

    var addr = linux.sockaddr.in{
        .port = std.mem.nativeToBig(u16, port),
        .addr = std.mem.nativeToBig(u32, 0x7f000001), // 127.0.0.1
    };
    const b = linux.bind(fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in));
    if (errno(b) != .SUCCESS) return Error.Bind;

    const l = linux.listen(fd, 16);
    if (errno(l) != .SUCCESS) return Error.Listen;
    return fd;
}

/// Blocks until a client connects; returns the connection fd.
pub fn accept(listen_fd: fd_t) Error!fd_t {
    const c = linux.accept4(listen_fd, null, null, linux.SOCK.CLOEXEC);
    if (errno(c) != .SUCCESS) return Error.Accept;
    return @intCast(c);
}

pub const close = dev.close_fd;

/// Reads up to buf.len bytes; returns slice actually read (0 on peer close).
pub fn read(fd: fd_t, buf: []u8) !usize {
    return dev.read(fd, buf);
}

pub fn writeAll(fd: fd_t, buf: []const u8) !void {
    return dev.writeAll(fd, buf);
}
