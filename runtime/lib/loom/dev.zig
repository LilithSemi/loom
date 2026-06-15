//! Raw device byte-IO and ioctl on an already-open fd.
//!
//! Files, directories and stdout all go through std.Io now; this module only
//! covers what std.Io deliberately does not model: ioctl (usbfs bulk transfers)
//! and termios-timed exact-length serial reads.

const std = @import("std");
const linux = std.os.linux;
const errno = std.posix.errno;

pub const fd_t = std.posix.fd_t;

pub const Error = error{ ReadFailed, WriteFailed, IoctlFailed, Timeout };

pub fn close_fd(fd: fd_t) void {
    _ = linux.close(fd);
}

pub fn read(fd: fd_t, buf: []u8) Error!usize {
    const rc = linux.read(fd, buf.ptr, buf.len);
    return switch (errno(rc)) {
        .SUCCESS => rc,
        else => Error.ReadFailed,
    };
}

pub fn write(fd: fd_t, buf: []const u8) Error!usize {
    const rc = linux.write(fd, buf.ptr, buf.len);
    return switch (errno(rc)) {
        .SUCCESS => rc,
        else => Error.WriteFailed,
    };
}

pub fn writeAll(fd: fd_t, buf: []const u8) Error!void {
    var off: usize = 0;
    while (off < buf.len) off += try write(fd, buf[off..]);
}

/// ioctl with a pointer argument; returns the raw non-negative result.
pub fn ioctl(fd: fd_t, request: u32, arg: usize) Error!usize {
    const rc = linux.ioctl(fd, request, arg);
    return switch (errno(rc)) {
        .SUCCESS => rc,
        .TIMEDOUT => Error.Timeout,
        else => Error.IoctlFailed,
    };
}
