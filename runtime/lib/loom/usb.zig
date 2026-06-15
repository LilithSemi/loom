//! USB transport for the Loom command protocol over the custom vendor device
//! (VID 0x1209, PID 0x10c0, bulk EP1 OUT 0x01 / EP1 IN 0x81).
//!
//! The device is found by scanning /sys/bus/usb/devices with std.Io, then the
//! /dev/bus/usb node is opened with std.Io and driven with usbfs ioctls (the
//! one thing std.Io does not model).

const std = @import("std");
const Io = std.Io;
const dev = @import("dev.zig");

pub const VID: u16 = 0x1209;
pub const PID: u16 = 0x10c0;
pub const EP_OUT: u8 = 0x01;
pub const EP_IN: u8 = 0x81;
const INTERFACE: u32 = 0;
const TIMEOUT_MS: u32 = 2000;
const SYS_DEVICES = "/sys/bus/usb/devices";

pub const Error = error{
    DeviceNotFound,
    ClaimFailed,
    BulkFailed,
    Timeout,
};

// _IOC encoding (asm-generic): (dir<<30)|(size<<16)|(type<<8)|nr.
const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
fn ioc(dir: u32, ty: u32, nr: u32, size: u32) u32 {
    return (dir << 30) | (size << 16) | (ty << 8) | nr;
}

const BulkTransfer = extern struct {
    ep: c_uint,
    len: c_uint,
    timeout: c_uint, // milliseconds
    data: ?*anyopaque,
};

const USBDEVFS_BULK: u32 = ioc(IOC_READ | IOC_WRITE, 'U', 2, @sizeOf(BulkTransfer));
const USBDEVFS_CLAIMINTERFACE: u32 = ioc(IOC_READ, 'U', 15, 4);
const USBDEVFS_RELEASEINTERFACE: u32 = ioc(IOC_READ, 'U', 16, 4);

pub const Device = struct {
    file: Io.File,

    /// Finds and opens the Loom vendor device, claiming its interface.
    pub fn open(io: Io) !Device {
        var path_buf: [64]u8 = undefined;
        const path = try findNode(io, &path_buf);
        const file = try Io.Dir.openFileAbsolute(io, path, .{ .mode = .read_write });
        errdefer file.close(io);

        var intf: u32 = INTERFACE;
        _ = dev.ioctl(file.handle, USBDEVFS_CLAIMINTERFACE, @intFromPtr(&intf)) catch
            return Error.ClaimFailed;
        return .{ .file = file };
    }

    pub fn close(self: Device, io: Io) void {
        var intf: u32 = INTERFACE;
        _ = dev.ioctl(self.file.handle, USBDEVFS_RELEASEINTERFACE, @intFromPtr(&intf)) catch {};
        self.file.close(io);
    }

    fn bulk(self: Device, ep: u8, data: []u8) !usize {
        var bt = BulkTransfer{
            .ep = ep,
            .len = @intCast(data.len),
            .timeout = TIMEOUT_MS,
            .data = if (data.len == 0) null else @ptrCast(data.ptr),
        };
        return dev.ioctl(self.file.handle, USBDEVFS_BULK, @intFromPtr(&bt)) catch |e| switch (e) {
            dev.Error.Timeout => Error.Timeout,
            else => Error.BulkFailed,
        };
    }

    /// Sends a whole command frame in one bulk-OUT transfer.
    pub fn writeFrame(self: Device, bytes: []const u8) !void {
        // usbfs needs a mutable buffer; frames are tiny (<= 11 bytes).
        var tmp: [64]u8 = undefined;
        @memcpy(tmp[0..bytes.len], bytes);
        _ = try self.bulk(EP_OUT, tmp[0..bytes.len]);
    }

    /// Reads exactly `buf.len` response bytes from a single bulk-IN transfer.
    pub fn readResponse(self: Device, buf: []u8) !void {
        const n = try self.bulk(EP_IN, buf);
        if (n != buf.len) return Error.BulkFailed;
    }
};

/// Scans /sys/bus/usb/devices for VID:PID and builds the /dev/bus/usb path.
fn findNode(io: Io, out: []u8) ![]const u8 {
    var sys = Io.Dir.openDirAbsolute(io, SYS_DEVICES, .{ .iterate = true }) catch
        return Error.DeviceNotFound;
    defer sys.close(io);

    var it = sys.iterate();
    while (it.next(io) catch return Error.DeviceNotFound) |entry| {
        if (entry.name.len == 0 or entry.name[0] == '.') continue;
        const vid = readAttrInt(io, sys, u16, entry.name, "idVendor", 16) orelse continue;
        const pid = readAttrInt(io, sys, u16, entry.name, "idProduct", 16) orelse continue;
        if (vid != VID or pid != PID) continue;
        const bus = readAttrInt(io, sys, u32, entry.name, "busnum", 10) orelse continue;
        const num = readAttrInt(io, sys, u32, entry.name, "devnum", 10) orelse continue;
        return std.fmt.bufPrint(out, "/dev/bus/usb/{d:0>3}/{d:0>3}", .{ bus, num });
    }
    return Error.DeviceNotFound;
}

/// Reads <name>/<attr> under the sysfs dir and parses an integer.
fn readAttrInt(io: Io, sys: Io.Dir, comptime T: type, name: []const u8, attr: []const u8, base: u8) ?T {
    var path_buf: [256]u8 = undefined;
    const rel = std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ name, attr }) catch return null;
    var buf: [32]u8 = undefined;
    const contents = sys.readFile(io, rel, &buf) catch return null;
    const s = std.mem.trim(u8, contents, " \r\n\t");
    return std.fmt.parseInt(T, s, base) catch null;
}

test "ioctl encodings match the kernel usbdevice_fs.h values" {
    try std.testing.expectEqual(@as(u32, 24), @sizeOf(BulkTransfer));
    try std.testing.expectEqual(@as(u32, 0xC0185502), USBDEVFS_BULK);
    try std.testing.expectEqual(@as(u32, 0x8004550F), USBDEVFS_CLAIMINTERFACE);
    try std.testing.expectEqual(@as(u32, 0x80045510), USBDEVFS_RELEASEINTERFACE);
}
