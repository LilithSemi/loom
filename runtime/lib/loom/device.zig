//! The Loom device interface: a fat pointer (context + vtable) that the model
//! runner (linear.zig / forward.zig) drives, without knowing whether it is
//! talking to real silicon over a transport or the in-process emulator. This is
//! what makes `runtime/lib/loom.zig` consumable and, later, C-bindable: the
//! runner depends on this concrete type, not an `anytype` generic.
//!
//! An implementation exposes a `device()` method returning one of these, wiring
//! its own methods into the vtable (see transport.zig / sim.zig).

/// The CSR/stream operations every Loom accelerator backend provides. All are
/// fallible (transport timeouts, sim allocation), hence `anyerror`.
pub const Device = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Write one 32-bit word to a CSR.
        reg_write: *const fn (ptr: *anyopaque, addr: u32, value: u32) anyerror!void,
        /// Read one 32-bit word from a CSR.
        reg_read: *const fn (ptr: *anyopaque, addr: u32) anyerror!u32,
        /// Write many {addr, value} pairs (the backend may coalesce them).
        write_regs: *const fn (ptr: *anyopaque, pairs: []const [2]u32) anyerror!void,
        /// Read `words.len` consecutive words starting at `addr` (a burst).
        read_regs: *const fn (ptr: *anyopaque, addr: u32, words: []u32) anyerror!void,
        /// Stream `values` as pushes to the fixed FIFO `addr`.
        write_stream: *const fn (ptr: *anyopaque, addr: u32, values: []const u32) anyerror!void,
        /// Burst-write `bytes` to INCREMENTING addresses from `addr` (loads an
        /// on-chip memory, e.g. the BRAM weight cache). `bytes.len` multiple of 4.
        write_mem: *const fn (ptr: *anyopaque, addr: u32, bytes: []const u8) anyerror!void,
    };

    pub fn regWrite(self: Device, addr: u32, value: u32) anyerror!void {
        return self.vtable.reg_write(self.ptr, addr, value);
    }
    pub fn regRead(self: Device, addr: u32) anyerror!u32 {
        return self.vtable.reg_read(self.ptr, addr);
    }
    pub fn writeRegs(self: Device, pairs: []const [2]u32) anyerror!void {
        return self.vtable.write_regs(self.ptr, pairs);
    }
    pub fn readRegs(self: Device, addr: u32, words: []u32) anyerror!void {
        return self.vtable.read_regs(self.ptr, addr, words);
    }
    pub fn writeStream(self: Device, addr: u32, values: []const u32) anyerror!void {
        return self.vtable.write_stream(self.ptr, addr, values);
    }
    pub fn writeMem(self: Device, addr: u32, bytes: []const u8) anyerror!void {
        return self.vtable.write_mem(self.ptr, addr, bytes);
    }
};
