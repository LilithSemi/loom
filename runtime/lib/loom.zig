//! Loom runtime library. Drives the Loom accelerator, real silicon over a
//! transport, or the emulator in unit tests, to run W4A8 transformer inference.
//!
//! `runtime/src/main.zig` is the CLI entrypoint and consumes this as
//! `@import("loom")`; this is also the surface future C bindings wrap. Consumers
//! open a backend (transport), get its `Device` interface, load a `model.Config`,
//! and call `forward.generate`.

/// The backend-agnostic accelerator interface (fat pointer + vtable) the model
/// runner drives. A backend exposes a `device()` returning one of these.
pub const device = @import("loom/device.zig");
pub const Device = device.Device;

/// Real-silicon backend (UART/USB) + the wire protocol.
pub const transport = @import("loom/transport.zig");
pub const protocol = @import("loom/protocol.zig");
pub const fp = @import("loom/fp.zig");
pub const quant = @import("loom/quant.zig");

/// Model config/manifest, ops, and the W4A8 forward pass.
pub const model = @import("loom/model.zig");
pub const ops = @import("loom/ops.zig");
pub const linear = @import("loom/linear.zig");
pub const forward = @import("loom/forward.zig");
pub const vision = @import("loom/vision.zig");
pub const image = @import("loom/image.zig");
pub const tokenizer = @import("loom/tokenizer.zig");
pub const bpe = @import("loom/bpe.zig");

/// Host-side OpenAI-compatible serving path.
pub const engine = @import("loom/engine.zig");
pub const serve = @import("loom/serve.zig");
pub const net = @import("loom/net.zig");

// sim.zig (the in-process emulator) is a unit-test backend, NOT part of the
// public/production surface: it is exposed only in test builds so downstream
// tests (e.g. the C bindings) can drive `forward.generate` against the same
// `Device` type without real hardware. In non-test builds this is `void`.
// The in-process emulator, exposed as a reference/testing backend so bindings
// (and downstream forks) can run without hardware. `loom_device_open_sim` wraps
// it; it is not part of the real-silicon datapath.
pub const sim = @import("loom/sim.zig");
