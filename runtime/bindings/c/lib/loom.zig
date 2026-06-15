//! C ABI over the Loom runtime (`@import("loom")`). Every fallible entrypoint
//! returns a `loom_status` int and fills a caller-owned `CError` (no global
//! error state, so callers can drive many instances concurrently). Handles are
//! opaque to C; here they are real Zig structs whose pointers cross the boundary
//! as the opaque `loom_runtime_t` / `loom_device_t` / `loom_model_t`.
//!
//! `runtime/src/main.zig` is the reference for the underlying flow (open a
//! transport, load a model, generate); this wraps that same path behind a C ABI.

const std = @import("std");
const loom = @import("loom");

/// Threadsafe, no libc dependency: fits a static/shared lib that may be linked
/// into a non-Zig host without pulling in a C allocator.
const gpa = std.heap.smp_allocator;

/// Mirrors `loom_status_t` in loom.h. Values are the C ABI contract.
const Status = enum(c_int) {
    ok = 0,
    err_alloc = -1,
    err_transport = -2,
    err_model_load = -3,
    err_generate = -4,
    err_invalid = -5,
    err_context = -6,
    err_eval = -7,
};

/// Mirrors `loom_error_t`: a caller-owned struct filled on failure.
const CError = extern struct {
    code: c_int,
    message: [256]u8,
};

/// Fills `err` (when non-null) with `st` and a NUL-terminated copy of `name`,
/// then returns `st` as the C status code.
fn fail(err: ?*CError, st: Status, name: []const u8) c_int {
    if (err) |e| {
        e.code = @intFromEnum(st);
        const n = @min(name.len, e.message.len - 1);
        @memcpy(e.message[0..n], name[0..n]);
        e.message[n] = 0;
    }
    return @intFromEnum(st);
}

/// Clears `err` (when non-null) and returns the ok status code.
fn ok(err: ?*CError) c_int {
    if (err) |e| {
        e.code = 0;
        e.message[0] = 0;
    }
    return @intFromEnum(Status.ok);
}

/// Root handle: owns the `std.Io` every device and model borrows. Heap-stored so
/// `threaded.io()` (which captures `*Threaded`) stays valid for the handle's
/// life. This is where the runtime obtains an `Io` without a `std.process.Init`.
const Runtime = struct {
    threaded: std.Io.Threaded,

    fn io(self: *Runtime) std.Io {
        return self.threaded.io();
    }
};

export fn loom_runtime_create(out: ?**Runtime, err: ?*CError) c_int {
    const o = out orelse return fail(err, .err_invalid, "null out");
    const rt = gpa.create(Runtime) catch return fail(err, .err_alloc, "OutOfMemory");
    rt.threaded = std.Io.Threaded.init(gpa, .{});
    o.* = rt;
    return ok(err);
}

export fn loom_runtime_destroy(rt: ?*Runtime) void {
    const r = rt orelse return;
    r.threaded.deinit();
    gpa.destroy(r);
}

export fn loom_version() [*:0]const u8 {
    return "loom-rt 0.1.0";
}

// ---- Host-side ops (no device): thin wrappers over ops.zig ----

export fn loom_rmsnorm(x: ?[*]const f32, gamma: ?[*]const f32, eps: f32, out: ?[*]f32, n: usize) void {
    const xp = x orelse return;
    const gp = gamma orelse return;
    const op = out orelse return;
    loom.ops.rmsNorm(xp[0..n], gp[0..n], eps, op[0..n]);
}

export fn loom_silu(x: ?[*]f32, n: usize) void {
    const xp = x orelse return;
    loom.ops.siluInPlace(xp[0..n]);
}

/// LayerNorm (Vision Transformer towers): out = (x-mean)/sqrt(var+eps)*gamma+beta.
export fn loom_layernorm(x: ?[*]const f32, gamma: ?[*]const f32, beta: ?[*]const f32, eps: f32, out: ?[*]f32, n: usize) void {
    const xp = x orelse return;
    const gp = gamma orelse return;
    const bp = beta orelse return;
    const op = out orelse return;
    loom.ops.layerNorm(xp[0..n], gp[0..n], bp[0..n], eps, op[0..n]);
}

/// GELU in place (tanh approximation), the ViT MLP activation.
export fn loom_gelu(x: ?[*]f32, n: usize) void {
    const xp = x orelse return;
    loom.ops.geluInPlace(xp[0..n]);
}

export fn loom_softmax(x: ?[*]f32, n: usize) void {
    const xp = x orelse return;
    loom.ops.softmaxInPlace(xp[0..n]);
}

export fn loom_rope(head: ?[*]f32, pos: usize, theta: f32, head_dim: usize) void {
    const hp = head orelse return;
    loom.ops.ropeHead(hp[0..head_dim], pos, theta);
}

/// Elementwise in-place add `x[i] += bias[i]` (for q/k/v bias in a composed
/// forward, e.g. Qwen2). The whole-model `loom_generate`/`loom_eval` path applies
/// bias internally, this is for callers assembling a forward from the host ops.
export fn loom_add(x: ?[*]f32, bias: ?[*]const f32, n: usize) void {
    const xp = x orelse return;
    const bp = bias orelse return;
    loom.ops.addInPlace(xp[0..n], bp[0..n]);
}

/// Mixture-of-Experts router selection: softmax `logits[0..num_experts]`, pick
/// the `top_k` highest, and write their indices to `idx[0..top_k]` and combine
/// weights to `w[0..top_k]` (renormalized to sum 1 when `norm_topk`). Returns
/// the number of experts selected (`min(top_k, num_experts)`). For callers
/// composing a MoE forward from the host ops + `loom_linear`; the whole-model
/// `loom_eval` path routes internally. `idx`/`w` must hold `top_k` elements.
export fn loom_moe_route(logits: ?[*]const f32, num_experts: usize, top_k: usize, norm_topk: bool, idx: ?[*]usize, w: ?[*]f32) usize {
    const lp = logits orelse return 0;
    const ip = idx orelse return 0;
    const wp = w orelse return 0;
    const k = @min(top_k, num_experts);
    return loom.ops.moeRoute(lp[0..num_experts], top_k, norm_topk, ip[0..k], wp[0..k]);
}

// ---- Tokenizer over the model's vocab ----

export fn loom_tokenize(model: ?*Model, text: ?[*:0]const u8, add_bos: bool, out_ids: ?[*]u32, cap: usize, n: ?*usize, err: ?*CError) c_int {
    const m = model orelse return fail(err, .err_invalid, "null model");
    const t = std.mem.span(text orelse return fail(err, .err_invalid, "null text"));
    const op = out_ids orelse return fail(err, .err_invalid, "null out_ids");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const ids = m.tok.encode(arena.allocator(), t, add_bos) catch |e|
        return fail(err, .err_generate, @errorName(e));
    if (ids.len > cap) return fail(err, .err_invalid, "out_ids too small");
    @memcpy(op[0..ids.len], ids);
    if (n) |np| np.* = ids.len;
    return ok(err);
}

export fn loom_detokenize(model: ?*Model, ids: ?[*]const u32, n: usize, buf: ?[*]u8, cap: usize, len: ?*usize, err: ?*CError) c_int {
    const m = model orelse return fail(err, .err_invalid, "null model");
    const ip = ids orelse return fail(err, .err_invalid, "null ids");
    const bp = buf orelse return fail(err, .err_invalid, "null buf");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const text = m.tok.decode(ip[0..n], arena.allocator()) catch |e|
        return fail(err, .err_generate, @errorName(e));
    if (text.len > cap) return fail(err, .err_invalid, "buf too small");
    @memcpy(bp[0..text.len], text);
    if (len) |lp| lp.* = text.len;
    return ok(err);
}

/// A sim emulator plus the flash byte buffers it reads, owned together so the
/// device can free them as a unit. `sim.device()` borrows these, so they must
/// outlive the device.
const SimBackend = struct {
    sim: loom.sim.Sim,
    weights: []u8,
    scales_flash: []u8,
};

/// An open accelerator backend. We own the heap `transport.Device` (so its
/// `.device()` fat-pointer stays stable across calls) or an owned `sim`.
const Device = struct {
    rt: *Runtime,
    backend: union(enum) {
        transport: *loom.transport.Device,
        sim: *SimBackend,
    },
    iface: loom.Device,
};

export fn loom_device_open_uart(rt: ?*Runtime, port: ?[*:0]const u8, baud: u32, out: ?**Device, err: ?*CError) c_int {
    const r = rt orelse return fail(err, .err_invalid, "null runtime");
    const o = out orelse return fail(err, .err_invalid, "null out");
    const p = std.mem.span(port orelse return fail(err, .err_invalid, "null port"));
    const t = gpa.create(loom.transport.Device) catch return fail(err, .err_alloc, "OutOfMemory");
    errdefer gpa.destroy(t);
    t.* = loom.transport.Device.openUart(r.io(), p, baud) catch |e|
        return fail(err, .err_transport, @errorName(e));
    const d = gpa.create(Device) catch return fail(err, .err_alloc, "OutOfMemory");
    d.* = .{ .rt = r, .backend = .{ .transport = t }, .iface = t.device() };
    o.* = d;
    return ok(err);
}

export fn loom_device_open_usb(rt: ?*Runtime, out: ?**Device, err: ?*CError) c_int {
    const r = rt orelse return fail(err, .err_invalid, "null runtime");
    const o = out orelse return fail(err, .err_invalid, "null out");
    const t = gpa.create(loom.transport.Device) catch return fail(err, .err_alloc, "OutOfMemory");
    errdefer gpa.destroy(t);
    t.* = loom.transport.Device.openUsb(r.io()) catch |e|
        return fail(err, .err_transport, @errorName(e));
    const d = gpa.create(Device) catch return fail(err, .err_alloc, "OutOfMemory");
    d.* = .{ .rt = r, .backend = .{ .transport = t }, .iface = t.device() };
    o.* = d;
    return ok(err);
}

/// Reference/testing backend: builds the in-process emulator from a genip dir's
/// flash images (weights.bin + scales_flash.bin) so eval/linear/generate run with
/// no hardware. The device owns the Sim and both byte buffers.
export fn loom_device_open_sim(rt: ?*Runtime, model: ?*Model, dir: ?[*:0]const u8, out: ?**Device, err: ?*CError) c_int {
    const r = rt orelse return fail(err, .err_invalid, "null runtime");
    const m = model orelse return fail(err, .err_invalid, "null model");
    const o = out orelse return fail(err, .err_invalid, "null out");
    const dpath = std.mem.span(dir orelse return fail(err, .err_invalid, "null dir"));
    const io = r.io();

    var md = loom.model.openModelDir(io, dpath) catch |e| return fail(err, .err_transport, @errorName(e));
    defer md.close(io);
    const cap = std.Io.Limit.limited(256 * 1024 * 1024);
    const weights = md.readFileAlloc(io, "weights.bin", gpa, cap) catch |e| return fail(err, .err_transport, @errorName(e));
    errdefer gpa.free(weights);
    const scales_flash = md.readFileAlloc(io, "scales_flash.bin", gpa, cap) catch |e| return fail(err, .err_transport, @errorName(e));
    errdefer gpa.free(scales_flash);

    const mf = m.cfg.manifest();
    const sb = gpa.create(SimBackend) catch return fail(err, .err_alloc, "OutOfMemory");
    errdefer gpa.destroy(sb);
    sb.* = .{
        .sim = loom.sim.Sim.init(gpa, mf.csr_base, mf.flash_weight_base, weights),
        .weights = weights,
        .scales_flash = scales_flash,
    };
    sb.sim.scale_flash_base = mf.scale_flash_base;
    sb.sim.scales_flash = sb.scales_flash;

    const d = gpa.create(Device) catch return fail(err, .err_alloc, "OutOfMemory");
    d.* = .{ .rt = r, .backend = .{ .sim = sb }, .iface = sb.sim.device() };
    o.* = d;
    return ok(err);
}

export fn loom_device_close(dev: ?*Device) void {
    const d = dev orelse return;
    switch (d.backend) {
        .transport => |t| {
            t.close();
            gpa.destroy(t);
        },
        .sim => |sb| {
            sb.sim.deinit();
            gpa.free(sb.weights);
            gpa.free(sb.scales_flash);
            gpa.destroy(sb);
        },
    }
    gpa.destroy(d);
}

/// A loaded genip model: the manifest/scales/glue (`Config`) plus the tokenizer,
/// borrowing the runtime for its `io` at generate time.
const Model = struct {
    rt: *Runtime,
    cfg: loom.model.Config,
    tok: loom.tokenizer.Any,
    /// The genip output dir, kept so `loom_device_prepare` can load the on-chip
    /// BRAM weight cache (weights_bram.bin/scales_bram.bin) onto a real device.
    dir_path: []const u8,
};

/// Loads loom.json + scales.bin + glue.bin (via `Config.load`) and tokenizer.bin
/// (BOS=1/EOS=2, matching the CLI) from a genip output directory. Any failure
/// returns err_model_load carrying the Zig error name.
export fn loom_model_load(rt: ?*Runtime, dir: ?[*:0]const u8, out: ?**Model, err: ?*CError) c_int {
    const r = rt orelse return fail(err, .err_invalid, "null runtime");
    const o = out orelse return fail(err, .err_invalid, "null out");
    const dpath = std.mem.span(dir orelse return fail(err, .err_invalid, "null dir"));
    const io = r.io();

    var cfg = loom.model.Config.load(gpa, io, dpath) catch |e|
        return fail(err, .err_model_load, @errorName(e));
    errdefer cfg.deinit();

    var d = loom.model.openModelDir(io, dpath) catch |e|
        return fail(err, .err_model_load, @errorName(e));
    defer d.close(io);
    const tok_bytes = d.readFileAlloc(io, "tokenizer.bin", gpa, std.Io.Limit.limited(4 * 1024 * 1024)) catch |e|
        return fail(err, .err_model_load, @errorName(e));
    defer gpa.free(tok_bytes);
    var tok = loom.tokenizer.Any.parse(gpa, tok_bytes, 1, 2) catch |e|
        return fail(err, .err_model_load, @errorName(e));
    errdefer tok.deinit();

    const dir_copy = gpa.dupe(u8, dpath) catch return fail(err, .err_alloc, "OutOfMemory");
    errdefer gpa.free(dir_copy);

    const m = gpa.create(Model) catch return fail(err, .err_alloc, "OutOfMemory");
    m.* = .{ .rt = r, .cfg = cfg, .tok = tok, .dir_path = dir_copy };
    o.* = m;
    return ok(err);
}

export fn loom_model_free(model: ?*Model) void {
    const m = model orelse return;
    gpa.free(m.dir_path);
    m.tok.deinit();
    m.cfg.deinit();
    gpa.destroy(m);
}

/// Loads a BRAM-weight-cache model's hot weights onto a real device once, before
/// generating (mirrors the CLI's startup step). A NO-OP for models built without
/// `--fp-bram-cache-kb` (bram_cache_kb == 0) and for the sim device, so it is
/// always safe to call. Without it, a cache model's hot matmuls read unloaded
/// on-chip BRAM. Call after opening the device, before loom_eval/loom_generate.
export fn loom_device_prepare(model: ?*Model, dev: ?*Device, err: ?*CError) c_int {
    const m = model orelse return fail(err, .err_invalid, "null model");
    const d = dev orelse return fail(err, .err_invalid, "null device");
    const io = m.rt.io();
    var md = loom.model.openModelDir(io, m.dir_path) catch |e|
        return fail(err, .err_model_load, @errorName(e));
    defer md.close(io);
    loom.forward.loadBramCache(gpa, io, d.iface, m.cfg, md) catch |e|
        return fail(err, .err_transport, @errorName(e));
    return ok(err);
}

/// Mirrors `loom_model_info_t`.
const CModelInfo = extern struct {
    hidden: u32,
    vocab: u32,
    layers: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    max_seq: u32,
    n_matrices: u32,
    rope_theta: f32,
    norm_eps: f32,
};

export fn loom_model_info(model: ?*Model, out: ?*CModelInfo) c_int {
    const m = model orelse return @intFromEnum(Status.err_invalid);
    const o = out orelse return @intFromEnum(Status.err_invalid);
    const mf = m.cfg.manifest();
    o.* = .{
        .hidden = mf.hidden,
        .vocab = mf.vocab,
        .layers = mf.layers,
        .num_heads = mf.num_heads,
        .num_kv_heads = mf.num_kv_heads,
        .head_dim = mf.head_dim,
        .intermediate = mf.intermediate,
        .max_seq = mf.max_seq,
        .n_matrices = @intCast(mf.matrices.len),
        .rope_theta = mf.rope_theta,
        .norm_eps = mf.norm_eps,
    };
    return @intFromEnum(Status.ok);
}

export fn loom_model_num_matrices(model: ?*Model) usize {
    const m = model orelse return 0;
    return m.cfg.manifest().matrices.len;
}

// A borrowed pointer into the model's matrix table crosses as the opaque
// `const loom_matrix_t*`. Valid until loom_model_free (cfg owns the table).
export fn loom_model_matrix(model: ?*Model, index: usize, out: ?**const loom.model.Matrix, err: ?*CError) c_int {
    const m = model orelse return fail(err, .err_invalid, "null model");
    const o = out orelse return fail(err, .err_invalid, "null out");
    const mats = m.cfg.manifest().matrices;
    if (index >= mats.len) return fail(err, .err_invalid, "index out of range");
    o.* = &mats[index];
    return ok(err);
}

export fn loom_model_matrix_by_name(model: ?*Model, name: ?[*:0]const u8, out: ?**const loom.model.Matrix, err: ?*CError) c_int {
    const m = model orelse return fail(err, .err_invalid, "null model");
    const o = out orelse return fail(err, .err_invalid, "null out");
    const nm = std.mem.span(name orelse return fail(err, .err_invalid, "null name"));
    for (m.cfg.manifest().matrices) |*mat| {
        if (std.mem.eql(u8, mat.name, nm)) {
            o.* = mat;
            return ok(err);
        }
    }
    return fail(err, .err_invalid, "no such matrix");
}

export fn loom_matrix_dims(mat: ?*const loom.model.Matrix, rows: ?*u32, cols: ?*u32) void {
    const mt = mat orelse return;
    if (rows) |r| r.* = mt.rows;
    if (cols) |c| c.* = mt.cols;
}

/// One flash-resident matmul: `out[mat.rows] = mat @ x[mat.cols]`. The device
/// reads the int4 weights + scales from flash; `x`/`out` are host fp32.
export fn loom_linear(model: ?*Model, dev: ?*Device, mat: ?*const loom.model.Matrix, x: ?[*]const f32, out: ?[*]f32, poll_timeout: usize, err: ?*CError) c_int {
    const m = model orelse return fail(err, .err_invalid, "null model");
    const d = dev orelse return fail(err, .err_invalid, "null device");
    const mt = mat orelse return fail(err, .err_invalid, "null matrix");
    const xp = x orelse return fail(err, .err_invalid, "null x");
    const op = out orelse return fail(err, .err_invalid, "null out");
    const mf = m.cfg.manifest();
    const timeout = if (poll_timeout == 0) loom.linear.DEFAULT_TIMEOUT else poll_timeout;
    loom.linear.linear(m.rt.io(), d.iface, mf.csr_base, mf.stores(), timeout, mt.*, xp[0..mt.cols], op[0..mt.rows]) catch |e|
        return fail(err, .err_generate, @errorName(e));
    return ok(err);
}

/// Column-tiled matmul for a matrix WIDER than the accelerator's column capacity
/// (`block_cols` = the device maxCols). Runs the wide matmul as consecutive
/// col-blocks and sums the partials, so a small accelerator handles any width via
/// more invocations. The row-scratch buffer is managed internally.
export fn loom_linear_col_tiled(model: ?*Model, dev: ?*Device, mat: ?*const loom.model.Matrix, block_cols: usize, x: ?[*]const f32, out: ?[*]f32, poll_timeout: usize, err: ?*CError) c_int {
    const m = model orelse return fail(err, .err_invalid, "null model");
    const d = dev orelse return fail(err, .err_invalid, "null device");
    const mt = mat orelse return fail(err, .err_invalid, "null matrix");
    const xp = x orelse return fail(err, .err_invalid, "null x");
    const op = out orelse return fail(err, .err_invalid, "null out");
    if (block_cols == 0) return fail(err, .err_invalid, "block_cols must be > 0");
    const partial = gpa.alloc(f32, mt.rows) catch return fail(err, .err_alloc, "OutOfMemory");
    defer gpa.free(partial);
    const mf = m.cfg.manifest();
    const timeout = if (poll_timeout == 0) loom.linear.DEFAULT_TIMEOUT else poll_timeout;
    loom.linear.linearColTiled(m.rt.io(), d.iface, mf.csr_base, mf.stores(), timeout, mt.*, block_cols, xp[0..mt.cols], op[0..mt.rows], partial) catch |e|
        return fail(err, .err_generate, @errorName(e));
    return ok(err);
}

/// Stateful decode context: owns a persistent KV cache (`forward.State`) sized
/// `n_ctx`, borrowing the model for its config + io.
const Context = struct {
    model: *Model,
    state: loom.forward.State,
};

export fn loom_context_create(model: ?*Model, n_ctx: usize, out: ?**Context, err: ?*CError) c_int {
    const m = model orelse return fail(err, .err_invalid, "null model");
    const o = out orelse return fail(err, .err_invalid, "null out");
    if (n_ctx == 0) return fail(err, .err_invalid, "n_ctx must be > 0");
    var state = loom.forward.State.init(gpa, m.cfg, n_ctx) catch |e|
        return fail(err, .err_context, @errorName(e));
    const c = gpa.create(Context) catch {
        state.deinit();
        return fail(err, .err_alloc, "OutOfMemory");
    };
    c.* = .{ .model = m, .state = state };
    o.* = c;
    return ok(err);
}

export fn loom_context_free(ctx: ?*Context) void {
    const c = ctx orelse return;
    c.state.deinit();
    gpa.destroy(c);
}

/// Steps `n` tokens from `pos`, updating the KV cache, and writes `vocab` logits
/// for the LAST token into `logits_out`. Caller samples and feeds the next token.
export fn loom_eval(ctx: ?*Context, dev: ?*Device, tokens: ?[*]const u32, n: usize, pos: usize, logits_out: ?[*]f32, err: ?*CError) c_int {
    const c = ctx orelse return fail(err, .err_invalid, "null ctx");
    const d = dev orelse return fail(err, .err_invalid, "null device");
    const toks = tokens orelse return fail(err, .err_invalid, "null tokens");
    const lo = logits_out orelse return fail(err, .err_invalid, "null logits_out");
    if (n == 0) return fail(err, .err_invalid, "n must be > 0");
    if (pos + n > c.state.n_ctx) return fail(err, .err_eval, "sequence exceeds n_ctx");
    const io = c.model.rt.io();
    const vocab = c.model.cfg.manifest().vocab;

    for (0..n) |i| {
        const want = i + 1 == n;
        const out_slice: ?[]f32 = if (want) lo[0..vocab] else null;
        c.state.step(io, d.iface, c.model.cfg, loom.linear.DEFAULT_TIMEOUT, toks[i], pos + i, out_slice) catch |e|
            return fail(err, .err_eval, @errorName(e));
    }
    return ok(err);
}

/// Multi-Token Prediction draft: prefills `tokens[0..n]`, then chains the MTP
/// heads to draft `num_modules + 1` tokens (the main next token plus one per
/// module) into `draft_out`. Writes the draft length to `k_out`. `draft_out`
/// must hold at least `num_modules + 1` ids. Fails if the model has no MTP heads.
/// This is the device-side draft primitive; the host runs the verify + accept
/// step (a single loom_eval over the drafted positions) to keep output greedy.
export fn loom_mtp_draft(ctx: ?*Context, dev: ?*Device, tokens: ?[*]const u32, n: usize, draft_out: ?[*]u32, k_out: ?*usize, err: ?*CError) c_int {
    const c = ctx orelse return fail(err, .err_invalid, "null ctx");
    const d = dev orelse return fail(err, .err_invalid, "null device");
    const toks = tokens orelse return fail(err, .err_invalid, "null tokens");
    const dout = draft_out orelse return fail(err, .err_invalid, "null draft_out");
    if (n == 0) return fail(err, .err_invalid, "n must be > 0");
    if (n > c.state.n_ctx) return fail(err, .err_eval, "sequence exceeds n_ctx");
    const mf = c.model.cfg.manifest();
    const mtp = mf.mtp orelse return fail(err, .err_eval, "model has no MTP heads");
    const io = c.model.rt.io();
    const k = c.state.mtpDraft(io, d.iface, c.model.cfg, loom.linear.DEFAULT_TIMEOUT, toks[0..n], dout[0 .. mtp.num_modules + 1]) catch |e|
        return fail(err, .err_eval, @errorName(e));
    if (k_out) |ko| ko.* = k;
    return ok(err);
}

/// Runs the vision tower + projector on `pixels` (`num_channels*image_size^2`,
/// channels-first, already resized + normalized), writing `seq_len*output_dim`
/// projected text-space embeddings into `out`. The needed dims are reported via
/// `seq_out`/`dim_out`; query them by inspecting the model's vision manifest, or
/// pass a generously sized `out` (out_cap guards it). The host then splices these
/// embeddings at the image placeholder token and runs loom_eval. Every matmul
/// runs on `dev`. Fails if the model has no vision tower / projector.
export fn loom_encode_image(model: ?*Model, dev: ?*Device, pixels: ?[*]const f32, npix: usize, out: ?[*]f32, out_cap: usize, seq_out: ?*usize, dim_out: ?*usize, err: ?*CError) c_int {
    const mo = model orelse return fail(err, .err_invalid, "null model");
    const d = dev orelse return fail(err, .err_invalid, "null device");
    const px = pixels orelse return fail(err, .err_invalid, "null pixels");
    const op = out orelse return fail(err, .err_invalid, "null out");
    const mf = mo.cfg.manifest();
    const v = mf.vision orelse return fail(err, .err_eval, "model has no vision tower");
    const proj = mf.projector orelse return fail(err, .err_eval, "model has no projector");
    const seq: usize = projectedTokens(v, proj);
    const dim: usize = proj.output_dim;
    if (npix != v.num_channels * v.image_size * v.image_size) return fail(err, .err_invalid, "wrong pixel count");
    if (out_cap < seq * dim) return fail(err, .err_invalid, "out too small for projected image tokens");
    loom.vision.encodeImage(gpa, mo.rt.io(), d.iface, mo.cfg, loom.linear.DEFAULT_TIMEOUT, px[0..npix], op[0 .. seq * dim]) catch |e|
        return fail(err, .err_eval, @errorName(e));
    if (seq_out) |s| s.* = seq;
    if (dim_out) |dd| dd.* = dim;
    return ok(err);
}

/// Number of projected image tokens: `vision.seq_len / scale^2` for an Idefics3
/// pixel-shuffle connector, else `vision.seq_len` (LLaVA).
fn projectedTokens(v: loom.model.Vision, proj: loom.model.Projector) usize {
    const s = proj.scale_factor;
    return if (s > 1) v.seq_len / (s * s) else v.seq_len;
}

/// One-shot "image file -> embeddings": decodes a baseline JPEG (`jpeg[0..n]`),
/// resizes + normalizes it per the model's vision config, runs the ViT tower +
/// projector on `dev`, and writes `seq_len*output_dim` text-space embeddings into
/// `out`. The full front door for feeding a photo to the VLM. Reports dims via
/// seq_out/dim_out. Fails if the model has no vision tower or the JPEG is
/// unsupported (progressive/arithmetic).
export fn loom_load_image(model: ?*Model, dev: ?*Device, jpeg: ?[*]const u8, n: usize, out: ?[*]f32, out_cap: usize, seq_out: ?*usize, dim_out: ?*usize, err: ?*CError) c_int {
    const mo = model orelse return fail(err, .err_invalid, "null model");
    const d = dev orelse return fail(err, .err_invalid, "null device");
    const jp = jpeg orelse return fail(err, .err_invalid, "null jpeg");
    const op = out orelse return fail(err, .err_invalid, "null out");
    const mf = mo.cfg.manifest();
    const v = mf.vision orelse return fail(err, .err_eval, "model has no vision tower");
    const proj = mf.projector orelse return fail(err, .err_eval, "model has no projector");
    const seq: usize = projectedTokens(v, proj);
    const dim: usize = proj.output_dim;
    if (out_cap < seq * dim) return fail(err, .err_invalid, "out too small for projected image tokens");
    const io = mo.rt.io();

    var img = loom.image.decodeJpeg(gpa, jp[0..n]) catch |e|
        return fail(err, .err_eval, @errorName(e));
    defer img.deinit(gpa);

    const size: usize = v.image_size;
    const pixels = gpa.alloc(f32, 3 * size * size) catch return fail(err, .err_alloc, "OutOfMemory");
    defer gpa.free(pixels);
    loom.image.preprocess(gpa, img.rgb, img.width, img.height, size, v.image_mean, v.image_std, pixels) catch |e|
        return fail(err, .err_eval, @errorName(e));

    loom.vision.encodeImage(gpa, io, d.iface, mo.cfg, loom.linear.DEFAULT_TIMEOUT, pixels, op[0 .. seq * dim]) catch |e|
        return fail(err, .err_eval, @errorName(e));
    if (seq_out) |s| s.* = seq;
    if (dim_out) |dd| dd.* = dim;
    return ok(err);
}

/// Vision-language eval: steps `n` tokens from `pos`, but each token equal to the
/// model's image_token_index consumes the next row of `vision_embeds`
/// (`num_vision * hidden`, from loom_encode_image) as its input embedding instead
/// of a token-embedding lookup. Writes the last position's `vocab` logits into
/// `logits_out`. The placeholder count must equal `num_vision`.
export fn loom_eval_vlm(ctx: ?*Context, dev: ?*Device, tokens: ?[*]const u32, n: usize, pos: usize, vision_embeds: ?[*]const f32, num_vision: usize, logits_out: ?[*]f32, err: ?*CError) c_int {
    const c = ctx orelse return fail(err, .err_invalid, "null ctx");
    const d = dev orelse return fail(err, .err_invalid, "null device");
    const toks = tokens orelse return fail(err, .err_invalid, "null tokens");
    const lo = logits_out orelse return fail(err, .err_invalid, "null logits_out");
    if (n == 0) return fail(err, .err_invalid, "n must be > 0");
    if (pos + n > c.state.n_ctx) return fail(err, .err_eval, "sequence exceeds n_ctx");
    const m = c.model.cfg.manifest();
    const img_tok = m.image_token_index orelse return fail(err, .err_eval, "model has no image_token_index");
    if (num_vision > 0 and vision_embeds == null) return fail(err, .err_invalid, "null vision_embeds");
    const hidden = c.state.hidden_n;
    const io = c.model.rt.io();
    const vocab = m.vocab;

    var vnext: usize = 0;
    for (0..n) |i| {
        const out_slice: ?[]f32 = if (i + 1 == n) lo[0..vocab] else null;
        if (toks[i] == img_tok) {
            if (vnext >= num_vision) return fail(err, .err_eval, "more image placeholders than vision embeds");
            const emb = vision_embeds.?[vnext * hidden ..][0..hidden];
            vnext += 1;
            c.state.stepEmbed(io, d.iface, c.model.cfg, loom.linear.DEFAULT_TIMEOUT, emb, pos + i, out_slice) catch |e|
                return fail(err, .err_eval, @errorName(e));
        } else {
            c.state.step(io, d.iface, c.model.cfg, loom.linear.DEFAULT_TIMEOUT, toks[i], pos + i, out_slice) catch |e|
                return fail(err, .err_eval, @errorName(e));
        }
    }
    if (vnext != num_vision) return fail(err, .err_eval, "unused vision embeds (placeholder count mismatch)");
    return ok(err);
}

const TokenCb = ?*const fn (id: u32, utf8: [*]const u8, len: usize, ud: ?*anyopaque) callconv(.c) void;

/// Mirrors `loom_generate_opts_t`.
const CGenOpts = extern struct {
    prompt: ?[*:0]const u8,
    max_tokens: usize,
    poll_timeout: usize,
    /// Loop-stop window (0 selects the default). Ends generation when the last
    /// N output tokens recur earlier; see forward.generate.
    repeat_window: usize,
};

/// Fallback backstop only for manifests missing max_seq; the normal cap is the
/// model's trained context and generation stops on the model's stop token.
const DEFAULT_MAX_TOKENS: usize = 256;
/// Default loop-stop window when CGenOpts.repeat_window is 0 (mirrors the CLI).
const DEFAULT_REPEAT_WINDOW: usize = 8;

/// Adapts the C callback into the `sink.emit(id)` shape `forward.generate`
/// drives. Renders each token to its UTF-8 piece (tracking `prev` for the BOS
/// leading-space rule), holds back an incomplete trailing UTF-8 sequence in
/// `pending` (a byte-level BPE token can be a fragment of a multibyte
/// codepoint), and hands the callback only the complete-prefix bytes. Mirrors
/// `main.zig`'s `StreamSink`.
const CSink = struct {
    tok: loom.tokenizer.Any,
    a: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    pending: *std.ArrayList(u8),
    prev: u32,
    cb: TokenCb,
    ud: ?*anyopaque,

    pub fn emit(self: *CSink, t: u32) !void {
        self.buf.clearRetainingCapacity();
        try self.tok.render(self.a, self.buf, t, self.prev);
        try self.pending.appendSlice(self.a, self.buf.items);
        // Flush up to the last complete UTF-8 boundary; keep the remainder.
        const cut = loom.tokenizer.completeUtf8Prefix(self.pending.items);
        if (self.cb) |cb| cb(t, self.pending.items.ptr, cut, self.ud);
        const rem = self.pending.items.len - cut;
        std.mem.copyForwards(u8, self.pending.items[0..rem], self.pending.items[cut..]);
        self.pending.shrinkRetainingCapacity(rem);
        self.prev = t;
    }
};

/// Greedily decodes from `opts.prompt` (BOS-only when empty), running every
/// matmul on `dev`, stopping at EOS or the BOS delimiter. All working memory is
/// a call-scoped arena. `out_produced` (nullable) receives the token count.
export fn loom_generate(
    model: ?*Model,
    dev: ?*Device,
    opts: ?*const CGenOpts,
    cb: TokenCb,
    user_data: ?*anyopaque,
    out_produced: ?*usize,
    err: ?*CError,
) c_int {
    const m = model orelse return fail(err, .err_invalid, "null model");
    const d = dev orelse return fail(err, .err_invalid, "null device");
    const o = opts orelse return fail(err, .err_invalid, "null opts");
    const io = m.rt.io();

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const prompt_text: []const u8 = if (o.prompt) |p| std.mem.span(p) else "";
    const prompt = if (prompt_text.len > 0)
        m.tok.encode(a, prompt_text, m.tok.addBos()) catch |e| return fail(err, .err_generate, @errorName(e))
    else
        a.dupe(u32, &[_]u32{m.tok.bosId()}) catch return fail(err, .err_alloc, "OutOfMemory");

    // max_tokens 0 = run until a stop token; the cap is a backstop bounded by the
    // model's trained context (mirrors the CLI). Falls back to DEFAULT_MAX_TOKENS
    // for manifests without max_seq.
    const mf = m.cfg.manifest();
    const cap = if (o.max_tokens != 0)
        o.max_tokens
    else if (mf.max_seq > prompt.len)
        mf.max_seq - prompt.len
    else
        DEFAULT_MAX_TOKENS;
    const timeout = if (o.poll_timeout == 0) loom.linear.DEFAULT_TIMEOUT else o.poll_timeout;
    // repeat_window 0 = the default loop-stop; a nonzero value is used as-is.
    const rep_window = if (o.repeat_window == 0) DEFAULT_REPEAT_WINDOW else o.repeat_window;
    const gen = a.alloc(u32, cap) catch return fail(err, .err_alloc, "OutOfMemory");

    var sbuf: std.ArrayList(u8) = .empty;
    var pending: std.ArrayList(u8) = .empty;
    var sink = CSink{ .tok = m.tok, .a = a, .buf = &sbuf, .pending = &pending, .prev = prompt[prompt.len - 1], .cb = cb, .ud = user_data };
    const stop = [_]u32{ m.tok.eosId(), m.tok.bosId() };

    const produced = loom.forward.generate(a, io, d.iface, m.cfg, timeout, prompt, gen, &stop, rep_window, &sink) catch |e|
        return fail(err, .err_generate, @errorName(e));
    // A final complete-but-unflushed codepoint (or any residual bytes) never
    // got a chance to hit a later token's flush; fire the callback once more.
    if (pending.items.len > 0) {
        if (cb) |c| c(sink.prev, pending.items.ptr, pending.items.len, user_data);
    }
    if (out_produced) |op| op.* = produced;
    return ok(err);
}

test "runtime create/destroy round-trips and yields a usable io" {
    var err: CError = undefined;
    var rt: *Runtime = undefined;
    try std.testing.expectEqual(@as(c_int, 0), loom_runtime_create(&rt, &err));
    _ = rt.io(); // constructible without std.process.Init
    loom_runtime_destroy(rt);
}

test "model load on a missing dir fills err_model_load with a message" {
    var err: CError = undefined;
    var rt: *Runtime = undefined;
    _ = loom_runtime_create(&rt, &err);
    defer loom_runtime_destroy(rt);
    var m: *Model = undefined;
    const rc = loom_model_load(rt, "/definitely/not/a/model", &m, &err);
    try std.testing.expectEqual(@as(c_int, @intFromEnum(Status.err_model_load)), rc);
    try std.testing.expectEqual(@as(c_int, @intFromEnum(Status.err_model_load)), err.code);
    try std.testing.expect(err.message[0] != 0); // carries the @errorName
}

test "host ops match ops.*" {
    var x = [_]f32{ 1.0, -2.0, 3.0, 0.5 };
    var y = [_]f32{ 1.0, -2.0, 3.0, 0.5 };
    loom_silu(&x, x.len);
    loom.ops.siluInPlace(&y);
    try std.testing.expectEqualSlices(f32, &y, &x);

    var sm = [_]f32{ 1, 2, 3, 4 };
    var sm2 = [_]f32{ 1, 2, 3, 4 };
    loom_softmax(&sm, sm.len);
    loom.ops.softmaxInPlace(&sm2);
    try std.testing.expectEqualSlices(f32, &sm2, &sm);

    var a = [_]f32{ 1, 2, 3, 4 };
    var a2 = [_]f32{ 1, 2, 3, 4 };
    const bias = [_]f32{ 5, -6, 7, -8 };
    loom_add(&a, &bias, a.len);
    loom.ops.addInPlace(&a2, &bias);
    try std.testing.expectEqualSlices(f32, &a2, &a);
}
