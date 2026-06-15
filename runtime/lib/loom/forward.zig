//! Incremental greedy decode with a per-layer KV cache: each new token computes
//! only its own Q/K/V and attends over the cached K/V (O(T), not the O(T^2)
//! full-sequence recompute). The matmuls run on the device via linear.zig; the
//! nonlinear glue (rmsnorm/rope/attention/silu/residual) runs host-side in f32.
//!
//! `State` holds the persistent KV cache + working buffers so a caller can decode
//! incrementally: `step(token, pos, logits_out?)` runs one token and, when asked,
//! returns the lm_head logits. `generate` is the greedy loop built on `State`.
//!
//! This is the Zig port of the dart `loom_hwgenflash.dart` `_generateKV`.

const std = @import("std");
const Io = std.Io;
const fp = @import("fp.zig");
const ops = @import("ops.zig");
const model = @import("model.zig");
const linear = @import("linear.zig");
const Device = @import("device.zig").Device;

/// A sink whose `emit` does nothing, for callers that don't want live tokens.
pub const NullSink = struct {
    pub fn emit(_: NullSink, _: u32) !void {}
};

/// One-time load of the tiered-cache hot weights + scales into the accelerator's
/// on-chip BRAM (`weights_bram.bin` -> `bram_weight_base`, `scales_bram.bin` ->
/// `bram_scale_base`) via burst writes. Call once after opening the device and
/// before decoding. No-op when the model was built without `--fp-bram-cache-kb`
/// (`bram_cache_kb == 0`): those matrices are all `store == .flash`.
pub fn loadBramCache(gpa: std.mem.Allocator, io: Io, device: Device, cfg: model.Config, dir: anytype) !void {
    const m = cfg.manifest();
    if (m.bram_cache_kb == 0) return;
    const w = try dir.readFileAlloc(io, "weights_bram.bin", gpa, Io.Limit.limited(4 * 1024 * 1024));
    try device.writeMem(m.bram_weight_base, w);
    const s = try dir.readFileAlloc(io, "scales_bram.bin", gpa, Io.Limit.limited(1 * 1024 * 1024));
    try device.writeMem(m.bram_scale_base, s);
}

/// Persistent per-decode state: the KV cache (filled up to the current position)
/// plus reused working buffers, all owned by an internal arena. Allocated once;
/// `step` mutates it in place so a caller can decode incrementally (feed one
/// token, read logits, repeat) without reallocating.
pub const State = struct {
    gpa: std.mem.Allocator,
    /// Single backing store for every buffer below. The working set is fixed and
    /// same-lifetime, so it is one allocation carved into views: one alloc, one
    /// free, contiguous.
    buf: []f32,
    n_ctx: usize,
    hidden_n: usize,
    q_dim: usize,
    kv_dim: usize,
    head_dim: usize,
    /// One layer's KV span (`n_ctx * kv_dim`); `kcache`/`vcache` are `layers` of these.
    kv_stride: usize,
    group: usize,
    scale: f32,
    eps: f32,
    theta: f32,
    kcache: []f32,
    vcache: []f32,
    hidden: []f32,
    norm: []f32,
    gamma: []f32,
    q: []f32,
    kcur: []f32,
    vcur: []f32,
    attn: []f32,
    scores: []f32,
    tmp_h: []f32,
    gate: []f32,
    up: []f32,
    /// Router logits scratch, sized `moe.num_experts` (empty for dense models).
    router_logits: []f32,
    /// MTP scratch (empty for models without MTP heads): the fusion concat
    /// (`2*hidden`), the module's running hidden (`hidden`), and a token-embed
    /// row (`hidden`).
    mtp_cat: []f32,
    mtp_combined: []f32,
    mtp_embed: []f32,
    logits: []f32,
    /// Row-output scratch for column-tiled matmuls (sized to the largest output).
    partial: []f32,
    name_buf: [64]u8,

    /// Allocates the KV cache (`n_ctx` positions) and working buffers for `cfg`
    /// as one contiguous block. Caller must `deinit`. `n_ctx` bounds the sequence
    /// length `step` accepts.
    pub fn init(gpa: std.mem.Allocator, cfg: model.Config, n_ctx: usize) !State {
        const m = cfg.manifest();
        const hidden_n: usize = m.hidden;
        const q_dim: usize = m.num_heads * m.head_dim;
        const kv_dim: usize = m.num_kv_heads * m.head_dim;
        const head_dim: usize = m.head_dim;
        const kv_stride = n_ctx * kv_dim;
        const kv_total = m.layers * kv_stride;

        // SwiGLU scratch must hold the widest FFN a layer runs: dense
        // `intermediate` or, for MoE models, an expert's `moe_intermediate`.
        const moe_inter: usize = if (m.moe) |mo| mo.moe_intermediate else 0;
        const mlp_dim = @max(@as(usize, m.intermediate), moe_inter);
        const n_experts: usize = if (m.moe) |mo| mo.num_experts else 0;
        // MTP scratch: 2*hidden (concat) + hidden (combined) + hidden (embed).
        const mtp_n: usize = if (m.mtp != null) 4 * hidden_n else 0;

        const total = 2 * kv_total + 4 * hidden_n + 2 * q_dim + 2 * kv_dim + n_ctx + 2 * mlp_dim + n_experts + mtp_n + 2 * m.vocab;
        const buf = try gpa.alloc(f32, total);
        errdefer gpa.free(buf);

        // Bump-carve the fixed layout out of `buf` (order is arbitrary).
        var c = Carver{ .buf = buf };
        const state: State = .{
            .gpa = gpa,
            .buf = buf,
            .n_ctx = n_ctx,
            .hidden_n = hidden_n,
            .q_dim = q_dim,
            .kv_dim = kv_dim,
            .head_dim = head_dim,
            .kv_stride = kv_stride,
            .group = m.num_heads / m.num_kv_heads,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .eps = m.norm_eps,
            .theta = m.rope_theta,
            .kcache = c.take(kv_total),
            .vcache = c.take(kv_total),
            .hidden = c.take(hidden_n),
            .norm = c.take(hidden_n),
            .gamma = c.take(hidden_n),
            .q = c.take(q_dim),
            .kcur = c.take(kv_dim),
            .vcur = c.take(kv_dim),
            .attn = c.take(q_dim),
            .scores = c.take(n_ctx),
            .tmp_h = c.take(hidden_n),
            .gate = c.take(mlp_dim),
            .up = c.take(mlp_dim),
            .router_logits = c.take(n_experts),
            .mtp_cat = c.take(if (m.mtp != null) 2 * hidden_n else 0),
            .mtp_combined = c.take(if (m.mtp != null) hidden_n else 0),
            .mtp_embed = c.take(if (m.mtp != null) hidden_n else 0),
            .logits = c.take(m.vocab),
            .partial = c.take(m.vocab),
            .name_buf = undefined,
        };
        std.debug.assert(c.off == total);
        return state;
    }

    /// Hands out successive non-overlapping views of a backing buffer.
    const Carver = struct {
        buf: []f32,
        off: usize = 0,
        fn take(self: *Carver, n: usize) []f32 {
            const s = self.buf[self.off..][0..n];
            self.off += n;
            return s;
        }
    };

    pub fn deinit(self: *State) void {
        self.gpa.free(self.buf);
    }

    fn matmul(self: *State, io: Io, dev: Device, cfg: model.Config, mm: model.Manifest, pt: usize, layer: usize, kind: []const u8, x: []const f32, o: []f32) !void {
        const name = try std.fmt.bufPrint(&self.name_buf, "layers.{d}.{s}", .{ layer, kind });
        try self.matmulNamed(io, dev, cfg, mm, pt, name, x, o);
    }

    /// Runs the matrix named `name` (col-tiled if wider than the accelerator).
    /// `name` may alias `self.name_buf`; the matrix is resolved (a value copy)
    /// before any further use of that buffer.
    fn matmulNamed(self: *State, io: Io, dev: Device, cfg: model.Config, mm: model.Manifest, pt: usize, name: []const u8, x: []const f32, o: []f32) !void {
        const mat = cfg.matrix(name) orelse return error.MissingMatrix;
        // Column-tile matrices wider than the accelerator's capacity (down_proj on
        // a small-maxCols build); otherwise a single call.
        if (mm.max_cols > 0 and mat.cols > mm.max_cols) {
            try linear.linearColTiled(io, dev, mm.csr_base, mm.stores(), pt, mat, mm.max_cols, x, o, self.partial);
        } else {
            try linear.linear(io, dev, mm.csr_base, mm.stores(), pt, mat, x, o);
        }
    }

    /// Fused SwiGLU: `o = silu(gate_proj @ x) * (up_proj @ x)`, folded on-device
    /// so the host reads back only the product (not gate and up separately) and
    /// the SiLU + elementwise multiply run in hardware. Resolves both matrices
    /// (value copies, so reusing name_buf is safe).
    fn matmulSwiGlu(self: *State, io: Io, dev: Device, cfg: model.Config, mm: model.Manifest, pt: usize, layer: usize, x: []const f32, o: []f32) !void {
        const gname = try std.fmt.bufPrint(&self.name_buf, "layers.{d}.gate_proj", .{layer});
        const gmat = cfg.matrix(gname) orelse return error.MissingMatrix;
        const uname = try std.fmt.bufPrint(&self.name_buf, "layers.{d}.up_proj", .{layer});
        const umat = cfg.matrix(uname) orelse return error.MissingMatrix;
        if (mm.max_cols > 0 and gmat.cols > mm.max_cols) {
            // Wide gate/up are stored col-block-contiguous, so the on-device fused
            // SwiGLU cannot read them; column-tile each, then fold silu(gate)*up on
            // the host. (gate -> self.up scratch, up -> o.)
            try linear.linearColTiled(io, dev, mm.csr_base, mm.stores(), pt, gmat, mm.max_cols, x, self.up[0..gmat.rows], self.partial);
            try linear.linearColTiled(io, dev, mm.csr_base, mm.stores(), pt, umat, mm.max_cols, x, o, self.partial);
            for (o, self.up[0..gmat.rows]) |*ov, g| ov.* *= g / (1.0 + @exp(-g));
        } else {
            try linear.linearSwiGlu(io, dev, mm.csr_base, mm.stores(), pt, gmat, umat, x, o);
        }
    }

    /// Mixture-of-Experts FFN for one token. Router (a tiny host fp16 matmul)
    /// softmaxes over experts, the top-k are selected, each runs a fused-SwiGLU
    /// + down_proj on-device, and the (optionally renormalized) router weights
    /// combine them additively into `self.hidden`. Mirrors golden/moe.dart's
    /// `moeMlp` exactly so the device path matches the fp64 reference.
    fn moeMlp(self: *State, io: Io, dev: Device, cfg: model.Config, mm: model.Manifest, pt: usize, layer: usize, moe: model.Moe, router_off: u32) !void {
        const ne: usize = moe.num_experts;

        // 1. Router logits over the post-attn norm (host-side, fp16 glue).
        const logits = self.router_logits[0..ne];
        cfg.matvecGlue(router_off, ne, self.hidden_n, self.norm, logits);

        // 2. Route: softmax + top-k + renormalized combine weights (shared with
        // the C ABI's loom_moe_route so host and device agree).
        var chosen: [ops.MAX_EXPERTS]usize = undefined;
        var weights: [ops.MAX_EXPERTS]f32 = undefined;
        const k = ops.moeRoute(logits, moe.top_k, moe.norm_topk, &chosen, &weights);

        // 3. Run each chosen expert and weight-accumulate into hidden.
        const inter = self.gate[0..moe.moe_intermediate];
        for (0..k) |j| {
            const e = chosen[j];
            const w = weights[j];

            // Fused SwiGLU: silu(gate @ norm) * (up @ norm) -> inter.
            const gname = try std.fmt.bufPrint(&self.name_buf, "layers.{d}.experts.{d}.gate_proj", .{ layer, e });
            const gmat = cfg.matrix(gname) orelse return error.MissingMatrix;
            const uname = try std.fmt.bufPrint(&self.name_buf, "layers.{d}.experts.{d}.up_proj", .{ layer, e });
            const umat = cfg.matrix(uname) orelse return error.MissingMatrix;
            try linear.linearSwiGlu(io, dev, mm.csr_base, mm.stores(), pt, gmat, umat, self.norm, inter);

            // down_proj (col-tiled when wider than the accelerator).
            const dname = try std.fmt.bufPrint(&self.name_buf, "layers.{d}.experts.{d}.down_proj", .{ layer, e });
            const dmat = cfg.matrix(dname) orelse return error.MissingMatrix;
            if (mm.max_cols > 0 and dmat.cols > mm.max_cols) {
                try linear.linearColTiled(io, dev, mm.csr_base, mm.stores(), pt, dmat, mm.max_cols, inter, self.tmp_h, self.partial);
            } else {
                try linear.linear(io, dev, mm.csr_base, mm.stores(), pt, dmat, inter, self.tmp_h);
            }
            for (self.hidden, self.tmp_h) |*hv, ov| hv.* += w * ov;
        }
    }

    /// Runs `token` at `pos` through all layers, writing this position's K/V into
    /// the cache and attending over positions 0..=pos. When `logits_out` is
    /// non-null, also applies the final norm + lm_head, writing `vocab` logits.
    pub fn step(self: *State, io: Io, device: Device, cfg: model.Config, poll_timeout: usize, token: u32, pos: usize, logits_out: ?[]f32) !void {
        const m = cfg.manifest();
        cfg.readGlue(m.embed_offset + @as(u32, @intCast(token * self.hidden_n)), self.hidden);
        try self.stepCore(io, device, cfg, poll_timeout, pos, logits_out);
    }

    /// Like [step] but the input embedding is supplied directly (for VLM image
    /// positions, whose embedding is a projected vision token rather than a
    /// looked-up token embedding). `embed` is `hidden`-length.
    pub fn stepEmbed(self: *State, io: Io, device: Device, cfg: model.Config, poll_timeout: usize, embed: []const f32, pos: usize, logits_out: ?[]f32) !void {
        @memcpy(self.hidden, embed[0..self.hidden_n]);
        try self.stepCore(io, device, cfg, poll_timeout, pos, logits_out);
    }

    /// Runs all layers over the input embedding already in `self.hidden` for
    /// position `pos`, updating the KV cache; writes logits when requested.
    fn stepCore(self: *State, io: Io, device: Device, cfg: model.Config, poll_timeout: usize, pos: usize, logits_out: ?[]f32) !void {
        const m = cfg.manifest();
        std.debug.assert(pos < self.n_ctx);

        for (0..m.layers) |l| {
            // This layer's slice of the flat KV cache.
            const kl = self.kcache[l * self.kv_stride ..][0..self.kv_stride];
            const vl = self.vcache[l * self.kv_stride ..][0..self.kv_stride];

            cfg.readGlue(m.layer_glue[l].input_norm, self.gamma);
            ops.rmsNorm(self.hidden, self.gamma, self.eps, self.norm);

            try self.matmul(io, device, cfg, m, poll_timeout, l, "q_proj", self.norm, self.q);
            try self.matmul(io, device, cfg, m, poll_timeout, l, "k_proj", self.norm, self.kcur);
            try self.matmul(io, device, cfg, m, poll_timeout, l, "v_proj", self.norm, self.vcur);

            // Qwen2 q/k/v bias (host-side glue, added before RoPE). A no-op for
            // archs whose manifest carries no bias offsets.
            if (m.layer_glue[l].q_bias) |off| cfg.addGlue(off, self.q);
            if (m.layer_glue[l].k_bias) |off| cfg.addGlue(off, self.kcur);
            if (m.layer_glue[l].v_bias) |off| cfg.addGlue(off, self.vcur);

            for (0..m.num_heads) |h| ops.ropeHead(self.q[h * self.head_dim ..][0..self.head_dim], pos, self.theta);
            for (0..m.num_kv_heads) |h| ops.ropeHead(self.kcur[h * self.head_dim ..][0..self.head_dim], pos, self.theta);

            @memcpy(kl[pos * self.kv_dim ..][0..self.kv_dim], self.kcur);
            @memcpy(vl[pos * self.kv_dim ..][0..self.kv_dim], self.vcur);

            const len = pos + 1;
            @memset(self.attn, 0);
            for (0..m.num_heads) |h| {
                const kvh = h / self.group;
                for (0..len) |s| {
                    var dot: f32 = 0;
                    for (0..self.head_dim) |d| {
                        dot += self.q[h * self.head_dim + d] * kl[s * self.kv_dim + kvh * self.head_dim + d];
                    }
                    self.scores[s] = dot * self.scale;
                }
                ops.softmaxInPlace(self.scores[0..len]);
                for (0..len) |s| {
                    const w = self.scores[s];
                    for (0..self.head_dim) |d| {
                        self.attn[h * self.head_dim + d] += w * vl[s * self.kv_dim + kvh * self.head_dim + d];
                    }
                }
            }

            try self.matmul(io, device, cfg, m, poll_timeout, l, "o_proj", self.attn, self.tmp_h);
            for (self.hidden, self.tmp_h) |*hv, ov| hv.* += ov;

            cfg.readGlue(m.layer_glue[l].post_norm, self.gamma);
            ops.rmsNorm(self.hidden, self.gamma, self.eps, self.norm);
            if (m.moe != null and m.layer_glue[l].router != null) {
                // Mixture-of-Experts: router top-k + per-expert SwiGLU, combined
                // straight into hidden.
                try self.moeMlp(io, device, cfg, m, poll_timeout, l, m.moe.?, m.layer_glue[l].router.?);
            } else {
                // Dense fused SwiGLU on-device: silu(gate @ norm) * (up @ norm) ->
                // gate, one pass reading back only the product. Then down_proj.
                try self.matmulSwiGlu(io, device, cfg, m, poll_timeout, l, self.norm, self.gate);
                try self.matmul(io, device, cfg, m, poll_timeout, l, "down_proj", self.gate, self.tmp_h);
                for (self.hidden, self.tmp_h) |*hv, ov| hv.* += ov;
            }
        }

        if (logits_out) |lo| {
            cfg.readGlue(m.final_norm_offset, self.gamma);
            ops.rmsNorm(self.hidden, self.gamma, self.eps, self.norm);
            // lm_head is wide too: col-tile it when the build does (else it would
            // read col-block-packed weights as a single tile-major image).
            try self.matmulNamed(io, device, cfg, m, poll_timeout, "lm_head", self.norm, lo);
        }
    }

    /// Runs one MTP module for the draft position: fuses `self.hidden` (the
    /// previous depth's hidden) with the embedding in `self.mtp_embed` via
    /// `eh_proj`, runs the module's transformer block on-device, and writes the
    /// result back into `self.hidden`. Mirrors golden/mtp.dart's single-position
    /// module forward (attention degenerates to the value projection).
    fn mtpModuleForward(self: *State, io: Io, dev: Device, cfg: model.Config, mm: model.Manifest, pt: usize, mi: usize, g: model.MtpModuleGlue) !void {
        const h = self.hidden_n;
        // Fuse: cat = [hnorm(prev) ; enorm(embed)]; combined = eh_proj @ cat.
        cfg.readGlue(g.hnorm, self.gamma);
        ops.rmsNorm(self.hidden, self.gamma, self.eps, self.mtp_cat[0..h]);
        cfg.readGlue(g.enorm, self.gamma);
        ops.rmsNorm(self.mtp_embed, self.gamma, self.eps, self.mtp_cat[h .. 2 * h]);
        var name = try std.fmt.bufPrint(&self.name_buf, "mtp.{d}.eh_proj", .{mi});
        try self.matmulNamed(io, dev, cfg, mm, pt, name, self.mtp_cat, self.mtp_combined);

        // Attention sub-block (single position): out head = its kv head's value.
        cfg.readGlue(g.input_norm, self.gamma);
        ops.rmsNorm(self.mtp_combined, self.gamma, self.eps, self.norm);
        name = try std.fmt.bufPrint(&self.name_buf, "mtp.{d}.v_proj", .{mi});
        try self.matmulNamed(io, dev, cfg, mm, pt, name, self.norm, self.vcur);
        for (0..mm.num_heads) |hd| {
            const kv = hd / self.group;
            @memcpy(self.attn[hd * self.head_dim ..][0..self.head_dim], self.vcur[kv * self.head_dim ..][0..self.head_dim]);
        }
        name = try std.fmt.bufPrint(&self.name_buf, "mtp.{d}.o_proj", .{mi});
        try self.matmulNamed(io, dev, cfg, mm, pt, name, self.attn, self.tmp_h);
        for (self.mtp_combined, self.tmp_h) |*c, o| c.* += o; // h1 = combined + o

        // Gated-SwiGLU FFN + residual, written back into hidden.
        cfg.readGlue(g.post_norm, self.gamma);
        ops.rmsNorm(self.mtp_combined, self.gamma, self.eps, self.norm);
        const gname = try std.fmt.bufPrint(&self.name_buf, "mtp.{d}.gate_proj", .{mi});
        const gmat = cfg.matrix(gname) orelse return error.MissingMatrix;
        const uname = try std.fmt.bufPrint(&self.name_buf, "mtp.{d}.up_proj", .{mi});
        const umat = cfg.matrix(uname) orelse return error.MissingMatrix;
        try linear.linearSwiGlu(io, dev, mm.csr_base, mm.stores(), pt, gmat, umat, self.norm, self.gate[0..mm.intermediate]);
        name = try std.fmt.bufPrint(&self.name_buf, "mtp.{d}.down_proj", .{mi});
        try self.matmulNamed(io, dev, cfg, mm, pt, name, self.gate[0..mm.intermediate], self.tmp_h);
        for (self.hidden, self.mtp_combined, self.tmp_h) |*out, c, d| out.* = c + d;
    }

    /// Multi-Token Prediction draft: prefills `tokens`, then chains the MTP
    /// modules to draft `mtp.num_modules + 1` tokens (the main model's next token
    /// plus one per module) into `draft_out`. Returns the draft length. Requires
    /// the model to carry MTP heads and `draft_out.len >= num_modules + 1`.
    pub fn mtpDraft(self: *State, io: Io, device: Device, cfg: model.Config, poll_timeout: usize, tokens: []const u32, draft_out: []u32) !usize {
        const m = cfg.manifest();
        const mtp = m.mtp orelse return error.NoMtpHeads;
        std.debug.assert(tokens.len > 0);

        // Main forward over the prompt: last step leaves self.hidden = the last
        // hidden (pre-final-norm) and self.logits = the main next-token logits.
        for (tokens, 0..) |tok, pos| {
            const last = pos + 1 == tokens.len;
            try self.step(io, device, cfg, poll_timeout, tok, pos, if (last) self.logits else null);
        }
        draft_out[0] = @intCast(argmax(self.logits));

        var tok = draft_out[0];
        for (0..mtp.num_modules) |mi| {
            cfg.readGlue(m.embed_offset + @as(u32, @intCast(tok * self.hidden_n)), self.mtp_embed);
            try self.mtpModuleForward(io, device, cfg, m, poll_timeout, mi, mtp.modules[mi]);
            // Read out through the shared final-norm + lm_head.
            cfg.readGlue(m.final_norm_offset, self.gamma);
            ops.rmsNorm(self.hidden, self.gamma, self.eps, self.norm);
            try self.matmulNamed(io, device, cfg, m, poll_timeout, "lm_head", self.norm, self.logits);
            draft_out[mi + 1] = @intCast(argmax(self.logits));
            tok = draft_out[mi + 1];
        }
        return mtp.num_modules + 1;
    }
};

/// Greedily decodes from `prompt`, running every matmul on `device` (real
/// transport or sim). Writes at most `out.len` tokens and stops early if it
/// produces any id in `stop_tokens` (the terminator is NOT written). Each kept
/// token is passed to `sink.emit(id)` as it is produced (for live streaming);
/// pass a `NullSink` to disable. Returns the number of tokens actually written.
/// Scans the generated prefix `gen` for the trailing `w`-token window recurring
/// at an earlier position (the signature of a decode loop).
fn loopedTail(gen: []const u32, w: usize) bool {
    if (w == 0 or gen.len < 2 * w) return false;
    const tail_start = gen.len - w;
    const tail = gen[tail_start..][0..w];
    var j: usize = 0;
    while (j < tail_start) : (j += 1) {
        if (std.mem.eql(u32, gen[j..][0..w], tail)) return true;
    }
    return false;
}

pub fn generate(
    gpa: std.mem.Allocator,
    io: Io,
    device: Device,
    cfg: model.Config,
    poll_timeout: usize,
    prompt: []const u32,
    out: []u32,
    stop_tokens: []const u32,
    /// Loop-stop window: end generation when the trailing this-many output tokens
    /// exactly recur earlier (a decode loop, common for tiny models that never
    /// emit a stop token). 0 disables. 8 exact tokens recurring is a strong loop
    /// signal coherent text rarely trips; loosen for a larger, less-loopy model.
    repeat_window: usize,
    sink: anytype,
) !usize {
    const max_pos = prompt.len + out.len;
    var state = try State.init(gpa, cfg, max_pos);
    defer state.deinit();

    // Positions: the whole prompt (prefill), then each generated token fed back.
    // The last prompt token produces out[0]; position p produces out[p+1-len].
    var produced: usize = 0;
    for (0..prompt.len + out.len - 1) |pos| {
        const token = if (pos < prompt.len) prompt[pos] else out[pos - prompt.len];
        const want_logits = pos + 1 >= prompt.len; // last prompt token and every decode step
        try state.step(io, device, cfg, poll_timeout, token, pos, if (want_logits) state.logits else null);
        if (want_logits) {
            const next: u32 = @intCast(argmax(state.logits));
            const oi = pos + 1 - prompt.len;
            // A stop token (EOS, or the BOS story delimiter) ends generation and
            // is not emitted, so the decoded text stops cleanly.
            for (stop_tokens) |s| {
                if (next == s) return produced;
            }
            out[oi] = next;
            produced = oi + 1;
            try sink.emit(next);
            // Loop-stop: the model is repeating a recent window, so nothing new
            // is coming. End here rather than churn to the context limit.
            if (loopedTail(out[0..produced], repeat_window)) return produced;
        }
    }
    return produced;
}

fn argmax(v: []const f32) usize {
    var best: usize = 0;
    for (v, 0..) |x, i| {
        if (x > v[best]) best = i;
    }
    return best;
}
