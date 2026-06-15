//! Device Vision Transformer encoder + multimodal projector. Mirrors the golden
//! reference golden/vision.dart + golden/projector.dart: patch-embed (im2col +
//! int4 matmul) + optional class token + learned position embeddings, then N
//! pre-norm ViT blocks (LayerNorm + bidirectional attention + GELU MLP), a final
//! LayerNorm, and the projector into the LLM's token-embedding space. Every
//! matmul runs on the device (W4A8); LayerNorm/GELU/attention are host glue.
const std = @import("std");
const Io = std.Io;

const model = @import("model.zig");
const linear = @import("linear.zig");
const ops = @import("ops.zig");
const Device = @import("device.zig").Device;

/// A vision matmul: column-tiled when the matrix is wider than the accelerator's
/// capacity (its weights are then stored col-block-contiguous), else single-shot.
/// `scratch` is col-tiling row scratch (>= mat.rows).
fn vmm(io: Io, dev: Device, m: model.Manifest, pt: usize, mat: model.Matrix, x: []const f32, out: []f32, scratch: []f32) !void {
    if (m.max_cols > 0 and mat.cols > m.max_cols) {
        try linear.linearColTiled(io, dev, m.csr_base, m.stores(), pt, mat, m.max_cols, x, out, scratch);
    } else {
        try linear.linear(io, dev, m.csr_base, m.stores(), pt, mat, x, out);
    }
}

/// Looks up `vision.layers.<li>.<kind>` (the returned Matrix does not alias
/// `buf`, so `buf` may be reused after).
fn vmat(cfg: model.Config, buf: []u8, li: usize, kind: []const u8) ?model.Matrix {
    const name = std.fmt.bufPrint(buf, "vision.layers.{d}.{s}", .{ li, kind }) catch return null;
    return cfg.matrix(name);
}

/// Full bidirectional (non-causal) multi-head attention over `seq` rows of
/// `q`/`k`/`vv` (`seq * nheads*hd` row-major), into `out`. `scores` is `seq`.
fn bidirAttention(q: []const f32, k: []const f32, vv: []const f32, seq: usize, nheads: usize, hd: usize, scores: []f32, out: []f32) void {
    const h = nheads * hd;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    @memset(out, 0);
    for (0..nheads) |hh| {
        const off = hh * hd;
        for (0..seq) |i| {
            for (0..seq) |j| {
                var dot: f32 = 0;
                for (0..hd) |d| dot += q[i * h + off + d] * k[j * h + off + d];
                scores[j] = dot * scale;
            }
            ops.softmaxInPlace(scores[0..seq]);
            for (0..seq) |j| {
                const w = scores[j];
                for (0..hd) |d| out[i * h + off + d] += w * vv[j * h + off + d];
            }
        }
    }
}

/// Encodes `pixels` (`num_channels*image_size*image_size`, channels-first, already
/// resized + normalized) into `seq_len` projected embeddings in the text space,
/// written row-major into `out` (`seq_len * projector.output_dim`). Allocates its
/// own scratch and frees it. Requires the model to carry a vision tower + projector.
pub fn encodeImage(gpa: std.mem.Allocator, io: Io, dev: Device, cfg: model.Config, pt: usize, pixels: []const f32, out: []f32) !void {
    const m = cfg.manifest();
    const v = m.vision orelse return error.NoVisionTower;
    const proj = m.projector orelse return error.NoProjector;
    const h: usize = v.hidden;
    const seq: usize = v.seq_len;
    const inter: usize = v.intermediate;
    const patch_len: usize = v.num_channels * v.patch_size * v.patch_size;
    const eps = v.layer_norm_eps;
    const out_dim: usize = proj.output_dim;

    // Idefics3 pixel shuffle merges scale^2 neighbouring patches before the
    // projector, so the projected token count shrinks by scale^2.
    const scale: usize = proj.scale_factor;
    const proj_tokens: usize = if (scale > 1) seq / (scale * scale) else seq;

    if (pixels.len != v.num_channels * v.image_size * v.image_size) return error.BadImageSize;
    if (out.len != proj_tokens * out_dim) return error.BadOutputSize;

    const rows = try gpa.alloc(f32, seq * h);
    defer gpa.free(rows);
    const q = try gpa.alloc(f32, seq * h);
    defer gpa.free(q);
    const k = try gpa.alloc(f32, seq * h);
    defer gpa.free(k);
    const vv = try gpa.alloc(f32, seq * h);
    defer gpa.free(vv);
    const attn = try gpa.alloc(f32, seq * h);
    defer gpa.free(attn);
    const norm = try gpa.alloc(f32, h);
    defer gpa.free(norm);
    const gamma = try gpa.alloc(f32, h);
    defer gpa.free(gamma);
    const beta = try gpa.alloc(f32, h);
    defer gpa.free(beta);
    const patch = try gpa.alloc(f32, patch_len);
    defer gpa.free(patch);
    const fc1o = try gpa.alloc(f32, inter);
    defer gpa.free(fc1o);
    const tmp = try gpa.alloc(f32, @max(h, inter));
    defer gpa.free(tmp);
    const projh = try gpa.alloc(f32, proj.hidden_dim);
    defer gpa.free(projh);
    const scores = try gpa.alloc(f32, seq);
    defer gpa.free(scores);
    // Column-tiling scratch, sized to the widest matmul's output rows.
    const cscratch = try gpa.alloc(f32, @max(inter, @max(h, out_dim)));
    defer gpa.free(cscratch);
    var name_buf: [64]u8 = undefined;

    const patch_mat = cfg.matrix("vision.patch_embed") orelse return error.MissingMatrix;

    // 1. Patch-embed. Class token (if any) is row 0; patches follow.
    const s = v.image_size;
    const p = v.patch_size;
    const pps = s / p;
    const first: usize = if (v.has_class_token) 1 else 0;
    if (v.has_class_token) cfg.readGlue(v.class_token.?, rows[0..h]);
    var pr: usize = 0;
    while (pr < pps * pps) : (pr += 1) {
        const py = pr / pps;
        const px = pr % pps;
        var idx: usize = 0;
        var ch: usize = 0;
        while (ch < v.num_channels) : (ch += 1) {
            var dy: usize = 0;
            while (dy < p) : (dy += 1) {
                var dx: usize = 0;
                while (dx < p) : (dx += 1) {
                    patch[idx] = pixels[ch * s * s + (py * p + dy) * s + (px * p + dx)];
                    idx += 1;
                }
            }
        }
        const row = rows[(first + pr) * h ..][0..h];
        try vmm(io, dev, m, pt, patch_mat, patch, row, cscratch);
        cfg.addGlue(v.patch_embed_bias, row);
    }
    // Add learned position embeddings.
    for (0..seq) |i| {
        cfg.readGlue(v.pos_embed + @as(u32, @intCast(i * h)), tmp[0..h]);
        for (rows[i * h ..][0..h], tmp[0..h]) |*r, pe| r.* += pe;
    }

    // 2. Optional pre-LayerNorm.
    if (v.pre_ln_gamma) |pg| {
        cfg.readGlue(pg, gamma);
        cfg.readGlue(v.pre_ln_beta.?, beta);
        for (0..seq) |i| {
            const r = rows[i * h ..][0..h];
            ops.layerNorm(r, gamma, beta, eps, norm);
            @memcpy(r, norm);
        }
    }

    // 3. Transformer blocks.
    for (v.layers, 0..) |lg, li| {
        const qmat = vmat(cfg, &name_buf, li, "q_proj") orelse return error.MissingMatrix;
        const kmat = vmat(cfg, &name_buf, li, "k_proj") orelse return error.MissingMatrix;
        const vmat_ = vmat(cfg, &name_buf, li, "v_proj") orelse return error.MissingMatrix;
        const omat = vmat(cfg, &name_buf, li, "out_proj") orelse return error.MissingMatrix;

        cfg.readGlue(lg.ln1_gamma, gamma);
        cfg.readGlue(lg.ln1_beta, beta);
        for (0..seq) |i| {
            const r = rows[i * h ..][0..h];
            ops.layerNorm(r, gamma, beta, eps, norm);
            try vmm(io, dev, m, pt, qmat, norm, q[i * h ..][0..h], cscratch);
            cfg.addGlue(lg.q_bias, q[i * h ..][0..h]);
            try vmm(io, dev, m, pt, kmat, norm, k[i * h ..][0..h], cscratch);
            cfg.addGlue(lg.k_bias, k[i * h ..][0..h]);
            try vmm(io, dev, m, pt, vmat_, norm, vv[i * h ..][0..h], cscratch);
            cfg.addGlue(lg.v_bias, vv[i * h ..][0..h]);
        }
        bidirAttention(q, k, vv, seq, v.num_heads, v.head_dim, scores, attn);
        for (0..seq) |i| {
            try vmm(io, dev, m, pt, omat, attn[i * h ..][0..h], tmp[0..h], cscratch);
            cfg.addGlue(lg.o_bias, tmp[0..h]);
            for (rows[i * h ..][0..h], tmp[0..h]) |*r, o| r.* += o;
        }

        // MLP.
        const fc1m = vmat(cfg, &name_buf, li, "fc1") orelse return error.MissingMatrix;
        const fc2m = vmat(cfg, &name_buf, li, "fc2") orelse return error.MissingMatrix;
        cfg.readGlue(lg.ln2_gamma, gamma);
        cfg.readGlue(lg.ln2_beta, beta);
        for (0..seq) |i| {
            const r = rows[i * h ..][0..h];
            ops.layerNorm(r, gamma, beta, eps, norm);
            try vmm(io, dev, m, pt, fc1m, norm, fc1o, cscratch);
            cfg.addGlue(lg.fc1_bias, fc1o);
            ops.geluInPlace(fc1o);
            try vmm(io, dev, m, pt, fc2m, fc1o, tmp[0..h], cscratch);
            cfg.addGlue(lg.fc2_bias, tmp[0..h]);
            for (r, tmp[0..h]) |*rr, d| rr.* += d;
        }
    }

    // 4. Final LayerNorm.
    cfg.readGlue(v.post_ln_gamma, gamma);
    cfg.readGlue(v.post_ln_beta, beta);
    for (0..seq) |i| {
        const r = rows[i * h ..][0..h];
        ops.layerNorm(r, gamma, beta, eps, norm);
        @memcpy(r, norm);
    }

    // 5. Projector into the text embedding space.
    const l1 = cfg.matrix("projector.linear_1") orelse return error.MissingMatrix;

    if (scale > 1) {
        // Idefics3: pixel-shuffle each output token (space-to-depth merge of a
        // scale x scale block, `h*scale^2` channels) then a single linear. Exact
        // replica of golden/projector.dart pixelShuffle's index mapping.
        const side = std.math.sqrt(seq); // seq is a perfect square (no class token)
        const os = side / scale;
        const es = h * scale; // e*scale
        const out_e = h * scale * scale;
        const shuf = try gpa.alloc(f32, out_e);
        defer gpa.free(shuf);
        for (0..proj_tokens) |o| {
            const h2 = o / os;
            const w2 = o % os;
            for (0..out_e) |c| {
                const hh = c / es;
                const rem = c % es;
                const ww = rem / h;
                const ec = rem % h;
                const in_tok = (h2 * scale + hh) * side + (w2 * scale + ww);
                shuf[c] = rows[in_tok * h + ec];
            }
            const dst = out[o * out_dim ..][0..out_dim];
            try vmm(io, dev, m, pt, l1, shuf, dst, cscratch);
            if (proj.bias1) |b| cfg.addGlue(b, dst);
        }
        return;
    }

    const l2: ?model.Matrix = if (proj.num_layers == 2)
        (cfg.matrix("projector.linear_2") orelse return error.MissingMatrix)
    else
        null;
    for (0..seq) |i| {
        const r = rows[i * h ..][0..h];
        const dst = out[i * out_dim ..][0..out_dim];
        if (l2) |l2m| {
            try vmm(io, dev, m, pt, l1, r, projh, cscratch);
            if (proj.bias1) |b| cfg.addGlue(b, projh);
            ops.geluInPlace(projh);
            try vmm(io, dev, m, pt, l2m, projh, dst, cscratch);
            if (proj.bias2) |b| cfg.addGlue(b, dst);
        } else {
            try vmm(io, dev, m, pt, l1, r, dst, cscratch);
            if (proj.bias1) |b| cfg.addGlue(b, dst);
        }
    }
}
