//! Loom model config + weight manifest, from the genip-emitted `loom.json` and
//! `scales.bin`. The runtime is model-agnostic: it reads what the generator
//! declared (dims, per-matrix flash offsets, fp16 scales) and never hardcodes.

const std = @import("std");
const Io = std.Io;
const fp = @import("fp.zig");

/// Opens a genip output directory, handling both absolute and cwd-relative
/// paths. Caller closes the returned Dir.
pub fn openModelDir(io: Io, path: []const u8) !Io.Dir {
    if (path.len > 0 and path[0] == '/') return Io.Dir.openDirAbsolute(io, path, .{});
    return Io.Dir.cwd().openDir(io, path, .{});
}

/// Decodes little-endian fp16 words from raw bytes into an owned []u16.
fn decodeU16(gpa: std.mem.Allocator, bytes: []const u8) ![]u16 {
    const n = bytes.len / 2;
    const out = try gpa.alloc(u16, n);
    errdefer gpa.free(out);
    for (0..n) |i| out[i] = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
    return out;
}

/// One linear weight matrix's placement in the flash weight image.
/// Which weight store a matrix lives in. `flash` = the SPI config flash (slow,
/// resident). `bram` = the on-chip tiered-cache SRAM (fast). The genip
/// `--fp-bram-cache-kb` split tags the biggest matrices `bram`; the manifest's
/// `bram_weight_base`/`bram_scale_base` locate that store.
pub const Store = enum { flash, bram };

pub const Matrix = struct {
    name: []const u8,
    /// Byte offset into this matrix's store image (weights.bin for flash,
    /// weights_bram.bin for bram); WEIGHT_BASE = <store>_weight_base + this.
    weight_offset: u32,
    /// Offset (in fp16 elements) into the host-side scales table (legacy push).
    /// Unused by the flash/BRAM-resident scale path; defaults so manifests that
    /// omit it (the bram-cache split) still parse.
    scale_offset: u32 = 0,
    /// Byte offset of this matrix's scales in its store's scale image (resident).
    scale_flash_offset: u32,
    rows: u32,
    cols: u32,
    col_tiles: u32,
    /// Number of per-row scale groups along the input dim. `1` = one scale per
    /// row (per-tensor-row W4A8). When a matrix col-tiles with a group per
    /// col-block, `groups` = number of col-blocks and the scale table is laid
    /// out group-major (block b's `rows` scales at scale_flash_offset+b*rows*4).
    /// Defaults to 1 so pre-group manifests parse unchanged.
    groups: u32 = 1,
    /// Which store this matrix's weights + scales live in. Defaults to `flash`
    /// so pre-cache manifests (no `store` field) parse unchanged.
    store: Store = .flash,
};

/// Per-layer RMSNorm gamma offsets (fp16 elements into glue.bin), plus the
/// optional q/k/v bias offsets for archs that use them (Qwen2). The bias fields
/// default to null so bias-free (llama) manifests parse unchanged.
pub const LayerGlue = struct {
    input_norm: u32,
    post_norm: u32,
    q_bias: ?u32 = null,
    k_bias: ?u32 = null,
    v_bias: ?u32 = null,
    /// fp16-element offset of this layer's MoE router weights `[num_experts x
    /// hidden]` in glue.bin. Present (non-null) iff the layer is Mixture-of-
    /// Experts; dense layers leave it null.
    router: ?u32 = null,
};

/// Shared Mixture-of-Experts config (uniform across a model's MoE layers).
/// Present on the manifest only for MoE models. A layer is MoE iff its
/// [LayerGlue.router] is non-null; its experts are the matrices named
/// `layers.<i>.experts.<e>.{gate,up,down}_proj`.
pub const Moe = struct {
    num_experts: u32,
    top_k: u32,
    norm_topk: bool = true,
    moe_intermediate: u32,
    num_shared: u32 = 0,
};

/// One Multi-Token Prediction module's fp16 glue offsets (enorm/hnorm + block
/// norms). Its eh_proj + transformer matrices are in `matrices`, named
/// `mtp.<m>.{eh_proj,q_proj,...}`.
pub const MtpModuleGlue = struct {
    enorm: u32,
    hnorm: u32,
    input_norm: u32,
    post_norm: u32,
};

/// Multi-Token Prediction heads (DeepSeek-V3 style), present only for models
/// with them. The token embedding, final norm, and lm_head are shared with the
/// main model.
pub const Mtp = struct {
    num_modules: u32,
    modules: []const MtpModuleGlue,
};

/// One ViT block's fp16 glue offsets: the two LayerNorms (gamma+beta) plus the
/// attention/MLP biases. The block's matmul matrices are in `matrices`, named
/// `vision.layers.<i>.{q_proj,k_proj,v_proj,out_proj,fc1,fc2}`.
pub const VisionLayerGlue = struct {
    ln1_gamma: u32,
    ln1_beta: u32,
    q_bias: u32,
    k_bias: u32,
    v_bias: u32,
    o_bias: u32,
    ln2_gamma: u32,
    ln2_beta: u32,
    fc1_bias: u32,
    fc2_bias: u32,
};

/// A Vision Transformer encoder tower (CLIP/SigLIP). Matmul matrices are int4 in
/// `matrices` (`vision.patch_embed`, `vision.layers.<i>.*`); LayerNorm params,
/// biases, the class token, and position embeddings are fp16 glue.
pub const Vision = struct {
    image_size: u32,
    patch_size: u32,
    num_channels: u32,
    hidden: u32,
    num_layers: u32,
    num_heads: u32,
    head_dim: u32,
    intermediate: u32,
    layer_norm_eps: f32,
    has_class_token: bool,
    seq_len: u32,
    /// Per-channel image normalization for host preprocessing.
    image_mean: [3]f32,
    image_std: [3]f32,
    patch_embed_bias: u32,
    class_token: ?u32 = null,
    pos_embed: u32,
    pre_ln_gamma: ?u32 = null,
    pre_ln_beta: ?u32 = null,
    post_ln_gamma: u32,
    post_ln_beta: u32,
    layers: []const VisionLayerGlue,
};

/// The vision->text projector (matrices `projector.linear_1`/`linear_2` int4;
/// biases fp16 glue). One linear, or a two-layer GELU MLP.
pub const Projector = struct {
    num_layers: u32,
    input_dim: u32,
    hidden_dim: u32,
    output_dim: u32,
    bias1: ?u32 = null,
    bias2: ?u32 = null,
    /// Pixel-shuffle merge factor (Idefics3/SmolVLM); 1 = no shuffle (LLaVA). The
    /// projected token count is `vision.seq_len / scale_factor^2`.
    scale_factor: u32 = 1,
};

/// The generated model manifest (`loom.json`), parsed field-for-field.
pub const Manifest = struct {
    name: []const u8,
    hidden: u32,
    vocab: u32,
    layers: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    /// Trained context length (llama2.c `seq_len`). Default generation cap when
    /// `--tokens` is omitted: generation runs until a stop token or this many
    /// positions (past it the model extrapolates RoPE into untrained ground).
    /// 0 = unknown (older manifests); the runtime falls back to a fixed default.
    max_seq: u32 = 0,
    /// Accelerator column capacity. Matrices wider than this are stored col-block
    /// contiguous and the runtime col-tiles them. 0 = no tiling (single block).
    max_cols: u32 = 0,
    /// Mixture-of-Experts config, null for dense models.
    moe: ?Moe = null,
    /// Multi-Token Prediction heads, null for models without them.
    mtp: ?Mtp = null,
    /// Vision tower + projector + image placeholder token, for VLMs.
    vision: ?Vision = null,
    projector: ?Projector = null,
    image_token_index: ?u32 = null,
    rope_theta: f32,
    norm_eps: f32,
    tie_embeddings: bool,
    csr_base: u32,
    flash_weight_base: u32,
    scale_flash_base: u32,
    /// On-chip tiered-cache SRAM bases for `store == .bram` matrices. Zero (and
    /// `bram_cache_kb == 0`) when the model was built without `--fp-bram-cache-kb`.
    bram_weight_base: u32 = 0,
    bram_scale_base: u32 = 0,
    bram_cache_kb: u32 = 0,
    embed_offset: u32,
    final_norm_offset: u32,
    layer_glue: []const LayerGlue,
    matrices: []const Matrix,

    /// The four store bases bundled for the linear driver, which picks per matrix
    /// by `mat.store`. Cheap value; call per-matmul.
    pub fn stores(self: Manifest) Stores {
        return .{
            .flash_weight_base = self.flash_weight_base,
            .scale_flash_base = self.scale_flash_base,
            .bram_weight_base = self.bram_weight_base,
            .bram_scale_base = self.bram_scale_base,
        };
    }
};

/// The weight/scale store bases the linear driver selects between per matrix.
pub const Stores = struct {
    flash_weight_base: u32,
    scale_flash_base: u32,
    bram_weight_base: u32 = 0,
    bram_scale_base: u32 = 0,

    /// This matrix's weight-store base (add `weight_offset` for its address).
    pub fn weightStore(self: Stores, mat: Matrix) u32 {
        return switch (mat.store) {
            .bram => self.bram_weight_base,
            .flash => self.flash_weight_base,
        };
    }
    /// This matrix's scale-store base (add `scale_flash_offset` for its address).
    pub fn scaleStore(self: Stores, mat: Matrix) u32 {
        return switch (mat.store) {
            .bram => self.bram_scale_base,
            .flash => self.scale_flash_base,
        };
    }
};

pub const Config = struct {
    parsed: std.json.Parsed(Manifest),
    /// fp16 per-row scales, decoded little-endian from scales.bin. Owned.
    scales: []u16,
    /// fp16 glue weights (embed table + norm gammas), from glue.bin. Owned.
    glue: []u16,
    gpa: std.mem.Allocator,

    /// Parses `loom.json` + raw `scales.bin` + `glue.bin` bytes into an owned
    /// Config. Borrows none of the inputs (manifest copied into the json arena,
    /// scales/glue decoded into owned buffers). Caller must `deinit`.
    pub fn parse(
        gpa: std.mem.Allocator,
        json: []const u8,
        scales_bytes: []const u8,
        glue_bytes: []const u8,
    ) !Config {
        const parsed = try std.json.parseFromSlice(
            Manifest,
            gpa,
            json,
            // alloc_always: copy every string into the parser's arena, so the
            // manifest stays valid after the caller frees the source `json`
            // (the default alloc_if_needed leaves unescaped strings pointing
            // into the source, which dangle once it's freed).
            .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
        );
        errdefer parsed.deinit();

        const scales = try decodeU16(gpa, scales_bytes);
        errdefer gpa.free(scales);
        const glue = try decodeU16(gpa, glue_bytes);
        errdefer gpa.free(glue);
        return .{ .parsed = parsed, .scales = scales, .glue = glue, .gpa = gpa };
    }

    /// Loads `loom.json` + `scales.bin` + `glue.bin` from a genip output
    /// directory. Caller must `deinit`.
    pub fn load(gpa: std.mem.Allocator, io: Io, dir_path: []const u8) !Config {
        var dir = try openModelDir(io, dir_path);
        defer dir.close(io);
        // Config/weight-side files are bounded (< a few hundred KB); cap it.
        const cap = Io.Limit.limited(64 * 1024 * 1024);
        const json = try dir.readFileAlloc(io, "loom.json", gpa, cap);
        defer gpa.free(json);
        const scales_bytes = try dir.readFileAlloc(io, "scales.bin", gpa, cap);
        defer gpa.free(scales_bytes);
        const glue_bytes = try dir.readFileAlloc(io, "glue.bin", gpa, cap);
        defer gpa.free(glue_bytes);
        return parse(gpa, json, scales_bytes, glue_bytes);
    }

    pub fn deinit(self: *Config) void {
        self.gpa.free(self.glue);
        self.gpa.free(self.scales);
        self.parsed.deinit();
    }

    /// Reads `out.len` fp16 glue weights starting at fp16-element `offset`,
    /// decoding to f32 into `out`.
    pub fn readGlue(self: Config, offset: u32, out: []f32) void {
        for (0..out.len) |i| out[i] = fp.fromBits(self.glue[offset + i]);
    }

    /// Adds `dst.len` fp16 glue weights starting at fp16-element `offset` into
    /// `dst` (decoding to f32). Used for the Qwen2 q/k/v bias-add.
    pub fn addGlue(self: Config, offset: u32, dst: []f32) void {
        for (0..dst.len) |i| dst[i] += fp.fromBits(self.glue[offset + i]);
    }

    /// Host matvec against an fp16 glue matrix `[rows x cols]` at fp16-element
    /// `offset`: `out[r] = sum_c glue[offset + r*cols + c] * x[c]`. Used for the
    /// tiny MoE router, which stays in fp16 glue and runs host-side (routing is
    /// precision-sensitive and cheap, so it never hits the int4 device path).
    pub fn matvecGlue(self: Config, offset: u32, rows: usize, cols: usize, x: []const f32, out: []f32) void {
        for (0..rows) |r| {
            var acc: f32 = 0;
            const base = offset + @as(u32, @intCast(r * cols));
            for (0..cols) |cc| acc += fp.fromBits(self.glue[base + cc]) * x[cc];
            out[r] = acc;
        }
    }

    pub fn manifest(self: Config) Manifest {
        return self.parsed.value;
    }

    /// The matrix named `name`, or null. The manifest is small (dozens of
    /// entries), so a linear scan is the right tool.
    pub fn matrix(self: Config, name: []const u8) ?Matrix {
        for (self.parsed.value.matrices) |m| {
            if (std.mem.eql(u8, m.name, name)) return m;
        }
        return null;
    }
};

const test_json =
    \\{ "name":"tiny", "hidden":8, "vocab":16, "layers":2,
    \\  "num_heads":2, "num_kv_heads":1, "head_dim":4, "intermediate":12,
    \\  "rope_theta":10000.0, "norm_eps":0.00001, "tie_embeddings":true,
    \\  "csr_base":65536, "flash_weight_base":538968064, "scale_flash_base":538968064,
    \\  "embed_offset":0, "final_norm_offset":100,
    \\  "layer_glue":[ {"input_norm":10,"post_norm":20}, {"input_norm":30,"post_norm":40} ],
    \\  "matrices":[
    \\    {"name":"layers.0.q_proj","weight_offset":0,"scale_offset":0,"scale_flash_offset":0,"rows":8,"cols":8,"col_tiles":4},
    \\    {"name":"layers.0.k_proj","weight_offset":64,"scale_offset":8,"scale_flash_offset":32,"rows":4,"cols":8,"col_tiles":4}
    \\  ] }
;

// fp16 1.0, 0.5 - reused for both the scale and glue tables in tests.
const test_fp16 = [_]u8{ 0x00, 0x3C, 0x00, 0x38 };

test "parse reads model dims and addresses" {
    var cfg = try Config.parse(std.testing.allocator, test_json, &test_fp16, &test_fp16);
    defer cfg.deinit();
    const m = cfg.manifest();
    try std.testing.expectEqual(@as(u32, 8), m.hidden);
    try std.testing.expectEqual(@as(u32, 2), m.layers);
    try std.testing.expectEqual(@as(u32, 1), m.num_kv_heads);
    try std.testing.expectEqual(@as(u32, 0x10000), m.csr_base);
    try std.testing.expectEqual(@as(u32, 0x20200000), m.flash_weight_base);
    try std.testing.expectEqualStrings("tiny", m.name);
}

test "parse reads the glue offsets" {
    var cfg = try Config.parse(std.testing.allocator, test_json, &test_fp16, &test_fp16);
    defer cfg.deinit();
    const m = cfg.manifest();
    try std.testing.expectEqual(@as(u32, 0), m.embed_offset);
    try std.testing.expectEqual(@as(u32, 100), m.final_norm_offset);
    try std.testing.expectEqual(@as(u32, 10), m.layer_glue[0].input_norm);
    try std.testing.expectEqual(@as(u32, 40), m.layer_glue[1].post_norm);
}

test "matrix lookup finds by name and returns null for a missing name" {
    var cfg = try Config.parse(std.testing.allocator, test_json, &test_fp16, &test_fp16);
    defer cfg.deinit();
    const q = cfg.matrix("layers.0.q_proj") orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 0), q.weight_offset);
    try std.testing.expectEqual(@as(u32, 8), q.cols);
    const k = cfg.matrix("layers.0.k_proj") orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 64), k.weight_offset);
    try std.testing.expectEqual(@as(u32, 4), k.rows);
    try std.testing.expect(cfg.matrix("layers.9.down_proj") == null);
}

test "scales and glue decode as little-endian fp16, readGlue -> f32" {
    // 0x3C00 = 1.0, 0x3800 = 0.5, stored little-endian.
    var cfg = try Config.parse(std.testing.allocator, test_json, &test_fp16, &test_fp16);
    defer cfg.deinit();
    try std.testing.expectEqual(@as(u16, 0x3C00), cfg.scales[0]);
    try std.testing.expectEqual(@as(u16, 0x3800), cfg.scales[1]);
    var out: [2]f32 = undefined;
    cfg.readGlue(0, &out);
    try std.testing.expectEqual(@as(f32, 1.0), out[0]);
    try std.testing.expectEqual(@as(f32, 0.5), out[1]);
}

test "layer_glue q/k/v bias offsets are optional, and addGlue accumulates" {
    // Qwen2-style: layer 0 carries q/k/v bias offsets, layer 1 has none.
    const json =
        \\{ "name":"q","hidden":8,"vocab":16,"layers":2,
        \\  "num_heads":2,"num_kv_heads":1,"head_dim":4,"intermediate":12,
        \\  "rope_theta":10000.0,"norm_eps":0.00001,"tie_embeddings":true,
        \\  "csr_base":65536,"flash_weight_base":0,"scale_flash_base":0,
        \\  "embed_offset":0,"final_norm_offset":100,
        \\  "layer_glue":[ {"input_norm":10,"post_norm":20,"q_bias":0,"k_bias":1,"v_bias":0},
        \\                 {"input_norm":30,"post_norm":40} ],
        \\  "matrices":[] }
    ;
    var cfg = try Config.parse(std.testing.allocator, json, &test_fp16, &test_fp16);
    defer cfg.deinit();
    const m = cfg.manifest();
    try std.testing.expectEqual(@as(?u32, 0), m.layer_glue[0].q_bias);
    try std.testing.expectEqual(@as(?u32, 1), m.layer_glue[0].k_bias);
    try std.testing.expectEqual(@as(?u32, null), m.layer_glue[1].q_bias);
    // glue = {1.0, 0.5}; addGlue at offset 0 adds {1.0, 0.5} into out.
    var out = [_]f32{ 1.0, 1.0 };
    cfg.addGlue(m.layer_glue[0].q_bias.?, &out);
    try std.testing.expectEqual(@as(f32, 2.0), out[0]);
    try std.testing.expectEqual(@as(f32, 1.5), out[1]);
}

test "parse reads MoE config and per-layer router glue offsets" {
    // A MoE model: top-level moe block, layer 0 routed (router offset), layer 1
    // dense (no router). Dense models omit the moe block entirely (defaults null).
    const json =
        \\{ "name":"moe","hidden":8,"vocab":16,"layers":2,
        \\  "num_heads":2,"num_kv_heads":1,"head_dim":4,"intermediate":12,
        \\  "moe":{"num_experts":4,"top_k":2,"norm_topk":true,"moe_intermediate":6,"num_shared":0},
        \\  "rope_theta":10000.0,"norm_eps":0.00001,"tie_embeddings":true,
        \\  "csr_base":65536,"flash_weight_base":0,"scale_flash_base":0,
        \\  "embed_offset":0,"final_norm_offset":100,
        \\  "layer_glue":[ {"input_norm":10,"post_norm":20,"router":50},
        \\                 {"input_norm":30,"post_norm":40} ],
        \\  "matrices":[] }
    ;
    var cfg = try Config.parse(std.testing.allocator, json, &test_fp16, &test_fp16);
    defer cfg.deinit();
    const m = cfg.manifest();
    const moe = m.moe orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 4), moe.num_experts);
    try std.testing.expectEqual(@as(u32, 2), moe.top_k);
    try std.testing.expectEqual(true, moe.norm_topk);
    try std.testing.expectEqual(@as(u32, 6), moe.moe_intermediate);
    try std.testing.expectEqual(@as(?u32, 50), m.layer_glue[0].router);
    try std.testing.expectEqual(@as(?u32, null), m.layer_glue[1].router);
}

test "dense manifest leaves the moe block null" {
    var cfg = try Config.parse(std.testing.allocator, test_json, &test_fp16, &test_fp16);
    defer cfg.deinit();
    try std.testing.expectEqual(@as(?Moe, null), cfg.manifest().moe);
    try std.testing.expectEqual(@as(?u32, null), cfg.manifest().layer_glue[0].router);
}

test "parse reads MTP heads (num_modules + per-module norm glue offsets)" {
    const json =
        \\{ "name":"mtp","hidden":8,"vocab":16,"layers":2,
        \\  "num_heads":2,"num_kv_heads":1,"head_dim":4,"intermediate":12,
        \\  "mtp":{"num_modules":2,"modules":[
        \\    {"enorm":10,"hnorm":20,"input_norm":30,"post_norm":40},
        \\    {"enorm":50,"hnorm":60,"input_norm":70,"post_norm":80} ]},
        \\  "rope_theta":10000.0,"norm_eps":0.00001,"tie_embeddings":true,
        \\  "csr_base":65536,"flash_weight_base":0,"scale_flash_base":0,
        \\  "embed_offset":0,"final_norm_offset":100,
        \\  "layer_glue":[ {"input_norm":0,"post_norm":1},{"input_norm":2,"post_norm":3} ],
        \\  "matrices":[] }
    ;
    var cfg = try Config.parse(std.testing.allocator, json, &test_fp16, &test_fp16);
    defer cfg.deinit();
    const m = cfg.manifest();
    const mtp = m.mtp orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 2), mtp.num_modules);
    try std.testing.expectEqual(@as(usize, 2), mtp.modules.len);
    try std.testing.expectEqual(@as(u32, 10), mtp.modules[0].enorm);
    try std.testing.expectEqual(@as(u32, 40), mtp.modules[0].post_norm);
    try std.testing.expectEqual(@as(u32, 60), mtp.modules[1].hnorm);
    // A dense manifest leaves mtp null.
    var dense = try Config.parse(std.testing.allocator, test_json, &test_fp16, &test_fp16);
    defer dense.deinit();
    try std.testing.expectEqual(@as(?Mtp, null), dense.manifest().mtp);
}

test "parse reads the vision tower + projector + image token" {
    const json =
        \\{ "name":"vlm","hidden":8,"vocab":16,"layers":1,
        \\  "num_heads":2,"num_kv_heads":2,"head_dim":4,"intermediate":16,
        \\  "vision":{"image_size":4,"patch_size":2,"num_channels":3,"hidden":8,
        \\    "num_layers":1,"num_heads":2,"head_dim":4,"intermediate":16,
        \\    "layer_norm_eps":0.00001,"has_class_token":true,"seq_len":5,
        \\    "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
        \\    "patch_embed_bias":10,"class_token":20,"pos_embed":30,
        \\    "pre_ln_gamma":40,"pre_ln_beta":48,"post_ln_gamma":56,"post_ln_beta":64,
        \\    "layers":[{"ln1_gamma":0,"ln1_beta":1,"q_bias":2,"k_bias":3,"v_bias":4,
        \\      "o_bias":5,"ln2_gamma":6,"ln2_beta":7,"fc1_bias":8,"fc2_bias":9}]},
        \\  "projector":{"num_layers":2,"input_dim":8,"hidden_dim":8,"output_dim":8,
        \\    "bias1":70,"bias2":78},
        \\  "image_token_index":7,
        \\  "rope_theta":10000.0,"norm_eps":0.00001,"tie_embeddings":true,
        \\  "csr_base":65536,"flash_weight_base":0,"scale_flash_base":0,
        \\  "embed_offset":0,"final_norm_offset":100,
        \\  "layer_glue":[{"input_norm":0,"post_norm":1}],"matrices":[] }
    ;
    var cfg = try Config.parse(std.testing.allocator, json, &test_fp16, &test_fp16);
    defer cfg.deinit();
    const m = cfg.manifest();
    const vis = m.vision orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 4), vis.image_size);
    try std.testing.expectEqual(@as(u32, 5), vis.seq_len);
    try std.testing.expectEqual(true, vis.has_class_token);
    try std.testing.expectEqual(@as(?u32, 20), vis.class_token);
    try std.testing.expectEqual(@as(?u32, 40), vis.pre_ln_gamma);
    try std.testing.expectEqual(@as(usize, 1), vis.layers.len);
    try std.testing.expectEqual(@as(u32, 8), vis.layers[0].fc1_bias);
    const proj = m.projector orelse return error.TestUnexpectedNull;
    try std.testing.expectEqual(@as(u32, 2), proj.num_layers);
    try std.testing.expectEqual(@as(?u32, 78), proj.bias2);
    try std.testing.expectEqual(@as(?u32, 7), m.image_token_index);
    // A text-only manifest leaves them null.
    var dense = try Config.parse(std.testing.allocator, test_json, &test_fp16, &test_fp16);
    defer dense.deinit();
    try std.testing.expectEqual(@as(?Vision, null), dense.manifest().vision);
    try std.testing.expectEqual(@as(?Projector, null), dense.manifest().projector);
}

test "matvecGlue computes a row-major fp16 matmul (router)" {
    // glue = {1.0, 0.5} repeated; build a 2x2 router: rows [1.0,0.5] and [0.5,1.0].
    // 0x3C00=1.0, 0x3800=0.5.
    const glue = [_]u8{ 0x00, 0x3C, 0x00, 0x38, 0x00, 0x38, 0x00, 0x3C };
    var cfg = try Config.parse(std.testing.allocator, test_json, &test_fp16, &glue);
    defer cfg.deinit();
    const x = [_]f32{ 2.0, 4.0 };
    var out: [2]f32 = undefined;
    cfg.matvecGlue(0, 2, 2, &x, &out);
    try std.testing.expectEqual(@as(f32, 1.0 * 2.0 + 0.5 * 4.0), out[0]); // 4.0
    try std.testing.expectEqual(@as(f32, 0.5 * 2.0 + 1.0 * 4.0), out[1]); // 5.0
}
