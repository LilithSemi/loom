//! The transformer's nonlinear glue in f32 (pure, no IO). These mirror the dart
//! golden ops (ip/lib/src/golden/ops.dart + attention.dart) and run host-side;
//! only the matmuls go to the device.

const std = @import("std");

/// RMSNorm: `out[i] = x[i] / sqrt(mean(x^2) + eps) * gamma[i]`. Slices are
/// equal length.
pub fn rmsNorm(x: []const f32, gamma: []const f32, eps: f32, out: []f32) void {
    std.debug.assert(x.len == gamma.len and x.len == out.len);
    var sum_sq: f32 = 0;
    for (x) |v| sum_sq += v * v;
    const mean_sq = sum_sq / @as(f32, @floatFromInt(x.len));
    const inv = 1.0 / @sqrt(mean_sq + eps);
    for (x, gamma, out) |v, g, *o| o.* = v * inv * g;
}

/// SiLU in place: `x[i] = x[i] * sigmoid(x[i])`.
pub fn siluInPlace(x: []f32) void {
    for (x) |*v| v.* = v.* / (1.0 + @exp(-v.*));
}

/// LayerNorm: normalize `x` to zero mean / unit variance (biased, /N) then scale
/// by `gamma` and shift by `beta` into `out`. Used by Vision Transformer towers
/// (LayerNorm rather than the LLM's RMSNorm). Mirrors golden/ops.dart layerNorm.
pub fn layerNorm(x: []const f32, gamma: []const f32, beta: []const f32, eps: f32, out: []f32) void {
    std.debug.assert(x.len == gamma.len and x.len == beta.len and x.len == out.len);
    const n: f32 = @floatFromInt(x.len);
    var mean: f32 = 0;
    for (x) |v| mean += v;
    mean /= n;
    var variance: f32 = 0;
    for (x) |v| {
        const d = v - mean;
        variance += d * d;
    }
    variance /= n;
    const inv = 1.0 / @sqrt(variance + eps);
    for (x, gamma, beta, out) |v, g, b, *o| o.* = (v - mean) * inv * g + b;
}

/// GELU in place, tanh approximation (`gelu_pytorch_tanh`, as SigLIP and most
/// ViTs use): `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`. Mirrors
/// golden/ops.dart gelu.
pub fn geluInPlace(x: []f32) void {
    const c: f32 = 0.7978845608028654; // sqrt(2/pi)
    for (x) |*v| {
        const val = v.*;
        const inner = c * (val + 0.044715 * val * val * val);
        const t = std.math.tanh(inner);
        v.* = 0.5 * val * (1.0 + t);
    }
}

/// Elementwise `dst[i] += src[i]`. Used for bias-add (Qwen2 q/k/v projections).
pub fn addInPlace(dst: []f32, src: []const f32) void {
    for (dst, src) |*d, s| d.* += s;
}

/// Softmax in place over `x`, numerically stable (subtract the max).
pub fn softmaxInPlace(x: []f32) void {
    if (x.len == 0) return;
    var max_v: f32 = x[0];
    for (x) |v| max_v = @max(max_v, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - max_v);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

/// Largest expert count `moeRoute` supports (DeepSeek-V3 tops out at 256).
pub const MAX_EXPERTS = 512;

/// Mixture-of-Experts router selection. Softmaxes `logits` over its experts,
/// picks the `top_k` highest, and writes their indices to `idx[0..top_k]` and
/// (optionally renormalized) combine weights to `w[0..top_k]`. `idx` and `w`
/// must each hold at least `top_k` elements; `top_k` is clamped to the expert
/// count. Mirrors golden/moe.dart's routing exactly. Returns the number of
/// experts selected (`min(top_k, logits.len)`).
pub fn moeRoute(logits: []const f32, top_k: usize, norm_topk: bool, idx: []usize, w: []f32) usize {
    const ne = logits.len;
    std.debug.assert(ne <= MAX_EXPERTS);
    const k = @min(top_k, ne);
    if (k == 0) return 0;

    // Numerically-stable softmax over all experts.
    var probs: [MAX_EXPERTS]f32 = undefined;
    var maxl: f32 = logits[0];
    for (logits) |v| maxl = @max(maxl, v);
    var sum: f32 = 0;
    for (0..ne) |i| {
        probs[i] = @exp(logits[i] - maxl);
        sum += probs[i];
    }
    for (0..ne) |i| probs[i] /= sum;

    // Top-k by probability (partial selection; ne is small).
    var taken = [_]bool{false} ** MAX_EXPERTS;
    for (0..k) |j| {
        var best: usize = 0;
        var bestp: f32 = -1;
        for (0..ne) |i| {
            if (!taken[i] and probs[i] > bestp) {
                bestp = probs[i];
                best = i;
            }
        }
        taken[best] = true;
        idx[j] = best;
    }

    // Combine weights = the chosen probs, renormalized to sum 1 (norm_topk).
    var wsum: f32 = 0;
    for (0..k) |j| wsum += probs[idx[j]];
    for (0..k) |j| {
        w[j] = probs[idx[j]];
        if (norm_topk and wsum > 0) w[j] /= wsum;
    }
    return k;
}

/// Applies RoPE to one attention head in place (HuggingFace half-split pairing:
/// coordinate `j` pairs with `j + head.len/2`). `head.len` must be even.
pub fn ropeHead(head: []f32, pos: usize, theta: f32) void {
    const hd = head.len;
    std.debug.assert(hd % 2 == 0);
    const half = hd / 2;
    const hd_f: f32 = @floatFromInt(hd);
    const pos_f: f32 = @floatFromInt(pos);
    for (0..half) |j| {
        const inv_freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * j)) / hd_f);
        const angle = pos_f * inv_freq;
        const c = @cos(angle);
        const s = @sin(angle);
        const x1 = head[j];
        const x2 = head[j + half];
        head[j] = x1 * c - x2 * s;
        head[j + half] = x2 * c + x1 * s;
    }
}

test "rmsNorm normalizes to unit RMS then scales by gamma" {
    const x = [_]f32{ 1.0, 2.0 };
    const gamma = [_]f32{ 1.0, 1.0 };
    var out: [2]f32 = undefined;
    rmsNorm(&x, &gamma, 0.0, &out);
    // mean(x^2)=2.5, inv=1/sqrt(2.5).
    const inv = 1.0 / @sqrt(@as(f32, 2.5));
    try std.testing.expectApproxEqAbs(1.0 * inv, out[0], 1e-6);
    try std.testing.expectApproxEqAbs(2.0 * inv, out[1], 1e-6);
}

test "siluInPlace: silu(0)=0 and large positive approaches identity" {
    var v = [_]f32{ 0.0, 20.0, -20.0 };
    siluInPlace(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), v[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v[2], 1e-6);
}

test "softmaxInPlace sums to one and is uniform for equal inputs" {
    var v = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    softmaxInPlace(&v);
    var sum: f32 = 0;
    for (v) |x| sum += x;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), v[0], 1e-6);
}

test "layerNorm normalizes to zero-mean/unit-var then scales+shifts" {
    const x = [_]f32{ 1, 2, 3, 4 };
    const gamma = [_]f32{ 1, 1, 1, 1 };
    const beta = [_]f32{ 0, 0, 0, 0 };
    var out: [4]f32 = undefined;
    layerNorm(&x, &gamma, &beta, 0.0, &out);
    // mean 2.5, var 1.25, inv = 1/sqrt(1.25).
    const inv = 1.0 / @sqrt(@as(f32, 1.25));
    try std.testing.expectApproxEqAbs(-1.5 * inv, out[0], 1e-6);
    try std.testing.expectApproxEqAbs(1.5 * inv, out[3], 1e-6);
    var sum: f32 = 0;
    for (out) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum, 1e-6);
}

test "geluInPlace: gelu(0)=0, matches the tanh approximation, saturates" {
    var v = [_]f32{ 0, 1, -1, 30 };
    geluInPlace(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8411919906082768), v[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.15880800939172324), v[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), v[3], 1e-4);
}

test "moeRoute picks the top-k experts and renormalizes their weights" {
    // logits rank experts 0>1>2>3; top-2 = {0,1}. With norm_topk the two chosen
    // softmax probs are rescaled to sum 1.
    const logits = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    var idx: [2]usize = undefined;
    var w: [2]f32 = undefined;
    const k = moeRoute(&logits, 2, true, &idx, &w);
    try std.testing.expectEqual(@as(usize, 2), k);
    try std.testing.expectEqual(@as(usize, 0), idx[0]);
    try std.testing.expectEqual(@as(usize, 1), idx[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), w[0] + w[1], 1e-6);
    // p0/p1 = e^1, so w0/w1 = e/(e+1), w1 = 1/(e+1).
    const e = @exp(@as(f32, 1.0));
    try std.testing.expectApproxEqAbs(e / (e + 1.0), w[0], 1e-5);
    try std.testing.expectApproxEqAbs(1.0 / (e + 1.0), w[1], 1e-5);
}

test "moeRoute without norm_topk keeps the raw softmax probs" {
    const logits = [_]f32{ 0.0, 0.0, 0.0, 0.0 }; // uniform -> each prob 0.25
    var idx: [2]usize = undefined;
    var w: [2]f32 = undefined;
    _ = moeRoute(&logits, 2, false, &idx, &w);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), w[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), w[1], 1e-6);
}

test "ropeHead leaves position 0 unchanged (angle 0)" {
    var head = [_]f32{ 1.0, -2.0, 3.0, 0.5 };
    ropeHead(&head, 0, 10000.0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, -2.0, 3.0, 0.5 }, &head);
}

test "regression: ropeHead rotates the (j, j+half) pair, not (j, j+1)" {
    // half-split: at pos 1, j=0 uses inv_freq=1 -> angle=1 rad; pair is (head[0],
    // head[2]). A (j,j+1) implementation would (wrongly) touch head[1] first.
    var head = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    ropeHead(&head, 1, 10000.0);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1.0)), head[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), head[1], 1e-6); // untouched
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1.0)), head[2], 1e-6);
}
