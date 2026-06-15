//! llama2.c tokenizer (tok512.bin) decode - the Zig port of the dart
//! Llama2cTokenizer. Byte format: `i32 max_token_length`, then `vocab_size`
//! entries of `{f32 score, i32 len, byte[len] piece}`. SentencePiece whitespace
//! (U+2581 "▁") renders as a space; BOS/EOS render empty; the leading space of
//! the first piece after BOS is dropped; a "<0xXX>" piece is a raw byte.

const std = @import("std");
const bpe_mod = @import("bpe.zig");

/// The 3-byte UTF-8 encoding of U+2581 ("▁"), SentencePiece's space marker.
const space_marker = [_]u8{ 0xE2, 0x96, 0x81 };

/// A tokenizer of either format, chosen by sniffing the image magic: "LTB1" ->
/// ByteLevel BPE (genip), otherwise the magic-less llama2.c SentencePiece
/// `tok512`. `bos`/`eos` are the SentencePiece defaults; BPE carries its own.
pub const Any = union(enum) {
    sentencepiece: Tokenizer,
    bpe: bpe_mod.Bpe,

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, bos: u32, eos: u32) !Any {
        if (bytes.len >= 4 and std.mem.eql(u8, bytes[0..4], "LTB1"))
            return .{ .bpe = try bpe_mod.Bpe.parse(gpa, bytes) };
        return .{ .sentencepiece = try Tokenizer.parse(gpa, bytes, bos, eos) };
    }

    pub fn deinit(self: *Any) void {
        switch (self.*) {
            .sentencepiece => |*t| t.deinit(),
            .bpe => |*t| t.deinit(),
        }
    }

    pub fn encode(self: *Any, gpa: std.mem.Allocator, text: []const u8, add_bos: bool) ![]u32 {
        return switch (self.*) {
            .sentencepiece => |*t| t.encode(gpa, text, add_bos),
            .bpe => |*t| t.encode(gpa, text, add_bos),
        };
    }

    pub fn decode(self: *Any, ids: []const u32, gpa: std.mem.Allocator) ![]u8 {
        return switch (self.*) {
            .sentencepiece => |*t| t.decode(ids, gpa),
            .bpe => |*t| t.decode(ids, gpa),
        };
    }

    /// Appends one token's display bytes. `prev` is used only by SentencePiece
    /// (the BOS leading-space rule); BPE ignores it.
    pub fn render(self: *Any, gpa: std.mem.Allocator, out: *std.ArrayList(u8), id: u32, prev: u32) !void {
        return switch (self.*) {
            .sentencepiece => |*t| t.renderToken(gpa, out, id, prev),
            .bpe => |*t| t.render(gpa, out, id),
        };
    }

    pub fn bosId(self: *const Any) u32 {
        return switch (self.*) {
            .sentencepiece => |t| t.bos,
            .bpe => |t| if (t.bos >= 0) @intCast(t.bos) else 0,
        };
    }

    pub fn eosId(self: *const Any) u32 {
        return switch (self.*) {
            .sentencepiece => |t| t.eos,
            .bpe => |t| if (t.eos >= 0) @intCast(t.eos) else 0,
        };
    }

    pub fn addBos(self: *const Any) bool {
        return switch (self.*) {
            .sentencepiece => true,
            .bpe => |t| t.add_bos,
        };
    }
};

/// Length of the longest prefix of `b` that ends on a UTF-8 codepoint boundary.
pub fn completeUtf8Prefix(b: []const u8) usize {
    var i: usize = 0;
    while (i < b.len) {
        const clen = std.unicode.utf8ByteSequenceLength(b[i]) catch {
            i += 1;
            continue;
        };
        if (i + clen > b.len) break; // incomplete trailing sequence
        i += clen;
    }
    return i;
}

pub const Tokenizer = struct {
    /// Token id -> piece bytes (▁ normalized to ' '). Each piece and the outer
    /// slice are owned.
    vocab: [][]u8,
    /// Token id -> merge score (SentencePiece log-prob; higher wins a merge).
    /// Parallel to `vocab`; owned.
    scores: []f32,
    bos: u32,
    eos: u32,
    gpa: std.mem.Allocator,

    /// Parses a tok512.bin image. Caller must `deinit`. Borrows nothing.
    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, bos: u32, eos: u32) !Tokenizer {
        var pieces: std.ArrayList([]u8) = .empty;
        errdefer {
            for (pieces.items) |p| gpa.free(p);
            pieces.deinit(gpa);
        }
        var scores: std.ArrayList(f32) = .empty;
        errdefer scores.deinit(gpa);
        var off: usize = 4; // skip i32 max_token_length
        while (off + 8 <= bytes.len) {
            const score: f32 = @bitCast(std.mem.readInt(u32, bytes[off..][0..4], .little));
            off += 4;
            const len: usize = @intCast(std.mem.readInt(i32, bytes[off..][0..4], .little));
            off += 4;
            if (len == 0 or off + len > bytes.len) {
                try pieces.append(gpa, try gpa.dupe(u8, ""));
                try scores.append(gpa, score);
                if (len == 0) continue else break;
            }
            const piece = try normalizeMarker(gpa, bytes[off .. off + len]);
            try pieces.append(gpa, piece);
            try scores.append(gpa, score);
            off += len;
        }
        return .{
            .vocab = try pieces.toOwnedSlice(gpa),
            .scores = try scores.toOwnedSlice(gpa),
            .bos = bos,
            .eos = eos,
            .gpa = gpa,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        for (self.vocab) |p| self.gpa.free(p);
        self.gpa.free(self.vocab);
        self.gpa.free(self.scores);
    }

    /// Id of the token whose piece exactly equals `s`, or null.
    fn lookup(self: Tokenizer, s: []const u8) ?u32 {
        for (self.vocab, 0..) |p, i| {
            if (std.mem.eql(u8, p, s)) return @intCast(i);
        }
        return null;
    }

    /// Encodes `text` to token ids the llama2.c way: optional BOS, a dummy
    /// leading-space token, per-codepoint vocab lookup with `<0xXX>` byte
    /// fallback (id = byte + 3), then greedy highest-score adjacent-pair merges.
    /// Returns an owned slice; caller frees.
    pub fn encode(self: Tokenizer, gpa: std.mem.Allocator, text: []const u8, add_bos: bool) ![]u32 {
        var toks: std.ArrayList(u32) = .empty;
        errdefer toks.deinit(gpa);
        if (add_bos) try toks.append(gpa, self.bos);
        // SentencePiece prepends a space to non-empty input.
        if (text.len > 0) {
            if (self.lookup(" ")) |id| try toks.append(gpa, id);
        }
        // Split into UTF-8 codepoints; look each up, else fall back to raw bytes.
        var i: usize = 0;
        while (i < text.len) {
            var clen: usize = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
            if (i + clen > text.len) clen = 1;
            const piece = text[i .. i + clen];
            if (self.lookup(piece)) |id| {
                try toks.append(gpa, id);
            } else {
                for (piece) |b| try toks.append(gpa, @as(u32, b) + 3);
            }
            i += clen;
        }
        // Greedily fold the adjacent pair whose concatenation is the
        // highest-scoring vocab entry, until none is left.
        var buf: [256]u8 = undefined;
        while (true) {
            var best_score: f32 = -std.math.inf(f32);
            var best_id: u32 = 0;
            var best_at: ?usize = null;
            var j: usize = 0;
            while (j + 1 < toks.items.len) : (j += 1) {
                const a = toks.items[j];
                const b = toks.items[j + 1];
                if (a >= self.vocab.len or b >= self.vocab.len) continue;
                const ap = self.vocab[a];
                const bp = self.vocab[b];
                if (ap.len + bp.len > buf.len) continue;
                @memcpy(buf[0..ap.len], ap);
                @memcpy(buf[ap.len..][0..bp.len], bp);
                if (self.lookup(buf[0 .. ap.len + bp.len])) |id| {
                    if (self.scores[id] > best_score) {
                        best_score = self.scores[id];
                        best_id = id;
                        best_at = j;
                    }
                }
            }
            const at = best_at orelse break;
            toks.items[at] = best_id;
            _ = toks.orderedRemove(at + 1);
        }
        return toks.toOwnedSlice(gpa);
    }

    /// Appends one token's display text to `out`, given `prev` (the preceding
    /// token, for the BOS leading-space rule; pass maxInt(u32) if none). BOS/EOS
    /// emit nothing; the leading space right after BOS is dropped; a "<0xXX>"
    /// piece becomes its raw byte. This is the per-token unit both `decode` and
    /// live streaming render through.
    pub fn renderToken(self: Tokenizer, gpa: std.mem.Allocator, out: *std.ArrayList(u8), t: u32, prev: u32) !void {
        if (t == self.bos or t == self.eos or t >= self.vocab.len) return;
        var piece = self.vocab[t];
        if (prev == self.bos and piece.len > 0 and piece[0] == ' ') piece = piece[1..];
        if (byteFallback(piece)) |b| {
            try out.append(gpa, b);
        } else {
            try out.appendSlice(gpa, piece);
        }
    }

    /// Decodes `tokens` to an owned string via `renderToken`.
    pub fn decode(self: Tokenizer, tokens: []const u32, gpa: std.mem.Allocator) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        for (tokens, 0..) |t, i| {
            const prev = if (i > 0) tokens[i - 1] else std.math.maxInt(u32);
            try self.renderToken(gpa, &out, t, prev);
        }
        return out.toOwnedSlice(gpa);
    }
};

/// Copies `src` with the SentencePiece space marker replaced by ' '.
fn normalizeMarker(gpa: std.mem.Allocator, src: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    while (i < src.len) {
        if (i + 3 <= src.len and std.mem.eql(u8, src[i .. i + 3], &space_marker)) {
            try out.append(gpa, ' ');
            i += 3;
        } else {
            try out.append(gpa, src[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(gpa);
}

/// If `piece` is a "<0xXX>" byte-fallback token, returns the byte value.
fn byteFallback(piece: []const u8) ?u8 {
    if (piece.len != 6 or !std.mem.startsWith(u8, piece, "<0x") or piece[5] != '>') return null;
    const hi = std.fmt.charToDigit(piece[3], 16) catch return null;
    const lo = std.fmt.charToDigit(piece[4], 16) catch return null;
    return hi * 16 + lo;
}

// Builds a minimal tok512-format image: max_token_length then the given pieces.
fn buildTestImage(gpa: std.mem.Allocator, pieces: []const []const u8) ![]u8 {
    var b: std.ArrayList(u8) = .empty;
    errdefer b.deinit(gpa);
    try b.appendSlice(gpa, &[_]u8{ 8, 0, 0, 0 }); // max_token_length
    for (pieces) |p| {
        try b.appendSlice(gpa, &[_]u8{ 0, 0, 0, 0 }); // score
        var lb: [4]u8 = undefined;
        std.mem.writeInt(i32, &lb, @intCast(p.len), .little);
        try b.appendSlice(gpa, &lb);
        try b.appendSlice(gpa, p);
    }
    return b.toOwnedSlice(gpa);
}

test "decode drops the leading space after BOS and skips BOS/EOS" {
    const gpa = std.testing.allocator;
    // token 0 <unk>, 1 <s> (BOS), 2 </s> (EOS), 3 "▁Once", 4 " a".
    const pieces = [_][]const u8{ "<unk>", "\n<s>\n", "\n</s>\n", "\xE2\x96\x81Once", "\xE2\x96\x81a" };
    const img = try buildTestImage(gpa, &pieces);
    defer gpa.free(img);
    var tok = try Tokenizer.parse(gpa, img, 1, 2);
    defer tok.deinit();

    const text = try tok.decode(&[_]u32{ 1, 3, 4 }, gpa); // <s> ▁Once ▁a
    defer gpa.free(text);
    try std.testing.expectEqualStrings("Once a", text); // BOS strips ▁Once's space
}

// Builds a tok512-format image with explicit per-piece scores.
fn buildScoredImage(gpa: std.mem.Allocator, pieces: []const []const u8, scores: []const f32) ![]u8 {
    var b: std.ArrayList(u8) = .empty;
    errdefer b.deinit(gpa);
    try b.appendSlice(gpa, &[_]u8{ 8, 0, 0, 0 }); // max_token_length
    for (pieces, scores) |p, s| {
        var sb: [4]u8 = undefined;
        std.mem.writeInt(u32, &sb, @bitCast(s), .little);
        try b.appendSlice(gpa, &sb);
        var lb: [4]u8 = undefined;
        std.mem.writeInt(i32, &lb, @intCast(p.len), .little);
        try b.appendSlice(gpa, &lb);
        try b.appendSlice(gpa, p);
    }
    return b.toOwnedSlice(gpa);
}

test "encode adds BOS + dummy space, then merges the best-scoring pair" {
    const gpa = std.testing.allocator;
    // ids: 0 <unk>, 1 <s>, 2 </s>, 3 " ", 4 "a", 5 "b", 6 "ab", 7 " a".
    const pieces = [_][]const u8{ "<unk>", "\n<s>\n", "\n</s>\n", " ", "a", "b", "ab", " a" };
    const scores = [_]f32{ 0, 0, 0, 0, 0, 0, 1.0, 0.5 }; // "ab" beats " a"
    const img = try buildScoredImage(gpa, &pieces, &scores);
    defer gpa.free(img);
    var tok = try Tokenizer.parse(gpa, img, 1, 2);
    defer tok.deinit();

    // "ab" -> [BOS, " ", a, b] -> merge a+b -> [BOS, " ", "ab"].
    const ids = try tok.encode(gpa, "ab", true);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 1, 3, 6 }, ids);

    // Round-trips back to the input (BOS drops the dummy space).
    const text = try tok.decode(ids, gpa);
    defer gpa.free(text);
    try std.testing.expectEqualStrings("ab", text);
}

test "encode without BOS keeps the leading space piece" {
    const gpa = std.testing.allocator;
    const pieces = [_][]const u8{ "<unk>", "\n<s>\n", "\n</s>\n", " ", "a", "b", "ab", " a" };
    const scores = [_]f32{ 0, 0, 0, 0, 0, 0, 1.0, 0.5 };
    const img = try buildScoredImage(gpa, &pieces, &scores);
    defer gpa.free(img);
    var tok = try Tokenizer.parse(gpa, img, 1, 2);
    defer tok.deinit();

    const ids = try tok.encode(gpa, "ab", false);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 3, 6 }, ids); // " ", "ab"
    const text = try tok.decode(ids, gpa);
    defer gpa.free(text);
    try std.testing.expectEqualStrings(" ab", text); // no BOS -> space kept
}

test "Any sniffs LTB1 as bpe and magic-less as sentencepiece" {
    const gpa = std.testing.allocator;
    // Magic-less tok512 -> sentencepiece.
    const sp_img = try buildTestImage(gpa, &[_][]const u8{ "<unk>", "\n<s>\n", "\n</s>\n", "\xE2\x96\x81Hi" });
    defer gpa.free(sp_img);
    var sp = try Any.parse(gpa, sp_img, 1, 2);
    defer sp.deinit();
    try std.testing.expect(sp == .sentencepiece);
    const t = try sp.decode(&[_]u32{ 1, 3 }, gpa);
    defer gpa.free(t);
    try std.testing.expectEqualStrings("Hi", t);

    // LTB1 magic -> bpe.
    var img: std.ArrayList(u8) = .empty;
    defer img.deinit(gpa);
    try img.appendSlice(gpa, "LTB1");
    try img.append(gpa, 0x01);
    try img.appendSlice(gpa, &[_]u8{ 0, 0, 0, 0 }); // bos 0
    try img.appendSlice(gpa, &[_]u8{ 0, 0, 0, 0 }); // eos 0
    try img.append(gpa, 0); // add_bos
    try img.appendSlice(gpa, &[_]u8{ 1, 0, 0, 0 }); // vocab_count 1
    try img.appendSlice(gpa, &[_]u8{ 1, 0 }); // len 1
    try img.appendSlice(gpa, "a");
    try img.appendSlice(gpa, &[_]u8{ 0, 0, 0, 0 }); // merges 0
    try img.appendSlice(gpa, &[_]u8{ 0, 0, 0, 0 }); // specials 0
    var bp = try Any.parse(gpa, img.items, 1, 2);
    defer bp.deinit();
    try std.testing.expect(bp == .bpe);
}

test "decode expands a <0xXX> byte-fallback token" {
    const gpa = std.testing.allocator;
    const pieces = [_][]const u8{ "<unk>", "\n<s>\n", "\n</s>\n", "<0x0A>" };
    const img = try buildTestImage(gpa, &pieces);
    defer gpa.free(img);
    var tok = try Tokenizer.parse(gpa, img, 1, 2);
    defer tok.deinit();

    const text = try tok.decode(&[_]u32{3}, gpa);
    defer gpa.free(text);
    try std.testing.expectEqualSlices(u8, "\n", text);
}
