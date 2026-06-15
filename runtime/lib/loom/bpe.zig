//! HuggingFace ByteLevel BPE tokenizer, the runtime port of the Dart
//! `BpeTokenizer` (ip/lib/src/tokenizer/bpe_tokenizer.dart). Reads genip's
//! LTB1 tokenizer.bin. See the BPE plan for the format.

const std = @import("std");

/// A merge key: the two vocab ids whose pieces concatenate. Ranked by order.
const Pair = struct { a: u32, b: u32 };
const PairCtx = struct {
    pub fn hash(_: PairCtx, p: Pair) u64 {
        return (@as(u64, p.a) << 32) | p.b;
    }
    pub fn eql(_: PairCtx, x: Pair, y: Pair) bool {
        return x.a == y.a and x.b == y.b;
    }
};

pub const Bpe = struct {
    gpa: std.mem.Allocator,
    /// id -> byte-level-unicode key bytes (owned, parallel outer slice owned).
    vocab: [][]u8,
    /// key bytes -> id.
    ids: std.StringHashMap(u32),
    /// (id_a,id_b) -> merge rank (lower merges first).
    ranks: std.HashMap(Pair, u32, PairCtx, std.hash_map.default_max_load_percentage),
    /// special content bytes -> id, and id -> content (owned content).
    special_ids: std.StringHashMap(u32),
    special_content: std.AutoHashMap(u32, []u8),
    /// byte value -> its UTF-8 codepoint bytes in the GPT-2 alphabet (owned).
    byte_to_uni: [256][]u8,
    /// codepoint -> byte, for decode (cps stay < 0x200).
    uni_to_byte: [0x200]i16,
    bos: i32,
    eos: i32,
    add_bos: bool,
    split_digits: bool,
    ignore_merges: bool,

    const Reader = struct {
        b: []const u8,
        p: usize = 0,
        fn u8v(r: *Reader) u8 {
            const v = r.b[r.p];
            r.p += 1;
            return v;
        }
        fn u16v(r: *Reader) u16 {
            const v = std.mem.readInt(u16, r.b[r.p..][0..2], .little);
            r.p += 2;
            return v;
        }
        fn u32v(r: *Reader) u32 {
            const v = std.mem.readInt(u32, r.b[r.p..][0..4], .little);
            r.p += 4;
            return v;
        }
        fn i32v(r: *Reader) i32 {
            const v = std.mem.readInt(i32, r.b[r.p..][0..4], .little);
            r.p += 4;
            return v;
        }
        fn bytes(r: *Reader, n: usize) []const u8 {
            const s = r.b[r.p .. r.p + n];
            r.p += n;
            return s;
        }
    };

    pub fn parse(gpa: std.mem.Allocator, image: []const u8) !Bpe {
        if (image.len < 4 or !std.mem.eql(u8, image[0..4], "LTB1"))
            return error.NotLtb1;
        var r = Reader{ .b = image, .p = 4 };
        const flags = r.u8v();
        const bos = r.i32v();
        const eos = r.i32v();
        const add_bos = r.u8v() != 0;

        var self = Bpe{
            .gpa = gpa,
            .vocab = &.{},
            .ids = std.StringHashMap(u32).init(gpa),
            .ranks = std.HashMap(Pair, u32, PairCtx, std.hash_map.default_max_load_percentage).init(gpa),
            .special_ids = std.StringHashMap(u32).init(gpa),
            .special_content = std.AutoHashMap(u32, []u8).init(gpa),
            .byte_to_uni = [_][]u8{&.{}} ** 256,
            .uni_to_byte = [_]i16{-1} ** 0x200,
            .bos = bos,
            .eos = eos,
            .add_bos = add_bos,
            .split_digits = (flags & 0x01) != 0,
            .ignore_merges = (flags & 0x02) != 0,
        };
        errdefer self.deinit();
        try self.buildAlphabet();

        const vn = r.u32v();
        const vocab = try gpa.alloc([]u8, vn);
        self.vocab = vocab;
        for (vocab) |*v| v.* = &.{};
        var i: u32 = 0;
        while (i < vn) : (i += 1) {
            const l = r.u16v();
            const key = try gpa.dupe(u8, r.bytes(l));
            vocab[i] = key;
            try self.ids.put(key, i);
        }
        const mn = r.u32v();
        var m: u32 = 0;
        while (m < mn) : (m += 1) {
            const a = r.u32v();
            const b = r.u32v();
            try self.ranks.put(.{ .a = a, .b = b }, m);
        }
        const sn = r.u32v();
        var s: u32 = 0;
        while (s < sn) : (s += 1) {
            const l = r.u16v();
            const content = try gpa.dupe(u8, r.bytes(l));
            const id = r.u32v();
            try self.special_ids.put(content, id);
            try self.special_content.put(id, content);
        }
        return self;
    }

    pub fn deinit(self: *Bpe) void {
        for (self.vocab) |v| if (v.len > 0) self.gpa.free(v);
        if (self.vocab.len > 0) self.gpa.free(self.vocab);
        self.ids.deinit();
        self.ranks.deinit();
        self.special_ids.deinit();
        var it = self.special_content.valueIterator();
        while (it.next()) |v| self.gpa.free(v.*);
        self.special_content.deinit();
        for (self.byte_to_uni) |u| if (u.len > 0) self.gpa.free(u);
    }

    /// The fixed GPT-2 byte-to-unicode map (matches `_buildByteToUnicode`).
    fn buildAlphabet(self: *Bpe) !void {
        var used = [_]bool{false} ** 256;
        var cps: [256]u21 = undefined;
        // Printable ranges map to themselves.
        inline for (.{ .{ '!', '~' }, .{ 0xA1, 0xAC }, .{ 0xAE, 0xFF } }) |rg| {
            var c: u21 = rg[0];
            while (c <= rg[1]) : (c += 1) {
                used[@intCast(c)] = true;
                cps[@intCast(c)] = c;
            }
        }
        var n: u21 = 0;
        var b: u16 = 0;
        while (b < 256) : (b += 1) {
            if (!used[b]) {
                cps[b] = 256 + n;
                n += 1;
            }
        }
        for (0..256) |byte| {
            var buf: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(cps[byte], &buf) catch unreachable;
            self.byte_to_uni[byte] = try self.gpa.dupe(u8, buf[0..len]);
            self.uni_to_byte[cps[byte]] = @intCast(byte);
        }
    }

    /// Appends one id's display bytes to `out`. A special id appends its
    /// content, a vocab id appends its bytes mapped back through the
    /// alphabet, and an unknown id appends nothing.
    pub fn render(self: *Bpe, gpa: std.mem.Allocator, out: *std.ArrayList(u8), id: u32) !void {
        if (self.special_content.get(id)) |content| {
            try out.appendSlice(gpa, content);
            return;
        }
        if (id >= self.vocab.len) return;
        const key = self.vocab[id];
        var i: usize = 0;
        while (i < key.len) {
            var clen: usize = std.unicode.utf8ByteSequenceLength(key[i]) catch 1;
            if (i + clen > key.len) clen = 1;
            const cp = std.unicode.utf8Decode(key[i .. i + clen]) catch {
                i += 1;
                continue;
            };
            if (cp < 0x200 and self.uni_to_byte[cp] >= 0)
                try out.append(gpa, @intCast(self.uni_to_byte[cp]));
            i += clen;
        }
    }

    /// Decodes ids to an owned byte string (matches `BpeTokenizer.decode`).
    pub fn decode(self: *Bpe, ids: []const u32, gpa: std.mem.Allocator) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        for (ids) |id| try self.render(gpa, &out, id);
        return out.toOwnedSlice(gpa);
    }

    fn isSpace(cp: u21) bool {
        return cp == ' ' or cp == '\t' or cp == '\n' or cp == '\r' or cp == 0x0B or cp == 0x0C;
    }
    fn isNumber(cp: u21) bool {
        return cp >= '0' and cp <= '9';
    }
    fn isLetter(cp: u21) bool {
        // ASCII + Latin-1 letters (the seam where a fuller BMP table drops in).
        if ((cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z')) return true;
        if (cp >= 0xC0 and cp <= 0xFF and cp != 0xD7 and cp != 0xF7) return true;
        return false;
    }
    fn cpAt(text: []const u8, i: usize) struct { cp: u21, len: usize } {
        const clen = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        const l = if (i + clen > text.len) 1 else clen;
        const cp = std.unicode.utf8Decode(text[i .. i + l]) catch text[i];
        return .{ .cp = cp, .len = l };
    }

    /// Encodes `text` to ids. Ports the Dart pipeline: split specials, then per
    /// segment split digits, GPT-2 pre-tokenize, byte-level map, BPE, id lookup.
    pub fn encode(self: *Bpe, gpa: std.mem.Allocator, text: []const u8, add_bos: bool) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(gpa);
        if (add_bos and self.bos >= 0) try out.append(gpa, @intCast(self.bos));

        var seg_start: usize = 0;
        var i: usize = 0;
        // Special-token split: scan for the longest special content at each pos.
        while (i < text.len) {
            if (try self.matchSpecial(text, i)) |m| {
                if (i > seg_start) try self.encodeSegment(gpa, text[seg_start..i], &out);
                try out.append(gpa, m.id);
                i += m.len;
                seg_start = i;
            } else i += 1;
        }
        if (seg_start < text.len) try self.encodeSegment(gpa, text[seg_start..], &out);
        return out.toOwnedSlice(gpa);
    }

    const SpecialMatch = struct { id: u32, len: usize };
    fn matchSpecial(self: *Bpe, text: []const u8, i: usize) !?SpecialMatch {
        var best: ?SpecialMatch = null;
        var it = self.special_ids.iterator();
        while (it.next()) |e| {
            const c = e.key_ptr.*;
            if (i + c.len <= text.len and std.mem.eql(u8, text[i .. i + c.len], c)) {
                if (best == null or c.len > best.?.len)
                    best = .{ .id = e.value_ptr.*, .len = c.len };
            }
        }
        return best;
    }

    fn encodeSegment(self: *Bpe, gpa: std.mem.Allocator, seg: []const u8, out: *std.ArrayList(u32)) !void {
        // Digit split: each digit becomes its own chunk (when split_digits).
        var chunk_start: usize = 0;
        var i: usize = 0;
        while (i < seg.len) {
            const c = cpAt(seg, i);
            if (self.split_digits and isNumber(c.cp)) {
                if (i > chunk_start) try self.pretokChunk(gpa, seg[chunk_start..i], out);
                try self.pretokChunk(gpa, seg[i .. i + c.len], out);
                i += c.len;
                chunk_start = i;
            } else i += c.len;
        }
        if (chunk_start < seg.len) try self.pretokChunk(gpa, seg[chunk_start..], out);
    }

    /// GPT-2 pre-tokenizer split of one chunk, then byte-level + BPE each word.
    fn pretokChunk(self: *Bpe, gpa: std.mem.Allocator, chunk: []const u8, out: *std.ArrayList(u32)) !void {
        var i: usize = 0;
        while (i < chunk.len) {
            const word_start = i;
            const first = cpAt(chunk, i);
            // 1. contractions: 's 't 're 've 'm 'll 'd (lowercase).
            if (first.cp == '\'') {
                const rest = chunk[i..];
                const seqs = [_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
                var matched = false;
                for (seqs) |sq| {
                    if (std.mem.startsWith(u8, rest, sq)) {
                        try self.emitWord(gpa, chunk[i .. i + sq.len], out);
                        i += sq.len;
                        matched = true;
                        break;
                    }
                }
                if (matched) continue;
            }
            // Whitespace run: \s+(?!\S) leaves the last space as next leading space.
            if (isSpace(first.cp)) {
                var j = i;
                var last_ws = i;
                while (j < chunk.len) {
                    const c = cpAt(chunk, j);
                    if (!isSpace(c.cp)) break;
                    last_ws = j;
                    j += c.len;
                }
                if (j >= chunk.len) { // run to end: one token
                    try self.emitWord(gpa, chunk[i..j], out);
                    i = j;
                    continue;
                }
                if (last_ws > i) try self.emitWord(gpa, chunk[i..last_ws], out);
                i = last_ws; // reprocess last space as leading space below
            }
            // 2/3/4: optional one leading space then a letters / number / other run.
            var m = i;
            if (m < chunk.len and cpAt(chunk, m).cp == ' ') m += 1;
            if (m >= chunk.len) { // trailing lone space
                try self.emitWord(gpa, chunk[word_start..chunk.len], out);
                i = chunk.len;
                continue;
            }
            const c2 = cpAt(chunk, m);
            if (isLetter(c2.cp)) {
                m += c2.len;
                while (m < chunk.len and isLetter(cpAt(chunk, m).cp)) m += cpAt(chunk, m).len;
            } else if (isNumber(c2.cp)) {
                m += c2.len;
                while (m < chunk.len and isNumber(cpAt(chunk, m).cp)) m += cpAt(chunk, m).len;
            } else { // other: non-space, non-letter, non-number
                m += c2.len;
                while (m < chunk.len) {
                    const c = cpAt(chunk, m);
                    if (isSpace(c.cp) or isLetter(c.cp) or isNumber(c.cp)) break;
                    m += c.len;
                }
            }
            try self.emitWord(gpa, chunk[i..m], out);
            i = m;
        }
    }

    /// Byte-level map one word, BPE it, look up ids.
    fn emitWord(self: *Bpe, gpa: std.mem.Allocator, word: []const u8, out: *std.ArrayList(u32)) !void {
        var bl: std.ArrayList(u8) = .empty;
        defer bl.deinit(gpa);
        for (word) |byte| try bl.appendSlice(gpa, self.byte_to_uni[byte]);
        try self.bpe(gpa, bl.items, out);
    }

    /// Merge-by-rank over the byte-level word's codepoint symbols, emitting ids.
    fn bpe(self: *Bpe, gpa: std.mem.Allocator, word: []const u8, out: *std.ArrayList(u32)) !void {
        if (self.ignore_merges) {
            if (self.ids.get(word)) |id| {
                try out.append(gpa, id);
                return;
            }
        }
        // symbols = list of byte-slice views into `word`, one per codepoint.
        var syms: std.ArrayList([]const u8) = .empty;
        defer syms.deinit(gpa);
        var i: usize = 0;
        while (i < word.len) {
            const clen = std.unicode.utf8ByteSequenceLength(word[i]) catch 1;
            try syms.append(gpa, word[i .. i + clen]);
            i += clen;
        }
        // Repeatedly fold the lowest-rank adjacent pair.
        var buf: [512]u8 = undefined;
        while (syms.items.len >= 2) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_at: ?usize = null;
            var j: usize = 0;
            while (j + 1 < syms.items.len) : (j += 1) {
                const ida = self.ids.get(syms.items[j]) orelse continue;
                const idb = self.ids.get(syms.items[j + 1]) orelse continue;
                if (self.ranks.get(.{ .a = ida, .b = idb })) |rk| {
                    if (rk < best_rank) {
                        best_rank = rk;
                        best_at = j;
                    }
                }
            }
            const at = best_at orelse break;
            const a = syms.items[at];
            const b = syms.items[at + 1];
            const merged = self.mergedKey(&buf, a, b) orelse break;
            syms.items[at] = merged;
            _ = syms.orderedRemove(at + 1);
        }
        for (syms.items) |sym| {
            if (self.ids.get(sym)) |id| try out.append(gpa, id);
        }
    }

    /// Returns the interned vocab-key slice for a+b (so it stays valid as a
    /// symbol across further merges). The concat must be a vocab entry.
    fn mergedKey(self: *Bpe, buf: []u8, a: []const u8, b: []const u8) ?[]const u8 {
        if (a.len + b.len > buf.len) return null;
        @memcpy(buf[0..a.len], a);
        @memcpy(buf[a.len..][0..b.len], b);
        const key = buf[0 .. a.len + b.len];
        // Return the interned key (owned by vocab) so the slice outlives `buf`.
        if (self.ids.getEntry(key)) |e| return e.key_ptr.*;
        return null;
    }
};

test "parse LTB1 and decode ids to bytes" {
    const gpa = std.testing.allocator;
    // vocab: 0 "<|endoftext|>", 1 "a", 2 "b", 3 "ab", one merge a+b, special eot=0.
    var img: std.ArrayList(u8) = .empty;
    defer img.deinit(gpa);
    try img.appendSlice(gpa, "LTB1");
    try img.append(gpa, 0x01); // flags: split_digits
    try appendI32(gpa, &img, 0); // bos
    try appendI32(gpa, &img, 0); // eos
    try img.append(gpa, 0); // add_bos
    try appendU32(gpa, &img, 4); // vocab_count
    try appendVocab(gpa, &img, "<|endoftext|>");
    try appendVocab(gpa, &img, "a");
    try appendVocab(gpa, &img, "b");
    try appendVocab(gpa, &img, "ab");
    try appendU32(gpa, &img, 1); // merges
    try appendU32(gpa, &img, 1);
    try appendU32(gpa, &img, 2);
    try appendU32(gpa, &img, 1); // specials
    try appendVocab(gpa, &img, "<|endoftext|>");
    try appendU32(gpa, &img, 0);

    var t = try Bpe.parse(gpa, img.items);
    defer t.deinit();
    const out = try t.decode(&[_]u32{ 3, 1 }, gpa); // "ab" + "a"
    defer gpa.free(out);
    try std.testing.expectEqualStrings("aba", out);
    // special id renders its content
    const eot = try t.decode(&[_]u32{0}, gpa);
    defer gpa.free(eot);
    try std.testing.expectEqualStrings("<|endoftext|>", eot);
}

test "encode merges to the golden ids and round-trips" {
    const gpa = std.testing.allocator;
    var img: std.ArrayList(u8) = .empty;
    defer img.deinit(gpa);
    // vocab: 0 eot, 1 " " (Ġ, byte 0x20 -> U+0120), 2 "a", 3 "b", 4 "ab", 5 " a"(Ġa)
    // Use the real byte-level chars so encode's alphabet map lines up.
    try img.appendSlice(gpa, "LTB1");
    try img.append(gpa, 0x01);
    try appendI32(gpa, &img, 0);
    try appendI32(gpa, &img, 0);
    try img.append(gpa, 0);
    try appendU32(gpa, &img, 6);
    try appendVocab(gpa, &img, "<|endoftext|>");
    try appendVocab(gpa, &img, "\xC4\xA0"); // U+0120 = byte-level ' '
    try appendVocab(gpa, &img, "a");
    try appendVocab(gpa, &img, "b");
    try appendVocab(gpa, &img, "ab");
    try appendVocab(gpa, &img, "\xC4\xA0a"); // ' a'
    try appendU32(gpa, &img, 1); // merges: a+b -> ab (rank 0)
    try appendU32(gpa, &img, 2);
    try appendU32(gpa, &img, 3);
    try appendU32(gpa, &img, 0); // no specials beyond added? include eot special
    // (specials_count 0 here, eot decode not needed for this test)

    var t = try Bpe.parse(gpa, img.items);
    defer t.deinit();
    const ids = try t.encode(gpa, "ab", false);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{4}, ids); // "a"+"b" -> "ab"
    const back = try t.decode(ids, gpa);
    defer gpa.free(back);
    try std.testing.expectEqualStrings("ab", back);
}

// Optional golden-vector check against the real emitted tokenizer.bin. Skips if
// the gitignored .cache artifact is absent.
test "encode reproduces the ternary fixture vector when present" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = "/home/ross/Midstall/loom/.cache/ternary-stories-genip/tokenizer.bin";
    const dir_path = std.fs.path.dirname(path).?;
    const base_name = std.fs.path.basename(path);
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return; // skip
    defer dir.close(io);
    const bytes = dir.readFileAlloc(io, base_name, gpa, std.Io.Limit.limited(8 * 1024 * 1024)) catch return; // skip
    defer gpa.free(bytes);
    var t = try Bpe.parse(gpa, bytes);
    defer t.deinit();
    const ids = try t.encode(gpa, "Once upon a time", false);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 432, 447, 259, 396 }, ids);
    const back = try t.decode(ids, gpa);
    defer gpa.free(back);
    try std.testing.expectEqualStrings("Once upon a time", back);
}

// Optional golden-vector check against the real SmolLM2-135M emit (vocab
// 49152). Model-agnostic: the expected ids are read from the fixture the
// genip compile writes alongside tokenizer.bin rather than hardcoded, since
// they differ from the ternary model's small-vocab ids above. Skips if the
// gitignored .cache artifact is absent.
test "encode reproduces the SmolLM2 fixture vector when present" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir_path = "/home/ross/Midstall/loom/.cache/smollm2-genip";
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return; // skip
    defer dir.close(io);
    const tok_bytes = dir.readFileAlloc(io, "tokenizer.bin", gpa, std.Io.Limit.limited(8 * 1024 * 1024)) catch return; // skip
    defer gpa.free(tok_bytes);
    const fixture_bytes = dir.readFileAlloc(io, "tokenizer_fixture.json", gpa, std.Io.Limit.limited(1024 * 1024)) catch return; // skip
    defer gpa.free(fixture_bytes);

    const Case = struct { text: []const u8, ids: []u32 };
    const Fixture = struct { cases: []Case };
    const parsed = try std.json.parseFromSlice(Fixture, gpa, fixture_bytes, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.cases.len > 0);
    const first = parsed.value.cases[0];

    var t = try Bpe.parse(gpa, tok_bytes);
    defer t.deinit();
    const ids = try t.encode(gpa, first.text, false);
    defer gpa.free(ids);
    try std.testing.expectEqualSlices(u32, first.ids, ids);
    const back = try t.decode(ids, gpa);
    defer gpa.free(back);
    try std.testing.expectEqualStrings(first.text, back);
}

// test helpers
fn appendU32(gpa: std.mem.Allocator, l: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try l.appendSlice(gpa, &b);
}
fn appendI32(gpa: std.mem.Allocator, l: *std.ArrayList(u8), v: i32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(i32, &b, v, .little);
    try l.appendSlice(gpa, &b);
}
fn appendVocab(gpa: std.mem.Allocator, l: *std.ArrayList(u8), s: []const u8) !void {
    var b: [2]u8 = undefined;
    std.mem.writeInt(u16, &b, @intCast(s.len), .little);
    try l.appendSlice(gpa, &b);
    try l.appendSlice(gpa, s);
}
