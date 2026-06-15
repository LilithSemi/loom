const std = @import("std");
const Io = std.Io;
const loom = @import("loom");
const proto = loom.protocol;
const quant = loom.quant;
const transport = loom.transport;
const serveMod = loom.serve;
const engineMod = loom.engine;
const fp = loom.fp;
const model = loom.model;
const tokenizer = loom.tokenizer;
const forward = loom.forward;
const linearMod = loom.linear;

const Device = transport.Device;
const DevIface = loom.Device;

const DEFAULT_PORT = "/dev/ttyACM0";
/// Fallback generation backstop ONLY for manifests missing `max_seq` (older
/// builds). Normal path caps at the model's trained context; generation ends on
/// the model's stop token well before either.
const DEFAULT_MAX_TOKENS: usize = 256;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.skip(); // argv[0]

    var port: []const u8 = DEFAULT_PORT;
    var kind: transport.Kind = .uart;
    var cmd: []const u8 = "help";
    var listen_port: u16 = 8080;
    var model_dir: []const u8 = "";
    var tokens: ?usize = null; // null = generate until the model stops (EOS/BOS)
    var prompt_text: []const u8 = "";
    var baud: u32 = 1500000; // must match the bitstream's UART divisor (new default)
    var stats = false; // print per-token round-trip + timing breakdown
    var poll_timeout: usize = linearMod.DEFAULT_TIMEOUT; // STATUS reads before giving up
    var repeat_window: usize = 8; // loop-stop window (0 disables); see forward.generate
    var repo: []const u8 = ""; // HF repo id for `fetch` (owner/name)
    var fetch_out: []const u8 = ""; // destination dir for `fetch`
    var hf_token: []const u8 = ""; // bearer token for gated `fetch` repos

    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--model") or std.mem.eql(u8, a, "-m")) {
            model_dir = args.next() orelse return usage(io, "--model needs a directory");
        } else if (std.mem.eql(u8, a, "--prompt") or std.mem.eql(u8, a, "-P")) {
            prompt_text = args.next() orelse return usage(io, "--prompt needs text");
        } else if (std.mem.eql(u8, a, "--tokens") or std.mem.eql(u8, a, "-n")) {
            const v = args.next() orelse return usage(io, "--tokens needs a count");
            tokens = std.fmt.parseInt(usize, v, 10) catch return usage(io, "--tokens invalid");
        } else if (std.mem.eql(u8, a, "--listen") or std.mem.eql(u8, a, "-l")) {
            const v = args.next() orelse return usage(io, "--listen needs a port");
            listen_port = std.fmt.parseInt(u16, v, 10) catch return usage(io, "--listen port invalid");
        } else if (std.mem.eql(u8, a, "--transport") or std.mem.eql(u8, a, "-t")) {
            const v = args.next() orelse return usage(io, "--transport needs a value");
            if (std.mem.eql(u8, v, "uart")) {
                kind = .uart;
            } else if (std.mem.eql(u8, v, "usb")) {
                kind = .usb;
            } else {
                return usage(io, "--transport must be uart or usb");
            }
        } else if (std.mem.eql(u8, a, "--port") or std.mem.eql(u8, a, "-p")) {
            port = args.next() orelse return usage(io, "--port needs a value");
        } else if (std.mem.eql(u8, a, "--baud") or std.mem.eql(u8, a, "-b")) {
            const v = args.next() orelse return usage(io, "--baud needs a value");
            baud = std.fmt.parseInt(u32, v, 10) catch return usage(io, "--baud invalid");
        } else if (std.mem.eql(u8, a, "--stats")) {
            stats = true;
        } else if (std.mem.eql(u8, a, "--poll-timeout")) {
            const v = args.next() orelse return usage(io, "--poll-timeout needs a value");
            poll_timeout = std.fmt.parseInt(usize, v, 10) catch return usage(io, "--poll-timeout invalid");
        } else if (std.mem.eql(u8, a, "--poll-delay-us")) {
            const v = args.next() orelse return usage(io, "--poll-delay-us needs a value");
            linearMod.poll_delay_us = std.fmt.parseInt(u64, v, 10) catch return usage(io, "--poll-delay-us invalid");
        } else if (std.mem.eql(u8, a, "--rep-window")) {
            // Loop-stop window: end generation when the last N output tokens
            // recur earlier (0 disables). Default catches tiny-model repeat loops.
            const v = args.next() orelse return usage(io, "--rep-window needs a value");
            repeat_window = std.fmt.parseInt(usize, v, 10) catch return usage(io, "--rep-window invalid");
        } else if (std.mem.eql(u8, a, "--repo")) {
            repo = args.next() orelse return usage(io, "--repo needs owner/name");
        } else if (std.mem.eql(u8, a, "--out")) {
            fetch_out = args.next() orelse return usage(io, "--out needs a directory");
        } else if (std.mem.eql(u8, a, "--hf-token")) {
            hf_token = args.next() orelse return usage(io, "--hf-token needs a value");
        } else if (std.mem.eql(u8, a, "-h") or std.mem.eql(u8, a, "--help")) {
            return usage(io, null);
        } else {
            cmd = a;
        }
    }

    if (std.mem.eql(u8, cmd, "help")) return usage(io, null);

    // One buffered stdout writer for the whole run; flushed on exit. The failure
    // paths below std.process.exit (which skips defers), so they flush by hand.
    const out_file = Io.File.stdout();
    var out_buf: [512]u8 = undefined;
    var out_writer = out_file.writer(io, &out_buf);
    const out = &out_writer.interface;
    defer out.flush() catch {};

    // `fetch` needs no device, it is a pure download, so handle it before we try
    // to open a transport (which would fail with no board attached).
    if (std.mem.eql(u8, cmd, "fetch")) {
        cmdFetch(io, out, repo, fetch_out, hf_token) catch |e| {
            try out.print("error: fetch failed: {s}\n", .{@errorName(e)});
            out.flush() catch {};
            std.process.exit(1);
        };
        return;
    }

    var device = openTransport(io, kind, port, baud) catch |e| {
        try out.print("error: could not open {s} transport: {s}\n", .{ @tagName(kind), @errorName(e) });
        out.flush() catch {};
        std.process.exit(1);
    };
    defer device.close();

    try out.print("[loom-runtime] transport = {s}\n", .{@tagName(kind)});

    if (std.mem.eql(u8, cmd, "version")) {
        try cmdVersion(device, out);
    } else if (std.mem.eql(u8, cmd, "probe")) {
        try cmdProbe(device, out);
    } else if (std.mem.eql(u8, cmd, "info")) {
        try cmdInfo(device, out);
    } else if (std.mem.eql(u8, cmd, "matmul")) {
        try cmdMatmul(device, out);
    } else if (std.mem.eql(u8, cmd, "serve")) {
        try out.flush(); // hand a clean stdout to serve, which writes the file directly
        try serveMod.run(.{ .device = device }, .{ .port = listen_port }, io, out_file);
    } else if (std.mem.eql(u8, cmd, "generate")) {
        try cmdGenerate(io, device.device(), out, model_dir, tokens, prompt_text, stats, poll_timeout, repeat_window);
    } else {
        try out.print("unknown command: {s}\n", .{cmd});
        return usage(io, null);
    }
}

fn openTransport(io: Io, kind: transport.Kind, port: []const u8, baud: u32) !Device {
    return switch (kind) {
        .uart => Device.openUart(io, port, baud),
        .usb => Device.openUsb(io),
    };
}

/// Downloads a HuggingFace model repo (config + tokenizer + safetensors, single
/// file or sharded) into a local directory over HTTPS via std.http, so genip can
/// compile it. This is the pipeline front door, `loom fetch --repo owner/name
/// --out dir` then `loom_genip --soc fp --model dir`.
fn cmdFetch(io: Io, out: *Io.Writer, repo: []const u8, out_dir: []const u8, token: []const u8) !void {
    if (repo.len == 0) return usage(io, "fetch needs --repo <owner/name>");
    if (out_dir.len == 0) return usage(io, "fetch needs --out <dir>");

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var dir = try Io.Dir.cwd().createDirPathOpen(io, out_dir, .{});
    defer dir.close(io);

    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();

    // Bearer auth for gated repos (optional).
    var auth_buf: [1024]u8 = undefined;
    var one_header: [1]std.http.Header = undefined;
    var extra: []const std.http.Header = &.{};
    if (token.len > 0) {
        one_header[0] = .{
            .name = "authorization",
            .value = try std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{token}),
        };
        extra = one_header[0..1];
    }

    try out.print("Fetching {s} -> {s}\n", .{ repo, out_dir });
    try out.flush();

    if (!try fetchFile(&client, io, dir, repo, "config.json", extra, out)) {
        try out.print("error: {s} has no config.json (not a HF model repo?)\n", .{repo});
        return error.NoConfig;
    }
    _ = fetchFile(&client, io, dir, repo, "tokenizer.json", extra, out) catch false;
    _ = fetchFile(&client, io, dir, repo, "generation_config.json", extra, out) catch false;

    // Weights: a sharded checkpoint (index.json + weight_map) wins, else the
    // single model.safetensors.
    const sharded = fetchFile(&client, io, dir, repo, "model.safetensors.index.json", extra, out) catch false;
    if (sharded) {
        const bytes = try dir.readFileAlloc(io, "model.safetensors.index.json", gpa, Io.Limit.limited(16 * 1024 * 1024));
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer parsed.deinit();
        const wm = parsed.value.object.get("weight_map") orelse return error.BadIndex;
        var seen = std.StringHashMap(void).init(gpa);
        var it = wm.object.iterator();
        while (it.next()) |e| {
            const shard = e.value_ptr.*.string;
            if (seen.contains(shard)) continue;
            try seen.put(shard, {});
            if (!try fetchFile(&client, io, dir, repo, shard, extra, out)) return error.ShardMissing;
        }
    } else {
        if (!try fetchFile(&client, io, dir, repo, "model.safetensors", extra, out)) {
            try out.print("error: no model.safetensors or index.json in {s}\n", .{repo});
            return error.NoWeights;
        }
    }
    try out.print("Done. Compile with: loom_genip --soc fp --model {s}\n", .{out_dir});
}

/// Fetches one repo file to [dir]/[name] over HTTPS (following redirects to the
/// HF CDN). Returns true on HTTP 200, false on a miss (deleting the stub file),
/// and errors only on a transport-level failure.
fn fetchFile(
    client: *std.http.Client,
    io: Io,
    dir: Io.Dir,
    repo: []const u8,
    name: []const u8,
    extra: []const std.http.Header,
    out: *Io.Writer,
) !bool {
    var url_buf: [512]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buf, "https://huggingface.co/{s}/resolve/main/{s}", .{ repo, name });

    var file = try dir.createFile(io, name, .{});
    var wbuf: [64 * 1024]u8 = undefined;
    var fw = file.writer(io, &wbuf);
    const res = client.fetch(.{
        .location = .{ .url = url },
        .response_writer = &fw.interface,
        .extra_headers = extra,
    }) catch |e| {
        file.close(io);
        return e;
    };
    fw.flush() catch {};
    file.close(io);

    if (res.status == .ok) {
        try out.print("  ok   {s}\n", .{name});
        try out.flush();
        return true;
    }
    dir.deleteFile(io, name) catch {};
    return false;
}

/// Transport reliability probe: hammer pure reads (no compute) of the constant
/// VERSION register and check every one. If singles or bursts ever come back
/// wrong, the transport read path is corrupting; if they're all clean, the
/// generate-time flakiness is in compute or the writes, not the reads.
fn cmdProbe(device: Device, out: *Io.Writer) !void {
    // The FP accelerator VERSION reg (csr_base 0x10000 + fp.REG_VERSION); the
    // transport-level proto.REG_VERSION is the OTHER (overlay) accel, which reads
    // 0 on the fp bitstream.
    const ver_addr: u32 = 0x10000 + fp.REG_VERSION;
    const N: usize = 5000;
    var bad_single: usize = 0;
    var first_bad: u32 = 0;
    for (0..N) |_| {
        const v = try device.regRead(ver_addr);
        if (v != fp.VERSION_MAGIC) {
            if (bad_single == 0) first_bad = v;
            bad_single += 1;
        }
    }
    try out.print("single-word VERSION reads: {d}/{d} bad", .{ bad_single, N });
    if (bad_single > 0) try out.print(" (first bad = 0x{X:0>8})", .{first_bad});
    try out.print("\n", .{});

    // Burst reads: word[0] of a 32-word burst from VERSION must be 'LOOM'; a
    // misaligned/corrupt burst read shows up as word[0] != VERSION_MAGIC.
    const W = 32;
    const M: usize = 2000;
    var bad_burst: usize = 0;
    var first_bad_burst: u32 = 0;
    for (0..M) |_| {
        var cur: [W]u32 = undefined;
        try device.readRegs(ver_addr, &cur);
        if (cur[0] != fp.VERSION_MAGIC) {
            if (bad_burst == 0) first_bad_burst = cur[0];
            bad_burst += 1;
        }
    }
    try out.print("burst({d}-word) VERSION reads: {d}/{d} bad", .{ W, bad_burst, M });
    if (bad_burst > 0) try out.print(" (first bad word[0] = 0x{X:0>8})", .{first_bad_burst});
    try out.print("\n", .{});

    // Write-readback: write a varying value to a RW CSR (COL_TILES, 16-bit) and
    // read it straight back. A mismatch means the write path corrupts (the CSR
    // setup that configures every matmul). Two forms: single regWrite, and the
    // burst writeRegs path the runtime actually uses per matmul.
    const col_addr: u32 = 0x10000 + fp.REG_COL_TILES;
    const row_addr: u32 = 0x10000 + fp.REG_ROW_BLOCKS;
    var bad_wr: usize = 0;
    var bad_wrburst: usize = 0;
    for (0..N) |i| {
        const val: u32 = @as(u32, @intCast((i *% 40503) & 0xFFFF));
        try device.regWrite(col_addr, val);
        if ((try device.regRead(col_addr)) != val) bad_wr += 1;
        const v2: u32 = @as(u32, @intCast((i *% 25173 +% 13849) & 0xFFFF));
        try device.writeRegs(&[_][2]u32{ .{ col_addr, val }, .{ row_addr, v2 } });
        if ((try device.regRead(col_addr)) != val or (try device.regRead(row_addr)) != v2) bad_wrburst += 1;
    }
    try out.print("single write-readback: {d}/{d} bad\n", .{ bad_wr, N });
    try out.print("burst  write-readback: {d}/{d} bad\n", .{ bad_wrburst, N });

    // Flash-read consistency: the accelerator reads weights from flash (0x20200000)
    // over USRMCLK/SPI, the one path generate exercises that the CSR probes don't.
    // Read a fixed 32-word window many times; any variation = flash-read flakiness
    // (the likely source of the generate-time token flips). Host reads go through
    // the same flash controller the accelerator uses.
    const flash_addr: u32 = 0x20200000;
    var fbase: [W]u32 = undefined;
    try device.readRegs(flash_addr, &fbase);
    var bad_flash: usize = 0;
    var flash_reads: usize = 0;
    for (0..M) |_| {
        var cur: [W]u32 = undefined;
        try device.readRegs(flash_addr, &cur);
        for (0..W) |k| {
            flash_reads += 1;
            if (cur[k] != fbase[k]) bad_flash += 1;
        }
    }
    try out.print("flash-read consistency: {d}/{d} words differ across {d} re-reads\n", .{ bad_flash, flash_reads, M });
    try out.flush();
}

fn cmdVersion(device: Device, out: *Io.Writer) !void {
    const v = try device.regRead(proto.REG_VERSION);
    const ok = v == proto.VERSION_MAGIC;
    try out.print("VERSION = 0x{X:0>8} ({s})\n", .{ v, if (ok) "OK 'LOOM'" else "BAD" });
    if (!ok) {
        out.flush() catch {};
        std.process.exit(1);
    }
}

fn cmdInfo(device: Device, out: *Io.Writer) !void {
    const v = try device.regRead(proto.REG_VERSION);
    var name_buf: [64]u8 = undefined;
    const name = device.readModelName(&name_buf) catch "?";
    const rows = try device.regRead(proto.REG_ROWS);
    const cols = try device.regRead(proto.REG_COLS);
    try out.print("VERSION = 0x{X:0>8} ({s})\n", .{ v, if (v == proto.VERSION_MAGIC) "OK" else "BAD" });
    try out.print("MODEL   = {s}\n", .{name});
    try out.print("ROWS    = {d}\nCOLS    = {d}\n", .{ rows, cols });
    try out.print("max tile = {d}x{d}\n", .{ proto.MAX_ROWS, proto.MAX_COLS });
}

fn cmdMatmul(device: Device, out: *Io.Writer) !void {
    // Demo case (matches tools/loom_uart_matmul.py): ROWS=2, COLS=4.
    const rows: usize = 2;
    const cols: usize = 4;
    const shift: u6 = 4;
    const weights = [_]i8{ 1, -2, 3, -4, 5, 6, -7, 8 };
    const acts = [_]i8{ 10, -3, 2, 1 };
    const mults = [_]u32{ 16, 8 };

    var expected: [rows]i8 = undefined;
    quant.matmul(&weights, &acts, &mults, shift, rows, cols, &expected);

    try out.print("Test case: ROWS={d} COLS={d} SHIFT={d}\n", .{ rows, cols, shift });
    try out.print("  expected (golden) = [{d}, {d}]\n", .{ expected[0], expected[1] });

    const v = try device.regRead(proto.REG_VERSION);
    if (v != proto.VERSION_MAGIC) {
        try out.print("FAIL: VERSION 0x{X:0>8}, wrong bitstream?\n", .{v});
        out.flush() catch {};
        std.process.exit(1);
    }

    // Weight buffer: byte index = row*MAX_COLS + col.
    var wbuf = [_]u8{0} ** (proto.MAX_ROWS * proto.MAX_COLS);
    for (0..rows) |r| {
        for (0..cols) |c| {
            wbuf[r * proto.MAX_COLS + c] = @bitCast(weights[r * cols + c]);
        }
    }
    try device.bufWrite(proto.BUF_WEIGHT, &wbuf);

    // Activation buffer: byte index = col.
    var abuf = [_]u8{0} ** proto.MAX_COLS;
    for (0..cols) |c| abuf[c] = @bitCast(acts[c]);
    try device.bufWrite(proto.BUF_ACT, &abuf);

    // RowMult buffer: uint16 per row.
    var mbuf = [_]u8{0} ** (proto.MAX_ROWS * 2);
    for (0..rows) |r| std.mem.writeInt(u16, mbuf[r * 2 ..][0..2], @intCast(mults[r]), .little);
    try device.bufWrite(proto.BUF_MULT, &mbuf);

    // CSRs.
    try device.regWrite(proto.REG_ROWS, rows);
    try device.regWrite(proto.REG_COLS, cols);
    try device.regWrite(proto.REG_SHIFT, shift);

    // Start.
    try device.regWrite(proto.REG_CONTROL, 0x1);

    // Poll STATUS.done (bit 1).
    var done = false;
    var status: u32 = 0;
    for (0..200) |_| {
        status = try device.regRead(proto.REG_STATUS);
        if (status & 0x2 != 0) {
            done = true;
            break;
        }
    }
    try out.print("  STATUS = 0x{X:0>8} (busy={d}, done={d})\n", .{ status, status & 1, (status >> 1) & 1 });
    if (!done) {
        try out.print("FAIL: accelerator never reported done\n", .{});
        out.flush() catch {};
        std.process.exit(1);
    }

    // Read result buffer.
    var rbuf = [_]u8{0} ** rows;
    try device.bufRead(proto.BUF_RESULT, &rbuf);
    const hw0 = quant.toI8(rbuf[0]);
    const hw1 = quant.toI8(rbuf[1]);
    try out.print("  hardware (silicon) = [{d}, {d}]\n", .{ hw0, hw1 });

    if (hw0 == expected[0] and hw1 == expected[1]) {
        try out.print("SUCCESS: the Loom accelerator works.\n", .{});
    } else {
        try out.print("FAIL: hardware != golden\n", .{});
        out.flush() catch {};
        std.process.exit(1);
    }
}

/// Generates text from a genip-emitted model directory (loom.json + scales.bin +
/// glue.bin + tokenizer.bin), running every matmul on the device (which reads
/// int4 weights from flash). Decodes from BOS.
/// Streams generated tokens to `out` as they arrive, rendering each through the
/// tokenizer (tracking `prev` for the BOS leading-space rule). Reuses one buffer.
const StreamSink = struct {
    tok: tokenizer.Any,
    out: *Io.Writer,
    gpa: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    pending: *std.ArrayList(u8),
    prev: u32,

    pub fn emit(self: *StreamSink, t: u32) !void {
        self.buf.clearRetainingCapacity();
        try self.tok.render(self.gpa, self.buf, t, self.prev);
        try self.pending.appendSlice(self.gpa, self.buf.items);
        // Flush up to the last complete UTF-8 boundary; keep the remainder.
        const cut = tokenizer.completeUtf8Prefix(self.pending.items);
        try self.out.writeAll(self.pending.items[0..cut]);
        try self.out.flush(); // a token can take seconds over UART; show it live
        const rem = self.pending.items.len - cut;
        std.mem.copyForwards(u8, self.pending.items[0..rem], self.pending.items[cut..]);
        self.pending.shrinkRetainingCapacity(rem);
        self.prev = t;
    }
};

fn cmdGenerate(io: Io, device: DevIface, out: *Io.Writer, model_dir: []const u8, tokens: ?usize, prompt_text: []const u8, stats: bool, poll_timeout: usize, repeat_window: usize) !void {
    if (model_dir.len == 0) return usage(io, "generate needs --model DIR (a genip output)");

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var cfg = model.Config.load(gpa, io, model_dir) catch |e| {
        try out.print("error: could not load model from {s}: {s}\n", .{ model_dir, @errorName(e) });
        out.flush() catch {};
        std.process.exit(1);
    };
    const m = cfg.manifest();

    var dir = try model.openModelDir(io, model_dir);
    defer dir.close(io);
    const tok_bytes = try dir.readFileAlloc(io, "tokenizer.bin", gpa, Io.Limit.limited(4 * 1024 * 1024));
    var tok = try tokenizer.Any.parse(gpa, tok_bytes, 1, 2);

    const ver = try device.regRead(m.csr_base + fp.REG_VERSION);
    if (ver != fp.VERSION_MAGIC) {
        try out.print("FAIL: VERSION 0x{X:0>8} - wrong bitstream, or weights not flashed?\n", .{ver});
        out.flush() catch {};
        std.process.exit(1);
    }

    // Tiered weight cache: load the hot matrices into on-chip BRAM once (no-op if
    // the model was built without --fp-bram-cache-kb). Cuts the flash weight-read
    // wall that dominates the token.
    if (m.bram_cache_kb != 0) {
        try out.print("[loom-runtime] loading {d}KB of hot weights into on-chip BRAM...\n", .{m.bram_cache_kb});
        try out.flush();
    }
    try forward.loadBramCache(gpa, io, device, cfg, dir);
    // Prompt tokens prime the KV cache; with no prompt, start bare from BOS.
    const prompt = if (prompt_text.len > 0)
        try tok.encode(gpa, prompt_text, tok.addBos())
    else
        try gpa.dupe(u32, &[_]u32{tok.bosId()});
    // No --tokens: run until the model emits a stop token. The cap is only a
    // backstop against a non-terminating model, and the natural bound is the
    // model's trained context (positions past it are untrained). Use that when
    // known; fall back to DEFAULT_MAX_TOKENS for manifests without max_seq.
    const cap = tokens orelse if (m.max_seq > prompt.len)
        m.max_seq - prompt.len
    else
        DEFAULT_MAX_TOKENS;
    if (tokens) |n|
        try out.print("[loom-runtime] model={s} ({d} layers, hidden {d}); prompt {d} tok, up to {d} more...\n", .{ m.name, m.layers, m.hidden, prompt.len, n })
    else
        try out.print("[loom-runtime] model={s} ({d} layers, hidden {d}); prompt {d} tok, until done (cap {d})...\n", .{ m.name, m.layers, m.hidden, prompt.len, cap });

    // Echo the prompt, then stream each generated token as it lands (a token can
    // take seconds over UART, so live output beats staging the whole reply).
    const ptext = try tok.decode(prompt, gpa);
    try out.writeAll(ptext);
    try out.flush();

    var sbuf: std.ArrayList(u8) = .empty;
    var pending: std.ArrayList(u8) = .empty;
    defer pending.deinit(gpa);
    var sink = StreamSink{ .tok = tok, .out = out, .gpa = gpa, .buf = &sbuf, .pending = &pending, .prev = prompt[prompt.len - 1] };

    // Stop as soon as the model emits EOS or the BOS story delimiter.
    const stop = [_]u32{ tok.eosId(), tok.bosId() };
    const gen = try gpa.alloc(u32, cap);
    linearMod.poll_reads = 0;
    linearMod.result_reads = 0;
    const produced = try forward.generate(gpa, io, device, cfg, poll_timeout, prompt, gen, &stop, repeat_window, &sink);
    try out.writeAll(pending.items);
    try out.flush();
    try out.writeAll("\n");
    if (stats) {
        const denom = if (produced == 0) 1 else produced;
        try out.print("[loom-runtime] round-trips/tok: poll={d} result={d}\n", .{ linearMod.poll_reads / denom, linearMod.result_reads / denom });
        try out.print("[loom-runtime] time ms/tok: write={d} poll={d} read={d}\n", .{ linearMod.t_write_ns / denom / 1_000_000, linearMod.t_poll_ns / denom / 1_000_000, linearMod.t_read_ns / denom / 1_000_000 });
    }
}

fn usage(io: Io, err: ?[]const u8) !void {
    var buf: [512]u8 = undefined;
    var w = Io.File.stderr().writer(io, &buf);
    const out = &w.interface;
    defer out.flush() catch {};
    if (err) |e| try out.print("error: {s}\n\n", .{e});
    try out.print(
        \\loom-cli: drive the Loom accelerator.
        \\
        \\usage: loom-cli [-t uart|usb] [-p PATH] <command>
        \\
        \\commands:
        \\  version   read VERSION CSR and verify the 'LOOM' magic
        \\  info      read VERSION + geometry CSRs
        \\  matmul    run a real int8 matmul on silicon, verify vs golden
        \\  generate  generate text from a genip model dir (every matmul on silicon)
        \\  serve     serve an OpenAI-compatible API
        \\
        \\options:
        \\  -t, --transport uart|usb   transport (default uart)
        \\  -p, --port PATH            serial device for uart (default {s})
        \\  -b, --baud RATE            uart baud, must match the bitstream (default 1500000)
        \\      --stats                print per-token round-trip + timing breakdown
        \\      --poll-timeout N       max STATUS reads per matmul before giving up
        \\  -m, --model DIR            genip output dir (for generate)
        \\  -P, --prompt TEXT          prompt to continue (default: bare BOS)
        \\  -n, --tokens N             cap generated tokens (default: model context)
        \\      --rep-window N         stop when the last N tokens loop (0 off, default 8)
        \\  -l, --listen PORT          serve port (default 8080)
        \\
    , .{DEFAULT_PORT});
}

test "matmul golden matches the demo case" {
    const weights = [_]i8{ 1, -2, 3, -4, 5, 6, -7, 8 };
    const acts = [_]i8{ 10, -3, 2, 1 };
    const mults = [_]u32{ 16, 8 };
    var out: [2]i8 = undefined;
    quant.matmul(&weights, &acts, &mults, 4, 2, 4, &out);
    try std.testing.expectEqual(@as(i8, 18), out[0]);
    try std.testing.expectEqual(@as(i8, 13), out[1]);
}
