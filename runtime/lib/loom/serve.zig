//! OpenAI-compatible HTTP server so agents like opencode can talk to the Loom
//! runtime. Implements GET /v1/models and POST /v1/chat/completions
//! (streaming SSE and non-streaming). One request per connection (Connection:
//! close) to stay simple and robust.
//!
//! The model id is NOT hardcoded: it is read from the silicon (MODEL_LEN CSR +
//! NAME region) at startup, so the runtime stays model agnostic.

const std = @import("std");
const Io = std.Io;
const net = @import("net.zig");
const engine = @import("engine.zig");

const FALLBACK_MODEL = "loom";

pub const Options = struct {
    port: u16 = 8080,
};

pub fn run(eng: engine.Engine, opts: Options, io: Io, out: Io.File) !void {
    var name_buf: [64]u8 = undefined;
    const model_id = blk: {
        const n = eng.device.readModelName(&name_buf) catch "";
        break :blk if (n.len == 0) FALLBACK_MODEL else n;
    };

    const listen_fd = try net.listen(opts.port);
    defer net.close(listen_fd);

    var lb: [160]u8 = undefined;
    const line = try std.fmt.bufPrint(&lb, "[loom-runtime] serving OpenAI API on http://127.0.0.1:{d} (model reported by silicon: {s})\n", .{ opts.port, model_id });
    try out.writeStreamingAll(io, line);

    while (true) {
        const conn = net.accept(listen_fd) catch continue;
        defer net.close(conn);
        handle(eng, model_id, conn) catch {};
    }
}

const Msg = struct {
    role: []const u8 = "",
    content: []const u8 = "",
};
const ChatReq = struct {
    model: []const u8 = "",
    stream: bool = false,
    messages: []const Msg = &.{},
};

fn handle(eng: engine.Engine, model_id: []const u8, conn: net.fd_t) !void {
    var buf: [64 * 1024]u8 = undefined;
    const req = readRequest(conn, &buf) catch {
        try sendStatus(conn, "400 Bad Request");
        return;
    };

    if (std.mem.startsWith(u8, req.line, "GET /v1/models")) {
        try sendModels(conn, model_id);
    } else if (std.mem.startsWith(u8, req.line, "POST /v1/chat/completions")) {
        try chatCompletions(eng, model_id, conn, req.body);
    } else if (std.mem.startsWith(u8, req.line, "GET /")) {
        try sendJson(conn, "{\"status\":\"loom-runtime ok\"}");
    } else {
        try sendStatus(conn, "404 Not Found");
    }
}

const Request = struct {
    line: []const u8,
    body: []const u8,
};

/// Reads an HTTP request: headers up to CRLFCRLF, then Content-Length bytes.
fn readRequest(conn: net.fd_t, buf: []u8) !Request {
    var n: usize = 0;
    var header_end: usize = 0;
    while (true) {
        if (n >= buf.len) return error.TooLarge;
        const r = try net.read(conn, buf[n..]);
        if (r == 0) return error.Closed;
        n += r;
        if (std.mem.indexOf(u8, buf[0..n], "\r\n\r\n")) |idx| {
            header_end = idx + 4;
            break;
        }
    }

    const head = buf[0..header_end];
    const line_end = std.mem.indexOf(u8, head, "\r\n") orelse return error.Bad;
    const line = head[0..line_end];

    const content_len = parseContentLength(head) orelse 0;
    var body_have = n - header_end;
    while (body_have < content_len) {
        if (n >= buf.len) return error.TooLarge;
        const r = try net.read(conn, buf[n..]);
        if (r == 0) break;
        n += r;
        body_have = n - header_end;
    }
    return .{ .line = line, .body = buf[header_end .. header_end + content_len] };
}

fn parseContentLength(head: []const u8) ?usize {
    var i: usize = 0;
    const needle = "content-length:";
    while (i + needle.len <= head.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(head[i .. i + needle.len], needle)) {
            var j = i + needle.len;
            while (j < head.len and (head[j] == ' ' or head[j] == '\t')) j += 1;
            var k = j;
            while (k < head.len and head[k] >= '0' and head[k] <= '9') k += 1;
            return std.fmt.parseInt(usize, head[j..k], 10) catch null;
        }
    }
    return null;
}

fn chatCompletions(eng: engine.Engine, model_id: []const u8, conn: net.fd_t, body: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const parsed = std.json.parseFromSlice(ChatReq, alloc, body, .{
        .ignore_unknown_fields = true,
    }) catch {
        try sendStatus(conn, "400 Bad Request");
        return;
    };
    const reqv = parsed.value;

    var prompt: []const u8 = "";
    var mi = reqv.messages.len;
    while (mi > 0) {
        mi -= 1;
        if (std.mem.eql(u8, reqv.messages[mi].role, "user")) {
            prompt = reqv.messages[mi].content;
            break;
        }
    }

    var reply_buf: [4096]u8 = undefined;
    const reply = eng.generate(prompt, &reply_buf) catch "error: generation failed";

    if (reqv.stream) {
        try streamChat(conn, model_id, reply);
    } else {
        try jsonChat(conn, model_id, reply);
    }
}

fn jsonChat(conn: net.fd_t, model_id: []const u8, reply: []const u8) !void {
    var body_buf: [8192]u8 = undefined;
    var esc_buf: [6144]u8 = undefined;
    const esc = jsonEscape(reply, &esc_buf);
    const body = try std.fmt.bufPrint(&body_buf,
        "{{\"id\":\"loom-1\",\"object\":\"chat.completion\",\"created\":0," ++
        "\"model\":\"{s}\",\"choices\":[{{\"index\":0," ++
        "\"message\":{{\"role\":\"assistant\",\"content\":\"{s}\"}}," ++
        "\"finish_reason\":\"stop\"}}]," ++
        "\"usage\":{{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}}}",
        .{ model_id, esc });
    try sendJson(conn, body);
}

fn streamChat(conn: net.fd_t, model_id: []const u8, reply: []const u8) !void {
    try net.writeAll(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\nConnection: close\r\n\r\n");

    var esc_buf: [6144]u8 = undefined;
    const esc = jsonEscape(reply, &esc_buf);
    var chunk_buf: [8192]u8 = undefined;
    const chunk = try std.fmt.bufPrint(&chunk_buf,
        "data: {{\"id\":\"loom-1\",\"object\":\"chat.completion.chunk\"," ++
        "\"model\":\"{s}\",\"choices\":[{{\"index\":0," ++
        "\"delta\":{{\"role\":\"assistant\",\"content\":\"{s}\"}}," ++
        "\"finish_reason\":null}}]}}\n\n",
        .{ model_id, esc });
    try net.writeAll(conn, chunk);

    var done_buf: [512]u8 = undefined;
    const done = try std.fmt.bufPrint(&done_buf,
        "data: {{\"id\":\"loom-1\",\"object\":\"chat.completion.chunk\",\"model\":\"{s}\"," ++
        "\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n",
        .{model_id});
    try net.writeAll(conn, done);
    try net.writeAll(conn, "data: [DONE]\n\n");
}

fn sendModels(conn: net.fd_t, model_id: []const u8) !void {
    var body_buf: [256]u8 = undefined;
    const body = try std.fmt.bufPrint(&body_buf,
        "{{\"object\":\"list\",\"data\":[{{\"id\":\"{s}\"," ++
        "\"object\":\"model\",\"created\":0,\"owned_by\":\"loom\"}}]}}",
        .{model_id});
    try sendJson(conn, body);
}

fn sendJson(conn: net.fd_t, body: []const u8) !void {
    var hdr: [128]u8 = undefined;
    const h = try std.fmt.bufPrint(&hdr,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" ++
        "Content-Length: {d}\r\nConnection: close\r\n\r\n", .{body.len});
    try net.writeAll(conn, h);
    try net.writeAll(conn, body);
}

fn sendStatus(conn: net.fd_t, status: []const u8) !void {
    var hdr: [128]u8 = undefined;
    const h = try std.fmt.bufPrint(&hdr,
        "HTTP/1.1 {s}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{status});
    try net.writeAll(conn, h);
}

/// Minimal JSON string escaping (quotes, backslash, control chars, newlines).
fn jsonEscape(s: []const u8, out: []u8) []const u8 {
    var n: usize = 0;
    for (s) |c| {
        if (n + 6 >= out.len) break;
        switch (c) {
            '"' => {
                out[n] = '\\';
                out[n + 1] = '"';
                n += 2;
            },
            '\\' => {
                out[n] = '\\';
                out[n + 1] = '\\';
                n += 2;
            },
            '\n' => {
                out[n] = '\\';
                out[n + 1] = 'n';
                n += 2;
            },
            '\r' => {
                out[n] = '\\';
                out[n + 1] = 'r';
                n += 2;
            },
            '\t' => {
                out[n] = '\\';
                out[n + 1] = 't';
                n += 2;
            },
            else => {
                if (c < 0x20) {
                    n += (std.fmt.bufPrint(out[n..], "\\u{x:0>4}", .{c}) catch break).len;
                } else {
                    out[n] = c;
                    n += 1;
                }
            },
        }
    }
    return out[0..n];
}

test "jsonEscape escapes quotes, backslashes and newlines" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("a\\\"b\\\\c\\nd", jsonEscape("a\"b\\c\nd", &buf));
}

test "parseContentLength is case-insensitive" {
    const head = "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 42\r\n\r\n";
    try std.testing.expectEqual(@as(?usize, 42), parseContentLength(head));
}
