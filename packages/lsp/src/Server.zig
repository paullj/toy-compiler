//! The JSON-RPC server core: an injectable reader/writer, the lifecycle handshake, the
//! request/notification dispatch, the document store, and didOpen/didChange -> check ->
//! publishDiagnostics. Requests are SERIALIZED — one message in, its response(s) out,
//! before the next is read. No concurrency, no cancellation, no incremental sync.
//!
//! Every REQUEST produces exactly one response (a result or an error) before dispatch
//! returns, so a client is never left hanging. Notifications never get a response. Only
//! OOM / a dead transport are fatal to the loop; a per-message fault is swallowed so one
//! bad message can neither drop a later response nor kill the session.

const std = @import("std");
const Io = std.Io;
const Reader = std.Io.Reader;
const Writer = std.Io.Writer;

const transport = @import("transport.zig");
const protocol = @import("protocol.zig");
const diagnostics = @import("diagnostics.zig");
const hover = @import("hover.zig");
const Documents = @import("Documents.zig");

pub const ExitCode = enum(u8) { ok = 0, no_shutdown = 1 };

/// Connect the server core to an injected reader + writer and run until `exit` or a clean
/// pipe close. `toy lsp` passes real stdin/stdout; the in-process test passes in-memory
/// buffers. Returns `.ok` when `shutdown` preceded `exit`, else `.no_shutdown` (exit 1).
pub fn serve(gpa: std.mem.Allocator, reader: *Reader, writer: *Writer) anyerror!ExitCode {
    var s = try Server.init(gpa);
    defer s.deinit();
    return s.run(reader, writer);
}

pub const Server = struct {
    gpa: std.mem.Allocator,
    /// Owns its own compute io (openCache / discover / checkGraph). This is separate from
    /// the transport reader/writer, whose io was already consumed by the caller.
    threaded: std.Io.Threaded,
    docs: Documents,
    /// One reused scratch dir + file under cwd: the check front-end reads from disk, so an
    /// unsaved buffer is round-tripped through this file (truncated + rewritten per check).
    scratch_dir: []u8,
    scratch_file: []u8,
    got_initialize: bool = false,
    shutdown_requested: bool = false,

    pub fn init(gpa: std.mem.Allocator) !Server {
        var threaded: std.Io.Threaded = .init(gpa, .{});
        errdefer threaded.deinit();
        const setup_io = threaded.io();

        // A per-instance dir name keeps two servers (or a test + a real run) from
        // colliding on the same scratch file under one cwd. The monotonic clock is a
        // sufficient discriminator — server instances are not created in a tight loop.
        const ns = Io.Clock.Timestamp.now(setup_io, .awake).raw.nanoseconds;
        const dir = try std.fmt.allocPrint(gpa, ".toy-lsp-{x}", .{@as(u64, @bitCast(@as(i64, @truncate(ns))))});
        errdefer gpa.free(dir);
        try Io.Dir.cwd().createDirPath(setup_io, dir);
        errdefer Io.Dir.cwd().deleteTree(setup_io, dir) catch {};
        const file = try std.fmt.allocPrint(gpa, "{s}/doc.toy", .{dir});
        errdefer gpa.free(file);

        return .{
            .gpa = gpa,
            .threaded = threaded,
            .docs = .{},
            .scratch_dir = dir,
            .scratch_file = file,
        };
    }

    pub fn deinit(self: *Server) void {
        Io.Dir.cwd().deleteTree(self.io(), self.scratch_dir) catch {};
        self.docs.deinit(self.gpa);
        self.gpa.free(self.scratch_file);
        self.gpa.free(self.scratch_dir);
        self.threaded.deinit();
        self.* = undefined;
    }

    /// The compute io, recomputed from the PINNED server each call: `Threaded.io()`
    /// captures `&self.threaded`, so it must never be cached across the by-value move out
    /// of `init`.
    fn io(self: *Server) Io {
        return self.threaded.io();
    }

    fn run(self: *Server, reader: *Reader, writer: *Writer) anyerror!ExitCode {
        while (true) {
            const body = transport.readMessage(self.gpa, reader) catch |e| switch (e) {
                // A clean pipe close at a message boundary ends the session.
                error.EndOfStream => break,
                // A malformed / truncated frame: terminate cleanly rather than hang.
                error.UnexpectedEof, error.StreamTooLong, error.MissingContentLength, error.InvalidHeader => break,
                error.OutOfMemory, error.ReadFailed => return e,
            };
            defer self.gpa.free(body);
            const keep = self.dispatch(writer, body) catch |e| switch (e) {
                // A dead transport or OOM is fatal; any per-message handler fault is not.
                error.OutOfMemory, error.WriteFailed => return e,
                else => true,
            };
            if (!keep) break;
        }
        return if (self.shutdown_requested) .ok else .no_shutdown;
    }

    /// Handle one message. Returns `false` ONLY on `exit` (stop the loop); `true` keeps
    /// serving. A parse failure or a message that isn't a request-for-us is ignored.
    fn dispatch(self: *Server, writer: *Writer, body: []const u8) anyerror!bool {
        var parsed = std.json.parseFromSlice(std.json.Value, self.gpa, body, .{}) catch return true;
        defer parsed.deinit();
        const root = parsed.value;
        if (root != .object) return true;

        // No `method` => this is a RESPONSE to us; we send no server->client requests, so
        // there is nothing to correlate. Ignore it.
        const method_v = root.object.get("method") orelse return true;
        if (method_v != .string) return true;
        const method = method_v.string;
        // Present iff this is a request; absent for a notification.
        const id = root.object.get("id");

        if (eql(method, "exit")) return false;

        if (eql(method, "initialize")) {
            if (self.got_initialize) {
                if (id) |iv| try protocol.writeError(self.gpa, writer, iv, protocol.err_code.invalid_request, "server already initialized");
                return true;
            }
            self.got_initialize = true;
            if (id) |iv| try protocol.writeResponse(self.gpa, writer, iv, protocol.InitializeResult{});
            return true;
        }

        // Everything except initialize/exit must wait for initialization.
        if (!self.got_initialize) {
            if (id) |iv| try protocol.writeError(self.gpa, writer, iv, protocol.err_code.server_not_initialized, "server not initialized");
            return true;
        }

        if (eql(method, "initialized")) return true;

        if (eql(method, "textDocument/didOpen")) {
            const td = field(root, "params", "textDocument") orelse return true;
            const uri = getStr(td, "uri") orelse return true;
            const text = getStr(td, "text") orelse return true;
            const version = getInt(td, "version") orelse 0;
            try self.docs.put(self.gpa, uri, text, version);
            try self.publish(writer, uri);
            return true;
        }

        if (eql(method, "textDocument/didChange")) {
            const params = objGet(root, "params") orelse return true;
            const td = objGet(params, "textDocument") orelse return true;
            const uri = getStr(td, "uri") orelse return true;
            const version = getInt(td, "version") orelse 0;
            const changes = objGet(params, "contentChanges") orelse return true;
            if (changes != .array or changes.array.items.len == 0) return true;
            // Full-text sync: the LAST change carries the whole document.
            const last = changes.array.items[changes.array.items.len - 1];
            const text = getStr(last, "text") orelse return true;
            try self.docs.put(self.gpa, uri, text, version);
            try self.publish(writer, uri);
            return true;
        }

        if (eql(method, "textDocument/didClose")) {
            const td = field(root, "params", "textDocument") orelse return true;
            const uri = getStr(td, "uri") orelse return true;
            const version = if (self.docs.get(uri)) |d| d.version else 0;
            self.docs.remove(self.gpa, uri);
            // Clear any diagnostics the client is still showing for the closed doc.
            try self.sendDiagnostics(writer, uri, version, &.{});
            return true;
        }

        if (eql(method, "textDocument/hover")) {
            const iv = id orelse return true; // a request must carry an id
            var hv: ?hover.Hover = null;
            defer if (hv) |*h| h.deinit(); // after writeResponse has serialized `value`
            var result: ?protocol.Hover = null;
            blk: {
                const params = objGet(root, "params") orelse break :blk;
                const td = objGet(params, "textDocument") orelse break :blk;
                const uri = getStr(td, "uri") orelse break :blk;
                const pos = objGet(params, "position") orelse break :blk;
                const line = getInt(pos, "line") orelse break :blk;
                const character = getInt(pos, "character") orelse break :blk;
                // Guard the WHOLE u32 range, not just `>= 0`: a valid i64 past u32 max would
                // panic the `@intCast` below, crashing the server on one malformed request.
                const max: i64 = std.math.maxInt(u32);
                if (line < 0 or character < 0 or line > max or character > max) break :blk;
                const doc = self.docs.get(uri) orelse break :blk; // hover before didOpen
                hv = hover.hoverAt(self.gpa, self.io(), self.scratch_file, doc.text, @intCast(line), @intCast(character)) catch break :blk;
                if (hv) |h| result = .{ .contents = .{ .kind = h.kind, .value = h.value } };
            }
            if (result) |r| {
                try protocol.writeResponse(self.gpa, writer, iv, r);
            } else {
                try protocol.writeResponse(self.gpa, writer, iv, std.json.Value{ .null = {} });
            }
            return true;
        }

        if (eql(method, "shutdown")) {
            self.shutdown_requested = true;
            if (id) |iv| try protocol.writeResponse(self.gpa, writer, iv, std.json.Value{ .null = {} });
            return true;
        }

        // Unknown method: a request is rejected; a notification (incl. `$/…`) is ignored.
        if (id) |iv| try protocol.writeError(self.gpa, writer, iv, protocol.err_code.method_not_found, "method not found");
        return true;
    }

    /// Check `uri`'s current buffer and publish the result. ALWAYS publishes (an empty
    /// array clears stale diagnostics on a now-clean document); a check fault publishes an
    /// empty set and continues (it is a notification — never a response, never a hang).
    fn publish(self: *Server, writer: *Writer, uri: []const u8) !void {
        const doc = self.docs.get(uri) orelse return;
        const version = doc.version;
        const text = doc.text;
        var mapped = diagnostics.checkBuffer(self.gpa, self.io(), self.scratch_file, uri, text) catch {
            try self.sendDiagnostics(writer, uri, version, &.{});
            return;
        };
        defer mapped.deinit();
        try self.sendDiagnostics(writer, uri, version, mapped.items);
    }

    fn sendDiagnostics(self: *Server, writer: *Writer, uri: []const u8, version: i64, items: []const protocol.LspDiagnostic) !void {
        try protocol.writeNotification(self.gpa, writer, "textDocument/publishDiagnostics", .{
            .uri = uri,
            .version = version,
            .diagnostics = items,
        });
    }
};

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn objGet(v: std.json.Value, key: []const u8) ?std.json.Value {
    if (v != .object) return null;
    return v.object.get(key);
}

/// `root.<a>.<b>` as a Value, or null if any hop is missing / not an object.
fn field(root: std.json.Value, a: []const u8, b: []const u8) ?std.json.Value {
    const first = objGet(root, a) orelse return null;
    return objGet(first, b);
}

fn getStr(v: std.json.Value, key: []const u8) ?[]const u8 {
    const f = objGet(v, key) orelse return null;
    return switch (f) {
        .string => |s| s,
        else => null,
    };
}

fn getInt(v: std.json.Value, key: []const u8) ?i64 {
    const f = objGet(v, key) orelse return null;
    return switch (f) {
        .integer => |i| i,
        else => null,
    };
}

const testing = std.testing;

/// Append one framed JSON message (built from `payload`) to `session`.
fn frameInto(gpa: std.mem.Allocator, session: *Writer.Allocating, payload: anytype) !void {
    var body: Writer.Allocating = .init(gpa);
    defer body.deinit();
    try std.json.Stringify.value(payload, .{ .emit_null_optional_fields = false }, &body.writer);
    try transport.writeMessage(&session.writer, body.written());
}

/// Re-drive `transport.readMessage` over the captured output until EndOfStream so the
/// frame count cannot be silently under-reported. Caller frees each body + the list.
fn splitFrames(gpa: std.mem.Allocator, raw: []const u8) !std.ArrayList([]u8) {
    var list: std.ArrayList([]u8) = .empty;
    errdefer {
        for (list.items) |b| gpa.free(b);
        list.deinit(gpa);
    }
    var r = Reader.fixed(raw);
    while (true) {
        const body = transport.readMessage(gpa, &r) catch |e| switch (e) {
            error.EndOfStream => break,
            else => return e,
        };
        try list.append(gpa, body);
    }
    return list;
}

/// The 0-based index of the line carrying the `#~ ERROR` annotation — computed from the
/// fixture itself, never from server output, so the range assertion is independent.
fn annotatedLine(src: []const u8) u32 {
    var i: u32 = 0;
    var it = std.mem.splitScalar(u8, src, '\n');
    while (it.next()) |line| : (i += 1) {
        if (std.mem.indexOf(u8, line, "#~ ERROR") != null) return i;
    }
    return 0;
}

/// The 0-based byte column of `needle` within line `line_idx` of `src` — computed from the
/// fixture, so hover positions are never hand-counted (ASCII, so byte == character).
fn colOf(src: []const u8, line_idx: u32, needle: []const u8) u32 {
    var it = std.mem.splitScalar(u8, src, '\n');
    var i: u32 = 0;
    while (it.next()) |line| : (i += 1) {
        if (i == line_idx) return @intCast(std.mem.indexOf(u8, line, needle).?);
    }
    unreachable;
}

fn hasCode(diags: std.json.Value, code: []const u8) bool {
    if (diags != .array) return false;
    for (diags.array.items) |d| {
        const c = getStr(d, "code") orelse continue;
        if (eql(c, code)) return true;
    }
    return false;
}

test "lsp e2e: initialize -> didOpen(diag) -> didChange(clean clears) -> broken(fault-tolerant) -> shutdown/exit" {
    const gpa = testing.allocator;
    var setup: std.Io.Threaded = .init(gpa, .{});
    defer setup.deinit();
    const sio = setup.io();

    const arity = try Io.Dir.cwd().readFileAlloc(sio, "tests/corpora/diagnostics/arity_mismatch.toy", gpa, .unlimited);
    defer gpa.free(arity);
    const broken = try Io.Dir.cwd().readFileAlloc(sio, "tests/corpora/diagnostics/parse_error.toy", gpa, .unlimited);
    defer gpa.free(broken);

    const arity_line = annotatedLine(arity);
    const clean = "fn main() -> int { return 0 }";
    const uri = "file:///d.toy";
    const Change = struct { text: []const u8 };

    // Build the whole session as one byte stream.
    var session: Writer.Allocating = .init(gpa);
    defer session.deinit();
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .id = @as(i64, 1),
        .method = "initialize",
        .params = .{ .capabilities = .{ .general = .{ .positionEncodings = &[_][]const u8{"utf-8"} } } },
    });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "initialized", .params = .{} });
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .method = "textDocument/didOpen",
        .params = .{ .textDocument = .{ .uri = uri, .languageId = "toy", .version = @as(i64, 1), .text = arity } },
    });
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .method = "textDocument/didChange",
        .params = .{ .textDocument = .{ .uri = uri, .version = @as(i64, 2) }, .contentChanges = &[_]Change{.{ .text = clean }} },
    });
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .method = "textDocument/didChange",
        .params = .{ .textDocument = .{ .uri = uri, .version = @as(i64, 3) }, .contentChanges = &[_]Change{.{ .text = broken }} },
    });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 2), .method = "shutdown" });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "exit" });

    // Drive the server end to end over in-memory buffers.
    var out: Writer.Allocating = .init(gpa);
    defer out.deinit();
    var reader = Reader.fixed(session.written());
    const code = try serve(gpa, &reader, &out.writer);
    try testing.expectEqual(ExitCode.ok, code); // shutdown preceded exit

    var frames = try splitFrames(gpa, out.written());
    defer {
        for (frames.items) |b| gpa.free(b);
        frames.deinit(gpa);
    }

    var saw_init = false;
    var saw_shutdown = false;
    var publishes: [8]std.json.Value = undefined;
    var publish_count: usize = 0;
    // One arena for every parsed frame; kept alive until all assertions are done.
    var parse_arena = std.heap.ArenaAllocator.init(gpa);
    defer parse_arena.deinit();
    const pa = parse_arena.allocator();

    for (frames.items) |body| {
        const v = (try std.json.parseFromSliceLeaky(std.json.Value, pa, body, .{}));
        if (objGet(v, "method")) |m| {
            if (m == .string and eql(m.string, "textDocument/publishDiagnostics")) {
                publishes[publish_count] = objGet(v, "params").?;
                publish_count += 1;
            }
            continue;
        }
        // A response: correlate by id.
        const idv = objGet(v, "id") orelse continue;
        if (idv == .integer and idv.integer == 1) {
            saw_init = true;
            const caps = field(v, "result", "capabilities").?;
            try testing.expectEqualStrings("utf-8", getStr(caps, "positionEncoding").?);
            const sync = objGet(caps, "textDocumentSync").?;
            try testing.expectEqual(@as(i64, 1), getInt(sync, "change").?);
            // The advertised capability set must be a SUBSET of what we implement — no
            // hover/completion/definition/etc. leaking in.
            var it = caps.object.iterator();
            while (it.next()) |e| {
                const k = e.key_ptr.*;
                try testing.expect(eql(k, "positionEncoding") or eql(k, "textDocumentSync") or eql(k, "hoverProvider"));
            }
        } else if (idv == .integer and idv.integer == 2) {
            saw_shutdown = true;
            try testing.expect(objGet(v, "result").? == .null);
        }
    }

    try testing.expectEqual(@as(usize, 3), publish_count); // three publishes, none dropped

    // v1: the arity error surfaces as T0039 on the annotated line, zero-width range.
    const d1 = objGet(publishes[0], "diagnostics").?;
    try testing.expect(hasCode(d1, "T0039"));
    for (d1.array.items) |d| {
        if (getStr(d, "code")) |c| if (eql(c, "T0039")) {
            const start = field(d, "range", "start").?;
            try testing.expectEqual(@as(i64, arity_line), getInt(start, "line").?);
            const end = field(d, "range", "end").?;
            try testing.expectEqual(getInt(start, "line").?, getInt(end, "line").?);
            try testing.expectEqual(getInt(start, "character").?, getInt(end, "character").?);
        };
    }

    // v2: the clean document CLEARS its diagnostics (the empty-publish assertion).
    const d2 = objGet(publishes[1], "diagnostics").?;
    try testing.expectEqual(@as(usize, 0), d2.array.items.len);

    // v3: a syntactically broken document still yields diagnostics (fault-tolerant).
    const d3 = objGet(publishes[2], "diagnostics").?;
    try testing.expect(hasCode(d3, "P0002"));

    try testing.expect(saw_init and saw_shutdown);
}

test "lsp hover: type at a binding/param/expr, signature at a callee, null over a gap and OOB" {
    const gpa = testing.allocator;

    // The leading indent on line 2 is the deliberate whitespace gap (id 6).
    const src = "fn add(a: int, b: int) -> int { return a + b }\nfn main() -> int {\n    x := add(1, 2)\n    return x\n}";
    const uri = "file:///h.toy";

    const Pos = struct { line: i64, character: i64 };
    const hoverReq = struct {
        fn make(id: i64, line: u32, character: u32) struct {
            jsonrpc: []const u8 = "2.0",
            id: i64,
            method: []const u8 = "textDocument/hover",
            params: struct { textDocument: struct { uri: []const u8 }, position: Pos },
        } {
            return .{ .id = id, .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = line, .character = character } } };
        }
    };

    var session: Writer.Allocating = .init(gpa);
    defer session.deinit();
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 1), .method = "initialize", .params = .{} });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "initialized", .params = .{} });
    // A hover BEFORE the document is opened -> null, no crash.
    try frameInto(gpa, &session, hoverReq.make(10, 0, 0));
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .method = "textDocument/didOpen",
        .params = .{ .textDocument = .{ .uri = uri, .languageId = "toy", .version = @as(i64, 1), .text = src } },
    });
    try frameInto(gpa, &session, hoverReq.make(2, 2, colOf(src, 2, "x :="))); // binding x
    try frameInto(gpa, &session, hoverReq.make(3, 0, colOf(src, 0, "a + b"))); // param-use a
    try frameInto(gpa, &session, hoverReq.make(4, 2, colOf(src, 2, "add("))); // callee add
    try frameInto(gpa, &session, hoverReq.make(5, 0, colOf(src, 0, "+ b"))); // expr +
    try frameInto(gpa, &session, hoverReq.make(6, 2, colOf(src, 2, "  "))); // leading-indent gap
    // Out-of-range line, and a character past u32 max (must NOT panic the @intCast).
    try frameInto(gpa, &session, hoverReq.make(7, 100000, 0));
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .id = @as(i64, 11),
        .method = "textDocument/hover",
        .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = @as(i64, 0), .character = @as(i64, 3000000000) } },
    });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 8), .method = "shutdown" });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "exit" });

    var out: Writer.Allocating = .init(gpa);
    defer out.deinit();
    var reader = Reader.fixed(session.written());
    const code = try serve(gpa, &reader, &out.writer);
    try testing.expectEqual(ExitCode.ok, code); // every request answered; no hang

    var frames = try splitFrames(gpa, out.written());
    defer {
        for (frames.items) |b| gpa.free(b);
        frames.deinit(gpa);
    }

    var parse_arena = std.heap.ArenaAllocator.init(gpa);
    defer parse_arena.deinit();
    const pa = parse_arena.allocator();

    // Collect every response by id (ids run 1..11).
    var by_id: [12]?std.json.Value = @splat(null);
    for (frames.items) |body| {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, pa, body, .{});
        if (objGet(v, "method") != null) continue; // a notification (publishDiagnostics)
        const idv = objGet(v, "id") orelse continue;
        if (idv == .integer and idv.integer >= 0 and idv.integer < 12) by_id[@intCast(idv.integer)] = v;
    }

    // id 1: hoverProvider is advertised as a boolean.
    const caps = field(by_id[1].?, "result", "capabilities").?;
    const hp = objGet(caps, "hoverProvider").?;
    try testing.expect(hp == .bool and hp.bool);

    // The positive cases FAIL on a null/absent value (non-vacuous).
    try testing.expectEqualStrings("int", hoverValue(by_id[2].?).?); // binding x
    try testing.expectEqualStrings("int", hoverValue(by_id[3].?).?); // param a
    try testing.expectEqualStrings("int", hoverValue(by_id[5].?).?); // expr +

    // id 4: callee signature.
    const callee = hoverValue(by_id[4].?).?;
    try testing.expectEqualStrings("fn add(int, int) -> int", callee);
    // The internal scratch-file module name must never leak into hover output.
    try testing.expect(std.mem.indexOf(u8, callee, "doc.") == null);

    // Every non-hit path answers with an explicit null result.
    try testing.expect(objGet(by_id[6].?, "result").? == .null); // whitespace gap
    try testing.expect(objGet(by_id[10].?, "result").? == .null); // before didOpen
    try testing.expect(objGet(by_id[7].?, "result").? == .null); // OOB line
    try testing.expect(objGet(by_id[11].?, "result").? == .null); // char > u32 max
}

/// The hover `result.contents.value`, or null if the response's result was JSON null.
fn hoverValue(resp: std.json.Value) ?[]const u8 {
    const result = objGet(resp, "result") orelse return null;
    const contents = objGet(result, "contents") orelse return null;
    return getStr(contents, "value");
}
