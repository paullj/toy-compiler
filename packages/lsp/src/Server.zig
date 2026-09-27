//! The JSON-RPC server core: an injectable reader/writer, the lifecycle handshake, the
//! request/notification dispatch, the document store, and didOpen/didChange -> check ->
//! publishDiagnostics. didChange uses INCREMENTAL sync (ranged deltas spliced in order,
//! with a whole-document replace as a fallback).
//!
//! The read model is anchor/drain/prescan/process: each turn does one blocking anchor read,
//! drains only the frames ALREADY buffered into a batch, prescans that batch for
//! `$/cancelRequest` ids, then processes the batch in order — a request whose id was
//! cancelled earlier in the same batch is answered RequestCancelled and does zero work. This
//! is best-effort in-batch cancellation: a serial one-at-a-time reader could never honor a
//! cancel that trails its request, so we drain what the client already sent before acting.
//!
//! Every REQUEST produces exactly one response (a result or an error), so a client is never
//! left hanging. Notifications never get a response. Only OOM / a dead transport are fatal to
//! the loop; a per-message fault is swallowed so one bad message can neither drop a later
//! response nor kill the session.

const std = @import("std");
const Io = std.Io;
const Reader = std.Io.Reader;
const Writer = std.Io.Writer;

const transport = @import("transport.zig");
const protocol = @import("protocol.zig");
const diagnostics = @import("diagnostics.zig");
const hover = @import("hover.zig");
const completion = @import("completion.zig");
const definition = @import("definition.zig");
const signature = @import("signature.zig");
const Documents = @import("Documents.zig");
const Workspace = @import("Workspace.zig");

pub const ExitCode = enum(u8) { ok = 0, no_shutdown = 1 };

/// Connect the server core to an injected reader + writer and run until `exit` or a clean
/// pipe close. `toy lsp` passes real stdin/stdout; the in-process test passes in-memory
/// buffers. Returns `.ok` when `shutdown` preceded `exit`, else `.no_shutdown` (exit 1).
pub fn serve(gpa: std.mem.Allocator, reader: *Reader, writer: *Writer) anyerror!ExitCode {
    var s = Server.init(gpa);
    defer s.deinit();
    return s.run(reader, writer);
}

pub const Server = struct {
    gpa: std.mem.Allocator,
    /// Owns its own compute io (discover / checkGraph). This is separate from the transport
    /// reader/writer, whose io was already consumed by the caller.
    threaded: std.Io.Threaded,
    /// Backs everything one message allocates (its JSON, the check, the response) and is
    /// reset after it, keeping its capacity: the heap then plateaus at the largest
    /// message instead of fragmenting (a wasm heap never shrinks). Documents use `gpa`.
    arena: std.heap.ArenaAllocator,
    docs: Documents,
    /// false on a host with no filesystem: imports resolve only against open documents.
    disk: bool = true,
    got_initialize: bool = false,
    shutdown_requested: bool = false,

    pub fn init(gpa: std.mem.Allocator) Server {
        return .{ .gpa = gpa, .threaded = .init(gpa, .{}), .arena = .init(gpa), .docs = .{} };
    }

    pub fn deinit(self: *Server) void {
        self.docs.deinit(self.gpa);
        self.arena.deinit();
        self.threaded.deinit();
        self.* = undefined;
    }

    /// Recomputed from the PINNED server each call: `Threaded.io()` captures
    /// `&self.threaded`, so it must never be cached across the by-value move out of `init`.
    fn workspace(self: *Server) Workspace {
        return .{ .io = self.threaded.io(), .docs = &self.docs, .disk = self.disk };
    }

    /// Serve every framed message in `input`, then return — for a host that pushes messages
    /// in (a browser worker) rather than the server pulling from a pipe. False once `exit`
    /// has been handled.
    pub fn feed(self: *Server, input: []const u8, writer: *Writer) anyerror!bool {
        var r = Reader.fixed(input);
        while (true) switch (try self.serveBatch(&r, writer)) {
            .more => {},
            .eof => return true,
            .exit => return false,
        };
    }

    fn run(self: *Server, reader: *Reader, writer: *Writer) anyerror!ExitCode {
        while (try self.serveBatch(reader, writer) == .more) {}
        return if (self.shutdown_requested) .ok else .no_shutdown;
    }

    const Turn = enum { more, eof, exit };

    /// One turn of the read model (see the file header): anchor, drain, prescan, process.
    fn serveBatch(self: *Server, reader: *Reader, writer: *Writer) anyerror!Turn {
        // P1 ANCHOR: one blocking read guarantees forward progress (or EOF => done). This is
        // the ONLY unconditional read; it blocks exactly like the old serial loop did.
        const first = transport.readMessage(self.gpa, reader) catch |e| switch (e) {
            error.EndOfStream => return .eof,
            error.UnexpectedEof, error.StreamTooLong, error.MissingContentLength, error.InvalidHeader => return .eof,
            error.OutOfMemory, error.ReadFailed => return e,
        };
        var batch: std.ArrayList([]u8) = .empty;
        defer {
            for (batch.items) |b| self.gpa.free(b);
            batch.deinit(self.gpa);
        }
        try batch.append(self.gpa, first);

        // P2 DRAIN: only frames ALREADY buffered. The bufferedLen()>0 gate is the sole thing
        // that stops us starting a read for a frame the client has not begun sending — never
        // drop it. It does NOT promise "never blocks": a partially-buffered trailing frame is
        // finished by readMessage (a bounded, client-is-mid-send wait, same as the old loop).
        // Fixed reader (test): the whole session is buffered, so this drains EVERY remaining
        // frame into one batch => a $/cancelRequest anywhere precedes its target in P4.
        // Streaming pipe (real client): only what a syscall already delivered.
        var terminal = false;
        while (reader.bufferedLen() > 0) {
            const b = transport.readMessage(self.gpa, reader) catch |e| switch (e) {
                // OOM/ReadFailed are fatal exactly as in the serial loop; queued-but-unprocessed
                // frames in this batch are dropped (freed by defer) — the transport is dying.
                error.OutOfMemory, error.ReadFailed => return e,
                else => {
                    terminal = true;
                    break;
                },
            };
            try batch.append(self.gpa, b);
        }

        // P3 PRESCAN: collect cancelled ids across the WHOLE batch, before processing any of it.
        var cancelled: CancelSet = .{ .gpa = self.gpa };
        defer cancelled.deinit();
        for (batch.items) |b| try collectCancel(self.gpa, b, &cancelled);

        // P4 PROCESS in order, honoring cancels. A per-message fault is not fatal (unchanged).
        for (batch.items) |b| {
            const keep = self.dispatch(writer, b, &cancelled) catch |e| switch (e) {
                error.OutOfMemory, error.WriteFailed => return e,
                else => true,
            };
            if (!keep) return .exit;
        }
        return if (terminal) .eof else .more;
    }

    /// Handle one message. Returns `false` ONLY on `exit` (stop the loop); `true` keeps
    /// serving. A parse failure or a message that isn't a request-for-us is ignored.
    fn dispatch(self: *Server, writer: *Writer, body: []const u8, cancelled: *const CancelSet) anyerror!bool {
        defer _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const root = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch return true;
        if (root != .object) return true;

        // No `method` => this is a RESPONSE to us; we send no server->client requests, so
        // there is nothing to correlate. Ignore it.
        const method_v = root.object.get("method") orelse return true;
        if (method_v != .string) return true;
        const method = method_v.string;
        // Present iff this is a request; absent for a notification.
        const id = root.object.get("id");

        // A request the client already cancelled: answer RequestCancelled and do NONE of the
        // work (no check, no feature call, no publish). Notifications carry
        // no top-level id and are never cancellable; `$/cancelRequest` carries its target in
        // params, not a top-level id, so it never self-matches — it falls through to the
        // ignored-notification path below (a notification is never answered). The gate is
        // uniform: a cancelled initialize/shutdown is also answered -32800 and its state side
        // effect is skipped — spec-legal (a server MAY cancel any request); left uniform.
        if (id) |iv| {
            if (cancelled.contains(iv)) {
                try protocol.writeError(a, writer, iv, protocol.err_code.request_cancelled, "request cancelled");
                return true;
            }
        }

        if (eql(method, "exit")) return false;

        if (eql(method, "initialize")) {
            if (self.got_initialize) {
                if (id) |iv| try protocol.writeError(a, writer, iv, protocol.err_code.invalid_request, "server already initialized");
                return true;
            }
            self.got_initialize = true;
            if (id) |iv| try protocol.writeResponse(a, writer, iv, protocol.InitializeResult{});
            return true;
        }

        // Everything except initialize/exit must wait for initialization.
        if (!self.got_initialize) {
            if (id) |iv| try protocol.writeError(a, writer, iv, protocol.err_code.server_not_initialized, "server not initialized");
            return true;
        }

        if (eql(method, "initialized")) return true;

        if (eql(method, "textDocument/didOpen")) {
            const td = field(root, "params", "textDocument") orelse return true;
            const uri = getStr(td, "uri") orelse return true;
            const text = getStr(td, "text") orelse return true;
            const version = getInt(td, "version") orelse 0;
            try self.docs.put(self.gpa, uri, text, version);
            try self.publish(a, writer, uri);
            return true;
        }

        if (eql(method, "textDocument/didChange")) {
            const params = objGet(root, "params") orelse return true;
            const td = objGet(params, "textDocument") orelse return true;
            const uri = getStr(td, "uri") orelse return true;
            const version = getInt(td, "version") orelse 0;
            const changes = objGet(params, "contentChanges") orelse return true;
            if (changes != .array) return true;
            // Incremental sync: apply each change IN ORDER, each relative to the doc AFTER the
            // previous one. A change with NO `range` is a whole-document replace (clients that
            // opt out of deltas). Protocol-exact SYNC only: publish() below still re-checks the
            // whole buffer and re-lexes/re-parses this edited file.
            for (changes.array.items) |change| {
                const text = getStr(change, "text") orelse continue;
                const range = objGet(change, "range") orelse {
                    try self.docs.put(self.gpa, uri, text, version); // full replace
                    continue;
                };
                const d = self.docs.get(uri) orelse continue; // a didChange before didOpen
                const s = objGet(range, "start") orelse continue;
                const e = objGet(range, "end") orelse continue;
                // Offsets are rebuilt against the CURRENT text: spliceRange reallocs d.text and
                // each delta's coords are against the post-previous-delta text.
                const doc_end: u32 = @intCast(d.text.len);
                const sl = coordU32(getInt(s, "line")) orelse continue;
                const sc = coordU32(getInt(s, "character")) orelse continue;
                const el = coordU32(getInt(e, "line")) orelse continue;
                const ec = coordU32(getInt(e, "character")) orelse continue;
                const start_off = (try hover.offsetIn(a, d.text, sl, sc)) orelse doc_end;
                var end_off = (try hover.offsetIn(a, d.text, el, ec)) orelse doc_end;
                if (end_off < start_off) end_off = start_off; // tolerate an inverted range
                try self.docs.spliceRange(self.gpa, uri, start_off, end_off, text);
            }
            self.docs.setVersion(uri, version);
            try self.publish(a, writer, uri);
            return true;
        }

        if (eql(method, "textDocument/didClose")) {
            const td = field(root, "params", "textDocument") orelse return true;
            const uri = getStr(td, "uri") orelse return true;
            const version = if (self.docs.get(uri)) |d| d.version else 0;
            self.docs.remove(self.gpa, uri);
            // Clear any diagnostics the client is still showing for the closed doc.
            try sendDiagnostics(a, writer, uri, version, &.{});
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
                hv = hover.hoverAt(a, self.workspace(), doc.text, @intCast(line), @intCast(character), uri) catch break :blk;
                if (hv) |h| result = .{ .contents = .{ .kind = h.kind, .value = h.value } };
            }
            if (result) |r| {
                try protocol.writeResponse(a, writer, iv, r);
            } else {
                try protocol.writeResponse(a, writer, iv, std.json.Value{ .null = {} });
            }
            return true;
        }

        if (eql(method, "textDocument/completion")) {
            const iv = id orelse return true; // a request must carry an id
            var comp: ?completion.Completions = null;
            defer if (comp) |*c| c.deinit(); // after writeResponse has serialized the items
            var items: []const protocol.CompletionItem = &.{};
            blk: {
                const params = objGet(root, "params") orelse break :blk;
                const td = objGet(params, "textDocument") orelse break :blk;
                const uri = getStr(td, "uri") orelse break :blk;
                const pos = objGet(params, "position") orelse break :blk;
                const line = getInt(pos, "line") orelse break :blk;
                const character = getInt(pos, "character") orelse break :blk;
                // Guard the whole u32 range (mirrors hover): an i64 past u32 max would
                // panic the `@intCast` and crash the server on one malformed request.
                const max: i64 = std.math.maxInt(u32);
                if (line < 0 or character < 0 or line > max or character > max) break :blk;
                const doc = self.docs.get(uri) orelse break :blk; // completion before didOpen
                comp = completion.completionsAt(a, self.workspace(), doc.text, @intCast(line), @intCast(character), uri) catch break :blk;
                if (comp) |c| items = c.items;
            }
            // Always answer with an array (an empty list is a valid "no candidates").
            try protocol.writeResponse(a, writer, iv, items);
            return true;
        }

        if (eql(method, "textDocument/definition")) {
            const iv = id orelse return true; // a request must carry an id
            var result: ?protocol.Location = null;
            var owned_uri: ?[]u8 = null;
            // Freed at if-branch scope AFTER writeResponse has serialized `result.uri`; must
            // NOT sit inside `blk` (the happy path sets result inside blk, so an in-blk defer
            // would free the uri before serialization — a use-after-free).
            defer if (owned_uri) |u| a.free(u);
            blk: {
                const params = objGet(root, "params") orelse break :blk;
                const td = objGet(params, "textDocument") orelse break :blk;
                const uri = getStr(td, "uri") orelse break :blk;
                const pos = objGet(params, "position") orelse break :blk;
                const line = getInt(pos, "line") orelse break :blk;
                const character = getInt(pos, "character") orelse break :blk;
                // Guard the whole u32 range (mirrors hover/completion): an i64 past u32 max
                // would panic the `@intCast` and crash the server on one malformed request.
                const max: i64 = std.math.maxInt(u32);
                if (line < 0 or character < 0 or line > max or character > max) break :blk;
                const doc = self.docs.get(uri) orelse break :blk; // definition before didOpen
                const d = (definition.definitionAt(a, self.workspace(), doc.text, @intCast(line), @intCast(character), uri) catch break :blk) orelse break :blk;
                // The uri is either the request's own open-doc uri or an imported module's
                // `file://` path.
                owned_uri = d.uri;
                result = .{ .uri = d.uri, .range = d.range };
            }
            if (result) |r| {
                try protocol.writeResponse(a, writer, iv, r);
            } else {
                try protocol.writeResponse(a, writer, iv, std.json.Value{ .null = {} });
            }
            return true;
        }

        if (eql(method, "textDocument/signatureHelp")) {
            const iv = id orelse return true; // a request must carry an id
            var sh: ?signature.Result = null;
            defer if (sh) |*x| x.deinit(); // after writeResponse has serialized the label
            var result: ?protocol.SignatureHelp = null;
            blk: {
                const params = objGet(root, "params") orelse break :blk;
                const td = objGet(params, "textDocument") orelse break :blk;
                const uri = getStr(td, "uri") orelse break :blk;
                const pos = objGet(params, "position") orelse break :blk;
                const line = getInt(pos, "line") orelse break :blk;
                const character = getInt(pos, "character") orelse break :blk;
                // Guard the whole u32 range (mirrors hover/completion/definition): an i64 past
                // u32 max would panic the `@intCast` and crash the server on one bad request.
                const max: i64 = std.math.maxInt(u32);
                if (line < 0 or character < 0 or line > max or character > max) break :blk;
                const doc = self.docs.get(uri) orelse break :blk; // signatureHelp before didOpen
                sh = signature.signatureHelpAt(a, self.workspace(), doc.text, @intCast(line), @intCast(character), uri) catch break :blk;
                if (sh) |x| result = x.help;
            }
            if (result) |r| {
                try protocol.writeResponse(a, writer, iv, r);
            } else {
                try protocol.writeResponse(a, writer, iv, std.json.Value{ .null = {} });
            }
            return true;
        }

        if (eql(method, "shutdown")) {
            self.shutdown_requested = true;
            if (id) |iv| try protocol.writeResponse(a, writer, iv, std.json.Value{ .null = {} });
            return true;
        }

        // Unknown method: a request is rejected; a notification (incl. `$/…`) is ignored.
        if (id) |iv| try protocol.writeError(a, writer, iv, protocol.err_code.method_not_found, "method not found");
        return true;
    }

    /// Check `uri`'s current buffer and publish the result. ALWAYS publishes (an empty
    /// array clears stale diagnostics on a now-clean document); a check fault publishes an
    /// empty set and continues (it is a notification — never a response, never a hang).
    fn publish(self: *Server, a: std.mem.Allocator, writer: *Writer, uri: []const u8) !void {
        const doc = self.docs.get(uri) orelse return;
        const version = doc.version;
        const text = doc.text;
        var mapped = diagnostics.checkBuffer(a, self.workspace(), uri, text) catch {
            try sendDiagnostics(a, writer, uri, version, &.{});
            return;
        };
        defer mapped.deinit();
        try sendDiagnostics(a, writer, uri, version, mapped.items);
    }
};

fn sendDiagnostics(a: std.mem.Allocator, writer: *Writer, uri: []const u8, version: i64, items: []const protocol.LspDiagnostic) !void {
    try protocol.writeNotification(a, writer, "textDocument/publishDiagnostics", .{
        .uri = uri,
        .version = version,
        .diagnostics = items,
    });
}

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

const CancelId = union(enum) { int: i64, str: []u8 };

/// Request ids cancelled within the CURRENT drain batch. Strings are OWNED — batch bodies
/// (and their parse arenas) are freed as processed, so a borrowed id would dangle. A fresh
/// set per batch is deliberate: a cancel only affects requests in its OWN batch, so a cancel
/// that arrives after its request was already answered (a later batch) is correctly a no-op
/// (no double-response), and a reused id in a later batch is never poisoned by a stale cancel.
const CancelSet = struct {
    gpa: std.mem.Allocator,
    ids: std.ArrayList(CancelId) = .empty,

    fn deinit(self: *CancelSet) void {
        for (self.ids.items) |c| switch (c) {
            .str => |s| self.gpa.free(s),
            .int => {},
        };
        self.ids.deinit(self.gpa);
    }
    fn add(self: *CancelSet, id: std.json.Value) !void {
        switch (id) {
            .integer => |i| try self.ids.append(self.gpa, .{ .int = i }),
            .string => |s| try self.ids.append(self.gpa, .{ .str = try self.gpa.dupe(u8, s) }),
            else => {},
        }
    }
    fn contains(self: *const CancelSet, id: std.json.Value) bool {
        for (self.ids.items) |c| switch (c) {
            .int => |i| if (id == .integer and id.integer == i) return true,
            .str => |s| if (id == .string and eql(id.string, s)) return true,
        };
        return false;
    }
};

/// Prescan one raw frame: if it is `$/cancelRequest`, record its `params.id`. A parse
/// failure or non-cancel frame is ignored (it is handled/rejected in the process phase).
fn collectCancel(gpa: std.mem.Allocator, body: []const u8, set: *CancelSet) !void {
    var parsed = std.json.parseFromSlice(std.json.Value, gpa, body, .{}) catch return;
    defer parsed.deinit();
    const root = parsed.value;
    const m = objGet(root, "method") orelse return;
    if (m != .string or !eql(m.string, "$/cancelRequest")) return;
    const params = objGet(root, "params") orelse return;
    const id = objGet(params, "id") orelse return;
    try set.add(id);
}

/// An LSP position coordinate -> u32 guarding the whole u32 range; null/negative/over-max
/// => null so a malformed range skips its change (mirrors the feature handlers' guards).
fn coordU32(v: ?i64) ?u32 {
    const i = v orelse return null;
    if (i < 0 or i > std.math.maxInt(u32)) return null;
    return @intCast(i);
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
            try testing.expectEqual(@as(i64, 2), getInt(sync, "change").?);
            // The advertised capability set must be a SUBSET of what we implement — no
            // hover/completion/definition/etc. leaking in.
            var it = caps.object.iterator();
            while (it.next()) |e| {
                const k = e.key_ptr.*;
                try testing.expect(eql(k, "positionEncoding") or eql(k, "textDocumentSync") or eql(k, "hoverProvider") or eql(k, "completionProvider") or eql(k, "definitionProvider") or eql(k, "signatureHelpProvider"));
            }
        } else if (idv == .integer and idv.integer == 2) {
            saw_shutdown = true;
            try testing.expect(objGet(v, "result").? == .null);
        }
    }

    try testing.expectEqual(@as(usize, 3), publish_count); // three publishes, none dropped

    // v1: the arity error surfaces as T0039 on the annotated line, over one line.
    const d1 = objGet(publishes[0], "diagnostics").?;
    try testing.expect(hasCode(d1, "T0039"));
    for (d1.array.items) |d| {
        if (getStr(d, "code")) |c| if (eql(c, "T0039")) {
            const start = field(d, "range", "start").?;
            try testing.expectEqual(@as(i64, arity_line), getInt(start, "line").?);
            const end = field(d, "range", "end").?;
            try testing.expectEqual(getInt(start, "line").?, getInt(end, "line").?);
            try testing.expect(getInt(start, "character").? < getInt(end, "character").?);
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
    try testing.expectEqualStrings("x: int", hoverValue(by_id[2].?).?); // binding x
    try testing.expectEqualStrings("a: int", hoverValue(by_id[3].?).?); // param a
    try testing.expectEqualStrings("int", hoverValue(by_id[5].?).?); // expr +

    // id 4: callee signature.
    const callee = hoverValue(by_id[4].?).?;
    try testing.expectEqualStrings("fn add(a: int, b: int) -> int", callee);
    // The entry module qualifier must never leak into hover output.
    try testing.expect(std.mem.indexOf(u8, callee, "doc.") == null);

    // Every non-hit path answers with an explicit null result.
    try testing.expect(objGet(by_id[6].?, "result").? == .null); // whitespace gap
    try testing.expect(objGet(by_id[10].?, "result").? == .null); // before didOpen
    try testing.expect(objGet(by_id[7].?, "result").? == .null); // OOB line
    try testing.expect(objGet(by_id[11].?, "result").? == .null); // char > u32 max
}

/// The hover `result.contents.value`, or null if the response's result was JSON null.
/// The hover's source text, unwrapped from the ```toy fence it is rendered in.
fn hoverValue(resp: std.json.Value) ?[]const u8 {
    const result = objGet(resp, "result") orelse return null;
    const contents = objGet(result, "contents") orelse return null;
    if (!eql(getStr(contents, "kind") orelse return null, "markdown")) return null;
    const v = getStr(contents, "value") orelse return null;
    const open = "```toy\n";
    const close = "\n```";
    if (!std.mem.startsWith(u8, v, open) or !std.mem.endsWith(u8, v, close)) return null;
    return v[open.len .. v.len - close.len];
}

/// A 0-based (line, character) position just PAST the last occurrence of `needle` in
/// `src` — computed from the fixture so completion cursors are never hand-counted
/// (ASCII, so byte == character).
fn posAfterLast(src: []const u8, needle: []const u8) protocol.Position {
    const idx = std.mem.lastIndexOf(u8, src, needle).? + needle.len;
    var line: u32 = 0;
    var col: u32 = 0;
    for (src[0..idx]) |ch| {
        if (ch == '\n') {
            line += 1;
            col = 0;
        } else col += 1;
    }
    return .{ .line = line, .character = col };
}

/// The response's `result` array (the completion items), or null.
fn resultArr(resp: std.json.Value) ?[]std.json.Value {
    const result = objGet(resp, "result") orelse return null;
    return if (result == .array) result.array.items else null;
}

/// The `kind` of the first completion item labeled `label`, or null if none.
fn labelKind(items: []const std.json.Value, label: []const u8) ?i64 {
    for (items) |it| {
        const l = getStr(it, "label") orelse continue;
        if (eql(l, label)) return getInt(it, "kind");
    }
    return null;
}

fn hasLabel(items: []const std.json.Value, label: []const u8) bool {
    return labelKind(items, label) != null;
}

test "lsp completion: scope, member (fields+methods), module members, and a broken buffer" {
    const gpa = testing.allocator;

    const scope_src =
        \\struct Counter { n: int }
        \\impl Counter { fn bump(self) -> int { return self.n } }
        \\fn helper(x: int) -> int { return x }
        \\fn main(arg: int) -> int {
        \\    c := Counter { n: 0 }
        \\    count := 10
        \\    return count
        \\}
    ;
    const member_src =
        \\struct Point { x: int, y: int }
        \\impl Point { fn mag(self) -> int { return self.x } }
        \\fn main() -> int {
        \\    p := Point { x: 1, y: 2 }
        \\    return p.x
        \\}
    ;
    const module_src =
        \\import std/io
        \\fn main() -> int {
        \\    io.println("hi")
        \\    return 0
        \\}
    ;
    // The trailing `.` is the ONLY parse-level error; the missing return is a
    // tolerated flow/type error.
    const broken_src =
        \\struct S { a: int, b: int }
        \\fn main() -> int {
        \\    s := S { a: 1, b: 2 }
        \\    s.
        \\}
    ;

    const u_scope = "file:///scope.toy";
    const u_member = "file:///member.toy";
    const u_module = "file:///module.toy";
    const u_broken = "file:///broken.toy";

    const Pos = struct { line: u32, character: u32 };
    const complReq = struct {
        fn make(id: i64, uri: []const u8, p: protocol.Position) struct {
            jsonrpc: []const u8 = "2.0",
            id: i64,
            method: []const u8 = "textDocument/completion",
            params: struct { textDocument: struct { uri: []const u8 }, position: Pos },
        } {
            return .{ .id = id, .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = p.line, .character = p.character } } };
        }
    };
    const openDoc = struct {
        fn make(uri: []const u8, text: []const u8) struct {
            jsonrpc: []const u8 = "2.0",
            method: []const u8 = "textDocument/didOpen",
            params: struct { textDocument: struct { uri: []const u8, languageId: []const u8 = "toy", version: i64 = 1, text: []const u8 } },
        } {
            return .{ .params = .{ .textDocument = .{ .uri = uri, .text = text } } };
        }
    };

    var session: Writer.Allocating = .init(gpa);
    defer session.deinit();
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 1), .method = "initialize", .params = .{} });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "initialized", .params = .{} });
    try frameInto(gpa, &session, openDoc.make(u_scope, scope_src));
    try frameInto(gpa, &session, openDoc.make(u_member, member_src));
    try frameInto(gpa, &session, openDoc.make(u_module, module_src));
    try frameInto(gpa, &session, openDoc.make(u_broken, broken_src));
    try frameInto(gpa, &session, complReq.make(20, u_scope, posAfterLast(scope_src, "count")));
    try frameInto(gpa, &session, complReq.make(21, u_member, posAfterLast(member_src, "p.")));
    try frameInto(gpa, &session, complReq.make(22, u_module, posAfterLast(module_src, "io.")));
    try frameInto(gpa, &session, complReq.make(23, u_broken, posAfterLast(broken_src, "s.")));
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 99), .method = "shutdown" });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "exit" });

    var out: Writer.Allocating = .init(gpa);
    defer out.deinit();
    var reader = Reader.fixed(session.written());
    const code = try serve(gpa, &reader, &out.writer);
    try testing.expectEqual(ExitCode.ok, code); // every request answered; no hang/crash

    var frames = try splitFrames(gpa, out.written());
    defer {
        for (frames.items) |b| gpa.free(b);
        frames.deinit(gpa);
    }

    var parse_arena = std.heap.ArenaAllocator.init(gpa);
    defer parse_arena.deinit();
    const pa = parse_arena.allocator();

    var by_id: [100]?std.json.Value = @splat(null);
    for (frames.items) |body| {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, pa, body, .{});
        if (objGet(v, "method") != null) continue;
        const idv = objGet(v, "id") orelse continue;
        if (idv == .integer and idv.integer >= 0 and idv.integer < 100) by_id[@intCast(idv.integer)] = v;
    }

    // Capability: the `.` trigger is advertised.
    const caps = field(by_id[1].?, "result", "capabilities").?;
    const cp = objGet(caps, "completionProvider").?;
    const trigs = objGet(cp, "triggerCharacters").?;
    try testing.expect(trigs == .array and trigs.array.items.len == 1);
    try testing.expectEqualStrings(".", trigs.array.items[0].string);

    // (1) SCOPE: enclosing-fn local + param + a visible top-level fn + a type + a keyword.
    const scope = resultArr(by_id[20].?).?;
    try testing.expectEqual(@as(?i64, 6), labelKind(scope, "count")); // local
    try testing.expectEqual(@as(?i64, 6), labelKind(scope, "arg")); // enclosing param
    try testing.expectEqual(@as(?i64, 3), labelKind(scope, "helper")); // top-level fn
    try testing.expectEqual(@as(?i64, 22), labelKind(scope, "Counter")); // top-level struct
    try testing.expectEqual(@as(?i64, 14), labelKind(scope, "if")); // keyword
    // A method must NOT leak into plain scope as a free fn; another fn's param must not
    // leak across the fn boundary; a field is not an in-scope name.
    try testing.expect(!hasLabel(scope, "bump"));
    try testing.expect(!hasLabel(scope, "x"));
    try testing.expect(!hasLabel(scope, "n"));

    // (2) MEMBER: the receiver struct's fields + method, and NOT unrelated globals.
    const member = resultArr(by_id[21].?).?;
    try testing.expectEqual(@as(?i64, 5), labelKind(member, "x")); // field
    try testing.expectEqual(@as(?i64, 5), labelKind(member, "y")); // field
    try testing.expectEqual(@as(?i64, 2), labelKind(member, "mag")); // method
    try testing.expect(!hasLabel(member, "main"));
    try testing.expect(!hasLabel(member, "Point"));
    try testing.expect(!hasLabel(member, "if"));

    // (3) MODULE: the imported module's pub fns, and no scratch-file qualifier leak.
    const module = resultArr(by_id[22].?).?;
    try testing.expectEqual(@as(?i64, 3), labelKind(module, "print"));
    try testing.expectEqual(@as(?i64, 3), labelKind(module, "println"));
    try testing.expect(!hasLabel(module, "main"));
    for (module) |it| {
        const l = getStr(it, "label").?;
        try testing.expect(std.mem.indexOfScalar(u8, l, '.') == null);
    }

    // (4) INCOMPLETE (fault-tolerant): a trailing `.` still yields the receiver's fields.
    const broken = resultArr(by_id[23].?).?;
    try testing.expectEqual(@as(?i64, 5), labelKind(broken, "a"));
    try testing.expectEqual(@as(?i64, 5), labelKind(broken, "b"));
}

/// The `range.{start,end}.{line,character}` of a definition `Location` response, or null if
/// the response's `result` was JSON null.
const DefRange = struct { start_line: i64, start_char: i64, end_line: i64, end_char: i64 };

fn defRange(resp: std.json.Value) ?DefRange {
    const result = objGet(resp, "result") orelse return null;
    if (result != .object) return null; // JSON null (no-hit) is not an object
    const start = field(result, "range", "start") orelse return null;
    const end = field(result, "range", "end") orelse return null;
    return .{
        .start_line = getInt(start, "line").?,
        .start_char = getInt(start, "character").?,
        .end_line = getInt(end, "line").?,
        .end_char = getInt(end, "character").?,
    };
}

test "lsp definition: local, param, top-level fn, type; null over a gap/OOB/pre-open; never leaks the scratch uri" {
    const gpa = testing.allocator;

    // Needles are pinned to unambiguous substrings so every asserted column comes from
    // `colOf`, never a hand count. Cross-file (imported symbols) is intentionally OUT of
    // scope here — this covers WITHIN-file navigation only.
    const src =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\struct Point { x: int, y: int }
        \\fn main() -> int {
        \\    p := Point { x: 1, y: 2 }
        \\    s := add(p.x, p.y)
        \\    return s
        \\}
    ;
    const uri = "file:///def.toy";

    const Pos = struct { line: i64, character: i64 };
    const defReq = struct {
        fn make(id: i64, line: u32, character: u32) struct {
            jsonrpc: []const u8 = "2.0",
            id: i64,
            method: []const u8 = "textDocument/definition",
            params: struct { textDocument: struct { uri: []const u8 }, position: Pos },
        } {
            return .{ .id = id, .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = line, .character = character } } };
        }
    };

    var session: Writer.Allocating = .init(gpa);
    defer session.deinit();
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 1), .method = "initialize", .params = .{} });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "initialized", .params = .{} });
    // A definition BEFORE the document is opened -> null, no crash.
    try frameInto(gpa, &session, defReq.make(10, 0, 0));
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .method = "textDocument/didOpen",
        .params = .{ .textDocument = .{ .uri = uri, .languageId = "toy", .version = @as(i64, 1), .text = src } },
    });
    try frameInto(gpa, &session, defReq.make(2, 5, colOf(src, 5, "s"))); // use of local `s`
    try frameInto(gpa, &session, defReq.make(3, 0, colOf(src, 0, "a + b"))); // use of param `a`
    try frameInto(gpa, &session, defReq.make(4, 4, colOf(src, 4, "add("))); // use of fn `add`
    try frameInto(gpa, &session, defReq.make(5, 3, colOf(src, 3, "Point"))); // use of type `Point`
    try frameInto(gpa, &session, defReq.make(6, 3, 0)); // leading-indent gap -> null
    try frameInto(gpa, &session, defReq.make(7, 100000, 0)); // OOB line -> null
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .id = @as(i64, 11),
        .method = "textDocument/definition",
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

    // Both the parsed value AND the raw frame bytes, by id (ids run 1..11): the leak check
    // asserts over the RAW response bytes, not the re-serialized parse.
    var by_id: [12]?std.json.Value = @splat(null);
    var raw_by_id: [12]?[]const u8 = @splat(null);
    for (frames.items) |body| {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, pa, body, .{});
        if (objGet(v, "method") != null) continue; // a notification
        const idv = objGet(v, "id") orelse continue;
        if (idv == .integer and idv.integer >= 0 and idv.integer < 12) {
            by_id[@intCast(idv.integer)] = v;
            raw_by_id[@intCast(idv.integer)] = body;
        }
    }

    // id 1: definitionProvider is advertised as a boolean.
    const caps = field(by_id[1].?, "result", "capabilities").?;
    const dp = objGet(caps, "definitionProvider").?;
    try testing.expect(dp == .bool and dp.bool);

    // The expected DECL ranges — every column derived from the fixture via `colOf`.
    const Case = struct { id: usize, line: i64, start: u32, len: u32 };
    const cases = [_]Case{
        .{ .id = 2, .line = 4, .start = colOf(src, 4, "s :="), .len = 1 }, // local `s`
        .{ .id = 3, .line = 0, .start = colOf(src, 0, "a: int"), .len = 1 }, // param `a`
        .{ .id = 4, .line = 0, .start = colOf(src, 0, "add"), .len = 3 }, // fn `add` (name, not `fn`)
        .{ .id = 5, .line = 1, .start = colOf(src, 1, "Point"), .len = 5 }, // type `Point`
    };
    for (cases) |c| {
        const r = defRange(by_id[c.id].?).?; // fails on a null result (non-vacuous)
        try testing.expectEqualStrings(uri, getStr(objGet(by_id[c.id].?, "result").?, "uri").?);
        try testing.expectEqual(c.line, r.start_line);
        try testing.expectEqual(@as(i64, c.start), r.start_char);
        try testing.expectEqual(c.line, r.end_line);
        try testing.expectEqual(@as(i64, c.start + c.len), r.end_char); // start != end -> non-vacuous
        // The internal scratch path must never surface in the response bytes.
        try testing.expect(std.mem.indexOf(u8, raw_by_id[c.id].?, ".toy-lsp") == null);
        try testing.expect(std.mem.indexOf(u8, raw_by_id[c.id].?, "doc.toy") == null);
    }

    // Every non-hit path answers with an explicit null result.
    try testing.expect(objGet(by_id[6].?, "result").? == .null); // whitespace gap
    try testing.expect(objGet(by_id[7].?, "result").? == .null); // OOB line
    try testing.expect(objGet(by_id[10].?, "result").? == .null); // before didOpen
    try testing.expect(objGet(by_id[11].?, "result").? == .null); // char > u32 max
}

/// The `result.signatures[0].label` of a signatureHelp response, or null if the result was
/// JSON null (no-hit).
fn sigLabel(resp: std.json.Value) ?[]const u8 {
    const result = objGet(resp, "result") orelse return null;
    if (result != .object) return null;
    const sigs = objGet(result, "signatures") orelse return null;
    if (sigs != .array or sigs.array.items.len == 0) return null;
    return getStr(sigs.array.items[0], "label");
}

/// The `result.activeParameter`, or null on a no-hit.
fn sigActive(resp: std.json.Value) ?i64 {
    const result = objGet(resp, "result") orelse return null;
    if (result != .object) return null;
    return getInt(result, "activeParameter");
}

/// The `[start,end)` label span of `result.signatures[0].parameters[i]`, or null.
fn sigParam(resp: std.json.Value, i: usize) ?[2]i64 {
    const result = objGet(resp, "result") orelse return null;
    if (result != .object) return null;
    const sigs = objGet(result, "signatures") orelse return null;
    if (sigs != .array or sigs.array.items.len == 0) return null;
    const params = objGet(sigs.array.items[0], "parameters") orelse return null;
    if (params != .array or i >= params.array.items.len) return null;
    const span = objGet(params.array.items[i], "label") orelse return null;
    if (span != .array or span.array.items.len != 2) return null;
    return .{ span.array.items[0].integer, span.array.items[1].integer };
}

test "lsp signatureHelp: active param, innermost nested callee, incomplete call, null outside/pre-open/OOB, no scratch leak" {
    const gpa = testing.allocator;

    const src =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn id(x: int) -> int { return x }
        \\fn main() -> int {
        \\    x := add(1, 2)
        \\    return x
        \\}
    ;
    // Cursor inside the INNER unary call must pick `id`, not the outer `add`.
    const nested =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn id(x: int) -> int { return x }
        \\fn main() -> int {
        \\    y := add(id(1), 2)
        \\    return y
        \\}
    ;
    // Truncated after the first comma+space: no `)`, no closing `}`.
    const incomplete = "fn add(a: int, b: int) -> int { return a + b }\nfn main() -> int {\n    x := add(1, ";

    const u_src = "file:///sig.toy";
    const u_nested = "file:///nested.toy";
    const u_inc = "file:///inc.toy";

    const Pos = struct { line: u32, character: u32 };
    const sigReq = struct {
        fn make(id: i64, uri: []const u8, p: protocol.Position) struct {
            jsonrpc: []const u8 = "2.0",
            id: i64,
            method: []const u8 = "textDocument/signatureHelp",
            params: struct { textDocument: struct { uri: []const u8 }, position: Pos },
        } {
            return .{ .id = id, .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = p.line, .character = p.character } } };
        }
    };
    const openDoc = struct {
        fn make(uri: []const u8, text: []const u8) struct {
            jsonrpc: []const u8 = "2.0",
            method: []const u8 = "textDocument/didOpen",
            params: struct { textDocument: struct { uri: []const u8, languageId: []const u8 = "toy", version: i64 = 1, text: []const u8 } },
        } {
            return .{ .params = .{ .textDocument = .{ .uri = uri, .text = text } } };
        }
    };

    const before_comma = posAfterLast(src, "add(1");
    const after_comma = posAfterLast(src, "add(1, ");
    const inner = posAfterLast(nested, "id(1");
    const inc_pos = posAfterLast(incomplete, "add(1, ");
    const outside: protocol.Position = .{ .line = 4, .character = colOf(src, 4, "return") };

    var session: Writer.Allocating = .init(gpa);
    defer session.deinit();
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 1), .method = "initialize", .params = .{} });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "initialized", .params = .{} });
    // A signatureHelp BEFORE the document is opened -> null, no crash.
    try frameInto(gpa, &session, sigReq.make(25, u_src, .{ .line = 0, .character = 0 }));
    try frameInto(gpa, &session, openDoc.make(u_src, src));
    try frameInto(gpa, &session, openDoc.make(u_nested, nested));
    try frameInto(gpa, &session, openDoc.make(u_inc, incomplete));
    try frameInto(gpa, &session, sigReq.make(20, u_src, before_comma));
    try frameInto(gpa, &session, sigReq.make(21, u_src, after_comma));
    try frameInto(gpa, &session, sigReq.make(22, u_nested, inner));
    try frameInto(gpa, &session, sigReq.make(23, u_inc, inc_pos));
    try frameInto(gpa, &session, sigReq.make(24, u_src, outside));
    // A character past u32 max must NOT panic the @intCast.
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .id = @as(i64, 26),
        .method = "textDocument/signatureHelp",
        .params = .{ .textDocument = .{ .uri = u_src }, .position = .{ .line = @as(i64, 0), .character = @as(i64, 3000000000) } },
    });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 99), .method = "shutdown" });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "exit" });

    var out: Writer.Allocating = .init(gpa);
    defer out.deinit();
    var reader = Reader.fixed(session.written());
    const code = try serve(gpa, &reader, &out.writer);
    try testing.expectEqual(ExitCode.ok, code); // every request answered; no hang/crash

    var frames = try splitFrames(gpa, out.written());
    defer {
        for (frames.items) |b| gpa.free(b);
        frames.deinit(gpa);
    }

    var parse_arena = std.heap.ArenaAllocator.init(gpa);
    defer parse_arena.deinit();
    const pa = parse_arena.allocator();

    var by_id: [100]?std.json.Value = @splat(null);
    var raw_by_id: [100]?[]const u8 = @splat(null);
    for (frames.items) |body| {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, pa, body, .{});
        if (objGet(v, "method") != null) continue; // a notification
        const idv = objGet(v, "id") orelse continue;
        if (idv == .integer and idv.integer >= 0 and idv.integer < 100) {
            by_id[@intCast(idv.integer)] = v;
            raw_by_id[@intCast(idv.integer)] = body;
        }
    }

    // Capability: the `(`/`,` triggers are advertised.
    const caps = field(by_id[1].?, "result", "capabilities").?;
    const shp = objGet(caps, "signatureHelpProvider").?;
    const trigs = objGet(shp, "triggerCharacters").?;
    try testing.expect(trigs == .array and trigs.array.items.len == 2);
    try testing.expectEqualStrings("(", trigs.array.items[0].string);
    try testing.expectEqualStrings(",", trigs.array.items[1].string);

    // Before the first comma -> active param 0; the label is the exact rendered signature.
    try testing.expectEqualStrings("fn add(int, int) -> int", sigLabel(by_id[20].?).?);
    try testing.expectEqual(@as(?i64, 0), sigActive(by_id[20].?));
    // The SECOND param's label span is the second `int` occurrence (proves the offset label).
    try testing.expectEqual([2]i64{ 12, 15 }, sigParam(by_id[20].?, 1).?);

    // After the first comma -> active param 1, same signature.
    try testing.expectEqualStrings("fn add(int, int) -> int", sigLabel(by_id[21].?).?);
    try testing.expectEqual(@as(?i64, 1), sigActive(by_id[21].?));

    // Nested: the cursor in the inner call resolves to the INNERMOST callee `id`.
    try testing.expectEqualStrings("fn id(int) -> int", sigLabel(by_id[22].?).?);
    try testing.expectEqual(@as(?i64, 0), sigActive(by_id[22].?));

    // Incomplete `add(1, ` at EOF: does NOT bail, returns the signature with active param 1.
    try testing.expectEqualStrings("fn add(int, int) -> int", sigLabel(by_id[23].?).?);
    try testing.expectEqual(@as(?i64, 1), sigActive(by_id[23].?));

    // Outside any call, before didOpen, and a char past u32 max all answer explicit null.
    try testing.expect(objGet(by_id[24].?, "result").? == .null); // on the `return` line
    try testing.expect(objGet(by_id[25].?, "result").? == .null); // before didOpen
    try testing.expect(objGet(by_id[26].?, "result").? == .null); // char > u32 max

    // The internal scratch path/module qualifier must never surface in the response bytes.
    try testing.expect(std.mem.indexOf(u8, raw_by_id[20].?, ".toy-lsp") == null);
    try testing.expect(std.mem.indexOf(u8, raw_by_id[20].?, "doc.") == null);
}

test "lsp incremental didChange: in-order ranged deltas + no-range full replace reflected by hover/diagnostics" {
    const gpa = testing.allocator;

    // Every column below is `colOf`-derived, so no offset is hand-counted.
    const T0 = "fn add(a: int) -> int {\n    return a\n}\nfn main() -> int {\n    return add(0)\n}";
    const c1_ins = "fn id(x: int) -> int {\n    return x\n}\n";
    const afterC1 = c1_ins ++ T0;
    const v2doc = "fn id(x: int) -> int {\n    return x\n}\nfn add(a: int) -> int {\n    return a\n}\nfn main() -> int {\n    return id(0)\n}";
    // v3doc is v2doc with line-7 `return id(0)` -> `return id(zzz)` (an undefined name).
    const v3doc = "fn id(x: int) -> int {\n    return x\n}\nfn add(a: int) -> int {\n    return a\n}\nfn main() -> int {\n    return id(zzz)\n}";
    const T2 = "fn f() -> int {\n    return 3\n}\nfn main() -> int {\n    return f()\n}";
    const uri = "file:///inc.toy";

    const Pos = struct { line: u32, character: u32 };
    const Rng = struct { start: Pos, end: Pos };
    const Ranged = struct { range: Rng, text: []const u8 };
    const FullReplace = struct { text: []const u8 };

    const hoverReq = struct {
        fn make(id: i64, line: u32, character: u32) struct {
            jsonrpc: []const u8 = "2.0",
            id: i64,
            method: []const u8 = "textDocument/hover",
            params: struct { textDocument: struct { uri: []const u8 }, position: struct { line: u32, character: u32 } },
        } {
            return .{ .id = id, .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = line, .character = character } } };
        }
    };

    var session: Writer.Allocating = .init(gpa);
    defer session.deinit();
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 1), .method = "initialize", .params = .{} });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "initialized", .params = .{} });
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .method = "textDocument/didOpen",
        .params = .{ .textDocument = .{ .uri = uri, .languageId = "toy", .version = @as(i64, 1), .text = T0 } },
    });
    // v2: TWO in-order ranged deltas. c1 inserts a new fn at (0,0); c2 then renames the call
    // `add` -> `id` on line 7 — a coordinate that only exists AFTER c1 was applied.
    const add_col = colOf(afterC1, 7, "add");
    const c1: Ranged = .{ .range = .{ .start = .{ .line = 0, .character = 0 }, .end = .{ .line = 0, .character = 0 } }, .text = c1_ins };
    const c2: Ranged = .{ .range = .{ .start = .{ .line = 7, .character = add_col }, .end = .{ .line = 7, .character = add_col + 3 } }, .text = "id" };
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .method = "textDocument/didChange",
        .params = .{ .textDocument = .{ .uri = uri, .version = @as(i64, 2) }, .contentChanges = &[_]Ranged{ c1, c2 } },
    });
    try frameInto(gpa, &session, hoverReq.make(30, 7, colOf(v2doc, 7, "id(")));
    // v3: ONE ranged delta, `0` -> `zzz`, coordinates against the post-v2 text.
    const zero_col = colOf(v2doc, 7, "0");
    const c3: Ranged = .{ .range = .{ .start = .{ .line = 7, .character = zero_col }, .end = .{ .line = 7, .character = zero_col + 1 } }, .text = "zzz" };
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .method = "textDocument/didChange",
        .params = .{ .textDocument = .{ .uri = uri, .version = @as(i64, 3) }, .contentChanges = &[_]Ranged{c3} },
    });
    // v4: a no-range change is a whole-document replace.
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .method = "textDocument/didChange",
        .params = .{ .textDocument = .{ .uri = uri, .version = @as(i64, 4) }, .contentChanges = &[_]FullReplace{.{ .text = T2 }} },
    });
    try frameInto(gpa, &session, hoverReq.make(31, 4, colOf(T2, 4, "f(")));
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 99), .method = "shutdown" });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "exit" });

    var out: Writer.Allocating = .init(gpa);
    defer out.deinit();
    var reader = Reader.fixed(session.written());
    const code = try serve(gpa, &reader, &out.writer);
    try testing.expectEqual(ExitCode.ok, code);

    var frames = try splitFrames(gpa, out.written());
    defer {
        for (frames.items) |b| gpa.free(b);
        frames.deinit(gpa);
    }

    var parse_arena = std.heap.ArenaAllocator.init(gpa);
    defer parse_arena.deinit();
    const pa = parse_arena.allocator();

    var by_id: [40]?std.json.Value = @splat(null);
    var publishes: [8]std.json.Value = undefined;
    var publish_count: usize = 0;
    for (frames.items) |body| {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, pa, body, .{});
        if (objGet(v, "method")) |m| {
            if (m == .string and eql(m.string, "textDocument/publishDiagnostics")) {
                publishes[publish_count] = objGet(v, "params").?;
                publish_count += 1;
            }
            continue;
        }
        const idv = objGet(v, "id") orelse continue;
        if (idv == .integer and idv.integer >= 0 and idv.integer < 40) by_id[@intCast(idv.integer)] = v;
    }

    // Capability: incremental sync advertised.
    const caps = field(by_id[1].?, "result", "capabilities").?;
    const sync = objGet(caps, "textDocumentSync").?;
    try testing.expectEqual(@as(i64, 2), getInt(sync, "change").?);

    // didOpen + three didChange = four publishes, none dropped/added.
    try testing.expectEqual(@as(usize, 4), publish_count);
    // T0 is clean.
    try testing.expectEqual(@as(usize, 0), objGet(publishes[0], "diagnostics").?.array.items.len);

    // In-order multi-edit: a hover on the renamed callee resolves to `id`. Fails under
    // last-wins (whole doc would become `"id"`) or if c2's line-7 coord hit pre-c1 text.
    try testing.expectEqualStrings("fn id(x: int) -> int", hoverValue(by_id[30].?).?);

    // Ranged single delta, offset-exact: the `zzz` edit yields a diagnostic pinned to BOTH the
    // line AND the exact column of `zzz` (a no-op/wrong-line/off-by-N splice all fail this).
    const d2 = objGet(publishes[2], "diagnostics").?;
    const zzz_col: i64 = colOf(v3doc, 7, "zzz");
    var found_zzz = false;
    for (d2.array.items) |d| {
        const start = field(d, "range", "start") orelse continue;
        if (getInt(start, "line") == @as(i64, 7) and getInt(start, "character") == zzz_col) found_zzz = true;
    }
    try testing.expect(found_zzz);

    // Full replace (not append): T2 is clean AND a hover on `f` resolves. Under append, the
    // (4, colOf(T2,4,"f(")) coordinate would land in the retained old prefix, not `f`.
    try testing.expectEqual(@as(usize, 0), objGet(publishes[3], "diagnostics").?.array.items.len);
    try testing.expectEqualStrings("fn f() -> int", hoverValue(by_id[31].?).?);
}

/// The response frame (parsed) whose top-level `id` is the string `want`, plus a count of
/// how many response frames carried it — for the double-response guard in the cancel test.
const StrIdHit = struct { resp: ?std.json.Value = null, count: usize = 0 };

test "lsp cancelRequest: a cancelled request answers -32800 with no result; a normal one still works" {
    const gpa = testing.allocator;

    const src = "fn main() -> int {\n    x := 1\n    return x\n}"; // line 1 == "    x := 1"
    const uri = "file:///cancel.toy";

    const x_col = colOf(src, 1, "x");

    var session: Writer.Allocating = .init(gpa);
    defer session.deinit();
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 1), .method = "initialize", .params = .{} });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "initialized", .params = .{} });
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .method = "textDocument/didOpen",
        .params = .{ .textDocument = .{ .uri = uri, .languageId = "toy", .version = @as(i64, 1), .text = src } },
    });
    // The request to cancel (a STRING id, exercising CancelId.str) ...
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .id = "cancelme",
        .method = "textDocument/hover",
        .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = @as(i64, 1), .character = @as(i64, x_col) } },
    });
    // ... its cancellation (a notification: target id in params, no top-level id) ...
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "$/cancelRequest", .params = .{ .id = "cancelme" } });
    // ... and a NON-cancelled control request (int id).
    try frameInto(gpa, &session, .{
        .jsonrpc = "2.0",
        .id = @as(i64, 6),
        .method = "textDocument/hover",
        .params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = @as(i64, 1), .character = @as(i64, x_col) } },
    });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .id = @as(i64, 7), .method = "shutdown" });
    try frameInto(gpa, &session, .{ .jsonrpc = "2.0", .method = "exit" });

    var out: Writer.Allocating = .init(gpa);
    defer out.deinit();
    var reader = Reader.fixed(session.written());
    const code = try serve(gpa, &reader, &out.writer);
    try testing.expectEqual(ExitCode.ok, code); // the loop did not hang

    var frames = try splitFrames(gpa, out.written());
    defer {
        for (frames.items) |b| gpa.free(b);
        frames.deinit(gpa);
    }

    var parse_arena = std.heap.ArenaAllocator.init(gpa);
    defer parse_arena.deinit();
    const pa = parse_arena.allocator();

    var by_id: [16]?std.json.Value = @splat(null);
    var cancel_hit: StrIdHit = .{};
    var publish_count: usize = 0;
    for (frames.items) |body| {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, pa, body, .{});
        if (objGet(v, "method")) |m| {
            if (m == .string and eql(m.string, "textDocument/publishDiagnostics")) publish_count += 1;
            continue; // a notification is never a response
        }
        const idv = objGet(v, "id") orelse continue;
        if (idv == .string and eql(idv.string, "cancelme")) {
            cancel_hit.count += 1;
            cancel_hit.resp = v;
        } else if (idv == .integer and idv.integer >= 0 and idv.integer < 16) {
            by_id[@intCast(idv.integer)] = v;
        }
    }

    // Exactly ONE response carries the cancelled id (no double-response) ...
    try testing.expectEqual(@as(usize, 1), cancel_hit.count);
    // ... it is RequestCancelled and produced NO normal work-product (no `result` key). This
    // response-level pair is the genuine skipped-work proof: the gate returns before any
    // check / feature call, so hover did zero work.
    const err_obj = objGet(cancel_hit.resp.?, "error").?;
    try testing.expectEqual(@as(i64, -32800), getInt(err_obj, "code").?);
    try testing.expect(objGet(cancel_hit.resp.?, "result") == null);

    // The non-cancelled control request is answered normally in the same session.
    try testing.expectEqualStrings("x: int", hoverValue(by_id[6].?).?);

    // Sanity (explicitly NOT the skip-work proof — hover never publishes): only the didOpen
    // publish was emitted; no stray notification leaked from the cancelled request.
    try testing.expectEqual(@as(usize, 1), publish_count);
}

test "lsp feed: each push is served to completion; the server reports exit" {
    const gpa = testing.allocator;
    var s = Server.init(gpa);
    defer s.deinit();
    s.disk = false;

    var in: Writer.Allocating = .init(gpa);
    defer in.deinit();
    var out: Writer.Allocating = .init(gpa);
    defer out.deinit();

    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .id = 1, .method = "initialize", .params = .{} });
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .method = "textDocument/didOpen", .params = .{
        .textDocument = .{ .uri = "file:///b/a.toy", .version = 1, .text = "fn main() -> int { return y }\n" },
    } });
    try testing.expect(try s.feed(in.written(), &out.writer));
    var frames = try splitFrames(gpa, out.written());
    defer {
        for (frames.items) |f| gpa.free(f);
        frames.deinit(gpa);
    }
    try testing.expectEqual(@as(usize, 2), frames.items.len);
    try testing.expect(std.mem.indexOf(u8, frames.items[1], "R0001") != null);

    in.clearRetainingCapacity();
    out.clearRetainingCapacity();
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .method = "exit" });
    try testing.expect(!try s.feed(in.written(), &out.writer));
}
