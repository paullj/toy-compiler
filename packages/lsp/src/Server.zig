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
const position = @import("position.zig");
const completion = @import("completion.zig");
const definition = @import("definition.zig");
const signature = @import("signature.zig");
const inlay = @import("inlay.zig");
const Analysis = @import("Analysis.zig");
const Documents = @import("Documents.zig");
const Workspace = @import("Workspace.zig");
const Graph = @import("toy_compiler").Graph;
const lsp_uri = @import("uri.zig");

pub const ExitCode = enum(u8) { ok = 0, no_shutdown = 1 };

/// A wasi build runs in a browser worker, where imports resolve only against open documents.
const host_has_disk = @import("builtin").os.tag != .wasi;

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
    /// Backs everything one message allocates (its JSON, a scratch check of a repaired
    /// buffer's answer, the response) and is reset after it, keeping its capacity: the heap
    /// then plateaus at the largest message instead of fragmenting (a wasm heap never
    /// shrinks). Documents and analyses outlive a message, so they use `gpa`.
    arena: std.heap.ArenaAllocator,
    docs: Documents,
    /// The latest analysis of each open document, by uri (the key is the analysis' own
    /// copy). Every feature request on an unchanged document reads it, so an edit costs one
    /// check however many requests follow; a change drops what it makes stale.
    analyses: std.StringHashMapUnmanaged(*Analysis) = .empty,
    /// The bundled std, parsed once for the session: it never changes.
    std_cache: Graph.BundledCache = .{},
    /// How many analyses the session has built.
    checks: u64 = 0,
    /// How the client counts `character`; utf-16 unless it offers utf-8 at `initialize`.
    encoding: position.Encoding = .utf16,
    got_initialize: bool = false,
    shutdown_requested: bool = false,
    /// The import search roots from `initialize`'s `initializationOptions` (see
    /// `configureRoots`); every string is owned.
    root: ?[]const u8 = null,
    include: std.ArrayList([]const u8) = .empty,

    pub fn init(gpa: std.mem.Allocator) Server {
        return .{ .gpa = gpa, .threaded = .init(gpa, .{}), .arena = .init(gpa), .docs = .{} };
    }

    pub fn deinit(self: *Server) void {
        var it = self.analyses.valueIterator();
        while (it.next()) |an| an.*.deinit();
        self.analyses.deinit(self.gpa);
        self.std_cache.deinit(self.gpa);
        if (self.root) |r| self.gpa.free(r);
        for (self.include.items) |r| self.gpa.free(r);
        self.include.deinit(self.gpa);
        self.docs.deinit(self.gpa);
        self.arena.deinit();
        self.threaded.deinit();
        self.* = undefined;
    }

    /// Recomputed from the PINNED server each call: `Threaded.io()` captures
    /// `&self.threaded`, so it must never be cached across the by-value move out of `init`.
    fn workspace(self: *Server) Workspace {
        return .{ .io = self.threaded.io(), .docs = &self.docs, .disk = host_has_disk, .encoding = self.encoding, .std_cache = &self.std_cache, .roots = .{ .root = self.root, .include = self.include.items } };
    }

    /// The analysis of `uri`'s current text, checked now unless an earlier request on this
    /// version already did; null for a document that is not open.
    fn analysisFor(self: *Server, uri: []const u8) !?*Analysis {
        const doc = self.docs.get(uri) orelse return null;
        if (self.analyses.get(uri)) |an| {
            if (an.version == doc.version) return an;
            self.drop(uri);
        }
        const an = try Analysis.init(self.gpa, self.workspace(), uri, doc.text, doc.version);
        errdefer an.deinit();
        try self.analyses.put(self.gpa, an.uri, an);
        self.checks += 1;
        return an;
    }

    /// Take the import search roots from `initializationOptions`: `root` (a string, the
    /// CLI's `--root`) and `include` (an array of strings, the CLI's `-I`s). A relative
    /// path is taken against the workspace folder (`rootUri`), so a project setting can
    /// be written once for every machine. A value of the wrong shape is ignored.
    fn configureRoots(self: *Server, params: std.json.Value) !void {
        const opts = objGet(params, "initializationOptions") orelse return;
        const base_uri = getStr(params, "rootUri");
        const base = if (base_uri) |u| try lsp_uri.uriToPath(self.gpa, u) else null;
        defer if (base) |b| self.gpa.free(b);
        if (getStr(opts, "root")) |r| self.root = try self.absolute(base, r);
        const list = objGet(opts, "include") orelse return;
        if (list != .array) return;
        for (list.array.items) |v| if (v == .string) {
            const abs = try self.absolute(base, v.string);
            errdefer self.gpa.free(abs);
            try self.include.append(self.gpa, abs);
        };
    }

    fn absolute(self: *Server, base: ?[]const u8, path: []const u8) ![]const u8 {
        if (std.fs.path.isAbsolute(path)) return self.gpa.dupe(u8, path);
        const b = base orelse return self.gpa.dupe(u8, path);
        return std.fs.path.join(self.gpa, &.{ b, path });
    }

    fn drop(self: *Server, uri: []const u8) void {
        const kv = self.analyses.fetchRemove(uri) orelse return;
        kv.value.deinit();
    }

    /// Drop every analysis a change to `uri`'s document makes stale: its own, and each one
    /// that read it (see `Analysis.reads`). Called before the change's publish, so even a
    /// client that does not bump the version never reads a stale analysis.
    fn invalidate(self: *Server, uri: []const u8) !void {
        self.drop(uri);
        const path = try Workspace.pathOf(self.gpa, uri);
        defer self.gpa.free(path);
        var stale: std.ArrayList([]const u8) = .empty;
        defer stale.deinit(self.gpa);
        var it = self.analyses.iterator();
        while (it.next()) |e| if (e.value_ptr.*.reads(path)) try stale.append(self.gpa, e.key_ptr.*);
        for (stale.items) |k| self.drop(k);
    }

    /// The analysis and position a `textDocument/<feature>` request names, or null for a
    /// malformed request or one about a document that is not open. The whole u32 range is
    /// guarded, not just `>= 0`: a valid i64 past u32 max would otherwise panic `@intCast`.
    fn positionAt(self: *Server, root: std.json.Value) !?Position {
        const params = objGet(root, "params") orelse return null;
        const uri = getStr(objGet(params, "textDocument") orelse return null, "uri") orelse return null;
        const pos = objGet(params, "position") orelse return null;
        const line = coordU32(getInt(pos, "line")) orelse return null;
        const character = coordU32(getInt(pos, "character")) orelse return null;
        const an = (try self.analysisFor(uri)) orelse return null;
        return .{ .an = an, .line = line, .character = character };
    }

    const Position = struct { an: *Analysis, line: u32, character: u32 };

    /// The analysis and range a `textDocument/inlayHint` request names, or null as for
    /// `positionAt`.
    fn rangeAt(self: *Server, root: std.json.Value) !?DocRange {
        const params = objGet(root, "params") orelse return null;
        const uri = getStr(objGet(params, "textDocument") orelse return null, "uri") orelse return null;
        const range = objGet(params, "range") orelse return null;
        const start = jsonPosition(objGet(range, "start") orelse return null) orelse return null;
        const end = jsonPosition(objGet(range, "end") orelse return null) orelse return null;
        const an = (try self.analysisFor(uri)) orelse return null;
        return .{ .an = an, .range = .{ .start = start, .end = end } };
    }

    const DocRange = struct { an: *Analysis, range: protocol.Range };

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
            self.encoding = negotiateEncoding(root);
            try self.configureRoots(objGet(root, "params") orelse .null);
            if (id) |iv| try protocol.writeResponse(a, writer, iv, protocol.InitializeResult{
                .capabilities = .{ .positionEncoding = self.encoding.name() },
            });
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
            try self.invalidate(uri);
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
                const start_off = (try position.offsetIn(a, d.text, sl, sc, self.encoding)) orelse doc_end;
                var end_off = (try position.offsetIn(a, d.text, el, ec, self.encoding)) orelse doc_end;
                if (end_off < start_off) end_off = start_off; // tolerate an inverted range
                try self.docs.spliceRange(self.gpa, uri, start_off, end_off, text);
            }
            self.docs.setVersion(uri, version);
            try self.invalidate(uri);
            try self.publish(a, writer, uri);
            return true;
        }

        if (eql(method, "textDocument/didClose")) {
            const td = field(root, "params", "textDocument") orelse return true;
            const uri = getStr(td, "uri") orelse return true;
            const version = if (self.docs.get(uri)) |d| d.version else 0;
            self.docs.remove(self.gpa, uri);
            try self.invalidate(uri);
            // Clear any diagnostics the client is still showing for the closed doc.
            try sendDiagnostics(a, writer, uri, version, &.{});
            return true;
        }

        // The position and range requests below allocate their answers from the per-message
        // arena, so nothing needs freeing: the arena reset after dispatch returns does it.
        if (eql(method, protocol.provider_requests.hoverProvider)) {
            const iv = id orelse return true; // a request must carry an id
            const result: ?protocol.Hover = blk: {
                const at = (self.positionAt(root) catch break :blk null) orelse break :blk null;
                const value = (hover.at(a, at.an, at.line, at.character) catch break :blk null) orelse break :blk null;
                break :blk .{ .contents = .{ .kind = "markdown", .value = value } };
            };
            try respond(a, writer, iv, result);
            return true;
        }

        if (eql(method, protocol.provider_requests.completionProvider)) {
            const iv = id orelse return true;
            const items: []const protocol.CompletionItem = blk: {
                const at = (self.positionAt(root) catch break :blk &.{}) orelse break :blk &.{};
                break :blk completion.at(a, self.gpa, self.workspace(), at.an, at.line, at.character) catch &.{};
            };
            // Always an array: an empty list is a valid "no candidates".
            try protocol.writeResponse(a, writer, iv, items);
            return true;
        }

        if (eql(method, protocol.provider_requests.definitionProvider)) {
            const iv = id orelse return true;
            const result: ?protocol.Location = blk: {
                const at = (self.positionAt(root) catch break :blk null) orelse break :blk null;
                break :blk definition.at(a, at.an, at.line, at.character) catch null;
            };
            try respond(a, writer, iv, result);
            return true;
        }

        if (eql(method, protocol.provider_requests.signatureHelpProvider)) {
            const iv = id orelse return true;
            const result: ?protocol.SignatureHelp = blk: {
                const at = (self.positionAt(root) catch break :blk null) orelse break :blk null;
                break :blk signature.at(a, self.gpa, self.workspace(), at.an, at.line, at.character) catch null;
            };
            try respond(a, writer, iv, result);
            return true;
        }

        if (eql(method, protocol.provider_requests.inlayHintProvider)) {
            const iv = id orelse return true;
            const hints: []const protocol.InlayHint = blk: {
                const at = (self.rangeAt(root) catch break :blk &.{}) orelse break :blk &.{};
                break :blk inlay.in(a, at.an, at.range) catch &.{};
            };
            // Always an array: an empty list is a valid "no hints".
            try protocol.writeResponse(a, writer, iv, hints);
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
        const an = (self.analysisFor(uri) catch null) orelse {
            try sendDiagnostics(a, writer, uri, version, &.{});
            return;
        };
        try sendDiagnostics(a, writer, uri, version, an.diagnostics);
    }
};

/// A request's answer, or an explicit JSON `null` result when there is none.
fn respond(a: std.mem.Allocator, writer: *Writer, id: std.json.Value, result: anytype) !void {
    if (result) |r| return protocol.writeResponse(a, writer, id, r);
    try protocol.writeResponse(a, writer, id, std.json.Value{ .null = {} });
}

/// utf-8 when the client lists it in `general.positionEncodings` (our native unit), else
/// utf-16, which the protocol makes every client support.
fn negotiateEncoding(root: std.json.Value) position.Encoding {
    const general = field(objGet(root, "params") orelse return .utf16, "capabilities", "general") orelse return .utf16;
    const offered = objGet(general, "positionEncodings") orelse return .utf16;
    if (offered != .array) return .utf16;
    for (offered.array.items) |e| if (e == .string and eql(e.string, "utf-8")) return .utf8;
    return .utf16;
}

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

/// A `{line, character}` object, each coordinate guarded as `coordU32` guards it.
fn jsonPosition(v: std.json.Value) ?protocol.Position {
    return .{
        .line = coordU32(getInt(v, "line")) orelse return null,
        .character = coordU32(getInt(v, "character")) orelse return null,
    };
}

/// An LSP position coordinate -> u32 guarding the whole u32 range; null/negative/over-max
/// => null, so a malformed position is rejected rather than panicking `@intCast`.
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

/// One whole client session for an end-to-end test: frame the client's messages, serve
/// them in one run through `serve`, then read the server's replies.
const TestSession = struct {
    gpa: std.mem.Allocator,
    in: Writer.Allocating,
    out: Writer.Allocating,
    /// Backs the parsed replies; kept until every assertion is done.
    arena: std.heap.ArenaAllocator,
    frames: std.ArrayList([]u8) = .empty,
    /// Every reply, parsed, in the order the server wrote it.
    replies: std.ArrayList(std.json.Value) = .empty,

    /// The id `finish` gives `shutdown`.
    const shutdown_id: i64 = 9999;

    fn init(gpa: std.mem.Allocator) TestSession {
        return .{ .gpa = gpa, .in = .init(gpa), .out = .init(gpa), .arena = .init(gpa) };
    }

    fn deinit(ts: *TestSession) void {
        for (ts.frames.items) |f| ts.gpa.free(f);
        ts.frames.deinit(ts.gpa);
        ts.arena.deinit();
        ts.in.deinit();
        ts.out.deinit();
    }

    fn send(ts: *TestSession, payload: anytype) !void {
        try frameInto(ts.gpa, &ts.in, payload);
    }

    /// `initialize` (id 1) with `params`, then `initialized`.
    fn initialize(ts: *TestSession, params: anytype) !void {
        try ts.send(.{ .jsonrpc = "2.0", .id = @as(i64, 1), .method = "initialize", .params = params });
        try ts.send(.{ .jsonrpc = "2.0", .method = "initialized", .params = .{} });
    }

    fn open(ts: *TestSession, uri: []const u8, text: []const u8) !void {
        try ts.send(.{
            .jsonrpc = "2.0",
            .method = "textDocument/didOpen",
            .params = .{ .textDocument = .{ .uri = uri, .languageId = "toy", .version = @as(i64, 1), .text = text } },
        });
    }

    /// A whole-document `didChange` to `version`.
    fn change(ts: *TestSession, uri: []const u8, version: i64, text: []const u8) !void {
        const Change = struct { text: []const u8 };
        try ts.send(.{
            .jsonrpc = "2.0",
            .method = "textDocument/didChange",
            .params = .{ .textDocument = .{ .uri = uri, .version = version }, .contentChanges = &[_]Change{.{ .text = text }} },
        });
    }

    fn request(ts: *TestSession, id: anytype, method: []const u8, params: anytype) !void {
        try ts.send(.{ .jsonrpc = "2.0", .id = id, .method = method, .params = params });
    }

    /// The params of a position request.
    fn at(uri: []const u8, line: i64, character: i64) struct {
        textDocument: struct { uri: []const u8 },
        position: struct { line: i64, character: i64 },
    } {
        return .{ .textDocument = .{ .uri = uri }, .position = .{ .line = line, .character = character } };
    }

    /// `shutdown` then `exit`, served: every request was answered (exit code ok) and
    /// `shutdown`'s result is null.
    fn finish(ts: *TestSession) !void {
        try ts.request(shutdown_id, "shutdown", .{});
        try ts.send(.{ .jsonrpc = "2.0", .method = "exit" });
        var reader = Reader.fixed(ts.in.written());
        try testing.expectEqual(ExitCode.ok, try serve(ts.gpa, &reader, &ts.out.writer));
        ts.frames = try splitFrames(ts.gpa, ts.out.written());
        for (ts.frames.items) |f| try ts.replies.append(ts.arena.allocator(), try std.json.parseFromSliceLeaky(std.json.Value, ts.arena.allocator(), f, .{}));
        try testing.expect(objGet(ts.response(shutdown_id).?, "result").? == .null);
    }

    fn index(ts: *const TestSession, id: i64) ?usize {
        for (ts.replies.items, 0..) |v, i| {
            if (objGet(v, "method") != null) continue; // a notification
            const idv = objGet(v, "id") orelse continue;
            if (idv == .integer and idv.integer == id) return i;
        }
        return null;
    }

    /// The response to request `id`.
    fn response(ts: *const TestSession, id: i64) ?std.json.Value {
        return ts.replies.items[ts.index(id) orelse return null];
    }

    /// The response to request `id`, as the server wrote it.
    fn raw(ts: *const TestSession, id: i64) ?[]const u8 {
        return ts.frames.items[ts.index(id) orelse return null];
    }

    fn capabilities(ts: *const TestSession) std.json.Value {
        return field(ts.response(1).?, "result", "capabilities").?;
    }

    /// The params of every `publishDiagnostics`, in order.
    fn publishes(ts: *TestSession) ![]const std.json.Value {
        var out: std.ArrayList(std.json.Value) = .empty;
        for (ts.replies.items) |v| {
            const m = getStr(v, "method") orelse continue;
            if (eql(m, "textDocument/publishDiagnostics")) try out.append(ts.arena.allocator(), objGet(v, "params").?);
        }
        return out.items;
    }
};

test "lsp roots: initializationOptions root and include resolve imports like --root and -I" {
    const gpa = testing.allocator;
    var ts = TestSession.init(gpa);
    defer ts.deinit();
    try ts.initialize(.{ .rootUri = "file:///toy-lsp-roots-ws", .initializationOptions = .{ .root = "/toy-lsp-roots-ws", .include = &[_][]const u8{"lib"} } });
    try ts.open("file:///toy-lsp-roots-ws/lib/extra.toy", "pub fn seven() -> int { return 7 }\n");
    try ts.open("file:///toy-lsp-roots-ws/vendor/more.toy", "pub fn one() -> int { return 1 }\n");
    try ts.open("file:///toy-lsp-roots-ws/src/main.toy", "import extra\nimport vendor/more\nfn main() -> int { return extra.seven() + more.one() }\n");
    try ts.finish();
    const pubs = try ts.publishes();
    const last = pubs[pubs.len - 1];
    try testing.expectEqualStrings("file:///toy-lsp-roots-ws/src/main.toy", getStr(last, "uri").?);
    try testing.expectEqual(@as(usize, 0), objGet(last, "diagnostics").?.array.items.len);
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

    var ts = TestSession.init(gpa);
    defer ts.deinit();
    try ts.initialize(.{ .capabilities = .{ .general = .{ .positionEncodings = &[_][]const u8{"utf-8"} } } });
    try ts.open(uri, arity);
    try ts.change(uri, 2, clean);
    try ts.change(uri, 3, broken);
    try ts.finish();

    const caps = ts.capabilities();
    try testing.expectEqualStrings("utf-8", getStr(caps, "positionEncoding").?);
    const sync = objGet(caps, "textDocumentSync").?;
    try testing.expectEqual(@as(i64, 2), getInt(sync, "change").?);
    // The advertised capability set is exactly `ServerCapabilities`: a provider appears
    // there only with its handler (see `protocol.provider_requests`).
    const declared = @typeInfo(protocol.ServerCapabilities).@"struct".fields;
    try testing.expectEqual(declared.len, caps.object.count());
    inline for (declared) |f| try testing.expect(caps.object.get(f.name) != null);

    const publishes = try ts.publishes();
    try testing.expectEqual(@as(usize, 3), publishes.len); // three publishes, none dropped

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
}

test "lsp: every advertised provider's request is answered" {
    const gpa = testing.allocator;
    const uri = "file:///p.toy";
    var ts = TestSession.init(gpa);
    defer ts.deinit();
    try ts.initialize(.{});
    try ts.open(uri, "fn add(a: int, b: int) -> int { a + b }\nfn main() -> int { return add(1, 2) }\n");
    const fields = @typeInfo(@TypeOf(protocol.provider_requests)).@"struct".fields;
    inline for (fields, 0..) |f, k| {
        const params = .{ .textDocument = .{ .uri = uri }, .position = .{ .line = @as(i64, 1), .character = @as(i64, 30) }, .range = .{ .start = .{ .line = @as(i64, 0), .character = @as(i64, 0) }, .end = .{ .line = @as(i64, 2), .character = @as(i64, 0) } } };
        try ts.request(@as(i64, 10 + k), @field(protocol.provider_requests, f.name), params);
    }
    try ts.request(@as(i64, 50), "textDocument/references", TestSession.at(uri, 1, 30));
    try ts.finish();
    inline for (fields, 0..) |_, k| {
        const r = ts.response(10 + k).?;
        try testing.expect(objGet(r, "error") == null);
        try testing.expect(objGet(r, "result") != null);
    }
    // A request no provider advertises is rejected.
    try testing.expectEqual(@as(i64, protocol.err_code.method_not_found), getInt(objGet(ts.response(50).?, "error").?, "code").?);
}

test "lsp hover: type at a binding/param/expr, signature at a callee, null over a gap and OOB" {
    const gpa = testing.allocator;

    // The leading indent on line 2 is the deliberate whitespace gap (id 6).
    const src = "fn add(a: int, b: int) -> int { return a + b }\nfn main() -> int {\n    x := add(1, 2)\n    return x\n}";
    const uri = "file:///h.toy";
    const hov = "textDocument/hover";
    const at = TestSession.at;

    var ts = TestSession.init(gpa);
    defer ts.deinit();
    try ts.initialize(.{});
    // A hover BEFORE the document is opened -> null, no crash.
    try ts.request(@as(i64, 10), hov, at(uri, 0, 0));
    try ts.open(uri, src);
    try ts.request(@as(i64, 2), hov, at(uri, 2, colOf(src, 2, "x :="))); // binding x
    try ts.request(@as(i64, 3), hov, at(uri, 0, colOf(src, 0, "a + b"))); // param-use a
    try ts.request(@as(i64, 4), hov, at(uri, 2, colOf(src, 2, "add("))); // callee add
    try ts.request(@as(i64, 5), hov, at(uri, 0, colOf(src, 0, "+ b"))); // expr +
    try ts.request(@as(i64, 6), hov, at(uri, 2, colOf(src, 2, "  "))); // leading-indent gap
    // Out-of-range line, and a character past u32 max (must NOT panic the @intCast).
    try ts.request(@as(i64, 7), hov, at(uri, 100000, 0));
    try ts.request(@as(i64, 11), hov, at(uri, 0, 3000000000));
    try ts.finish();

    // hoverProvider is advertised as a boolean.
    const hp = objGet(ts.capabilities(), "hoverProvider").?;
    try testing.expect(hp == .bool and hp.bool);

    // The positive cases FAIL on a null/absent value (non-vacuous).
    try testing.expectEqualStrings("x: int", hoverValue(ts.response(2).?).?); // binding x
    try testing.expectEqualStrings("a: int", hoverValue(ts.response(3).?).?); // param a
    try testing.expectEqualStrings("int", hoverValue(ts.response(5).?).?); // expr +

    // id 4: callee signature.
    const callee = hoverValue(ts.response(4).?).?;
    try testing.expectEqualStrings("fn add(a: int, b: int) -> int", callee);
    // The entry module qualifier must never leak into hover output.
    try testing.expect(std.mem.indexOf(u8, callee, "doc.") == null);

    // Every non-hit path answers with an explicit null result.
    try testing.expect(objGet(ts.response(6).?, "result").? == .null); // whitespace gap
    try testing.expect(objGet(ts.response(10).?, "result").? == .null); // before didOpen
    try testing.expect(objGet(ts.response(7).?, "result").? == .null); // OOB line
    try testing.expect(objGet(ts.response(11).?, "result").? == .null); // char > u32 max
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

const posAfterLast = @import("test_util.zig").posAfterLast;

/// The params of a position request at `p`.
fn atPos(uri: []const u8, p: protocol.Position) @TypeOf(TestSession.at("", 0, 0)) {
    return TestSession.at(uri, p.line, p.character);
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
    const compl = "textDocument/completion";

    var ts = TestSession.init(gpa);
    defer ts.deinit();
    try ts.initialize(.{});
    try ts.open(u_scope, scope_src);
    try ts.open(u_member, member_src);
    try ts.open(u_module, module_src);
    try ts.open(u_broken, broken_src);
    try ts.request(@as(i64, 20), compl, atPos(u_scope, posAfterLast(scope_src, "count")));
    try ts.request(@as(i64, 21), compl, atPos(u_member, posAfterLast(member_src, "p.")));
    try ts.request(@as(i64, 22), compl, atPos(u_module, posAfterLast(module_src, "io.")));
    try ts.request(@as(i64, 23), compl, atPos(u_broken, posAfterLast(broken_src, "s.")));
    try ts.finish();

    // Capability: the `.` trigger is advertised.
    const cp = objGet(ts.capabilities(), "completionProvider").?;
    const trigs = objGet(cp, "triggerCharacters").?;
    try testing.expect(trigs == .array and trigs.array.items.len == 1);
    try testing.expectEqualStrings(".", trigs.array.items[0].string);

    // (1) SCOPE: enclosing-fn local + param + a visible top-level fn + a type + a keyword.
    const scope = resultArr(ts.response(20).?).?;
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
    const member = resultArr(ts.response(21).?).?;
    try testing.expectEqual(@as(?i64, 5), labelKind(member, "x")); // field
    try testing.expectEqual(@as(?i64, 5), labelKind(member, "y")); // field
    try testing.expectEqual(@as(?i64, 2), labelKind(member, "mag")); // method
    try testing.expect(!hasLabel(member, "main"));
    try testing.expect(!hasLabel(member, "Point"));
    try testing.expect(!hasLabel(member, "if"));

    // (3) MODULE: the imported module's pub fns, and no scratch-file qualifier leak.
    const module = resultArr(ts.response(22).?).?;
    try testing.expectEqual(@as(?i64, 3), labelKind(module, "print"));
    try testing.expectEqual(@as(?i64, 3), labelKind(module, "println"));
    try testing.expect(!hasLabel(module, "main"));
    for (module) |it| {
        const l = getStr(it, "label").?;
        try testing.expect(std.mem.indexOfScalar(u8, l, '.') == null);
    }

    // (4) INCOMPLETE (fault-tolerant): a trailing `.` still yields the receiver's fields.
    const broken = resultArr(ts.response(23).?).?;
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
    const def = "textDocument/definition";
    const at = TestSession.at;

    var ts = TestSession.init(gpa);
    defer ts.deinit();
    try ts.initialize(.{});
    // A definition BEFORE the document is opened -> null, no crash.
    try ts.request(@as(i64, 10), def, at(uri, 0, 0));
    try ts.open(uri, src);
    try ts.request(@as(i64, 2), def, at(uri, 5, colOf(src, 5, "s"))); // use of local `s`
    try ts.request(@as(i64, 3), def, at(uri, 0, colOf(src, 0, "a + b"))); // use of param `a`
    try ts.request(@as(i64, 4), def, at(uri, 4, colOf(src, 4, "add("))); // use of fn `add`
    try ts.request(@as(i64, 5), def, at(uri, 3, colOf(src, 3, "Point"))); // use of type `Point`
    try ts.request(@as(i64, 6), def, at(uri, 3, 0)); // leading-indent gap -> null
    try ts.request(@as(i64, 7), def, at(uri, 100000, 0)); // OOB line -> null
    try ts.request(@as(i64, 11), def, at(uri, 0, 3000000000));
    try ts.finish();

    // definitionProvider is advertised as a boolean.
    const dp = objGet(ts.capabilities(), "definitionProvider").?;
    try testing.expect(dp == .bool and dp.bool);

    // The expected DECL ranges — every column derived from the fixture via `colOf`.
    const Case = struct { id: i64, line: i64, start: u32, len: u32 };
    const cases = [_]Case{
        .{ .id = 2, .line = 4, .start = colOf(src, 4, "s :="), .len = 1 }, // local `s`
        .{ .id = 3, .line = 0, .start = colOf(src, 0, "a: int"), .len = 1 }, // param `a`
        .{ .id = 4, .line = 0, .start = colOf(src, 0, "add"), .len = 3 }, // fn `add` (name, not `fn`)
        .{ .id = 5, .line = 1, .start = colOf(src, 1, "Point"), .len = 5 }, // type `Point`
    };
    for (cases) |c| {
        const r = defRange(ts.response(c.id).?).?; // fails on a null result (non-vacuous)
        try testing.expectEqualStrings(uri, getStr(objGet(ts.response(c.id).?, "result").?, "uri").?);
        try testing.expectEqual(c.line, r.start_line);
        try testing.expectEqual(@as(i64, c.start), r.start_char);
        try testing.expectEqual(c.line, r.end_line);
        try testing.expectEqual(@as(i64, c.start + c.len), r.end_char); // start != end -> non-vacuous
        // The internal scratch path must never surface in the response bytes.
        try testing.expect(std.mem.indexOf(u8, ts.raw(c.id).?, ".toy-lsp") == null);
        try testing.expect(std.mem.indexOf(u8, ts.raw(c.id).?, "doc.toy") == null);
    }

    // Every non-hit path answers with an explicit null result.
    try testing.expect(objGet(ts.response(6).?, "result").? == .null); // whitespace gap
    try testing.expect(objGet(ts.response(7).?, "result").? == .null); // OOB line
    try testing.expect(objGet(ts.response(10).?, "result").? == .null); // before didOpen
    try testing.expect(objGet(ts.response(11).?, "result").? == .null); // char > u32 max
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
    const sig = "textDocument/signatureHelp";

    const before_comma = posAfterLast(src, "add(1");
    const after_comma = posAfterLast(src, "add(1, ");
    const inner = posAfterLast(nested, "id(1");
    const inc_pos = posAfterLast(incomplete, "add(1, ");
    const outside: protocol.Position = .{ .line = 4, .character = colOf(src, 4, "return") };

    var ts = TestSession.init(gpa);
    defer ts.deinit();
    try ts.initialize(.{});
    // A signatureHelp BEFORE the document is opened -> null, no crash.
    try ts.request(@as(i64, 25), sig, TestSession.at(u_src, 0, 0));
    try ts.open(u_src, src);
    try ts.open(u_nested, nested);
    try ts.open(u_inc, incomplete);
    try ts.request(@as(i64, 20), sig, atPos(u_src, before_comma));
    try ts.request(@as(i64, 21), sig, atPos(u_src, after_comma));
    try ts.request(@as(i64, 22), sig, atPos(u_nested, inner));
    try ts.request(@as(i64, 23), sig, atPos(u_inc, inc_pos));
    try ts.request(@as(i64, 24), sig, atPos(u_src, outside));
    // A character past u32 max must NOT panic the @intCast.
    try ts.request(@as(i64, 26), sig, TestSession.at(u_src, 0, 3000000000));
    try ts.finish();

    // Capability: the `(`/`,` triggers are advertised.
    const shp = objGet(ts.capabilities(), "signatureHelpProvider").?;
    const trigs = objGet(shp, "triggerCharacters").?;
    try testing.expect(trigs == .array and trigs.array.items.len == 2);
    try testing.expectEqualStrings("(", trigs.array.items[0].string);
    try testing.expectEqualStrings(",", trigs.array.items[1].string);

    // Before the first comma -> active param 0; the label is the exact rendered signature.
    try testing.expectEqualStrings("fn add(int, int) -> int", sigLabel(ts.response(20).?).?);
    try testing.expectEqual(@as(?i64, 0), sigActive(ts.response(20).?));
    // The SECOND param's label span is the second `int` occurrence (proves the offset label).
    try testing.expectEqual([2]i64{ 12, 15 }, sigParam(ts.response(20).?, 1).?);

    // After the first comma -> active param 1, same signature.
    try testing.expectEqualStrings("fn add(int, int) -> int", sigLabel(ts.response(21).?).?);
    try testing.expectEqual(@as(?i64, 1), sigActive(ts.response(21).?));

    // Nested: the cursor in the inner call resolves to the INNERMOST callee `id`.
    try testing.expectEqualStrings("fn id(int) -> int", sigLabel(ts.response(22).?).?);
    try testing.expectEqual(@as(?i64, 0), sigActive(ts.response(22).?));

    // Incomplete `add(1, ` at EOF: does NOT bail, returns the signature with active param 1.
    try testing.expectEqualStrings("fn add(int, int) -> int", sigLabel(ts.response(23).?).?);
    try testing.expectEqual(@as(?i64, 1), sigActive(ts.response(23).?));

    // Outside any call, before didOpen, and a char past u32 max all answer explicit null.
    try testing.expect(objGet(ts.response(24).?, "result").? == .null); // on the `return` line
    try testing.expect(objGet(ts.response(25).?, "result").? == .null); // before didOpen
    try testing.expect(objGet(ts.response(26).?, "result").? == .null); // char > u32 max

    // The internal scratch path/module qualifier must never surface in the response bytes.
    try testing.expect(std.mem.indexOf(u8, ts.raw(20).?, ".toy-lsp") == null);
    try testing.expect(std.mem.indexOf(u8, ts.raw(20).?, "doc.") == null);
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

    var ts = TestSession.init(gpa);
    defer ts.deinit();
    try ts.initialize(.{});
    try ts.open(uri, T0);
    // v2: TWO in-order ranged deltas. c1 inserts a new fn at (0,0); c2 then renames the call
    // `add` -> `id` on line 7 — a coordinate that only exists AFTER c1 was applied.
    const add_col = colOf(afterC1, 7, "add");
    const c1: Ranged = .{ .range = .{ .start = .{ .line = 0, .character = 0 }, .end = .{ .line = 0, .character = 0 } }, .text = c1_ins };
    const c2: Ranged = .{ .range = .{ .start = .{ .line = 7, .character = add_col }, .end = .{ .line = 7, .character = add_col + 3 } }, .text = "id" };
    try ts.send(.{
        .jsonrpc = "2.0",
        .method = "textDocument/didChange",
        .params = .{ .textDocument = .{ .uri = uri, .version = @as(i64, 2) }, .contentChanges = &[_]Ranged{ c1, c2 } },
    });
    try ts.request(@as(i64, 30), "textDocument/hover", TestSession.at(uri, 7, colOf(v2doc, 7, "id(")));
    // v3: ONE ranged delta, `0` -> `zzz`, coordinates against the post-v2 text.
    const zero_col = colOf(v2doc, 7, "0");
    const c3: Ranged = .{ .range = .{ .start = .{ .line = 7, .character = zero_col }, .end = .{ .line = 7, .character = zero_col + 1 } }, .text = "zzz" };
    try ts.send(.{
        .jsonrpc = "2.0",
        .method = "textDocument/didChange",
        .params = .{ .textDocument = .{ .uri = uri, .version = @as(i64, 3) }, .contentChanges = &[_]Ranged{c3} },
    });
    // v4: a no-range change is a whole-document replace.
    try ts.change(uri, 4, T2);
    try ts.request(@as(i64, 31), "textDocument/hover", TestSession.at(uri, 4, colOf(T2, 4, "f(")));
    try ts.finish();

    // Capability: incremental sync advertised.
    const sync = objGet(ts.capabilities(), "textDocumentSync").?;
    try testing.expectEqual(@as(i64, 2), getInt(sync, "change").?);

    // didOpen + three didChange = four publishes, none dropped/added.
    const publishes = try ts.publishes();
    try testing.expectEqual(@as(usize, 4), publishes.len);
    // T0 is clean.
    try testing.expectEqual(@as(usize, 0), objGet(publishes[0], "diagnostics").?.array.items.len);

    // In-order multi-edit: a hover on the renamed callee resolves to `id`. Fails under
    // last-wins (whole doc would become `"id"`) or if c2's line-7 coord hit pre-c1 text.
    try testing.expectEqualStrings("fn id(x: int) -> int", hoverValue(ts.response(30).?).?);

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
    try testing.expectEqualStrings("fn f() -> int", hoverValue(ts.response(31).?).?);
}

test "lsp cancelRequest: a cancelled request answers -32800 with no result; a normal one still works" {
    const gpa = testing.allocator;

    const src = "fn main() -> int {\n    x := 1\n    return x\n}"; // line 1 == "    x := 1"
    const uri = "file:///cancel.toy";

    const x_col = colOf(src, 1, "x");

    var ts = TestSession.init(gpa);
    defer ts.deinit();
    try ts.initialize(.{});
    try ts.open(uri, src);
    // The request to cancel (a STRING id, exercising CancelId.str) ...
    try ts.request("cancelme", "textDocument/hover", TestSession.at(uri, 1, x_col));
    // ... its cancellation (a notification: target id in params, no top-level id) ...
    try ts.send(.{ .jsonrpc = "2.0", .method = "$/cancelRequest", .params = .{ .id = "cancelme" } });
    // ... and a NON-cancelled control request (int id).
    try ts.request(@as(i64, 6), "textDocument/hover", TestSession.at(uri, 1, x_col));
    try ts.finish();

    var cancel_hits: usize = 0;
    var cancel_resp: ?std.json.Value = null;
    for (ts.replies.items) |v| {
        if (objGet(v, "method") != null) continue; // a notification is never a response
        const idv = objGet(v, "id") orelse continue;
        if (idv == .string and eql(idv.string, "cancelme")) {
            cancel_hits += 1;
            cancel_resp = v;
        }
    }

    // Exactly ONE response carries the cancelled id (no double-response) ...
    try testing.expectEqual(@as(usize, 1), cancel_hits);
    // ... it is RequestCancelled and produced NO normal work-product (no `result` key). This
    // response-level pair is the genuine skipped-work proof: the gate returns before any
    // check / feature call, so hover did zero work.
    const err_obj = objGet(cancel_resp.?, "error").?;
    try testing.expectEqual(@as(i64, -32800), getInt(err_obj, "code").?);
    try testing.expect(objGet(cancel_resp.?, "result") == null);

    // The non-cancelled control request is answered normally in the same session.
    try testing.expectEqualStrings("x: int", hoverValue(ts.response(6).?).?);

    // Sanity (explicitly NOT the skip-work proof — hover never publishes): only the didOpen
    // publish was emitted; no stray notification leaked from the cancelled request.
    try testing.expectEqual(@as(usize, 1), (try ts.publishes()).len);
}

test "lsp feed: each push is served to completion; the server reports exit" {
    const gpa = testing.allocator;
    var s = Server.init(gpa);
    defer s.deinit();

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

test "lsp positions: utf-16 unless the client offers utf-8, both directions" {
    const gpa = testing.allocator;
    // `日本` is 6 bytes but 2 UTF-16 units, so `zz` sits at byte column 30 but unit 26.
    const src = "fn main() -> bool {\n    return \"\u{65e5}\u{672c}\" == \"x\" || zz\n}\n";
    for ([_]struct { offer: ?[]const u8, want_char: i64 }{
        .{ .offer = null, .want_char = 26 },
        .{ .offer = "utf-8", .want_char = 30 },
    }) |c| {
        var s = Server.init(gpa);
        defer s.deinit();
        var in: Writer.Allocating = .init(gpa);
        defer in.deinit();
        var out: Writer.Allocating = .init(gpa);
        defer out.deinit();
        if (c.offer) |o| {
            try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .id = 1, .method = "initialize", .params = .{ .capabilities = .{ .general = .{ .positionEncodings = &[_][]const u8{o} } } } });
        } else {
            try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .id = 1, .method = "initialize", .params = .{} });
        }
        try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .method = "textDocument/didOpen", .params = .{
            .textDocument = .{ .uri = "file:///b/u.toy", .version = 1, .text = src },
        } });
        // A ranged edit in the client's units: rename `zz` to `yy`.
        try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .method = "textDocument/didChange", .params = .{
            .textDocument = .{ .uri = "file:///b/u.toy", .version = 2 },
            .contentChanges = &[_]struct { range: protocol.Range, text: []const u8 }{.{
                .range = .{ .start = .{ .line = 1, .character = @intCast(c.want_char) }, .end = .{ .line = 1, .character = @intCast(c.want_char + 2) } },
                .text = "yy",
            }},
        } });
        _ = try s.feed(in.written(), &out.writer);
        try testing.expect(std.mem.indexOf(u8, s.docs.textOf("file:///b/u.toy").?, "|| yy") != null);

        var frames = try splitFrames(gpa, out.written());
        defer {
            for (frames.items) |f| gpa.free(f);
            frames.deinit(gpa);
        }
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const last = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), frames.items[frames.items.len - 1], .{});
        const d = objGet(last, "params").?.object.get("diagnostics").?.array.items[0];
        try testing.expectEqual(c.want_char, getInt(field(d, "range", "start").?, "character").?);
    }
}

test "lsp inlayHint: advertised, answered in range over stdio and through feed alike, [] before didOpen" {
    const gpa = testing.allocator;
    const src =
        \\fn inc(mut n: int) { n = n + 1 }
        \\fn run(f: fn()) { f() }
        \\fn main() -> int {
        \\    c := 1
        \\    run(fn() { inc(c) })
        \\    inc(c)
        \\    return c
        \\}
        \\
    ;
    const uri = "file:///inlay.toy";
    const range = struct {
        fn of(u: []const u8, first: i64, last: i64) struct {
            textDocument: struct { uri: []const u8 },
            range: struct { start: struct { line: i64, character: i64 }, end: struct { line: i64, character: i64 } },
        } {
            return .{ .textDocument = .{ .uri = u }, .range = .{ .start = .{ .line = first, .character = 0 }, .end = .{ .line = last, .character = 0 } } };
        }
    }.of;
    const ih = "textDocument/inlayHint";

    var ts = TestSession.init(gpa);
    defer ts.deinit();
    try ts.initialize(.{});
    try ts.request(@as(i64, 2), ih, range(uri, 0, 9));
    try ts.open(uri, src);
    try ts.request(@as(i64, 3), ih, range(uri, 0, 9));
    try ts.request(@as(i64, 4), ih, range(uri, 5, 6));
    // A negative coordinate is malformed: no hints, no crash.
    try ts.request(@as(i64, 5), ih, range(uri, -1, 9));
    try ts.finish();

    // The wasm build drives the same core through `feed`: the same bytes in, the same out.
    var fed: Writer.Allocating = .init(gpa);
    defer fed.deinit();
    var s = Server.init(gpa);
    defer s.deinit();
    try testing.expect(!try s.feed(ts.in.written(), &fed.writer));
    try testing.expectEqualStrings(ts.out.written(), fed.written());

    const ihp = objGet(ts.capabilities(), "inlayHintProvider").?;
    try testing.expect(ihp == .bool and ihp.bool);

    try testing.expectEqual(@as(usize, 0), resultArr(ts.response(2).?).?.len); // before didOpen
    try testing.expectEqual(@as(usize, 0), resultArr(ts.response(5).?).?.len); // malformed range

    const all = resultArr(ts.response(3).?).?;
    try testing.expectEqual(@as(usize, 3), all.len);
    const Want = struct { line: i64, character: i64, label: []const u8, tooltip: []const u8 };
    const want = [_]Want{
        .{ .line = 4, .character = colOf(src, 4, "{ inc"), .label = "captures c (shared)", .tooltip = "c: shared" },
        .{ .line = 4, .character = colOf(src, 4, "c) })"), .label = "mut", .tooltip = "mutated by inc" },
        .{ .line = 5, .character = colOf(src, 5, "c)"), .label = "mut", .tooltip = "mutated by inc" },
    };
    for (want, all) |w, h| {
        try testing.expectEqual(w.line, getInt(objGet(h, "position").?, "line").?);
        try testing.expectEqual(w.character, getInt(objGet(h, "position").?, "character").?);
        try testing.expectEqualStrings(w.label, getStr(h, "label").?);
        try testing.expectEqualStrings(w.tooltip, getStr(h, "tooltip").?);
    }
    try testing.expectEqual(@as(?i64, protocol.inlay_kind.parameter), getInt(all[1], "kind"));
    try testing.expect(objGet(all[0], "kind") == null);

    // Lines 5..6 hold only the second `mut` argument.
    const ranged = resultArr(ts.response(4).?).?;
    try testing.expectEqual(@as(usize, 1), ranged.len);
    try testing.expectEqual(@as(i64, 5), getInt(objGet(ranged[0], "position").?, "line").?);
}

test "lsp: one check per document version, however many requests read it; a change to an import re-checks" {
    const gpa = testing.allocator;
    var s = Server.init(gpa);
    defer s.deinit();
    var in: Writer.Allocating = .init(gpa);
    defer in.deinit();
    var out: Writer.Allocating = .init(gpa);
    defer out.deinit();

    const main_uri = "file:///w/main.toy";
    const helper_uri = "file:///w/helper.toy";
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .id = 1, .method = "initialize", .params = .{} });
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .method = "textDocument/didOpen", .params = .{
        .textDocument = .{ .uri = helper_uri, .version = 1, .text = "pub fn seven() -> int { return 7 }\n" },
    } });
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .method = "textDocument/didOpen", .params = .{
        .textDocument = .{ .uri = main_uri, .version = 1, .text = "import helper\nfn main() -> int { return helper.seven() }\n" },
    } });
    _ = try s.feed(in.written(), &out.writer);
    try testing.expectEqual(@as(u64, 2), s.checks);

    // Requests on the same version read the analysis the publish built.
    in.clearRetainingCapacity();
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .id = 2, .method = "textDocument/hover", .params = TestSession.at(main_uri, 1, 34) });
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .id = 3, .method = "textDocument/inlayHint", .params = .{
        .textDocument = .{ .uri = main_uri },
        .range = .{ .start = .{ .line = 0, .character = 0 }, .end = .{ .line = 2, .character = 0 } },
    } });
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .id = 4, .method = "textDocument/definition", .params = TestSession.at(main_uri, 1, 34) });
    _ = try s.feed(in.written(), &out.writer);
    try testing.expectEqual(@as(u64, 2), s.checks);
    try testing.expect(std.mem.indexOf(u8, out.written(), "fn seven() -> int") != null);

    // An edit to the import drops the importer's analysis; its next request re-checks.
    in.clearRetainingCapacity();
    out.clearRetainingCapacity();
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .method = "textDocument/didChange", .params = .{
        .textDocument = .{ .uri = helper_uri, .version = 2 },
        .contentChanges = &[_]struct { text: []const u8 }{.{ .text = "pub fn seven() -> str { return \"7\" }\n" }},
    } });
    _ = try s.feed(in.written(), &out.writer);
    try testing.expectEqual(@as(u64, 3), s.checks);
    try testing.expect(s.analyses.get(main_uri) == null);
    in.clearRetainingCapacity();
    out.clearRetainingCapacity();
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .id = 5, .method = "textDocument/hover", .params = TestSession.at(main_uri, 1, 34) });
    _ = try s.feed(in.written(), &out.writer);
    try testing.expectEqual(@as(u64, 4), s.checks);
    try testing.expect(std.mem.indexOf(u8, out.written(), "fn seven() -> str") != null);

    // Closing a document drops its analysis.
    in.clearRetainingCapacity();
    try frameInto(gpa, &in, .{ .jsonrpc = "2.0", .method = "textDocument/didClose", .params = .{ .textDocument = .{ .uri = main_uri } } });
    _ = try s.feed(in.written(), &out.writer);
    try testing.expect(s.analyses.get(main_uri) == null);
}

test "lsp: each module of a multi-module program publishes its own diagnostics under its own uri" {
    const gpa = testing.allocator;
    var ts = TestSession.init(gpa);
    defer ts.deinit();

    // The import sits in a subdirectory of the entry's.
    const main_uri = "file:///w/main.toy";
    const helper_uri = "file:///w/util/helper.toy";
    const helper_src = "fn add(a: int, b: int) -> int { return a + b }\npub fn get() -> int {\n    add(1)\n}\n";
    const main_src = "import util/helper\n\nfn main() -> int {\n    z: int = \"b\"\n    helper.get() + z\n}\n";
    try ts.initialize(.{});
    try ts.open(helper_uri, helper_src);
    try ts.open(main_uri, main_src);
    try ts.finish();

    const publishes = try ts.publishes();
    try testing.expectEqual(@as(usize, 2), publishes.len);

    const helper_pub = publishes[0];
    try testing.expectEqualStrings(helper_uri, getStr(helper_pub, "uri").?);
    const hd = objGet(helper_pub, "diagnostics").?.array.items;
    try testing.expectEqual(@as(usize, 1), hd.len);
    try testing.expectEqualStrings("T0039", getStr(hd[0], "code").?);
    try testing.expectEqual(@as(i64, 2), field(hd[0], "range", "start").?.object.get("line").?.integer);
    const rel = objGet(hd[0], "relatedInformation").?.array.items[0];
    try testing.expectEqualStrings(helper_uri, getStr(objGet(rel, "location").?, "uri").?);
    try testing.expectEqual(@as(i64, 0), field(objGet(rel, "location").?, "range", "start").?.object.get("line").?.integer);

    // The importer's publish carries only its own error: the helper's belongs to the helper.
    const main_pub = publishes[1];
    try testing.expectEqualStrings(main_uri, getStr(main_pub, "uri").?);
    const md = objGet(main_pub, "diagnostics").?.array.items;
    try testing.expectEqual(@as(usize, 1), md.len);
    try testing.expectEqualStrings("T0041", getStr(md[0], "code").?);
    try testing.expectEqual(@as(i64, 3), field(md[0], "range", "start").?.object.get("line").?.integer);
}
