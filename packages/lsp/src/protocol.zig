//! LSP / JSON-RPC wire types and envelope writers. Everything is serialized with
//! `std.json` (no external LSP dependency, per house style). `emit_null_optional_fields
//! = false` on every serialization means a null optional (an absent diagnostic `code`,
//! no `relatedInformation`) is OMITTED rather than emitted as `null`.

const std = @import("std");
const Writer = std.Io.Writer;
const transport = @import("transport.zig");
const toyc = @import("toy_compiler");

/// JSON-RPC / LSP error codes we can produce.
pub const err_code = struct {
    pub const parse: i32 = -32700;
    pub const invalid_request: i32 = -32600;
    pub const method_not_found: i32 = -32601;
    pub const invalid_params: i32 = -32602;
    pub const internal: i32 = -32603;
    /// LSP: a request arrived before `initialize`.
    pub const server_not_initialized: i32 = -32002;
};

/// LSP `DiagnosticSeverity`.
pub const severity = struct {
    pub const err: u8 = 1;
    pub const warning: u8 = 2;
    pub const information: u8 = 3;
    pub const hint: u8 = 4;
};

/// A 0-based position. `character` is a UTF-8 byte offset within the line (we advertise
/// `positionEncoding: "utf-8"`, so for a client that honors it this is exact; for ASCII
/// source — the overwhelming majority of toy code — it equals the UTF-16 column too).
pub const Position = struct { line: u32, character: u32 };
pub const Range = struct { start: Position, end: Position };
pub const Location = struct { uri: []const u8, range: Range };
pub const Related = struct { location: Location, message: []const u8 };

pub const LspDiagnostic = struct {
    range: Range,
    severity: u8,
    /// The stable code string ("T0039"); omitted for an uncoded diagnostic.
    code: ?[]const u8 = null,
    source: []const u8 = "toy",
    message: []const u8,
    relatedInformation: ?[]const Related = null,
};

/// The subset of `ServerCapabilities` we actually implement — advertised as-is. No
/// hover/completion/definition/etc.: advertising a provider we do not implement would
/// make the client send requests we can only reject.
const ServerCapabilities = struct {
    /// utf-8 so a byte column equals `character` for ASCII (see `Position`).
    positionEncoding: []const u8 = "utf-8",
    /// `1` == Full-text sync; `openClose` so the client sends didOpen/didClose.
    textDocumentSync: struct {
        openClose: bool = true,
        change: u8 = 1,
    } = .{},
};

const ServerInfo = struct { name: []const u8, version: []const u8 };

pub const InitializeResult = struct {
    capabilities: ServerCapabilities = .{},
    serverInfo: ServerInfo = .{ .name = "toy-lsp", .version = toyc.version.semver },
};

const json_opts: std.json.Stringify.Options = .{ .emit_null_optional_fields = false };

/// Serialize `payload` to a JSON body then frame + flush it over `writer`. The body is
/// staged in a `Writer.Allocating` so its final length is known for the `Content-Length`.
fn send(gpa: std.mem.Allocator, writer: *Writer, payload: anytype) !void {
    var aw: Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try std.json.Stringify.value(payload, json_opts, &aw.writer);
    try transport.writeMessage(writer, aw.written());
}

/// A successful response. `id` is echoed verbatim as the raw request id (int or string);
/// `result` is any serializable value (pass `std.json.Value{ .null = {} }` for a `null`
/// result — shutdown — so the key is PRESENT as JSON `null`, not omitted).
pub fn writeResponse(gpa: std.mem.Allocator, writer: *Writer, id: std.json.Value, result: anytype) !void {
    try send(gpa, writer, .{ .jsonrpc = "2.0", .id = id, .result = result });
}

/// An error response. `id` is `null` (JSON `null`) when the offending message had no id.
pub fn writeError(gpa: std.mem.Allocator, writer: *Writer, id: ?std.json.Value, code: i32, message: []const u8) !void {
    const id_val = id orelse std.json.Value{ .null = {} };
    try send(gpa, writer, .{
        .jsonrpc = "2.0",
        .id = id_val,
        .@"error" = .{ .code = code, .message = message },
    });
}

/// A server-to-client notification (no id).
pub fn writeNotification(gpa: std.mem.Allocator, writer: *Writer, method: []const u8, params: anytype) !void {
    try send(gpa, writer, .{ .jsonrpc = "2.0", .method = method, .params = params });
}

const testing = std.testing;

test "InitializeResult advertises only the implemented capabilities" {
    var aw: Writer.Allocating = .init(testing.allocator);
    defer aw.deinit();
    const r: InitializeResult = .{};
    try std.json.Stringify.value(r, json_opts, &aw.writer);
    const out = aw.written();
    try testing.expect(std.mem.indexOf(u8, out, "\"positionEncoding\":\"utf-8\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"change\":1") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"openClose\":true") != null);
    // Nothing we do not implement leaks into the advertisement.
    try testing.expect(std.mem.indexOf(u8, out, "hoverProvider") == null);
    try testing.expect(std.mem.indexOf(u8, out, "completionProvider") == null);
    try testing.expect(std.mem.indexOf(u8, out, "definitionProvider") == null);
}

test "writeResponse echoes an integer id and omits null optionals in the result" {
    var aw: Writer.Allocating = .init(testing.allocator);
    defer aw.deinit();
    const diag: LspDiagnostic = .{ .range = .{ .start = .{ .line = 0, .character = 0 }, .end = .{ .line = 0, .character = 0 } }, .severity = severity.err, .message = "x" };
    try writeResponse(testing.allocator, &aw.writer, .{ .integer = 7 }, diag);
    const out = aw.written();
    try testing.expect(std.mem.indexOf(u8, out, "\"id\":7") != null);
    // `code`/`relatedInformation` were null -> omitted.
    try testing.expect(std.mem.indexOf(u8, out, "\"code\"") == null);
    try testing.expect(std.mem.indexOf(u8, out, "relatedInformation") == null);
}

test "writeResponse emits an explicit null result for shutdown" {
    var aw: Writer.Allocating = .init(testing.allocator);
    defer aw.deinit();
    try writeResponse(testing.allocator, &aw.writer, .{ .integer = 2 }, std.json.Value{ .null = {} });
    try testing.expect(std.mem.indexOf(u8, aw.written(), "\"result\":null") != null);
}
