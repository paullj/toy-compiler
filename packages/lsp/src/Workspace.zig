//! What a feature request checks against: the open-document store layered over (optionally)
//! the disk. An open buffer is usually UNSAVED, so the front-end reads it — and every other
//! open document an import names — from memory through a `Graph.Overlay`; only a module no
//! editor has open falls through to disk. Nothing is written, and the query cache is
//! disabled: per-keystroke content would grow a persistent cache without bound.

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");
const Documents = @import("Documents.zig");
const lsp_uri = @import("uri.zig");

const Graph = toyc.Graph;
const Cache = toyc.Cache;

const Workspace = @This();

io: Io,
docs: *const Documents,
/// false on a host with no filesystem (the browser): an import resolves only against open
/// documents.
disk: bool = true,

/// Discover the module graph rooted at `uri`, whose current text is `text` (which may differ
/// from the stored document: completion and signature help check a repaired copy). A
/// non-`file://` uri has no directory, so it can import only bundled modules.
pub fn discover(ws: Workspace, gpa: std.mem.Allocator, uri: []const u8, text: []const u8) !Graph.Graph {
    const path = (try lsp_uri.uriToPath(gpa, uri)) orelse try gpa.dupe(u8, untitled_path);
    defer gpa.free(path);
    const root = std.fs.path.dirname(path) orelse "";

    const layer: Layer = .{ .entry_path = path, .entry_text = text, .docs = ws.docs };
    const overlay: Graph.Overlay = .{ .ctx = &layer, .getFn = Layer.get, .disk = ws.disk };
    return Graph.discoverWith(gpa, ws.io, Cache.disabled, "native", path, null, root, overlay);
}

const untitled_path = "untitled.toy";
const file_scheme = "file://";

const Layer = struct {
    entry_path: []const u8,
    entry_text: []const u8,
    docs: *const Documents,

    fn get(ctx: *const anyopaque, path: []const u8) ?[]const u8 {
        const l: *const Layer = @ptrCast(@alignCast(ctx));
        if (std.mem.eql(u8, path, l.entry_path)) return l.entry_text;
        // Open documents are keyed by uri (`uri.pathToUri`'s spelling, without allocating).
        var buf: [file_scheme.len + std.fs.max_path_bytes]u8 = undefined;
        const uri = std.fmt.bufPrint(&buf, file_scheme ++ "{s}", .{path}) catch return null;
        return l.docs.textOf(uri);
    }
};

const testing = std.testing;

test "an import resolves to another OPEN document's unsaved text, with no disk access" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    try docs.put(gpa, "file:///ws/helper.toy", "pub fn seven() -> int { return 7 }\n", 1);

    const ws: Workspace = .{ .io = Io.failing, .docs = &docs, .disk = false };
    var g = try ws.discover(gpa, "file:///ws/main.toy", "import helper\nfn main() -> int { return helper.seven() }\n");
    defer g.deinit(gpa);
    try testing.expect(g.err == null);
    try testing.expectEqual(@as(usize, 2), g.modules.len);
}

test "with the disk off, an import no document provides is a missing module, not an I/O error" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);

    const ws: Workspace = .{ .io = Io.failing, .docs = &docs, .disk = false };
    var g = try ws.discover(gpa, "file:///ws/main.toy", "import nowhere\nfn main() -> int { return 0 }\n");
    defer g.deinit(gpa);
    try testing.expect(g.err != null);
    try testing.expectEqual(Graph.Error.Kind.missing, g.err.?.kind);
}

test "an untitled buffer still checks and can import bundled std" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);

    const ws: Workspace = .{ .io = Io.failing, .docs = &docs, .disk = false };
    var g = try ws.discover(gpa, "untitled:Untitled-1", "import std/math\nfn main() -> int { return 0 }\n");
    defer g.deinit(gpa);
    try testing.expect(g.err == null);
}
