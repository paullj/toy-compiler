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
const position = @import("position.zig");

const Graph = toyc.Graph;
const Cache = toyc.Cache;

const Workspace = @This();

io: Io,
docs: *const Documents,
/// false on a host with no filesystem (the browser): an import resolves only against open
/// documents.
disk: bool = true,
/// How the client counts `character` in the positions it sends and reads back.
encoding: position.Encoding = .utf16,

/// Discover the module graph rooted at `uri`, whose current text is `text` (which may differ
/// from the stored document: completion and signature help check a repaired copy). A
/// non-`file://` uri has no directory, so it can import only bundled modules.
pub fn discover(ws: Workspace, gpa: std.mem.Allocator, uri: []const u8, text: []const u8) !Graph.Graph {
    const path = (try lsp_uri.uriToPath(gpa, uri)) orelse try gpa.dupe(u8, untitled_path);
    defer gpa.free(path);
    const root = std.fs.path.dirname(path) orelse "";

    // The front-end asks for sources by decoded path, but a client's uri may percent-encode
    // it (`file:///a%20b/x.toy`): index the open documents by path, not by rebuilt uri.
    var by_path: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer {
        var it = by_path.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        by_path.deinit(gpa);
    }
    var docs = ws.docs.map.iterator();
    while (docs.next()) |e| {
        const doc_path = (try lsp_uri.uriToPath(gpa, e.key_ptr.*)) orelse continue;
        const gop = try by_path.getOrPut(gpa, doc_path);
        if (gop.found_existing) gpa.free(doc_path);
        gop.value_ptr.* = e.value_ptr.text;
    }
    // The entry's current text (possibly a repaired copy) wins over its stored document.
    const entry = try by_path.getOrPut(gpa, path);
    if (!entry.found_existing) entry.key_ptr.* = try gpa.dupe(u8, path);
    entry.value_ptr.* = text;

    const overlay: Graph.Overlay = .{ .ctx = &by_path, .getFn = lookup, .disk = ws.disk };
    return Graph.discoverWith(gpa, ws.io, Cache.disabled, "native", path, null, root, overlay);
}

const untitled_path = "untitled.toy";

fn lookup(ctx: *const anyopaque, path: []const u8) ?[]const u8 {
    const by_path: *const std.StringHashMapUnmanaged([]const u8) = @ptrCast(@alignCast(ctx));
    return by_path.get(path);
}

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

test "the entry's text wins over its own open document, without leaking" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    try docs.put(gpa, "file:///ws/main.toy", "fn main() -> int { return nope }\n", 1);

    const ws: Workspace = .{ .io = Io.failing, .docs = &docs, .disk = false };
    var g = try ws.discover(gpa, "file:///ws/main.toy", "fn main() -> int { return 0 }\n");
    defer g.deinit(gpa);
    try testing.expectEqualStrings("fn main() -> int { return 0 }\n", g.entry().source);
}

test "an import finds an open document whose uri percent-encodes its path" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    try docs.put(gpa, "file:///my%20ws/helper.toy", "pub fn seven() -> int { return 7 }\n", 1);

    const ws: Workspace = .{ .io = Io.failing, .docs = &docs, .disk = false };
    var g = try ws.discover(gpa, "file:///my%20ws/main.toy", "import helper\nfn main() -> int { return helper.seven() }\n");
    defer g.deinit(gpa);
    try testing.expect(g.err == null);
    try testing.expectEqual(@as(usize, 2), g.modules.len);
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
