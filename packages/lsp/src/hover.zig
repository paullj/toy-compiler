//! textDocument/hover: what the token under the cursor is (see `describe.zig`), rendered as
//! a ```toy block the client highlights.
//!
//! Checks through the same `Workspace` as `diagnostics.checkBuffer`, but instead of mapping
//! diagnostics it RETAINS the graph/resolve/typecheck results to describe the token, renders
//! the answer into an owned arena, THEN tears the compiler results down. The render must
//! precede those deinits: a `Sig.name` and a `Layout.name` are BORROWED from the
//! resolve/graph tables, so the arena copy is what outlives them.
//!
//! Selection is token-keyed because the AST stores only a `main_token` per node — no span,
//! no subtree-exit marker. The innermost node covering an offset is therefore "the node
//! whose `main_token` is the token under the cursor". A `{` / `(` / keyword token thus
//! hovers its enclosing node's type by design (a `block` -> `()`, a `call`'s `(` -> the
//! call's result); a whitespace/comment GAP carries no token and yields null.

const std = @import("std");
const toyc = @import("toy_compiler");
const Workspace = @import("Workspace.zig");
const position = @import("position.zig");
const describe = @import("describe.zig");

const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const SourceMap = toyc.term.render.SourceMap;

/// Owns the arena backing `value`. `kind` is a static string. `value` stays valid until
/// `deinit`; the compiler results it was rendered from have already been torn down.
pub const Hover = struct {
    arena: std.heap.ArenaAllocator,
    value: []const u8,
    kind: []const u8,

    pub fn deinit(self: *Hover) void {
        self.arena.deinit();
    }
};

/// The type/signature at (`line`, `character`) in `source`, or null when there is nothing
/// meaningful under the cursor (a gap, an out-of-range or poisoned position, or a document
/// that does not parse). Resolve and type errors elsewhere do not suppress it: mid-edit, the
/// rest of the buffer still types. Never errors on a compile problem — only a hard I/O / OOM
/// fault propagates.
pub fn hoverAt(
    gpa: std.mem.Allocator,
    ws: Workspace,
    source: []const u8,
    line: u32,
    character: u32,
    doc_uri: []const u8,
) !?Hover {
    var graph = try ws.discover(gpa, doc_uri, source);
    defer graph.deinit(gpa);
    if (graph.err != null) return null;

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, ws.io, 0);
    defer tc.deinit(gpa);

    const m = graph.entry();

    var sm = try SourceMap.init(gpa, m.file, m.source);
    defer sm.deinit(gpa);

    const off = position.positionToOffset(&sm, line, character, ws.encoding) orelse return null;
    const tok = position.tokenAt(m.tokens, off) orelse return null;

    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    var d: describe.Describer = .{ .a = arena.allocator(), .graph = &graph, .res = &res, .tc = &tc };
    if (!try d.describe(tok)) {
        arena.deinit();
        return null;
    }
    // Fenced so the client highlights it as toy source.
    const value = try std.fmt.allocPrint(d.a, "```toy\n{s}\n```", .{d.buf.items});
    return .{ .arena = arena, .value = value, .kind = "markdown" };
}

const testing = std.testing;

test "hover still types a binding when another line has a resolve error" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src = "fn main() -> int {\n    count := 3\n    return count + missing\n}\n";
    var h = (try hoverAt(gpa, ws, src, 1, 5, "file:///b/main.toy")) orelse return error.HoverSuppressed;
    defer h.deinit();
    try testing.expectEqualStrings("```toy\ncount: int\n```", h.value);
}

test "hover describes declarations, bindings, type names, fields, and variants" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\pub struct Square { side: int }
        \\
        \\pub fn grow(sq: Square, by: int) -> Square {
        \\    bigger := Square { side: sq.side + by }
        \\    return bigger
        \\}
        \\
        \\pub enum Shape { Circle(int), Box(int) }
        \\
        \\pub fn area(shape: Shape) -> int {
        \\    match shape {
        \\        .Circle(r) -> 3 * r * r,
        \\        .Box(side) -> side * side,
        \\    }
        \\}
        \\
    ;
    const cases = [_]struct { line: u32, needle: []const u8, want: []const u8 }{
        .{ .line = 0, .needle = "Square", .want = "pub struct Square { side: int }" },
        .{ .line = 0, .needle = "side", .want = "side: int" },
        .{ .line = 2, .needle = "grow", .want = "fn grow(sq: Square, by: int) -> Square" },
        .{ .line = 2, .needle = "sq", .want = "sq: Square" },
        .{ .line = 2, .needle = "Square", .want = "pub struct Square { side: int }" },
        .{ .line = 3, .needle = "bigger", .want = "bigger: Square" },
        .{ .line = 4, .needle = "bigger", .want = "bigger: Square" },
        .{ .line = 7, .needle = "Shape", .want = "pub enum Shape { Circle(int), Box(int) }" },
        .{ .line = 7, .needle = "Circle", .want = "Shape.Circle(int)" },
        .{ .line = 10, .needle = "shape", .want = "shape: Shape" },
        .{ .line = 11, .needle = "Circle", .want = "Shape.Circle(int)" },
        .{ .line = 11, .needle = "r)", .want = "r: int" },
        .{ .line = 11, .needle = "r * r", .want = "r: int" },
    };
    for (cases) |c| {
        var it = std.mem.splitScalar(u8, src, '\n');
        var i: u32 = 0;
        const col: u32 = while (it.next()) |l| : (i += 1) {
            if (i == c.line) break @intCast(std.mem.indexOf(u8, l, c.needle).?);
        } else unreachable;
        var h = (try hoverAt(gpa, ws, src, c.line, col, "file:///b/shapes.toy")) orelse {
            std.debug.print("no hover for '{s}' on line {d}\n", .{ c.needle, c.line });
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover shows generic, protocol, and prelude types as the user spells them" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct Pair[A, B] { left: A, right: B }
        \\protocol Area {
        \\    fn area(self) -> int
        \\}
        \\struct Sq { side: int }
        \\impl Sq has Area {
        \\    fn area(self) -> int { self.side * self.side }
        \\}
        \\fn main() -> int {
        \\    point := Pair[int, int] { left: 3, right: 39 }
        \\    first: Option[int] = Option.some(point.left)
        \\    return Sq { side: 2 }.area()
        \\}
        \\
    ;
    const cases = [_]struct { line: u32, needle: []const u8, want: []const u8 }{
        .{ .line = 0, .needle = "Pair", .want = "struct Pair[A, B] { left: A, right: B }" },
        .{ .line = 0, .needle = "left", .want = "left: A" },
        .{ .line = 1, .needle = "Area", .want = "protocol Area { fn area(self) -> int }" },
        .{ .line = 2, .needle = "area", .want = "fn area(self) -> int" },
        .{ .line = 5, .needle = "Area", .want = "protocol Area { fn area(self) -> int }" },
        .{ .line = 6, .needle = "area", .want = "fn area(self: Sq) -> int" },
        .{ .line = 9, .needle = "point", .want = "point: Pair[int, int]" },
        .{ .line = 9, .needle = "left", .want = "left: int" },
        .{ .line = 10, .needle = "Option", .want = "enum Option[T] { some(T), none }" },
        .{ .line = 10, .needle = "first", .want = "first: Option[int]" },
        .{ .line = 10, .needle = "some", .want = "Option[int].some(int)" },
    };
    for (cases) |c| {
        var it = std.mem.splitScalar(u8, src, '\n');
        var i: u32 = 0;
        const col: u32 = while (it.next()) |l| : (i += 1) {
            if (i == c.line) break @intCast(std.mem.indexOf(u8, l, c.needle).?);
        } else unreachable;
        var h = (try hoverAt(gpa, ws, src, c.line, col, "file:///b/main.toy")) orelse {
            std.debug.print("no hover for '{s}' on line {d}\n", .{ c.needle, c.line });
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}
