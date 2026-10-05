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

test "hover on a local fn shows it as a binding of its function type" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src = "fn main() -> int {\n    fn step(k: int) -> int { k }\n    return step(1)\n}\n";
    for ([_]struct { line: u32, col: u32 }{ .{ .line = 1, .col = 7 }, .{ .line = 2, .col = 11 } }) |c| {
        var h = (try hoverAt(gpa, ws, src, c.line, c.col, "file:///b/main.toy")) orelse return error.NoHover;
        defer h.deinit();
        try testing.expectEqualStrings("```toy\nstep: fn(int) -> int\n```", h.value);
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

test "hover on a piped callee shows its signature, also on a leading-pipe line" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn main() -> int {
        \\    x := 1 |> add(2)
        \\    y := x
        \\        |> add(3)
        \\    acc := Acc { n: y }
        \\    z := acc |> .plus(4)
        \\    return acc
        \\        |> .plus(z)
        \\}
        \\
    ;
    const cases = [_]struct { needle: []const u8, want: []const u8 }{
        .{ .needle = "add(2)", .want = "fn add(a: int, b: int) -> int" },
        .{ .needle = "add(3)", .want = "fn add(a: int, b: int) -> int" },
        .{ .needle = "plus(4)", .want = "fn plus(self: Acc, k: int) -> int" },
        .{ .needle = "plus(z)", .want = "fn plus(self: Acc, k: int) -> int" },
    };
    for (cases) |c| {
        const p = @import("test_util.zig").posOf(src, c.needle);
        var h = (try hoverAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no hover for '{s}'\n", .{c.needle});
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on a capture's '_' shows the type the call gives it" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn main() -> int {
        \\    inc := add(1, _)
        \\    return inc(2)
        \\}
        \\
    ;
    const p = @import("test_util.zig").posOf(src, "_)");
    var h = (try hoverAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse return error.NoHover;
    defer h.deinit();
    try testing.expectEqualStrings("```toy\n_: int\n```", h.value);
}

test "hover on a method callee shows the selected method" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const test_util = @import("test_util.zig");
    const src = test_util.method_fixture;
    const plus = "fn plus(self: Acc, a: int, b: int) -> int";
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "plus(1, 2)"), .want = plus },
        .{ .at = test_util.posOf(src, "plus(2)"), .want = plus },
        .{ .at = test_util.posOf(src, "make(3)"), .want = "fn make(v: int) -> Box[int]" },
        .{ .at = test_util.posOf(src, "get()"), .want = "fn get(self: Box[int]) -> int" },
        // A bounded receiver's method is the protocol's, shown as written.
        .{ .at = test_util.posOf(src, "dbl()"), .want = "fn dbl(self) -> int" },
        .{ .at = test_util.posOf(src, "concat("), .want = "fn concat(self: str, other: str) -> str" },
        .{ .at = test_util.posOf(src, "eq(3)"), .want = "fn eq(self: int, other: int) -> bool" },
        .{ .at = test_util.posOf(src, "unwrap()"), .want = "fn unwrap(self: Option[int]) -> int" },
        .{ .at = test_util.posOf(src, "unwrap_or(0)"), .want = "fn unwrap_or(self: Option[int], default: int) -> int" },
        // A derive's method row has no fn of its own (`fn_id` 0 names some other fn).
        .{ .at = test_util.posOf(src, "hash()"), .want = "fn hash(self: Key) -> int" },
        .{ .at = test_util.posOf(src, "len()"), .want = "fn len(self: str) -> int" },
        // A variant sharing a method's name selects no method: it keeps its type hover.
        .{ .at = test_util.posAfterLast(src, "Kind."), .want = "Kind" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///p/main.toy")) orelse {
            std.debug.print("no hover for '{s}'\n", .{c.want});
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on a generic method call whose args do not check shows the method as declared" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\    fn map[U](self, f: fn(T) -> U) -> Box[U] { Box { v: f(self.v) } }
        \\}
        \\fn main() -> int {
        \\    b := Box { v: "s" }
        \\    k := b.map(fn(s) { s.len() })
        \\    z := b.map(fn(s) { s.nope() })
        \\    return k.v + z.v
        \\}
        \\
    ;
    const p = @import("test_util.zig").posAfterLast(src, "z := b.");
    var h = (try hoverAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse return error.NoHover;
    defer h.deinit();
    try testing.expectEqualStrings("```toy\nfn map(self, f: fn(T) -> U) -> Box[U]\n```", h.value);
}

test "hover inside a bounded template: a prelude bound spells Self, a scratch receiver shows its name" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const test_util = @import("test_util.zig");
    const src =
        \\protocol One { fn one(self) -> int }
        \\impl int has One { fn one(self) -> int { self } }
        \\fn h[T has Hash](v: T) -> int {
        \\    e := if (3).eq(3) { 1 } else { 0 }
        \\    return v.hash() + e
        \\}
        \\fn o[T has Ord](v: T, w: T) -> Ordering { v.cmp(w) }
        \\fn f[T has One](v: T) -> int { v.one() }
        \\fn main() -> int {
        \\    _o := o(1, 2)
        \\    return h(7) + f(1)
        \\}
        \\
    ;
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "hash()"), .want = "fn hash(self: Self) -> int" },
        .{ .at = test_util.posOf(src, "cmp(w)"), .want = "fn cmp(self: Self, other: Self) -> Ordering" },
        // A one-line protocol's method shows without the protocol's braces.
        .{ .at = test_util.posOf(src, "one() }"), .want = "fn one(self) -> int" },
        // The template body's node types are scratch, so the receiver type is unknown here.
        .{ .at = test_util.posOf(src, "eq(3)"), .want = "fn eq(self, other) -> bool" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///p/main.toy")) orelse {
            std.debug.print("no hover for '{s}'\n", .{c.want});
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on a method call with an argument error still shows the selected method" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const test_util = @import("test_util.zig");
    const src =
        \\struct Key { a: int }
        \\fn main() -> int {
        \\    k := Key { a: 1 }
        \\    return k.hash(1) + k.to_string(2).len()
        \\}
        \\
    ;
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "hash(1)"), .want = "fn hash(self: Key) -> int" },
        .{ .at = test_util.posOf(src, "to_string(2)"), .want = "fn to_string(self: Key) -> str" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///p/main.toy")) orelse {
            std.debug.print("no hover for '{s}'\n", .{c.want});
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on a callee spells its mut params, a mut self receiver included" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct P { x: int }
        \\impl P {
        \\    fn bump(mut self, mut d: int) { self.x = self.x + d }
        \\}
        \\fn inc(mut n: int, k: int) { n = n + k }
        \\fn main() -> int {
        \\    c := 1
        \\    inc(c, 2)
        \\    p := P { x: 1 }
        \\    p.bump(c)
        \\    return c + p.x
        \\}
        \\
    ;
    const cases = [_]struct { needle: []const u8, want: []const u8 }{
        .{ .needle = "inc(c", .want = "fn inc(mut n: int, k: int) -> ()" },
        .{ .needle = "bump(c)", .want = "fn bump(mut self: P, mut d: int) -> ()" },
    };
    for (cases) |c| {
        const p = @import("test_util.zig").posOf(src, c.needle);
        var h = (try hoverAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no hover for '{s}'\n", .{c.needle});
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}
