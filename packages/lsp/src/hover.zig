//! textDocument/hover: what the token under the cursor is (see `describe.zig`), rendered as
//! a ```toy block the client highlights. A bound method value, a `_` capture, a captured
//! name and a `mut` argument also get a prose note after the block (see `places.zig`).
//!
//! A `{` / `(` / keyword token hovers its enclosing node's type by design (a `block` ->
//! `()`, a `call`'s `(` -> the call's result); a whitespace/comment GAP carries no token and
//! yields null (see `Analysis` on token-keyed selection).

const std = @import("std");
const toyc = @import("toy_compiler");
const Analysis = @import("Analysis.zig");
const describe = @import("describe.zig");
const places = @import("places.zig");
const render = @import("render.zig");

const Type = toyc.Typecheck.Type;

/// The hover's markdown at (`line`, `character`), allocated in `a`, or null when there is
/// nothing meaningful under the cursor (a gap, an out-of-range or poisoned position, or a
/// document that does not parse). Resolve and type errors elsewhere do not suppress it:
/// mid-edit, the rest of the buffer still types.
pub fn at(a: std.mem.Allocator, an: *Analysis, line: u32, character: u32) !?[]const u8 {
    if (an.checked == null) return null;
    const tok = an.tokenAt(line, character) orelse return null;
    var d: describe.Describer = .{ .a = a, .an = an };
    if (!try d.describe(tok)) return null;
    var value: std.ArrayList(u8) = .empty;
    // Fenced so the client highlights it as toy source.
    try value.print(a, "```toy\n{s}\n```", .{d.buf.items});
    try appendValueNotes(a, &value, an, tok);
    try appendFieldNote(a, &value, an, tok);
    try appendPlaceNotes(a, &value, an, tok);
    return value.items;
}

/// A field of another module's visibility and that module, as prose after the fence: the
/// fence shows its type, and a use never spells whether it may be made. In its own module
/// every field may be used, so the note would say nothing.
fn appendFieldNote(a: std.mem.Allocator, buf: *std.ArrayList(u8), an: *const Analysis, tok: u32) !void {
    const f = an.fieldAt(tok) orelse return;
    if (f.module == an.entry()) return;
    const path = an.graph.modules[f.module].path;
    if (an.fieldIsPub(f)) {
        try buf.print(a, "\n\npub field of module {s}", .{path});
    } else {
        try buf.print(a, "\n\nprivate to module {s}", .{path});
    }
}

/// Which function value `tok` makes, as prose after the fence: the member of a bound method
/// value names its receiver's type (the fence shows the method), and the `_` of a capture
/// gives the capture's fn type (the fence shows the `_` itself).
fn appendValueNotes(a: std.mem.Allocator, buf: *std.ArrayList(u8), an: *const Analysis, tok: u32) !void {
    const m = an.module();
    const nts = an.nodeTypes();
    const tc = an.tc();
    for (an.nodesAt(tok)) |i| {
        if (i >= nts.len) continue;
        const n = m.nodes[i];
        const ty: Type, const what: []const u8 = switch (n.tag) {
            .field_access => .{ an.boundReceiver(i) orelse continue, "bound method of " },
            .lambda => if (toyc.Ast.isCaptureLambda(m.tree(), m.tokens, toyc.Ast.Index.from(i)))
                .{ nts[i], "capture: " }
            else
                continue,
            else => continue,
        };
        if (!render.isRenderable(ty)) continue;
        try buf.print(a, "\n\n{s}", .{what});
        try render.renderType(a, buf, ty, tc.layouts, tc.enum_layouts);
        return;
    }
}

/// What the source never spells about the place `tok` names, as prose after the fence:
/// how a closure holds it, and which call writes it through a `mut` param.
fn appendPlaceNotes(a: std.mem.Allocator, buf: *std.ArrayList(u8), an: *Analysis, tok: u32) !void {
    if (places.captureAt(an, tok)) |h| {
        try buf.appendSlice(a, "\n\ncaptured");
        if (h != .unknown) {
            try buf.appendSlice(a, ": ");
            try h.append(a, buf);
        }
    }
    if (try places.mutatedBy(an, tok)) |callee| try buf.print(a, "\n\nmutated by {s}", .{callee});
}

const testing = std.testing;
const Workspace = @import("Workspace.zig");

/// A test's hover: its analysis and its answer, torn down together.
const TestHover = struct {
    an: *Analysis,
    arena: std.heap.ArenaAllocator,
    value: []const u8,

    fn deinit(h: *TestHover) void {
        h.arena.deinit();
        h.an.deinit();
    }
};

/// The hover at (`line`, `character`) in `src`, checked as the document `uri` in `ws`.
fn hoverAt(gpa: std.mem.Allocator, ws: Workspace, src: []const u8, line: u32, character: u32, uri: []const u8) !?TestHover {
    const an = try @import("test_util.zig").analyzeIn(gpa, ws, uri, src);
    errdefer an.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const value = (try at(arena.allocator(), an, line, character)) orelse {
        arena.deinit();
        an.deinit();
        return null;
    };
    return .{ .an = an, .arena = arena, .value = value };
}

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

test "hover on a capture's '_' shows the type the call gives it and the fn type of the capture" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
        \\struct Pt(int, str)
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn main() -> int {
        \\    acc := Acc { n: 1 }
        \\    inc := add(1, _)
        \\    p := acc.plus(_)
        \\    g := acc.plus
        \\    q := g(_)
        \\    mk := Pt(_, "s")
        \\    k := 5 |> add(2, _)
        \\    return inc(2) + p(3) + q(4) + mk(5).0 + k
        \\}
        \\
    ;
    const test_util = @import("test_util.zig");
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "_)\n    p :="), .want = "_: int\n```\n\ncapture: fn(int) -> int" },
        .{ .at = test_util.posOf(src, "_)\n    g :="), .want = "_: int\n```\n\ncapture: fn(int) -> int" },
        .{ .at = test_util.posOf(src, "_)\n    mk :="), .want = "_: int\n```\n\ncapture: fn(int) -> int" },
        .{ .at = test_util.posOf(src, "_, \"s\""), .want = "_: int\n```\n\ncapture: fn(int) -> Pt" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///p/main.toy")) orelse {
            std.debug.print("no hover at {d}:{d}\n", .{ c.at.line, c.at.character });
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
    // A pipe fills its `_` with the piped value: a plain call, no capture.
    const pipe_hole = test_util.posOf(src, "_)\n    return");
    try testing.expect(try hoverAt(gpa, ws, src, pipe_hole.line, pipe_hole.character, "file:///p/main.toy") == null);
}

test "hover on a bound method value shows its method and names the receiver's type" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct Acc { n: int }
        \\impl Acc {
        \\    fn plus(self, k: int) -> int { self.n + k }
        \\    fn bump(mut self) { self.n = self.n + 1 }
        \\}
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\    fn get(self) -> T { self.v }
        \\    fn mk(v: T) -> Box[T] { Box { v: v } }
        \\}
        \\fn main() -> int {
        \\    acc := Acc { n: 1 }
        \\    g := acc.plus
        \\    b := Box { v: "s" }
        \\    bg := b.get
        \\    up := acc.bump
        \\    up()
        \\    mk := Box[int].mk
        \\    return g(1) + bg().len() + acc.plus(2) + mk(3).v
        \\}
        \\
    ;
    const test_util = @import("test_util.zig");
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "plus\n"), .want = "fn plus(self: Acc, k: int) -> int\n```\n\nbound method of Acc" },
        .{ .at = test_util.posOf(src, "get\n"), .want = "fn get(self: Box[str]) -> str\n```\n\nbound method of Box[str]" },
        .{ .at = test_util.posOf(src, "bump\n"), .want = "fn bump(mut self: Acc) -> ()\n```\n\nbound method of Acc" },
        // A call of the method is not a value.
        .{ .at = test_util.posOf(src, "plus(2)"), .want = "fn plus(self: Acc, k: int) -> int\n```" },
        // An associated fn as a value binds no receiver.
        .{ .at = test_util.posOf(src, "mk\n"), .want = "fn mk(v: int) -> Box[int]\n```" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///p/main.toy")) orelse return error.NoHover;
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on the callee of a function value's call shows its type" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\import std/vec
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
        \\struct H { run: fn(int) -> int }
        \\fn dbl(x: int) -> int { x * 2 }
        \\fn apply(f: fn(int) -> int, v: int) -> int { f(v) }
        \\fn main() -> int {
        \\    acc := Acc { n: 1 }
        \\    g := acc.plus
        \\    h := H { run: dbl }
        \\    fs := [dbl]
        \\    return g(1) + h.run(2) + fs[0](3) + apply(g, 4) + 5 |> h.run()
        \\}
        \\
    ;
    const test_util = @import("test_util.zig");
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "f(v)"), .want = "f: fn(int) -> int" },
        .{ .at = test_util.posOf(src, "g(1)"), .want = "g: fn(int) -> int" },
        // A field read shows its type, as any field read does.
        .{ .at = test_util.posOf(src, "run(2)"), .want = "fn(int) -> int" },
        .{ .at = test_util.posOf(src, "run()"), .want = "fn(int) -> int" },
        // An element callee has no name token: its `[` heads it.
        .{ .at = test_util.posOf(src, "[0](3)"), .want = "fn(int) -> int" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///p/main.toy")) orelse return error.NoHover;
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on a method of a qualified-protocol conformance shows the method's name" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\import std/iter
        \\struct C { n: int }
        \\impl C has iter.Iterator[int] {
        \\    fn next(mut self) -> Option[int] {
        \\        self.n = self.n + 1
        \\        return Option.some(self.n)
        \\    }
        \\}
        \\fn main() -> int {
        \\    c := C { n: 0 }
        \\    _x := c.next()
        \\    return 0
        \\}
        \\
    ;
    const test_util = @import("test_util.zig");
    const want = "```toy\nfn next(mut self: C) -> Option[int]\n```";
    for ([_][]const u8{ "next(mut", "next()" }) |needle| {
        const pos = test_util.posOf(src, needle);
        var h = (try hoverAt(gpa, ws, src, pos.line, pos.character, "file:///p/main.toy")) orelse return error.NoHover;
        defer h.deinit();
        try testing.expectEqualStrings(want, h.value);
    }
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
        .{ .at = test_util.posOf(src, "expect(\"no value\")"), .want = "fn expect(self: Option[int], msg: str) -> int" },
        .{ .at = test_util.posOf(src, "or(0)"), .want = "fn or(self: Option[int], default: int) -> int" },
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

test "hover on a non-generic type's associated fn callee shows the fn" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const test_util = @import("test_util.zig");
    const src =
        \\struct K { n: int }
        \\impl K { fn make(n: int) -> K { K { n: n } } }
        \\enum E { A }
        \\impl E { fn first() -> E { E.A } }
        \\fn main() -> int {
        \\    k := K.make(1)
        \\    _e := E.first()
        \\    return k.n
        \\}
        \\
    ;
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "make(1)"), .want = "fn make(n: int) -> K" },
        .{ .at = test_util.posOf(src, "first()"), .want = "fn first() -> E" },
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
    try testing.expectEqualStrings("```toy\nfn map(self: Box[T], f: fn(T) -> U) -> Box[U]\n```", h.value);
}

test "hover on a non-generic type's own-generic method call shows the method as declared" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const tu = @import("test_util.zig");
    const src =
        \\struct K { v: int }
        \\impl K {
        \\    fn map[U](self, f: fn(int) -> U) -> U { f(self.v) }
        \\    fn with[U](v: int, _u: U) -> K { K { v: v } }
        \\}
        \\fn main() -> int {
        \\    k := K.with(2, "s")
        \\    n := k.map(fn(x) { x + 1 })
        \\    z := k.map(fn(s) { s.nope() })
        \\    return n + z
        \\}
        \\
    ;
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = tu.posAfterLast(src, "n := k."), .want = "fn map(self: K, f: fn(int) -> int) -> int" },
        .{ .at = tu.posAfterLast(src, "z := k."), .want = "fn map(self: K, f: fn(int) -> U) -> U" },
        .{ .at = tu.posAfterLast(src, "k := K."), .want = "fn with(v: int, _u: str) -> K" },
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

test "hover inside a bounded template: a prelude bound spells Self, a concrete receiver shows its type" {
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
        // The template body's own node types are scratch; its one instance knows the receiver.
        .{ .at = test_util.posOf(src, "eq(3)"), .want = "fn eq(self: int, other: int) -> bool" },
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

test "hover on a captured name says how its closure holds it" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
        \\fn apply(f: fn(int) -> int, v: int) -> int { f(v) }
        \\fn counter() -> fn() -> int {
        \\    n := 0
        \\    fn() {
        \\        n = n + 1
        \\        n
        \\    }
        \\}
        \\fn main() -> int {
        \\    total := 0
        \\    acc := Acc { n: 2 }
        \\    r := apply(fn(v) { total = total + v + acc.n }, 5)
        \\    plus := acc.plus
        \\    return r + total + counter()() + plus(1)
        \\}
        \\
    ;
    const test_util = @import("test_util.zig");
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "n = n"), .want = "n: int\n```\n\ncaptured: shared (boxed)" },
        .{ .at = test_util.posOf(src, "total = total"), .want = "total: int\n```\n\ncaptured: shared" },
        .{ .at = test_util.posOf(src, "acc.n }"), .want = "acc: Acc\n```\n\ncaptured: copied (8 bytes)" },
        .{ .at = test_util.posOf(src, "acc.plus"), .want = "acc: Acc\n```\n\ncaptured: copied (8 bytes)" },
        // The local outside its closure is not a capture.
        .{ .at = test_util.posOf(src, "total + counter"), .want = "total: int\n```" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///p/main.toy")) orelse return error.NoHover;
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on a compound assignment's target and value types them as with '='" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct P { x: int }
        \\fn main() -> int {
        \\    count := 3
        \\    step := 2
        \\    count += step
        \\    p := P { x: 1 }
        \\    p.x <<= count
        \\    bump := fn() { count -= 1 }
        \\    bump()
        \\    return count + p.x
        \\}
        \\
    ;
    const test_util = @import("test_util.zig");
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "count += step"), .want = "count: int\n```" },
        .{ .at = test_util.posOf(src, "step\n    p :="), .want = "step: int\n```" },
        .{ .at = test_util.posOf(src, "x <<="), .want = "int\n```" },
        .{ .at = test_util.posOf(src, "count -= 1"), .want = "count: int\n```\n\ncaptured: shared" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///c/main.toy")) orelse return error.NoHover;
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on a mut argument or a mut self receiver names the call that mutates it" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct P { x: int, y: int }
        \\impl P {
        \\    fn bump(mut self) { self.x = self.x + 1 }
        \\}
        \\fn inc(mut n: int, k: int) { n = n + k }
        \\fn run(f: fn()) { f() }
        \\fn main() -> int {
        \\    c := 1
        \\    p := P { x: 1, y: 2 }
        \\    inc(c, 2)
        \\    inc(p.y, c)
        \\    p.bump()
        \\    run(fn() { inc(c, 1) })
        \\    return c + p.x
        \\}
        \\
    ;
    const test_util = @import("test_util.zig");
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "c, 2)"), .want = "c: int\n```\n\nmutated by inc" },
        .{ .at = test_util.posOf(src, "y, c)"), .want = "int\n```\n\nmutated by inc" },
        .{ .at = test_util.posOf(src, "p.bump"), .want = "p: P\n```\n\nmutated by bump" },
        // Passed to a `mut` param inside a closure: captured, and boxed as an assigned one is.
        .{ .at = test_util.posOf(src, "c, 1)"), .want = "c: int\n```\n\ncaptured: shared\n\nmutated by inc" },
        // A by-value argument says nothing more.
        .{ .at = test_util.posOf(src, "c)\n    p.bump"), .want = "c: int\n```" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///p/main.toy")) orelse return error.NoHover;
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on a generic fn: a call shows the instance it runs, its declaration and a call in a template body as declared" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const test_util = @import("test_util.zig");
    const src =
        \\fn id[T](x: T) -> T { x }
        \\fn pair[T has Hash](v: T) -> int { id(v).hash() }
        \\fn main() -> int {
        \\    a := id(7)
        \\    b := id[str]("s")
        \\    return a + b.len() + pair(3)
        \\}
        \\
    ;
    const cases = [_]struct { at: @import("protocol.zig").Position, want: []const u8 }{
        .{ .at = test_util.posOf(src, "id(7)"), .want = "fn id(x: int) -> int" },
        .{ .at = test_util.posOf(src, "id[str]"), .want = "fn id(x: str) -> str" },
        .{ .at = test_util.posOf(src, "id[T]"), .want = "fn id(x: T) -> T" },
        .{ .at = test_util.posOf(src, "id(v)"), .want = "fn id(x: T) -> T" },
        .{ .at = test_util.posOf(src, "pair(3)"), .want = "fn pair(v: int) -> int" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///p/main.toy")) orelse return error.NoHover;
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on method calls inside generic bodies (#33)" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const test_util = @import("test_util.zig");
    const src = test_util.generic_body_fixture;
    const P = @import("protocol.zig").Position;
    const after = struct {
        fn f(p: P, n: u32) P {
            return .{ .line = p.line, .character = p.character + n };
        }
    }.f;
    const cases = [_]struct { at: P, want: []const u8 }{
        // A generic method's call follows the template's params: shown as declared.
        .{ .at = after(test_util.posOf(src, "self.get()"), 5), .want = "fn get(self: Box[T]) -> T\n```" },
        .{ .at = after(test_util.posOf(src, "b.get() }"), 2), .want = "fn get(self: Box[T]) -> T\n```" },
        // A concrete type's method is the same in every instance.
        .{ .at = test_util.posOf(src, "plus(2)"), .want = "fn plus(self: Acc, k: int) -> int\n```" },
        .{ .at = test_util.posOf(src, "plus\n"), .want = "fn plus(self: Acc, k: int) -> int\n```\n\nbound method of Acc" },
        // Outside a generic body, unchanged.
        .{ .at = test_util.posOf(src, "get()\n"), .want = "fn get(self: Box[int]) -> int\n```" },
    };
    for (cases) |c| {
        var h = (try hoverAt(gpa, ws, src, c.at.line, c.at.character, "file:///p/main.toy")) orelse {
            std.debug.print("no hover for '{s}'\n", .{c.want});
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on the struct of 'lib.S { .. }' never shows a same-named variant" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    try docs.put(gpa, "file:///q/lib.toy", "pub struct Pt { x: int }\n", 1);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\import lib
        \\
        \\enum Other { Pt(int), none }
        \\
        \\fn main() -> int {
        \\    p := lib.Pt { x: 1 }
        \\    return p.x
        \\}
        \\
    ;
    const pos = @import("test_util.zig").posOf(src, "Pt { x");
    var h = (try hoverAt(gpa, ws, src, pos.line, pos.character, "file:///q/main.toy")) orelse return error.NoHover;
    defer h.deinit();
    try testing.expectEqualStrings("```toy\nPt\n```", h.value);
}

test "hover on a field of another module names its visibility and that module" {
    const gpa = testing.allocator;
    const tu = @import("test_util.zig");
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws = try tu.privacyWorkspace(gpa, &docs);
    const src = tu.privacy_main;
    const ret = std.mem.indexOf(u8, src, "return").?;
    const cases = [_]struct { at: usize, want: []const u8 }{
        .{ .at = ret + std.mem.indexOf(u8, src[ret..], "x +").?, .want = "int\n```\n\npub field of module lib" },
        // The read T0078 rejects still hovers.
        .{ .at = ret + std.mem.indexOf(u8, src[ret..], "y +").?, .want = "int\n```\n\nprivate to module lib" },
        // In its own module a field's visibility changes nothing, so no note.
        .{ .at = ret + std.mem.indexOf(u8, src[ret..], "b\n").?, .want = "int\n```" },
        .{ .at = std.mem.indexOf(u8, src, "b: int").?, .want = "b: int\n```" },
        .{ .at = std.mem.indexOf(u8, src, "a: int").?, .want = "a: int\n```" },
    };
    for (cases) |c| {
        const pos = tu.posAt(src, c.at);
        var h = (try hoverAt(gpa, ws, src, pos.line, pos.character, tu.privacy_main_uri)) orelse return error.NoHover;
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on a struct literal's field names its visibility" {
    const gpa = testing.allocator;
    const tu = @import("test_util.zig");
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws = try tu.privacyWorkspace(gpa, &docs);
    const src = tu.privacy_literal;
    const cases = [_]struct { use: []const u8, want: []const u8 }{
        // The one place a private field can be set from outside is where T0078 lands.
        .{ .use = "y: 3", .want = "y: int\n```\n\nprivate to module lib" },
        .{ .use = "x: 4", .want = "x: int\n```\n\npub field of module lib" },
    };
    for (cases) |c| {
        const pos = tu.posOf(src, c.use);
        var h = (try hoverAt(gpa, ws, src, pos.line, pos.character, tu.privacy_main_uri)) orelse return error.NoHover;
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover on a selectively imported name under its alias shows the item it names" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    try docs.put(gpa, "file:///q/lib.toy", "pub struct Pt { pub x: int }\npub fn area(p: Pt) -> int { p.x }\n", 1);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\import lib.{area as size, Pt as P}
        \\
        \\fn main() -> int {
        \\    p := P { x: 1 }
        \\    return size(p)
        \\}
        \\
    ;
    const tu = @import("test_util.zig");
    const cases = [_]struct { at: []const u8, want: []const u8 }{
        .{ .at = "size(p)", .want = "fn area(p: Pt) -> int" },
        .{ .at = "P { x", .want = "pub struct Pt { pub x: int }" },
        .{ .at = "p := ", .want = "p: Pt" },
    };
    for (cases) |c| {
        const pos = tu.posOf(src, c.at);
        var h = (try hoverAt(gpa, ws, src, pos.line, pos.character, "file:///q/main.toy")) orelse return error.NoHover;
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}
