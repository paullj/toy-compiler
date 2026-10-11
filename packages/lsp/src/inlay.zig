//! textDocument/inlayHint: what a closure or a call does to a place, shown where the
//! source has no word for it (see `places.zig` for the facts).
//!
//! * A lambda or local fn that captures gets its capture list before its body,
//!   `fn(v) captures total (shared) { .. }`, each capture's storage detail in the tooltip.
//!   A `_` capture and a bound method value get none: every local they capture is spelled
//!   in the one expression that makes them.
//! * An argument a `mut` param takes gets `mut` before it, since the call site carries no
//!   marker (closures.md, `mut` parameters); the tooltip names the callee. A `mut self`
//!   receiver gets none: as in Rust and Swift, a receiver is unmarked and its method names
//!   what happens to it. Hover says "mutated by f" on both.

const std = @import("std");
const protocol = @import("protocol.zig");
const places = @import("places.zig");
const Analysis = @import("Analysis.zig");

/// The hints whose position falls in `range` (both ends included), in source order,
/// allocated in `a`. A document that does not parse has none; resolve and type errors
/// elsewhere do not suppress them.
pub fn in(a: std.mem.Allocator, an: *Analysis, range: protocol.Range) ![]const protocol.InlayHint {
    if (an.checked == null) return &.{};
    const m = an.module();
    const lo = an.offsetAt(range.start.line, range.start.character) orelse return &.{};
    const hi = an.offsetAt(range.end.line, range.end.character) orelse @as(u32, @intCast(m.source.len));

    const At = struct { off: u32, hint: protocol.InlayHint };
    var found: std.ArrayList(At) = .empty;

    for (try places.capturing(an, a)) |cl| {
        const off = m.tokens[cl.body].start;
        if (off < lo or off > hi) continue;
        var label: std.ArrayList(u8) = .empty;
        var tip: std.ArrayList(u8) = .empty;
        try label.appendSlice(a, "captures ");
        for (cl.items, 0..) |it, k| {
            if (k != 0) {
                try label.appendSlice(a, ", ");
                try tip.append(a, '\n');
            }
            try label.appendSlice(a, it.name);
            if (it.hold.word()) |w| try label.print(a, " ({s})", .{w});
            try tip.appendSlice(a, it.name);
            if (it.hold != .unknown) {
                try tip.appendSlice(a, ": ");
                try it.hold.append(a, &tip);
            }
        }
        try found.append(a, .{ .off = off, .hint = .{
            .position = an.positionOf(off),
            .label = label.items,
            .tooltip = tip.items,
            .paddingRight = true,
        } });
    }

    for (try an.mutArgs()) |ma| {
        if (ma.receiver) continue;
        const off = an.span(ma.node).start;
        if (off < lo or off > hi) continue;
        try found.append(a, .{ .off = off, .hint = .{
            .position = an.positionOf(off),
            .label = "mut",
            .kind = protocol.inlay_kind.parameter,
            .tooltip = try std.fmt.allocPrint(a, "mutated by {s}", .{ma.callee}),
            .paddingRight = true,
        } });
    }

    std.mem.sort(At, found.items, {}, struct {
        fn lt(_: void, x: At, y: At) bool {
            return x.off < y.off;
        }
    }.lt);
    const hints = try a.alloc(protocol.InlayHint, found.items.len);
    for (found.items, hints) |f, *h| h.* = f.hint;
    return hints;
}

const testing = std.testing;
const Documents = @import("Documents.zig");
const Workspace = @import("Workspace.zig");
const position = @import("position.zig");
const test_util = @import("test_util.zig");

/// A test's hints: their analysis and the answer, torn down together.
const Result = struct {
    an: *Analysis,
    arena: std.heap.ArenaAllocator,
    hints: []const protocol.InlayHint,

    fn deinit(r: *Result) void {
        r.arena.deinit();
        r.an.deinit();
    }
};

/// The hints in `range` of `source`, checked as the document `doc_uri` in `ws`.
fn hintsIn(gpa: std.mem.Allocator, ws: Workspace, source: []const u8, range: protocol.Range, doc_uri: []const u8) !Result {
    const an = try test_util.analyzeIn(gpa, ws, doc_uri, source);
    errdefer an.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const hints = try in(arena.allocator(), an, range);
    return .{ .an = an, .arena = arena, .hints = hints };
}

fn whole(src: []const u8) protocol.Range {
    return .{ .start = .{ .line = 0, .character = 0 }, .end = test_util.posAfterLast(src, "}") };
}

const fixture =
    \\fn apply(f: fn(int) -> int, v: int) -> int { f(v) }
    \\fn add(a: int, b: int) -> int { a + b }
    \\fn inc(mut n: int, k: int) { n = n + k }
    \\struct Acc { n: int }
    \\impl Acc {
    \\    fn plus(self, k: int) -> int { self.n + k }
    \\    fn bump(mut self) { self.n = self.n + 1 }
    \\}
    \\fn main() -> int {
    \\    base := 10
    \\    total := 0
    \\    r := apply(fn(v) { total = total + v + base }, 5)
    \\    fn fact(k: int) -> int { if k <= 1 { base } else { k * fact(k - 1) } }
    \\    plain := apply(fn(v) { v * 2 }, 1)
    \\    addb := add(base, _)
    \\    acc := Acc { n: 1 }
    \\    bound := acc.plus
    \\    inc(total, 2)
    \\    acc.bump()
    \\    return r + fact(3) + plain + addb(1) + bound(1) + total
    \\}
    \\
;

test "hintsIn: capture lists before closure bodies and mut before mut arguments, in source order" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    var r = try hintsIn(gpa, ws, fixture, whole(fixture), "file:///p/main.toy");
    defer r.deinit();

    const Want = struct { at: protocol.Position, label: []const u8, tooltip: []const u8, kind: ?u8 = null };
    const want = [_]Want{
        .{ .at = test_util.posOf(fixture, "{ total = total"), .label = "captures base (copied), total (shared)", .tooltip = "base: copied (8 bytes)\ntotal: shared" },
        .{ .at = test_util.posOf(fixture, "{ if k"), .label = "captures base (copied)", .tooltip = "base: copied (8 bytes)" },
        .{ .at = test_util.posOf(fixture, "total, 2)"), .label = "mut", .tooltip = "mutated by inc", .kind = protocol.inlay_kind.parameter },
    };
    try testing.expectEqual(want.len, r.hints.len);
    for (want, r.hints) |w, h| {
        try testing.expectEqual(w.at, h.position);
        try testing.expectEqualStrings(w.label, h.label);
        try testing.expectEqualStrings(w.tooltip, h.tooltip.?);
        try testing.expectEqual(w.kind, h.kind);
        try testing.expectEqual(@as(?bool, true), h.paddingRight);
    }
}

test "hintsIn keeps only the requested range, counted in the client's encoding" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    // `é` is 2 bytes but 1 UTF-16 unit, so the `mut` hint after it sits one column
    // further in utf-8.
    const src =
        \\fn inc(mut n: int) -> int {
        \\    n = n + 1
        \\    n
        \\}
        \\fn main() -> int {
        \\    c := 1
        \\    t := "é".len() + inc(c)
        \\    k := 2
        \\    g := fn() { k }
        \\    return c + t + g()
        \\}
        \\
    ;
    const line6: protocol.Range = .{ .start = .{ .line = 6, .character = 0 }, .end = .{ .line = 7, .character = 0 } };
    for ([_]position.Encoding{ .utf8, .utf16 }) |enc| {
        const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false, .encoding = enc };
        var r = try hintsIn(gpa, ws, src, line6, "file:///p/main.toy");
        defer r.deinit();
        try testing.expectEqual(@as(usize, 1), r.hints.len);
        try testing.expectEqualStrings("mut", r.hints[0].label);
        const bytes = test_util.posOf(src, "c)\n    k");
        const want: protocol.Position = .{ .line = bytes.line, .character = if (enc == .utf8) bytes.character else bytes.character - 1 };
        try testing.expectEqual(want, r.hints[0].position);

        var all = try hintsIn(gpa, ws, src, whole(src), "file:///p/main.toy");
        defer all.deinit();
        try testing.expectEqual(@as(usize, 2), all.hints.len);
        try testing.expectEqualStrings("captures k (copied)", all.hints[1].label);
    }
}

test "hintsIn: no hints for a document that does not parse, or a range past its end" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    var broken = try hintsIn(gpa, ws, "fn main( -> int {", whole("}"), "file:///p/main.toy");
    defer broken.deinit();
    try testing.expectEqual(@as(usize, 0), broken.hints.len);
    const past: protocol.Range = .{ .start = .{ .line = 900, .character = 0 }, .end = .{ .line = 901, .character = 0 } };
    var none = try hintsIn(gpa, ws, fixture, past, "file:///p/main.toy");
    defer none.deinit();
    try testing.expectEqual(@as(usize, 0), none.hints.len);
}
