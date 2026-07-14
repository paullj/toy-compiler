//! The stdlib public symbols the front/middle-end reaches for by name — the single
//! source of truth for those strings. Surface sugar (for-in, list literals, indexing)
//! desugars against these methods/types, and the checker, discovery, and lowerer must
//! agree on the exact spelling. A rename in the corresponding std/*.toy otherwise
//! silently breaks that sugar with no compiler error; centralizing the strings makes
//! the rename a one-line edit here.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");

/// The list/vector struct (`std/vec.toy`) list-literal and index sugar target.
pub const vec_struct = "Vec";

/// The hashed-container struct (`std/map.toy`): the index gate rejects `m[k]` by this
/// name, and the `for k, v` desugar reads `Entry`'s two fields to bind key/val.
pub const map_struct = "Map";
pub const entry_struct = "Entry";

/// The protocol a `for-in` receiver's `iter()` must yield (`std/iter.toy`), matched by
/// name against `model.protocols`.
pub const iter_protocol = "Iterator";

/// Associated / instance method names the surface sugar desugars to.
pub const method_new = "new";
pub const method_push = "push";
pub const method_iter = "iter";
pub const method_next = "next";

/// The `core/mem` element-read primitive `xs[i]` desugars to, by qualified name.
pub const ga_at = "core/mem.ga_at";

/// Where a receiver-method desugar reads the receiver type it selects the method on.
pub const RecvSource = enum {
    /// The sugar node's own type (`node_types[i]`).
    node,
    /// The node's `rhs` child type (`node_types[n.rhs]`).
    rhs,
};

/// One receiver method a surface sugar expands to.
pub const DesugarMethod = struct { recv: RecvSource, name: []const u8 };

/// The receiver methods each surface sugar expands to, in expansion order — the single
/// enumeration mono discovery (`scanCalls`) and the lowerer both read, so the set they
/// discover and the set they emit provably agree (a method added here reaches both, and
/// lower's positional reads assert the length so a divergence is a compile error, not a
/// miscompile). Sugars that desugar to a free primitive rather than a receiver method
/// (`xs[i]` -> `ga_at`) are not receiver-method desugars and are not listed here.
pub fn desugarMethods(tag: Ast.Node.Tag) []const DesugarMethod {
    return switch (tag) {
        .empty_list => &.{.{ .recv = .node, .name = method_new }},
        .list_literal => &.{
            .{ .recv = .node, .name = method_new },
            .{ .recv = .node, .name = method_push },
        },
        .for_in_stmt => &.{
            .{ .recv = .rhs, .name = method_iter },
            .{ .recv = .node, .name = method_next },
        },
        // `for k, v in m` desugars identically to `for x in m`: `iter()` on the iterable
        // (`.rhs`), `next()` on the resulting iterator type (`.node` = node_types[stmt],
        // set by checkForIn2). The two bindings are a lower-only concern (the Entry field
        // split), so the discovered method set is the same as the single-binding form.
        .for_in2_stmt => &.{
            .{ .recv = .rhs, .name = method_iter },
            .{ .recv = .node, .name = method_next },
        },
        else => &.{},
    };
}

test "method names are distinct, non-empty, and ga_at is qualified" {
    const methods = [_][]const u8{ method_new, method_push, method_iter, method_next };
    for (methods) |m| try std.testing.expect(m.len != 0);
    for (methods, 0..) |a, i| for (methods[i + 1 ..]) |b|
        try std.testing.expect(!std.mem.eql(u8, a, b));
    try std.testing.expect(std.mem.indexOfScalar(u8, ga_at, '.') != null);
    try std.testing.expect(vec_struct.len != 0 and iter_protocol.len != 0);
}

test "desugarMethods enumerates each sugar's receiver methods in expansion order" {
    const empty = desugarMethods(.empty_list);
    try std.testing.expectEqual(@as(usize, 1), empty.len);
    try std.testing.expect(std.mem.eql(u8, empty[0].name, method_new) and empty[0].recv == .node);

    const list = desugarMethods(.list_literal);
    try std.testing.expectEqual(@as(usize, 2), list.len);
    try std.testing.expect(std.mem.eql(u8, list[0].name, method_new) and list[0].recv == .node);
    try std.testing.expect(std.mem.eql(u8, list[1].name, method_push) and list[1].recv == .node);

    const fin = desugarMethods(.for_in_stmt);
    try std.testing.expectEqual(@as(usize, 2), fin.len);
    try std.testing.expect(std.mem.eql(u8, fin[0].name, method_iter) and fin[0].recv == .rhs);
    try std.testing.expect(std.mem.eql(u8, fin[1].name, method_next) and fin[1].recv == .node);

    const fin2 = desugarMethods(.for_in2_stmt);
    try std.testing.expectEqual(@as(usize, 2), fin2.len);
    try std.testing.expect(std.mem.eql(u8, fin2[0].name, method_iter) and fin2[0].recv == .rhs);
    try std.testing.expect(std.mem.eql(u8, fin2[1].name, method_next) and fin2[1].recv == .node);

    try std.testing.expect(map_struct.len != 0 and entry_struct.len != 0);

    // A sugar with no receiver-method desugar (indexing goes through `ga_at`).
    try std.testing.expectEqual(@as(usize, 0), desugarMethods(.index).len);
}
