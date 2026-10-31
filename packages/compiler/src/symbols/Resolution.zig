//! What an identifier expression resolves to. A peer data module: resolve
//! produces it, types/codegen consume it — each depends on this DATA, not on the
//! resolve stage sideways.

const Ast = @import("../ast/Ast.zig");

/// What an identifier expression refers to, once resolved.
pub const Resolution = union(enum) {
    /// Not (yet) resolved, or resolution failed (an error was reported).
    unresolved,
    /// A parameter or `:=` local, named by its per-function slot index.
    local: u32,
    /// A top-level function, named by its index into the function table. In a
    /// single-file resolve this is the per-file fn index; in a whole-graph resolve
    /// (`resolve_graph`) it is the GRAPH-GLOBAL fn id (an index into the merged
    /// program-wide function table). Either way Typecheck/Codegen index their fn
    /// table by it — the table is built in the same id space the resolver used.
    func: u32,
    /// An imported module namespace bound by an `import` decl (the `.` 4th
    /// meaning). Carries the GRAPH-GLOBAL module id of the imported module.
    /// Written onto the *receiver* identifier of a `mod.member` field-access; the
    /// member itself (e.g. the callee `mod.fn`) is resolved on the field_access
    /// node to a `.func` (cross-module) or stays quiet for a qualified type/enum
    /// reference (resolved later by Typecheck against that module's tables).
    module: u32,
    /// On a `break`/`continue` node carrying a `@label`: the target's INNER
    /// construct node (the `loop_expr`/`while_stmt`/`for_stmt`/`block` that the
    /// `labeled` wrapper wraps). Typecheck/Codegen match this against their
    /// context stacks to pick the named (not innermost) target.
    label: Ast.Index,
};

/// A bracketed postfix `head[..]` is a VALUE index — not a `Type[args]` turbofish
/// — when the LEFTMOST identifier of the head chain resolves to a value. Walk the
/// `field_access` receiver chain down to its leftmost identifier and test it:
/// `b.items[i]` (leftmost `b` -> .local) is an index; `mod.Box[int]` (leftmost
/// `mod` -> .module) and bare `Vec[int]` (leftmost `Vec` -> .unresolved) stay
/// turbofishes. Params and `:=` locals are both `.local`, so one test suffices.
/// The three stages (resolve/check/lower) share this single authority so their
/// value-index predicates cannot drift.
pub fn leftmostHeadIsValue(tree: Ast.Tree, resolutions: []const Resolution, head: Ast.Index) bool {
    var cur = head;
    while (cur != Ast.none) {
        const node = tree.nodes[cur.int()];
        switch (node.tag) {
            .field_access => cur = node.lhs,
            .identifier => return resolutions[cur.int()] == .local,
            else => return false,
        }
    }
    return false;
}

/// Whether the Vec an `.index` (or a chain of them) reads is held by a name or a field
/// (`xs[i]`, `h.xs[i][j]`), not made on the spot by a call, block or literal
/// (`mk()[i]`). An element of such a temporary is no place: writing it is an error like
/// writing the temporary itself.
pub fn indexesHeldVec(nodes: []const Ast.Node, index: Ast.Index) bool {
    var cur = index;
    while (nodes[cur.int()].tag == .index) cur = nodes[cur.int()].lhs;
    return switch (nodes[cur.int()].tag) {
        .identifier, .field_access, .tuple_field, .type_app, .error_node => true,
        else => false,
    };
}

test indexesHeldVec {
    const std = @import("std");
    const idx = Ast.Index.from;
    // 0: xs  1: xs[0]  2: xs[0][1]  3: h.xs  4: h.xs[0]  5: mk()  6: mk()[0]  7: mk()[0][1]
    const nodes = [_]Ast.Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .index, .main_token = 1, .lhs = idx(0), .rhs = Ast.none },
        .{ .tag = .index, .main_token = 2, .lhs = idx(1), .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 3, .lhs = idx(0), .rhs = Ast.none },
        .{ .tag = .index, .main_token = 4, .lhs = idx(3), .rhs = Ast.none },
        .{ .tag = .call, .main_token = 5, .lhs = idx(0), .rhs = Ast.none },
        .{ .tag = .index, .main_token = 6, .lhs = idx(5), .rhs = Ast.none },
        .{ .tag = .index, .main_token = 7, .lhs = idx(6), .rhs = Ast.none },
    };
    try std.testing.expect(indexesHeldVec(&nodes, idx(1)));
    try std.testing.expect(indexesHeldVec(&nodes, idx(2)));
    try std.testing.expect(indexesHeldVec(&nodes, idx(4)));
    try std.testing.expect(!indexesHeldVec(&nodes, idx(6)));
    try std.testing.expect(!indexesHeldVec(&nodes, idx(7)));
}

/// The Vec element a place chain (`xs[i]`, `ps[i].q.a`, or a value-headed `b.ps[i].x`)
/// is rooted at: an `.index`, or a `type_app` whose head is a value, which the parser could
/// not tell from a turbofish. Null for a place rooted anywhere else. The one walker the
/// checker and lower share, so a written element is marked exactly where it is lowered.
pub fn elementRoot(tree: Ast.Tree, resolutions: []const Resolution, place: Ast.Index) ?Ast.Index {
    var cur = place;
    while (true) {
        const n = tree.nodes[cur.int()];
        switch (n.tag) {
            .field_access, .tuple_field => cur = n.lhs,
            .index => return if (indexesHeldVec(tree.nodes, cur)) cur else null,
            .type_app => return if (leftmostHeadIsValue(tree, resolutions, n.lhs)) cur else null,
            else => return null,
        }
    }
}

/// The local a place (`x`, `x.f`, `x.0`, or a chain of them) is rooted at: its resolver
/// slot, or null for a place rooted anywhere else (an index, a call), whose store writes
/// no local. The one walker the checker, lower and the LSP share.
pub fn placeRoot(tree: Ast.Tree, resolutions: []const Resolution, place: Ast.Index) ?u32 {
    var cur = place;
    while (cur != Ast.none and cur.int() < resolutions.len) {
        const n = tree.nodes[cur.int()];
        switch (n.tag) {
            .field_access, .tuple_field => cur = n.lhs,
            .identifier => return switch (resolutions[cur.int()]) {
                .local => |slot| slot,
                else => null,
            },
            else => return null,
        }
    }
    return null;
}

/// Whether the `field_access` `access` is a qualified unit-variant construction (`N.V`,
/// `mod.N.V`, `Opt[int].V`) rather than a field read: its receiver names a type, not a
/// value. Being rooted at no local is not enough, as `ps[0].s` and `f().s` read fields
/// off receivers that are no place.
pub fn isQualifiedVariant(tree: Ast.Tree, resolutions: []const Resolution, access: Ast.Index) bool {
    const recv_idx = tree.nodes[access.int()].lhs;
    const recv = tree.nodes[recv_idx.int()];
    return switch (recv.tag) {
        .identifier => resolutions[recv_idx.int()] == .unresolved,
        .field_access => resolutions[recv.lhs.int()] == .module,
        .type_app => !leftmostHeadIsValue(tree, resolutions, recv.lhs),
        else => false,
    };
}

test isQualifiedVariant {
    const std = @import("std");
    const Node = Ast.Node;
    const idx = Ast.Index.from;

    // 0: identifier N (unresolved)  1: field_access(0) `N.V`
    // 2: identifier mod (module)  3: field_access(2) `mod.N`  4: field_access(3) `mod.N.V`
    // 5: identifier ps (local)  6: index(5)  7: field_access(6) `ps[0].s`
    // 8: field_access(5) `ps.s`  9: call  10: field_access(9) `f().s`
    var nodes = [_]Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 1, .lhs = idx(0), .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 2, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 3, .lhs = idx(2), .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 4, .lhs = idx(3), .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 5, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .index, .main_token = 6, .lhs = idx(5), .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 7, .lhs = idx(6), .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 8, .lhs = idx(5), .rhs = Ast.none },
        .{ .tag = .call, .main_token = 9, .lhs = idx(0), .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 10, .lhs = idx(9), .rhs = Ast.none },
    };
    const tree: Ast.Tree = .{ .nodes = &nodes, .extra = &.{} };
    const res = [_]Resolution{ .unresolved, .unresolved, .{ .module = 1 }, .unresolved, .unresolved, .{ .local = 0 }, .unresolved, .unresolved, .unresolved, .unresolved, .unresolved };

    try std.testing.expect(isQualifiedVariant(tree, &res, idx(1)));
    try std.testing.expect(isQualifiedVariant(tree, &res, idx(4)));
    try std.testing.expect(!isQualifiedVariant(tree, &res, idx(7)));
    try std.testing.expect(!isQualifiedVariant(tree, &res, idx(8)));
    try std.testing.expect(!isQualifiedVariant(tree, &res, idx(10)));
}

test placeRoot {
    const std = @import("std");
    const Node = Ast.Node;
    const idx = Ast.Index.from;

    // 0: identifier (local 3)  1: field_access(0)  2: tuple_field(1)
    // 3: identifier (module)  4: field_access(3)  5: call
    var nodes = [_]Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 1, .lhs = idx(0), .rhs = Ast.none },
        .{ .tag = .tuple_field, .main_token = 2, .lhs = idx(1), .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 3, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 4, .lhs = idx(3), .rhs = Ast.none },
        .{ .tag = .call, .main_token = 5, .lhs = idx(0), .rhs = Ast.none },
    };
    const tree: Ast.Tree = .{ .nodes = &nodes, .extra = &.{} };
    const res = [_]Resolution{ .{ .local = 3 }, .unresolved, .unresolved, .{ .module = 0 }, .unresolved, .unresolved };

    try std.testing.expectEqual(@as(?u32, 3), placeRoot(tree, &res, idx(0)));
    try std.testing.expectEqual(@as(?u32, 3), placeRoot(tree, &res, idx(2)));
    try std.testing.expectEqual(@as(?u32, null), placeRoot(tree, &res, idx(4)));
    try std.testing.expectEqual(@as(?u32, null), placeRoot(tree, &res, idx(5)));
    try std.testing.expectEqual(@as(?u32, null), placeRoot(tree, &res, Ast.none));
}

test leftmostHeadIsValue {
    const std = @import("std");
    const Node = Ast.Node;
    const idx = Ast.Index.from;

    // 0: identifier (local)  1: identifier (module)
    // 2: field_access(0, .items)  3: field_access(2, .g)
    // 4: field_access(1, .Box)
    var nodes = [_]Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 1, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 2, .lhs = idx(0), .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 3, .lhs = idx(2), .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 4, .lhs = idx(1), .rhs = Ast.none },
    };
    const tree: Ast.Tree = .{ .nodes = &nodes, .extra = &.{} };
    const res = [_]Resolution{ .{ .local = 0 }, .{ .module = 0 }, .unresolved, .unresolved, .unresolved };

    try std.testing.expect(leftmostHeadIsValue(tree, &res, idx(0))); // bare local
    try std.testing.expect(leftmostHeadIsValue(tree, &res, idx(3))); // b.items.g -> leftmost local
    try std.testing.expect(!leftmostHeadIsValue(tree, &res, idx(1))); // bare module
    try std.testing.expect(!leftmostHeadIsValue(tree, &res, idx(4))); // mod.Box -> leftmost module
    try std.testing.expect(!leftmostHeadIsValue(tree, &res, Ast.none));
}
