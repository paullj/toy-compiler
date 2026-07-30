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
