//! What a call node, or a value-position function name, selects, recorded once by the
//! checker. A peer data module: types produces it, lower / the fingerprint / the LSP
//! consume it — each depends on this DATA, not on the checker's dispatch rules.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");

/// The selection for one call, stored on the call's slot (see `callSlot`), or for one
/// function named as a value, stored on that name's own node, in a slice parallel to
/// `node_types`. Zero is `.none`, so a fresh table is all-unselected.
pub const Callee = extern struct {
    tag: Tag = .none,
    /// `.builtin`: the `Builtin` ordinal.
    sub: u8 = 0,
    /// `.protocol`: the method's index within its protocol.
    aux: u16 = 0,
    /// `.method`: an index into `GraphResult.methods`. `.protocol`: the protocol id.
    /// `.fn_ref`: the referenced fn's global id.
    id: u32 = 0,

    /// `.pending` is checker-internal: the method row only exists once the mono tail
    /// has reified the receiver, so the seal resolves it; no consumer ever sees it.
    /// `.value`: a call of a function-typed value (`f(1)`, `h.run(4)`, `fs[i](x)`).
    /// `.fn_ref`: a named function used as a value (`f := add`), on the name's node.
    pub const Tag = enum(u8) { none, method, protocol, builtin, pending, value, fn_ref };

    /// The compiler-provided methods that have no `methods` row. The four `opt_*`
    /// ops mirror `Typecheck.OptResultMethod` (`opt_is_tag0` is `is_some`/`is_ok`);
    /// a reader wanting the spelled name reads the member token.
    pub const Builtin = enum(u8) {
        scalar_eq,
        scalar_hash,
        str_concat,
        str_len,
        str_byte_at,
        scalar_to_string,
        display_to_string,
        conv_into,
        conv_try_into,
        opt_is_tag0,
        opt_is_tag1,
        opt_unwrap,
        opt_unwrap_or,
    };

    pub const ProtocolMethod = struct { protocol: u32, method: u16 };

    pub const View = union(enum) {
        none,
        method: u32,
        protocol: ProtocolMethod,
        builtin: Builtin,
        value,
        fn_ref: u32,
    };

    pub const pending: Callee = .{ .tag = .pending };

    pub const value_call: Callee = .{ .tag = .value };

    pub fn fnRef(gid: u32) Callee {
        return .{ .tag = .fn_ref, .id = gid };
    }

    pub fn method(index: u32) Callee {
        return .{ .tag = .method, .id = index };
    }

    pub fn protocolMethod(protocol: u32, k: u16) Callee {
        return .{ .tag = .protocol, .aux = k, .id = protocol };
    }

    pub fn builtinMethod(k: Builtin) Callee {
        return .{ .tag = .builtin, .sub = @intFromEnum(k) };
    }

    /// The only way a consumer reads a slot.
    pub fn view(c: Callee) View {
        return switch (c.tag) {
            .none, .pending => .none,
            .method => .{ .method = c.id },
            .protocol => .{ .protocol = .{ .protocol = c.id, .method = c.aux } },
            .builtin => .{ .builtin = @enumFromInt(c.sub) },
            .value => .value,
            .fn_ref => .{ .fn_ref = c.id },
        };
    }

    /// The slot of a `.call` node: its member node when it has one, else its callee
    /// node, whatever that callee's shape (`f(..)`, `fs[i](..)`, `get()(..)`).
    pub fn callSlot(tree: Ast.Tree, call: Ast.Node) ?Ast.Index {
        if (call.tag != .call or call.lhs == Ast.none) return null;
        return memberNode(tree, call) orelse call.lhs;
    }

    /// The slot of a `.call` node: its `field_access` callee (`v.m(..)`), or the
    /// `field_access` under a `type_app` callee (`v.m[T](..)`). Null for any other
    /// callee shape. A qualified `mod.f()` also yields its field_access; that slot is
    /// simply never written.
    pub fn memberNode(tree: Ast.Tree, call: Ast.Node) ?Ast.Index {
        if (call.tag != .call or call.lhs == Ast.none) return null;
        const cn = tree.nodes[call.lhs.int()];
        switch (cn.tag) {
            .field_access => return call.lhs,
            .type_app => {
                if (cn.lhs == Ast.none) return null;
                return if (tree.nodes[cn.lhs.int()].tag == .field_access) cn.lhs else null;
            },
            else => return null,
        }
    }

    comptime {
        std.debug.assert(@sizeOf(Callee) == 8);
    }
};

test "Callee is 8 bytes and zero is none" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(Callee));
    try std.testing.expectEqual(Callee.View.none, std.mem.zeroes(Callee).view());
    try std.testing.expectEqual(Callee.View.none, (Callee{}).view());
    try std.testing.expectEqual(Callee.View.none, Callee.pending.view());
}

test "view round-trips every constructor" {
    try std.testing.expectEqual(Callee.View{ .method = 7 }, Callee.method(7).view());
    try std.testing.expectEqual(Callee.View{ .protocol = .{ .protocol = 3, .method = 2 } }, Callee.protocolMethod(3, 2).view());
    inline for (@typeInfo(Callee.Builtin).@"enum".fields) |f| {
        const k: Callee.Builtin = @enumFromInt(f.value);
        try std.testing.expectEqual(Callee.View{ .builtin = k }, Callee.builtinMethod(k).view());
    }
}

test "view round-trips value and fn_ref" {
    try std.testing.expectEqual(Callee.View.value, Callee.value_call.view());
    try std.testing.expectEqual(Callee.View{ .fn_ref = 42 }, Callee.fnRef(42).view());
    try std.testing.expect(!std.meta.eql(Callee.value_call, Callee{}));
}

test "memberNode: plain, explicit-args and non-member callees" {
    const Node = Ast.Node;
    const idx = Ast.Index.from;

    // 0: identifier v   1: field_access(0, .m)   2: identifier int
    // 3: type_app(1, [2])   4: identifier f   5: type_app(4, [2])
    // 6: call(1)  v.m()   7: call(3)  v.m[int]()   8: call(4)  f()   9: call(5)  f[int]()
    var extra = [_]u32{ 0, 0, 2, 1, 2 };
    var nodes = [_]Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 2, .lhs = idx(0), .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 4, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .type_app, .main_token = 3, .lhs = idx(1), .rhs = idx(2) },
        .{ .tag = .identifier, .main_token = 6, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .type_app, .main_token = 7, .lhs = idx(4), .rhs = idx(2) },
        .{ .tag = .call, .main_token = 3, .lhs = idx(1), .rhs = idx(0) },
        .{ .tag = .call, .main_token = 6, .lhs = idx(3), .rhs = idx(0) },
        .{ .tag = .call, .main_token = 7, .lhs = idx(4), .rhs = idx(0) },
        .{ .tag = .call, .main_token = 9, .lhs = idx(5), .rhs = idx(0) },
    };
    const tree: Ast.Tree = .{ .nodes = &nodes, .extra = &extra };

    try std.testing.expectEqual(@as(?Ast.Index, idx(1)), Callee.memberNode(tree, nodes[6]));
    try std.testing.expectEqual(@as(?Ast.Index, idx(1)), Callee.memberNode(tree, nodes[7]));
    try std.testing.expectEqual(@as(?Ast.Index, null), Callee.memberNode(tree, nodes[8]));
    try std.testing.expectEqual(@as(?Ast.Index, null), Callee.memberNode(tree, nodes[9]));
    try std.testing.expectEqual(@as(?Ast.Index, null), Callee.memberNode(tree, nodes[1]));
}

test "callSlot: member callee, bare, index, call and lambda-shaped callees" {
    const Node = Ast.Node;
    const idx = Ast.Index.from;

    // 0: identifier v   1: field_access(0, .m)   2: identifier f
    // 3: index(2, 0)  f[v]   4: call(2)  f()   5: call(1)  v.m()
    // 6: call(3)  f[v]()   7: call(4)  f()()   8: block   9: call(8)  { .. }()
    var extra = [_]u32{ 0, 0 };
    var nodes = [_]Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 2, .lhs = idx(0), .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 3, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .index, .main_token = 4, .lhs = idx(2), .rhs = idx(0) },
        .{ .tag = .call, .main_token = 4, .lhs = idx(2), .rhs = idx(0) },
        .{ .tag = .call, .main_token = 3, .lhs = idx(1), .rhs = idx(0) },
        .{ .tag = .call, .main_token = 6, .lhs = idx(3), .rhs = idx(0) },
        .{ .tag = .call, .main_token = 7, .lhs = idx(4), .rhs = idx(0) },
        .{ .tag = .block, .main_token = 8, .lhs = idx(0), .rhs = Ast.none },
        .{ .tag = .call, .main_token = 9, .lhs = idx(8), .rhs = idx(0) },
    };
    const tree: Ast.Tree = .{ .nodes = &nodes, .extra = &extra };

    try std.testing.expectEqual(@as(?Ast.Index, idx(1)), Callee.callSlot(tree, nodes[5]));
    try std.testing.expectEqual(@as(?Ast.Index, idx(2)), Callee.callSlot(tree, nodes[4]));
    try std.testing.expectEqual(@as(?Ast.Index, idx(3)), Callee.callSlot(tree, nodes[6]));
    try std.testing.expectEqual(@as(?Ast.Index, idx(4)), Callee.callSlot(tree, nodes[7]));
    try std.testing.expectEqual(@as(?Ast.Index, idx(8)), Callee.callSlot(tree, nodes[9]));
    try std.testing.expectEqual(@as(?Ast.Index, null), Callee.callSlot(tree, nodes[1]));
}
