//! What a call node, or a function or method named as a value, selects, recorded once
//! by the checker. A peer data module: types produces it, lower / the fingerprint / the LSP
//! consume it — each depends on this DATA, not on the checker's dispatch rules.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");

/// The selection for one call, stored on the call's slot (see `callSlot`), or for one
/// function or method named as a value, stored on that name's own node, in a slice
/// parallel to `node_types`. Zero is `.none`, so a fresh table is all-unselected.
pub const Callee = extern struct {
    tag: Tag = .none,
    /// `.builtin`: the `Builtin` ordinal.
    sub: u8 = 0,
    /// `.protocol`: the method's index within its protocol.
    aux: u16 = 0,
    /// `.method`, `.bound_method`: an index into `GraphResult.methods`. `.protocol`: the
    /// protocol id. `.fn_ref`, `.direct`, `.generic`: the fn's global id. `.instance`: an
    /// index into `GraphResult.instances`.
    id: u32 = 0,

    /// The `pending_*` tags are checker-internal: the method row only exists once the
    /// mono tail has reified the receiver, so the seal resolves them; no consumer ever
    /// sees one. `.pending_method` becomes `.method`, `.pending_bound` `.bound_method`,
    /// and `.pending_call` (a call of a method with its own type params, or with explicit
    /// type args, `id` the call node) the `.method` row the call's arg types pick.
    /// `.pending_template` sits on such a call's own node when its explicit type args
    /// picked the template (`id`, its fn id; `aux` 1 when the args are the method's own
    /// type params), which the arg types alone cannot tell apart; the seal clears it.
    /// `.value`: a call of a function-typed value (`f(1)`, `h.run(4)`, `fs[i](x)`).
    /// `.fn_ref`: a named function used as a value (`f := add`), on the name's node.
    /// `.bound_method`: a method used as a value (`f := acc.plus`), on the `field_access`.
    /// `.template`: a call of a method with its own type params that selected no instance
    /// (its args did not check). It views as `.none`; only `templateFn` reads it, so the
    /// LSP can still show the method a half-typed `xs.map(` names.
    /// `.direct`: a call naming a non-generic fn (`f(x)`, `mod.f(x)`), builtins included.
    /// `.instance`: a call of a generic fn (`id(7)`, `id[int](7)`), by the instance it
    /// runs. `.generic`: a call of a generic fn whose instance is not known here: one in a
    /// template's own body, or one whose args did not check. The seal records these
    /// once mono has finished, so every reader shares one selection.
    /// `.instance_unkeyed`: an `.instance` call its caller's codegen key does not fold: a
    /// turbofish of a generic fn whose value params name no type param
    /// (`mem.ga_new[T]()`). The fold never took such a call for a generic one, so a change
    /// to that instance's signature does not reach the caller's key ([C2]); folding it
    /// changes every such caller's key, so it is a change of its own. Views as `.instance`.
    pub const Tag = enum(u8) { none, method, protocol, builtin, pending_method, pending_bound, pending_call, value, fn_ref, bound_method, template, direct, instance, generic, instance_unkeyed, pending_template };

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
        bound_method: u32,
        direct: u32,
        instance: u32,
        generic: u32,
    };

    pub const pending: Callee = .{ .tag = .pending_method };

    /// A bound method on a generic-type receiver, resolved by the seal like `pending`.
    pub const pending_bound: Callee = .{ .tag = .pending_bound };

    /// A call of a method with its own type params. One instance row per instantiation
    /// shares the receiver and name, so the seal picks the row by the call's arg types.
    pub fn pendingCall(call: Ast.Index) Callee {
        return .{ .tag = .pending_call, .id = call.int() };
    }

    /// The template an explicit-args call picked (see `.pending_template`).
    pub fn pendingTemplate(gid: u32, own: bool) Callee {
        return .{ .tag = .pending_template, .aux = @intFromBool(own), .id = gid };
    }

    pub fn isPending(c: Callee) bool {
        return switch (c.tag) {
            .pending_method, .pending_bound, .pending_call, .pending_template => true,
            else => false,
        };
    }

    pub const value_call: Callee = .{ .tag = .value };

    pub fn fnRef(gid: u32) Callee {
        return .{ .tag = .fn_ref, .id = gid };
    }

    pub fn method(index: u32) Callee {
        return .{ .tag = .method, .id = index };
    }

    /// `gid` is the generic method's own fn id.
    pub fn template(gid: u32) Callee {
        return .{ .tag = .template, .id = gid };
    }

    /// The method an unresolved own-generic call names (see `.template`), else null.
    pub fn templateFn(c: Callee) ?u32 {
        return if (c.tag == .template) c.id else null;
    }

    pub fn direct(gid: u32) Callee {
        return .{ .tag = .direct, .id = gid };
    }

    pub fn instanceCall(ii: u32) Callee {
        return .{ .tag = .instance, .id = ii };
    }

    pub fn genericCall(gid: u32) Callee {
        return .{ .tag = .generic, .id = gid };
    }

    pub fn boundMethod(index: u32) Callee {
        return .{ .tag = .bound_method, .id = index };
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
            .none, .pending_method, .pending_bound, .pending_call, .pending_template, .template => .none,
            .method => .{ .method = c.id },
            .protocol => .{ .protocol = .{ .protocol = c.id, .method = c.aux } },
            .builtin => .{ .builtin = @enumFromInt(c.sub) },
            .value => .value,
            .fn_ref => .{ .fn_ref = c.id },
            .bound_method => .{ .bound_method = c.id },
            .direct => .{ .direct = c.id },
            .instance, .instance_unkeyed => .{ .instance = c.id },
            .generic => .{ .generic = c.id },
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

test "view round-trips the free-fn calls" {
    try std.testing.expectEqual(Callee.View{ .direct = 3 }, Callee.direct(3).view());
    try std.testing.expectEqual(Callee.View{ .instance = 4 }, Callee.instanceCall(4).view());
    try std.testing.expectEqual(Callee.View{ .generic = 5 }, Callee.genericCall(5).view());
}

test "view round-trips value and fn_ref" {
    try std.testing.expectEqual(Callee.View.value, Callee.value_call.view());
    try std.testing.expectEqual(Callee.View{ .fn_ref = 42 }, Callee.fnRef(42).view());
    try std.testing.expect(!std.meta.eql(Callee.value_call, Callee{}));
}

test "view round-trips bound_method, and a pending bound method reads as none" {
    try std.testing.expectEqual(Callee.View{ .bound_method = 9 }, Callee.boundMethod(9).view());
    try std.testing.expectEqual(Callee.View.none, Callee.pending_bound.view());
    try std.testing.expect(!std.meta.eql(Callee.pending_bound, Callee.pending));
}

test "pendingCall views as none and keeps its call" {
    const c = Callee.pendingCall(Ast.Index.from(42));
    try std.testing.expectEqual(Callee.View.none, c.view());
    try std.testing.expect(!std.meta.eql(c, Callee.pending));
    try std.testing.expect(!std.meta.eql(c, Callee.pending_bound));
    try std.testing.expectEqual(@as(u32, 42), c.id);
    for ([_]Callee{ c, Callee.pending, Callee.pending_bound }) |p| try std.testing.expect(p.isPending());
    try std.testing.expect(!Callee.method(1).isPending() and !(Callee{}).isPending());
}

test "a template callee views as none and only templateFn reads it" {
    const c = Callee.template(5);
    try std.testing.expectEqual(Callee.View.none, c.view());
    try std.testing.expectEqual(@as(?u32, 5), c.templateFn());
    try std.testing.expectEqual(@as(?u32, null), Callee.method(5).templateFn());
    try std.testing.expectEqual(@as(?u32, null), (Callee{}).templateFn());
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
