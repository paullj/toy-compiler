//! How a function takes each parameter: by value, or by address (`mut`). A peer data
//! module: lower and Spec ask `argMask` which VALUE args a call passes by address, so the
//! two cannot disagree on them. A receiver's mode is not in that mask: lowerCall reads it
//! off `Method.mut_self`, and the checker off its own `CallSig`. A mask is one `u64`, bit
//! `i` for param `i`; a method's `self` is param 0.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Token = @import("../ast/Token.zig").Token;
const Callee = @import("Callee.zig").Callee;

/// The parser rejects `mut` past this many params, so every mode fits one mask (Spec's
/// eligibility masks are 64-bit for the same reason).
pub const max_params: u32 = 64;

/// Whether bit `i` of `mask` is set: param (or value arg) `i` is passed by address.
pub fn isMut(mask: u64, i: usize) bool {
    return i < max_params and mask & (@as(u64, 1) << @intCast(i)) != 0;
}

/// The mode mask of a fn's own declaration, one bit per `mut` param.
pub fn ofProto(tree: Ast.Tree, tokens: []const Token, proto: Ast.FnProto) u64 {
    var mask: u64 = 0;
    for (proto.params[0..@min(proto.params.len, max_params)], 0..) |p, i| {
        if (Ast.isMutParam(tree, tokens, p)) mask |= @as(u64, 1) << @intCast(i);
    }
    return mask;
}

/// Which VALUE args of `call` its statically known callee takes by address (bit `i` =
/// value arg `i`, so a method's receiver bit is shifted out). It mirrors the callee
/// selection `lowerCall` makes; a value call, a builtin and a protocol call carry no
/// modes (the checker keeps `mut` off every such path).
///
/// `view` supplies `.tree`, `.resolutions`, `.callees`, `.sigs` and `.methods`.
pub fn argMask(view: anytype, call: Ast.Index) u64 {
    const tree = view.tree;
    const n = tree.nodes[call.int()];
    const slot = Callee.callSlot(tree, n) orelse return 0;
    if (slot.int() >= view.callees.len) return 0;
    switch (view.callees[slot.int()].view()) {
        .method => |mi| {
            if (mi >= view.methods.len) return 0;
            const m = view.methods[mi];
            // A reified generic-impl row's `fn_id` is its template's, whose modes hold for
            // every instance (they are syntactic).
            if (m.derive != null or m.fn_id >= view.sigs.len) return 0;
            const mask = view.sigs[m.fn_id].mut_mask;
            return if (m.is_static) mask else mask >> 1;
        },
        .none => {
            const head_node = tree.nodes[n.lhs.int()];
            const head = if (head_node.tag == .type_app) head_node.lhs else n.lhs;
            if (head == Ast.none or head.int() >= view.resolutions.len) return 0;
            return switch (view.resolutions[head.int()]) {
                .func => |gid| if (gid < view.sigs.len) view.sigs[gid].mut_mask else 0,
                else => 0,
            };
        },
        .protocol, .builtin, .value, .fn_ref, .bound_method => return 0,
    }
}

const testing = std.testing;

test "ofProto sets one bit per mut param, the receiver first" {
    const gpa = testing.allocator;
    const src =
        \\struct P { x: int }
        \\impl P {
        \\    fn m(mut self, a: int, mut d: int) { self.x = a + d }
        \\    fn r(self, mut d: int) -> int { self.x + d }
        \\}
        \\fn g(a: int, mut b: int) { b = a }
        \\fn h(a: int) -> int { a }
        \\
    ;
    const tokens = try @import("../lex.zig").tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try @import("../parse.zig").expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    const want = [_]u64{ 0b101, 0b10, 0b10, 0 };
    var seen: usize = 0;
    for (tree.nodes) |n| {
        if (n.tag != .fn_decl) continue;
        try testing.expectEqual(want[seen], ofProto(tree, tokens, Ast.protoAt(tree, n.lhs.int())));
        seen += 1;
    }
    try testing.expectEqual(want.len, seen);
}

test "isMut is false at and beyond 64" {
    const all: u64 = std.math.maxInt(u64);
    try testing.expect(isMut(all, 0));
    try testing.expect(isMut(all, 63));
    try testing.expect(!isMut(all, 64));
    try testing.expect(!isMut(all, 1000));
    try testing.expect(!isMut(0b10, 0));
    try testing.expect(isMut(0b10, 1));
}
