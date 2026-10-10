//! How a function takes each parameter: by value, or by address (`mut`). A peer data
//! module: lower and Spec ask `argMask` which VALUE args a call passes by address, so the
//! two cannot disagree on them. A receiver's mode is not in that mask: lowerCall reads it
//! off `Method.mut_self`, and the checker off its own `CallSig`. A mask is one `u64`, bit
//! `i` for param `i`; a method's `self` is param 0.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Token = @import("../ast/Token.zig").Token;
const Callee = @import("Callee.zig").Callee;
const Body = @import("Body.zig");

/// The parser rejects `mut` past this many params, so every mode fits one mask (Spec's
/// eligibility masks are 64-bit for the same reason).
pub const max_params: u32 = 64;

/// Whether bit `i` of `mask` is set: param (or value arg) `i` is passed by address.
pub fn isMut(mask: u64, i: usize) bool {
    return i < max_params and mask & (@as(u64, 1) << @intCast(i)) != 0;
}

/// The mode mask of a fn's own declaration, one bit per `mut` param. Read once per fn,
/// when its signature is decoded (`FnSym.mut_mask`, copied to `Sig.mut_mask` and each
/// body's sig); the resolver, which runs before any signature exists, reads it too.
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
pub fn argMask(body: Body.BodyView, prog: Body.ProgramView, call: Ast.Index) u64 {
    const tree = body.tree;
    const n = tree.nodes[call.int()];
    const slot = Callee.callSlot(tree, n) orelse return 0;
    if (slot.int() >= body.callees.len) return 0;
    return switch (body.callees[slot.int()].view()) {
        .method => |mi| {
            if (mi >= prog.methods.len) return 0;
            const m = prog.methods[mi];
            // A reified generic-impl row's `fn_id` is its template's, whose modes hold for
            // every instance (they are syntactic).
            if (m.derive != null or m.fn_id >= prog.sigs.len) return 0;
            const mask = prog.sigs[m.fn_id].mut_mask;
            return if (m.is_static) mask else mask >> 1;
        },
        .direct, .generic => |gid| if (gid < prog.sigs.len) prog.sigs[gid].mut_mask else 0,
        // An instance's modes are its template's.
        .instance => |ii| if (ii < prog.instances.len and prog.instances[ii].template_gid < prog.sigs.len) prog.sigs[prog.instances[ii].template_gid].mut_mask else 0,
        .none, .protocol, .builtin, .value, .fn_ref, .bound_method => 0,
    };
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
