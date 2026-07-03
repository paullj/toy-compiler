//! Local type-arg inference matcher (M3) — a peer data module beside `Mono.zig`
//! and `Sig.zig`.
//!
//! This is the SINGLE home for the one-sided structural match that turns a bare
//! (no-explicit-args) generic call `id(7)` into a concrete type-arg tuple. Four
//! consumers run it and MUST agree byte-for-byte on which `Mono.Instance` a bare
//! call selects: Pass-C checking (`BodyChecker`), mono discovery (`types.scanCalls`),
//! lowering (`lower.lowerCall`), and the caller-side fingerprint fold
//! (`AstWalk.CallVisitor`). Keeping the algorithm — and especially the never/invalid
//! skip rule — in ONE place is what stops them drifting.
//!
//! This is the DESIGNATED-DROPPABLE layer of the generics spine: it is pure and
//! allocation-light (`match` allocates nothing), stores no `type_var` anywhere, and
//! is invoked only from small guarded branches, so the M2 monomorphization core stays
//! honorable if M3 is ever removed.
//!
//! NO HM, no persistent unification vars crossing fn boundaries: the matcher runs and
//! is discarded at the call node. First-binding-wins per type-var ordinal (ascending
//! param order); a rebind to a DIFFERENT concrete type is a conflict (naming both
//! source positions); a var left unbound after all params is uninferable.

const std = @import("std");
const Type = @import("../layout/Engine.zig").Type;

pub const Outcome = union(enum) {
    ok,
    /// Type-var `ord` was bound at `first_pos` then matched a different type at
    /// `second_pos` (always the later index; symmetric via `Type.eql`).
    conflict: struct { ord: u32, first_pos: usize, second_pos: usize },
    /// Type-var `ord` was never bound by any param (return-only / uninferable).
    unbound: struct { ord: u32 },
};

/// One-sided match of `template_params` against `arg_types`, filling `out_args`
/// (indexed by type-var ordinal). Assumes `template_params.len == arg_types.len`
/// (the caller arity-checks). `out_args`/`bound`/`first_pos` are caller-owned and
/// `len == generic_param_count`. Allocates nothing.
///
/// The never/invalid skip lives HERE so it cannot drift across the four consumers:
/// a `never` or `invalid` argument binds NOTHING (so a var still bound from another
/// arg still infers, and an already-reported `invalid` never cascades a new error).
pub fn match(
    generic_param_count: u32,
    template_params: []const Type,
    arg_types: []const Type,
    out_args: []Type,
    bound: []bool,
    first_pos: []usize,
) Outcome {
    @memset(bound, false);
    for (template_params, 0..) |p, i| {
        if (!p.isTypeVar()) continue;
        const a = arg_types[i];
        if (a.kind == .never or a.kind == .invalid) continue;
        const ord = p.typeVarOrd();
        if (ord >= generic_param_count) continue; // defensive: an out-of-range ordinal is inert
        if (!bound[ord]) {
            out_args[ord] = a;
            bound[ord] = true;
            first_pos[ord] = i;
        } else if (!Type.eql(out_args[ord], a)) {
            return .{ .conflict = .{ .ord = ord, .first_pos = first_pos[ord], .second_pos = i } };
        }
    }
    for (0..generic_param_count) |ord| {
        if (!bound[ord]) return .{ .unbound = .{ .ord = @intCast(ord) } };
    }
    return .ok;
}

pub const FillOutcome = union(enum) {
    ok,
    /// Type-var `ord` was arg-bound to `arg` but the expected type wants `expected`.
    conflict: struct { ord: u32, arg: Type, expected: Type },
    /// Type-var `ord` is still open after the target-fill (lowest such ordinal).
    unbound: struct { ord: u32 },
};

/// M7 target-fill reconcile, applied AFTER `match` at a CONSTRUCTION check site ONLY
/// (enum/struct). It is NOT one of the four call-discovery consumers of `match`:
/// enum/struct instances are discovered from `node_types` by `reifyApps`, never
/// re-inferred, so filling here cannot drift them — that is why this lives beside
/// `match` yet needs no consumer edit and function-call inference stays untouched.
///
/// Precedence (locked): arg-derived bindings are authoritative. A bound var whose
/// `exp_args` entry disagrees is a `.conflict` (arg wins, but the disagreement is a
/// hard error); an open var with `exp_args` present is filled; `exp_args == null` (no
/// usable target) or an arity mismatch leaves everything as-is. Two passes so the
/// reported `.unbound` is the LOWEST still-open ordinal AFTER all fills, matching
/// `match`'s contract. Allocation-free.
pub fn fillExpected(out: []Type, bound: []bool, exp_args: ?[]const Type) FillOutcome {
    if (exp_args) |ea| if (ea.len == out.len) {
        for (0..out.len) |i| {
            if (bound[i]) {
                if (!Type.eql(out[i], ea[i]))
                    return .{ .conflict = .{ .ord = @intCast(i), .arg = out[i], .expected = ea[i] } };
            } else {
                out[i] = ea[i];
                bound[i] = true;
            }
        }
    };
    for (0..out.len) |i| if (!bound[i]) return .{ .unbound = .{ .ord = @intCast(i) } };
    return .ok;
}

/// Convenience for the two Sig-only reconstruction sites (`lower`, `CallVisitor`),
/// which have the template params but not its generic-param count. The caller passes
/// the authoritative `generic_param_count` (`1 + max typeVarOrd` over the sig's params,
/// which equals `generic_params.len` for any call that survived Pass C — every var of
/// a surviving call is bound, hence appears in a param). Returns the OWNED inferred
/// arg tuple on `.ok`, else `null` (arity mismatch, conflict, or unbound).
pub fn infer(
    gpa: std.mem.Allocator,
    generic_param_count: u32,
    template_params: []const Type,
    arg_types: []const Type,
) !?[]Type {
    if (arg_types.len != template_params.len) return null;
    const out = try gpa.alloc(Type, generic_param_count);
    errdefer gpa.free(out);
    const bound = try gpa.alloc(bool, generic_param_count);
    defer gpa.free(bound);
    const first_pos = try gpa.alloc(usize, generic_param_count);
    defer gpa.free(first_pos);
    switch (match(generic_param_count, template_params, arg_types, out, bound, first_pos)) {
        .ok => return out,
        else => {
            gpa.free(out);
            return null;
        },
    }
}

const testing = std.testing;

/// Run `match` over stack scratch sized to `n` and return the outcome + the filled args.
fn runMatch(n: u32, params: []const Type, args: []const Type) struct { out: Outcome, args: [4]Type } {
    var out_args: [4]Type = undefined;
    var bound: [4]bool = undefined;
    var first_pos: [4]usize = undefined;
    const o = match(n, params, args, out_args[0..n], bound[0..n], first_pos[0..n]);
    return .{ .out = o, .args = out_args };
}

test "first-binding-wins over [T,U]" {
    // snd[T,U](a:T,b:U): T<-bool at 0, U<-int at 1.
    const r = runMatch(2, &.{ Type.typeVar(0), Type.typeVar(1) }, &.{ Type.@"bool", Type.int });
    try testing.expectEqual(Outcome.ok, r.out);
    try testing.expect(Type.eql(r.args[0], Type.@"bool"));
    try testing.expect(Type.eql(r.args[1], Type.int));
}

test "first-binding-wins over [T,T] with agreeing args" {
    const r = runMatch(1, &.{ Type.typeVar(0), Type.typeVar(0) }, &.{ Type.int, Type.int });
    try testing.expectEqual(Outcome.ok, r.out);
    try testing.expect(Type.eql(r.args[0], Type.int));
}

test "conflict returns BOTH source positions" {
    const r = runMatch(1, &.{ Type.typeVar(0), Type.typeVar(0) }, &.{ Type.int, Type.@"bool" });
    switch (r.out) {
        .conflict => |c| {
            try testing.expectEqual(@as(u32, 0), c.ord);
            try testing.expectEqual(@as(usize, 0), c.first_pos);
            try testing.expectEqual(@as(usize, 1), c.second_pos);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "conflict is symmetric — swapping the two args yields the same ord, positions track source order" {
    const r = runMatch(1, &.{ Type.typeVar(0), Type.typeVar(0) }, &.{ Type.@"bool", Type.int });
    switch (r.out) {
        .conflict => |c| {
            try testing.expectEqual(@as(u32, 0), c.ord);
            // second_pos is ALWAYS the later index, whichever order the args come in.
            try testing.expectEqual(@as(usize, 0), c.first_pos);
            try testing.expectEqual(@as(usize, 1), c.second_pos);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "never AND invalid arg types bind nothing (the var still infers from another arg)" {
    // same[T](a:T,b:T): never at 0 skipped, int at 1 binds T=int.
    const r1 = runMatch(1, &.{ Type.typeVar(0), Type.typeVar(0) }, &.{ Type.never, Type.int });
    try testing.expectEqual(Outcome.ok, r1.out);
    try testing.expect(Type.eql(r1.args[0], Type.int));
    // invalid at 0 skipped, int at 1 binds T=int (no cascade).
    const r2 = runMatch(1, &.{ Type.typeVar(0), Type.typeVar(0) }, &.{ Type.invalid, Type.int });
    try testing.expectEqual(Outcome.ok, r2.out);
    try testing.expect(Type.eql(r2.args[0], Type.int));
}

test "never/invalid never conflicts with a concrete binding, regardless of order" {
    // int at 0 binds T, never at 1 is skipped (not a conflict).
    const r = runMatch(1, &.{ Type.typeVar(0), Type.typeVar(0) }, &.{ Type.int, Type.never });
    try testing.expectEqual(Outcome.ok, r.out);
    try testing.expect(Type.eql(r.args[0], Type.int));
}

test "unbound ordinal (return-only) detected" {
    // ro[T]() -> T: no params bind T.
    const r = runMatch(1, &.{}, &.{});
    switch (r.out) {
        .unbound => |u| try testing.expectEqual(@as(u32, 0), u.ord),
        else => return error.TestUnexpectedResult,
    }
}

test "unbound reports the LOWEST still-open ordinal" {
    // [T,U](a:U): T (ord 0) unbound; U (ord 1) bound — lowest-unbound is T.
    const r = runMatch(2, &.{Type.typeVar(1)}, &.{Type.int});
    switch (r.out) {
        .unbound => |u| try testing.expectEqual(@as(u32, 0), u.ord),
        else => return error.TestUnexpectedResult,
    }
}

test "infer returns owned args on ok" {
    const gpa = testing.allocator;
    const got = (try infer(gpa, 2, &.{ Type.typeVar(0), Type.typeVar(1) }, &.{ Type.int, Type.@"bool" })).?;
    defer gpa.free(got);
    try testing.expect(Type.eql(got[0], Type.int));
    try testing.expect(Type.eql(got[1], Type.@"bool"));
}

test "infer returns null on arity mismatch and on conflict" {
    const gpa = testing.allocator;
    try testing.expectEqual(@as(?[]Type, null), try infer(gpa, 1, &.{Type.typeVar(0)}, &.{ Type.int, Type.int }));
    try testing.expectEqual(@as(?[]Type, null), try infer(gpa, 1, &.{ Type.typeVar(0), Type.typeVar(0) }, &.{ Type.int, Type.@"bool" }));
}

/// Run `fillExpected` over stack scratch seeded from `out_init`/`bound_init`.
fn runFill(out_init: []const Type, bound_init: []const bool, exp: ?[]const Type) struct { out: FillOutcome, args: [4]Type } {
    var out_args: [4]Type = undefined;
    var bnd: [4]bool = undefined;
    for (out_init, 0..) |t, i| out_args[i] = t;
    for (bound_init, 0..) |b, i| bnd[i] = b;
    const n = out_init.len;
    const o = fillExpected(out_args[0..n], bnd[0..n], exp);
    return .{ .out = o, .args = out_args };
}

test "fillExpected fills a still-open var from the target" {
    const r = runFill(&.{Type.invalid}, &.{false}, &.{Type.int});
    try testing.expectEqual(FillOutcome.ok, r.out);
    try testing.expect(Type.eql(r.args[0], Type.int));
}

test "fillExpected: an arg-bound var that AGREES with the target is no conflict" {
    const r = runFill(&.{Type.int}, &.{true}, &.{Type.int});
    try testing.expectEqual(FillOutcome.ok, r.out);
}

test "fillExpected: an arg-bound var that DISAGREES with the target is a conflict" {
    const r = runFill(&.{Type.@"bool"}, &.{true}, &.{Type.int});
    switch (r.out) {
        .conflict => |c| {
            try testing.expectEqual(@as(u32, 0), c.ord);
            try testing.expect(Type.eql(c.arg, Type.@"bool"));
            try testing.expect(Type.eql(c.expected, Type.int));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "fillExpected: null target leaves an open var unbound" {
    const r = runFill(&.{Type.invalid}, &.{false}, null);
    switch (r.out) {
        .unbound => |u| try testing.expectEqual(@as(u32, 0), u.ord),
        else => return error.TestUnexpectedResult,
    }
}

test "fillExpected: null target with all vars already bound is ok" {
    const r = runFill(&.{Type.int}, &.{true}, null);
    try testing.expectEqual(FillOutcome.ok, r.out);
}

test "fillExpected: two arg-bound vars agreeing with the target is ok" {
    const r = runFill(&.{ Type.int, Type.@"bool" }, &.{ true, true }, &.{ Type.int, Type.@"bool" });
    try testing.expectEqual(FillOutcome.ok, r.out);
}

test "fillExpected: partial — one arg-bound, one filled from the target" {
    const r = runFill(&.{ Type.int, Type.invalid }, &.{ true, false }, &.{ Type.int, Type.@"bool" });
    try testing.expectEqual(FillOutcome.ok, r.out);
    try testing.expect(Type.eql(r.args[1], Type.@"bool"));
}

test "fillExpected: an agreeing ord0 does not mask a conflicting ord1" {
    const r = runFill(&.{ Type.@"bool", Type.@"bool" }, &.{ true, true }, &.{ Type.@"bool", Type.int });
    switch (r.out) {
        .conflict => |c| try testing.expectEqual(@as(u32, 1), c.ord),
        else => return error.TestUnexpectedResult,
    }
}

test "fillExpected: an arity mismatch is treated as no target (no-op)" {
    const r = runFill(&.{Type.invalid}, &.{false}, &.{ Type.int, Type.@"bool" });
    switch (r.out) {
        .unbound => |u| try testing.expectEqual(@as(u32, 0), u.ord),
        else => return error.TestUnexpectedResult,
    }
}
