//! Local type-arg inference matcher — a peer data module beside `Mono.zig`
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
//! is invoked only from small guarded branches, so the monomorphization core stays
//! honorable if it is ever removed.
//!
//! NO HM, no persistent unification vars crossing fn boundaries: the matcher runs and
//! is discarded at the call node. First-binding-wins per type-var ordinal (ascending
//! param order); a rebind to a DIFFERENT concrete type is a conflict (naming both
//! source positions); a var left unbound after all params is uninferable.

const std = @import("std");
const Type = @import("../layout/Type.zig").Type;

pub const Outcome = union(enum) {
    ok,
    /// Type-var `ord` was bound at `first_pos` then matched a different type at
    /// `second_pos` (always the later index; symmetric via `Type.eql`). `prev`/`cur`
    /// are the two clashing LEAF types — the descended element leaves for a
    /// container clash (e.g. `int` vs `bool` inside `Vec[int]`/`Vec[bool]`), NOT the
    /// whole `Vec[..]` — so callers name the real conflict, not the container ctor.
    conflict: struct { ord: u32, first_pos: usize, second_pos: usize, prev: Type, cur: Type },
    /// Type-var `ord` was never bound by any param (return-only / uninferable).
    unbound: struct { ord: u32 },
};

/// One decomposed `App`: its ctor id, whether that id lives in the enum id space,
/// and its type-args. Type-erased so this leaf never learns the `Composite` table.
pub const Decomposed = struct { ctor: u32, is_enum: bool, args: []const Type };

/// A type-erased hook that turns an `App` `Type` into its `(ctor, is_enum, args)` —
/// the ONLY way this pure leaf can see through a container param (`Vec[T]`) to its
/// type-var args. The composite table lives in the checker, so it is injected rather
/// than imported: `Infer` still depends on nothing but `std` + `Type`. A caller with
/// no composite in scope (the find-regime consumers) passes `null`, and structural
/// descent degrades to the flat rule.
pub const Decomposer = struct {
    ctx: *anyopaque,
    func: *const fn (*anyopaque, Type) ?Decomposed,
    inline fn call(d: Decomposer, ty: Type) ?Decomposed {
        return d.func(d.ctx, ty);
    }
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
    dec: ?Decomposer,
) Outcome {
    @memset(bound, false);
    for (template_params, 0..) |p, i| switch (unifyOne(p, arg_types[i], i, generic_param_count, out_args, bound, first_pos, dec)) {
        .ok => {},
        else => |o| return o,
    };
    for (0..generic_param_count) |ord| {
        if (!bound[ord]) return .{ .unbound = .{ .ord = @intCast(ord) } };
    }
    return .ok;
}

/// Unify one template param `p` against one arg `a`. `pos` is the SOURCE arg index
/// `i` from `match`'s top-level loop, threaded UNCHANGED through the structural
/// descent so a conflict caret pair always names the two source args (the T0015
/// span the callers map with `args[first_pos]`/`args[second_pos]`), even when the
/// clash is inside a container.
///
/// Structural descent (`p` an `App`) is recursive: it needs a `dec` to see through a
/// container, and terminates because interned `App`s are a finite, acyclic,
/// depth-bounded (T0017) structure. With `dec == null` an `.app` param binds nothing
/// (the pre-descent behavior — a container param left its var unbound), so passing
/// `null` reproduces the flat matcher byte-for-byte.
fn unifyOne(p: Type, a: Type, pos: usize, gpc: u32, out: []Type, bound: []bool, first_pos: []usize, dec: ?Decomposer) Outcome {
    // The never/invalid skip, at EVERY level: a `never`/`invalid` arg (or arg slot,
    // e.g. `Vec[never]` from an empty-literal arg) binds nothing and never cascades.
    if (a.kind == .never or a.kind == .invalid) return .ok;
    if (p.isTypeVar()) {
        const ord = p.typeVarOrd();
        if (ord >= gpc) return .ok; // defensive: an out-of-range ordinal is inert
        if (!bound[ord]) {
            out[ord] = a;
            bound[ord] = true;
            first_pos[ord] = pos;
        } else if (!Type.eql(out[ord], a)) {
            return .{ .conflict = .{ .ord = ord, .first_pos = first_pos[ord], .second_pos = pos, .prev = out[ord], .cur = a } };
        }
        return .ok;
    }
    if (p.isApp()) {
        const d = dec orelse return .ok; // no composite ⇒ can't descend; binds nothing
        const pe = d.call(p) orelse return .ok;
        const ae = d.call(a) orelse return .ok; // arg not an App ⇒ no structural match
        // Same ctor, same id-space, same arity — else these are unrelated containers
        // and there is nothing sound to bind (a genuine arg mismatch is diagnosed
        // elsewhere; here we only fill still-open vars).
        if (pe.ctor != ae.ctor or pe.is_enum != ae.is_enum or pe.args.len != ae.args.len) return .ok;
        for (pe.args, ae.args) |pk, ak| switch (unifyOne(pk, ak, pos, gpc, out, bound, first_pos, dec)) {
            .ok => {},
            else => |o| return o,
        };
        return .ok;
    }
    return .ok; // a concrete param binds nothing
}

pub const FillOutcome = union(enum) {
    ok,
    /// Type-var `ord` was arg-bound to `arg` but the expected type wants `expected`.
    conflict: struct { ord: u32, arg: Type, expected: Type },
    /// Type-var `ord` is still open after the target-fill (lowest such ordinal).
    unbound: struct { ord: u32 },
};

/// Target-fill reconcile, applied AFTER `match` at a CONSTRUCTION check site ONLY
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
    dec: ?Decomposer,
) !?[]Type {
    if (arg_types.len != template_params.len) return null;
    const out = try gpa.alloc(Type, generic_param_count);
    errdefer gpa.free(out);
    const bound = try gpa.alloc(bool, generic_param_count);
    defer gpa.free(bound);
    const first_pos = try gpa.alloc(usize, generic_param_count);
    defer gpa.free(first_pos);
    switch (match(generic_param_count, template_params, arg_types, out, bound, first_pos, dec)) {
        .ok => return out,
        else => {
            gpa.free(out);
            return null;
        },
    }
}

const testing = std.testing;

const MatchResult = struct { out: Outcome, args: [4]Type };

/// Run `match` over stack scratch sized to `n` and return the outcome + the filled
/// args. `dec` defaults to `null` (the flat matcher); structural tests pass a fake.
fn runMatch(n: u32, params: []const Type, args: []const Type) MatchResult {
    return runMatchDec(n, params, args, null);
}

fn runMatchDec(n: u32, params: []const Type, args: []const Type, dec: ?Decomposer) MatchResult {
    var out_args: [4]Type = undefined;
    var bound: [4]bool = undefined;
    var first_pos: [4]usize = undefined;
    const o = match(n, params, args, out_args[0..n], bound[0..n], first_pos[0..n], dec);
    return .{ .out = o, .args = out_args };
}

/// A test-only decomposer backed by a static `(app-index -> Decomposed)` table, so
/// the structural-descent tests exercise `unifyOne` WITHOUT importing `Composite`
/// (the leaf stays pure). A `Type` is treated as an `App` iff its `appIdx` indexes
/// the table.
const FakeComposite = struct {
    entries: []const Decomposed,
    fn thunk(ctx: *anyopaque, ty: Type) ?Decomposed {
        if (!ty.isApp()) return null;
        const self: *const FakeComposite = @ptrCast(@alignCast(ctx));
        const idx = ty.appIdx();
        if (idx >= self.entries.len) return null;
        return self.entries[idx];
    }
    fn decomposer(self: *FakeComposite) Decomposer {
        return .{ .ctx = self, .func = thunk };
    }
};

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
            // For a flat param the leaf IS the whole arg.
            try testing.expect(Type.eql(c.prev, Type.int));
            try testing.expect(Type.eql(c.cur, Type.@"bool"));
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

/// A fixed table of `App`s exercised by the structural-descent tests. Ctors: 1=`Vec`,
/// 2=`Map`, 3 as a struct-App AND (separately) as an enum-App to prove the `is_enum`
/// guard. Nested entries (5,6) reference inner-App indices via `Type.app`.
const fake_entries = [_]Decomposed{
    .{ .ctor = 1, .is_enum = false, .args = &.{Type.typeVar(0)} }, // 0: Vec[T]
    .{ .ctor = 1, .is_enum = false, .args = &.{Type.int} }, // 1: Vec[int]
    .{ .ctor = 1, .is_enum = false, .args = &.{Type.@"bool"} }, // 2: Vec[bool]
    .{ .ctor = 2, .is_enum = false, .args = &.{ Type.typeVar(0), Type.typeVar(1) } }, // 3: Map[K,V]
    .{ .ctor = 2, .is_enum = false, .args = &.{ Type.str, Type.int } }, // 4: Map[str,int]
    .{ .ctor = 1, .is_enum = false, .args = &.{Type.app(0)} }, // 5: Vec[Vec[T]]
    .{ .ctor = 1, .is_enum = false, .args = &.{Type.app(1)} }, // 6: Vec[Vec[int]]
    .{ .ctor = 3, .is_enum = true, .args = &.{Type.int} }, // 7: enum E3[int]
    .{ .ctor = 2, .is_enum = false, .args = &.{ Type.typeVar(0), Type.typeVar(0) } }, // 8: Map[T,T]
    .{ .ctor = 2, .is_enum = false, .args = &.{ Type.int, Type.@"bool" } }, // 9: Map[int,bool]
    .{ .ctor = 3, .is_enum = false, .args = &.{Type.typeVar(0)} }, // 10: struct S3[T]
    .{ .ctor = 2, .is_enum = false, .args = &.{Type.typeVar(0)} }, // 11: bad-arity Map[T]
    .{ .ctor = 1, .is_enum = false, .args = &.{Type.never} }, // 12: Vec[never]
};

fn runStructMatch(n: u32, params: []const Type, args: []const Type) MatchResult {
    var fc = FakeComposite{ .entries = &fake_entries };
    return runMatchDec(n, params, args, fc.decomposer());
}

test "structural: Vec[T] param vs Vec[int] arg binds T=int" {
    const r = runStructMatch(1, &.{Type.app(0)}, &.{Type.app(1)});
    try testing.expectEqual(Outcome.ok, r.out);
    try testing.expect(Type.eql(r.args[0], Type.int));
}

test "structural: Map[K,V] vs Map[str,int] binds both" {
    const r = runStructMatch(2, &.{Type.app(3)}, &.{Type.app(4)});
    try testing.expectEqual(Outcome.ok, r.out);
    try testing.expect(Type.eql(r.args[0], Type.str));
    try testing.expect(Type.eql(r.args[1], Type.int));
}

test "structural: recursion — Vec[Vec[T]] vs Vec[Vec[int]] binds T=int" {
    const r = runStructMatch(1, &.{Type.app(5)}, &.{Type.app(6)});
    try testing.expectEqual(Outcome.ok, r.out);
    try testing.expect(Type.eql(r.args[0], Type.int));
}

test "structural: [Vec[T],Vec[T]] vs [Vec[int],Vec[bool]] conflicts at the two SOURCE args" {
    const r = runStructMatch(1, &.{ Type.app(0), Type.app(0) }, &.{ Type.app(1), Type.app(2) });
    switch (r.out) {
        .conflict => |c| {
            try testing.expectEqual(@as(u32, 0), c.ord);
            try testing.expectEqual(@as(usize, 0), c.first_pos);
            try testing.expectEqual(@as(usize, 1), c.second_pos);
            // The clashing LEAF types are the descended container elements, not `Vec[..]`.
            try testing.expect(Type.eql(c.prev, Type.int));
            try testing.expect(Type.eql(c.cur, Type.@"bool"));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "structural: intra-arg conflict [Map[T,T]] vs [Map[int,bool]] carets the single arg" {
    const r = runStructMatch(1, &.{Type.app(8)}, &.{Type.app(9)});
    switch (r.out) {
        .conflict => |c| {
            try testing.expectEqual(@as(u32, 0), c.ord);
            // Both positions are the single source arg (no synthetic sub-arg spans).
            try testing.expectEqual(@as(usize, 0), c.first_pos);
            try testing.expectEqual(@as(usize, 0), c.second_pos);
            try testing.expect(Type.eql(c.prev, Type.int));
            try testing.expect(Type.eql(c.cur, Type.@"bool"));
        },
        else => return error.TestUnexpectedResult,
    }
}

test "structural guard: a different ctor binds nothing (var stays unbound)" {
    const r = runStructMatch(1, &.{Type.app(0)}, &.{Type.app(7)}); // Vec[T] vs enum E3[int]
    switch (r.out) {
        .unbound => |u| try testing.expectEqual(@as(u32, 0), u.ord),
        else => return error.TestUnexpectedResult,
    }
}

test "structural guard: struct-App vs enum-App with the SAME ctor binds nothing" {
    // struct S3[T] (10) vs enum E3[int] (7): same ctor id 3, differing `is_enum`.
    const r = runStructMatch(1, &.{Type.app(10)}, &.{Type.app(7)});
    switch (r.out) {
        .unbound => |u| try testing.expectEqual(@as(u32, 0), u.ord),
        else => return error.TestUnexpectedResult,
    }
}

test "structural guard: differing arity binds nothing" {
    const r = runStructMatch(1, &.{Type.app(11)}, &.{Type.app(9)}); // Map[T] vs Map[int,bool]
    switch (r.out) {
        .unbound => |u| try testing.expectEqual(@as(u32, 0), u.ord),
        else => return error.TestUnexpectedResult,
    }
}

test "structural guard: a non-App arg against an App param binds nothing" {
    const r = runStructMatch(1, &.{Type.app(0)}, &.{Type.int}); // Vec[T] vs int
    switch (r.out) {
        .unbound => |u| try testing.expectEqual(@as(u32, 0), u.ord),
        else => return error.TestUnexpectedResult,
    }
}

test "structural: a never element skips at the nested level (var stays unbound)" {
    const r = runStructMatch(1, &.{Type.app(0)}, &.{Type.app(12)}); // Vec[T] vs Vec[never]
    switch (r.out) {
        .unbound => |u| try testing.expectEqual(@as(u32, 0), u.ord),
        else => return error.TestUnexpectedResult,
    }
}

test "structural: dec==null reproduces the flat matcher (App param binds nothing)" {
    const r = runMatchDec(1, &.{Type.app(0)}, &.{Type.app(1)}, null);
    switch (r.out) {
        .unbound => |u| try testing.expectEqual(@as(u32, 0), u.ord),
        else => return error.TestUnexpectedResult,
    }
}

test "infer returns owned args on ok" {
    const gpa = testing.allocator;
    const got = (try infer(gpa, 2, &.{ Type.typeVar(0), Type.typeVar(1) }, &.{ Type.int, Type.@"bool" }, null)).?;
    defer gpa.free(got);
    try testing.expect(Type.eql(got[0], Type.int));
    try testing.expect(Type.eql(got[1], Type.@"bool"));
}

test "infer returns null on arity mismatch and on conflict" {
    const gpa = testing.allocator;
    try testing.expectEqual(@as(?[]Type, null), try infer(gpa, 1, &.{Type.typeVar(0)}, &.{ Type.int, Type.int }, null));
    try testing.expectEqual(@as(?[]Type, null), try infer(gpa, 1, &.{ Type.typeVar(0), Type.typeVar(0) }, &.{ Type.int, Type.@"bool" }, null));
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
