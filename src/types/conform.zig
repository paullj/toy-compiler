//! The conformance DECISION, deepened into one module: "does ground/abstract type
//! `T` satisfy protocol `P`, and via which tier". Extracted from the `Typecheck`
//! mega-struct so the existence check, the type_var bound-as-axiom rule, the composed
//! operator verdict, the recursive structural engine, and the derive-blocker leaf live
//! behind ONE small interface instead of being re-derived at each caller. The witness
//! PICK (`resolveConformanceMethod`) stays in the parent as the shared leaf all three
//! dispatch consumers (checker, lower, fingerprint) already agree through — this module
//! decides conformance, it does not pick the method.
//!
//! Imports the parent as `Typecheck` (the `src/types/` submodule convention shared with
//! `derive_synth.zig`/`coherence.zig`/`reify.zig`).

const std = @import("std");
const Typecheck = @import("../types.zig");
const LayoutEngine = @import("../layout/Engine.zig");
const Composite = @import("../symbols/Composite.zig");
const StdNames = @import("../symbols/StdNames.zig");

const Type = Typecheck.Type;
const Model = Typecheck.Model;
const Conformance = Typecheck.Conformance;
const NonConformingField = Typecheck.NonConformingField;
const StructSym = LayoutEngine.StructSym;
const EnumSym = LayoutEngine.EnumSym;
const eqlTypeVec = Typecheck.eqlTypeVec;

/// Whether the ground type `recv` conforms to protocol `pid`. A pure,
/// thread-order-free linear scan over the frozen `model.conformances` by protocol id
/// + `Type.eql` — the SAME determinism discipline `findMethod` uses, so bound
/// resolution at the mono worklist is byte-identical at any `-jN`. A generic-App
/// type-arg necessarily misses (conformances only record concrete `structT`/`enumT`/
/// scalar receivers — `impl Box[T] has P` is rejected at parse), which is the correct
/// "not conforming" answer.
pub fn existence(model: *const Model, pid: u32, recv: Type, protocol_args: []const Type) bool {
    for (model.conformances) |c| {
        // Exact match first (a user `impl int8 has P` row, or any concrete receiver).
        if (c.protocol == pid and Type.eql(c.recv, recv) and eqlTypeVec(c.protocol_args, protocol_args)) return true;
        // Then the integer-width normalization (see `intMatchesPlatformRow`): keeps the
        // row table minimal (no per-width rows).
        if (intMatchesPlatformRow(recv, c.recv) and c.protocol == pid and eqlTypeVec(c.protocol_args, protocol_args)) return true;
    }
    return false;
}

/// The builtin-protocol integer-width normalization: a width receiver (`int8`/`uint32`/…)
/// matches the canonical platform `Type.int` conformance row. The builtin `conf` rows list
/// only `Type.int`, and every width shares one uniform builtin-protocol verdict (Eq/Ord/
/// Hash/Display/Add/…), so a width conforms exactly where `int` does. Single-sourced across
/// `existence` and `conformsRec`'s scalar arm so the two verdicts cannot drift — the
/// memo-key soundness argument (a width collapses to the `(pid,.int,…)` slot) rests on it.
fn intMatchesPlatformRow(recv: Type, row_recv: Type) bool {
    return recv.isInteger() and Type.eql(row_recv, Type.int);
}

/// The type_var bound-as-axiom rule, single-sourced. Inside a bounded template's
/// definition check a `type_var(ord)` conforms to `pid` iff its declared bound IS `pid`.
/// O(1), never memoized. A non-type_var `recv`, an out-of-range ordinal, or an unbounded
/// (`null`) slot all deny — so a stray type_var never spuriously conforms and this is safe
/// to fold into `direct` alongside `existence`. This is the ONE copy of a rule formerly
/// written verbatim in `conformsRec`, `conformsTo`, and `conformsToArith`.
pub fn axiom(recv: Type, pid: u32, bound_protocols: []const ?u32) bool {
    if (!recv.isTypeVar()) return false;
    const ord = recv.typeVarOrd();
    if (ord >= bound_protocols.len) return false;
    return (bound_protocols[ord] orelse return false) == pid;
}

/// The DIRECT verdict: an existing (explicit/prelude/refinement) conformance OR the
/// type_var bound-as-axiom rule. Non-fallible, no memo/alloc. This is exactly the arith
/// operator predicate and the first two tiers of the derivable-operator predicate.
pub fn direct(model: *const Model, recv: Type, pid: u32, bound_protocols: []const ?u32) bool {
    if (existence(model, pid, recv, &.{})) return true;
    return axiom(recv, pid, bound_protocols);
}

/// Which tier of the composed operator verdict fired. `.structural` is the ONLY tier
/// that carries a derive side-effect obligation (the caller records the request), so the
/// caller gates on it without re-running the decision.
pub const Tier = enum { none, direct, structural };

/// The composed operator-gating verdict for the STRUCTURALLY-DERIVABLE protocols (Eq/Ord/
/// Hash/Display): `direct` (existence ∨ axiom) FIRST, then — only when `recv` is an
/// aggregate — the recursive structural engine. The arith/derivable split is which-fn-you-
/// call, not a flag: arithmetic protocols (Add/Sub/…) are NOT structurally derivable (a
/// struct of Add-conforming fields must NOT type-check for `+`), so their caller
/// (`conformsToArith`) calls `direct` directly and never reaches the structural tier. `direct`-
/// first keeps an explicit/refinement conformance off the structural path (no double-fire).
/// Fallible only through `structural` (its memo/subst allocate).
pub fn classify(
    model: *const Model,
    composite: *Composite,
    bound_protocols: []const ?u32,
    memo: *std.AutoHashMapUnmanaged(u64, bool),
    gpa: std.mem.Allocator,
    recv: Type,
    pid: u32,
) error{OutOfMemory}!Tier {
    if (direct(model, recv, pid, bound_protocols)) return .direct;
    switch (recv.kind) {
        .@"struct", .@"enum", .app => {},
        else => return .none,
    }
    if (try structural(model.structs, model.enums, model.conformances, recv, pid, memo, gpa, composite, bound_protocols))
        return .structural;
    return .none;
}

/// Whether the GROUND type `recv` conforms to protocol `pid`, recursively — the
/// shared substrate for on-demand structural derive. `existence`-FIRST (an
/// explicit `impl`, a prelude scalar, or the Ord-refinement `(Eq,T)` entry means it
/// already conforms — the double-fire guard), THEN structural: a struct conforms iff
/// EVERY field conforms; an enum conforms iff EVERY variant's payload fields conform
/// (pid-parameterized, so it powers payload-enum `Eq` AND payload-enum/struct `Ord`;
/// an empty-payload enum trivially conforms since there are no payload fields to check).
///
/// Memoized by `(pid, kind, type-id)` in the caller-owned `memo` (thread-local in Pass
/// C, local in the synthesis barrier — never shared, so race-free). `memo` caches only
/// SETTLED verdicts, so conformance is a pure function of source. NOTE the memo key
/// omits the integer sign/width byte, so every integer width collapses to one
/// `(pid, .int, no_struct)` slot — this is BENIGN because a scalar `.int` receiver
/// returns in the `else` arm (via `Type.int`-normalization) BEFORE the memo is written,
/// and every width shares one uniform builtin-protocol verdict, so the collision is
/// absent rather than wrong (pinned by the uniform-verdict test). A generic template may
/// be self- OR mutually-recursive (`next: Node[T]`, `A[T]`↔`B[T]`), so termination rides
/// a per-query in-progress stack that coinductively assumes conformance on a back-edge;
/// see `conformsRec` for why an assumed verdict is never written back to `memo`.
pub fn structural(
    structs: []const StructSym,
    enums: []const EnumSym,
    conformances: []const Conformance,
    recv: Type,
    pid: u32,
    memo: *std.AutoHashMapUnmanaged(u64, bool),
    gpa: std.mem.Allocator,
    composite: *Composite,
    bound_protocols: []const ?u32,
) error{OutOfMemory}!bool {
    var stack: std.AutoHashMapUnmanaged(u64, u32) = .empty;
    defer stack.deinit(gpa);
    const r = try conformsRec(structs, enums, conformances, recv, pid, memo, &stack, 0, gpa, composite, bound_protocols);
    return r.ok;
}

/// A depth of `no_assumption` in `ConfStep.low` means the verdict rests on no live
/// coinductive assumption, so it is safe to settle into `memo`.
const no_assumption: u32 = std.math.maxInt(u32);

/// A `conformsRec` verdict plus `low`: the shallowest recursion depth of an in-progress
/// (coinductively-assumed) node the verdict leaned on, or `no_assumption` if it leaned on
/// none. A node settles its `memo` entry only when `low >= own_depth` — Tarjan-style, it
/// is then the node that closed the cycle, so no STRICT-ancestor assumption is still live
/// and the verdict cannot change. Members deeper in a multi-node cycle carry `low` from a
/// shallower ancestor, stay unsettled, and are recomputed on demand instead of caching a
/// verdict that was true only under the entry's assumption (the poisoning bug).
const ConfStep = struct { ok: bool, low: u32 };

fn conformsRec(
    structs: []const StructSym,
    enums: []const EnumSym,
    conformances: []const Conformance,
    recv: Type,
    pid: u32,
    memo: *std.AutoHashMapUnmanaged(u64, bool),
    stack: *std.AutoHashMapUnmanaged(u64, u32),
    depth: u32,
    gpa: std.mem.Allocator,
    composite: *Composite,
    bound_protocols: []const ?u32,
) error{OutOfMemory}!ConfStep {
    switch (recv.kind) {
        .@"struct", .@"enum" => {},
        .type_var => {
            // Conditional conformance, the bound-as-axiom leaf: inside a bounded
            // template's definition check `App(G,[T])` reduces the pattern to `T`, which
            // conforms iff its declared bound IS `pid`. O(1) — never memoized. Empty
            // `bound_protocols` (a non-generic/unbounded context) denies, so a stray
            // `type_var` never spuriously conforms.
            return .{ .ok = axiom(recv, pid, bound_protocols), .low = no_assumption };
        },
        .app => {
            // Conditional conformance, the recursive step: `App(G,[args])` conforms
            // iff every ctor field / enum-variant payload PATTERN, with `args` substituted
            // in, conforms. Memoized by the interned App index (a kind byte distinct from
            // struct/enum) so nested `Box[Box[Point]]` is not re-queried exponentially.
            const ai = recv.appIdx();
            const akey: u64 = (@as(u64, pid) << 40) | (@as(u64, @intFromEnum(recv.kind)) << 32) | @as(u64, ai);
            if (memo.get(akey)) |v| return .{ .ok = v, .low = no_assumption };
            // Back-edge onto a node still being computed (`A[T]` reached again via `B[T]`):
            // coinductively assume conformance, tagging the verdict with THAT node's depth
            // so every node between here and it learns its answer was assumption-dependent.
            if (stack.get(akey)) |d| return .{ .ok = true, .low = d };
            try stack.put(gpa, akey, depth);
            const e = composite.at(ai);
            var aok = true;
            var low: u32 = no_assumption;
            if (e.ctor_is_enum) {
                if (e.ctor < enums.len) {
                    outer: for (enums[e.ctor].variants) |v| {
                        for (v.field_types) |ft| {
                            const sub = try substPattern(composite, gpa, ft, e.args);
                            const r = try conformsRec(structs, enums, conformances, sub, pid, memo, stack, depth + 1, gpa, composite, bound_protocols);
                            if (r.low < low) low = r.low;
                            if (!r.ok) {
                                aok = false;
                                break :outer;
                            }
                        }
                    }
                } else aok = false;
            } else {
                if (e.ctor < structs.len) {
                    for (structs[e.ctor].field_types) |ft| {
                        const sub = try substPattern(composite, gpa, ft, e.args);
                        const r = try conformsRec(structs, enums, conformances, sub, pid, memo, stack, depth + 1, gpa, composite, bound_protocols);
                        if (r.low < low) low = r.low;
                        if (!r.ok) {
                            aok = false;
                            break;
                        }
                    }
                } else aok = false;
            }
            _ = stack.remove(akey);
            if (low >= depth) {
                try memo.put(gpa, akey, aok);
                return .{ .ok = aok, .low = no_assumption };
            }
            return .{ .ok = aok, .low = low };
        },
        else => {
            // Scalar / non-aggregate: only an explicit/prelude conformance counts (no
            // structural rule); a poison never conforms here. An integer WIDTH normalizes
            // to the canonical platform `Type.int` row (mirroring `existence`) — dead
            // for top-level scalar operators (they resolve via `existence` before ever
            // reaching `structural`), but the correct recursive answer for a future struct/enum
            // field of a width type deriving Eq/Ord.
            for (conformances) |c| {
                if (c.protocol == pid and Type.eql(c.recv, recv) and c.protocol_args.len == 0) return .{ .ok = true, .low = no_assumption };
                if (intMatchesPlatformRow(recv, c.recv) and c.protocol == pid and c.protocol_args.len == 0) return .{ .ok = true, .low = no_assumption };
            }
            return .{ .ok = false, .low = no_assumption };
        },
    }
    const id: u32 = if (recv.kind == .@"enum") recv.enum_id else recv.struct_id;
    const key: u64 = (@as(u64, pid) << 40) | (@as(u64, @intFromEnum(recv.kind)) << 32) | @as(u64, id);
    if (memo.get(key)) |v| return .{ .ok = v, .low = no_assumption };
    if (stack.get(key)) |d| return .{ .ok = true, .low = d };
    try stack.put(gpa, key, depth);

    var ok = false;
    var low: u32 = no_assumption;
    find: {
        // Explicit / prelude / Ord-refinement conformance wins — never double-derive.
        for (conformances) |c| if (c.protocol == pid and Type.eql(c.recv, recv) and c.protocol_args.len == 0) {
            ok = true;
            break :find;
        };
        if (recv.kind == .@"struct") {
            if (recv.struct_id < structs.len) {
                ok = true;
                for (structs[recv.struct_id].field_types) |ft| {
                    const r = try conformsRec(structs, enums, conformances, ft, pid, memo, stack, depth + 1, gpa, composite, bound_protocols);
                    if (r.low < low) low = r.low;
                    if (!r.ok) {
                        ok = false;
                        break;
                    }
                }
            }
        } else { // enum: every variant's payload fields must conform
            if (recv.enum_id < enums.len) {
                ok = true;
                outer: for (enums[recv.enum_id].variants) |v| {
                    for (v.field_types) |ft| {
                        const r = try conformsRec(structs, enums, conformances, ft, pid, memo, stack, depth + 1, gpa, composite, bound_protocols);
                        if (r.low < low) low = r.low;
                        if (!r.ok) {
                            ok = false;
                            break :outer;
                        }
                    }
                }
            }
        }
    }
    _ = stack.remove(key);
    if (low >= depth) {
        try memo.put(gpa, key, ok);
        return .{ .ok = ok, .low = no_assumption };
    }
    return .{ .ok = ok, .low = low };
}

/// Substitute a generic ctor's field/payload PATTERN through a concrete arg tuple for the
/// conditional-conformance query — a free-function mirror of `Typecheck.substType`
/// that needs only the composite table (`structural` has no `*Typecheck`): a `type_var(ord)`
/// becomes `args[ord]`; a nested `App(ctor,[pat..])` substitutes each arg and re-interns
/// (so a `Box[T]` field grounds to `Box[int]`, itself re-queried through the App arm);
/// anything else passes through. Interning here mints transient query-only App indices
/// (never reified/fingerprinted), so the run-order-dependent index is harmless. Public
/// because the checker's T0026/T0027 `deepestNonConforming` error-message walk re-descends
/// the same pattern graph.
pub fn substPattern(composite: *Composite, gpa: std.mem.Allocator, pat: Type, args: []const Type) error{OutOfMemory}!Type {
    if (pat.isTypeVar()) {
        const ord = pat.typeVarOrd();
        return if (ord < args.len) args[ord] else Type.invalid;
    }
    if (pat.isApp()) {
        const e = composite.at(pat.appIdx());
        var buf: [8]Type = undefined;
        const sub: []Type = if (e.args.len <= buf.len) buf[0..e.args.len] else try gpa.alloc(Type, e.args.len);
        defer if (e.args.len > buf.len) gpa.free(sub);
        for (e.args, 0..) |a, i| sub[i] = try substPattern(composite, gpa, a, args);
        return Type.app(try composite.intern(gpa, e.ctor, sub, e.ctor_is_enum));
    }
    return pat;
}

/// The `Item` type of an iterator type `iter_app` (`VecIter[int]` -> `int`): scan the
/// GENERIC-conformance table for a row whose receiver ctor matches `iter_app`'s and whose
/// protocol is named `Iterator`, then substitute the recorded arg PATTERN through
/// `iter_app`'s concrete args. Null when `iter_app` is not an App or conforms to no
/// `Iterator`. This is the ONLY reader of `model.template_conformances`, so recording a
/// generic conformance there perturbs no other verdict. A malformed row (no args) yields
/// null rather than crash.
pub fn iteratorItem(model: *const Model, composite: *Composite, gpa: std.mem.Allocator, iter_app: Type) error{OutOfMemory}!?Type {
    if (!iter_app.isApp()) return null;
    const e = composite.at(iter_app.appIdx());
    for (model.template_conformances) |row| {
        if (row.recv_ctor != e.ctor or row.recv_is_enum != e.ctor_is_enum) continue;
        if (row.protocol_id >= model.protocols.len) continue;
        if (!std.mem.eql(u8, model.protocols[row.protocol_id].name, StdNames.iter_protocol)) continue;
        if (row.protocol_args.len == 0) return null;
        return try substPattern(composite, gpa, row.protocol_args[0], e.args);
    }
    return null;
}

/// The first struct field (in declaration order) whose type does NOT conform to `pid`
/// — the field a T0029 use-site diagnostic names when a struct would derive `Eq` but a
/// field blocks it. Null when `recv` is not a struct or every field conforms.
pub fn firstNonConformingField(
    structs: []const StructSym,
    enums: []const EnumSym,
    conformances: []const Conformance,
    recv: Type,
    pid: u32,
    memo: *std.AutoHashMapUnmanaged(u64, bool),
    gpa: std.mem.Allocator,
    composite: *Composite,
    bound_protocols: []const ?u32,
) error{OutOfMemory}!?NonConformingField {
    if (recv.kind != .@"struct" or recv.struct_id >= structs.len) return null;
    const s = structs[recv.struct_id];
    for (s.field_types, 0..) |ft, i| {
        if (!try structural(structs, enums, conformances, ft, pid, memo, gpa, composite, bound_protocols)) {
            return .{ .name = if (i < s.field_names.len) s.field_names[i] else "?", .ty = ft };
        }
    }
    return null;
}

const testing = std.testing;
const Ast = @import("../ast/Ast.zig");
const VariantSym = LayoutEngine.VariantSym;

test "structural conformance: scalar/struct/nested/empty-enum/payload-enum recurse" {
    const gpa = testing.allocator;
    const eq_pid: u32 = 0;
    // struct#0 {x:int, y:bool} — all scalars conform; struct#1 {inner: struct#0} — nested;
    // struct#2 {c: enum#1} — an all-conforming payload-enum field; struct#3 {d: enum#2}
    // — a NON-conforming payload-enum field (str payload, no prelude Eq).
    const s0_ft = [_]Type{ Type.int, Type.@"bool" };
    const s0_fn = [_][]const u8{ "x", "y" };
    const s1_ft = [_]Type{Type.structT(0)};
    const s1_fn = [_][]const u8{"inner"};
    const s2_ft = [_]Type{Type.enumT(1)};
    const s2_fn = [_][]const u8{"c"};
    const s3_ft = [_]Type{Type.enumT(2)};
    const s3_fn = [_][]const u8{"d"};
    const structs = [_]StructSym{
        .{ .decl_node = Ast.none, .name = "S0", .field_types = @constCast(&s0_ft), .field_names = @constCast(&s0_fn) },
        .{ .decl_node = Ast.none, .name = "S1", .field_types = @constCast(&s1_ft), .field_names = @constCast(&s1_fn) },
        .{ .decl_node = Ast.none, .name = "S2", .field_types = @constCast(&s2_ft), .field_names = @constCast(&s2_fn) },
        .{ .decl_node = Ast.none, .name = "S3", .field_types = @constCast(&s3_ft), .field_names = @constCast(&s3_fn) },
    };
    var e0_vars = [_]VariantSym{ .{ .name = "A", .form = .unit }, .{ .name = "B", .form = .unit } };
    var e1_pl = [_]Type{Type.int};
    var e1_vars = [_]VariantSym{ .{ .name = "R", .form = .tuple, .field_types = &e1_pl }, .{ .name = "G", .form = .unit } };
    var e2_pl = [_]Type{Type.str};
    var e2_vars = [_]VariantSym{ .{ .name = "X", .form = .tuple, .field_types = &e2_pl }, .{ .name = "Y", .form = .unit } };
    const enums = [_]EnumSym{
        .{ .decl_node = Ast.none, .name = "E0", .variants = &e0_vars },
        .{ .decl_node = Ast.none, .name = "E1", .variants = &e1_vars },
        .{ .decl_node = Ast.none, .name = "E2", .variants = &e2_vars },
    };
    const confs = [_]Conformance{ .{ .protocol = eq_pid, .recv = Type.int }, .{ .protocol = eq_pid, .recv = Type.@"bool" } };

    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);
    var co: Composite = .{};
    defer co.deinit(gpa);
    const C = struct {
        fn q(st: []const StructSym, en: []const EnumSym, cf: []const Conformance, ty: Type, m: *std.AutoHashMapUnmanaged(u64, bool), g: std.mem.Allocator, cp: *Composite) !bool {
            return structural(st, en, cf, ty, 0, m, g, cp, &.{});
        }
    };
    try testing.expect(try C.q(&structs, &enums, &confs, Type.int, &memo, gpa, &co));
    try testing.expect(try C.q(&structs, &enums, &confs, Type.structT(0), &memo, gpa, &co));
    try testing.expect(try C.q(&structs, &enums, &confs, Type.structT(1), &memo, gpa, &co));
    try testing.expect(try C.q(&structs, &enums, &confs, Type.enumT(0), &memo, gpa, &co));
    try testing.expect(try C.q(&structs, &enums, &confs, Type.enumT(1), &memo, gpa, &co));
    try testing.expect(try C.q(&structs, &enums, &confs, Type.structT(2), &memo, gpa, &co));
    try testing.expect(!try C.q(&structs, &enums, &confs, Type.enumT(2), &memo, gpa, &co));
    try testing.expect(!try C.q(&structs, &enums, &confs, Type.structT(3), &memo, gpa, &co));
    try testing.expect(!try C.q(&structs, &enums, &confs, Type.str, &memo, gpa, &co));

    // The blocked struct#3 names its first offending field (`d: E2`).
    const off = try firstNonConformingField(&structs, &enums, &confs, Type.structT(3), 0, &memo, gpa, &co, &.{});
    try testing.expect(off != null);
    try testing.expectEqualStrings("d", off.?.name);
    try testing.expect(Type.eql(Type.enumT(2), off.?.ty));
}

test "structural conformance: App types, nested memoization, and the type_var bound axiom" {
    const gpa = testing.allocator;
    const ord_pid: u32 = 0;
    const doubler_pid: u32 = 1;

    var co: Composite = .{};
    defer co.deinit(gpa);

    // struct Box[T] { v: T } — a generic TEMPLATE whose field is the pattern `type_var(0)`.
    var box_ft = [_]Type{Type.typeVar(0)};
    var box_fn = [_][]const u8{"v"};
    const box_params = [_][]const u8{"T"};
    const box_int = try co.intern(gpa, 0, &.{Type.int}, false);
    const box_box_int = try co.intern(gpa, 0, &.{Type.app(box_int)}, false);
    const box_tv = try co.intern(gpa, 0, &.{Type.typeVar(0)}, false);
    var sbox_ft = [_]Type{Type.app(box_int)};
    var sbox_fn = [_][]const u8{"b"};

    const structs = [_]StructSym{
        .{ .decl_node = Ast.none, .name = "Box", .field_types = @constCast(&box_ft), .field_names = @constCast(&box_fn), .is_generic = true, .generic_params = &box_params },
        .{ .decl_node = Ast.none, .name = "SBox", .field_types = @constCast(&sbox_ft), .field_names = @constCast(&sbox_fn) },
    };
    const enums = [_]EnumSym{};
    const confs = [_]Conformance{.{ .protocol = ord_pid, .recv = Type.int }};

    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);

    try testing.expect(try structural(&structs, &enums, &confs, Type.app(box_int), ord_pid, &memo, gpa, &co, &.{}));
    try testing.expect(!try structural(&structs, &enums, &confs, Type.app(box_int), doubler_pid, &memo, gpa, &co, &.{}));

    try testing.expect(try structural(&structs, &enums, &confs, Type.app(box_box_int), ord_pid, &memo, gpa, &co, &.{}));
    const n = memo.count();
    try testing.expect(try structural(&structs, &enums, &confs, Type.app(box_box_int), ord_pid, &memo, gpa, &co, &.{}));
    try testing.expectEqual(n, memo.count());

    // Abstract Box[T]: discharges from the axiom `T has Ord` iff the bound IS Ord.
    const bp_ord = [_]?u32{ord_pid};
    const bp_doubler = [_]?u32{doubler_pid};
    var memo_ord: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo_ord.deinit(gpa);
    var memo_dbl: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo_dbl.deinit(gpa);
    try testing.expect(try structural(&structs, &enums, &confs, Type.app(box_tv), ord_pid, &memo_ord, gpa, &co, &bp_ord));
    try testing.expect(!try structural(&structs, &enums, &confs, Type.app(box_tv), ord_pid, &memo_dbl, gpa, &co, &bp_doubler));

    var memo_s: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo_s.deinit(gpa);
    try testing.expect(try structural(&structs, &enums, &confs, Type.structT(1), ord_pid, &memo_s, gpa, &co, &.{}));
}

test "iteratorItem: a matching generic-conformance row substitutes Item through the App args" {
    const gpa = testing.allocator;
    const iter_pid: u32 = 0;
    const other_pid: u32 = 1;

    var co: Composite = .{};
    defer co.deinit(gpa);
    // VecIter[int] (ctor 0) and a Box[int] (ctor 1) that conforms to no Iterator.
    const veciter_int = Type.app(try co.intern(gpa, 0, &.{Type.int}, false));
    const box_int = Type.app(try co.intern(gpa, 1, &.{Type.int}, false));

    const protocols = [_]Typecheck.ProtocolSym{
        .{ .name = "Iterator", .mod = 0, .pub_export = true, .decl_node = Ast.none, .methods = &.{} },
        .{ .name = "Other", .mod = 0, .pub_export = true, .decl_node = Ast.none, .methods = &.{} },
    };
    // The recorded arg PATTERN is `[type_var(0)]` (Item = the impl's sole type-param).
    const pat = [_]Type{Type.typeVar(0)};
    const rows = [_]Typecheck.TemplateConformance{
        .{ .protocol_id = iter_pid, .recv_ctor = 0, .recv_is_enum = false, .protocol_args = &pat },
        .{ .protocol_id = other_pid, .recv_ctor = 1, .recv_is_enum = false, .protocol_args = &pat },
    };

    var model: Model = undefined;
    model.protocols = &protocols;
    model.template_conformances = &rows;

    // VecIter[int] -> Item = int (type_var(0) grounded through the App's `[int]`).
    const item = try iteratorItem(&model, &co, gpa, veciter_int);
    try testing.expect(item != null);
    try testing.expect(Type.eql(Type.int, item.?));
    // Box[int] matches ctor 1 but its protocol is "Other" (not "Iterator") -> null.
    try testing.expect((try iteratorItem(&model, &co, gpa, box_int)) == null);
    // A non-App receiver is never iterable here.
    try testing.expect((try iteratorItem(&model, &co, gpa, Type.int)) == null);
}

test "axiom + direct: the bound-as-axiom rule, single-sourced" {
    // `axiom` denies for a non-type_var, an out-of-range ordinal, and an unbounded slot;
    // discharges iff the ordinal's bound IS `pid`. This is the ONE copy the former three
    // verbatim rules (conformsRec/conformsTo/conformsToArith) collapse into.
    const ord_pid: u32 = 3;
    const other_pid: u32 = 4;
    const bounds = [_]?u32{ ord_pid, null };
    try testing.expect(axiom(Type.typeVar(0), ord_pid, &bounds)); // T0 has Ord
    try testing.expect(!axiom(Type.typeVar(0), other_pid, &bounds)); // T0 not bound by other
    try testing.expect(!axiom(Type.typeVar(1), ord_pid, &bounds)); // T1 unbounded (null)
    try testing.expect(!axiom(Type.typeVar(9), ord_pid, &bounds)); // out of range
    try testing.expect(!axiom(Type.int, ord_pid, &bounds)); // not a type_var
    try testing.expect(!axiom(Type.typeVar(0), ord_pid, &.{})); // empty bounds deny

    // `direct` = existence ∨ axiom: with an empty conformance table, only the axiom fires.
    var model: Model = undefined;
    model.conformances = &.{};
    try testing.expect(direct(&model, Type.typeVar(0), ord_pid, &bounds));
    try testing.expect(!direct(&model, Type.typeVar(0), other_pid, &bounds));
    try testing.expect(!direct(&model, Type.int, ord_pid, &bounds));
}

test "keystone: the checker verdict and the shared witness leaf agree across the table" {
    // The property formerly trusted only by comment (`lower.zig:887`, `BodyChecker.zig:1266`):
    // wherever the conformance DECISION accepts `(recv, pid)`, the shared witness LEAF
    // (`resolveConformanceMethod` / `builtinScalarMethod`) resolves exactly one witness — so
    // the checker never accepts an operator that lower/fingerprint then cannot bind.
    const gpa = testing.allocator;
    const eq_pid: u32 = 0;
    const ord_pid: u32 = 1;

    var co: Composite = .{};
    defer co.deinit(gpa);

    // A user struct S with an explicit `impl S has Eq { fn eq }` (Method carrying a fn_id),
    // a derived struct D (Method carrying a `.derive` recipe index), plus the platform int row.
    const s_ft = [_]Type{Type.int};
    const s_fn = [_][]const u8{"x"};
    const structs = [_]StructSym{
        .{ .decl_node = Ast.none, .name = "S", .field_types = @constCast(&s_ft), .field_names = @constCast(&s_fn) },
        .{ .decl_node = Ast.none, .name = "D", .field_types = @constCast(&s_ft), .field_names = @constCast(&s_fn) },
    };
    const enums = [_]EnumSym{};
    // int conforms to Eq/Ord (platform rows); S has an explicit Eq impl row.
    const confs = [_]Conformance{
        .{ .protocol = eq_pid, .recv = Type.int },
        .{ .protocol = ord_pid, .recv = Type.int },
        .{ .protocol = eq_pid, .recv = Type.structT(0) },
    };
    const methods = [_]Typecheck.Method{
        .{ .recv = Type.structT(0), .name = "eq", .fn_id = 42, .protocol_id = eq_pid },
        .{ .recv = Type.structT(1), .name = "eq", .fn_id = 0, .derive = 7, .protocol_id = eq_pid },
    };

    var model: Model = undefined;
    model.structs = &structs;
    model.enums = &enums;
    model.conformances = &confs;

    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);

    // int-width receiver: existence/direct accept (width-norm), and the SCALAR witness leaf
    // both lower and the fingerprint fold use fires (int8/uint32 collapse to the int row).
    for ([_]Type{ Type.int8, Type.uint32 }) |w| {
        try testing.expect(existence(&model, eq_pid, w, &.{}));
        try testing.expect(direct(&model, w, ord_pid, &.{}));
        try testing.expect(Typecheck.builtinScalarMethod(w, "eq") != null);
    }

    // explicit-impl struct S: direct accepts (existence), and the witness leaf resolves to
    // the one Method with fn_id 42.
    try testing.expect(direct(&model, Type.structT(0), eq_pid, &.{}));
    switch (Typecheck.resolveConformanceMethod(&methods, Type.structT(0), "eq", eq_pid, null)) {
        .one => |m| try testing.expectEqual(@as(u32, 42), m.fn_id),
        else => return error.WitnessNotResolved,
    }

    // derived struct D: NO explicit row, so classify falls to the structural tier, and the
    // witness leaf resolves the same Method carrying `.derive = 7` (the identity
    // witnessCallee/foldWitness key their mangled symbol on).
    try testing.expectEqual(Tier.structural, try classify(&model, &co, &.{}, &memo, gpa, Type.structT(1), eq_pid));
    switch (Typecheck.resolveConformanceMethod(&methods, Type.structT(1), "eq", eq_pid, null)) {
        .one => |m| try testing.expectEqual(@as(u32, 7), m.derive.?),
        else => return error.WitnessNotResolved,
    }

    // type_var bound `[T has Ord]`: direct accepts via the axiom (no witness pick — a bound
    // discharges structurally at the instance re-check, not here).
    const bounds = [_]?u32{ord_pid};
    try testing.expect(direct(&model, Type.typeVar(0), ord_pid, &bounds));
}
