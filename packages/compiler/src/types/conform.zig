//! The conformance DECISION, deepened into one module: "does ground/abstract type
//! `T` satisfy protocol `P`, and via which tier". Extracted from the `Typecheck`
//! mega-struct so the existence check, the type_var bound-as-axiom rule, the composed
//! operator verdict, the recursive structural engine, and the derive-blocker leaf live
//! behind ONE small interface instead of being re-derived at each caller. The witness
//! PICK (`resolveConformanceMethod`) stays in the parent: the checker records a method
//! call's pick as a Callee, and the operator/derive witness sites call it directly — this
//! module decides conformance, it does not pick the method.
//!
//! Imports the parent as `Typecheck` (the `src/types/` submodule convention shared with
//! `derive_synth.zig`/`coherence.zig`/`reify.zig`).

const std = @import("std");
const Typecheck = @import("../types.zig");
const LayoutEngine = @import("../layout/Engine.zig");
const Composite = @import("../symbols/Composite.zig");
const StdNames = @import("../symbols/StdNames.zig");
const Infer = @import("../symbols/Infer.zig");

const Type = Typecheck.Type;
const Model = Typecheck.Model;
const Conformance = Typecheck.Conformance;
const TemplateConformance = Typecheck.TemplateConformance;
const NonConformingField = Typecheck.NonConformingField;
const StructSym = LayoutEngine.StructSym;
const EnumSym = LayoutEngine.EnumSym;
const eqlTypeVec = Typecheck.eqlTypeVec;

/// Whether the ground type `recv` conforms to protocol `pid`. A pure,
/// thread-order-free linear scan over the frozen `model.conformances` by protocol id
/// + `Type.eql` — the SAME determinism discipline `findMethod` uses, so bound
/// resolution at the mono worklist is byte-identical at any `-jN`. A generic-App
/// receiver misses here: a generic `impl Box[T] has P` is a template row, which
/// `generic` matches.
pub fn existence(model: *const Model, pid: u32, recv: Type, protocol_args: []const Type) bool {
    if (protocol_args.len == 0 and fnDisplays(model.structs, .of(model), recv, pid)) return true;
    return existsIn(model.conformances, pid, recv, protocol_args);
}

/// Whether `recv` is a function type (or the pair it reifies to) asked for Display. A
/// function value renders as `<fn>` without a witness, since no impl can be written for a
/// function type. Display is its only protocol: identity is all it has to compare or hash.
fn fnDisplays(structs: []const StructSym, gen: Generic, recv: Type, pid: u32) bool {
    const pre = gen.prelude orelse return false;
    if (pre.protocols.display != pid) return false;
    return recv.isFunc() or LayoutEngine.isFnStruct(recv, structs);
}

pub fn existsIn(conformances: []const Conformance, pid: u32, recv: Type, protocol_args: []const Type) bool {
    for (conformances) |c| {
        // Exact match first (a user `impl int8 has P` row, or any concrete receiver).
        if (c.protocol == pid and Type.eql(c.recv, recv) and eqlTypeVec(c.protocol_args, protocol_args)) return true;
        // Then the integer-width normalization (see `intMatchesPlatformRow`): keeps the
        // row table minimal (no per-width rows).
        if (intMatchesPlatformRow(recv, c.recv) and c.protocol == pid and eqlTypeVec(c.protocol_args, protocol_args)) return true;
    }
    return false;
}

/// What a pre-reify conformance query needs to see generic conformances: the template
/// rows, and the prelude, whose derivable protocols an impl-param bound may be met by
/// structurally. The empty value sees none, which is right post-reify (a reached generic
/// conformance is a concrete `conformances` row by then) and for a prelude-less caller.
pub const Generic = struct {
    rows: []const TemplateConformance = &.{},
    prelude: ?Typecheck.Prelude = null,

    pub fn of(model: *const Model) Generic {
        return .{ .rows = model.template_conformances, .prelude = model.prelude };
    }
};

/// The tables `generic` reads, bundled so the recursion through impl-param bounds passes
/// one value.
pub const Tables = struct {
    structs: []const StructSym,
    enums: []const EnumSym,
    conformances: []const Conformance,
    gen: Generic,
    composite: *Composite,
    gpa: std.mem.Allocator,
    bound_protocols: []const ?u32,
    /// Set when called from inside `conformsRec`, so a bound's structural check joins
    /// its in-progress stack (cycle-safe) and reports the assumptions it leaned on.
    rec: ?Rec = null,

    const Rec = struct { stack: *std.AutoHashMapUnmanaged(u64, u32), depth: u32, low: *u32 };
};

/// A generic conformance an App receiver meets: its row, and the impl's params bound
/// from the receiver's args (OWNED by the caller).
pub const GenericMatch = struct {
    row: *const TemplateConformance,
    params: []Type,

    pub fn deinit(m: GenericMatch, gpa: std.mem.Allocator) void {
        gpa.free(m.params);
    }
};

/// How a generic conformance `impl Ctor[T] has P[..]` decides `pid[protocol_args]` for the
/// App `recv`: no row covers `recv` (`none`); a row binds the impl's params from `recv`'s
/// args, its arg pattern grounds to `protocol_args`, and every impl-param bound
/// (`impl Box[T has Display]`) holds (`met`); or a row covers `recv` but a bound fails
/// (`unmet`). An explicit impl wins over the structural derive even when it does not
/// apply, so `unmet` is a denial, not a fall-through. `recv` may be abstract (a bounded
/// template's `Box[T]`): a bound on `T` then holds by the axiom rule. Coherence keeps at
/// most one row per `(pid, ctor, args)`.
pub const Verdict = union(enum) { none, met: GenericMatch, unmet: UnmetBound };

pub fn generic(tb: Tables, memo: *std.AutoHashMapUnmanaged(u64, bool), recv: Type, pid: u32, protocol_args: []const Type) error{OutOfMemory}!Verdict {
    const row = (try bindRow(tb, recv, pid, protocol_args)) orelse return .none;
    if (try firstUnmetBound(tb, memo, row)) |u| {
        row.deinit(tb.gpa);
        return .{ .unmet = u };
    }
    return .{ .met = row };
}

/// The impl-param bound a generic row leaves unmet: the bound arg and its protocol.
pub const UnmetBound = struct { arg: Type, pid: u32 };

/// The first impl-param bound of `m` whose arg does not conform, else null.
fn firstUnmetBound(tb: Tables, memo: *std.AutoHashMapUnmanaged(u64, bool), m: GenericMatch) error{OutOfMemory}!?UnmetBound {
    for (m.row.bounds, 0..) |maybe_bp, ord| {
        const bp = maybe_bp orelse continue;
        if (ord >= m.params.len) continue;
        const arg = m.params[ord];
        const pat: []const Type = if (ord < m.row.bound_args.len) m.row.bound_args[ord] else &.{};
        var buf: [4]Type = undefined;
        const bargs: []Type = if (pat.len <= buf.len) buf[0..pat.len] else try tb.gpa.alloc(Type, pat.len);
        defer if (pat.len > buf.len) tb.gpa.free(bargs);
        for (pat, bargs) |p, *b| b.* = try substPattern(tb.composite, tb.gpa, p, m.params);
        if (!try meets(tb, memo, arg, bp, bargs)) return .{ .arg = arg, .pid = bp };
    }
    return null;
}

/// The row of `pid[protocol_args]` whose receiver pattern binds from the App `recv`,
/// before its impl-param bounds are checked.
fn bindRow(tb: Tables, recv: Type, pid: u32, protocol_args: []const Type) error{OutOfMemory}!?GenericMatch {
    if (!recv.isApp()) return null;
    const e = tb.composite.at(recv.appIdx());
    if (e.is_fn) return null;
    for (tb.gen.rows) |*row| {
        if (row.protocol_id != pid or row.recv_ctor != e.ctor or row.recv_is_enum != e.ctor_is_enum) continue;
        if (row.protocol_args.len != protocol_args.len or !row.recv.isApp()) continue;
        const pat = tb.composite.at(row.recv.appIdx()).args;
        if (pat.len != e.args.len) continue;
        const params = try tb.gpa.alloc(Type, row.n_params);
        errdefer tb.gpa.free(params);
        const bnd = try tb.gpa.alloc(bool, row.n_params);
        defer tb.gpa.free(bnd);
        const fp = try tb.gpa.alloc(usize, row.n_params);
        defer tb.gpa.free(fp);
        if (Infer.match(row.n_params, pat, e.args, params, bnd, fp, tb.composite.decomposer()) != .ok) {
            tb.gpa.free(params);
            continue;
        }
        var args_ok = true;
        for (row.protocol_args, protocol_args) |pa, want| {
            if (!Type.eql(try substPattern(tb.composite, tb.gpa, pa, params), want)) {
                args_ok = false;
                break;
            }
        }
        if (!args_ok) {
            tb.gpa.free(params);
            continue;
        }
        return .{ .row = row, .params = params };
    }
    return null;
}

/// The protocol args the first generic conformance of `pid` covering the App `recv` gives
/// it, grounded through `recv`'s args and OWNED by the caller; null when none covers it.
/// Names what an impl gives when a bound asks `pid` at other args.
pub fn givenArgs(tb: Tables, recv: Type, pid: u32) error{OutOfMemory}!?[]Type {
    if (!recv.isApp()) return null;
    const e = tb.composite.at(recv.appIdx());
    for (tb.gen.rows) |row| {
        if (row.protocol_id != pid or row.recv_ctor != e.ctor or row.recv_is_enum != e.ctor_is_enum) continue;
        if (row.n_params != e.args.len) continue;
        const out = try tb.gpa.alloc(Type, row.protocol_args.len);
        for (row.protocol_args, out) |pa, *o| o.* = try substPattern(tb.composite, tb.gpa, pa, e.args);
        return out;
    }
    return null;
}

/// Whether `ty` meets the bound `[T has pid[protocol_args]]`, by every route a bound at
/// an instantiation accepts: a concrete conformance, the axiom rule, a generic
/// conformance, or a structural derive of a zero-arg derivable protocol.
fn meets(tb: Tables, memo: *std.AutoHashMapUnmanaged(u64, bool), ty: Type, pid: u32, protocol_args: []const Type) error{OutOfMemory}!bool {
    if (existsIn(tb.conformances, pid, ty, protocol_args)) return true;
    if (axiom(ty, pid, tb.bound_protocols)) return true;
    switch (try generic(tb, memo, ty, pid, protocol_args)) {
        .none => {},
        .met => |m| {
            m.deinit(tb.gpa);
            return true;
        },
        .unmet => return false,
    }
    if (protocol_args.len != 0) return false;
    const pre = tb.gen.prelude orelse return false;
    if (!pre.isDerivable(pid)) return false;
    if (tb.rec) |r| {
        const st = try conformsRec(tb.structs, tb.enums, tb.conformances, tb.gen, ty, pid, memo, r.stack, r.depth + 1, tb.gpa, tb.composite, tb.bound_protocols);
        if (st.low < r.low.*) r.low.* = st.low;
        return st.ok;
    }
    return structural(tb.structs, tb.enums, tb.conformances, tb.gen, ty, pid, memo, tb.gpa, tb.composite, tb.bound_protocols);
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

/// Which tier of the composed operator verdict fired. `.generic` and `.structural` carry
/// a side-effect obligation the caller records as a derive request: the generic impl's
/// instance for the mono tail to instantiate, or the structural recipe for the synthesis
/// barrier. They stay distinct tiers because only `.structural` is a derive.
pub const Tier = enum { none, direct, generic, structural };

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
    const tb: Tables = .{ .structs = model.structs, .enums = model.enums, .conformances = model.conformances, .gen = .of(model), .composite = composite, .gpa = gpa, .bound_protocols = bound_protocols };
    switch (try generic(tb, memo, recv, pid, &.{})) {
        .none => {},
        .met => |m| {
            m.deinit(gpa);
            return .generic;
        },
        .unmet => return .none,
    }
    if (try structural(model.structs, model.enums, model.conformances, .of(model), recv, pid, memo, gpa, composite, bound_protocols))
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
    gen: Generic,
    recv: Type,
    pid: u32,
    memo: *std.AutoHashMapUnmanaged(u64, bool),
    gpa: std.mem.Allocator,
    composite: *Composite,
    bound_protocols: []const ?u32,
) error{OutOfMemory}!bool {
    var stack: std.AutoHashMapUnmanaged(u64, u32) = .empty;
    defer stack.deinit(gpa);
    const r = try conformsRec(structs, enums, conformances, gen, recv, pid, memo, &stack, 0, gpa, composite, bound_protocols);
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
    gen: Generic,
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
            // An explicit generic conformance wins over the structural walk, as a
            // concrete `impl` does in the nominal arm below. Its impl-param bounds recurse
            // on this stack: a bound arg may hold this very App (`Node[T]` through `Box`).
            const tb: Tables = .{ .structs = structs, .enums = enums, .conformances = conformances, .gen = gen, .composite = composite, .gpa = gpa, .bound_protocols = bound_protocols, .rec = .{ .stack = stack, .depth = depth, .low = &low } };
            const covered = switch (try generic(tb, memo, recv, pid, &.{})) {
                .none => false,
                .met => |m| blk: {
                    m.deinit(gpa);
                    break :blk true;
                },
                .unmet => blk: {
                    aok = false;
                    break :blk true;
                },
            };
            if (covered) {} else if (neverDerives(structs, composite, gen, recv, pid)) {
                aok = false;
            } else if (e.ctor_is_enum) {
                if (e.ctor < enums.len) {
                    outer: for (enums[e.ctor].variants) |v| {
                        for (v.field_types) |ft| {
                            const sub = try substPattern(composite, gpa, ft, e.args);
                            const r = try conformsRec(structs, enums, conformances, gen, sub, pid, memo, stack, depth + 1, gpa, composite, bound_protocols);
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
                        const r = try conformsRec(structs, enums, conformances, gen, sub, pid, memo, stack, depth + 1, gpa, composite, bound_protocols);
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
            if (fnDisplays(structs, gen, recv, pid)) return .{ .ok = true, .low = no_assumption };
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
        if (LayoutEngine.isFnStruct(recv, structs)) {
            ok = fnDisplays(structs, gen, recv, pid);
            break :find;
        }
        if (neverDerives(structs, composite, gen, recv, pid)) break :find;
        if (recv.kind == .@"struct") {
            if (recv.struct_id < structs.len) {
                ok = true;
                for (structs[recv.struct_id].field_types) |ft| {
                    const r = try conformsRec(structs, enums, conformances, gen, ft, pid, memo, stack, depth + 1, gpa, composite, bound_protocols);
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
                        const r = try conformsRec(structs, enums, conformances, gen, ft, pid, memo, stack, depth + 1, gpa, composite, bound_protocols);
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
/// `Iterator`. A malformed row (no args) yields null rather than crash.
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
    gen: Generic,
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
        if (!try structural(structs, enums, conformances, gen, ft, pid, memo, gpa, composite, bound_protocols)) {
            return .{ .name = if (i < s.field_names.len) s.field_names[i] else "?", .ty = ft };
        }
    }
    return null;
}

/// Whether `recv` IS a managed box (`Ref`/`gc_array`): a reified structT carrying the
/// reference marker (post-reify), OR a struct-`App` over the prelude box template (pre-reify,
/// as the bound check may see). Single leaf test for both stages.
fn isManagedBox(structs: []const StructSym, composite: *Composite, recv: Type, ref_ctor: ?u32, gc_array_ctor: ?u32) bool {
    if (LayoutEngine.isRefStruct(recv, structs)) return true;
    if (recv.isApp()) {
        const e = composite.at(recv.appIdx());
        if (e.ctor_is_enum) return false;
        if (ref_ctor) |rc| if (e.ctor == rc) return true;
        if (gc_array_ctor) |gc| if (e.ctor == gc) return true;
    }
    return false;
}

/// Whether `recv` is a managed box asked for a derived `Display`. A box's only field is
/// its cell pointer, so a derive would print an address: a nondeterministic render that
/// leaks the private `gc_array` storage of every collection. Eq and Hash keep their own
/// rules (identity `==`, the `containsManaged` key guard). Needs `gen.prelude` to name
/// Display.
fn neverDerives(structs: []const StructSym, composite: *Composite, gen: Generic, recv: Type, pid: u32) bool {
    const pre = gen.prelude orelse return false;
    if (pre.protocols.display != pid) return false;
    return isManagedBox(structs, composite, recv, pre.ref_struct, pre.gc_array_struct);
}

/// The managed box that keeps `recv` from deriving Display, or null: `recv` itself, or
/// a field (or variant payload) that fails Display, down the by-value chain. Lets the
/// T0023 hint name the box rather than ask for a collection's private `gc_array` to conform.
pub fn displayBlockingBox(structs: []const StructSym, enums: []const EnumSym, conformances: []const Conformance, gen: Generic, composite: *Composite, recv: Type, pid: u32, memo: *std.AutoHashMapUnmanaged(u64, bool), gpa: std.mem.Allocator) error{OutOfMemory}!?Type {
    if (neverDerives(structs, composite, gen, recv, pid)) {
        // A box that an impl covers (std's `impl Ref[T has Display]`) fails on a type arg
        // instead, which the plain field hint names.
        if (recv.kind != .app) return recv;
        const tb: Tables = .{ .structs = structs, .enums = enums, .conformances = conformances, .gen = gen, .composite = composite, .gpa = gpa, .bound_protocols = &.{} };
        return switch (try generic(tb, memo, recv, pid, &.{})) {
            .none => recv,
            .met => |m| blk: {
                m.deinit(gpa);
                break :blk null;
            },
            .unmet => null,
        };
    }
    var comps: std.ArrayList(Type) = .empty;
    defer comps.deinit(gpa);
    switch (recv.kind) {
        .@"struct" => if (recv.struct_id < structs.len) try comps.appendSlice(gpa, structs[recv.struct_id].field_types),
        .@"enum" => if (recv.enum_id < enums.len) for (enums[recv.enum_id].variants) |v| try comps.appendSlice(gpa, v.field_types),
        .app => {
            const e = composite.at(recv.appIdx());
            if (e.ctor_is_enum) {
                if (e.ctor < enums.len) for (enums[e.ctor].variants) |v| for (v.field_types) |ft| try comps.append(gpa, try substPattern(composite, gpa, ft, e.args));
            } else if (e.ctor < structs.len) for (structs[e.ctor].field_types) |ft| try comps.append(gpa, try substPattern(composite, gpa, ft, e.args));
        },
        else => return null,
    }
    // Terminates: a box is a leaf (answered above) and by-value cycles are a compile error.
    for (comps.items) |ft| {
        if (try structural(structs, enums, conformances, gen, ft, pid, memo, gpa, composite, &.{})) continue;
        if (try displayBlockingBox(structs, enums, conformances, gen, composite, ft, pid, memo, gpa)) |box| return box;
    }
    return null;
}

/// Whether `recv` is, or transitively (by-value) HOLDS, a managed box. A box has no stable
/// hashable value (its identity is a heap cell), so a Hashable KEY must exclude it — the Ref
/// guard the structural Hash relation itself lacks (a box's reified field is `int`, which
/// `structural(hash)` would wrongly accept). Mirrors `derive_synth.needsTrace`'s by-value
/// walk but pure over the tables so both callers share it; a box is a LEAF (never recursed).
/// Terminates: by-value cycles are already a compile error and a box's field is `int`, so the
/// by-value graph is a finite DAG. `memo` keyed on (nominal id + enum flag) collapses shared
/// subtrees; the transient `App` arm is unmemoized (bounded by the finite by-value DAG).
fn containsManaged(structs: []const StructSym, enums: []const EnumSym, composite: *Composite, recv: Type, ref_ctor: ?u32, gc_array_ctor: ?u32, memo: *std.AutoHashMapUnmanaged(u64, bool), gpa: std.mem.Allocator) error{OutOfMemory}!bool {
    if (isManagedBox(structs, composite, recv, ref_ctor, gc_array_ctor)) return true;
    switch (recv.kind) {
        .@"struct", .@"enum" => {
            const is_enum = recv.kind == .@"enum";
            const id = recv.nominalId();
            const key = (@as(u64, id) << 1) | @intFromBool(is_enum);
            if (memo.get(key)) |v| return v;
            try memo.put(gpa, key, false); // cheap insurance against a (should-not-occur) by-value cycle
            var result = false;
            if (is_enum) {
                if (id < enums.len) outer: for (enums[id].variants) |v| for (v.field_types) |ft| {
                    if (try containsManaged(structs, enums, composite, ft, ref_ctor, gc_array_ctor, memo, gpa)) {
                        result = true;
                        break :outer;
                    }
                };
            } else if (id < structs.len) for (structs[id].field_types) |ft| {
                if (try containsManaged(structs, enums, composite, ft, ref_ctor, gc_array_ctor, memo, gpa)) {
                    result = true;
                    break;
                }
            };
            try memo.put(gpa, key, result);
            return result;
        },
        .app => {
            const e = composite.at(recv.appIdx());
            if (e.ctor_is_enum) {
                if (e.ctor < enums.len) for (enums[e.ctor].variants) |v| for (v.field_types) |ft| {
                    if (try containsManaged(structs, enums, composite, try substPattern(composite, gpa, ft, e.args), ref_ctor, gc_array_ctor, memo, gpa)) return true;
                };
            } else if (e.ctor < structs.len) for (structs[e.ctor].field_types) |ft| {
                if (try containsManaged(structs, enums, composite, try substPattern(composite, gpa, ft, e.args), ref_ctor, gc_array_ctor, memo, gpa)) return true;
            };
            return false;
        },
        else => return false,
    }
}

/// The SINGLE Hashable-key authority, shared by the `[K has Hashable]` bound and the
/// descriptor's erased hash/eq gate so "passes the gate" ≡ "the erased hash/eq units can be
/// built AND call a resolvable witness". `existence`-FIRST (a builtin scalar row OR a
/// top-level explicit `impl` — honored, and short-circuiting the Ref guard so an explicit box
/// impl is respected), then the STRUCTURAL leg keyed on `hash_pid` (the pid the Hash/Eq
/// emitters resolve each field through), minus managed-box keys (`containsManaged`). `gen`
/// lets a pre-reify query see an explicit generic `Hash` impl, which wins over the derive
/// as a concrete one does; post-reify its reached instances are concrete rows.
pub fn hashable(structs: []const StructSym, enums: []const EnumSym, conformances: []const Conformance, gen: Generic, composite: *Composite, recv: Type, hashable_pid: u32, hash_pid: u32, ref_ctor: ?u32, gc_array_ctor: ?u32, memo: *std.AutoHashMapUnmanaged(u64, bool), gpa: std.mem.Allocator) error{OutOfMemory}!bool {
    for (conformances) |c| {
        if (c.protocol == hashable_pid and c.protocol_args.len == 0 and (Type.eql(c.recv, recv) or intMatchesPlatformRow(recv, c.recv))) return true;
    }
    switch (recv.kind) {
        .@"struct", .@"enum", .app => {},
        else => return false,
    }
    var mm: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer mm.deinit(gpa);
    if (try containsManaged(structs, enums, composite, recv, ref_ctor, gc_array_ctor, &mm, gpa)) return false;
    return structural(structs, enums, conformances, gen, recv, hash_pid, memo, gpa, composite, &.{});
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
    const s0_ft = [_]Type{ Type.int, Type.bool };
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
    const confs = [_]Conformance{ .{ .protocol = eq_pid, .recv = Type.int }, .{ .protocol = eq_pid, .recv = Type.bool } };

    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);
    var co: Composite = .{};
    defer co.deinit(gpa);
    const C = struct {
        fn q(st: []const StructSym, en: []const EnumSym, cf: []const Conformance, ty: Type, m: *std.AutoHashMapUnmanaged(u64, bool), g: std.mem.Allocator, cp: *Composite) !bool {
            return structural(st, en, cf, .{}, ty, 0, m, g, cp, &.{});
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
    const off = try firstNonConformingField(&structs, &enums, &confs, .{}, Type.structT(3), 0, &memo, gpa, &co, &.{});
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

    try testing.expect(try structural(&structs, &enums, &confs, .{}, Type.app(box_int), ord_pid, &memo, gpa, &co, &.{}));
    try testing.expect(!try structural(&structs, &enums, &confs, .{}, Type.app(box_int), doubler_pid, &memo, gpa, &co, &.{}));

    try testing.expect(try structural(&structs, &enums, &confs, .{}, Type.app(box_box_int), ord_pid, &memo, gpa, &co, &.{}));
    const n = memo.count();
    try testing.expect(try structural(&structs, &enums, &confs, .{}, Type.app(box_box_int), ord_pid, &memo, gpa, &co, &.{}));
    try testing.expectEqual(n, memo.count());

    // Abstract Box[T]: discharges from the axiom `T has Ord` iff the bound IS Ord.
    const bp_ord = [_]?u32{ord_pid};
    const bp_doubler = [_]?u32{doubler_pid};
    var memo_ord: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo_ord.deinit(gpa);
    var memo_dbl: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo_dbl.deinit(gpa);
    try testing.expect(try structural(&structs, &enums, &confs, .{}, Type.app(box_tv), ord_pid, &memo_ord, gpa, &co, &bp_ord));
    try testing.expect(!try structural(&structs, &enums, &confs, .{}, Type.app(box_tv), ord_pid, &memo_dbl, gpa, &co, &bp_doubler));

    var memo_s: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo_s.deinit(gpa);
    try testing.expect(try structural(&structs, &enums, &confs, .{}, Type.structT(1), ord_pid, &memo_s, gpa, &co, &.{}));
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
    // Wherever the conformance DECISION accepts `(recv, pid)`, the witness LEAF the
    // operator sites resolve through (`resolveConformanceMethod` / `builtinScalarMethod`)
    // yields exactly one witness — so the checker never accepts an operator that
    // lower/fingerprint then cannot bind.
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

test "hashable: scalar/struct/nested/Ref/float/enum/explicit truth table" {
    const gpa = testing.allocator;
    const hash_pid: u32 = 0;
    const hashable_pid: u32 = 1;

    var co: Composite = .{};
    defer co.deinit(gpa);

    // struct#0 {x:int, y:int} — all Hashable scalars; struct#1 Box (native_family=.ref) —
    // a reified managed box; struct#2 {r: Box} — holds a box (excluded); struct#3 {f:float}
    // — a non-Hashable field; struct#4 {b: Box} but carries an EXPLICIT (hashable, S4) row.
    const s0_ft = [_]Type{ Type.int, Type.int };
    const s0_fn = [_][]const u8{ "x", "y" };
    const box_ft = [_]Type{Type.int};
    const box_fn = [_][]const u8{"0"};
    const s2_ft = [_]Type{Type.structT(1)};
    const s2_fn = [_][]const u8{"r"};
    const s3_ft = [_]Type{Type.float};
    const s3_fn = [_][]const u8{"f"};
    const s4_ft = [_]Type{Type.structT(1)};
    const s4_fn = [_][]const u8{"b"};
    const structs = [_]StructSym{
        .{ .decl_node = Ast.none, .name = "S0", .field_types = @constCast(&s0_ft), .field_names = @constCast(&s0_fn) },
        .{ .decl_node = Ast.none, .name = "Box", .field_types = @constCast(&box_ft), .field_names = @constCast(&box_fn), .native_family = .ref },
        .{ .decl_node = Ast.none, .name = "S2", .field_types = @constCast(&s2_ft), .field_names = @constCast(&s2_fn) },
        .{ .decl_node = Ast.none, .name = "S3", .field_types = @constCast(&s3_ft), .field_names = @constCast(&s3_fn) },
        .{ .decl_node = Ast.none, .name = "S4", .field_types = @constCast(&s4_ft), .field_names = @constCast(&s4_fn) },
    };
    var e0_vars = [_]VariantSym{ .{ .name = "A", .form = .unit }, .{ .name = "B", .form = .unit } };
    var e1_pl = [_]Type{Type.int};
    var e1_vars = [_]VariantSym{ .{ .name = "R", .form = .tuple, .field_types = &e1_pl }, .{ .name = "G", .form = .unit } };
    var e2_pl = [_]Type{Type.float};
    var e2_vars = [_]VariantSym{ .{ .name = "X", .form = .tuple, .field_types = &e2_pl }, .{ .name = "Y", .form = .unit } };
    const enums = [_]EnumSym{
        .{ .decl_node = Ast.none, .name = "E0", .variants = &e0_vars },
        .{ .decl_node = Ast.none, .name = "E1", .variants = &e1_vars },
        .{ .decl_node = Ast.none, .name = "E2", .variants = &e2_vars },
    };
    const confs = [_]Conformance{
        .{ .protocol = hash_pid, .recv = Type.int },
        .{ .protocol = hashable_pid, .recv = Type.int },
        .{ .protocol = hashable_pid, .recv = Type.structT(4) },
    };
    // The Box ctor id in the composite is struct#1; a bare `Ref`-ctor App is that ctor.
    const box_app = Type.app(try co.intern(gpa, 1, &.{Type.int}, false));
    const ref_ctor: ?u32 = 1;

    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);
    const H = struct {
        fn q(st: []const StructSym, en: []const EnumSym, cf: []const Conformance, cp: *Composite, ty: Type, rc: ?u32, m: *std.AutoHashMapUnmanaged(u64, bool), g: std.mem.Allocator) !bool {
            return hashable(st, en, cf, .{}, cp, ty, 1, 0, rc, null, m, g);
        }
    };
    try testing.expect(try H.q(&structs, &enums, &confs, &co, Type.int, ref_ctor, &memo, gpa)); // scalar (Hashable row)
    try testing.expect(try H.q(&structs, &enums, &confs, &co, Type.structT(0), ref_ctor, &memo, gpa)); // {x,y:int}
    try testing.expect(!try H.q(&structs, &enums, &confs, &co, Type.structT(2), ref_ctor, &memo, gpa)); // {r:Box}
    try testing.expect(!try H.q(&structs, &enums, &confs, &co, box_app, ref_ctor, &memo, gpa)); // bare Ref-ctor App
    try testing.expect(!try H.q(&structs, &enums, &confs, &co, Type.structT(3), ref_ctor, &memo, gpa)); // {f:float}
    try testing.expect(try H.q(&structs, &enums, &confs, &co, Type.enumT(0), ref_ctor, &memo, gpa)); // empty enum
    try testing.expect(try H.q(&structs, &enums, &confs, &co, Type.enumT(1), ref_ctor, &memo, gpa)); // payload enum {int}
    try testing.expect(!try H.q(&structs, &enums, &confs, &co, Type.enumT(2), ref_ctor, &memo, gpa)); // payload enum {float}
    try testing.expect(try H.q(&structs, &enums, &confs, &co, Type.structT(4), ref_ctor, &memo, gpa)); // explicit impl over a box
}

test "generic: a template row meets, fails or misses by its impl-param bounds and arg pattern" {
    const gpa = testing.allocator;
    const show: u32 = 0;
    const into: u32 = 1;
    var co: Composite = .{};
    defer co.deinit(gpa);
    // `Box[T]` (ctor 0) with `impl Box[T has Show] has Show` and `impl Box[T] has Into[T]`.
    const box_t = Type.app(try co.intern(gpa, 0, &.{Type.typeVar(0)}, false));
    const box_int = Type.app(try co.intern(gpa, 0, &.{Type.int}, false));
    const box_bool = Type.app(try co.intern(gpa, 0, &.{Type.bool}, false));
    const box_box_int = Type.app(try co.intern(gpa, 0, &.{box_int}, false));
    const other_int = Type.app(try co.intern(gpa, 1, &.{Type.int}, false));
    const bounds = [_]?u32{show};
    const no_bounds = [_]?u32{null};
    const into_pat = [_]Type{Type.typeVar(0)};
    const rows = [_]TemplateConformance{
        .{ .protocol_id = show, .recv_ctor = 0, .recv_is_enum = false, .recv = box_t, .n_params = 1, .bounds = &bounds },
        .{ .protocol_id = into, .recv_ctor = 0, .recv_is_enum = false, .recv = box_t, .n_params = 1, .bounds = &no_bounds, .protocol_args = &into_pat },
    };
    // Only `int` conforms to Show concretely.
    const confs = [_]Conformance{.{ .protocol = show, .recv = Type.int }};
    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);
    const tb: Tables = .{ .structs = &.{}, .enums = &.{}, .conformances = &confs, .gen = .{ .rows = &rows }, .composite = &co, .gpa = gpa, .bound_protocols = &.{} };

    const met = try generic(tb, &memo, box_int, show, &.{});
    try testing.expect(met == .met);
    met.met.deinit(gpa);
    // The bound recurses through the same row: `Box[Box[int]]` holds because `Box[int]` does.
    const nested = try generic(tb, &memo, box_box_int, show, &.{});
    try testing.expect(nested == .met);
    nested.met.deinit(gpa);
    const unmet = try generic(tb, &memo, box_bool, show, &.{});
    try testing.expect(unmet == .unmet);
    try testing.expect(Type.eql(unmet.unmet.arg, Type.bool));
    try testing.expect((try generic(tb, &memo, other_int, show, &.{})) == .none);
    try testing.expect((try generic(tb, &memo, Type.int, show, &.{})) == .none);
    // The arg pattern grounds through the bound params: `Box[int]` is `Into[int]`, not `Into[bool]`.
    const into_int = try generic(tb, &memo, box_int, into, &.{Type.int});
    try testing.expect(into_int == .met);
    into_int.met.deinit(gpa);
    try testing.expect((try generic(tb, &memo, box_int, into, &.{Type.bool})) == .none);
    // In a bounded template's check, `Box[T]` meets the bound by the axiom on `T`.
    const bp = [_]?u32{show};
    var abstract = tb;
    abstract.bound_protocols = &bp;
    const ax = try generic(abstract, &memo, box_t, show, &.{});
    try testing.expect(ax == .met);
    ax.met.deinit(gpa);
    try testing.expect((try generic(tb, &memo, box_t, show, &.{})) == .unmet);
}

test "a Ref field blocks the Display derive but not Eq" {
    const gpa = testing.allocator;
    const eq_pid: u32 = 0;
    const disp_pid: u32 = 1;
    const box_ft = [_]Type{Type.int};
    const box_fn = [_][]const u8{"0"};
    const h_ft = [_]Type{Type.structT(0)};
    const h_fn = [_][]const u8{"r"};
    const structs = [_]StructSym{
        .{ .decl_node = Ast.none, .name = "Ref", .field_types = @constCast(&box_ft), .field_names = @constCast(&box_fn), .native_family = .ref },
        .{ .decl_node = Ast.none, .name = "H", .field_types = @constCast(&h_ft), .field_names = @constCast(&h_fn) },
    };
    const confs = [_]Conformance{ .{ .protocol = eq_pid, .recv = Type.int }, .{ .protocol = disp_pid, .recv = Type.int } };
    const gen: Generic = .{ .prelude = .{ .protocols = .{ .eq = eq_pid, .display = disp_pid } } };

    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);
    var co: Composite = .{};
    defer co.deinit(gpa);
    try testing.expect(try structural(&structs, &.{}, &confs, gen, Type.structT(1), eq_pid, &memo, gpa, &co, &.{}));
    try testing.expect(!try structural(&structs, &.{}, &confs, gen, Type.structT(1), disp_pid, &memo, gpa, &co, &.{}));
    try testing.expect(!try structural(&structs, &.{}, &confs, gen, Type.structT(0), disp_pid, &memo, gpa, &co, &.{}));
    try testing.expect(Type.eql((try displayBlockingBox(&structs, &.{}, &confs, gen, &co, Type.structT(1), disp_pid, &memo, gpa)).?, Type.structT(0)));

    // An explicit impl on the box wins, as it will for std's `impl Ref[T has Display]`.
    const with_impl = confs ++ [_]Conformance{.{ .protocol = disp_pid, .recv = Type.structT(0) }};
    var memo2: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo2.deinit(gpa);
    try testing.expect(try structural(&structs, &.{}, &with_impl, gen, Type.structT(1), disp_pid, &memo2, gpa, &co, &.{}));
}

test "a function value conforms to Display, and to no other protocol" {
    const gpa = testing.allocator;
    const eq_pid: u32 = 0;
    const disp_pid: u32 = 1;
    const pair_ft = [_]Type{ Type.int, Type.int };
    const pair_fn = [_][]const u8{ "code", "env" };
    const b_ft = [_]Type{ Type.int, Type.structT(0) };
    const b_fn = [_][]const u8{ "n", "f" };
    const structs = [_]StructSym{
        .{ .decl_node = Ast.none, .name = "fn()", .field_types = @constCast(&pair_ft), .field_names = @constCast(&pair_fn), .fn_sig = .{ .params = &.{}, .ret = Type.unit } },
        .{ .decl_node = Ast.none, .name = "B", .field_types = @constCast(&b_ft), .field_names = @constCast(&b_fn) },
    };
    const confs = [_]Conformance{ .{ .protocol = eq_pid, .recv = Type.int }, .{ .protocol = disp_pid, .recv = Type.int } };
    const gen: Generic = .{ .prelude = .{ .protocols = .{ .eq = eq_pid, .display = disp_pid } } };

    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);
    var co: Composite = .{};
    defer co.deinit(gpa);
    try testing.expect(try structural(&structs, &.{}, &confs, gen, Type.structT(0), disp_pid, &memo, gpa, &co, &.{}));
    try testing.expect(try structural(&structs, &.{}, &confs, gen, Type.structT(1), disp_pid, &memo, gpa, &co, &.{}));
    try testing.expect(!try structural(&structs, &.{}, &confs, gen, Type.structT(0), eq_pid, &memo, gpa, &co, &.{}));
    try testing.expect(!try structural(&structs, &.{}, &confs, gen, Type.structT(1), eq_pid, &memo, gpa, &co, &.{}));
}
