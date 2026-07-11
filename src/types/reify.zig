//! The App-reification engine, extracted from the `Typecheck` mega-struct as free
//! functions over `*Typecheck` (Zig has no struct-field privacy, so these read the
//! checker's fields directly). `reifyApps` is the whole-program driver run from
//! `monomorphize`: it collects every reachable ground `App`, reifies each to a concrete
//! `structT`/`enumT` in a deterministic order, and rewrites all `.app` occurrences to it.
//! `reifyAppTo` is the single dispatcher/memo/termination-guard that the `layoutEnv`
//! thunk also drives on-demand. `max_instantiation_depth` (the T0017 cap) lives in
//! types.zig, shared with the mono tail.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Mono = @import("../symbols/Mono.zig");
const LayoutEngine = @import("../layout/Engine.zig");
const Typecheck = @import("../types.zig");
const Type = Typecheck.Type;
const VariantSym = LayoutEngine.VariantSym;

/// One entry in the reification sort: a composite `App` index + its precomputed
/// nesting depth + its INDEX-INDEPENDENT structural key. Sorting by `(depth, key)`
/// makes reified-`struct_id` assignment a pure function of source (inner `App`s get
/// lower ids than outer; independent `App`s ordered by structural key), so the ids —
/// hence any `s<id>` mangling downstream — are byte-identical at `-j1`/`-jN`.
const ReifyItem = struct { idx: u32, depth: u32, key: []const u8 };

fn reifyItemLess(_: void, a: ReifyItem, b: ReifyItem) bool {
    if (a.depth != b.depth) return a.depth < b.depth;
    return std.mem.lessThan(u8, a.key, b.key);
}

/// Recursively add `ty`'s composite index (and its nested `App` args) to `set` if it
/// is an `App`. Dedup on the index so each ground `App` is reified once.
fn collectApp(t: *Typecheck, ty: Type, set: *std.AutoArrayHashMapUnmanaged(u32, void)) !void {
    if (!ty.isApp()) return;
    const gop = try set.getOrPut(t.gpa, ty.appIdx());
    if (gop.found_existing) return;
    const e = t.composite.at(ty.appIdx());
    for (e.args) |a| try collectApp(t, a, set);
}

/// If `ty` is an `App`, rewrite it in place to the concrete type it reified to
/// (`structT` for a struct-App, `enumT` for an enum-App). The stored `Type` was decided
/// once at reification, so this never has to re-dispatch on `ctor_is_enum`.
fn rewriteApp(t: *Typecheck, ty: *Type) void {
    if (ty.isApp()) {
        if (t.reify_map.get(ty.appIdx())) |rt| ty.* = rt;
    }
}

/// Reify a composite `App` to the concrete `Type` it stands for — the single
/// dispatcher + memo + termination guard. Memoizes on the composite index so each
/// ground `App` reifies to exactly one concrete type; dispatches to `reifyAppToStruct`
/// (-> `structT`) or `reifyAppToEnum` (-> `enumT`) on the `ctor_is_enum` discriminator.
/// The `reify_depth` counter latches T0017 past `max_instantiation_depth` — an unbounded
/// generic-enum type-growth chain (`enum L[T] { cons(T, L[Box[T]]) }`) recurses through
/// `reifyAppToEnum -> substReify -> reifyAppTo` with a strictly-growing arg, and each
/// level is a fresh distinct id whose args are grounded (so `Composite.appDepth` stays
/// 1 and cannot catch it). Bailing with `.invalid` (UNmemoized, so a re-entry re-bails)
/// truncates the chain deterministically. The memo check runs BEFORE the depth
/// increment, so already-reified inners cost nothing and only true nesting accumulates.
pub fn reifyAppTo(t: *Typecheck, app_idx: u32) error{OutOfMemory}!Type {
    if (t.reify_map.get(app_idx)) |rt| return rt;
    t.reify_depth += 1;
    defer t.reify_depth -= 1;
    if (t.reify_depth > Typecheck.max_instantiation_depth) {
        if (!t.mono_depth_capped) {
            const e = t.composite.at(app_idx);
            const mod = if (e.ctor_is_enum) t.enums.items[e.ctor].mod else t.structs.items[e.ctor].mod;
            _ = t.gphSelect(mod);
            const decl_node = if (e.ctor_is_enum) t.enums.items[e.ctor].decl_node else t.structs.items[e.ctor].decl_node;
            // An AST-less prelude template (Option/Result) has decl_node == Ast.none; anchor
            // the diagnostic at byte 0 rather than OOB-derefing the tree on maxInt(u32).
            const at: u32 = if (decl_node == Ast.none) 0 else t.byteOf(t.tree.nodes[decl_node.int()].main_token);
            try t.sink.emitCode(.T0017, at, "instantiation too deep: generic type nesting exceeds the depth limit");
            t.mono_depth_capped = true;
        }
        return Type.invalid; // NOT memoized: a re-entry re-bails + re-latches (quietly)
    }
    const e = t.composite.at(app_idx);
    if (e.ctor_is_enum) return reifyAppToEnum(t, app_idx);
    return Type.structT(try reifyAppToStruct(t, app_idx));
}

/// Reify a struct-`App` (`Box[int]`) to a fresh concrete `struct_id`, memoized by
/// the `reifyAppTo` dispatcher (which owns the memo check + depth guard). Grounds the
/// `App`'s own args (a nested `App` arg -> its reified concrete type via `reifyAppTo`,
/// bottom-up, so a mixed nest `Box[Either[int,bool]]` grounds its enum arg to `enumT`),
/// mints a fresh struct with the template's field names + the field patterns
/// substituted through the grounded args (`substReify`), and lays it out via
/// `layoutReified` on the live tables. The minted name (`Box$int`) is a pure function
/// of the template name + the concrete args.
fn reifyAppToStruct(t: *Typecheck, app_idx: u32) error{OutOfMemory}!u32 {
    const e = t.composite.at(app_idx);
    var abuf: [8]Type = undefined;
    const cargs: []Type = if (e.args.len <= abuf.len) abuf[0..e.args.len] else try t.gpa.alloc(Type, e.args.len);
    defer if (e.args.len > abuf.len) t.gpa.free(cargs);
    for (e.args, 0..) |a, i| cargs[i] = if (a.isApp()) try reifyAppTo(t, a.appIdx()) else a;

    const tmpl = t.structs.items[e.ctor]; // value copy; its field slices are stable
    const sid: u32 = @intCast(t.structs.items.len);
    try t.reify_map.put(t.gpa, app_idx, Type.structT(sid));

    const name = try Mono.mangle(t.gpa, tmpl.name, cargs);
    try t.reified_names.append(t.gpa, name); // owns the minted name (freed at teardown)
    // Claim `sid` with a placeholder BEFORE `substReify` runs: a generic-struct-typed
    // field (`struct Wrapper[T] { b: Box[T] }`) makes `substReify` recurse into
    // `reifyAppToStruct` for the field's `App`, which mints from `structs.items.len`.
    // Appending here first advances the length past `sid`, so that inner struct lands
    // at `sid+1` instead of stealing `sid`. The `.laying` guard still catches a
    // genuinely self-referential generic (the memo returns this `sid` on re-entry).
    try t.structs.append(t.gpa, .{
        .decl_node = tmpl.decl_node,
        .name = name,
        .mod = tmpl.mod,
        .is_generic = false,
        // Tag the reified instance by the PRELUDE template id (not the mangled name — a
        // user `struct Ref` would mangle to the same `Ref$int`), so the reference
        // predicate recognizes a managed box per-instance. Mirrors `reifyAppToEnum`.
        .native_family = if (t.prelude) |p| p.refFamily(e.ctor) else .none,
    });
    const fnames = try t.gpa.dupe([]const u8, tmpl.field_names);
    errdefer t.gpa.free(fnames);
    const ftypes = try t.gpa.alloc(Type, tmpl.field_types.len);
    errdefer t.gpa.free(ftypes);
    for (tmpl.field_types, 0..) |ft, j| ftypes[j] = try substReify(t, ft, cargs);
    t.structs.items[sid].field_names = fnames;
    t.structs.items[sid].field_types = ftypes;
    try LayoutEngine.layoutReified(t.layoutEnv(), sid);
    return sid;
}

/// Reify an enum-`App` (`Either[int,bool]`) to a fresh concrete `enum_id`,
/// memoized by the `reifyAppTo` dispatcher. Mirrors `reifyAppToStruct`: grounds the
/// `App`'s own args (via `reifyAppTo`), copies the template EnumSym, claims a fresh
/// `enum_id` via a placeholder append (so a nested reify lands past it), substitutes
/// each variant's payload patterns through the grounded args (`substReify`), and lays
/// out the concrete variants via `layoutReifiedEnum` on the live tables BEFORE the enum
/// snapshot, so lower/codegen/Abi see only a plain tagged union. The minted name
/// (`Either$int$bool`) is a pure function of the template name + the concrete args.
fn reifyAppToEnum(t: *Typecheck, app_idx: u32) error{OutOfMemory}!Type {
    const e = t.composite.at(app_idx);
    var abuf: [8]Type = undefined;
    const cargs: []Type = if (e.args.len <= abuf.len) abuf[0..e.args.len] else try t.gpa.alloc(Type, e.args.len);
    defer if (e.args.len > abuf.len) t.gpa.free(cargs);
    for (e.args, 0..) |a, i| cargs[i] = if (a.isApp()) try reifyAppTo(t, a.appIdx()) else a;

    const tmpl = t.enums.items[e.ctor]; // value copy; its variant slices are stable
    const eid: u32 = @intCast(t.enums.items.len);
    try t.reify_map.put(t.gpa, app_idx, Type.enumT(eid));

    const name = try Mono.mangle(t.gpa, tmpl.name, cargs);
    try t.reified_names.append(t.gpa, name); // owns the minted name (freed at teardown)
    // Claim `eid` with a placeholder BEFORE substituting variants: a variant payload
    // that re-applies a generic (`w(Box[T])` / `cons(T, L[Box[T]])`) makes `substReify`
    // recurse + append, which must land PAST `eid` rather than steal it (mirrors
    // reifyAppToStruct's placeholder discipline).
    try t.enums.append(t.gpa, .{
        .decl_node = tmpl.decl_node,
        .name = name,
        .mod = tmpl.mod,
        .is_generic = false,
        // Tag the reified instance by the PRELUDE template id (not the mangled name — a
        // user `enum Option` would mangle to the same `Option$int`), so `lower` recognizes
        // its native inherent methods per-instance.
        .native_family = if (t.prelude) |p| p.optResultFamily(e.ctor) else .none,
    });
    const src_variants = tmpl.variants;
    const variants = try t.gpa.alloc(VariantSym, src_variants.len);
    errdefer t.gpa.free(variants);
    var vbuilt: usize = 0;
    errdefer for (variants[0..vbuilt]) |v| {
        t.gpa.free(v.field_names);
        t.gpa.free(v.field_types);
    };
    for (src_variants, 0..) |sv, vi| {
        const fnames = try t.gpa.dupe([]const u8, sv.field_names);
        errdefer t.gpa.free(fnames);
        const ftypes = try t.gpa.alloc(Type, sv.field_types.len);
        errdefer t.gpa.free(ftypes);
        for (sv.field_types, 0..) |ft, k| ftypes[k] = try substReify(t, ft, cargs);
        variants[vi] = .{ .name = sv.name, .form = sv.form, .field_names = fnames, .field_types = ftypes };
        vbuilt += 1;
    }
    t.enums.items[eid].variants = variants;
    try LayoutEngine.layoutReifiedEnum(t.layoutEnv(), eid);
    return Type.enumT(eid);
}

/// Substitute a template field-type PATTERN through a reified instance's concrete args,
/// always yielding a CONCRETE type: a `type_var(ord)` becomes `cargs[ord]`; a
/// nested `App(c, [pat..])` (a field like `b: Box[T]`) grounds its args, re-interns,
/// and reifies to `structT` (bottom-up, so `layoutReified` only ever sees `structT`);
/// anything else passes through.
fn substReify(t: *Typecheck, ty: Type, cargs: []const Type) error{OutOfMemory}!Type {
    if (ty.isTypeVar()) {
        const ord = ty.typeVarOrd();
        return if (ord < cargs.len) cargs[ord] else Type.invalid;
    }
    if (ty.isApp()) {
        const e = t.composite.at(ty.appIdx());
        var sbuf: [8]Type = undefined;
        const sub: []Type = if (e.args.len <= sbuf.len) sbuf[0..e.args.len] else try t.gpa.alloc(Type, e.args.len);
        defer if (e.args.len > sbuf.len) t.gpa.free(sub);
        for (e.args, 0..) |a, i| sub[i] = try substReify(t, a, cargs);
        const new_idx = try t.internApp(e.ctor, sub, e.ctor_is_enum);
        // Dispatch struct-vs-enum on the interned discriminator: a `Box[T]` field grounds
        // to `structT`, an `Either[T,U]`/`L[Box[T]]` payload to `enumT`. The
        // `reifyAppTo` depth guard makes an unbounded enum type-growth chain terminate.
        return reifyAppTo(t, new_idx);
    }
    return ty;
}

/// Reify every reachable ground `App` to a concrete `structT`/`enumT` and rewrite all
/// `.app` occurrences to it. See the call site in `monomorphize` for why this
/// runs where it does. `nts` is the per-module node_types (`t.gph_node_types`).
pub fn reifyApps(t: *Typecheck, nts: [][]Type) !void {
    // (1) Collect every reachable ground `App` index (+ nested) from the slot sets an
    // `App` can reach lower/codegen/the snapshot through: all module node_types, every
    // non-generic fn's sig (lowered directly), and every instance's node_types + sig.
    var to_reify: std.AutoArrayHashMapUnmanaged(u32, void) = .empty;
    defer to_reify.deinit(t.gpa);
    for (nts) |mnt| for (mnt) |ty| try collectApp(t, ty, &to_reify);
    for (t.fns.items) |f| {
        if (f.isGeneric()) continue; // a template sig carries type_var patterns, never lowered
        for (f.params) |p| try collectApp(t, p, &to_reify);
        try collectApp(t, f.ret, &to_reify);
    }
    // A NON-generic struct with a concrete generic-struct field (`struct S { b:
    // Box[int] }`) carries a ground `App` in its `field_types`; reify + rewrite it so
    // the snapshot/fingerprint/lower see a plain `structT`. A generic TEMPLATE's fields
    // are type_var patterns (never ground, never lowered), so skip it; a reified
    // struct's fields are already `structT`, so `collectApp` is a no-op there.
    for (t.structs.items) |s| {
        if (s.is_generic) continue;
        for (s.field_types) |ft| try collectApp(t, ft, &to_reify);
    }
    // A non-generic enum with a concrete generic-aggregate payload (`enum E {
    // v(Box[int]) }` / `enum E { v(Either[int,bool]) }`) carries a ground `App` in a
    // variant's `field_types`; collect it so the same reify+rewrite erases it before the
    // enum snapshot/fingerprint. A generic ENUM TEMPLATE carries `type_var`/
    // App-over-type_var PATTERNS in its variant payloads (never ground, never lowered) —
    // skip it, exactly as generic structs are skipped above; a reified enum's payloads
    // are already concrete (grounded by `substReify`), so `collectApp` is a no-op there.
    for (t.enums.items) |en| {
        if (en.is_generic) continue;
        for (en.variants) |v| for (v.field_types) |ft| try collectApp(t, ft, &to_reify);
    }
    for (t.mono.items) |inst| {
        for (inst.node_types) |ty| try collectApp(t, ty, &to_reify);
        for (inst.args) |a| try collectApp(t, a, &to_reify);
        for (inst.params) |p| try collectApp(t, p, &to_reify);
        try collectApp(t, inst.ret, &to_reify);
        // A bounded instance's `[T has P]` conformance carries a `conform_ty` copy (a
        // separate Type from `inst.args`), so collecting args alone leaves a ground `App`
        // un-reified there; collect it too, exactly as derive_reqs.conform_ty below.
        for (inst.conformances) |rc| try collectApp(t, rc.conform_ty, &to_reify);
    }
    // A conditional-conformance derive request carries a ground `App` conform_ty
    // (`Box[int] < ..` records `App(Box,[int])`); collect it so the same reify pass mints
    // its concrete `struct_id`, and step (3) rewrites the request to that `structT` — else
    // `synthesizeDerives` (post-reify) would key `Derive.writeKey` off a stale App index.
    for (t.derive_reqs.items) |r| try collectApp(t, r.conform_ty, &to_reify);
    // The captured fallible-char-conversion Results are ground `App`s (`Result[char,ConvErr]`)
    // the synthesis barrier reads AFTER this pass; collect them so their concrete enum id is
    // minted here and step (3) rewrites the capture to that `enumT` (else `synthesizeDerives`
    // would deref a stale App index as `enum_id`).
    if (t.conv_int_char_result) |r| try collectApp(t, r, &to_reify);
    if (t.conv_char_byte_result) |r| try collectApp(t, r, &to_reify);
    if (t.conv_float_int_result) |r| try collectApp(t, r, &to_reify);

    // (2) Reify in (depth, structural-key) order so ids are a pure function of source.
    // Apps reified on-demand during Phase 0b (concrete generic-aggregate fields) are
    // already memoized; the `reifyAppTo` dispatcher returns their existing ids here.
    if (to_reify.count() > 0) {
        const items = try t.gpa.alloc(ReifyItem, to_reify.count());
        defer {
            for (items) |it| t.gpa.free(@constCast(it.key));
            t.gpa.free(items);
        }
        for (to_reify.keys(), 0..) |ai, i| {
            var kb: std.ArrayList(u8) = .empty;
            errdefer kb.deinit(t.gpa);
            try t.composite.writeStructuralKey(t.gpa, Type.app(ai), &kb);
            items[i] = .{ .idx = ai, .depth = t.composite.appDepth(Type.app(ai)), .key = try kb.toOwnedSlice(t.gpa) };
        }
        std.mem.sort(ReifyItem, items, {}, reifyItemLess);
        for (items) |it| _ = try reifyAppTo(t, it.idx);
    }

    // (3) Rewrite every `.app` -> its reified `structT`/`enumT` across the SAME slot set,
    // so no `App` survives into the snapshot / instance table / fn sigs. A field's
    // reified type has the same size as the `App` it laid out from, so offsets are stable.
    for (nts) |mnt| for (mnt) |*ty| rewriteApp(t, ty);
    for (t.fns.items) |*f| {
        if (f.isGeneric()) continue;
        for (f.params) |*p| rewriteApp(t, p);
        rewriteApp(t, &f.ret);
    }
    for (t.structs.items) |s| {
        if (s.is_generic) continue;
        for (s.field_types) |*ft| rewriteApp(t, ft);
    }
    for (t.enums.items) |en| {
        if (en.is_generic) continue; // a template's payload patterns are never lowered
        for (en.variants) |v| for (v.field_types) |*ft| rewriteApp(t, ft);
    }
    for (t.mono.items) |*inst| {
        for (inst.node_types) |*ty| rewriteApp(t, ty);
        for (@constCast(inst.args)) |*a| rewriteApp(t, a);
        for (@constCast(inst.params)) |*p| rewriteApp(t, p);
        rewriteApp(t, &inst.ret);
        for (@constCast(inst.conformances)) |*rc| rewriteApp(t, &rc.conform_ty);
    }
    // rewrite each conditional-conformance derive request's `App` conform_ty to its
    // reified `structT`/`enumT`, so `synthesizeDerives` sees the concrete type.
    for (t.derive_reqs.items) |*r| rewriteApp(t, &r.conform_ty);
    // rewrite the captured conv Results to the concrete `enumT` the synthesis barrier's
    // witness recipes carry (their emitter reads `d.ret.enum_id`).
    if (t.conv_int_char_result) |*r| rewriteApp(t, r);
    if (t.conv_char_byte_result) |*r| rewriteApp(t, r);
    if (t.conv_float_int_result) |*r| rewriteApp(t, r);
}
