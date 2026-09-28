//! The auto-derive synthesis routines, extracted from the `Typecheck` mega-struct as
//! free functions over `*Typecheck` (Zig has no struct-field privacy, so these read the
//! checker's fields directly). The sole entry point is `synthesizeDerives`, driven once from
//! `monomorphize`; the rest are its resolver/fixpoint helpers.

const std = @import("std");
const Typecheck = @import("../types.zig");
const Type = Typecheck.Type;
const Method = Typecheck.Method;
const DeriveRecipe = Typecheck.DeriveRecipe;
const Derive = @import("../symbols/Derive.zig");
const Abi = @import("../codegen/abi/Abi.zig");

const conforms = Typecheck.conform.structural;

/// The sentinel `protocol_id` for a `Trace` recipe. `Trace` is not a source protocol (no
/// row in `t.protocols`), so it borrows a value above every real protocol id — the
/// canonical sort then places every trace recipe LAST, after eq/ord/hash/display/try_into,
/// so no existing recipe's sorted position (and thus its minted name / synthetic id) shifts.
/// No derive path indexes `t.protocols.items[protocol_id]` for a trace recipe (mangle keys
/// off the name string; writeKey/lessThan use the id as an opaque u32).
const trace_pid: u32 = std.math.maxInt(u32);
const resolveConformanceMethod = Typecheck.resolveConformanceMethod;

/// Whether `ty` is a managed box (`Ref[T]`/`gc_array[T]`): a reified struct carrying the
/// reference-family marker. Such a type is never given a structural Eq/Ord witness (its
/// `==` is compared inline by cell identity) and blocks a `Hash` derive. The post-reify
/// predicate over the live struct table; `lower.isRefTy` runs the same authority
/// (`Typecheck.isRefStruct`) over the layout snapshot.
fn isRefType(t: *const Typecheck, ty: Type) bool {
    return Typecheck.isRefStruct(ty, t.structs.items);
}

/// Whether `ty` transitively holds a MANAGED (reference) field — i.e. tracing it is not a
/// no-op. A `Ref`/`gc_array` box component is an immediate `true`; a by-value struct/enum
/// component recurses (a `Ref` is a leaf boundary, NEVER recursed through). Terminates:
/// by-value cycles are already a compile error (`Engine.layoutReferent`), and a `Ref`'s
/// single field is `int`, so the by-value graph is a finite acyclic DAG. `memo` (keyed on the
/// nominal id + the enum flag) collapses the DAG's shared subtrees.
///
/// `count_str` decides whether a `str` component (a managed leaf whose heap buffer is reached
/// only by tracing the enclosing cell) makes the aggregate trace-requiring. The descriptor's
/// erased trace unit — the one the collector actually dispatches through a container element
/// walk — must see `str` as managed (it word-scans the cell, so the buffer pointer is
/// marked). The vestigial natural-derive `Trace` fixpoint (emitted but never called under
/// descriptor-driven tracing) passes `false`, so it synthesizes no dead unit for a str-only
/// aggregate — which would otherwise enlarge a heap-free program's image.
fn needsTrace(t: *Typecheck, ty: Type, count_str: bool, memo: *std.AutoHashMapUnmanaged(u64, bool)) error{OutOfMemory}!bool {
    switch (ty.kind) {
        .@"struct", .@"enum" => {},
        else => return false,
    }
    const key = (@as(u64, ty.nominalId()) << 1) | @intFromBool(ty.kind == .@"enum");
    if (memo.get(key)) |v| return v;
    // Pre-seed `false` so a self-referential probe (should not occur for by-value types, but
    // cheap insurance) terminates rather than recursing forever.
    try memo.put(t.gpa, key, false);

    var comps: std.ArrayList(Type) = .empty;
    defer comps.deinit(t.gpa);
    try collectComponentTypes(t, ty, &comps);
    var result = false;
    for (comps.items) |ft| {
        if (count_str and ft.kind == .str) {
            result = true;
            break;
        }
        switch (ft.kind) {
            .@"struct", .@"enum" => {},
            else => continue,
        }
        if (isRefType(t, ft) or try needsTrace(t, ft, count_str, memo)) {
            result = true;
            break;
        }
    }
    try memo.put(t.gpa, key, result);
    return result;
}

/// Resolve one component field of a `Trace` recipe to its `FieldWitness`: a managed field
/// is the `.trace_mark` boundary; a by-value aggregate that itself holds managed fields has
/// a sibling `.trace` recipe (its presence IS the `needsTrace` signal — no recompute), so
/// call it; everything else is `.inline_kind` (not managed → the emitter no-ops it). Read
/// AFTER all trace names are minted, so the sibling's `name` is live.
fn resolveTraceFieldWitness(t: *const Typecheck, ft: Type) Derive.FieldWitness {
    switch (ft.kind) {
        .@"struct", .@"enum" => {},
        else => return .inline_kind,
    }
    if (isRefType(t, ft)) return .trace_mark;
    for (t.derives.items) |d| {
        if (d.kind == .trace and Type.eql(d.conform_ty, ft)) return .{ .trace_call = d.name.? };
    }
    return .inline_kind;
}

/// True when `recv` has an EXPLICIT/prelude/Ord-refinement `(pid, recv)` conformance in
/// the live table (the double-fire guard: such a type is never structurally derived).
fn hasConformanceLive(t: *const Typecheck, pid: u32, recv: Type) bool {
    for (t.conformances.items) |c| {
        if (c.protocol == pid and Type.eql(c.recv, recv) and c.protocol_args.len == 0) return true;
    }
    return false;
}

/// The synthesis barrier: turn the (parallel-Pass-C, fn-id-ordered) derive requests
/// into the canonical `t.derives` recipe table + synthetic `Method` entries.
///
/// DETERMINISM: the request set is deduped by `Derive.writeKey`, a fixpoint enqueues each
/// aggregate field's own derive, then the whole set is CANONICALLY SORTED
/// (`Derive.lessThan`) BEFORE any name is minted — so the recipes, their synthetic ids/
/// names, and the codegen enumeration order are a pure function of source (never the
/// hashmap/thread order Pass C discovered them in). Field witnesses are resolved ONCE
/// here on the live method table (after ALL synthetic methods are appended), so the
/// emitter never re-resolves and the fingerprint can fold the resolved identity.
pub fn synthesizeDerives(t: *Typecheck) !void {
    // A conv-only program (a fallible char conversion but no `==`/`.hash`/`print` derive)
    // still needs the barrier to run, so gate on BOTH request sources. A descriptor'd
    // Hashable key with no explicit derive also seeds Eq/Hash below (the erased dispatch's
    // witnesses), so an otherwise-derive-free Map program must run the barrier too.
    if (t.derive_reqs.items.len == 0 and t.descriptor_reqs.items.len == 0 and
        t.conv_int_char_result == null and t.conv_char_byte_result == null and
        t.conv_float_int_result == null) return;
    const pre = t.prelude orelse return;
    const eq_pid = pre.protocols.eq orelse return;
    const eq_name = t.protocols.items[eq_pid].name;
    const ord_pid_opt = pre.protocols.ord;
    const hash_pid_opt = pre.protocols.hash;
    const display_pid_opt = pre.protocols.display;
    const gpa = t.gpa;

    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);

    // Ord fixpoint.
    // `ord_seen` doubles as the ord-type SET (keyed `writeKey(ord_pid, .ord, ty)`); the Eq
    // fixpoint consults it so an Ord type is NEVER given a separate Eq recipe — a derived
    // `Ord` fills the single `(Eq, T)` slot (precedence), and `==` routes through its
    // `cmp`. The fixpoint chases struct fields AND every enum variant's payload fields,
    // skipping any field with a live explicit conformance (it uses its own witness).
    var ord_seen: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var it = ord_seen.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        ord_seen.deinit(gpa);
    }
    var ord_work: std.ArrayList(Type) = .empty;
    defer ord_work.deinit(gpa);
    var comps: std.ArrayList(Type) = .empty;
    defer comps.deinit(gpa);

    if (ord_pid_opt) |ord_pid| {
        for (t.derive_reqs.items) |req| {
            if (req.protocol_id != ord_pid) continue;
            try enqueueDerive(gpa, &ord_seen, &ord_work, ord_pid, .ord, req.conform_ty);
        }
        var oi: usize = 0;
        while (oi < ord_work.items.len) : (oi += 1) {
            comps.clearRetainingCapacity();
            try collectComponentTypes(t, ord_work.items[oi], &comps);
            for (comps.items) |ft| {
                switch (ft.kind) {
                    .@"struct", .@"enum" => {},
                    else => continue,
                }
                if (hasConformanceLive(t, ord_pid, ft)) continue; // explicit Ord field: reuse its witness
                if (try conforms(t.structs.items, t.enums.items, t.conformances.items, ft, ord_pid, &memo, gpa, t.composite, &.{}))
                    try enqueueDerive(gpa, &ord_seen, &ord_work, ord_pid, .ord, ft);
            }
        }
    }

    // Descriptor'd Hashable keys with NO explicit `impl` need the DERIVED Hash + Eq witnesses so
    // the erased key dispatch (which resolves the KEY's Hashable hash/eq) has something to call.
    // Seed them into the Eq + Hash worklists below — BEFORE those fixpoints run — so each key's
    // fields are chased identically to an `==`/`.hash()`-driven request. Gated on the SAME
    // `conform.hashable` verdict (over the SAME `t.descriptor_reqs`) that `synthesizeDescriptors`
    // gates the erased unit on, so the recipe set and the erased-unit set are in lockstep: no key
    // gets an erased unit without a resolvable witness. The Ref guard excludes managed-box keys
    // (no phantom recipe / T0030); an explicit-impl key is skipped (it supplies its own witnesses).
    var desc_keys: std.ArrayList(Type) = .empty;
    defer desc_keys.deinit(gpa);
    if (pre.protocols.hashable) |hashable_pid| if (hash_pid_opt) |hash_pid| {
        for (t.descriptor_reqs.items) |k| {
            switch (k.kind) {
                .@"struct", .@"enum" => {},
                else => continue,
            }
            var dup = false;
            for (desc_keys.items) |e| if (Type.eql(e, k)) {
                dup = true;
                break;
            };
            if (dup or hasConformanceLive(t, hashable_pid, k)) continue;
            if (try Typecheck.conform.hashable(t.structs.items, t.enums.items, t.conformances.items, t.composite, k, hashable_pid, hash_pid, pre.ref_struct, pre.gc_array_struct, &memo, gpa))
                try desc_keys.append(gpa, k);
        }
    };

    // Eq fixpoint, SKIPPING any Ord type.
    var eq_seen: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var it = eq_seen.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        eq_seen.deinit(gpa);
    }
    var eq_work: std.ArrayList(Type) = .empty;
    defer eq_work.deinit(gpa);

    for (t.derive_reqs.items) |req| {
        if (req.protocol_id != eq_pid) continue;
        if (isRefType(t, req.conform_ty)) continue; // a Ref is compared inline (cell identity), no witness
        if (try ordFills(t, &ord_seen, ord_pid_opt, req.conform_ty)) continue; // Ord fills Eq
        try enqueueDerive(gpa, &eq_seen, &eq_work, eq_pid, .eq, req.conform_ty);
    }
    for (desc_keys.items) |k| {
        // A descriptor'd key carrying an EXPLICIT `impl has Eq`/`Ord` supplies its own `==`
        // witness (the erased eq dispatch resolves it via `structEqAtSlots`); minting a
        // derived Eq alongside it would either collide into an `.ambiguous` resolve (an
        // internal codegen diagnostic) or, for an Ord-refined key with no explicit eq, register
        // a spurious structural `eq` that silently rebinds the program's `==`. Reuse the live
        // witness — the SAME guard the aggregate-field Eq walk below applies.
        if (hasConformanceLive(t, eq_pid, k)) continue; // explicit Eq: reuse its witness
        if (ord_pid_opt) |op| if (hasConformanceLive(t, op, k)) continue; // explicit Ord: its cmp fills eq
        if (try ordFills(t, &ord_seen, ord_pid_opt, k)) continue; // a derived Ord fills the (Eq,K) slot
        try enqueueDerive(gpa, &eq_seen, &eq_work, eq_pid, .eq, k);
    }
    var ei: usize = 0;
    while (ei < eq_work.items.len) : (ei += 1) {
        comps.clearRetainingCapacity();
        try collectComponentTypes(t, eq_work.items[ei], &comps);
        for (comps.items) |ft| {
            switch (ft.kind) {
                .@"struct", .@"enum" => {},
                else => continue,
            }
            if (isRefType(t, ft)) continue; // a Ref field is compared inline (cell identity), no witness
            if (hasConformanceLive(t, eq_pid, ft)) continue; // explicit Eq field: reuse its witness
            if (try ordFills(t, &ord_seen, ord_pid_opt, ft)) continue; // Ord fills Eq: the field's cmp witness serves `==`
            if (try conforms(t.structs.items, t.enums.items, t.conformances.items, ft, eq_pid, &memo, gpa, t.composite, &.{}))
                try enqueueDerive(gpa, &eq_seen, &eq_work, eq_pid, .eq, ft);
        }
    }

    // Hash fixpoint, INDEPENDENT of Eq/Ord (Hash is not a refinement of
    // either, so there is no `ordFills`-style skip). Seeded from the `hash_pid` requests,
    // it chases every aggregate field the same way the Eq/Ord fixpoints do — struct fields
    // AND every enum variant's payload — skipping any field with a live explicit `Hash`
    // conformance (it uses its own witness). The SAME `collectComponentTypes` order the
    // emitter walks, so the derived hash is `Eq`-consistent by construction.
    var hash_seen: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var it = hash_seen.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        hash_seen.deinit(gpa);
    }
    var hash_work: std.ArrayList(Type) = .empty;
    defer hash_work.deinit(gpa);
    // Parallel to `hash_work`: the site of the request each item was reached from. The Hash
    // fixpoint is the one that can reject a type (a `Ref` field), and must say where.
    var hash_sites: std.ArrayList(?Typecheck.DeriveSite) = .empty;
    defer hash_sites.deinit(gpa);

    if (hash_pid_opt) |hash_pid| {
        for (t.derive_reqs.items) |req| {
            if (req.protocol_id != hash_pid) continue;
            if (try claimDerive(gpa, &hash_seen, hash_pid, .hash, req.conform_ty))
                try pushHash(gpa, &hash_work, &hash_sites, req.conform_ty, req.site);
        }
        for (desc_keys.items) |k| {
            // A key with an EXPLICIT `impl has Hash` supplies its own hash witness (which the
            // erased hash dispatch resolves via `hashAtSlot`); a derived Hash alongside it would
            // collide into an `.ambiguous` resolve. Reuse the live witness — any deterministic
            // hash is consistent with the derived structural eq the map pairs it with.
            if (hasConformanceLive(t, hash_pid, k)) continue; // explicit Hash: reuse its witness
            if (try claimDerive(gpa, &hash_seen, hash_pid, .hash, k))
                try pushHash(gpa, &hash_work, &hash_sites, k, null);
        }
        var hi: usize = 0;
        while (hi < hash_work.items.len) : (hi += 1) {
            const site = hash_sites.items[hi];
            comps.clearRetainingCapacity();
            try collectComponentTypes(t, hash_work.items[hi], &comps);
            for (comps.items) |ft| {
                switch (ft.kind) {
                    .@"struct", .@"enum" => {},
                    else => continue,
                }
                // A managed box has no hashable value — its identity is a heap cell, not a
                // stable key — so a Hash derive over a field that holds one is a compile error.
                if (isRefType(t, ft)) {
                    try emitRefHash(t, site, hash_work.items[hi], ft);
                    continue;
                }
                if (hasConformanceLive(t, hash_pid, ft)) continue; // explicit Hash field: reuse its witness
                if (try conforms(t.structs.items, t.enums.items, t.conformances.items, ft, hash_pid, &memo, gpa, t.composite, &.{}))
                    if (try claimDerive(gpa, &hash_seen, hash_pid, .hash, ft))
                        try pushHash(gpa, &hash_work, &hash_sites, ft, site);
            }
        }
    }

    // Display fixpoint, INDEPENDENT of Eq/Ord/Hash (Display is not a
    // refinement of any, so there is no `ordFills`-style skip). Seeded from the `to_string`
    // / display-derive requests, it chases every aggregate field the same way the other fixpoints do —
    // struct fields AND every enum variant's payload — skipping any field with a live
    // explicit `Display` conformance (it uses its own witness). The SAME
    // `collectComponentTypes` order the emitter walks, so a nested field's `display` witness
    // is a sibling recipe.
    var disp_seen: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var it = disp_seen.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        disp_seen.deinit(gpa);
    }
    var disp_work: std.ArrayList(Type) = .empty;
    defer disp_work.deinit(gpa);

    if (display_pid_opt) |disp_pid| {
        for (t.derive_reqs.items) |req| {
            if (req.protocol_id != disp_pid) continue;
            try enqueueDerive(gpa, &disp_seen, &disp_work, disp_pid, .display, req.conform_ty);
        }
        var wi: usize = 0;
        while (wi < disp_work.items.len) : (wi += 1) {
            comps.clearRetainingCapacity();
            try collectComponentTypes(t, disp_work.items[wi], &comps);
            for (comps.items) |ft| {
                switch (ft.kind) {
                    .@"struct", .@"enum" => {},
                    else => continue,
                }
                if (hasConformanceLive(t, disp_pid, ft)) continue; // explicit Display field: reuse its witness
                if (try conforms(t.structs.items, t.enums.items, t.conformances.items, ft, disp_pid, &memo, gpa, t.composite, &.{}))
                    try enqueueDerive(gpa, &disp_seen, &disp_work, disp_pid, .display, ft);
            }
        }
    }

    // Trace fixpoint, the FOURTH co-product of the one walk. Seeded from the UNION of the
    // four derive worklists intersected with `needsTrace` (a type earns a trace unit as a
    // co-product of being derived), it chases every by-value aggregate component that itself
    // holds a managed field — a `Ref` field is the trace BOUNDARY and is never enqueued
    // (its cell is marked, its pointee never recursed). The SAME `collectComponentTypes`
    // order the other fixpoints walk, so trace's field projection agrees with eq/hash.
    var trace_seen: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var it = trace_seen.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        trace_seen.deinit(gpa);
    }
    var trace_work: std.ArrayList(Type) = .empty;
    defer trace_work.deinit(gpa);
    {
        var tmemo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
        defer tmemo.deinit(gpa);
        for ([_][]const Type{ ord_work.items, eq_work.items, hash_work.items, disp_work.items }) |lst| {
            for (lst) |ty| {
                switch (ty.kind) {
                    .@"struct", .@"enum" => {},
                    else => continue,
                }
                if (try needsTrace(t, ty, false, &tmemo))
                    try enqueueDerive(gpa, &trace_seen, &trace_work, trace_pid, .trace, ty);
            }
        }
        var ti: usize = 0;
        while (ti < trace_work.items.len) : (ti += 1) {
            comps.clearRetainingCapacity();
            try collectComponentTypes(t, trace_work.items[ti], &comps);
            for (comps.items) |ft| {
                switch (ft.kind) {
                    .@"struct", .@"enum" => {},
                    else => continue,
                }
                if (isRefType(t, ft)) continue; // the trace boundary — mark the cell, never recurse
                if (try needsTrace(t, ft, false, &tmemo))
                    try enqueueDerive(gpa, &trace_seen, &trace_work, trace_pid, .trace, ft);
            }
        }
    }

    // Materialize recipes (names/params/field_witnesses filled after the sort).
    const ordering_ty: Type = if (pre.ordering_enum) |oid| Type.enumT(oid) else .{ .kind = .invalid };
    if (ord_pid_opt) |ord_pid| {
        const ord_name = t.protocols.items[ord_pid].name;
        for (ord_work.items) |ty| try t.derives.append(gpa, .{
            .protocol_id = ord_pid,
            .protocol_name = ord_name,
            .kind = .ord,
            .conform_ty = ty,
            .ret = ordering_ty,
        });
    }
    for (eq_work.items) |ty| try t.derives.append(gpa, .{
        .protocol_id = eq_pid,
        .protocol_name = eq_name,
        .kind = .eq,
        .conform_ty = ty,
        .ret = Type.@"bool",
    });
    if (hash_pid_opt) |hash_pid| {
        const hash_name = t.protocols.items[hash_pid].name;
        for (hash_work.items) |ty| try t.derives.append(gpa, .{
            .protocol_id = hash_pid,
            .protocol_name = hash_name,
            .kind = .hash,
            .conform_ty = ty,
            .ret = Type.int,
        });
    }
    if (display_pid_opt) |disp_pid| {
        const disp_name = t.protocols.items[disp_pid].name;
        for (disp_work.items) |ty| try t.derives.append(gpa, .{
            .protocol_id = disp_pid,
            .protocol_name = disp_name,
            .kind = .display,
            .conform_ty = ty,
            .ret = Type.str,
        });
    }

    // Trace recipes. The sentinel `trace_pid` sorts these LAST, so no eq/ord/hash/display
    // recipe's position or minted name shifts.
    for (trace_work.items) |ty| {
        try t.derives.append(gpa, .{
            .protocol_id = trace_pid,
            .protocol_name = "Trace",
            .kind = .trace,
            .conform_ty = ty,
            .ret = Type.unit,
        });
    }

    // The two FALLIBLE char conversions become shared source-less witnesses (ONE each,
    // CALLed per site) so the ~24/~12-cell validate+Result-build never inlines (the
    // frame-overflow fix). Keyed on the `char` struct + separated by KIND; `try_into` has
    // no dispatch wiring, so these get NO `t.methods` row. `ret` is the concrete Result the
    // checker already interned, so the witness reads the byte-identical enum layout. The
    // `try_into` protocol id (10) is higher than eq/ord/hash/display, so the canonical sort
    // places these LAST — no existing recipe's sorted position / minted name shifts.
    if (pre.protocols.try_into) |ti_pid| if (pre.char_struct) |char_id| {
        const ti_name = t.protocols.items[ti_pid].name;
        const cty = Type.structT(char_id);
        if (t.conv_int_char_result) |rty| try t.derives.append(gpa, .{
            .protocol_id = ti_pid,
            .protocol_name = ti_name,
            .kind = .conv_int_char,
            .conform_ty = cty,
            .ret = rty,
        });
        if (t.conv_char_byte_result) |rty| try t.derives.append(gpa, .{
            .protocol_id = ti_pid,
            .protocol_name = ti_name,
            .kind = .conv_char_byte,
            .conform_ty = cty,
            .ret = rty,
        });
    };

    // float -> int rides the same shared-witness path as the char convs, but anchors on
    // `Type.float` (no char involvement) so a float-only program with no `char` struct still
    // mints it. `carriesIntDesc(.float)==false`, so mangle/writeKey accept it -> `TryInto$float_to_int$s0`.
    if (pre.protocols.try_into) |ti_pid| {
        if (t.conv_float_int_result) |rty| try t.derives.append(gpa, .{
            .protocol_id = ti_pid,
            .protocol_name = t.protocols.items[ti_pid].name,
            .kind = .conv_float_int,
            .conform_ty = Type.float,
            .ret = rty,
        });
    }

    // Canonical sort — the SOLE ordering driver (never discovery/thread order).
    std.mem.sort(DeriveRecipe, t.derives.items, {}, Derive.lessThan);

    // Mint names + params + register the synthetic `Method` for each recipe,
    // BEFORE resolving field witnesses (so a nested aggregate field resolves to its sibling
    // recipe's synthetic method). An Ord recipe registers a `cmp` Method{protocol_id=ord};
    // an Eq recipe an `eq` Method{protocol_id=eq}. `derive=index` routes lower/fold to the
    // synthetic unit.
    for (t.derives.items, 0..) |*d, di| {
        d.name = try Derive.mangle(gpa, d.protocol_name, d.kind, d.conform_ty);
        // A `Hash` witness takes ONLY `self` (`hash(self) -> int`); `Eq`/`Ord` are
        // homogeneous 2-ary (`m(self, other) -> _`). The param count feeds the ABI + the
        // fingerprint Sig fold, so it MUST match the emitter's declared param count.
        d.params = switch (d.kind) {
            // `trace` is a 1-ary unit like `hash`: `trace(self) -> ()`.
            .hash, .display, .trace => try gpa.dupe(Type, &[_]Type{d.conform_ty}),
            .eq, .ord => try gpa.dupe(Type, &[_]Type{ d.conform_ty, d.conform_ty }),
            // A conv witness takes the source scalar as a plain `int` (the raw value the old
            // inline consumed): `int_to_char(int) -> Result[char,ConvErr]`,
            // `char_to_byte(int) -> Result[byte,ConvErr]`.
            .conv_int_char, .conv_char_byte => try gpa.dupe(Type, &[_]Type{Type.int}),
            .conv_float_int => try gpa.dupe(Type, &[_]Type{Type.float}),
        };
        // A conv recipe is NOT a conformance method (`try_into` has no dispatch wiring), so
        // it gets no `t.methods` row — the call site scans `t.derives` for it by kind.
        switch (d.kind) {
            // `trace` has no dispatch wiring (it is not a source protocol method), so — like
            // the conv witnesses — it gets no `t.methods` row; the emitter reads the sibling
            // trace unit off the recipe's resolved `field_witnesses`, not the method table.
            .conv_int_char, .conv_char_byte, .conv_float_int, .trace => {},
            else => try t.methods.append(gpa, .{
                .recv = d.conform_ty,
                .name = Derive.methodName(d.kind),
                .fn_id = 0,
                .protocol_id = d.protocol_id,
                .derive = @intCast(di),
            }),
        }
    }
    // Minted names are a pure function of (protocol, kind, type-id); distinct recipes
    // never collide (Debug/ReleaseSafe guard, mirroring the Mono self-collision assert).
    if (std.debug.runtime_safety) {
        for (t.derives.items, 0..) |a, i| {
            for (t.derives.items[i + 1 ..]) |b| std.debug.assert(!std.mem.eql(u8, a.name.?, b.name.?));
        }
    }

    // Resolve each recipe's field witnesses on the LIVE method table (now carrying
    // the synthetic entries), for BOTH structs and enums (enum payloads flattened in
    // variant-decl-then-field order). These fold into the derive fingerprint; the emitter
    // re-resolves from the same table, so the two stay in lockstep.
    for (t.derives.items) |*d| d.field_witnesses = try resolveDeriveFields(t, d.*);
}

/// The descriptor ordering class: scalar (0), struct (1), enum (2). Within a class,
/// scalars sort by a fixed kind rank and aggregates by nominal id — a source-pure,
/// index-independent, `-jN`-stable total order.
fn descClass(ty: Type) u8 {
    return switch (ty.kind) {
        .@"struct" => 1,
        .@"enum" => 2,
        else => 0,
    };
}

fn scalarRank(ty: Type) u8 {
    return switch (ty.kind) {
        .int => 0,
        .bool => 1,
        .str => 2,
        .unit => 3,
        .float => 4,
        .rawptr => 5,
        else => 6,
    };
}

fn descLess(_: void, a: Type, b: Type) bool {
    const ca = descClass(a);
    const cb = descClass(b);
    if (ca != cb) return ca < cb;
    return switch (ca) {
        // A tie on scalar rank means both are `.int` (every other scalar has a unique
        // rank); break it on the sign/width byte so distinct int widths get a stable
        // order — else the unstable sort could reorder them across `-jN`.
        0 => if (scalarRank(a) != scalarRank(b))
            scalarRank(a) < scalarRank(b)
        else
            @as(u8, @bitCast(a.int_desc)) < @as(u8, @bitCast(b.int_desc)),
        else => a.nominalId() < b.nominalId(),
    };
}

/// Resolve the `descriptor_of[T]` requests into the descriptor plan + erased hash/eq
/// witnesses. Runs AFTER `synthesizeDerives` so the natural Hash/Eq recipes + methods the
/// erased struct/enum walk delegates to already exist. Determinism is anchored by the
/// canonical sort + names minted after the sort — never discovery/thread order.
pub fn synthesizeDescriptors(t: *Typecheck) !void {
    if (t.descriptor_reqs.items.len == 0) return;
    const pre = t.prelude orelse return;
    const hash_pid_opt = pre.protocols.hash;
    const gpa = t.gpa;

    // 1) dedup (reified concrete types → structural equality suffices).
    var uniq: std.ArrayList(Type) = .empty;
    defer uniq.deinit(gpa);
    outer: for (t.descriptor_reqs.items) |r| {
        for (uniq.items) |u| if (Type.eql(u, r)) continue :outer;
        try uniq.append(gpa, r);
    }

    // 2) canonical sort (the SOLE ordering driver).
    std.mem.sort(Type, uniq.items, {}, descLess);

    // 3) resolve each: a descriptor entry, plus erased hash+eq units when `Hashable`.
    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);
    // A SEPARATE memo for the trace-ability probe (distinct semantics from `hashable`, though
    // keyed the same way, so it must not share `memo`).
    var tmemo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer tmemo.deinit(gpa);
    for (uniq.items) |ty| {
        const hashable = if (pre.protocols.hashable) |hpid| (if (hash_pid_opt) |hp|
            try Typecheck.conform.hashable(t.structs.items, t.enums.items, t.conformances.items, t.composite, ty, hpid, hp, pre.ref_struct, pre.gc_array_struct, &memo, gpa)
        else
            false) else false;
        try t.descriptor_types.append(gpa, .{ .ty = ty, .hashable = hashable });
        if (hashable) {
            try t.erased_units.append(gpa, .{ .ty = ty, .kind = .hash, .name = try Derive.erasedMangle(gpa, .hash, ty) });
            try t.erased_units.append(gpa, .{ .ty = ty, .kind = .eq, .name = try Derive.erasedMangle(gpa, .eq, ty) });
        }
        // A container stashes `descriptor_of[Elem]`/[K]/[V] in its header; the collector reads
        // that descriptor's `trace_off` to mark the managed object each live cell references.
        // Every MANAGED cell type gets an erased trace unit here (the ONE place its descriptor
        // is minted): a heap `str` buffer, a `Ref`/`gc_array` box pointer, or a by-value
        // aggregate transitively holding one (`needsTrace`). A scalar / managed-free aggregate
        // needs none — its cells are leaves the leaf-marked backing already keeps.
        if (ty.kind == .str or isRefType(t, ty) or try needsTrace(t, ty, true, &tmemo)) {
            try t.erased_units.append(gpa, .{ .ty = ty, .kind = .trace, .name = try Derive.erasedMangle(gpa, .trace, ty) });
        }
    }
}

/// Enqueue `ty` for derive `(pid, kind)` once, deduped on the canonical recipe key so a
/// repeated request / nested field is synthesized a single time. `seen` OWNS the key bytes.
/// `container`'s Hash derive reached a managed-box field. Reported at the expression that
/// demanded the derive; a derive seeded only by a descriptor key has no such site, so it
/// falls back to the box's declaration name.
fn emitRefHash(t: *Typecheck, site: ?Typecheck.DeriveSite, container: Type, ft: Type) !void {
    const sym = t.structs.items[ft.struct_id];
    const at: Typecheck.DeriveSite = site orelse blk: {
        const m = t.graph.mods[sym.mod];
        const tok = m.tree.nodes[sym.decl_node.int()].main_token;
        break :blk .{ .scope = sym.mod, .span = .{ .start = m.tokens[tok].start, .end = m.tokens[tok].end } };
    };
    const prev = t.sink.cur_scope;
    defer t.sink.setScope(prev);
    t.sink.setScope(at.scope);
    try t.sink.emitFmtCodeSpan(.T0030, at.span, "cannot derive 'Hash' for '{s}': Ref-containing type has no auto Hash", .{t.typeName(container)});
}

fn pushHash(gpa: std.mem.Allocator, work: *std.ArrayList(Type), sites: *std.ArrayList(?Typecheck.DeriveSite), ty: Type, site: ?Typecheck.DeriveSite) !void {
    try work.append(gpa, ty);
    try sites.append(gpa, site);
}

fn enqueueDerive(gpa: std.mem.Allocator, seen: *std.StringHashMapUnmanaged(void), work: *std.ArrayList(Type), pid: u32, kind: Derive.Kind, ty: Type) !void {
    if (try claimDerive(gpa, seen, pid, kind, ty)) try work.append(gpa, ty);
}

/// Mark `(pid, kind, ty)` as seen; false when it already was.
fn claimDerive(gpa: std.mem.Allocator, seen: *std.StringHashMapUnmanaged(void), pid: u32, kind: Derive.Kind, ty: Type) !bool {
    var kb: std.ArrayList(u8) = .empty;
    defer kb.deinit(gpa);
    try Derive.writeKey(gpa, &kb, pid, kind, ty);
    if (seen.contains(kb.items)) return false;
    const owned = try kb.toOwnedSlice(gpa);
    errdefer gpa.free(owned);
    try seen.put(gpa, owned, {});
    return true;
}



/// True when `ty` already has (or will have) a DERIVED `Ord` recipe — so its `cmp` fills the
/// single `(Eq, ty)` slot and the Eq fixpoint must not synthesize a separate Eq unit for it.
fn ordFills(t: *Typecheck, ord_seen: *const std.StringHashMapUnmanaged(void), ord_pid_opt: ?u32, ty: Type) !bool {
    const ord_pid = ord_pid_opt orelse return false;
    var kb: std.ArrayList(u8) = .empty;
    defer kb.deinit(t.gpa);
    try Derive.writeKey(t.gpa, &kb, ord_pid, .ord, ty);
    return ord_seen.contains(kb.items);
}

/// Append `ty`'s component field types in the CANONICAL derive walk order (the order the
/// emitter and the witness-fold both use): a struct's fields in declaration order; an enum's
/// every-variant payload fields in variant-decl-then-field order. A scalar `ty` appends none.
fn collectComponentTypes(t: *Typecheck, ty: Type, out: *std.ArrayList(Type)) !void {
    switch (ty.kind) {
        .@"struct" => if (ty.struct_id < t.structs.items.len)
            try out.appendSlice(t.gpa, t.structs.items[ty.struct_id].field_types),
        .@"enum" => if (ty.enum_id < t.enums.items.len)
            for (t.enums.items[ty.enum_id].variants) |v| try out.appendSlice(t.gpa, v.field_types),
        else => {},
    }
}

/// Resolve one recipe's per-field witnesses: the flattened component fields (see
/// `collectComponentTypes`) each mapped to their `FieldWitness` via `resolveFieldWitness`
/// parameterized by the recipe kind. Returns the borrowed-empty slice
/// for a no-field recipe (empty struct / empty-payload enum) so teardown's `len > 0` free
/// guard stays correct. OWNED outer slice (freed by `freeDeriveEntries`).
fn resolveDeriveFields(t: *Typecheck, d: DeriveRecipe) ![]const Derive.FieldWitness {
    // A conv witness is hand-emitted (not a per-field structural walk), so it has no field
    // witnesses — the empty slice keeps teardown's `len > 0` free guard correct.
    switch (d.kind) {
        .conv_int_char, .conv_char_byte, .conv_float_int => return &.{},
        else => {},
    }
    var ftys: std.ArrayList(Type) = .empty;
    defer ftys.deinit(t.gpa);
    try collectComponentTypes(t, d.conform_ty, &ftys);
    if (ftys.items.len == 0) return &.{};
    const fw = try t.gpa.alloc(Derive.FieldWitness, ftys.items.len);
    errdefer t.gpa.free(fw);
    for (ftys.items, 0..) |ft, i| fw[i] = switch (d.kind) {
        .eq => resolveFieldWitness(t, .eq, ft),
        .ord => resolveFieldWitness(t, .ord, ft),
        .hash => resolveFieldWitness(t, .hash, ft),
        .display => resolveFieldWitness(t, .display, ft),
        .trace => resolveTraceFieldWitness(t, ft),
        .conv_int_char, .conv_char_byte, .conv_float_int => unreachable, // guarded above
    };
    return fw;
}

/// Resolve one struct/enum field type to its `FieldWitness` for a derive of `kind`:
/// scalars/str/unit are handled inline by the emitter (`inline_kind`); an
/// aggregate field dispatches to its per-protocol witness (a sibling derive, a Mono
/// instance method, or a user `impl` fn). An `Eq` derive is the one two-lookup case — it
/// falls back to the field's `cmp` witness when it has no `eq` (`==` as `cmp(..) ==
/// Ordering.eq`). Read only on the LIVE method table AFTER all synthetic entries are
/// appended, so a sibling derive resolves correctly. A witness miss on a conforming field
/// is unreachable; degrade to `inline_kind`.
fn resolveFieldWitness(t: *const Typecheck, comptime kind: Derive.Kind, ft: Type) Derive.FieldWitness {
    switch (ft.kind) {
        .@"struct", .@"enum" => {},
        else => return .inline_kind,
    }
    const method, const variant = switch (kind) {
        .eq => .{ "eq", "eq_call" },
        .ord => .{ "cmp", "cmp_call" },
        .hash => .{ "hash", "hash_call" },
        .display => .{ "display", "display_call" },
        // `trace` is resolved by `resolveTraceFieldWitness`, never this generic path.
        .trace, .conv_int_char, .conv_char_byte, .conv_float_int => unreachable, // never instantiated (guarded in resolveDeriveFields)
    };
    const pr = Typecheck.gatherPreludeIds(t);
    const pid = switch (kind) {
        .eq => pr.eq,
        .ord => pr.ord,
        .hash => pr.hash,
        .display => pr.display,
        .trace, .conv_int_char, .conv_char_byte, .conv_float_int => unreachable,
    };
    switch (resolveConformanceMethod(t.methods.items, ft, method, pid, null)) {
        .one => |m| return @unionInit(Derive.FieldWitness, variant, witnessName(t, m)),
        .none, .ambiguous => {},
    }
    if (kind == .eq) switch (resolveConformanceMethod(t.methods.items, ft, "cmp", pr.ord, null)) {
        .one => |m| return .{ .cmp_eq = witnessName(t, m) },
        .none, .ambiguous => {},
    };
    return .inline_kind;
}

/// The emitted symbol name a resolved witness `Method` lowers to: a synthetic
/// derive's minted name, a Mono instance's mangled name, else the fn's qualified name.
/// All three outlive codegen (owned by `GraphResult.derives`/`.instances`, or resolve
/// result), so a borrowing `FieldWitness` slice stays valid.
fn witnessName(t: *const Typecheck, m: Method) []const u8 {
    if (m.derive) |di| return t.derives.items[di].name.?;
    if (m.instance) |ii| return t.mono.items[ii].name.?;
    return if (t.gph_fn_names) |fns| fns[m.fn_id] else "";
}
