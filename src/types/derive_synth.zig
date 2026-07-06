//! The M18-M22 auto-derive synthesis cluster, extracted from the `Typecheck` mega-struct as
//! free functions over `*Typecheck` (Zig has no struct-field privacy, so these read the
//! checker's fields directly). The sole entry point is `synthesizeDerives`, driven once from
//! `monomorphize`; the rest are its resolver/fixpoint helpers.

const std = @import("std");
const Typecheck = @import("../types.zig");
const Type = Typecheck.Type;
const Method = Typecheck.Method;
const DeriveRecipe = Typecheck.DeriveRecipe;
const Derive = @import("../symbols/Derive.zig");

const conforms = Typecheck.conforms;
const resolveConformanceMethod = Typecheck.resolveConformanceMethod;

/// True when `recv` has an EXPLICIT/prelude/Ord-refinement `(pid, recv)` conformance in
/// the live table (the double-fire guard: such a type is never structurally derived).
fn hasConformanceLive(t: *const Typecheck, pid: u32, recv: Type) bool {
    for (t.conformances.items) |c| {
        if (c.protocol == pid and Type.eql(c.recv, recv) and c.protocol_args.len == 0) return true;
    }
    return false;
}

/// The M18 synthesis barrier: turn the (parallel-Pass-C, fn-id-ordered) derive requests
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
    if (t.derive_reqs.items.len == 0) return;
    const eq_pid = t.eq_protocol_id orelse return;
    const eq_name = t.protocols.items[eq_pid].name;
    const ord_pid_opt = t.ord_protocol_id;
    const hash_pid_opt = t.hash_protocol_id;
    const display_pid_opt = t.display_protocol_id;
    const gpa = t.gpa;

    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);

    // ---- (1) Ord fixpoint (M19) -------------------------------------------------------
    // `ord_seen` doubles as the ord-type SET (keyed `writeKey(ord_pid, .ord, ty)`); the Eq
    // fixpoint consults it so an Ord type is NEVER given a separate Eq recipe — a derived
    // `Ord` fills the single `(Eq, T)` slot (M16 precedence), and `==` routes through its
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

    // ---- (2) Eq fixpoint (M18), SKIPPING any Ord type ---------------------------------
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
        if (try ordFills(t, &ord_seen, ord_pid_opt, req.conform_ty)) continue; // Ord fills Eq
        try enqueueDerive(gpa, &eq_seen, &eq_work, eq_pid, .eq, req.conform_ty);
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
            if (hasConformanceLive(t, eq_pid, ft)) continue; // explicit Eq field: reuse its witness
            if (try ordFills(t, &ord_seen, ord_pid_opt, ft)) continue; // Ord fills Eq: the field's cmp witness serves `==`
            if (try conforms(t.structs.items, t.enums.items, t.conformances.items, ft, eq_pid, &memo, gpa, t.composite, &.{}))
                try enqueueDerive(gpa, &eq_seen, &eq_work, eq_pid, .eq, ft);
        }
    }

    // ---- (2b) Hash fixpoint (M20), INDEPENDENT of Eq/Ord (Hash is not a refinement of
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

    if (hash_pid_opt) |hash_pid| {
        for (t.derive_reqs.items) |req| {
            if (req.protocol_id != hash_pid) continue;
            try enqueueDerive(gpa, &hash_seen, &hash_work, hash_pid, .hash, req.conform_ty);
        }
        var hi: usize = 0;
        while (hi < hash_work.items.len) : (hi += 1) {
            comps.clearRetainingCapacity();
            try collectComponentTypes(t, hash_work.items[hi], &comps);
            for (comps.items) |ft| {
                switch (ft.kind) {
                    .@"struct", .@"enum" => {},
                    else => continue,
                }
                if (hasConformanceLive(t, hash_pid, ft)) continue; // explicit Hash field: reuse its witness
                if (try conforms(t.structs.items, t.enums.items, t.conformances.items, ft, hash_pid, &memo, gpa, t.composite, &.{}))
                    try enqueueDerive(gpa, &hash_seen, &hash_work, hash_pid, .hash, ft);
            }
        }
    }

    // ---- (2c) Display fixpoint (M22), INDEPENDENT of Eq/Ord/Hash (Display is not a
    // refinement of any, so there is no `ordFills`-style skip). Seeded from the `print(x)`
    // requests, it chases every aggregate field the same way the other fixpoints do —
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

    // ---- (3) Materialize recipes (names/params/field_witnesses filled after the sort) --
    const ordering_ty: Type = if (t.ordering_enum_id) |oid| Type.enumT(oid) else .{ .kind = .invalid };
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
            .ret = Type.unit,
        });
    }

    // ---- (4) Canonical sort — the SOLE ordering driver (never discovery/thread order) --
    std.mem.sort(DeriveRecipe, t.derives.items, {}, Derive.lessThan);

    // ---- (5) Mint names + params + register the synthetic `Method` for each recipe,
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
            .hash, .display => try gpa.dupe(Type, &[_]Type{d.conform_ty}),
            .eq, .ord => try gpa.dupe(Type, &[_]Type{ d.conform_ty, d.conform_ty }),
        };
        try t.methods.append(gpa, .{
            .recv = d.conform_ty,
            .name = Derive.methodName(d.kind),
            .fn_id = 0,
            .recv_generic = false,
            .protocol_id = d.protocol_id,
            .derive = @intCast(di),
        });
    }
    // Minted names are a pure function of (protocol, kind, type-id); distinct recipes
    // never collide (Debug/ReleaseSafe guard, mirroring the Mono self-collision assert).
    if (std.debug.runtime_safety) {
        for (t.derives.items, 0..) |a, i| {
            for (t.derives.items[i + 1 ..]) |b| std.debug.assert(!std.mem.eql(u8, a.name.?, b.name.?));
        }
    }

    // ---- (6) Resolve each recipe's field witnesses on the LIVE method table (now carrying
    // the synthetic entries), for BOTH structs and enums (enum payloads flattened in
    // variant-decl-then-field order). These fold into the derive fingerprint; the emitter
    // re-resolves from the same table, so the two stay in lockstep.
    for (t.derives.items) |*d| d.field_witnesses = try resolveDeriveFields(t, d.*);
}

/// Enqueue `ty` for derive `(pid, kind)` once, deduped on the canonical recipe key so a
/// repeated request / nested field is synthesized a single time. `seen` OWNS the key bytes.
fn enqueueDerive(gpa: std.mem.Allocator, seen: *std.StringHashMapUnmanaged(void), work: *std.ArrayList(Type), pid: u32, kind: Derive.Kind, ty: Type) !void {
    var kb: std.ArrayList(u8) = .empty;
    defer kb.deinit(gpa);
    try Derive.writeKey(gpa, &kb, pid, kind, ty);
    if (seen.contains(kb.items)) return;
    const owned = try kb.toOwnedSlice(gpa);
    errdefer gpa.free(owned);
    try seen.put(gpa, owned, {});
    try work.append(gpa, ty);
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

/// Resolve one recipe's per-field witnesses (M19): the flattened component fields (see
/// `collectComponentTypes`) each mapped to their `FieldWitness` via `resolveFieldWitness`
/// parameterized by the recipe kind. Returns the borrowed-empty slice
/// for a no-field recipe (empty struct / empty-payload enum) so teardown's `len > 0` free
/// guard stays correct. OWNED outer slice (freed by `freeDeriveEntries`).
fn resolveDeriveFields(t: *Typecheck, d: DeriveRecipe) ![]const Derive.FieldWitness {
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
    };
    return fw;
}

/// Resolve one struct/enum field type to its `FieldWitness` for a derive of `kind`
/// (M18-M22): scalars/str/unit are handled inline by the emitter (`inline_kind`); an
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
    };
    const pid = switch (kind) {
        .eq => t.eq_protocol_id,
        .ord => t.ord_protocol_id,
        .hash => t.hash_protocol_id,
        .display => t.display_protocol_id,
    };
    switch (resolveConformanceMethod(t.methods.items, ft, method, pid, null)) {
        .one => |m| return @unionInit(Derive.FieldWitness, variant, witnessName(t, m)),
        .none, .ambiguous => {},
    }
    if (kind == .eq) switch (resolveConformanceMethod(t.methods.items, ft, "cmp", t.ord_protocol_id, null)) {
        .one => |m| return .{ .cmp_eq = witnessName(t, m) },
        .none, .ambiguous => {},
    };
    return .inline_kind;
}

/// The emitted symbol name a resolved witness `Method` lowers to (M18): a synthetic
/// derive's minted name, a Mono instance's mangled name, else the fn's qualified name.
/// All three outlive codegen (owned by `GraphResult.derives`/`.instances`, or resolve
/// result), so a borrowing `FieldWitness` slice stays valid.
fn witnessName(t: *const Typecheck, m: Method) []const u8 {
    if (m.derive) |di| return t.derives.items[di].name.?;
    if (m.instance) |ii| return t.mono.items[ii].name.?;
    return if (t.gph_fn_names) |fns| fns[m.fn_id] else "";
}
