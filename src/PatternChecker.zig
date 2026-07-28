const std = @import("std");
const Ast = @import("ast/Ast.zig");
const LayoutEngine = @import("layout/Engine.zig");
const Type = @import("layout/Type.zig").Type;
const VariantSym = LayoutEngine.VariantSym;
const ControlFlow = @import("ControlFlow.zig");

// The pattern/match subsystem, carved out of `BodyChecker` as free functions over
// a `*BodyChecker`. `substTy` stays in BodyChecker (it is shared by the struct/method
// paths too); the accessors these functions reach for are `pub` on BodyChecker for the
// same reason, and the compiler enforces that pub-ness.
const BC = @import("BodyChecker.zig");
const BodyChecker = BC.BodyChecker;

/// Coverage state for a match, by scrutinee kind. Enum: a per-variant seen bitmap.
/// Bool: which of true/false a literal arm has covered. Int: nothing (an infinite
/// domain — exhaustiveness only via `_`).
const Cov = union(enum) {
    @"enum": []bool,
    @"bool": *BoolCov,
    int,
};
const BoolCov = struct { t: bool = false, f: bool = false };

/// Whether the prior UNGUARDED arms already cover every value the scrutinee can take.
/// Enum: every variant seen. Bool: both cases seen. Int: only a wildcard (its domain
/// is infinite). An arm reached while this holds can never match.
fn matchSaturated(cov: Cov, has_wildcard: bool) bool {
    if (has_wildcard) return true;
    return switch (cov) {
        .@"enum" => |sv| for (sv) |s| {
            if (!s) break false;
        } else true,
        .@"bool" => |b| b.t and b.f,
        .int => false,
    };
}

pub fn typeOfMatch(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    const st = try bc.typeOf(n.lhs);
    const arms = Ast.rangeSlice(bc.tree, (n.rhs).int());
    if (st.kind == .invalid) {
        // Poison-absorb: still walk arms (bodies may have their own errors) but
        // don't emit a scrutinee or exhaustiveness error.
        for (arms) |arm_idx| _ = try bc.typeOf(Ast.armHeaderAt(bc.tree, (bc.tree.nodes[(arm_idx).int()].rhs).int()).body);
        bc.node_types[(node_idx).int()] = .invalid;
        return .invalid;
    }
    // A generic-enum instance scrutinee is an `App` (reified to `enumT` in the mono
    // tail); its coverage runs over the template enum's variants.
    const enum_id = bc.scrutEnumId(st);
    if (enum_id == null and st.kind != .int and st.kind != .bool) {
        for (arms) |arm_idx| _ = try bc.typeOf(Ast.armHeaderAt(bc.tree, (bc.tree.nodes[(arm_idx).int()].rhs).int()).body);
        try bc.sink.emitFmt(bc.byteOf(n.main_token), "match scrutinee must be an enum, int, or bool, got {s}", .{bc.typeName(st)});
        bc.node_types[(node_idx).int()] = .invalid;
        return .invalid;
    }

    var seen: []bool = &.{};
    var bool_cov: BoolCov = .{};
    var cov: Cov = if (enum_id) |eid| blk: {
        seen = try bc.gpa.alloc(bool, bc.model.enums[eid].variants.len);
        @memset(seen, false);
        break :blk .{ .@"enum" = seen };
    } else switch (st.kind) {
        .bool => .{ .@"bool" = &bool_cov },
        else => .int,
    };
    defer if (enum_id != null) bc.gpa.free(seen);

    var has_wildcard = false;
    var warned = false;
    var result: Type = Type.never;
    for (arms) |arm_idx| {
        const arm = bc.tree.nodes[(arm_idx).int()];
        const h = Ast.armHeaderAt(bc.tree, (arm.rhs).int());
        const guarded = h.guard != Ast.none;
        // Tested BEFORE this arm contributes, so the arm that COMPLETES coverage
        // (and the required int `_`) is reachable; only an arm reached while already
        // saturated by prior unguarded arms warns. Argument-free so per-generic-instance
        // rechecks dedupe. First one only — no cascade.
        if (!warned and matchSaturated(cov, has_wildcard)) {
            try bc.sink.emitFmtCode(.W0006, bc.byteOf(bc.tree.nodes[(arm.lhs).int()].main_token), "unreachable match arm; every value is already matched by an earlier arm", .{});
            warned = true;
        }
        try checkPattern(bc, arm.lhs, st, &cov, &has_wildcard, !guarded);
        if (guarded) {
            const gt = try bc.typeOf(h.guard);
            if (gt.kind != .invalid and gt.kind != .bool)
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(h.guard).int()].main_token), "match guard must be bool, got {s}", .{bc.typeName(gt)});
        }
        const body_ty0 = try bc.typeOfExpected(h.body, bc.expected);
        const body_ty: Type = if (armDiverges(bc, h.body)) Type.never else body_ty0;
        result = try bc.merge(n.main_token, result, body_ty);
    }
    if (!has_wildcard) switch (cov) {
        .@"enum" => |sv| {
            const e = bc.model.enums[enum_id.?];
            var names: std.ArrayList(u8) = .empty;
            defer names.deinit(bc.gpa);
            var n_missing: usize = 0;
            for (e.variants, 0..) |v, i| if (!sv[i]) {
                if (n_missing != 0) try names.appendSlice(bc.gpa, ", ");
                try names.append(bc.gpa, '\'');
                try names.appendSlice(bc.gpa, v.name);
                try names.append(bc.gpa, '\'');
                n_missing += 1;
            };
            if (n_missing > 0)
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "non-exhaustive match: missing {s} {s}; add the missing arm(s) or a '_' arm", .{ if (n_missing == 1) "variant" else "variants", names.items });
        },
        .bool => |bcov| if (!(bcov.t and bcov.f))
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "non-exhaustive match: bool requires both true and false (or '_')", .{}),
        .int => try bc.sink.emitFmt(bc.byteOf(n.main_token), "non-exhaustive match: int match requires '_'", .{}),
    };
    bc.node_types[(node_idx).int()] = result;
    return result;
}

// Pattern irrefutability lives in `ControlFlow.zig` too (pure walks over the
// tree + enum table); the pattern checker below delegates through these thin
// wrappers. `variantPayloadIrrefutable` is the one that takes a resolved
// `VariantSym` directly (the caller already has it).

fn irrefutable(bc: *const BodyChecker, pat_idx: Ast.Index, ty: Type) bool {
    return ControlFlow.irrefutable(bc.cflow(), pat_idx, ty);
}

fn variantPayloadIrrefutable(bc: *const BodyChecker, pat_idx: Ast.Index, variant: VariantSym) bool {
    return ControlFlow.variantPayloadIrrefutable(bc.cflow(), pat_idx, variant);
}

fn armDiverges(bc: *const BodyChecker, node_idx: Ast.Index) bool {
    return ControlFlow.armDiverges(bc.cflow(), node_idx);
}

fn checkPattern(bc: *BodyChecker, pat_idx: Ast.Index, expected: Type, cov: *Cov, has_wildcard: *bool, count_cov: bool) error{OutOfMemory}!void {
    const pat = bc.tree.nodes[(pat_idx).int()];
    switch (pat.tag) {
        .pattern_wildcard => if (count_cov) {
            has_wildcard.* = true;
        },
        .pattern_binding => {
            // Record the type this binding matched AGAINST on its own node. The
            // binding's slot is SHARED across or-pattern alternatives (Resolve), so
            // the slot type is overwritten and can't reveal a `.A(x) | .B(x)` type
            // divergence; the per-node matched type can (read by `collectBindings`).
            bc.node_types[(pat_idx).int()] = expected;
            if (pat.rhs == Ast.none) {
                // Bind-whole: type by value; a top-level bare binding is irrefutable.
                if (bc.resolutions[(pat_idx).int()] == .local) try bc.setSlot(bc.resolutions[(pat_idx).int()].local, expected);
                if (count_cov) has_wildcard.* = true;
            } else {
                try checkPattern(bc, pat.rhs, expected, cov, has_wildcard, count_cov);
            }
        },
        .pattern_literal => {
            // Type a numeric pattern via the shared literal helper, adopting the
            // scrutinee as the expected width: a narrow-int scrutinee (`match x:int8`)
            // adopts int8 + range-checks the pattern literal, while a non-integer
            // scrutinee yields platform `int` so the mismatch diagnostic below still fires.
            const lt: Type = if (bc.tokens[pat.main_token].tag == .number)
                try bc.typeNumericLiteral(pat.main_token, expected)
            else
                Type.@"bool";
            if (expected.kind != .invalid and !Type.eql(lt, expected))
                try bc.sink.emitFmt(bc.byteOf(pat.main_token), "literal pattern type {s} does not match scrutinee {s}", .{ bc.typeName(lt), bc.typeName(expected) });
            // A bool literal records its case toward coverage; int never covers.
            if (count_cov) switch (cov.*) {
                .bool => |bcov| {
                    if (lt.kind == .bool) {
                        if (std.mem.eql(u8, bc.nameText(pat.main_token), "true")) bcov.t = true else bcov.f = true;
                    }
                },
                else => {},
            };
        },
        .pattern_or => {
            for (Ast.rangeSlice(bc.tree, (pat.lhs).int())) |a| try checkPattern(bc, a, expected, cov, has_wildcard, count_cov);
            try checkOrBindings(bc, pat_idx);
        },
        .pattern_variant => try checkVariantPattern(bc, pat_idx, expected, cov, has_wildcard, count_cov),
        else => {},
    }
}

fn checkVariantPattern(bc: *BodyChecker, pat_idx: Ast.Index, expected: Type, cov: *Cov, has_wildcard: *bool, count_cov: bool) error{OutOfMemory}!void {
    const pat = bc.tree.nodes[(pat_idx).int()];
    // A generic-enum instance scrutinee is an `App`; its enum id + instance
    // type-args come off the composite entry. A plain `enumT` has no type-args, so
    // `substTy(..., &.{})` below is the identity — byte-identical to the earlier behavior.
    const enum_id = bc.scrutEnumId(expected) orelse {
        if (expected.kind != .invalid)
            try bc.sink.emitFmt(bc.byteOf(pat.main_token), "variant pattern on a non-enum scrutinee {s}", .{bc.typeName(expected)});
        return;
    };
    const targs: []const Type = if (expected.isApp()) bc.composite.at(expected.appIdx()).args else &.{};
    const e = bc.model.enums[enum_id];
    // A qualified `N.V` pattern: the type-name must name the scrutinee enum.
    if (pat.lhs != Ast.none) {
        const tname = bc.nameText(bc.tree.nodes[(pat.lhs).int()].main_token);
        if (bc.activeEnumMap().get(tname)) |qid| {
            if (qid != enum_id)
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(pat.lhs).int()].main_token), "pattern enum '{s}' does not match scrutinee '{s}'", .{ tname, e.name });
        } else {
            try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(pat.lhs).int()].main_token), "'{s}' is not an enum type", .{tname});
        }
    }
    const vname = bc.nameText(pat.main_token);
    var vi: ?usize = null;
    for (e.variants, 0..) |v, i| {
        if (std.mem.eql(u8, v.name, vname)) {
            vi = i;
            break;
        }
    }
    const variant = if (vi) |i| blk: {
        // Cover variant i only when this arm counts AND variant i's payload is fully
        // matched (`.V(_)`/`.V(x)` cover it; `.V(0)` or `.V(.W(x))` over a multi-variant
        // inner enum do not — caught by the type-aware payload check). For a generic
        // instance the payload types must be SUBSTITUTED through `targs` first so the
        // irrefutability decision sees concrete types (a bind/wildcard is irrefutable
        // regardless, so `Either[int,bool]`'s `.left(n)`/`.right(_)` cover correctly).
        if (count_cov and try variantPayloadIrrefutableSubst(bc, pat_idx, e.variants[i], targs)) cov.@"enum"[i] = true;
        break :blk e.variants[i];
    } else {
        try bc.emitNoVariant(bc.byteOf(pat.main_token), e.name, vname, e.variants);
        return;
    };
    const binders = if (pat.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (pat.rhs).int());
    switch (variant.form) {
        .unit => {
            if (binders.len != 0)
                try bc.sink.emitFmt(bc.byteOf(pat.main_token), "unit variant '{s}.{s}' binds no payload", .{ e.name, vname });
        },
        .tuple => {
            if (binders.len != variant.field_types.len) {
                try bc.sink.emitFmt(bc.byteOf(pat.main_token), "variant '{s}.{s}' binds {d} value(s), got {d}", .{ e.name, vname, variant.field_types.len, binders.len });
                return;
            }
            for (binders, variant.field_types) |b_idx, fty_pat| {
                try checkPattern(bc, b_idx, BC.substTy(bc, fty_pat, targs), cov, has_wildcard, false);
            }
        },
        .@"struct" => {
            for (binders) |b_idx| {
                const b = bc.tree.nodes[(b_idx).int()];
                // A struct binding's SOURCE field name is the rename source (lhs),
                // or the bound name itself when punning.
                const src_name = if (b.lhs != Ast.none) bc.nameText(bc.tree.nodes[(b.lhs).int()].main_token) else bc.nameText(b.main_token);
                var fty: Type = .invalid;
                var found = false;
                for (variant.field_names, 0..) |dn, j| {
                    if (std.mem.eql(u8, dn, src_name)) {
                        fty = BC.substTy(bc, variant.field_types[j], targs);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    try bc.sink.emitFmt(bc.byteOf(b.main_token), "no field '{s}' in '{s}.{s}'", .{ src_name, e.name, vname });
                }
                // The carrier IS a pattern_binding: if it has a sub-pattern, match
                // the field against it; else bind the whole field by value.
                try checkPattern(bc, b_idx, fty, cov, has_wildcard, false);
            }
        },
    }
}

/// App-aware `irrefutable`. A non-`App` type delegates verbatim to
/// `irrefutable` (= `ControlFlow.irrefutable`), so every scalar / plain-`enumT` /
/// `structT` payload is byte-identical. For an `App` payload (a substituted
/// generic-enum/struct instance — the case `ControlFlow` cannot resolve because its
/// `Ctx` has no composite table) we handle the pattern here: a nested variant/or
/// pattern resolves the enum via `scrutEnumId` + the App's own type-args and recurses.
/// Terminates: it descends the FINITE sub-pattern tree, not the (possibly cyclic) type.
fn irrefutableTy(bc: *BodyChecker, pat_idx: Ast.Index, ty: Type) error{OutOfMemory}!bool {
    if (!ty.isApp()) return irrefutable(bc, pat_idx, ty);
    const pat = bc.tree.nodes[(pat_idx).int()];
    return switch (pat.tag) {
        .pattern_wildcard => true,
        .pattern_binding => pat.rhs == Ast.none or try irrefutableTy(bc, pat.rhs, ty),
        .pattern_literal => false,
        .pattern_or => try orCoversTyApp(bc, pat_idx, ty),
        .pattern_variant => blk: {
            // A single-variant enum instance is the only variant pattern that can be
            // total (the tag test cannot fail); its payload must then be irrefutable.
            const eid = bc.scrutEnumId(ty) orelse break :blk false;
            const e = bc.model.enums[eid];
            if (e.variants.len != 1) break :blk false;
            break :blk try variantPayloadIrrefutableSubst(bc, pat_idx, e.variants[0], bc.composite.at(ty.appIdx()).args);
        },
        else => false,
    };
}

/// `ControlFlow.orCoversType` mirrored, App-aware (payloads substituted through the
/// instance's `targs` before the per-variant irrefutability check).
fn orCoversTyApp(bc: *BodyChecker, or_idx: Ast.Index, ty: Type) error{OutOfMemory}!bool {
    const alts = Ast.rangeSlice(bc.tree, (bc.tree.nodes[(or_idx).int()].lhs).int());
    for (alts) |a| if (try irrefutableTy(bc, a, ty)) return true;
    const eid = bc.scrutEnumId(ty) orelse return false;
    const e = bc.model.enums[eid];
    const targs = bc.composite.at(ty.appIdx()).args;
    var seen = [_]bool{false} ** ControlFlow.max_cover_variants;
    if (e.variants.len > seen.len) return false;
    for (alts) |a| {
        const ap = bc.tree.nodes[(a).int()];
        if (ap.tag != .pattern_variant) continue;
        const vname = bc.nameText(ap.main_token);
        for (e.variants, 0..) |v, i| {
            if (std.mem.eql(u8, v.name, vname) and try variantPayloadIrrefutableSubst(bc, a, v, targs)) seen[i] = true;
        }
    }
    for (e.variants, 0..) |_, i| if (!seen[i]) return false;
    return true;
}

/// `variantPayloadIrrefutable` over a variant whose payload types have been
/// SUBSTITUTED through the enum instance's `targs`. For `targs.len == 0` (a
/// non-generic / plain-`enumT` scrutinee) this delegates to the borrowed-variant
/// call, byte-identical to the earlier behavior. For a generic instance each binder is checked via
/// the App-aware `irrefutableTy` against its substituted field type — this is
/// what makes a nested variant/or pattern over a substituted generic-enum-instance
/// (`App`) payload compute correctly rather than always-refutable.
fn variantPayloadIrrefutableSubst(bc: *BodyChecker, pat_idx: Ast.Index, variant: VariantSym, targs: []const Type) error{OutOfMemory}!bool {
    if (targs.len == 0) return variantPayloadIrrefutable(bc, pat_idx, variant);
    const pat = bc.tree.nodes[(pat_idx).int()];
    const binders = if (pat.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (pat.rhs).int());
    switch (variant.form) {
        .unit => return binders.len == 0,
        .tuple => {
            if (binders.len != variant.field_types.len) return false;
            for (binders, variant.field_types) |b, fpat| {
                if (!try irrefutableTy(bc, b, BC.substTy(bc, fpat, targs))) return false;
            }
            return true;
        },
        .@"struct" => {
            for (binders) |b_idx| {
                const b = bc.tree.nodes[(b_idx).int()];
                const src = if (b.lhs != Ast.none) bc.nameText(bc.tree.nodes[(b.lhs).int()].main_token) else bc.nameText(b.main_token);
                var fty: Type = .invalid;
                for (variant.field_names, 0..) |dn, j| if (std.mem.eql(u8, dn, src)) {
                    fty = BC.substTy(bc, variant.field_types[j], targs);
                    break;
                };
                if (!try irrefutableTy(bc, b_idx, fty)) return false;
            }
            return true;
        },
    }
}

fn checkOrBindings(bc: *BodyChecker, or_idx: Ast.Index) error{OutOfMemory}!void {
    const alts = Ast.rangeSlice(bc.tree, (bc.tree.nodes[(or_idx).int()].lhs).int());
    if (alts.len < 2) return;
    var first_map: std.StringHashMapUnmanaged(Type) = .empty;
    defer first_map.deinit(bc.gpa);
    try collectBindings(bc, alts[0], &first_map);
    var ok = true;
    for (alts[1..]) |alt| {
        var m: std.StringHashMapUnmanaged(Type) = .empty;
        defer m.deinit(bc.gpa);
        try collectBindings(bc, alt, &m);
        if (m.count() != first_map.count()) {
            ok = false;
        } else {
            var it = m.iterator();
            while (it.next()) |entry| {
                const want = first_map.get(entry.key_ptr.*) orelse {
                    ok = false;
                    break;
                };
                if (!Type.eql(want, entry.value_ptr.*)) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) break;
    }
    if (!ok)
        try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(or_idx).int()].main_token), "or-pattern alternatives must bind the same names and types", .{});
}

fn collectBindings(bc: *BodyChecker, pat_idx: Ast.Index, out: *std.StringHashMapUnmanaged(Type)) error{OutOfMemory}!void {
    const pat = bc.tree.nodes[(pat_idx).int()];
    switch (pat.tag) {
        .pattern_binding => {
            const name = bc.nameText(pat.main_token);
            // The PER-NODE matched type (set in checkPattern), NOT the shared slot
            // type — so two alternatives binding the same name at different field
            // types are seen as different and rejected.
            const ty: Type = bc.node_types[(pat_idx).int()];
            try out.put(bc.gpa, name, ty);
            if (pat.rhs != Ast.none) try collectBindings(bc, pat.rhs, out);
        },
        .pattern_variant => if (pat.rhs != Ast.none)
            for (Ast.rangeSlice(bc.tree, (pat.rhs).int())) |c| try collectBindings(bc, c, out),
        .pattern_or => for (Ast.rangeSlice(bc.tree, (pat.lhs).int())) |a| try collectBindings(bc, a, out),
        else => {},
    }
}

const testing = std.testing;

test "matchSaturated: wildcard short-circuits; enum needs every variant; bool needs both; int only via wildcard" {
    // A wildcard saturates regardless of the per-kind coverage state.
    try testing.expect(matchSaturated(.int, true));
    var one_unseen = [_]bool{false};
    try testing.expect(matchSaturated(.{ .@"enum" = &one_unseen }, true));

    // Enum: saturated iff every variant bit is set; an empty variant set is vacuously so.
    var all_seen = [_]bool{ true, true };
    var partial = [_]bool{ true, false };
    try testing.expect(matchSaturated(.{ .@"enum" = &all_seen }, false));
    try testing.expect(!matchSaturated(.{ .@"enum" = &partial }, false));
    try testing.expect(matchSaturated(.{ .@"enum" = &.{} }, false));

    // Bool: both cases required.
    var both = BoolCov{ .t = true, .f = true };
    var one = BoolCov{ .t = true, .f = false };
    try testing.expect(matchSaturated(.{ .@"bool" = &both }, false));
    try testing.expect(!matchSaturated(.{ .@"bool" = &one }, false));

    // Int: an infinite domain — only a wildcard can saturate it.
    try testing.expect(!matchSaturated(.int, false));
}
