//! `derive_emit`: the structural auto-derive IR emitter. A SMALL outward interface —
//! one `lower` dispatcher over four kind entry points — hiding a LARGE layout-walking
//! implementation (Eq/Ord/Hash/Display over structs and payload enums). Split out of
//! `lower.zig` so the derive bodies are directly unit-testable and the shared
//! scalar/control-flow lowering substrate stays a peer.
//!
//! DEPENDENCY IS ONE-WAY: `derive_emit → lower`. It borrows the mutable `L.Builder`
//! one-function state, the shared witness/str/print/`brTo`/`finishFn` emitters, and the
//! frozen `Ordering`/hash constants from `lower.zig`; `lower.zig` never imports back, so
//! there is no cycle. The two external callers (`Codegen`, `Engine`) invoke `lower` here.
//!
//! DETERMINISM: every emitter is PURE of `(recipe, layouts, method table)` — a fixed
//! field/variant-order walk that hands out ids monotonically and reads no map — so a
//! double-lower is byte-identical under `-jN` (what VERIFY relies on).

const std = @import("std");
const L = @import("../lower.zig");
const Ir = @import("../ir/Ir.zig");
const Typecheck = @import("../types.zig");
const Link = @import("../link/Link.zig");
const Derive = @import("../symbols/Derive.zig");
const Diagnostic = @import("../diagnostics/Diagnostic.zig").Diagnostic;
const arith = @import("../opt/arith.zig");
const testing = std.testing;

/// The `Hash` derive's field multiplier (the FNV-64 prime), re-materialized at each
/// `hashCombine` use (a pure iconst the opt folds), so the emitter reads no shared state.
/// Lives beside its sole user; the paired `hash_seed` (and the frozen-constant rationale)
/// stay `pub` in `lower.zig` because the builtin `.hash()` path shares that seed. MUST fit
/// in i64 — the fold wraps via the IR's `*%`/`+%` (never traps).
const hash_mult: i64 = 0x100000001B3;

/// Struct/enum equality of two operands already MATERIALIZED into slots `lslot`/`rslot`
///: resolve the `Eq` witness and call `witness(lslot, rslot) -> bool`, or (an
/// Ord-only type: the Ord-refinement filled `(Eq,T)` but added no `eq` method) call the
/// `cmp` witness, read the returned `Ordering` tag, and compare `== eq`. The
/// slot-operand sibling of `lowerStructEq`/`lowerCmpDiscriminant`, so the auto-derive
/// emitter's aggregate FIELD path stays in lockstep with the top-level `==` (a wrong
/// witness here would be a `conforms`/emitter mismatch). Returns a bool value.
fn structEqAtSlots(b: *L.Builder, ty: Typecheck.Type, lslot: Ir.SlotId, rslot: Ir.SlotId) error{OutOfMemory}!Ir.ValueId {
    switch (Typecheck.resolveConformanceMethod(b.in.methods, ty, "eq", b.in.prelude_ids.eq, null)) {
        .one => |m| {
            const callee = L.witnessCallee(b, m);
            const args = try b.gpa.alloc(Ir.Operand, 2);
            errdefer b.gpa.free(args);
            args[0] = .{ .slot = lslot };
            args[1] = .{ .slot = rslot };
            return try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, Typecheck.Type.@"bool");
        },
        .none, .ambiguous => {},
    }
    switch (Typecheck.resolveConformanceMethod(b.in.methods, ty, "cmp", b.in.prelude_ids.ord, null)) {
        .one => |m| {
            // `==` as `cmp(a,b) == Ordering.eq`: the witness returns `Ordering`, whose ret
            // carries the enum type (the ret_slot ABI + `get_tag` layout). A DERIVED `cmp`
            // (`fn_id == 0`) reads it from the recipe via `witnessRet` (derive-first).
            const ret_ty = L.witnessRet(b, m);
            const callee = L.witnessCallee(b, m);
            const args = try b.gpa.alloc(Ir.Operand, 2);
            args[0] = .{ .slot = lslot };
            args[1] = .{ .slot = rslot };
            const rslot_ord = try b.addSlot(ret_ty);
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = rslot_ord } }, null);
            const base = try b.emit(.{ .slot_addr = rslot_ord }, Typecheck.Type.int);
            const tag = try b.emit(.{ .get_tag = base }, Typecheck.Type.int);
            const k = try b.emit(.{ .iconst = L.ord_eq }, Typecheck.Type.int);
            return try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = tag, .rhs = k } }, Typecheck.Type.@"bool");
        },
        .none, .ambiguous => {
            // The synthesis barrier proved every aggregate field conforms before emitting
            // this unit, so a miss here is an internal invariant break — note-and-drop to
            // stay well-formed rather than miscompile.
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Eq: no witness for an aggregate field in lower" });
            b.had_error = true;
            return try b.emit(.{ .bconst = false }, Typecheck.Type.@"bool");
        },
    }
}

/// The bool eq of struct field `i` (at byte `off`, type `fty`) between the two receiver
/// bases: int/bool inline `icmp eq`; `str` via the `{ptr,len}` byte-loop
/// (`strEqAtPtrs`); a struct/enum field is copied into fresh temp slots (a `field_addr`
/// is a ptr VALUE, not a slot operand — the ABI needs the field's own slot) then routed
/// through `structEqAtSlots`. Unit fields are rejected by T0007, so never occur.
fn deriveFieldEq(b: *L.Builder, fty: Typecheck.Type, off: u32, self_base: Ir.ValueId, other_base: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    switch (fty.kind) {
        .int, .bool => {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            const lv = try b.emit(.{ .load = .{ .addr = la, .ty = fty } }, fty);
            const ra = try b.emit(.{ .field_addr = .{ .base = other_base, .off = off, .ty = fty } }, int_ty);
            const rv = try b.emit(.{ .load = .{ .addr = ra, .ty = fty } }, fty);
            return try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = lv, .rhs = rv } }, Typecheck.Type.@"bool");
        },
        .str => {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            const ra = try b.emit(.{ .field_addr = .{ .base = other_base, .off = off, .ty = fty } }, int_ty);
            return try L.strEqAtPtrs(b, la, ra);
        },
        .@"struct", .@"enum" => {
            const lslot = try b.addSlot(fty);
            const ld = try b.emit(.{ .slot_addr = lslot }, int_ty);
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = ld, .src = la, .ty = fty } }, null);
            const rslot = try b.addSlot(fty);
            const rd = try b.emit(.{ .slot_addr = rslot }, int_ty);
            const ra = try b.emit(.{ .field_addr = .{ .base = other_base, .off = off, .ty = fty } }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = rd, .src = ra, .ty = fty } }, null);
            return try structEqAtSlots(b, fty, lslot, rslot);
        },
        else => {
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Eq: unsupported field type in lower" });
            b.had_error = true;
            return try b.emit(.{ .bconst = false }, Typecheck.Type.@"bool");
        },
    }
}

/// The enum tag-dispatch ladder shared by every payload-enum derive: walk variants in fixed
/// decl order, gating each on `icmp .eq tag_val, vi`; the last variant is the else-fallthrough.
/// `perVariant(b, ctx, vi)` emits that variant's body (each derive delivers its own result to a
/// join). Kept in one place so Eq/Cmp/Hash/Display can never disagree on variant walk order.
fn emitVariantLadder(
    b: *L.Builder,
    e: Typecheck.EnumLayout,
    tag_val: Ir.ValueId,
    ctx: anytype,
    comptime perVariant: anytype,
) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    for (e.variants, 0..) |_, vi| {
        const last = vi + 1 == e.variants.len;
        if (last) {
            try perVariant(b, ctx, vi);
            break;
        }
        const vk = try b.emit(.{ .iconst = @intCast(vi) }, int_ty);
        const is_vi = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = tag_val, .rhs = vk } }, bool_ty);
        const body = try b.addBlock();
        const next = try b.addBlock();
        b.setTerm(.{ .cond_br = .{ .cond = is_vi, .t = body, .f = next } });
        b.switchTo(body);
        try perVariant(b, ctx, vi);
        b.switchTo(next);
    }
}

/// Payload-enum structural `Eq`: equal iff the discriminants match AND, for that
/// variant, every payload field is equal. A `get_tag` compare gates a tag-dispatch ladder
/// (fixed variant-decl order) where each variant's payload fields multiply-accumulate to a
/// bool; every path delivers the bool through the join's merge param. Pure of `(layout,
/// method table)` — a fixed walk handing ids monotonically — so a double-lower is identical.
fn deriveEnumEq(b: *L.Builder, cty: Typecheck.Type, self_base: Ir.ValueId, other_base: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    const e = b.in.enum_layouts[cty.enum_id];

    const lt = try b.emit(.{ .get_tag = self_base }, int_ty);
    const rt = try b.emit(.{ .get_tag = other_base }, int_ty);
    const tags_eq = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = lt, .rhs = rt } }, bool_ty);

    const join = try b.addBlock();
    const merge = try b.addParam(join, bool_ty);
    const dispatch = try b.addBlock();
    const false_blk = try b.addBlock();
    b.setTerm(.{ .cond_br = .{ .cond = tags_eq, .t = dispatch, .f = false_blk } });

    b.switchTo(false_blk);
    const fv = try b.emit(.{ .bconst = false }, bool_ty);
    try L.brTo(b, join, .{ .value = fv });

    b.switchTo(dispatch);
    try emitVariantLadder(b, e, lt, .{ .e = e, .self_base = self_base, .other_base = other_base, .join = join }, struct {
        fn f(bb: *L.Builder, c: anytype, vi: usize) error{OutOfMemory}!void {
            try emitVariantPayloadEq(bb, c.e, vi, c.self_base, c.other_base, c.join);
        }
    }.f);

    b.switchTo(join);
    return merge;
}

/// The bool payload-equality of variant `vi` (at absolute payload offsets), delivered to
/// `join`: `true` for an empty variant, else a multiply-accumulate over the payload fields.
fn emitVariantPayloadEq(b: *L.Builder, e: Typecheck.EnumLayout, vi: usize, self_base: Ir.ValueId, other_base: Ir.ValueId, join: Ir.BlockId) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    const v = e.variants[vi];
    if (v.field_types.len == 0) {
        const tv = try b.emit(.{ .bconst = true }, bool_ty);
        try L.brTo(b, join, .{ .value = tv });
        return;
    }
    var acc = try b.emit(.{ .iconst = 1 }, int_ty);
    for (v.field_types, v.offsets) |fty, poff| {
        const feq = try deriveFieldEq(b, fty, e.payload_off + poff, self_base, other_base);
        acc = try b.emit(.{ .mul = .{ .lhs = acc, .rhs = feq } }, int_ty);
    }
    const zero = try b.emit(.{ .iconst = 0 }, int_ty);
    const res = try b.emit(.{ .icmp = .{ .cc = .ne, .lhs = acc, .rhs = zero } }, bool_ty);
    try L.brTo(b, join, .{ .value = res });
}

/// The branch-free 3-way discriminant (0=lt/1=eq/2=gt) of two int/bool VALUES:
/// `(lhs > rhs) - (lhs < rhs) + 1`. `unsigned` selects the unsigned magnitude conds:
/// a derived `Ord` on a `uint64` field must compare unsigned or a high-bit-set value
/// mis-orders as negative. Signed conds are correct for signed ints, bools (0/1), and
/// the small non-negative enum tags.
fn threeWayInt(b: *L.Builder, lv: Ir.ValueId, rv: Ir.ValueId, unsigned: bool) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    const gt = try b.emit(.{ .icmp = .{ .cc = if (unsigned) .ugt else .gt, .lhs = lv, .rhs = rv } }, bool_ty);
    const lt = try b.emit(.{ .icmp = .{ .cc = if (unsigned) .ult else .lt, .lhs = lv, .rhs = rv } }, bool_ty);
    const diff = try b.emit(.{ .sub = .{ .lhs = gt, .rhs = lt } }, int_ty);
    const one = try b.emit(.{ .iconst = 1 }, int_ty);
    return try b.emit(.{ .add = .{ .lhs = diff, .rhs = one } }, int_ty);
}

/// The int 3-way `Ord` discriminant (0=lt/1=eq/2=gt) of field `i` (at byte `off`, type
/// `fty`) between the two receiver bases: int/bool via the branch-free `threeWayInt`;
/// `str` via the `{ptr,len}` lexicographic byte-loop (`strCmpAtPtrs`); a struct/enum field is
/// copied into fresh temp slots then routed through `cmpAtSlots`. Unit fields are rejected by
/// T0007, so never occur. Mirrors `deriveFieldEq`.
fn deriveFieldCmp(b: *L.Builder, fty: Typecheck.Type, off: u32, self_base: Ir.ValueId, other_base: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    switch (fty.kind) {
        .int, .bool => {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            const lv = try b.emit(.{ .load = .{ .addr = la, .ty = fty } }, fty);
            const ra = try b.emit(.{ .field_addr = .{ .base = other_base, .off = off, .ty = fty } }, int_ty);
            const rv = try b.emit(.{ .load = .{ .addr = ra, .ty = fty } }, fty);
            return try threeWayInt(b, lv, rv, fty.isUnsignedInt());
        },
        .str => {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            const ra = try b.emit(.{ .field_addr = .{ .base = other_base, .off = off, .ty = fty } }, int_ty);
            return try L.strCmpAtPtrs(b, la, ra);
        },
        .@"struct", .@"enum" => {
            const lslot = try b.addSlot(fty);
            const ld = try b.emit(.{ .slot_addr = lslot }, int_ty);
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = ld, .src = la, .ty = fty } }, null);
            const rslot = try b.addSlot(fty);
            const rd = try b.emit(.{ .slot_addr = rslot }, int_ty);
            const ra = try b.emit(.{ .field_addr = .{ .base = other_base, .off = off, .ty = fty } }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = rd, .src = ra, .ty = fty } }, null);
            return try cmpAtSlots(b, fty, lslot, rslot);
        },
        else => {
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Ord: unsupported field type in lower" });
            b.had_error = true;
            return try b.emit(.{ .iconst = L.ord_eq }, int_ty);
        },
    }
}

/// The int 3-way `Ord` discriminant of two operands already MATERIALIZED into slots:
/// resolve the `cmp` witness, call `cmp(lslot, rslot) -> Ordering`, and `get_tag` its result.
/// The slot-operand sibling of `lowerCmpDiscriminant`, so the auto-derive Ord emitter's
/// aggregate FIELD path stays in lockstep with the top-level `<`. Ret sized via `witnessRet`
/// (a DERIVED `cmp` witness has `fn_id == 0`). Returns an int value.
fn cmpAtSlots(b: *L.Builder, ty: Typecheck.Type, lslot: Ir.SlotId, rslot: Ir.SlotId) error{OutOfMemory}!Ir.ValueId {
    switch (Typecheck.resolveConformanceMethod(b.in.methods, ty, "cmp", b.in.prelude_ids.ord, null)) {
        .one => |m| {
            const ret_ty = L.witnessRet(b, m);
            const callee = L.witnessCallee(b, m);
            const args = try b.gpa.alloc(Ir.Operand, 2);
            args[0] = .{ .slot = lslot };
            args[1] = .{ .slot = rslot };
            const ord_slot = try b.addSlot(ret_ty);
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = ord_slot } }, null);
            const base = try b.emit(.{ .slot_addr = ord_slot }, Typecheck.Type.int);
            return try b.emit(.{ .get_tag = base }, Typecheck.Type.int);
        },
        .none, .ambiguous => {
            // The synthesis barrier proved every aggregate field conforms before emitting
            // this unit, so a miss here is an internal invariant break — note-and-drop.
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Ord: no cmp witness for an aggregate field in lower" });
            b.had_error = true;
            return try b.emit(.{ .iconst = L.ord_eq }, Typecheck.Type.int);
        },
    }
}

/// Emit a lexicographic short-circuit chain over `ftys` (at `base_off + offs[i]`), delivering
/// the deciding 3-way discriminant to `join`: the FIRST non-`eq` field decides; a tie
/// falls through to the next; the LAST field's cmp is the answer regardless. An empty field
/// list delivers `eq`. Reused for a struct (base_off 0) and a variant payload (base_off =
/// `payload_off`). Pure of `(layout, method table)`, so `--verify`-stable.
fn deriveLexChain(b: *L.Builder, ftys: []const Typecheck.Type, offs: []const u32, base_off: u32, self_base: Ir.ValueId, other_base: Ir.ValueId, join: Ir.BlockId) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    if (ftys.len == 0) {
        const eqc = try b.emit(.{ .iconst = L.ord_eq }, int_ty);
        try L.brTo(b, join, .{ .value = eqc });
        return;
    }
    for (ftys, offs, 0..) |fty, off, i| {
        const c = try deriveFieldCmp(b, fty, base_off + off, self_base, other_base);
        if (i + 1 == ftys.len) {
            // Last field: its cmp is the whole answer whether eq or not.
            try L.brTo(b, join, .{ .value = c });
            return;
        }
        const eqk = try b.emit(.{ .iconst = L.ord_eq }, int_ty);
        const is_eq = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = c, .rhs = eqk } }, bool_ty);
        const cont = try b.addBlock();
        const decide = try b.addBlock();
        b.setTerm(.{ .cond_br = .{ .cond = is_eq, .t = cont, .f = decide } });
        // decide: this field's cmp decides the ordering.
        b.switchTo(decide);
        try L.brTo(b, join, .{ .value = c });
        // cont: fields so far tied; keep comparing.
        b.switchTo(cont);
    }
}

/// Payload-enum structural `Ord`: compare discriminants (case-declaration order) first;
/// on equal tag compare that variant's payload lexicographically — the SE-0266 total order.
/// Delivers the deciding 3-way discriminant to `join`. A tag-dispatch ladder (fixed
/// variant-decl order) mirrors `deriveEnumEq`.
fn deriveEnumCmp(b: *L.Builder, cty: Typecheck.Type, self_base: Ir.ValueId, other_base: Ir.ValueId, join: Ir.BlockId) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    const e = b.in.enum_layouts[cty.enum_id];

    const st = try b.emit(.{ .get_tag = self_base }, int_ty);
    const ot = try b.emit(.{ .get_tag = other_base }, int_ty);
    const tag_disc = try threeWayInt(b, st, ot, false);
    const tags_eq = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = st, .rhs = ot } }, bool_ty);

    const dispatch = try b.addBlock();
    const tag_decide = try b.addBlock();
    b.setTerm(.{ .cond_br = .{ .cond = tags_eq, .t = dispatch, .f = tag_decide } });

    b.switchTo(tag_decide);
    try L.brTo(b, join, .{ .value = tag_disc });

    b.switchTo(dispatch);
    try emitVariantLadder(b, e, st, .{ .e = e, .self_base = self_base, .other_base = other_base, .join = join }, struct {
        fn f(bb: *L.Builder, c: anytype, vi: usize) error{OutOfMemory}!void {
            const v = c.e.variants[vi];
            try deriveLexChain(bb, v.field_types, v.offsets, c.e.payload_off, c.self_base, c.other_base, c.join);
        }
    }.f);
}

// The source-less auto-derive emitters (`Eq`/`Ord`/`Hash`/`Display`) share a determinism
// contract: each is PURE of `(recipe, layouts, method table)` — a fixed field/variant-order
// walk that hands out ids monotonically and reads no map, so a double-lower is byte-identical
// under `-jN`.

/// Lower a SOURCE-LESS auto-derive `Eq` unit: the spike's layout-walking emitter.
/// Two params (the two receiver values, by slot), a single straight-line
/// multiply-accumulate over the struct's fields (`acc *= field_eq`, then
/// `ret = acc != 0`) — no `cond_br` except the self-contained str/aggregate sub-graphs,
/// so it is maximally `--verify`-stable — or, for an empty-payload enum, `get_tag(self)
/// == get_tag(other)`.
pub fn lower(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    d: Derive.Derive,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    return switch (d.kind) {
        .eq => lowerDeriveEq(gpa, in, d, sym, out_diags),
        .ord => lowerDeriveOrd(gpa, in, d, sym, out_diags),
        .hash => lowerDeriveHash(gpa, in, d, sym, out_diags),
        .display => lowerDeriveDisplay(gpa, in, d, sym, out_diags),
    };
}

fn lowerDeriveEq(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    d: Derive.Derive,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    const cty = d.conform_ty;

    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = bool_ty, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_self = try b.addSlot(cty);
    try params.append(gpa, p_self);
    const p_other = try b.addSlot(cty);
    try params.append(gpa, p_other);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.ret_param = try b.addParam(exit, bool_ty);
    b.blocks.items[exit].term = .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    const self_base = try b.emit(.{ .slot_addr = p_self }, int_ty);
    const other_base = try b.emit(.{ .slot_addr = p_other }, int_ty);

    const result: Ir.ValueId = switch (cty.kind) {
        .@"struct" => blk: {
            const layout = b.in.layouts[cty.struct_id];
            var acc = try b.emit(.{ .iconst = 1 }, int_ty);
            for (layout.field_types, layout.offsets) |fty, off| {
                const feq = try deriveFieldEq(&b, fty, off, self_base, other_base);
                acc = try b.emit(.{ .mul = .{ .lhs = acc, .rhs = feq } }, int_ty);
            }
            const zero = try b.emit(.{ .iconst = 0 }, int_ty);
            break :blk try b.emit(.{ .icmp = .{ .cc = .ne, .lhs = acc, .rhs = zero } }, bool_ty);
        },
        .@"enum" => blk: {
            const e = b.in.enum_layouts[cty.enum_id];
            var any_payload = false;
            for (e.variants) |v| if (v.field_types.len != 0) {
                any_payload = true;
                break;
            };
            if (!any_payload) {
                // Empty-payload enum: equal iff the discriminants match (`get_tag` at off 0).
                const lt = try b.emit(.{ .get_tag = self_base }, int_ty);
                const rt = try b.emit(.{ .get_tag = other_base }, int_ty);
                break :blk try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = lt, .rhs = rt } }, bool_ty);
            }
            // Payload enum: equal iff the tags match AND, for that variant, every
            // payload field is equal. The tag-dispatch ladder + join deliver the bool.
            break :blk try deriveEnumEq(&b, cty, self_base, other_base);
        },
        else => blk: {
            // Unreachable: the synthesis barrier only authorizes struct/enum recipes.
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Eq: unsupported conform type in lower" });
            b.had_error = true;
            break :blk try b.emit(.{ .bconst = false }, bool_ty);
        },
    };

    if (!b.termSet()) try L.brTo(&b, exit, .{ .value = result });
    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

/// Lower a SOURCE-LESS auto-derive `Ord` unit: a layout-walking emitter returning the
/// prelude `Ordering` value. Two params (the two receivers, by slot). A shared `join` block
/// carries the deciding 3-way discriminant; a struct emits a lexicographic short-circuit chain
/// over its fields (declaration order), a payload enum compares discriminants then the equal
/// variant's payload lexicographically (SE-0266 total order). At `join` the discriminant is
/// stored as the `Ordering` value's tag (offset 0) and returned via the exit param (aggregate
/// sret ABI, exactly as a user `-> Ordering` cmp).
fn lowerDeriveOrd(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    d: Derive.Derive,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    const int_ty = Typecheck.Type.int;
    const cty = d.conform_ty;
    const ord_ty = d.ret; // the prelude `Ordering` enum

    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = ord_ty, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_self = try b.addSlot(cty);
    try params.append(gpa, p_self);
    const p_other = try b.addSlot(cty);
    try params.append(gpa, p_other);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.ret_param = try b.addParam(exit, ord_ty);
    b.blocks.items[exit].term = .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    const self_base = try b.emit(.{ .slot_addr = p_self }, int_ty);
    const other_base = try b.emit(.{ .slot_addr = p_other }, int_ty);

    // A shared join whose one int param is the deciding 3-way discriminant. Every decision
    // path branches here; the discriminant IS the `Ordering` tag (lt=0/eq=1/gt=2).
    const join = try b.addBlock();
    const disc = try b.addParam(join, int_ty);

    switch (cty.kind) {
        .@"struct" => {
            const layout = b.in.layouts[cty.struct_id];
            try deriveLexChain(&b, layout.field_types, layout.offsets, 0, self_base, other_base, join);
        },
        .@"enum" => try deriveEnumCmp(&b, cty, self_base, other_base, join),
        else => {
            // Unreachable: the synthesis barrier only authorizes struct/enum recipes.
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Ord: unsupported conform type in lower" });
            b.had_error = true;
            const eqc = try b.emit(.{ .iconst = L.ord_eq }, int_ty);
            try L.brTo(&b, join, .{ .value = eqc });
        },
    }

    // join: materialize the `Ordering` value (its tag = the discriminant, stored @0) and
    // return it through the exit param.
    b.switchTo(join);
    const ord_slot = try b.addSlot(ord_ty);
    const ord_addr = try b.emit(.{ .slot_addr = ord_slot }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = ord_addr, .val = disc, .ty = int_ty } }, null);
    if (!b.termSet()) try L.brTo(&b, exit, .{ .slot = ord_slot });

    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

/// Fold one more field/tag into the running hash: `h := h*MULT + add_val`. The
/// single mixing primitive of the structural `Hash` derive; `MULT` is re-materialized at
/// each use (a pure iconst the opt folds), so the emitter reads no shared state and a
/// double-lower is byte-identical. Wraps past i64 via the IR's wrapping `*`/`+` (never
/// traps), so a large accumulator is safe.
fn hashCombine(b: *L.Builder, h: Ir.ValueId, add_val: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const mult = try b.emit(.{ .iconst = hash_mult }, int_ty);
    const hm = try b.emit(.{ .mul = .{ .lhs = h, .rhs = mult } }, int_ty);
    return try b.emit(.{ .add = .{ .lhs = hm, .rhs = add_val } }, int_ty);
}

/// The int hash of an aggregate operand already MATERIALIZED into slot `slot`:
/// resolve the `hash` witness and call `witness(slot) -> int`. The slot-operand sibling of
/// the top-level derive, so a nested aggregate FIELD stays in lockstep with the callee's own
/// derived unit. A miss is unreachable for a conforming field (the synthesis barrier proved
/// it) — note-and-drop rather than miscompile. Returns an int value.
fn hashAtSlot(b: *L.Builder, ty: Typecheck.Type, slot: Ir.SlotId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    switch (Typecheck.resolveConformanceMethod(b.in.methods, ty, "hash", b.in.prelude_ids.hash, null)) {
        .one => |m| {
            const callee = L.witnessCallee(b, m);
            const args = try b.gpa.alloc(Ir.Operand, 1);
            errdefer b.gpa.free(args);
            args[0] = .{ .slot = slot };
            return try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, int_ty);
        },
        .none, .ambiguous => {
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Hash: no hash witness for an aggregate field in lower" });
            b.had_error = true;
            return try b.emit(.{ .iconst = 0 }, int_ty);
        },
    }
}

/// The int hash of struct field `i` (at byte `off`, type `fty`) of receiver base `self_base`
///: int/bool hash to their own loaded VALUE (identity — an int is its own hash, a bool
/// is 0/1); `str` via the `{ptr,len}` byte polynomial (`hashStrAtPtr`); a struct/enum field is
/// copied into a fresh temp slot then routed through `hashAtSlot`. Unit fields are rejected by
/// T0007, so never occur. Mirrors `deriveFieldEq`/`deriveFieldCmp` (single receiver — hash is
/// 1-ary).
fn deriveFieldHash(b: *L.Builder, fty: Typecheck.Type, off: u32, self_base: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    switch (fty.kind) {
        .int, .bool => {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            return try b.emit(.{ .load = .{ .addr = la, .ty = fty } }, fty);
        },
        .str => {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            return try L.hashStrAtPtr(b, la);
        },
        .@"struct", .@"enum" => {
            const slot = try b.addSlot(fty);
            const d = try b.emit(.{ .slot_addr = slot }, int_ty);
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = d, .src = la, .ty = fty } }, null);
            return try hashAtSlot(b, fty, slot);
        },
        else => {
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Hash: unsupported field type in lower" });
            b.had_error = true;
            return try b.emit(.{ .iconst = 0 }, int_ty);
        },
    }
}

/// Fold variant `vi`'s active payload into the running hash `h0` and deliver the result to
/// `join`: an empty variant delivers `h0` unchanged; else each payload field
/// multiply-accumulates. Mirrors `emitVariantPayloadEq`.
fn emitVariantPayloadHash(b: *L.Builder, e: Typecheck.EnumLayout, vi: usize, self_base: Ir.ValueId, h0: Ir.ValueId, join: Ir.BlockId) error{OutOfMemory}!void {
    const v = e.variants[vi];
    var h = h0;
    for (v.field_types, v.offsets) |fty, poff| {
        const fh = try deriveFieldHash(b, fty, e.payload_off + poff, self_base);
        h = try hashCombine(b, h, fh);
    }
    try L.brTo(b, join, .{ .value = h });
}

/// Payload-enum structural `Hash`: fold the discriminant into the fixed seed
/// (`combine(seed, tag)`), then dispatch on the tag (case-declaration order) to fold the
/// ACTIVE variant's payload into that base. A tag-dispatch ladder + a shared int `join`
/// merges each variant's result — the SAME structure `deriveEnumEq` uses, so the walked
/// field order matches `Eq` (equal enum values hash equal). Returns the int hash value.
fn deriveEnumHash(b: *L.Builder, cty: Typecheck.Type, self_base: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const e = b.in.enum_layouts[cty.enum_id];

    const tag = try b.emit(.{ .get_tag = self_base }, int_ty);
    const seed = try b.emit(.{ .iconst = L.hash_seed }, int_ty);
    const h0 = try hashCombine(b, seed, tag);

    const join = try b.addBlock();
    const merge = try b.addParam(join, int_ty);
    try emitVariantLadder(b, e, tag, .{ .e = e, .self_base = self_base, .h0 = h0, .join = join }, struct {
        fn f(bb: *L.Builder, c: anytype, vi: usize) error{OutOfMemory}!void {
            try emitVariantPayloadHash(bb, c.e, vi, c.self_base, c.h0, c.join);
        }
    }.f);

    b.switchTo(join);
    return merge;
}

/// Lower a SOURCE-LESS auto-derive `Hash` unit: a layout-walking emitter returning an
/// int. ONE param (the receiver, by slot). A struct folds a fixed seed through its fields in
/// layout order (`h := h*MULT + fieldhash`); an empty-payload enum folds the discriminant
/// (`combine(seed, tag)`); a payload enum folds the discriminant then the active variant's
/// payload via a tag-dispatch ladder. The fixed seed makes the hash reproducible run-to-run.
fn lowerDeriveHash(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    d: Derive.Derive,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    const int_ty = Typecheck.Type.int;
    const cty = d.conform_ty;

    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = int_ty, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_self = try b.addSlot(cty);
    try params.append(gpa, p_self);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.ret_param = try b.addParam(exit, int_ty);
    b.blocks.items[exit].term = .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    const self_base = try b.emit(.{ .slot_addr = p_self }, int_ty);

    const result: Ir.ValueId = switch (cty.kind) {
        .@"struct" => blk: {
            const layout = b.in.layouts[cty.struct_id];
            var h = try b.emit(.{ .iconst = L.hash_seed }, int_ty);
            for (layout.field_types, layout.offsets) |fty, off| {
                const fh = try deriveFieldHash(&b, fty, off, self_base);
                h = try hashCombine(&b, h, fh);
            }
            break :blk h;
        },
        .@"enum" => blk: {
            const e = b.in.enum_layouts[cty.enum_id];
            var any_payload = false;
            for (e.variants) |v| if (v.field_types.len != 0) {
                any_payload = true;
                break;
            };
            if (!any_payload) {
                // Empty-payload enum: fold the discriminant into the seed (bare tag hash).
                const tag = try b.emit(.{ .get_tag = self_base }, int_ty);
                const seed = try b.emit(.{ .iconst = L.hash_seed }, int_ty);
                break :blk try hashCombine(&b, seed, tag);
            }
            break :blk try deriveEnumHash(&b, cty, self_base);
        },
        else => blk: {
            // Unreachable: the synthesis barrier only authorizes struct/enum recipes.
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Hash: unsupported conform type in lower" });
            b.had_error = true;
            break :blk try b.emit(.{ .iconst = 0 }, int_ty);
        },
    };

    if (!b.termSet()) try L.brTo(&b, exit, .{ .value = result });
    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

/// Display field `i` (at byte `off`, type `fty`) of receiver base `self_base`:
/// int/bool render inline (via `__display_int` / the bool cond); a `str` field writes its
/// raw bytes (the SAME `str {ptr,len}` write path a top-level `str` uses — no quotes); a
/// struct/enum field is copied into a fresh temp slot then routed through `displayAtSlot`.
/// Unit fields are rejected by T0007, so never occur. Mirrors `deriveFieldHash`.
fn deriveFieldDisplay(b: *L.Builder, fty: Typecheck.Type, off: u32, self_base: Ir.ValueId) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    switch (fty.kind) {
        .int => {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            const v = try b.emit(.{ .load = .{ .addr = la, .ty = fty } }, fty);
            try L.emitDisplayIntValue(b, v);
        },
        .bool => {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            const v = try b.emit(.{ .load = .{ .addr = la, .ty = fty } }, fty);
            try L.emitDisplayBoolValue(b, v);
        },
        .str => {
            // Copy the 16-byte {ptr,len} header into a fresh str slot then `print` it —
            // ONE shared raw-bytes path, so a nested `str` field renders identically to a
            // top-level `str` (raw bytes, no surrounding quotes).
            const slot = try b.addSlot(fty);
            const d = try b.emit(.{ .slot_addr = slot }, int_ty);
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = d, .src = la, .ty = fty } }, null);
            try L.emitPrintSlot(b, slot);
        },
        .@"struct", .@"enum" => {
            const slot = try b.addSlot(fty);
            const d = try b.emit(.{ .slot_addr = slot }, int_ty);
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = d, .src = la, .ty = fty } }, null);
            try L.displayAtSlot(b, fty, slot);
        },
        else => {
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Display: unsupported field type in lower" });
            b.had_error = true;
        },
    }
}

/// Render variant `vi`'s active payload to fd 1 and branch to `join`: write the
/// bare `variant` name, and for a payload variant `variant(<v0>, <v1>)` (fields in
/// declaration order, `, `-separated), using ABSOLUTE payload offsets. Mirrors
/// `emitVariantPayloadEq`'s ladder-arm shape but writes for effect (unit).
fn emitVariantDisplay(b: *L.Builder, e: Typecheck.EnumLayout, vi: usize, self_base: Ir.ValueId, join: Ir.BlockId) error{OutOfMemory}!void {
    const v = e.variants[vi];
    try L.emitWriteLiteral(b, v.name);
    if (v.field_types.len != 0) {
        try L.emitWriteLiteral(b, "(");
        for (v.field_types, v.offsets, 0..) |fty, poff, j| {
            try deriveFieldDisplay(b, fty, e.payload_off + poff, self_base);
            if (j + 1 != v.field_types.len) try L.emitWriteLiteral(b, ", ");
        }
        try L.emitWriteLiteral(b, ")");
    }
    if (!b.termSet()) try L.brTo(b, join, .none);
}

/// Lower a SOURCE-LESS auto-derive `Display` unit: a unit-returning, layout-walking
/// emitter that WRITES the value's structural rendering directly to the output fd — never a
/// returned `str`. ONE param (the receiver, by slot). A struct writes `Name{field: <v>, ...}`
/// (fields in layout order); a payload enum does a `get_tag` dispatch ladder writing a bare
/// `variant` or `variant(<v0>, <v1>)`; scalar fields render inline (int->__display_int, bool
/// inline, str->raw bytes), aggregate fields call the sibling `display` witness.
fn lowerDeriveDisplay(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    d: Derive.Derive,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    const int_ty = Typecheck.Type.int;
    const unit_ty = Typecheck.Type.unit;
    const cty = d.conform_ty;

    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = unit_ty, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_self = try b.addSlot(cty);
    try params.append(gpa, p_self);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.blocks.items[exit].term = .{ .ret = .none };
    b.blocks.items[exit].term_set = true;

    const self_base = try b.emit(.{ .slot_addr = p_self }, int_ty);

    switch (cty.kind) {
        .@"struct" => {
            const layout = b.in.layouts[cty.struct_id];
            try L.emitWriteLiteral(&b, layout.name);
            try L.emitWriteLiteral(&b, "{");
            for (layout.field_names, layout.field_types, layout.offsets, 0..) |fname, fty, off, i| {
                try L.emitWriteLiteral(&b, fname);
                try L.emitWriteLiteral(&b, ": ");
                try deriveFieldDisplay(&b, fty, off, self_base);
                if (i + 1 != layout.field_types.len) try L.emitWriteLiteral(&b, ", ");
            }
            try L.emitWriteLiteral(&b, "}");
        },
        .@"enum" => {
            const e = b.in.enum_layouts[cty.enum_id];
            const tag = try b.emit(.{ .get_tag = self_base }, int_ty);
            const join = try b.addBlock();
            try emitVariantLadder(&b, e, tag, .{ .e = e, .self_base = self_base, .join = join }, struct {
                fn f(bb: *L.Builder, c: anytype, vi: usize) error{OutOfMemory}!void {
                    try emitVariantDisplay(bb, c.e, vi, c.self_base, c.join);
                }
            }.f);
            b.switchTo(join);
        },
        else => {
            // Unreachable: the synthesis barrier only authorizes struct/enum recipes.
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Display: unsupported conform type in lower" });
            b.had_error = true;
        },
    }

    if (!b.termSet()) try L.brTo(&b, exit, .none);
    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

// ===========================================================================
// Boundary tests — the seam these emitters finally make reachable. Each builds a
// bare `L.Inputs` (all-int-field aggregate → the field walkers never touch the
// method/derive tables) and asserts an OBSERVABLE fact through the derive interface,
// not internal state.

/// The `Op` of the instruction that defines value `vid` (searching every block), or
/// null. Boundary tests walk the emitted SSA graph back from a result.
fn defOf(func: *const Ir.Function, vid: Ir.ValueId) ?Ir.Op {
    for (func.blocks) |blk| for (blk.instrs) |ins| {
        if (ins.result == vid) return ins.op;
    };
    return null;
}

/// The scalar value delivered to the exit block (the `br exit(v)` arg). For a
/// straight-line derive there is exactly one such edge.
fn retValue(func: *const Ir.Function) ?Ir.ValueId {
    for (func.blocks) |blk| switch (blk.term) {
        .br => |br| if (br.dest == func.exit and br.args.len == 1) switch (br.args[0]) {
            .value => |v| return v,
            else => {},
        },
        else => {},
    };
    return null;
}

/// The `self`-side field offsets touched, in block/instr (i.e. emission) order: every
/// `field_addr` whose base is the `slot_addr` of param slot 0.
fn selfFieldOffsets(gpa: std.mem.Allocator, func: *const Ir.Function) error{OutOfMemory}![]u32 {
    var self_base: ?Ir.ValueId = null;
    outer: for (func.blocks) |blk| for (blk.instrs) |ins| switch (ins.op) {
        .slot_addr => |s| if (s == 0) {
            self_base = ins.result;
            break :outer;
        },
        else => {},
    };
    var list: std.ArrayList(u32) = .empty;
    errdefer list.deinit(gpa);
    for (func.blocks) |blk| for (blk.instrs) |ins| switch (ins.op) {
        .field_addr => |fa| if (fa.base == self_base.?) try list.append(gpa, fa.off),
        else => {},
    };
    return list.toOwnedSlice(gpa);
}

/// True if any emitted `icmp` uses condition `cc`.
fn hasCond(func: *const Ir.Function, cc: Ir.Cond) bool {
    for (func.blocks) |blk| for (blk.instrs) |ins| switch (ins.op) {
        .icmp => |c| if (c.cc == cc) return true,
        else => {},
    };
    return false;
}

fn intStructInputs(layouts: []const Typecheck.Layout) L.Inputs {
    return .{
        .tree = undefined, // a source-less derive reads layouts, never the tree
        .tokens = &.{},
        .source = "",
        .resolutions = &.{},
        .node_types = &.{},
        .layouts = layouts,
        .enum_layouts = &.{},
        .names = &.{},
        .methods = &.{},
        .derives = &.{},
    };
}

test "derive Hash: mixer chain is h*MULT+field, binding constants + operand order to arith.foldBin" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);

    var fnames = [_][]const u8{ "a", "b" };
    var ftys = [_]Typecheck.Type{ Typecheck.Type.int, Typecheck.Type.int };
    var offs = [_]u32{ 0, 8 };
    var layouts = [_]Typecheck.Layout{.{ .name = "S", .field_names = &fnames, .field_types = &ftys, .offsets = &offs, .size = 16, .@"align" = 8 }};
    const d: Derive.Derive = .{ .protocol_id = 0, .protocol_name = "Hash", .kind = .hash, .conform_ty = Typecheck.Type.structT(0), .ret = Typecheck.Type.int };

    var func = try lowerDeriveHash(gpa, intStructInputs(&layouts), d, .{ .kind = .user_fn, .name = "H" }, &diags);
    defer func.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);

    // Trace the accumulator back from the returned value: h2 = (h1 * MULT) + fld1,
    // h1 = (seed * MULT) + fld0.
    const h2 = retValue(&func).?;
    const add1 = defOf(&func, h2).?.add;
    const mul1 = defOf(&func, add1.lhs).?.mul;
    const mult1 = defOf(&func, mul1.rhs).?.iconst;
    const add0 = defOf(&func, mul1.lhs).?.add;
    const mul0 = defOf(&func, add0.lhs).?.mul;
    const mult0 = defOf(&func, mul0.rhs).?.iconst;
    const seed = defOf(&func, mul0.lhs).?.iconst;

    // The per-field addends are the raw field loads (h*MULT + field, not field-then-mul).
    try testing.expect(std.meta.activeTag(defOf(&func, add0.rhs).?) == .load);
    try testing.expect(std.meta.activeTag(defOf(&func, add1.rhs).?) == .load);

    // Constants are the frozen seed/mult — a swapped operand order or drifted constant fails.
    try testing.expectEqual(L.hash_seed, seed);
    try testing.expectEqual(hash_mult, mult0);
    try testing.expectEqual(hash_mult, mult1);

    // Folding the emitted constants with arith.foldBin (mul-before-add per field) equals an
    // independent Zig reference over arbitrary field values.
    const v0: i64 = 7;
    const v1: i64 = 11;
    const folded = arith.foldBin(.add, arith.foldBin(.mul, arith.foldBin(.add, arith.foldBin(.mul, seed, mult0), v0), mult1), v1);
    const ref = ((L.hash_seed *% hash_mult) +% v0) *% hash_mult +% v1;
    try testing.expectEqual(ref, folded);
}

test "derive Eq/Ord/Hash walk struct fields in the same layout order" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);

    var fnames = [_][]const u8{ "a", "b", "c" };
    var ftys = [_]Typecheck.Type{ Typecheck.Type.int, Typecheck.Type.int, Typecheck.Type.int };
    var offs = [_]u32{ 0, 8, 16 };
    var layouts = [_]Typecheck.Layout{.{ .name = "S", .field_names = &fnames, .field_types = &ftys, .offsets = &offs, .size = 24, .@"align" = 8 }};
    const in = intStructInputs(&layouts);
    const ct = Typecheck.Type.structT(0);
    const sym: Link.SymName = .{ .kind = .user_fn, .name = "D" };

    var eq_f = try lowerDeriveEq(gpa, in, .{ .protocol_id = 0, .protocol_name = "Eq", .kind = .eq, .conform_ty = ct, .ret = Typecheck.Type.@"bool" }, sym, &diags);
    defer eq_f.deinit(gpa);
    var ord_f = try lowerDeriveOrd(gpa, in, .{ .protocol_id = 0, .protocol_name = "Ord", .kind = .ord, .conform_ty = ct, .ret = Typecheck.Type.int }, sym, &diags);
    defer ord_f.deinit(gpa);
    var hash_f = try lowerDeriveHash(gpa, in, .{ .protocol_id = 0, .protocol_name = "Hash", .kind = .hash, .conform_ty = ct, .ret = Typecheck.Type.int }, sym, &diags);
    defer hash_f.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);

    const eq_offs = try selfFieldOffsets(gpa, &eq_f);
    defer gpa.free(eq_offs);
    const ord_offs = try selfFieldOffsets(gpa, &ord_f);
    defer gpa.free(ord_offs);
    const hash_offs = try selfFieldOffsets(gpa, &hash_f);
    defer gpa.free(hash_offs);

    const expected = [_]u32{ 0, 8, 16 };
    try testing.expectEqualSlices(u32, &expected, eq_offs);
    try testing.expectEqualSlices(u32, &expected, ord_offs);
    try testing.expectEqualSlices(u32, &expected, hash_offs);
}

test "derive Ord compares an unsigned field unsigned, a signed field signed" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const sym: Link.SymName = .{ .kind = .user_fn, .name = "D" };

    {
        var fnames = [_][]const u8{"a"};
        var ftys = [_]Typecheck.Type{Typecheck.Type.uint64};
        var offs = [_]u32{0};
        var layouts = [_]Typecheck.Layout{.{ .name = "U", .field_names = &fnames, .field_types = &ftys, .offsets = &offs, .size = 8, .@"align" = 8 }};
        var func = try lowerDeriveOrd(gpa, intStructInputs(&layouts), .{ .protocol_id = 0, .protocol_name = "Ord", .kind = .ord, .conform_ty = Typecheck.Type.structT(0), .ret = Typecheck.Type.int }, sym, &diags);
        defer func.deinit(gpa);
        try testing.expect(hasCond(&func, .ugt) and hasCond(&func, .ult));
        try testing.expect(!hasCond(&func, .gt) and !hasCond(&func, .lt));
    }
    {
        var fnames = [_][]const u8{"a"};
        var ftys = [_]Typecheck.Type{Typecheck.Type.int};
        var offs = [_]u32{0};
        var layouts = [_]Typecheck.Layout{.{ .name = "I", .field_names = &fnames, .field_types = &ftys, .offsets = &offs, .size = 8, .@"align" = 8 }};
        var func = try lowerDeriveOrd(gpa, intStructInputs(&layouts), .{ .protocol_id = 0, .protocol_name = "Ord", .kind = .ord, .conform_ty = Typecheck.Type.structT(0), .ret = Typecheck.Type.int }, sym, &diags);
        defer func.deinit(gpa);
        try testing.expect(hasCond(&func, .gt) and hasCond(&func, .lt));
        try testing.expect(!hasCond(&func, .ugt) and !hasCond(&func, .ult));
    }
}
