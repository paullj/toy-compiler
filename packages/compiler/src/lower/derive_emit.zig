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
const Abi = @import("../codegen/abi/Abi.zig");
const Derive = @import("../symbols/Derive.zig");
const Diagnostic = @import("../diagnostics/Diagnostic.zig").Diagnostic;
const arith = @import("../opt/arith.zig");
const testing = std.testing;

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

/// Eq of two aggregate KEYs already MATERIALIZED into slots: dispatch to the HASHABLE (pid 11)
/// `eq` witness first (explicit `impl` — honored), else delegate to `structEqAtSlots` (pid 0
/// `eq`, then the Ord-fills-Eq `cmp` fallback). The one authority the Map's key eq is.
fn eqKeyAtSlots(b: *L.Builder, ty: Typecheck.Type, lslot: Ir.SlotId, rslot: Ir.SlotId) error{OutOfMemory}!Ir.ValueId {
    if (b.in.prelude_ids.hashable) |hp| switch (Typecheck.resolveConformanceMethod(b.in.methods, ty, "eq", hp, null)) {
        .one => |m| {
            const callee = L.witnessCallee(b, m);
            const args = try b.gpa.alloc(Ir.Operand, 2);
            errdefer b.gpa.free(args);
            args[0] = .{ .slot = lslot };
            args[1] = .{ .slot = rslot };
            return try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, Typecheck.Type.@"bool");
        },
        .none, .ambiguous => {},
    };
    return structEqAtSlots(b, ty, lslot, rslot);
}

/// The bool eq of struct field `i` (at byte `off`, type `fty`) between the two receiver
/// bases: int/bool inline `icmp eq`; `str` via the `{ptr,len}` byte-loop
/// (`strEqAtPtrs`); a struct/enum field is copied into fresh temp slots (a `field_addr`
/// is a ptr VALUE, not a slot operand — the ABI needs the field's own slot) then routed
/// through `structEqAtSlots`. A `()` field is zero-sized: eq true (all `()` are equal).
fn deriveFieldEq(b: *L.Builder, fty: Typecheck.Type, off: u32, self_base: Ir.ValueId, other_base: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    // A managed-box FIELD compares by CELL IDENTITY: an 8-byte pointer `icmp eq`, not a
    // structural deref (the box's `int` field IS the cell pointer).
    if (L.isRefTy(b, fty)) {
        const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = int_ty } }, int_ty);
        const lv = try b.emit(.{ .load = .{ .addr = la, .ty = int_ty } }, int_ty);
        const ra = try b.emit(.{ .field_addr = .{ .base = other_base, .off = off, .ty = int_ty } }, int_ty);
        const rv = try b.emit(.{ .load = .{ .addr = ra, .ty = int_ty } }, int_ty);
        return try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = lv, .rhs = rv } }, Typecheck.Type.@"bool");
    }
    switch (fty.kind) {
        .int, .bool => {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            const lv = try b.emit(.{ .load = .{ .addr = la, .ty = fty } }, fty);
            const ra = try b.emit(.{ .field_addr = .{ .base = other_base, .off = off, .ty = fty } }, int_ty);
            const rv = try b.emit(.{ .load = .{ .addr = ra, .ty = fty } }, fty);
            return try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = lv, .rhs = rv } }, Typecheck.Type.@"bool");
        },
        .float => {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            const lv = try b.emit(.{ .load = .{ .addr = la, .ty = fty } }, fty);
            const ra = try b.emit(.{ .field_addr = .{ .base = other_base, .off = off, .ty = fty } }, int_ty);
            const rv = try b.emit(.{ .load = .{ .addr = ra, .ty = fty } }, fty);
            return try b.emit(.{ .fcmp = .{ .cc = .eq, .lhs = lv, .rhs = rv } }, Typecheck.Type.@"bool");
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
        // A `()` field is zero-sized: two `()` are always equal (the identity in the
        // AND-fold), so no field_addr/load — emit the constant `true`.
        .unit => return try b.emit(.{ .bconst = true }, Typecheck.Type.@"bool"),
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
/// copied into fresh temp slots then routed through `cmpAtSlots`. A `()` field is zero-sized:
/// cmp Equal (all `()` are equal). Mirrors `deriveFieldEq`.
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
        // A `()` field is zero-sized: two `()` compare Equal (the identity in the
        // lexicographic fold), so no field_addr/load — emit the `Equal` discriminant.
        .unit => return try b.emit(.{ .iconst = L.ord_eq }, int_ty),
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
        .conv_int_char => lowerConvIntChar(gpa, in, d, sym, out_diags),
        .conv_char_byte => lowerConvCharByte(gpa, in, d, sym, out_diags),
        .conv_float_int => lowerConvFloatInt(gpa, in, d, sym, out_diags),
        .trace => lowerDeriveTrace(gpa, in, d, sym, out_diags),
    };
}

/// Lower the SOURCE-LESS `int -> char` fallible-conversion witness:
/// `int_to_char(int) -> Result[char, ConvErr]`. Reads the source int from its param slot,
/// stores it verbatim (as `uint32`) into the char payload iff it is a valid Unicode scalar
/// (`validScalarValue`), else `ConvErr.out_of_range`. The predicate + the Result tail are
/// the SAME emitters the old inline used, over the SAME interned Result layout (`d.ret`), so
/// the runtime bytes are byte-identical to the pre-fix inline.
fn lowerConvIntChar(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    d: Derive.Derive,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    const int_ty = Typecheck.Type.int;
    const ret = d.ret; // Result[char, ConvErr]

    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = ret, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_v = try b.addSlot(int_ty);
    try params.append(gpa, p_v);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.ret_param = try b.addParam(exit, ret);
    b.blocks.items[exit].term = .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    const e = b.in.enum_layouts[ret.enum_id];
    const vbase = try b.emit(.{ .slot_addr = p_v }, int_ty);
    const v = try b.emit(.{ .load = .{ .addr = vbase, .ty = int_ty } }, int_ty);
    const slot = try b.addSlot(ret);
    const base = try b.emit(.{ .slot_addr = slot }, int_ty);
    const valid = try L.validScalarValue(&b, v);
    const ok_blk = try b.addBlock();
    const err_blk = try b.addBlock();
    const join = try b.addBlock();
    b.setTerm(.{ .cond_br = .{ .cond = valid, .t = ok_blk, .f = err_blk } });
    try L.emitConvResultTail(&b, e, base, ok_blk, err_blk, join, v, Typecheck.Type.uint32);
    b.switchTo(join);
    try L.brTo(&b, exit, .{ .slot = slot });
    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

/// Lower the SOURCE-LESS `char -> byte` fallible-conversion witness:
/// `char_to_byte(int) -> Result[byte, ConvErr]`. The receiver's codepoint arrives (raw) in
/// the param slot; `Ok(masked)` iff `masked` (v truncated + re-extended to `uint8`) equals
/// v — the same-signedness narrow fits-check (a codepoint is unsigned, so no `sign_blk`).
fn lowerConvCharByte(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    d: Derive.Derive,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    const ret = d.ret; // Result[byte, ConvErr]

    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = ret, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_v = try b.addSlot(int_ty);
    try params.append(gpa, p_v);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.ret_param = try b.addParam(exit, ret);
    b.blocks.items[exit].term = .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    const e = b.in.enum_layouts[ret.enum_id];
    const vbase = try b.emit(.{ .slot_addr = p_v }, int_ty);
    const v = try b.emit(.{ .load = .{ .addr = vbase, .ty = int_ty } }, int_ty);
    const slot = try b.addSlot(ret);
    const base = try b.emit(.{ .slot_addr = slot }, int_ty);
    const masked = try L.recanonToWidth(&b, v, Typecheck.Type.uint8);
    const fits = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = masked, .rhs = v } }, bool_ty);
    const ok_blk = try b.addBlock();
    const err_blk = try b.addBlock();
    const join = try b.addBlock();
    b.setTerm(.{ .cond_br = .{ .cond = fits, .t = ok_blk, .f = err_blk } });
    try L.emitConvResultTail(&b, e, base, ok_blk, err_blk, join, masked, Typecheck.Type.uint8);
    b.switchTo(join);
    try L.brTo(&b, exit, .{ .slot = slot });
    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

/// Lower the SOURCE-LESS `float -> int` fallible-conversion witness:
/// `float_to_int(float) -> Result[int, ConvErr]`. Reads the f64 param, then Ok(trunc-toward-zero)
/// iff `-2^63 <= f < 2^63` else Err. fcvtzs SATURATES on NaN/+-inf/out-of-range, but the ruling
/// wants Err — so the fcmp range check runs first. `.ge`/`.lt` are NaN-safe (false on unordered),
/// so NaN and +-inf fall to Err; an explicit `f != f` is redundant and deliberately omitted. The
/// strict `< 2^63` excludes the exactly-representable 2^63 (whose fcvtzs would saturate to i64_max).
fn lowerConvFloatInt(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    d: Derive.Derive,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    const int_ty = Typecheck.Type.int;
    const float_ty = Typecheck.Type.float;
    const bool_ty = Typecheck.Type.@"bool";
    const ret = d.ret; // Result[int, ConvErr]

    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = ret, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_v = try b.addSlot(float_ty); // float param -> v0 via the fp ABI
    try params.append(gpa, p_v);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.ret_param = try b.addParam(exit, ret);
    b.blocks.items[exit].term = .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    const e = b.in.enum_layouts[ret.enum_id];
    const vbase = try b.emit(.{ .slot_addr = p_v }, int_ty);
    const f = try b.emit(.{ .load = .{ .addr = vbase, .ty = float_ty } }, float_ty);
    const slot = try b.addSlot(ret);
    const base = try b.emit(.{ .slot_addr = slot }, int_ty);

    const lo = try b.emit(.{ .fconst = -0x1p63 }, float_ty); // -2^63 exactly (0xC3E0000000000000)
    const hi = try b.emit(.{ .fconst = 0x1p63 }, float_ty); // +2^63 exactly (0x43E0000000000000)
    const ge_lo = try b.emit(.{ .fcmp = .{ .cc = .ge, .lhs = f, .rhs = lo } }, bool_ty);
    const lt_hi = try b.emit(.{ .fcmp = .{ .cc = .lt, .lhs = f, .rhs = hi } }, bool_ty);
    const in_range = try b.emit(.{ .mul = .{ .lhs = ge_lo, .rhs = lt_hi } }, int_ty); // branch-free bool AND
    const zero = try b.emit(.{ .iconst = 0 }, int_ty);
    const valid = try b.emit(.{ .icmp = .{ .cc = .ne, .lhs = in_range, .rhs = zero } }, bool_ty);

    const iv = try b.emit(.{ .fcvtzs = f }, int_ty); // stored only on the ok path
    const ok_blk = try b.addBlock();
    const err_blk = try b.addBlock();
    const join = try b.addBlock();
    b.setTerm(.{ .cond_br = .{ .cond = valid, .t = ok_blk, .f = err_blk } });
    try L.emitConvResultTail(&b, e, base, ok_blk, err_blk, join, iv, Typecheck.Type.int);
    b.switchTo(join);
    try L.brTo(&b, exit, .{ .slot = slot });
    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
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

/// Hash of an aggregate KEY already MATERIALIZED into `slot`: dispatch to the key's HASHABLE
/// (pid 11) `hash` witness first (an explicit `impl has Hashable` — honored), else delegate to
/// the structural Hash witness (`hashAtSlot`, pid 6). The one authority the Map's key hash is.
fn hashKeyAtSlot(b: *L.Builder, ty: Typecheck.Type, slot: Ir.SlotId) error{OutOfMemory}!Ir.ValueId {
    if (b.in.prelude_ids.hashable) |hp| switch (Typecheck.resolveConformanceMethod(b.in.methods, ty, "hash", hp, null)) {
        .one => |m| {
            const callee = L.witnessCallee(b, m);
            const args = try b.gpa.alloc(Ir.Operand, 1);
            errdefer b.gpa.free(args);
            args[0] = .{ .slot = slot };
            return try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, Typecheck.Type.int);
        },
        .none, .ambiguous => {},
    };
    return hashAtSlot(b, ty, slot);
}

/// The int hash of struct field `i` (at byte `off`, type `fty`) of receiver base `self_base`
///: int/bool hash to their own loaded VALUE (identity — an int is its own hash, a bool
/// is 0/1); `str` via the `{ptr,len}` byte polynomial (`hashStrAtPtr`); a struct/enum field is
/// copied into a fresh temp slot then routed through `hashAtSlot`. A `()` field is zero-sized:
/// hash `hash_seed` (Eq-consistent). Mirrors `deriveFieldEq`/`deriveFieldCmp` (single receiver — hash is
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
        // A `()` field is zero-sized: fold a fixed constant (Eq-consistent, since all
        // `()` are equal; matches the bare-unit `.hash()` seed in lower).
        .unit => return try b.emit(.{ .iconst = L.hash_seed }, int_ty),
        else => {
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Hash: unsupported field type in lower" });
            b.had_error = true;
            return try b.emit(.{ .iconst = 0 }, int_ty);
        },
    }
}

/// Fold variant `vi`'s active payload into the running hash `h0` and deliver the result to
/// `join`: an empty variant delivers `h0` unchanged; else each payload field is folded
/// through the fxhash `hashMix`. Mirrors `emitVariantPayloadEq`.
fn emitVariantPayloadHash(b: *L.Builder, e: Typecheck.EnumLayout, vi: usize, self_base: Ir.ValueId, h0: Ir.ValueId, join: Ir.BlockId) error{OutOfMemory}!void {
    const v = e.variants[vi];
    var h = h0;
    for (v.field_types, v.offsets) |fty, poff| {
        const fh = try deriveFieldHash(b, fty, e.payload_off + poff, self_base);
        h = try L.hashMix(b, h, fh);
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
    const h0 = try L.hashMix(b, seed, tag);

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
/// layout order (folding each field hash through the fxhash `hashMix`); an empty-payload enum folds the discriminant
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
                h = try L.hashMix(&b, h, fh);
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
                break :blk try L.hashMix(&b, seed, tag);
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

/// Re-export so callers name the erased kind through this emitter.
pub const ErasedKind = Derive.ErasedKind;

/// Lower an ERASED key witness: the same structural walk as the natural `hash`/`eq`
/// derive, but the receiver base(s) arrive as `rawptr` params (the key's byte address)
/// instead of by-value self slots. The loaded pointer IS the key base, so the shared
/// `deriveFieldHash`/`deriveFieldEq`/`hashStrAtPtr`/`strEqAtPtrs` walks apply unchanged —
/// `field_addr(base, off)` reads through the passed pointer identically to a frame-slot
/// base. Uncached and pure of `(ty, kind, layout)`, so a double-lower is byte-identical.
/// Replicates the 3-line struct/enum folds rather than editing `lowerDeriveHash`/
/// `lowerDeriveEq`, deliberately keeping the natural-derive corpus untouched.
pub fn lowerErased(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    ty: Typecheck.Type,
    kind: Derive.ErasedKind,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    const rawptr_ty = Typecheck.Type.rawptr;
    return switch (kind) {
        .hash => lowerErasedHash(gpa, in, ty, sym, out_diags, int_ty, rawptr_ty),
        .eq => lowerErasedEq(gpa, in, ty, sym, out_diags, int_ty, bool_ty, rawptr_ty),
        .trace => lowerErasedTrace(gpa, in, ty, sym, out_diags, int_ty, rawptr_ty),
    };
}

/// Emit `gc_mark(cell)` / `gc_mark_leaf(cell)` for a value already in `cell`.
fn emitMarkCall(b: *L.Builder, callee: Link.SymName, cell: Ir.ValueId) error{OutOfMemory}!void {
    const args = try b.gpa.alloc(Ir.Operand, 1);
    errdefer b.gpa.free(args);
    args[0] = .{ .value = cell };
    _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, null);
}

/// Lower an ERASED trace unit `descriptor$trace$<ty>(self: rawptr) -> ()`: called by the
/// collector (and by a container walk) with `self` = the byte address of a `ty`-typed cell,
/// it marks the managed heap object(s) that cell references. Three shapes:
///  * `str` — the cell holds a heap buffer pointer at offset 0 → `gc_mark_leaf` it (a leaf:
///    the buffer holds no further managed pointers, so the bit alone keeps it, no scan).
///  * a managed box (`Ref`/`gc_array`) — the cell holds a pointee/header pointer at offset 0
///    → `gc_mark` it (conservative: the pointee may itself hold managed pointers, so it is
///    pushed and scanned; a leaf pointee like `Ref[int]`'s int cell scans to a no-op).
///  * a managed by-value aggregate (a struct/enum transitively holding a box) — conservatively
///    `gc_mark` each 8-byte word of the in-place cell; every genuine managed pointer within
///    (in any field / any enum variant) is thereby pushed + scanned, a sound superset that
///    needs no per-field layout walk. Uncached and pure of `(ty, kind, layout)`, so a
///    double-lower is byte-identical.
fn lowerErasedTrace(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    ty: Typecheck.Type,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
    int_ty: Typecheck.Type,
    rawptr_ty: Typecheck.Type,
) error{OutOfMemory}!Ir.Function {
    const unit_ty = Typecheck.Type.unit;

    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = unit_ty, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_ptr = try b.addSlot(rawptr_ty);
    try params.append(gpa, p_ptr);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.blocks.items[exit].term = .{ .ret = .none };
    b.blocks.items[exit].term_set = true;

    const p_addr = try b.emit(.{ .slot_addr = p_ptr }, int_ty);
    const self_base = try b.emit(.{ .load = .{ .addr = p_addr, .ty = rawptr_ty } }, rawptr_ty);

    if (ty.kind == .str) {
        const buf = try b.emit(.{ .load = .{ .addr = self_base, .ty = int_ty } }, int_ty);
        try emitMarkCall(&b, L.gc_mark_leaf_sym, buf);
    } else if (L.isRefTy(&b, ty)) {
        const pointee = try b.emit(.{ .load = .{ .addr = self_base, .ty = int_ty } }, int_ty);
        try emitMarkCall(&b, L.gc_mark_sym, pointee);
    } else {
        // A by-value aggregate: conservatively mark every 8-byte word of the in-place cell.
        const size = Abi.typeSize(ty, in.layouts, in.enum_layouts);
        const words = (size + 7) / 8;
        var k: u32 = 0;
        while (k < words) : (k += 1) {
            const wa = try b.emit(.{ .field_addr = .{ .base = self_base, .off = k * 8, .ty = int_ty } }, int_ty);
            const w = try b.emit(.{ .load = .{ .addr = wa, .ty = int_ty } }, int_ty);
            try emitMarkCall(&b, L.gc_mark_sym, w);
        }
    }

    if (!b.termSet()) try L.brTo(&b, exit, .none);
    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

fn lowerErasedHash(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    ty: Typecheck.Type,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
    int_ty: Typecheck.Type,
    rawptr_ty: Typecheck.Type,
) error{OutOfMemory}!Ir.Function {
    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = int_ty, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_ptr = try b.addSlot(rawptr_ty);
    try params.append(gpa, p_ptr);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.ret_param = try b.addParam(exit, int_ty);
    b.blocks.items[exit].term = .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    const p_addr = try b.emit(.{ .slot_addr = p_ptr }, int_ty);
    const self_base = try b.emit(.{ .load = .{ .addr = p_addr, .ty = rawptr_ty } }, rawptr_ty);

    const result: Ir.ValueId = switch (ty.kind) {
        .int, .bool => try b.emit(.{ .load = .{ .addr = self_base, .ty = ty } }, ty),
        .str => try L.hashStrAtPtr(&b, self_base),
        // A bare `()` key is zero-sized: nothing to load off the passed pointer, so fold
        // the fixed seed (Eq-consistent, since all `()` are equal). Mirrors the `.unit`
        // field arm in `deriveFieldHash`.
        .unit => try b.emit(.{ .iconst = L.hash_seed }, int_ty),
        .@"struct", .@"enum" => blk: {
            // Materialize the key into a fresh by-value slot off the loaded pointer, then
            // dispatch to the key's Hashable `hash` witness (an explicit `impl` OR the derived
            // structural unit) — the SAME authority the `[K has Hashable]` bound gates on.
            const slot = try b.addSlot(ty);
            const d = try b.emit(.{ .slot_addr = slot }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = d, .src = self_base, .ty = ty } }, null);
            break :blk try hashKeyAtSlot(&b, ty, slot);
        },
        else => blk: {
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "erased hash: unsupported key type in lower" });
            b.had_error = true;
            break :blk try b.emit(.{ .iconst = 0 }, int_ty);
        },
    };

    if (!b.termSet()) try L.brTo(&b, exit, .{ .value = result });
    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

fn lowerErasedEq(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    ty: Typecheck.Type,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
    int_ty: Typecheck.Type,
    bool_ty: Typecheck.Type,
    rawptr_ty: Typecheck.Type,
) error{OutOfMemory}!Ir.Function {
    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = bool_ty, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_a = try b.addSlot(rawptr_ty);
    try params.append(gpa, p_a);
    const p_b = try b.addSlot(rawptr_ty);
    try params.append(gpa, p_b);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.ret_param = try b.addParam(exit, bool_ty);
    b.blocks.items[exit].term = .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    const pa_addr = try b.emit(.{ .slot_addr = p_a }, int_ty);
    const self_base = try b.emit(.{ .load = .{ .addr = pa_addr, .ty = rawptr_ty } }, rawptr_ty);
    const pb_addr = try b.emit(.{ .slot_addr = p_b }, int_ty);
    const other_base = try b.emit(.{ .load = .{ .addr = pb_addr, .ty = rawptr_ty } }, rawptr_ty);

    const result: Ir.ValueId = switch (ty.kind) {
        .int, .bool => blk: {
            const lv = try b.emit(.{ .load = .{ .addr = self_base, .ty = ty } }, ty);
            const rv = try b.emit(.{ .load = .{ .addr = other_base, .ty = ty } }, ty);
            break :blk try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = lv, .rhs = rv } }, bool_ty);
        },
        .str => try L.strEqAtPtrs(&b, self_base, other_base),
        // A bare `()` key is zero-sized: no bytes to compare off either pointer, so two
        // `()` are always equal. Mirrors the `.unit` field arm in `deriveFieldEq`.
        .unit => try b.emit(.{ .bconst = true }, bool_ty),
        .@"struct", .@"enum" => blk: {
            // Materialize both keys into fresh by-value slots off their loaded pointers, then
            // dispatch to the key's Hashable `eq` witness (an explicit `impl` OR the derived
            // structural unit) — the SAME authority the `[K has Hashable]` bound gates on.
            const lslot = try b.addSlot(ty);
            const ld = try b.emit(.{ .slot_addr = lslot }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = ld, .src = self_base, .ty = ty } }, null);
            const rslot = try b.addSlot(ty);
            const rd = try b.emit(.{ .slot_addr = rslot }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = rd, .src = other_base, .ty = ty } }, null);
            break :blk try eqKeyAtSlots(&b, ty, lslot, rslot);
        },
        else => blk: {
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "erased eq: unsupported key type in lower" });
            b.had_error = true;
            break :blk try b.emit(.{ .bconst = false }, bool_ty);
        },
    };

    if (!b.termSet()) try L.brTo(&b, exit, .{ .value = result });
    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

/// Render field `i` (at byte `off`, type `fty`) of receiver base `self_base` to a fresh
/// `str` slot through the SHARED `displayToSlot` authority — the SAME renderer `print` /
/// `to_string` / a top-level value use, so a field never drifts from a standalone value.
/// int/bool load the field value; str/struct/enum copy the 16-/N-byte field into a temp
/// slot (self by slot); a `()` field is zero-sized (a `.none` operand → the const-data
/// `"()"`). The caller `appendPiece`s the returned slot. Mirrors `deriveFieldHash`.
fn deriveFieldDisplay(b: *L.Builder, fty: Typecheck.Type, off: u32, self_base: Ir.ValueId) error{OutOfMemory}!Ir.SlotId {
    const int_ty = Typecheck.Type.int;
    const op: Ir.Operand = switch (fty.kind) {
        .int, .bool => blk: {
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            break :blk .{ .value = try b.emit(.{ .load = .{ .addr = la, .ty = fty } }, fty) };
        },
        .str, .@"struct", .@"enum" => blk: {
            const slot = try b.addSlot(fty);
            const d = try b.emit(.{ .slot_addr = slot }, int_ty);
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = d, .src = la, .ty = fty } }, null);
            break :blk .{ .slot = slot };
        },
        .unit => .none,
        else => {
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Display: unsupported field type in lower" });
            b.had_error = true;
            return try L.emitStrLiteralSlot(b, "");
        },
    };
    return try L.displayToSlot(b, fty, op);
}

/// Render variant `vi`'s active payload into the shared `acc` and branch to `join`: append
/// the bare `variant` name, and for a payload variant `variant(<v0>, <v1>)` (fields in
/// declaration order, `, `-separated), using ABSOLUTE payload offsets. The variant name +
/// `"("` are fused into one interned literal. Mirrors `emitVariantPayloadEq`'s ladder-arm
/// shape but builds a str instead of comparing.
fn emitVariantDisplay(b: *L.Builder, acc: Ir.SlotId, e: Typecheck.EnumLayout, vi: usize, self_base: Ir.ValueId, join: Ir.BlockId) error{OutOfMemory}!void {
    const v = e.variants[vi];
    if (v.field_types.len == 0) {
        try L.appendLiteral(b, acc, v.name);
    } else {
        const open = try std.fmt.allocPrint(b.gpa, "{s}(", .{v.name});
        defer b.gpa.free(open);
        try L.appendLiteral(b, acc, open);
        for (v.field_types, v.offsets, 0..) |fty, poff, j| {
            const piece = try deriveFieldDisplay(b, fty, e.payload_off + poff, self_base);
            try L.appendPiece(b, acc, piece);
            if (j + 1 != v.field_types.len) try L.appendLiteral(b, acc, ", ");
        }
        try L.appendLiteral(b, acc, ")");
    }
    if (!b.termSet()) try L.brTo(b, join, .none);
}

/// Lower a SOURCE-LESS auto-derive `Display` unit: a `str`-RETURNING, layout-walking emitter
/// that BUILDS the value's structural rendering into a fresh gc `str` (the exact bytes it
/// used to write to fd 1) and returns it. `print` writes the result; `to_string` returns it.
/// ONE param (the receiver, by slot). A struct builds `Name{field: <v>, ...}` (fields in
/// layout order); a payload enum does a `get_tag` dispatch ladder producing a bare `variant`
/// or `variant(<v0>, <v1>)`. Every fixed run (name + separators) is one interned literal;
/// each dynamic field is rendered by the shared `displayToSlot` and appended into the reused
/// `acc` str slot (one slot for the whole render — the frame-capacity fix). A scalar field
/// renders through `displayToSlot`'s scalar arms; an aggregate field calls its sibling
/// `display` witness — one authority, no drift.
fn lowerDeriveDisplay(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    d: Derive.Derive,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    const int_ty = Typecheck.Type.int;
    const str_ty = Typecheck.Type.str;
    const cty = d.conform_ty;

    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = str_ty, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const p_self = try b.addSlot(cty);
    try params.append(gpa, p_self);

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    b.ret_param = try b.addParam(exit, str_ty);
    b.blocks.items[exit].term = .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    if (L.isCharTy(&b, cty)) {
        // char's Display OVERRIDES the structural tuple walk with the hand-written UTF-8
        // encoder — the same override the per-site inline shunt used to provide, now emitted
        // ONCE here and CALLed from every char-display site (the frame-overflow fix).
        const s = try L.lowerCharDisplay(&b, p_self);
        if (!b.termSet()) try L.brTo(&b, exit, .{ .slot = s });
        return try L.finishFn(&b, gpa, sym, &params, entry, exit);
    }

    const self_base = try b.emit(.{ .slot_addr = p_self }, int_ty);
    const acc = try L.emitStrLiteralSlot(&b, "");

    switch (cty.kind) {
        .@"struct" => {
            const layout = b.in.layouts[cty.struct_id];
            if (layout.is_tuple) {
                const open = try std.fmt.allocPrint(gpa, "{s}(", .{layout.name});
                defer gpa.free(open);
                try L.appendLiteral(&b, acc, open);
                for (layout.field_types, layout.offsets, 0..) |fty, off, i| {
                    try L.appendPiece(&b, acc, try deriveFieldDisplay(&b, fty, off, self_base));
                    if (i + 1 != layout.field_types.len) try L.appendLiteral(&b, acc, ", ");
                }
                try L.appendLiteral(&b, acc, ")");
            } else {
                for (layout.field_names, layout.field_types, layout.offsets, 0..) |fname, fty, off, i| {
                    // Fuse the (separator + field name + `: `) run into ONE interned literal:
                    // `Name{f0: ` for the first field, `, f1: ` for the rest.
                    const lit = if (i == 0)
                        try std.fmt.allocPrint(gpa, "{s}{{{s}: ", .{ layout.name, fname })
                    else
                        try std.fmt.allocPrint(gpa, ", {s}: ", .{fname});
                    defer gpa.free(lit);
                    try L.appendLiteral(&b, acc, lit);
                    try L.appendPiece(&b, acc, try deriveFieldDisplay(&b, fty, off, self_base));
                }
                // A record struct always has >= 1 field (T0005 rejects an empty struct), so
                // the loop emitted the fused `Name{f0: ` open; close the brace.
                try L.appendLiteral(&b, acc, "}");
            }
            if (!b.termSet()) try L.brTo(&b, exit, .{ .slot = acc });
        },
        .@"enum" => {
            const e = b.in.enum_layouts[cty.enum_id];
            const tag = try b.emit(.{ .get_tag = self_base }, int_ty);
            const join = try b.addBlock();
            try emitVariantLadder(&b, e, tag, .{ .e = e, .acc = acc, .self_base = self_base, .join = join }, struct {
                fn f(bb: *L.Builder, c: anytype, vi: usize) error{OutOfMemory}!void {
                    try emitVariantDisplay(bb, c.acc, c.e, vi, c.self_base, c.join);
                }
            }.f);
            b.switchTo(join);
            if (!b.termSet()) try L.brTo(&b, exit, .{ .slot = acc });
        },
        else => {
            // Unreachable: the synthesis barrier only authorizes struct/enum recipes.
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Display: unsupported conform type in lower" });
            b.had_error = true;
            if (!b.termSet()) try L.brTo(&b, exit, .{ .slot = acc });
        },
    }

    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

/// Trace one component field per its RESOLVED `FieldWitness` (the recipe carries the
/// managed-vs-by-value decision, so the emitter reads no method table — the same purity the
/// other derive emitters hold). A `.trace_mark` (managed) field loads the 8-byte cell
/// pointer and `bl gc_mark`s it — the trace BOUNDARY, never recursing into the pointee. A
/// `.trace_call` by-value aggregate is copied into a fresh slot then routed to its sibling
/// `trace` witness (mirrors `deriveFieldHash`'s aggregate path). Everything else
/// (`.inline_kind`: scalar/str/unmanaged aggregate) is a no-op — not managed.
fn deriveFieldTrace(b: *L.Builder, fty: Typecheck.Type, off: u32, self_base: Ir.ValueId, fw: Derive.FieldWitness) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    switch (fw) {
        .trace_mark => {
            const fa = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = int_ty } }, int_ty);
            const cell = try b.emit(.{ .load = .{ .addr = fa, .ty = int_ty } }, int_ty);
            const args = try b.gpa.alloc(Ir.Operand, 1);
            errdefer b.gpa.free(args);
            args[0] = .{ .value = cell };
            _ = try b.emit(.{ .call = .{ .callee = L.gc_mark_sym, .args = args, .ret_slot = Ir.none_slot } }, null);
        },
        .trace_call => |name| {
            const slot = try b.addSlot(fty);
            const d = try b.emit(.{ .slot_addr = slot }, int_ty);
            const la = try b.emit(.{ .field_addr = .{ .base = self_base, .off = off, .ty = fty } }, int_ty);
            _ = try b.emit(.{ .copy = .{ .dst = d, .src = la, .ty = fty } }, null);
            const args = try b.gpa.alloc(Ir.Operand, 1);
            errdefer b.gpa.free(args);
            args[0] = .{ .slot = slot };
            _ = try b.emit(.{ .call = .{ .callee = .{ .kind = .user_fn, .name = name }, .args = args, .ret_slot = Ir.none_slot } }, null);
        },
        else => {},
    }
}

/// Trace variant `vi`'s active payload (its slice of the flattened `field_witnesses`) and
/// branch to `join`. Mirrors `emitVariantPayloadEq`/`emitVariantDisplay`'s ladder-arm shape,
/// writing for effect (unit). The per-variant base into the flat witness list is the sum of
/// the prior variants' field counts — the SAME variant-decl-then-field order
/// `collectComponentTypes` (and thus `resolveDeriveFields`) walked.
fn emitVariantTrace(b: *L.Builder, e: Typecheck.EnumLayout, vi: usize, self_base: Ir.ValueId, field_witnesses: []const Derive.FieldWitness, join: Ir.BlockId) error{OutOfMemory}!void {
    const v = e.variants[vi];
    var base: usize = 0;
    for (e.variants[0..vi]) |pv| base += pv.field_types.len;
    for (v.field_types, v.offsets, 0..) |fty, poff, j| {
        try deriveFieldTrace(b, fty, e.payload_off + poff, self_base, field_witnesses[base + j]);
    }
    if (!b.termSet()) try L.brTo(b, join, .none);
}

/// Lower a SOURCE-LESS auto-derive `Trace` unit: a unit-returning, layout-walking emitter
/// that marks each MANAGED field of the receiver. ONE param (the receiver, by slot). A
/// struct walks its fields in layout order; a payload enum does a `get_tag` dispatch ladder
/// over each variant's payload. Every managed field emits a `bl gc_mark` (the trace
/// boundary); by-value aggregates that hold managed fields call their sibling trace witness;
/// scalar/str/unmanaged fields are no-ops. VESTIGIAL: this typed unit is emitted but never
/// called. Descriptor-driven tracing dispatches per object through the ERASED per-type trace
/// units named in the descriptor table (`traceNameFor` -> `lowerErasedTrace`), not this one,
/// so it has no runtime caller and is dead-stripped (removing the dead emission is a follow-up).
fn lowerDeriveTrace(
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
            for (layout.field_types, layout.offsets, d.field_witnesses) |fty, off, fw| {
                try deriveFieldTrace(&b, fty, off, self_base, fw);
            }
        },
        .@"enum" => {
            const e = b.in.enum_layouts[cty.enum_id];
            const tag = try b.emit(.{ .get_tag = self_base }, int_ty);
            const join = try b.addBlock();
            try emitVariantLadder(&b, e, tag, .{ .e = e, .self_base = self_base, .fw = d.field_witnesses, .join = join }, struct {
                fn f(bb: *L.Builder, c: anytype, vi: usize) error{OutOfMemory}!void {
                    try emitVariantTrace(bb, c.e, vi, c.self_base, c.fw, c.join);
                }
            }.f);
            b.switchTo(join);
        },
        else => {
            // Unreachable: the synthesis barrier only authorizes struct/enum recipes.
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Trace: unsupported conform type in lower" });
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

/// Render a Function to owned text — used to pin `-jN` byte-identity of the erased units.
fn renderFn(gpa: std.mem.Allocator, func: *const Ir.Function, in: L.Inputs) ![]u8 {
    var buf: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try Ir.render(&w, func, in.layouts, in.enum_layouts);
    return gpa.dupe(u8, w.buffered());
}

/// Whether any block emits an op whose active tag matches `tag`.
fn hasOp(func: *const Ir.Function, comptime tag: std.meta.Tag(Ir.Op)) bool {
    for (func.blocks) |blk| for (blk.instrs) |ins| {
        if (std.meta.activeTag(ins.op) == tag) return true;
    };
    return false;
}

/// The name of the first `call` whose callee is a `user_fn`, or null.
fn firstCallName(func: *const Ir.Function) ?[]const u8 {
    for (func.blocks) |blk| for (blk.instrs) |ins| switch (ins.op) {
        .call => |c| if (c.callee.kind == .user_fn) return c.callee.name,
        else => {},
    };
    return null;
}

test "erased hash on an int key returns the loaded value; byte-identical across relowers" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const in = intStructInputs(&.{});
    const sym: Link.SymName = .{ .kind = .user_fn, .name = "descriptor$hash$int" };

    var f1 = try lowerErased(gpa, in, Typecheck.Type.int, .hash, sym, &diags);
    defer f1.deinit(gpa);
    var f2 = try lowerErased(gpa, in, Typecheck.Type.int, .hash, sym, &diags);
    defer f2.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);

    const t1 = try renderFn(gpa, &f1, in);
    defer gpa.free(t1);
    const t2 = try renderFn(gpa, &f2, in);
    defer gpa.free(t2);
    try testing.expectEqualStrings(t1, t2);
}

test "erased str trace leaf-marks the buffer once (gc_mark_leaf over load[self,0]); byte-identical" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const in = intStructInputs(&.{});
    const sym: Link.SymName = .{ .kind = .user_fn, .name = "descriptor$trace$str" };

    var f1 = try lowerErased(gpa, in, Typecheck.Type.str, .trace, sym, &diags);
    defer f1.deinit(gpa);
    var f2 = try lowerErased(gpa, in, Typecheck.Type.str, .trace, sym, &diags);
    defer f2.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);

    // Exactly one leaf-mark of the heap buffer (the str's ptr half), and NOT a full gc_mark
    // (the buffer holds no further managed pointers). The single `.call` arg is the value
    // loaded from the passed pointer at offset 0.
    var leaf_calls: usize = 0;
    var full_calls: usize = 0;
    var loaded_cell: ?Ir.ValueId = null;
    for (f1.blocks) |blk| for (blk.instrs) |ins| switch (ins.op) {
        .call => |c| {
            if (c.callee.kind == .builtin and std.mem.eql(u8, c.callee.name, "gc_mark_leaf")) {
                leaf_calls += 1;
                try testing.expectEqual(@as(usize, 1), c.args.len);
                loaded_cell = c.args[0].value;
            }
            if (c.callee.kind == .builtin and std.mem.eql(u8, c.callee.name, "gc_mark")) full_calls += 1;
        },
        else => {},
    };
    try testing.expectEqual(@as(usize, 1), leaf_calls);
    try testing.expectEqual(@as(usize, 0), full_calls);
    // The marked cell is a `.load` (the buffer pointer read off self).
    try testing.expect(loaded_cell != null);
    switch (defOf(&f1, loaded_cell.?).?) {
        .load => {},
        else => try testing.expect(false),
    }

    const t1 = try renderFn(gpa, &f1, in);
    defer gpa.free(t1);
    const t2 = try renderFn(gpa, &f2, in);
    defer gpa.free(t2);
    try testing.expectEqualStrings(t1, t2);
}

test "erased struct hash copies the key into a slot and calls its Hash witness; byte-identical across relowers" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);

    var fnames = [_][]const u8{ "a", "b", "c" };
    var ftys = [_]Typecheck.Type{ Typecheck.Type.int, Typecheck.Type.int, Typecheck.Type.int };
    var offs = [_]u32{ 0, 8, 16 };
    var layouts = [_]Typecheck.Layout{.{ .name = "S", .field_names = &fnames, .field_types = &ftys, .offsets = &offs, .size = 24, .@"align" = 8 }};
    const ct = Typecheck.Type.structT(0);
    const hash_pid: u32 = 6;

    // The erased unit no longer walks the key's fields inline; it materializes the key and
    // calls the key's resolved structural `Hash` witness. Wire a one-entry derive + method
    // table so that witness resolves to `Hash$hash$s0`.
    var derives = [_]Derive.Derive{.{ .protocol_id = hash_pid, .protocol_name = "Hash", .kind = .hash, .conform_ty = ct, .ret = Typecheck.Type.int, .name = "Hash$hash$s0" }};
    var methods = [_]Typecheck.Method{.{ .recv = ct, .name = "hash", .fn_id = 0, .derive = 0, .protocol_id = hash_pid }};
    var in = intStructInputs(&layouts);
    in.methods = &methods;
    in.derives = &derives;
    in.prelude_ids = .{ .hash = hash_pid };

    const sym: Link.SymName = .{ .kind = .user_fn, .name = "descriptor$hash$s0" };

    var f1 = try lowerErased(gpa, in, ct, .hash, sym, &diags);
    defer f1.deinit(gpa);
    var f2 = try lowerErased(gpa, in, ct, .hash, sym, &diags);
    defer f2.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);

    // (a) the key is copied into a by-value slot; (b) the witness is CALLed (no `field_addr`
    // walk off the key base).
    try testing.expect(hasOp(&f1, .copy));
    try testing.expect(!hasOp(&f1, .field_addr));
    try testing.expectEqualStrings("Hash$hash$s0", firstCallName(&f1).?);

    const t1 = try renderFn(gpa, &f1, in);
    defer gpa.free(t1);
    const t2 = try renderFn(gpa, &f2, in);
    defer gpa.free(t2);
    try testing.expectEqualStrings(t1, t2);
}

test "erased eq on an int key emits an icmp eq over two loaded pointers" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const in = intStructInputs(&.{});
    const sym: Link.SymName = .{ .kind = .user_fn, .name = "descriptor$eq$int" };

    var f = try lowerErased(gpa, in, Typecheck.Type.int, .eq, sym, &diags);
    defer f.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    try testing.expect(hasCond(&f, .eq));
}

test "derive Hash: mixer is fxhash (rotl(h,5)^word)*%K — constants/shifts/lshr + operand order pinned" {
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

    // One fxhash round unwound from an accumulator `out`: out = (rotl(h,5) ^ word) *% K,
    // rotl(h,5) = shl(h,5) | lshr(h,59). Returns the inbound accumulator, the emitted K, and
    // the word value so the caller chains rounds. The `.lshr` unwrap is the guard: an
    // `.ashr` regression fails the union-field access here.
    const Round = struct {
        h_in: Ir.ValueId,
        k: i64,
        word: Ir.ValueId,
        fn of(f: *const Ir.Function, out: Ir.ValueId) !@This() {
            const mul = defOf(f, out).?.mul;
            const bx = defOf(f, mul.lhs).?.bxor;
            const bor = defOf(f, bx.lhs).?.bor;
            const shl = defOf(f, bor.lhs).?.shl;
            const lshr = defOf(f, bor.rhs).?.lshr;
            try testing.expectEqual(@as(i64, 5), defOf(f, shl.rhs).?.iconst);
            try testing.expectEqual(@as(i64, 59), defOf(f, lshr.rhs).?.iconst);
            try testing.expectEqual(shl.lhs, lshr.lhs); // both rotates read the SAME accumulator
            try testing.expect(std.meta.activeTag(defOf(f, bx.rhs).?) == .load); // word = raw field load
            return .{ .h_in = shl.lhs, .k = defOf(f, mul.rhs).?.iconst, .word = bx.rhs };
        }
    };

    const h2 = retValue(&func).?;
    const r1 = try Round.of(&func, h2); // field b (off 8)
    const r0 = try Round.of(&func, r1.h_in); // field a (off 0)
    const seed = defOf(&func, r0.h_in).?.iconst;

    // Pin the emitted constants to INDEPENDENT literals (NOT L.hash_k / L.hash_seed): a drift
    // of either symbol emits the new value and fails here, catching a silent constant change.
    try testing.expectEqual(@as(i64, 0x243F6A8885A308D3), seed);
    try testing.expectEqual(@as(i64, 0x517cc1b727220a95), r0.k);
    try testing.expectEqual(@as(i64, 0x517cc1b727220a95), r1.k);

    // Cross-check the FORMULA two independent ways over arbitrary words: fold the emitted seed
    // with the opt-stage `arith` helpers (native lshr, width 64) vs a raw-u64 fxhash reference.
    // A wrong shift amount, .ashr, swapped order, or FNV shape diverges.
    const v0: i64 = 7;
    const v1: i64 = 11;
    const mixFold = struct {
        fn f(h: i64, w: i64) i64 {
            const rot = arith.foldBin(.bor, arith.foldShift(.shl, h, 5, 64), arith.foldShift(.lshr, h, 59, 64));
            return arith.foldBin(.mul, arith.foldBin(.bxor, rot, w), 0x517cc1b727220a95);
        }
    }.f;
    const fxRef = struct {
        fn f(h: i64, w: i64) i64 {
            const u: u64 = @bitCast(h);
            const rot: i64 = @bitCast((u << 5) | (u >> 59));
            return (rot ^ w) *% 0x517cc1b727220a95;
        }
    }.f;
    try testing.expectEqual(fxRef(fxRef(seed, v0), v1), mixFold(mixFold(seed, v0), v1));
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

test "derive Eq on a Ref field compares by pointer identity (icmp eq, no witness call)" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);

    // layouts[0] = `struct S { r: Ref[int] }`; layouts[1] = the reified `Ref$int` box,
    // tagged native_family .ref so `isRefTy` recognizes the field as a managed box.
    var s_fnames = [_][]const u8{"r"};
    var s_ftys = [_]Typecheck.Type{Typecheck.Type.structT(1)};
    var s_offs = [_]u32{0};
    var ref_fnames = [_][]const u8{"0"};
    var ref_ftys = [_]Typecheck.Type{Typecheck.Type.int};
    var ref_offs = [_]u32{0};
    var layouts = [_]Typecheck.Layout{
        .{ .name = "S", .field_names = &s_fnames, .field_types = &s_ftys, .offsets = &s_offs, .size = 8, .@"align" = 8 },
        .{ .name = "Ref$int", .field_names = &ref_fnames, .field_types = &ref_ftys, .offsets = &ref_offs, .size = 8, .@"align" = 8, .native_family = .ref },
    };
    const sym: Link.SymName = .{ .kind = .user_fn, .name = "D" };
    var func = try lowerDeriveEq(gpa, intStructInputs(&layouts), .{ .protocol_id = 0, .protocol_name = "Eq", .kind = .eq, .conform_ty = Typecheck.Type.structT(0), .ret = Typecheck.Type.@"bool" }, sym, &diags);
    defer func.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);

    // Identity compare: an `icmp eq`, and crucially NO witness `.call` (a structural deref).
    try testing.expect(hasCond(&func, .eq));
    var calls: usize = 0;
    for (func.blocks) |blk| for (blk.instrs) |ins| switch (ins.op) {
        .call => calls += 1,
        else => {},
    };
    try testing.expectEqual(@as(usize, 0), calls);
}

/// Count the `bl gc_mark` sites in an emitted trace body: every `.call` whose callee is
/// the `gc_mark` builtin.
fn gcMarkCallSites(func: *const Ir.Function) usize {
    var n: usize = 0;
    for (func.blocks) |blk| for (blk.instrs) |ins| switch (ins.op) {
        .call => |c| if (c.callee.kind == .builtin and std.mem.eql(u8, c.callee.name, "gc_mark")) {
            n += 1;
        },
        else => {},
    };
    return n;
}

/// Count the sibling-`trace` recursion sites in an emitted trace body: every `.call` to a
/// named `user_fn` (the by-value-aggregate `.trace_call` witness), excluding the `gc_mark`
/// builtin boundary.
fn userFnCallSites(func: *const Ir.Function) usize {
    var n: usize = 0;
    for (func.blocks) |blk| for (blk.instrs) |ins| switch (ins.op) {
        .call => |c| if (c.callee.kind == .user_fn) {
            n += 1;
        },
        else => {},
    };
    return n;
}

test "derive Trace marks exactly the managed fields; a scalar struct traces to a no-op" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const sym: Link.SymName = .{ .kind = .user_fn, .name = "T" };

    // struct S { r: Ref[int], n: int, r2: Ref[int] } — two managed fields flanking a scalar.
    // layouts[1] is the reified Ref box (native_family .ref). The recipe's field witnesses
    // are the resolver's output: mark / no-op / mark, in layout order.
    {
        var s_fnames = [_][]const u8{ "r", "n", "r2" };
        var s_ftys = [_]Typecheck.Type{ Typecheck.Type.structT(1), Typecheck.Type.int, Typecheck.Type.structT(1) };
        var s_offs = [_]u32{ 0, 8, 16 };
        var ref_fnames = [_][]const u8{"0"};
        var ref_ftys = [_]Typecheck.Type{Typecheck.Type.int};
        var ref_offs = [_]u32{0};
        var layouts = [_]Typecheck.Layout{
            .{ .name = "S", .field_names = &s_fnames, .field_types = &s_ftys, .offsets = &s_offs, .size = 24, .@"align" = 8 },
            .{ .name = "Ref$int", .field_names = &ref_fnames, .field_types = &ref_ftys, .offsets = &ref_offs, .size = 8, .@"align" = 8, .native_family = .ref },
        };
        var fws = [_]Derive.FieldWitness{ .trace_mark, .inline_kind, .trace_mark };
        const d: Derive.Derive = .{ .protocol_id = 0, .protocol_name = "Trace", .kind = .trace, .conform_ty = Typecheck.Type.structT(0), .ret = Typecheck.Type.unit, .field_witnesses = &fws };
        var func = try lowerDeriveTrace(gpa, intStructInputs(&layouts), d, sym, &diags);
        defer func.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), diags.items.len);

        // One `bl gc_mark` per managed field (2), and the scalar field is NOT marked.
        try testing.expectEqual(@as(usize, 2), gcMarkCallSites(&func));
        // The two marked offsets are the managed fields' offsets (0 and 16), in layout order.
        const offs = try selfFieldOffsets(gpa, &func);
        defer gpa.free(offs);
        try testing.expectEqualSlices(u32, &[_]u32{ 0, 16 }, offs);
    }

    // A plain scalar struct { a: int, b: int } traces to an EMPTY body — no mark sites.
    {
        var fnames = [_][]const u8{ "a", "b" };
        var ftys = [_]Typecheck.Type{ Typecheck.Type.int, Typecheck.Type.int };
        var offs = [_]u32{ 0, 8 };
        var layouts = [_]Typecheck.Layout{.{ .name = "P", .field_names = &fnames, .field_types = &ftys, .offsets = &offs, .size = 16, .@"align" = 8 }};
        var fws = [_]Derive.FieldWitness{ .inline_kind, .inline_kind };
        const d: Derive.Derive = .{ .protocol_id = 0, .protocol_name = "Trace", .kind = .trace, .conform_ty = Typecheck.Type.structT(0), .ret = Typecheck.Type.unit, .field_witnesses = &fws };
        var func = try lowerDeriveTrace(gpa, intStructInputs(&layouts), d, sym, &diags);
        defer func.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), diags.items.len);
        try testing.expectEqual(@as(usize, 0), gcMarkCallSites(&func));
    }
}

test "derive Trace recurses into a by-value aggregate field, marks nothing itself (the boundary is the callee)" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const sym: Link.SymName = .{ .kind = .user_fn, .name = "T" };

    // struct Outer { i: Inner, n: int } where Inner { r: Ref[int] } holds the managed field.
    // Outer's own trace does NOT mark: field `i` is a by-value aggregate that itself derives
    // Trace, so Outer routes it to Inner's sibling witness (`.trace_call`); `n` is a no-op.
    // The `gc_mark` boundary lives one level down, in Inner's body — this is the spike-core
    // by-value recursion path.
    var outer_fnames = [_][]const u8{ "i", "n" };
    var outer_ftys = [_]Typecheck.Type{ Typecheck.Type.structT(1), Typecheck.Type.int };
    var outer_offs = [_]u32{ 0, 8 };
    var inner_fnames = [_][]const u8{"r"};
    var inner_ftys = [_]Typecheck.Type{Typecheck.Type.structT(2)};
    var inner_offs = [_]u32{0};
    var ref_fnames = [_][]const u8{"0"};
    var ref_ftys = [_]Typecheck.Type{Typecheck.Type.int};
    var ref_offs = [_]u32{0};
    var layouts = [_]Typecheck.Layout{
        .{ .name = "Outer", .field_names = &outer_fnames, .field_types = &outer_ftys, .offsets = &outer_offs, .size = 16, .@"align" = 8 },
        .{ .name = "Inner", .field_names = &inner_fnames, .field_types = &inner_ftys, .offsets = &inner_offs, .size = 8, .@"align" = 8 },
        .{ .name = "Ref$int", .field_names = &ref_fnames, .field_types = &ref_ftys, .offsets = &ref_offs, .size = 8, .@"align" = 8, .native_family = .ref },
    };

    // Outer: `.trace_call` for the Inner field, no-op for the scalar.
    var outer_fws = [_]Derive.FieldWitness{ .{ .trace_call = "trace$Inner" }, .inline_kind };
    const outer_d: Derive.Derive = .{ .protocol_id = 0, .protocol_name = "Trace", .kind = .trace, .conform_ty = Typecheck.Type.structT(0), .ret = Typecheck.Type.unit, .field_witnesses = &outer_fws };
    var outer_f = try lowerDeriveTrace(gpa, intStructInputs(&layouts), outer_d, sym, &diags);
    defer outer_f.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    // Outer routes one recursion to Inner's witness and marks NOTHING itself.
    try testing.expectEqual(@as(usize, 1), userFnCallSites(&outer_f));
    try testing.expectEqual(@as(usize, 0), gcMarkCallSites(&outer_f));

    // Inner: the managed `Ref` field marks — the boundary the recursion terminates at.
    var inner_fws = [_]Derive.FieldWitness{.trace_mark};
    const inner_d: Derive.Derive = .{ .protocol_id = 0, .protocol_name = "Trace", .kind = .trace, .conform_ty = Typecheck.Type.structT(1), .ret = Typecheck.Type.unit, .field_witnesses = &inner_fws };
    var inner_f = try lowerDeriveTrace(gpa, intStructInputs(&layouts), inner_d, sym, &diags);
    defer inner_f.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    try testing.expectEqual(@as(usize, 1), gcMarkCallSites(&inner_f));
    try testing.expectEqual(@as(usize, 0), userFnCallSites(&inner_f));
}

test "derive Trace on a payload enum marks the managed field of the active variant only" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const sym: Link.SymName = .{ .kind = .user_fn, .name = "T" };

    // enum E { Empty, Boxed(Ref[int]) } — the flattened field-witness list is the variants'
    // payloads concatenated in decl order: Empty contributes none, Boxed contributes one
    // `.trace_mark`. The get_tag dispatch ladder emits ONE `gc_mark` (on the Boxed arm); the
    // base-offset arithmetic must land it on the right slice of the flat witness list.
    var ref_fnames = [_][]const u8{"0"};
    var ref_ftys = [_]Typecheck.Type{Typecheck.Type.int};
    var ref_offs = [_]u32{0};
    var layouts = [_]Typecheck.Layout{
        .{ .name = "Ref$int", .field_names = &ref_fnames, .field_types = &ref_ftys, .offsets = &ref_offs, .size = 8, .@"align" = 8, .native_family = .ref },
    };

    var empty_offs = [_]u32{};
    var empty_fnames = [_][]const u8{};
    var empty_ftys = [_]Typecheck.Type{};
    var boxed_fnames = [_][]const u8{"0"};
    var boxed_ftys = [_]Typecheck.Type{Typecheck.Type.structT(0)};
    var boxed_offs = [_]u32{0};
    var variants = [_]Typecheck.VariantLayout{
        .{ .name = "Empty", .form = .unit, .field_names = &empty_fnames, .field_types = &empty_ftys, .offsets = &empty_offs },
        .{ .name = "Boxed", .form = .tuple, .field_names = &boxed_fnames, .field_types = &boxed_ftys, .offsets = &boxed_offs },
    };
    var enum_layouts = [_]Typecheck.EnumLayout{
        .{ .name = "E", .variants = &variants, .tag_size = 8, .payload_off = 8, .size = 16, .@"align" = 8 },
    };

    var in = intStructInputs(&layouts);
    in.enum_layouts = &enum_layouts;

    var fws = [_]Derive.FieldWitness{.trace_mark}; // flattened: [] ++ [Ref]
    const d: Derive.Derive = .{ .protocol_id = 0, .protocol_name = "Trace", .kind = .trace, .conform_ty = Typecheck.Type.enumT(0), .ret = Typecheck.Type.unit, .field_witnesses = &fws };
    var func = try lowerDeriveTrace(gpa, in, d, sym, &diags);
    defer func.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    try testing.expectEqual(@as(usize, 1), gcMarkCallSites(&func));
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
