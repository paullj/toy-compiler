//! The pure `Type` algebra: kinds, constants, predicates, relations, and the
//! flat leaf serializer.
//!
//! A byte-foldable value type (not a tagged union) plus the operations over it —
//! `eql`/`assignable`/`orderLeaf`, the integer descriptor, and `appendKeyBytes`.
//! Imports `std` only, so it is a leaf module with no dependency on the AST or the
//! layout engine: the single home for the type model the checker, fingerprint,
//! dedup, and mangling all route through.
//!
//! HIDES: that `IntDesc` bit-packs sign+width into the spare pad byte; that a
//! `type_var` ordinal and an `app` index both alias `struct_id`; the exact leaf
//! key byte layout.

const std = @import("std");

/// The type kind. `invalid` is the poison/error type: it absorbs further errors
/// so one mistake produces one diagnostic. `@"struct"` carries a `struct_id`
/// indexing the per-program struct table.
///
/// `type_var` (APPENDED — ordinals are frozen) is a CHECK-TIME type variable:
/// it reuses `struct_id` as the generic-parameter ordinal and exists only inside a
/// generic template's decoded signature. It is substituted away to a concrete kind
/// during the serial monomorphization tail, BEFORE any `node_types` slot is frozen,
/// so it never reaches lower/codegen/layout (Debug-asserted there).
///
/// `app` (APPENDED — ordinals frozen) is a CHECK-TIME composite type
/// `Ctor[args..]` (a generic-struct application like `Box[int]`): it reuses
/// `struct_id` as an index into the interned composite table (`symbols/Composite.zig`).
/// Every reachable ground `app` is REIFIED to a fresh ordinary `struct_id` (its
/// `Layout` registered on the live tables) during the serial monomorphization tail,
/// BEFORE the layout snapshot, so codegen/fingerprint/cache see only plain `structT`
/// and an `app` never reaches lower/codegen (Debug-asserted there).
/// `rawptr` (APPENDED — ordinals frozen) is an opaque, UNMANAGED C `void*`: a
/// plain scalar (size=align=8) with no laid-out referent and no nominal id, so
/// `idField(.rawptr) == .none` and all rawptrs are equal by kind. It carries no
/// heap-reference semantics (`isReference()` is false); its only functional client
/// this milestone is as an `extern` param/return type.
pub const Kind = enum(u8) { invalid, unit, int, bool, str, never, @"struct", @"enum", type_var, app, float, rawptr };

/// The width class of an integer type. `plat` is the platform-word `int`/`uint`
/// (currently 64-bit); the numbered classes are the fixed-width `int8..int64`.
/// `enum(u3)` so it fits the 8-bit `IntDesc` packed struct with room to spare.
pub const IntWidth = enum(u3) { plat, w8, w16, w32, w64 };

/// The sign+width descriptor an `.int`-kind `Type` carries in its otherwise-spare
/// pad byte. Defaults to signed platform `int`, so `Type.int` (`.{ .kind = .int }`)
/// stays byte-identical to before this field existed. `packed struct(u8)` keeps
/// `@sizeOf(Type)==12` and makes the descriptor a single deterministic byte for the
/// fingerprint/dedup/order serializers.
pub const IntDesc = packed struct(u8) { signed: bool = true, width: IntWidth = .plat, _pad: u4 = 0 };

/// Which struct field of a `Type` carries a given kind's nominal identity. The single
/// home for the per-kind id-selection every identity serializer (`eql`, the flat/order
/// keys, the coherence key, the derive/mono mangles) must agree on: `.none` kinds have
/// no nominal id.
pub const IdField = enum { none, struct_id, enum_id };

/// A type. A byte-foldable struct (not a tagged union) so it preserves `@memset`,
/// `node_types` triviality, and a stable fingerprint basis. A `@"struct"` kind
/// carries an index into the struct table; a `@"enum"` kind an index into the
/// parallel enum table; all other kinds leave both `no_struct`.
pub const Type = struct {
    kind: Kind,
    int_desc: IntDesc = .{},
    struct_id: u32 = no_struct,
    enum_id: u32 = no_struct,

    pub const no_struct: u32 = std.math.maxInt(u32);

    pub const invalid: Type = .{ .kind = .invalid };
    pub const unit: Type = .{ .kind = .unit };
    pub const int: Type = .{ .kind = .int };
    pub const @"bool": Type = .{ .kind = .bool };
    pub const str: Type = .{ .kind = .str };
    pub const never: Type = .{ .kind = .never };
    pub const float: Type = .{ .kind = .float };
    pub const rawptr: Type = .{ .kind = .rawptr };

    pub const uint: Type = .{ .kind = .int, .int_desc = .{ .signed = false } };
    pub const int8: Type = .{ .kind = .int, .int_desc = .{ .width = .w8 } };
    pub const int16: Type = .{ .kind = .int, .int_desc = .{ .width = .w16 } };
    pub const int32: Type = .{ .kind = .int, .int_desc = .{ .width = .w32 } };
    pub const int64: Type = .{ .kind = .int, .int_desc = .{ .width = .w64 } };
    pub const uint8: Type = .{ .kind = .int, .int_desc = .{ .signed = false, .width = .w8 } };
    pub const uint16: Type = .{ .kind = .int, .int_desc = .{ .signed = false, .width = .w16 } };
    pub const uint32: Type = .{ .kind = .int, .int_desc = .{ .signed = false, .width = .w32 } };
    pub const uint64: Type = .{ .kind = .int, .int_desc = .{ .signed = false, .width = .w64 } };

    pub fn structT(id: u32) Type {
        return .{ .kind = .@"struct", .struct_id = id };
    }

    pub fn enumT(id: u32) Type {
        return .{ .kind = .@"enum", .enum_id = id };
    }

    /// A check-time type variable for generic-parameter `ordinal`. Reuses
    /// `struct_id` as the ordinal — NO widening, `@sizeOf(Type)` is unchanged.
    pub fn typeVar(ordinal: u32) Type {
        return .{ .kind = .type_var, .struct_id = ordinal };
    }

    pub fn isTypeVar(t: Type) bool {
        return t.kind == .type_var;
    }

    /// The generic-parameter ordinal of a `type_var` (read of the reused id field).
    pub fn typeVarOrd(t: Type) u32 {
        return t.struct_id;
    }

    /// A check-time composite `Ctor[args..]` type. Reuses `struct_id` as the
    /// interned composite-table index (`symbols/Composite.zig`) — NO widening.
    pub fn app(idx: u32) Type {
        return .{ .kind = .app, .struct_id = idx };
    }

    pub fn isApp(t: Type) bool {
        return t.kind == .app;
    }

    /// The composite-table index of an `app` (read of the reused id field). Two
    /// structurally-equal `app`s share an index within a run (content-addressed
    /// interning), so `eql`-by-id is correct.
    pub fn appIdx(t: Type) u32 {
        return t.struct_id;
    }

    /// Which field carries this kind's nominal identity. Exhaustive (no `else`) so a
    /// NEW `Kind` is a build error here rather than a silent identity drop in one of the
    /// serializers that route through this. `type_var`/`app` alias `struct_id` (the id
    /// space they reuse), so this hides that aliasing from every caller.
    pub fn idField(kind: Kind) IdField {
        return switch (kind) {
            .@"struct", .type_var, .app => .struct_id,
            .@"enum" => .enum_id,
            .invalid, .unit, .int, .bool, .str, .never, .float, .rawptr => .none,
        };
    }

    /// This type's nominal id (0 for a `.none` kind), read through the field `idField`
    /// selects — the one accessor every serializer's per-kind id read collapses to.
    pub fn nominalId(t: Type) u32 {
        return switch (idField(t.kind)) {
            .none => 0,
            .struct_id => t.struct_id,
            .enum_id => t.enum_id,
        };
    }

    /// Whether this kind's identity includes the `int_desc` sign/width byte. The single
    /// home for the int-only descriptor rule the coherence key once dropped: an identity
    /// serializer either folds `int_desc` for exactly these kinds or under-discriminates.
    pub fn carriesIntDesc(kind: Kind) bool {
        return kind == .int;
    }

    pub fn eql(a: Type, b: Type) bool {
        if (a.kind != b.kind) return false;
        if (carriesIntDesc(a.kind) and @as(u8, @bitCast(a.int_desc)) != @as(u8, @bitCast(b.int_desc)))
            return false;
        return switch (idField(a.kind)) {
            .none => true,
            .struct_id, .enum_id => a.nominalId() == b.nominalId(),
        };
    }

    /// The one assignability relation: is a value of type `got` acceptable where a
    /// `want` is expected? Mirrors `merge`'s discipline so the two stay siblings —
    /// `invalid` is poison (either side absorbs, so a root error yields exactly one
    /// diagnostic and never cascades), `never` is bottom (a value that never exists
    /// fits any slot), and otherwise assignability is structural equality with NO
    /// implicit coercion. Returning `true` for poison/`never` means a caller guards
    /// `if (!assignable(want, got)) emit(...)` and stays silent on already-reported
    /// or unreachable values — exactly the manual `kind != .invalid` guards it
    /// replaces.
    pub fn assignable(want: Type, got: Type) bool {
        if (want.kind == .invalid or got.kind == .invalid) return true;
        if (got.kind == .never) return true;
        return eql(want, got);
    }

    pub fn isStruct(t: Type) bool {
        return t.kind == .@"struct";
    }

    pub fn isEnum(t: Type) bool {
        return t.kind == .@"enum";
    }

    pub fn isInteger(t: Type) bool {
        return t.kind == .int;
    }

    pub fn isFloat(t: Type) bool {
        return t.kind == .float;
    }

    pub fn isNumeric(t: Type) bool {
        return t.kind == .int or t.kind == .float;
    }

    pub fn isSigned(t: Type) bool {
        return t.kind == .int and t.int_desc.signed;
    }

    pub fn isUnsignedInt(t: Type) bool {
        return t.kind == .int and !t.int_desc.signed;
    }

    pub fn isPlatformInt(t: Type) bool {
        return t.kind == .int and t.int_desc.width == .plat;
    }

    /// The bit width of an integer `Type` (platform `int`/`uint` counts as 64).
    pub fn intBits(t: Type) u16 {
        return switch (t.int_desc.width) {
            .plat, .w64 => 64,
            .w8 => 8,
            .w16 => 16,
            .w32 => 32,
        };
    }

    pub fn isRawPtr(t: Type) bool {
        return t.kind == .rawptr;
    }

    /// Leaf-`Type` view: ALWAYS false — do not use it to detect a managed box. A bare
    /// `Type` has no table, and reference-ness lives in a side table: an unmanaged `rawptr`
    /// carries NO heap-reference semantics (a bare C `void*`), and a managed box is a
    /// marker-tagged struct only the table-aware authority can recognize. To ask whether a
    /// reified type is a box, call `Engine.isRefStruct(t, table)` (the layout snapshot or
    /// the live struct table); pre-reify, use `BodyChecker.isRefPayload`. This stub exists
    /// only so the `rawptr`-is-not-a-reference invariant reads at the leaf.
    pub fn isReference(t: Type) bool {
        _ = t;
        return false;
    }

    /// A builtin scalar type — one with no laid-out referent (`int`/`bool`/`str`/`unit`).
    pub fn isScalar(t: Type) bool {
        return t.kind == .int or t.kind == .bool or t.kind == .str or t.kind == .unit or t.kind == .rawptr;
    }

    /// The source spelling of an integer type (`int`/`uint`/`int8`.../`uint64`).
    /// Exhaustive over `IntWidth` (no `else` arm) so a new width class is a build
    /// error here rather than a silently-wrong name. Static strings — no alloc, so a
    /// borrowing accessor (`refs.typeName`, `Mono.mangle`) can return the slice directly.
    pub fn intName(t: Type) []const u8 {
        return switch (t.int_desc.width) {
            .plat => if (t.int_desc.signed) "int" else "uint",
            .w8 => if (t.int_desc.signed) "int8" else "uint8",
            .w16 => if (t.int_desc.signed) "int16" else "uint16",
            .w32 => if (t.int_desc.signed) "int32" else "uint32",
            .w64 => if (t.int_desc.signed) "int64" else "uint64",
        };
    }

    /// Append this type's flat leaf key — `(kind, struct_id, enum_id)` at fixed
    /// width — to `buf`. The single serializer every dedup/structural key routes
    /// through (`Composite.writeFlatKey`/`writeStructuralKey`, `Mono.writeKey`), so
    /// the byte encoding stays identical across them.
    pub fn appendKeyBytes(t: Type, gpa: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        try buf.append(gpa, @intFromEnum(t.kind));
        // The sign/width descriptor only for `.int` (self-delimiting off the leading
        // kind byte), so every non-int key stays byte-identical while `int8`/`int64`
        // never fold to one interned/dedup/mangle key.
        if (carriesIntDesc(t.kind)) try buf.append(gpa, @as(u8, @bitCast(t.int_desc)));
        var w: [4]u8 = undefined;
        std.mem.writeInt(u32, &w, t.struct_id, .little);
        try buf.appendSlice(gpa, &w);
        std.mem.writeInt(u32, &w, t.enum_id, .little);
        try buf.appendSlice(gpa, &w);
    }

    /// The canonical leaf ordering over `(kind, struct_id, enum_id)` — the tie-break
    /// spine of `Mono.lessThan`'s total instance order.
    pub fn orderLeaf(a: Type, b: Type) std.math.Order {
        if (a.kind != b.kind) return std.math.order(@intFromEnum(a.kind), @intFromEnum(b.kind));
        // Two integer widths sharing the `.int` kind tie-break on the descriptor byte,
        // so `Mono.lessThan` stays a total order across widths (else the `-jN` mono sort
        // is nondeterministic).
        if (carriesIntDesc(a.kind) and @as(u8, @bitCast(a.int_desc)) != @as(u8, @bitCast(b.int_desc)))
            return std.math.order(@as(u8, @bitCast(a.int_desc)), @as(u8, @bitCast(b.int_desc)));
        if (a.struct_id != b.struct_id) return std.math.order(a.struct_id, b.struct_id);
        return std.math.order(a.enum_id, b.enum_id);
    }

    comptime {
        // The byte-foldable `Type` never widens (locked decision): `type_var` reuses
        // `struct_id` as the ordinal, so appending the kind does not grow the struct.
        std.debug.assert(@sizeOf(Type) == 12);
    }
};

const testing = std.testing;

test "algebra: assignable structural equality with no coercion" {
    try testing.expect(Type.assignable(Type.int, Type.int));
    try testing.expect(Type.assignable(Type.structT(3), Type.structT(3)));
    try testing.expect(!Type.assignable(Type.int, Type.bool));
    try testing.expect(!Type.assignable(Type.structT(1), Type.structT(2)));
    try testing.expect(!Type.assignable(Type.int, Type.str));
}

test "algebra: invalid is poison (either side absorbs)" {
    try testing.expect(Type.assignable(Type.invalid, Type.int));
    try testing.expect(Type.assignable(Type.int, Type.invalid));
    try testing.expect(Type.assignable(Type.invalid, Type.invalid));
}

test "algebra: never is bottom (fits any want)" {
    try testing.expect(Type.assignable(Type.int, Type.never));
    try testing.expect(Type.assignable(Type.structT(0), Type.never));
    try testing.expect(!Type.assignable(Type.never, Type.int));
}

test "algebra: eql discriminates struct/enum ids" {
    try testing.expect(Type.eql(Type.structT(2), Type.structT(2)));
    try testing.expect(!Type.eql(Type.structT(2), Type.structT(3)));
    try testing.expect(Type.eql(Type.enumT(1), Type.enumT(1)));
    try testing.expect(!Type.eql(Type.enumT(1), Type.enumT(2)));
    // same id, different kind: not equal (a struct id is not an enum id).
    try testing.expect(!Type.eql(Type.structT(0), Type.enumT(0)));
}

test "algebra: type_var round-trips its ordinal and eql is per-ordinal" {
    // The byte-foldable Type never widens: the ordinal rides `struct_id`.
    try testing.expectEqual(@as(usize, 12), @sizeOf(Type));
    const t0 = Type.typeVar(0);
    const t1 = Type.typeVar(1);
    try testing.expect(t0.isTypeVar());
    try testing.expectEqual(@as(u32, 0), t0.typeVarOrd());
    try testing.expectEqual(@as(u32, 1), t1.typeVarOrd());
    // Distinct ordinals are distinct types; same ordinal is equal.
    try testing.expect(Type.eql(t0, Type.typeVar(0)));
    try testing.expect(!Type.eql(t0, t1));
    // A type_var is not a struct even though it reuses struct_id (kind discriminates).
    try testing.expect(!Type.eql(t0, Type.structT(0)));
    try testing.expect(!t0.isStruct());
}

test "algebra: app round-trips its composite index and eql is per-index" {
    // The byte-foldable Type never widens: the composite index rides `struct_id`.
    try testing.expectEqual(@as(usize, 12), @sizeOf(Type));
    const a0 = Type.app(0);
    const a1 = Type.app(1);
    try testing.expect(a0.isApp());
    try testing.expectEqual(@as(u32, 0), a0.appIdx());
    try testing.expectEqual(@as(u32, 1), a1.appIdx());
    // Same interned index => eql (structurally-equal Apps share an index); distinct
    // indices are distinct types.
    try testing.expect(Type.eql(a0, Type.app(0)));
    try testing.expect(!Type.eql(a0, a1));
    // An app is neither a struct nor a type_var even though all three reuse struct_id
    // (kind discriminates), so eql-by-id never confuses them.
    try testing.expect(!Type.eql(a0, Type.structT(0)));
    try testing.expect(!Type.eql(a0, Type.typeVar(0)));
    try testing.expect(!a0.isStruct());
    // assignable delegates to eql: same-index apps are assignable, distinct are not.
    try testing.expect(Type.assignable(a0, Type.app(0)));
    try testing.expect(!Type.assignable(a0, a1));
}

test "algebra: appendKeyBytes discriminates exactly what eql discriminates" {
    // The dedup/interning correctness property the flat serializer exists to
    // guarantee: an under-discriminating key folds two distinct types into one
    // interned entry (miscompile), an over-discriminating one splits equal types.
    // Assert eql(a,b) <=> byteEqual(key(a), key(b)) over the canonical constructor
    // set — every kind, every IntWidth x sign, a few struct/enum/type_var/app ids.
    // Restricted to canonical values (non-id kinds carry no_struct): eql ignores
    // struct_id/enum_id for those kinds while appendKeyBytes always writes them.
    const gpa = testing.allocator;
    const canon = [_]Type{
        Type.invalid,   Type.unit,       Type.bool,       Type.str,        Type.never,
        Type.int,       Type.uint,       Type.int8,       Type.int16,      Type.int32,
        Type.int64,     Type.uint8,      Type.uint16,     Type.uint32,     Type.uint64,
        Type.structT(0), Type.structT(1), Type.enumT(0),  Type.enumT(1),   Type.typeVar(0),
        Type.typeVar(1), Type.app(0),     Type.app(1),     Type.float,      Type.rawptr,
    };
    for (canon) |a| {
        for (canon) |b| {
            var ka: std.ArrayList(u8) = .empty;
            defer ka.deinit(gpa);
            var kb: std.ArrayList(u8) = .empty;
            defer kb.deinit(gpa);
            try a.appendKeyBytes(gpa, &ka);
            try b.appendKeyBytes(gpa, &kb);
            try testing.expectEqual(Type.eql(a, b), std.mem.eql(u8, ka.items, kb.items));
        }
    }
}

test "algebra: float is a distinct nominal-id-free type" {
    try testing.expectEqual(@as(usize, 12), @sizeOf(Type));
    try testing.expect(Type.float.isFloat());
    try testing.expect(Type.float.isNumeric());
    try testing.expect(Type.int.isNumeric());
    try testing.expect(!Type.float.isInteger());
    try testing.expect(!Type.bool.isNumeric());
    // float carries no int descriptor and no nominal id; distinct from int.
    try testing.expect(!Type.eql(Type.float, Type.int));
    try testing.expect(Type.eql(Type.float, Type.float));
    try testing.expectEqual(IdField.none, Type.idField(.float));
}

test "algebra: rawptr is a distinct nominal-id-free 8-byte scalar" {
    try testing.expectEqual(@as(usize, 12), @sizeOf(Type));
    try testing.expect(Type.rawptr.isRawPtr());
    try testing.expect(Type.rawptr.isScalar());
    try testing.expect(!Type.rawptr.isReference());
    // A distinct type: not int, not str, no nominal id; equal only to itself.
    try testing.expect(Type.eql(Type.rawptr, Type.rawptr));
    try testing.expect(!Type.eql(Type.rawptr, Type.int));
    try testing.expect(!Type.eql(Type.rawptr, Type.str));
    try testing.expectEqual(IdField.none, Type.idField(.rawptr));
}

test "algebra: integer widths are distinct byte-foldable types" {
    // The width/sign descriptor rides the otherwise-spare pad byte: no widening.
    try testing.expectEqual(@as(usize, 12), @sizeOf(Type));

    // eql discriminates width AND sign; plain int is unchanged.
    try testing.expect(!Type.eql(Type.int8, Type.int64));
    try testing.expect(!Type.eql(Type.int8, Type.int));
    try testing.expect(!Type.eql(Type.int, Type.uint));
    try testing.expect(Type.eql(Type.int, Type.int));
    try testing.expect(Type.eql(Type.int8, Type.int8));

    // Predicates.
    try testing.expect(Type.int8.isInteger() and Type.int8.isSigned() and !Type.int8.isPlatformInt());
    try testing.expect(Type.uint8.isInteger() and !Type.uint8.isSigned());
    try testing.expect(Type.int.isPlatformInt());
    try testing.expect(!Type.bool.isInteger());

    // Names.
    try testing.expectEqualStrings("int8", Type.int8.intName());
    try testing.expectEqualStrings("uint32", Type.uint32.intName());
    try testing.expectEqualStrings("int", Type.int.intName());
    try testing.expectEqualStrings("uint", Type.uint.intName());

    // appendKeyBytes separates widths but leaves plain int's key at one byte + ids.
    const gpa = testing.allocator;
    var k8: std.ArrayList(u8) = .empty;
    defer k8.deinit(gpa);
    var k64: std.ArrayList(u8) = .empty;
    defer k64.deinit(gpa);
    var ki: std.ArrayList(u8) = .empty;
    defer ki.deinit(gpa);
    try Type.int8.appendKeyBytes(gpa, &k8);
    try Type.int64.appendKeyBytes(gpa, &k64);
    try Type.int.appendKeyBytes(gpa, &ki);
    try testing.expect(!std.mem.eql(u8, k8.items, k64.items));
    try testing.expect(!std.mem.eql(u8, k8.items, ki.items));

    // orderLeaf is a strict total order across widths.
    try testing.expect(Type.orderLeaf(Type.int8, Type.int64) != .eq);
    try testing.expect(Type.orderLeaf(Type.int8, Type.int64) == Type.orderLeaf(Type.int8, Type.int64));
}
