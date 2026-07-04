//! Auto-derive recipe table — a peer data module (M18).
//!
//! A `Derive` is one authorized structural-derive `(protocol, concrete type)` pair
//! reified to a SOURCE-LESS synthetic per-fn codegen unit (no AST). The type
//! checker's serial monomorphization/synthesis barrier produces the table (from the
//! derive requests Pass C recorded, plus a fixpoint over nested aggregate fields);
//! codegen enumerates it as a THIRD class of lowerable units after the base fns +
//! Mono instances; `lower` walks the concrete layout to emit the unit's IR with no
//! `node_types`. This module is the SINGLE home for the canonical logic (the sort
//! key, the worklist dedup key, the mangled `SymName`) so the consumers cannot drift.
//!
//! Determinism is SACRED: derive requests are discovered in PARALLEL Pass C, so the
//! table is deduped (`writeKey`) then CANONICALLY SORTED (`lessThan`, by
//! `(protocol, kind, type-id)`) BEFORE any name is minted or any unit enumerated —
//! so the recipe set, its synthetic ids/names, and the enumeration order are a pure
//! function of source, never hashmap/thread order.

const std = @import("std");
const Type = @import("../layout/Engine.zig").Type;

/// Which derive this recipe carries. Append-only (mirrors `Token.Tag`/`Node.Tag`
/// discipline): M19 adds `.ord`, M20 `.hash`, M22 `.display`. The ordinal folds into
/// the mangled name + the sort key, so it must stay stable.
pub const Kind = enum(u8) { eq };

/// The method a `Kind` synthesizes (a pure function of the kind). Used for the
/// mangled name segment; `Eq` derives an `eq` method.
pub fn methodName(k: Kind) []const u8 {
    return switch (k) {
        .eq => "eq",
    };
}

/// How one field/payload of the conforming type contributes to the derived method,
/// RESOLVED ONCE in the synthesis barrier (on the live method table) so the emitter
/// never re-resolves and the fingerprint can fold the resolved identity. The `name`
/// slices are BORROWED (a sibling derive's minted name, a Mono instance's mangled
/// name, or a resolve-result fn name) — all outlive codegen, none owned here.
pub const FieldEq = union(enum) {
    /// A scalar (int/bool/str) field the emitter compares inline by layout kind —
    /// no witness symbol. (Unit fields are rejected by T0007, so never occur.)
    inline_kind,
    /// An aggregate (struct / empty-payload enum) field with an `eq` witness: call it
    /// `witness(field_self, field_other) -> bool`.
    eq_call: []const u8,
    /// An Ord-only aggregate field: no `eq` witness, but a `cmp` witness exists — call
    /// `cmp(field_self, field_other)`, read the returned `Ordering` tag, compare `== eq`.
    cmp_eq: []const u8,
};

/// One authorized structural-derive recipe. `field_witnesses`, `params`, and `name`
/// are OWNED (freed by the owning `GraphResult`); `protocol_name` borrows the
/// prelude/source `ProtocolSym.name` (outlives codegen); the `FieldEq` name slices
/// are borrowed (see `FieldEq`).
pub const Derive = struct {
    /// The protocol this recipe witnesses (`Eq` in M18).
    protocol_id: u32,
    protocol_name: []const u8,
    kind: Kind,
    /// The ground concrete type this derives the protocol for (a `structT`/`enumT`).
    conform_ty: Type,
    /// Per struct field (in layout/field order) / empty for an empty-payload enum:
    /// the resolved field-eq recipe. OWNED outer slice.
    field_witnesses: []const FieldEq = &.{},
    /// The synthetic method's params `[conform_ty, conform_ty]` — an OWNED heap slice
    /// so `AstWalk.CallVisitor.foldWitness` may borrow it into a `Fingerprint.Sig`
    /// (which stores the slice by reference; a by-value `[2]Type` local would dangle).
    params: []const Type = &.{},
    /// Owning module id (a sentinel; a source-less unit belongs to no real decl, but
    /// codegen threads a module for the per-module tree/resolutions it never reads).
    mod: u32 = 0,
    /// The mangled synthetic symbol. Minted in canonical order after the sort;
    /// `undefined` until then.
    name: []const u8 = undefined,
};

/// True when `conform_ty` is an enum (the id then lives in `enum_id`, not `struct_id`).
fn isEnum(t: Type) bool {
    return t.kind == .@"enum";
}

/// The ground type-id used as the canonical ordering / dedup key: `enum_id` for an
/// enum recipe, else `struct_id`.
fn typeId(t: Type) u32 {
    return if (isEnum(t)) t.enum_id else t.struct_id;
}

/// The canonical total order: by protocol id, then derive kind, then enum-vs-struct,
/// then the ground type-id. Drives BOTH the synthetic-name minting and the codegen
/// enumeration order, so the whole table is a pure function of source (never
/// hashmap/thread/discovery order).
pub fn lessThan(_: void, a: Derive, b: Derive) bool {
    if (a.protocol_id != b.protocol_id) return a.protocol_id < b.protocol_id;
    if (a.kind != b.kind) return @intFromEnum(a.kind) < @intFromEnum(b.kind);
    const ae = isEnum(a.conform_ty);
    const be = isEnum(b.conform_ty);
    if (ae != be) return @intFromBool(ae) < @intFromBool(be);
    return typeId(a.conform_ty) < typeId(b.conform_ty);
}

/// Append the canonical dedup key for `(protocol_id, kind, conform_ty)` to `buf`:
/// injective — two distinct recipes never share a key (the kind + enum flag + id are
/// fixed-width). Used by the synthesis worklist dedup.
pub fn writeKey(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), protocol_id: u32, kind: Kind, conform_ty: Type) !void {
    var w: [4]u8 = undefined;
    std.mem.writeInt(u32, &w, protocol_id, .little);
    try buf.appendSlice(gpa, &w);
    try buf.append(gpa, @intFromEnum(kind));
    try buf.append(gpa, @intFromBool(isEnum(conform_ty)));
    std.mem.writeInt(u32, &w, typeId(conform_ty), .little);
    try buf.appendSlice(gpa, &w);
}

/// The mangled synthetic symbol: `<Protocol>$<method>$<s|e><id>` (e.g. `Eq$eq$s0`,
/// `Eq$eq$e2`). `$` is not a valid source identifier char, so this can never collide
/// with a user fn; the `<Protocol>$<method>$` prefix + the `s`/`e`-tagged id keep it
/// disjoint from a Mono instance's `<template>$<args>` mangling too. The id is only a
/// SEPARATION layer — a layout edit that keeps the id still flips the unit's
/// fingerprint via the derive-fingerprint's layout fold, so the name never needs the
/// layout.
pub fn mangle(gpa: std.mem.Allocator, protocol_name: []const u8, kind: Kind, conform_ty: Type) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);
    try buf.appendSlice(gpa, protocol_name);
    try buf.append(gpa, '$');
    try buf.appendSlice(gpa, methodName(kind));
    try buf.append(gpa, '$');
    var nb: [16]u8 = undefined;
    const tag: u8 = if (isEnum(conform_ty)) 'e' else 's';
    try buf.append(gpa, tag);
    try buf.appendSlice(gpa, std.fmt.bufPrint(&nb, "{d}", .{typeId(conform_ty)}) catch unreachable);
    return buf.toOwnedSlice(gpa);
}

const testing = std.testing;

test "mangle is distinct per (protocol, kind, type)" {
    const gpa = testing.allocator;
    const a = try mangle(gpa, "Eq", .eq, Type.structT(0));
    defer gpa.free(a);
    try testing.expectEqualStrings("Eq$eq$s0", a);
    const b = try mangle(gpa, "Eq", .eq, Type.structT(1));
    defer gpa.free(b);
    try testing.expectEqualStrings("Eq$eq$s1", b);
    const c = try mangle(gpa, "Eq", .eq, Type.enumT(0));
    defer gpa.free(c);
    try testing.expectEqualStrings("Eq$eq$e0", c);
    // a struct#0 and an enum#0 mangle distinctly (the s/e tag separates the two id
    // spaces), and neither collides with a Mono `<template>$<args>` name.
    try testing.expect(!std.mem.eql(u8, a, c));
    try testing.expect(!std.mem.eql(u8, a, b));
}

test "lessThan is a total canonical order (protocol, kind, enum-flag, id)" {
    const s0 = Derive{ .protocol_id = 0, .protocol_name = "Eq", .kind = .eq, .conform_ty = Type.structT(0) };
    const s1 = Derive{ .protocol_id = 0, .protocol_name = "Eq", .kind = .eq, .conform_ty = Type.structT(1) };
    const e0 = Derive{ .protocol_id = 0, .protocol_name = "Eq", .kind = .eq, .conform_ty = Type.enumT(0) };
    // structs sort by id; a struct sorts before an enum (enum-flag false < true).
    try testing.expect(lessThan({}, s0, s1));
    try testing.expect(!lessThan({}, s1, s0));
    try testing.expect(lessThan({}, s0, e0));
    try testing.expect(lessThan({}, s1, e0));
    // antisymmetry / irreflexivity.
    try testing.expect(!lessThan({}, s0, s0));
    try testing.expect(!lessThan({}, e0, e0));
}

test "writeKey is injective across kind/enum-flag/id" {
    const gpa = testing.allocator;
    var a: std.ArrayList(u8) = .empty;
    defer a.deinit(gpa);
    var b: std.ArrayList(u8) = .empty;
    defer b.deinit(gpa);
    try writeKey(gpa, &a, 0, .eq, Type.structT(0));
    try writeKey(gpa, &b, 0, .eq, Type.enumT(0));
    // struct#0 vs enum#0 differ ONLY in the enum flag; the key must separate them.
    try testing.expect(!std.mem.eql(u8, a.items, b.items));
    var c: std.ArrayList(u8) = .empty;
    defer c.deinit(gpa);
    try writeKey(gpa, &c, 0, .eq, Type.structT(0));
    try testing.expectEqualSlices(u8, a.items, c.items); // same recipe => same key
}
