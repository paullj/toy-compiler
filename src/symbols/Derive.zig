//! Auto-derive recipe table — a peer data module.
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
const Type = @import("../layout/Type.zig").Type;

/// Which derive this recipe carries. Append-only (mirrors `Token.Tag`/`Node.Tag`
/// discipline): successive kinds are `.ord`, `.hash`, `.display`, then the two FALLIBLE
/// char conversions `.conv_int_char`/`.conv_char_byte` (source-less `TryInto` witnesses,
/// not real conformance methods — see `derive_synth`). The ordinal folds into the mangled
/// name + the sort key, so it must stay stable.
pub const Kind = enum(u8) { eq, ord, hash, display, conv_int_char, conv_char_byte, conv_float_int };

/// The method a `Kind` synthesizes (a pure function of the kind). Used for the
/// mangled name segment; `Eq` derives an `eq` method, `Ord` a `cmp` method, `Hash` a
/// `hash` method, `Display` a `display` method. The two conv kinds mint the witness's
/// `TryInto$<method>$s<charId>` name — they have no dispatch method-table row.
pub fn methodName(k: Kind) []const u8 {
    return switch (k) {
        .eq => "eq",
        .ord => "cmp",
        .hash => "hash",
        .display => "display",
        .conv_int_char => "int_to_char",
        .conv_char_byte => "char_to_byte",
        .conv_float_int => "float_to_int",
    };
}

/// How one field/payload of the conforming type contributes to the derived method,
/// RESOLVED ONCE in the synthesis barrier (on the live method table) so the emitter
/// never re-resolves and the fingerprint can fold the resolved identity. The `name`
/// slices are BORROWED (a sibling derive's minted name, a Mono instance's mangled
/// name, or a resolve-result fn name) — all outlive codegen, none owned here.
pub const FieldWitness = union(enum) {
    /// A scalar (int/bool/str) field the emitter compares inline by layout kind —
    /// no witness symbol. (Unit fields are rejected by T0007, so never occur.)
    inline_kind,
    /// An aggregate (struct / empty-payload enum) field with an `eq` witness: call it
    /// `witness(field_self, field_other) -> bool`.
    eq_call: []const u8,
    /// An Ord-only aggregate field: no `eq` witness, but a `cmp` witness exists — call
    /// `cmp(field_self, field_other)`, read the returned `Ordering` tag, compare `== eq`.
    cmp_eq: []const u8,
    /// An aggregate (struct / enum) field of an `Ord` derive with a `cmp` witness:
    /// call `cmp(field_self, field_other)`, read the returned `Ordering` tag, and use it as
    /// the 3-way field discriminant in the lexicographic chain. Append-only (ordinal 3).
    cmp_call: []const u8,
    /// An aggregate (struct / enum) field of a `Hash` derive with a `hash` witness:
    /// call `hash(field_self) -> int` and fold the returned int into the accumulator.
    /// Append-only (ordinal 4).
    hash_call: []const u8,
    /// An aggregate (struct / enum) field of a `Display` derive with a `display` witness:
    /// call `display(field_self) -> ()`, which writes the field's rendering directly
    /// to the output fd. Append-only (ordinal 5).
    display_call: []const u8,
};

/// One authorized structural-derive recipe. `field_witnesses`, `params`, and `name`
/// are OWNED (freed by the owning `GraphResult`); `protocol_name` borrows the
/// prelude/source `ProtocolSym.name` (outlives codegen); the `FieldWitness` name slices
/// are borrowed (see `FieldWitness`).
pub const Derive = struct {
    /// The protocol this recipe witnesses (`Eq`).
    protocol_id: u32,
    protocol_name: []const u8,
    kind: Kind,
    /// The ground concrete type this derives the protocol for (a `structT`/`enumT`).
    conform_ty: Type,
    /// The synthetic method's return type: `bool` for an `Eq` derive, the prelude
    /// `Ordering` enum for an `Ord` derive, `int` for a `Hash` derive. The
    /// emitter types its ret slot / exit
    /// param from this, and a witness-ret lookup reads it for a DERIVED `cmp` field (whose
    /// `fn_id == 0` would otherwise mis-read `sigs[0].ret`). A pure function of `kind`, so
    /// it is not folded into the derive fingerprint (kind already discriminates the key).
    ret: Type = .{ .kind = .invalid },
    /// Per struct field (in layout/field order) / empty for an empty-payload enum:
    /// the resolved field-eq recipe. OWNED outer slice.
    field_witnesses: []const FieldWitness = &.{},
    /// The synthetic method's params: `[conform_ty, conform_ty]` for a homogeneous `Eq`/
    /// `Ord` derive, `[conform_ty]` (self only) for a `Hash` derive. An OWNED heap
    /// slice so `AstWalk.CallVisitor.foldWitness` may borrow it into a `Fingerprint.Sig`
    /// (which stores the slice by reference; a by-value array local would dangle).
    params: []const Type = &.{},
    /// Owning module id (a sentinel; a source-less unit belongs to no real decl, but
    /// codegen threads a module for the per-module tree/resolutions it never reads).
    mod: u32 = 0,
    /// The mangled synthetic symbol, minted in canonical order after the sort. `null`
    /// until then, so a read (`.name.?`) traps loudly on out-of-order use.
    name: ?[]const u8 = null,
};

/// True when `conform_ty` is an enum (the id then lives in `enum_id`, not `struct_id`).
/// Routes through `Type.idField` so the enum-vs-struct id-space split is single-sourced.
fn isEnum(t: Type) bool {
    return Type.idField(t.kind) == .enum_id;
}

/// The ground type-id used as the canonical ordering / dedup key: the nominal id, read
/// through the field `Type.idField` selects (`enum_id` for an enum recipe, else `struct_id`).
fn typeId(t: Type) u32 {
    return t.nominalId();
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
    // Derive recipes only ever target struct/enum nominal types; the key deliberately
    // omits the int descriptor, which is only sound because no `.int` conformer reaches
    // here. Pin that so the omission is principled, not an accidental drop.
    std.debug.assert(!Type.carriesIntDesc(conform_ty.kind));
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
    // See `writeKey`: the mangled name drops the int descriptor for the same reason
    // (no `.int` conformer ever reaches derive synthesis).
    std.debug.assert(!Type.carriesIntDesc(conform_ty.kind));
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

test "mangle distinguishes Ord `cmp` from Eq `eq` per (kind, type)" {
    const gpa = testing.allocator;
    const oe = try mangle(gpa, "Ord", .ord, Type.enumT(0));
    defer gpa.free(oe);
    try testing.expectEqualStrings("Ord$cmp$e0", oe);
    const os = try mangle(gpa, "Ord", .ord, Type.structT(0));
    defer gpa.free(os);
    try testing.expectEqualStrings("Ord$cmp$s0", os);
    // The Ord `cmp` name is disjoint from the Eq `eq` name for the same struct id.
    const es = try mangle(gpa, "Eq", .eq, Type.structT(0));
    defer gpa.free(es);
    try testing.expect(!std.mem.eql(u8, os, es));
}

test "mangle produces a distinct `Hash$hash$` name per (struct/enum, id)" {
    const gpa = testing.allocator;
    const hs = try mangle(gpa, "Hash", .hash, Type.structT(0));
    defer gpa.free(hs);
    try testing.expectEqualStrings("Hash$hash$s0", hs);
    const he = try mangle(gpa, "Hash", .hash, Type.enumT(0));
    defer gpa.free(he);
    try testing.expectEqualStrings("Hash$hash$e0", he);
    // The Hash `hash` name is disjoint from the Eq `eq` and Ord `cmp` names for the same id.
    const es = try mangle(gpa, "Eq", .eq, Type.structT(0));
    defer gpa.free(es);
    const os = try mangle(gpa, "Ord", .ord, Type.structT(0));
    defer gpa.free(os);
    try testing.expect(!std.mem.eql(u8, hs, es));
    try testing.expect(!std.mem.eql(u8, hs, os));
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

test "lessThan orders eq-kind before ord-kind for the same protocol/type" {
    // Same protocol id + same type; the kind ordinal (eq=0 < ord=1) is the tiebreaker.
    const eq0 = Derive{ .protocol_id = 0, .protocol_name = "Eq", .kind = .eq, .conform_ty = Type.structT(0) };
    const ord0 = Derive{ .protocol_id = 0, .protocol_name = "Ord", .kind = .ord, .conform_ty = Type.structT(0) };
    try testing.expect(lessThan({}, eq0, ord0));
    try testing.expect(!lessThan({}, ord0, eq0));
}

test "lessThan orders eq < ord < hash for the same protocol/type" {
    // The kind ordinal (eq=0 < ord=1 < hash=2) is the tiebreaker within one (protocol, type);
    // a Hash recipe sorts AFTER Eq and Ord so its synthetic id/name is minted last.
    const eq0 = Derive{ .protocol_id = 0, .protocol_name = "Eq", .kind = .eq, .conform_ty = Type.structT(0) };
    const ord0 = Derive{ .protocol_id = 0, .protocol_name = "Ord", .kind = .ord, .conform_ty = Type.structT(0) };
    const hash0 = Derive{ .protocol_id = 0, .protocol_name = "Hash", .kind = .hash, .conform_ty = Type.structT(0) };
    try testing.expect(lessThan({}, eq0, hash0));
    try testing.expect(lessThan({}, ord0, hash0));
    try testing.expect(!lessThan({}, hash0, eq0));
    try testing.expect(!lessThan({}, hash0, ord0));
    try testing.expect(!lessThan({}, hash0, hash0));
}

test "mangle produces a distinct `Display$display$` name; kind orders last" {
    const gpa = testing.allocator;
    const ds = try mangle(gpa, "Display", .display, Type.structT(0));
    defer gpa.free(ds);
    try testing.expectEqualStrings("Display$display$s0", ds);
    const de = try mangle(gpa, "Display", .display, Type.enumT(2));
    defer gpa.free(de);
    try testing.expectEqualStrings("Display$display$e2", de);
    // The Display `display` name is disjoint from the Eq/Ord/Hash names for the same id.
    const es = try mangle(gpa, "Eq", .eq, Type.structT(0));
    defer gpa.free(es);
    const os = try mangle(gpa, "Ord", .ord, Type.structT(0));
    defer gpa.free(os);
    const hs = try mangle(gpa, "Hash", .hash, Type.structT(0));
    defer gpa.free(hs);
    try testing.expect(!std.mem.eql(u8, ds, es));
    try testing.expect(!std.mem.eql(u8, ds, os));
    try testing.expect(!std.mem.eql(u8, ds, hs));
    // The kind ordinal (eq=0 < ord=1 < hash=2 < display=3) makes a Display recipe sort LAST
    // within one (protocol, type), so its synthetic id/name is minted after all others.
    const eq0 = Derive{ .protocol_id = 0, .protocol_name = "Eq", .kind = .eq, .conform_ty = Type.structT(0) };
    const hash0 = Derive{ .protocol_id = 0, .protocol_name = "Hash", .kind = .hash, .conform_ty = Type.structT(0) };
    const disp0 = Derive{ .protocol_id = 0, .protocol_name = "Display", .kind = .display, .conform_ty = Type.structT(0) };
    try testing.expect(lessThan({}, eq0, disp0));
    try testing.expect(lessThan({}, hash0, disp0));
    try testing.expect(!lessThan({}, disp0, hash0));
    try testing.expect(!lessThan({}, disp0, disp0));
    // writeKey separates the display kind from eq/ord/hash for the same (protocol, type).
    var a: std.ArrayList(u8) = .empty;
    defer a.deinit(gpa);
    var d: std.ArrayList(u8) = .empty;
    defer d.deinit(gpa);
    try writeKey(gpa, &a, 0, .hash, Type.structT(0));
    try writeKey(gpa, &d, 0, .display, Type.structT(0));
    try testing.expect(!std.mem.eql(u8, a.items, d.items));
}

test "mangle produces distinct `TryInto$` conv names, disjoint from Display/Eq" {
    const gpa = testing.allocator;
    const ic = try mangle(gpa, "TryInto", .conv_int_char, Type.structT(3));
    defer gpa.free(ic);
    try testing.expectEqualStrings("TryInto$int_to_char$s3", ic);
    const cb = try mangle(gpa, "TryInto", .conv_char_byte, Type.structT(3));
    defer gpa.free(cb);
    try testing.expectEqualStrings("TryInto$char_to_byte$s3", cb);
    // The two conv names are disjoint from each other and from a Display/Eq name for the
    // same struct id (the `<method>` segment separates them).
    try testing.expect(!std.mem.eql(u8, ic, cb));
    const ds = try mangle(gpa, "Display", .display, Type.structT(3));
    defer gpa.free(ds);
    const es = try mangle(gpa, "Eq", .eq, Type.structT(3));
    defer gpa.free(es);
    try testing.expect(!std.mem.eql(u8, ic, ds));
    try testing.expect(!std.mem.eql(u8, ic, es));
    try testing.expect(!std.mem.eql(u8, cb, ds));
    // The float->int witness anchors on `Type.float` (no char struct); `carriesIntDesc(.float)`
    // is false, so mangle/writeKey accept it and mint the `s0` name distinct from the char convs.
    const fi = try mangle(gpa, "TryInto", .conv_float_int, Type.float);
    defer gpa.free(fi);
    try testing.expectEqualStrings("TryInto$float_to_int$s0", fi);
    try testing.expect(!std.mem.eql(u8, fi, ic));
    try testing.expect(!std.mem.eql(u8, fi, cb));
    // writeKey separates the conv kinds for the same (protocol, type).
    var a: std.ArrayList(u8) = .empty;
    defer a.deinit(gpa);
    var b: std.ArrayList(u8) = .empty;
    defer b.deinit(gpa);
    try writeKey(gpa, &a, 10, .conv_int_char, Type.structT(3));
    try writeKey(gpa, &b, 10, .conv_char_byte, Type.structT(3));
    try testing.expect(!std.mem.eql(u8, a.items, b.items));
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
    // the same (protocol, type) under distinct kinds (eq vs ord) key distinctly.
    var d: std.ArrayList(u8) = .empty;
    defer d.deinit(gpa);
    try writeKey(gpa, &d, 0, .ord, Type.structT(0));
    try testing.expect(!std.mem.eql(u8, a.items, d.items));
    // the hash kind keys distinctly from both eq and ord for the same (protocol, type).
    var e: std.ArrayList(u8) = .empty;
    defer e.deinit(gpa);
    try writeKey(gpa, &e, 0, .hash, Type.structT(0));
    try testing.expect(!std.mem.eql(u8, a.items, e.items));
    try testing.expect(!std.mem.eql(u8, d.items, e.items));
}
