//! Monomorphization instance table — a peer data module (M2).
//!
//! An `Instance` is one reachable `(generic template, concrete-arg-tuple)` pair
//! reified to an ordinary concrete per-fn codegen unit. The type checker's serial
//! monomorphization tail produces the table; codegen enumerates it as extra
//! lowerable units; `lower`/`Fingerprint`(`CallVisitor`) resolve a generic call
//! site to its instance through it. This module is the SINGLE home for the
//! canonical logic (`find`, the sort key, the mangled `SymName`) so the four
//! consumers cannot drift.
//!
//! Determinism is SACRED: `lessThan` is a total order over `(template gid, arg
//! bytes)` and `mangle` is a pure function of the template name + the arg tuple,
//! so the instance list, its ids, and its mangled symbols are a pure function of
//! source (never hashmap/thread order).

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Type = @import("../layout/Engine.zig").Type;

/// One resolved bound `[T has P]` on a monomorphized instance (M13): the witnessing
/// `impl <conform_ty> has P` chosen at the mono worklist by conformance lookup. This
/// is the structural datum folded (ordered, never XOR) into the instance's content
/// fingerprint so toggling a sibling-module conformance invalidates EXACTLY the
/// dependent monomorphizations. NOT the impl body-fp (that would over-invalidate) —
/// only the protocol identity + the conforming type's layout + the witnessing method
/// symbols in protocol-declared order.
///
/// LIFETIME (getting this wrong is a use-after-free in codegen):
///   * `protocol_name` borrows the SOURCE-backed `ProtocolSym.name` (stable; the
///     `t.protocols` array is freed at typecheck teardown but the name BYTES are
///     source slices that outlive codegen).
///   * `witness_syms` element slices borrow `gph_fn_names` (resolve-result-backed,
///     outlives codegen); the OUTER `witness_syms` slice is OWNED by the Instance.
///   * `conform_ty` is a 12-byte POD (the ground conforming `structT`/`enumT`/scalar).
/// Only the outer `conformances` slice + each `witness_syms` outer slice are owned.
pub const ResolvedConformance = struct {
    protocol_name: []const u8,
    conform_ty: Type,
    witness_syms: []const []const u8,
};

/// One reified generic instance. All slices are OWNED (freed by the owning
/// `GraphResult`).
pub const Instance = struct {
    /// Global fn id of the generic template this instance specializes.
    template_gid: u32,
    /// The concrete type-args, in generic-param order.
    args: []const Type,
    /// A per-instance `node_types` slice sized to the template module's node count
    /// (written by the per-instance re-check). NEVER the module's shared slice, so
    /// two instances of one template do not alias each other's node slots.
    node_types: []Type,
    /// The template's params/ret with every `type_var` substituted to concrete.
    params: []const Type,
    ret: Type,
    /// The mangled instance symbol name (e.g. `id$int`). Minted in canonical order
    /// after the sort; `undefined` until then.
    name: []const u8,
    /// Owning module id (the template's module) and the template's decl node.
    mod: u32,
    decl_node: Ast.Index,
    /// The resolved bounds `[T has P]` witnessing conformances for this instance
    /// (M13), one per bounded generic param, in generic-param order. Empty (`&.{}`)
    /// for an unbounded template's instance so its fingerprint fold is byte-identical
    /// (warm cache preserved). The OUTER slice + each entry's `witness_syms` outer
    /// slice are OWNED by the owning `GraphResult`; see `ResolvedConformance`.
    conformances: []const ResolvedConformance = &.{},
};

/// The index of the instance for `(gid, args)`, or null. A deterministic linear
/// scan comparing the template gid + each arg by `Type.eql`.
pub fn find(insts: []const Instance, gid: u32, args: []const Type) ?usize {
    for (insts, 0..) |inst, i| {
        if (inst.template_gid != gid or inst.args.len != args.len) continue;
        var ok = true;
        for (inst.args, args) |a, b| {
            if (!Type.eql(a, b)) {
                ok = false;
                break;
            }
        }
        if (ok) return i;
    }
    return null;
}

/// The canonical total order: by template gid, then lexicographically by the arg
/// tuple's `(kind, struct_id, enum_id)` triples. Drives BOTH the instance
/// enumeration order and the name-minting order, so the whole table is a pure
/// function of source.
pub fn lessThan(_: void, a: Instance, b: Instance) bool {
    if (a.template_gid != b.template_gid) return a.template_gid < b.template_gid;
    const n = @min(a.args.len, b.args.len);
    for (a.args[0..n], b.args[0..n]) |x, y| {
        if (x.kind != y.kind) return @intFromEnum(x.kind) < @intFromEnum(y.kind);
        if (x.struct_id != y.struct_id) return x.struct_id < y.struct_id;
        if (x.enum_id != y.enum_id) return x.enum_id < y.enum_id;
    }
    return a.args.len < b.args.len;
}

/// Append the canonical dedup key for `(gid, args)` to `buf`: the gid then each
/// arg's `(kind, struct_id, enum_id)` in a fixed width. Injective — two distinct
/// `(gid, args)` never share a key (the gid fixes the arity, so lengths never
/// collide). Used by the mono tail's worklist dedup.
pub fn writeKey(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), gid: u32, args: []const Type) !void {
    var w: [4]u8 = undefined;
    std.mem.writeInt(u32, &w, gid, .little);
    try buf.appendSlice(gpa, &w);
    for (args) |a| {
        try buf.append(gpa, @intFromEnum(a.kind));
        std.mem.writeInt(u32, &w, a.struct_id, .little);
        try buf.appendSlice(gpa, &w);
        std.mem.writeInt(u32, &w, a.enum_id, .little);
        try buf.appendSlice(gpa, &w);
    }
}

/// The mangled instance symbol: `<template>$<arg0>$<arg1>...`. `$` is not a valid
/// source identifier char, so a mangled name can never collide with a user fn.
/// Struct/enum args mangle to `s<id>`/`e<id>` (the global id is a deterministic
/// function of source); the id is only a SEPARATION layer — a struct-layout edit
/// that keeps the id still flips the instance's fingerprint via the type-arg
/// layout fold, so the mangled name never needs to encode the layout.
pub fn mangle(gpa: std.mem.Allocator, template_name: []const u8, args: []const Type) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);
    try buf.appendSlice(gpa, template_name);
    var nb: [16]u8 = undefined;
    for (args) |a| {
        try buf.append(gpa, '$');
        switch (a.kind) {
            .int => try buf.appendSlice(gpa, "int"),
            .bool => try buf.appendSlice(gpa, "bool"),
            .str => try buf.appendSlice(gpa, "str"),
            .unit => try buf.appendSlice(gpa, "unit"),
            .@"struct" => try buf.appendSlice(gpa, std.fmt.bufPrint(&nb, "s{d}", .{a.struct_id}) catch unreachable),
            .@"enum" => try buf.appendSlice(gpa, std.fmt.bufPrint(&nb, "e{d}", .{a.enum_id}) catch unreachable),
            else => try buf.appendSlice(gpa, "x"),
        }
    }
    return buf.toOwnedSlice(gpa);
}

const testing = std.testing;

test "mangle is index-free-ish and distinct per arg tuple" {
    const gpa = testing.allocator;
    const a = try mangle(gpa, "id", &.{Type.int});
    defer gpa.free(a);
    try testing.expectEqualStrings("id$int", a);
    const b = try mangle(gpa, "id", &.{Type.structT(0)});
    defer gpa.free(b);
    try testing.expectEqualStrings("id$s0", b);
    try testing.expect(!std.mem.eql(u8, a, b));
    const c = try mangle(gpa, "pair", &.{ Type.int, Type.bool });
    defer gpa.free(c);
    try testing.expectEqualStrings("pair$int$bool", c);
}

test "find matches on gid + args by Type.eql" {
    const insts = [_]Instance{
        .{ .template_gid = 0, .args = &.{Type.int}, .node_types = &.{}, .params = &.{Type.int}, .ret = Type.int, .name = "id$int", .mod = 0, .decl_node = Ast.none },
        .{ .template_gid = 0, .args = &.{Type.structT(0)}, .node_types = &.{}, .params = &.{Type.structT(0)}, .ret = Type.structT(0), .name = "id$s0", .mod = 0, .decl_node = Ast.none },
    };
    try testing.expectEqual(@as(?usize, 0), find(&insts, 0, &.{Type.int}));
    try testing.expectEqual(@as(?usize, 1), find(&insts, 0, &.{Type.structT(0)}));
    try testing.expectEqual(@as(?usize, null), find(&insts, 0, &.{Type.bool}));
    try testing.expectEqual(@as(?usize, null), find(&insts, 1, &.{Type.int}));
}

test "an instance carries its resolved conformances (M13)" {
    const witness = [_][]const u8{"lib.P.dbl"};
    const confs = [_]ResolvedConformance{.{ .protocol_name = "Doubler", .conform_ty = Type.structT(3), .witness_syms = &witness }};
    const inst = Instance{
        .template_gid = 5,
        .args = &.{Type.structT(3)},
        .node_types = &.{},
        .params = &.{Type.structT(3)},
        .ret = Type.int,
        .name = "twice$s3",
        .mod = 0,
        .decl_node = Ast.none,
        .conformances = &confs,
    };
    try testing.expectEqual(@as(usize, 1), inst.conformances.len);
    try testing.expectEqualStrings("Doubler", inst.conformances[0].protocol_name);
    try testing.expect(Type.eql(Type.structT(3), inst.conformances[0].conform_ty));
    try testing.expectEqualStrings("lib.P.dbl", inst.conformances[0].witness_syms[0]);
    // An instance built WITHOUT conformances defaults to the empty slice (warm-cache
    // byte-identity for unbounded templates).
    const plain = Instance{ .template_gid = 0, .args = &.{Type.int}, .node_types = &.{}, .params = &.{Type.int}, .ret = Type.int, .name = "id$int", .mod = 0, .decl_node = Ast.none };
    try testing.expectEqual(@as(usize, 0), plain.conformances.len);
}

test "lessThan is a total canonical order (gid then arg triples)" {
    const a = Instance{ .template_gid = 0, .args = &.{Type.int}, .node_types = &.{}, .params = &.{}, .ret = Type.int, .name = "", .mod = 0, .decl_node = Ast.none };
    const b = Instance{ .template_gid = 0, .args = &.{Type.structT(0)}, .node_types = &.{}, .params = &.{}, .ret = Type.int, .name = "", .mod = 0, .decl_node = Ast.none };
    const c = Instance{ .template_gid = 1, .args = &.{Type.int}, .node_types = &.{}, .params = &.{}, .ret = Type.int, .name = "", .mod = 0, .decl_node = Ast.none };
    // int (kind 2) < struct (kind 6), so a < b; gid 0 < gid 1, so a < c.
    try testing.expect(lessThan({}, a, b));
    try testing.expect(!lessThan({}, b, a));
    try testing.expect(lessThan({}, a, c));
    try testing.expect(lessThan({}, b, c));
}
