//! The symbol names the compiler mints, in one place. A minted name is its owner's symbol
//! (or a fixed prefix) followed by `$`-separated pieces. `$` is not a source identifier
//! char, so no minted name equals a user fn's, and each family spells its pieces so that
//! no two families meet (see the uniqueness test). The spellings reach every binary and
//! cache key, so changing one is a behavior change.

const std = @import("std");
const Type = @import("../layout/Type.zig").Type;
const Closure = @import("Closure.zig");
const Spec = @import("Spec.zig");
const Derive = @import("Derive.zig");

/// A generic instance: `<template>$<arg0>$<arg1>...`. Struct/enum args mangle to
/// `s<id>`/`e<id>` (the global id is a deterministic function of source); the id is only a
/// SEPARATION layer — a struct-layout edit that keeps the id still flips the instance's
/// fingerprint via the type-arg layout fold, so the mangled name never needs to encode the
/// layout.
pub fn instance(gpa: std.mem.Allocator, template_name: []const u8, args: []const Type) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);
    try buf.appendSlice(gpa, template_name);
    var nb: [16]u8 = undefined;
    for (args) |a| {
        try buf.append(gpa, '$');
        switch (a.kind) {
            .int => try buf.appendSlice(gpa, a.intName()),
            .bool => try buf.appendSlice(gpa, "bool"),
            .str => try buf.appendSlice(gpa, "str"),
            .unit => try buf.appendSlice(gpa, "unit"),
            .float => try buf.appendSlice(gpa, "float"),
            .@"struct" => try buf.appendSlice(gpa, std.fmt.bufPrint(&nb, "s{d}", .{a.nominalId()}) catch unreachable),
            .@"enum" => try buf.appendSlice(gpa, std.fmt.bufPrint(&nb, "e{d}", .{a.nominalId()}) catch unreachable),
            else => try buf.appendSlice(gpa, "x"),
        }
    }
    return buf.toOwnedSlice(gpa);
}

/// The lambda unit of rank `rank` in its owner: `<owner>$l<rank>`.
pub fn lambda(gpa: std.mem.Allocator, owner_sym: []const u8, rank: u32) ![]u8 {
    return std.fmt.allocPrint(gpa, "{s}$l{d}", .{ owner_sym, rank });
}

/// The fn-value thunk or bound adapter for `target_sym`: `<target>$fnval`, `$bound`,
/// `$boundbox` or `$boundref` as `kind` says.
pub fn adapter(gpa: std.mem.Allocator, target_sym: []const u8, kind: Closure.AdapterKind) ![]u8 {
    const suffix = switch (kind) {
        .fn_value, .assoc_value => "$fnval",
        .bound_copy => "$bound",
        .bound_box => "$boundbox",
        .bound_ref => "$boundref",
    };
    return std.mem.concat(gpa, u8, &.{ target_sym, suffix });
}

/// A spec unit: `<inst>$sp<p>$<sym>` per target, ascending p. A pure function of the
/// instance and target symbols; the pieces split back apart, so distinct target lists give
/// distinct names.
pub fn spec(gpa: std.mem.Allocator, inst_name: []const u8, targets: []const Spec.Target) error{OutOfMemory}![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, inst_name);
    for (targets) |t| try out.print(gpa, "$sp{d}${s}", .{ t.param, t.sym.name });
    return out.toOwnedSlice(gpa);
}

/// A derive witness: `<Protocol>$<method>$<s|e><id>` (e.g. `Eq$eq$s0`, `Eq$eq$e2`). The id
/// is only a SEPARATION layer — a layout edit that keeps the id still flips the unit's
/// fingerprint via the derive-fingerprint's layout fold, so the name never needs the layout.
pub fn derive(gpa: std.mem.Allocator, protocol_name: []const u8, kind: Derive.Kind, conform_ty: Type) ![]u8 {
    // See `Derive.writeKey`: the name drops the int descriptor for the same reason (no
    // `.int` conformer ever reaches derive synthesis).
    std.debug.assert(!Type.carriesIntDesc(conform_ty.kind));
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);
    try buf.appendSlice(gpa, protocol_name);
    try buf.append(gpa, '$');
    try buf.appendSlice(gpa, Derive.methodName(kind));
    try buf.append(gpa, '$');
    var nb: [16]u8 = undefined;
    const tag: u8 = if (Derive.isEnum(conform_ty)) 'e' else 's';
    try buf.append(gpa, tag);
    try buf.appendSlice(gpa, std.fmt.bufPrint(&nb, "{d}", .{conform_ty.nominalId()}) catch unreachable);
    return buf.toOwnedSlice(gpa);
}

/// An erased key-witness unit: `descriptor$hash$<tag>` / `descriptor$eq$<tag>` /
/// `descriptor$trace$<tag>`, where `<tag>` is `s<id>`/`e<id>` for a struct/enum, or the
/// kind spelling for a scalar (a `Map[int, V]` key is an `int`, which `derive` cannot name).
pub fn erased(gpa: std.mem.Allocator, kind: Derive.ErasedKind, ty: Type) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);
    try buf.appendSlice(gpa, "descriptor$");
    try buf.appendSlice(gpa, switch (kind) {
        .hash => "hash",
        .eq => "eq",
        .trace => "trace",
    });
    try buf.append(gpa, '$');
    switch (ty.kind) {
        .@"struct", .@"enum" => {
            try buf.append(gpa, if (Derive.isEnum(ty)) 'e' else 's');
            var nb: [16]u8 = undefined;
            try buf.appendSlice(gpa, std.fmt.bufPrint(&nb, "{d}", .{ty.nominalId()}) catch unreachable);
        },
        // Per-width spelling (`int`/`uint`/`int8`…/`uint64`), so each int width mints a
        // distinct erased symbol — mirroring `appendKeyBytes`/`descHash` folding the
        // sign/width byte into the type key. A bare "int" would alias every width onto ONE
        // hash/eq implementation (a silent wrong-width read).
        .int => try buf.appendSlice(gpa, ty.intName()),
        .bool => try buf.appendSlice(gpa, "bool"),
        .str => try buf.appendSlice(gpa, "str"),
        .unit => try buf.appendSlice(gpa, "unit"),
        .float => try buf.appendSlice(gpa, "float"),
        .rawptr => try buf.appendSlice(gpa, "rawptr"),
        else => try buf.appendSlice(gpa, "other"),
    }
    return buf.toOwnedSlice(gpa);
}

const testing = std.testing;
const Sym = @import("Sym.zig");

/// A gpa-owned name list, freed together.
const Names = struct {
    items: std.ArrayList([]u8) = .empty,

    fn add(n: *Names, name: anyerror![]u8) !void {
        const owned = try name;
        errdefer testing.allocator.free(owned);
        try n.items.append(testing.allocator, owned);
    }

    fn deinit(n: *Names) void {
        for (n.items.items) |s| testing.allocator.free(s);
        n.items.deinit(testing.allocator);
    }
};

test "each family spells its names" {
    const gpa = testing.allocator;
    const add: Sym.SymName = .{ .kind = .user_fn, .name = "main.add" };
    const lam: Sym.SymName = .{ .kind = .user_fn, .name = "main$l0" };
    const Case = struct { got: anyerror![]u8, want: []const u8 };
    const cases = [_]Case{
        .{ .got = instance(gpa, "id", &.{Type.int}), .want = "id$int" },
        .{ .got = instance(gpa, "id", &.{Type.structT(0)}), .want = "id$s0" },
        .{ .got = instance(gpa, "pair", &.{ Type.int, Type.bool }), .want = "pair$int$bool" },
        .{ .got = instance(gpa, "id", &.{Type.int8}), .want = "id$int8" },
        .{ .got = instance(gpa, "id", &.{Type.uint32}), .want = "id$uint32" },
        .{ .got = lambda(gpa, "f", 0), .want = "f$l0" },
        .{ .got = lambda(gpa, "f", 12), .want = "f$l12" },
        .{ .got = adapter(gpa, "geometry/rect.area", .fn_value), .want = "geometry/rect.area$fnval" },
        .{ .got = adapter(gpa, "f", .bound_copy), .want = "f$bound" },
        .{ .got = adapter(gpa, "f", .bound_box), .want = "f$boundbox" },
        .{ .got = adapter(gpa, "f", .bound_ref), .want = "f$boundref" },
        .{ .got = spec(gpa, "main.apply$int", &.{.{ .param = 0, .kind = .named, .sym = add }}), .want = "main.apply$int$sp0$main.add" },
        .{ .got = spec(gpa, "main.apply$int", &.{ .{ .param = 0, .kind = .named, .sym = add }, .{ .param = 1, .kind = .lambda, .sym = lam } }), .want = "main.apply$int$sp0$main.add$sp1$main$l0" },
        .{ .got = derive(gpa, "Eq", .eq, Type.structT(0)), .want = "Eq$eq$s0" },
        .{ .got = derive(gpa, "Eq", .eq, Type.enumT(0)), .want = "Eq$eq$e0" },
        .{ .got = derive(gpa, "Ord", .ord, Type.structT(0)), .want = "Ord$cmp$s0" },
        .{ .got = derive(gpa, "Hash", .hash, Type.enumT(0)), .want = "Hash$hash$e0" },
        .{ .got = derive(gpa, "Display", .display, Type.enumT(2)), .want = "Display$display$e2" },
        .{ .got = derive(gpa, "TryInto", .conv_int_char, Type.structT(3)), .want = "TryInto$int_to_char$s3" },
        .{ .got = derive(gpa, "TryInto", .conv_char_byte, Type.structT(3)), .want = "TryInto$char_to_byte$s3" },
        // The float->int witness anchors on `Type.float` (no char struct): `s0`.
        .{ .got = derive(gpa, "TryInto", .conv_float_int, Type.float), .want = "TryInto$float_to_int$s0" },
        .{ .got = derive(gpa, "Trace", .trace, Type.enumT(1)), .want = "Trace$trace$e1" },
        .{ .got = erased(gpa, .hash, Type.structT(0)), .want = "descriptor$hash$s0" },
        .{ .got = erased(gpa, .eq, Type.structT(0)), .want = "descriptor$eq$s0" },
        .{ .got = erased(gpa, .hash, Type.int), .want = "descriptor$hash$int" },
        .{ .got = erased(gpa, .hash, Type.int8), .want = "descriptor$hash$int8" },
        .{ .got = erased(gpa, .hash, Type.uint8), .want = "descriptor$hash$uint8" },
        .{ .got = erased(gpa, .hash, Type.str), .want = "descriptor$hash$str" },
        .{ .got = erased(gpa, .trace, Type.str), .want = "descriptor$trace$str" },
    };
    var got: Names = .{};
    defer got.deinit();
    for (cases) |c| try got.add(c.got);
    for (cases, got.items.items) |c, g| try testing.expectEqualStrings(c.want, g);
}

test "minted names never collide, within a family or across families" {
    const gpa = testing.allocator;
    var names: Names = .{};
    defer names.deinit();

    const args = [_]Type{ Type.int, Type.int8, Type.uint32, Type.bool, Type.str, Type.unit, Type.float, Type.structT(0), Type.structT(3), Type.enumT(0), Type.enumT(4), Type.rawptr };
    for (args) |a| try names.add(instance(gpa, "f", &.{a}));
    for (args[0..3]) |a| for (args[3..6]) |b| try names.add(instance(gpa, "f", &.{ a, b }));
    for ([_]u32{ 0, 1, 12 }) |r| {
        try names.add(lambda(gpa, "f", r));
        try names.add(lambda(gpa, "f$int", r));
    }
    for ([_]Closure.AdapterKind{ .fn_value, .bound_copy, .bound_box, .bound_ref }) |k| {
        try names.add(adapter(gpa, "f", k));
        try names.add(adapter(gpa, "f$int", k));
    }
    const g: Sym.SymName = .{ .kind = .user_fn, .name = "f" };
    const l: Sym.SymName = .{ .kind = .user_fn, .name = "f$l0" };
    const target_lists = [_][]const Spec.Target{
        &.{.{ .param = 0, .kind = .named, .sym = g }},
        &.{.{ .param = 1, .kind = .named, .sym = g }},
        &.{.{ .param = 0, .kind = .lambda, .sym = l }},
        &.{ .{ .param = 0, .kind = .named, .sym = g }, .{ .param = 1, .kind = .lambda, .sym = l } },
        &.{ .{ .param = 0, .kind = .lambda, .sym = l }, .{ .param = 1, .kind = .named, .sym = g } },
    };
    for (target_lists) |ts| {
        try names.add(spec(gpa, "f$int", ts));
        try names.add(spec(gpa, "f$bool", ts));
    }
    // A protocol may share a template's name; the method piece keeps them apart.
    const derives = [_]struct { p: []const u8, k: Derive.Kind }{
        .{ .p = "Eq", .k = .eq },                 .{ .p = "Ord", .k = .ord },
        .{ .p = "Hash", .k = .hash },             .{ .p = "Display", .k = .display },
        .{ .p = "TryInto", .k = .conv_int_char }, .{ .p = "TryInto", .k = .conv_char_byte },
        .{ .p = "Trace", .k = .trace },           .{ .p = "f", .k = .eq },
    };
    for (derives) |d| for ([_]Type{ Type.structT(0), Type.structT(3), Type.enumT(0) }) |t| try names.add(derive(gpa, d.p, d.k, t));
    try names.add(derive(gpa, "TryInto", .conv_float_int, Type.float));
    for ([_]Derive.ErasedKind{ .hash, .eq, .trace }) |k| for (args) |t| try names.add(erased(gpa, k, t));

    const all = names.items.items;
    for (all, 0..) |a, i| for (all[i + 1 ..]) |b| {
        if (std.mem.eql(u8, a, b)) {
            std.debug.print("collision: {s}\n", .{a});
            return error.TestUnexpectedResult;
        }
    };
}
