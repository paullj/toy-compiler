//! The function-value units the typecheck tail mints for codegen, and their names. A
//! peer data module like `Callee`: types builds the tables serially in canonical order,
//! the driver enumerates them as codegen units, lower reads them by key.

const std = @import("std");
const Sym = @import("Sym.zig");
const LayoutEngine = @import("../layout/Engine.zig");

/// A body that owns closure units: a base fn (its global id) or a generic instance (its
/// index in the instance table).
pub const Owner = union(enum) { base: u32, instance: u32 };

pub const AdapterKind = enum(u8) { fn_value };

/// A source-less unit that turns a named callee into the `(env, params..)` shape every
/// function value is called with. `.fn_value` is the thunk a named fn taken as a value
/// points at: it ignores the env and calls the fn directly.
pub const Adapter = struct {
    kind: AdapterKind,
    /// `.fn_value`: the referenced fn's global id.
    key: u32,
    /// Owned.
    name: []const u8,
    /// The callee's symbol; the name is borrowed from the typecheck result's tables.
    target: Sym.SymName,
    /// The value signature, without the env: borrowed from the callee's typecheck sig.
    sig: LayoutEngine.FnSig,
};

/// The thunk / adapter symbol for `target_sym`. `$` is not a source identifier char, and
/// no generic-instance mangle spells an arg `fnval` (see the test), so the name is
/// disjoint from every user fn, instance, derive and erased unit.
pub fn adapterName(gpa: std.mem.Allocator, target_sym: []const u8, kind: AdapterKind) ![]u8 {
    const suffix = switch (kind) {
        .fn_value => "$fnval",
    };
    return std.mem.concat(gpa, u8, &.{ target_sym, suffix });
}

fn adapterLess(kind: AdapterKind, key: u32, a: Adapter) std.math.Order {
    if (kind != a.kind) return std.math.order(@intFromEnum(kind), @intFromEnum(a.kind));
    return std.math.order(key, a.key);
}

/// Binary search over `ads`, which is sorted by `(kind, key)`.
pub fn findAdapter(ads: []const Adapter, kind: AdapterKind, key: u32) ?u32 {
    const Key = struct { kind: AdapterKind, key: u32 };
    const i = std.sort.binarySearch(Adapter, ads, Key{ .kind = kind, .key = key }, struct {
        fn order(k: Key, a: Adapter) std.math.Order {
            return adapterLess(k.kind, k.key, a);
        }
    }.order) orelse return null;
    return @intCast(i);
}

const testing = std.testing;

test "adapterName appends the kind suffix" {
    const nm = try adapterName(testing.allocator, "geometry/rect.area", .fn_value);
    defer testing.allocator.free(nm);
    try testing.expectEqualStrings("geometry/rect.area$fnval", nm);
}

test "an adapter name never equals a generic-instance mangle" {
    const Mono = @import("Mono.zig");
    const Type = @import("../layout/Type.zig").Type;
    const spellings = [_]Type{ Type.int, Type.bool, Type.str, Type.unit, Type.float, Type.structT(3), Type.enumT(4), Type.rawptr };
    const ad = try adapterName(testing.allocator, "f", .fn_value);
    defer testing.allocator.free(ad);
    for (spellings) |t| {
        const m = try Mono.mangle(testing.allocator, "f", &.{t});
        defer testing.allocator.free(m);
        try testing.expect(!std.mem.eql(u8, m, ad));
    }
}

test "findAdapter binary-searches by (kind, key)" {
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = @import("../layout/Type.zig").Type.unit };
    const t: Sym.SymName = .{ .kind = .user_fn, .name = "f" };
    const ads = [_]Adapter{
        .{ .kind = .fn_value, .key = 2, .name = "a", .target = t, .sig = sig },
        .{ .kind = .fn_value, .key = 5, .name = "b", .target = t, .sig = sig },
        .{ .kind = .fn_value, .key = 9, .name = "c", .target = t, .sig = sig },
    };
    try testing.expectEqual(@as(?u32, 0), findAdapter(&ads, .fn_value, 2));
    try testing.expectEqual(@as(?u32, 1), findAdapter(&ads, .fn_value, 5));
    try testing.expectEqual(@as(?u32, 2), findAdapter(&ads, .fn_value, 9));
    try testing.expectEqual(@as(?u32, null), findAdapter(&ads, .fn_value, 4));
    try testing.expectEqual(@as(?u32, null), findAdapter(&.{}, .fn_value, 4));
}
