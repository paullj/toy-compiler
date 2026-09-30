//! The function-value units the typecheck tail mints for codegen, and their names. A
//! peer data module like `Callee`: resolve records each fn's lambda boundaries, types
//! builds the unit tables serially in canonical order, the driver enumerates them as
//! codegen units, lower reads them by key.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Sym = @import("Sym.zig");
const LayoutEngine = @import("../layout/Engine.zig");

/// A local named inside a lambda but declared outside it: `slot` is the owner fn's
/// resolver slot, `first_use` the first identifier node (in resolve order) naming it.
pub const Capture = struct { slot: u32, first_use: Ast.Index };

/// One lambda in a fn body, as the resolver saw it.
pub const Boundary = struct {
    /// The `.lambda` node.
    node: Ast.Index,
    /// Pre-order index among the owner fn's boundaries: the lambda's unit name suffix.
    rank: u32,
    /// Every slot the lambda declares is at least this; a local below it is declared
    /// outside (resolver slots are handed out monotonically per fn).
    slot_floor: u32,
    /// Ascending slot, deduped; each keeps its first use.
    captures: []const Capture,
};

/// The lambdas of one top-level fn (or method), recorded only for a fn that has one.
pub const FnClosures = struct {
    /// The enclosing `fn_decl`.
    fn_node: Ast.Index,
    /// Sorted by node index.
    boundaries: []const Boundary,
    /// The slots written by `x = ..` or by a place store rooted at `x` (`x.f = ..`),
    /// sorted and deduped.
    assigned: []const u32,
};

pub fn deinitFnClosures(gpa: std.mem.Allocator, list: []const FnClosures) void {
    for (list) |fc| {
        for (fc.boundaries) |b| gpa.free(b.captures);
        gpa.free(fc.boundaries);
        gpa.free(fc.assigned);
    }
    gpa.free(list);
}

/// A body that owns closure units: a base fn (its global id) or a generic instance (its
/// index in the instance table).
pub const Owner = union(enum) { base: u32, instance: u32 };

/// The canonical owner order: every base fn before every instance, then by id.
pub fn ownerLess(a: Owner, b: Owner) bool {
    return ownerOrder(a, b) == .lt;
}

fn ownerOrder(a: Owner, b: Owner) std.math.Order {
    const ta: u8 = @intFromEnum(std.meta.activeTag(a));
    const tb: u8 = @intFromEnum(std.meta.activeTag(b));
    if (ta != tb) return std.math.order(ta, tb);
    const ia = switch (a) {
        inline else => |v| v,
    };
    const ib = switch (b) {
        inline else => |v| v,
    };
    return std.math.order(ia, ib);
}

/// A lambda's own codegen unit: one per lambda per lowerable owner, so a lambda inside a
/// generic fn gets one unit per instance (each instance types the body into its own
/// `node_types`).
pub const LambdaUnit = struct {
    owner: Owner,
    /// The owner's module.
    mod: u32,
    /// The owner's `fn_decl`.
    fn_node: Ast.Index,
    /// The `.lambda` node.
    node: Ast.Index,
    rank: u32,
    /// Owned.
    name: []const u8,
    /// The reified value signature, without the env; `params` owned.
    sig: LayoutEngine.FnSig,
    /// The resolver slot of each param, in order; owned.
    param_slots: []const u32,
};

pub fn freeLambdaUnits(gpa: std.mem.Allocator, units: []const LambdaUnit) void {
    for (units) |u| {
        gpa.free(u.name);
        gpa.free(u.sig.params);
        gpa.free(u.param_slots);
    }
    gpa.free(units);
}

/// The unit symbol of the owner's lambda of rank `rank`. `$` is not a source identifier
/// char and no instance mangle spells an arg `l<n>` (see the test).
pub fn lambdaName(gpa: std.mem.Allocator, owner_sym: []const u8, rank: u32) ![]u8 {
    return std.fmt.allocPrint(gpa, "{s}$l{d}", .{ owner_sym, rank });
}

/// Binary search over `list`, which is sorted by `fn_node`.
pub fn findFn(list: []const FnClosures, fn_node: Ast.Index) ?*const FnClosures {
    const i = std.sort.binarySearch(FnClosures, list, fn_node, struct {
        fn order(key: Ast.Index, fc: FnClosures) std.math.Order {
            return std.math.order(key.int(), fc.fn_node.int());
        }
    }.order) orelse return null;
    return &list[i];
}

/// Binary search over `units`, which is sorted by `(owner, node)`.
pub fn findUnit(units: []const LambdaUnit, owner: Owner, node: Ast.Index) ?u32 {
    const Key = struct { owner: Owner, node: Ast.Index };
    const i = std.sort.binarySearch(LambdaUnit, units, Key{ .owner = owner, .node = node }, struct {
        fn order(k: Key, u: LambdaUnit) std.math.Order {
            return switch (ownerOrder(k.owner, u.owner)) {
                .eq => std.math.order(k.node.int(), u.node.int()),
                else => |o| o,
            };
        }
    }.order) orelse return null;
    return @intCast(i);
}

/// Whether ascending `sorted` holds `x`.
pub fn contains(sorted: []const u32, x: u32) bool {
    return std.sort.binarySearch(u32, sorted, x, struct {
        fn order(key: u32, y: u32) std.math.Order {
            return std.math.order(key, y);
        }
    }.order) != null;
}

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

test "lambda and adapter names never equal a generic-instance mangle or each other" {
    const Mono = @import("Mono.zig");
    const Type = @import("../layout/Type.zig").Type;
    const gpa = testing.allocator;
    const spellings = [_]Type{ Type.int, Type.bool, Type.str, Type.unit, Type.float, Type.structT(3), Type.enumT(4), Type.rawptr };
    const ad = try adapterName(gpa, "f", .fn_value);
    defer gpa.free(ad);
    const l0 = try lambdaName(gpa, "f", 0);
    defer gpa.free(l0);
    const l12 = try lambdaName(gpa, "f", 12);
    defer gpa.free(l12);
    try testing.expectEqualStrings("f$l0", l0);
    try testing.expectEqualStrings("f$l12", l12);
    for ([_][]const u8{ ad, l0, l12 }) |unit| {
        for (spellings) |t| {
            const m = try Mono.mangle(gpa, "f", &.{t});
            defer gpa.free(m);
            try testing.expect(!std.mem.eql(u8, m, unit));
        }
    }
    try testing.expect(!std.mem.eql(u8, ad, l0));
}

test "findFn and contains binary-search their sorted tables" {
    const idx = Ast.Index.from;
    const bs = [_]Boundary{
        .{ .node = idx(4), .rank = 1, .slot_floor = 2, .captures = &.{} },
        .{ .node = idx(9), .rank = 0, .slot_floor = 1, .captures = &.{} },
    };
    const fcs = [_]FnClosures{
        .{ .fn_node = idx(10), .boundaries = &bs, .assigned = &.{} },
        .{ .fn_node = idx(30), .boundaries = &.{}, .assigned = &.{} },
    };
    try testing.expectEqual(idx(30), findFn(&fcs, idx(30)).?.fn_node);
    try testing.expect(findFn(&fcs, idx(20)) == null);
    try testing.expect(findFn(&.{}, idx(10)) == null);
    try testing.expect(contains(&.{ 1, 3, 7 }, 3));
    try testing.expect(!contains(&.{ 1, 3, 7 }, 4));
    try testing.expect(!contains(&.{}, 0));
}

test "findUnit orders base owners before instances, then by node" {
    const idx = Ast.Index.from;
    const Type = @import("../layout/Type.zig").Type;
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = Type.unit };
    const mk = struct {
        fn f(o: Owner, n: u32, s: LayoutEngine.FnSig) LambdaUnit {
            return .{ .owner = o, .mod = 0, .fn_node = idx(100), .node = idx(n), .rank = 0, .name = "", .sig = s, .param_slots = &.{} };
        }
    }.f;
    const units = [_]LambdaUnit{
        mk(.{ .base = 2 }, 5, sig),
        mk(.{ .base = 2 }, 8, sig),
        mk(.{ .base = 7 }, 1, sig),
        mk(.{ .instance = 0 }, 5, sig),
        mk(.{ .instance = 3 }, 2, sig),
    };
    for (units[1..], units[0 .. units.len - 1]) |cur, prev| try testing.expect(ownerLess(prev.owner, cur.owner) or (std.meta.eql(prev.owner, cur.owner) and prev.node.int() < cur.node.int()));
    try testing.expectEqual(@as(?u32, 1), findUnit(&units, .{ .base = 2 }, idx(8)));
    try testing.expectEqual(@as(?u32, 3), findUnit(&units, .{ .instance = 0 }, idx(5)));
    try testing.expectEqual(@as(?u32, 4), findUnit(&units, .{ .instance = 3 }, idx(2)));
    try testing.expectEqual(@as(?u32, null), findUnit(&units, .{ .instance = 2 }, idx(5)));
    try testing.expectEqual(@as(?u32, null), findUnit(&units, .{ .base = 7 }, idx(5)));
    try testing.expect(ownerLess(.{ .base = 99 }, .{ .instance = 0 }));
    try testing.expect(!ownerLess(.{ .instance = 0 }, .{ .base = 99 }));
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
