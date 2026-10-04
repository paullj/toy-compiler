//! The function-value units the typecheck tail mints for codegen, and their names. A
//! peer data module like `Callee`: resolve records each fn's lambda boundaries, types
//! builds the unit tables serially in canonical order, the driver enumerates them as
//! codegen units, lower reads them by key.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Sym = @import("Sym.zig");
const LayoutEngine = @import("../layout/Engine.zig");
const Type = @import("../layout/Type.zig").Type;
const Resolution = @import("Resolution.zig").Resolution;
const Modes = @import("Modes.zig");

/// A local named inside a lambda but declared outside it: `slot` is the owner fn's
/// resolver slot, `first_use` the first identifier node (in resolve order) naming it.
pub const Capture = struct { slot: u32, first_use: Ast.Index };

/// One lambda or local fn in a fn body, as the resolver saw it.
pub const Boundary = struct {
    /// The `.lambda` or `.local_fn` node.
    node: Ast.Index,
    /// Pre-order index among the owner fn's boundaries: the lambda's unit name suffix.
    rank: u32,
    /// Every slot the lambda declares is at least this; a local below it is declared
    /// outside (resolver slots are handed out monotonically per fn).
    slot_floor: u32,
    /// A local fn's own name slot, declared just below `slot_floor`. Its body names itself
    /// through this slot without capturing it: the unit rebuilds its own value from its env.
    self_slot: ?u32 = null,
    /// Ascending slot, deduped; each keeps its first use.
    captures: []const Capture,
};

/// The lambdas and local fns of one top-level fn (or method), recorded only for a fn
/// that has one.
pub const FnClosures = struct {
    /// The enclosing `fn_decl`.
    fn_node: Ast.Index,
    /// Sorted by node index.
    boundaries: []const Boundary,
};

pub fn deinitFnClosures(gpa: std.mem.Allocator, list: []const FnClosures) void {
    for (list) |fc| {
        for (fc.boundaries) |b| gpa.free(b.captures);
        gpa.free(fc.boundaries);
    }
    gpa.free(list);
}

/// A body that owns closure units: a base fn (its global id) or a generic instance (its
/// index in the instance table).
pub const Owner = union(enum) { base: u32, instance: u32 };

/// An owner's boxed locals: the slots a closure captures and something assigns, which
/// live in a GC cell for their whole lifetime. Recorded only for an owner with one.
pub const OwnerClosure = struct {
    owner: Owner,
    /// Ascending; owned.
    boxed: []const u32,
    /// The boxed slots that only closures which never outlive their maker capture, and
    /// that no bound method value holds: such a closure can hold the slot's address
    /// instead of a cell. Ascending; owned.
    unboxable: []const u32 = &.{},
    /// `boxed` without `unboxable`: the slots that keep their cell under frame storage,
    /// because a closure that escapes or a bound method value holds them, or because they
    /// are bigger than `Escape.unbox_max`. Ascending; owned.
    kept_boxed: []const u32 = &.{},
};

pub fn freeOwnerClosures(gpa: std.mem.Allocator, list: []const OwnerClosure) void {
    for (list) |oc| {
        gpa.free(oc.boxed);
        gpa.free(oc.unboxable);
        gpa.free(oc.kept_boxed);
    }
    gpa.free(list);
}

/// How one lowering stores an owner's captured locals: `boxed` in cells, `unboxed` in
/// their own slots, held by address by the closures that capture them.
pub const Storage = struct { boxed: []const u32 = &.{}, unboxed: []const u32 = &.{} };

/// `owner`'s storage, by binary search over `list` (sorted by owner); empty for an owner
/// with no record. With `frame` (frame storage, which the driver turns on with any opt
/// pass), the `unboxable` slots leave their cells and the closures that capture them hold
/// their address; without it, every box stays.
pub fn ownerStorage(list: []const OwnerClosure, owner: Owner, frame: bool) Storage {
    const i = std.sort.binarySearch(OwnerClosure, list, owner, struct {
        fn order(key: Owner, oc: OwnerClosure) std.math.Order {
            return ownerOrder(key, oc.owner);
        }
    }.order) orelse return .{};
    const oc = list[i];
    return if (frame) .{ .boxed = oc.kept_boxed, .unboxed = oc.unboxable } else .{ .boxed = oc.boxed };
}

/// Whether any of `owner`'s lambda units keeps its env in a frame slot. `units` is sorted
/// by `(owner, node)`, so the owner's units are one run of it.
pub fn ownerHasFrameEnv(units: []const LambdaUnit, owner: Owner) bool {
    var i = std.sort.lowerBound(LambdaUnit, units, owner, struct {
        fn order(key: Owner, u: LambdaUnit) std.math.Order {
            return ownerOrder(key, u.owner);
        }
    }.order);
    while (i < units.len and ownerOrder(owner, units[i].owner) == .eq) : (i += 1) {
        if (units[i].frame_env != null) return true;
    }
    return false;
}

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

/// How a closure holds a captured local: `copy` is the local's value, taken when the
/// closure is made, which is sound only while nothing assigns the local; `box` is a
/// pointer to the GC cell the local lives in, shared with its owner and every other
/// closure that captures it.
/// `ref` is the address of the captured storage: a `mut` param's place, which the closure
/// can hold only because it never outlives the call that lent it. Under frame storage a
/// `box` capture of a slot in `Storage.unboxed` is lowered the way `ref` is: the word holds
/// the address of the owner's own slot, since no closure capturing it outlives the owner.
pub const CaptureMode = enum(u8) { copy, box, ref };

/// One captured local's place in its closure's env cell: owner slot, type, mode and byte
/// offset. The creating body writes it there and the lambda unit reads it back, both from
/// this one record.
pub const EnvSlot = struct { slot: u32, ty: Type, mode: CaptureMode, off: u32 };

/// A lambda's or local fn's own codegen unit: one per lambda per lowerable owner, so a
/// lambda inside a generic fn gets one unit per instance (each instance types the body
/// into its own `node_types`).
pub const LambdaUnit = struct {
    owner: Owner,
    /// The owner's module.
    mod: u32,
    /// The owner's `fn_decl`.
    fn_node: Ast.Index,
    /// The `.lambda` or `.local_fn` node.
    node: Ast.Index,
    rank: u32,
    /// Owned.
    name: []const u8,
    /// The reified value signature, without the env; `params` owned.
    sig: LayoutEngine.FnSig,
    /// The resolver slot of each param, in order; owned.
    param_slots: []const u32,
    /// A local fn's own name slot, bound in its prologue to `{its code, its env}`.
    self_slot: ?u32 = null,
    /// One per capture, ascending slot; owned.
    env: []const EnvSlot = &.{},
    /// The env cell's byte size; 0 means the lambda's value carries a null env.
    env_size: u32 = 0,
    /// The struct id of this env laid out as a frame slot (`envStruct`), when the closure
    /// never outlives the body that makes it and its env fits that body's budget; null
    /// keeps the env in a GC cell. Lowering uses it only with `frame_envs` on.
    frame_env: ?u32 = null,
};

pub fn freeLambdaUnit(gpa: std.mem.Allocator, u: LambdaUnit) void {
    gpa.free(u.name);
    gpa.free(u.sig.params);
    gpa.free(u.param_slots);
    gpa.free(u.env);
}

pub fn freeLambdaUnits(gpa: std.mem.Allocator, units: []const LambdaUnit) void {
    for (units) |u| freeLambdaUnit(gpa, u);
    gpa.free(units);
}

/// Lay the captures out in order, each at the next multiple of its alignment, writing
/// each offset to `out_offs`. Returns the env cell size rounded up to 8, or 0 when every
/// capture is zero-sized, so a closure that needs no storage allocates nothing.
pub fn envLayout(sizes: []const u32, aligns: []const u32, out_offs: []u32) u32 {
    std.debug.assert(sizes.len == aligns.len and sizes.len == out_offs.len);
    var off: u32 = 0;
    for (sizes, aligns, out_offs) |s, a, *o| {
        off = std.mem.alignForward(u32, off, @max(a, 1));
        o.* = off;
        off += s;
    }
    if (off == 0) return 0;
    return std.mem.alignForward(u32, off, 8);
}

/// The closure facts an owner's own code depends on that its body walk cannot see: the
/// owner's boxed slots under `st`, then per lambda unit (in rank order) its rank, env size,
/// frame-env bit (with `frame`) and env slots, then `st.unboxed` when it is not empty.
/// Index-free bytes for the owner's fingerprint; empty when there is no env slot and no
/// boxed slot, so a body without captures keeps its key. `units` are the owner's, and `st`
/// is `ownerStorage(.., frame)`, the same storage its lowering gets.
pub fn ownerFacts(gpa: std.mem.Allocator, units: []const LambdaUnit, st: Storage, frame: bool) error{OutOfMemory}![]const u8 {
    const boxed = st.boxed;
    const any_env = for (units) |u| {
        if (u.env.len != 0) break true;
    } else false;
    if (!any_env and boxed.len == 0) return &.{};

    const order = try gpa.alloc(u32, units.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, units, struct {
        fn lt(us: []const LambdaUnit, a: u32, b: u32) bool {
            return us[a].rank < us[b].rank;
        }
    }.lt);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    const w = struct {
        fn u32le(o: *std.ArrayList(u8), g: std.mem.Allocator, v: u32) error{OutOfMemory}!void {
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, v, .little);
            try o.appendSlice(g, &buf);
        }
    }.u32le;
    try w(&out, gpa, @intCast(boxed.len));
    for (boxed) |s| try w(&out, gpa, s);
    try w(&out, gpa, @intCast(units.len));
    for (order) |i| {
        const u = units[i];
        try w(&out, gpa, u.rank);
        try w(&out, gpa, u.env_size);
        // A frame env changes the owner's code but not its body, and whether the closure
        // escapes is a fact about its callees' bodies.
        if (frame) try out.append(gpa, @intFromBool(u.frame_env != null));
        try w(&out, gpa, @intCast(u.env.len));
        for (u.env) |es| {
            try w(&out, gpa, es.slot);
            try out.append(gpa, @intFromEnum(es.mode));
            try w(&out, gpa, es.off);
        }
    }
    // Last, and only when there is one, so every other owner keeps its key: the stream
    // before it is self-delimiting.
    if (st.unboxed.len != 0) {
        try w(&out, gpa, @intCast(st.unboxed.len));
        for (st.unboxed) |s| try w(&out, gpa, s);
    }
    return out.toOwnedSlice(gpa);
}

/// `unit`'s env as a struct, for a frame slot the precise root map can read: each copy is
/// typed as the capture, a cell pointer as `rawptr`, and an address (a `ref`, or a box in
/// `unboxable`) as `int`, so a frame address is never a root.
pub fn envStruct(gpa: std.mem.Allocator, unit: LambdaUnit, unboxable: []const u32) error{OutOfMemory}!LayoutEngine.Layout {
    const name = try std.mem.concat(gpa, u8, &.{ "env$", unit.name });
    errdefer gpa.free(name);
    const names = try gpa.alloc([]const u8, unit.env.len);
    var named: usize = 0;
    errdefer {
        for (names[0..named]) |n| gpa.free(n);
        gpa.free(names);
    }
    for (unit.env, names) |es, *n| {
        n.* = try std.fmt.allocPrint(gpa, "s{d}", .{es.slot});
        named += 1;
    }
    const tys = try gpa.alloc(Type, unit.env.len);
    errdefer gpa.free(tys);
    const offs = try gpa.alloc(u32, unit.env.len);
    for (unit.env, tys, offs) |es, *t, *o| {
        o.* = es.off;
        t.* = switch (es.mode) {
            .copy => es.ty,
            .box => if (contains(unboxable, es.slot)) Type.int else Type.rawptr,
            .ref => Type.int,
        };
    }
    return .{ .name = name, .field_names = @ptrCast(names), .field_types = tys, .offsets = offs, .size = unit.env_size, .@"align" = 8 };
}

pub fn freeEnvStruct(gpa: std.mem.Allocator, l: LayoutEngine.Layout) void {
    gpa.free(l.name);
    for (l.field_names) |n| gpa.free(n);
    gpa.free(l.field_names);
    gpa.free(l.field_types);
    gpa.free(l.offsets);
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

pub const AdapterKind = enum(u8) { fn_value, bound_copy, bound_box, bound_ref };

/// A source-less unit that turns a named callee into the `(env, params..)` shape every
/// function value is called with. `.fn_value` is the thunk a named fn taken as a value
/// points at: it ignores the env and calls the fn directly. A bound method's adapter
/// passes the receiver its env holds as `self`: `.bound_copy` holds a copy of the
/// receiver value, `.bound_box` the cell of a boxed local receiver, `.bound_ref` the
/// address of a `mut` param receiver (as an `int`, so the word is never a root).
pub const Adapter = struct {
    kind: AdapterKind,
    /// `.fn_value`: the referenced fn's global id. `.bound_*`: the method's index.
    key: u32,
    /// Owned.
    name: []const u8,
    /// The callee's symbol; the name is borrowed from the typecheck result's tables.
    target: Sym.SymName,
    /// The value signature, without the env or `self`: borrowed from the callee's
    /// typecheck sig.
    sig: LayoutEngine.FnSig,
    /// `.bound_*`: the method's receiver type.
    recv: Type = Type.unit,
    /// `.bound_*`: the method takes `mut self`, so it is passed the receiver's address.
    mut_self: bool = false,
};

/// The thunk / adapter symbol for `target_sym`. `$` is not a source identifier char, and
/// no generic-instance mangle spells an arg `fnval`, `bound`, `boundbox` or `boundref`
/// (see the test), so the name is disjoint from every user fn, instance, derive and erased unit.
pub fn adapterName(gpa: std.mem.Allocator, target_sym: []const u8, kind: AdapterKind) ![]u8 {
    const suffix = switch (kind) {
        .fn_value => "$fnval",
        .bound_copy => "$bound",
        .bound_box => "$boundbox",
        .bound_ref => "$boundref",
    };
    return std.mem.concat(gpa, u8, &.{ target_sym, suffix });
}

/// The adapter a bound method value goes through: the receiver's address when it is one
/// of the owner's `mut` params (`places`, as `Modes.ofProto`), the receiver's cell when it
/// is a local its owner boxes (the closure then sees later writes to it), else a copy of
/// the receiver taken when the value is made.
pub fn boundKind(recv_local: ?u32, boxed: []const u32, places: u64) AdapterKind {
    const slot = recv_local orelse return .bound_copy;
    if (Modes.isMut(places, slot)) return .bound_ref;
    return if (contains(boxed, slot)) .bound_box else .bound_copy;
}

/// The slot of bound method `fa`'s receiver when it is a bare local (`acc` in
/// `acc.plus`): the one receiver a bound value captures as a variable rather than as a
/// value computed once.
pub fn boundRecvLocal(tree: Ast.Tree, resolutions: []const Resolution, fa: Ast.Index) ?u32 {
    const recv = tree.nodes[fa.int()].lhs;
    if (tree.nodes[recv.int()].tag != .identifier) return null;
    return switch (resolutions[recv.int()]) {
        .local => |slot| slot,
        else => null,
    };
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
    const gpa = testing.allocator;
    const spellings = [_]Type{ Type.int, Type.bool, Type.str, Type.unit, Type.float, Type.structT(3), Type.enumT(4), Type.rawptr };
    const ad = try adapterName(gpa, "f", .fn_value);
    defer gpa.free(ad);
    const bc = try adapterName(gpa, "f", .bound_copy);
    defer gpa.free(bc);
    const bb = try adapterName(gpa, "f", .bound_box);
    defer gpa.free(bb);
    const br = try adapterName(gpa, "f", .bound_ref);
    defer gpa.free(br);
    const l0 = try lambdaName(gpa, "f", 0);
    defer gpa.free(l0);
    const l12 = try lambdaName(gpa, "f", 12);
    defer gpa.free(l12);
    try testing.expectEqualStrings("f$l0", l0);
    try testing.expectEqualStrings("f$l12", l12);
    try testing.expectEqualStrings("f$bound", bc);
    try testing.expectEqualStrings("f$boundbox", bb);
    try testing.expectEqualStrings("f$boundref", br);
    const units = [_][]const u8{ ad, bc, bb, br, l0, l12 };
    for (units) |unit| {
        for (spellings) |t| {
            const m = try Mono.mangle(gpa, "f", &.{t});
            defer gpa.free(m);
            try testing.expect(!std.mem.eql(u8, m, unit));
        }
    }
    for (units, 0..) |a, i| for (units[i + 1 ..]) |b| try testing.expect(!std.mem.eql(u8, a, b));
}

test "boundKind boxes only a boxed local receiver" {
    try testing.expectEqual(AdapterKind.bound_box, boundKind(3, &.{ 1, 3 }, 0));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(2, &.{ 1, 3 }, 0));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(null, &.{ 1, 3 }, 0));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(1, &.{}, 0));
}

test "boundKind holds a mut param receiver by address, else as before" {
    const places: u64 = 0b101;
    try testing.expectEqual(AdapterKind.bound_ref, boundKind(0, &.{}, places));
    try testing.expectEqual(AdapterKind.bound_ref, boundKind(2, &.{2}, places));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(1, &.{}, places));
    try testing.expectEqual(AdapterKind.bound_box, boundKind(3, &.{3}, places));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(null, &.{}, places));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(70, &.{}, ~@as(u64, 0)));
}

test "ownerHasFrameEnv reads only the owner's own run" {
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = Type.unit };
    const idx = Ast.Index.from;
    const units = [_]LambdaUnit{
        .{ .owner = .{ .base = 1 }, .mod = 0, .fn_node = idx(9), .node = idx(3), .rank = 0, .name = "", .sig = sig, .param_slots = &.{} },
        .{ .owner = .{ .base = 1 }, .mod = 0, .fn_node = idx(9), .node = idx(5), .rank = 1, .name = "", .sig = sig, .param_slots = &.{}, .frame_env = 4 },
        .{ .owner = .{ .base = 2 }, .mod = 0, .fn_node = idx(19), .node = idx(13), .rank = 0, .name = "", .sig = sig, .param_slots = &.{} },
        .{ .owner = .{ .instance = 0 }, .mod = 0, .fn_node = idx(29), .node = idx(23), .rank = 0, .name = "", .sig = sig, .param_slots = &.{}, .frame_env = 5 },
    };
    try testing.expect(ownerHasFrameEnv(&units, .{ .base = 1 }));
    try testing.expect(!ownerHasFrameEnv(&units, .{ .base = 2 }));
    try testing.expect(!ownerHasFrameEnv(&units, .{ .base = 7 }));
    try testing.expect(ownerHasFrameEnv(&units, .{ .instance = 0 }));
    try testing.expect(!ownerHasFrameEnv(&units, .{ .instance = 1 }));
    try testing.expect(!ownerHasFrameEnv(&.{}, .{ .base = 1 }));
}

test "findFn and contains binary-search their sorted tables" {
    const idx = Ast.Index.from;
    const bs = [_]Boundary{
        .{ .node = idx(4), .rank = 1, .slot_floor = 2, .captures = &.{} },
        .{ .node = idx(9), .rank = 0, .slot_floor = 1, .captures = &.{} },
    };
    const fcs = [_]FnClosures{
        .{ .fn_node = idx(10), .boundaries = &bs },
        .{ .fn_node = idx(30), .boundaries = &.{} },
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
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = Type.unit };
    const t: Sym.SymName = .{ .kind = .user_fn, .name = "f" };
    const ads = [_]Adapter{
        .{ .kind = .fn_value, .key = 2, .name = "a", .target = t, .sig = sig },
        .{ .kind = .fn_value, .key = 5, .name = "b", .target = t, .sig = sig },
        .{ .kind = .fn_value, .key = 9, .name = "c", .target = t, .sig = sig },
        .{ .kind = .bound_copy, .key = 2, .name = "d", .target = t, .sig = sig },
        .{ .kind = .bound_box, .key = 2, .name = "e", .target = t, .sig = sig },
    };
    try testing.expectEqual(@as(?u32, 0), findAdapter(&ads, .fn_value, 2));
    try testing.expectEqual(@as(?u32, 1), findAdapter(&ads, .fn_value, 5));
    try testing.expectEqual(@as(?u32, 2), findAdapter(&ads, .fn_value, 9));
    try testing.expectEqual(@as(?u32, 3), findAdapter(&ads, .bound_copy, 2));
    try testing.expectEqual(@as(?u32, 4), findAdapter(&ads, .bound_box, 2));
    try testing.expectEqual(@as(?u32, null), findAdapter(&ads, .bound_copy, 9));
    try testing.expectEqual(@as(?u32, null), findAdapter(&ads, .fn_value, 4));
    try testing.expectEqual(@as(?u32, null), findAdapter(&.{}, .fn_value, 4));
}

test "envLayout packs by alignment, rounds to 8, returns 0 when all zero-sized" {
    var offs: [4]u32 = undefined;
    try testing.expectEqual(@as(u32, 24), envLayout(&.{ 1, 8, 2, 4 }, &.{ 1, 8, 2, 4 }, &offs));
    try testing.expectEqualSlices(u32, &.{ 0, 8, 16, 20 }, &offs);
    // An int, a zero-sized unit and a 16-byte str: the unit takes no room.
    var offs3: [3]u32 = undefined;
    try testing.expectEqual(@as(u32, 24), envLayout(&.{ 8, 0, 16 }, &.{ 8, 1, 8 }, &offs3));
    try testing.expectEqualSlices(u32, &.{ 0, 8, 8 }, &offs3);
    var one: [1]u32 = undefined;
    try testing.expectEqual(@as(u32, 8), envLayout(&.{1}, &.{1}, &one));
    var units2: [2]u32 = undefined;
    try testing.expectEqual(@as(u32, 0), envLayout(&.{ 0, 0 }, &.{ 1, 1 }, &units2));
    try testing.expectEqual(@as(u32, 0), envLayout(&.{}, &.{}, &.{}));
}

test "ownerFacts is empty with no env slot and no boxed slot" {
    const gpa = testing.allocator;
    const idx = Ast.Index.from;
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = Type.unit };
    const bare = [_]LambdaUnit{
        .{ .owner = .{ .base = 1 }, .mod = 0, .fn_node = idx(50), .node = idx(3), .rank = 1, .name = "", .sig = sig, .param_slots = &.{} },
        .{ .owner = .{ .base = 1 }, .mod = 0, .fn_node = idx(50), .node = idx(7), .rank = 0, .name = "", .sig = sig, .param_slots = &.{} },
    };
    try testing.expectEqual(@as(usize, 0), (try ownerFacts(gpa, &bare, .{}, false)).len);
    try testing.expectEqual(@as(usize, 0), (try ownerFacts(gpa, &.{}, .{}, false)).len);

    const env = [_]EnvSlot{.{ .slot = 2, .ty = Type.int, .mode = .copy, .off = 0 }};
    var with_env = bare;
    with_env[1].env = &env;
    with_env[1].env_size = 8;
    const f1 = try ownerFacts(gpa, &with_env, .{}, false);
    defer gpa.free(f1);
    try testing.expect(f1.len != 0);
    const f2 = try ownerFacts(gpa, &bare, .{ .boxed = &.{3} }, false);
    defer gpa.free(f2);
    try testing.expect(f2.len != 0);
    try testing.expect(!std.mem.eql(u8, f1, f2));
    // Rank order, not table order: the rank-0 unit (the capturing one) folds first.
    try testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, f1[8..12], .little));
    try testing.expectEqual(@as(u32, 8), std.mem.readInt(u32, f1[12..16], .little));
}

test "ownerStorage finds an owner's record and is empty for an owner without one" {
    const ocs = [_]OwnerClosure{
        .{ .owner = .{ .base = 3 }, .boxed = &.{ 1, 4 } },
        .{ .owner = .{ .base = 9 }, .boxed = &.{0} },
        .{ .owner = .{ .instance = 2 }, .boxed = &.{ 2, 5, 6 } },
    };
    try testing.expectEqualSlices(u32, &.{ 1, 4 }, ownerStorage(&ocs, .{ .base = 3 }, false).boxed);
    try testing.expectEqualSlices(u32, &.{0}, ownerStorage(&ocs, .{ .base = 9 }, false).boxed);
    try testing.expectEqualSlices(u32, &.{ 2, 5, 6 }, ownerStorage(&ocs, .{ .instance = 2 }, false).boxed);
    try testing.expectEqual(@as(usize, 0), ownerStorage(&ocs, .{ .base = 2 }, false).boxed.len);
    try testing.expectEqual(@as(usize, 0), ownerStorage(&ocs, .{ .instance = 3 }, false).boxed.len);
    try testing.expectEqual(@as(usize, 0), ownerStorage(&.{}, .{ .base = 3 }, false).boxed.len);
}

test "ownerStorage keeps every box without frame storage and only the kept ones with it" {
    const ocs = [_]OwnerClosure{
        .{ .owner = .{ .base = 1 }, .boxed = &.{ 2, 5 }, .unboxable = &.{5}, .kept_boxed = &.{2} },
        .{ .owner = .{ .instance = 0 }, .boxed = &.{3}, .unboxable = &.{}, .kept_boxed = &.{3} },
    };
    const off = ownerStorage(&ocs, .{ .base = 1 }, false);
    try testing.expectEqualSlices(u32, &.{ 2, 5 }, off.boxed);
    try testing.expectEqualSlices(u32, &.{}, off.unboxed);
    const on = ownerStorage(&ocs, .{ .base = 1 }, true);
    try testing.expectEqualSlices(u32, &.{2}, on.boxed);
    try testing.expectEqualSlices(u32, &.{5}, on.unboxed);
    try testing.expectEqualSlices(u32, &.{3}, ownerStorage(&ocs, .{ .instance = 0 }, true).boxed);
    try testing.expectEqual(@as(usize, 0), ownerStorage(&ocs, .{ .base = 7 }, true).boxed.len);
}

test "envStruct types a copy as its capture, a cell as rawptr and an address as int" {
    const gpa = testing.allocator;
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = Type.unit };
    const env = [_]EnvSlot{
        .{ .slot = 1, .ty = Type.str, .mode = .copy, .off = 0 },
        .{ .slot = 2, .ty = Type.int, .mode = .box, .off = 16 },
        .{ .slot = 3, .ty = Type.int, .mode = .box, .off = 24 },
        .{ .slot = 0, .ty = Type.int, .mode = .ref, .off = 32 },
    };
    const u: LambdaUnit = .{ .owner = .{ .base = 0 }, .mod = 0, .fn_node = Ast.Index.from(9), .node = Ast.Index.from(4), .rank = 0, .name = "m.f$l0", .sig = sig, .param_slots = &.{}, .env = &env, .env_size = 40 };
    const l = try envStruct(gpa, u, &.{3});
    defer freeEnvStruct(gpa, l);
    try testing.expectEqualStrings("env$m.f$l0", l.name);
    try testing.expectEqualSlices(Type, &.{ Type.str, Type.rawptr, Type.int, Type.int }, l.field_types);
    try testing.expectEqualSlices(u32, &.{ 0, 16, 24, 32 }, l.offsets);
    try testing.expectEqual(@as(u32, 40), l.size);
    try testing.expectEqual(@as(u32, 8), l.@"align");
}

test "ownerFacts folds each unit's frame-env bit only with frame storage" {
    const gpa = testing.allocator;
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = Type.unit };
    const env = [_]EnvSlot{.{ .slot = 2, .ty = Type.int, .mode = .copy, .off = 0 }};
    var u: LambdaUnit = .{ .owner = .{ .base = 1 }, .mod = 0, .fn_node = Ast.Index.from(50), .node = Ast.Index.from(3), .rank = 0, .name = "", .sig = sig, .param_slots = &.{}, .env = &env, .env_size = 8 };
    const heap_o0 = try ownerFacts(gpa, &.{u}, .{}, false);
    defer gpa.free(heap_o0);
    const heap_o1 = try ownerFacts(gpa, &.{u}, .{}, true);
    defer gpa.free(heap_o1);
    u.frame_env = 12;
    const frame_o0 = try ownerFacts(gpa, &.{u}, .{}, false);
    defer gpa.free(frame_o0);
    const frame_o1 = try ownerFacts(gpa, &.{u}, .{}, true);
    defer gpa.free(frame_o1);
    try testing.expectEqualSlices(u8, heap_o0, frame_o0);
    try testing.expect(!std.mem.eql(u8, heap_o1, frame_o1));
    try testing.expectEqual(@as(usize, 0), (try ownerFacts(gpa, &.{}, .{}, true)).len);
}

test "ownerFacts folds which boxes a lowering keeps and which it unboxes" {
    const gpa = testing.allocator;
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = Type.unit };
    const env = [_]EnvSlot{
        .{ .slot = 2, .ty = Type.int, .mode = .box, .off = 0 },
        .{ .slot = 3, .ty = Type.int, .mode = .box, .off = 8 },
    };
    const u: LambdaUnit = .{ .owner = .{ .base = 1 }, .mod = 0, .fn_node = Ast.Index.from(50), .node = Ast.Index.from(3), .rank = 0, .name = "", .sig = sig, .param_slots = &.{}, .env = &env, .env_size = 16 };
    const cells = try ownerFacts(gpa, &.{u}, .{ .boxed = &.{ 2, 3 } }, true);
    defer gpa.free(cells);
    const one = try ownerFacts(gpa, &.{u}, .{ .boxed = &.{2}, .unboxed = &.{3} }, true);
    defer gpa.free(one);
    const other = try ownerFacts(gpa, &.{u}, .{ .boxed = &.{3}, .unboxed = &.{2} }, true);
    defer gpa.free(other);
    const none = try ownerFacts(gpa, &.{u}, .{ .unboxed = &.{ 2, 3 } }, true);
    defer gpa.free(none);
    const all = [_][]const u8{ cells, one, other, none };
    for (all, 0..) |a, i| for (all[i + 1 ..]) |b| try testing.expect(!std.mem.eql(u8, a, b));
}
