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
const Body = @import("Body.zig");

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

/// A body that owns closure units, by its index in the program's body list
/// (`Body.List`): a base fn or a generic instance.
pub const Owner = Body.Index;

/// The escape pass's storage decision for one owner with a boxed local. Recorded only for
/// an owner whose body boxes a local.
pub const OwnerClosure = struct {
    owner: Owner,
    /// The boxed slots that frame storage takes out of their cells: only closures which
    /// never outlive their maker capture them, no bound method value holds them, and they
    /// are at most `Escape.unbox_max` bytes. Ascending; owned.
    unboxable: []const u32 = &.{},
};

pub fn freeOwnerClosures(gpa: std.mem.Allocator, list: []const OwnerClosure) void {
    for (list) |oc| gpa.free(oc.unboxable);
    gpa.free(list);
}

/// How one lowering of an owner holds a local that its closures (or a bound method value)
/// capture. `copy` is the local's value, taken when the closure is made, which is sound
/// only while nothing assigns the local. `cell` is the GC cell the local lives in for its
/// whole lifetime, shared with its owner and every closure that captures it. `frame_addr`
/// is the address of the owner's own slot, for a boxed local that frame storage takes out
/// of its cell because no closure capturing it outlives the owner. `param_addr` is the
/// caller's place of a `mut` param (`mut self` included), which a closure can hold only
/// because it never outlives the call that lent it.
pub const Hold = enum(u8) { copy, cell, frame_addr, param_addr };

/// One lowering's storage for an owner's captured locals: the one answer the env layout,
/// lower, the adapters, the fingerprint and the LSP read.
pub const Storage = struct {
    /// The owner's boxed locals, ascending: each in a cell unless in `frame_addrs`.
    boxed: []const u32 = &.{},
    /// The boxed locals this lowering keeps in the owner's frame, ascending.
    frame_addrs: []const u32 = &.{},
    /// The owner's `mut` params, `mut self` first (its sig's `mut_mask`).
    places: u64 = 0,

    pub fn hold(st: Storage, slot: u32) Hold {
        if (Modes.isMut(st.places, slot)) return .param_addr;
        if (!contains(st.boxed, slot)) return .copy;
        return if (contains(st.frame_addrs, slot)) .frame_addr else .cell;
    }
};

/// Body `b`'s storage with every box in its cell: what typecheck lays the envs and the
/// bound adapters out against, and what a lowering without frame storage gets.
pub fn heapStorage(b: Body) Storage {
    return .{ .boxed = b.boxed, .places = b.sig.mut_mask };
}

/// `owner`'s storage in one lowering. With `frame` (frame storage, which the driver turns
/// on with any opt pass) the owner's `unboxable` slots (`list`, sorted by owner) leave
/// their cells; without it, every box stays.
pub fn ownerStorage(bodies: Body.List, list: []const OwnerClosure, owner: Owner, frame: bool) Storage {
    var st = heapStorage(bodies.items[owner]);
    if (!frame) return st;
    const i = std.sort.binarySearch(OwnerClosure, list, owner, struct {
        fn order(key: Owner, oc: OwnerClosure) std.math.Order {
            return std.math.order(key, oc.owner);
        }
    }.order) orelse return st;
    st.frame_addrs = list[i].unboxable;
    return st;
}

/// Whether any of `owner`'s lambda units keeps its env in a frame slot.
pub fn ownerHasFrameEnv(units: []const LambdaUnit, owner: Owner) bool {
    return for (ownerUnits(units, owner)) |u| {
        if (u.frame_env != null) break true;
    } else false;
}

/// `owner`'s lambda units: one run of `units`, which is sorted by `(owner, node)`.
pub fn ownerUnits(units: []const LambdaUnit, owner: Owner) []const LambdaUnit {
    const lo = std.sort.lowerBound(LambdaUnit, units, owner, struct {
        fn order(key: Owner, u: LambdaUnit) std.math.Order {
            return std.math.order(key, u.owner);
        }
    }.order);
    var hi = lo;
    while (hi < units.len and units[hi].owner == owner) hi += 1;
    return units[lo..hi];
}

/// One captured local's place in its closure's env cell: owner slot, type and byte offset.
/// The creating body writes it there and the lambda unit reads it back, both from this one
/// record, holding the local as the owner's `Storage` says.
pub const EnvSlot = struct { slot: u32, ty: Type, off: u32 };

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
            return switch (std.math.order(k.owner, u.owner)) {
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

pub const AdapterKind = enum(u8) {
    fn_value,
    bound_copy,
    bound_box,
    bound_ref,

    /// How a bound adapter's env holds the receiver.
    pub fn recvHold(k: AdapterKind) Hold {
        return switch (k) {
            .bound_copy => .copy,
            .bound_box => .cell,
            .bound_ref => .param_addr,
            .fn_value => unreachable,
        };
    }
};

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

/// The adapter a bound method value goes through: it holds a bare local receiver as the
/// owner's storage `st` holds it (a `mut` param's address, a boxed local's cell, so the
/// value and its owner see each other's writes, or a copy), and any other receiver as a
/// copy taken when the value is made. The adapter table is minted once, from the heap
/// storage, so frame storage never takes a bound receiver's cell away (`analyzeEscapes`).
pub fn boundKind(recv_local: ?u32, st: Storage) AdapterKind {
    const slot = recv_local orelse return .bound_copy;
    return switch (st.hold(slot)) {
        .copy => .bound_copy,
        .cell => .bound_box,
        .param_addr => .bound_ref,
        .frame_addr => unreachable,
    };
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
const Sig = @import("Sig.zig").Sig;

test "boundKind boxes only a boxed local receiver" {
    const st: Storage = .{ .boxed = &.{ 1, 3 } };
    try testing.expectEqual(AdapterKind.bound_box, boundKind(3, st));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(2, st));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(null, st));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(1, .{}));
}

test "boundKind holds a mut param receiver by address, else as before" {
    const places: u64 = 0b101;
    try testing.expectEqual(AdapterKind.bound_ref, boundKind(0, .{ .places = places }));
    try testing.expectEqual(AdapterKind.bound_ref, boundKind(2, .{ .boxed = &.{2}, .places = places }));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(1, .{ .places = places }));
    try testing.expectEqual(AdapterKind.bound_box, boundKind(3, .{ .boxed = &.{3}, .places = places }));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(null, .{ .places = places }));
    try testing.expectEqual(AdapterKind.bound_copy, boundKind(70, .{ .places = ~@as(u64, 0) }));
    for ([_]AdapterKind{ .bound_copy, .bound_box, .bound_ref }) |k| {
        const st: Storage = .{ .boxed = &.{1}, .places = 0b100 };
        const slot: u32 = switch (k.recvHold()) {
            .copy => 0,
            .cell => 1,
            .param_addr => 2,
            .frame_addr => unreachable,
        };
        try testing.expectEqual(k, boundKind(slot, st));
    }
}

test "hold: a mut param by address, a boxed local in its cell or frame, any other a copy" {
    const st: Storage = .{ .boxed = &.{ 2, 4, 5 }, .frame_addrs = &.{4}, .places = 0b1 };
    try testing.expectEqual(Hold.param_addr, st.hold(0));
    try testing.expectEqual(Hold.copy, st.hold(1));
    try testing.expectEqual(Hold.cell, st.hold(2));
    try testing.expectEqual(Hold.frame_addr, st.hold(4));
    try testing.expectEqual(Hold.cell, st.hold(5));
    try testing.expectEqual(Hold.copy, st.hold(70));
}

test "ownerHasFrameEnv reads only the owner's own run" {
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = Type.unit };
    const idx = Ast.Index.from;
    const units = [_]LambdaUnit{
        .{ .owner = 1, .mod = 0, .fn_node = idx(9), .node = idx(3), .rank = 0, .name = "", .sig = sig, .param_slots = &.{} },
        .{ .owner = 1, .mod = 0, .fn_node = idx(9), .node = idx(5), .rank = 1, .name = "", .sig = sig, .param_slots = &.{}, .frame_env = 4 },
        .{ .owner = 2, .mod = 0, .fn_node = idx(19), .node = idx(13), .rank = 0, .name = "", .sig = sig, .param_slots = &.{} },
        .{ .owner = 100, .mod = 0, .fn_node = idx(29), .node = idx(23), .rank = 0, .name = "", .sig = sig, .param_slots = &.{}, .frame_env = 5 },
    };
    try testing.expect(ownerHasFrameEnv(&units, 1));
    try testing.expect(!ownerHasFrameEnv(&units, 2));
    try testing.expect(!ownerHasFrameEnv(&units, 7));
    try testing.expect(ownerHasFrameEnv(&units, 100));
    try testing.expect(!ownerHasFrameEnv(&units, 101));
    try testing.expect(!ownerHasFrameEnv(&.{}, 1));
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

test "findUnit orders by owner, then by node" {
    const idx = Ast.Index.from;
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = Type.unit };
    const mk = struct {
        fn f(o: Owner, n: u32, s: LayoutEngine.FnSig) LambdaUnit {
            return .{ .owner = o, .mod = 0, .fn_node = idx(100), .node = idx(n), .rank = 0, .name = "", .sig = s, .param_slots = &.{} };
        }
    }.f;
    const units = [_]LambdaUnit{
        mk(2, 5, sig),
        mk(2, 8, sig),
        mk(7, 1, sig),
        mk(100, 5, sig),
        mk(103, 2, sig),
    };
    for (units[1..], units[0 .. units.len - 1]) |cur, prev| try testing.expect(prev.owner < cur.owner or (prev.owner == cur.owner and prev.node.int() < cur.node.int()));
    try testing.expectEqual(@as(?u32, 1), findUnit(&units, 2, idx(8)));
    try testing.expectEqual(@as(?u32, 3), findUnit(&units, 100, idx(5)));
    try testing.expectEqual(@as(?u32, 4), findUnit(&units, 103, idx(2)));
    try testing.expectEqual(@as(?u32, null), findUnit(&units, 102, idx(5)));
    try testing.expectEqual(@as(?u32, null), findUnit(&units, 7, idx(5)));
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

test "ownerStorage keeps every box without frame storage and moves the unboxable ones with it" {
    const sig: Sig = .{ .kind = .user_fn, .name = "f", .params = &.{}, .ret = Type.unit };
    const mut_sig: Sig = .{ .kind = .user_fn, .name = "g", .params = &.{}, .ret = Type.unit, .mut_mask = 0b1 };
    const body = struct {
        fn f(s: Sig, boxed: []const u32) Body {
            return .{ .of = .{ .base = 0 }, .mod = 0, .decl = Ast.none, .sig = s, .nodes = .{}, .boxed = boxed };
        }
    }.f;
    const items = [_]Body{ body(sig, &.{ 2, 5 }), body(mut_sig, &.{3}), body(sig, &.{}) };
    const bodies: Body.List = .{ .items = &items, .n_base = 3 };
    const ocs = [_]OwnerClosure{
        .{ .owner = 0, .unboxable = &.{5} },
        .{ .owner = 1, .unboxable = &.{} },
    };
    const off = ownerStorage(bodies, &ocs, 0, false);
    try testing.expectEqualSlices(u32, &.{ 2, 5 }, off.boxed);
    try testing.expectEqualSlices(u32, &.{}, off.frame_addrs);
    try testing.expectEqual(Hold.cell, off.hold(5));
    const on = ownerStorage(bodies, &ocs, 0, true);
    try testing.expectEqualSlices(u32, &.{ 2, 5 }, on.boxed);
    try testing.expectEqualSlices(u32, &.{5}, on.frame_addrs);
    try testing.expectEqual(Hold.cell, on.hold(2));
    try testing.expectEqual(Hold.frame_addr, on.hold(5));
    try testing.expectEqual(Hold.param_addr, ownerStorage(bodies, &ocs, 1, true).hold(0));
    try testing.expectEqual(Hold.cell, ownerStorage(bodies, &ocs, 1, true).hold(3));
    try testing.expectEqual(@as(usize, 0), ownerStorage(bodies, &ocs, 2, true).boxed.len);
    try testing.expectEqual(@as(usize, 0), ownerStorage(bodies, &.{}, 0, true).frame_addrs.len);
}
