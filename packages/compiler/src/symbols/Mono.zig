//! Monomorphization instance table — a peer data module.
//!
//! An `Instance` is one reachable `(generic template, concrete-arg-tuple)` pair
//! reified to an ordinary concrete per-fn codegen unit. The type checker's serial
//! monomorphization tail produces the table; codegen enumerates it as extra
//! lowerable units; the checker's call seal records each generic call's instance
//! (`instanceOf`) on its callee slot, and every later reader takes it from there. This
//! module is the SINGLE home for the canonical logic (`find`, the sort key) so the
//! consumers cannot drift; the instance's symbol is `Mangle.instance`.
//!
//! Determinism is SACRED: `lessThan` is a total order over `(template gid, arg
//! bytes)` and `Mangle.instance` is a pure function of the template name + the arg tuple,
//! so the instance list, its ids, and its mangled symbols are a pure function of
//! source (never hashmap/thread order).

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Type = @import("../layout/Type.zig").Type;
const Callee = @import("Callee.zig").Callee;
const Infer = @import("Infer.zig");
const Sig = @import("Sig.zig").Sig;
const Resolution = @import("Resolution.zig").Resolution;
const Body = @import("Body.zig");

const Mono = @This();

/// One resolved bound `[T has P]` on a monomorphized instance: the witnessing
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
///   * `protocol_args` are the bound's protocol type-args (`[T has Into[int]]` ->
///     `[int]`), substituted through the instance args — 12-byte POD Types. Empty for a
///     non-generic protocol. The OUTER slice is OWNED by the Instance. Folded structurally
///     into the (e) fingerprint component so an `Into[int]` -> `Into[bool]` edit can't
///     serve a stale cached witness.
/// Only the outer `conformances` slice + each `witness_syms`/`protocol_args` outer slice
/// are owned.
pub const ResolvedConformance = struct {
    protocol_name: []const u8,
    conform_ty: Type,
    witness_syms: []const []const u8,
    protocol_args: []const Type = &.{},
};

/// One reified generic instance. All slices are OWNED (freed by the owning
/// `GraphResult`).
pub const Instance = struct {
    /// Global fn id of the generic template this instance specializes.
    template_gid: u32,
    /// The concrete type-args, in generic-param order.
    args: []const Type,
    /// The per-instance types of the template's decl range, from module node `node_base`
    /// (written by the per-instance re-check; read through `nodes`). NEVER the module's
    /// shared slice, so two instances of one template do not alias each other's node slots.
    node_types: []Type,
    /// This instance's method-call selections, parallel to `node_types` (and likewise never
    /// the module's shared slice).
    callees: []Callee,
    node_base: u32 = 0,
    /// The template's params/ret with every `type_var` substituted to concrete.
    params: []const Type,
    ret: Type,
    /// The mangled instance symbol name (e.g. `id$int`), minted in canonical order
    /// after the sort. `null` until then, so a read (`.name.?`) traps loudly on
    /// out-of-order use.
    name: ?[]const u8 = null,
    /// Owning module id (the template's module) and the template's decl node.
    mod: u32,
    decl_node: Ast.Index,
    /// The resolved bounds `[T has P]` witnessing conformances for this instance,
    /// one per bounded generic param, in generic-param order. Empty (`&.{}`)
    /// for an unbounded template's instance so its fingerprint fold is byte-identical
    /// (warm cache preserved). The OUTER slice + each entry's `witness_syms` outer
    /// slice are OWNED by the owning `GraphResult`; see `ResolvedConformance`.
    conformances: []const ResolvedConformance = &.{},
    /// This instance's boxed locals (see `Closure.OwnerClosure`), ascending; owned. Each
    /// instance computes its own: a `mut self` pick, and so a box, can differ per args.
    boxed: []const u32 = &.{},

    pub fn nodes(inst: Instance) Body.Nodes {
        return .{ .base = inst.node_base, .types = inst.node_types, .callees = inst.callees };
    }
};

/// A generic template's inference facts: `params` is the template's value params; `count`
/// is the generic-param count fed to `Infer.infer` and used as the explicit-turbofish
/// arity. Discovery derives it from the `FnSym` (`generic_params.len`); later readers from
/// the `Sig` (`Sig.genericParamCount`) — they agree for any call that survives Pass C.
pub const TemplateRef = struct { params: []const Type, count: u32 };

/// Fn `gid`'s template facts from the `Sig` table, or null when it is no generic template.
pub fn templateRef(sigs: []const Sig, gid: u32) ?TemplateRef {
    if (gid >= sigs.len or !sigs[gid].isGenericTemplate()) return null;
    return .{ .params = sigs[gid].params, .count = sigs[gid].genericParamCount() };
}

/// Where `callInstanceRef` learns a callee's template facts.
pub const Templates = union(enum) {
    /// After the check, from each fn's `Sig`. The composite is gone, so a container param
    /// cannot be re-inferred: such a call resolves by `findByReifiedParams` instead.
    sigs: []const Sig,
    /// Per fn id, its facts as the caller holds them (null for a fn that is no generic
    /// template), and during discovery the live composite, so inference sees through a
    /// container param (`Vec[T]`) exactly as the body check did.
    refs: struct { of: []const ?TemplateRef, decomposer: ?Infer.Decomposer = null },

    fn of(t: Templates, gid: u32) ?TemplateRef {
        return switch (t) {
            .sigs => |sigs| templateRef(sigs, gid),
            .refs => |r| if (gid < r.of.len) r.of[gid] else null,
        };
    }

    fn decomposer(t: Templates) ?Infer.Decomposer {
        return switch (t) {
            .sigs => null,
            .refs => |r| r.decomposer,
        };
    }
};

/// What `callInstanceRef` reads: one body's tree, resolutions and node types, and where
/// its callees' template facts come from.
pub const CallView = struct {
    tree: Ast.Tree,
    resolutions: []const Resolution,
    nodes: Body.Nodes,
    templates: Templates,
};

/// A recovered generic call site: the template `gid` and its concrete type-args
/// (OWNED — the caller frees `args`).
pub const CallRef = struct { gid: u32, args: []Type };

/// Recover `(template gid, concrete type-args)` from a `.call` node's callee — the
/// single arg-extraction both mono discovery (`types.scanCalls`) and the call seal
/// (`instanceOf`) run, so they cannot drift on which `Instance` a call selects. Two
/// call-shapes:
///   * explicit turbofish `id[int](..)` — the callee is a `type_app`; the type-args
///     are read straight from `node_types`.
///   * bare inferred `id(7)` — the callee is a plain `identifier`, OR a qualified
///     module generic fn `mod.id(7)` whose `.field_access` callee `resolveModuleMember`
///     bound to `.func`; both carry the SAME gid on `call.lhs`, so the type-args are
///     inferred from the value-arg `node_types` via `Infer.infer` identically.
/// The leading `if (bres != .func)` is the discriminator: a value-method field_access
/// stays `.unresolved` (discovery binds impl params; the checker records the selected
/// method — genuinely different leaves) and a same-module `N.V` variant
/// access is `.unresolved` too, so both still return null here.
///
/// Returns null for any non-generic / non-`.func` / out-of-range / uninferable call; on
/// `.ok` the caller owns `.args`.
pub fn callInstanceRef(gpa: std.mem.Allocator, view: CallView, call_node: Ast.Node) error{OutOfMemory}!?CallRef {
    if (call_node.tag != .call or call_node.lhs == Ast.none) return null;
    const tree = view.tree;
    const callee = tree.nodes[call_node.lhs.int()];
    switch (callee.tag) {
        .type_app => {
            const bres = view.resolutions[callee.lhs.int()];
            if (bres != .func) return null;
            const gid = bres.func;
            if (view.templates.of(gid) == null) return null;
            const targ_nodes = Ast.rangeSlice(tree, callee.rhs.int());
            const args = try gpa.alloc(Type, targ_nodes.len);
            errdefer gpa.free(args);
            for (targ_nodes, 0..) |tn, k| {
                if (!view.nodes.has(tn)) {
                    gpa.free(args);
                    return null; // pre-typecheck view
                }
                args[k] = view.nodes.typeOf(tn);
            }
            return .{ .gid = gid, .args = args };
        },
        .identifier, .field_access => {
            const bres = view.resolutions[call_node.lhs.int()];
            if (bres != .func) return null;
            const gid = bres.func;
            const tmpl = view.templates.of(gid) orelse return null;
            const value_args = Ast.rangeSlice(tree, call_node.rhs.int());
            const arg_types = try gpa.alloc(Type, value_args.len);
            defer gpa.free(arg_types);
            for (value_args, 0..) |va, k| {
                if (!view.nodes.has(va)) return null; // pre-typecheck view
                arg_types[k] = view.nodes.typeOf(va);
            }
            const targs = (try Infer.infer(gpa, tmpl.count, tmpl.params, arg_types, view.templates.decomposer())) orelse return null;
            return .{ .gid = gid, .args = targs };
        },
        else => return null,
    }
}

/// The instance generic call `call` runs, once the check is done: the explicit or inferred
/// type-args `callInstanceRef` recovers, found in `insts`. The composite is gone by then, so
/// a pure-container call recovers none; it falls to the instance whose reified params
/// accept the call's reified args (`findByReifiedParams`). Null when there is none.
pub fn instanceOf(gpa: std.mem.Allocator, view: CallView, insts: *const Lookup, call: Ast.Index) error{OutOfMemory}!?usize {
    const call_node = view.tree.nodes[call.int()];
    if (try callInstanceRef(gpa, view, call_node)) |ref| {
        defer gpa.free(ref.args);
        if (insts.find(ref.gid, ref.args)) |ii| return ii;
    }
    const bres = view.resolutions[call_node.lhs.int()];
    if (bres != .func) return null;
    const value_args = Ast.rangeSlice(view.tree, call_node.rhs.int());
    // Heap scratch, not a fixed stack buffer, so a call of any arity still finds its
    // instance (a silent miss would drop the caller's recompile dependency).
    const buf = try gpa.alloc(Type, value_args.len);
    defer gpa.free(buf);
    for (value_args, 0..) |va, k| {
        if (!view.nodes.has(va)) return null;
        buf[k] = view.nodes.typeOf(va);
    }
    return insts.findByReifiedParams(bres.func, buf);
}

/// `find` and `findByReifiedParams` over one fixed instance table, by hash. The call seal
/// asks once per generic call, and a scan of the table each time made N calls of N
/// instances cost O(N^2) (#41). Each answer is the scan's: the first match in table order.
pub const Lookup = struct {
    insts: []const Instance,
    by_args: Map = .empty,
    by_params: Map = .empty,
    /// Templates with an instance whose reified params hold an `invalid` slot. `assignable`
    /// lets any arg into that slot, so only the scan can answer a params lookup for them.
    loose: std.AutoHashMapUnmanaged(u32, void) = .empty,

    const Key = struct { gid: u32, types: []const Type };
    const Map = std.HashMapUnmanaged(Key, usize, KeyContext, std.hash_map.default_max_load_percentage);
    /// Hashes exactly what `Type.eql` compares, so two keys `eql` treats as one collide.
    const KeyContext = struct {
        pub fn hash(_: KeyContext, k: Key) u64 {
            var h = std.hash.Wyhash.init(k.gid);
            for (k.types) |t| {
                h.update(&.{@intFromEnum(t.kind)});
                if (Type.carriesIntDesc(t.kind)) h.update(&.{@as(u8, @bitCast(t.int_desc))});
                h.update(std.mem.asBytes(&t.nominalId()));
            }
            return h.final();
        }
        pub fn eql(_: KeyContext, a: Key, b: Key) bool {
            if (a.gid != b.gid or a.types.len != b.types.len) return false;
            for (a.types, b.types) |x, y| if (!Type.eql(x, y)) return false;
            return true;
        }
    };

    pub fn init(gpa: std.mem.Allocator, insts: []const Instance) error{OutOfMemory}!Lookup {
        var l: Lookup = .{ .insts = insts };
        errdefer l.deinit(gpa);
        for (insts, 0..) |inst, i| {
            const a = try l.by_args.getOrPut(gpa, .{ .gid = inst.template_gid, .types = inst.args });
            if (!a.found_existing) a.value_ptr.* = i;
            const loose = for (inst.params) |p| {
                if (p.kind == .invalid) break true;
            } else false;
            if (loose) {
                try l.loose.put(gpa, inst.template_gid, {});
                continue;
            }
            const p = try l.by_params.getOrPut(gpa, .{ .gid = inst.template_gid, .types = inst.params });
            if (!p.found_existing) p.value_ptr.* = i;
        }
        return l;
    }

    pub fn deinit(l: *Lookup, gpa: std.mem.Allocator) void {
        l.by_args.deinit(gpa);
        l.by_params.deinit(gpa);
        l.loose.deinit(gpa);
    }

    pub fn find(l: *const Lookup, gid: u32, args: []const Type) ?usize {
        return l.by_args.get(.{ .gid = gid, .types = args });
    }

    pub fn findByReifiedParams(l: *const Lookup, gid: u32, arg_types: []const Type) ?usize {
        // A `never` or `invalid` arg fits any param, which no single hash key can stand for.
        for (arg_types) |a| if (a.kind == .invalid or a.kind == .never) return Mono.findByReifiedParams(l.insts, gid, arg_types);
        if (l.loose.contains(gid)) return Mono.findByReifiedParams(l.insts, gid, arg_types);
        return l.by_params.get(.{ .gid = gid, .types = arg_types });
    }
};

/// Find the instance whose reified value-PARAMS accept the call's reified value-ARG
/// types — the FIND-regime resolver for a bare container call `total(xs)`, where the
/// composite table is gone (so re-inference is impossible in principle) but every
/// instance already carries its fully-reified `params`. `Type.assignable(param, arg)`
/// — NOT `eql` — so a `never`/`invalid` arg slot is accepted (mirroring
/// `Infer.match`'s skip rule); for concrete types it reduces to `eql` (no coercion).
/// Deterministic linear scan; allocation-free.
pub fn findByReifiedParams(insts: []const Instance, gid: u32, arg_types: []const Type) ?usize {
    for (insts, 0..) |inst, i| {
        if (inst.template_gid != gid or inst.params.len != arg_types.len) continue;
        var ok = true;
        for (inst.params, arg_types) |p, a| {
            if (!Type.assignable(p, a)) {
                ok = false;
                break;
            }
        }
        if (ok) return i;
    }
    return null;
}

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
        switch (x.orderLeaf(y)) {
            .lt => return true,
            .gt => return false,
            .eq => {},
        }
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
    for (args) |a| try a.appendKeyBytes(gpa, buf);
}

const testing = std.testing;

/// A view over a tiny tree whose template facts answer for exactly one gid (below 10).
fn fakeView(tree: Ast.Tree, resolutions: []const Resolution, node_types: []const Type, refs: *[10]?TemplateRef, gen_gid: u32, gen_params: []const Type, gen_count: u32) CallView {
    refs.* = .{null} ** 10;
    refs[gen_gid] = .{ .params = gen_params, .count = gen_count };
    return .{ .tree = tree, .resolutions = resolutions, .nodes = .{ .types = node_types }, .templates = .{ .refs = .{ .of = refs } } };
}

test "callInstanceRef recovers (gid, args) from an explicit turbofish callee" {
    const gpa = testing.allocator;
    // Tree for `id[int](x)`: n0=id, n1=int(type-arg), n2=type_app(lhs=0,rhs=hdr@0),
    // n3=x(value-arg), n4=call(lhs=2,rhs=hdr@3).
    var nodes = [_]Ast.Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .type_app, .main_token = 0, .lhs = Ast.Index.from(0), .rhs = Ast.Index.from(0) },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .call, .main_token = 0, .lhs = Ast.Index.from(2), .rhs = Ast.Index.from(3) },
    };
    var extra = [_]u32{ 2, 1, 1, 4, 1, 3 };
    const tree = Ast.Tree{ .nodes = &nodes, .extra = &extra };
    var resolutions = [_]Resolution{.unresolved} ** 5;
    resolutions[0] = .{ .func = 7 };
    const node_types = [_]Type{ Type.invalid, Type.int, Type.invalid, Type.invalid, Type.invalid };
    var refs: [10]?TemplateRef = undefined;
    const view = fakeView(tree, &resolutions, &node_types, &refs, 7, &.{}, 1);

    const ref = (try callInstanceRef(gpa, view, nodes[4])).?;
    defer gpa.free(ref.args);
    try testing.expectEqual(@as(u32, 7), ref.gid);
    try testing.expectEqual(@as(usize, 1), ref.args.len);
    try testing.expect(Type.eql(ref.args[0], Type.int));
}

test "callInstanceRef infers (gid, args) from a bare generic callee" {
    const gpa = testing.allocator;
    // Tree for `id(x)`: n0=id, n1=x(value-arg), n2=call(lhs=0,rhs=hdr@0).
    var nodes = [_]Ast.Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .call, .main_token = 0, .lhs = Ast.Index.from(0), .rhs = Ast.Index.from(0) },
    };
    var extra = [_]u32{ 2, 1, 1 };
    const tree = Ast.Tree{ .nodes = &nodes, .extra = &extra };
    var resolutions = [_]Resolution{.unresolved} ** 3;
    resolutions[0] = .{ .func = 5 };
    const node_types = [_]Type{ Type.invalid, Type.int, Type.invalid };
    const params = [_]Type{Type.typeVar(0)};
    var refs: [10]?TemplateRef = undefined;
    const view = fakeView(tree, &resolutions, &node_types, &refs, 5, &params, 1);

    const ref = (try callInstanceRef(gpa, view, nodes[2])).?;
    defer gpa.free(ref.args);
    try testing.expectEqual(@as(u32, 5), ref.gid);
    try testing.expectEqual(@as(usize, 1), ref.args.len);
    try testing.expect(Type.eql(ref.args[0], Type.int));
}

test "callInstanceRef returns null for a non-generic or non-func callee" {
    const gpa = testing.allocator;
    var nodes = [_]Ast.Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .call, .main_token = 0, .lhs = Ast.Index.from(0), .rhs = Ast.Index.from(0) },
    };
    var extra = [_]u32{ 2, 1, 1 };
    const tree = Ast.Tree{ .nodes = &nodes, .extra = &extra };
    var resolutions = [_]Resolution{.unresolved} ** 3;
    resolutions[0] = .{ .func = 5 };
    const node_types = [_]Type{ Type.invalid, Type.int, Type.invalid };
    // gen_gid 9 != the callee's gid 5 -> no template facts -> not a template.
    var refs: [10]?TemplateRef = undefined;
    const view = fakeView(tree, &resolutions, &node_types, &refs, 9, &.{}, 1);
    try testing.expectEqual(@as(?CallRef, null), try callInstanceRef(gpa, view, nodes[2]));
}

test "callInstanceRef infers (gid, args) from a qualified bare generic callee" {
    const gpa = testing.allocator;
    // Tree for `mod.id(x)`: n0=mod(receiver, unused by the arm), n1=field_access
    // (call.lhs, bound to .func by resolveModuleMember), n2=x(value-arg),
    // n3=call(lhs=1,rhs=hdr@0). The arm reads resolutions[call.lhs] + call.rhs only.
    var nodes = [_]Ast.Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 0, .lhs = Ast.Index.from(0), .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .call, .main_token = 0, .lhs = Ast.Index.from(1), .rhs = Ast.Index.from(0) },
    };
    var extra = [_]u32{ 2, 1, 2 };
    const tree = Ast.Tree{ .nodes = &nodes, .extra = &extra };
    var resolutions = [_]Resolution{.unresolved} ** 4;
    resolutions[1] = .{ .func = 5 };
    const node_types = [_]Type{ Type.invalid, Type.invalid, Type.int, Type.invalid };
    const params = [_]Type{Type.typeVar(0)};
    var refs: [10]?TemplateRef = undefined;
    const view = fakeView(tree, &resolutions, &node_types, &refs, 5, &params, 1);

    const ref = (try callInstanceRef(gpa, view, nodes[3])).?;
    defer gpa.free(ref.args);
    try testing.expectEqual(@as(u32, 5), ref.gid);
    try testing.expectEqual(@as(usize, 1), ref.args.len);
    try testing.expect(Type.eql(ref.args[0], Type.int));
}

test "callInstanceRef returns null for a value-method field_access callee" {
    const gpa = testing.allocator;
    // A `recv.m(x)` value method: the field_access callee stays .unresolved (no .func),
    // so the `bres != .func` discriminator excludes it — a method is not a bare-inferred
    // free fn.
    var nodes = [_]Ast.Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .field_access, .main_token = 0, .lhs = Ast.Index.from(0), .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .call, .main_token = 0, .lhs = Ast.Index.from(1), .rhs = Ast.Index.from(0) },
    };
    var extra = [_]u32{ 2, 1, 2 };
    const tree = Ast.Tree{ .nodes = &nodes, .extra = &extra };
    const resolutions = [_]Resolution{.unresolved} ** 4;
    const node_types = [_]Type{ Type.invalid, Type.invalid, Type.int, Type.invalid };
    const params = [_]Type{Type.typeVar(0)};
    var refs: [10]?TemplateRef = undefined;
    const view = fakeView(tree, &resolutions, &node_types, &refs, 5, &params, 1);
    try testing.expectEqual(@as(?CallRef, null), try callInstanceRef(gpa, view, nodes[3]));
}

/// A decomposer over a static decomposed table, so `callInstanceRef` can infer a container
/// type-arg WITHOUT a real `Composite`.
const FakeComposite = struct {
    entries: []const Infer.Decomposed,
    fn thunk(ctx: *anyopaque, ty: Type) ?Infer.Decomposed {
        if (!ty.isApp()) return null;
        const self: *const FakeComposite = @ptrCast(@alignCast(ctx));
        const idx = ty.appIdx();
        if (idx >= self.entries.len) return null;
        return self.entries[idx];
    }
    fn decomposer(self: *const FakeComposite) Infer.Decomposer {
        return .{ .ctx = @constCast(self), .func = thunk };
    }
};

test "callInstanceRef infers a CONTAINER type-arg through the live decomposer" {
    const gpa = testing.allocator;
    // `total(xs)`: n0=id, n1=xs(value-arg, typed Vec[int] = app(1)), n2=call.
    var nodes = [_]Ast.Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .call, .main_token = 0, .lhs = Ast.Index.from(0), .rhs = Ast.Index.from(0) },
    };
    var extra = [_]u32{ 2, 1, 1 };
    const tree = Ast.Tree{ .nodes = &nodes, .extra = &extra };
    var resolutions = [_]Resolution{.unresolved} ** 3;
    resolutions[0] = .{ .func = 5 };
    // node_types[1] = Vec[int] = app(1).
    const node_types = [_]Type{ Type.invalid, Type.app(1), Type.invalid };
    const params = [_]Type{Type.app(0)}; // param Vec[T] = app(0)
    const entries = [_]Infer.Decomposed{
        .{ .ctor = 1, .is_enum = false, .args = &.{Type.typeVar(0)} }, // 0: Vec[T]
        .{ .ctor = 1, .is_enum = false, .args = &.{Type.int} }, // 1: Vec[int]
    };
    const composite: FakeComposite = .{ .entries = &entries };
    var refs: [10]?TemplateRef = undefined;
    var view = fakeView(tree, &resolutions, &node_types, &refs, 5, &params, 1);
    view.templates.refs.decomposer = composite.decomposer();

    const ref = (try callInstanceRef(gpa, view, nodes[2])).?;
    defer gpa.free(ref.args);
    try testing.expectEqual(@as(u32, 5), ref.gid);
    try testing.expectEqual(@as(usize, 1), ref.args.len);
    try testing.expect(Type.eql(ref.args[0], Type.int));
}

test "findByReifiedParams matches on reified params, tolerating a never/invalid slot" {
    const insts = [_]Instance{
        .{ .template_gid = 3, .args = &.{Type.int}, .node_types = &.{}, .callees = &.{}, .params = &.{Type.structT(7)}, .ret = Type.int, .name = "total$s7", .mod = 0, .decl_node = Ast.none },
        .{ .template_gid = 3, .args = &.{Type.bool}, .node_types = &.{}, .callees = &.{}, .params = &.{ Type.structT(8), Type.int }, .ret = Type.int, .name = "pair$s8", .mod = 0, .decl_node = Ast.none },
    };
    // Exact reified match.
    try testing.expectEqual(@as(?usize, 0), findByReifiedParams(&insts, 3, &.{Type.structT(7)}));
    // Wrong gid / wrong reified param -> no match.
    try testing.expectEqual(@as(?usize, null), findByReifiedParams(&insts, 4, &.{Type.structT(7)}));
    try testing.expectEqual(@as(?usize, null), findByReifiedParams(&insts, 3, &.{Type.structT(9)}));
    // A `never` arg slot is accepted via `assignable`; the concrete slot disambiguates.
    try testing.expectEqual(@as(?usize, 1), findByReifiedParams(&insts, 3, &.{ Type.structT(8), Type.never }));
}

test "find matches on gid + args by Type.eql" {
    const insts = [_]Instance{
        .{ .template_gid = 0, .args = &.{Type.int}, .node_types = &.{}, .callees = &.{}, .params = &.{Type.int}, .ret = Type.int, .name = "id$int", .mod = 0, .decl_node = Ast.none },
        .{ .template_gid = 0, .args = &.{Type.structT(0)}, .node_types = &.{}, .callees = &.{}, .params = &.{Type.structT(0)}, .ret = Type.structT(0), .name = "id$s0", .mod = 0, .decl_node = Ast.none },
    };
    try testing.expectEqual(@as(?usize, 0), find(&insts, 0, &.{Type.int}));
    try testing.expectEqual(@as(?usize, 1), find(&insts, 0, &.{Type.structT(0)}));
    try testing.expectEqual(@as(?usize, null), find(&insts, 0, &.{Type.bool}));
    try testing.expectEqual(@as(?usize, null), find(&insts, 1, &.{Type.int}));
}

test "Lookup answers every find and findByReifiedParams as the scans do" {
    const gpa = testing.allocator;
    const insts = [_]Instance{
        .{ .template_gid = 0, .args = &.{Type.int}, .node_types = &.{}, .callees = &.{}, .params = &.{Type.int}, .ret = Type.int, .mod = 0, .decl_node = Ast.none },
        .{ .template_gid = 0, .args = &.{Type.structT(0)}, .node_types = &.{}, .callees = &.{}, .params = &.{Type.structT(0)}, .ret = Type.int, .mod = 0, .decl_node = Ast.none },
        .{ .template_gid = 1, .args = &.{Type.int}, .node_types = &.{}, .callees = &.{}, .params = &.{ Type.structT(7), Type.int }, .ret = Type.int, .mod = 0, .decl_node = Ast.none },
        .{ .template_gid = 1, .args = &.{Type.bool}, .node_types = &.{}, .callees = &.{}, .params = &.{ Type.structT(8), Type.int }, .ret = Type.int, .mod = 0, .decl_node = Ast.none },
        // A poisoned param slot takes any arg, so gid 2 must fall back to the scan.
        .{ .template_gid = 2, .args = &.{Type.int}, .node_types = &.{}, .callees = &.{}, .params = &.{ Type.invalid, Type.int }, .ret = Type.int, .mod = 0, .decl_node = Ast.none },
        .{ .template_gid = 2, .args = &.{Type.bool}, .node_types = &.{}, .callees = &.{}, .params = &.{ Type.bool, Type.int }, .ret = Type.int, .mod = 0, .decl_node = Ast.none },
    };
    var l = try Lookup.init(gpa, &insts);
    defer l.deinit(gpa);
    const queries = [_]struct { gid: u32, types: []const Type }{
        .{ .gid = 0, .types = &.{Type.int} },
        .{ .gid = 0, .types = &.{Type.structT(0)} },
        .{ .gid = 0, .types = &.{Type.bool} },
        .{ .gid = 0, .types = &.{Type.never} },
        .{ .gid = 1, .types = &.{Type.int} },
        .{ .gid = 1, .types = &.{ Type.structT(8), Type.int } },
        .{ .gid = 1, .types = &.{ Type.structT(8), Type.never } },
        .{ .gid = 1, .types = &.{ Type.invalid, Type.int } },
        .{ .gid = 1, .types = &.{Type.structT(7)} },
        .{ .gid = 2, .types = &.{ Type.bool, Type.int } },
        .{ .gid = 2, .types = &.{Type.bool} },
        .{ .gid = 3, .types = &.{Type.int} },
    };
    for (queries) |q| {
        try testing.expectEqual(find(&insts, q.gid, q.types), l.find(q.gid, q.types));
        try testing.expectEqual(findByReifiedParams(&insts, q.gid, q.types), l.findByReifiedParams(q.gid, q.types));
    }
    // The scan picks gid 2's poisoned instance first; so must the lookup.
    try testing.expectEqual(@as(?usize, 4), l.findByReifiedParams(2, &.{ Type.bool, Type.int }));
}

test "an instance carries its resolved conformances" {
    const witness = [_][]const u8{"lib.P.into$Into$int"};
    const pargs = [_]Type{Type.int};
    const confs = [_]ResolvedConformance{.{ .protocol_name = "Into", .conform_ty = Type.structT(3), .witness_syms = &witness, .protocol_args = &pargs }};
    const inst = Instance{
        .template_gid = 5,
        .args = &.{Type.structT(3)},
        .node_types = &.{},
        .callees = &.{},
        .params = &.{Type.structT(3)},
        .ret = Type.int,
        .name = "use$s3",
        .mod = 0,
        .decl_node = Ast.none,
        .conformances = &confs,
    };
    try testing.expectEqual(@as(usize, 1), inst.conformances.len);
    try testing.expectEqualStrings("Into", inst.conformances[0].protocol_name);
    try testing.expect(Type.eql(Type.structT(3), inst.conformances[0].conform_ty));
    try testing.expectEqualStrings("lib.P.into$Into$int", inst.conformances[0].witness_syms[0]);
    // The bound's protocol type-args ride the conformance (folded into the (e) fp).
    try testing.expectEqual(@as(usize, 1), inst.conformances[0].protocol_args.len);
    try testing.expect(Type.eql(Type.int, inst.conformances[0].protocol_args[0]));
    // An instance built WITHOUT conformances defaults to the empty slice (warm-cache
    // byte-identity for unbounded templates); its conformance's protocol_args default too.
    const plain = Instance{ .template_gid = 0, .args = &.{Type.int}, .node_types = &.{}, .callees = &.{}, .params = &.{Type.int}, .ret = Type.int, .name = "id$int", .mod = 0, .decl_node = Ast.none };
    try testing.expectEqual(@as(usize, 0), plain.conformances.len);
}

test "lessThan is a total canonical order (gid then arg triples)" {
    const a = Instance{ .template_gid = 0, .args = &.{Type.int}, .node_types = &.{}, .callees = &.{}, .params = &.{}, .ret = Type.int, .name = "", .mod = 0, .decl_node = Ast.none };
    const b = Instance{ .template_gid = 0, .args = &.{Type.structT(0)}, .node_types = &.{}, .callees = &.{}, .params = &.{}, .ret = Type.int, .name = "", .mod = 0, .decl_node = Ast.none };
    const c = Instance{ .template_gid = 1, .args = &.{Type.int}, .node_types = &.{}, .callees = &.{}, .params = &.{}, .ret = Type.int, .name = "", .mod = 0, .decl_node = Ast.none };
    // int (kind 2) < struct (kind 6), so a < b; gid 0 < gid 1, so a < c.
    try testing.expect(lessThan({}, a, b));
    try testing.expect(!lessThan({}, b, a));
    try testing.expect(lessThan({}, a, c));
    try testing.expect(lessThan({}, b, c));
}
