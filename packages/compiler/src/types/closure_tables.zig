//! The closure tables the typecheck tail builds once its tables are final: the adapters,
//! the lambda units with their env layouts, the escape pass (T0052, and which boxes and
//! envs frame storage may keep in a frame), the owner-closure table and the frame env
//! structs. Serial over the finished tables in canonical body order, so identical at any `-j`.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Resolution = @import("../symbols/Resolution.zig").Resolution;
const Sig = @import("../symbols/Sig.zig").Sig;
const Callee = @import("../symbols/Callee.zig").Callee;
const Closure = @import("../symbols/Closure.zig");
const Modes = @import("../symbols/Modes.zig");
const Escape = @import("../symbols/Escape.zig");
const Body = @import("../symbols/Body.zig");
const Mono = @import("../symbols/Mono.zig");
const Mangle = @import("../symbols/Mangle.zig");
const Walk = @import("../ast/Walk.zig");
const LayoutEngine = @import("../layout/Engine.zig");
const Abi = @import("../codegen/abi/Abi.zig");
const Typecheck = @import("../types.zig");
const Type = Typecheck.Type;
const Layout = Typecheck.Layout;
const EnumLayout = Typecheck.EnumLayout;
const DeriveRecipe = Typecheck.DeriveRecipe;

pub fn freeAdapters(gpa: std.mem.Allocator, adapters: []const Closure.Adapter) void {
    for (adapters) |a| gpa.free(a.name);
    gpa.free(adapters);
}

/// The adapter table, sorted by `(kind, key)` and deduped: one `.fn_value` thunk per fn a
/// module body or an instance body names as a value (`Callee.fn_ref`), and one bound
/// adapter per method bound as a value (`Callee.bound_method`), boxed or copying as its
/// receiver is held in the owner. Serial over the finished tables, so it is identical at
/// any `-j`.
pub fn buildAdapters(
    gpa: std.mem.Allocator,
    graph: *const Typecheck.GraphCtx,
    bodies: Body.List,
    callees: []const []const Callee,
    instances: []const Mono.Instance,
    fns: []const Typecheck.GraphFnInput,
    sigs: []const Sig,
    methods: []const Typecheck.Method,
    derives: []const DeriveRecipe,
) ![]Closure.Adapter {
    const Key = struct { kind: Closure.AdapterKind, key: u32 };
    var keys: std.ArrayList(Key) = .empty;
    defer keys.deinit(gpa);
    var any_bound = false;
    for (callees) |cs| for (cs) |c| switch (c.view()) {
        .fn_ref => |gid| try keys.append(gpa, .{ .kind = .fn_value, .key = gid }),
        .bound_method => any_bound = true,
        .none, .method, .protocol, .builtin, .value, .direct, .instance, .generic => {},
    };
    for (instances) |inst| for (inst.callees) |c| switch (c.view()) {
        .fn_ref => |gid| try keys.append(gpa, .{ .kind = .fn_value, .key = gid }),
        .bound_method => any_bound = true,
        .none, .method, .protocol, .builtin, .value, .direct, .instance, .generic => {},
    };
    // Whether a bound value boxes depends on its owner's boxed list, so bound values are
    // found per owner, by walking the owner's body.
    if (any_bound) for (bodies.items) |b| try collectBound(gpa, &graph.mods[b.mod], b, &keys);
    std.mem.sort(Key, keys.items, {}, struct {
        fn lt(_: void, a: Key, b: Key) bool {
            if (a.kind != b.kind) return @intFromEnum(a.kind) < @intFromEnum(b.kind);
            return a.key < b.key;
        }
    }.lt);
    var n: usize = 0;
    for (keys.items) |k| {
        if (n > 0 and std.meta.eql(keys.items[n - 1], k)) continue;
        keys.items[n] = k;
        n += 1;
    }
    const out = try gpa.alloc(Closure.Adapter, n);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |a| gpa.free(a.name);
        gpa.free(out);
    }
    for (keys.items[0..n], out) |k, *a| {
        switch (k.kind) {
            .fn_value => a.* = .{
                .kind = .fn_value,
                .key = k.key,
                .name = try Mangle.adapter(gpa, fns[k.key].name, .fn_value),
                .target = .{ .kind = fns[k.key].kind, .name = fns[k.key].name },
                .sig = .{ .params = sigs[k.key].params, .ret = sigs[k.key].ret },
            },
            .bound_copy, .bound_box, .bound_ref => {
                const m = methods[k.key];
                const target = Typecheck.witnessSym(m, derives, instances, fns).?;
                const sig = Typecheck.witnessSig(m, derives, instances, sigs).?;
                a.* = .{
                    .kind = k.kind,
                    .key = k.key,
                    .name = try Mangle.adapter(gpa, target.name, k.kind),
                    .target = target,
                    .sig = .{ .params = sig.params[@min(sig.params.len, 1)..], .ret = sig.ret },
                    .recv = m.recv,
                    .mut_self = m.mut_self,
                };
            },
        }
        built += 1;
    }
    return out;
}

/// Append the `(kind, method)` adapter key of every bound method value in body `b` (its
/// lambdas included), read from its own callees.
fn collectBound(
    gpa: std.mem.Allocator,
    mc: *const Typecheck.GraphCtx.ModuleCtx,
    b: Body,
    keys: anytype,
) error{OutOfMemory}!void {
    const V = struct {
        gpa: std.mem.Allocator,
        tree: Ast.Tree,
        resolutions: []const Resolution,
        body: Body,
        keys: @TypeOf(keys),
        pub fn on(self: *@This(), ev: Walk.Event) error{OutOfMemory}!void {
            const e = switch (ev) {
                .enter => |e| e,
                else => return,
            };
            if (e.idx.int() >= self.body.callees.len) return;
            switch (self.body.callees[e.idx.int()].view()) {
                .bound_method => |mi| try self.keys.append(self.gpa, .{
                    .kind = Closure.boundKind(Closure.boundRecvLocal(self.tree, self.resolutions, e.idx), Closure.heapStorage(self.body)),
                    .key = mi,
                }),
                .none, .method, .protocol, .builtin, .value, .fn_ref, .direct, .instance, .generic => {},
            }
        }
    };
    var v: V = .{ .gpa = gpa, .tree = mc.tree, .resolutions = mc.resolutions, .body = b, .keys = keys };
    try Walk.walk(.{ .tree = mc.tree, .tokens = mc.tokens, .source = mc.source }, b.decl, &v);
}

/// The lambda-unit table: for each body in order, one unit per lambda or local fn in it, in
/// node order. Each unit's signature is the reified pair the lambda's node types to, and
/// its env is laid out here once, for both the body that makes the closure and the unit
/// that reads it. Serial over the finished tables, so it is identical at any `-j`.
pub fn buildLambdaUnits(
    gpa: std.mem.Allocator,
    graph: *const Typecheck.GraphCtx,
    bodies: Body.List,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
) ![]Closure.LambdaUnit {
    var units: std.ArrayList(Closure.LambdaUnit) = .empty;
    errdefer {
        for (units.items) |u| Closure.freeLambdaUnit(gpa, u);
        units.deinit(gpa);
    }
    for (bodies.items, 0..) |b, i| {
        const mc = &graph.mods[b.mod];
        const fc = Closure.findFn(mc.closures, b.decl) orelse continue;
        try appendLambdaUnits(gpa, &units, @intCast(i), b, fc, mc, layouts, enum_layouts);
    }
    return units.toOwnedSlice(gpa);
}

/// Whether global fn `f` is a generic user fn with a body: a template, lowered only
/// through its instances, of which a program may make none.
fn isTemplateOwner(f: Typecheck.GraphFnInput, fsym: Typecheck.FnSym) bool {
    return f.decl_node != Ast.none and f.kind == .user_fn and fsym.isGeneric();
}

/// The owner-closure table: each body with a boxed local, in body order, its unboxable
/// slots. Serial over the finished tables, so it is identical at any `-j`.
pub fn buildOwnerClosures(gpa: std.mem.Allocator, bodies: Body.List, unboxable: []const []const u32) ![]Closure.OwnerClosure {
    var out: std.ArrayList(Closure.OwnerClosure) = .empty;
    errdefer {
        for (out.items) |oc| gpa.free(oc.unboxable);
        out.deinit(gpa);
    }
    for (bodies.items, unboxable, 0..) |b, ub, i| {
        if (b.boxed.len == 0) continue;
        try out.ensureUnusedCapacity(gpa, 1);
        out.appendAssumeCapacity(.{ .owner = @intCast(i), .unboxable = try gpa.dupe(u32, ub) });
    }
    return out.toOwnedSlice(gpa);
}

/// The escape pass's results for the closure tables: each body's unboxable slots (indexed
/// like the body list, empty for most), and each lambda unit's frame-env eligibility
/// (indexed like the unit table).
const EscapeTables = struct {
    unboxable: [][]u32,
    frame: []bool,

    pub fn deinit(e: *EscapeTables, gpa: std.mem.Allocator) void {
        for (e.unboxable) |u| gpa.free(u);
        gpa.free(e.unboxable);
        gpa.free(e.frame);
        e.* = undefined;
    }
};

/// One escape unit: a body, or a generic template read for T0052 alone.
const EscapeUnit = struct {
    mod: u32,
    decl: Ast.Index,
    /// Its `mut` params, `mut self` first.
    modes: u64,
    params: []const Type,
    node_types: []const Type,
    callees: []const Callee,
};

/// The escape pass, serial over the finished tables in body order (so identical at any
/// `-j`) and run at every opt level, since it decides legality: a closure that escapes
/// cannot capture a `mut` param or `mut self` (T0052, at the first such capture in rank
/// order), and a bound method value on one cannot escape either. It also finds the boxed
/// slots that only non-escaping closures capture and the lambdas whose env may live in a
/// frame (non-escaping, within the owner's budget). Units `[0, bodies)` are the bodies;
/// the templates follow, and no call targets one.
pub fn analyzeEscapes(
    t: *Typecheck,
    fns: []const Typecheck.GraphFnInput,
    sigs: []const Sig,
    node_types: []const []const Type,
    callees: []const []const Callee,
    bodies: Body.List,
    instances: []const Mono.Instance,
    methods: []const Typecheck.Method,
    lambda_units: []const Closure.LambdaUnit,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
) !EscapeTables {
    const gpa = t.gpa;
    var units: std.ArrayList(EscapeUnit) = .empty;
    defer units.deinit(gpa);
    for (bodies.items) |b| try units.append(gpa, .{ .mod = b.mod, .decl = b.decl, .modes = b.sig.mut_mask, .params = b.sig.params, .node_types = b.node_types, .callees = b.callees });
    for (fns, t.fns.items, 0..) |f, fs, gid| {
        if (!isTemplateOwner(f, fs)) continue;
        try units.append(gpa, .{ .mod = f.module, .decl = f.decl_node, .modes = fs.mut_mask, .params = sigs[gid].params, .node_types = node_types[f.module], .callees = callees[f.module] });
    }
    const n_bodies = bodies.items.len;
    const prog: Body.ProgramView = .{ .sigs = sigs, .methods = methods, .instances = instances, .layouts = layouts };

    const facts = try gpa.alloc(?Escape.Facts, units.items.len);
    @memset(facts, null);
    defer {
        for (facts) |*f| if (f.*) |*ff| ff.deinit(gpa);
        gpa.free(facts);
    }
    for (units.items, 0..) |eu, u| {
        const template = u >= n_bodies;
        const mc = &t.graph.mods[eu.mod];
        // Only a body with a closure, a param a closure can be passed to, or a `mut` param a
        // bound method value can hold needs facts.
        const fc = Closure.findFn(mc.closures, eu.decl);
        const has_mut = eu.modes != 0;
        // A template is read for T0052 alone, which needs a closure and a `mut` param.
        if (template and (fc == null or !has_mut)) continue;
        const takes_fn = for (eu.params) |p| {
            if (LayoutEngine.isFnStruct(p, layouts)) break true;
        } else false;
        if (fc == null and !takes_fn and !has_mut) continue;
        const proto = Ast.protoAt(mc.tree, mc.tree.nodes[eu.decl.int()].lhs.int());
        const body: Body.BodyView = .{ .tree = mc.tree, .resolutions = mc.resolutions, .node_types = eu.node_types, .callees = eu.callees };
        facts[u] = try Escape.factsOf(gpa, .{ .tree = mc.tree, .tokens = mc.tokens, .source = mc.source }, body, prog, bodies, eu.decl, @intCast(proto.params.len), fc, if (template) .template else .lowered);
    }
    var sol = try Escape.solve(gpa, facts);
    defer sol.deinit(gpa);

    const unboxable = try gpa.alloc([]u32, n_bodies);
    for (unboxable) |*x| x.* = &.{};
    errdefer {
        for (unboxable) |x| gpa.free(x);
        gpa.free(unboxable);
    }
    const frame = try gpa.alloc(bool, lambda_units.len);
    errdefer gpa.free(frame);
    @memset(frame, false);

    var reported: std.ArrayList(u32) = .empty;
    defer reported.deinit(gpa);
    var escaping_slots: std.ArrayList(u32) = .empty;
    defer escaping_slots.deinit(gpa);
    var ui: usize = 0;
    for (units.items, 0..) |eu, u| {
        const f = facts[u] orelse continue;
        const mc = &t.graph.mods[eu.mod];
        reported.clearRetainingCapacity();
        escaping_slots.clearRetainingCapacity();
        const fc = Closure.findFn(mc.closures, eu.decl);
        const by_rank = try gpa.alloc(u32, if (fc) |c| c.boundaries.len else 0);
        defer gpa.free(by_rank);
        if (fc) |c| for (c.boundaries, 0..) |b, i| {
            by_rank[b.rank] = @intCast(i);
        };
        for (by_rank) |bi| {
            const b = fc.?.boundaries[bi];
            const use = f.siteUse(b.node) orelse .escapes;
            if (!sol.siteEscapes(f, @intCast(u), use)) continue;
            for (b.captures) |c| {
                try escaping_slots.append(gpa, c.slot);
                if (!Modes.isMut(eu.modes, c.slot) or std.mem.indexOfScalar(u32, reported.items, c.slot) != null) continue;
                try reported.append(gpa, c.slot);
                try reportEscapingCapture(t, mc, eu.mod, c.first_use);
            }
        }
        // A bound value on a `mut` param holds its address too: in node order, its first
        // escaping one, unless a closure already reported the param.
        for (f.sites) |site| {
            const p = site.recv orelse continue;
            if (site.kind != .bound or !Modes.isMut(eu.modes, p) or std.mem.indexOfScalar(u32, reported.items, p) != null) continue;
            if (!sol.siteEscapes(f, @intCast(u), site.use)) continue;
            try reported.append(gpa, p);
            try reportEscapingCapture(t, mc, eu.mod, mc.tree.nodes[site.node.int()].lhs);
        }
        // A template has no storage of its own to decide: only its instances are lowered.
        if (u >= n_bodies) continue;
        const fcl = fc orelse continue;

        // This owner's units are one run of the (owner, node)-sorted table.
        const owner: Closure.Owner = @intCast(u);
        while (ui < lambda_units.len and lambda_units[ui].owner < owner) ui += 1;
        const lo = ui;
        while (ui < lambda_units.len and lambda_units[ui].owner == owner) ui += 1;
        const run = lambda_units[lo..ui];

        var ub: std.ArrayList(u32) = .empty;
        errdefer ub.deinit(gpa);
        for (bodies.items[u].boxed) |sl| {
            if (std.mem.indexOfScalar(u32, escaping_slots.items, sl) != null) continue;
            // A bound method value's receiver keeps its cell: its adapter (`boundKind`) is
            // minted once from the heap storage, for every opt level.
            if (Closure.contains(f.bound_recv, sl)) continue;
            // A slot of 16 B or less never makes the frame bigger than at -O0: it adds at
            // most 8 slot bytes and drops the 16 bytes of values its box costs, so only
            // frame envs can overflow a frame, and the fallback reverts exactly those.
            const ty = find: for (run) |lu| {
                for (lu.env) |es| if (es.slot == sl) break :find es.ty;
            } else continue;
            if (std.mem.alignForward(u32, Abi.typeSize(ty, layouts, enum_layouts), 8) > Escape.unbox_max) continue;
            try ub.append(gpa, sl);
        }
        unboxable[u] = try ub.toOwnedSlice(gpa);

        var budget: u32 = Escape.frame_budget;
        for (by_rank) |bi| {
            const b = fcl.boundaries[bi];
            const k = Closure.findUnit(run, owner, b.node) orelse continue;
            const lu = run[k];
            if (lu.env_size == 0 or lu.env_size > budget) continue;
            if (sol.siteEscapes(f, @intCast(u), f.siteUse(b.node) orelse .escapes)) continue;
            frame[lo + k] = true;
            budget -= lu.env_size;
        }
    }
    t.sink.sort();
    return .{ .unboxable = unboxable, .frame = frame };
}

/// T0052 for the capture of a `mut` param (or `mut self`) at `use`, by a closure that
/// escapes. Reported in the owner's module scope, since the pass runs after every body.
fn reportEscapingCapture(t: *Typecheck, mc: *const Typecheck.GraphCtx.ModuleCtx, mod: u32, use: Ast.Index) error{OutOfMemory}!void {
    const tok = mc.tree.nodes[use.int()].main_token;
    const name = mc.tokens[tok].text(mc.source);
    const what = if (std.mem.eql(u8, name, "self")) "the 'mut self' receiver" else "the 'mut' parameter";
    try t.sink.report(.{ .span = .{ .start = mc.tokens[tok].start, .end = mc.tokens[tok].end }, .code = .T0052, .scope = mod }, "a closure that escapes cannot capture {s} '{s}'; copy it to a local first", .{ what, name });
}

/// Append one env struct per frame-env lambda unit to `layouts` (after every existing id,
/// so no id moves) and point the unit at it. Returns the grown table, which replaces
/// `layouts` (freed) only on success.
pub fn appendEnvLayouts(
    gpa: std.mem.Allocator,
    layouts: []Layout,
    lambda_units: []Closure.LambdaUnit,
    frame: []const bool,
    bodies: Body.List,
    owner_closures: []const Closure.OwnerClosure,
) ![]Layout {
    var envs: std.ArrayList(Layout) = .empty;
    defer envs.deinit(gpa);
    errdefer for (envs.items) |l| freeEnvStruct(gpa, l);
    for (lambda_units, frame) |lu, s| {
        if (!s) continue;
        try envs.ensureUnusedCapacity(gpa, 1);
        envs.appendAssumeCapacity(try envStruct(gpa, lu, Closure.ownerStorage(bodies, owner_closures, lu.owner, true)));
    }
    if (envs.items.len == 0) return layouts;
    const out = try gpa.alloc(Layout, layouts.len + envs.items.len);
    @memcpy(out[0..layouts.len], layouts);
    @memcpy(out[layouts.len..], envs.items);
    var id: u32 = @intCast(layouts.len);
    for (lambda_units, frame) |*lu, s| if (s) {
        lu.frame_env = id;
        id += 1;
    };
    gpa.free(layouts);
    envs.clearRetainingCapacity();
    return out;
}

fn appendLambdaUnits(
    gpa: std.mem.Allocator,
    units: *std.ArrayList(Closure.LambdaUnit),
    owner: Closure.Owner,
    b: Body,
    fc: *const Closure.FnClosures,
    mc: *const Typecheck.GraphCtx.ModuleCtx,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
) !void {
    for (fc.boundaries) |bd| {
        const t = b.node_types[bd.node.int()];
        // Only a program with type errors leaves a lambda untyped, and it is never lowered.
        if (!LayoutEngine.isFnStruct(t, layouts)) continue;
        const fs = layouts[t.struct_id].fn_sig.?;
        const proto = Ast.protoAt(mc.tree, mc.tree.nodes[bd.node.int()].lhs.int());
        const name = try Mangle.lambda(gpa, b.sig.name, bd.rank);
        errdefer gpa.free(name);
        const params = try gpa.dupe(Type, fs.params);
        errdefer gpa.free(params);
        const slots = try gpa.alloc(u32, proto.params.len);
        errdefer gpa.free(slots);
        for (proto.params, slots) |p, *sl| sl.* = switch (mc.resolutions[p.int()]) {
            .local => |l| l,
            else => std.math.maxInt(u32),
        };
        const env = try gpa.alloc(Closure.EnvSlot, bd.captures.len);
        errdefer gpa.free(env);
        const env_size = try layoutCaptureEnv(gpa, bd.captures, b.node_types, Closure.heapStorage(b), layouts, enum_layouts, env);
        try units.append(gpa, .{
            .owner = owner,
            .mod = b.mod,
            .fn_node = fc.fn_node,
            .node = bd.node,
            .rank = bd.rank,
            .name = name,
            .sig = .{ .params = params, .ret = fs.ret },
            .param_slots = slots,
            .self_slot = bd.self_slot,
            .env = env,
            .env_size = env_size,
        });
    }
}

/// Fill `out` with one env slot per capture (ascending slot, as the resolver lists them),
/// typed as the owner typed the capture's first use, and return the env size. A capture the
/// owner's heap storage `st` holds by cell or address takes one 8-byte word, which frame
/// storage reuses for a frame address; a copy takes the value's size. A capture left
/// `.invalid` by an earlier error sizes as 0; such a program is never lowered.
fn layoutCaptureEnv(
    gpa: std.mem.Allocator,
    captures: []const Closure.Capture,
    node_types: []const Type,
    st: Closure.Storage,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
    out: []Closure.EnvSlot,
) !u32 {
    const sizes = try gpa.alloc(u32, captures.len);
    defer gpa.free(sizes);
    const aligns = try gpa.alloc(u32, captures.len);
    defer gpa.free(aligns);
    const offs = try gpa.alloc(u32, captures.len);
    defer gpa.free(offs);
    for (captures, sizes, aligns) |c, *sz, *al| {
        if (st.hold(c.slot) != .copy) {
            sz.* = 8;
            al.* = 8;
            continue;
        }
        const ty = node_types[c.first_use.int()];
        sz.* = Abi.typeSize(ty, layouts, enum_layouts);
        al.* = Abi.typeAlign(ty, layouts, enum_layouts);
    }
    const size = envLayout(sizes, aligns, offs);
    for (captures, offs, out) |c, off, *es| es.* = .{ .slot = c.slot, .ty = node_types[c.first_use.int()], .off = off };
    return size;
}

/// Every closure table names its owner by body index: a lambda unit sits in its owner's
/// decl and module, and owners ascend through both tables as the bodies do.
pub fn checkOwners(bodies: Body.List, lambda_units: []const Closure.LambdaUnit, owner_closures: []const Closure.OwnerClosure) void {
    for (bodies.items[0..bodies.n_base], 0..) |b, i| std.debug.assert(b.of == .base and (i == 0 or bodies.items[i - 1].of.base < b.of.base));
    for (bodies.items[bodies.n_base..], 0..) |b, k| std.debug.assert(b.of.instance == k);
    for (lambda_units, 0..) |u, i| {
        const b = bodies.items[u.owner];
        std.debug.assert(b.decl == u.fn_node and b.mod == u.mod);
        std.debug.assert(i == 0 or lambda_units[i - 1].owner <= u.owner);
    }
    for (owner_closures, 0..) |oc, i| {
        const st = Closure.ownerStorage(bodies, owner_closures, oc.owner, true);
        for (oc.unboxable) |sl| std.debug.assert(st.hold(sl) == .frame_addr);
        std.debug.assert(i == 0 or owner_closures[i - 1].owner < oc.owner);
    }
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

/// `unit`'s env as a struct, for a frame slot the precise root map can read: each copy is
/// typed as the capture, a cell pointer as `rawptr`, and an address as `int`, so a frame
/// address is never a root. `st` is its owner's frame storage.
pub fn envStruct(gpa: std.mem.Allocator, unit: Closure.LambdaUnit, st: Closure.Storage) error{OutOfMemory}!LayoutEngine.Layout {
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
        t.* = switch (st.hold(es.slot)) {
            .copy => es.ty,
            .cell => Type.rawptr,
            .frame_addr, .param_addr => Type.int,
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

const testing = std.testing;

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

test "envStruct types a copy as its capture, a cell as rawptr and an address as int" {
    const gpa = testing.allocator;
    const sig: LayoutEngine.FnSig = .{ .params = &.{}, .ret = Type.unit };
    const env = [_]Closure.EnvSlot{
        .{ .slot = 1, .ty = Type.str, .off = 0 },
        .{ .slot = 2, .ty = Type.int, .off = 16 },
        .{ .slot = 3, .ty = Type.int, .off = 24 },
        .{ .slot = 0, .ty = Type.int, .off = 32 },
    };
    const u: Closure.LambdaUnit = .{ .owner = 0, .mod = 0, .fn_node = Ast.Index.from(9), .node = Ast.Index.from(4), .rank = 0, .name = "m.f$l0", .sig = sig, .param_slots = &.{}, .env = &env, .env_size = 40 };
    const l = try envStruct(gpa, u, .{ .boxed = &.{ 2, 3 }, .frame_addrs = &.{3}, .places = 0b1 });
    defer freeEnvStruct(gpa, l);
    try testing.expectEqualStrings("env$m.f$l0", l.name);
    try testing.expectEqualSlices(Type, &.{ Type.str, Type.rawptr, Type.int, Type.int }, l.field_types);
    try testing.expectEqualSlices(u32, &.{ 0, 16, 24, 32 }, l.offsets);
    try testing.expectEqual(@as(u32, 40), l.size);
    try testing.expectEqual(@as(u32, 8), l.@"align");
}
