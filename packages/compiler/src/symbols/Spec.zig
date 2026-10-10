//! Hybrid specialization of generic calls. A call of a generic fn, or of a method of a
//! generic type, whose function-typed argument is known at compile time (a named fn or a
//! lambda literal) goes to a copy of the erased instance of its own, which calls that
//! argument directly; any other argument (a local holding a function, a field, a call
//! result) goes through the erased instance.
//! A peer data module like `Closure`: the driver builds the table (`build`) serially before
//! codegen, deciding each call's unit once, and lower and the fingerprint both look it up
//! (`Table.unitAt`), so the call a body makes and the identity its key folds cannot
//! disagree.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Sym = @import("Sym.zig");
const Type = @import("../layout/Type.zig").Type;
const Closure = @import("Closure.zig");
const Walk = @import("../ast/Walk.zig");
const Callee = @import("Callee.zig").Callee;
const Modes = @import("Modes.zig");
const Resolution = @import("Resolution.zig").Resolution;
const Body = @import("Body.zig");
const Mangle = @import("Mangle.zig");

pub const TargetKind = enum(u8) { named, lambda };

/// A fn-typed param of a spec unit and the code a call of it jumps to: a named fn (called
/// without an env) or a lambda unit (called with the env of the pair the param holds).
pub const Target = struct { param: u32, kind: TargetKind, sym: Sym.SymName };

/// A specializable call: the instance it calls, and how many params precede its first
/// argument (1 for a method's `self`, which is the receiver, not an argument).
pub const Site = struct { ii: usize, self_n: u32 };

/// One specialized copy of instance `inst`. `name` and `targets` are owned by the table;
/// each target's `sym.name` is borrowed from the fn names or the lambda-unit table.
pub const Unit = struct { inst: u32, name: []const u8, targets: []const Target };

/// Bit `p` set: param `p` of template `gid` may be specialized.
pub const TemplateMask = struct { gid: u32, eligible: u64 };

/// A call that goes to a spec unit: the `.call` node `call` in body `owner` (its lambdas
/// included) runs `units[unit]`.
pub const Call = struct { owner: Closure.Owner, call: Ast.Index, unit: u32 };

pub const Table = struct {
    /// Sorted by name.
    units: []const Unit = &.{},
    /// Sorted by gid; only templates with an eligible param.
    masks: []const TemplateMask = &.{},
    /// Every call that goes to a unit, sorted by `(owner, call)`: decided once, when the
    /// table is built, so lower and the fingerprint look a call's unit up.
    calls: []const Call = &.{},

    pub fn deinit(t: Table, gpa: std.mem.Allocator) void {
        for (t.units) |u| {
            gpa.free(u.name);
            gpa.free(u.targets);
        }
        gpa.free(t.units);
        gpa.free(t.masks);
        gpa.free(t.calls);
    }

    /// The spec unit call `call` in body `owner` goes to, or null for the erased instance.
    pub fn unitAt(t: Table, owner: ?Closure.Owner, call: Ast.Index) ?*const Unit {
        const o = owner orelse return null;
        const Key = struct { owner: u32, call: u32 };
        const i = std.sort.binarySearch(Call, t.calls, Key{ .owner = o, .call = call.int() }, struct {
            fn order(k: Key, c: Call) std.math.Order {
                return switch (std.math.order(k.owner, c.owner)) {
                    .eq => std.math.order(k.call, c.call.int()),
                    else => |ord| ord,
                };
            }
        }.order) orelse return null;
        return &t.units[t.calls[i].unit];
    }
};

/// Spec units per template, over all its instances. Past it a call uses the erased
/// instance; which sites keep theirs is decided by unit name, never by walk order.
pub const cap_per_template: u32 = 8;

/// The params of the template at `fn_decl` a call may specialize: typed as a function in
/// the template's own signature `params`, never assigned in its body nor passed to a `mut`
/// param (a spec unit assumes the param still holds the caller's argument at every call of
/// it), and below 64. A template with a lambda or local fn (`has_closures`) specializes
/// nothing, so every lambda unit belongs to exactly one lowered body.
pub fn eligibleMask(src: Walk.Source, body: Body.BodyView, prog: Body.ProgramView, fn_decl: Ast.Index, params: []const Type, has_closures: bool) u64 {
    if (has_closures) return 0;
    var fn_params: u64 = 0;
    for (params[0..@min(params.len, 64)], 0..) |p, i| {
        if (p.isFunc()) fn_params |= @as(u64, 1) << @intCast(i);
    }
    if (fn_params == 0) return 0;
    const V = struct {
        body: Body.BodyView,
        prog: Body.ProgramView,
        mask: u64,

        fn clearLocal(self: *@This(), node: Ast.Index) void {
            const tree = self.body.tree;
            if (tree.nodes[node.int()].tag != .identifier or node.int() >= self.body.resolutions.len) return;
            switch (self.body.resolutions[node.int()]) {
                .local => |slot| if (slot < 64) {
                    self.mask &= ~(@as(u64, 1) << @intCast(slot));
                },
                else => {},
            }
        }

        pub fn on(self: *@This(), ev: Walk.Event) void {
            switch (ev) {
                .enter => |e| if (e.tag == .assign) self.clearLocal(self.body.tree.nodes[e.idx.int()].lhs),
                .callee => |c| {
                    const arg_mut = Modes.argMask(self.body, self.prog, c.call);
                    if (arg_mut == 0) return;
                    const args = Ast.rangeSlice(self.body.tree, self.body.tree.nodes[c.call.int()].rhs.int());
                    for (args, 0..) |a, i| if (Modes.isMut(arg_mut, i)) self.clearLocal(a);
                },
                else => {},
            }
        }
    };
    var v: V = .{ .body = body, .prog = prog, .mask = fn_params };
    Walk.walk(src, fn_decl, &v) catch unreachable;
    return v.mask;
}

/// The lambda units a body's lambda literals lower to: those of its owner.
pub const Lambdas = struct { owner: ?Closure.Owner, units: []const Closure.LambdaUnit };

/// The targets of generic call `call` resolving to instance `ii`, ascending by param: for
/// each eligible param, an argument naming a fn (`callees[arg]` is `.fn_ref`) or a lambda
/// literal of the body's owner. A local fn is a local holding a function, like any other
/// variable, so it is not a target. Argument `a` is param `a + self_n`.
pub fn targetsOf(gpa: std.mem.Allocator, body: Body.BodyView, prog: Body.ProgramView, lambdas: Lambdas, call: Ast.Index, site: Site, eligible: u64, out: *std.ArrayList(Target)) error{OutOfMemory}!void {
    const tree = body.tree;
    const args = Ast.rangeSlice(tree, tree.nodes[call.int()].rhs.int());
    const nparams = prog.instances[site.ii].params.len;
    for (args, @as(usize, site.self_n)..) |arg, p| {
        if (p >= nparams or p >= 64) break;
        if (eligible & (@as(u64, 1) << @intCast(p)) == 0) continue;
        const an = tree.nodes[arg.int()];
        switch (an.tag) {
            .identifier, .field_access => {
                if (arg.int() >= body.callees.len) continue;
                switch (body.callees[arg.int()].view()) {
                    .fn_ref => |gid| if (gid < prog.names.len) try out.append(gpa, .{ .param = @intCast(p), .kind = .named, .sym = prog.names[gid] }),
                    .none, .method, .protocol, .builtin, .value, .bound_method, .direct, .instance, .generic => {},
                }
            },
            .lambda => {
                const owner = lambdas.owner orelse continue;
                const ui = Closure.findUnit(lambdas.units, owner, arg) orelse continue;
                try out.append(gpa, .{ .param = @intCast(p), .kind = .lambda, .sym = .{ .kind = .user_fn, .name = lambdas.units[ui].name } });
            },
            else => {},
        }
    }
}

/// Binary search over `t.units` by name.
pub fn find(t: Table, name: []const u8) ?*const Unit {
    const i = std.sort.binarySearch(Unit, t.units, name, struct {
        fn order(key: []const u8, u: Unit) std.math.Order {
            return std.mem.order(u8, key, u.name);
        }
    }.order) orelse return null;
    return &t.units[i];
}

/// The eligible params of template `gid`; 0 when it has none.
pub fn mask(t: Table, gid: u32) u64 {
    const i = std.sort.binarySearch(TemplateMask, t.masks, gid, struct {
        fn order(key: u32, m: TemplateMask) std.math.Order {
            return std.math.order(key, m.gid);
        }
    }.order) orelse return 0;
    return t.masks[i].eligible;
}

/// A spec unit some call site wants, keyed by its template for the cap.
pub const Candidate = struct { gid: u32, unit: Unit };

/// Record the unit a call of instance `ii` with `targets` would get, unless an equal one is
/// already in `out`, and return its index in `out`. `targets` is copied.
pub fn addCandidate(gpa: std.mem.Allocator, out: *std.ArrayList(Candidate), gid: u32, ii: usize, inst_name: []const u8, targets: []const Target) error{OutOfMemory}!u32 {
    const name = try Mangle.spec(gpa, inst_name, targets);
    errdefer gpa.free(name);
    for (out.items, 0..) |c, k| if (std.mem.eql(u8, c.unit.name, name)) {
        gpa.free(name);
        return @intCast(k);
    };
    const owned = try gpa.dupe(Target, targets);
    errdefer gpa.free(owned);
    try out.append(gpa, .{ .gid = gid, .unit = .{ .inst = @intCast(ii), .name = name, .targets = owned } });
    return @intCast(out.items.len - 1);
}

/// A call `collectCandidates` saw, by the candidate it wants (an index into the candidate
/// list).
pub const PendingCall = struct { owner: Closure.Owner, call: Ast.Index, cand: u32 };

/// The table from every candidate: per template, the first `cap_per_template` by name,
/// then all sorted by name, and each call's unit (a call whose candidate the cap dropped
/// takes the erased instance). Consumes `cands` and `masks` (both freed, on error too).
pub fn finish(gpa: std.mem.Allocator, cands: *std.ArrayList(Candidate), masks: []const TemplateMask, pending: []const PendingCall) error{OutOfMemory}!Table {
    defer cands.deinit(gpa);
    errdefer gpa.free(masks);
    errdefer freeUnits(gpa, cands.items);
    const n = cands.items.len;
    const order = try gpa.alloc(u32, n);
    defer gpa.free(order);
    for (order, 0..) |*o, k| o.* = @intCast(k);
    std.mem.sort(u32, order, cands.items, struct {
        fn lt(cs: []const Candidate, x: u32, y: u32) bool {
            const a = cs[x];
            const b = cs[y];
            if (a.gid != b.gid) return a.gid < b.gid;
            return std.mem.order(u8, a.unit.name, b.unit.name) == .lt;
        }
    }.lt);
    // Per candidate, its unit index once kept; maxInt while dropped.
    const unit_of = try gpa.alloc(u32, n);
    defer gpa.free(unit_of);
    @memset(unit_of, std.math.maxInt(u32));
    var kept: std.ArrayList(u32) = .empty;
    defer kept.deinit(gpa);
    var run_gid: u32 = 0;
    var run_len: u32 = 0;
    for (order, 0..) |k, i| {
        const c = cands.items[k];
        if (i == 0 or c.gid != run_gid) {
            run_gid = c.gid;
            run_len = 0;
        }
        if (run_len == cap_per_template) continue;
        run_len += 1;
        try kept.append(gpa, k);
    }
    std.mem.sort(u32, kept.items, cands.items, struct {
        fn lt(cs: []const Candidate, x: u32, y: u32) bool {
            return std.mem.order(u8, cs[x].unit.name, cs[y].unit.name) == .lt;
        }
    }.lt);
    for (kept.items, 0..) |k, ui| unit_of[k] = @intCast(ui);

    var calls: std.ArrayList(Call) = .empty;
    defer calls.deinit(gpa);
    for (pending) |pc| if (unit_of[pc.cand] != std.math.maxInt(u32))
        try calls.append(gpa, .{ .owner = pc.owner, .call = pc.call, .unit = unit_of[pc.cand] });
    std.mem.sort(Call, calls.items, {}, struct {
        fn lt(_: void, a: Call, b: Call) bool {
            if (a.owner != b.owner) return a.owner < b.owner;
            return a.call.int() < b.call.int();
        }
    }.lt);
    const calls_out = try calls.toOwnedSlice(gpa);
    errdefer gpa.free(calls_out);
    const units = try gpa.alloc(Unit, kept.items.len);
    for (kept.items, units) |k, *u| u.* = cands.items[k].unit;
    for (cands.items, unit_of) |c, ui| if (ui == std.math.maxInt(u32)) {
        gpa.free(c.unit.name);
        gpa.free(c.unit.targets);
    };
    cands.clearRetainingCapacity();
    return .{ .units = units, .masks = masks, .calls = calls_out };
}

/// Free the owned name and targets of each candidate's unit.
pub fn freeUnits(gpa: std.mem.Allocator, cands: []const Candidate) void {
    for (cands) |c| {
        gpa.free(c.unit.name);
        gpa.free(c.unit.targets);
    }
}

/// What `build` reads: per-module tables by module id, the bodies, and the program-wide
/// tables.
pub const Source = struct {
    /// Each module's tree, tokens and source.
    modules: []const Walk.Source,
    resolutions: []const []const Resolution,
    closures: []const []const Closure.FnClosures,
    bodies: Body.List,
    prog: Body.ProgramView,
    lambda_units: []const Closure.LambdaUnit,

    fn view(src: Source, b: Body) Body.BodyView {
        return b.view(src.modules[b.mod].tree, src.resolutions[b.mod]);
    }
};

/// The spec table: a mask per template with an eligible param, and a unit per distinct
/// generic call site (in any body, its lambdas included) whose argument for an eligible
/// param is a named fn or a lambda literal. Serial, and the cap and the unit order go by
/// name, so it is identical at any `-j`.
pub fn build(gpa: std.mem.Allocator, src: Source) error{OutOfMemory}!Table {
    const masks = try templateMasks(gpa, src);
    if (masks.len == 0) {
        gpa.free(masks);
        return .{};
    }
    var cands: std.ArrayList(Candidate) = .empty;
    var pending: std.ArrayList(PendingCall) = .empty;
    defer pending.deinit(gpa);
    collectCandidates(gpa, src, masks, &cands, &pending) catch |e| {
        freeUnits(gpa, cands.items);
        cands.deinit(gpa);
        gpa.free(masks);
        return e;
    };
    return finish(gpa, &cands, masks, pending.items);
}

/// The site of generic call `call`, from its callee: a generic fn's instance, or a method
/// of one of a generic type's instances (`b.apply(f)` on a `Box[int]`). Null for any other
/// call.
pub fn siteOf(body: Body.BodyView, prog: Body.ProgramView, call: Ast.Index) ?Site {
    const slot = Callee.callSlot(body.tree, body.tree.nodes[call.int()]) orelse return null;
    if (slot.int() >= body.callees.len) return null;
    return switch (body.callees[slot.int()].view()) {
        .instance => |ii| .{ .ii = ii, .self_n = 0 },
        .method => |mi| {
            if (mi >= prog.methods.len) return null;
            const m = prog.methods[mi];
            return .{ .ii = m.instance orelse return null, .self_n = @intFromBool(!m.is_static) };
        },
        else => null,
    };
}

/// The eligible params of each instantiated template that has one, sorted by gid. Every
/// instance of a template shares its body, so its first instance stands for all of them.
fn templateMasks(gpa: std.mem.Allocator, src: Source) error{OutOfMemory}![]TemplateMask {
    var masks: std.ArrayList(TemplateMask) = .empty;
    errdefer masks.deinit(gpa);
    var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer seen.deinit(gpa);
    for (src.prog.instances, 0..) |inst, k| {
        const gid = inst.template_gid;
        if ((try seen.getOrPut(gpa, gid)).found_existing) continue;
        if (gid >= src.prog.sigs.len) continue;
        const fcs: []const Closure.FnClosures = if (inst.mod < src.closures.len) src.closures[inst.mod] else &.{};
        const b = src.bodies.items[src.bodies.ofInstance(@intCast(k))];
        const m = eligibleMask(src.modules[inst.mod], src.view(b), src.prog, inst.decl_node, src.prog.sigs[gid].params, Closure.findFn(fcs, inst.decl_node) != null);
        if (m != 0) try masks.append(gpa, .{ .gid = gid, .eligible = m });
    }
    std.mem.sort(TemplateMask, masks.items, {}, struct {
        fn lt(_: void, a: TemplateMask, b: TemplateMask) bool {
            return a.gid < b.gid;
        }
    }.lt);
    return masks.toOwnedSlice(gpa);
}

fn collectCandidates(gpa: std.mem.Allocator, src: Source, masks: []const TemplateMask, cands: *std.ArrayList(Candidate), pending: *std.ArrayList(PendingCall)) error{OutOfMemory}!void {
    const V = struct {
        gpa: std.mem.Allocator,
        body: Body.BodyView,
        prog: Body.ProgramView,
        lambdas: Lambdas,
        masks: Table,
        cands: *std.ArrayList(Candidate),
        pending: *std.ArrayList(PendingCall),
        targets: std.ArrayList(Target) = .empty,

        pub fn on(self: *@This(), ev: Walk.Event) error{OutOfMemory}!void {
            const e = switch (ev) {
                .enter => |e| e,
                else => return,
            };
            if (e.tag != .call or self.body.tree.nodes[e.idx.int()].lhs == Ast.none) return;
            const site = siteOf(self.body, self.prog, e.idx) orelse return;
            const inst = self.prog.instances[site.ii];
            const m = mask(self.masks, inst.template_gid);
            if (m == 0) return;
            self.targets.clearRetainingCapacity();
            try targetsOf(self.gpa, self.body, self.prog, self.lambdas, e.idx, site, m, &self.targets);
            if (self.targets.items.len == 0) return;
            const cand = try addCandidate(self.gpa, self.cands, inst.template_gid, site.ii, inst.name.?, self.targets.items);
            try self.pending.append(self.gpa, .{ .owner = self.lambdas.owner.?, .call = e.idx, .cand = cand });
        }
    };
    for (src.bodies.items, 0..) |b, i| {
        var v: V = .{
            .gpa = gpa,
            .body = src.view(b),
            .prog = src.prog,
            .lambdas = .{ .owner = @intCast(i), .units = src.lambda_units },
            .masks = .{ .masks = masks },
            .cands = cands,
            .pending = pending,
        };
        defer v.targets.deinit(gpa);
        try Walk.walk(src.modules[b.mod], b.decl, &v);
    }
}

const testing = std.testing;

test "finish caps each template by name and sorts the units by name" {
    const gpa = testing.allocator;
    var cands: std.ArrayList(Candidate) = .empty;
    const lam_names = [_][]const u8{ "m$l9", "m$l3", "m$l0", "m$l7", "m$l1", "m$l8", "m$l2", "m$l6", "m$l5", "m$l4" };
    for (lam_names) |ln| _ = try addCandidate(gpa, &cands, 4, 0, "g$int", &.{.{ .param = 0, .kind = .lambda, .sym = .{ .kind = .user_fn, .name = ln } }});
    // A repeat of an earlier site adds nothing.
    try testing.expectEqual(@as(u32, 1), try addCandidate(gpa, &cands, 4, 0, "g$int", &.{.{ .param = 0, .kind = .lambda, .sym = .{ .kind = .user_fn, .name = "m$l3" } }}));
    _ = try addCandidate(gpa, &cands, 2, 1, "a$int", &.{.{ .param = 0, .kind = .named, .sym = .{ .kind = .user_fn, .name = "m.f" } }});
    try testing.expectEqual(@as(usize, 11), cands.items.len);
    const masks = try gpa.dupe(TemplateMask, &.{ .{ .gid = 2, .eligible = 1 }, .{ .gid = 4, .eligible = 1 } });
    // Calls by the candidate they want: `m$l7` (kept), `m$l9` (cut by the cap), `m.f`.
    const pending = [_]PendingCall{
        .{ .owner = 3, .call = Ast.Index.from(20), .cand = 3 },
        .{ .owner = 1, .call = Ast.Index.from(9), .cand = 0 },
        .{ .owner = 1, .call = Ast.Index.from(4), .cand = 10 },
    };
    const t = try finish(gpa, &cands, masks, &pending);
    defer t.deinit(gpa);
    try testing.expectEqual(@as(usize, 1 + cap_per_template), t.units.len);
    for (t.units[1..], t.units[0 .. t.units.len - 1]) |cur, prev| try testing.expect(std.mem.order(u8, prev.name, cur.name) == .lt);
    try testing.expect(find(t, "a$int$sp0$m.f") != null);
    try testing.expect(find(t, "g$int$sp0$m$l7") != null);
    try testing.expect(find(t, "g$int$sp0$m$l8") == null);
    try testing.expect(find(t, "g$int$sp0$m$l9") == null);
    try testing.expectEqual(@as(u64, 1), mask(t, 4));
    try testing.expectEqual(@as(u64, 0), mask(t, 3));
    try testing.expectEqualStrings("g$int$sp0$m$l7", t.unitAt(3, Ast.Index.from(20)).?.name);
    try testing.expectEqual(@as(?*const Unit, null), t.unitAt(1, Ast.Index.from(9)));
    try testing.expectEqualStrings("a$int$sp0$m.f", t.unitAt(1, Ast.Index.from(4)).?.name);
    try testing.expectEqual(@as(?*const Unit, null), t.unitAt(null, Ast.Index.from(4)));
    try testing.expectEqual(@as(usize, 2), t.calls.len);
}

/// One module resolved and typechecked, with `names` in global fn-id order, so the spec
/// helpers read the checker's real tables.
const Checked = struct {
    const Token = @import("../ast/Token.zig").Token;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");

    src: []const u8,
    tokens: []Token,
    tree: Ast.Tree,
    g: Graph.Graph,
    res: ResolveGraph.GraphResult,
    tc: TypecheckGraph.GraphResult,
    names: []Sym.SymName,

    fn create(gpa: std.mem.Allocator, src: []const u8) !*Checked {
        const self = try gpa.create(Checked);
        errdefer gpa.destroy(self);
        self.src = src;
        self.tokens = try @import("../lex.zig").tokenize(gpa, src);
        errdefer gpa.free(self.tokens);
        self.tree = try @import("../parse.zig").expectTree(gpa, self.tokens, src);
        errdefer {
            gpa.free(self.tree.nodes);
            gpa.free(self.tree.extra);
        }
        self.g = try Graph.single(gpa, "main", "", src, self.tokens, self.tree.nodes, self.tree.extra, self.tree.pub_bits);
        errdefer self.g.deinit(gpa);
        self.res = try ResolveGraph.resolveGraph(gpa, &self.g);
        errdefer self.res.deinit(gpa);
        self.tc = try TypecheckGraph.checkGraph(gpa, &self.g, &self.res, null, 0);
        errdefer self.tc.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), self.tc.diags.len);
        self.names = try gpa.alloc(Sym.SymName, self.res.fns.len);
        for (self.res.fns, 0..) |gf, i| {
            const kind: Sym.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
            self.names[i] = .{ .kind = kind, .name = gf.name };
        }
        return self;
    }

    fn destroy(self: *Checked, gpa: std.mem.Allocator) void {
        gpa.free(self.names);
        self.tc.deinit(gpa);
        self.res.deinit(gpa);
        self.g.deinit(gpa);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
        gpa.free(self.tokens);
        gpa.destroy(self);
    }

    fn gid(self: *const Checked, name: []const u8) !u32 {
        for (self.res.fns, 0..) |gf, i| if (std.mem.eql(u8, gf.name, name)) return @intCast(i);
        return error.TestExpectedFn;
    }

    /// The eligible mask of template `name`, read over its module's tables.
    fn templateMask(self: *const Checked, name: []const u8) !u64 {
        const g = try self.gid(name);
        const f = self.res.fns[g];
        const has_closures = Closure.findFn(self.res.closures[f.module], f.decl_node) != null;
        return eligibleMask(.{ .tree = self.tree, .tokens = self.tokens, .source = self.src }, self.body(), self.prog(), f.decl_node, self.tc.sigs[g].params, has_closures);
    }

    fn body(self: *const Checked) Body.BodyView {
        return .{ .tree = self.tree, .resolutions = self.res.resolutions[0], .node_types = self.tc.node_types[0], .callees = self.tc.callees[0] };
    }

    fn prog(self: *const Checked) Body.ProgramView {
        return .{ .sigs = self.tc.sigs, .methods = self.tc.methods, .instances = self.tc.instances, .names = self.names, .layouts = self.tc.layouts };
    }

    /// The lambda units of fn `owner`'s body.
    fn lambdasOf(self: *const Checked, owner: []const u8) !Lambdas {
        return .{ .owner = self.tc.bodies.ofFn(try self.gid(owner)).?, .units = self.tc.lambda_units };
    }

    /// The targets of every call in fn `owner` of an instance of `template`, all params
    /// eligible, read in `owner`'s body.
    fn targetsIn(self: *const Checked, gpa: std.mem.Allocator, owner: []const u8, template: []const u8, out: *std.ArrayList(Target)) !void {
        const tg = try self.gid(template);
        const lambdas = try self.lambdasOf(owner);
        for (self.tree.nodes, 0..) |n, i| {
            if (n.tag != .call) continue;
            const callee = self.tree.nodes[n.lhs.int()];
            if (callee.tag != .identifier or self.res.resolutions[0][n.lhs.int()] != .func) continue;
            if (self.res.resolutions[0][n.lhs.int()].func != tg) continue;
            const ii = for (self.tc.instances, 0..) |inst, k| {
                if (inst.template_gid == tg) break k;
            } else return error.TestExpectedInstance;
            try targetsOf(gpa, self.body(), self.prog(), lambdas, Ast.Index.from(@intCast(i)), .{ .ii = ii, .self_n = 0 }, ~@as(u64, 0), out);
        }
    }
};

test "an assigned fn param is ineligible" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\fn inc(x: int) -> int { x + 1 }
        \\fn keep[T](f: fn(T) -> T, g: fn(T) -> T, x: T) -> T { g(f(x)) }
        \\fn swap[T](f: fn(T) -> T, g: fn(T) -> T, x: T) -> T {
        \\    g = f
        \\    g(x)
        \\}
        \\fn main() -> int { keep(inc, inc, 1) + swap(inc, inc, 1) }
        \\
    );
    defer fx.destroy(gpa);
    try testing.expectEqual(@as(u64, 0b011), try fx.templateMask("main.keep"));
    try testing.expectEqual(@as(u64, 0b001), try fx.templateMask("main.swap"));
}

test "a fn-typed param passed to a mut param is not eligible" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\fn inc(x: int) -> int { x + 1 }
        \\fn dec(x: int) -> int { x - 1 }
        \\fn reset(mut g: fn(int) -> int) { g = dec }
        \\fn run[T](f: fn(int) -> int, x: T) -> T {
        \\    reset(f)
        \\    _y := f(1)
        \\    x
        \\}
        \\fn keep[T](f: fn(int) -> int, x: T) -> T {
        \\    _y := f(1)
        \\    x
        \\}
        \\fn main() -> int { run(inc, 1) + keep(inc, 1) }
        \\
    );
    defer fx.destroy(gpa);
    try testing.expectEqual(@as(u64, 0), try fx.templateMask("main.run"));
    try testing.expectEqual(@as(u64, 1), try fx.templateMask("main.keep"));
}

test "a template containing a lambda is ineligible" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\fn inc(x: int) -> int { x + 1 }
        \\fn wrap[T](f: fn(T) -> T, x: T) -> T {
        \\    g := fn(y: T) -> T { f(y) }
        \\    g(x)
        \\}
        \\fn plain[T](f: fn(T) -> T, x: T) -> T { f(x) }
        \\fn main() -> int { wrap(inc, 1) + plain(inc, 1) }
        \\
    );
    defer fx.destroy(gpa);
    try testing.expectEqual(@as(u64, 0), try fx.templateMask("main.wrap"));
    try testing.expectEqual(@as(u64, 1), try fx.templateMask("main.plain"));
}

test "a local fn value is not a target" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\fn inc(x: int) -> int { x + 1 }
        \\fn apply[T](f: fn(T) -> T, x: T) -> T { f(x) }
        \\fn main() -> int {
        \\    fn twice(x: int) -> int { x * 2 }
        \\    h := inc
        \\    apply(twice, 1) + apply(h, 2) + apply(inc, 3) + apply(fn(x: int) { x - 1 }, 4)
        \\}
        \\
    );
    defer fx.destroy(gpa);
    var targets: std.ArrayList(Target) = .empty;
    defer targets.deinit(gpa);
    try fx.targetsIn(gpa, "main", "main.apply", &targets);
    // Only the named fn and the lambda literal: the local fn and the local holding `inc`
    // are values.
    try testing.expectEqual(@as(usize, 2), targets.items.len);
    try testing.expectEqual(TargetKind.named, targets.items[0].kind);
    try testing.expectEqualStrings("main.inc", targets.items[0].sym.name);
    try testing.expectEqual(TargetKind.lambda, targets.items[1].kind);
    try testing.expectEqualStrings("main$l1", targets.items[1].sym.name);
}

test "a generic method's fn argument is a target at its param, after self and earlier params" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\    fn twice(self, _k: int, f: fn(T) -> T) -> T { f(f(self.v)) }
        \\}
        \\fn inc(x: int) -> int { x + 1 }
        \\fn main() -> int {
        \\    b := Box[int] { v: 4 }
        \\    b.twice(0, inc)
        \\}
        \\
    );
    defer fx.destroy(gpa);
    const body = fx.body();
    const prog = fx.prog();
    const lambdas = try fx.lambdasOf("main");
    var sites: usize = 0;
    for (fx.tree.nodes, 0..) |n, i| {
        if (n.tag != .call) continue;
        const call = Ast.Index.from(@intCast(i));
        const site = siteOf(body, prog, call) orelse continue;
        sites += 1;
        try testing.expectEqual(@as(u32, 1), site.self_n);
        try testing.expectEqualStrings("main.Box.twice$int", fx.tc.instances[site.ii].name.?);
        var targets: std.ArrayList(Target) = .empty;
        defer targets.deinit(gpa);
        try targetsOf(gpa, body, prog, lambdas, call, site, ~@as(u64, 0), &targets);
        try testing.expectEqual(@as(usize, 1), targets.items.len);
        try testing.expectEqual(@as(u32, 2), targets.items[0].param);
        try testing.expectEqualStrings("main.inc", targets.items[0].sym.name);
    }
    try testing.expectEqual(@as(usize, 1), sites);
    // The template's mask names the same param.
    try testing.expectEqual(@as(u64, 0b100), try fx.templateMask("main.Box.twice"));
}
