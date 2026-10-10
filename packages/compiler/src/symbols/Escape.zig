//! Escape analysis of function values. Which params of each lowered body may let a value
//! passed to them outlive the call, and which lambdas and local fns never outlive the body
//! that makes them. A peer data module like `Spec`: types runs it serially once every
//! body is checked, at every opt level, because it decides T0052; codegen reads only the
//! storage it allows, through the closure tables.
//!
//! A value escapes unless each use of it is a call of it, or passes it whole to a by-value
//! param of a known body whose summary says that param does not escape. Summaries are
//! solved bottom-up over the call graph's strongly connected components; an argument to a
//! body in the caller's own component escapes, so recursion never needs a fixpoint.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const AstWalk = @import("../query/AstWalk.zig");
const Callee = @import("Callee.zig").Callee;
const Closure = @import("Closure.zig");
const Modes = @import("Modes.zig");
const LayoutEngine = @import("../layout/Engine.zig");
const Body = @import("Body.zig");

/// The env bytes one owner may keep in frames, over all its lambdas; a unit past it keeps a
/// GC cell, so frame envs grow a frame by at most this much.
pub const frame_budget: u32 = 256;

/// The largest captured local (`roundUp8` of its size; a `str` is 16) that a lowering may
/// keep unboxed: no bigger than the cell pointer and box setup it replaces, so unboxing
/// never grows a frame.
pub const unbox_max: u32 = 16;

/// The body a call runs, by its index in the body list (its escape unit), and how many
/// params precede its first argument (1 for a method's receiver). A generic template read
/// for T0052 alone is a unit past the bodies, and no call targets one.
pub const Target = struct { unit: u32, self_n: u32 };

/// The body `call` runs, or null when it is not known here: a call of a function value, a
/// builtin, a protocol method, a derive, or a fn with no lowered body (an extern, an
/// intrinsic).
pub fn callTarget(gpa: std.mem.Allocator, body: Body.BodyView, prog: Body.ProgramView, bodies: Body.List, call: Ast.Index) error{OutOfMemory}!?Target {
    const tree = body.tree;
    const n = tree.nodes[call.int()];
    const slot = Callee.callSlot(tree, n) orelse return null;
    if (slot.int() >= body.callees.len) return null;
    switch (body.callees[slot.int()].view()) {
        .method => |mi| {
            if (mi >= prog.methods.len) return null;
            const m = prog.methods[mi];
            if (m.derive != null) return null;
            const self_n: u32 = @intFromBool(!m.is_static);
            if (m.instance) |ii| return .{ .unit = bodies.ofInstance(ii), .self_n = self_n };
            return .{ .unit = bodies.ofFn(m.fn_id) orelse return null, .self_n = self_n };
        },
        .none => {
            if (try AstWalk.instanceOfCall(gpa, body, prog, call)) |ii| return .{ .unit = bodies.ofInstance(@intCast(ii)), .self_n = 0 };
            const head_node = tree.nodes[n.lhs.int()];
            const head = if (head_node.tag == .type_app) head_node.lhs else n.lhs;
            if (head.int() >= body.resolutions.len) return null;
            return switch (body.resolutions[head.int()]) {
                .func => |gid| .{ .unit = bodies.ofFn(gid) orelse return null, .self_n = 0 },
                else => null,
            };
        },
        .protocol, .builtin, .value, .fn_ref, .bound_method => return null,
    }
}

/// A use that passes a value whole to `param` of body `unit`.
pub const Dep = struct { unit: u32, param: u32 };

/// How the closure a site makes is used.
pub const SiteUse = union(enum) {
    /// Called where it is made.
    called,
    /// Kept, returned, compared, or passed where the analysis cannot follow it.
    escapes,
    /// Passed whole to a known body's param.
    arg: Dep,
    /// Bound to this local (a `:=`, or a local fn's own name): its uses decide.
    local: u32,
};

/// A closure value a body makes, and how it is used.
pub const Site = struct {
    /// The `.lambda` or `.local_fn` node, or the `.field_access` of a bound method value.
    node: Ast.Index,
    use: SiteUse,
    kind: enum(u8) { lambda, local_fn, bound },
    /// `.bound`: the receiver's slot when it is a bare local.
    recv: ?u32 = null,
};

pub const SlotDep = struct { slot: u32, dep: Dep };

/// What one walk of a body records.
pub const Facts = struct {
    /// The body's own params: resolver slots `0..nparams`.
    nparams: u32 = 0,
    /// Bit `s`: local `s` is used where a value outlives the call (or is captured), whatever
    /// its callees do.
    escapes: std.DynamicBitSetUnmanaged = .{},
    /// Every whole-argument use of a local by a known body, sorted by slot.
    deps: []SlotDep = &.{},
    /// Every known body this one passes a function value to, in walk order (repeats
    /// allowed): the only call edges a closure can travel.
    calls: []u32 = &.{},
    /// Every lambda, local fn and bound method value, sorted by node.
    sites: []Site = &.{},
    /// The locals a bound method value holds as its receiver, ascending.
    bound_recv: []u32 = &.{},

    pub fn deinit(f: *Facts, gpa: std.mem.Allocator) void {
        f.escapes.deinit(gpa);
        gpa.free(f.deps);
        gpa.free(f.calls);
        gpa.free(f.sites);
        gpa.free(f.bound_recv);
        f.* = undefined;
    }

    /// The use of the closure made at `node`, or null when the walk saw no such site.
    pub fn siteUse(f: Facts, node: Ast.Index) ?SiteUse {
        const i = std.sort.binarySearch(Site, f.sites, node, struct {
            fn order(key: Ast.Index, s: Site) std.math.Order {
                return std.math.order(key.int(), s.node.int());
            }
        }.order) orelse return null;
        return f.sites[i].use;
    }

    fn depsOf(f: Facts, slot: u32) []const SlotDep {
        const lo = std.sort.lowerBound(SlotDep, f.deps, slot, struct {
            fn order(key: u32, d: SlotDep) std.math.Order {
                return std.math.order(key, d.slot);
            }
        }.order);
        var hi = lo;
        while (hi < f.deps.len and f.deps[hi].slot == slot) hi += 1;
        return f.deps[lo..hi];
    }
};

/// How `factsOf` reads a body.
pub const Mode = enum {
    /// A lowered body, with the types and callees its own check recorded: what may escape
    /// decides storage, so anything the walk cannot follow escapes.
    lowered,
    /// A generic template no instance may ever lower, read only to report T0052 once for
    /// every instance: its nodes have no types and its calls mostly no callee, so a value
    /// passed where the walk cannot follow it is taken not to escape. What escapes then
    /// escapes in every instance, so the report is never one an instance would not make.
    template,
};

/// Walk the body at `decl` once. `closures` is its lambdas as the resolver saw them. In a
/// lowered body only a call that passes a function value is resolved: no other argument can
/// carry a closure.
pub fn factsOf(
    gpa: std.mem.Allocator,
    src: AstWalk.Source,
    body: Body.BodyView,
    prog: Body.ProgramView,
    bodies: Body.List,
    decl: Ast.Index,
    nparams: u32,
    closures: ?*const Closure.FnClosures,
    mode: Mode,
) error{OutOfMemory}!Facts {
    // `opaque_arg`: an argument of a call a template's walk cannot follow.
    const Role = union(enum) { value_callee, arg: ?Dep, opaque_arg, init: u32 };
    const V = struct {
        gpa: std.mem.Allocator,
        body: Body.BodyView,
        prog: Body.ProgramView,
        bodies: Body.List,
        mode: Mode,
        roles: std.AutoHashMapUnmanaged(u32, Role) = .empty,
        escapes: std.DynamicBitSetUnmanaged = .{},
        deps: std.ArrayList(SlotDep) = .empty,
        calls: std.ArrayList(u32) = .empty,
        sites: std.ArrayList(Site) = .empty,
        bound_recv: std.ArrayList(u32) = .empty,

        fn local(self: *const @This(), node: Ast.Index) ?u32 {
            if (node.int() >= self.body.resolutions.len) return null;
            return switch (self.body.resolutions[node.int()]) {
                .local => |s| s,
                else => null,
            };
        }

        /// How the closure value made at `node` is used, from the position it sits in.
        fn roleUse(self: *const @This(), node: Ast.Index) SiteUse {
            const role = self.roles.get(node.int()) orelse return .escapes;
            return switch (role) {
                .value_callee, .opaque_arg => .called,
                .arg => |d| if (d) |dep| .{ .arg = dep } else .escapes,
                .init => |t| .{ .local = t },
            };
        }

        fn escape(self: *@This(), slot: u32) error{OutOfMemory}!void {
            if (slot >= self.escapes.bit_length) try self.escapes.resize(self.gpa, slot + 1, false);
            self.escapes.set(slot);
        }

        pub fn on(self: *@This(), ev: AstWalk.Event) error{OutOfMemory}!void {
            const e = switch (ev) {
                .enter => |e| e,
                else => return,
            };
            const tree = self.body.tree;
            const n = tree.nodes[e.idx.int()];
            switch (e.tag) {
                .call => {
                    if (n.lhs == Ast.none) return;
                    if (Callee.callSlot(tree, n)) |slot| if (slot.int() < self.body.callees.len and self.body.callees[slot.int()].view() == .value)
                        try self.roles.put(self.gpa, n.lhs.int(), .value_callee);
                    const args = Ast.rangeSlice(tree, n.rhs.int());
                    if (self.mode == .template) return self.templateCall(e.idx, n, args);
                    const passes_fn = for (args) |a| {
                        if (a.int() < self.body.node_types.len and LayoutEngine.isFnStruct(self.body.node_types[a.int()], self.prog.layouts)) break true;
                    } else false;
                    if (!passes_fn) return;
                    const target = try callTarget(self.gpa, self.body, self.prog, self.bodies, e.idx);
                    if (target) |t| try self.calls.append(self.gpa, t.unit);
                    const mask = Modes.argMask(self.body, self.prog, e.idx);
                    for (args, 0..) |a, i| {
                        const dep: ?Dep = if (target) |t| blk: {
                            const p = i + t.self_n;
                            if (Modes.isMut(mask, i) or p >= 64) break :blk null;
                            break :blk .{ .unit = t.unit, .param = @intCast(p) };
                        } else null;
                        try self.roles.put(self.gpa, a.int(), .{ .arg = dep });
                    }
                },
                .var_decl => if (self.local(e.idx)) |s| if (n.lhs != Ast.none) try self.roles.put(self.gpa, n.lhs.int(), .{ .init = s }),
                .identifier => if (self.local(e.idx)) |s| {
                    const role = self.roles.get(e.idx.int()) orelse return self.escape(s);
                    switch (role) {
                        .value_callee, .opaque_arg => {},
                        .arg => |d| if (d) |dep| try self.deps.append(self.gpa, .{ .slot = s, .dep = dep }) else try self.escape(s),
                        .init => try self.escape(s),
                    }
                },
                .lambda => try self.sites.append(self.gpa, .{ .node = e.idx, .use = self.roleUse(e.idx), .kind = .lambda }),
                .local_fn => try self.sites.append(self.gpa, .{ .node = e.idx, .use = if (self.local(e.idx)) |s| .{ .local = s } else .escapes, .kind = .local_fn }),
                .field_access => if (e.idx.int() < self.body.callees.len and self.body.callees[e.idx.int()].view() == .bound_method) {
                    const recv = Closure.boundRecvLocal(tree, self.body.resolutions, e.idx);
                    if (recv) |s| try self.bound_recv.append(self.gpa, s);
                    try self.sites.append(self.gpa, .{ .node = e.idx, .use = self.roleUse(e.idx), .kind = .bound, .recv = recv });
                },
                else => {},
            }
        }

        /// A template's call: its callee is known only when it names a lowered body
        /// directly, and a local in callee position is a function value it calls.
        fn templateCall(self: *@This(), call: Ast.Index, n: Ast.Node, args: []const Ast.Index) error{OutOfMemory}!void {
            if (self.local(n.lhs) != null) try self.roles.put(self.gpa, n.lhs.int(), .value_callee);
            const target = try callTarget(self.gpa, self.body, self.prog, self.bodies, call);
            if (target) |t| try self.calls.append(self.gpa, t.unit);
            const mask = Modes.argMask(self.body, self.prog, call);
            for (args, 0..) |a, i| {
                const role: Role = if (target) |t| blk: {
                    const p = i + t.self_n;
                    if (Modes.isMut(mask, i) or p >= 64) break :blk .{ .arg = null };
                    break :blk .{ .arg = .{ .unit = t.unit, .param = @intCast(p) } };
                } else .opaque_arg;
                try self.roles.put(self.gpa, a.int(), role);
            }
        }
    };
    var v: V = .{ .gpa = gpa, .body = body, .prog = prog, .bodies = bodies, .mode = mode };
    defer v.roles.deinit(gpa);
    errdefer {
        v.escapes.deinit(gpa);
        v.deps.deinit(gpa);
        v.calls.deinit(gpa);
        v.sites.deinit(gpa);
        v.bound_recv.deinit(gpa);
    }
    try AstWalk.walk(src, decl, &v);
    // A captured local is copied or shared into a closure the walk does not follow.
    if (closures) |fc| for (fc.boundaries) |b| for (b.captures) |c| try v.escape(c.slot);

    std.mem.sort(SlotDep, v.deps.items, {}, struct {
        fn lt(_: void, a: SlotDep, b: SlotDep) bool {
            return a.slot < b.slot;
        }
    }.lt);
    std.mem.sort(Site, v.sites.items, {}, struct {
        fn lt(_: void, a: Site, b: Site) bool {
            return a.node.int() < b.node.int();
        }
    }.lt);
    std.mem.sort(u32, v.bound_recv.items, {}, std.sort.asc(u32));
    var f: Facts = .{ .nparams = nparams, .escapes = v.escapes };
    v.escapes = .{};
    errdefer f.escapes.deinit(gpa);
    f.deps = try v.deps.toOwnedSlice(gpa);
    f.calls = try v.calls.toOwnedSlice(gpa);
    f.sites = try v.sites.toOwnedSlice(gpa);
    f.bound_recv = try v.bound_recv.toOwnedSlice(gpa);
    return f;
}

/// The solved summaries: bit `p` of `summary[u]` means param `p` of body `u` may escape.
pub const Solution = struct {
    summary: []u64,
    /// The strongly connected component of each body.
    scc: []u32,

    pub fn deinit(s: *Solution, gpa: std.mem.Allocator) void {
        gpa.free(s.summary);
        gpa.free(s.scc);
        s.* = undefined;
    }

    /// Whether a value body `caller` passes to `d` may escape.
    pub fn depEscapes(s: Solution, caller: u32, d: Dep) bool {
        if (d.param >= 64 or s.scc[d.unit] == std.math.maxInt(u32) or s.scc[d.unit] == s.scc[caller]) return true;
        return s.summary[d.unit] & (@as(u64, 1) << @intCast(d.param)) != 0;
    }

    /// Whether the value in local `slot` of body `u` may escape.
    pub fn slotEscapes(s: Solution, f: Facts, u: u32, slot: u32) bool {
        if (slot < f.escapes.bit_length and f.escapes.isSet(slot)) return true;
        for (f.depsOf(slot)) |d| if (s.depEscapes(u, d.dep)) return true;
        return false;
    }

    /// Whether the closure a site of body `u` makes may outlive it.
    pub fn siteEscapes(s: Solution, f: Facts, u: u32, use: SiteUse) bool {
        return switch (use) {
            .called => false,
            .escapes => true,
            .arg => |d| s.depEscapes(u, d),
            .local => |t| s.slotEscapes(f, u, t),
        };
    }
};

/// Solve every body's summary. `facts[u]` is null for a number that is no lowered body.
/// Tarjan's algorithm visits bodies in index order and calls in walk order, and finishes a
/// component only after every component it calls into, so each summary reads finished ones.
pub fn solve(gpa: std.mem.Allocator, facts: []const ?Facts) error{OutOfMemory}!Solution {
    const none = std.math.maxInt(u32);
    const summary = try gpa.alloc(u64, facts.len);
    errdefer gpa.free(summary);
    @memset(summary, 0);
    const scc = try gpa.alloc(u32, facts.len);
    errdefer gpa.free(scc);
    @memset(scc, none);
    const index = try gpa.alloc(u32, facts.len);
    defer gpa.free(index);
    @memset(index, none);
    const low = try gpa.alloc(u32, facts.len);
    defer gpa.free(low);
    var stack: std.ArrayList(u32) = .empty;
    defer stack.deinit(gpa);
    const Frame = struct { u: u32, next: u32 };
    var frames: std.ArrayList(Frame) = .empty;
    defer frames.deinit(gpa);
    var members: std.ArrayList(u32) = .empty;
    defer members.deinit(gpa);
    var counter: u32 = 0;
    var comps: u32 = 0;
    const sol: Solution = .{ .summary = summary, .scc = scc };

    for (facts, 0..) |root_f, root| {
        if (root_f == null or index[root] != none) continue;
        try frames.append(gpa, .{ .u = @intCast(root), .next = 0 });
        index[root] = counter;
        low[root] = counter;
        counter += 1;
        try stack.append(gpa, @intCast(root));
        while (frames.items.len > 0) {
            const top = &frames.items[frames.items.len - 1];
            const u = top.u;
            const calls = facts[u].?.calls;
            if (top.next < calls.len) {
                const w = calls[top.next];
                top.next += 1;
                if (w >= facts.len or facts[w] == null) continue;
                if (index[w] == none) {
                    index[w] = counter;
                    low[w] = counter;
                    counter += 1;
                    try stack.append(gpa, w);
                    try frames.append(gpa, .{ .u = w, .next = 0 });
                } else if (scc[w] == none) {
                    low[u] = @min(low[u], index[w]);
                }
                continue;
            }
            _ = frames.pop();
            if (frames.items.len > 0) {
                const parent = frames.items[frames.items.len - 1].u;
                low[parent] = @min(low[parent], low[u]);
            }
            if (low[u] != index[u]) continue;
            members.clearRetainingCapacity();
            while (true) {
                const w = stack.pop().?;
                scc[w] = comps;
                try members.append(gpa, w);
                if (w == u) break;
            }
            comps += 1;
            for (members.items) |w| {
                const f = facts[w].?;
                var bits: u64 = 0;
                for (0..@min(f.nparams, 64)) |p| {
                    if (sol.slotEscapes(f, w, @intCast(p))) bits |= @as(u64, 1) << @intCast(p);
                }
                summary[w] = bits;
            }
        }
    }
    return sol;
}

const testing = std.testing;

fn testFacts(gpa: std.mem.Allocator, nparams: u32, escaping: []const u32, deps: []const SlotDep, calls: []const u32) !Facts {
    var f: Facts = .{ .nparams = nparams };
    try f.escapes.resize(gpa, 8, false);
    for (escaping) |s| f.escapes.set(s);
    f.deps = try gpa.dupe(SlotDep, deps);
    f.calls = try gpa.dupe(u32, calls);
    return f;
}

test "a param escapes through a callee's escaping param, not through a non-escaping one" {
    const gpa = testing.allocator;
    // 0 keeps its param 0; 1 only calls its param 0; 2 passes param 0 to 1 and param 1 to 0.
    var fs = [_]?Facts{
        try testFacts(gpa, 1, &.{0}, &.{}, &.{}),
        try testFacts(gpa, 1, &.{}, &.{}, &.{}),
        try testFacts(gpa, 2, &.{}, &.{ .{ .slot = 0, .dep = .{ .unit = 1, .param = 0 } }, .{ .slot = 1, .dep = .{ .unit = 0, .param = 0 } } }, &.{ 1, 0 }),
    };
    defer for (&fs) |*f| f.*.?.deinit(gpa);
    var s = try solve(gpa, &fs);
    defer s.deinit(gpa);
    try testing.expectEqual(@as(u64, 0b1), s.summary[0]);
    try testing.expectEqual(@as(u64, 0), s.summary[1]);
    try testing.expectEqual(@as(u64, 0b10), s.summary[2]);
}

test "an argument to a body in the caller's own component escapes" {
    const gpa = testing.allocator;
    // 0 and 1 call each other with their param; 2 calls itself; 3 calls 0 from outside.
    var fs = [_]?Facts{
        try testFacts(gpa, 1, &.{}, &.{.{ .slot = 0, .dep = .{ .unit = 1, .param = 0 } }}, &.{1}),
        try testFacts(gpa, 1, &.{}, &.{.{ .slot = 0, .dep = .{ .unit = 0, .param = 0 } }}, &.{0}),
        try testFacts(gpa, 1, &.{}, &.{.{ .slot = 0, .dep = .{ .unit = 2, .param = 0 } }}, &.{2}),
        try testFacts(gpa, 1, &.{}, &.{}, &.{0}),
    };
    defer for (&fs) |*f| f.*.?.deinit(gpa);
    var s = try solve(gpa, &fs);
    defer s.deinit(gpa);
    try testing.expectEqual(s.scc[0], s.scc[1]);
    try testing.expect(s.scc[2] != s.scc[0]);
    try testing.expectEqual(@as(u64, 1), s.summary[0]);
    try testing.expectEqual(@as(u64, 1), s.summary[1]);
    try testing.expectEqual(@as(u64, 1), s.summary[2]);
    try testing.expect(s.depEscapes(3, .{ .unit = 3, .param = 0 }));
    try testing.expect(s.depEscapes(3, .{ .unit = 0, .param = 0 }));
}

test "a site escapes by its use: called, a non-escaping arg, a local" {
    const gpa = testing.allocator;
    var fs = [_]?Facts{
        try testFacts(gpa, 1, &.{}, &.{}, &.{}),
        try testFacts(gpa, 0, &.{4}, &.{.{ .slot = 3, .dep = .{ .unit = 0, .param = 0 } }}, &.{0}),
    };
    defer for (&fs) |*f| f.*.?.deinit(gpa);
    var s = try solve(gpa, &fs);
    defer s.deinit(gpa);
    const f = fs[1].?;
    try testing.expect(!s.siteEscapes(f, 1, .called));
    try testing.expect(s.siteEscapes(f, 1, .escapes));
    try testing.expect(!s.siteEscapes(f, 1, .{ .arg = .{ .unit = 0, .param = 0 } }));
    try testing.expect(s.siteEscapes(f, 1, .{ .arg = .{ .unit = 0, .param = 64 } }));
    try testing.expect(!s.siteEscapes(f, 1, .{ .local = 3 }));
    try testing.expect(s.siteEscapes(f, 1, .{ .local = 4 }));
    try testing.expect(!s.siteEscapes(f, 1, .{ .local = 5 }));
}

test "a bound method site takes its use from its position, like a lambda in that position" {
    const gpa = testing.allocator;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const src =
        \\struct P { x: int }
        \\struct H { f: fn() -> int }
        \\impl P { fn get(self) -> int { self.x } }
        \\fn run(f: fn() -> int) -> int { f() }
        \\fn main() -> int {
        \\ p := P { x: 1 }
        \\ g := p.get
        \\ l := fn() -> int { 2 }
        \\ a := run(p.get) + run(fn() -> int { 3 })
        \\ h := H { f: p.get }
        \\ k := H { f: fn() -> int { 4 } }
        \\ return g() + l() + a + h.f() + k.f()
        \\}
        \\
    ;
    const tokens = try @import("../lex.zig").tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try @import("../parse.zig").expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    var g = try Graph.single(gpa, "main", "", src, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinit(gpa);
    var res = try ResolveGraph.resolveGraph(gpa, &g);
    defer res.deinit(gpa);
    var tc = try TypecheckGraph.checkGraph(gpa, &g, &res, null, 0);
    defer tc.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    var main_gid: ?u32 = null;
    var run_gid: ?u32 = null;
    for (res.fns, 0..) |f, i| {
        if (std.mem.eql(u8, f.name, "main")) main_gid = @intCast(i);
        if (std.mem.eql(u8, f.name, "main.run")) run_gid = @intCast(i);
    }
    const body: Body.BodyView = .{ .tree = tree, .resolutions = res.resolutions[0], .node_types = tc.node_types[0], .callees = tc.callees[0] };
    const prog: Body.ProgramView = .{ .sigs = tc.sigs, .methods = tc.methods, .instances = tc.instances, .layouts = tc.layouts };
    const decl = res.fns[main_gid.?].decl_node;
    var f = try factsOf(gpa, .{ .tree = tree, .tokens = tokens, .source = src }, body, prog, tc.bodies, decl, 0, Closure.findFn(res.closures[0], decl), .lowered);
    defer f.deinit(gpa);
    const run_unit = tc.bodies.ofFn(run_gid.?).?;

    // Pair each bound site with the lambda in the same position, in source order.
    var bound: std.ArrayList(Site) = .empty;
    defer bound.deinit(gpa);
    var lambdas: std.ArrayList(Site) = .empty;
    defer lambdas.deinit(gpa);
    for (f.sites) |s| switch (s.kind) {
        .bound => try bound.append(gpa, s),
        .lambda => try lambdas.append(gpa, s),
        .local_fn => return error.TestUnexpectedResult,
    };
    try testing.expectEqual(@as(usize, 3), bound.items.len);
    try testing.expectEqual(@as(usize, 3), lambdas.items.len);
    for (bound.items, lambdas.items) |b, l| {
        try testing.expectEqual(@as(?u32, 0), b.recv);
        try testing.expectEqual(std.meta.activeTag(l.use), std.meta.activeTag(b.use));
    }
    try testing.expect(bound.items[0].use == .local);
    try testing.expectEqual(Dep{ .unit = run_unit, .param = 0 }, bound.items[1].use.arg);
    try testing.expectEqual(Dep{ .unit = run_unit, .param = 0 }, lambdas.items[1].use.arg);
    try testing.expect(bound.items[2].use == .escapes);
    try testing.expectEqualSlices(u32, &.{ 0, 0, 0 }, f.bound_recv);
}
