//! Hybrid specialization of generic calls. A call of a generic fn, or of a method of a
//! generic type, whose function-typed argument is known at compile time (a named fn or a
//! lambda literal) goes to a copy of the erased instance of its own, which calls that
//! argument directly; any other argument (a local holding a function, a field, a call
//! result) goes through the erased instance.
//! A peer data module like `Closure`: the driver builds the table serially before codegen,
//! and lower and the fingerprint both pick a call's unit through `select`, so the call a
//! body makes and the identity its key folds cannot disagree.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Sym = @import("Sym.zig");
const Type = @import("../layout/Type.zig").Type;
const Resolution = @import("Resolution.zig").Resolution;
const Closure = @import("Closure.zig");
const AstWalk = @import("../query/AstWalk.zig");
const Callee = @import("Callee.zig").Callee;

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

pub const Table = struct {
    /// Sorted by name.
    units: []const Unit = &.{},
    /// Sorted by gid; only templates with an eligible param.
    masks: []const TemplateMask = &.{},

    pub fn deinit(t: Table, gpa: std.mem.Allocator) void {
        for (t.units) |u| {
            gpa.free(u.name);
            gpa.free(u.targets);
        }
        gpa.free(t.units);
        gpa.free(t.masks);
    }
};

/// Spec units per template, over all its instances. Past it a call uses the erased
/// instance; which sites keep theirs is decided by unit name, never by walk order.
pub const cap_per_template: u32 = 8;

/// The params of the template at `fn_decl` a call may specialize: typed as a function in
/// the template's own signature `params`, never assigned in its body (a spec unit assumes
/// the param still holds the caller's argument at every call of it), and below 64. A
/// template with a lambda or local fn (`has_closures`) specializes nothing, so every
/// lambda unit belongs to exactly one lowered body.
pub fn eligibleMask(src: AstWalk.Source, resolutions: []const Resolution, fn_decl: Ast.Index, params: []const Type, has_closures: bool) u64 {
    if (has_closures) return 0;
    var fn_params: u64 = 0;
    for (params[0..@min(params.len, 64)], 0..) |p, i| {
        if (p.isFunc()) fn_params |= @as(u64, 1) << @intCast(i);
    }
    if (fn_params == 0) return 0;
    const V = struct {
        tree: Ast.Tree,
        resolutions: []const Resolution,
        mask: u64,
        pub fn on(self: *@This(), ev: AstWalk.Event) void {
            const e = switch (ev) {
                .enter => |e| e,
                else => return,
            };
            if (e.tag != .assign) return;
            const target = self.tree.nodes[e.idx.int()].lhs;
            if (self.tree.nodes[target.int()].tag != .identifier or target.int() >= self.resolutions.len) return;
            switch (self.resolutions[target.int()]) {
                .local => |slot| if (slot < 64) {
                    self.mask &= ~(@as(u64, 1) << @intCast(slot));
                },
                else => {},
            }
        }
    };
    var v: V = .{ .tree = src.tree, .resolutions = resolutions, .mask = fn_params };
    AstWalk.walk(src, fn_decl, &v) catch unreachable;
    return v.mask;
}

/// The targets of generic call `call` resolving to instance `ii`, ascending by param: for
/// each eligible param, an argument naming a fn (`callees[arg]` is `.fn_ref`) or a lambda
/// literal of the view's owner. A local fn is a local holding a function, like any other
/// variable, so it is not a target. Argument `a` is param `a + self_n`.
///
/// `view` supplies `.tree`, `.callees`, `.names`, `.instances` and `.closure` (its
/// `lambda_units` and `owner`).
pub fn targetsOf(gpa: std.mem.Allocator, view: anytype, call: Ast.Index, site: Site, eligible: u64, out: *std.ArrayList(Target)) error{OutOfMemory}!void {
    const tree = view.tree;
    const args = Ast.rangeSlice(tree, tree.nodes[call.int()].rhs.int());
    const nparams = view.instances[site.ii].params.len;
    for (args, @as(usize, site.self_n)..) |arg, p| {
        if (p >= nparams or p >= 64) break;
        if (eligible & (@as(u64, 1) << @intCast(p)) == 0) continue;
        const an = tree.nodes[arg.int()];
        switch (an.tag) {
            .identifier, .field_access => {
                if (arg.int() >= view.callees.len) continue;
                switch (view.callees[arg.int()].view()) {
                    .fn_ref => |gid| if (gid < view.names.len) try out.append(gpa, .{ .param = @intCast(p), .kind = .named, .sym = view.names[gid] }),
                    .none, .method, .protocol, .builtin, .value, .bound_method => {},
                }
            },
            .lambda => {
                const owner = view.closure.owner orelse continue;
                const ui = Closure.findUnit(view.closure.lambda_units, owner, arg) orelse continue;
                try out.append(gpa, .{ .param = @intCast(p), .kind = .lambda, .sym = .{ .kind = .user_fn, .name = view.closure.lambda_units[ui].name } });
            },
            else => {},
        }
    }
}

/// `<inst>$sp<p>$<sym>` per target, ascending p. A pure function of the instance and target
/// symbols. No instance, fn or lambda symbol has a `$`-separated piece spelled `sp<n>`, so
/// the pieces split back apart and distinct target lists give distinct names.
pub fn nameOf(gpa: std.mem.Allocator, inst_name: []const u8, targets: []const Target) error{OutOfMemory}![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, inst_name);
    for (targets) |t| try out.print(gpa, "$sp{d}${s}", .{ t.param, t.sym.name });
    return out.toOwnedSlice(gpa);
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

/// The site of method call `call` when its method is one of a generic type's instances
/// (`b.apply(f)` on a `Box[int]`), or null. `view` supplies `.tree`, `.callees` and
/// `.methods`.
pub fn methodSite(view: anytype, call: Ast.Index) ?Site {
    const fa = Callee.memberNode(view.tree, view.tree.nodes[call.int()]) orelse return null;
    if (fa.int() >= view.callees.len) return null;
    const v = view.callees[fa.int()].view();
    if (v != .method or v.method >= view.methods.len) return null;
    const m = view.methods[v.method];
    return .{ .ii = m.instance orelse return null, .self_n = @intFromBool(!m.is_static) };
}

/// The spec unit generic call `call` at `site` goes to, or null for the erased instance;
/// the name is borrowed from the table. `view` is as for `targetsOf`, plus
/// `.closure.specs`.
pub fn select(gpa: std.mem.Allocator, view: anytype, call: Ast.Index, site: Site) error{OutOfMemory}!?[]const u8 {
    const t: Table = view.closure.specs;
    if (t.units.len == 0) return null;
    const inst = view.instances[site.ii];
    const m = mask(t, inst.template_gid);
    if (m == 0) return null;
    var targets: std.ArrayList(Target) = .empty;
    defer targets.deinit(gpa);
    try targetsOf(gpa, view, call, site, m, &targets);
    if (targets.items.len == 0) return null;
    const name = try nameOf(gpa, inst.name.?, targets.items);
    defer gpa.free(name);
    const u = find(t, name) orelse return null;
    return u.name;
}

/// A spec unit some call site wants, keyed by its template for the cap.
pub const Candidate = struct { gid: u32, unit: Unit };

/// Record the unit a call of instance `ii` with `targets` would get, unless an equal one is
/// already in `out`. `targets` is copied.
pub fn addCandidate(gpa: std.mem.Allocator, out: *std.ArrayList(Candidate), gid: u32, ii: usize, inst_name: []const u8, targets: []const Target) error{OutOfMemory}!void {
    const name = try nameOf(gpa, inst_name, targets);
    errdefer gpa.free(name);
    for (out.items) |c| if (std.mem.eql(u8, c.unit.name, name)) {
        gpa.free(name);
        return;
    };
    const owned = try gpa.dupe(Target, targets);
    errdefer gpa.free(owned);
    try out.append(gpa, .{ .gid = gid, .unit = .{ .inst = @intCast(ii), .name = name, .targets = owned } });
}

/// The table from every candidate: per template, the first `cap_per_template` by name,
/// then all sorted by name. Consumes `cands` and `masks` (both freed, on error too).
pub fn finish(gpa: std.mem.Allocator, cands: *std.ArrayList(Candidate), masks: []const TemplateMask) error{OutOfMemory}!Table {
    defer cands.deinit(gpa);
    errdefer gpa.free(masks);
    std.mem.sort(Candidate, cands.items, {}, struct {
        fn lt(_: void, a: Candidate, b: Candidate) bool {
            if (a.gid != b.gid) return a.gid < b.gid;
            return std.mem.order(u8, a.unit.name, b.unit.name) == .lt;
        }
    }.lt);
    var units = std.ArrayList(Unit).initCapacity(gpa, cands.items.len) catch |e| {
        freeUnits(gpa, cands.items);
        return e;
    };
    var run_gid: u32 = 0;
    var run_len: u32 = 0;
    for (cands.items, 0..) |c, i| {
        if (i == 0 or c.gid != run_gid) {
            run_gid = c.gid;
            run_len = 0;
        }
        if (run_len == cap_per_template) {
            gpa.free(c.unit.name);
            gpa.free(c.unit.targets);
            continue;
        }
        run_len += 1;
        units.appendAssumeCapacity(c.unit);
    }
    cands.clearRetainingCapacity();
    std.mem.sort(Unit, units.items, {}, struct {
        fn lt(_: void, a: Unit, b: Unit) bool {
            return std.mem.order(u8, a.name, b.name) == .lt;
        }
    }.lt);
    const out = units.toOwnedSlice(gpa) catch |e| {
        for (units.items) |u| {
            gpa.free(u.name);
            gpa.free(u.targets);
        }
        units.deinit(gpa);
        return e;
    };
    return .{ .units = out, .masks = masks };
}

/// Free the owned name and targets of each candidate's unit.
pub fn freeUnits(gpa: std.mem.Allocator, cands: []const Candidate) void {
    for (cands) |c| {
        gpa.free(c.unit.name);
        gpa.free(c.unit.targets);
    }
}

const testing = std.testing;

test "nameOf is content-derived and injective" {
    const gpa = testing.allocator;
    const add: Sym.SymName = .{ .kind = .user_fn, .name = "main.add" };
    const lam: Sym.SymName = .{ .kind = .user_fn, .name = "main$l0" };
    const a = try nameOf(gpa, "main.apply$int", &.{.{ .param = 0, .kind = .named, .sym = add }});
    defer gpa.free(a);
    try testing.expectEqualStrings("main.apply$int$sp0$main.add", a);
    const b = try nameOf(gpa, "main.apply$int", &.{.{ .param = 0, .kind = .named, .sym = add }});
    defer gpa.free(b);
    try testing.expectEqualStrings(a, b);

    // Distinct instances, params, targets and target lists give distinct names.
    const variants = [_][]const Target{
        &.{.{ .param = 1, .kind = .named, .sym = add }},
        &.{.{ .param = 0, .kind = .lambda, .sym = lam }},
        &.{ .{ .param = 0, .kind = .named, .sym = add }, .{ .param = 1, .kind = .lambda, .sym = lam } },
        &.{ .{ .param = 0, .kind = .lambda, .sym = lam }, .{ .param = 1, .kind = .named, .sym = add } },
    };
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    try names.append(gpa, try gpa.dupe(u8, a));
    try names.append(gpa, try nameOf(gpa, "main.apply$bool", &.{.{ .param = 0, .kind = .named, .sym = add }}));
    for (variants) |v| try names.append(gpa, try nameOf(gpa, "main.apply$int", v));
    for (names.items, 0..) |x, i| for (names.items[i + 1 ..]) |y| try testing.expect(!std.mem.eql(u8, x, y));
    try testing.expectEqualStrings("main.apply$int$sp0$main.add$sp1$main$l0", names.items[4]);

    // A spec piece is never a generic-instance arg spelling.
    const Mono = @import("Mono.zig");
    for ([_]Type{ Type.int, Type.bool, Type.str, Type.unit, Type.float, Type.structT(3), Type.enumT(4), Type.rawptr }) |t| {
        const m = try Mono.mangle(gpa, "f", &.{t});
        defer gpa.free(m);
        try testing.expect(std.mem.indexOf(u8, m, "$sp") == null);
    }
}

test "finish caps each template by name and sorts the units by name" {
    const gpa = testing.allocator;
    var cands: std.ArrayList(Candidate) = .empty;
    const lam_names = [_][]const u8{ "m$l9", "m$l3", "m$l0", "m$l7", "m$l1", "m$l8", "m$l2", "m$l6", "m$l5", "m$l4" };
    for (lam_names) |ln| try addCandidate(gpa, &cands, 4, 0, "g$int", &.{.{ .param = 0, .kind = .lambda, .sym = .{ .kind = .user_fn, .name = ln } }});
    // A repeat of an earlier site adds nothing.
    try addCandidate(gpa, &cands, 4, 0, "g$int", &.{.{ .param = 0, .kind = .lambda, .sym = .{ .kind = .user_fn, .name = "m$l3" } }});
    try addCandidate(gpa, &cands, 2, 1, "a$int", &.{.{ .param = 0, .kind = .named, .sym = .{ .kind = .user_fn, .name = "m.f" } }});
    try testing.expectEqual(@as(usize, 11), cands.items.len);
    const masks = try gpa.dupe(TemplateMask, &.{ .{ .gid = 2, .eligible = 1 }, .{ .gid = 4, .eligible = 1 } });
    const t = try finish(gpa, &cands, masks);
    defer t.deinit(gpa);
    try testing.expectEqual(@as(usize, 1 + cap_per_template), t.units.len);
    for (t.units[1..], t.units[0 .. t.units.len - 1]) |cur, prev| try testing.expect(std.mem.order(u8, prev.name, cur.name) == .lt);
    try testing.expect(find(t, "a$int$sp0$m.f") != null);
    try testing.expect(find(t, "g$int$sp0$m$l7") != null);
    try testing.expect(find(t, "g$int$sp0$m$l8") == null);
    try testing.expect(find(t, "g$int$sp0$m$l9") == null);
    try testing.expectEqual(@as(u64, 1), mask(t, 4));
    try testing.expectEqual(@as(u64, 0), mask(t, 3));
}

/// One module resolved and typechecked, with `names` in global fn-id order, so the spec
/// helpers read the checker's real tables.
const Checked = struct {
    const Token = @import("../ast/Token.zig").Token;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const LowerInputs = @import("../lower.zig").ClosureInputs;
    const Mono = @import("Mono.zig");

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

    /// The eligible mask of template `name`, as the driver computes it.
    fn templateMask(self: *const Checked, name: []const u8) !u64 {
        const g = try self.gid(name);
        const f = self.res.fns[g];
        const has_closures = Closure.findFn(self.res.closures[f.module], f.decl_node) != null;
        return eligibleMask(.{ .tree = self.tree, .tokens = self.tokens, .source = self.src }, self.res.resolutions[0], f.decl_node, self.tc.sigs[g].params, has_closures);
    }

    const View = struct {
        tree: Ast.Tree,
        callees: []const @import("Callee.zig").Callee,
        names: []const Sym.SymName,
        instances: []const Mono.Instance,
        methods: []const @import("../types.zig").Method,
        closure: LowerInputs,
    };

    fn view(self: *const Checked, owner: []const u8) !View {
        return .{
            .tree = self.tree,
            .callees = self.tc.callees[0],
            .names = self.names,
            .instances = self.tc.instances,
            .methods = self.tc.methods,
            .closure = .{ .adapters = self.tc.adapters, .lambda_units = self.tc.lambda_units, .owner = .{ .base = try self.gid(owner) } },
        };
    }

    /// The targets of every call in fn `owner` of an instance of `template`, all params
    /// eligible, read under `owner`'s view.
    fn targetsIn(self: *const Checked, gpa: std.mem.Allocator, owner: []const u8, template: []const u8, out: *std.ArrayList(Target)) !void {
        const tg = try self.gid(template);
        const v = try self.view(owner);
        for (self.tree.nodes, 0..) |n, i| {
            if (n.tag != .call) continue;
            const callee = self.tree.nodes[n.lhs.int()];
            if (callee.tag != .identifier or self.res.resolutions[0][n.lhs.int()] != .func) continue;
            if (self.res.resolutions[0][n.lhs.int()].func != tg) continue;
            const ii = for (self.tc.instances, 0..) |inst, k| {
                if (inst.template_gid == tg) break k;
            } else return error.TestExpectedInstance;
            try targetsOf(gpa, &v, Ast.Index.from(@intCast(i)), .{ .ii = ii, .self_n = 0 }, ~@as(u64, 0), out);
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
    const v = try fx.view("main");
    var sites: usize = 0;
    for (fx.tree.nodes, 0..) |n, i| {
        if (n.tag != .call) continue;
        const call = Ast.Index.from(@intCast(i));
        const site = methodSite(&v, call) orelse continue;
        sites += 1;
        try testing.expectEqual(@as(u32, 1), site.self_n);
        try testing.expectEqualStrings("main.Box.twice$int", fx.tc.instances[site.ii].name.?);
        var targets: std.ArrayList(Target) = .empty;
        defer targets.deinit(gpa);
        try targetsOf(gpa, &v, call, site, ~@as(u64, 0), &targets);
        try testing.expectEqual(@as(usize, 1), targets.items.len);
        try testing.expectEqual(@as(u32, 2), targets.items[0].param);
        try testing.expectEqualStrings("main.inc", targets.items[0].sym.name);
    }
    try testing.expectEqual(@as(usize, 1), sites);
    // The template's mask names the same param.
    try testing.expectEqual(@as(u64, 0b100), try fx.templateMask("main.Box.twice"));
}
