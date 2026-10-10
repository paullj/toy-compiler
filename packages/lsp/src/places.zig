//! What the source does not spell about a place: how a closure holds each local it
//! captures, which call writes a local through a `mut` param, and whether a local still
//! holds what it was bound to. Hover, inlay hints and signature help show these.
//! Everything is read from what the compiler already decided: the lambda units' envs and
//! the owners' boxed slots (`Closure`), and the arg modes lower passes by
//! (`Modes.argMask`), so the editor cannot disagree with the binary.
//!
//! The LSP does not know the opt level, and since escape analysis a captured-and-assigned
//! local may live in its owner's frame instead of a GC cell. So a capture is described by
//! its language-level mode, and a storage word is added only where every build agrees:
//! "shared (boxed)" when the cell survives frame storage too, "shared (by address)" for a
//! `mut` param, plain "shared" when the opt level decides.
//!
//! KNOWN LIMITATIONS:
//!   * A closure in a generic fn has an env only per instance. With no instance its
//!     captures show no mode, and a copy's size shows only when every instance agrees.
//!   * A method call inside an unbounded generic body records no selection (see
//!     `callee_sig`), so its `mut` arguments and receiver are not reported.

const std = @import("std");
const toyc = @import("toy_compiler");
const render = @import("render.zig");

const Ast = toyc.Ast;
const Abi = toyc.Abi;
const Callee = toyc.Callee;
const Closure = toyc.Closure;
const Body = toyc.Body;
const Modes = toyc.Modes;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Type = toyc.Typecheck.Type;

/// How a closure holds one captured local.
pub const Hold = union(enum) {
    /// A copy taken when the closure is made: its size in the env, or null when the
    /// instances of a generic fn disagree on it.
    copied: ?u32,
    /// Captured and assigned, and in a GC cell in every build: a closure that captures it
    /// escapes, a bound method value holds it, or it is too big to leave its cell.
    boxed,
    /// Captured and assigned: a cell without frame storage, the owner's own slot with it.
    shared,
    /// A `mut` param (`mut self` included): the caller's place, held by address.
    by_address,
    /// No checked env to read: a generic fn with no instance, or a lambda an error left
    /// untyped.
    unknown,

    /// The mode in one word, or null when it is unknown.
    pub fn word(h: Hold) ?[]const u8 {
        return switch (h) {
            .copied => "copied",
            .boxed, .shared, .by_address => "shared",
            .unknown => null,
        };
    }

    /// The mode with its storage detail ("copied (8 bytes)", "shared (boxed)"); nothing
    /// for an unknown one.
    pub fn append(h: Hold, a: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        switch (h) {
            .copied => |size| if (size) |n| try buf.print(a, "copied ({d} bytes)", .{n}) else try buf.appendSlice(a, "copied"),
            .boxed => try buf.appendSlice(a, "shared (boxed)"),
            .shared => try buf.appendSlice(a, "shared"),
            .by_address => try buf.appendSlice(a, "shared (by address)"),
            .unknown => {},
        }
    }

    /// One hold for every instance of a generic fn: a detail survives only when they all
    /// agree on it.
    fn merge(x: Hold, y: Hold) Hold {
        return switch (x) {
            .copied => |sx| switch (y) {
                .copied => |sy| .{ .copied = if (sx != null and sy != null and sx.? == sy.?) sx else null },
                else => .unknown,
            },
            .boxed => if (y == .boxed) .boxed else if (y == .shared) .shared else .unknown,
            .shared => if (y == .boxed or y == .shared) .shared else .unknown,
            .by_address => if (y == .by_address) .by_address else .unknown,
            .unknown => .unknown,
        };
    }
};

/// One local a closure captures, named as at its first use inside the closure.
pub const Item = struct { name: []const u8, hold: Hold };

/// A lambda or local fn that captures, with its captures in slot order.
pub const Captures = struct {
    /// The `.lambda` or `.local_fn` node.
    node: u32,
    /// The `{` token that opens its body.
    body: u32,
    items: []const Item,
};

/// An argument a callee writes through: one a `mut` param takes, or a `mut self`
/// receiver.
pub const MutArg = struct {
    /// The argument: a local, or a field / tuple-field chain rooted at one.
    node: u32,
    /// The callee as the user knows it, unqualified.
    callee: []const u8,
    receiver: bool,
};

/// The checked entry module of one request. Every slice borrows the compiler results.
pub const Facts = struct {
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const TypecheckGraph.GraphResult,

    fn entry(f: Facts) u32 {
        return f.graph.entry_index;
    }

    fn module(f: Facts) *const Graph.Module {
        return f.graph.entry();
    }

    fn span(f: Facts, node: u32) toyc.Walk.Span {
        const m = f.module();
        return toyc.Walk.nodeSpan(.{ .tree = m.tree(), .tokens = m.tokens, .source = m.source }, Ast.Index.from(node));
    }

    fn fnClosures(f: Facts) []const Closure.FnClosures {
        return if (f.entry() < f.res.closures.len) f.res.closures[f.entry()] else &.{};
    }

    /// How the local `tok` names is held, when `tok` is a capture: by the innermost lambda
    /// or local fn around it that captures it, else by the bound method value whose
    /// receiver it is. Null for any other token.
    pub fn captureAt(f: Facts, tok: u32) ?Hold {
        const m = f.module();
        const resolutions = f.res.resolutions[f.entry()];
        for (m.nodes, 0..) |n, i| {
            if (n.tag != .identifier or n.main_token != tok) continue;
            const slot = switch (resolutions[i]) {
                .local => |s| s,
                else => continue,
            };
            if (f.innermostCapturing(tok, slot)) |node| return f.envHold(node, slot);
            if (f.isBoundReceiver(@intCast(i))) return f.boundHold(tok, @intCast(i), slot);
        }
        return null;
    }

    /// Every lambda and local fn in the entry module that captures a local. A `_` capture
    /// is left out: each local it captures is spelled in its one call.
    pub fn capturing(f: Facts, a: std.mem.Allocator) ![]const Captures {
        const m = f.module();
        const tree = m.tree();
        var out: std.ArrayList(Captures) = .empty;
        for (f.fnClosures()) |fc| for (fc.boundaries) |b| {
            if (b.captures.len == 0 or Ast.isCaptureLambda(tree, m.tokens, b.node)) continue;
            const body = m.nodes[b.node.int()].rhs;
            if (body == Ast.none or m.nodes[body.int()].tag != .block) continue;
            const items = try a.alloc(Item, b.captures.len);
            for (b.captures, items) |c, *it| it.* = .{
                .name = m.tokens[m.nodes[c.first_use.int()].main_token].text(m.source),
                .hold = f.envHold(b.node, c.slot),
            };
            try out.append(a, .{ .node = b.node.int(), .body = m.nodes[body.int()].main_token, .items = items });
        };
        return out.toOwnedSlice(a);
    }

    /// Every argument a `mut` param takes and every `mut self` receiver in the entry
    /// module, in call-node order.
    pub fn mutArgs(f: Facts, a: std.mem.Allocator) ![]const MutArg {
        const m = f.module();
        const tree = m.tree();
        const e = f.entry();
        const body: Body.BodyView = .{ .tree = tree, .resolutions = f.res.resolutions[e], .node_types = f.tc.node_types[e], .callees = f.tc.callees[e] };
        const prog: Body.ProgramView = .{ .sigs = f.tc.sigs, .methods = f.tc.methods, .instances = f.tc.instances };
        var out: std.ArrayList(MutArg) = .empty;
        for (m.nodes, 0..) |n, c| {
            if (n.tag != .call) continue;
            const mask = Modes.argMask(body, prog, Ast.Index.from(@intCast(c)));
            const recv = f.mutReceiver(n);
            if (mask == 0 and recv == null) continue;
            const name = f.calleeName(n) orelse continue;
            if (recv) |r| try out.append(a, .{ .node = r, .callee = name, .receiver = true });
            for (Ast.rangeSlice(tree, n.rhs.int()), 0..) |arg, k| {
                if (Modes.isMut(mask, k)) try out.append(a, .{ .node = arg.int(), .callee = name, .receiver = false });
            }
        }
        return out.toOwnedSlice(a);
    }

    /// The callee that writes the place `tok` is part of, when that place is a `mut`
    /// argument or a `mut self` receiver: its root local or any field on the way to it.
    pub fn mutatedBy(f: Facts, a: std.mem.Allocator, tok: u32) !?[]const u8 {
        const nodes = f.module().nodes;
        for (try f.mutArgs(a)) |ma| {
            var cur = ma.node;
            while (true) {
                const n = nodes[cur];
                if (n.main_token == tok) return ma.callee;
                switch (n.tag) {
                    .field_access, .tuple_field => cur = n.lhs.int(),
                    else => break,
                }
            }
        }
        return null;
    }

    /// The binder (a `:=` or a local fn) of the local that the identifier node `ident`
    /// names, when nothing in its fn writes that local again: no assignment rooted at it
    /// and no `mut` argument rooted at it. Null for a param, a loop or match binding, or
    /// a local written after its binding.
    pub fn soleBinding(f: Facts, a: std.mem.Allocator, ident: u32) !?u32 {
        const m = f.module();
        const resolutions = f.res.resolutions[f.entry()];
        if (ident >= resolutions.len) return null;
        const slot = switch (resolutions[ident]) {
            .local => |s| s,
            else => return null,
        };
        const fn_node = f.enclosingFn(m.nodes[ident].main_token) orelse return null;
        // Slots number a whole top-level fn, so the span keeps another fn's same slot out.
        const sp = f.span(fn_node);
        var binder: ?u32 = null;
        for (m.nodes[0..@min(m.nodes.len, resolutions.len)], 0..) |n, i| {
            const at = m.tokens[n.main_token].start;
            if (at < sp.start or at >= sp.end) continue;
            switch (n.tag) {
                .var_decl, .local_fn => if (resolutions[i] == .local and resolutions[i].local == slot) {
                    binder = @intCast(i);
                },
                .assign => if (toyc.Resolve.placeRoot(m.tree(), resolutions, n.lhs) == slot) return null,
                else => {},
            }
        }
        for (try f.mutArgs(a)) |ma| {
            const at = m.tokens[m.nodes[ma.node].main_token].start;
            if (at >= sp.start and at < sp.end and toyc.Resolve.placeRoot(m.tree(), resolutions, Ast.Index.from(ma.node)) == slot) return null;
        }
        return binder;
    }

    /// The smallest lambda or local fn around `tok` that captures `slot`. Slots number a
    /// whole top-level fn, so the span test also keeps another fn's same slot out.
    fn innermostCapturing(f: Facts, tok: u32, slot: u32) ?Ast.Index {
        const at = f.module().tokens[tok].start;
        var best: ?Ast.Index = null;
        var best_len: u32 = std.math.maxInt(u32);
        for (f.fnClosures()) |fc| for (fc.boundaries) |b| {
            const holds = for (b.captures) |c| {
                if (c.slot == slot) break true;
            } else false;
            if (!holds) continue;
            const sp = f.span(b.node.int());
            if (at < sp.start or at >= sp.end or sp.end - sp.start >= best_len) continue;
            best = b.node;
            best_len = sp.end - sp.start;
        };
        return best;
    }

    /// How closure `node` holds `slot`, from the env every lowered unit of it lays out:
    /// one for a non-generic owner, one per instance of a generic one.
    fn envHold(f: Facts, node: Ast.Index, slot: u32) Hold {
        var hold: ?Hold = null;
        for (f.tc.lambda_units) |u| {
            if (u.mod != f.entry() or u.node != node) continue;
            for (u.env) |es| {
                if (es.slot != slot) continue;
                const h = f.ownerHold(u.owner, slot, f.sizeOf(es.ty));
                hold = if (hold) |prev| prev.merge(h) else h;
            }
        }
        return hold orelse .unknown;
    }

    /// How `owner` holds `slot` in every build: the compiler's storage without frame
    /// storage and with it, merged, so a storage word survives only where both agree.
    /// `copy_size` is the bytes a copy takes.
    fn ownerHold(f: Facts, owner: Closure.Owner, slot: u32, copy_size: ?u32) Hold {
        var hold: ?Hold = null;
        for ([_]bool{ false, true }) |frame| {
            const h: Hold = switch (Closure.ownerStorage(f.tc.bodies, f.tc.owner_closures, owner, frame).hold(slot)) {
                .copy => .{ .copied = copy_size },
                .cell => .boxed,
                .frame_addr => .shared,
                .param_addr => .by_address,
            };
            hold = if (hold) |prev| prev.merge(h) else h;
        }
        return hold.?;
    }

    fn isBoundReceiver(f: Facts, recv: u32) bool {
        const e = f.entry();
        for (f.module().nodes, 0..) |n, fa| {
            if (n.tag != .field_access or n.lhs.int() != recv) continue;
            if (fa < f.tc.callees[e].len and f.tc.callees[e][fa].view() == .bound_method) return true;
            for (f.tc.instances) |inst| {
                if (inst.mod == e and fa < inst.callees.len and inst.callees[fa].view() == .bound_method) return true;
            }
        }
        return false;
    }

    /// How a bound method value holds its local receiver: as its owner holds it, which
    /// `Closure.boundKind` turns into its adapter. A `mut` param is held by address even in
    /// a generic fn with no instance.
    fn boundHold(f: Facts, tok: u32, recv: u32, slot: u32) Hold {
        const fn_node = Ast.Index.from(f.enclosingFn(tok) orelse return .unknown);
        const e = f.entry();
        const gid = for (f.res.fns, 0..) |gf, g| {
            if (gf.module == e and gf.decl_node == fn_node) break g;
        } else return .unknown;
        if (Modes.isMut(f.tc.sigs[gid].mut_mask, slot)) return .by_address;
        // One body for a non-generic fn, one per instance of a generic one.
        var hold: ?Hold = null;
        for (f.tc.bodies.items, 0..) |b, i| {
            if (b.mod != e or b.decl != fn_node) continue;
            const h = f.ownerHold(@intCast(i), slot, if (recv < b.node_types.len) f.sizeOf(b.node_types[recv]) else null);
            hold = if (hold) |prev| prev.merge(h) else h;
        }
        return hold orelse .unknown;
    }

    /// The top-level fn (or method) whose body holds `tok`.
    fn enclosingFn(f: Facts, tok: u32) ?u32 {
        const m = f.module();
        const at = m.tokens[tok].start;
        for (m.nodes, 0..) |n, i| {
            if (n.tag != .fn_decl or n.rhs == Ast.none) continue;
            const sp = f.span(@intCast(i));
            if (at >= sp.start and at < sp.end) return @intCast(i);
        }
        return null;
    }

    /// The bytes a copy of a `ty` value takes in an env, as `layoutCaptureEnv` sizes it.
    fn sizeOf(f: Facts, ty: Type) ?u32 {
        switch (ty.kind) {
            .invalid, .type_var, .app, .func => return null,
            .@"struct" => if (ty.struct_id >= f.tc.layouts.len) return null,
            .@"enum" => if (ty.enum_id >= f.tc.enum_layouts.len) return null,
            else => {},
        }
        return Abi.typeSize(ty, f.tc.layouts, f.tc.enum_layouts);
    }

    /// A `mut self` method call's receiver node.
    fn mutReceiver(f: Facts, call: Ast.Node) ?u32 {
        const m = f.module();
        const fa = Callee.memberNode(m.tree(), call) orelse return null;
        const callees = f.tc.callees[f.entry()];
        if (fa.int() >= callees.len) return null;
        const mi = switch (callees[fa.int()].view()) {
            .method => |mi| mi,
            else => return null,
        };
        if (mi >= f.tc.methods.len) return null;
        const meth = f.tc.methods[mi];
        if (!meth.mut_self or meth.is_static) return null;
        return m.nodes[fa.int()].lhs.int();
    }

    /// The callee `argMask` read the modes of: a selected method, or a top-level fn.
    fn calleeName(f: Facts, call: Ast.Node) ?[]const u8 {
        const tree = f.module().tree();
        const callees = f.tc.callees[f.entry()];
        const slot = Callee.callSlot(tree, call) orelse return null;
        if (slot.int() >= callees.len) return null;
        const gid = switch (callees[slot.int()].view()) {
            .method => |mi| return if (mi < f.tc.methods.len) f.tc.methods[mi].name else null,
            .direct, .generic => |gid| gid,
            .instance => |ii| if (ii < f.tc.instances.len) f.tc.instances[ii].template_gid else return null,
            else => return null,
        };
        return if (gid < f.tc.sigs.len) render.unmangled(render.bareName(f.tc.sigs[gid].name)) else null;
    }
};

const testing = std.testing;
const Documents = @import("Documents.zig");
const Workspace = @import("Workspace.zig");

/// A checked fixture: the compiler results `Facts` borrows, torn down together.
const Checked = struct {
    docs: Documents = .{},
    graph: Graph.Graph,
    res: ResolveGraph.GraphResult,
    tc: TypecheckGraph.GraphResult,
    arena: std.heap.ArenaAllocator,

    fn init(gpa: std.mem.Allocator, src: []const u8) !*Checked {
        const c = try gpa.create(Checked);
        errdefer gpa.destroy(c);
        c.docs = .{};
        const ws: Workspace = .{ .io = std.Io.failing, .docs = &c.docs, .disk = false };
        c.graph = try ws.discover(gpa, "file:///p/main.toy", src);
        errdefer c.graph.deinit(gpa);
        try testing.expect(c.graph.err == null);
        c.res = try ResolveGraph.resolveGraph(gpa, &c.graph);
        errdefer c.res.deinit(gpa);
        c.tc = try TypecheckGraph.checkGraph(gpa, &c.graph, &c.res, ws.io, 0);
        c.arena = .init(gpa);
        return c;
    }

    fn deinit(c: *Checked, gpa: std.mem.Allocator) void {
        c.arena.deinit();
        c.tc.deinit(gpa);
        c.res.deinit(gpa);
        c.graph.deinit(gpa);
        c.docs.deinit(gpa);
        gpa.destroy(c);
    }

    fn facts(c: *const Checked) Facts {
        return .{ .graph = &c.graph, .res = &c.res, .tc = &c.tc };
    }

    /// The token at the first occurrence of `needle`, plus `skip` bytes.
    fn tokOf(c: *const Checked, needle: []const u8, skip: u32) u32 {
        const m = c.graph.entry();
        const off: u32 = @intCast(std.mem.indexOf(u8, m.source, needle).? + skip);
        return @import("position.zig").tokenAt(m.tokens, off).?;
    }

    fn holdText(c: *Checked, needle: []const u8, skip: u32) !?[]const u8 {
        const h = c.facts().captureAt(c.tokOf(needle, skip)) orelse return null;
        var buf: std.ArrayList(u8) = .empty;
        try h.append(c.arena.allocator(), &buf);
        return buf.items;
    }
};

const capture_fixture =
    \\struct Big { a: int, b: int, c: int }
    \\struct Acc { n: int }
    \\impl Acc {
    \\    fn plus(self, k: int) -> int { self.n + k }
    \\    fn bump(mut self) { self.n = self.n + 1 }
    \\}
    \\fn apply(f: fn(int) -> int, v: int) -> int { f(v) }
    \\fn add(a: int, b: int) -> int { a + b }
    \\fn counter() -> fn() -> int {
    \\    n := 0
    \\    fn() { n = n + 1
    \\        n }
    \\}
    \\fn grow(mut m: int) -> int { apply(fn(v) { m + v }, 1) }
    \\fn main() -> int {
    \\    big := Big { a: 1, b: 2, c: 3 }
    \\    total := 0
    \\    r := apply(fn(v) { total = total + v
    \\        total + big.a }, 5)
    \\    base := 10
    \\    fn fact(k: int) -> int { if k <= 1 { base } else { k * fact(k - 1) } }
    \\    inc := add(base, _)
    \\    acc := Acc { n: 1 }
    \\    f := acc.plus
    \\    held := Acc { n: 2 }
    \\    g := held.plus
    \\    held.bump()
    \\    nested := apply(fn(v) { apply(fn(w) { w + base }, v) }, 2)
    \\    return r + fact(3) + inc(1) + f(1) + g(1) + counter()() + grow(r) + nested
    \\}
    \\
;

test "captureAt: a copy shows its env size; a shared local says whether every build boxes it" {
    const gpa = testing.allocator;
    const c = try Checked.init(gpa, capture_fixture);
    defer c.deinit(gpa);
    const cases = [_]struct { needle: []const u8, skip: u32 = 0, want: []const u8 }{
        // Returned from `counter`, so the closure escapes and `n` keeps its cell.
        .{ .needle = "n = n + 1", .want = "shared (boxed)" },
        .{ .needle = "n = n + 1", .skip = 4, .want = "shared (boxed)" },
        // Passed only to `apply`, which calls it: frame storage may unbox `total`.
        .{ .needle = "total = total", .want = "shared" },
        .{ .needle = "big.a", .want = "copied (24 bytes)" },
        .{ .needle = "base } else", .want = "copied (8 bytes)" },
        // A `_` capture captures the locals its call names.
        .{ .needle = "base, _)", .want = "copied (8 bytes)" },
        .{ .needle = "m + v", .want = "shared (by address)" },
        // A bound method value captures its receiver.
        .{ .needle = "acc.plus", .want = "copied (8 bytes)" },
        .{ .needle = "held.plus", .want = "shared (boxed)" },
        // The innermost closure is the one that holds it; both copy it.
        .{ .needle = "w + base", .skip = 4, .want = "copied (8 bytes)" },
    };
    for (cases) |cs| {
        const got = (try c.holdText(cs.needle, cs.skip)) orelse {
            std.debug.print("no capture at '{s}'+{d}\n", .{ cs.needle, cs.skip });
            return error.NoCapture;
        };
        try testing.expectEqualStrings(cs.want, got);
    }
    // Not captures: the outer declaration and use, a lambda's own param, a local fn
    // naming itself, and a method call's receiver.
    for ([_][]const u8{ "total := 0", "r + fact", "v\n        total", "fact(k - 1)", "held.bump" }) |needle| {
        try testing.expect(c.facts().captureAt(c.tokOf(needle, 0)) == null);
    }
}

test "capturing: lambdas and local fns list their captures; a '_' capture does not" {
    const gpa = testing.allocator;
    const c = try Checked.init(gpa, capture_fixture);
    defer c.deinit(gpa);
    const a = c.arena.allocator();
    const m = c.graph.entry();
    var seen: std.ArrayList(u8) = .empty;
    for (try c.facts().capturing(a)) |cl| {
        try testing.expectEqual(toyc.Tag.l_brace, m.tokens[cl.body].tag);
        for (cl.items, 0..) |it, k| {
            try seen.appendSlice(a, if (k == 0) "[" else ",");
            try seen.appendSlice(a, it.name);
            try seen.append(a, '=');
            try seen.appendSlice(a, it.hold.word().?);
        }
        try seen.append(a, ']');
    }
    try testing.expectEqualStrings("[n=shared][m=shared][big=copied,total=shared][base=copied][base=copied][base=copied]", seen.items);
}

test "a generic fn's capture keeps only what every instance agrees on" {
    const gpa = testing.allocator;
    const src =
        \\fn hold[T](x: T) -> fn() -> T { fn() { x } }
        \\fn never[T](x: T) -> fn() -> T { fn() { x } }
        \\fn main() -> int {
        \\    a := hold(1)
        \\    b := hold("s")
        \\    return a() + b().len()
        \\}
        \\
    ;
    const c = try Checked.init(gpa, src);
    defer c.deinit(gpa);
    try testing.expectEqualStrings("copied", (try c.holdText("x } }", 0)).?);
    const never = c.facts().captureAt(c.tokOf("x } }\nfn main", 0)).?;
    try testing.expect(never == .unknown);
    try testing.expect(never.word() == null);
}

test "mutArgs: mut args and mut self receivers, named by their callee" {
    const gpa = testing.allocator;
    const src =
        \\struct P { x: int, y: int }
        \\impl P {
        \\    fn bump(mut self, mut d: int) { self.x = self.x + d }
        \\    fn peek(self) -> int { self.x }
        \\}
        \\fn inc(mut n: int, k: int) { n = n + k }
        \\fn main() -> int {
        \\    c := 1
        \\    p := P { x: 1, y: 2 }
        \\    inc(c, 2)
        \\    inc(p.y, c)
        \\    p.bump(c)
        \\    c |> inc(3)
        \\    return c + p.peek()
        \\}
        \\
    ;
    const c = try Checked.init(gpa, src);
    defer c.deinit(gpa);
    const f = c.facts();
    const a = c.arena.allocator();
    const cases = [_]struct { needle: []const u8, skip: u32 = 0, want: ?[]const u8 }{
        .{ .needle = "c, 2)", .want = "inc" },
        .{ .needle = "p.y, c", .want = "inc" },
        .{ .needle = "p.y, c", .skip = 2, .want = "inc" },
        .{ .needle = "p.bump", .want = "bump" },
        .{ .needle = "bump(c)", .skip = 5, .want = "bump" },
        .{ .needle = "c |> inc", .want = "inc" },
        // By-value args, a by-value receiver, and the callee itself.
        .{ .needle = "c)\n    p.bump", .want = null },
        .{ .needle = "2)\n", .want = null },
        .{ .needle = "p.peek", .want = null },
        .{ .needle = "inc(c", .want = null },
    };
    for (cases) |cs| {
        const got = try f.mutatedBy(a, c.tokOf(cs.needle, cs.skip));
        if (cs.want) |w| {
            try testing.expectEqualStrings(w, got orelse {
                std.debug.print("no mut arg at '{s}'+{d}\n", .{ cs.needle, cs.skip });
                return error.NoMutArg;
            });
        } else try testing.expect(got == null);
    }
    var receivers: usize = 0;
    for (try f.mutArgs(a)) |ma| receivers += @intFromBool(ma.receiver);
    try testing.expectEqual(@as(usize, 1), receivers);
}

test "soleBinding: a := or a local fn nothing writes again; not a param, a loop var or a written local" {
    const gpa = testing.allocator;
    const src =
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
        \\fn reset(mut f: fn(int) -> int) { f = fn(v) { v } }
        \\fn main(arg: int) -> int {
        \\    acc := Acc { n: 1 }
        \\    kept := acc.plus
        \\    moved := acc.plus
        \\    moved = acc.plus
        \\    lent := acc.plus
        \\    reset(lent)
        \\    inner := acc.plus
        \\    w := fn() { inner = acc.plus }
        \\    fn step(k: int) -> int { k + arg }
        \\    t := 0
        \\    for i in 0..2 { t = t + i }
        \\    w()
        \\    return kept(1) + moved(1) + lent(1) + inner(1) + step(1) + arg + t
        \\}
        \\fn other() -> int {
        \\    kept := 1
        \\    kept = 2
        \\    return kept
        \\}
        \\
    ;
    const c = try Checked.init(gpa, src);
    defer c.deinit(gpa);
    const f = c.facts();
    const a = c.arena.allocator();
    const m = c.graph.entry();
    const cases = [_]struct { needle: []const u8, binder: ?[]const u8 }{
        // `other` has its own `kept`, whose write does not reach this one.
        .{ .needle = "kept(1)", .binder = "kept :=" },
        .{ .needle = "step(1)", .binder = "step(k" },
        .{ .needle = "moved(1)", .binder = null },
        .{ .needle = "lent(1)", .binder = null },
        // Written inside a closure, which shares the local.
        .{ .needle = "inner(1)", .binder = null },
        .{ .needle = "arg + t", .binder = null },
        .{ .needle = "i }", .binder = null },
    };
    for (cases) |cs| {
        const tok = c.tokOf(cs.needle, 0);
        const ident: u32 = for (m.nodes, 0..) |n, i| {
            if (n.tag == .identifier and n.main_token == tok) break @intCast(i);
        } else return error.NoIdentifier;
        const got = try f.soleBinding(a, ident);
        if (cs.binder) |want| {
            const b = got orelse {
                std.debug.print("no sole binding for '{s}'\n", .{cs.needle});
                return error.NoSoleBinding;
            };
            try testing.expectEqual(c.tokOf(want, 0), m.nodes[b].main_token);
        } else try testing.expect(got == null);
    }
}

test "Hold.merge keeps a detail only when both agree" {
    try testing.expectEqual(Hold{ .copied = 8 }, Hold.merge(.{ .copied = 8 }, .{ .copied = 8 }));
    try testing.expectEqual(Hold{ .copied = null }, Hold.merge(.{ .copied = 8 }, .{ .copied = 16 }));
    try testing.expectEqual(Hold{ .copied = null }, Hold.merge(.{ .copied = null }, .{ .copied = 8 }));
    try testing.expectEqual(Hold.shared, Hold.merge(.boxed, .shared));
    try testing.expectEqual(Hold.boxed, Hold.merge(.boxed, .boxed));
    try testing.expectEqual(Hold.unknown, Hold.merge(.boxed, .{ .copied = 8 }));
    try testing.expectEqual(Hold.by_address, Hold.merge(.by_address, .by_address));
}
