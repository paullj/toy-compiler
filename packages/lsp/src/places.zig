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
//!   * A call inside a generic body is read from the body's own table, which records no
//!     method selection for an unbounded body (only its instances check it), so its `mut`
//!     arguments and receiver are not reported.

const std = @import("std");
const toyc = @import("toy_compiler");
const render = @import("render.zig");

const Ast = toyc.Ast;
const Abi = toyc.Abi;
const Callee = toyc.Callee;
const Closure = toyc.Closure;
const Body = toyc.Body;
const Modes = toyc.Modes;
const Type = toyc.Typecheck.Type;
const Analysis = @import("Analysis.zig");

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

/// How the local `tok` names is held, when `tok` is a capture: by the innermost lambda or
/// local fn around it that captures it, else by the bound method value whose receiver it
/// is. Null for any other token.
pub fn captureAt(an: *const Analysis, tok: u32) ?Hold {
    const m = an.module();
    const rs = an.resolutions();
    for (an.nodesAt(tok)) |i| {
        if (m.nodes[i].tag != .identifier or i >= rs.len) continue;
        const slot = switch (rs[i]) {
            .local => |s| s,
            else => continue,
        };
        if (innermostCapturing(an, tok, slot)) |node| return envHold(an, node, slot);
        if (isBoundReceiver(an, i)) return boundHold(an, tok, i, slot);
    }
    return null;
}

/// Every lambda and local fn in the entry module that captures a local. A `_` capture is
/// left out: each local it captures is spelled in its one call.
pub fn capturing(an: *const Analysis, a: std.mem.Allocator) ![]const Captures {
    const m = an.module();
    const tree = m.tree();
    var out: std.ArrayList(Captures) = .empty;
    for (fnClosures(an)) |fc| for (fc.boundaries) |b| {
        if (b.captures.len == 0 or Ast.isCaptureLambda(tree, m.tokens, b.node)) continue;
        const body = m.nodes[b.node.int()].rhs;
        if (body == Ast.none or m.nodes[body.int()].tag != .block) continue;
        const items = try a.alloc(Item, b.captures.len);
        for (b.captures, items) |c, *it| it.* = .{
            .name = m.tokens[m.nodes[c.first_use.int()].main_token].text(m.source),
            .hold = envHold(an, b.node, c.slot),
        };
        try out.append(a, .{ .node = b.node.int(), .body = m.nodes[body.int()].main_token, .items = items });
    };
    return out.toOwnedSlice(a);
}

/// Every argument a `mut` param takes and every `mut self` receiver in the entry module,
/// in call-node order. `Analysis.mutArgs` keeps the answer.
pub fn mutArgs(an: *const Analysis, a: std.mem.Allocator) ![]const MutArg {
    const m = an.module();
    const tree = m.tree();
    const t = an.tc();
    const body: Body.BodyView = .{ .tree = tree, .resolutions = an.resolutions(), .node_types = an.nodeTypes(), .callees = an.callees() };
    const prog: Body.ProgramView = .{ .sigs = t.sigs, .methods = t.methods, .instances = t.instances };
    var out: std.ArrayList(MutArg) = .empty;
    for (m.nodes, 0..) |n, c| {
        if (n.tag != .call) continue;
        const mask = Modes.argMask(body, prog, Ast.Index.from(@intCast(c)));
        const recv = mutReceiver(an, n);
        if (mask == 0 and recv == null) continue;
        const name = calleeName(an, n) orelse continue;
        if (recv) |r| try out.append(a, .{ .node = r, .callee = name, .receiver = true });
        for (Ast.rangeSlice(tree, n.rhs.int()), 0..) |arg, k| {
            if (Modes.isMut(mask, k)) try out.append(a, .{ .node = arg.int(), .callee = name, .receiver = false });
        }
    }
    return out.toOwnedSlice(a);
}

/// The callee that writes the place `tok` is part of, when that place is a `mut` argument
/// or a `mut self` receiver: its root local or any field on the way to it.
pub fn mutatedBy(an: *Analysis, tok: u32) !?[]const u8 {
    const nodes = an.module().nodes;
    for (try an.mutArgs()) |ma| {
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

/// The binder (a `:=` or a local fn) of the local that the identifier node `ident` names,
/// when nothing in its fn writes that local again: no assignment rooted at it and no `mut`
/// argument rooted at it. Null for a param, a loop or match binding, or a local written
/// after its binding.
pub fn soleBinding(an: *Analysis, ident: u32) !?u32 {
    const m = an.module();
    const rs = an.resolutions();
    if (ident >= rs.len) return null;
    const slot = switch (rs[ident]) {
        .local => |s| s,
        else => return null,
    };
    // Slots number a whole top-level fn, so the fn keeps another fn's same slot out.
    const fn_node = an.fnOf(m.nodes[ident].main_token) orelse return null;
    var binder: ?u32 = null;
    for (m.nodes[0..@min(m.nodes.len, rs.len)], 0..) |n, i| {
        if (an.fnOf(n.main_token) != fn_node) continue;
        switch (n.tag) {
            .var_decl, .local_fn => if (rs[i] == .local and rs[i].local == slot) {
                binder = @intCast(i);
            },
            .assign => if (toyc.Resolve.placeRoot(m.tree(), rs, n.lhs) == slot) return null,
            else => {},
        }
    }
    for (try an.mutArgs()) |ma| {
        if (an.fnOf(m.nodes[ma.node].main_token) == fn_node and toyc.Resolve.placeRoot(m.tree(), rs, Ast.Index.from(ma.node)) == slot) return null;
    }
    return binder;
}

fn fnClosures(an: *const Analysis) []const Closure.FnClosures {
    const all = an.res().closures;
    return if (an.entry() < all.len) all[an.entry()] else &.{};
}

/// The smallest lambda or local fn around `tok` that captures `slot`. Slots number a whole
/// top-level fn, so the span test also keeps another fn's same slot out.
fn innermostCapturing(an: *const Analysis, tok: u32, slot: u32) ?Ast.Index {
    const at = an.module().tokens[tok].start;
    var best: ?Ast.Index = null;
    var best_len: u32 = std.math.maxInt(u32);
    for (fnClosures(an)) |fc| for (fc.boundaries) |b| {
        const holds = for (b.captures) |c| {
            if (c.slot == slot) break true;
        } else false;
        if (!holds) continue;
        const sp = an.span(b.node.int());
        if (at < sp.start or at >= sp.end or sp.end - sp.start >= best_len) continue;
        best = b.node;
        best_len = sp.end - sp.start;
    };
    return best;
}

/// How closure `node` holds `slot`, from the env every lowered unit of it lays out: one
/// for a non-generic owner, one per instance of a generic one.
fn envHold(an: *const Analysis, node: Ast.Index, slot: u32) Hold {
    var hold: ?Hold = null;
    for (an.tc().lambda_units) |u| {
        if (u.mod != an.entry() or u.node != node) continue;
        for (u.env) |es| {
            if (es.slot != slot) continue;
            const h = ownerHold(an, u.owner, slot, sizeOf(an, es.ty));
            hold = if (hold) |prev| prev.merge(h) else h;
        }
    }
    return hold orelse .unknown;
}

/// How `owner` holds `slot` in every build: the compiler's storage without frame storage
/// and with it, merged, so a storage word survives only where both agree. `copy_size` is
/// the bytes a copy takes.
fn ownerHold(an: *const Analysis, owner: Closure.Owner, slot: u32, copy_size: ?u32) Hold {
    const t = an.tc();
    var hold: ?Hold = null;
    for ([_]bool{ false, true }) |frame| {
        const h: Hold = switch (Closure.ownerStorage(t.bodies, t.owner_closures, owner, frame).hold(slot)) {
            .copy => .{ .copied = copy_size },
            .cell => .boxed,
            .frame_addr => .shared,
            .param_addr => .by_address,
        };
        hold = if (hold) |prev| prev.merge(h) else h;
    }
    return hold.?;
}

fn isBoundReceiver(an: *const Analysis, recv: u32) bool {
    for (an.module().nodes, 0..) |n, fa| {
        if (n.tag == .field_access and n.lhs.int() == recv and an.bindsMethod(@intCast(fa))) return true;
    }
    return false;
}

/// How a bound method value holds its local receiver: as its owner holds it, which
/// `Closure.boundKind` turns into its adapter. A `mut` param is held by address even in a
/// generic fn with no instance.
fn boundHold(an: *const Analysis, tok: u32, recv: u32, slot: u32) Hold {
    const fn_node = an.fnOf(tok) orelse return .unknown;
    const t = an.tc();
    const gid = an.gidOf(fn_node) orelse return .unknown;
    if (Modes.isMut(t.sigs[gid].mut_mask, slot)) return .by_address;
    // One body for a non-generic fn, one per instance of a generic one.
    var hold: ?Hold = null;
    for (t.bodies.items, 0..) |b, i| {
        if (b.mod != an.entry() or b.decl.int() != fn_node) continue;
        const h = ownerHold(an, @intCast(i), slot, if (recv < b.node_types.len) sizeOf(an, b.node_types[recv]) else null);
        hold = if (hold) |prev| prev.merge(h) else h;
    }
    return hold orelse .unknown;
}

/// The bytes a copy of a `ty` value takes in an env, as `layoutCaptureEnv` sizes it.
fn sizeOf(an: *const Analysis, ty: Type) ?u32 {
    const t = an.tc();
    switch (ty.kind) {
        .invalid, .type_var, .app, .func => return null,
        .@"struct" => if (ty.struct_id >= t.layouts.len) return null,
        .@"enum" => if (ty.enum_id >= t.enum_layouts.len) return null,
        else => {},
    }
    return Abi.typeSize(ty, t.layouts, t.enum_layouts);
}

/// A `mut self` method call's receiver node.
fn mutReceiver(an: *const Analysis, call: Ast.Node) ?u32 {
    const m = an.module();
    const fa = Callee.memberNode(m.tree(), call) orelse return null;
    const cs = an.callees();
    if (fa.int() >= cs.len) return null;
    const mi = switch (cs[fa.int()].view()) {
        .method => |mi| mi,
        else => return null,
    };
    const t = an.tc();
    if (mi >= t.methods.len) return null;
    const meth = t.methods[mi];
    if (!meth.mut_self or meth.is_static) return null;
    return m.nodes[fa.int()].lhs.int();
}

/// The callee `argMask` read the modes of: a selected method, or a top-level fn.
fn calleeName(an: *const Analysis, call: Ast.Node) ?[]const u8 {
    const tree = an.module().tree();
    const cs = an.callees();
    const t = an.tc();
    const slot = Callee.callSlot(tree, call) orelse return null;
    if (slot.int() >= cs.len) return null;
    const gid = switch (cs[slot.int()].view()) {
        .method => |mi| return if (mi < t.methods.len) t.methods[mi].name else null,
        .direct, .generic => |gid| gid,
        .instance => |ii| if (ii < t.instances.len) t.instances[ii].template_gid else return null,
        else => return null,
    };
    return if (gid < t.sigs.len) render.unmangled(render.bareName(t.sigs[gid].name)) else null;
}

const testing = std.testing;

/// A checked fixture and an arena for the answers, torn down together.
const Checked = struct {
    an: *Analysis,
    arena: std.heap.ArenaAllocator,

    fn init(gpa: std.mem.Allocator, src: []const u8) !Checked {
        const an = try @import("test_util.zig").analyze(gpa, src);
        try testing.expect(an.checked != null);
        return .{ .an = an, .arena = .init(gpa) };
    }

    fn deinit(c: *Checked) void {
        c.arena.deinit();
        c.an.deinit();
    }

    /// The token at the first occurrence of `needle`, plus `skip` bytes.
    fn tokOf(c: *const Checked, needle: []const u8, skip: u32) u32 {
        const m = c.an.module();
        const off: u32 = @intCast(std.mem.indexOf(u8, m.source, needle).? + skip);
        return @import("position.zig").tokenAt(m.tokens, off).?;
    }

    fn holdText(c: *Checked, needle: []const u8, skip: u32) !?[]const u8 {
        const h = captureAt(c.an, c.tokOf(needle, skip)) orelse return null;
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
    var c = try Checked.init(gpa, capture_fixture);
    defer c.deinit();
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
        try testing.expect(captureAt(c.an, c.tokOf(needle, 0)) == null);
    }
}

test "capturing: lambdas and local fns list their captures; a '_' capture does not" {
    const gpa = testing.allocator;
    var c = try Checked.init(gpa, capture_fixture);
    defer c.deinit();
    const a = c.arena.allocator();
    const m = c.an.module();
    var seen: std.ArrayList(u8) = .empty;
    for (try capturing(c.an, a)) |cl| {
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
    var c = try Checked.init(gpa, src);
    defer c.deinit();
    try testing.expectEqualStrings("copied", (try c.holdText("x } }", 0)).?);
    const never = captureAt(c.an, c.tokOf("x } }\nfn main", 0)).?;
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
    var c = try Checked.init(gpa, src);
    defer c.deinit();
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
        const got = try mutatedBy(c.an, c.tokOf(cs.needle, cs.skip));
        if (cs.want) |w| {
            try testing.expectEqualStrings(w, got orelse {
                std.debug.print("no mut arg at '{s}'+{d}\n", .{ cs.needle, cs.skip });
                return error.NoMutArg;
            });
        } else try testing.expect(got == null);
    }
    var receivers: usize = 0;
    for (try c.an.mutArgs()) |ma| receivers += @intFromBool(ma.receiver);
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
    var c = try Checked.init(gpa, src);
    defer c.deinit();
    const m = c.an.module();
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
        const got = try soleBinding(c.an, ident);
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
