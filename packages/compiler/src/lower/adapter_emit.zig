//! `adapter_emit`: the source-less closure-adapter IR emitter. An adapter gives a named
//! callee the `(env, params..)` shape every function value is called with, so a call of
//! a function value never needs to know what it holds.
//!
//! DEPENDENCY IS ONE-WAY: `adapter_emit → lower` (the `derive_emit` pattern). The emitter
//! is pure of its `Adapter` entry and the layouts, so a double-lower is byte-identical.

const std = @import("std");
const L = @import("../lower.zig");
const Ir = @import("../ir/Ir.zig");
const Typecheck = @import("../types.zig");
const Link = @import("../link/Link.zig");
const Closure = @import("../symbols/Closure.zig");
const Diagnostic = @import("../diagnostics/Diagnostic.zig").Diagnostic;

/// Lower adapter `a` to `sym`: param 0 is the env, the rest are the value signature's
/// params, forwarded to a direct call of `a.target`. A `.fn_value` or `.assoc_value` thunk
/// ignores the env; a bound adapter first passes the receiver the env holds as `self`.
pub fn lowerAdapter(
    gpa: std.mem.Allocator,
    in: L.Inputs,
    a: Closure.Adapter,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    var b: L.Builder = .{ .gpa = gpa, .in = in, .ret_type = a.sig.ret, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    try params.append(gpa, try b.addSlot(Typecheck.Type.rawptr));
    for (a.sig.params) |p| try params.append(gpa, try b.addSlot(p));

    const blocks = try L.openBodyBlocks(&b, a.sig.ret);

    const self_n: usize = switch (a.kind) {
        .fn_value, .assoc_value => 0,
        .bound_copy, .bound_box, .bound_ref => 1,
    };
    const args = try gpa.alloc(Ir.Operand, self_n + a.sig.params.len);
    errdefer gpa.free(args);
    if (self_n == 1) args[0] = try receiverOperand(&b, a, params.items[0]);
    for (a.sig.params, params.items[1..], self_n..) |p, sid, i| args[i] = try L.slotOperand(&b, sid, p);
    const result = try L.emitCall(&b, .{ .direct = a.target }, args, a.sig.ret, 0);

    if (!b.termSet()) try L.brTo(&b, blocks.exit, if (b.ret_param == Ir.none_value) .none else result);
    return try L.finishFn(&b, gpa, sym, &params, blocks.entry, blocks.exit);
}

/// The `self` a bound adapter passes: the address of the receiver for a `mut self`
/// method, else its value. The env holds the receiver as its owner held it
/// (`AdapterKind.recvHold`): a copy holds the receiver itself, a cell word points at the
/// boxed local's cell, where the receiver lives, and an address word, an `int`, at the
/// `mut` param it lives in.
fn receiverOperand(b: *L.Builder, a: Closure.Adapter, env_slot: Ir.SlotId) error{OutOfMemory}!Ir.Operand {
    const rawptr = Typecheck.Type.rawptr;
    const env_addr = try b.emit(.{ .slot_addr = env_slot }, Typecheck.Type.int);
    const env = try b.emit(.{ .load = .{ .addr = env_addr, .ty = rawptr } }, rawptr);
    const recv_addr = switch (a.kind.recvHold()) {
        .copy => env,
        .cell => try b.emit(.{ .load = .{ .addr = env, .ty = rawptr } }, rawptr),
        .param_addr, .frame_addr => try b.emit(.{ .load = .{ .addr = env, .ty = Typecheck.Type.int } }, Typecheck.Type.int),
    };
    if (a.mut_self) return .{ .value = recv_addr };
    return try L.addrOperand(b, recv_addr, a.recv);
}

const testing = std.testing;

fn adapterInputs(layouts: []const Typecheck.Layout) L.Inputs {
    return .{
        .tree = undefined, // an adapter reads layouts, never the tree
        .tokens = &.{},
        .source = "",
        .resolutions = &.{},
        .nodes = .{},
        .layouts = layouts,
        .enum_layouts = &.{},
        .names = &.{},
        .methods = &.{},
        .derives = &.{},
        .closure = .none,
    };
}

fn renderFn(func: *const Ir.Function, in: L.Inputs, buf: []u8) ![]const u8 {
    var w = std.Io.Writer.fixed(buf);
    try Ir.render(&w, func, in.layouts, in.enum_layouts);
    return w.buffered();
}

test "a fn-value thunk takes the env first and forwards its params to a direct call" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const in = adapterInputs(&.{});
    const params = [_]Typecheck.Type{ Typecheck.Type.int, Typecheck.Type.bool };
    const a: Closure.Adapter = .{
        .kind = .fn_value,
        .key = 3,
        .name = "add$fnval",
        .target = .{ .kind = .user_fn, .name = "add" },
        .sig = .{ .params = &params, .ret = Typecheck.Type.int },
    };
    var func = try lowerAdapter(gpa, in, a, .{ .kind = .user_fn, .name = a.name }, &diags);
    defer func.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    try testing.expectEqual(@as(usize, 3), func.params.len);
    try testing.expectEqual(Typecheck.Kind.rawptr, func.slots[func.params[0]].type.kind);
    var buf: [2048]u8 = undefined;
    const text = try renderFn(&func, in, &buf);
    try testing.expect(std.mem.indexOf(u8, text, "call @add(%") != null);
    try testing.expect(std.mem.indexOf(u8, text, "ret %") != null);
}

test "a unit-returning thunk returns nothing" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const in = adapterInputs(&.{});
    const a: Closure.Adapter = .{
        .kind = .fn_value,
        .key = 0,
        .name = "tick$fnval",
        .target = .{ .kind = .user_fn, .name = "tick" },
        .sig = .{ .params = &.{}, .ret = Typecheck.Type.unit },
    };
    var func = try lowerAdapter(gpa, in, a, .{ .kind = .user_fn, .name = a.name }, &diags);
    defer func.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), func.params.len);
    var buf: [1024]u8 = undefined;
    const text = try renderFn(&func, in, &buf);
    try testing.expect(std.mem.indexOf(u8, text, "call @tick()") != null);
}

test "a bound adapter passes its env's receiver copy as self, then the params" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const in = adapterInputs(&.{});
    const params = [_]Typecheck.Type{Typecheck.Type.int};
    const a: Closure.Adapter = .{
        .kind = .bound_copy,
        .key = 4,
        .name = "main.int.add3$bound",
        .target = .{ .kind = .user_fn, .name = "main.int.add3" },
        .sig = .{ .params = &params, .ret = Typecheck.Type.int },
        .recv = Typecheck.Type.int,
    };
    var func = try lowerAdapter(gpa, in, a, .{ .kind = .user_fn, .name = a.name }, &diags);
    defer func.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    var buf: [2048]u8 = undefined;
    try testing.expectEqualStrings(
        \\fn main.int.add3$bound(s0, s1) -> int {
        \\  slots: s0:rawptr s1:int
        \\b0:
        \\  %1 = slot_addr s0
        \\  %2 = load %1 : rawptr
        \\  %3 = load %2 : int
        \\  %4 = slot_addr s1
        \\  %5 = load %4 : int
        \\  %6 = call @main.int.add3(%3, %5)
        \\  br b1(%6)
        \\b1(%0:int):
        \\  ret %0
        \\}
        \\
    , try renderFn(&func, in, &buf));
}

test "a boxed adapter reads the receiver through its cell: the address for mut self, else the value" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const in = adapterInputs(&.{});
    const bump: Closure.Adapter = .{
        .kind = .bound_box,
        .key = 1,
        .name = "main.C.bump$boundbox",
        .target = .{ .kind = .user_fn, .name = "main.C.bump" },
        .sig = .{ .params = &.{}, .ret = Typecheck.Type.unit },
        .recv = Typecheck.Type.structT(0),
        .mut_self = true,
    };
    var f1 = try lowerAdapter(gpa, in, bump, .{ .kind = .user_fn, .name = bump.name }, &diags);
    defer f1.deinit(gpa);
    var buf: [2048]u8 = undefined;
    try testing.expectEqualStrings(
        \\fn main.C.bump$boundbox(s0) -> unit {
        \\  slots: s0:rawptr
        \\b0:
        \\  %0 = slot_addr s0
        \\  %1 = load %0 : rawptr
        \\  %2 = load %1 : rawptr
        \\  call @main.C.bump(%2)
        \\  br b1
        \\b1:
        \\  ret
        \\}
        \\
    , try renderFn(&f1, in, &buf));
    const get: Closure.Adapter = .{
        .kind = .bound_box,
        .key = 2,
        .name = "main.int.get$boundbox",
        .target = .{ .kind = .user_fn, .name = "main.int.get" },
        .sig = .{ .params = &.{}, .ret = Typecheck.Type.int },
        .recv = Typecheck.Type.int,
    };
    var f2 = try lowerAdapter(gpa, in, get, .{ .kind = .user_fn, .name = get.name }, &diags);
    defer f2.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    const text = try renderFn(&f2, in, &buf);
    try testing.expect(std.mem.indexOf(u8, text, "%3 = load %2 : rawptr\n  %4 = load %3 : int\n  %5 = call @main.int.get(%4)") != null);
}

test "a bound_ref adapter loads the receiver address from its env as int and passes it as self" {
    const gpa = testing.allocator;
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const in = adapterInputs(&.{});
    const bump: Closure.Adapter = .{
        .kind = .bound_ref,
        .key = 1,
        .name = "main.C.bump$boundref",
        .target = .{ .kind = .user_fn, .name = "main.C.bump" },
        .sig = .{ .params = &.{}, .ret = Typecheck.Type.unit },
        .recv = Typecheck.Type.structT(0),
        .mut_self = true,
    };
    var f1 = try lowerAdapter(gpa, in, bump, .{ .kind = .user_fn, .name = bump.name }, &diags);
    defer f1.deinit(gpa);
    var buf: [2048]u8 = undefined;
    try testing.expectEqualStrings(
        \\fn main.C.bump$boundref(s0) -> unit {
        \\  slots: s0:rawptr
        \\b0:
        \\  %0 = slot_addr s0
        \\  %1 = load %0 : rawptr
        \\  %2 = load %1 : int
        \\  call @main.C.bump(%2)
        \\  br b1
        \\b1:
        \\  ret
        \\}
        \\
    , try renderFn(&f1, in, &buf));
    const get: Closure.Adapter = .{
        .kind = .bound_ref,
        .key = 2,
        .name = "main.int.get$boundref",
        .target = .{ .kind = .user_fn, .name = "main.int.get" },
        .sig = .{ .params = &.{}, .ret = Typecheck.Type.int },
        .recv = Typecheck.Type.int,
    };
    var f2 = try lowerAdapter(gpa, in, get, .{ .kind = .user_fn, .name = get.name }, &diags);
    defer f2.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    const text = try renderFn(&f2, in, &buf);
    try testing.expect(std.mem.indexOf(u8, text, "%3 = load %2 : int\n  %4 = load %3 : int\n  %5 = call @main.int.get(%4)") != null);
}
