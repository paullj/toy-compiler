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

/// Lower adapter `a` to `sym`: param 0 is the env (unused by a `.fn_value` thunk), the
/// rest are the value signature's params, forwarded to a direct call of `a.target`.
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

    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    if (L.exitParamType(&b, a.sig.ret)) |pt| b.ret_param = try b.addParam(exit, pt);
    b.blocks.items[exit].term = if (b.ret_param == Ir.none_value) .{ .ret = .none } else .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    const args = try gpa.alloc(Ir.Operand, a.sig.params.len);
    errdefer gpa.free(args);
    for (a.sig.params, params.items[1..], 0..) |p, sid, i| args[i] = try L.slotOperand(&b, sid, p);
    const result = try L.emitCall(&b, .{ .direct = a.target }, args, a.sig.ret, 0);

    if (!b.termSet()) try L.brTo(&b, exit, if (b.ret_param == Ir.none_value) .none else result);
    return try L.finishFn(&b, gpa, sym, &params, entry, exit);
}

const testing = std.testing;

fn adapterInputs(layouts: []const Typecheck.Layout) L.Inputs {
    return .{
        .tree = undefined, // an adapter reads layouts, never the tree
        .tokens = &.{},
        .source = "",
        .resolutions = &.{},
        .node_types = &.{},
        .callees = &.{},
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
