//! Shared ValueId operand-walk — the SINGLE place that enumerates every site a
//! `ValueId` can appear in a `Function`. DCE liveness-marking, value renumbering,
//! and store→load forwarding's use-rewrite all route through this so they can
//! never drift (a missed site is a silent miscompile).
//!
//! `forEachValueUse` visits every value USE (operands that read a value) and
//! `forEachValueDef` visits every value DEF (the `Instr.result` + block params).
//! Both skip `none_value`. Block ids (entry/exit, br.dest, cond_br.t/.f) and slot
//! operands are NOT values and are intentionally not visited here.
//!
//! The USE enumeration lives in ONE exhaustive-`switch` spine — `eachInstrUseOperand`
//! + `eachTermUseOperand` — that yields a `*ValueId` per operand slot. The three
//! public passes are thin adaptors over it, so the "which fields are value uses"
//! mapping exists once: adding an `Ir.Op`/`Terminator` variant is a compile-time
//! break in exactly one place (it was formerly caught only by the opt-on/opt-off
//! differential at runtime).

const std = @import("std");
const Ir = @import("../ir/Ir.zig");

/// The ONE enumeration of every value-USE operand slot of an instruction: calls
/// `each(ctx, *Ir.ValueId)` on each operand pointer, in the fixed order below
/// (binary lhs before rhs, store addr before val, call args left-to-right).
/// `none_value` slots ARE yielded; callers filter/remap. Slot operands and the
/// unit (`.none`) operand of a call are NOT values → never yielded; block ids
/// live in the terminator. Exhaustive (no `else`) so a new `Op` variant fails to
/// compile here rather than silently escaping the walk.
fn eachInstrUseOperand(op: *Ir.Op, ctx: anytype, comptime each: anytype) void {
    switch (op.*) {
        .iconst, .bconst, .unit, .slot_addr, .cstr_ptr => {},
        .add, .sub, .mul, .sdiv, .udiv, .smod, .umod, .band, .bor, .bxor, .shl, .lshr, .ashr => |*bin| {
            each(ctx, &bin.lhs);
            each(ctx, &bin.rhs);
        },
        .neg, .bnot, .get_tag, .load_byte, .bcompl => |*v| each(ctx, v),
        .icmp => |*c| {
            each(ctx, &c.lhs);
            each(ctx, &c.rhs);
        },
        .field_addr => |*fa| each(ctx, &fa.base),
        .load => |*l| each(ctx, &l.addr),
        .store => |*s| {
            each(ctx, &s.addr);
            each(ctx, &s.val);
        },
        .copy => |*c| {
            each(ctx, &c.dst);
            each(ctx, &c.src);
        },
        .call => |*c| for (c.args) |*a| switch (a.*) {
            .value => |*v| each(ctx, v),
            .slot, .none => {},
        },
    }
}

/// The value-USE operand slots of a terminator (`cond_br` cond, `br`/`ret` value
/// args), yielded as `*Ir.ValueId`. Block-id destinations are NOT values.
/// Exhaustive over `Terminator`, mirroring `eachInstrUseOperand`.
fn eachTermUseOperand(term: *Ir.Terminator, ctx: anytype, comptime each: anytype) void {
    switch (term.*) {
        .br => |*br| for (br.args) |*a| switch (a.*) {
            .value => |*v| each(ctx, v),
            .slot, .none => {},
        },
        .cond_br => |*c| each(ctx, &c.cond),
        .ret => |*o| switch (o.*) {
            .value => |*v| each(ctx, v),
            .slot, .none => {},
        },
        .@"unreachable", .trap, .panic => {},
    }
}

/// Visit every value-USE in the function (operands that read a value), in a
/// deterministic block-id then instruction order. `f` receives each used
/// `ValueId` (never `none_value`).
pub fn forEachValueUse(func: *const Ir.Function, ctx: anytype, comptime f: fn (@TypeOf(ctx), Ir.ValueId) void) void {
    const Adaptor = struct {
        fn each(c: @TypeOf(ctx), p: *Ir.ValueId) void {
            if (p.* != Ir.none_value) f(c, p.*);
        }
    };
    // `blocks`/`instrs` are non-const slices even behind `*const Function`, so the
    // spine's `*ValueId` is valid; this read-only pass simply never writes through it.
    for (func.blocks) |*b| {
        for (b.instrs) |*ins| eachInstrUseOperand(&ins.op, ctx, Adaptor.each);
        eachTermUseOperand(&b.term, ctx, Adaptor.each);
    }
}

/// Set `flags[v]` for an in-bounds `v` — the shared visitor for marking a value
/// used (with `forEachValueUse`) or defined (with `forEachValueDef`).
pub fn markUsed(flags: []bool, v: Ir.ValueId) void {
    if (v < flags.len) flags[v] = true;
}

/// Visit every value-DEF (each instr result + each block param), skipping
/// `none_value` results, in block-id then instruction order.
pub fn forEachValueDef(func: *const Ir.Function, ctx: anytype, comptime f: fn (@TypeOf(ctx), Ir.ValueId) void) void {
    for (func.blocks) |b| {
        for (b.params) |p| if (p != Ir.none_value) f(ctx, p);
        for (b.instrs) |ins| if (ins.result != Ir.none_value) f(ctx, ins.result);
    }
}

/// Rewrite EVERY ValueId site (defs + uses) in-place through `map`, a function
/// from old id to new id. `none_value` is passed through `map` too (callers'
/// maps must preserve it). Used by value-pruning to renumber after compaction.
pub fn remapValues(func: *Ir.Function, ctx: anytype, comptime map: fn (@TypeOf(ctx), Ir.ValueId) Ir.ValueId) void {
    const Adaptor = struct {
        fn each(c: @TypeOf(ctx), p: *Ir.ValueId) void {
            p.* = map(c, p.*);
        }
    };
    for (func.blocks) |*b| {
        // DEF remapping (result + params) stays inline — it is not part of the USE
        // spine (mirrors the forEachValueDef/forEachValueUse split above).
        for (b.params) |*p| p.* = map(ctx, p.*);
        for (b.instrs) |*ins| {
            ins.result = map(ctx, ins.result);
            eachInstrUseOperand(&ins.op, ctx, Adaptor.each);
        }
        eachTermUseOperand(&b.term, ctx, Adaptor.each);
    }
}

/// Rewrite every value-USE site (operands that READ a value) in-place through
/// `map`, leaving DEFS (instr results, block params) untouched — the mutating
/// counterpart of `forEachValueUse`. `none_value` is passed through `map` too
/// (callers' maps must preserve it). Used by store→load forwarding, which
/// redirects readers of a forwarded load WITHOUT redefining anything (rewriting
/// the forwarded load's own result would duplicate a def).
pub fn remapUses(func: *Ir.Function, ctx: anytype, comptime map: fn (@TypeOf(ctx), Ir.ValueId) Ir.ValueId) void {
    const Adaptor = struct {
        fn each(c: @TypeOf(ctx), p: *Ir.ValueId) void {
            p.* = map(c, p.*);
        }
    };
    for (func.blocks) |*b| {
        for (b.instrs) |*ins| eachInstrUseOperand(&ins.op, ctx, Adaptor.each);
        eachTermUseOperand(&b.term, ctx, Adaptor.each);
    }
}

// ---------------------------------------------------------------------------
// Tests — pin the operand-USE spine at its interface. Building one function with
// an instr of every value-bearing op, a value-less op of each shape, and all
// three USE-bearing terminators makes "every op's uses are walked" a single
// enumeration assertion; a new unhandled op fails to COMPILE, a mis-wired one
// fails test #1. `none_value` slots (a call `.none` arg vs a literal `none_value`
// operand) pin the filter-vs-route contract.
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Build a 4-block function exercising every USE-bearing shape, from `a` (an
/// arena — no `Function.deinit` needed). Results occupy 100.., value-operand ids
/// occupy 1.. so a `+1000` remap keeps the two ranges distinguishable. `bnot`
/// takes a literal `none_value` operand to pin `none_value` routing. `values`/
/// `slots` are left empty: the walk never dereferences them.
fn buildUseFixture(a: std.mem.Allocator) !Ir.Function {
    var instrs = try a.alloc(Ir.Instr, 29);
    instrs[0] = .{ .result = 100, .op = .{ .add = .{ .lhs = 1, .rhs = 2 } } };
    instrs[1] = .{ .result = 101, .op = .{ .sub = .{ .lhs = 3, .rhs = 4 } } };
    instrs[2] = .{ .result = 102, .op = .{ .mul = .{ .lhs = 5, .rhs = 6 } } };
    instrs[3] = .{ .result = 103, .op = .{ .sdiv = .{ .lhs = 7, .rhs = 8 } } };
    instrs[4] = .{ .result = 104, .op = .{ .udiv = .{ .lhs = 9, .rhs = 10 } } };
    instrs[5] = .{ .result = 105, .op = .{ .neg = 11 } };
    instrs[6] = .{ .result = 106, .op = .{ .bnot = Ir.none_value } }; // none_value operand
    instrs[7] = .{ .result = 107, .op = .{ .icmp = .{ .cc = .eq, .lhs = 13, .rhs = 14 } } };
    instrs[8] = .{ .result = 108, .op = .{ .field_addr = .{ .base = 15, .off = 8, .ty = Ir.Type.int } } };
    instrs[9] = .{ .result = 109, .op = .{ .load = .{ .addr = 16, .ty = Ir.Type.int } } };
    instrs[10] = .{ .result = 110, .op = .{ .load_byte = 17 } };
    instrs[11] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 18, .val = 19, .ty = Ir.Type.int } } };
    instrs[12] = .{ .result = Ir.none_value, .op = .{ .copy = .{ .dst = 20, .src = 21, .ty = Ir.Type.int } } };
    instrs[13] = .{ .result = 111, .op = .{ .get_tag = 22 } };
    // value-less ops: yield nothing.
    instrs[14] = .{ .result = 112, .op = .{ .iconst = 42 } };
    instrs[15] = .{ .result = 113, .op = .{ .bconst = true } };
    instrs[16] = .{ .result = Ir.none_value, .op = .unit };
    instrs[17] = .{ .result = 114, .op = .{ .slot_addr = 0 } };
    instrs[18] = .{ .result = 115, .op = .{ .cstr_ptr = 0xabc } };
    var call_args = try a.alloc(Ir.Operand, 4);
    call_args[0] = .{ .value = 23 };
    call_args[1] = .{ .slot = 0 };
    call_args[2] = .none;
    call_args[3] = .{ .value = 24 };
    instrs[19] = .{ .result = 116, .op = .{ .call = .{
        .callee = .{ .kind = .user_fn, .name = "g" },
        .args = call_args,
        .ret_slot = Ir.none_slot,
    } } };
    instrs[20] = .{ .result = 117, .op = .{ .band = .{ .lhs = 40, .rhs = 41 } } };
    instrs[21] = .{ .result = 118, .op = .{ .bor = .{ .lhs = 42, .rhs = 43 } } };
    instrs[22] = .{ .result = 119, .op = .{ .bxor = .{ .lhs = 44, .rhs = 45 } } };
    instrs[23] = .{ .result = 120, .op = .{ .shl = .{ .lhs = 46, .rhs = 47 } } };
    instrs[24] = .{ .result = 121, .op = .{ .lshr = .{ .lhs = 48, .rhs = 49 } } };
    instrs[25] = .{ .result = 122, .op = .{ .ashr = .{ .lhs = 50, .rhs = 51 } } };
    instrs[26] = .{ .result = 123, .op = .{ .bcompl = 52 } };
    instrs[27] = .{ .result = 124, .op = .{ .smod = .{ .lhs = 53, .rhs = 54 } } };
    instrs[28] = .{ .result = 125, .op = .{ .umod = .{ .lhs = 55, .rhs = 56 } } };

    var br_args = try a.alloc(Ir.Operand, 3);
    br_args[0] = .{ .value = 30 };
    br_args[1] = .{ .slot = 0 };
    br_args[2] = .none;

    var blocks = try a.alloc(Ir.Block, 4);
    blocks[0] = .{ .params = &.{}, .instrs = instrs, .term = .{ .br = .{ .dest = 1, .args = br_args } } };
    const p1 = try a.alloc(Ir.ValueId, 1);
    p1[0] = 200; // a block param DEF, to prove remapUses leaves it alone
    blocks[1] = .{ .params = p1, .instrs = &.{}, .term = .{ .cond_br = .{ .cond = 31, .t = 2, .f = 3 } } };
    blocks[2] = .{ .params = &.{}, .instrs = &.{}, .term = .{ .ret = .{ .value = 32 } } };
    blocks[3] = .{ .params = &.{}, .instrs = &.{}, .term = .{ .ret = .none } }; // yields nothing

    return .{
        .name = .{ .kind = .user_fn, .name = "f" },
        .params = &.{},
        .ret_type = Ir.Type.int,
        .slots = &.{},
        .values = &.{},
        .blocks = blocks,
        .entry = 0,
        .exit = 2,
    };
}

// Every value operand id present in the fixture, in enumeration order.
const expected_uses = [_]Ir.ValueId{
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // add..udiv
    11, // neg (bnot's none_value operand is skipped by forEachValueUse)
    13, 14, // icmp
    15, // field_addr
    16, // load
    17, // load_byte
    18, 19, // store
    20, 21, // copy
    22, // get_tag
    23, 24, // call (slot + none args skipped)
    40, 41, 42, 43, 44, 45, // band/bor/bxor
    46, 47, 48, 49, 50, 51, // shl/lshr/ashr
    52, // bcompl
    53, 54, 55, 56, // smod/umod
    30, // br arg (slot + none skipped)
    31, // cond_br cond
    32, // ret value
};

const Collector = struct {
    list: *std.ArrayList(Ir.ValueId),
    gpa: std.mem.Allocator,
    fn visit(c: @This(), v: Ir.ValueId) void {
        c.list.append(c.gpa, v) catch @panic("oom");
    }
};

test "forEachValueUse enumerates exactly every value operand, skipping none_value/slot/none" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var func = try buildUseFixture(arena.allocator());

    var got: std.ArrayList(Ir.ValueId) = .empty;
    defer got.deinit(testing.allocator);
    forEachValueUse(&func, Collector{ .list = &got, .gpa = testing.allocator }, Collector.visit);

    try testing.expectEqualSlices(Ir.ValueId, &expected_uses, got.items);
    for (got.items) |v| try testing.expect(v != Ir.none_value);
}

const Shift = struct {
    // none_value-preserving remap: the contract callers must honor.
    fn map(_: void, v: Ir.ValueId) Ir.ValueId {
        return if (v == Ir.none_value) Ir.none_value else v + 1000;
    }
};

test "remapUses rewrites uses only, leaves results + block params untouched" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var func = try buildUseFixture(arena.allocator());

    remapUses(&func, {}, Shift.map);

    // Every value USE shifted by 1000; the none_value operand stayed none_value.
    var got: std.ArrayList(Ir.ValueId) = .empty;
    defer got.deinit(testing.allocator);
    forEachValueUse(&func, Collector{ .list = &got, .gpa = testing.allocator }, Collector.visit);
    try testing.expectEqual(expected_uses.len, got.items.len);
    for (expected_uses, got.items) |want, have| try testing.expectEqual(want + 1000, have);
    try testing.expectEqual(Ir.none_value, func.blocks[0].instrs[6].op.bnot);

    // DEFS untouched by remapUses.
    try testing.expectEqual(@as(Ir.ValueId, 100), func.blocks[0].instrs[0].result);
    try testing.expectEqual(@as(Ir.ValueId, 116), func.blocks[0].instrs[19].result);
    try testing.expectEqual(@as(Ir.ValueId, 200), func.blocks[1].params[0]);
}

test "remapValues rewrites defs and uses" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var func = try buildUseFixture(arena.allocator());

    remapValues(&func, {}, Shift.map);

    // USES shifted.
    var got: std.ArrayList(Ir.ValueId) = .empty;
    defer got.deinit(testing.allocator);
    forEachValueUse(&func, Collector{ .list = &got, .gpa = testing.allocator }, Collector.visit);
    try testing.expectEqual(expected_uses.len, got.items.len);
    for (expected_uses, got.items) |want, have| try testing.expectEqual(want + 1000, have);

    // DEFS shifted; a none_value result stayed none_value.
    try testing.expectEqual(@as(Ir.ValueId, 1100), func.blocks[0].instrs[0].result);
    try testing.expectEqual(@as(Ir.ValueId, 1116), func.blocks[0].instrs[19].result);
    try testing.expectEqual(Ir.none_value, func.blocks[0].instrs[11].result); // store
    try testing.expectEqual(@as(Ir.ValueId, 1200), func.blocks[1].params[0]);
}

const RouteProbe = struct {
    // Identity EXCEPT it rewrites none_value → 777, so a surviving 777 proves the
    // none_value slot was actually routed through `map` (not skipped).
    fn map(_: void, v: Ir.ValueId) Ir.ValueId {
        return if (v == Ir.none_value) 777 else v;
    }
};

test "none_value discipline: forEachValueUse skips it, remap routes it through map" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var func = try buildUseFixture(arena.allocator());

    remapUses(&func, {}, RouteProbe.map);
    try testing.expectEqual(@as(Ir.ValueId, 777), func.blocks[0].instrs[6].op.bnot);

    var func2 = try buildUseFixture(arena.allocator());
    remapValues(&func2, {}, RouteProbe.map);
    try testing.expectEqual(@as(Ir.ValueId, 777), func2.blocks[0].instrs[6].op.bnot);
}
