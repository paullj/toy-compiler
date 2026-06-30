//! PASS a — constant folding + propagation (on SSA values).
//!
//! An SSA value defined by `iconst`/`bconst` is a constant. This pass folds
//! `add`/`sub`/`mul`/`sdiv`/`neg`/`bnot`/`icmp` whose operands are all constant
//! into a single `iconst`/`bconst`, IN PLACE (the op is rewritten; neither the
//! new nor the displaced op owns a nested slice, so this is leak-free and
//! allocation-free — see Opt.zig memory strategy (a)).
//!
//! Exact aarch64 arithmetic comes from `arith.zig` (wrapping add/sub/mul; sdiv
//! /0=0 and INT_MIN/-1=INT_MIN; signed icmp; neg wraps; bnot is logical-not on a
//! bool). Diverging here would break the opt-on≡opt-off differential.
//!
//! Propagation is transitive within a single forward pass: rewriting a def to an
//! `iconst` AND recording it in the indexed `known` table lets later same-pass
//! consumers (which, by SSA def-before-use and monotonic source-order ids, have
//! LARGER value ids) immediately see it as constant. The Opt fixpoint also
//! re-runs the whole pipeline, but a const-only chain collapses in ONE fold pass.
//!
//! CONSERVATIVE: block-param (merge/phi) values are NEVER treated as constant —
//! they are defined by predecessor stores, not by an `iconst` we can see here.
//! Only `iconst`/`bconst` instruction results seed the table; everything else is
//! `.none`. Traversal is strict block-id then instruction order (deterministic,
//! Driver [C11]); no hash-map iteration.

const std = @import("std");
const Ir = @import("../ir/Ir.zig");
const Opt = @import("Opt.zig");
const arith = @import("arith.zig");

/// Indexed known-constant lattice for one value id. `none` = not known constant
/// (or not yet defined / a non-const def). Kept as a flat array indexed by
/// ValueId (NOT a hash map) so traversal stays deterministic.
const Known = union(enum) {
    none,
    int: i64,
    bool: bool,
};

pub fn run(gpa: std.mem.Allocator, func: *Ir.Function, stats: *Opt.Stats) error{OutOfMemory}!bool {
    if (func.values.len == 0) return false;

    const known = try gpa.alloc(Known, func.values.len);
    defer gpa.free(known);
    @memset(known, .none);

    var changed = false;

    // Strict block-id then instruction order. SSA def-before-use holds within
    // straight-line code, and ids are monotonic source-order, so a producer's
    // `known` entry is always set before any consumer with a larger id reads it.
    for (func.blocks) |*b| {
        for (b.instrs) |*ins| {
            const res = ins.result;
            switch (ins.op) {
                .iconst => |v| {
                    if (res != Ir.none_value) known[res] = .{ .int = v };
                },
                .bconst => |v| {
                    if (res != Ir.none_value) known[res] = .{ .bool = v };
                },
                .add, .sub, .mul, .sdiv => |bin| {
                    const l = constInt(known, bin.lhs) orelse continue;
                    const r = constInt(known, bin.rhs) orelse continue;
                    const kind: arith.BinKind = switch (ins.op) {
                        .add => .add,
                        .sub => .sub,
                        .mul => .mul,
                        .sdiv => .sdiv,
                        else => unreachable,
                    };
                    const folded = arith.foldBin(kind, l, r);
                    ins.op = .{ .iconst = folded };
                    if (res != Ir.none_value) known[res] = .{ .int = folded };
                    stats.consts_folded += 1;
                    changed = true;
                },
                .neg => |v| {
                    const x = constInt(known, v) orelse continue;
                    const folded = arith.neg(x);
                    ins.op = .{ .iconst = folded };
                    if (res != Ir.none_value) known[res] = .{ .int = folded };
                    stats.consts_folded += 1;
                    changed = true;
                },
                .bnot => |v| {
                    const x = constBool(known, v) orelse continue;
                    const folded = arith.bnot(x);
                    ins.op = .{ .bconst = folded };
                    if (res != Ir.none_value) known[res] = .{ .bool = folded };
                    stats.consts_folded += 1;
                    changed = true;
                },
                .icmp => |c| {
                    const l = constInt(known, c.lhs) orelse continue;
                    const r = constInt(known, c.rhs) orelse continue;
                    const folded = arith.icmp(c.cc, l, r);
                    ins.op = .{ .bconst = folded };
                    if (res != Ir.none_value) known[res] = .{ .bool = folded };
                    stats.consts_folded += 1;
                    changed = true;
                },
                // Everything else (load/store/copy/call/slot_addr/field_addr/
                // get_tag/cstr_ptr/unit) is not foldable.
                else => {},
            }
        }
    }

    return changed;
}

fn constInt(known: []const Known, v: Ir.ValueId) ?i64 {
    if (v == Ir.none_value or v >= known.len) return null;
    return switch (known[v]) {
        .int => |x| x,
        else => null,
    };
}

fn constBool(known: []const Known, v: Ir.ValueId) ?bool {
    if (v == Ir.none_value or v >= known.len) return null;
    return switch (known[v]) {
        .bool => |x| x,
        else => null,
    };
}

// ---------------------------------------------------------------------------
// Tests — hand-built IR, run under std.testing.allocator (catches leak/double
// -free). Each builds a single-block function so def-before-use holds trivially.
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Build a one-block function owning heap slices for the given values+instrs.
/// Caller owns; `func.deinit(gpa)` frees everything.
fn buildFn(
    gpa: std.mem.Allocator,
    values: []const Ir.ValueDef,
    instrs: []const Ir.Instr,
    term: Ir.Terminator,
) !Ir.Function {
    const v = try gpa.dupe(Ir.ValueDef, values);
    const ins = try gpa.dupe(Ir.Instr, instrs);
    var blocks = try gpa.alloc(Ir.Block, 1);
    blocks[0] = .{
        .params = try gpa.alloc(Ir.ValueId, 0),
        .instrs = ins,
        .term = term,
    };
    return .{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.int,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = v,
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
    };
}

fn expectIconst(ins: Ir.Instr, want: i64) !void {
    try testing.expect(ins.op == .iconst);
    try testing.expectEqual(want, ins.op.iconst);
}

fn expectBconst(ins: Ir.Instr, want: bool) !void {
    try testing.expect(ins.op == .bconst);
    try testing.expectEqual(want, ins.op.bconst);
}

test "fold add of two iconsts -> iconst, with propagation chain in one pass" {
    const gpa = testing.allocator;
    // %0=iconst 2; %1=iconst 3; %2=iconst 4; %3=mul %1,%2; %4=add %0,%3; %5=sub %4,%0
    // => mul=12, add=14, sub=12 (14-2). All fold in one forward pass.
    const values = [_]Ir.ValueDef{.{ .type = Ir.Type.int }} ** 6;
    const instrs = [_]Ir.Instr{
        .{ .result = 0, .op = .{ .iconst = 2 } },
        .{ .result = 1, .op = .{ .iconst = 3 } },
        .{ .result = 2, .op = .{ .iconst = 4 } },
        .{ .result = 3, .op = .{ .mul = .{ .lhs = 1, .rhs = 2 } } },
        .{ .result = 4, .op = .{ .add = .{ .lhs = 0, .rhs = 3 } } },
        .{ .result = 5, .op = .{ .sub = .{ .lhs = 4, .rhs = 0 } } },
    };
    var func = try buildFn(gpa, &values, &instrs, .{ .ret = .{ .value = 5 } });
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    try testing.expectEqual(@as(usize, 3), st.consts_folded);

    const got = func.blocks[0].instrs;
    try expectIconst(got[3], 12);
    try expectIconst(got[4], 14);
    try expectIconst(got[5], 12);
}

test "fold sdiv by zero is 0; INT_MIN/-1 is INT_MIN (matches aarch64, no trap)" {
    const gpa = testing.allocator;
    const min = std.math.minInt(i64);
    // %0=5; %1=0; %2=sdiv %0,%1  ;  %3=INT_MIN; %4=-1; %5=sdiv %3,%4
    const values = [_]Ir.ValueDef{.{ .type = Ir.Type.int }} ** 6;
    const instrs = [_]Ir.Instr{
        .{ .result = 0, .op = .{ .iconst = 5 } },
        .{ .result = 1, .op = .{ .iconst = 0 } },
        .{ .result = 2, .op = .{ .sdiv = .{ .lhs = 0, .rhs = 1 } } },
        .{ .result = 3, .op = .{ .iconst = min } },
        .{ .result = 4, .op = .{ .iconst = -1 } },
        .{ .result = 5, .op = .{ .sdiv = .{ .lhs = 3, .rhs = 4 } } },
    };
    var func = try buildFn(gpa, &values, &instrs, .{ .ret = .{ .value = 5 } });
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    _ = try run(gpa, &func, &st);
    try expectIconst(func.blocks[0].instrs[2], 0);
    try expectIconst(func.blocks[0].instrs[5], min);
}

test "fold neg(INT_MIN) wraps to INT_MIN; wrapping add overflow" {
    const gpa = testing.allocator;
    const min = std.math.minInt(i64);
    const max = std.math.maxInt(i64);
    // %0=INT_MIN; %1=neg %0  ;  %2=MAX; %3=1; %4=add %2,%3 (wraps to INT_MIN)
    const values = [_]Ir.ValueDef{.{ .type = Ir.Type.int }} ** 5;
    const instrs = [_]Ir.Instr{
        .{ .result = 0, .op = .{ .iconst = min } },
        .{ .result = 1, .op = .{ .neg = 0 } },
        .{ .result = 2, .op = .{ .iconst = max } },
        .{ .result = 3, .op = .{ .iconst = 1 } },
        .{ .result = 4, .op = .{ .add = .{ .lhs = 2, .rhs = 3 } } },
    };
    var func = try buildFn(gpa, &values, &instrs, .{ .ret = .{ .value = 4 } });
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    _ = try run(gpa, &func, &st);
    try expectIconst(func.blocks[0].instrs[1], min);
    try expectIconst(func.blocks[0].instrs[4], min);
}

test "fold signed icmp + bnot to bconst" {
    const gpa = testing.allocator;
    // %0=-3; %1=0; %2=icmp lt %0,%1 (true); %3=bnot %2 (false)
    const values = [_]Ir.ValueDef{
        .{ .type = Ir.Type.int },
        .{ .type = Ir.Type.int },
        .{ .type = Ir.Type.bool },
        .{ .type = Ir.Type.bool },
    };
    const instrs = [_]Ir.Instr{
        .{ .result = 0, .op = .{ .iconst = -3 } },
        .{ .result = 1, .op = .{ .iconst = 0 } },
        .{ .result = 2, .op = .{ .icmp = .{ .cc = .lt, .lhs = 0, .rhs = 1 } } },
        .{ .result = 3, .op = .{ .bnot = 2 } },
    };
    var func = try buildFn(gpa, &values, &instrs, .{ .ret = .{ .value = 3 } });
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    _ = try run(gpa, &func, &st);
    try expectBconst(func.blocks[0].instrs[2], true);
    try expectBconst(func.blocks[0].instrs[3], false);
    try testing.expectEqual(@as(usize, 2), st.consts_folded);
}

test "non-const operand is not folded; load result never const" {
    const gpa = testing.allocator;
    // %0=slot_addr s0 (none); %1=load %0 (none); %2=iconst 5; %3=add %1,%2 (NOT folded)
    const values = [_]Ir.ValueDef{.{ .type = Ir.Type.int }} ** 4;
    const instrs = [_]Ir.Instr{
        .{ .result = 0, .op = .{ .slot_addr = 0 } },
        .{ .result = 1, .op = .{ .load = .{ .addr = 0, .ty = Ir.Type.int } } },
        .{ .result = 2, .op = .{ .iconst = 5 } },
        .{ .result = 3, .op = .{ .add = .{ .lhs = 1, .rhs = 2 } } },
    };
    var func = try buildFn(gpa, &values, &instrs, .{ .ret = .{ .value = 3 } });
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(!changed);
    try testing.expectEqual(@as(usize, 0), st.consts_folded);
    try testing.expect(func.blocks[0].instrs[3].op == .add);
}

test "block-param (merge) value is NOT treated as const" {
    const gpa = testing.allocator;
    // Two blocks. b1 has param %0 (a merge value). b1: %1=iconst 7; %2=add %0,%1.
    // %0 must NOT be const, so the add must NOT fold.
    const values = [_]Ir.ValueDef{.{ .type = Ir.Type.int }} ** 3;

    const b0_instrs = try gpa.alloc(Ir.Instr, 0);
    var b1_instrs = try gpa.alloc(Ir.Instr, 2);
    b1_instrs[0] = .{ .result = 1, .op = .{ .iconst = 7 } };
    b1_instrs[1] = .{ .result = 2, .op = .{ .add = .{ .lhs = 0, .rhs = 1 } } };

    var b1_params = try gpa.alloc(Ir.ValueId, 1);
    b1_params[0] = 0;

    var blocks = try gpa.alloc(Ir.Block, 2);
    blocks[0] = .{
        .params = try gpa.alloc(Ir.ValueId, 0),
        .instrs = b0_instrs,
        .term = .{ .br = .{ .dest = 1, .args = try gpa.alloc(Ir.Operand, 0) } },
    };
    blocks[1] = .{
        .params = b1_params,
        .instrs = b1_instrs,
        .term = .{ .ret = .{ .value = 2 } },
    };

    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.int,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = try gpa.dupe(Ir.ValueDef, &values),
        .blocks = blocks,
        .entry = 0,
        .exit = 1,
    };
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(!changed);
    try testing.expect(func.blocks[1].instrs[1].op == .add);
}
