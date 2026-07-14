//! PASS c — dead-instruction + dead-value elimination.
//!
//! PHASE 1 (dead-instr elim): an instruction is droppable iff its result is
//! UNUSED *and* its op is side-effect-free (PURE). PURE = iconst, bconst, unit,
//! add, sub, mul, sdiv, neg, bnot, icmp, slot_addr, field_addr, load, get_tag,
//! cstr_ptr (a load is a pure read — safe to drop when dead). NEVER drop `call`
//! (may print / have effects), `store`, or `copy`, regardless of result use.
//! Usage is computed by the SHARED `walk.zig` operand-walk (slot operands don't
//! mark values). We mark-then-compact per block, repeating to a local fixpoint so
//! chains collapse (dropping a dead `load` makes its `slot_addr` dead next round).
//! Each block compacts with one alloc-copy-free-old.
//!
//! PHASE 2 (dead-value pruning — the frame/machine-code win): after phase 1, the
//! live value set is { every surviving instr's result } ∪ { every block param }.
//! Rebuild `func.values` to the live values only (alloc-copy-free-old), build an
//! ascending-old-id remap (preserves monotonic source-order determinism),
//! and rewrite EVERY ValueId site through `walk.remapValues` (shared with liveness
//! so they cannot drift). `none_value` maps to itself. `entry`/`exit` are BlockIds
//! — untouched here. Slots are left stable (no dead-slot pass — out of scope).
//! Shrinking `func.values.len` makes FrameLayout allocate fewer cells → fewer
//! emitted prologue/marshal/load/store words (and can dodge FrameTooLarge).
//!
//! Block params are ALWAYS kept live (they are merge/phi slots written by
//! predecessor stores; dropping one would desync the edge). Determinism: every
//! traversal is block-id then instruction order; no hash-map iteration.

const std = @import("std");
const Ir = @import("../ir/Ir.zig");
const Opt = @import("Opt.zig");
const walk = @import("walk.zig");

pub fn run(gpa: std.mem.Allocator, func: *Ir.Function, stats: *Opt.Stats) error{OutOfMemory}!bool {
    if (func.values.len == 0) return false;

    const used = try gpa.alloc(bool, func.values.len);
    defer gpa.free(used);

    var any_instr_dropped = false;
    // Bounded by total instr count: each round drops ≥1 instr or stops.
    while (true) {
        // Recompute usage from the current instrs/terminators.
        @memset(used, false);
        walk.forEachValueUse(func, used, walk.markUsed);

        var round_changed = false;
        for (func.blocks) |*b| {
            // Count survivors first.
            var kept: usize = 0;
            for (b.instrs) |ins| {
                if (instrLive(ins, used)) kept += 1;
            }
            if (kept == b.instrs.len) continue; // nothing dead in this block

            const new_instrs = try gpa.alloc(Ir.Instr, kept);
            var j: usize = 0;
            for (b.instrs) |ins| {
                if (instrLive(ins, used)) {
                    new_instrs[j] = ins;
                    j += 1;
                } else {
                    // Dropped a pure-unused instr. PURE ops own no nested slice
                    // (call is never droppable), so nothing to free on the instr.
                    stats.instrs_dced += 1;
                }
            }
            gpa.free(b.instrs);
            b.instrs = new_instrs;
            round_changed = true;
            any_instr_dropped = true;
        }
        if (!round_changed) break;
    }

    const pruned = try pruneValues(gpa, func, stats);
    return any_instr_dropped or pruned;
}

/// An instr survives iff it produces no value, OR its result is used, OR its op
/// has a side effect (call/store/copy are never dropped).
fn instrLive(ins: Ir.Instr, used: []const bool) bool {
    if (hasSideEffect(ins.op)) return true;
    if (ins.result == Ir.none_value) return true; // a pure op with no result: keep (shouldn't occur)
    return used[ins.result];
}

fn hasSideEffect(op: Ir.Op) bool {
    return switch (op) {
        .call, .call_indirect, .store, .store_byte, .copy => true,
        else => false,
    };
}

/// Rebuild `func.values` to only the live values (results of surviving instrs +
/// all block params), remapping every ValueId site. Returns true if any pruned.
fn pruneValues(gpa: std.mem.Allocator, func: *Ir.Function, stats: *Opt.Stats) error{OutOfMemory}!bool {
    const old_n = func.values.len;
    if (old_n == 0) return false;

    const live = try gpa.alloc(bool, old_n);
    defer gpa.free(live);
    @memset(live, false);
    // Defs that survive: every surviving instr result + every block param.
    walk.forEachValueDef(func, live, walk.markUsed);

    var kept: usize = 0;
    for (live) |l| {
        if (l) kept += 1;
    }
    if (kept == old_n) return false;

    // Ascending-old-id remap; dead ids map to none_value (must never be reached
    // by a surviving site — guaranteed because only live defs survive and uses of
    // a dead value were the dropped instrs).
    const remap = try gpa.alloc(Ir.ValueId, old_n);
    defer gpa.free(remap);
    {
        var next: Ir.ValueId = 0;
        for (live, 0..) |l, i| {
            if (l) {
                remap[i] = next;
                next += 1;
            } else {
                remap[i] = Ir.none_value;
            }
        }
    }

    // Rebuild values (alloc-copy-free-old), in old-id order.
    const new_values = try gpa.alloc(Ir.ValueDef, kept);
    {
        var j: usize = 0;
        for (func.values, 0..) |vd, i| {
            if (live[i]) {
                new_values[j] = vd;
                j += 1;
            }
        }
    }
    gpa.free(func.values);
    func.values = new_values;

    // Renumber every ValueId site through the shared walk. none_value -> none_value.
    walk.remapValues(func, remap, remapFn);

    stats.values_pruned += (old_n - kept);
    return true;
}

fn remapFn(remap: []const Ir.ValueId, v: Ir.ValueId) Ir.ValueId {
    if (v == Ir.none_value) return Ir.none_value;
    return remap[v];
}

// Tests — hand-built IR under std.testing.allocator (catches leak/double-free).

const testing = std.testing;

fn intVals(gpa: std.mem.Allocator, n: usize) ![]Ir.ValueDef {
    const v = try gpa.alloc(Ir.ValueDef, n);
    for (v) |*x| x.* = .{ .type = Ir.Type.int };
    return v;
}

fn oneBlock(gpa: std.mem.Allocator, instrs: []Ir.Instr, term: Ir.Terminator, nvals: usize) !Ir.Function {
    var blocks = try gpa.alloc(Ir.Block, 1);
    blocks[0] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = instrs, .term = term };
    return .{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.int,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = try intVals(gpa, nvals),
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
    };
}

test "unused iconst is dropped; value pruned + ret remapped" {
    const gpa = testing.allocator;
    // %0=iconst 9 (dead); %1=iconst 5 (returned). Drop %0 -> values shrink to 1,
    // %1 renumbers to %0, ret follows.
    var instrs = try gpa.alloc(Ir.Instr, 2);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 9 } };
    instrs[1] = .{ .result = 1, .op = .{ .iconst = 5 } };
    var func = try oneBlock(gpa, instrs, .{ .ret = .{ .value = 1 } }, 2);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    try testing.expectEqual(@as(usize, 1), st.instrs_dced);
    try testing.expectEqual(@as(usize, 1), st.values_pruned);
    try testing.expectEqual(@as(usize, 1), func.blocks[0].instrs.len);
    try testing.expectEqual(@as(usize, 1), func.values.len);
    // surviving iconst 5 renumbered to result %0; ret %0.
    try testing.expectEqual(@as(Ir.ValueId, 0), func.blocks[0].instrs[0].result);
    try testing.expectEqual(@as(i64, 5), func.blocks[0].instrs[0].op.iconst);
    try testing.expectEqual(@as(Ir.ValueId, 0), func.blocks[0].term.ret.value);
}

test "dead chain (load + its slot_addr) both dropped" {
    const gpa = testing.allocator;
    // %0=slot_addr s0; %1=load %0 (dead); %2=iconst 7 (returned).
    // First round drops the load (unused), exposing slot_addr as dead next round.
    var instrs = try gpa.alloc(Ir.Instr, 3);
    instrs[0] = .{ .result = 0, .op = .{ .slot_addr = 0 } };
    instrs[1] = .{ .result = 1, .op = .{ .load = .{ .addr = 0, .ty = Ir.Type.int } } };
    instrs[2] = .{ .result = 2, .op = .{ .iconst = 7 } };
    var slots = try gpa.alloc(Ir.Slot, 1);
    slots[0] = .{ .type = Ir.Type.int };
    var blocks = try gpa.alloc(Ir.Block, 1);
    blocks[0] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = instrs, .term = .{ .ret = .{ .value = 2 } } };
    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.int,
        .slots = slots,
        .values = try intVals(gpa, 3),
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
    };
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    try testing.expectEqual(@as(usize, 2), st.instrs_dced); // load + slot_addr
    try testing.expectEqual(@as(usize, 1), func.blocks[0].instrs.len);
    try testing.expectEqual(@as(usize, 1), func.values.len);
    try testing.expectEqual(@as(i64, 7), func.blocks[0].instrs[0].op.iconst);
}

test "call/store/copy are KEPT even with unused result" {
    const gpa = testing.allocator;
    // %0=slot_addr s0; %1=iconst 3; store %0,%1; %2=slot_addr s1;
    // call @f() -> (void, result none); copy %2 <- %0.
    // store/copy/call must all survive; slot_addrs feeding them survive too.
    var instrs = try gpa.alloc(Ir.Instr, 5);
    instrs[0] = .{ .result = 0, .op = .{ .slot_addr = 0 } };
    instrs[1] = .{ .result = 1, .op = .{ .iconst = 3 } };
    instrs[2] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 0, .val = 1, .ty = Ir.Type.int } } };
    instrs[3] = .{ .result = 2, .op = .{ .slot_addr = 1 } };
    instrs[4] = .{ .result = Ir.none_value, .op = .{ .call = .{ .callee = .{ .kind = .user_fn, .name = "f" }, .args = try gpa.alloc(Ir.Operand, 0), .ret_slot = Ir.none_slot } } };

    var slots = try gpa.alloc(Ir.Slot, 2);
    slots[0] = .{ .type = Ir.Type.int };
    slots[1] = .{ .type = Ir.Type.int };
    var blocks = try gpa.alloc(Ir.Block, 1);
    blocks[0] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = instrs, .term = .{ .ret = .none } };
    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.unit,
        .slots = slots,
        .values = try intVals(gpa, 3),
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
    };
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    // %2 (slot_addr s1) is unused (call has no args) -> dropped. store/call kept.
    // %0,%1 used by store. So 1 instr dropped, 1 value pruned.
    try testing.expect(changed);
    try testing.expectEqual(@as(usize, 1), st.instrs_dced);
    try testing.expectEqual(@as(usize, 4), func.blocks[0].instrs.len);
    // store and call still present.
    var has_store = false;
    var has_call = false;
    for (func.blocks[0].instrs) |ins| {
        if (ins.op == .store) has_store = true;
        if (ins.op == .call) has_call = true;
    }
    try testing.expect(has_store and has_call);
}

test "block param is kept live + remapped; none_value preserved" {
    const gpa = testing.allocator;
    // b0: %1=iconst 9 (DEAD); br b1(%2-via-store?) — keep it simple:
    // b0: %1=iconst 9 (dead); %2=iconst 4 (br arg); br b1(%2)
    // b1(%0): ret %0
    var b0_instrs = try gpa.alloc(Ir.Instr, 2);
    b0_instrs[0] = .{ .result = 1, .op = .{ .iconst = 9 } };
    b0_instrs[1] = .{ .result = 2, .op = .{ .iconst = 4 } };
    var br_args = try gpa.alloc(Ir.Operand, 1);
    br_args[0] = .{ .value = 2 };
    var b1_params = try gpa.alloc(Ir.ValueId, 1);
    b1_params[0] = 0;

    var blocks = try gpa.alloc(Ir.Block, 2);
    blocks[0] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = b0_instrs, .term = .{ .br = .{ .dest = 1, .args = br_args } } };
    blocks[1] = .{ .params = b1_params, .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .ret = .{ .value = 0 } } };

    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.int,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = try intVals(gpa, 3),
        .blocks = blocks,
        .entry = 0,
        .exit = 1,
    };
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    // %1 (iconst 9) dead -> dropped. Live values: %0 (param), %2 (br arg).
    // Remap ascending: %0->0, %2->1. values.len = 2.
    try testing.expectEqual(@as(usize, 1), st.instrs_dced);
    try testing.expectEqual(@as(usize, 1), st.values_pruned);
    try testing.expectEqual(@as(usize, 2), func.values.len);
    // b1 param renumbers %0 -> %0 (still 0). ret %0.
    try testing.expectEqual(@as(Ir.ValueId, 0), func.blocks[1].params[0]);
    try testing.expectEqual(@as(Ir.ValueId, 0), func.blocks[1].term.ret.value);
    // surviving iconst 4 result %2 -> %1; br arg follows.
    try testing.expectEqual(@as(Ir.ValueId, 1), func.blocks[0].instrs[0].result);
    try testing.expectEqual(@as(Ir.ValueId, 1), func.blocks[0].term.br.args[0].value);
}

test "no-op when nothing is dead" {
    const gpa = testing.allocator;
    // %0=iconst 1; %1=iconst 2; %2=add %0,%1; ret %2 — all live.
    var instrs = try gpa.alloc(Ir.Instr, 3);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 1 } };
    instrs[1] = .{ .result = 1, .op = .{ .iconst = 2 } };
    instrs[2] = .{ .result = 2, .op = .{ .add = .{ .lhs = 0, .rhs = 1 } } };
    var func = try oneBlock(gpa, instrs, .{ .ret = .{ .value = 2 } }, 3);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(!changed);
    try testing.expectEqual(@as(usize, 0), st.instrs_dced);
    try testing.expectEqual(@as(usize, 3), func.values.len);
}
