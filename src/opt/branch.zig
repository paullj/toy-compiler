//! PASS b — branch folding + unreachable-block elimination.
//!
//! PHASE 1 (branch-fold): a `cond_br cond,t,f` whose `cond` is a constant bool
//! (an SSA value defined by `bconst`) becomes an argless unconditional `br` to the
//! taken target — `t` if true, `f` if false. `cond_br` is argless by design, so
//! the replacement carries no merge args; `br.args` is a static empty slice
//! (`Function.deinit`'s `gpa.free` no-ops on a 0-len slice — Opt.zig strategy (b)).
//!
//! PHASE 2 (unreachable-elim): whenever phase 1 changed a terminator, recompute
//! reachability from `func.entry` (iterate-to-fixpoint over successors, ascending
//! block id for determinism). Drop every block not reachable: free its owned
//! slices through the shared `Ir.Block.freeOwned` (once), then compact survivors
//! with a single alloc-copy-free-old. Rewrite every surviving terminator's block refs and
//! `func.entry`/`func.exit` through an ascending-id block remap.
//!
//! INVARIANTS: only WHOLE unreachable blocks are dropped, so no live predecessor
//! loses a `br.arg` and no block-param/predecessor pairing is broken (`cond_br` is
//! argless; `br.args` reference values/slots, not blocks). The single EXIT block
//! stays reachable in well-formed lowered IR (asserted in tests). Deterministic:
//! all worklists/remaps walk block ids ascending — no hash-map iteration.
//!
//! This pass does NOT re-derive fold's in-pass `known` table; it independently
//! scans `bconst` defs (cheap, deterministic). The Opt fixpoint runs fold BEFORE
//! branch, so an `icmp`/`bnot`-folded `bconst` is already visible here.

const std = @import("std");
const Ir = @import("../ir/Ir.zig");
const Opt = @import("Opt.zig");

pub fn run(gpa: std.mem.Allocator, func: *Ir.Function, stats: *Opt.Stats) error{OutOfMemory}!bool {
    if (func.values.len == 0 and func.blocks.len == 0) return false;

    // Indexed const-bool table by ValueId (NOT a hash map). Only `bconst` defs
    // are recorded; everything else stays null. Re-derived each pass.
    const value_const = try gpa.alloc(?bool, func.values.len);
    defer gpa.free(value_const);
    @memset(value_const, null);
    for (func.blocks) |b| {
        for (b.instrs) |ins| {
            if (ins.result == Ir.none_value) continue;
            switch (ins.op) {
                .bconst => |v| value_const[ins.result] = v,
                else => {},
            }
        }
    }

    var folded_any = false;
    for (func.blocks) |*b| {
        switch (b.term) {
            .cond_br => |c| {
                const taken_const = constBool(value_const, c.cond) orelse continue;
                const taken: Ir.BlockId = if (taken_const) c.t else c.f;
                // Replace argless cond_br with an argless br to the taken target;
                // `br.args` is a static empty slice (deinit's free no-ops on len 0).
                b.term = .{ .br = .{ .dest = taken, .args = &.{} } };
                stats.branches_folded += 1;
                folded_any = true;
            },
            else => {},
        }
    }

    // Run whenever phase 1 folded a branch (a fold can orphan blocks). Also safe
    // to skip entirely when nothing folded — no fold means no new unreachables
    // this pass (lower never emits unreachable blocks that fold didn't create…
    // EXCEPT the lowered `never` merge block, which IS already orphaned. We only
    // run elim after a fold to keep this pass a pure no-op at O0-equivalent and to
    // avoid touching IR the fixpoint didn't change).
    if (!folded_any) return false;

    const dropped = try elimUnreachable(gpa, func, stats);
    return folded_any or dropped;
}

fn constBool(value_const: []const ?bool, v: Ir.ValueId) ?bool {
    if (v == Ir.none_value or v >= value_const.len) return null;
    return value_const[v];
}

/// Successor block ids of a terminator, written into `out` (cap 2), returns count.
fn successors(term: Ir.Terminator, out: *[2]Ir.BlockId) usize {
    return switch (term) {
        .br => |br| blk: {
            out[0] = br.dest;
            break :blk 1;
        },
        .cond_br => |c| blk: {
            out[0] = c.t;
            out[1] = c.f;
            break :blk 2;
        },
        .ret, .@"unreachable" => 0,
    };
}

/// Drop blocks unreachable from `func.entry`. Returns true if any were dropped.
/// Frees dropped blocks' owned slices once, compacts survivors, remaps refs.
fn elimUnreachable(gpa: std.mem.Allocator, func: *Ir.Function, stats: *Opt.Stats) error{OutOfMemory}!bool {
    const n = func.blocks.len;
    if (n == 0) return false;

    const reachable = try gpa.alloc(bool, n);
    defer gpa.free(reachable);
    @memset(reachable, false);

    // Iterate-to-fixpoint reachability from entry, ascending-id sweeps (no queue
    // ordering dependence => deterministic). entry is always reachable.
    if (func.entry < n) reachable[func.entry] = true;
    var changed = true;
    while (changed) {
        changed = false;
        for (func.blocks, 0..) |b, i| {
            if (!reachable[i]) continue;
            var succ: [2]Ir.BlockId = undefined;
            const cnt = successors(b.term, &succ);
            for (succ[0..cnt]) |s| {
                if (s < n and !reachable[s]) {
                    reachable[s] = true;
                    changed = true;
                }
            }
        }
    }

    // Count survivors; bail if all reachable.
    var kept: usize = 0;
    for (reachable) |r| {
        if (r) kept += 1;
    }
    if (kept == n) return false;

    // Build ascending-id remap old->new for survivors (holes removed).
    const remap = try gpa.alloc(Ir.BlockId, n);
    defer gpa.free(remap);
    {
        var next: Ir.BlockId = 0;
        for (reachable, 0..) |r, i| {
            if (r) {
                remap[i] = next;
                next += 1;
            } else {
                remap[i] = Ir.none_block;
            }
        }
    }

    // Free each DROPPED block's owned slices through the shared `freeOwned`.
    for (func.blocks, 0..) |b, i| {
        if (reachable[i]) continue;
        b.freeOwned(gpa);
    }

    // Compact survivors (single alloc-copy-free-old), in old-id order.
    const new_blocks = try gpa.alloc(Ir.Block, kept);
    {
        var j: usize = 0;
        for (func.blocks, 0..) |b, i| {
            if (reachable[i]) {
                new_blocks[j] = b;
                j += 1;
            }
        }
    }
    gpa.free(func.blocks);
    func.blocks = new_blocks;

    // Rewrite surviving terminators' block refs and entry/exit through remap.
    for (func.blocks) |*b| {
        switch (b.term) {
            .br => |*br| br.dest = remap[br.dest],
            .cond_br => |*c| {
                c.t = remap[c.t];
                c.f = remap[c.f];
            },
            .ret, .@"unreachable" => {},
        }
    }
    if (func.entry < n) func.entry = remap[func.entry];
    if (func.exit < n and remap[func.exit] != Ir.none_block) func.exit = remap[func.exit];

    stats.blocks_removed += (n - kept);
    return true;
}

// Tests — hand-built IR under std.testing.allocator (catches leak/double-free).

const testing = std.testing;

fn intVals(gpa: std.mem.Allocator, n: usize) ![]Ir.ValueDef {
    const v = try gpa.alloc(Ir.ValueDef, n);
    for (v) |*x| x.* = .{ .type = Ir.Type.int };
    return v;
}

test "cond_br on bconst true -> br t with empty args" {
    const gpa = testing.allocator;
    // b0: %0=bconst true; cond_br %0,b1,b2
    // b1: ret  ;  b2: ret  (both kept reachable only via the cond_br targets)
    var b0_instrs = try gpa.alloc(Ir.Instr, 1);
    b0_instrs[0] = .{ .result = 0, .op = .{ .bconst = true } };
    var blocks = try gpa.alloc(Ir.Block, 3);
    blocks[0] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = b0_instrs, .term = .{ .cond_br = .{ .cond = 0, .t = 1, .f = 2 } } };
    blocks[1] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .ret = .none } };
    blocks[2] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .ret = .none } };

    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.unit,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = try intVals(gpa, 1),
        .blocks = blocks,
        .entry = 0,
        .exit = 1,
    };
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    try testing.expectEqual(@as(usize, 1), st.branches_folded);
    try testing.expectEqual(@as(usize, 1), st.blocks_removed);
    // b2 (false target) dropped. b0 now br -> (new id of b1). Survivors: b0,b1.
    try testing.expectEqual(@as(usize, 2), func.blocks.len);
    try testing.expect(func.blocks[0].term == .br);
    // b1 was id 1, b0 kept id 0, so b1 -> 1. The br must point at it.
    try testing.expectEqual(@as(Ir.BlockId, 1), func.blocks[0].term.br.dest);
    try testing.expectEqual(@as(usize, 0), func.blocks[0].term.br.args.len);
    try testing.expectEqual(@as(Ir.BlockId, 1), func.exit);
}

test "cond_br on bconst false -> br f, drops the true target" {
    const gpa = testing.allocator;
    var b0_instrs = try gpa.alloc(Ir.Instr, 1);
    b0_instrs[0] = .{ .result = 0, .op = .{ .bconst = false } };
    var blocks = try gpa.alloc(Ir.Block, 3);
    blocks[0] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = b0_instrs, .term = .{ .cond_br = .{ .cond = 0, .t = 1, .f = 2 } } };
    blocks[1] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .ret = .none } };
    blocks[2] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .ret = .none } };

    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.unit,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = try intVals(gpa, 1),
        .blocks = blocks,
        .entry = 0,
        .exit = 2,
    };
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    // true target b1 dropped; b2 was the false target, kept. Survivors b0,b2.
    try testing.expectEqual(@as(usize, 2), func.blocks.len);
    try testing.expect(func.blocks[0].term == .br);
    // b2 (old id 2) remaps to new id 1 (b0=0, b1 dropped, b2->1).
    try testing.expectEqual(@as(Ir.BlockId, 1), func.blocks[0].term.br.dest);
    try testing.expectEqual(@as(Ir.BlockId, 1), func.exit);
}

test "unreachable-elim drops + remaps a 4-block function" {
    const gpa = testing.allocator;
    // b0: %0=bconst true; cond_br %0,b1,b2
    // b1: br b3
    // b2: br b3   (becomes unreachable after fold)
    // b3: ret
    var b0_instrs = try gpa.alloc(Ir.Instr, 1);
    b0_instrs[0] = .{ .result = 0, .op = .{ .bconst = true } };
    var blocks = try gpa.alloc(Ir.Block, 4);
    blocks[0] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = b0_instrs, .term = .{ .cond_br = .{ .cond = 0, .t = 1, .f = 2 } } };
    blocks[1] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .br = .{ .dest = 3, .args = try gpa.alloc(Ir.Operand, 0) } } };
    blocks[2] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .br = .{ .dest = 3, .args = try gpa.alloc(Ir.Operand, 0) } } };
    blocks[3] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .ret = .none } };

    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.unit,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = try intVals(gpa, 1),
        .blocks = blocks,
        .entry = 0,
        .exit = 3,
    };
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    // b2 unreachable after folding to br b1; survivors b0,b1,b3 (3 blocks).
    try testing.expectEqual(@as(usize, 1), st.blocks_removed);
    try testing.expectEqual(@as(usize, 3), func.blocks.len);
    // b0 -> br b1 (b1 keeps new id 1). b1 -> br b3 (b3 remaps 3->2).
    try testing.expectEqual(@as(Ir.BlockId, 1), func.blocks[0].term.br.dest);
    try testing.expectEqual(@as(Ir.BlockId, 2), func.blocks[1].term.br.dest);
    try testing.expectEqual(@as(Ir.BlockId, 2), func.exit);
}

test "no-op when cond is not a constant" {
    const gpa = testing.allocator;
    // b0: %0=slot_addr s0; %1=load %0; cond_br %1,b1,b2 — %1 not const.
    var b0_instrs = try gpa.alloc(Ir.Instr, 2);
    b0_instrs[0] = .{ .result = 0, .op = .{ .slot_addr = 0 } };
    b0_instrs[1] = .{ .result = 1, .op = .{ .load = .{ .addr = 0, .ty = Ir.Type.bool } } };
    var blocks = try gpa.alloc(Ir.Block, 3);
    blocks[0] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = b0_instrs, .term = .{ .cond_br = .{ .cond = 1, .t = 1, .f = 2 } } };
    blocks[1] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .ret = .none } };
    blocks[2] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .ret = .none } };

    var slots = try gpa.alloc(Ir.Slot, 1);
    slots[0] = .{ .type = Ir.Type.bool };
    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.unit,
        .slots = slots,
        .values = try intVals(gpa, 2),
        .blocks = blocks,
        .entry = 0,
        .exit = 1,
    };
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(!changed);
    try testing.expectEqual(@as(usize, 0), st.branches_folded);
    try testing.expect(func.blocks[0].term == .cond_br);
    try testing.expectEqual(@as(usize, 3), func.blocks.len);
}
