//! IR optimization stage (M13): a small pass manager driving ir→ir transforms
//! between `lower` and `codegen`. Peer to `lower`/`codegen`; reachable through
//! the `toy_compiler` module (src/root.zig). The pipeline becomes
//! lex→parse→resolve→types→lower→OPT→codegen→link.
//!
//! PASSES (each its own file): fold, branch, dce, forward. They run behind this
//! manager's bounded fixpoint so fold→branch→forward→DCE cascade.
//!
//! === MEMORY STRATEGY (no arena; in-place mutate + mark-then-compact-once) ===
//! `Ir.Function.deinit` frees, in order: each block's params; each `.call`
//! instr's `args`; the block's `instrs`; the block's `br.args`; then `blocks`,
//! `params`, `slots`, `values`, `literals.bytes`, `literals`. Every transform
//! MUST preserve that invariant so deinit neither leaks nor double-frees:
//!   (a) FOLD replaces an `Instr.op` with `.iconst`/`.bconst` IN PLACE. Neither
//!       the new op nor the displaced add/icmp/neg/bnot owns a nested slice, so
//!       this is leak-free and allocation-free.
//!   (b) BRANCH-FOLD replaces a `.cond_br` (argless, owns nothing) with
//!       `.br{ dest, args = &.{} }` — a static empty slice; deinit's `gpa.free`
//!       no-ops on a 0-len slice (freeing a NON-empty static slice is the bug).
//!   (c) UNREACHABLE-ELIM frees each DROPPED block's nested slices in deinit
//!       order (each `.call`'s args; then instrs; then params; then `br.args`)
//!       exactly once, then compacts survivors with one alloc-copy-free-old.
//!       Per-block instr DCE compacts with one alloc-copy-free-old too.
//!   (d) VALUE PRUNING rebuilds `func.values` (alloc-copy-free-old) and renumbers
//!       EVERY ValueId site through the shared `walk.zig` helper (so liveness and
//!       remap can't drift). `entry`/`exit` are BlockIds (remapped by the block
//!       remap, never the value remap). Slots are left stable.
//! Use `std.testing.allocator` in every pass unit test to catch leak/double-free.

const std = @import("std");
const Ir = @import("../ir/Ir.zig");

pub const arith = @import("arith.zig");
pub const walk = @import("walk.zig");
pub const fold = @import("fold.zig");
pub const branch = @import("branch.zig");
pub const dce = @import("dce.zig");
pub const forward = @import("forward.zig");

/// The passes, in fixpoint order. The enum's declaration order IS the order the
/// fixpoint loop runs them each round.
pub const Pass = enum(u3) { fold, branch, forward, dce };

/// Which passes are enabled. A `packed struct` over a fixed `u8` so its `bits()`
/// is a stable wire byte for the codegen fingerprint mix — toggling `-O` lands on
/// a different cache key. Field order is locked (do not reorder).
pub const Config = packed struct(u8) {
    fold: bool = false,
    branch: bool = false,
    forward: bool = false,
    dce: bool = false,
    _pad: u4 = 0,

    pub const O0: Config = .{};
    pub const O1: Config = .{ .fold = true, .branch = true, .forward = true, .dce = true };

    pub fn has(self: Config, p: Pass) bool {
        return switch (p) {
            .fold => self.fold,
            .branch => self.branch,
            .forward => self.forward,
            .dce => self.dce,
        };
    }

    pub fn set(self: *Config, p: Pass, on: bool) void {
        switch (p) {
            .fold => self.fold = on,
            .branch => self.branch = on,
            .forward => self.forward = on,
            .dce => self.dce = on,
        }
    }

    /// The fixed wire byte folded into the codegen fingerprint. `_pad` is always
    /// 0 so this is deterministic.
    pub fn bits(self: Config) u8 {
        return @bitCast(self);
    }

    pub fn any(self: Config) bool {
        return self.fold or self.branch or self.forward or self.dce;
    }
};

/// Per-pass counters, summed deterministically across functions and fixpoint
/// rounds (fixed field order). Surfaced via `--opt-stats`.
pub const Stats = struct {
    consts_folded: usize = 0,
    branches_folded: usize = 0,
    blocks_removed: usize = 0,
    instrs_dced: usize = 0,
    values_pruned: usize = 0,
    loads_forwarded: usize = 0,
    rounds: usize = 0,
    ir_instrs_before: usize = 0,
    ir_instrs_after: usize = 0,

    pub fn add(self: *Stats, o: Stats) void {
        self.consts_folded += o.consts_folded;
        self.branches_folded += o.branches_folded;
        self.blocks_removed += o.blocks_removed;
        self.instrs_dced += o.instrs_dced;
        self.values_pruned += o.values_pruned;
        self.loads_forwarded += o.loads_forwarded;
        self.rounds += o.rounds;
        self.ir_instrs_before += o.ir_instrs_before;
        self.ir_instrs_after += o.ir_instrs_after;
    }
};

const PassFn = *const fn (std.mem.Allocator, *Ir.Function, *Stats) error{OutOfMemory}!bool;

/// Dispatch table: Pass → its transform. Indexed by `@intFromEnum(Pass)`.
fn dispatch(p: Pass) PassFn {
    return switch (p) {
        .fold => fold.run,
        .branch => branch.run,
        .forward => forward.run,
        .dce => dce.run,
    };
}

/// Bounded fixpoint cap; documented in the locked design (≤8 rounds).
pub const MAX_ROUNDS = 8;

/// Run the enabled passes on `func` in-place to a bounded fixpoint. Records the
/// dual-metric IR instruction count (before/after) and per-pass counters into
/// `stats`. With no passes enabled this is a pure no-op (the O0==O1 scaffold gate).
pub fn run(gpa: std.mem.Allocator, func: *Ir.Function, cfg: Config, stats: *Stats) error{OutOfMemory}!void {
    stats.ir_instrs_before = Ir.instrCount(func);
    defer stats.ir_instrs_after = Ir.instrCount(func);

    if (!cfg.any()) return;

    // Count rounds actually executed (a round is one full pass over the enabled
    // passes). The loop runs at least once; the final round that reports no change
    // still counts as executed.
    var rounds: usize = 0;
    while (rounds < MAX_ROUNDS) {
        rounds += 1;
        var changed = false;
        inline for (.{ .fold, .branch, .forward, .dce }) |p| {
            if (cfg.has(p)) {
                const c = try dispatch(p)(gpa, func, stats);
                changed = changed or c;
            }
        }
        if (!changed) break;
    }
    stats.rounds = rounds;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test {
    _ = arith;
    _ = walk;
    _ = fold;
    _ = branch;
    _ = dce;
    _ = forward;
}

test "Config.bits is a stable wire byte; O0 != O1" {
    try std.testing.expectEqual(@as(u8, 0), Config.O0.bits());
    try std.testing.expect(Config.O1.bits() != Config.O0.bits());
    // Individual toggles produce distinct bytes.
    var c: Config = .{};
    c.set(.fold, true);
    try std.testing.expect(c.bits() != Config.O0.bits());
    try std.testing.expect(c.has(.fold) and !c.has(.dce));
}

test "run with no passes is a no-op recording instr count" {
    const gpa = std.testing.allocator;
    var blocks = try gpa.alloc(Ir.Block, 1);
    blocks[0] = .{
        .params = try gpa.alloc(Ir.ValueId, 0),
        .instrs = try gpa.alloc(Ir.Instr, 0),
        .term = .{ .ret = .none },
    };
    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "main" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.unit,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = try gpa.alloc(Ir.ValueDef, 0),
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
    };
    defer func.deinit(gpa);

    var st: Stats = .{};
    try run(gpa, &func, .O0, &st);
    try std.testing.expectEqual(@as(usize, 0), st.ir_instrs_before);
    try std.testing.expectEqual(@as(usize, 0), st.ir_instrs_after);
    try std.testing.expectEqual(@as(usize, 0), st.rounds);
}
