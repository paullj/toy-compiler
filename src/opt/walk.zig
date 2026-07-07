//! Shared ValueId operand-walk — the SINGLE place that enumerates every site a
//! `ValueId` can appear in a `Function`. DCE liveness-marking, value renumbering,
//! and store→load forwarding's use-rewrite all route through this so they can
//! never drift (a missed site is a silent miscompile, caught only by the
//! differential).
//!
//! `forEachValueUse` visits every value USE (operands that read a value) and
//! `forEachValueDef` visits every value DEF (the `Instr.result` + block params).
//! Both skip `none_value`. Block ids (entry/exit, br.dest, cond_br.t/.f) and slot
//! operands are NOT values and are intentionally not visited here.

const std = @import("std");
const Ir = @import("../ir/Ir.zig");

/// Visit every value-USE in the function (operands that read a value), in a
/// deterministic block-id then instruction order. `f` receives each used
/// `ValueId` (never `none_value`).
pub fn forEachValueUse(func: *const Ir.Function, ctx: anytype, comptime f: fn (@TypeOf(ctx), Ir.ValueId) void) void {
    for (func.blocks) |b| {
        for (b.instrs) |ins| {
            switch (ins.op) {
                .iconst, .bconst, .unit, .slot_addr, .cstr_ptr => {},
                .add, .sub, .mul, .sdiv, .udiv => |bin| {
                    use(ctx, f, bin.lhs);
                    use(ctx, f, bin.rhs);
                },
                .neg, .bnot, .get_tag, .load_byte => |v| use(ctx, f, v),
                .icmp => |c| {
                    use(ctx, f, c.lhs);
                    use(ctx, f, c.rhs);
                },
                .field_addr => |fa| use(ctx, f, fa.base),
                .load => |l| use(ctx, f, l.addr),
                .store => |s| {
                    use(ctx, f, s.addr);
                    use(ctx, f, s.val);
                },
                .copy => |c| {
                    use(ctx, f, c.dst);
                    use(ctx, f, c.src);
                },
                .call => |c| for (c.args) |a| switch (a) {
                    .value => |v| use(ctx, f, v),
                    .slot, .none => {},
                },
            }
        }
        switch (b.term) {
            .br => |br| for (br.args) |a| switch (a) {
                .value => |v| use(ctx, f, v),
                .slot, .none => {},
            },
            .cond_br => |c| use(ctx, f, c.cond),
            .ret => |o| switch (o) {
                .value => |v| use(ctx, f, v),
                .slot, .none => {},
            },
            .@"unreachable", .trap => {},
        }
    }
}

inline fn use(ctx: anytype, comptime f: fn (@TypeOf(ctx), Ir.ValueId) void, v: Ir.ValueId) void {
    if (v != Ir.none_value) f(ctx, v);
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
    for (func.blocks) |*b| {
        for (b.params) |*p| p.* = map(ctx, p.*);
        for (b.instrs) |*ins| {
            ins.result = map(ctx, ins.result);
            switch (ins.op) {
                .iconst, .bconst, .unit, .slot_addr, .cstr_ptr => {},
                .add, .sub, .mul, .sdiv, .udiv => |*bin| {
                    bin.lhs = map(ctx, bin.lhs);
                    bin.rhs = map(ctx, bin.rhs);
                },
                .neg, .bnot, .get_tag, .load_byte => |*v| v.* = map(ctx, v.*),
                .icmp => |*c| {
                    c.lhs = map(ctx, c.lhs);
                    c.rhs = map(ctx, c.rhs);
                },
                .field_addr => |*fa| fa.base = map(ctx, fa.base),
                .load => |*l| l.addr = map(ctx, l.addr),
                .store => |*s| {
                    s.addr = map(ctx, s.addr);
                    s.val = map(ctx, s.val);
                },
                .copy => |*c| {
                    c.dst = map(ctx, c.dst);
                    c.src = map(ctx, c.src);
                },
                .call => |*c| for (c.args) |*a| switch (a.*) {
                    .value => |*v| v.* = map(ctx, v.*),
                    .slot, .none => {},
                },
            }
        }
        switch (b.term) {
            .br => |*br| for (br.args) |*a| switch (a.*) {
                .value => |*v| v.* = map(ctx, v.*),
                .slot, .none => {},
            },
            .cond_br => |*c| c.cond = map(ctx, c.cond),
            .ret => |*o| switch (o.*) {
                .value => |*v| v.* = map(ctx, v.*),
                .slot, .none => {},
            },
            .@"unreachable", .trap => {},
        }
    }
}

/// Rewrite every value-USE site (operands that READ a value) in-place through
/// `map`, leaving DEFS (instr results, block params) untouched — the mutating
/// counterpart of `forEachValueUse`. `none_value` is passed through `map` too
/// (callers' maps must preserve it). Used by store→load forwarding, which
/// redirects readers of a forwarded load WITHOUT redefining anything (rewriting
/// the forwarded load's own result would duplicate a def).
pub fn remapUses(func: *Ir.Function, ctx: anytype, comptime map: fn (@TypeOf(ctx), Ir.ValueId) Ir.ValueId) void {
    for (func.blocks) |*b| {
        for (b.instrs) |*ins| {
            switch (ins.op) {
                .iconst, .bconst, .unit, .slot_addr, .cstr_ptr => {},
                .add, .sub, .mul, .sdiv, .udiv => |*bin| {
                    bin.lhs = map(ctx, bin.lhs);
                    bin.rhs = map(ctx, bin.rhs);
                },
                .neg, .bnot, .get_tag, .load_byte => |*v| v.* = map(ctx, v.*),
                .icmp => |*c| {
                    c.lhs = map(ctx, c.lhs);
                    c.rhs = map(ctx, c.rhs);
                },
                .field_addr => |*fa| fa.base = map(ctx, fa.base),
                .load => |*l| l.addr = map(ctx, l.addr),
                .store => |*s| {
                    s.addr = map(ctx, s.addr);
                    s.val = map(ctx, s.val);
                },
                .copy => |*c| {
                    c.dst = map(ctx, c.dst);
                    c.src = map(ctx, c.src);
                },
                .call => |*c| for (c.args) |*a| switch (a.*) {
                    .value => |*v| v.* = map(ctx, v.*),
                    .slot, .none => {},
                },
            }
        }
        switch (b.term) {
            .br => |*br| for (br.args) |*a| switch (a.*) {
                .value => |*v| v.* = map(ctx, v.*),
                .slot, .none => {},
            },
            .cond_br => |*c| c.cond = map(ctx, c.cond),
            .ret => |*o| switch (o.*) {
                .value => |*v| v.* = map(ctx, v.*),
                .slot, .none => {},
            },
            .@"unreachable", .trap => {},
        }
    }
}
