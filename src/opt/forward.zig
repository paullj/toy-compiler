//! PASS d — store→load forwarding (the memory-heavy-IR analog of copy-prop, and
//! where the real wins are: lowered locals are slots, so everything is store/load
//! traffic). Forward the value of a `store(addr=A, val=V)` into a later
//! `load(addr=A')` when A and A' provably name the SAME slot AND nothing clobbers
//! that slot between them. THE HIGHEST-RISK PASS — the alias rule is deliberately
//! CONSERVATIVE; when in ANY doubt, do NOT forward. The opt-on≡opt-off differential
//! is the backstop.
//!
//! INTRA-BLOCK ONLY (first cut): `avail` is reset at every block boundary. We never
//! reason across edges, so a store in one block can never be forwarded into a load
//! in another.
//!
//! ABSTRACT ADDRESS. `resolve(ptr)` traces a ptr value's defining op:
//!   slot_addr(S)            -> .slot{ S, off = 0 }
//!   field_addr{base,off}    -> resolve(base); if that is .slot{S,k} -> .slot{S,k+off}
//!                              else .unknown
//!   anything else           -> .unknown  (a load result, a call result, a param,
//!                              a block-param ptr, …)
//! MUST-ALIAS(A,A') := both .slot, SAME S, SAME off.
//!
//! ESCAPE. A slot's address ESCAPES (and is then PERMANENTLY un-forwardable) if a
//! ptr value resolving to that slot is ever used as a `.call` arg, a `store.val`,
//! or a `copy.src` — i.e. the address (not the loaded scalar) leaves our analysis.
//! `escaped[S]` is computed in a function-wide pre-pass and is conservative.
//!
//! CLOBBER (within a block, walking instrs in order):
//!   store{addr,val}: a=resolve(addr).
//!       .slot{S,off} && !escaped[S]  -> avail[{S,off}] = val   (record)
//!       .unknown                     -> clear ALL avail        (might write anywhere)
//!   load{addr}: a=resolve(addr).
//!       .slot{S,off} && !escaped[S] && avail has {S,off}=V -> FORWARD: rewrite every
//!           use of this load's result to V (SSA: defined once). The load (and its
//!           feeder slot_addr/field_addr) then go dead -> DCE prunes them next round
//!           -> the real dual-metric win (fewer values, smaller frame, fewer words).
//!   call:  clear ALL avail (may write through an escaped addr / have effects).
//!   copy{dst}: dst resolves .slot{S,*} -> clear every avail key for slot S (an
//!       aggregate write touches the whole slot). dst .unknown -> clear ALL.
//!
//! MEMORY: this pass only REWRITES ValueId operands in place (no slice grows/shrinks);
//! the now-dead load + its feeders are left for DCE to compact. Zero alloc on the
//! Function's owned slices besides scratch we free. Deterministic: blocks in id
//! order, instrs in order, `avail` is a flat indexed array keyed by (slot,off) —
//! never a hash-map iteration.

const std = @import("std");
const Ir = @import("../ir/Ir.zig");
const Opt = @import("Opt.zig");
const walk = @import("walk.zig");

const AbstractAddr = union(enum) {
    /// Names slot S at byte offset `off`.
    slot: struct { s: Ir.SlotId, off: u32 },
    /// Could name anything — never forward through / into this.
    unknown,
};

/// One available store: the slot+offset it wrote and the value it stored.
const Avail = struct {
    s: Ir.SlotId,
    off: u32,
    val: Ir.ValueId,
    live: bool,
};

pub fn run(gpa: std.mem.Allocator, func: *Ir.Function, stats: *Opt.Stats) error{OutOfMemory}!bool {
    if (func.values.len == 0) return false;

    // def table: ValueId -> its defining Op (for resolve())
    // SSA: each value is defined once. We only need the op kind to trace ptrs, so
    // store an optional op per value id. Indexed (no hash map) -> deterministic.
    const def = try gpa.alloc(?Ir.Op, func.values.len);
    defer gpa.free(def);
    @memset(def, null);
    for (func.blocks) |b| {
        for (b.instrs) |ins| {
            if (ins.result != Ir.none_value and ins.result < def.len) def[ins.result] = ins.op;
        }
    }

    // escape pre-pass: any slot whose ADDRESS leaves our analysis
    const escaped = try gpa.alloc(bool, func.slots.len);
    defer gpa.free(escaped);
    @memset(escaped, false);
    markEscapes(func, def, escaped);

    // forward, block by block (intra-block only)
    // `remap[v]` (default identity) rewrites every value use after we forward. We
    // forward inside the same block then apply remap to the WHOLE function (the
    // load's result can only be used at >= its id, and only after this block, but a
    // function-wide apply is simplest and correct: a value is defined once).
    const remap = try gpa.alloc(Ir.ValueId, func.values.len);
    defer gpa.free(remap);
    for (remap, 0..) |*r, i| r.* = @intCast(i);

    // A load is only worth forwarding if its result is actually USED. Without this
    // guard the pass never reaches a fixpoint: it leaves the (now-unused) load in
    // place for DCE, so a forward-only pipeline would re-forward the same load every
    // round forever (capping at MAX_ROUNDS). Computing usage once and skipping
    // already-dead loads makes the pass idempotent → it reports no change once every
    // forwardable load has been rewritten.
    const used = try gpa.alloc(bool, func.values.len);
    defer gpa.free(used);
    @memset(used, false);
    walk.forEachValueUse(func, used, walk.markUsed);

    // Scratch avail list, reused per block (cleared at each boundary). Bounded by
    // the number of stores in the block; grown as needed.
    var avail: std.ArrayListUnmanaged(Avail) = .empty;
    defer avail.deinit(gpa);

    var forwarded_any = false;
    for (func.blocks) |*b| {
        avail.clearRetainingCapacity();
        for (b.instrs) |ins| {
            switch (ins.op) {
                .store => |s| {
                    switch (resolve(def, s.addr)) {
                        .slot => |sl| {
                            if (sl.s < escaped.len and escaped[sl.s]) {
                                // Escaped slot: do not record (cannot trust must-alias).
                                continue;
                            }
                            recordStore(gpa, &avail, sl.s, sl.off, applyRemap(remap, s.val)) catch return error.OutOfMemory;
                        },
                        .unknown => clearAll(&avail),
                    }
                },
                .load => |l| {
                    if (ins.result == Ir.none_value) continue;
                    // Skip a load whose result is already unused (e.g. forwarded in a
                    // prior round). Keeps the pass idempotent / convergent.
                    if (ins.result < used.len and !used[ins.result]) continue;
                    switch (resolve(def, l.addr)) {
                        .slot => |sl| {
                            if (sl.s < escaped.len and escaped[sl.s]) continue;
                            if (lookup(&avail, sl.s, sl.off)) |v| {
                                // Forward: the load's result becomes the stored value.
                                // Apply remap to v in case the stored value was itself
                                // a previously-forwarded id (keeps the chain coherent).
                                remap[ins.result] = applyRemap(remap, v);
                                stats.loads_forwarded += 1;
                                forwarded_any = true;
                            }
                        },
                        .unknown => {}, // unknown load addr: nothing to forward.
                    }
                },
                .call, .call_indirect => clearAll(&avail), // may write through an escaped addr / effects.
                .copy => |c| {
                    switch (resolve(def, c.dst)) {
                        .slot => |sl| clearSlot(&avail, sl.s), // whole-aggregate write.
                        .unknown => clearAll(&avail),
                    }
                },
                else => {},
            }
        }
    }

    if (!forwarded_any) return false;

    // Apply the remap to every value USE (operands that read a value) — NOT to defs
    // (instr.result, block params). Rewriting the forwarded load's own result would
    // duplicate a def; instead we leave it intact and now UNUSED, so DCE prunes the
    // load + its feeder slot_addr/field_addr next round (the real win). none_value
    // is left untouched by mapUse.
    walk.remapUses(func, remap, mapUse);
    return true;
}

fn mapUse(remap: []const Ir.ValueId, v: Ir.ValueId) Ir.ValueId {
    if (v == Ir.none_value or v >= remap.len) return v;
    return remap[v];
}

/// Follow a chain of prior forwards (the stored value may itself be a forwarded id).
fn applyRemap(remap: []const Ir.ValueId, v: Ir.ValueId) Ir.ValueId {
    if (v == Ir.none_value or v >= remap.len) return v;
    // remap is built so a forwarded id points at an EARLIER def (the stored value),
    // which is never itself rewritten to point further (stores record the current
    // remapped val), so a single hop suffices. Guard anyway against a cycle.
    var cur = v;
    var hops: usize = 0;
    while (cur < remap.len and remap[cur] != cur and hops < remap.len) : (hops += 1) cur = remap[cur];
    return cur;
}

/// Trace a ptr value's defining op to an abstract address.
fn resolve(def: []const ?Ir.Op, ptr: Ir.ValueId) AbstractAddr {
    if (ptr == Ir.none_value or ptr >= def.len) return .unknown;
    const op = def[ptr] orelse return .unknown;
    return switch (op) {
        .slot_addr => |s| .{ .slot = .{ .s = s, .off = 0 } },
        .field_addr => |fa| switch (resolve(def, fa.base)) {
            .slot => |sl| .{ .slot = .{ .s = sl.s, .off = sl.off + fa.off } },
            .unknown => .unknown,
        },
        else => .unknown,
    };
}

/// Mark a slot escaped when a ptr value resolving to it is used as a call arg,
/// a store.val, or a copy.src (the ADDRESS itself leaves our analysis).
fn markEscapes(func: *const Ir.Function, def: []const ?Ir.Op, escaped: []bool) void {
    for (func.blocks) |b| {
        for (b.instrs) |ins| {
            switch (ins.op) {
                .store => |s| escapeOf(def, escaped, s.val),
                .copy => |c| escapeOf(def, escaped, c.src),
                .call => |c| for (c.args) |a| switch (a) {
                    .value => |v| escapeOf(def, escaped, v),
                    .slot => |sid| if (sid < escaped.len) {
                        escaped[sid] = true; // slot passed directly by reference.
                    },
                    .none => {},
                },
                .call_indirect => |c| {
                    escapeOf(def, escaped, c.target);
                    for (c.args) |a| switch (a) {
                        .value => |v| escapeOf(def, escaped, v),
                        .slot => |sid| if (sid < escaped.len) {
                            escaped[sid] = true;
                        },
                        .none => {},
                    };
                },
                else => {},
            }
        }
        // Terminator operands can also carry a ptr out (br args / ret value).
        switch (b.term) {
            .br => |br| for (br.args) |a| switch (a) {
                .value => |v| escapeOf(def, escaped, v),
                .slot => |sid| if (sid < escaped.len) {
                    escaped[sid] = true;
                },
                .none => {},
            },
            .ret => |o| switch (o) {
                .value => |v| escapeOf(def, escaped, v),
                .slot => |sid| if (sid < escaped.len) {
                    escaped[sid] = true;
                },
                .none => {},
            },
            .cond_br, .@"unreachable", .trap, .panic => {},
        }
    }
}

fn escapeOf(def: []const ?Ir.Op, escaped: []bool, v: Ir.ValueId) void {
    switch (resolve(def, v)) {
        .slot => |sl| if (sl.s < escaped.len) {
            escaped[sl.s] = true;
        },
        .unknown => {},
    }
}

// avail bookkeeping (flat indexed list; deterministic, no hash-map iter)

fn recordStore(gpa: std.mem.Allocator, avail: *std.ArrayListUnmanaged(Avail), s: Ir.SlotId, off: u32, val: Ir.ValueId) !void {
    // A later store to the same {S,off} supersedes the earlier one: overwrite.
    for (avail.items) |*a| {
        if (a.live and a.s == s and a.off == off) {
            a.val = val;
            return;
        }
    }
    try avail.append(gpa, .{ .s = s, .off = off, .val = val, .live = true });
}

fn lookup(avail: *std.ArrayListUnmanaged(Avail), s: Ir.SlotId, off: u32) ?Ir.ValueId {
    for (avail.items) |a| {
        if (a.live and a.s == s and a.off == off) return a.val;
    }
    return null;
}

fn clearAll(avail: *std.ArrayListUnmanaged(Avail)) void {
    for (avail.items) |*a| a.live = false;
}

fn clearSlot(avail: *std.ArrayListUnmanaged(Avail), s: Ir.SlotId) void {
    for (avail.items) |*a| {
        if (a.s == s) a.live = false;
    }
}

// Tests — hand-built IR under std.testing.allocator (catches leak/double-free).

const testing = std.testing;

fn intVals(gpa: std.mem.Allocator, n: usize) ![]Ir.ValueDef {
    const v = try gpa.alloc(Ir.ValueDef, n);
    for (v) |*x| x.* = .{ .type = Ir.Type.int };
    return v;
}

/// Build a single-block int function from instrs + a ret terminator.
fn oneBlock(gpa: std.mem.Allocator, slots_n: usize, instrs: []Ir.Instr, ret: Ir.ValueId, nvals: usize) !Ir.Function {
    const slots = try gpa.alloc(Ir.Slot, slots_n);
    for (slots) |*s| s.* = .{ .type = Ir.Type.int };
    var blocks = try gpa.alloc(Ir.Block, 1);
    blocks[0] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = instrs, .term = .{ .ret = .{ .value = ret } } };
    return .{
        .name = .{ .kind = .user_fn, .name = "t" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Ir.Type.int,
        .slots = slots,
        .values = try intVals(gpa, nvals),
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
    };
}

test "forward: store then load same slot -> load result becomes stored value" {
    const gpa = testing.allocator;
    // %0=iconst 10; %1=slot_addr s0; store %1,%0; %2=slot_addr s0; %3=load %2; ret %3
    // Forward: %3 -> %0. ret should then read %0.
    var instrs = try gpa.alloc(Ir.Instr, 5);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 10 } };
    instrs[1] = .{ .result = 1, .op = .{ .slot_addr = 0 } };
    instrs[2] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 1, .val = 0, .ty = Ir.Type.int } } };
    instrs[3] = .{ .result = 2, .op = .{ .slot_addr = 0 } };
    instrs[4] = .{ .result = 3, .op = .{ .load = .{ .addr = 2, .ty = Ir.Type.int } } };
    var func = try oneBlock(gpa, 1, instrs, 3, 4);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    try testing.expectEqual(@as(usize, 1), st.loads_forwarded);
    try testing.expectEqual(@as(Ir.ValueId, 0), func.blocks[0].term.ret.value);
    // The load instr is still present (DCE removes it later) but unused now.
    try testing.expectEqual(@as(Ir.ValueId, 3), func.blocks[0].instrs[4].result);
}

test "forward: NOT across a clobbering store to the same slot" {
    const gpa = testing.allocator;
    // store s0,%0 ; store s0,%1 ; load s0 -> must read %1, not %0.
    var instrs = try gpa.alloc(Ir.Instr, 7);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 1 } };
    instrs[1] = .{ .result = 1, .op = .{ .iconst = 2 } };
    instrs[2] = .{ .result = 2, .op = .{ .slot_addr = 0 } };
    instrs[3] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 2, .val = 0, .ty = Ir.Type.int } } };
    instrs[4] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 2, .val = 1, .ty = Ir.Type.int } } };
    instrs[5] = .{ .result = 3, .op = .{ .slot_addr = 0 } };
    instrs[6] = .{ .result = 4, .op = .{ .load = .{ .addr = 3, .ty = Ir.Type.int } } };
    var func = try oneBlock(gpa, 1, instrs, 4, 5);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    try testing.expectEqual(@as(usize, 1), st.loads_forwarded);
    // Forwarded the LATEST store's value %1, not %0.
    try testing.expectEqual(@as(Ir.ValueId, 1), func.blocks[0].term.ret.value);
}

test "forward: NOT through a call between store and load" {
    const gpa = testing.allocator;
    // store s0,%0 ; call @f() ; load s0 -> call clobbers, do NOT forward.
    var instrs = try gpa.alloc(Ir.Instr, 6);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 5 } };
    instrs[1] = .{ .result = 1, .op = .{ .slot_addr = 0 } };
    instrs[2] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 1, .val = 0, .ty = Ir.Type.int } } };
    instrs[3] = .{ .result = Ir.none_value, .op = .{ .call = .{ .callee = .{ .kind = .user_fn, .name = "f" }, .args = try gpa.alloc(Ir.Operand, 0), .ret_slot = Ir.none_slot } } };
    instrs[4] = .{ .result = 2, .op = .{ .slot_addr = 0 } };
    instrs[5] = .{ .result = 3, .op = .{ .load = .{ .addr = 2, .ty = Ir.Type.int } } };
    var func = try oneBlock(gpa, 1, instrs, 3, 4);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(!changed);
    try testing.expectEqual(@as(usize, 0), st.loads_forwarded);
    try testing.expectEqual(@as(Ir.ValueId, 3), func.blocks[0].term.ret.value);
}

test "forward: NOT from a different slot" {
    const gpa = testing.allocator;
    // store s0,%0 ; load s1 -> different slot, no avail entry, do NOT forward.
    var instrs = try gpa.alloc(Ir.Instr, 5);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 7 } };
    instrs[1] = .{ .result = 1, .op = .{ .slot_addr = 0 } };
    instrs[2] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 1, .val = 0, .ty = Ir.Type.int } } };
    instrs[3] = .{ .result = 2, .op = .{ .slot_addr = 1 } };
    instrs[4] = .{ .result = 3, .op = .{ .load = .{ .addr = 2, .ty = Ir.Type.int } } };
    var func = try oneBlock(gpa, 2, instrs, 3, 4);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(!changed);
    try testing.expectEqual(@as(usize, 0), st.loads_forwarded);
}

test "forward: NOT a distinct field offset of the same slot" {
    const gpa = testing.allocator;
    // %1=slot_addr s0; %2=field_addr %1,0; store %2,%0 ;
    // %3=slot_addr s0; %4=field_addr %3,8; load %4 -> off 8 != off 0, no forward.
    var instrs = try gpa.alloc(Ir.Instr, 7);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 3 } };
    instrs[1] = .{ .result = 1, .op = .{ .slot_addr = 0 } };
    instrs[2] = .{ .result = 2, .op = .{ .field_addr = .{ .base = 1, .off = 0, .ty = Ir.Type.int } } };
    instrs[3] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 2, .val = 0, .ty = Ir.Type.int } } };
    instrs[4] = .{ .result = 3, .op = .{ .slot_addr = 0 } };
    instrs[5] = .{ .result = 4, .op = .{ .field_addr = .{ .base = 3, .off = 8, .ty = Ir.Type.int } } };
    instrs[6] = .{ .result = 5, .op = .{ .load = .{ .addr = 4, .ty = Ir.Type.int } } };
    var func = try oneBlock(gpa, 1, instrs, 5, 6);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(!changed);
    try testing.expectEqual(@as(usize, 0), st.loads_forwarded);
}

test "forward: same field offset DOES forward" {
    const gpa = testing.allocator;
    // store field(s0,8),%0 ; load field(s0,8) -> forward %0.
    var instrs = try gpa.alloc(Ir.Instr, 7);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 42 } };
    instrs[1] = .{ .result = 1, .op = .{ .slot_addr = 0 } };
    instrs[2] = .{ .result = 2, .op = .{ .field_addr = .{ .base = 1, .off = 8, .ty = Ir.Type.int } } };
    instrs[3] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 2, .val = 0, .ty = Ir.Type.int } } };
    instrs[4] = .{ .result = 3, .op = .{ .slot_addr = 0 } };
    instrs[5] = .{ .result = 4, .op = .{ .field_addr = .{ .base = 3, .off = 8, .ty = Ir.Type.int } } };
    instrs[6] = .{ .result = 5, .op = .{ .load = .{ .addr = 4, .ty = Ir.Type.int } } };
    var func = try oneBlock(gpa, 1, instrs, 5, 6);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    try testing.expectEqual(@as(usize, 1), st.loads_forwarded);
    try testing.expectEqual(@as(Ir.ValueId, 0), func.blocks[0].term.ret.value);
}

test "forward: NOT from an escaped slot (addr passed to a call)" {
    const gpa = testing.allocator;
    // store s0,%0 ; %addr=slot_addr s0; call @f(%addr) ; load s0 ->
    // s0 escaped (addr handed to call) => never forward, even disregarding the call
    // clobber. (We assert no forward.)
    var call_args = try gpa.alloc(Ir.Operand, 1);
    call_args[0] = .{ .value = 3 }; // %3 = slot_addr s0
    var instrs = try gpa.alloc(Ir.Instr, 7);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 9 } };
    instrs[1] = .{ .result = 1, .op = .{ .slot_addr = 0 } };
    instrs[2] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 1, .val = 0, .ty = Ir.Type.int } } };
    instrs[3] = .{ .result = 3, .op = .{ .slot_addr = 0 } };
    instrs[4] = .{ .result = Ir.none_value, .op = .{ .call = .{ .callee = .{ .kind = .user_fn, .name = "f" }, .args = call_args, .ret_slot = Ir.none_slot } } };
    instrs[5] = .{ .result = 4, .op = .{ .slot_addr = 0 } };
    instrs[6] = .{ .result = 5, .op = .{ .load = .{ .addr = 4, .ty = Ir.Type.int } } };
    var func = try oneBlock(gpa, 1, instrs, 5, 6);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(!changed);
    try testing.expectEqual(@as(usize, 0), st.loads_forwarded);
}

test "forward: NOT through an unknown-address store" {
    const gpa = testing.allocator;
    // store s0,%0 ; store (load-result-ptr),%1 ; load s0 -> the middle store has an
    // unknown addr -> clears all avail -> do NOT forward the s0 load.
    // %2=slot_addr s0; store %2,%0; %3=load %2 (a ptr value, unknown);
    // store %3,%1 (unknown addr); %4=slot_addr s0; %5=load %4.
    var instrs = try gpa.alloc(Ir.Instr, 8);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 1 } };
    instrs[1] = .{ .result = 1, .op = .{ .iconst = 2 } };
    instrs[2] = .{ .result = 2, .op = .{ .slot_addr = 0 } };
    instrs[3] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 2, .val = 0, .ty = Ir.Type.int } } };
    instrs[4] = .{ .result = 3, .op = .{ .load = .{ .addr = 2, .ty = Ir.Type.int } } }; // %3 = loaded scalar, used as a ptr below (unknown resolve)
    instrs[5] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 3, .val = 1, .ty = Ir.Type.int } } }; // unknown addr
    instrs[6] = .{ .result = 4, .op = .{ .slot_addr = 0 } };
    instrs[7] = .{ .result = 5, .op = .{ .load = .{ .addr = 4, .ty = Ir.Type.int } } };
    var func = try oneBlock(gpa, 1, instrs, 5, 6);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    // NOTE: %3 (the first load of s0) IS forwardable to %0 — that load reads s0 with
    // an avail entry present. So a forward DOES happen on %3, but the SECOND load
    // (%5) must NOT forward because the unknown-addr store cleared avail. Assert the
    // second load's user (ret) is unchanged.
    try testing.expectEqual(@as(Ir.ValueId, 5), func.blocks[0].term.ret.value);
    _ = changed;
}

test "forward: NOT across a clobbering copy to the same slot" {
    const gpa = testing.allocator;
    // store s0,%0 ; copy s0 <- s1 ; load s0 -> copy writes s0, clears it, no forward.
    var instrs = try gpa.alloc(Ir.Instr, 7);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 4 } };
    instrs[1] = .{ .result = 1, .op = .{ .slot_addr = 0 } };
    instrs[2] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 1, .val = 0, .ty = Ir.Type.int } } };
    instrs[3] = .{ .result = 2, .op = .{ .slot_addr = 1 } };
    instrs[4] = .{ .result = Ir.none_value, .op = .{ .copy = .{ .dst = 1, .src = 2, .ty = Ir.Type.int } } };
    instrs[5] = .{ .result = 3, .op = .{ .slot_addr = 0 } };
    instrs[6] = .{ .result = 4, .op = .{ .load = .{ .addr = 3, .ty = Ir.Type.int } } };
    var func = try oneBlock(gpa, 2, instrs, 4, 5);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(!changed);
    try testing.expectEqual(@as(usize, 0), st.loads_forwarded);
}

test "forward: chain forwards both reads of a local (forward_chain shape)" {
    const gpa = testing.allocator;
    // a:=5; b:=a+a — store s0,%0; load s0; load s0; add -> both loads forward to %0.
    // %0=iconst 5; %1=slot_addr s0; store %1,%0;
    // %2=slot_addr s0; %3=load %2; %4=slot_addr s0; %5=load %4; %6=add %3,%5; ret %6
    var instrs = try gpa.alloc(Ir.Instr, 8);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 5 } };
    instrs[1] = .{ .result = 1, .op = .{ .slot_addr = 0 } };
    instrs[2] = .{ .result = Ir.none_value, .op = .{ .store = .{ .addr = 1, .val = 0, .ty = Ir.Type.int } } };
    instrs[3] = .{ .result = 2, .op = .{ .slot_addr = 0 } };
    instrs[4] = .{ .result = 3, .op = .{ .load = .{ .addr = 2, .ty = Ir.Type.int } } };
    instrs[5] = .{ .result = 4, .op = .{ .slot_addr = 0 } };
    instrs[6] = .{ .result = 5, .op = .{ .load = .{ .addr = 4, .ty = Ir.Type.int } } };
    instrs[7] = .{ .result = 6, .op = .{ .add = .{ .lhs = 3, .rhs = 5 } } };
    var func = try oneBlock(gpa, 1, instrs, 6, 7);
    defer func.deinit(gpa);

    var st: Opt.Stats = .{};
    const changed = try run(gpa, &func, &st);
    try testing.expect(changed);
    try testing.expectEqual(@as(usize, 2), st.loads_forwarded);
    // add now reads %0, %0.
    try testing.expectEqual(@as(Ir.ValueId, 0), func.blocks[0].instrs[7].op.add.lhs);
    try testing.expectEqual(@as(Ir.ValueId, 0), func.blocks[0].instrs[7].op.add.rhs);
}
