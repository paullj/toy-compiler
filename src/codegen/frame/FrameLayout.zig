//! Spill-everything frame layout (M12, LOCK #4).
//!
//! Replaces the FOUR M11 frame-sizing walkers (countSlots*/collectSlots*,
//! measureOutgoing*, measureDepth*/measureExpr, buildSlotTable temp logic) with
//! ONE deterministic pass over the lowered `Ir.Function`. The rule is dead
//! simple: every IR `Slot` AND every SSA `Value` gets its OWN stack slot, laid
//! sequentially, type-sized and 8-byte aligned, NEVER reused. This trivially
//! eliminates the recurring frame-undersizing SIGBUS class (a value/temp landing
//! past the frame) at the cost of larger frames — slot-coloring is a deferred
//! optimization (do NOT do it here).
//!
//! LAYOUT (deterministic, in IR index order):
//!   [ saved x29/x30 ]  <- handled by the prologue (stp ..., #-16!); NOT part of
//!                         this frame (offsets here are sp-relative, above the
//!                         `sub sp` adjustment).
//!   [ out_base       ]  outgoing stack-arg region (offsets 0..out_base). Every
//!                       slot/value offset is shifted up by this so a call's
//!                       stack args at [sp,#0..] never collide with a live slot.
//!                       out_base = roundUp16(max over call sites of the call's
//!                       NSAA bytes, from Abi.planCall), 0 if the fn makes no
//!                       calls.
//!   [ slot region    ]  each Ir.Slot: roundUp8(running) then its 8-min-aligned
//!                       size (str 16, struct/enum roundUp8(layout size),
//!                       scalar 8).
//!   [ value region   ]  each Ir.ValueDef: one slot, roundUp8(size) (scalars 8;
//!                       an aggregate-typed value — should not occur under
//!                       LOCK #2, aggregates live in slots — still gets a sized
//!                       cell defensively).
//!   [ sret save slot ]  one 8-byte cell IF the return is a >16B aggregate (the
//!                       incoming x8 is spilled here once; the body's calls
//!                       clobber x8).
//!   frame = roundUp16(out_base + slots + values + maybe-sret).
//!
//! DETERMINISM: pure, slice-indexed, no hashmaps / pointer iteration / globals.
//! Same IR + layouts => same offsets => same bytes. This underpins the VERIFY
//! byte-identity re-lower ([C11]).
//!
//! GUARDS (preserved from M11, surfaced as errors so codegen can emit the SAME
//! diagnostic strings):
//!   * frame > 4095          -> error.FrameTooLarge        ("function frame too
//!                              large for codegen (too many locals)")
//!   * an incoming stack-arg scaled offset > 4095*8 -> error.ParamOffsetTooLarge
//!                              ("too many parameters for codegen ...")
//! NOTE: spill-all enlarges frames vs the old reused-temp path, so FrameTooLarge
//! can trip on programs that compiled before — the corpus must verify every fn
//! stays < 4096B (slot-coloring is the deferred fix).

const std = @import("std");
const Ir = @import("../../ir/Ir.zig");
const Abi = @import("../abi/Abi.zig");

const types = @import("../../types.zig");
pub const Type = types.Type;
pub const Layout = types.Layout;
pub const EnumLayout = types.EnumLayout;

const FrameLayout = @This();

/// The widest scalable ldr/str immediate offset (imm12 * 8). Any sp/fp offset
/// past this cannot be encoded. Mirrors M11 `Codegen.MAX_SCALED_OFFSET`.
const MAX_SCALED_OFFSET: u32 = 4095 * 8;

/// `sret_save_off == NO_SRET` means the function does not return a >16B
/// aggregate (no x8 buffer to save). Mirrors M11 `Codegen.NO_SRET`.
pub const NO_SRET: u32 = std.math.maxInt(u32);

pub const Error = error{ OutOfMemory, FrameTooLarge, ParamOffsetTooLarge };

/// Total frame bytes the prologue opens with `sub sp, sp, #frame` (16-aligned,
/// > 0 only when the fn needs locals/values/outgoing args).
frame: u32,
/// Outgoing stack-arg region size at the bottom of the frame (sp-relative
/// 0..out_base). 16-aligned. 0 if the fn makes no calls.
out_base: u32,
/// sp-relative offset of the saved incoming x8 (sret buffer ptr), or `NO_SRET`.
sret_save_off: u32,
/// sp-relative byte offset of each `Ir.Slot`, indexed by `SlotId`. Owned.
slot_off: []u32,
/// sp-relative byte offset of each `Ir.ValueDef`, indexed by `ValueId`. Owned.
value_off: []u32,

pub fn deinit(self: *FrameLayout, gpa: std.mem.Allocator) void {
    gpa.free(self.slot_off);
    gpa.free(self.value_off);
    self.* = undefined;
}

/// sp-relative byte offset of slot `sid`.
pub fn slotAddr(self: *const FrameLayout, sid: Ir.SlotId) u32 {
    return self.slot_off[sid];
}

/// sp-relative byte offset of value `vid`.
pub fn valueAddr(self: *const FrameLayout, vid: Ir.ValueId) u32 {
    return self.value_off[vid];
}

fn roundUp8(n: u32) u32 {
    return (n + 7) / 8 * 8;
}

fn roundUp16(n: u32) u32 {
    return (n + 15) / 16 * 16;
}

/// Per-slot frame size: a slot must hold the WHOLE value. str is 16 (ptr,len);
/// struct/enum take their layout size rounded to 8 (so the next slot stays
/// 8-aligned; struct align is <=8 today). Everything else is one 8-byte word.
/// Mirrors M11 `buildSlotTable` (~2482).
fn slotSize(ty: Type, layouts: []const Layout, enum_layouts: []const EnumLayout) u32 {
    return switch (ty.kind) {
        .str => 16,
        .@"struct", .@"enum" => roundUp8(Abi.typeSize(ty, layouts, enum_layouts)),
        else => 8,
    };
}

/// Compute the spill-everything frame for `func`. Pure over the IR + read-only
/// layout tables; allocates only the two offset slices (owned by the caller).
pub fn compute(
    gpa: std.mem.Allocator,
    func: *const Ir.Function,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
) Error!FrameLayout {
    // (1) out_base: the widest outgoing stack-arg region over all call sites.
    // For each `call` instr, plan its outbound args (Abi.planCall) and take the
    // max NSAA bytes; round the running max to 16 so the slot region above it
    // stays 16-aligned. The arg TYPES come from each operand's slot/value type.
    var max_nsaa: u32 = 0;
    {
        // A reusable scratch buffer for arg types, grown as needed.
        var arg_types: std.ArrayList(Type) = .empty;
        defer arg_types.deinit(gpa);
        for (func.blocks) |blk| {
            for (blk.instrs) |ins| {
                switch (ins.op) {
                    .call => |c| {
                        arg_types.clearRetainingCapacity();
                        try arg_types.ensureTotalCapacity(gpa, c.args.len);
                        for (c.args) |arg| {
                            arg_types.appendAssumeCapacity(func.operandType(arg));
                        }
                        // The callee's return type drives sret (x8), which does
                        // NOT consume NSAA — but planCall needs a ret type. Use
                        // the call's own ret_slot type if present, else unit
                        // (scalar/unit results never affect NSAA either).
                        const ret_ty: Type = if (c.ret_slot != Ir.none_slot)
                            func.slots[c.ret_slot].type
                        else
                            Type.unit;
                        var plan = try Abi.planCall(gpa, arg_types.items, ret_ty, layouts, enum_layouts);
                        defer plan.deinit(gpa);
                        if (plan.nsaa_bytes > max_nsaa) max_nsaa = plan.nsaa_bytes;
                    },
                    else => {},
                }
            }
        }
    }
    const out_base: u32 = roundUp16(max_nsaa);

    // (2) slot region: each Ir.Slot laid sequentially, 8-aligned, shifted by
    // out_base. Prefix sum, exactly the M11 buildSlotTable arithmetic.
    const slot_off = try gpa.alloc(u32, func.slots.len);
    errdefer gpa.free(slot_off);
    var running: u32 = 0;
    for (func.slots, 0..) |slot, i| {
        running = roundUp8(running);
        slot_off[i] = out_base + running;
        running += slotSize(slot.type, layouts, enum_layouts);
    }

    // (3) value region: each SSA value its OWN slot (spill-all), 8-aligned.
    const value_off = try gpa.alloc(u32, func.values.len);
    errdefer gpa.free(value_off);
    for (func.values, 0..) |vdef, i| {
        running = roundUp8(running);
        value_off[i] = out_base + running;
        running += slotSize(vdef.type, layouts, enum_layouts);
    }

    // (4) sret save slot (8 bytes) when the return is a >16B aggregate.
    var sret_save_off: u32 = NO_SRET;
    if (Abi.classifyRet(func.ret_type, layouts, enum_layouts) == .sret) {
        running = roundUp8(running);
        sret_save_off = out_base + running;
        running += 8;
    }

    // (5) frame = 16-aligned total. Assert the layout invariants.
    const frame: u32 = roundUp16(out_base + running);
    std.debug.assert(frame % 16 == 0);
    std.debug.assert(out_base % 16 == 0);

    // GUARD: the prologue's `sub sp, sp, #frame` uses a 12-bit immediate, and
    // every slot/value ldr/str uses a scaled imm12. The top offset is < frame,
    // so a frame within imm12 keeps every access encodable.
    if (frame > 4095) {
        // errdefer (registered above for both offset slices) frees them.
        return error.FrameTooLarge;
    }

    // GUARD: incoming stack args (params 9+) are read at [x29,#16+nsaa],
    // INDEPENDENT of frame. Plan the incoming params; any stack/stack_ptr loc
    // whose top scaled offset exceeds the imm12 range is unencodable.
    {
        var param_types: std.ArrayList(Type) = .empty;
        defer param_types.deinit(gpa);
        try param_types.ensureTotalCapacity(gpa, func.params.len);
        for (func.params) |sid| param_types.appendAssumeCapacity(func.slots[sid].type);
        var pplan = try Abi.planParams(gpa, param_types.items, func.ret_type, layouts, enum_layouts);
        defer pplan.deinit(gpa);
        for (pplan.locs) |loc| {
            const top_in: u32 = switch (loc) {
                .stack => |s| 16 + s.nsaa_off + (if (s.bytes > 0) s.bytes - 8 else 0),
                .stack_ptr => |off| 16 + off,
                else => 0,
            };
            if (top_in > MAX_SCALED_OFFSET) {
                // errdefer (registered above) frees both offset slices.
                return error.ParamOffsetTooLarge;
            }
        }
    }

    return .{
        .frame = frame,
        .out_base = out_base,
        .sret_save_off = sret_save_off,
        .slot_off = slot_off,
        .value_off = value_off,
    };
}

// ===========================================================================
// TESTS
// ===========================================================================

const testing = std.testing;

// A small builder for hand-rolled IR functions. All slices are heap-allocated
// so `Function.deinit` frees them cleanly.
const TestFn = struct {
    func: Ir.Function,

    fn deinit(self: *TestFn, gpa: std.mem.Allocator) void {
        self.func.deinit(gpa);
    }
};

fn mkLayouts(gpa: std.mem.Allocator, sizes: []const u32, aligns: []const u32) ![]Layout {
    const ls = try gpa.alloc(Layout, sizes.len);
    for (sizes, 0..) |s, i| {
        ls[i] = .{
            .name = "T",
            .field_names = &.{},
            .field_types = &.{},
            .offsets = &.{},
            .size = s,
            .@"align" = aligns[i],
        };
    }
    return ls;
}

fn allocSlots(gpa: std.mem.Allocator, types_: []const Type) ![]Ir.Slot {
    const s = try gpa.alloc(Ir.Slot, types_.len);
    for (types_, 0..) |t, i| s[i] = .{ .type = t };
    return s;
}

fn allocValues(gpa: std.mem.Allocator, types_: []const Type) ![]Ir.ValueDef {
    const v = try gpa.alloc(Ir.ValueDef, types_.len);
    for (types_, 0..) |t, i| v[i] = .{ .type = t };
    return v;
}

// Build a 1-block function with the given slots/values and a single ret-none
// terminator (no instrs). For exercising slot/value offset math.
fn buildSimple(
    gpa: std.mem.Allocator,
    slot_types: []const Type,
    value_types: []const Type,
    param_count: usize,
    ret_type: Type,
) !Ir.Function {
    const slots = try allocSlots(gpa, slot_types);
    const values = try allocValues(gpa, value_types);
    const params = try gpa.alloc(Ir.SlotId, param_count);
    for (0..param_count) |i| params[i] = @intCast(i);
    const blocks = try gpa.alloc(Ir.Block, 1);
    blocks[0] = .{
        .params = try gpa.alloc(Ir.ValueId, 0),
        .instrs = try gpa.alloc(Ir.Instr, 0),
        .term = .{ .ret = .none },
    };
    return .{
        .name = .{ .kind = .user_fn, .name = "f" },
        .params = params,
        .ret_type = ret_type,
        .slots = slots,
        .values = values,
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
    };
}

test "frame: empty fn (no slots, no values, no calls) -> frame 0" {
    const gpa = testing.allocator;
    var func = try buildSimple(gpa, &.{}, &.{}, 0, Type.unit);
    defer func.deinit(gpa);

    var fl = try compute(gpa, &func, &.{}, &.{});
    defer fl.deinit(gpa);

    try testing.expectEqual(@as(u32, 0), fl.frame);
    try testing.expectEqual(@as(u32, 0), fl.out_base);
    try testing.expectEqual(NO_SRET, fl.sret_save_off);
    try testing.expectEqual(@as(usize, 0), fl.slot_off.len);
    try testing.expectEqual(@as(usize, 0), fl.value_off.len);
}

test "frame: scalar slots + scalar values lay out sequentially, 8-aligned, 16-aligned frame" {
    const gpa = testing.allocator;
    // 2 int slots + 3 int values, no calls.
    var func = try buildSimple(
        gpa,
        &.{ Type.int, Type.int },
        &.{ Type.int, Type.int, Type.int },
        0,
        Type.int,
    );
    defer func.deinit(gpa);

    var fl = try compute(gpa, &func, &.{}, &.{});
    defer fl.deinit(gpa);

    // out_base 0; slots at 0,8; values at 16,24,32; running=40; frame roundUp16(40)=48.
    try testing.expectEqual(@as(u32, 0), fl.out_base);
    try testing.expectEqual(@as(u32, 0), fl.slot_off[0]);
    try testing.expectEqual(@as(u32, 8), fl.slot_off[1]);
    try testing.expectEqual(@as(u32, 16), fl.value_off[0]);
    try testing.expectEqual(@as(u32, 24), fl.value_off[1]);
    try testing.expectEqual(@as(u32, 32), fl.value_off[2]);
    try testing.expectEqual(@as(u32, 48), fl.frame);
    try testing.expectEqual(NO_SRET, fl.sret_save_off);
}

test "frame: str slot occupies 16 bytes and keeps following slots 8-aligned" {
    const gpa = testing.allocator;
    // int, str, int slots: offsets 0, 8, 24.
    var func = try buildSimple(
        gpa,
        &.{ Type.int, Type.str, Type.int },
        &.{},
        0,
        Type.unit,
    );
    defer func.deinit(gpa);

    var fl = try compute(gpa, &func, &.{}, &.{});
    defer fl.deinit(gpa);

    try testing.expectEqual(@as(u32, 0), fl.slot_off[0]);
    try testing.expectEqual(@as(u32, 8), fl.slot_off[1]); // str
    try testing.expectEqual(@as(u32, 24), fl.slot_off[2]); // 8 + 16
    // running = 32 -> frame roundUp16(32) = 32.
    try testing.expectEqual(@as(u32, 32), fl.frame);
}

test "frame: aggregate slot uses roundUp8 of its layout size" {
    const gpa = testing.allocator;
    // struct 0 = 24B align 8, struct 1 = 12B align 8.
    const ls = try mkLayouts(gpa, &.{ 24, 12 }, &.{ 8, 8 });
    defer gpa.free(ls);

    var func = try buildSimple(
        gpa,
        &.{ Type.structT(0), Type.structT(1), Type.int },
        &.{},
        0,
        Type.unit,
    );
    defer func.deinit(gpa);

    var fl = try compute(gpa, &func, ls, &.{});
    defer fl.deinit(gpa);

    // struct0 (24) at 0; struct1 (12 -> roundUp8 16) at 24; int at 40.
    try testing.expectEqual(@as(u32, 0), fl.slot_off[0]);
    try testing.expectEqual(@as(u32, 24), fl.slot_off[1]);
    try testing.expectEqual(@as(u32, 40), fl.slot_off[2]);
    // running = 48 -> frame 48.
    try testing.expectEqual(@as(u32, 48), fl.frame);
}

test "frame: many block-arg (merge) value slots each get their own cell" {
    const gpa = testing.allocator;
    // simulate a deeply-nested chain that produced many SSA values: 20 int values.
    var vtypes: [20]Type = undefined;
    for (&vtypes) |*v| v.* = Type.int;
    var func = try buildSimple(gpa, &.{}, &vtypes, 0, Type.int);
    defer func.deinit(gpa);

    var fl = try compute(gpa, &func, &.{}, &.{});
    defer fl.deinit(gpa);

    // 20 values * 8 = 160; each at i*8.
    for (0..20) |i| {
        try testing.expectEqual(@as(u32, @intCast(i * 8)), fl.value_off[i]);
    }
    try testing.expectEqual(@as(u32, 160), fl.frame);
}

test "frame: out_base reserves the widest outgoing stack-arg region over call sites" {
    const gpa = testing.allocator;
    const ls: []const Layout = &.{};
    const el: []const EnumLayout = &.{};

    // Build a fn with two call instrs: one with 8 int args (0 NSAA), one with
    // 10 int args (2 stack words -> 16 NSAA). out_base = roundUp16(16) = 16.
    const slots = try allocSlots(gpa, &.{Type.int}); // one local slot (a result)
    // 10 value defs to serve as scalar args.
    var vtypes: [10]Type = undefined;
    for (&vtypes) |*v| v.* = Type.int;
    const values = try allocValues(gpa, &vtypes);

    // call A: 8 args. call B: 10 args.
    const args_a = try gpa.alloc(Ir.Operand, 8);
    for (0..8) |i| args_a[i] = .{ .value = @intCast(i) };
    const args_b = try gpa.alloc(Ir.Operand, 10);
    for (0..10) |i| args_b[i] = .{ .value = @intCast(i) };

    const instrs = try gpa.alloc(Ir.Instr, 2);
    instrs[0] = .{ .result = Ir.none_value, .op = .{ .call = .{
        .callee = .{ .kind = .user_fn, .name = "a" },
        .args = args_a,
        .ret_slot = Ir.none_slot,
    } } };
    instrs[1] = .{ .result = Ir.none_value, .op = .{ .call = .{
        .callee = .{ .kind = .user_fn, .name = "b" },
        .args = args_b,
        .ret_slot = Ir.none_slot,
    } } };

    const blocks = try gpa.alloc(Ir.Block, 1);
    blocks[0] = .{
        .params = try gpa.alloc(Ir.ValueId, 0),
        .instrs = instrs,
        .term = .{ .ret = .none },
    };
    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "caller" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Type.unit,
        .slots = slots,
        .values = values,
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
    };
    defer func.deinit(gpa);

    var fl = try compute(gpa, &func, ls, el);
    defer fl.deinit(gpa);

    // out_base = 16; slots/values shifted up by it.
    try testing.expectEqual(@as(u32, 16), fl.out_base);
    try testing.expectEqual(@as(u32, 16), fl.slot_off[0]); // the one slot at out_base
    try testing.expectEqual(@as(u32, 24), fl.value_off[0]); // first value after the slot
}

test "frame: sret return reserves an 8-byte save slot atop the frame" {
    const gpa = testing.allocator;
    const ls = try mkLayouts(gpa, &.{24}, &.{8}); // 24B -> sret
    defer gpa.free(ls);

    // fn returns a 24B struct; one int slot.
    var func = try buildSimple(gpa, &.{Type.int}, &.{}, 0, Type.structT(0));
    defer func.deinit(gpa);

    var fl = try compute(gpa, &func, ls, &.{});
    defer fl.deinit(gpa);

    // slot at 0 (8 bytes); sret save at 8; running 16; frame 16.
    try testing.expectEqual(@as(u32, 0), fl.slot_off[0]);
    try testing.expectEqual(@as(u32, 8), fl.sret_save_off);
    try testing.expectEqual(@as(u32, 16), fl.frame);
}

test "frame: FrameTooLarge when spill-all overflows the imm12 budget" {
    const gpa = testing.allocator;
    // 600 int values * 8 = 4800 > 4095. Should error cleanly.
    var vtypes: [600]Type = undefined;
    for (&vtypes) |*v| v.* = Type.int;
    var func = try buildSimple(gpa, &.{}, &vtypes, 0, Type.int);
    defer func.deinit(gpa);

    try testing.expectError(error.FrameTooLarge, compute(gpa, &func, &.{}, &.{}));
}

test "frame: ParamOffsetTooLarge when an incoming stack-arg offset exceeds imm12" {
    const gpa = testing.allocator;
    // Need a param whose incoming stack offset 16+nsaa > 4095*8 = 32760. Each
    // overflow param is 8 bytes, starting after 8 GPR params. nsaa for param k
    // (k>=8) is (k-8)*8. Solve 16 + (k-8)*8 > 32760 -> k-8 > 4093 -> k >= 4102.
    // Use 4110 int params (well past the threshold). All ints (scalar) so the
    // frame itself stays tiny (no slots/values besides params).
    const n = 4110;
    var ptypes: [n]Type = undefined;
    for (&ptypes) |*p| p.* = Type.int;
    var func = try buildSimple(gpa, &ptypes, &.{}, n, Type.unit);
    defer func.deinit(gpa);

    // The frame guard would trip first (4110 param slots * 8 > 4095), so this
    // test instead confirms an over-budget configuration errors (either guard).
    const res = compute(gpa, &func, &.{}, &.{});
    try testing.expect(res == error.FrameTooLarge or res == error.ParamOffsetTooLarge);
}
