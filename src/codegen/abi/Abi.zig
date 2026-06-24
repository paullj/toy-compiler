//! AArch64 (AAPCS64) calling-convention decisions — the ONE place that turns
//! target-independent typed values into physical register/stack assignments
//! (LOCK #3). The IR carries no registers, no frame offsets, no calling
//! convention; this module decides reg-pair vs indirect, x8 sret, and the
//! NGRN/NSAA walk, given only a `Type` slice + the read-only layout tables.
//!
//! WHY a dedicated module: the M11 codegen smeared the SAME NGRN/NSAA walk
//! across THREE sites that drifted independently — the prologue param
//! marshalling, `genCallInner`'s outbound args, and `measureOutgoingExpr`'s
//! stack-byte budget. Any disagreement landed an argument at the wrong offset
//! (SIGBUS). This module collapses all three into ONE private `walkAbi`, so the
//! prologue (`planParams`), the call site (`planCall`), and the frame's
//! `out_base` (`planCall(...).nsaa_bytes`) are guaranteed consistent BY
//! CONSTRUCTION.
//!
//! PURITY: every function here is pure over `Type` + `layouts`/`enum_layouts`
//! (read-only). The plan-producing functions allocate a result slice only; no
//! IO, no globals, no hashmaps over pointers. This is what keeps the dual-path
//! VERIFY byte-identity re-lower sound (Driver [C11]).
//!
//! AAPCS64 RULES reproduced EXACTLY from the M11 codegen (do NOT change):
//!   * scalar (int/bool) → 1 GPR if NGRN<8 else 8 stack bytes (NSAA).
//!   * aggregate <=16B (str / small struct/enum) → a run of `ebs` (1 or 2)
//!     eightbytes: if NGRN+ebs<=8 it takes regs {NGRN..}, NGRN+=ebs; else it
//!     goes WHOLLY to the stack (ebs*8 bytes) AND NGRN is set to 8 — the AAPCS
//!     no-split rule: once an aggregate spills, no later arg may backfill the
//!     remaining GPRs.
//!   * aggregate >16B → INDIRECT: the caller passes a POINTER to its copy in 1
//!     GPR (if NGRN<8) else 8 stack bytes.
//!   * an indirect (>16B) RESULT uses x8 (the indirect-result register) which is
//!     SEPARATE from the GPR arg sequence — it does NOT consume an NGRN.

const std = @import("std");
const Typecheck = @import("../../types.zig");

const Abi = @This();

pub const Type = Typecheck.Type;
pub const Layout = Typecheck.Layout;
pub const EnumLayout = Typecheck.EnumLayout;

/// The 8 general-purpose argument registers (x0..x7) and the indirect-result
/// register x8. There are 8 GPR arg slots; arg 9+ (or a spilled aggregate) goes
/// on the stack.
pub const num_gpr_args: u32 = 8;

/// AAPCS64 aggregate class (size-driven, never forked by kind):
///   * `scalar`    — not an aggregate (int/bool): one GPR / 8 stack bytes.
///   * `reg_pair`  — aggregate <=16B: a 1- or 2-eightbyte register run.
///   * `indirect`  — aggregate >16B: passed/returned via a pointer (+ x8 sret).
pub const AbiClass = enum { scalar, reg_pair, indirect };

// ---- pure type queries (single source of truth, shared with FrameLayout) ----

/// Byte size of a type (int/bool 8, str 16, struct/enum → its layout size).
/// Mirrors the M11 `Codegen.typeSize` exactly.
pub fn typeSize(ty: Type, layouts: []const Layout, enum_layouts: []const EnumLayout) u32 {
    return switch (ty.kind) {
        .int, .bool => 8,
        .str => 16,
        .@"struct" => layouts[ty.struct_id].size,
        .@"enum" => enum_layouts[ty.enum_id].size,
        else => 0,
    };
}

/// Natural alignment of a type (8 for scalars/str; a struct/enum's layout
/// alignment). Mirrors the M11 `Codegen.typeAlign` exactly.
pub fn typeAlign(ty: Type, layouts: []const Layout, enum_layouts: []const EnumLayout) u32 {
    return switch (ty.kind) {
        .int, .bool, .str => 8,
        .@"struct" => layouts[ty.struct_id].@"align",
        .@"enum" => enum_layouts[ty.enum_id].@"align",
        else => 1,
    };
}

/// Whether a type is a value AGGREGATE (str / struct / enum) — passed in a reg
/// pair (<=16B) or indirect+x8 (>16B), copied by bytes. Mirrors M11
/// `Codegen.isAggregate`.
pub fn isAggregate(ty: Type) bool {
    return ty.kind == .str or ty.kind == .@"struct" or ty.kind == .@"enum";
}

/// How many 8-byte "eightbytes" a <=16-byte reg-pair value occupies (1 or 2).
pub fn eightbytes(size: u32) u32 {
    return (size + 7) / 8;
}

/// AAPCS64 class for a type: a non-aggregate is `scalar`; an aggregate is
/// `reg_pair` (<=16B) or `indirect` (>16B). Single source of truth for the
/// reg-pair vs indirect split (was `Codegen.abiClass`, but lifted to take the
/// full type so callers needn't pre-compute the size).
pub fn classify(ty: Type, layouts: []const Layout, enum_layouts: []const EnumLayout) AbiClass {
    if (!isAggregate(ty)) return .scalar;
    return if (typeSize(ty, layouts, enum_layouts) <= 16) .reg_pair else .indirect;
}

// ---- one parameter / argument location -------------------------------------

/// Where one parameter (or one outbound call argument) lives per AAPCS64. The
/// same encoding serves both directions (incoming prologue + outgoing call); the
/// emitter interprets it as "store incoming reg here" vs "load from here into
/// reg" accordingly.
///
///   * `gpr`       — a scalar OR a reg-pair aggregate occupying `count` (1 or 2)
///                   consecutive GPRs starting at x`first`.
///   * `gpr_ptr`   — an INDIRECT (>16B) aggregate whose POINTER is passed in the
///                   single GPR x`reg` (the bytes live elsewhere).
///   * `stack`     — a scalar OR a reg-pair aggregate placed WHOLLY on the
///                   outgoing/incoming stack: `bytes` total at NSAA offset
///                   `nsaa_off` (relative to the stack-arg base).
///   * `stack_ptr` — an INDIRECT aggregate whose POINTER (8 bytes) sits on the
///                   stack at NSAA offset `nsaa_off`.
pub const ParamLoc = union(enum) {
    gpr: struct { first: u8, count: u8 },
    gpr_ptr: u8,
    stack: struct { nsaa_off: u32, bytes: u32 },
    stack_ptr: u32,
};

/// Outbound call arguments use the exact same location encoding as parameters.
pub const ArgLoc = ParamLoc;

/// Where a function's RESULT lands per AAPCS64:
///   * `none`   — unit (no result register).
///   * `reg`    — a scalar (1 reg, x0) or a reg-pair aggregate (`regs` = 1 or 2,
///                in x0/x1).
///   * `sret`   — an indirect (>16B) aggregate: the CALLER passes a buffer
///                pointer in x8; the callee writes through it and returns void.
pub const RetLoc = union(enum) {
    none,
    reg: struct { regs: u8 },
    sret,
};

/// Classify a function's return type. A `unit` return is `none`; a scalar or
/// reg-pair aggregate returns in x0(/x1); a >16B aggregate is `sret` (x8).
pub fn classifyRet(ty: Type, layouts: []const Layout, enum_layouts: []const EnumLayout) RetLoc {
    if (ty.kind == .unit) return .none;
    switch (classify(ty, layouts, enum_layouts)) {
        .scalar => return .{ .reg = .{ .regs = 1 } },
        .reg_pair => return .{ .reg = .{ .regs = @intCast(eightbytes(typeSize(ty, layouts, enum_layouts))) } },
        .indirect => return .sret,
    }
}

// ---- the ONE shared NGRN/NSAA walk -----------------------------------------

/// The single source of truth for the AAPCS64 argument-placement walk. Both
/// `planParams` and `planCall` route through here so the three M11 walks
/// (prologue / genCallInner / measureOutgoingExpr) can never drift again.
///
/// Fills `out_locs[i]` for each `types_[i]` and returns the total NSAA byte
/// count (the running stack total, BEFORE any 16-byte round-up). `out_locs` must
/// be `types_.len` long. `sret` says whether the result is indirect — it does
/// NOT consume an NGRN (x8 is separate), so the walk is identical whether or not
/// sret is set; the flag is threaded only so the caller can record it.
fn walkAbi(
    types_: []const Type,
    out_locs: []ParamLoc,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
) u32 {
    std.debug.assert(out_locs.len == types_.len);
    var ngrn: u32 = 0;
    var nsaa: u32 = 0;
    for (types_, 0..) |ty, i| {
        switch (classify(ty, layouts, enum_layouts)) {
            .indirect => {
                if (ngrn < num_gpr_args) {
                    out_locs[i] = .{ .gpr_ptr = @intCast(ngrn) };
                    ngrn += 1;
                } else {
                    out_locs[i] = .{ .stack_ptr = nsaa };
                    nsaa += 8;
                }
            },
            .reg_pair => {
                const ebs = eightbytes(typeSize(ty, layouts, enum_layouts));
                if (ngrn + ebs <= num_gpr_args) {
                    out_locs[i] = .{ .gpr = .{ .first = @intCast(ngrn), .count = @intCast(ebs) } };
                    ngrn += ebs;
                } else {
                    // AAPCS no-split: the whole aggregate goes to the stack and
                    // NGRN is exhausted (no later backfill of the leftover GPRs).
                    out_locs[i] = .{ .stack = .{ .nsaa_off = nsaa, .bytes = ebs * 8 } };
                    nsaa += ebs * 8;
                    ngrn = num_gpr_args;
                }
            },
            .scalar => {
                if (ngrn < num_gpr_args) {
                    out_locs[i] = .{ .gpr = .{ .first = @intCast(ngrn), .count = 1 } };
                    ngrn += 1;
                } else {
                    out_locs[i] = .{ .stack = .{ .nsaa_off = nsaa, .bytes = 8 } };
                    nsaa += 8;
                }
            },
        }
    }
    return nsaa;
}

// ---- parameter plan (incoming, prologue) -----------------------------------

/// The full incoming-parameter plan for a function: where each param arrives,
/// plus whether the indirect-result pointer arrives in x8.
pub const ParamPlan = struct {
    /// True when the return type is a >16B aggregate (sret): the caller hands a
    /// result-buffer pointer in x8 (the callee must save it before the body
    /// clobbers x8).
    sret_in_x8: bool,
    /// One location per parameter, in declaration order. Owned by the caller;
    /// free with `gpa.free`.
    locs: []ParamLoc,

    pub fn deinit(self: *ParamPlan, gpa: std.mem.Allocator) void {
        gpa.free(self.locs);
        self.* = undefined;
    }
};

/// Plan the incoming parameters of a function with signature `params -> ret`.
/// The result `locs` slice is allocated with `gpa`. Reproduces the M11 prologue
/// NGRN/NSAA walk (Codegen.lowerFn ~519) exactly.
pub fn planParams(
    gpa: std.mem.Allocator,
    params: []const Type,
    ret: Type,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
) error{OutOfMemory}!ParamPlan {
    const locs = try gpa.alloc(ParamLoc, params.len);
    errdefer gpa.free(locs);
    _ = walkAbi(params, locs, layouts, enum_layouts);
    return .{
        .sret_in_x8 = classifyRet(ret, layouts, enum_layouts) == .sret,
        .locs = locs,
    };
}

// ---- call plan (outgoing, call site) ---------------------------------------

/// The full outbound-argument plan for one call site: where each argument goes,
/// the total NSAA stack-arg bytes (drives the frame's `out_base`), and whether
/// the call needs x8 (an indirect result the caller must point at its buffer).
pub const CallPlan = struct {
    /// One location per argument, in source order. Owned by the caller; free
    /// with `gpa.free`.
    locs: []ArgLoc,
    /// Total outgoing stack-arg bytes (NSAA), BEFORE the 16-byte round-up the
    /// frame applies. `FrameLayout.out_base` is `roundUp16(max over calls)`.
    nsaa_bytes: u32,
    /// True when the callee returns a >16B aggregate: the caller must set x8 to
    /// its result buffer before the `bl`.
    sret_in_x8: bool,

    pub fn deinit(self: *CallPlan, gpa: std.mem.Allocator) void {
        gpa.free(self.locs);
        self.* = undefined;
    }
};

/// Plan the outbound arguments of a call to a callee with signature
/// `args -> ret`. The result `locs` slice is allocated with `gpa`. Reproduces
/// the M11 `genCallInner` NGRN/NSAA walk (~992) exactly, so `nsaa_bytes` equals
/// the stack bytes the emission writes (matching what the frame reserves).
pub fn planCall(
    gpa: std.mem.Allocator,
    args: []const Type,
    ret: Type,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
) error{OutOfMemory}!CallPlan {
    const locs = try gpa.alloc(ArgLoc, args.len);
    errdefer gpa.free(locs);
    const nsaa = walkAbi(args, locs, layouts, enum_layouts);
    return .{
        .locs = locs,
        .nsaa_bytes = nsaa,
        .sret_in_x8 = classifyRet(ret, layouts, enum_layouts) == .sret,
    };
}

// ===========================================================================
// TESTS
// ===========================================================================

const testing = std.testing;

// Build minimal layout tables for tests. Struct id i has size `sizes[i]`,
// align 8. (The ABI only consults size for aggregates.)
fn mkLayouts(gpa: std.mem.Allocator, sizes: []const u32) ![]Layout {
    const ls = try gpa.alloc(Layout, sizes.len);
    for (sizes, 0..) |s, i| {
        ls[i] = .{
            .name = "T",
            .field_names = &.{},
            .field_types = &.{},
            .offsets = &.{},
            .size = s,
            .@"align" = 8,
        };
    }
    return ls;
}

fn structOf(id: u32) Type {
    return Type.structT(id);
}

test "classify: scalar / reg_pair / indirect" {
    const gpa = testing.allocator;
    // struct ids: 0 => 8B (reg_pair, 1 ebs), 1 => 16B (reg_pair, 2 ebs),
    //             2 => 24B (indirect)
    const ls = try mkLayouts(gpa, &.{ 8, 16, 24 });
    defer gpa.free(ls);
    const el: []const EnumLayout = &.{};

    try testing.expectEqual(AbiClass.scalar, classify(Type.int, ls, el));
    try testing.expectEqual(AbiClass.scalar, classify(Type.bool, ls, el));
    try testing.expectEqual(AbiClass.reg_pair, classify(Type.str, ls, el)); // 16B
    try testing.expectEqual(AbiClass.reg_pair, classify(structOf(0), ls, el));
    try testing.expectEqual(AbiClass.reg_pair, classify(structOf(1), ls, el));
    try testing.expectEqual(AbiClass.indirect, classify(structOf(2), ls, el));

    try testing.expectEqual(@as(u32, 1), eightbytes(8));
    try testing.expectEqual(@as(u32, 2), eightbytes(16));
    try testing.expectEqual(@as(u32, 2), eightbytes(9));
}

test "classifyRet: unit / scalar / reg_pair / sret" {
    const gpa = testing.allocator;
    const ls = try mkLayouts(gpa, &.{ 8, 16, 24 });
    defer gpa.free(ls);
    const el: []const EnumLayout = &.{};

    try testing.expectEqual(RetLoc.none, classifyRet(Type.unit, ls, el));
    try testing.expectEqual(@as(u8, 1), classifyRet(Type.int, ls, el).reg.regs);
    try testing.expectEqual(@as(u8, 2), classifyRet(Type.str, ls, el).reg.regs); // 16B pair
    try testing.expectEqual(@as(u8, 1), classifyRet(structOf(0), ls, el).reg.regs); // 8B
    try testing.expectEqual(@as(u8, 2), classifyRet(structOf(1), ls, el).reg.regs); // 16B
    try testing.expectEqual(RetLoc.sret, classifyRet(structOf(2), ls, el)); // 24B
}

test "planParams: all-scalar in x0..x7" {
    const gpa = testing.allocator;
    const ls: []const Layout = &.{};
    const el: []const EnumLayout = &.{};
    const params = [_]Type{ Type.int, Type.bool, Type.int };
    var plan = try planParams(gpa, &params, Type.int, ls, el);
    defer plan.deinit(gpa);

    try testing.expect(!plan.sret_in_x8);
    try testing.expectEqual(@as(u8, 0), plan.locs[0].gpr.first);
    try testing.expectEqual(@as(u8, 1), plan.locs[0].gpr.count);
    try testing.expectEqual(@as(u8, 1), plan.locs[1].gpr.first);
    try testing.expectEqual(@as(u8, 2), plan.locs[2].gpr.first);
}

test "planParams: str occupies a 2-reg run (eightbytes pair)" {
    const gpa = testing.allocator;
    const ls: []const Layout = &.{};
    const el: []const EnumLayout = &.{};
    // int, str, int: str takes x1,x2; the trailing int gets x3.
    const params = [_]Type{ Type.int, Type.str, Type.int };
    var plan = try planParams(gpa, &params, Type.unit, ls, el);
    defer plan.deinit(gpa);

    try testing.expectEqual(@as(u8, 0), plan.locs[0].gpr.first);
    try testing.expectEqual(@as(u8, 1), plan.locs[1].gpr.first);
    try testing.expectEqual(@as(u8, 2), plan.locs[1].gpr.count);
    try testing.expectEqual(@as(u8, 3), plan.locs[2].gpr.first);
}

test "planParams: <=16B struct reg-pair, >16B struct indirect (gpr_ptr)" {
    const gpa = testing.allocator;
    const ls = try mkLayouts(gpa, &.{ 16, 24 });
    defer gpa.free(ls);
    const el: []const EnumLayout = &.{};
    const params = [_]Type{ structOf(0), structOf(1) }; // 16B pair, 24B indirect
    var plan = try planParams(gpa, &params, Type.unit, ls, el);
    defer plan.deinit(gpa);

    // 16B struct => 2-reg pair x0,x1.
    try testing.expectEqual(@as(u8, 0), plan.locs[0].gpr.first);
    try testing.expectEqual(@as(u8, 2), plan.locs[0].gpr.count);
    // 24B struct => indirect pointer in x2.
    try testing.expectEqual(@as(u8, 2), plan.locs[1].gpr_ptr);
}

test "planParams: sret return reserves x8 but does NOT consume a GPR" {
    const gpa = testing.allocator;
    const ls = try mkLayouts(gpa, &.{24}); // 24B -> sret
    defer gpa.free(ls);
    const el: []const EnumLayout = &.{};
    const params = [_]Type{ Type.int, Type.int };
    var plan = try planParams(gpa, &params, structOf(0), ls, el);
    defer plan.deinit(gpa);

    try testing.expect(plan.sret_in_x8);
    // x8 is separate: the two int params STILL start at x0.
    try testing.expectEqual(@as(u8, 0), plan.locs[0].gpr.first);
    try testing.expectEqual(@as(u8, 1), plan.locs[1].gpr.first);
}

test "planCall: >=9 scalar args overflow to the stack (NSAA)" {
    const gpa = testing.allocator;
    const ls: []const Layout = &.{};
    const el: []const EnumLayout = &.{};
    const args = [_]Type{ Type.int, Type.int, Type.int, Type.int, Type.int, Type.int, Type.int, Type.int, Type.int, Type.int };
    var plan = try planCall(gpa, &args, Type.int, ls, el);
    defer plan.deinit(gpa);

    // First 8 -> x0..x7.
    var i: u8 = 0;
    while (i < 8) : (i += 1) {
        try testing.expectEqual(i, plan.locs[i].gpr.first);
        try testing.expectEqual(@as(u8, 1), plan.locs[i].gpr.count);
    }
    // args 9,10 -> stack at NSAA 0, 8.
    try testing.expectEqual(@as(u32, 0), plan.locs[8].stack.nsaa_off);
    try testing.expectEqual(@as(u32, 8), plan.locs[8].stack.bytes);
    try testing.expectEqual(@as(u32, 8), plan.locs[9].stack.nsaa_off);
    try testing.expectEqual(@as(u32, 16), plan.nsaa_bytes);
    try testing.expect(!plan.sret_in_x8);
}

test "planCall: AAPCS no-split — reg-pair that doesn't fit goes wholly to stack and forbids backfill" {
    const gpa = testing.allocator;
    const ls: []const Layout = &.{};
    const el: []const EnumLayout = &.{};
    // 7 ints (x0..x6, NGRN=7) then a str (2 eightbytes): 7+2 > 8, so the str
    // spills WHOLLY to the stack and NGRN becomes 8 — a trailing int must NOT
    // backfill x7; it spills too.
    const args = [_]Type{ Type.int, Type.int, Type.int, Type.int, Type.int, Type.int, Type.int, Type.str, Type.int };
    var plan = try planCall(gpa, &args, Type.unit, ls, el);
    defer plan.deinit(gpa);

    var i: u8 = 0;
    while (i < 7) : (i += 1) try testing.expectEqual(i, plan.locs[i].gpr.first);
    // str: wholly on stack (16 bytes) at NSAA 0.
    try testing.expectEqual(@as(u32, 0), plan.locs[7].stack.nsaa_off);
    try testing.expectEqual(@as(u32, 16), plan.locs[7].stack.bytes);
    // trailing int: NO backfill into x7 — it lands on the stack at NSAA 16.
    try testing.expectEqual(@as(u32, 16), plan.locs[8].stack.nsaa_off);
    try testing.expectEqual(@as(u32, 24), plan.nsaa_bytes);
}

test "planCall: indirect aggregate beyond x7 passes its pointer on the stack (stack_ptr)" {
    const gpa = testing.allocator;
    const ls = try mkLayouts(gpa, &.{24}); // 24B indirect
    defer gpa.free(ls);
    const el: []const EnumLayout = &.{};
    // 8 ints fill x0..x7; the 9th arg (a >16B struct) passes its pointer on the
    // stack (8 bytes), NOT in a GPR.
    const args = [_]Type{ Type.int, Type.int, Type.int, Type.int, Type.int, Type.int, Type.int, Type.int, structOf(0) };
    var plan = try planCall(gpa, &args, Type.unit, ls, el);
    defer plan.deinit(gpa);

    try testing.expectEqual(@as(u32, 0), plan.locs[8].stack_ptr);
    try testing.expectEqual(@as(u32, 8), plan.nsaa_bytes);
}

test "planCall: aggregate return sets sret_in_x8" {
    const gpa = testing.allocator;
    const ls = try mkLayouts(gpa, &.{24});
    defer gpa.free(ls);
    const el: []const EnumLayout = &.{};
    const args = [_]Type{Type.int};
    var plan = try planCall(gpa, &args, structOf(0), ls, el);
    defer plan.deinit(gpa);

    try testing.expect(plan.sret_in_x8);
    // the result does not consume a GPR; the int arg is still x0.
    try testing.expectEqual(@as(u8, 0), plan.locs[0].gpr.first);
    try testing.expectEqual(@as(u32, 0), plan.nsaa_bytes);
}

test "planCall: no args -> empty locs, zero nsaa" {
    const gpa = testing.allocator;
    const ls: []const Layout = &.{};
    const el: []const EnumLayout = &.{};
    const args = [_]Type{};
    var plan = try planCall(gpa, &args, Type.int, ls, el);
    defer plan.deinit(gpa);

    try testing.expectEqual(@as(usize, 0), plan.locs.len);
    try testing.expectEqual(@as(u32, 0), plan.nsaa_bytes);
}
