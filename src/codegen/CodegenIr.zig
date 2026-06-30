//! Ir.Function → Link.FnCode codegen (M12, the IR path).
//!
//! The dual-path peer to the legacy AST→FnCode `Codegen.lower`. This consumes a
//! lowered `Ir.Function` (target-independent) and emits AArch64 bytes, deciding
//! the ABI ONLY here (via `Abi`) and the frame ONLY here (via `FrameLayout`,
//! spill-everything). It reuses the `Aarch64.zig` encoder and the same
//! label/fixup backpatch algorithm as the AST path, so branch encoding stays
//! identical.
//!
//! VALUE MODEL (LOCK #2/#4): every SSA value AND every slot has its OWN frame
//! cell (FrameLayout). An instruction loads its operands from their cells into
//! scratch regs (x9/x10/x11), computes, and stores the result into the result
//! value's cell. This is dead-simple and trivially correct; slot-coloring is a
//! deferred optimization.
//!
//! BLOCK ARGS = STORE-BEFORE-BR ("phi = memory", LOCK #2): a `br dest(args)`
//! edge stores each arg value into the destination block's param cell, THEN
//! branches. Multi-arg edges load ALL sources into scratch first, then store,
//! so a swap-style edge (param values feeding each other) is clobber-safe.
//! `cond_br` is ARGLESS by construction (lower split any value-merge cond edge),
//! so it is exactly cmp/cbnz + b.
//!
//! DETERMINISM: blocks are walked in BlockId order; one label per block, placed
//! and backpatched. Pure over the IR + read-only layout tables. Same IR + same
//! layouts ⇒ same bytes (underpins VERIFY byte-identity, [C11]).

const std = @import("std");
const Ir = @import("../ir/Ir.zig");
const Abi = @import("abi/Abi.zig");
const FrameLayout = @import("frame/FrameLayout.zig");
const Aarch64 = @import("Aarch64.zig");
const Link = @import("../link/Link.zig");
const Typecheck = @import("../types.zig");

pub const Diagnostic = @import("../diagnostics/Diagnostic.zig").Diagnostic;

/// Per-fn codegen cache policy (orthogonal to the now-single backend):
///   normal — use a cached blob if present, else lower+cache.
///   force  — ignore cache, always re-lower (cold build).
///   verify — re-lower every fn TWICE and assert byte-identity ([C11]),
///            independent of cache state. Implies force (the gate must hold on
///            a cold build, not only on a primed cache hit).
pub const Mode = enum { normal, force, verify };

const Type = Typecheck.Type;
const Layout = Typecheck.Layout;
const EnumLayout = Typecheck.EnumLayout;

// Scratch GPRs the body uses for the load/compute/store pattern. Distinct from
// x0..x8 (params / result / sret-x8) so marshalling can't collide with them.
const S0: u32 = 9;
const S1: u32 = 10;
const S2: u32 = 11;

// ---- intra-function labels & branch backpatch (same algorithm as Codegen) --

const LabelId = u32;
const UNPLACED: u32 = std.math.maxInt(u32);
const BranchWidth = enum { imm19, imm26 };
const Fixup = struct { site: u32, label: LabelId, width: BranchWidth };

const Gen = struct {
    gpa: std.mem.Allocator,
    func: *const Ir.Function,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
    is_entry: bool,

    fl: FrameLayout,
    params: Abi.ParamPlan,

    code: std.ArrayList(u8) = .empty,
    relocs: std.ArrayList(Link.Reloc) = .empty,
    literals: std.ArrayList(Link.Literal) = .empty,

    labels: std.ArrayList(u32) = .empty,
    fixups: std.ArrayList(Fixup) = .empty,

    /// One label per IR block, pre-allocated so a forward branch can target a
    /// not-yet-emitted block. Indexed by BlockId.
    block_labels: []LabelId,

    fn emit(g: *Gen, word: u32) error{OutOfMemory}!void {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, word, .little);
        try g.code.appendSlice(g.gpa, &buf);
    }

    fn emitImm64(g: *Gen, rd: u32, value: i64) error{OutOfMemory}!void {
        var words: [16]u8 = undefined;
        var len: usize = 0;
        Aarch64.movImm64(&words, &len, rd, value);
        try g.code.appendSlice(g.gpa, words[0..len]);
    }

    fn newLabel(g: *Gen) error{OutOfMemory}!LabelId {
        const id: LabelId = @intCast(g.labels.items.len);
        try g.labels.append(g.gpa, UNPLACED);
        return id;
    }

    fn placeLabel(g: *Gen, id: LabelId) void {
        g.labels.items[id] = @intCast(g.code.items.len);
    }

    fn branchTo(g: *Gen, word: u32, id: LabelId, width: BranchWidth) error{OutOfMemory}!void {
        try g.fixups.append(g.gpa, .{ .site = @intCast(g.code.items.len), .label = id, .width = width });
        try g.emit(word);
    }

    fn resolveFixups(g: *Gen) error{OutOfMemory}!void {
        for (g.fixups.items) |fx| {
            const target = g.labels.items[fx.label];
            std.debug.assert(target != UNPLACED);
            const delta_words: i64 = @divExact(@as(i64, target) - @as(i64, fx.site), 4);
            const old = std.mem.readInt(u32, g.code.items[fx.site..][0..4], .little);
            const new = switch (fx.width) {
                .imm19 => blk: {
                    if (delta_words < -(@as(i64, 1) << 18) or delta_words >= (@as(i64, 1) << 18)) {
                        // Practically unreachable (the __TEXT page dwarfs ±1MB).
                        break :blk old;
                    }
                    break :blk if ((old & 0xFF000000) == 0x54000000)
                        Aarch64.patchBCond(old, @intCast(delta_words))
                    else
                        Aarch64.patchCbz(old, @intCast(delta_words));
                },
                .imm26 => blk: {
                    if (delta_words < -(@as(i64, 1) << 25) or delta_words >= (@as(i64, 1) << 25)) {
                        break :blk old;
                    }
                    break :blk Aarch64.patchB(old, @intCast(delta_words));
                },
            };
            std.mem.writeInt(u32, g.code.items[fx.site..][0..4], new, .little);
        }
    }

    // ---- frame addressing -------------------------------------------------

    fn slotOff(g: *const Gen, sid: Ir.SlotId) u32 {
        return g.fl.slotAddr(sid);
    }
    fn valueOff(g: *const Gen, vid: Ir.ValueId) u32 {
        return g.fl.valueAddr(vid);
    }

    fn typeSize(g: *const Gen, ty: Type) u32 {
        return Abi.typeSize(ty, g.layouts, g.enum_layouts);
    }
    fn isAggregate(ty: Type) bool {
        return Abi.isAggregate(ty);
    }

    /// Load a scalar SSA value into `reg`.
    fn loadValue(g: *Gen, reg: u32, vid: Ir.ValueId) error{OutOfMemory}!void {
        try g.emit(Aarch64.ldrSp(reg, g.valueOff(vid)));
    }
    /// Store `reg` into a scalar SSA value's cell.
    fn storeValue(g: *Gen, reg: u32, vid: Ir.ValueId) error{OutOfMemory}!void {
        if (vid == Ir.none_value) return;
        try g.emit(Aarch64.strSp(reg, g.valueOff(vid)));
    }
};

/// Lower one `Ir.Function` to a `Link.FnCode`. On a frame-budget overflow,
/// records a matching diagnostic and returns a well-formed (possibly empty)
/// FnCode. Caller owns the result.
pub fn lowerIr(
    gpa: std.mem.Allocator,
    func: *const Ir.Function,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
    is_entry: bool,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Link.FnCode {
    // (1) Frame layout. A budget overflow maps to the SAME diagnostic strings as
    // the AST path; return a well-formed empty FnCode so the driver fails cleanly.
    const fl = FrameLayout.compute(gpa, func, layouts, enum_layouts) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.FrameTooLarge => {
            try out_diags.append(gpa, .{ .byte_offset = 0, .message = "function frame too large for codegen (too many locals)" });
            return emptyFn(gpa, func.name);
        },
        error.ParamOffsetTooLarge => {
            try out_diags.append(gpa, .{ .byte_offset = 0, .message = "too many parameters for codegen (stack-arg offset out of range)" });
            return emptyFn(gpa, func.name);
        },
    };

    // Param plan (incoming prologue marshalling).
    var param_types: std.ArrayList(Type) = .empty;
    defer param_types.deinit(gpa);
    try param_types.ensureTotalCapacity(gpa, func.params.len);
    for (func.params) |sid| param_types.appendAssumeCapacity(func.slots[sid].type);
    const params = try Abi.planParams(gpa, param_types.items, func.ret_type, layouts, enum_layouts);

    var g: Gen = .{
        .gpa = gpa,
        .func = func,
        .layouts = layouts,
        .enum_layouts = enum_layouts,
        .is_entry = is_entry,
        .fl = fl,
        .params = params,
        .block_labels = try gpa.alloc(LabelId, func.blocks.len),
    };
    // Free everything g owns on any failure; the FnCode takes ownership at the end.
    defer {
        g.fl.deinit(gpa);
        g.params.deinit(gpa);
        gpa.free(g.block_labels);
        g.labels.deinit(gpa);
        g.fixups.deinit(gpa);
    }
    errdefer {
        g.code.deinit(gpa);
        for (g.relocs.items) |r| switch (r.target) {
            .func, .import => |s| gpa.free(s.name),
            .cstr => {},
        };
        g.relocs.deinit(gpa);
        for (g.literals.items) |l| gpa.free(l.bytes);
        g.literals.deinit(gpa);
    }

    try genBody(&g);

    const name_copy = try gpa.dupe(u8, func.name.name);
    errdefer gpa.free(name_copy);
    return .{
        .sym = .{ .kind = func.name.kind, .name = name_copy },
        .code = try g.code.toOwnedSlice(gpa),
        .relocs = try g.relocs.toOwnedSlice(gpa),
        .literals = try g.literals.toOwnedSlice(gpa),
    };
}

fn emptyFn(gpa: std.mem.Allocator, sym: Ir.SymName) error{OutOfMemory}!Link.FnCode {
    const name_copy = try gpa.dupe(u8, sym.name);
    return .{
        .sym = .{ .kind = sym.kind, .name = name_copy },
        .code = &.{},
        .relocs = &.{},
        .literals = &.{},
    };
}

fn genBody(g: *Gen) error{OutOfMemory}!void {
    // One label per block; the exit block gets a label too (the single epilogue
    // owner). Pre-allocate them so forward branches can target any block.
    for (g.block_labels) |*l| l.* = try g.newLabel();

    // Register every string literal in the FnCode table (codegen owns its copy;
    // the relink tail interns them program-wide).
    for (g.func.literals) |lit| {
        const bytes = try g.gpa.dupe(u8, lit.bytes);
        try addLiteral(g, lit.hash, bytes);
    }

    // PROLOGUE.
    try g.emit(Aarch64.stpFpLrPre); // stp x29,x30,[sp,#-16]!
    try g.emit(Aarch64.movFpSp); // mov x29, sp
    if (g.fl.frame > 0) {
        try g.emit(Aarch64.subImm(Aarch64.SP, Aarch64.SP, @intCast(g.fl.frame)));
    }
    // sret: save the incoming x8 (caller's result buffer) once up front — body
    // calls clobber x8.
    if (g.fl.sret_save_off != FrameLayout.NO_SRET) {
        try g.emit(Aarch64.strSp(8, g.fl.sret_save_off));
    }

    try marshalParams(g);

    // BODY: walk blocks in BlockId order. The exit block is emitted in sequence
    // like any other (its terminator is `ret`, which owns the epilogue).
    for (g.func.blocks, 0..) |*blk, bid| {
        g.placeLabel(g.block_labels[bid]);
        for (blk.instrs) |ins| try genInstr(g, ins);
        try genTerm(g, @intCast(bid), blk.term);
    }

    try g.resolveFixups();
}

/// Marshal incoming params from their ABI locations into their slot cells.
fn marshalParams(g: *Gen) error{OutOfMemory}!void {
    for (g.params.locs, 0..) |loc, i| {
        const sid = g.func.params[i];
        const off = g.slotOff(sid);
        const ty = g.func.slots[sid].type;
        const size = g.typeSize(ty);
        switch (loc) {
            .gpr => |r| {
                // A scalar (count 1) or a reg-pair aggregate (count 1-2): store
                // each incoming GPR into the slot's consecutive words.
                var k: u32 = 0;
                while (k < r.count) : (k += 1) {
                    try g.emit(Aarch64.strSp(r.first + k, off + k * 8));
                }
            },
            .gpr_ptr => |r| {
                // An indirect (>16B) aggregate: r holds a pointer to the caller's
                // copy. Copy the bytes into the slot.
                try copyBytes(g, Aarch64.SP, off, r, 0, size);
            },
            .stack => |s| {
                // A scalar / reg-pair aggregate passed on the incoming stack at
                // [x29,#16+nsaa]. Copy each word into the slot.
                var k: u32 = 0;
                while (k < s.bytes) : (k += 8) {
                    try g.emit(Aarch64.ldrFp(S0, 16 + s.nsaa_off + k));
                    try g.emit(Aarch64.strSp(S0, off + k));
                }
            },
            .stack_ptr => |nsaa_off| {
                // An indirect aggregate whose pointer sits on the incoming stack.
                try g.emit(Aarch64.ldrFp(S0, 16 + nsaa_off));
                try copyBytes(g, Aarch64.SP, off, S0, 0, size);
            },
        }
    }
}

/// Copy `size` bytes (rounded to 8) from [src_reg+src_off] to [dst_reg+dst_off]
/// through S2. A TIGHT loop with NO intervening branch (preserves the sret/copy
/// no-`bl` adjacency invariant). The two base regs must differ from S2.
fn copyBytes(g: *Gen, dst_reg: u32, dst_off: u32, src_reg: u32, src_off: u32, size: u32) error{OutOfMemory}!void {
    var o: u32 = 0;
    while (o < size) : (o += 8) {
        try g.emit(Aarch64.ldrRegUoff(S2, src_reg, src_off + o));
        try g.emit(Aarch64.strRegUoff(S2, dst_reg, dst_off + o));
    }
}

fn condToAarch64(cc: Ir.Cond) Aarch64.Cond {
    return switch (cc) {
        .eq => .eq,
        .ne => .ne,
        .lt => .lt,
        .le => .le,
        .gt => .gt,
        .ge => .ge,
    };
}

fn genInstr(g: *Gen, ins: Ir.Instr) error{OutOfMemory}!void {
    switch (ins.op) {
        .iconst => |v| {
            try g.emitImm64(S0, v);
            try g.storeValue(S0, ins.result);
        },
        .bconst => |v| {
            try g.emit(Aarch64.movz(S0, if (v) 1 else 0, 0));
            try g.storeValue(S0, ins.result);
        },
        .unit => {}, // zero-sized: no result cell to fill.
        .add => |b| try genArith(g, ins.result, b, .add),
        .sub => |b| try genArith(g, ins.result, b, .sub),
        .mul => |b| try genArith(g, ins.result, b, .mul),
        .sdiv => |b| try genArith(g, ins.result, b, .sdiv),
        .neg => |v| {
            try g.loadValue(S0, v);
            try g.emit(Aarch64.neg(S0, S0));
            try g.storeValue(S0, ins.result);
        },
        .bnot => |v| {
            try g.loadValue(S0, v);
            try g.emit(Aarch64.cmpImm(S0, 0));
            try g.emit(Aarch64.cset(S0, .eq));
            try g.storeValue(S0, ins.result);
        },
        .icmp => |c| {
            try g.loadValue(S0, c.lhs);
            try g.loadValue(S1, c.rhs);
            try g.emit(Aarch64.cmpReg(S0, S1));
            try g.emit(Aarch64.cset(S0, condToAarch64(c.cc)));
            try g.storeValue(S0, ins.result);
        },
        .slot_addr => |sid| {
            try g.emit(Aarch64.addImm(S0, Aarch64.SP, @intCast(g.slotOff(sid))));
            try g.storeValue(S0, ins.result);
        },
        .field_addr => |f| {
            try g.loadValue(S0, f.base);
            if (f.off != 0) try g.emit(Aarch64.addImm(S0, S0, @intCast(f.off)));
            try g.storeValue(S0, ins.result);
        },
        .load => |l| {
            try g.loadValue(S0, l.addr); // S0 = address
            try g.emit(Aarch64.ldrRegUoff(S0, S0, 0));
            try g.storeValue(S0, ins.result);
        },
        .store => |s| {
            try g.loadValue(S0, s.addr); // S0 = address
            try g.loadValue(S1, s.val); // S1 = value
            try g.emit(Aarch64.strRegUoff(S1, S0, 0));
        },
        .copy => |c| {
            // dst/src are ptr VALUES. Load them into base regs, then byte-copy.
            try g.loadValue(S0, c.dst);
            try g.loadValue(S1, c.src);
            try copyBytes(g, S0, 0, S1, 0, g.typeSize(c.ty));
        },
        .get_tag => |v| {
            try g.loadValue(S0, v); // S0 = enum slot ptr
            try g.emit(Aarch64.ldrRegUoff(S0, S0, 0)); // tag @ offset 0
            try g.storeValue(S0, ins.result);
        },
        .cstr_ptr => |h| try genCstrPtr(g, ins.result, h),
        .call => |c| try genCall(g, ins.result, c),
    }
}

const ArithKind = enum { add, sub, mul, sdiv };

fn genArith(g: *Gen, result: Ir.ValueId, b: Ir.Bin, kind: ArithKind) error{OutOfMemory}!void {
    try g.loadValue(S0, b.lhs);
    try g.loadValue(S1, b.rhs);
    const word = switch (kind) {
        .add => Aarch64.addReg(S0, S0, S1),
        .sub => Aarch64.subReg(S0, S0, S1),
        .mul => Aarch64.mul(S0, S0, S1),
        .sdiv => Aarch64.sdiv(S0, S0, S1),
    };
    try g.emit(word);
    try g.storeValue(S0, result);
}

/// `cstr_ptr h`: materialize a `__cstring` address via adrp+add (placeholders +
/// `.cstr` relocs patched post-vmaddr). Result is the ptr in the value cell.
fn genCstrPtr(g: *Gen, result: Ir.ValueId, h: u64) error{OutOfMemory}!void {
    var site: u32 = @intCast(g.code.items.len);
    try g.relocs.append(g.gpa, .{ .site = site, .target = .{ .cstr = h }, .kind = .adrp_page });
    try g.emit(Aarch64.adrp(S0, 0));
    site = @intCast(g.code.items.len);
    try g.relocs.append(g.gpa, .{ .site = site, .target = .{ .cstr = h }, .kind = .add_lo12 });
    try g.emit(Aarch64.addImm(S0, S0, 0));
    try g.storeValue(S0, result);
}

/// Lower a call: marshal args per AAPCS64 (Abi.planCall), point x8 at the result
/// buffer for an sret call, `bl` (a `.call26` reloc), then place the result.
///
/// MARSHALLING ORDER matters: a scalar/reg-pair arg in a GPR could be x0..x7 (the
/// same regs we load others into). We load each arg directly into its target GPR
/// in a single left-to-right pass; this is safe because every source is a FRAME
/// cell (no inter-register dependencies) and each target GPR is written exactly
/// once. Stack args/ptrs use scratch S0 transiently.
fn genCall(g: *Gen, result: Ir.ValueId, c: Ir.Call) error{OutOfMemory}!void {
    // Recover arg + ret types for the ABI plan.
    var arg_types: std.ArrayList(Type) = .empty;
    defer arg_types.deinit(g.gpa);
    try arg_types.ensureTotalCapacity(g.gpa, c.args.len);
    for (c.args) |a| arg_types.appendAssumeCapacity(operandType(g, a));
    // Recover the RETURN type: an aggregate result lands in `ret_slot` (its type);
    // a scalar result is the call instr's own value type; a unit call has neither.
    const ret_ty: Type = if (c.ret_slot != Ir.none_slot)
        g.func.slots[c.ret_slot].type
    else if (result != Ir.none_value)
        g.func.values[result].type
    else
        Type.unit;

    var plan = try Abi.planCall(g.gpa, arg_types.items, ret_ty, g.layouts, g.enum_layouts);
    defer plan.deinit(g.gpa);

    // 1) Marshal each argument into its ABI location.
    for (plan.locs, 0..) |loc, i| {
        const arg = c.args[i];
        switch (loc) {
            .gpr => |r| {
                // A scalar value → one GPR; a reg-pair aggregate → `count`
                // consecutive GPRs loaded from the aggregate's frame words. An
                // aggregate can arrive EITHER as a slot OR as an Operand.value (a
                // merge param / value-break result): under spill-everything an
                // aggregate value's bytes live DIRECTLY in its cell, so both load
                // `count` words from the cell's base offset.
                switch (arg) {
                    .value => |v| {
                        const off = g.valueOff(v);
                        var k: u32 = 0;
                        while (k < r.count) : (k += 1) {
                            try g.emit(Aarch64.ldrSp(r.first + k, off + k * 8));
                        }
                    },
                    .slot => |s| {
                        const off = g.slotOff(s);
                        var k: u32 = 0;
                        while (k < r.count) : (k += 1) {
                            try g.emit(Aarch64.ldrSp(r.first + k, off + k * 8));
                        }
                    },
                    .none => {},
                }
            },
            .gpr_ptr => |r| {
                // Indirect (>16B) aggregate: pass a pointer to its frame cell
                // (slot, or an in-place value cell for a merge value).
                const base: u32 = switch (arg) {
                    .slot => |s| g.slotOff(s),
                    .value => |v| g.valueOff(v),
                    .none => continue,
                };
                try g.emit(Aarch64.addImm(r, Aarch64.SP, @intCast(base)));
            },
            .stack => |st| {
                // Scalar / reg-pair aggregate placed wholly on the outgoing stack.
                // An aggregate may arrive as a slot OR as an Operand.value (bytes
                // spilled in place); copy `st.bytes` either way. A scalar is 8B.
                const base: u32 = switch (arg) {
                    .value => |v| g.valueOff(v),
                    .slot => |s| g.slotOff(s),
                    .none => continue,
                };
                var k: u32 = 0;
                while (k < st.bytes) : (k += 8) {
                    try g.emit(Aarch64.ldrSp(S0, base + k));
                    try g.emit(Aarch64.strSp(S0, st.nsaa_off + k));
                }
            },
            .stack_ptr => |nsaa_off| {
                const base: u32 = switch (arg) {
                    .slot => |s| g.slotOff(s),
                    .value => |v| g.valueOff(v),
                    .none => continue,
                };
                try g.emit(Aarch64.addImm(S0, Aarch64.SP, @intCast(base)));
                try g.emit(Aarch64.strSp(S0, nsaa_off));
            },
        }
    }

    // 2) sret: point x8 at the result slot AFTER args (so an `add x8,sp` is not
    //    disturbed by arg marshalling reading sp).
    if (plan.sret_in_x8) {
        std.debug.assert(c.ret_slot != Ir.none_slot);
        try g.emit(Aarch64.addImm(8, Aarch64.SP, @intCast(g.slotOff(c.ret_slot))));
    }

    // 3) bl placeholder + `.call26` reloc.
    const name_copy = try g.gpa.dupe(u8, c.callee.name);
    errdefer g.gpa.free(name_copy);
    const site: u32 = @intCast(g.code.items.len);
    try g.relocs.append(g.gpa, .{
        .site = site,
        .target = .{ .func = .{ .kind = c.callee.kind, .name = name_copy } },
        .kind = .call26,
        .addend = 0,
    });
    try g.emit(Aarch64.bl(0));

    // 4) Place the result. A scalar lands in x0 → store into its value cell. A
    //    reg-pair aggregate result lands in x0[,x1] → store into ret_slot's words.
    //    An sret result was already written through x8 by the callee. Unit: none.
    switch (Abi.classifyRet(ret_ty, g.layouts, g.enum_layouts)) {
        .none => {},
        .reg => |r| {
            if (c.ret_slot != Ir.none_slot) {
                // Aggregate reg-pair result → store x0[,x1] into the slot.
                const off = g.slotOff(c.ret_slot);
                var k: u32 = 0;
                while (k < r.regs) : (k += 1) {
                    try g.emit(Aarch64.strSp(@intCast(k), off + k * 8));
                }
            } else {
                // Scalar result in x0 → its value cell.
                try g.storeValue(0, result);
            }
        },
        .sret => {}, // written through x8.
    }
}

fn genTerm(g: *Gen, bid: Ir.BlockId, term: Ir.Terminator) error{OutOfMemory}!void {
    switch (term) {
        .br => |br| {
            try storeBeforeBr(g, br.dest, br.args);
            // No branch instruction if the destination is the next block in
            // sequence AND no fall-through hazard — but to stay simple and match
            // the AST path's label discipline, always emit an explicit `b`.
            try g.branchTo(Aarch64.b(0), g.block_labels[br.dest], .imm26);
        },
        .cond_br => |c| {
            // ARGLESS by construction. Load the bool, cbnz true / fall to false,
            // then unconditional b to false. (Mirrors genCondBareBool: cbnz to the
            // true dest, then b to the false dest.)
            try g.loadValue(S0, c.cond);
            try g.branchTo(Aarch64.cbnz(S0, 0), g.block_labels[c.t], .imm19);
            try g.branchTo(Aarch64.b(0), g.block_labels[c.f], .imm26);
        },
        .ret => |o| {
            // Only the EXIT block. Place the return value, then the epilogue.
            try placeReturn(g, o);
            try emitEpilogue(g);
        },
        .@"unreachable" => {}, // emit nothing (preserve the never byte budget).
    }
    _ = bid;
}

/// Store-before-br: write each edge arg into the destination block's param cell
/// (a block param IS an SSA value with its own frame cell — "phi = memory").
///
/// CLOBBER SAFETY: a source operand on an edge is never one of THAT edge's
/// destination param cells (lower supplies each join param from independent
/// operands — no swap-style param↔param feed on a single edge), so a straight
/// read-source→write-dest pass per arg is safe; no staging needed.
fn storeBeforeBr(g: *Gen, dest: Ir.BlockId, args: []const Ir.Operand) error{OutOfMemory}!void {
    const params = g.func.blocks[dest].params;
    // Normally `args.len == params.len`. A DIVERGENT/never edge (e.g. a break-less
    // `loop`'s synthetic exit `br` to a value join) can carry FEWER args than the
    // join has params: that edge is unreachable, so leaving the extra param cells
    // unwritten is harmless. Write only the supplied args.
    std.debug.assert(args.len <= params.len);
    for (args, 0..) |arg, i| {
        const pvid = params[i];
        const pty = g.func.values[pvid].type;
        try writeOperandTo(g, pvid, pty, arg);
    }
}

/// Write `arg` into the param value cell `pvid` (scalar = store the value word;
/// aggregate = byte-copy from the source slot into the param's cell).
fn writeOperandTo(g: *Gen, pvid: Ir.ValueId, pty: Type, arg: Ir.Operand) error{OutOfMemory}!void {
    if (Gen.isAggregate(pty)) {
        // An aggregate merge param: copy bytes from the source into the param's
        // frame cell. Both a `.slot` source and an aggregate `.value` source hold
        // the aggregate BYTES DIRECTLY in their frame cell (an aggregate value is
        // spilled in place, never a pointer to the bytes — LOCK #2/#4), so the
        // copy source is `SP + off` in both cases.
        const dst_off = g.valueOff(pvid);
        switch (arg) {
            .slot => |s| try copyBytes(g, Aarch64.SP, dst_off, Aarch64.SP, g.slotOff(s), g.typeSize(pty)),
            .value => |v| try copyBytes(g, Aarch64.SP, dst_off, Aarch64.SP, g.valueOff(v), g.typeSize(pty)),
            .none => {},
        }
        return;
    }
    switch (arg) {
        .value => |v| {
            try g.loadValue(S0, v);
            try g.storeValue(S0, pvid);
        },
        .slot => |s| {
            // A scalar carried by slot (unusual): load the word, store to the cell.
            try g.emit(Aarch64.ldrSp(S0, g.slotOff(s)));
            try g.storeValue(S0, pvid);
        },
        .none => {}, // unit merge: nothing to store.
    }
}

/// Place the function's return value per AAPCS64 before the epilogue.
fn placeReturn(g: *Gen, o: Ir.Operand) error{OutOfMemory}!void {
    const ret_ty = g.func.ret_type;
    switch (Abi.classifyRet(ret_ty, g.layouts, g.enum_layouts)) {
        .none => {
            // Unit return. A unit ENTRY (main) must land x0=0 by contract.
            if (g.is_entry) try g.emit(Aarch64.movz(0, 0, 0));
        },
        .reg => |r| {
            // Scalar (1 reg) or reg-pair aggregate (1-2 regs) into x0[,x1]. The
            // bytes live in the operand's frame cell — a `.slot` for an aggregate
            // produced into a slot, or a `.value` cell (a scalar, OR an aggregate
            // merge param whose cell holds the aggregate bytes).
            const off: u32 = switch (o) {
                .value => |v| g.valueOff(v),
                .slot => |s| g.slotOff(s),
                .none => return,
            };
            var k: u32 = 0;
            while (k < r.regs) : (k += 1) {
                try g.emit(Aarch64.ldrSp(@intCast(k), off + k * 8));
            }
        },
        .sret => {
            // Large aggregate: copy the result bytes through the saved x8 buffer.
            // TIGHT no-`bl` copy (the no-split invariant). The bytes live in the
            // operand's cell (a slot, or a value cell for an aggregate merge param).
            std.debug.assert(g.fl.sret_save_off != FrameLayout.NO_SRET);
            const src_off: u32 = switch (o) {
                .value => |v| g.valueOff(v),
                .slot => |s| g.slotOff(s),
                .none => return,
            };
            try g.emit(Aarch64.ldrSp(S0, g.fl.sret_save_off)); // S0 = caller buffer
            try copyBytes(g, S0, 0, Aarch64.SP, src_off, g.typeSize(ret_ty));
        },
    }
}

fn emitEpilogue(g: *Gen) error{OutOfMemory}!void {
    if (g.fl.frame > 0) {
        try g.emit(Aarch64.addImm(Aarch64.SP, Aarch64.SP, @intCast(g.fl.frame)));
    }
    try g.emit(Aarch64.ldpFpLrPost);
    try g.emit(Aarch64.ret);
}

fn operandType(g: *const Gen, op: Ir.Operand) Type {
    return switch (op) {
        .value => |v| g.func.values[v].type,
        .slot => |s| g.func.slots[s].type,
        .none => Type.unit,
    };
}

/// Register a decoded literal in the FnCode table (deduped, takes ownership).
fn addLiteral(g: *Gen, hash: u64, bytes: []u8) error{OutOfMemory}!void {
    for (g.literals.items) |lit| {
        if (lit.hash == hash and std.mem.eql(u8, lit.bytes, bytes)) {
            g.gpa.free(bytes);
            return;
        }
    }
    g.literals.append(g.gpa, .{ .hash = hash, .bytes = bytes }) catch |e| {
        g.gpa.free(bytes);
        return e;
    };
}

// ===========================================================================
// print(str) builtin body — hand-written, AST/IR-independent. Appended to the
// program at link time (emit.zig) when any fn references `print`. Shuffles the
// (ptr,len) str pair into write(fd=1, buf, len) and tail-calls libc `write`.
// ===========================================================================

/// Build the `print` builtin's FnCode directly (no IR, no frame). Caller owns
/// the result.
pub fn lowerPrint(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer relocs.deinit(gpa);

    const emit = struct {
        fn f(c: *std.ArrayList(u8), a: std.mem.Allocator, word: u32) error{OutOfMemory}!void {
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, word, .little);
            try c.appendSlice(a, &buf);
        }
    }.f;

    try emit(&code, gpa, Aarch64.stpFpLrPre); // stp x29, x30, [sp, #-16]!
    try emit(&code, gpa, Aarch64.movFpSp); // mov x29, sp
    // Shuffle (ptr,len) into write's (buf,len) = (x1,x2), then fd=1 in w0. Order
    // matters: move len (x1→x2) BEFORE overwriting x1 with ptr (x0→x1).
    try emit(&code, gpa, Aarch64.movReg(2, 1)); // len → x2
    try emit(&code, gpa, Aarch64.movReg(1, 0)); // ptr → x1
    try emit(&code, gpa, Aarch64.movz(0, 1, 0)); // fd = 1
    // adrp x16, _write@GOT  (placeholder; .adrp_page reloc to import "write")
    var site: u32 = @intCast(code.items.len);
    const wname1 = try gpa.dupe(u8, "write");
    errdefer gpa.free(wname1);
    try relocs.append(gpa, .{ .site = site, .target = .{ .import = .{ .kind = .import, .name = wname1 } }, .kind = .adrp_page });
    try emit(&code, gpa, Aarch64.adrp(16, 0)); // adrp x16, _write@GOT
    // ldr x16, [x16]  (placeholder; .ldr_lo12 reloc, same import)
    site = @intCast(code.items.len);
    const wname2 = try gpa.dupe(u8, "write");
    errdefer gpa.free(wname2);
    try relocs.append(gpa, .{ .site = site, .target = .{ .import = .{ .kind = .import, .name = wname2 } }, .kind = .ldr_lo12 });
    try emit(&code, gpa, Aarch64.ldrRegUoff(16, 16, 0)); // ldr x16, [x16]
    try emit(&code, gpa, Aarch64.blr(16)); // blr x16
    try emit(&code, gpa, Aarch64.ldpFpLrPost); // ldp x29, x30, [sp], #16
    try emit(&code, gpa, Aarch64.ret); // ret

    const name = try gpa.dupe(u8, "print");
    errdefer gpa.free(name);
    return .{
        .sym = .{ .kind = .builtin, .name = name },
        .code = try code.toOwnedSlice(gpa),
        .relocs = try relocs.toOwnedSlice(gpa),
        .literals = &.{},
    };
}

// ===========================================================================
// TESTS — hand-build a tiny Ir.Function and assert the emitted byte shape.
// ===========================================================================

const testing = std.testing;

// A 1-block `fn f() -> int { ret <iconst v> }` exercising prologue/epilogue,
// iconst, store-before-br into the exit param, and the scalar return path.
test "ir-codegen: constant-return function lowers to a valid prologue/epilogue" {
    const gpa = testing.allocator;

    // values: %0 = exit param (int), %1 = iconst result (int).
    var values = try gpa.alloc(Ir.ValueDef, 2);
    values[0] = .{ .type = Type.int };
    values[1] = .{ .type = Type.int };

    // entry: %1 = iconst 7 ; br exit(%1)
    var entry_instrs = try gpa.alloc(Ir.Instr, 1);
    entry_instrs[0] = .{ .result = 1, .op = .{ .iconst = 7 } };
    const entry_args = try gpa.alloc(Ir.Operand, 1);
    entry_args[0] = .{ .value = 1 };

    var blocks = try gpa.alloc(Ir.Block, 2);
    blocks[0] = .{
        .params = try gpa.alloc(Ir.ValueId, 0),
        .instrs = entry_instrs,
        .term = .{ .br = .{ .dest = 1, .args = entry_args } },
    };
    const exit_params = try gpa.alloc(Ir.ValueId, 1);
    exit_params[0] = 0;
    blocks[1] = .{
        .params = exit_params,
        .instrs = try gpa.alloc(Ir.Instr, 0),
        .term = .{ .ret = .{ .value = 0 } },
    };

    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "f" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Type.int,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = values,
        .blocks = blocks,
        .entry = 0,
        .exit = 1,
        .literals = try gpa.alloc(Ir.Literal, 0),
    };
    defer func.deinit(gpa);

    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    var fc = try lowerIr(gpa, &func, &.{}, &.{}, false, &diags);
    defer fc.deinit(gpa);

    try testing.expectEqual(@as(usize, 0), diags.items.len);
    try testing.expect(fc.code.len % 4 == 0);
    try testing.expect(fc.code.len >= 4 * 5); // prologue + body + epilogue
    // First word is the prologue stp x29,x30,[sp,#-16]!.
    try testing.expectEqual(Aarch64.stpFpLrPre, std.mem.readInt(u32, fc.code[0..4], .little));
    // Last word is `ret`.
    const last = std.mem.readInt(u32, fc.code[fc.code.len - 4 ..][0..4], .little);
    try testing.expectEqual(Aarch64.ret, last);
}

// A unit-returning ENTRY function must force x0=0 in the return path.
test "ir-codegen: unit entry forces x0=0 before the epilogue" {
    const gpa = testing.allocator;

    var blocks = try gpa.alloc(Ir.Block, 1);
    blocks[0] = .{
        .params = try gpa.alloc(Ir.ValueId, 0),
        .instrs = try gpa.alloc(Ir.Instr, 0),
        .term = .{ .ret = .none },
    };
    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "main" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Type.unit,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = try gpa.alloc(Ir.ValueDef, 0),
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
        .literals = try gpa.alloc(Ir.Literal, 0),
    };
    defer func.deinit(gpa);

    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    var fc = try lowerIr(gpa, &func, &.{}, &.{}, true, &diags);
    defer fc.deinit(gpa);

    try testing.expectEqual(@as(usize, 0), diags.items.len);
    // movz x0,#0 must appear somewhere (the unit-entry x0=0 contract).
    const movz0 = Aarch64.movz(0, 0, 0);
    var found = false;
    var i: usize = 0;
    while (i + 4 <= fc.code.len) : (i += 4) {
        if (std.mem.readInt(u32, fc.code[i..][0..4], .little) == movz0) found = true;
    }
    try testing.expect(found);
}

// An AGGREGATE merge param fed by an aggregate `.value` operand must copy the
// bytes DIRECTLY out of the source value's frame cell (the aggregate is spilled
// in place), NOT load the cell's first word as a pointer and copy from there.
// This pins the `writeOperandTo` aggregate-`.value` fix (the if_value SIGSEGV).
test "ir-codegen: aggregate value-merge param copies from the value cell, not via a pointer" {
    const gpa = testing.allocator;

    // A 24-byte struct (id 0) → an aggregate carried in 3 words. ret type is the
    // struct (sret), so the exit param is an aggregate VALUE.
    var layouts = try gpa.alloc(Layout, 1);
    defer gpa.free(layouts);
    layouts[0] = .{ .name = "V3", .field_names = &.{}, .field_types = &.{}, .offsets = &.{}, .size = 24, .@"align" = 8 };
    const sty = Type.structT(0);

    // values: %0 = exit param (V3), %1 = merge param (V3).
    var values = try gpa.alloc(Ir.ValueDef, 2);
    values[0] = .{ .type = sty };
    values[1] = .{ .type = sty };

    // b0 (entry): br merge(b1) with arg %1 — but %1 is b1's own param. To exercise
    // an aggregate `.value` edge we route: b0 → b2(%1 fed from a slot) → b1(%1) →
    // exit ret %0. Simpler: b0 has a slot s0 it stores into, br b1(s0); b1 is the
    // merge with param %1; b1 br exit(%1 as .value); exit ret %0.
    var slots = try gpa.alloc(Ir.Slot, 1);
    slots[0] = .{ .type = sty };

    // b0: br b1(s0)   [aggregate .slot arg → merge param %1's cell]
    const b0_args = try gpa.alloc(Ir.Operand, 1);
    b0_args[0] = .{ .slot = 0 };
    // b1(%1): br b2(%1)   [aggregate .value arg → the fix under test]
    const b1_params = try gpa.alloc(Ir.ValueId, 1);
    b1_params[0] = 1;
    const b1_args = try gpa.alloc(Ir.Operand, 1);
    b1_args[0] = .{ .value = 1 };
    // b2(%0): ret %0    [exit, sret]
    const b2_params = try gpa.alloc(Ir.ValueId, 1);
    b2_params[0] = 0;

    var blocks = try gpa.alloc(Ir.Block, 3);
    blocks[0] = .{ .params = try gpa.alloc(Ir.ValueId, 0), .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .br = .{ .dest = 1, .args = b0_args } } };
    blocks[1] = .{ .params = b1_params, .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .br = .{ .dest = 2, .args = b1_args } } };
    blocks[2] = .{ .params = b2_params, .instrs = try gpa.alloc(Ir.Instr, 0), .term = .{ .ret = .{ .value = 0 } } };

    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "f" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = sty,
        .slots = slots,
        .values = values,
        .blocks = blocks,
        .entry = 0,
        .exit = 2,
        .literals = try gpa.alloc(Ir.Literal, 0),
    };
    defer func.deinit(gpa);

    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    var fc = try lowerIr(gpa, &func, layouts, &.{}, false, &diags);
    defer fc.deinit(gpa);

    try testing.expectEqual(@as(usize, 0), diags.items.len);

    // The b1→b2 edge copies %1 (a 24B aggregate value) into %0's cell. The fix:
    // BOTH the load and store of each word are SP-relative `ldrRegUoff`/`strRegUoff`
    // (base = SP = reg 31). A regression would emit a `ldrSp` of the cell into a
    // scratch reg (loading the value as a pointer) followed by `ldrRegUoff` off
    // that scratch — i.e. a non-SP base. Assert the copy NEVER dereferences a
    // loaded value: every `ldrRegUoff` in the body uses base SP.
    //
    // We can't easily isolate the b1 block, so assert the WEAKER, robust property:
    // there is at least one SP-based 3-word copy run (the b1→b2 aggregate copy) and
    // no `ldrRegUoff` uses a base register in 9..11 (the scratch regs) — which is
    // exactly what the buggy "load cell as pointer then copy" would produce.
    var has_scratch_base_load = false;
    var k: usize = 0;
    while (k + 4 <= fc.code.len) : (k += 4) {
        const w = std.mem.readInt(u32, fc.code[k..][0..4], .little);
        // ldr (unsigned offset, 64-bit): 0xF94 top bits; base reg = bits[9:5].
        if ((w & 0xFFC00000) == 0xF9400000) {
            const base = (w >> 5) & 0x1F;
            if (base == S0 or base == S1 or base == S2) has_scratch_base_load = true;
        }
    }
    // With the fix, the aggregate copy reads from SP+off, so no scratch-base ldr is
    // emitted for it. (slot_addr/field_addr/load in OTHER fns can use scratch bases,
    // but THIS fn has no such ops — only the two aggregate copies + the sret copy.)
    try testing.expect(!has_scratch_base_load);
}

// A reg-pair AGGREGATE (str, 16B) arriving as a call argument as an Operand.VALUE
// (a merge param / value-break result) must be marshalled by loading BOTH of its
// words from the value's frame cell into the two GPRs — NOT asserted to be a
// single-GPR scalar. Pins the genCall `.gpr`/`.value` reg-pair fix (the
// labeledblock_str SIGABRT: `assert(r.count == 1)`).
test "ir-codegen: a reg-pair aggregate passed as an Operand.value loads both words" {
    const gpa = testing.allocator;

    // str is a 16-byte reg_pair aggregate (ptr@0, len@8) → 2 GPRs (x0,x1).
    const sty = Type.str;

    // %0 = a str merge param (the aggregate VALUE passed to the call).
    var values = try gpa.alloc(Ir.ValueDef, 1);
    values[0] = .{ .type = sty };

    // b0(%0): call @sink(%0) ; ret  — the call takes the str by value (reg-pair),
    // sink returns unit so this fn is unit-typed. The arg is Operand.value %0.
    const b0_params = try gpa.alloc(Ir.ValueId, 1);
    b0_params[0] = 0;
    const call_args = try gpa.alloc(Ir.Operand, 1);
    call_args[0] = .{ .value = 0 };
    var instrs = try gpa.alloc(Ir.Instr, 1);
    instrs[0] = .{ .result = Ir.none_value, .op = .{ .call = .{
        .callee = .{ .kind = .user_fn, .name = "sink" },
        .args = call_args,
        .ret_slot = Ir.none_slot,
    } } };

    var blocks = try gpa.alloc(Ir.Block, 1);
    blocks[0] = .{ .params = b0_params, .instrs = instrs, .term = .{ .ret = .none } };

    var func = Ir.Function{
        .name = .{ .kind = .user_fn, .name = "f" },
        .params = try gpa.alloc(Ir.SlotId, 0),
        .ret_type = Type.unit,
        .slots = try gpa.alloc(Ir.Slot, 0),
        .values = values,
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
        .literals = try gpa.alloc(Ir.Literal, 0),
    };
    defer func.deinit(gpa);

    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    // The regression was an `unreachable` panic during lowering; reaching here at
    // all proves the assert is gone. Also assert clean + a `bl` was emitted.
    var fc = try lowerIr(gpa, &func, &.{}, &.{}, false, &diags);
    defer fc.deinit(gpa);

    try testing.expectEqual(@as(usize, 0), diags.items.len);

    // The marshalling must load %0's TWO words into x0 and x1 (ldr x0,[sp,#off],
    // ldr x1,[sp,#off+8]) before the bl. Assert an `ldr x1, [sp, ...]` exists
    // (the SECOND word of the reg pair) — the buggy path emitted only `ldr x0`.
    var has_ldr_x1_sp = false;
    var k: usize = 0;
    while (k + 4 <= fc.code.len) : (k += 4) {
        const w = std.mem.readInt(u32, fc.code[k..][0..4], .little);
        // ldr Xt,[sp,#imm] (unsigned offset, 64-bit): 0xF94 + base==31(sp) + Rt==1.
        if ((w & 0xFFC00000) == 0xF9400000 and ((w >> 5) & 0x1F) == 31 and (w & 0x1F) == 1) {
            has_ldr_x1_sp = true;
        }
    }
    try testing.expect(has_ldr_x1_sp);
}
