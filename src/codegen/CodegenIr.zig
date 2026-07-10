//! Ir.Function → Link.FnCode codegen (the IR path).
//!
//! The dual-path peer to the legacy AST→FnCode `Codegen.lower`. This consumes a
//! lowered `Ir.Function` (target-independent) and emits AArch64 bytes, deciding
//! the ABI ONLY here (via `Abi`) and the frame ONLY here (via `FrameLayout`,
//! spill-everything). It reuses the `Aarch64.zig` encoder and the same
//! label/fixup backpatch algorithm as the AST path, so branch encoding stays
//! identical.
//!
//! VALUE MODEL: every SSA value AND every slot has its OWN frame
//! cell (FrameLayout). An instruction loads its operands from their cells into
//! scratch regs (x9/x10/x11), computes, and stores the result into the result
//! value's cell. This is dead-simple and trivially correct; slot-coloring is a
//! deferred optimization.
//!
//! BLOCK ARGS = STORE-BEFORE-BR ("phi = memory"): a `br dest(args)`
//! edge stores each arg value into the destination block's param cell, THEN
//! branches. Multi-arg edges load ALL sources into scratch first, then store,
//! so a swap-style edge (param values feeding each other) is clobber-safe.
//! `cond_br` is ARGLESS by construction (lower split any value-merge cond edge),
//! so it is exactly cmp/cbnz + b.
//!
//! DETERMINISM: blocks are walked in BlockId order; one label per block, placed
//! and backpatched. Pure over the IR + read-only layout tables. Same IR + same
//! layouts ⇒ same bytes (underpins VERIFY byte-identity).

const std = @import("std");
const Ir = @import("../ir/Ir.zig");
const Abi = @import("abi/Abi.zig");
const FrameLayout = @import("frame/FrameLayout.zig");
const Aarch64 = @import("Aarch64.zig");
const Link = @import("../link/Link.zig");
const Typecheck = @import("../types.zig");

pub const Diagnostic = @import("../diagnostics/Diagnostic.zig").Diagnostic;

const Type = Typecheck.Type;
const Layout = Typecheck.Layout;
const EnumLayout = Typecheck.EnumLayout;

// Scratch GPRs the body uses for the load/compute/store pattern. Distinct from
// x0..x8 (params / result / sret-x8) so marshalling can't collide with them.
const S0: u32 = 9;
const S1: u32 = 10;
const S2: u32 = 11;

// Scratch D-registers (SIMD&FP file, caller-saved) for the float load/compute/store
// pattern. Distinct index space from the GPR scratch above; D-regs appear ONLY in
// genFloatArith/fcmp — float values otherwise ride GPR cells via storeValue.
const D0: u32 = 16;
const D1: u32 = 17;

// The float ARGUMENT/RESULT V-registers are the low physical D-registers v0..v7
// (raw indices 0..7), a separate index space from the D16/D17 scratch above.
// `ldrFpSp`/`strFpSp` encode the raw physical D-number, so arg regs pass 0..7
// and a float result rides v0 — never colliding with the arith scratch.
const V0: u32 = 0;

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
    /// Load a float SSA value's cell into D-register `d`.
    fn loadFpValue(g: *Gen, d: u32, vid: Ir.ValueId) error{OutOfMemory}!void {
        try g.emit(Aarch64.ldrFpSp(d, g.valueOff(vid)));
    }
    /// Store D-register `d` into a float SSA value's cell.
    fn storeFpValue(g: *Gen, d: u32, vid: Ir.ValueId) error{OutOfMemory}!void {
        if (vid == Ir.none_value) return;
        try g.emit(Aarch64.strFpSp(d, g.valueOff(vid)));
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
        for (g.relocs.items) |r| if (r.target.name()) |nm| gpa.free(nm);
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

    // Walk blocks in BlockId order. The exit block is emitted in sequence like
    // any other (its terminator is `ret`, which owns the epilogue).
    for (g.func.blocks, 0..) |*blk, bid| {
        g.placeLabel(g.block_labels[bid]);
        for (blk.instrs) |ins| try genInstr(g, ins);
        try genTerm(g, blk.term);
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
            .fpr => |v| {
                // An incoming bare float in v`v` (a D-register): store it into the
                // slot's 8-byte cell.
                try g.emit(Aarch64.strFpSp(v, off));
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
        .ult => .lo,
        .ule => .ls,
        .ugt => .hi,
        .uge => .hs,
    };
}

/// The NaN-safe aarch64 cond for a float compare against FCMP's NZCV (unordered →
/// C=1,V=1). `<`/`<=` use `mi`/`ls` (NOT `lt`/`le`, which are TRUE on unordered);
/// `>`/`>=`/`==` are already false on unordered, `!=` true — matching IEEE
/// (any comparison with NaN is false except `!=`).
fn fcondToAarch64(cc: Ir.FCond) Aarch64.Cond {
    return switch (cc) {
        .eq => .eq,
        .ne => .ne,
        .lt => .mi,
        .le => .ls,
        .gt => .gt,
        .ge => .ge,
    };
}

const FloatArithKind = enum { fadd, fsub, fmul, fdiv };

fn genFloatArith(g: *Gen, result: Ir.ValueId, b: Ir.Bin, comptime kind: FloatArithKind) error{OutOfMemory}!void {
    try g.loadFpValue(D0, b.lhs);
    try g.loadFpValue(D1, b.rhs);
    try g.emit(switch (kind) {
        .fadd => Aarch64.fadd(D0, D0, D1),
        .fsub => Aarch64.fsub(D0, D0, D1),
        .fmul => Aarch64.fmul(D0, D0, D1),
        .fdiv => Aarch64.fdiv(D0, D0, D1),
    });
    // No normalizeWidth: f64 is never narrowed.
    try g.storeFpValue(D0, result);
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
        .udiv => |b| try genArith(g, ins.result, b, .udiv),
        .smod => |b| try genRem(g, ins.result, b, .smod),
        .umod => |b| try genRem(g, ins.result, b, .umod),
        .neg => |v| {
            try g.loadValue(S0, v);
            try g.emit(Aarch64.neg(S0, S0));
            try storeNormalized(g, ins.result, S0);
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
        .load_byte => |v| {
            try g.loadValue(S0, v); // S0 = byte address
            try g.emit(Aarch64.ldrbRegUoff(S0, S0, 0)); // zero-extended single byte
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
        .band => |b| try genArith(g, ins.result, b, .band),
        .bor => |b| try genArith(g, ins.result, b, .bor),
        .bxor => |b| try genArith(g, ins.result, b, .bxor),
        .shl => |b| try genShift(g, ins.result, b, .shl),
        .lshr => |b| try genShift(g, ins.result, b, .lshr),
        .ashr => |b| try genShift(g, ins.result, b, .ashr),
        .bcompl => |v| {
            try g.loadValue(S0, v);
            try g.emit(Aarch64.mvn(S0, S0));
            try storeNormalized(g, ins.result, S0);
        },
        .fconst => |f| {
            try g.emitImm64(S0, @bitCast(f)); // f64 bit-pattern into a GPR
            try g.storeValue(S0, ins.result); // value cell holds the raw bits
        },
        .fadd => |b| try genFloatArith(g, ins.result, b, .fadd),
        .fsub => |b| try genFloatArith(g, ins.result, b, .fsub),
        .fmul => |b| try genFloatArith(g, ins.result, b, .fmul),
        .fdiv => |b| try genFloatArith(g, ins.result, b, .fdiv),
        .fcmp => |c| {
            try g.loadFpValue(D0, c.lhs);
            try g.loadFpValue(D1, c.rhs);
            try g.emit(Aarch64.fcmp(D0, D1));
            try g.emit(Aarch64.cset(S0, fcondToAarch64(c.cc))); // bool → GPR cell
            try g.storeValue(S0, ins.result);
        },
        .scvtf => |v| {
            try g.loadValue(S0, v); // source int in a GPR (width-canonicalized)
            try g.emit(Aarch64.scvtf(D0, S0));
            try g.storeFpValue(D0, ins.result);
        },
        .fcvtzs => |v| {
            try g.loadFpValue(D0, v); // source f64 in a D reg
            try g.emit(Aarch64.fcvtzs(S0, D0));
            try g.storeValue(S0, ins.result); // plat-width int result; no normalize needed
        },
    }
}

/// Re-canonicalize a narrow-integer result to its width then store it — the codegen
/// mirror of the opt side's `commitFoldedInt`. Every narrow-int-producing op must
/// normalize before storing (signed narrow sign-extends, unsigned narrow masks) so a
/// later full-width use reads a canonical value; routing all producers through one
/// helper keeps that invariant hard to forget.
fn storeNormalized(g: *Gen, result: Ir.ValueId, reg: u32) error{OutOfMemory}!void {
    try normalizeWidth(g, reg, g.func.values[result].type);
    try g.storeValue(reg, result);
}

const ArithKind = enum { add, sub, mul, sdiv, udiv, band, bor, bxor };

fn genArith(g: *Gen, result: Ir.ValueId, b: Ir.Bin, kind: ArithKind) error{OutOfMemory}!void {
    try g.loadValue(S0, b.lhs);
    try g.loadValue(S1, b.rhs);
    const word = switch (kind) {
        .add => Aarch64.addReg(S0, S0, S1),
        .sub => Aarch64.subReg(S0, S0, S1),
        .mul => Aarch64.mul(S0, S0, S1),
        .sdiv => Aarch64.sdiv(S0, S0, S1),
        .udiv => Aarch64.udiv(S0, S0, S1),
        .band => Aarch64.andReg(S0, S0, S1),
        .bor => Aarch64.orrReg(S0, S0, S1),
        .bxor => Aarch64.eorReg(S0, S0, S1),
    };
    try g.emit(word);
    try storeNormalized(g, result, S0);
}

const RemKind = enum { smod, umod };

/// Integer remainder r = a - (a/b)*b: divide into S2, then `msub`. Signed vs
/// unsigned rides on `sdiv`/`udiv` (mirroring genArith); `normalizeWidth`
/// re-canonicalizes a narrow result exactly as the divide path does. S2 is the
/// standard scratch (genShift uses it identically within one instruction).
fn genRem(g: *Gen, result: Ir.ValueId, b: Ir.Bin, kind: RemKind) error{OutOfMemory}!void {
    try g.loadValue(S0, b.lhs); // a
    try g.loadValue(S1, b.rhs); // b
    try g.emit(if (kind == .smod) Aarch64.sdiv(S2, S0, S1) else Aarch64.udiv(S2, S0, S1)); // q = a/b
    try g.emit(Aarch64.msub(S0, S2, S1, S0)); // r = a - q*b
    try storeNormalized(g, result, S0);
}

const ShiftKind = enum { shl, lshr, ashr };

/// Go-semantics shift via a runtime guard. aarch64 lslv/lsrv/asrv mask the amount
/// mod 64, so amt >= width (in particular amt >= 64, which the mask folds back
/// below width) would keep bits instead of zeroing / sign-filling. The
/// `cmp amt,#width; csel` (compared UNSIGNED, `hs`) forces the >=width result, so
/// -O0 here ≡ -O1 (arith.foldShift) for EVERY amount. `asr #63` broadcasts the
/// sign bit for the signed fill (the operand is stored sign-extended to 64 bits).
fn genShift(g: *Gen, result: Ir.ValueId, b: Ir.Bin, kind: ShiftKind) error{OutOfMemory}!void {
    const ty = g.func.values[result].type;
    const width: u12 = @intCast(ty.intBits()); // 8/16/32/64, all fit u12
    try g.loadValue(S0, b.lhs);
    try g.loadValue(S1, b.rhs);
    switch (kind) {
        .shl => {
            try g.emit(Aarch64.lslv(S0, S0, S1));
            try g.emit(Aarch64.cmpImm(S1, width));
            try g.emit(Aarch64.csel(S0, Aarch64.XZR, S0, .hs));
        },
        .lshr => {
            try g.emit(Aarch64.lsrv(S0, S0, S1));
            try g.emit(Aarch64.cmpImm(S1, width));
            try g.emit(Aarch64.csel(S0, Aarch64.XZR, S0, .hs));
        },
        .ashr => {
            try g.emit(Aarch64.asrv(S2, S0, S1)); // raw (masked) result → S2
            try g.emit(Aarch64.asrImm(S0, S0, 63)); // sign-fill → S0 (lhs already consumed into S2)
            try g.emit(Aarch64.cmpImm(S1, width));
            try g.emit(Aarch64.csel(S0, S0, S2, .hs)); // amt>=width ? sign-fill : raw
        },
    }
    try storeNormalized(g, result, S0);
}

/// Re-canonicalize a narrow-integer result in `reg` to its full 64-bit form after
/// a full-width op may have left high bits stale: sign-extend a signed narrow,
/// zero-extend (mask) an unsigned narrow. No-op for non-integers and platform/w64
/// ints, so existing programs emit byte-identically. Exhaustive over `IntWidth`
/// (no `else`) and value-domain-identical to `arith.wrapTo` (opt-on ≡ opt-off).
fn normalizeWidth(g: *Gen, reg: u32, ty: Ir.Type) error{OutOfMemory}!void {
    if (!ty.isInteger()) return;
    switch (ty.int_desc.width) {
        .plat, .w64 => {},
        .w8 => try g.emit(if (ty.int_desc.signed) Aarch64.sxtb(reg, reg) else Aarch64.andLowBits(reg, reg, 7)),
        .w16 => try g.emit(if (ty.int_desc.signed) Aarch64.sxth(reg, reg) else Aarch64.andLowBits(reg, reg, 15)),
        .w32 => try g.emit(if (ty.int_desc.signed) Aarch64.sxtw(reg, reg) else Aarch64.andLowBits(reg, reg, 31)),
    }
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
    for (c.args) |a| arg_types.appendAssumeCapacity(g.func.operandType(a));
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
            .fpr => |v| {
                // An outgoing bare float → load its frame cell into arg V-reg v`v`.
                // A separate register file from the GPR targets, so it can't clobber
                // any x0..x7 arg in this single left-to-right pass.
                const off: u32 = switch (arg) {
                    .value => |vv| g.valueOff(vv),
                    .slot => |s| g.slotOff(s),
                    .none => continue,
                };
                try g.emit(Aarch64.ldrFpSp(v, off));
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

    // sret: point x8 at the result slot AFTER args (so an `add x8,sp` is not
    // disturbed by arg marshalling reading sp).
    if (plan.sret_in_x8) {
        std.debug.assert(c.ret_slot != Ir.none_slot);
        try g.emit(Aarch64.addImm(8, Aarch64.SP, @intCast(g.slotOff(c.ret_slot))));
    }

    // bl placeholder + `.call26` reloc.
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

    // Place the result. A scalar lands in x0 → store into its value cell. A
    // reg-pair aggregate result lands in x0[,x1] → store into ret_slot's words.
    // An sret result was already written through x8 by the callee. Unit: none.
    switch (Abi.classifyRet(ret_ty, g.layouts, g.enum_layouts)) {
        .none => {},
        .fp_reg => try g.storeFpValue(V0, result), // float result in v0 → its value cell.
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

/// Same seed as lower.zig's `lit_seed`, so a panic message that ALSO appears as a
/// display literal dedups to one `__cstring` blob entry; distinct hashes are harmless.
const panic_lit_seed: u64 = 0x10c5_7e87;

/// Retarget a `.panic`/`.trap` terminator to `panic(msg)`: intern `msg` as a
/// `__cstring` on THIS FnCode, put its ptr in x0 (adrp+add, `.cstr` relocs) and its
/// byte-length in x1 (every message < 64 KiB -> one `movz`), then `bl panic` (a
/// `.call26` reloc to the builtin, appended once at link time). The block has no
/// successors; a `brk #0` backstop follows the never-returning call.
fn emitPanicCall(g: *Gen, msg: []const u8) error{OutOfMemory}!void {
    const h = std.hash.Wyhash.hash(panic_lit_seed, msg);
    try addLiteral(g, h, try g.gpa.dupe(u8, msg));
    var site: u32 = @intCast(g.code.items.len);
    try g.relocs.append(g.gpa, .{ .site = site, .target = .{ .cstr = h }, .kind = .adrp_page });
    try g.emit(Aarch64.adrp(0, 0)); // adrp x0, msg@page
    site = @intCast(g.code.items.len);
    try g.relocs.append(g.gpa, .{ .site = site, .target = .{ .cstr = h }, .kind = .add_lo12 });
    try g.emit(Aarch64.addImm(0, 0, 0)); // add x0, x0, #msg@lo12
    try g.emit(Aarch64.movz(1, @intCast(msg.len), 0)); // x1 = len
    const name = try g.gpa.dupe(u8, "panic");
    errdefer g.gpa.free(name);
    site = @intCast(g.code.items.len);
    try g.relocs.append(g.gpa, .{ .site = site, .target = .{ .func = .{ .kind = .builtin, .name = name } }, .kind = .call26, .addend = 0 });
    try g.emit(Aarch64.bl(0)); // bl panic
    try g.emit(Aarch64.brk0); // unreachable backstop (panic never returns)
}

fn genTerm(g: *Gen, term: Ir.Terminator) error{OutOfMemory}!void {
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
            // then unconditional b to false.
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
        .trap => try emitPanicCall(g, "unwrap of empty Option/Result"),
        .panic => |reason| try emitPanicCall(g, switch (reason) {
            .div_by_zero => "division by zero",
            .rem_by_zero => "remainder by zero",
        }),
    }
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
        // spilled in place, never a pointer to the bytes), so the
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
        .fp_reg => {
            // A bare float result: load the return operand's cell into v0.
            const off: u32 = switch (o) {
                .value => |v| g.valueOff(v),
                .slot => |s| g.slotOff(s),
                .none => return,
            };
            try g.emit(Aarch64.ldrFpSp(V0, off));
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

// print(str) builtin body — hand-written, AST/IR-independent. Appended to the
// program at link time (emit.zig) when any fn references `print`. Shuffles the
// (ptr,len) str pair into write(fd=1, buf, len) and tail-calls libc `write`.

// A tiny shared harness for the hand-emitted `write`-calling builtins
// (`lowerPrint`/`lowerDisplayInt`/`lowerPanic`). Each still writes its own body; these
// own only the boilerplate every one repeats verbatim — opening a frame, the `write`
// GOT-import preamble (byte-identical bar the destination register), the error-path
// reloc-name cleanup, and packaging the finished FnCode.

/// Append one little-endian AArch64 instruction word to a builtin's code buffer.
fn emitWord(c: *std.ArrayList(u8), a: std.mem.Allocator, word: u32) error{OutOfMemory}!void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, word, .little);
    try c.appendSlice(a, &buf);
}

/// Open a standard frame: `stp x29,x30,[sp,#-16]! ; mov x29,sp`. A valid frame chain
/// is required both to call `write` and for `panic` to walk it.
fn emitFramePrologue(code: *std.ArrayList(u8), gpa: std.mem.Allocator) error{OutOfMemory}!void {
    try emitWord(code, gpa, Aarch64.stpFpLrPre);
    try emitWord(code, gpa, Aarch64.movFpSp);
}

/// Emit the `write` GOT-import preamble — `adrp x16, write@GOT ; ldr xRd,[x16]` — and
/// append its two `.import` relocs (patched to the `__got` slot after layout). The fn
/// pointer lands in xRd; the caller issues `blr xRd`. Each name is freed on this fn's
/// own error path; on success `relocs` owns them (freed via `deinitBuiltinRelocs` or,
/// once packaged, `FnCode.deinit`).
fn emitWriteImport(code: *std.ArrayList(u8), relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator, rd: u32) error{OutOfMemory}!void {
    {
        const nm = try gpa.dupe(u8, "write");
        errdefer gpa.free(nm);
        try relocs.append(gpa, .{ .site = @intCast(code.items.len), .target = .{ .import = .{ .kind = .import, .name = nm } }, .kind = .adrp_page });
    }
    try emitWord(code, gpa, Aarch64.adrp(16, 0));
    {
        const nm = try gpa.dupe(u8, "write");
        errdefer gpa.free(nm);
        try relocs.append(gpa, .{ .site = @intCast(code.items.len), .target = .{ .import = .{ .kind = .import, .name = nm } }, .kind = .ldr_lo12 });
    }
    try emitWord(code, gpa, Aarch64.ldrRegUoff(rd, 16, 0));
}

/// Free a partially-built builtin's reloc target names + the reloc array (deinit of the
/// array alone does NOT free the interned `.func`/`.import` names). The builtins'
/// error-path errdefer; on success ownership passes to the FnCode.
fn deinitBuiltinRelocs(relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator) void {
    for (relocs.items) |r| if (r.target.name()) |nm| gpa.free(nm);
    relocs.deinit(gpa);
}

/// Package a hand-emitted builtin: name it `builtin` and take ownership of its code +
/// relocs. On failure the caller's `code`/`relocs` errdefers reclaim the buffers.
fn finishBuiltin(code: *std.ArrayList(u8), relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator, name_str: []const u8) error{OutOfMemory}!Link.FnCode {
    const name = try gpa.dupe(u8, name_str);
    errdefer gpa.free(name);
    return .{
        .sym = .{ .kind = .builtin, .name = name },
        .code = try code.toOwnedSlice(gpa),
        .relocs = try relocs.toOwnedSlice(gpa),
        .literals = &.{},
    };
}

/// Build the `print` builtin's FnCode directly (no IR, no frame). Caller owns
/// the result.
pub fn lowerPrint(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    const emit = emitWord;

    try emitFramePrologue(&code, gpa);
    // Shuffle (ptr,len) into write's (buf,len) = (x1,x2), then fd=1 in w0. Order
    // matters: move len (x1→x2) BEFORE overwriting x1 with ptr (x0→x1).
    try emit(&code, gpa, Aarch64.movReg(2, 1)); // len → x2
    try emit(&code, gpa, Aarch64.movReg(1, 0)); // ptr → x1
    try emit(&code, gpa, Aarch64.movz(0, 1, 0)); // fd = 1
    try emitWriteImport(&code, &relocs, gpa, 16); // x16 = &write
    try emit(&code, gpa, Aarch64.blr(16)); // blr x16
    try emit(&code, gpa, Aarch64.ldpFpLrPost); // ldp x29, x30, [sp], #16
    try emit(&code, gpa, Aarch64.ret); // ret

    return finishBuiltin(&code, &relocs, gpa, "print");
}

// __display_int(n) builtin body — hand-written, AST/IR-independent. The
// heap-free `int`->decimal renderer: format the digits of the i64 in x0 into a
// fixed 32-byte STACK buffer (backward, so no reversal), then tail into
// write(fd=1, buf, len). Appended at link time (emit.zig) when any fn references
// `__display_int`; like `print` it uses only the stack + the `write` syscall, so
// a generated program that displays an int touches NO allocator.
//
// Digit extraction avoids i64::MIN overflow by NEVER negating the running value:
// each step computes q = n/10 (sdiv, truncates toward zero) and the single-digit
// remainder r = n - q*10 (range -9..9), takes |r| (safe — |r| <= 9), and stores
// '0'+|r|. The original sign (saved in x9) prepends a '-'. n==0 emits one '0' via
// the do-while shape. All branch offsets are compile-time-known local words.

/// Build the `__display_int` builtin's FnCode directly (no IR, no frame beyond the
/// 32-byte digit buffer). Caller owns the result.
pub fn lowerDisplayInt(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    const emit = emitWord;

    const A = Aarch64;
    // Prologue + reserve a 32-byte digit buffer at [sp, sp+32); saved fp/lr sit above.
    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 32)); // sub sp, sp, #32
    try emit(&code, gpa, A.movReg(9, 0)); // x9 = n (saved for the sign test)
    try emit(&code, gpa, A.movz(11, 10, 0)); // x11 = 10 (divisor)
    try emit(&code, gpa, A.movz(14, 0x30, 0)); // x14 = '0'
    try emit(&code, gpa, A.addImm(10, A.SP, 24)); // x10 = &buf_end (write cursor, grows down)
    // do-while digit loop (LOOP = this sdiv):
    try emit(&code, gpa, A.sdiv(12, 0, 11)); // x12 = n / 10
    try emit(&code, gpa, A.mul(13, 12, 11)); // x13 = q * 10
    try emit(&code, gpa, A.subReg(13, 0, 13)); // x13 = n - q*10  (signed remainder)
    try emit(&code, gpa, A.cmpImm(13, 0)); // cmp r, #0
    try emit(&code, gpa, A.bCond(.ge, 2)); // b.ge +2  (skip the neg if r >= 0)
    try emit(&code, gpa, A.neg(13, 13)); // x13 = -r   (|r|, safe: |r| <= 9)
    try emit(&code, gpa, A.addReg(13, 13, 14)); // x13 = '0' + digit
    try emit(&code, gpa, A.subImm(10, 10, 1)); // --cursor
    try emit(&code, gpa, A.strb(13, 10, 0)); // *cursor = digit char
    try emit(&code, gpa, A.movReg(0, 12)); // n = q
    try emit(&code, gpa, A.cbnz(0, -10)); // cbnz n, LOOP  (back 10 words to the sdiv)
    // Sign: if the original value was negative, prepend '-'.
    try emit(&code, gpa, A.cmpImm(9, 0)); // cmp orig, #0
    try emit(&code, gpa, A.bCond(.ge, 4)); // b.ge +4  (skip the 3 sign instrs if >= 0)
    try emit(&code, gpa, A.movz(13, 0x2D, 0)); // x13 = '-'
    try emit(&code, gpa, A.subImm(10, 10, 1)); // --cursor
    try emit(&code, gpa, A.strb(13, 10, 0)); // *cursor = '-'
    // write(fd=1, buf=cursor, len=buf_end-cursor).
    try emit(&code, gpa, A.addImm(1, A.SP, 24)); // x1 = &buf_end
    try emit(&code, gpa, A.subReg(2, 1, 10)); // x2 = len = end - cursor
    try emit(&code, gpa, A.movReg(1, 10)); // x1 = buf = cursor
    try emit(&code, gpa, A.movz(0, 1, 0)); // x0 = fd = 1
    try emitWriteImport(&code, &relocs, gpa, 16); // x16 = &write
    try emit(&code, gpa, A.blr(16));
    // Epilogue.
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 32)); // sub-buffer teardown
    try emit(&code, gpa, A.ldpFpLrPost); // ldp x29,x30,[sp],#16
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "__display_int");
}

// panic(str) builtin body — hand-written, AST/IR-independent. Appended once at link
// time (emit.zig) when any fn references `panic`. Writes the message {ptr,len} to fd 2
// (STDERR), then a SYMBOLIZED-backtrace rung-A dump — one `0x<offset>` line per frame
// walked off the x29 FP chain — then exits NONZERO via SYS_exit(1) (raw `svc`, no libc):
// a deterministic, uncatchable abort. Never returns; the trailing brk #0 is an
// unreachable backstop.
//
// Each frame line is `ra - text_base - 4` in hex — the CALL SITE's slide-independent
// __text offset (post-processable by `atos`). `text_base` is self-located at runtime:
// `adr x9,.` yields this instruction's runtime address and the linker bakes that
// instruction's static __text offset into the following movz/movk (self-relative
// `.movw_g0`/`.movw_g1` relocs), so `text_base = adr_addr - baked_offset` holds under
// PIE/ASLR. Loop invariants (text_base, fp cursor, counter, buffer base, shift consts)
// live in x19-x24/x26 — callee-saved, so they survive each `write` call.
pub fn lowerPanic(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    const A = Aarch64;
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    const emit = emitWord;

    // Prologue + 64-byte scratch buffer for the hex line (`0x` + 16 nibbles + `\n`).
    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 64)); // scratch buffer
    try emit(&code, gpa, A.addImm(22, A.SP, 0)); // x22 = buffer base

    // write(2, msg_ptr, msg_len): incoming x0=ptr, x1=len. Move len (x1->x2) BEFORE
    // overwriting x1 with ptr (x0->x1). Load &write from its GOT slot ONCE into the
    // callee-saved x26, then `blr x26` for the message and every frame line.
    try emit(&code, gpa, A.movReg(2, 1)); // len -> x2
    try emit(&code, gpa, A.movReg(1, 0)); // ptr -> x1
    try emit(&code, gpa, A.movz(0, 2, 0)); // fd = 2 (STDERR)
    try emitWriteImport(&code, &relocs, gpa, 26); // x26 = &write
    try emit(&code, gpa, A.blr(26)); // write the message

    // One '\n' separator: messages carry no trailing newline, so this keeps the
    // first frame line off the message regardless of the message's contents.
    try emit(&code, gpa, A.movz(9, '\n', 0));
    try emit(&code, gpa, A.strb(9, 22, 0)); // buffer[0] = '\n'
    try emit(&code, gpa, A.movz(2, 1, 0)); // len 1
    try emit(&code, gpa, A.movReg(1, 22)); // buf
    try emit(&code, gpa, A.movz(0, 2, 0)); // fd 2
    try emit(&code, gpa, A.blr(26)); // write '\n'

    // Self-locate text_base into x19 (see the doc comment). The two movw relocs bake
    // the `adr`'s own absolute __text offset (addend = adr_site - mov_site); they need
    // no target (the baked value is site-relative), so they carry `.none`.
    const adr_pos: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.adr(9, 0)); // x9 = text_base + adr_off
    var site: u32 = @intCast(code.items.len);
    try relocs.append(gpa, .{ .site = site, .target = .none, .kind = .movw_g0, .addend = @as(i64, adr_pos) - @as(i64, site) });
    try emit(&code, gpa, A.movz(10, 0, 0)); // x10 = adr_off (lo)
    site = @intCast(code.items.len);
    try relocs.append(gpa, .{ .site = site, .target = .none, .kind = .movw_g1, .addend = @as(i64, adr_pos) - @as(i64, site) });
    try emit(&code, gpa, A.movk(10, 0, 1)); // x10 |= adr_off (hi)
    try emit(&code, gpa, A.subReg(19, 9, 10)); // x19 = text_base

    // Load the backtrace symbol table base into the callee-saved x27 (survives every
    // write() call). The relink tail reserved `symtab_base_hash` at the table's
    // __cstring offset, so this resolves like a string-literal pointer (adrp+add).
    site = @intCast(code.items.len);
    try relocs.append(gpa, .{ .site = site, .target = .{ .cstr = Link.symtab_base_hash }, .kind = .adrp_page });
    try emit(&code, gpa, A.adrp(27, 0)); // adrp x27, symtab@page
    site = @intCast(code.items.len);
    try relocs.append(gpa, .{ .site = site, .target = .{ .cstr = Link.symtab_base_hash }, .kind = .add_lo12 });
    try emit(&code, gpa, A.addImm(27, 27, 0)); // x27 = &symtab

    // Walk state: x20 = fp cursor (this frame), x21 = frame cap, x23/x24 = hex shifts.
    try emit(&code, gpa, A.movReg(20, A.FP)); // x20 = x29
    try emit(&code, gpa, A.movz(21, 64, 0)); // x21 = 64 (frame cap)
    try emit(&code, gpa, A.movz(23, 4, 0)); // x23 = 4
    try emit(&code, gpa, A.movz(24, 60, 0)); // x24 = 60

    const loop_top: u32 = @intCast(code.items.len);
    const cbz_fp: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(20, 0)); // fp == 0 -> DONE
    const cbz_cnt: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(21, 0)); // counter == 0 -> DONE
    try emit(&code, gpa, A.ldrRegUoff(25, 20, 8)); // x25 = ra = *(fp+8)
    try emit(&code, gpa, A.subReg(25, 25, 19)); // ra - text_base
    try emit(&code, gpa, A.subImm(25, 25, 4)); // -> call site (x25 = frame offset)

    // Symbol lookup: linear-scan the sorted table for the greatest entry off <= x25;
    // x28 = its name ptr (0 = none). Entries ascend by off, so the first off > target
    // ends the scan; x28 then holds the enclosing fn's name (or the `{text_size,""}`
    // sentinel for a frame past __text). x4-x10 are scratch (no write() runs here).
    try emit(&code, gpa, A.ldrRegUoff(4, 27, 0)); // x4 = count
    try emit(&code, gpa, A.addImm(5, 27, 8)); // x5 = &entry[0]
    try emit(&code, gpa, A.movz(6, 0, 0)); // x6 = i
    try emit(&code, gpa, A.movz(28, 0, 0)); // x28 = best name ptr (none)
    const scan_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cmpReg(6, 4)); // i vs count
    const scan_bhs: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.hs, 0)); // i >= count -> scan_done
    try emit(&code, gpa, A.ldrRegUoff(9, 5, 0)); // x9 = entry.off
    try emit(&code, gpa, A.cmpReg(9, 25)); // off vs target
    const scan_bhi: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.hi, 0)); // off > target -> scan_done
    try emit(&code, gpa, A.ldrRegUoff(10, 5, 8)); // x10 = entry.name_off
    try emit(&code, gpa, A.addReg(28, 27, 10)); // x28 = base + name_off
    try emit(&code, gpa, A.addImm(5, 5, 16)); // ++entry
    try emit(&code, gpa, A.addImm(6, 6, 1)); // ++i
    const scan_b: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // -> scan_top
    const scan_done: u32 = @intCast(code.items.len);

    // Prefix "0x" into the buffer.
    try emit(&code, gpa, A.movz(9, '0', 0));
    try emit(&code, gpa, A.strb(9, 22, 0));
    try emit(&code, gpa, A.movz(9, 'x', 0));
    try emit(&code, gpa, A.strb(9, 22, 1));
    try emit(&code, gpa, A.addImm(11, 22, 2)); // x11 = write cursor
    try emit(&code, gpa, A.movz(12, 16, 0)); // x12 = 16 nibbles
    const hex_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.lsrv(13, 25, 24)); // top nibble = val >> 60
    try emit(&code, gpa, A.lslv(25, 25, 23)); // val <<= 4
    try emit(&code, gpa, A.andLowBits(13, 13, 3)); // & 0xF
    try emit(&code, gpa, A.cmpImm(13, 10));
    try emit(&code, gpa, A.addImm(14, 13, '0')); // '0' + n
    try emit(&code, gpa, A.addImm(15, 13, 'a' - 10)); // 'a'-10 + n
    try emit(&code, gpa, A.csel(14, 14, 15, .lo)); // n < 10 ? digit : letter
    try emit(&code, gpa, A.strb(14, 11, 0));
    try emit(&code, gpa, A.addImm(11, 11, 1));
    try emit(&code, gpa, A.subImm(12, 12, 1));
    const cbnz_hex: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbnz(12, 0)); // more nibbles -> HEX

    // Append ' ' then FLUSH the prefix. The buffer only ever holds "0x" + 16 nibbles +
    // ' ' (19 bytes ≤ 64), so it cannot overflow whatever the (arbitrary-length) name is.
    try emit(&code, gpa, A.movz(9, ' ', 0));
    try emit(&code, gpa, A.strb(9, 11, 0));
    try emit(&code, gpa, A.addImm(11, 11, 1));
    try emit(&code, gpa, A.subReg(2, 11, 22)); // len = cursor - base
    try emit(&code, gpa, A.movReg(1, 22)); // buf
    try emit(&code, gpa, A.movz(0, 2, 0)); // fd 2
    try emit(&code, gpa, A.blr(26)); // write "0x<hex> "

    // The symbol name, written DIRECTLY from its read-only __cstring pointer (x28):
    // strlen then write — no copy into the fixed buffer, so no length bound on names.
    // A missing entry (x28 == 0) or the sentinel's empty name writes nothing.
    const name_guard: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(28, 0)); // no entry -> name_done
    try emit(&code, gpa, A.movReg(9, 28)); // x9 = scan ptr
    const strlen_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.ldrbRegUoff(10, 9, 0)); // w10 = *ptr
    const strlen_cbz: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(10, 0)); // NUL -> strlen_done
    try emit(&code, gpa, A.addImm(9, 9, 1));
    const strlen_b: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // -> strlen_top
    const strlen_done: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.subReg(2, 9, 28)); // len = ptr - name
    try emit(&code, gpa, A.movReg(1, 28)); // buf = name ptr
    try emit(&code, gpa, A.movz(0, 2, 0)); // fd 2
    try emit(&code, gpa, A.blr(26)); // write the name
    const name_done: u32 = @intCast(code.items.len);

    // Trailing newline (always, even for an unnamed frame).
    try emit(&code, gpa, A.movz(9, '\n', 0));
    try emit(&code, gpa, A.strb(9, 22, 0)); // buffer[0] = '\n'
    try emit(&code, gpa, A.movz(2, 1, 0)); // len 1
    try emit(&code, gpa, A.movReg(1, 22)); // buf
    try emit(&code, gpa, A.movz(0, 2, 0)); // fd 2
    try emit(&code, gpa, A.blr(26)); // write '\n'
    try emit(&code, gpa, A.ldrRegUoff(20, 20, 0)); // fp = *fp
    try emit(&code, gpa, A.subImm(21, 21, 1)); // counter--
    const b_loop: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // -> LOOP

    const done_pos: u32 = @intCast(code.items.len);
    // SYS_exit(1): x0 = status, x16 = SYS_exit, svc #0x80. Never returns.
    try emit(&code, gpa, A.movz(0, 1, 0));
    try emit(&code, gpa, A.movz(16, 1, 0));
    try emit(&code, gpa, A.svc0x80);
    try emit(&code, gpa, A.brk0); // unreachable backstop

    // Backpatch the intra-fn branches (signed word deltas).
    const buf = code.items;
    patchCbzTo(buf, cbz_fp, done_pos);
    patchCbzTo(buf, cbz_cnt, done_pos);
    patchCbzTo(buf, cbnz_hex, hex_top);
    patchBTo(buf, b_loop, loop_top);
    patchBCondTo(buf, scan_bhs, scan_done);
    patchBCondTo(buf, scan_bhi, scan_done);
    patchBTo(buf, scan_b, scan_top);
    patchCbzTo(buf, name_guard, name_done);
    patchCbzTo(buf, strlen_cbz, strlen_done);
    patchBTo(buf, strlen_b, strlen_top);

    return finishBuiltin(&code, &relocs, gpa, "panic");
}

/// Backpatch a `cbz`/`cbnz` placeholder at byte `site` to branch to byte `target`
/// (preserving opcode + rt), given the resolved word delta.
fn patchCbzTo(buf: []u8, site: u32, target: u32) void {
    const delta: i19 = @intCast(@divExact(@as(i64, target) - @as(i64, site), 4));
    const word = std.mem.readInt(u32, buf[site..][0..4], .little);
    std.mem.writeInt(u32, buf[site..][0..4], Aarch64.patchCbz(word, delta), .little);
}

/// Backpatch a `b.cond` placeholder at byte `site` to branch to byte `target`
/// (preserving opcode + cond).
fn patchBCondTo(buf: []u8, site: u32, target: u32) void {
    const delta: i19 = @intCast(@divExact(@as(i64, target) - @as(i64, site), 4));
    const word = std.mem.readInt(u32, buf[site..][0..4], .little);
    std.mem.writeInt(u32, buf[site..][0..4], Aarch64.patchBCond(word, delta), .little);
}

/// Backpatch an unconditional `b` placeholder at byte `site` to branch to `target`.
fn patchBTo(buf: []u8, site: u32, target: u32) void {
    const delta: i26 = @intCast(@divExact(@as(i64, target) - @as(i64, site), 4));
    std.mem.writeInt(u32, buf[site..][0..4], Aarch64.b(delta), .little);
}

// TESTS — hand-build a tiny Ir.Function and assert the emitted byte shape.

const testing = std.testing;

test "ir-codegen: __display_int builtin byte shape (prologue, buffer, digit loop, write, ret)" {
    const gpa = testing.allocator;
    var fc = try lowerDisplayInt(gpa);
    defer fc.deinit(gpa);

    // A hand-asm builtin named `__display_int`.
    try testing.expectEqualStrings("__display_int", fc.sym.name);
    try testing.expectEqual(Link.SymKind.builtin, fc.sym.kind);
    // 4-byte aligned instruction stream; opens with the frame prologue.
    try testing.expect(fc.code.len % 4 == 0);
    try testing.expectEqual(Aarch64.stpFpLrPre, std.mem.readInt(u32, fc.code[0..4], .little));
    // Reserves the 32-byte digit buffer right after `mov x29,sp`.
    try testing.expectEqual(Aarch64.subImm(Aarch64.SP, Aarch64.SP, 32), std.mem.readInt(u32, fc.code[8..12], .little));
    // Ends with `ret`; the two instructions before it are the frame teardown.
    const n = fc.code.len;
    try testing.expectEqual(Aarch64.ret, std.mem.readInt(u32, fc.code[n - 4 ..][0..4], .little));
    try testing.expectEqual(Aarch64.ldpFpLrPost, std.mem.readInt(u32, fc.code[n - 8 ..][0..4], .little));
    try testing.expectEqual(Aarch64.addImm(Aarch64.SP, Aarch64.SP, 32), std.mem.readInt(u32, fc.code[n - 12 ..][0..4], .little));
    // The body contains the backward loop back-edge (`cbnz x0, -10`) and at least one
    // byte store (`strb w13,[x10]`) — the digit-emitting core.
    var saw_cbnz = false;
    var saw_strb = false;
    var i: usize = 0;
    while (i < fc.code.len) : (i += 4) {
        const w = std.mem.readInt(u32, fc.code[i..][0..4], .little);
        if (w == Aarch64.cbnz(0, -10)) saw_cbnz = true;
        if (w == Aarch64.strb(13, 10, 0)) saw_strb = true;
    }
    try testing.expect(saw_cbnz);
    try testing.expect(saw_strb);
    // Two relocs to the `write` import (adrp_page + ldr_lo12), like `print`.
    try testing.expectEqual(@as(usize, 2), fc.relocs.len);
    for (fc.relocs) |r| {
        try testing.expect(r.target == .import);
        try testing.expectEqualStrings("write", r.target.import.name);
    }
}

test "lowerPanic: prologue, self-locate movw relocs, hex frame loop, then SYS_exit(1)" {
    const gpa = testing.allocator;
    var fc = try lowerPanic(gpa);
    defer fc.deinit(gpa);
    try testing.expectEqual(Link.SymKind.builtin, fc.sym.kind);
    try testing.expectEqualStrings("panic", fc.sym.name);
    try testing.expect(fc.code.len % 4 == 0);
    // Opens with the frame prologue.
    try testing.expectEqual(Aarch64.stpFpLrPre, std.mem.readInt(u32, fc.code[0..4], .little));
    // Ends with the never-returning SYS_exit(1) + brk backstop.
    const n = fc.code.len;
    try testing.expectEqual(Aarch64.brk0, std.mem.readInt(u32, fc.code[n - 4 ..][0..4], .little));
    try testing.expectEqual(Aarch64.svc0x80, std.mem.readInt(u32, fc.code[n - 8 ..][0..4], .little));
    // The self-locate `adr x9, .` and the hex-loop `csel x14,x14,x15,lo` are present.
    var saw_adr = false;
    var saw_csel = false;
    var i: usize = 0;
    while (i < fc.code.len) : (i += 4) {
        const word = std.mem.readInt(u32, fc.code[i..][0..4], .little);
        if (word == Aarch64.adr(9, 0)) saw_adr = true;
        if (word == Aarch64.csel(14, 14, 15, .lo)) saw_csel = true;
    }
    try testing.expect(saw_adr);
    try testing.expect(saw_csel);
    // Six relocs: two `write` imports (adrp_page + ldr_lo12) reached via the GOT, the
    // two self-relative movw relocs that bake the text-base offset, then the adrp+add
    // pair that loads the backtrace symbol table base (a `.cstr` to the reserved hash).
    try testing.expectEqual(@as(usize, 6), fc.relocs.len);
    try testing.expectEqual(Link.RelocKind.adrp_page, fc.relocs[0].kind);
    try testing.expectEqualStrings("write", fc.relocs[0].target.import.name);
    try testing.expectEqual(Link.RelocKind.ldr_lo12, fc.relocs[1].kind);
    try testing.expectEqualStrings("write", fc.relocs[1].target.import.name);
    try testing.expectEqual(Link.RelocKind.movw_g0, fc.relocs[2].kind);
    try testing.expectEqual(Link.RelocKind.movw_g1, fc.relocs[3].kind);
    // The movw addends name the `adr` site relative to each mov site (−4, −8).
    try testing.expectEqual(@as(i64, -4), fc.relocs[2].addend);
    try testing.expectEqual(@as(i64, -8), fc.relocs[3].addend);
    try testing.expectEqual(Link.RelocKind.adrp_page, fc.relocs[4].kind);
    try testing.expectEqual(Link.symtab_base_hash, fc.relocs[4].target.cstr);
    try testing.expectEqual(Link.RelocKind.add_lo12, fc.relocs[5].kind);
    try testing.expectEqual(Link.symtab_base_hash, fc.relocs[5].target.cstr);
}

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

// Lower `fn f() -> ty { ret (iconst 100 + iconst 100) : ty }` and return the code.
fn lowerAddOfWidth(gpa: std.mem.Allocator, ty: Type) !Link.FnCode {
    var values = try gpa.alloc(Ir.ValueDef, 4);
    values[0] = .{ .type = ty }; // exit param
    values[1] = .{ .type = ty };
    values[2] = .{ .type = ty };
    values[3] = .{ .type = ty }; // add result

    var entry_instrs = try gpa.alloc(Ir.Instr, 3);
    entry_instrs[0] = .{ .result = 1, .op = .{ .iconst = 100 } };
    entry_instrs[1] = .{ .result = 2, .op = .{ .iconst = 100 } };
    entry_instrs[2] = .{ .result = 3, .op = .{ .add = .{ .lhs = 1, .rhs = 2 } } };
    const entry_args = try gpa.alloc(Ir.Operand, 1);
    entry_args[0] = .{ .value = 3 };

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
        .ret_type = ty,
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
    return lowerIr(gpa, &func, &.{}, &.{}, false, &diags);
}

fn codeHasWord(fc: Link.FnCode, w: u32) bool {
    var i: usize = 0;
    while (i + 4 <= fc.code.len) : (i += 4) {
        if (std.mem.readInt(u32, fc.code[i..][0..4], .little) == w) return true;
    }
    return false;
}

test "ir-codegen: genArith width-normalizes narrow results (and platform int does not)" {
    const gpa = testing.allocator;

    // int8 add re-canonicalizes with sxtb.
    var fc8 = try lowerAddOfWidth(gpa, Type.int8);
    defer fc8.deinit(gpa);
    try testing.expect(codeHasWord(fc8, Aarch64.sxtb(S0, S0)));

    // uint8 add zero-extends with `and #0xff`.
    var fcu8 = try lowerAddOfWidth(gpa, Type.uint8);
    defer fcu8.deinit(gpa);
    try testing.expect(codeHasWord(fcu8, Aarch64.andLowBits(S0, S0, 7)));

    // Platform int add emits NEITHER — byte-identity guard for existing programs.
    var fci = try lowerAddOfWidth(gpa, Type.int);
    defer fci.deinit(gpa);
    try testing.expect(!codeHasWord(fci, Aarch64.sxtb(S0, S0)));
    try testing.expect(!codeHasWord(fci, Aarch64.andLowBits(S0, S0, 7)));
}

test "ir-codegen: condToAarch64 maps unsigned conds to the unsigned aarch64 forms" {
    // A transposition here (e.g. .ule => .lo) compiles green with no unsigned
    // compare program in the suite, so pin the four mappings directly.
    try testing.expectEqual(Aarch64.Cond.lo, condToAarch64(.ult));
    try testing.expectEqual(Aarch64.Cond.ls, condToAarch64(.ule));
    try testing.expectEqual(Aarch64.Cond.hi, condToAarch64(.ugt));
    try testing.expectEqual(Aarch64.Cond.hs, condToAarch64(.uge));
}

test "ir-codegen: fcondToAarch64 uses the NaN-safe forms for < and <=" {
    // `<`/`<=` must be `mi`/`ls`, NOT `lt`/`le` (which are TRUE on unordered).
    // A transposition compiles green with no NaN program pinning it, so pin directly.
    try testing.expectEqual(Aarch64.Cond.eq, fcondToAarch64(.eq));
    try testing.expectEqual(Aarch64.Cond.ne, fcondToAarch64(.ne));
    try testing.expectEqual(Aarch64.Cond.mi, fcondToAarch64(.lt));
    try testing.expectEqual(Aarch64.Cond.ls, fcondToAarch64(.le));
    try testing.expectEqual(Aarch64.Cond.gt, fcondToAarch64(.gt));
    try testing.expectEqual(Aarch64.Cond.ge, fcondToAarch64(.ge));
}

// The value transform an emitted normalizeWidth word performs, decoded from the
// ISA meaning of the exact encoders codegen uses — derived independently of
// `arith.wrapTo` so a divergence between the two width tables surfaces here.
const WidthXform = enum { identity, sxtb, sxth, sxtw, mask8, mask16, mask32 };

fn decodeWidthWord(words: []const u8) WidthXform {
    if (words.len == 0) return .identity;
    std.debug.assert(words.len == 4);
    const w = std.mem.readInt(u32, words[0..4], .little);
    if (w == Aarch64.sxtb(S0, S0)) return .sxtb;
    if (w == Aarch64.sxth(S0, S0)) return .sxth;
    if (w == Aarch64.sxtw(S0, S0)) return .sxtw;
    if (w == Aarch64.andLowBits(S0, S0, 7)) return .mask8;
    if (w == Aarch64.andLowBits(S0, S0, 15)) return .mask16;
    if (w == Aarch64.andLowBits(S0, S0, 31)) return .mask32;
    @panic("normalizeWidth emitted an unrecognized word");
}

fn applyWidthXform(x: WidthXform, v: i64) i64 {
    return switch (x) {
        .identity => v,
        .sxtb => @as(i64, @as(i8, @truncate(v))),
        .sxth => @as(i64, @as(i16, @truncate(v))),
        .sxtw => @as(i64, @as(i32, @truncate(v))),
        .mask8 => v & 0xFF,
        .mask16 => v & 0xFFFF,
        .mask32 => v & 0xFFFFFFFF,
    };
}

// Ties codegen's `normalizeWidth` (an instruction table) to `arith.wrapTo` (a
// value table) that CodegenIr.zig:434 claims mirror each other. The test-local
// import keeps the production graph free of a codegen→opt edge. For every
// int-width × sign (plus a non-integer identity arm), the value meaning of the
// instruction codegen emits must equal what wrapTo computes; any future
// divergence (a width added to one table, or a sign/zero-extend chosen
// differently) becomes a unit failure rather than a differential-only catch.
test "ir-codegen: normalizeWidth value-domain matches arith.wrapTo across all widths" {
    const arith = @import("../opt/arith.zig");

    var g = Gen{
        .gpa = testing.allocator,
        .func = undefined,
        .layouts = undefined,
        .enum_layouts = undefined,
        .is_entry = undefined,
        .fl = undefined,
        .params = undefined,
        .block_labels = undefined,
    };
    defer g.code.deinit(g.gpa);

    const cases = [_]Type{
        Type.int,   Type.uint,
        Type.int8,  Type.uint8,
        Type.int16, Type.uint16,
        Type.int32, Type.uint32,
        Type.int64, Type.uint64,
        Type.@"bool", // non-integer → identity arm
    };
    const probes = [_]i64{
        0x1122_3344_5566_7788, -1, 0x80, 0x8000, 0x8000_0000, 0,
    };

    for (cases) |ty| {
        g.code.clearRetainingCapacity();
        try normalizeWidth(&g, S0, ty);
        const xform = decodeWidthWord(g.code.items);
        for (probes) |p| {
            try testing.expectEqual(arith.wrapTo(ty, p), applyWidthXform(xform, p));
        }
    }
}
