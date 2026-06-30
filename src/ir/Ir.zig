//! Target-independent intermediate representation (M12).
//!
//! A flat, index-based SSA-ish IR sitting between the front-end (Ast + resolved
//! types) and codegen. This is a PEER DATA MODULE: it depends ONLY on
//! `types.Type` and `Link.SymName`, never on codegen, lower, or the AST. codegen
//! depends on `Ir` (Ir→FnCode), and `lower` produces it (Ast→Ir). It carries NO
//! physical registers, frame offsets, or calling convention — the ABI is decided
//! entirely in codegen by the `Abi` module (LOCK #3).
//!
//! VALUE MODEL (LOCK #2): scalars (int/bool temporaries) and control-flow
//! merge-values are SSA `Value`s — each defined exactly once. Named locals and
//! ALL aggregates (struct/enum/str) live in memory `Slot`s, addressed via
//! `slot_addr`/`field_addr` + `load`/`store`/`copy`. Merge values are carried as
//! block params (`phi = memory`): each predecessor stores the arg into the param
//! slot, then branches (store-before-br, lowered in codegen).
//!
//! DETERMINISM: every `BlockId`/`ValueId`/`SlotId` is handed out monotonically in
//! a strict source-order walk by `lower`. Blocks render in `BlockId` order. This
//! is what underpins the VERIFY byte-identity re-lower (Driver [C11]).

const std = @import("std");
const types = @import("../types.zig");
const Link = @import("../link/Link.zig");

pub const Type = types.Type;
pub const SymName = Link.SymName;

pub const ValueId = u32;
pub const SlotId = u32;
pub const BlockId = u32;

/// Sentinels. A value-less instruction (store/copy/unit/void-call) has
/// `result == none_value`; a scalar/unit call has `ret_slot == none_slot`.
pub const none_value: ValueId = std.math.maxInt(u32);
pub const none_slot: SlotId = std.math.maxInt(u32);
pub const none_block: BlockId = std.math.maxInt(u32);

/// Target-independent comparison condition (signed semantics). codegen maps this
/// to `Aarch64.Cond`; lower maps the source comparison token to this.
pub const Cond = enum(u8) { eq, ne, lt, le, gt, ge };

/// An operand of a call / terminator. Scalars travel by `value`; str/struct/enum
/// aggregates travel by `slot` (passed by reference, ABI decided in codegen);
/// `none` is the unit operand (materializes to nothing).
pub const Operand = union(enum) {
    value: ValueId,
    slot: SlotId,
    none,
};

/// A memory slot: one frame location. Carries only its type; the byte offset is
/// assigned by codegen's `FrameLayout` (spill-everything). Locals, params, the
/// for-loop induction var, match bindings, and every aggregate live in slots.
pub const Slot = struct {
    type: Type,
};

/// An SSA value definition: a scalar temporary or a merge (block-param) value.
/// Carries only its type; its frame slot is assigned by codegen (spill-all).
pub const ValueDef = struct {
    type: Type,
};

/// A binary op's two value operands.
pub const Bin = struct { lhs: ValueId, rhs: ValueId };

/// One decoded string literal a function references, keyed by its content hash
/// (the same hash a `cstr_ptr` op carries). `bytes` are the decoded runtime
/// bytes WITHOUT a trailing NUL. lower decodes these (it holds the source); a
/// `cstr_ptr` op carries only the hash, so codegen recovers the bytes from this
/// table to populate `FnCode.literals` for the relink tail (which interns them
/// program-wide). Owned by the `Function`.
pub const Literal = struct {
    hash: u64,
    bytes: []u8,
};

/// A call. `callee` is the stable symbol identity (`{kind,name}`). `args` are
/// typed operands (scalar→value, aggregate→slot). `ret_slot` is the destination
/// slot for an aggregate result, or `none_slot` for a scalar/unit result (whose
/// `Instr.result` then holds the scalar value, or `none_value` for unit). The
/// ABI (reg-pair vs indirect, x8 sret, NGRN/NSAA) is decided in codegen.
pub const Call = struct {
    callee: SymName,
    args: []Operand,
    ret_slot: SlotId,
};

/// An operation producing (at most) one value. Aggregate construction never
/// produces a value — it writes into a slot via store/copy/field_addr.
pub const Op = union(enum) {
    /// Scalar i64 constant → int value.
    iconst: i64,
    /// Boolean constant → bool value.
    bconst: bool,
    /// The unit value (materializes to nothing in codegen).
    unit,

    add: Bin,
    sub: Bin,
    mul: Bin,
    sdiv: Bin,

    neg: ValueId,
    bnot: ValueId,

    icmp: struct { cc: Cond, lhs: ValueId, rhs: ValueId },

    /// Address of a frame slot → ptr value.
    slot_addr: SlotId,
    /// base ptr + BYTE offset → ptr value. lower (which holds the layouts)
    /// resolves field-index → byte offset; codegen stays layout-free. `ty` is the
    /// field's type (load/store width for a subsequent access).
    field_addr: struct { base: ValueId, off: u32, ty: Type },
    /// Scalar load from a ptr value → value of `ty`.
    load: struct { addr: ValueId, ty: Type },
    /// Scalar store; no result.
    store: struct { addr: ValueId, val: ValueId, ty: Type },
    /// Aggregate byte copy (dst ptr ← src ptr, sizeof `ty`); no result. codegen
    /// lowers to a tight load/store loop with NO intervening branch.
    copy: struct { dst: ValueId, src: ValueId, ty: Type },

    /// 8-byte tag at offset 0 of an enum slot ptr → int value.
    get_tag: ValueId,
    /// A `.cstr` literal reference (content-hash, registered in `FnCode`) → ptr
    /// value (the ptr half of a str aggregate).
    cstr_ptr: u64,

    call: Call,
};

/// One instruction: an op plus its result value id (`none_value` when the op
/// produces no value: store/copy/unit/void-call).
pub const Instr = struct {
    result: ValueId,
    op: Op,
};

/// How a block ends.
pub const Terminator = union(enum) {
    /// Unconditional branch carrying merge args (store-before-br at the edge).
    br: struct { dest: BlockId, args: []Operand },
    /// Two-way branch on a bool value. ARGLESS by design: lower splits any
    /// value-merge cond edge into an arg-store pre-block, so codegen's `cond_br`
    /// is exactly the cmp+b.cond+b machinery, with no edge args to thread.
    cond_br: struct { cond: ValueId, t: BlockId, f: BlockId },
    /// Function return. Emitted only by the single EXIT block; `.none` for unit.
    ret: Operand,
    @"unreachable",
};

/// A basic block: a list of block params (the merge slots), straight-line
/// instructions, and exactly one terminator.
pub const Block = struct {
    params: []ValueId,
    instrs: []Instr,
    term: Terminator,
};

/// A lowered function. All index spaces (slots/values/blocks) are dense and
/// source-order. `entry` is the first executed block; `exit` is the single EXIT
/// block whose one param is the return value and whose terminator is `ret`.
pub const Function = struct {
    name: SymName,
    /// Param slots, in source order. ABI placement decided in codegen.
    params: []SlotId,
    ret_type: Type,
    slots: []Slot,
    values: []ValueDef,
    blocks: []Block,
    entry: BlockId,
    exit: BlockId,
    /// Decoded string literals this function references, content-hash keyed. A
    /// `cstr_ptr` op names a hash; codegen looks the bytes up here. Owned.
    literals: []Literal = &.{},

    /// Free every owned slice. `name.name` is borrowed from the names table
    /// (owned by the caller), so it is NOT freed here.
    pub fn deinit(self: *Function, gpa: std.mem.Allocator) void {
        for (self.blocks) |*b| {
            gpa.free(b.params);
            for (b.instrs) |*ins| {
                switch (ins.op) {
                    .call => |c| gpa.free(c.args),
                    else => {},
                }
            }
            gpa.free(b.instrs);
            switch (b.term) {
                .br => |br| gpa.free(br.args),
                else => {},
            }
        }
        gpa.free(self.blocks);
        gpa.free(self.params);
        gpa.free(self.slots);
        gpa.free(self.values);
        for (self.literals) |l| gpa.free(l.bytes);
        gpa.free(self.literals);
        self.* = undefined;
    }
};

/// Total straight-line instruction count across all blocks (terminators are
/// NOT counted — they are a fixed per-block cost). The opt-stage dual metric's
/// IR-side number; documented to exclude terminators so it is stable to compare.
pub fn instrCount(func: *const Function) usize {
    var n: usize = 0;
    for (func.blocks) |b| n += b.instrs.len;
    return n;
}

// ---------------------------------------------------------------------------
// Textual renderer (`--emit ir`) — a deterministic golden-test surface.
// ---------------------------------------------------------------------------

/// Render `fn` to `out` in the agreed grammar. Type names come from `layouts`/
/// `enum_layouts` for struct/enum kinds; scalars use their fixed spellings.
/// Deterministic: all ids are source-order, blocks emit in `BlockId` order.
pub fn render(
    out: *std.Io.Writer,
    func: *const Function,
    layouts: []const types.Layout,
    enum_layouts: []const types.EnumLayout,
) anyerror!void {
    try out.print("fn {s}(", .{func.name.name});
    for (func.params, 0..) |sid, i| {
        if (i != 0) try out.writeAll(", ");
        try out.print("s{d}", .{sid});
    }
    try out.writeAll(") -> ");
    try renderType(out, func.ret_type, layouts, enum_layouts);
    try out.writeAll(" {\n");

    // slots line
    try out.writeAll("  slots:");
    for (func.slots, 0..) |slot, i| {
        try out.print(" s{d}:", .{i});
        try renderType(out, slot.type, layouts, enum_layouts);
    }
    try out.writeAll("\n");

    for (func.blocks, 0..) |blk, bid| {
        try out.print("b{d}", .{bid});
        if (blk.params.len > 0) {
            try out.writeAll("(");
            for (blk.params, 0..) |vid, i| {
                if (i != 0) try out.writeAll(", ");
                try out.print("%{d}:", .{vid});
                try renderType(out, valueType(func, vid), layouts, enum_layouts);
            }
            try out.writeAll(")");
        }
        try out.writeAll(":\n");

        for (blk.instrs) |ins| try renderInstr(out, ins, layouts, enum_layouts);
        try renderTerm(out, blk.term);
    }

    try out.writeAll("}\n");
}

fn valueType(func: *const Function, vid: ValueId) Type {
    if (vid == none_value or vid >= func.values.len) return Type.invalid;
    return func.values[vid].type;
}

fn renderType(
    out: *std.Io.Writer,
    ty: Type,
    layouts: []const types.Layout,
    enum_layouts: []const types.EnumLayout,
) anyerror!void {
    switch (ty.kind) {
        .invalid => try out.writeAll("invalid"),
        .unit => try out.writeAll("unit"),
        .int => try out.writeAll("int"),
        .bool => try out.writeAll("bool"),
        .str => try out.writeAll("str"),
        .never => try out.writeAll("never"),
        .@"struct" => {
            if (ty.struct_id < layouts.len) {
                try out.writeAll(layouts[ty.struct_id].name);
            } else {
                try out.print("struct#{d}", .{ty.struct_id});
            }
        },
        .@"enum" => {
            if (ty.enum_id < enum_layouts.len) {
                try out.writeAll(enum_layouts[ty.enum_id].name);
            } else {
                try out.print("enum#{d}", .{ty.enum_id});
            }
        },
    }
}

fn renderOperand(out: *std.Io.Writer, op: Operand) anyerror!void {
    switch (op) {
        .value => |v| try out.print("%{d}", .{v}),
        .slot => |s| try out.print("s{d}", .{s}),
        .none => try out.writeAll("()"),
    }
}

fn condName(cc: Cond) []const u8 {
    return switch (cc) {
        .eq => "eq",
        .ne => "ne",
        .lt => "lt",
        .le => "le",
        .gt => "gt",
        .ge => "ge",
    };
}

fn renderInstr(
    out: *std.Io.Writer,
    ins: Instr,
    layouts: []const types.Layout,
    enum_layouts: []const types.EnumLayout,
) anyerror!void {
    // store/copy and void calls have no result; render without the `%N =` lead.
    const has_result = ins.result != none_value;
    try out.writeAll("  ");
    if (has_result) try out.print("%{d} = ", .{ins.result});

    switch (ins.op) {
        .iconst => |v| try out.print("iconst {d}\n", .{v}),
        .bconst => |v| try out.print("bconst {}\n", .{v}),
        .unit => try out.writeAll("unit\n"),
        .add => |b| try out.print("add %{d}, %{d}\n", .{ b.lhs, b.rhs }),
        .sub => |b| try out.print("sub %{d}, %{d}\n", .{ b.lhs, b.rhs }),
        .mul => |b| try out.print("mul %{d}, %{d}\n", .{ b.lhs, b.rhs }),
        .sdiv => |b| try out.print("sdiv %{d}, %{d}\n", .{ b.lhs, b.rhs }),
        .neg => |v| try out.print("neg %{d}\n", .{v}),
        .bnot => |v| try out.print("bnot %{d}\n", .{v}),
        .icmp => |c| try out.print("icmp {s} %{d}, %{d}\n", .{ condName(c.cc), c.lhs, c.rhs }),
        .slot_addr => |s| try out.print("slot_addr s{d}\n", .{s}),
        .field_addr => |f| {
            try out.print("field_addr %{d}, {d} : ", .{ f.base, f.off });
            try renderType(out, f.ty, layouts, enum_layouts);
            try out.writeAll("\n");
        },
        .load => |l| {
            try out.print("load %{d} : ", .{l.addr});
            try renderType(out, l.ty, layouts, enum_layouts);
            try out.writeAll("\n");
        },
        .store => |s| {
            try out.print("store %{d}, %{d} : ", .{ s.addr, s.val });
            try renderType(out, s.ty, layouts, enum_layouts);
            try out.writeAll("\n");
        },
        .copy => |c| {
            try out.print("copy %{d} <- %{d} : ", .{ c.dst, c.src });
            try renderType(out, c.ty, layouts, enum_layouts);
            try out.writeAll("\n");
        },
        .get_tag => |v| try out.print("get_tag %{d}\n", .{v}),
        .cstr_ptr => |h| try out.print("cstr_ptr #{x}\n", .{h}),
        .call => |c| {
            try out.print("call @{s}(", .{c.callee.name});
            for (c.args, 0..) |a, i| {
                if (i != 0) try out.writeAll(", ");
                try renderOperand(out, a);
            }
            if (c.ret_slot != none_slot) {
                try out.print(" -> s{d}", .{c.ret_slot});
            }
            try out.writeAll(")\n");
        },
    }
}

fn renderTerm(out: *std.Io.Writer, term: Terminator) anyerror!void {
    switch (term) {
        .br => |br| {
            try out.print("  br b{d}", .{br.dest});
            if (br.args.len > 0) {
                try out.writeAll("(");
                for (br.args, 0..) |a, i| {
                    if (i != 0) try out.writeAll(", ");
                    try renderOperand(out, a);
                }
                try out.writeAll(")");
            }
            try out.writeAll("\n");
        },
        .cond_br => |c| try out.print("  cond_br %{d}, b{d}, b{d}\n", .{ c.cond, c.t, c.f }),
        .ret => |o| {
            switch (o) {
                .none => try out.writeAll("  ret\n"),
                else => {
                    try out.writeAll("  ret ");
                    try renderOperand(out, o);
                    try out.writeAll("\n");
                },
            }
        },
        .@"unreachable" => try out.writeAll("  unreachable\n"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "render: minimal unit function with entry+exit" {
    const gpa = std.testing.allocator;

    // fn main() -> unit { b0: ret }  with a single (entry==exit) block.
    var blocks = try gpa.alloc(Block, 1);
    blocks[0] = .{ .params = &.{}, .instrs = &.{}, .term = .{ .ret = .none } };

    var func = Function{
        .name = .{ .kind = .user_fn, .name = "main" },
        .params = &.{},
        .ret_type = Type.unit,
        .slots = &.{},
        .values = &.{},
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
    };
    // deinit frees blocks; params/slots/values are empty literals — replace with
    // heap allocs so deinit's frees are valid.
    func.params = try gpa.alloc(SlotId, 0);
    func.slots = try gpa.alloc(Slot, 0);
    func.values = try gpa.alloc(ValueDef, 0);
    defer func.deinit(gpa);

    var buf: [256]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try render(&w, &func, &.{}, &.{});

    const got = w.buffered();
    const want =
        "fn main() -> unit {\n" ++
        "  slots:\n" ++
        "b0:\n" ++
        "  ret\n" ++
        "}\n";
    try std.testing.expectEqualStrings(want, got);
}

test "render: arithmetic + slots + call" {
    const gpa = std.testing.allocator;

    var values = try gpa.alloc(ValueDef, 3);
    values[0] = .{ .type = Type.int };
    values[1] = .{ .type = Type.int };
    values[2] = .{ .type = Type.int };

    var slots = try gpa.alloc(Slot, 1);
    slots[0] = .{ .type = Type.int };

    var instrs = try gpa.alloc(Instr, 3);
    instrs[0] = .{ .result = 0, .op = .{ .iconst = 7 } };
    instrs[1] = .{ .result = 1, .op = .{ .iconst = 3 } };
    instrs[2] = .{ .result = 2, .op = .{ .add = .{ .lhs = 0, .rhs = 1 } } };

    var blocks = try gpa.alloc(Block, 1);
    blocks[0] = .{
        .params = try gpa.alloc(ValueId, 0),
        .instrs = instrs,
        .term = .{ .ret = .{ .value = 2 } },
    };

    var func = Function{
        .name = .{ .kind = .user_fn, .name = "f" },
        .params = try gpa.alloc(SlotId, 0),
        .ret_type = Type.int,
        .slots = slots,
        .values = values,
        .blocks = blocks,
        .entry = 0,
        .exit = 0,
    };
    defer func.deinit(gpa);

    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try render(&w, &func, &.{}, &.{});

    const want =
        "fn f() -> int {\n" ++
        "  slots: s0:int\n" ++
        "b0:\n" ++
        "  %0 = iconst 7\n" ++
        "  %1 = iconst 3\n" ++
        "  %2 = add %0, %1\n" ++
        "  ret %2\n" ++
        "}\n";
    try std.testing.expectEqualStrings(want, w.buffered());
}

test "sentinels are distinct from valid ids" {
    try std.testing.expect(none_value == std.math.maxInt(u32));
    try std.testing.expect(none_slot == std.math.maxInt(u32));
}
