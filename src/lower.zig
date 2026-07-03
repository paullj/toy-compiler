//! `lower`: the Ast→Ir transformation stage, a TOP-LEVEL pipeline peer to
//! `parse` (Ast) and `types` (the resolved type side-tables). It is PURE of its
//! frozen inputs — no globals, no map-iteration-order — so the IR is a
//! deterministic function of (tree, tokens, source, resolutions, node_types,
//! layouts, enum_layouts, names, fn_decl). That determinism is what lets the
//! codegen cache stay keyed on the AST fingerprint (the IR never escapes the
//! per-fn query) and lets VERIFY re-lower byte-identically.
//!
//! STAGE "lower-core": this implements the SCALAR + CONTROL-FLOW surface:
//! int/bool/unit literals, identifiers, arithmetic/comparison/`!`, `&&`/`||`
//! (short-circuit via blocks), var_decl + assign (locals are memory slots,
//! load/store), scalar calls, if/else (merge via a block param), while/for (the
//! induction var is a slot)/loop (break value via a block param), labels +
//! break/continue/return resolving to the right block. `str` is also lowered
//! (literal/identifier/var_decl/assign/arg/result + the `print` builtin) since it
//! is a 16-byte aggregate carried by slot — cheap and needed for I/O. Aggregate
//! struct/enum construction, field access, and `match` are LATER stages: they
//! surface a clean diagnostic here so the IR stays well-formed.
//!
//! VALUE MODEL (LOCK #2): every scalar temporary is an SSA `Value` (defined once);
//! merge values (if/loop/labeled-block/the fn return value) are carried as block
//! params. A block param is itself a memory slot, written store-before-br at each
//! predecessor (`phi = memory`) — but we emit that as `Terminator.br` carrying
//! `args`, and codegen does the store-before-br. cond_br is ARGLESS: any value
//! merge on a two-way edge is split into an arg-store pre-block whose only job is
//! to `br dest(value)`.

const std = @import("std");
const Ast = @import("ast/Ast.zig");
const Token = @import("ast/Token.zig").Token;
const TokenTag = @import("ast/Token.zig").Tag;
const Resolve = @import("resolve.zig");
const Typecheck = @import("types.zig");
const Link = @import("link/Link.zig");
const Ir = @import("ir/Ir.zig");
const Sig = @import("symbols/Sig.zig").Sig;
const Mono = @import("symbols/Mono.zig");
const Infer = @import("symbols/Infer.zig");
const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;

/// The read-only front-end inputs a lowering needs. Mirrors the slice of
/// `Driver.Frozen` that codegen consumes today; bundling them keeps the call
/// site (and the eventual dual-path fork) tidy.
pub const Inputs = struct {
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const Resolve.Resolution,
    node_types: []const Typecheck.Type,
    layouts: []const Typecheck.Layout,
    enum_layouts: []const Typecheck.EnumLayout,
    names: []const Link.SymName,
    /// This fn's typecheck-resolved signature (param/return `Type`s carrying the
    /// ABI-correct GLOBAL struct/enum ids), or `null` in single-file table mode.
    ///
    /// In graph mode two modules may each declare a same-named `struct Point` with
    /// DISTINCT global layout ids/sizes. `typeFromRef`'s bare-name first-match scan
    /// cannot tell them apart, so a fn naming its OWN `Point` could be lowered with
    /// the wrong layout — taking the <=16B reg-pair vs >16B sret ABI decision on the
    /// wrong size and miscompiling/hanging. When present, this sig is the ABI-
    /// deciding source for struct/enum param/return types (it already carries the
    /// owning module's id, including a qualified cross-module `b: rect.Rect` resolved
    /// by `Typecheck.typeFromQualified`); `typeFromRef` is only the fallback.
    sig: ?Sig = null,
    /// The monomorphization instance table (M2). A generic call `id[int](..)` in
    /// this body resolves its callee to the reified instance's mangled SymName by
    /// matching `(template gid + the concrete type-args read from node_types)`
    /// against this table. Empty for a program with no generics. Resolution reads
    /// only concrete `node_types` — no substitution map is threaded here (the mono
    /// tail already substituted every `type_var` away before `node_types` froze).
    instances: []const Mono.Instance = &.{},
    /// Program-wide callee signatures by global fn id (M3). A bare inferred generic
    /// call `id(7)` resolves its callee to a generic template's `.func`; that is
    /// detectable because the template's `sigs[func].params` carry `type_var`s. When
    /// so, `lowerCall` re-runs the shared `Infer` matcher over the value-arg
    /// `node_types` to select the SAME `Mono.Instance` Pass C created, instead of
    /// emitting the template symbol. Empty for a program with no generics.
    sigs: []const Sig = &.{},
    /// The program-wide inherent-method table (M8). A call whose callee is a
    /// `field_access` over a VALUE receiver dispatches through this: the receiver's
    /// concrete type + the member name select the method's global fn id (the mangled
    /// SymName in `names`), and the receiver is prepended as the `self` arg. Empty for
    /// a program with no methods; content-keyed (`findMethod`), so `-jN` deterministic.
    ///
    /// NO default: a build site that forgets to thread this must be a COMPILE error,
    /// not a silent "no methods" miss (a method call would then hit "call target
    /// unsupported" / lose its mut-self ABI). Every `lower.Inputs` build site MUST set
    /// it — pass `&.{}` only when the program provably has no methods.
    methods: []const Typecheck.Method,
};

/// The mutable builder state for ONE function lowering. All index spaces
/// (slots/values/blocks) are handed out monotonically as the source-order walk
/// proceeds, which is what makes the result deterministic.
const Builder = struct {
    gpa: std.mem.Allocator,
    in: Inputs,

    slots: std.ArrayList(Ir.Slot) = .empty,
    values: std.ArrayList(Ir.ValueDef) = .empty,
    blocks: std.ArrayList(BlockBuild) = .empty,
    /// Decoded string literals referenced by this fn (content-hash keyed, deduped).
    /// A `cstr_ptr` op carries only the hash; codegen recovers the bytes here.
    literals: std.ArrayList(Ir.Literal) = .empty,

    /// The block currently being appended to.
    cur: Ir.BlockId = 0,
    /// The single EXIT block: its one param is the return value, terminator `ret`.
    exit: Ir.BlockId = 0,
    /// The function return type (the exit param's type), so a bare `return` /
    /// fall-off-end can carry the right unit/scalar operand.
    ret_type: Typecheck.Type = .{ .kind = .invalid },
    /// The exit block's param value (or `none_value` for a unit function).
    ret_param: Ir.ValueId = Ir.none_value,

    /// The slot holding the `mut self` receiver's ADDRESS (M9), or `none_slot` for
    /// any non-mut fn. A mut-self param slot is typed `int` (an 8B pointer) rather
    /// than the struct, so its scalar/1-GPR ABI carries the caller's slot address;
    /// `rootAddr` loads through it so `self`/`self.f` reach the caller's storage.
    /// `none_slot` never equals a live slot id, so every non-mut fn is byte-identical.
    mut_self_slot: Ir.SlotId = Ir.none_slot,

    /// resolutions[].local index → the IR slot bound to that local. Allocated
    /// lazily the first time a local is touched (so id order tracks source order).
    local_slots: std.AutoHashMapUnmanaged(u32, Ir.SlotId) = .empty,

    /// The enclosing loop/labeled-block contexts (innermost last).
    loops: std.ArrayList(LoopCtx) = .empty,

    diags: *std.ArrayList(Diagnostic),
    had_error: bool = false,

    /// A block under construction: a growable instruction list + (once set) a
    /// terminator. `term_set` guards against emitting two terminators into one
    /// block (a divergent construct sets its own terminator; the enclosing code
    /// must not also branch).
    const BlockBuild = struct {
        params: std.ArrayList(Ir.ValueId) = .empty,
        instrs: std.ArrayList(Ir.Instr) = .empty,
        term: Ir.Terminator = .@"unreachable",
        term_set: bool = false,
    };

    fn deinit(b: *Builder) void {
        for (b.blocks.items) |*bb| {
            bb.params.deinit(b.gpa);
            for (bb.instrs.items) |*ins| switch (ins.op) {
                .call => |c| b.gpa.free(c.args),
                else => {},
            };
            bb.instrs.deinit(b.gpa);
            switch (bb.term) {
                .br => |br| b.gpa.free(br.args),
                else => {},
            }
        }
        b.blocks.deinit(b.gpa);
        b.slots.deinit(b.gpa);
        b.values.deinit(b.gpa);
        for (b.literals.items) |l| b.gpa.free(l.bytes);
        b.literals.deinit(b.gpa);
        b.local_slots.deinit(b.gpa);
        b.loops.deinit(b.gpa);
    }

    /// Register a decoded string literal (content-hash keyed; deduped within the
    /// fn). TAKES OWNERSHIP of `bytes` (frees a within-fn dup). Mirrors codegen's
    /// `addLiteral` so the FnCode literal table matches.
    fn addLiteral(b: *Builder, hash: u64, bytes: []u8) error{OutOfMemory}!void {
        for (b.literals.items) |lit| {
            if (lit.hash == hash and std.mem.eql(u8, lit.bytes, bytes)) {
                b.gpa.free(bytes);
                return;
            }
        }
        b.literals.append(b.gpa, .{ .hash = hash, .bytes = bytes }) catch |e| {
            b.gpa.free(bytes);
            return e;
        };
    }

    fn addSlot(b: *Builder, ty: Typecheck.Type) error{OutOfMemory}!Ir.SlotId {
        const id: Ir.SlotId = @intCast(b.slots.items.len);
        try b.slots.append(b.gpa, .{ .type = ty });
        return id;
    }

    fn addValue(b: *Builder, ty: Typecheck.Type) error{OutOfMemory}!Ir.ValueId {
        const id: Ir.ValueId = @intCast(b.values.items.len);
        try b.values.append(b.gpa, .{ .type = ty });
        return id;
    }

    fn addBlock(b: *Builder) error{OutOfMemory}!Ir.BlockId {
        const id: Ir.BlockId = @intCast(b.blocks.items.len);
        try b.blocks.append(b.gpa, .{});
        return id;
    }

    /// Append a new block param (a merge slot) to `bid`, returning its value id.
    fn addParam(b: *Builder, bid: Ir.BlockId, ty: Typecheck.Type) error{OutOfMemory}!Ir.ValueId {
        const v = try b.addValue(ty);
        try b.blocks.items[bid].params.append(b.gpa, v);
        return v;
    }

    /// Emit an op into the CURRENT block, producing a value of `ty` (or pass
    /// `null` for a result-less op like store/copy/void-call → `none_value`).
    fn emit(b: *Builder, op: Ir.Op, ty: ?Typecheck.Type) error{OutOfMemory}!Ir.ValueId {
        const result: Ir.ValueId = if (ty) |t| try b.addValue(t) else Ir.none_value;
        try b.blocks.items[b.cur].instrs.append(b.gpa, .{ .result = result, .op = op });
        return result;
    }

    /// Set the current block's terminator (idempotent guard: the FIRST terminator
    /// wins, so a divergent sub-construct that already terminated is not clobbered
    /// by an enclosing fall-through branch).
    fn setTerm(b: *Builder, term: Ir.Terminator) void {
        const bb = &b.blocks.items[b.cur];
        if (bb.term_set) {
            // A second terminator would be malformed; drop it (the block already
            // diverged). Free any args we were handed so nothing leaks.
            switch (term) {
                .br => |br| b.gpa.free(br.args),
                else => {},
            }
            return;
        }
        bb.term = term;
        bb.term_set = true;
    }

    fn termSet(b: *Builder) bool {
        return b.blocks.items[b.cur].term_set;
    }

    /// Switch the cursor to `bid` (the block subsequent code appends to).
    fn switchTo(b: *Builder, bid: Ir.BlockId) void {
        b.cur = bid;
    }

    fn note(b: *Builder, tok: u32, msg: []const u8) error{OutOfMemory}!void {
        b.had_error = true;
        try b.diags.append(b.gpa, .{ .byte_offset = b.in.tokens[tok].start, .message = msg });
    }
};

/// How a value-`break` to this context delivers its value — the two merge shapes
/// are mutually exclusive, so a tagged union makes the illegal combinations
/// unrepresentable (and `lowerBreak` switches it exhaustively).
const Merge = union(enum) {
    /// Non-value loop/block: a `break` carries no value.
    none,
    /// SCALAR/str value: a value-break's value rides POSITIONALLY as the join
    /// block's br arg, so the join param id need not be stored here.
    scalar,
    /// AGGREGATE value: a value-break produces INTO `ptr` (of `ty`), then branches
    /// argless to `break_bb` (the exit reads the destination, not a block-arg merge).
    aggregate: struct { ptr: Ir.ValueId, ty: Typecheck.Type },
};

/// The lowering context for one enclosing loop / labeled bare block. `break`
/// branches to `break_bb` (carrying a value per `merge`); `continue` branches
/// to `continue_bb` (the loop header for while/loop; the inc block for `for`;
/// `none_block` for a labeled bare block).
const LoopCtx = struct {
    kind: enum { loop, while_for, labeled_block },
    label: ?[]const u8,
    construct_node: Ast.Index,
    break_bb: Ir.BlockId,
    continue_bb: Ir.BlockId,
    merge: Merge,
};

/// Lower one function declaration to an `Ir.Function`. Pure of `in` (frozen
/// inputs) — handing out slot/value/block ids in a strict source-order walk so
/// the result is deterministic. Unsupported (not-yet-lowered) constructs append a
/// `Diagnostic` to `out_diags`; the IR returned is still well-formed and frees
/// normally. Caller owns the returned `Function` and must `deinit` it.
pub fn lowerFn(
    gpa: std.mem.Allocator,
    in: Inputs,
    fn_decl: Ast.Index,
    sym: Link.SymName,
    is_entry: bool,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    _ = is_entry; // the is_entry unit-main x0=0 contract is a codegen concern.

    const decl = in.tree.nodes[(fn_decl).int()];
    const proto = Ast.protoAt(in.tree, (decl.lhs).int());
    const ret_type = returnType(in, proto);

    var b: Builder = .{ .gpa = gpa, .in = in, .ret_type = ret_type, .diags = out_diags };
    errdefer b.deinit();

    // Params become slots in source order. Bind each param's resolution-local
    // index to its slot so an identifier reading the param finds the same slot.
    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    for (proto.params, 0..) |_, i| {
        // A `mut self` receiver (M9): type param-0's slot as `int` (an 8B pointer to
        // the caller's live slot), NOT the struct. This bypasses `paramType` — which
        // would return the struct type from the Sig — so the existing scalar-param ABI
        // carries the address in one GPR, with no Abi/Codegen change. `rootAddr` then
        // loads through this slot for every `self`/`self.field` use.
        const is_mut_self = i == 0 and Ast.isMutParam(in.tree, in.tokens, proto.params[0]);
        const pty = if (is_mut_self) Typecheck.Type.int else paramType(in, proto, @intCast(i));
        const sid = try b.addSlot(pty);
        if (is_mut_self) b.mut_self_slot = sid;
        try params.append(gpa, sid);
        // resolve assigns local slot indices in declaration order, params FIRST
        // (it declares each param but does not write `.local` onto the param node).
        // So a body identifier referencing param `i` resolves to `.local(i)`. Bind
        // local index `i` → this param's IR slot.
        try b.local_slots.put(gpa, @intCast(i), sid);
    }

    // entry block (b0) and the single EXIT block. The exit's one param is the
    // return value (none for a unit function). Allocate entry first so it is b0.
    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    if (ret_type.kind != .unit and ret_type.kind != .never) {
        b.ret_param = try b.addParam(exit, ret_type);
    }
    // The exit block returns its param (or unit).
    b.blocks.items[exit].term = if (b.ret_param == Ir.none_value)
        .{ .ret = .none }
    else
        .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;

    // Lower the body. A non-unit fn's body is a VALUE block (implicit return):
    // its trailing expression is the return value, branched to the exit. A unit fn
    // lowers for effect and falls through with no value. If the body diverges
    // (every path returns), the trailing fall-through edge is simply not emitted.
    const body_is_value = (b.ret_param != Ir.none_value);
    if (body_is_value) {
        const v = try lowerBlockValue(&b, decl.rhs, ret_type);
        if (!b.termSet()) try brTo(&b, exit, v);
    } else {
        try lowerBlockStmts(&b, decl.rhs);
        if (!b.termSet()) try brTo(&b, exit, .none);
    }

    // Materialize the builder into a flat Function. Do every fallible allocation
    // BEFORE tearing the builder down, so the `errdefer b.deinit()` stays a valid
    // cleanup for the whole builder until we have fully transferred ownership.
    const params_owned = try params.toOwnedSlice(gpa);
    errdefer gpa.free(params_owned);

    const blocks = try gpa.alloc(Ir.Block, b.blocks.items.len);
    errdefer gpa.free(blocks);
    // Convert each builder block. On a mid-loop OOM, `errdefer b.deinit()` frees
    // the still-owned builder blocks; the already-converted ones were emptied by
    // `toOwnedSlice` (deinit sees empty lists), and `errdefer gpa.free(blocks)`
    // frees the outer array (the moved-in inner slices leak only on OOM — benign).
    for (b.blocks.items, 0..) |*bb, i| {
        blocks[i] = .{
            .params = try bb.params.toOwnedSlice(gpa),
            .instrs = try bb.instrs.toOwnedSlice(gpa),
            .term = bb.term,
        };
        // The terminator's args slice (if any) is now owned by `blocks[i]`; clear
        // the builder copy so deinit doesn't double-free it.
        bb.term = .@"unreachable";
        bb.term_set = false;
    }

    const slots = try b.slots.toOwnedSlice(gpa);
    const values = try b.values.toOwnedSlice(gpa);
    const literals = try b.literals.toOwnedSlice(gpa);

    // All fallible work is done. Disarm `errdefer b.deinit()` by freeing exactly
    // the builder-owned remnants (the emptied block list, the local map, the loop
    // stack) here; slots/values/blocks/params/literals have moved into the result.
    b.blocks.deinit(gpa);
    b.local_slots.deinit(gpa);
    b.loops.deinit(gpa);
    b.blocks = .empty;
    b.local_slots = .empty;
    b.loops = .empty;
    b.slots = .empty;
    b.values = .empty;
    b.literals = .empty;

    return .{
        .name = sym,
        .params = params_owned,
        .ret_type = ret_type,
        .slots = slots,
        .values = values,
        .blocks = blocks,
        .entry = entry,
        .exit = exit,
        .literals = literals,
    };
}

/// Lower a block's statements in order into the current block (no merge value).
fn lowerBlockStmts(b: *Builder, block_idx: Ast.Index) error{OutOfMemory}!void {
    const block = b.in.tree.nodes[(block_idx).int()];
    for (Ast.rangeSlice(b.in.tree, (block.lhs).int())) |stmt_idx| {
        if (b.termSet()) break; // unreachable tail (after a return/divergent stmt)
        try lowerStmt(b, stmt_idx);
    }
}

fn lowerStmt(b: *Builder, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const stmt = b.in.tree.nodes[(stmt_idx).int()];
    switch (stmt.tag) {
        .var_decl => {
            const slot = try localSlot(b, stmt_idx, b.in.node_types[(stmt_idx).int()]);
            try storeInto(b, slot, stmt.lhs, b.in.node_types[(stmt_idx).int()]);
        },
        .assign => {
            const target = b.in.tree.nodes[(stmt.lhs).int()];
            const place_ty = b.in.node_types[(stmt.lhs).int()];
            if (target.tag == .field_access) {
                try lowerFieldStore(b, stmt.lhs, stmt.rhs, place_ty);
                return;
            }
            const slot = try localSlot(b, stmt.lhs, place_ty);
            // Whole-`self` reassignment inside a `mut self` method (`self = expr`): the
            // slot holds a POINTER, so produce the RHS through it into the caller's
            // storage rather than overwriting the local pointer (M9).
            if (slot == b.mut_self_slot) {
                try lowerExprInto(b, stmt.rhs, try rootAddr(b, slot), place_ty);
                return;
            }
            try storeInto(b, slot, stmt.rhs, place_ty);
        },
        .return_stmt => {
            const arg: Ir.Operand = if (stmt.lhs == Ast.none)
                .none
            else
                try lowerExpr(b, stmt.lhs);
            try brTo(b, b.exit, arg);
        },
        .expr_stmt => {
            _ = try lowerExpr(b, stmt.lhs); // evaluate for effect, discard
        },
        .block => try lowerBlockStmts(b, stmt_idx),
        .if_stmt => try lowerIfStmt(b, stmt_idx),
        .while_stmt => try lowerWhile(b, stmt_idx, null),
        .for_stmt => try lowerFor(b, stmt_idx, null),
        .labeled => try lowerLabeledStmt(b, stmt_idx),
        .break_stmt => try lowerBreak(b, stmt_idx),
        .continue_stmt => try lowerContinue(b, stmt_idx),
        // A poison leaf must never reach lower: a tainted tree is gated out before
        // codegen (a later milestone), and it is not produced anywhere yet.
        .error_node => unreachable,
        else => try b.note(stmt.main_token, "statement unsupported in lower"),
    }
}

/// Store the value of `expr` into `slot`. A scalar emits a store; an aggregate
/// (str/struct/enum) is PRODUCED INTO the slot directly via `lowerExprInto` (the
/// produce-into-slot path: construct/copy/match write straight to the destination,
/// never value-then-copy).
fn storeInto(b: *Builder, slot: Ir.SlotId, expr: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!void {
    switch (ty.kind) {
        .int, .bool => {
            const v = try lowerExpr(b, expr);
            const addr = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            _ = try b.emit(.{ .store = .{ .addr = addr, .val = operandValue(v), .ty = ty } }, null);
        },
        .str, .@"struct", .@"enum" => {
            const dst = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            try lowerExprInto(b, expr, dst, ty);
        },
        .unit => {
            _ = try lowerExpr(b, expr); // for effect
        },
        else => try b.note(b.in.tree.nodes[(expr).int()].main_token, "aggregate var/assign unsupported in lower"),
    }
}

/// Lower an expression in VALUE context: a scalar yields `Operand.value`; a str
/// (16-byte aggregate) yields `Operand.slot`; unit yields `.none`.
fn lowerExpr(b: *Builder, node_idx: Ast.Index) error{OutOfMemory}!Ir.Operand {
    if (node_idx == Ast.none) return .none;
    const n = b.in.tree.nodes[(node_idx).int()];
    const ty = b.in.node_types[(node_idx).int()];
    // A check-time `type_var` (M2) / composite `App` (M4) is substituted/reified to a
    // concrete kind BEFORE lowering; if one reaches here, the mono tail missed a
    // node_types slot — trip loudly in Debug/ReleaseSafe rather than miscompile.
    std.debug.assert(ty.kind != .type_var and ty.kind != .app);
    switch (n.tag) {
        .literal_number => {
            const v = parseInt(b.in.tokens[n.main_token].text(b.in.source)) orelse {
                try b.note(n.main_token, "integer literal out of range for codegen (i64)");
                return .{ .value = try b.emit(.{ .iconst = 0 }, Typecheck.Type.int) };
            };
            return .{ .value = try b.emit(.{ .iconst = v }, Typecheck.Type.int) };
        },
        .literal_bool => {
            const t = b.in.tokens[n.main_token].tag == .kw_true;
            return .{ .value = try b.emit(.{ .bconst = t }, Typecheck.Type.@"bool") };
        },
        .literal_unit => {
            _ = try b.emit(.unit, null);
            return .none;
        },
        .literal_string => return try lowerStrLiteral(b, node_idx),
        .identifier => return try lowerIdentifier(b, node_idx, ty),
        .unary => return try lowerUnary(b, node_idx, n),
        .binary => return try lowerBinary(b, node_idx, n),
        .call => {
            if (isQualifiedVariantCtorCall(b, n, ty)) {
                return try aggregateValue(b, node_idx, ty);
            }
            return try lowerCall(b, node_idx, n);
        },
        .block => return try lowerBlockValue(b, node_idx, ty),
        .if_stmt => return try lowerIfValue(b, node_idx, ty),
        .loop_expr => return try lowerLoopValue(b, node_idx, ty, null),
        .labeled => return try lowerLabeledValue(b, node_idx, ty),
        .field_access => return try lowerFieldAccess(b, node_idx, ty),
        .match_expr => return try lowerMatchValue(b, node_idx, ty),
        .struct_init, .enum_init_unit, .enum_init_tuple, .enum_init_struct => {
            // An aggregate-producing expression in value context: materialize it
            // into a fresh temp slot and yield Operand.slot.
            return try aggregateValue(b, node_idx, ty);
        },
        // A poison leaf must never reach lower: a tainted tree is gated out before
        // codegen (a later milestone), and it is not produced anywhere yet.
        .error_node => unreachable,
        else => {
            try b.note(n.main_token, "expression unsupported in lower");
            return .none;
        },
    }
}

fn lowerStrLiteral(b: *Builder, node_idx: Ast.Index) error{OutOfMemory}!Ir.Operand {
    const n = b.in.tree.nodes[(node_idx).int()];
    const bytes = (try decodeStringLiteral(b, n.main_token)) orelse return .none;
    // Content-hash the decoded bytes; `cstr_ptr` carries the hash and the bytes
    // are registered in the fn's literal table (codegen recovers them there and
    // emits the adrp+add). `addLiteral` takes ownership of `bytes`.
    const h = std.hash.Wyhash.hash(lit_seed, bytes);
    const len: i64 = @intCast(bytes.len);
    try b.addLiteral(h, bytes);

    const slot = try b.addSlot(Typecheck.Type.str);
    const base = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
    const p = try b.emit(.{ .cstr_ptr = h }, Typecheck.Type.int);
    _ = try b.emit(.{ .store = .{ .addr = base, .val = p, .ty = Typecheck.Type.int } }, null);
    const len_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = 8, .ty = Typecheck.Type.int } }, Typecheck.Type.int);
    const lenv = try b.emit(.{ .iconst = len }, Typecheck.Type.int);
    _ = try b.emit(.{ .store = .{ .addr = len_addr, .val = lenv, .ty = Typecheck.Type.int } }, null);
    return .{ .slot = slot };
}

fn lowerIdentifier(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const slot = try localSlot(b, node_idx, ty);
    switch (ty.kind) {
        .int, .bool => {
            const addr = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            const v = try b.emit(.{ .load = .{ .addr = addr, .ty = ty } }, ty);
            return .{ .value = v };
        },
        .str, .@"struct", .@"enum" => {
            // Whole-`self` value read inside a `mut self` method (`return self`, or
            // passing `self` by value): the slot holds a POINTER, not the struct, so
            // copy the pointee into a fresh temp and yield that (M9). An ordinary
            // aggregate local is passed by slot directly (no copy).
            if (slot == b.mut_self_slot) {
                const tmp = try b.addSlot(ty);
                const dst = try b.emit(.{ .slot_addr = tmp }, Typecheck.Type.int);
                _ = try b.emit(.{ .copy = .{ .dst = dst, .src = try rootAddr(b, slot), .ty = ty } }, null);
                return .{ .slot = tmp };
            }
            return .{ .slot = slot }; // aggregate: pass by slot, no load
        },
        .unit => return .none,
        else => {
            try b.note(b.in.tree.nodes[(node_idx).int()].main_token, "identifier type unsupported in lower");
            return .none;
        },
    }
}

fn lowerUnary(b: *Builder, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Ir.Operand {
    const op = b.in.tokens[n.main_token].tag;
    switch (op) {
        .minus => {
            const v = try lowerExpr(b, n.lhs);
            return .{ .value = try b.emit(.{ .neg = operandValue(v) }, Typecheck.Type.int) };
        },
        .bang => {
            const v = try lowerExpr(b, n.lhs);
            return .{ .value = try b.emit(.{ .bnot = operandValue(v) }, Typecheck.Type.@"bool") };
        },
        else => {
            try b.note(n.main_token, "unary operator unsupported in lower");
            _ = node_idx;
            return .none;
        },
    }
}

fn lowerBinary(b: *Builder, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Ir.Operand {
    _ = node_idx;
    const op = b.in.tokens[n.main_token].tag;
    switch (op) {
        .plus, .minus, .star, .slash => {
            const lhs = operandValue(try lowerExpr(b, n.lhs));
            const rhs = operandValue(try lowerExpr(b, n.rhs));
            const bin: Ir.Bin = .{ .lhs = lhs, .rhs = rhs };
            const ir_op: Ir.Op = switch (op) {
                .plus => .{ .add = bin },
                .minus => .{ .sub = bin },
                .star => .{ .mul = bin },
                .slash => .{ .sdiv = bin },
                else => unreachable,
            };
            return .{ .value = try b.emit(ir_op, Typecheck.Type.int) };
        },
        .lt, .lt_eq, .gt, .gt_eq, .eq_eq, .bang_eq => {
            const lhs = operandValue(try lowerExpr(b, n.lhs));
            const rhs = operandValue(try lowerExpr(b, n.rhs));
            const cc = condFromToken(op);
            return .{ .value = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.@"bool") };
        },
        .amp_amp, .pipe_pipe => return try lowerAndOrValue(b, n, op),
        else => {
            try b.note(n.main_token, "binary operator unsupported in lower");
            return .none;
        },
    }
}

/// `a && b` / `a || b` as a VALUE: short-circuit via blocks. The merge is a bool
/// block param on the join. For `&&`: a-false stores `false` and branches to join;
/// a-true falls through to evaluate b, which branches to join with its own value.
/// For `||`: a-true stores `true`; a-false evaluates b.
fn lowerAndOrValue(b: *Builder, n: Ast.Node, op: TokenTag) error{OutOfMemory}!Ir.Operand {
    const eval_b = try b.addBlock();
    const short = try b.addBlock(); // the short-circuit constant block
    const join = try b.addBlock();
    const merge = try b.addParam(join, Typecheck.Type.@"bool");

    // Condition on `a`: for &&, a-true → eval_b, a-false → short. For ||, swap.
    if (op == .amp_amp) {
        try genCond(b, n.lhs, eval_b, short);
    } else {
        try genCond(b, n.lhs, short, eval_b);
    }

    // short block: the short-circuit constant (&&→false, ||→true) → join(const).
    b.switchTo(short);
    const c = try b.emit(.{ .bconst = (op == .pipe_pipe) }, Typecheck.Type.@"bool");
    try brTo(b, join, .{ .value = c });

    // eval_b block: evaluate b, branch to join with its value.
    b.switchTo(eval_b);
    const bv = try lowerExpr(b, n.rhs);
    if (!b.termSet()) try brTo(b, join, bv);

    b.switchTo(join);
    return .{ .value = merge };
}

/// True when `sig` is a generic template (some param is a check-time `type_var`).
/// Bare inferred calls resolve to the template's `.func`; genericness is detected
/// from the callee sig so `lowerCall` selects the reified instance, not the template.
fn sigHasTypeVar(sig: Sig) bool {
    for (sig.params) |p| if (p.isTypeVar()) return true;
    return false;
}

/// `1 + max type_var ordinal` over `sig.params` — the generic-param count `Infer.infer`
/// needs. Equals `generic_params.len` for any bare call that reached lower: such a call
/// passed Pass C, so every type-var was bound, hence appears in a value param.
fn genericParamCount(sig: Sig) u32 {
    var m: u32 = 0;
    for (sig.params) |p| if (p.isTypeVar() and p.typeVarOrd() > m) {
        m = p.typeVarOrd();
    };
    return m + 1;
}

fn lowerCall(b: *Builder, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Ir.Operand {
    const callee_node = b.in.tree.nodes[(n.lhs).int()];
    var callee: Link.SymName = undefined;
    // A method call `recv.m(args)` (M8): dispatch to the method's mangled symbol and
    // PREPEND the receiver as the `self` arg (arg 0). `self_recv` set ⟺ this is a
    // method call; the receiver expr is lowered as arg 0 below.
    var self_recv: Ast.Index = Ast.none;
    // A `mut self` method (M9): pass the receiver's ADDRESS (a place) as arg 0 instead
    // of a by-value copy, so the callee mutates the caller's storage.
    var self_mut = false;
    if (callee_node.tag == .type_app) {
        // A generic call `id[int](..)` (M2): the callee is a `type_app` whose base
        // identifier carries the template gid. Resolve to the reified instance's
        // mangled SymName by matching (gid + the concrete type-args the checker
        // wrote into node_types) against the instance table — a pure read of
        // concrete types, no substitution needed here.
        const base_res = b.in.resolutions[(callee_node.lhs).int()];
        if (base_res != .func) {
            try b.note(n.main_token, "call target unsupported in lower");
            return .none;
        }
        const targ_nodes = Ast.rangeSlice(b.in.tree, (callee_node.rhs).int());
        const targs = try b.gpa.alloc(Typecheck.Type, targ_nodes.len);
        defer b.gpa.free(targs);
        for (targ_nodes, 0..) |tn, i| targs[i] = b.in.node_types[(tn).int()];
        const ii = Mono.find(b.in.instances, base_res.func, targs) orelse {
            try b.note(callee_node.main_token, "unresolved generic instance in lower");
            return .none;
        };
        callee = .{ .kind = .user_fn, .name = b.in.instances[ii].name };
    } else if (methodGidOf(b, n)) |m| {
        // Method dispatch (M8): callee = the method's mangled global symbol; the
        // receiver (`callee_node.lhs`) is prepended as `self` in the arg build below.
        callee = b.in.names[m.fn_id];
        self_recv = callee_node.lhs;
        self_mut = m.mut_self;
    } else {
        // The callee identifier resolves to a `.func` index into `names` (this also
        // covers the `print` builtin, whose name index points at the synthetic entry).
        const callee_res = b.in.resolutions[(n.lhs).int()];
        if (callee_res != .func) {
            try b.note(n.main_token, "call target unsupported in lower");
            return .none;
        }
        if (callee_node.tag == .identifier and callee_res.func < b.in.sigs.len and sigHasTypeVar(b.in.sigs[callee_res.func])) {
            // A bare inferred generic call `id(7)` (M3): the plain-identifier callee
            // resolves to a generic template (its sig params carry `type_var`s). Re-run
            // the SHARED matcher over the value-arg node_types to pick the SAME instance
            // Pass C / scanCalls selected, then use its mangled name (mirroring the
            // type_app branch). A miss (arity/conflict/unbound/unminted) is an internal
            // invariant break — Pass C already gated it — so note-and-drop.
            const sig = b.in.sigs[callee_res.func];
            const value_args = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const arg_types = try b.gpa.alloc(Typecheck.Type, value_args.len);
            defer b.gpa.free(arg_types);
            for (value_args, 0..) |va, i| arg_types[i] = b.in.node_types[(va).int()];
            const targs = (try Infer.infer(b.gpa, genericParamCount(sig), sig.params, arg_types)) orelse {
                try b.note(callee_node.main_token, "unresolved generic instance in lower");
                return .none;
            };
            defer b.gpa.free(targs);
            const ii = Mono.find(b.in.instances, callee_res.func, targs) orelse {
                try b.note(callee_node.main_token, "unresolved generic instance in lower");
                return .none;
            };
            callee = .{ .kind = .user_fn, .name = b.in.instances[ii].name };
        } else {
            callee = b.in.names[callee_res.func];
        }
    }
    const result_ty = b.in.node_types[(node_idx).int()];

    // Evaluate every arg left-to-right into an Operand (scalar→value, str→slot). For a
    // method call the receiver is the FIRST arg (`self`, by value — reusing the M2
    // struct-arg reg-pair/sret path), followed by the source args in order.
    const arg_nodes = Ast.rangeSlice(b.in.tree, (n.rhs).int());
    const self_n: usize = if (self_recv != Ast.none) 1 else 0;
    const args = try b.gpa.alloc(Ir.Operand, arg_nodes.len + self_n);
    errdefer b.gpa.free(args);
    if (self_recv != Ast.none) {
        if (self_mut) {
            // Pass the receiver's place ADDRESS (an int value) as `self`. The receiver
            // was proven a mutable place by the body checker (T0019); a non-place here
            // is an internal invariant break — note-and-drop rather than miscompile.
            const addr = try lowerPlaceAddr(b, self_recv);
            if (addr == Ir.none_value) {
                try b.note(callee_node.main_token, "mut-self receiver is not a place in lower");
                b.gpa.free(args);
                return .none;
            }
            args[0] = .{ .value = addr };
        } else {
            args[0] = try lowerExpr(b, self_recv);
        }
    }
    for (arg_nodes, 0..) |arg, i| {
        args[self_n + i] = try lowerExpr(b, arg);
    }

    // Result placement: scalar → an Instr.result value; aggregate → a fresh
    // ret_slot; unit → neither. The ABI (reg vs sret) is decided in codegen.
    switch (result_ty.kind) {
        .int, .bool => {
            const v = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, result_ty);
            return .{ .value = v };
        },
        .str, .@"struct", .@"enum" => {
            // Aggregate result lands in a fresh slot; the ABI (reg-pair vs sret)
            // is decided in codegen from `ret_slot`.
            const slot = try b.addSlot(result_ty);
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = slot } }, null);
            return .{ .slot = slot };
        },
        .unit, .never => {
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, null);
            return .none;
        },
        else => {
            const slot = try b.addSlot(result_ty);
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = slot } }, null);
            try b.note(n.main_token, "call result type unsupported in lower");
            return .{ .slot = slot };
        },
    }
}

/// PRODUCE-INTO-SLOT: lower the aggregate (or scalar/str) `expr` writing its value
/// directly to the memory at the ptr value `dst_ptr` (of element type `ty`). This
/// generalizes the codegen `genStructTo*` family: struct/enum construction, a
/// value-if/block/loop/labeled/match, a copy of an identifier/field/call result —
/// all write the destination, never value-then-copy at the top level. `dst_ptr` is
/// a ptr VALUE (from slot_addr or field_addr).
/// True when a `.call` node is a qualified tuple-variant CONSTRUCTION `N.V(args)`
/// rather than a cross-module enum-RETURNING call `mod.fn(args)`. Both parse as a
/// `.call` over a `field_access` callee and are told apart ONLY by the callee's
/// resolution: a real call binds its field_access callee to a `.func`; a variant
/// constructor's receiver binds to a type, so the callee stays unresolved.
/// Misclassifying a call as construction inlines a wrong-layout variant and drops
/// the call → silent miscompile / infinite recursion.
fn isQualifiedVariantCtorCall(b: *Builder, n: Ast.Node, ty: Typecheck.Type) bool {
    // A method returning an enum ALSO parses as `.call` over an unresolved
    // field_access callee, so it would otherwise be misrouted to variant
    // construction. Distinguish by the receiver being a VALUE with a resolved method.
    return ty.kind == .@"enum" and b.in.tree.nodes[(n.lhs).int()].tag == .field_access and
        b.in.resolutions[(n.lhs).int()] != .func and methodGidOf(b, n) == null;
}

/// The `Method` a method call `recv.m(args)` dispatches to, or null when `n` is
/// not a method call. A method callee is a `field_access` NOT bound to a `.func`
/// (that is a qualified module call) whose receiver types to a concrete struct/enum
/// with a matching entry in the method table. A pure content-keyed lookup (no
/// hashmap/thread order), so it is identical at any `-jN`. Returns the whole
/// `Method` (not just `fn_id`) so the caller reads `mut_self` for the by-address
/// receiver ABI (M9); `null`-semantics are unchanged for the ctor-call classifier.
fn methodGidOf(b: *Builder, n: Ast.Node) ?Typecheck.Method {
    const cn = b.in.tree.nodes[(n.lhs).int()];
    if (cn.tag != .field_access) return null;
    if (b.in.resolutions[(n.lhs).int()] == .func) return null;
    const recv = b.in.node_types[(cn.lhs).int()];
    if (recv.kind != .@"struct" and recv.kind != .@"enum") return null;
    const member = b.in.tokens[cn.main_token].text(b.in.source);
    return Typecheck.findMethod(b.in.methods, recv, member);
}

fn lowerExprInto(b: *Builder, expr: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(expr).int()];
    switch (ty.kind) {
        .int, .bool => {
            const v = operandValue(try lowerExpr(b, expr));
            _ = try b.emit(.{ .store = .{ .addr = dst_ptr, .val = v, .ty = ty } }, null);
            return;
        },
        .unit => {
            _ = try lowerExpr(b, expr);
            return;
        },
        .str, .@"struct", .@"enum" => {},
        else => {
            try b.note(n.main_token, "produce-into-slot type unsupported in lower");
            return;
        },
    }

    // Aggregate: dispatch on the producing construct.
    switch (n.tag) {
        .literal_string => try storeStrLiteralInto(b, expr, dst_ptr),
        .struct_init => try lowerStructInitInto(b, expr, dst_ptr),
        .enum_init_unit, .enum_init_tuple, .enum_init_struct => try lowerEnumInitInto(b, expr, dst_ptr, ty),
        // qualified `N.V` (field_access) / `N.V(args)` (call) that typecheck
        // classified as enum-valued are variant constructions, NOT field reads /
        // real calls. Detect by enum type + shape.
        .field_access => {
            if (ty.kind == .@"enum" and !isLocalRootedPlace(b, expr)) {
                try lowerEnumInitInto(b, expr, dst_ptr, ty);
            } else {
                try copyAggInto(b, expr, dst_ptr, ty);
            }
        },
        .call => {
            if (isQualifiedVariantCtorCall(b, n, ty)) {
                try lowerEnumInitInto(b, expr, dst_ptr, ty);
            } else {
                try copyAggInto(b, expr, dst_ptr, ty);
            }
        },
        .if_stmt => try lowerIfValueInto(b, expr, dst_ptr, ty),
        .block => try lowerBlockValueInto(b, expr, dst_ptr, ty),
        .loop_expr => try lowerLoopValueInto(b, expr, dst_ptr, ty, null),
        .labeled => try lowerLabeledValueInto(b, expr, dst_ptr, ty),
        .match_expr => try lowerMatchInto(b, expr, dst_ptr, ty),
        .identifier => try copyAggInto(b, expr, dst_ptr, ty),
        else => try b.note(n.main_token, "aggregate expression unsupported in lower"),
    }
}

/// An aggregate value yielded by an identifier / field-access / call: lower it to
/// an Operand.slot, then byte-copy that slot into `dst_ptr`.
fn copyAggInto(b: *Builder, expr: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const op = try lowerExpr(b, expr);
    const src_ptr = try operandPtr(b, op);
    if (src_ptr == Ir.none_value) return; // a diagnostic was already emitted
    _ = try b.emit(.{ .copy = .{ .dst = dst_ptr, .src = src_ptr, .ty = ty } }, null);
}

/// The ptr VALUE backing an aggregate Operand (its slot address). For a scalar/
/// none Operand this is a lowering bug → `none_value`.
fn operandPtr(b: *Builder, op: Ir.Operand) error{OutOfMemory}!Ir.ValueId {
    return switch (op) {
        .slot => |s| try b.emit(.{ .slot_addr = s }, Typecheck.Type.int),
        else => Ir.none_value,
    };
}

/// Store a str literal's (cstr_ptr, len) pair into the destination ptr; the
/// `ptr@0, len@8` field layout is pinned by the `lower-aggregates: str literal` test.
fn storeStrLiteralInto(b: *Builder, expr: Ast.Index, dst_ptr: Ir.ValueId) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(expr).int()];
    const bytes = (try decodeStringLiteral(b, n.main_token)) orelse return;
    const h = std.hash.Wyhash.hash(lit_seed, bytes);
    const len: i64 = @intCast(bytes.len);
    try b.addLiteral(h, bytes);
    const p = try b.emit(.{ .cstr_ptr = h }, Typecheck.Type.int);
    _ = try b.emit(.{ .store = .{ .addr = dst_ptr, .val = p, .ty = Typecheck.Type.int } }, null);
    const len_addr = try b.emit(.{ .field_addr = .{ .base = dst_ptr, .off = 8, .ty = Typecheck.Type.int } }, Typecheck.Type.int);
    const lenv = try b.emit(.{ .iconst = len }, Typecheck.Type.int);
    _ = try b.emit(.{ .store = .{ .addr = len_addr, .val = lenv, .ty = Typecheck.Type.int } }, null);
}

/// Materialize an aggregate-producing expression into a fresh temp slot and return
/// Operand.slot. Used in value context (an arg, a nested field initializer source,
/// a scrutinee spill).
fn aggregateValue(b: *Builder, expr: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const slot = try b.addSlot(ty);
    const dst = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
    try lowerExprInto(b, expr, dst, ty);
    return .{ .slot = slot };
}

/// Read a `field_access` in VALUE context. A scalar field → slot_addr/field_addr +
/// load → value. A str/struct/enum field → Operand.slot of a temp the field bytes
/// are copied into (so a downstream consumer treats it like any other aggregate
/// slot). A local-rooted base reads in place; an rvalue base (a call/construction)
/// is materialized to a temp first.
fn lowerFieldAccess(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    // A qualified enum unit-variant `N.V` reaches here as field_access but is a
    // construction, not a field read.
    if (ty.kind == .@"enum" and !isLocalRootedPlace(b, node_idx)) {
        return try aggregateValue(b, node_idx, ty);
    }
    const addr = try lowerPlaceAddr(b, node_idx);
    if (addr == Ir.none_value) return .none;
    switch (ty.kind) {
        .int, .bool => {
            const v = try b.emit(.{ .load = .{ .addr = addr, .ty = ty } }, ty);
            return .{ .value = v };
        },
        .str, .@"struct", .@"enum" => {
            // Copy the field bytes into a temp slot so it is a self-contained
            // aggregate operand (the place address may be reused/clobbered).
            const tmp = try b.addSlot(ty);
            const dst = try b.emit(.{ .slot_addr = tmp }, Typecheck.Type.int);
            _ = try b.emit(.{ .copy = .{ .dst = dst, .src = addr, .ty = ty } }, null);
            return .{ .slot = tmp };
        },
        .unit => return .none,
        else => {
            try b.note(b.in.tree.nodes[(node_idx).int()].main_token, "field access type unsupported in lower");
            return .none;
        },
    }
}

/// Store into a `field_access` place (`p.x = v`, `p.a.b = v`). The place address is
/// a field_addr chain rooted at a local slot; scalar → store, aggregate → produce
/// into the place address.
fn lowerFieldStore(b: *Builder, place: Ast.Index, value: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!void {
    const addr = try lowerPlaceAddr(b, place);
    if (addr == Ir.none_value) {
        try b.note(b.in.tree.nodes[(place).int()].main_token, "field store target unsupported in lower");
        return;
    }
    switch (ty.kind) {
        .int, .bool => {
            const v = operandValue(try lowerExpr(b, value));
            _ = try b.emit(.{ .store = .{ .addr = addr, .val = v, .ty = ty } }, null);
        },
        .str, .@"struct", .@"enum" => try lowerExprInto(b, value, addr, ty),
        else => try b.note(b.in.tree.nodes[(place).int()].main_token, "field store type unsupported in lower"),
    }
}

/// The address of a slot's CONTENTS. For an ordinary slot that is `slot_addr(slot)`.
/// For the `mut self` receiver slot (M9) the slot holds a POINTER to the caller's
/// place, so the address of the receiver's storage is that pointer — a `load` of
/// `slot_addr(slot)`. Every non-mut fn has `mut_self_slot == none_slot`, which no
/// live slot equals, so this collapses to a bare `slot_addr` (byte-identical).
fn rootAddr(b: *Builder, slot: Ir.SlotId) error{OutOfMemory}!Ir.ValueId {
    const sa = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
    if (slot == b.mut_self_slot)
        return try b.emit(.{ .load = .{ .addr = sa, .ty = Typecheck.Type.int } }, Typecheck.Type.int);
    return sa;
}

/// The ptr VALUE of a LOCAL-ROOTED place (`p`, `p.x`, `p.a.b`): a slot_addr at the
/// root + a field_addr per `.field` hop, resolving each field name → byte offset
/// from the layout. Returns `none_value` for a non-local-rooted place.
fn lowerPlaceAddr(b: *Builder, node_idx: Ast.Index) error{OutOfMemory}!Ir.ValueId {
    const n = b.in.tree.nodes[(node_idx).int()];
    switch (n.tag) {
        .identifier => {
            if (b.in.resolutions[(node_idx).int()] != .local) return Ir.none_value;
            const slot = try localSlot(b, node_idx, b.in.node_types[(node_idx).int()]);
            return try rootAddr(b, slot);
        },
        .field_access => {
            const base_addr = try lowerPlaceAddr(b, n.lhs);
            if (base_addr == Ir.none_value) return Ir.none_value;
            const base_ty = b.in.node_types[(n.lhs).int()];
            if (base_ty.kind != .@"struct") return Ir.none_value;
            const layout = b.in.layouts[base_ty.struct_id];
            const fname = b.in.tokens[n.main_token].text(b.in.source);
            for (layout.field_names, 0..) |dn, j| {
                if (std.mem.eql(u8, dn, fname)) {
                    return try b.emit(.{ .field_addr = .{ .base = base_addr, .off = layout.offsets[j], .ty = layout.field_types[j] } }, Typecheck.Type.int);
                }
            }
            return base_addr; // unreachable on a well-typed program
        },
        else => return Ir.none_value,
    }
}

/// Write a `Name { field: value, ... }` construction into the destination ptr.
/// Each field initializer writes at `layout.offsets[idx]` (mapped by name, so a
/// reordered initializer list is fine), scalars via store, aggregates recursing
/// through `lowerExprInto` at a field_addr.
fn lowerStructInitInto(b: *Builder, node_idx: Ast.Index, dst_ptr: Ir.ValueId) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(node_idx).int()];
    const id = b.in.node_types[(node_idx).int()].struct_id;
    const layout = b.in.layouts[id];
    for (Ast.rangeSlice(b.in.tree, (n.rhs).int())) |fi_idx| {
        const fi = b.in.tree.nodes[(fi_idx).int()];
        const fname = b.in.tokens[fi.main_token].text(b.in.source);
        var foff: u32 = 0;
        var fty = Typecheck.Type.int;
        for (layout.field_names, 0..) |dn, j| {
            if (std.mem.eql(u8, dn, fname)) {
                foff = layout.offsets[j];
                fty = layout.field_types[j];
                break;
            }
        }
        const faddr = try b.emit(.{ .field_addr = .{ .base = dst_ptr, .off = foff, .ty = fty } }, Typecheck.Type.int);
        try lowerExprInto(b, fi.lhs, faddr, fty);
    }
}

/// Decoded view of a variant-construction node, normalizing inferred enum_init_*
/// and qualified `N.V`/`N.V(...)` (field_access/call) into one shape.
const VariantCtor = struct {
    vtok: u32,
    payload: Ast.Index, // Range header over arg/field-init nodes, or none
    is_struct_form: bool,
};

fn decodeVariantCtor(b: *Builder, node_idx: Ast.Index) VariantCtor {
    const n = b.in.tree.nodes[(node_idx).int()];
    return switch (n.tag) {
        .enum_init_unit => .{ .vtok = n.main_token, .payload = Ast.none, .is_struct_form = false },
        .enum_init_tuple => .{ .vtok = n.main_token, .payload = n.rhs, .is_struct_form = false },
        .enum_init_struct => .{ .vtok = n.main_token, .payload = n.rhs, .is_struct_form = true },
        .field_access => .{ .vtok = n.main_token, .payload = Ast.none, .is_struct_form = false },
        .call => .{ .vtok = b.in.tree.nodes[(n.lhs).int()].main_token, .payload = n.rhs, .is_struct_form = false },
        else => unreachable,
    };
}

/// Build a variant value into the destination ptr: tag (iconst variant index) at
/// offset 0, then each payload element at `payload_off + variant.offsets[i]`
/// (struct-form fields mapped by name, reorder-safe).
fn lowerEnumInitInto(b: *Builder, node_idx: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const e = b.in.enum_layouts[ty.enum_id];
    const ctor = decodeVariantCtor(b, node_idx);
    const vname = b.in.tokens[ctor.vtok].text(b.in.source);
    var vi: u32 = 0;
    for (e.variants, 0..) |v, i| {
        if (std.mem.eql(u8, v.name, vname)) {
            vi = @intCast(i);
            break;
        }
    }
    const variant = e.variants[vi];

    const tagv = try b.emit(.{ .iconst = @intCast(vi) }, Typecheck.Type.int);
    _ = try b.emit(.{ .store = .{ .addr = dst_ptr, .val = tagv, .ty = Typecheck.Type.int } }, null);

    if (ctor.payload == Ast.none) return;
    const elems = Ast.rangeSlice(b.in.tree, (ctor.payload).int());
    for (elems, 0..) |elem_idx, i| {
        const value: Ast.Index = if (ctor.is_struct_form) b.in.tree.nodes[(elem_idx).int()].lhs else elem_idx;
        // Map a struct-form field-init by name (reorder-safe); positional otherwise.
        var fi_idx: usize = i;
        if (ctor.is_struct_form) {
            const fname = b.in.tokens[b.in.tree.nodes[(elem_idx).int()].main_token].text(b.in.source);
            for (variant.field_names, 0..) |dn, j| {
                if (std.mem.eql(u8, dn, fname)) {
                    fi_idx = j;
                    break;
                }
            }
        }
        const abs_off = e.payload_off + variant.offsets[fi_idx];
        const fty = variant.field_types[fi_idx];
        const faddr = try b.emit(.{ .field_addr = .{ .base = dst_ptr, .off = abs_off, .ty = fty } }, Typecheck.Type.int);
        try lowerExprInto(b, value, faddr, fty);
    }
}

/// A `match` in VALUE context yielding a SCALAR (or str/struct/enum). Materialize
/// the result via `lowerMatchInto` into a fresh slot, then yield: a scalar loads
/// from the slot to a value; an aggregate yields Operand.slot.
fn lowerMatchValue(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    switch (ty.kind) {
        .int, .bool => {
            const slot = try b.addSlot(ty);
            const dst = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            try lowerMatchInto(b, node_idx, dst, ty);
            const addr = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            const v = try b.emit(.{ .load = .{ .addr = addr, .ty = ty } }, ty);
            return .{ .value = v };
        },
        .str, .@"struct", .@"enum" => return try aggregateValue(b, node_idx, ty),
        .unit, .never => {
            const slot = try b.addSlot(Typecheck.Type.int);
            const dst = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            try lowerMatchInto(b, node_idx, dst, ty);
            return .none;
        },
        else => {
            try b.note(b.in.tree.nodes[(node_idx).int()].main_token, "match result type unsupported in lower");
            return .none;
        },
    }
}

/// Lower a `match` writing the selected arm body into `dst_ptr`. Spills the
/// scrutinee to a slot, then a linear FIRST-MATCH-WINS chain of structural tests
/// (each arm has a `next` block for a mismatch / failed guard) dispatches to each
/// arm body, which produces into `dst_ptr` and branches to a shared `join`. NO
/// switch terminator (preserves overlapping-pattern order).
fn lowerMatchInto(b: *Builder, node_idx: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(node_idx).int()];
    const scrut_ty = b.in.node_types[(n.lhs).int()];

    // Spill the scrutinee into a slot (every match reads it by address).
    const scrut_op = try lowerExpr(b, n.lhs);
    const scrut_slot = try spillScrutinee(b, scrut_op, scrut_ty, b.in.tree.nodes[(n.lhs).int()].main_token);
    if (scrut_slot == Ir.none_slot) return; // diagnostic already emitted

    const join = try b.addBlock();
    const arms = Ast.rangeSlice(b.in.tree, (n.rhs).int());
    for (arms) |arm_idx| {
        const arm = b.in.tree.nodes[(arm_idx).int()];
        const h = Ast.armHeaderAt(b.in.tree, (arm.rhs).int());
        const next = try b.addBlock();
        // Structural test: a mismatch branches to `next`. Bindings are stored as a
        // side effect of a successful (sub)match.
        try testPattern(b, arm.lhs, scrut_slot, 0, scrut_ty, next);
        // Guard (after the pattern matched): false → next (first-match-wins).
        if (h.guard != Ast.none) {
            const body_bb = try b.addBlock();
            try genCond(b, h.guard, body_bb, next);
            b.switchTo(body_bb);
        }
        try lowerExprInto(b, h.body, dst_ptr, ty);
        if (!b.termSet()) try brTo(b, join, .none);
        b.switchTo(next);
    }
    // Exhaustiveness is enforced by types.zig; a fall-through past the last arm is
    // unreachable. Leave `next`(the final one)'s terminator as the default
    // `unreachable`, and continue lowering at `join`.
    try brTo(b, join, .none); // the trailing `next` block branches to join (dead but well-formed)
    b.switchTo(join);
}

/// Spill a scrutinee Operand into a slot, returning the slot id. A scalar is stored
/// to a fresh int/bool slot; an aggregate already lives in a slot.
fn spillScrutinee(b: *Builder, op: Ir.Operand, ty: Typecheck.Type, tok: u32) error{OutOfMemory}!Ir.SlotId {
    switch (op) {
        .slot => |s| return s,
        .value => |v| {
            const slot = try b.addSlot(ty);
            const addr = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            _ = try b.emit(.{ .store = .{ .addr = addr, .val = v, .ty = ty } }, null);
            return slot;
        },
        .none => {
            try b.note(tok, "match scrutinee unsupported in lower");
            return Ir.none_slot;
        },
    }
}

/// Recursively test a pattern against the value at byte offset `off` within the
/// slot `base` (type `val_ty`), branching to `fail` on a mismatch and storing
/// bound leaves into their slots. Addresses via
/// field_addr + the enum tag via get_tag.
fn testPattern(b: *Builder, pat_idx: Ast.Index, base: Ir.SlotId, off: u32, val_ty: Typecheck.Type, fail: Ir.BlockId) error{OutOfMemory}!void {
    const pat = b.in.tree.nodes[(pat_idx).int()];
    switch (pat.tag) {
        .pattern_wildcard => {},
        .pattern_binding => {
            if (pat.rhs == Ast.none) {
                try bindLeaf(b, pat_idx, base, off, val_ty);
            } else {
                try testPattern(b, pat.rhs, base, off, val_ty, fail);
            }
        },
        .pattern_literal => {
            const text = b.in.tokens[pat.main_token].text(b.in.source);
            const lit: i64 = if (b.in.tokens[pat.main_token].tag == .number)
                parseIntLit(text)
            else if (std.mem.eql(u8, text, "true")) 1 else 0;
            const addr = try slotFieldAddr(b, base, off, val_ty);
            const cur = try b.emit(.{ .load = .{ .addr = addr, .ty = val_ty } }, val_ty);
            const litv = try b.emit(.{ .iconst = lit }, Typecheck.Type.int);
            const c = try b.emit(.{ .icmp = .{ .cc = .ne, .lhs = cur, .rhs = litv } }, Typecheck.Type.@"bool");
            const ok = try b.addBlock();
            b.setTerm(.{ .cond_br = .{ .cond = c, .t = fail, .f = ok } });
            b.switchTo(ok);
        },
        .pattern_or => {
            const alts = Ast.rangeSlice(b.in.tree, (pat.lhs).int());
            const body = try b.addBlock();
            for (alts, 0..) |alt, i| {
                if (i + 1 == alts.len) {
                    // Last alt: a mismatch is the whole or-pattern's failure.
                    try testPattern(b, alt, base, off, val_ty, fail);
                    if (!b.termSet()) try brTo(b, body, .none);
                } else {
                    const alt_fail = try b.addBlock();
                    try testPattern(b, alt, base, off, val_ty, alt_fail);
                    if (!b.termSet()) try brTo(b, body, .none);
                    b.switchTo(alt_fail);
                }
            }
            b.switchTo(body);
        },
        .pattern_variant => {
            const e = b.in.enum_layouts[val_ty.enum_id];
            const vname = b.in.tokens[pat.main_token].text(b.in.source);
            var vi: u32 = 0;
            for (e.variants, 0..) |v, i| {
                if (std.mem.eql(u8, v.name, vname)) {
                    vi = @intCast(i);
                    break;
                }
            }
            // Compare the tag at `off` (offset 0 of the enum value).
            const ptr = try b.emit(.{ .slot_addr = base }, Typecheck.Type.int);
            const enum_ptr = if (off == 0) ptr else try b.emit(.{ .field_addr = .{ .base = ptr, .off = off, .ty = val_ty } }, Typecheck.Type.int);
            const tagv = try b.emit(.{ .get_tag = enum_ptr }, Typecheck.Type.int);
            const want = try b.emit(.{ .iconst = @intCast(vi) }, Typecheck.Type.int);
            const c = try b.emit(.{ .icmp = .{ .cc = .ne, .lhs = tagv, .rhs = want } }, Typecheck.Type.@"bool");
            const ok = try b.addBlock();
            b.setTerm(.{ .cond_br = .{ .cond = c, .t = fail, .f = ok } });
            b.switchTo(ok);
            // Recurse into payload sub-patterns IN PLACE.
            if (pat.rhs != Ast.none) {
                const variant = e.variants[vi];
                const binders = Ast.rangeSlice(b.in.tree, (pat.rhs).int());
                for (binders, 0..) |bnd_idx, i| {
                    const bnd = b.in.tree.nodes[(bnd_idx).int()];
                    var fi: usize = i;
                    if (variant.form == .@"struct") {
                        const src_name = if (bnd.lhs != Ast.none) b.in.tokens[b.in.tree.nodes[(bnd.lhs).int()].main_token].text(b.in.source) else b.in.tokens[bnd.main_token].text(b.in.source);
                        for (variant.field_names, 0..) |dn, j| {
                            if (std.mem.eql(u8, dn, src_name)) {
                                fi = j;
                                break;
                            }
                        }
                    }
                    const child_off = off + e.payload_off + variant.offsets[fi];
                    try testPattern(b, bnd_idx, base, child_off, variant.field_types[fi], fail);
                }
            }
        },
        else => {},
    }
}

/// Bind a whole-value `pattern_binding` leaf: copy the value at [base+off] into the
/// binding's slot (scalar load/store, aggregate copy). A binding with no `.local`
/// (e.g. an unused field) is a no-op.
fn bindLeaf(b: *Builder, bind_idx: Ast.Index, base: Ir.SlotId, off: u32, ty: Typecheck.Type) error{OutOfMemory}!void {
    if (b.in.resolutions[(bind_idx).int()] != .local) return;
    const dst_slot = try localSlot(b, bind_idx, ty);
    const src = try slotFieldAddr(b, base, off, ty);
    const dst = try b.emit(.{ .slot_addr = dst_slot }, Typecheck.Type.int);
    switch (ty.kind) {
        .int, .bool => {
            const v = try b.emit(.{ .load = .{ .addr = src, .ty = ty } }, ty);
            _ = try b.emit(.{ .store = .{ .addr = dst, .val = v, .ty = ty } }, null);
        },
        else => _ = try b.emit(.{ .copy = .{ .dst = dst, .src = src, .ty = ty } }, null),
    }
}

/// A ptr value addressing [slot + off] (the field `ty` carried for load/store
/// width). `off == 0` collapses to a bare `slot_addr`.
fn slotFieldAddr(b: *Builder, base: Ir.SlotId, off: u32, ty: Typecheck.Type) error{OutOfMemory}!Ir.ValueId {
    const ptr = try b.emit(.{ .slot_addr = base }, Typecheck.Type.int);
    if (off == 0) return ptr;
    return try b.emit(.{ .field_addr = .{ .base = ptr, .off = off, .ty = ty } }, Typecheck.Type.int);
}

/// Whether a `field_access`/identifier place is rooted at a local.
fn isLocalRootedPlace(b: *Builder, node_idx: Ast.Index) bool {
    const n = b.in.tree.nodes[(node_idx).int()];
    return switch (n.tag) {
        .identifier => b.in.resolutions[(node_idx).int()] == .local,
        .field_access => isLocalRootedPlace(b, n.lhs),
        else => false,
    };
}

/// `if cond { then } [else ...]` as a STATEMENT (value discarded). cond lowers in
/// CONTROL context to a cond_br; the arms run for effect; a `join` block collects
/// fall-through. An else-less if jumps straight to join on false. A divergent arm
/// (already terminated) does not branch to join.
fn lowerIfStmt(b: *Builder, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const stmt = b.in.tree.nodes[(stmt_idx).int()];
    const h = Ast.ifHeaderAt(b.in.tree, (stmt.rhs).int());
    const then_bb = try b.addBlock();
    const join = try b.addBlock();
    const else_bb = if (h.else_node == Ast.none) join else try b.addBlock();

    try genCond(b, stmt.lhs, then_bb, else_bb);

    b.switchTo(then_bb);
    try lowerBlockStmts(b, h.then_block);
    if (!b.termSet()) try brTo(b, join, .none);

    if (h.else_node != Ast.none) {
        b.switchTo(else_bb);
        if (b.in.tree.nodes[(h.else_node).int()].tag == .if_stmt) {
            try lowerIfStmt(b, h.else_node); // else if
        } else {
            try lowerBlockStmts(b, h.else_node);
        }
        if (!b.termSet()) try brTo(b, join, .none);
    }

    b.switchTo(join);
}

/// `if cond { then } else { else }` as a VALUE: both arms produce the merge value
/// carried as a block param on `join`. Typecheck guarantees an else.
fn lowerIfValue(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const stmt = b.in.tree.nodes[(node_idx).int()];
    const h = Ast.ifHeaderAt(b.in.tree, (stmt.rhs).int());
    std.debug.assert(h.else_node != Ast.none);

    const then_bb = try b.addBlock();
    const else_bb = try b.addBlock();
    const join = try b.addBlock();
    const merge = try b.addParam(join, ty);

    try genCond(b, stmt.lhs, then_bb, else_bb);

    b.switchTo(then_bb);
    const tv = try lowerBlockValue(b, h.then_block, ty);
    if (!b.termSet()) try brTo(b, join, tv);

    b.switchTo(else_bb);
    const ev = if (b.in.tree.nodes[(h.else_node).int()].tag == .if_stmt)
        try lowerIfValue(b, h.else_node, ty)
    else
        try lowerBlockValue(b, h.else_node, ty);
    if (!b.termSet()) try brTo(b, join, ev);

    b.switchTo(join);
    return paramOperand(ty, merge);
}

/// A `{ ... }` block in VALUE position: non-final stmts for effect, the trailing
/// expression is the block's value. A block with no trailing expression yields
/// unit. The caller switches to the produced block after this returns.
/// If `stmt_idx` is a value-producing trailing statement, return the expression
/// node to lower in value context; otherwise `null` (a true statement: a
/// `return`, a `while`/`for`, etc.). An `expr_stmt` unwraps to its inner expr.
/// A bare block-like construct (`if`/`match`/`loop`/`labeled`/`block`) is NOT
/// wrapped in an `expr_stmt` by the parser when it appears trailing, but it
/// still yields the block's value, so recognize it by tag and lower it directly.
fn trailingValueExpr(b: *Builder, stmt_idx: Ast.Index) ?Ast.Index {
    const tag = b.in.tree.nodes[(stmt_idx).int()].tag;
    return switch (tag) {
        .expr_stmt => b.in.tree.nodes[(stmt_idx).int()].lhs,
        .if_stmt, .match_expr, .loop_expr, .labeled, .block => stmt_idx,
        else => null,
    };
}

fn lowerBlockValue(b: *Builder, block_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const stmts = Ast.rangeSlice(b.in.tree, (b.in.tree.nodes[(block_idx).int()].lhs).int());
    if (stmts.len == 0) return .none;
    for (stmts[0 .. stmts.len - 1]) |s| {
        if (b.termSet()) return .none;
        try lowerStmt(b, s);
    }
    if (b.termSet()) return .none;
    const last = stmts[stmts.len - 1];
    if (trailingValueExpr(b, last)) |expr| {
        return try lowerExpr(b, expr);
    }
    _ = ty;
    try lowerStmt(b, last); // a trailing stmt (e.g. return / while) → unit value
    return .none;
}

/// `while cond { body }`: header (cond_br to body|done); body; back-edge to
/// header. break→done, continue→header. A `()` statement.
fn lowerWhile(b: *Builder, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
    const stmt = b.in.tree.nodes[(stmt_idx).int()];
    const header = try b.addBlock();
    const body = try b.addBlock();
    const done = try b.addBlock();

    try brTo(b, header, .none);
    b.switchTo(header);
    try genCond(b, stmt.lhs, body, done);

    b.switchTo(body);
    try b.loops.append(b.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .break_bb = done, .continue_bb = header, .merge = .none });
    try lowerBlockStmts(b, stmt.rhs);
    _ = b.loops.pop();
    if (!b.termSet()) try brTo(b, header, .none); // back-edge

    b.switchTo(done);
}

/// `for i in lo..hi { body }` (half-open). `i` is a slot (the for_stmt's `.local`,
/// int). init store lo; header: re-eval hi, load i, `i >= hi` cond_br done|body;
/// body; inc: i = i+1, br header. continue→inc, break→done.
fn lowerFor(b: *Builder, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
    const stmt = b.in.tree.nodes[(stmt_idx).int()];
    const h = Ast.forHeaderAt(b.in.tree, (stmt.rhs).int());
    const islot = try localSlot(b, stmt_idx, Typecheck.Type.int);

    // i := lo
    const lo = operandValue(try lowerExpr(b, h.lo));
    {
        const addr = try b.emit(.{ .slot_addr = islot }, Typecheck.Type.int);
        _ = try b.emit(.{ .store = .{ .addr = addr, .val = lo, .ty = Typecheck.Type.int } }, null);
    }

    const header = try b.addBlock();
    const body = try b.addBlock();
    const inc = try b.addBlock();
    const done = try b.addBlock();

    try brTo(b, header, .none);
    b.switchTo(header);
    const hi = operandValue(try lowerExpr(b, h.hi)); // re-eval each iteration
    const iaddr = try b.emit(.{ .slot_addr = islot }, Typecheck.Type.int);
    const iv = try b.emit(.{ .load = .{ .addr = iaddr, .ty = Typecheck.Type.int } }, Typecheck.Type.int);
    const cmp = try b.emit(.{ .icmp = .{ .cc = .ge, .lhs = iv, .rhs = hi } }, Typecheck.Type.@"bool");
    b.setTerm(.{ .cond_br = .{ .cond = cmp, .t = done, .f = body } });

    b.switchTo(body);
    try b.loops.append(b.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .break_bb = done, .continue_bb = inc, .merge = .none });
    try lowerBlockStmts(b, stmt.lhs);
    _ = b.loops.pop();
    if (!b.termSet()) try brTo(b, inc, .none);

    b.switchTo(inc);
    {
        const a = try b.emit(.{ .slot_addr = islot }, Typecheck.Type.int);
        const cur = try b.emit(.{ .load = .{ .addr = a, .ty = Typecheck.Type.int } }, Typecheck.Type.int);
        const one = try b.emit(.{ .iconst = 1 }, Typecheck.Type.int);
        const next = try b.emit(.{ .add = .{ .lhs = cur, .rhs = one } }, Typecheck.Type.int);
        const a2 = try b.emit(.{ .slot_addr = islot }, Typecheck.Type.int);
        _ = try b.emit(.{ .store = .{ .addr = a2, .val = next, .ty = Typecheck.Type.int } }, null);
    }
    try brTo(b, header, .none);

    b.switchTo(done);
}

/// VALUE `loop { body }`: infinite loop yielding via `break <expr>`. `top` is the
/// back-edge target; `exit` carries the merge param. A break-less (`never`) loop
/// never reaches `exit` → its body just back-edges; the exit stays unreachable.
fn lowerLoopValue(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type, label: ?[]const u8) error{OutOfMemory}!Ir.Operand {
    const n = b.in.tree.nodes[(node_idx).int()];
    const top = try b.addBlock();
    const exit = try b.addBlock();
    const is_value = ty.kind != .unit and ty.kind != .never;
    const merge: Ir.ValueId = if (is_value) try b.addParam(exit, ty) else Ir.none_value;

    try brTo(b, top, .none);
    b.switchTo(top);
    try b.loops.append(b.gpa, .{ .kind = .loop, .label = label, .construct_node = node_idx, .break_bb = exit, .continue_bb = top, .merge = if (is_value) .scalar else .none });
    try lowerBlockStmts(b, n.lhs);
    _ = b.loops.pop();
    if (!b.termSet()) try brTo(b, top, .none); // back-edge

    b.switchTo(exit);
    return paramOperand(ty, merge);
}

/// A `labeled` wrapper as a STATEMENT: dispatch to the inner construct, threading
/// the label. loop/block are value-yielding but discarded here.
fn lowerLabeledStmt(b: *Builder, node_idx: Ast.Index) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(node_idx).int()];
    const label = b.in.tokens[n.main_token].text(b.in.source);
    const inner = b.in.tree.nodes[(n.lhs).int()];
    switch (inner.tag) {
        .while_stmt => try lowerWhile(b, n.lhs, label),
        .for_stmt => try lowerFor(b, n.lhs, label),
        .loop_expr => _ = try lowerLoopValue(b, n.lhs, b.in.node_types[(n.lhs).int()], label),
        .block => _ = try lowerLabeledBlock(b, n.lhs, b.in.node_types[(node_idx).int()], label),
        else => try b.note(n.main_token, "labeled construct unsupported in lower"),
    }
}

/// A `labeled` wrapper as a VALUE expression.
fn lowerLabeledValue(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const n = b.in.tree.nodes[(node_idx).int()];
    const label = b.in.tokens[n.main_token].text(b.in.source);
    const inner = b.in.tree.nodes[(n.lhs).int()];
    switch (inner.tag) {
        .loop_expr => return try lowerLoopValue(b, n.lhs, ty, label),
        .block => return try lowerLabeledBlock(b, n.lhs, ty, label),
        .while_stmt => {
            try lowerWhile(b, n.lhs, label);
            return .none;
        },
        .for_stmt => {
            try lowerFor(b, n.lhs, label);
            return .none;
        },
        else => {
            try b.note(n.main_token, "labeled construct unsupported in lower");
            return .none;
        },
    }
}

/// VALUE labeled BARE BLOCK: like a loop MINUS the back-edge PLUS a trailing-value
/// store. `break @L <expr>` and the trailing expression both branch to `exit` with
/// the value. The exit's merge param is the block's value.
fn lowerLabeledBlock(b: *Builder, block_idx: Ast.Index, ty: Typecheck.Type, label: []const u8) error{OutOfMemory}!Ir.Operand {
    const exit = try b.addBlock();
    const is_value = ty.kind != .unit and ty.kind != .never;
    const merge: Ir.ValueId = if (is_value) try b.addParam(exit, ty) else Ir.none_value;

    try b.loops.append(b.gpa, .{ .kind = .labeled_block, .label = label, .construct_node = block_idx, .break_bb = exit, .continue_bb = Ir.none_block, .merge = if (is_value) .scalar else .none });
    const tv = try lowerBlockValue(b, block_idx, ty);
    _ = b.loops.pop();
    if (!b.termSet()) try brTo(b, exit, tv); // fall-through value

    b.switchTo(exit);
    return paramOperand(ty, merge);
}

// These mirror the value-yielding control-flow lowerings but write each arm's
// trailing value into a caller-supplied destination ptr (an aggregate sink: a var
// slot, a field_addr, or an outer join slot), so a struct/enum/str result works in
// every sink without a value-then-copy. A divergent arm sets its own terminator
// and does NOT branch to join.

fn lowerIfValueInto(b: *Builder, node_idx: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const stmt = b.in.tree.nodes[(node_idx).int()];
    const h = Ast.ifHeaderAt(b.in.tree, (stmt.rhs).int());
    std.debug.assert(h.else_node != Ast.none);
    const then_bb = try b.addBlock();
    const else_bb = try b.addBlock();
    const join = try b.addBlock();

    try genCond(b, stmt.lhs, then_bb, else_bb);

    b.switchTo(then_bb);
    try lowerBlockValueInto(b, h.then_block, dst_ptr, ty);
    if (!b.termSet()) try brTo(b, join, .none);

    b.switchTo(else_bb);
    if (b.in.tree.nodes[(h.else_node).int()].tag == .if_stmt)
        try lowerIfValueInto(b, h.else_node, dst_ptr, ty)
    else
        try lowerBlockValueInto(b, h.else_node, dst_ptr, ty);
    if (!b.termSet()) try brTo(b, join, .none);

    b.switchTo(join);
}

fn lowerBlockValueInto(b: *Builder, block_idx: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const stmts = Ast.rangeSlice(b.in.tree, (b.in.tree.nodes[(block_idx).int()].lhs).int());
    if (stmts.len == 0) return;
    for (stmts[0 .. stmts.len - 1]) |s| {
        if (b.termSet()) return;
        try lowerStmt(b, s);
    }
    if (b.termSet()) return;
    const last = stmts[stmts.len - 1];
    if (trailingValueExpr(b, last)) |expr| {
        try lowerExprInto(b, expr, dst_ptr, ty);
    } else {
        try lowerStmt(b, last); // a trailing stmt (return/divergent) → no value
    }
}

fn lowerLoopValueInto(b: *Builder, node_idx: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type, label: ?[]const u8) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(node_idx).int()];
    const top = try b.addBlock();
    const exit = try b.addBlock();

    try brTo(b, top, .none);
    b.switchTo(top);
    try b.loops.append(b.gpa, .{ .kind = .loop, .label = label, .construct_node = node_idx, .break_bb = exit, .continue_bb = top, .merge = .{ .aggregate = .{ .ptr = dst_ptr, .ty = ty } } });
    try lowerBlockStmts(b, n.lhs);
    _ = b.loops.pop();
    if (!b.termSet()) try brTo(b, top, .none); // back-edge

    b.switchTo(exit);
}

fn lowerLabeledValueInto(b: *Builder, node_idx: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(node_idx).int()];
    const label = b.in.tokens[n.main_token].text(b.in.source);
    const inner = b.in.tree.nodes[(n.lhs).int()];
    switch (inner.tag) {
        .loop_expr => try lowerLoopValueInto(b, n.lhs, dst_ptr, ty, label),
        .block => try lowerLabeledBlockInto(b, n.lhs, dst_ptr, ty, label),
        else => try b.note(n.main_token, "labeled aggregate construct unsupported in lower"),
    }
}

fn lowerLabeledBlockInto(b: *Builder, block_idx: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type, label: []const u8) error{OutOfMemory}!void {
    const exit = try b.addBlock();
    try b.loops.append(b.gpa, .{ .kind = .labeled_block, .label = label, .construct_node = block_idx, .break_bb = exit, .continue_bb = Ir.none_block, .merge = .{ .aggregate = .{ .ptr = dst_ptr, .ty = ty } } });
    try lowerBlockValueInto(b, block_idx, dst_ptr, ty);
    _ = b.loops.pop();
    if (!b.termSet()) try brTo(b, exit, .none);
    b.switchTo(exit);
}

fn lowerBreak(b: *Builder, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const stmt = b.in.tree.nodes[(stmt_idx).int()];
    const ctx = targetLoop(b, stmt_idx);
    switch (ctx.merge) {
        .aggregate => |agg| if (stmt.lhs != Ast.none) {
            // Produce the value INTO the destination ptr, then branch argless to the
            // exit (which the producer already switched away from — the exit reads
            // the destination, no block-arg merge).
            try lowerExprInto(b, stmt.lhs, agg.ptr, agg.ty);
            try brTo(b, ctx.break_bb, .none);
        } else try brTo(b, ctx.break_bb, .none),
        .scalar => if (stmt.lhs != Ast.none) {
            const v = try lowerExpr(b, stmt.lhs);
            try brTo(b, ctx.break_bb, v);
        } else try brTo(b, ctx.break_bb, .none),
        .none => {
            if (stmt.lhs != Ast.none) _ = try lowerExpr(b, stmt.lhs); // for effect, discard
            try brTo(b, ctx.break_bb, .none);
        },
    }
}

fn lowerContinue(b: *Builder, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const ctx = targetLoop(b, stmt_idx);
    try brTo(b, ctx.continue_bb, .none); // continue carries no args
}

/// Select the loop/labeled context a break/continue targets: a labeled one finds
/// the NAMED context by construct node; a bare one finds the innermost LOOP
/// (skipping labeled bare blocks). Typecheck guarantees a match exists.
fn targetLoop(b: *Builder, stmt_idx: Ast.Index) LoopCtx {
    const items = b.loops.items;
    if (b.in.resolutions[(stmt_idx).int()] == .label) {
        const target = b.in.resolutions[(stmt_idx).int()].label;
        var i = items.len;
        while (i > 0) {
            i -= 1;
            if (items[i].construct_node == target) return items[i];
        }
        unreachable; // typecheck-guaranteed
    }
    var i = items.len;
    while (i > 0) {
        i -= 1;
        if (items[i].kind != .labeled_block) return items[i];
    }
    unreachable; // typecheck-guaranteed
}

/// Lower a bool expression in CONTROL context: emit a `cond_br` (or a chain) so
/// control reaches `true_bb` when the expression is true and `false_bb` when
/// false. Comparisons fold the `cc` into an icmp + cond_br; `!a` swaps the dests;
/// `&&`/`||` chain through fresh intermediate blocks; a bare bool cond_brs on its
/// value. cond_br is argless by design (the value-merge edges are arranged by the
/// if/loop value paths, which `brTo` with the merge operand).
fn genCond(b: *Builder, node_idx: Ast.Index, true_bb: Ir.BlockId, false_bb: Ir.BlockId) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(node_idx).int()];
    switch (n.tag) {
        .binary => {
            const op = b.in.tokens[n.main_token].tag;
            switch (op) {
                .lt, .lt_eq, .gt, .gt_eq, .eq_eq, .bang_eq => {
                    const lhs = operandValue(try lowerExpr(b, n.lhs));
                    const rhs = operandValue(try lowerExpr(b, n.rhs));
                    const cc = condFromToken(op);
                    const c = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.@"bool");
                    b.setTerm(.{ .cond_br = .{ .cond = c, .t = true_bb, .f = false_bb } });
                },
                .amp_amp => {
                    // a-true → eval b; a-false → false_bb.
                    const mid = try b.addBlock();
                    try genCond(b, n.lhs, mid, false_bb);
                    b.switchTo(mid);
                    try genCond(b, n.rhs, true_bb, false_bb);
                },
                .pipe_pipe => {
                    // a-true → true_bb; a-false → eval b.
                    const mid = try b.addBlock();
                    try genCond(b, n.lhs, true_bb, mid);
                    b.switchTo(mid);
                    try genCond(b, n.rhs, true_bb, false_bb);
                },
                else => try genCondBareBool(b, node_idx, true_bb, false_bb),
            }
        },
        .unary => {
            if (b.in.tokens[n.main_token].tag == .bang) {
                try genCond(b, n.lhs, false_bb, true_bb); // SWAP dests
            } else {
                try genCondBareBool(b, node_idx, true_bb, false_bb);
            }
        },
        else => try genCondBareBool(b, node_idx, true_bb, false_bb),
    }
}

fn genCondBareBool(b: *Builder, node_idx: Ast.Index, true_bb: Ir.BlockId, false_bb: Ir.BlockId) error{OutOfMemory}!void {
    const v = operandValue(try lowerExpr(b, node_idx));
    b.setTerm(.{ .cond_br = .{ .cond = v, .t = true_bb, .f = false_bb } });
}

/// Branch the current block to `dest`, carrying `arg` as the single merge edge
/// value (or no args for a `.none` arg). Sets the terminator (guarded).
fn brTo(b: *Builder, dest: Ir.BlockId, arg: Ir.Operand) error{OutOfMemory}!void {
    if (b.termSet()) return;
    const args: []Ir.Operand = switch (arg) {
        .none => &.{},
        else => blk: {
            const a = try b.gpa.alloc(Ir.Operand, 1);
            a[0] = arg;
            break :blk a;
        },
    };
    b.setTerm(.{ .br = .{ .dest = dest, .args = args } });
}

/// The Operand that reads a join's merge param: `.none` for a unit merge (the
/// param value is `none_value`), else `.value` of the param.
fn paramOperand(ty: Typecheck.Type, merge: Ir.ValueId) Ir.Operand {
    _ = ty;
    if (merge == Ir.none_value) return .none;
    return .{ .value = merge };
}

/// Extract the `value` of an Operand for an op that needs a scalar value id. A
/// `.none`/`.slot` here is a lowering bug for scalar context; default to
/// `none_value` so we never index garbage (a diagnostic was already emitted).
fn operandValue(op: Ir.Operand) Ir.ValueId {
    return switch (op) {
        .value => |v| v,
        else => Ir.none_value,
    };
}

/// Get (or lazily bind) the IR slot for the local that `node_idx` resolves to.
/// Param slots are pre-bound in `lowerFn`; a `:=` local / for-var is bound here
/// the first time it is touched. Allocating in touch order keeps slot ids
/// deterministic for a given source.
fn localSlot(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.SlotId {
    const res = b.in.resolutions[(node_idx).int()];
    std.debug.assert(res == .local);
    const li = res.local;
    if (b.local_slots.get(li)) |sid| return sid;
    const sid = try b.addSlot(ty);
    try b.local_slots.put(b.gpa, li, sid);
    return sid;
}

/// Resolve a function's declared return type. No ret-type node (or `()`) is unit;
/// named refs map to scalars or to a struct/enum layout by name.
fn returnType(in: Inputs, proto: Ast.FnProto) Typecheck.Type {
    if (proto.ret_type == Ast.none) return Typecheck.Type.unit;
    if (in.tree.nodes[(proto.ret_type).int()].tag == .literal_unit) return Typecheck.Type.unit;
    // Prefer the typecheck-resolved sig for EVERY kind: it carries the GLOBAL
    // struct/enum id (incl. a cross-module qualified `mod.Type`) so the ABI decision
    // is taken on the correct layout, AND — for a monomorphized instance (M2) — it
    // is the SUBSTITUTED concrete type, so a ret spelled `T` (whose token would
    // otherwise fall to `typeFromRef`'s `int` default, miscompiling `id[bool]`/
    // `id[str]`) is resolved correctly. Byte-identical for a non-generic fn, whose
    // `s.ret` equals what `typeFromRef` would compute for the concrete spelling.
    if (in.sig) |s| {
        std.debug.assert(!s.ret.isTypeVar() and !s.ret.isApp()); // reified-away before lower (M2/M4)
        return s.ret;
    }
    return typeFromRef(in, proto.ret_type);
}

/// A `param` node's type: prefer the type checker's `node_types`, falling back to
/// its type-ref token spelling.
fn paramType(in: Inputs, proto: Ast.FnProto, slot: u32) Typecheck.Type {
    const param_node = proto.params[slot];
    // Prefer the typecheck-resolved sig for EVERY kind (see `returnType`): correct
    // GLOBAL id for aggregates AND the SUBSTITUTED concrete type for an instance's
    // `T`-spelled param, so `id[bool]`/`id[str]` are not lost to the `int` default.
    if (in.sig) |s| if (slot < s.params.len) {
        std.debug.assert(!s.params[slot].isTypeVar() and !s.params[slot].isApp()); // reified-away before lower (M2/M4)
        return s.params[slot];
    };
    if (param_node.int() < in.node_types.len) {
        const t = in.node_types[(param_node).int()];
        if (t.kind != .invalid) return t;
    }
    const pn = in.tree.nodes[(param_node).int()];
    if (pn.lhs != Ast.none) return typeFromRef(in, pn.lhs);
    return Typecheck.Type.int;
}

/// Map a type-ref node's token spelling to a `Type`.
fn typeFromRef(in: Inputs, ref: Ast.Index) Typecheck.Type {
    const name = in.tokens[in.tree.nodes[(ref).int()].main_token].text(in.source);
    if (std.mem.eql(u8, name, "str")) return Typecheck.Type.str;
    if (std.mem.eql(u8, name, "bool")) return Typecheck.Type.@"bool";
    if (std.mem.eql(u8, name, "int")) return Typecheck.Type.int;
    for (in.layouts, 0..) |l, id| {
        if (std.mem.eql(u8, l.name, name)) return Typecheck.Type.structT(@intCast(id));
    }
    for (in.enum_layouts, 0..) |e, id| {
        if (std.mem.eql(u8, e.name, name)) return Typecheck.Type.enumT(@intCast(id));
    }
    // A generic-param spelling `T` never reaches here on the lowered path: the
    // instance's sig (preferred above) already carries the concrete substitution.
    return Typecheck.Type.int;
}

/// Map a comparison token to its signed IR condition.
fn condFromToken(tag: TokenTag) Ir.Cond {
    return switch (tag) {
        .lt => .lt,
        .lt_eq => .le,
        .gt => .gt,
        .gt_eq => .ge,
        .eq_eq => .eq,
        .bang_eq => .ne,
        else => unreachable,
    };
}

/// Seed for the literal content hash. FROZEN: changing it re-hashes every literal,
/// so the `.cstr` reloc targets (the hashes) in existing cached FnCode blobs would
/// no longer line up with the literal table — cached blobs rot.
const lit_seed: u64 = 0x10c5_7e87;

/// Parse a number-literal token text into i64, stripping `_` separators. Returns
/// null if it does not fit i64 (range-checked). The
/// asymmetry vs the NON-range-checked pattern-literal parse is preserved for a
/// later match stage.
fn parseInt(raw: []const u8) ?i64 {
    var buf: [24]u8 = undefined;
    var n: usize = 0;
    for (raw) |c| {
        if (c == '_') continue;
        if (n >= buf.len) return null;
        buf[n] = c;
        n += 1;
    }
    return std.fmt.parseInt(i64, buf[0..n], 10) catch null;
}

/// Parse a (possibly `_`-separated) decimal int literal as written in source, for
/// a PATTERN literal. NON-range-checked — the
/// asymmetry vs the range-checked `parseInt` for `literal_number` is intentional.
fn parseIntLit(text: []const u8) i64 {
    var v: i64 = 0;
    for (text) |c| {
        if (c == '_') continue;
        v = v * 10 + @as(i64, c - '0');
    }
    return v;
}

/// Decode a string-literal token into its runtime bytes (quotes stripped, escapes
/// `\n \t \\ \"` decoded). Returns null + a diagnostic on a malformed token /
/// unknown escape. Caller owns the returned slice. Mirrors
/// Codegen.decodeStringLiteral byte-for-byte so the content hash matches.
fn decodeStringLiteral(b: *Builder, tok: u32) error{OutOfMemory}!?[]u8 {
    const raw = b.in.tokens[tok].text(b.in.source);
    if (raw.len < 2 or raw[0] != '"' or raw[raw.len - 1] != '"') {
        try b.note(tok, "malformed string literal");
        return null;
    }
    const body = raw[1 .. raw.len - 1];
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(b.gpa);
    var i: usize = 0;
    while (i < body.len) : (i += 1) {
        const c = body[i];
        if (c != '\\') {
            try out.append(b.gpa, c);
            continue;
        }
        i += 1;
        if (i >= body.len) {
            out.deinit(b.gpa);
            try b.note(tok, "string literal ends with a dangling backslash");
            return null;
        }
        const decoded: u8 = switch (body[i]) {
            'n' => 0x0A,
            't' => 0x09,
            '\\' => '\\',
            '"' => '"',
            else => {
                out.deinit(b.gpa);
                try b.note(tok, "unknown escape in string literal");
                return null;
            },
        };
        try out.append(b.gpa, decoded);
    }
    return try out.toOwnedSlice(b.gpa);
}

const testing = std.testing;
const Graph = @import("driver/Graph.zig");
const ResolveGraph = @import("resolve_graph.zig");
const TypecheckGraph = @import("types_graph.zig");

/// The front-end results a lower test feeds into `Inputs`: the whole-graph
/// resolve + typecheck results stored WHOLE (a one-module graph — the ONE
/// front-end). The resolve result owns the fn-name strings the typecheck
/// `sigs[].name` borrow; both are torn down together. The node-parallel reads are
/// the entry module: `resolve.resolutions[0]` / `typecheck.node_types[0]`;
/// `.layouts`/`.enum_layouts` are program-global.
const FrontEnd = struct {
    resolve: ResolveGraph.GraphResult,
    typecheck: Typecheck.GraphResult,

    fn deinit(self: *FrontEnd, gpa: std.mem.Allocator) void {
        self.typecheck.deinit(gpa);
        self.resolve.deinit(gpa);
    }
};

/// Resolve + typecheck `src` (already lexed/parsed) as a one-module graph. The
/// graph BORROWS `tokens`/`tree`/`src` (the caller keeps them alive past the
/// returned `FrontEnd`), so it is torn down immediately; the results own their
/// arrays. `io = null` => serial Pass-C (deterministic).
fn frontEnd(gpa: std.mem.Allocator, tokens: []const Token, tree: Ast.Tree, src: []const u8) !FrontEnd {
    var g = try Graph.single(gpa, "main", "", src, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinit(gpa);
    var res = try ResolveGraph.resolveGraph(gpa, &g);
    errdefer res.deinit(gpa);
    const tc = try TypecheckGraph.checkGraph(gpa, &g, &res, null, 0);
    return .{ .resolve = res, .typecheck = tc };
}

/// Parse + resolve + typecheck `src`, lower the named fn, render it, and compare
/// the rendered IR text to `want`. A focused integration harness for lower-core.
fn expectLowered(src: []const u8, fn_name: []const u8, want: []const u8) !void {
    const gpa = testing.allocator;
    const Lexer = @import("lex.zig");
    const Parser = @import("parse.zig");

    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }

    var fe = try frontEnd(gpa, tokens, .{ .nodes = tree.nodes, .extra = tree.extra }, src);
    defer fe.deinit(gpa);
    const rr = fe.resolve;
    const tc = fe.typecheck;

    const prog = tree.nodes[(Ast.root(tree.nodes)).int()];
    var fn_nodes: std.ArrayList(Ast.Index) = .empty;
    defer fn_nodes.deinit(gpa);
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, (prog.lhs).int())) |idx| {
        if (tree.nodes[(idx).int()].tag == .fn_decl) try fn_nodes.append(gpa, idx);
    }

    const names = try gpa.alloc(Link.SymName, fn_nodes.items.len + 1);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (fn_nodes.items, 0..) |idx, i| {
        const nm = tokens[tree.nodes[(idx).int()].main_token].text(src);
        names[i] = .{ .kind = .user_fn, .name = try gpa.dupe(u8, nm) };
    }
    names[fn_nodes.items.len] = .{ .kind = .builtin, .name = try gpa.dupe(u8, "print") };

    const in = Inputs{
        .tree = .{ .nodes = tree.nodes, .extra = tree.extra },
        .tokens = tokens,
        .source = src,
        .resolutions = rr.resolutions[0],
        .node_types = tc.node_types[0],
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = names,
        .methods = tc.methods,
    };

    var target: Ast.Index = Ast.none;
    var target_i: usize = 0;
    for (fn_nodes.items, 0..) |idx, i| {
        if (std.mem.eql(u8, tokens[tree.nodes[(idx).int()].main_token].text(src), fn_name)) {
            target = idx;
            target_i = i;
            break;
        }
    }
    try testing.expect(target != Ast.none);

    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const is_entry = std.mem.eql(u8, fn_name, "main");
    var func = try lowerFn(gpa, in, target, names[target_i], is_entry, &diags);
    defer func.deinit(gpa);

    var buf: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try Ir.render(&w, &func, in.layouts, in.enum_layouts);
    try testing.expectEqualStrings(want, w.buffered());
}

/// Like `expectLowered`, but resolves fns through the GRAPH-GLOBAL fn table
/// (`res.fns`) so a method (declared inside an `impl`, absent from the program's
/// direct children) — and a fn that CALLS one — lower with `names`/`sig` indexed by
/// global fn id. Mirrors `Codegen.renderGraphIr`'s single-module path. The M9
/// mut-self lowering lives on the method path, which the top-level `expectLowered`
/// cannot reach.
fn expectLoweredG(src: []const u8, fn_name: []const u8, want: []const u8) !void {
    const gpa = testing.allocator;
    const Lexer = @import("lex.zig");
    const Parser = @import("parse.zig");

    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }

    var fe = try frontEnd(gpa, tokens, .{ .nodes = tree.nodes, .extra = tree.extra }, src);
    defer fe.deinit(gpa);
    const rr = fe.resolve;
    const tc = fe.typecheck;

    // Build `names` in global fn-id order (the space method dispatch/`sig` use).
    const names = try gpa.alloc(Link.SymName, rr.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (rr.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

    // Match `fn_name` against the LAST dot-segment: a top-level fn is named bare
    // (`main`), a method is mangled (`main.P.bump` → matched by `bump`).
    var target_gid: ?usize = null;
    for (rr.fns, 0..) |gf, i| {
        if (gf.decl_node == Ast.none) continue;
        const last = if (std.mem.lastIndexOfScalar(u8, gf.name, '.')) |dot| gf.name[dot + 1 ..] else gf.name;
        if (std.mem.eql(u8, last, fn_name)) {
            target_gid = i;
            break;
        }
    }
    const gid = target_gid orelse return error.TestUnexpectedResult;

    const in = Inputs{
        .tree = .{ .nodes = tree.nodes, .extra = tree.extra },
        .tokens = tokens,
        .source = src,
        .resolutions = rr.resolutions[0],
        .node_types = tc.node_types[0],
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = names,
        .sig = if (gid < tc.sigs.len) tc.sigs[gid] else null,
        .instances = tc.instances,
        .sigs = tc.sigs,
        .methods = tc.methods,
    };

    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    const is_entry = std.mem.eql(u8, fn_name, "main");
    var func = try lowerFn(gpa, in, rr.fns[gid].decl_node, names[gid], is_entry, &diags);
    defer func.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);

    var buf: [8192]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try Ir.render(&w, &func, in.layouts, in.enum_layouts);
    try testing.expectEqualStrings(want, w.buffered());
}

test "lower-core: arithmetic return" {
    try expectLowered(
        "fn f() -> int { return 7 + 3 }\n",
        "f",
        "fn f() -> int {\n" ++
            "  slots:\n" ++
            "b0:\n" ++
            "  %1 = iconst 7\n" ++
            "  %2 = iconst 3\n" ++
            "  %3 = add %1, %2\n" ++
            "  br b1(%3)\n" ++
            "b1(%0:int):\n" ++
            "  ret %0\n" ++
            "}\n",
    );
}

test "lower-core: var_decl + identifier load/store" {
    try expectLowered(
        "fn f() -> int { x := 5\n return x }\n",
        "f",
        "fn f() -> int {\n" ++
            "  slots: s0:int\n" ++
            "b0:\n" ++
            "  %1 = iconst 5\n" ++
            "  %2 = slot_addr s0\n" ++
            "  store %2, %1 : int\n" ++
            "  %3 = slot_addr s0\n" ++
            "  %4 = load %3 : int\n" ++
            "  br b1(%4)\n" ++
            "b1(%0:int):\n" ++
            "  ret %0\n" ++
            "}\n",
    );
}

test "lower-core: while loop with break/continue is well-formed" {
    const gpa = testing.allocator;
    const Lexer = @import("lex.zig");
    const Parser = @import("parse.zig");
    const src =
        "fn main() -> int {\n" ++
        "  i := 0\n  s := 0\n" ++
        "  while i < 5 { i = i + 1\n if i == 2 { continue }\n s = s + i }\n" ++
        "  return s\n}\n";
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    var fe = try frontEnd(gpa, tokens, .{ .nodes = tree.nodes, .extra = tree.extra }, src);
    defer fe.deinit(gpa);
    const rr = fe.resolve;
    const tc = fe.typecheck;

    const prog = tree.nodes[(Ast.root(tree.nodes)).int()];
    var fn_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, (prog.lhs).int())) |idx| {
        if (tree.nodes[(idx).int()].tag == .fn_decl) fn_decl = idx;
    }
    var names = [_]Link.SymName{.{ .kind = .user_fn, .name = try gpa.dupe(u8, "main") }};
    defer gpa.free(names[0].name);
    const in = Inputs{
        .tree = .{ .nodes = tree.nodes, .extra = tree.extra },
        .tokens = tokens,
        .source = src,
        .resolutions = rr.resolutions[0],
        .node_types = tc.node_types[0],
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = &names,
        .methods = tc.methods,
    };
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    var func = try lowerFn(gpa, in, fn_decl, names[0], true, &diags);
    defer func.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    try testing.expect(func.blocks.len >= 5); // entry + exit + header/body/done
}

test "lower-core: out-of-range int literal yields a diagnostic" {
    const gpa = testing.allocator;
    const Lexer = @import("lex.zig");
    const Parser = @import("parse.zig");
    const src = "fn f() -> int { return 99999999999999999999 }\n";
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    var fe = try frontEnd(gpa, tokens, .{ .nodes = tree.nodes, .extra = tree.extra }, src);
    defer fe.deinit(gpa);
    const rr = fe.resolve;
    const tc = fe.typecheck;
    const prog = tree.nodes[(Ast.root(tree.nodes)).int()];
    var fn_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, (prog.lhs).int())) |idx| {
        if (tree.nodes[(idx).int()].tag == .fn_decl) fn_decl = idx;
    }
    var names = [_]Link.SymName{.{ .kind = .user_fn, .name = try gpa.dupe(u8, "f") }};
    defer gpa.free(names[0].name);
    const in = Inputs{
        .tree = .{ .nodes = tree.nodes, .extra = tree.extra },
        .tokens = tokens,
        .source = src,
        .resolutions = rr.resolutions[0],
        .node_types = tc.node_types[0],
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = &names,
        .methods = tc.methods,
    };
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    var func = try lowerFn(gpa, in, fn_decl, names[0], false, &diags);
    defer func.deinit(gpa);
    try testing.expect(diags.items.len >= 1);
}

test "lower-aggregates: struct construct + field read" {
    try expectLowered(
        "struct P { x: int, y: int }\nfn f() -> int { p := P { x: 6, y: 7 }\n return p.x }\n",
        "f",
        "fn f() -> int {\n" ++
            "  slots: s0:P\n" ++
            "b0:\n" ++
            "  %1 = slot_addr s0\n" ++
            "  %2 = field_addr %1, 0 : int\n" ++
            "  %3 = iconst 6\n" ++
            "  store %2, %3 : int\n" ++
            "  %4 = field_addr %1, 8 : int\n" ++
            "  %5 = iconst 7\n" ++
            "  store %4, %5 : int\n" ++
            "  %6 = slot_addr s0\n" ++
            "  %7 = field_addr %6, 0 : int\n" ++
            "  %8 = load %7 : int\n" ++
            "  br b1(%8)\n" ++
            "b1(%0:int):\n" ++
            "  ret %0\n" ++
            "}\n",
    );
}

test "lower-aggregates: enum match dispatch is well-formed + leak-clean" {
    const gpa = testing.allocator;
    const Lexer = @import("lex.zig");
    const Parser = @import("parse.zig");
    const src =
        "enum E { C(int), N }\n" ++
        "fn f(e: E) -> int { match e { .C(r) -> r, .N -> 0 } }\n";
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    var fe = try frontEnd(gpa, tokens, .{ .nodes = tree.nodes, .extra = tree.extra }, src);
    defer fe.deinit(gpa);
    const rr = fe.resolve;
    const tc = fe.typecheck;

    const prog = tree.nodes[(Ast.root(tree.nodes)).int()];
    var fn_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, (prog.lhs).int())) |idx| {
        if (tree.nodes[(idx).int()].tag == .fn_decl) fn_decl = idx;
    }
    var names = [_]Link.SymName{.{ .kind = .user_fn, .name = try gpa.dupe(u8, "f") }};
    defer gpa.free(names[0].name);
    const in = Inputs{
        .tree = .{ .nodes = tree.nodes, .extra = tree.extra },
        .tokens = tokens,
        .source = src,
        .resolutions = rr.resolutions[0],
        .node_types = tc.node_types[0],
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = &names,
        .methods = tc.methods,
    };
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    var func = try lowerFn(gpa, in, fn_decl, names[0], false, &diags);
    defer func.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    // entry + exit + per-arm next/body blocks (>= 5).
    try testing.expect(func.blocks.len >= 5);
    // The dispatch reads the tag via get_tag at least once.
    var saw_get_tag = false;
    for (func.blocks) |blk| {
        for (blk.instrs) |ins| if (ins.op == .get_tag) {
            saw_get_tag = true;
        };
    }
    try testing.expect(saw_get_tag);
}

/// Parse + resolve + typecheck `src`, lower the named fn, and assert it produced at
/// least one diagnostic (a `b.note`). The aggregate produce-into-slot family is the
/// most miscompile-prone code here, so these pin the lowering-stage error paths that
/// the whole-program byte-diff cannot reach (the source typechecks, then fails to
/// lower).
fn expectLowerDiag(src: []const u8, fn_name: []const u8) !void {
    const gpa = testing.allocator;
    const Lexer = @import("lex.zig");
    const Parser = @import("parse.zig");

    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    var fe = try frontEnd(gpa, tokens, .{ .nodes = tree.nodes, .extra = tree.extra }, src);
    defer fe.deinit(gpa);
    const rr = fe.resolve;
    const tc = fe.typecheck;
    const prog = tree.nodes[(Ast.root(tree.nodes)).int()];
    var fn_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, (prog.lhs).int())) |idx| {
        if (tree.nodes[(idx).int()].tag == .fn_decl and
            std.mem.eql(u8, tokens[tree.nodes[(idx).int()].main_token].text(src), fn_name)) fn_decl = idx;
    }
    var names = [_]Link.SymName{.{ .kind = .user_fn, .name = try gpa.dupe(u8, fn_name) }};
    defer gpa.free(names[0].name);
    const in = Inputs{
        .tree = .{ .nodes = tree.nodes, .extra = tree.extra },
        .tokens = tokens,
        .source = src,
        .resolutions = rr.resolutions[0],
        .node_types = tc.node_types[0],
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = &names,
        .methods = tc.methods,
    };
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    var func = try lowerFn(gpa, in, fn_decl, names[0], false, &diags);
    defer func.deinit(gpa);
    try testing.expect(diags.items.len >= 1);
}

test "lower-aggregates: str literal produced into a var slot (ptr@0, len@8)" {
    try expectLowered(
        "fn f() {\n s := \"hi\"\n}\n",
        "f",
        "fn f() -> unit {\n" ++
            "  slots: s0:str\n" ++
            "b0:\n" ++
            "  %0 = slot_addr s0\n" ++
            "  %1 = cstr_ptr #c9c8bb6f84f1f9cf\n" ++
            "  store %0, %1 : int\n" ++
            "  %2 = field_addr %0, 8 : int\n" ++
            "  %3 = iconst 2\n" ++
            "  store %2, %3 : int\n" ++
            "  br b1\n" ++
            "b1:\n" ++
            "  ret\n" ++
            "}\n",
    );
}

test "lower-aggregates: nested struct init writes inner fields at nested offsets" {
    try expectLowered(
        "struct Inner { a: int, b: int }\nstruct Outer { p: Inner, q: int }\n" ++
            "fn f() {\n o := Outer { p: Inner { a: 1, b: 2 }, q: 3 }\n}\n",
        "f",
        "fn f() -> unit {\n" ++
            "  slots: s0:Outer\n" ++
            "b0:\n" ++
            "  %0 = slot_addr s0\n" ++
            "  %1 = field_addr %0, 0 : Inner\n" ++
            "  %2 = field_addr %1, 0 : int\n" ++
            "  %3 = iconst 1\n" ++
            "  store %2, %3 : int\n" ++
            "  %4 = field_addr %1, 8 : int\n" ++
            "  %5 = iconst 2\n" ++
            "  store %4, %5 : int\n" ++
            "  %6 = field_addr %0, 16 : int\n" ++
            "  %7 = iconst 3\n" ++
            "  store %6, %7 : int\n" ++
            "  br b1\n" ++
            "b1:\n" ++
            "  ret\n" ++
            "}\n",
    );
}

test "lower-aggregates: tuple-variant enum init stores tag@0, payload@8" {
    try expectLowered(
        "enum E { C(int), N }\nfn f() {\n e := E.C(5)\n}\n",
        "f",
        "fn f() -> unit {\n" ++
            "  slots: s0:E\n" ++
            "b0:\n" ++
            "  %0 = slot_addr s0\n" ++
            "  %1 = iconst 0\n" ++
            "  store %0, %1 : int\n" ++
            "  %2 = field_addr %0, 8 : int\n" ++
            "  %3 = iconst 5\n" ++
            "  store %2, %3 : int\n" ++
            "  br b1\n" ++
            "b1:\n" ++
            "  ret\n" ++
            "}\n",
    );
}

test "lower-aggregates: struct-variant enum init stores tag@0, fields@8/16" {
    try expectLowered(
        "enum E { R { w: int, h: int }, N }\nfn f() {\n e := E.R { w: 3, h: 4 }\n}\n",
        "f",
        "fn f() -> unit {\n" ++
            "  slots: s0:E\n" ++
            "b0:\n" ++
            "  %0 = slot_addr s0\n" ++
            "  %1 = iconst 0\n" ++
            "  store %0, %1 : int\n" ++
            "  %2 = field_addr %0, 8 : int\n" ++
            "  %3 = iconst 3\n" ++
            "  store %2, %3 : int\n" ++
            "  %4 = field_addr %0, 16 : int\n" ++
            "  %5 = iconst 4\n" ++
            "  store %4, %5 : int\n" ++
            "  br b1\n" ++
            "b1:\n" ++
            "  ret\n" ++
            "}\n",
    );
}

test "lower-aggregates: value-if returning a struct produces into each arm slot" {
    try expectLowered(
        "struct V { a: int, b: int }\n" ++
            "fn f(c: int) -> V {\n if c == 1 { V { a: 10, b: 20 } } else { V { a: 1, b: 1 } }\n}\n",
        "f",
        "fn f(s0) -> V {\n" ++
            "  slots: s0:int s1:V s2:V\n" ++
            "b0:\n" ++
            "  %2 = slot_addr s0\n" ++
            "  %3 = load %2 : int\n" ++
            "  %4 = iconst 1\n" ++
            "  %5 = icmp eq %3, %4\n" ++
            "  cond_br %5, b2, b3\n" ++
            "b1(%0:V):\n" ++
            "  ret %0\n" ++
            "b2:\n" ++
            "  %6 = slot_addr s1\n" ++
            "  %7 = field_addr %6, 0 : int\n" ++
            "  %8 = iconst 10\n" ++
            "  store %7, %8 : int\n" ++
            "  %9 = field_addr %6, 8 : int\n" ++
            "  %10 = iconst 20\n" ++
            "  store %9, %10 : int\n" ++
            "  br b4(s1)\n" ++
            "b3:\n" ++
            "  %11 = slot_addr s2\n" ++
            "  %12 = field_addr %11, 0 : int\n" ++
            "  %13 = iconst 1\n" ++
            "  store %12, %13 : int\n" ++
            "  %14 = field_addr %11, 8 : int\n" ++
            "  %15 = iconst 1\n" ++
            "  store %14, %15 : int\n" ++
            "  br b4(s2)\n" ++
            "b4(%1:V):\n" ++
            "  br b1(%1)\n" ++
            "}\n",
    );
}

test "lower-aggregates: value-loop break of a struct produces into the merge slot" {
    try expectLowered(
        "struct V { a: int, b: int }\nfn f() -> V {\n loop { break V { a: 1, b: 2 } }\n}\n",
        "f",
        "fn f() -> V {\n" ++
            "  slots: s0:V\n" ++
            "b0:\n" ++
            "  br b2\n" ++
            "b1(%0:V):\n" ++
            "  ret %0\n" ++
            "b2:\n" ++
            "  %2 = slot_addr s0\n" ++
            "  %3 = field_addr %2, 0 : int\n" ++
            "  %4 = iconst 1\n" ++
            "  store %3, %4 : int\n" ++
            "  %5 = field_addr %2, 8 : int\n" ++
            "  %6 = iconst 2\n" ++
            "  store %5, %6 : int\n" ++
            "  br b3(s0)\n" ++
            "b3(%1:V):\n" ++
            "  br b1(%1)\n" ++
            "}\n",
    );
}

test "lower-aggregates: labeled-block break of a struct produces into the merge slot" {
    try expectLowered(
        "struct V { a: int, b: int }\nfn f() -> V {\n @blk { break @blk V { a: 1, b: 2 } }\n}\n",
        "f",
        "fn f() -> V {\n" ++
            "  slots: s0:V\n" ++
            "b0:\n" ++
            "  %2 = slot_addr s0\n" ++
            "  %3 = field_addr %2, 0 : int\n" ++
            "  %4 = iconst 1\n" ++
            "  store %3, %4 : int\n" ++
            "  %5 = field_addr %2, 8 : int\n" ++
            "  %6 = iconst 2\n" ++
            "  store %5, %6 : int\n" ++
            "  br b2(s0)\n" ++
            "b1(%0:V):\n" ++
            "  ret %0\n" ++
            "b2(%1:V):\n" ++
            "  br b1(%1)\n" ++
            "}\n",
    );
}

test "lower-aggregates: match-into a struct per arm (get_tag dispatch)" {
    try expectLowered(
        "enum E { C(int), N }\nstruct V { a: int }\n" ++
            "fn f(e: E) -> V {\n match e { .C(r) -> V { a: r }, .N -> V { a: 0 } }\n}\n",
        "f",
        "fn f(s0) -> V {\n" ++
            "  slots: s0:E s1:V s2:int\n" ++
            "b0:\n" ++
            "  %1 = slot_addr s1\n" ++
            "  %2 = slot_addr s0\n" ++
            "  %3 = get_tag %2\n" ++
            "  %4 = iconst 0\n" ++
            "  %5 = icmp ne %3, %4\n" ++
            "  cond_br %5, b3, b4\n" ++
            "b1(%0:V):\n" ++
            "  ret %0\n" ++
            "b2:\n" ++
            "  br b1(s1)\n" ++
            "b3:\n" ++
            "  %13 = slot_addr s0\n" ++
            "  %14 = get_tag %13\n" ++
            "  %15 = iconst 1\n" ++
            "  %16 = icmp ne %14, %15\n" ++
            "  cond_br %16, b5, b6\n" ++
            "b4:\n" ++
            "  %6 = slot_addr s0\n" ++
            "  %7 = field_addr %6, 8 : int\n" ++
            "  %8 = slot_addr s2\n" ++
            "  %9 = load %7 : int\n" ++
            "  store %8, %9 : int\n" ++
            "  %10 = field_addr %1, 0 : int\n" ++
            "  %11 = slot_addr s2\n" ++
            "  %12 = load %11 : int\n" ++
            "  store %10, %12 : int\n" ++
            "  br b2\n" ++
            "b5:\n" ++
            "  br b2\n" ++
            "b6:\n" ++
            "  %17 = field_addr %1, 0 : int\n" ++
            "  %18 = iconst 0\n" ++
            "  store %17, %18 : int\n" ++
            "  br b2\n" ++
            "}\n",
    );
}

test "lower-aggregates: aggregate identifier copy-into emits a copy" {
    try expectLowered(
        "struct V { a: int, b: int }\nfn f(p: V) {\n q := p\n}\n",
        "f",
        "fn f(s0) -> unit {\n" ++
            "  slots: s0:V s1:V\n" ++
            "b0:\n" ++
            "  %0 = slot_addr s1\n" ++
            "  %1 = slot_addr s0\n" ++
            "  copy %0 <- %1 : V\n" ++
            "  br b1\n" ++
            "b1:\n" ++
            "  ret\n" ++
            "}\n",
    );
}

test "lower-diagnostics: an unknown string escape fails to lower" {
    try expectLowerDiag("fn f() {\n s := \"\\q\"\n}\n", "f");
}

test "M9: a mut-self method body loads self through the pointer slot (s0:int)" {
    try expectLoweredG(
        "struct P { x: int, y: int }\n" ++
            "impl P { fn bump(mut self, d: int) { self.x = self.x + d } }\n" ++
            "fn main() -> int { p := P{ x: 40, y: 0 }\n p.bump(2)\n return p.x }\n",
        "bump",
        // s0 is the mut-self POINTER slot (int, not P). Every `self.x` read/write
        // loads the pointer (`slot_addr s0; load`) then a `field_addr`, so the store
        // lands in the caller's storage.
        "fn main.P.bump(s0, s1) -> unit {\n" ++
            "  slots: s0:int s1:int\n" ++
            "b0:\n" ++
            "  %0 = slot_addr s0\n" ++
            "  %1 = load %0 : int\n" ++
            "  %2 = field_addr %1, 0 : int\n" ++
            "  %3 = slot_addr s0\n" ++
            "  %4 = load %3 : int\n" ++
            "  %5 = field_addr %4, 0 : int\n" ++
            "  %6 = load %5 : int\n" ++
            "  %7 = slot_addr s1\n" ++
            "  %8 = load %7 : int\n" ++
            "  %9 = add %6, %8\n" ++
            "  store %2, %9 : int\n" ++
            "  br b1\n" ++
            "b1:\n" ++
            "  ret\n" ++
            "}\n",
    );
}

test "M9: the caller passes the receiver place ADDRESS as arg 0 (a scalar value)" {
    try expectLoweredG(
        "struct P { x: int, y: int }\n" ++
            "impl P { fn bump(mut self, d: int) { self.x = self.x + d } }\n" ++
            "fn main() -> int { p := P{ x: 40, y: 0 }\n p.bump(2)\n return p.x }\n",
        "main",
        "fn main() -> int {\n" ++
            "  slots: s0:P\n" ++
            "b0:\n" ++
            "  %1 = slot_addr s0\n" ++
            "  %2 = field_addr %1, 0 : int\n" ++
            "  %3 = iconst 40\n" ++
            "  store %2, %3 : int\n" ++
            "  %4 = field_addr %1, 8 : int\n" ++
            "  %5 = iconst 0\n" ++
            "  store %4, %5 : int\n" ++
            "  %6 = slot_addr s0\n" ++
            "  %7 = iconst 2\n" ++
            "  call @main.P.bump(%6, %7)\n" ++
            "  %8 = slot_addr s0\n" ++
            "  %9 = field_addr %8, 0 : int\n" ++
            "  %10 = load %9 : int\n" ++
            "  br b1(%10)\n" ++
            "b1(%0:int):\n" ++
            "  ret %0\n" ++
            "}\n",
    );
}

test "M9: whole-self value read copies the pointee into a fresh temp" {
    try expectLoweredG(
        "struct P { x: int, y: int }\n" ++
            "impl P { fn ident(mut self) -> P { return self } }\n" ++
            "fn main() -> int { p := P{ x: 1, y: 2 }\n q := p.ident()\n return q.x }\n",
        "ident",
        // Whole-`self` read: the slot holds a pointer, so `self` is COPIED from the
        // pointee (`slot_addr s0; load` → the pointer) into a fresh temp (s1).
        "fn main.P.ident(s0) -> P {\n" ++
            "  slots: s0:int s1:P\n" ++
            "b0:\n" ++
            "  %1 = slot_addr s1\n" ++
            "  %2 = slot_addr s0\n" ++
            "  %3 = load %2 : int\n" ++
            "  copy %1 <- %3 : P\n" ++
            "  br b1(s1)\n" ++
            "b1(%0:P):\n" ++
            "  ret %0\n" ++
            "}\n",
    );
}
