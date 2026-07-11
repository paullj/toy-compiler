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
//! VALUE MODEL: every scalar temporary is an SSA `Value` (defined once);
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
const Abi = @import("codegen/abi/Abi.zig");
const Sig = @import("symbols/Sig.zig").Sig;
const Mono = @import("symbols/Mono.zig");
const Derive = @import("symbols/Derive.zig");
const Infer = @import("symbols/Infer.zig");
const Intrinsic = @import("symbols/Intrinsic.zig");
const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;
const Literal = @import("types/literal.zig");

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
    /// The monomorphization instance table. A generic call `id[int](..)` in
    /// this body resolves its callee to the reified instance's mangled SymName by
    /// matching `(template gid + the concrete type-args read from node_types)`
    /// against this table. Empty for a program with no generics. Resolution reads
    /// only concrete `node_types` — no substitution map is threaded here (the mono
    /// tail already substituted every `type_var` away before `node_types` froze).
    instances: []const Mono.Instance = &.{},
    /// Program-wide callee signatures by global fn id. A bare inferred generic
    /// call `id(7)` resolves its callee to a generic template's `.func`; that is
    /// detectable because the template's `sigs[func].params` carry `type_var`s. When
    /// so, `lowerCall` re-runs the shared `Infer` matcher over the value-arg
    /// `node_types` to select the SAME `Mono.Instance` Pass C created, instead of
    /// emitting the template symbol. Empty for a program with no generics.
    sigs: []const Sig = &.{},
    /// The program-wide inherent-method table. A call whose callee is a
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
    /// The authorized structural auto-derive recipes, so `lowerStructEq`'s `.one`
    /// arm can resolve a derived `Eq` witness (`m.derive`) to the synthetic unit's
    /// mangled name, and `lowerDeriveEq` can resolve a nested aggregate field's witness.
    /// Empty for a program with no derives. NO default (same COMPILE-error discipline as
    /// `methods`): every build site threads it explicitly.
    derives: []const Derive.Derive,
    /// The prelude protocol ids, so each operator/derive/`?`-widen witness site
    /// resolves by its SPECIFIC protocol — a sibling protocol reusing `eq`/`cmp`/… on the
    /// operand type cannot be selected (which would make the resolver `.ambiguous` and abort
    /// codegen on a checked program). MUST equal the id the fingerprint fold uses (same
    /// checker snapshot) or warm-cache/`-jN` determinism breaks. Defaults to all-null (a
    /// prelude-less test caller): witness sites then fall back to name-only resolution.
    prelude_ids: Typecheck.PreludeProtocolIds = .{},
    /// The compiler-provided `char` struct id (see `Typecheck.Prelude.char_struct`), so
    /// the `.into()`/`.try_into()` recognizer + char-literal lowering key the char cases
    /// off the same id the checker used. Null for a prelude-less test caller (no char).
    char_struct: ?u32 = null,
};

/// The mutable builder state for ONE function lowering. All index spaces
/// (slots/values/blocks) are handed out monotonically as the source-order walk
/// proceeds, which is what makes the result deterministic.
///
/// `pub` (with the derive-facing methods `pub`) so the sibling `lower/derive_emit.zig`
/// can build the SAME one-function state for a source-less auto-derive unit; the
/// dependency stays one-way (`derive_emit → lower`).
pub const Builder = struct {
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

    /// The slot holding the `mut self` receiver's ADDRESS, or `none_slot` for
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

    pub fn deinit(b: *Builder) void {
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
    pub fn addLiteral(b: *Builder, hash: u64, bytes: []u8) error{OutOfMemory}!void {
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

    pub fn addSlot(b: *Builder, ty: Typecheck.Type) error{OutOfMemory}!Ir.SlotId {
        const id: Ir.SlotId = @intCast(b.slots.items.len);
        try b.slots.append(b.gpa, .{ .type = ty });
        return id;
    }

    fn addValue(b: *Builder, ty: Typecheck.Type) error{OutOfMemory}!Ir.ValueId {
        const id: Ir.ValueId = @intCast(b.values.items.len);
        try b.values.append(b.gpa, .{ .type = ty });
        return id;
    }

    pub fn addBlock(b: *Builder) error{OutOfMemory}!Ir.BlockId {
        const id: Ir.BlockId = @intCast(b.blocks.items.len);
        try b.blocks.append(b.gpa, .{});
        return id;
    }

    /// Append a new block param (a merge slot) to `bid`, returning its value id.
    pub fn addParam(b: *Builder, bid: Ir.BlockId, ty: Typecheck.Type) error{OutOfMemory}!Ir.ValueId {
        const v = try b.addValue(ty);
        try b.blocks.items[bid].params.append(b.gpa, v);
        return v;
    }

    /// Emit an op into the CURRENT block, producing a value of `ty` (or pass
    /// `null` for a result-less op like store/copy/void-call → `none_value`).
    pub fn emit(b: *Builder, op: Ir.Op, ty: ?Typecheck.Type) error{OutOfMemory}!Ir.ValueId {
        const result: Ir.ValueId = if (ty) |t| try b.addValue(t) else Ir.none_value;
        try b.blocks.items[b.cur].instrs.append(b.gpa, .{ .result = result, .op = op });
        return result;
    }

    /// Set the current block's terminator (idempotent guard: the FIRST terminator
    /// wins, so a divergent sub-construct that already terminated is not clobbered
    /// by an enclosing fall-through branch).
    pub fn setTerm(b: *Builder, term: Ir.Terminator) void {
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

    pub fn termSet(b: *Builder) bool {
        return b.blocks.items[b.cur].term_set;
    }

    /// Switch the cursor to `bid` (the block subsequent code appends to).
    pub fn switchTo(b: *Builder, bid: Ir.BlockId) void {
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
        // A `mut self` receiver: type param-0's slot as `int` (an 8B pointer to
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
        // A managed box returns as an 8-byte scalar cell pointer (in a register), NOT via
        // the struct reg-pair/sret ABI, so the exit param is `int`. The body value flows
        // in as a scalar (every Ref producer yields `.value`), and the call site reads it
        // as a scalar too.
        const ret_param_ty = if (isRefTy(&b, ret_type)) Typecheck.Type.int else ret_type;
        b.ret_param = try b.addParam(exit, ret_param_ty);
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

    return try finishFn(&b, gpa, sym, &params, entry, exit);
}

/// Materialize the builder into a flat `Ir.Function`, transferring ownership of every
/// index space (slots/values/blocks/params/literals) out of `b` — which is left EMPTY
/// so the caller's `errdefer b.deinit()` frees nothing it no longer owns. Shared by
/// `lowerFn` (AST path) and `lowerDeriveEq` (source-less path). Every fallible alloc
/// happens BEFORE tearing the builder down, so the caller's `errdefer b.deinit()` +
/// this fn's own `errdefer`s stay valid cleanups until ownership fully transfers.
pub fn finishFn(
    b: *Builder,
    gpa: std.mem.Allocator,
    sym: Link.SymName,
    params: *std.ArrayList(Ir.SlotId),
    entry: Ir.BlockId,
    exit: Ir.BlockId,
) error{OutOfMemory}!Ir.Function {
    const params_owned = try params.toOwnedSlice(gpa);
    errdefer gpa.free(params_owned);

    const blocks = try gpa.alloc(Ir.Block, b.blocks.items.len);
    errdefer gpa.free(blocks);
    // Convert each builder block. On a mid-loop OOM, `errdefer b.deinit()` (the caller's)
    // frees the still-owned builder blocks; the already-converted ones were emptied by
    // `toOwnedSlice` (deinit sees empty lists), and `errdefer gpa.free(blocks)` frees the
    // outer array (the moved-in inner slices leak only on OOM — benign).
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

    // All fallible work is done. Disarm the caller's `errdefer b.deinit()` by freeing
    // exactly the builder-owned remnants (the emptied block list, the local map, the
    // loop stack) here; slots/values/blocks/params/literals have moved into the result.
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
        .ret_type = b.ret_type,
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
            // `*r = v`: a deref place stores through the box, reusing the field-store tail
            // (its address comes from `lowerPlaceAddr`'s `.unary` arm).
            const is_deref = target.tag == .unary and b.in.tokens[target.main_token].tag == .star;
            if (target.tag == .field_access or target.tag == .tuple_field or is_deref) {
                try lowerFieldStore(b, stmt.lhs, stmt.rhs, place_ty);
                return;
            }
            const slot = try localSlot(b, stmt.lhs, place_ty);
            // Whole-`self` reassignment inside a `mut self` method (`self = expr`): the
            // slot holds a POINTER, so produce the RHS through it into the caller's
            // storage rather than overwriting the local pointer.
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
        .unsafe_block => try lowerBlockStmts(b, stmt.lhs),
        .if_stmt => try lowerIfStmt(b, stmt_idx),
        .while_stmt => try lowerWhile(b, stmt_idx, null),
        .for_stmt => try lowerFor(b, stmt_idx, null),
        .labeled => try lowerLabeledStmt(b, stmt_idx),
        .break_stmt => try lowerBreak(b, stmt_idx),
        .continue_stmt => try lowerContinue(b, stmt_idx),
        // A poison leaf must never reach lower: a tainted tree is gated out before
        // codegen, and it is not produced anywhere yet.
        .error_node => unreachable,
        else => try b.note(stmt.main_token, "statement unsupported in lower"),
    }
}

/// Store the value of `expr` into `slot`. A scalar emits a store; an aggregate
/// (str/struct/enum) is PRODUCED INTO the slot directly via `lowerExprInto` (the
/// produce-into-slot path: construct/copy/match write straight to the destination,
/// never value-then-copy).
fn storeInto(b: *Builder, slot: Ir.SlotId, expr: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!void {
    // A managed box is a struct type but travels as an 8-byte scalar cell pointer, not an
    // aggregate slot, so store it like a scalar rather than routing to `lowerExprInto`.
    if (isRefTy(b, ty)) {
        const v = try lowerExpr(b, expr);
        const addr = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
        _ = try b.emit(.{ .store = .{ .addr = addr, .val = operandValue(v), .ty = Typecheck.Type.int } }, null);
        return;
    }
    switch (ty.kind) {
        .int, .bool, .float, .rawptr => {
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
    // A check-time `type_var` / composite `App` is substituted/reified to a
    // concrete kind BEFORE lowering; if one reaches here, the mono tail missed a
    // node_types slot — trip loudly in Debug/ReleaseSafe rather than miscompile.
    std.debug.assert(ty.kind != .type_var and ty.kind != .app);
    switch (n.tag) {
        .literal_number => {
            // A literal past 2^64-1 is now a T0034 at the checker, which gates codegen,
            // so this fallback is unreachable for a checked program; keep the `iconst 0`
            // so an unchecked lower test can't crash.
            const v = Literal.value(b.in.tokens[n.main_token].text(b.in.source)) orelse
                return .{ .value = try b.emit(.{ .iconst = 0 }, Typecheck.Type.int) };
            return .{ .value = try b.emit(.{ .iconst = v }, Typecheck.Type.int) };
        },
        .literal_float => {
            const f = Literal.floatValue(b.in.tokens[n.main_token].text(b.in.source)) orelse 0.0;
            return .{ .value = try b.emit(.{ .fconst = f }, Typecheck.Type.float) };
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
        // A char literal is a `char` struct VALUE: materialize it into a fresh temp slot.
        .literal_char => return try aggregateValue(b, node_idx, ty),
        .identifier => return try lowerIdentifier(b, node_idx, ty),
        .unary => return try lowerUnary(b, node_idx, n),
        .binary => return try lowerBinary(b, node_idx, n),
        .call => {
            if (isQualifiedVariantCtorCall(b, n, ty)) {
                return try aggregateValue(b, node_idx, ty);
            }
            if (isTupleStructCtorCall(b, n, ty)) {
                return try aggregateValue(b, node_idx, ty);
            }
            return try lowerCall(b, node_idx, n);
        },
        .block => return try lowerBlockValue(b, node_idx, ty),
        // `unsafe { .. }` lowers transparently as its inner block — the unsafe context
        // is a check-time concern (BodyChecker gates `store`/`load`), invisible here.
        .unsafe_block => return try lowerBlockValue(b, n.lhs, ty),
        .if_stmt => return try lowerIfValue(b, node_idx, ty),
        .loop_expr => return try lowerLoopValue(b, node_idx, ty, null),
        .labeled => return try lowerLabeledValue(b, node_idx, ty),
        .field_access, .tuple_field => return try lowerFieldAccess(b, node_idx, ty),
        .match_expr => return try lowerMatchValue(b, node_idx, ty),
        .try_expr => return try lowerTryValue(b, node_idx, ty),
        .struct_init, .enum_init_unit, .enum_init_tuple, .enum_init_struct => {
            // An aggregate-producing expression in value context: materialize it
            // into a fresh temp slot and yield Operand.slot.
            return try aggregateValue(b, node_idx, ty);
        },
        // A poison leaf must never reach lower: a tainted tree is gated out before
        // codegen, and it is not produced anywhere yet.
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
    // A managed box reads as an 8-byte scalar cell pointer (its own value), not by slot.
    if (isRefTy(b, ty)) {
        const addr = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
        return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = Typecheck.Type.int } }, Typecheck.Type.int) };
    }
    switch (ty.kind) {
        .int, .bool, .float, .rawptr => {
            const addr = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            const v = try b.emit(.{ .load = .{ .addr = addr, .ty = ty } }, ty);
            return .{ .value = v };
        },
        .str, .@"struct", .@"enum" => {
            // Whole-`self` value read inside a `mut self` method (`return self`, or
            // passing `self` by value): the slot holds a POINTER, not the struct, so
            // copy the pointee into a fresh temp and yield that. An ordinary
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
            // The checked type of the negation node carries its integer width, so
            // codegen re-canonicalizes a narrow `neg` (sxtb/and) at the right size.
            return .{ .value = try b.emit(.{ .neg = operandValue(v) }, b.in.node_types[(node_idx).int()]) };
        },
        .bang => {
            const v = try lowerExpr(b, n.lhs);
            return .{ .value = try b.emit(.{ .bnot = operandValue(v) }, Typecheck.Type.@"bool") };
        },
        .tilde => {
            const v = try lowerExpr(b, n.lhs);
            return .{ .value = try b.emit(.{ .bcompl = operandValue(v) }, b.in.node_types[(node_idx).int()]) };
        },
        // `&x`: box `x` into a fresh `gc_alloc`'d cell; the `Ref[T]` value IS the 8-byte
        // cell pointer. The cell is never freed (correct for a terminating program).
        .amp => {
            const t = b.in.node_types[(n.lhs).int()];
            const size_v = try b.emit(.{ .iconst = @intCast(Abi.typeSize(t, b.in.layouts, b.in.enum_layouts)) }, Typecheck.Type.int);
            // Scope the `args` errdefer to the alloc→emit window: once emit succeeds the
            // block instruction owns `args`, so a later OOM must NOT re-free it here.
            const cellptr = cell: {
                const args = try b.gpa.alloc(Ir.Operand, 1);
                errdefer b.gpa.free(args);
                args[0] = .{ .value = size_v };
                break :cell try b.emit(.{ .call = .{ .callee = gc_alloc_sym, .args = args, .ret_slot = Ir.none_slot } }, Typecheck.Type.int);
            };
            if (isRefTy(b, t)) {
                const v = try lowerExpr(b, n.lhs);
                _ = try b.emit(.{ .store = .{ .addr = cellptr, .val = operandValue(v), .ty = Typecheck.Type.int } }, null);
            } else switch (t.kind) {
                .int, .bool, .float, .rawptr => {
                    const v = try lowerExpr(b, n.lhs);
                    _ = try b.emit(.{ .store = .{ .addr = cellptr, .val = operandValue(v), .ty = t } }, null);
                },
                .str, .@"struct", .@"enum" => try lowerExprInto(b, n.lhs, cellptr, t),
                .unit => _ = try lowerExpr(b, n.lhs),
                else => try b.note(n.main_token, "boxed type unsupported in lower"),
            }
            return .{ .value = cellptr };
        },
        // `*r`: place-read the boxed `T` through the cell pointer (offset 0).
        .star => {
            const addr = try lowerPlaceAddr(b, node_idx);
            if (addr == Ir.none_value) return .none;
            const t = b.in.node_types[(node_idx).int()];
            if (isRefTy(b, t)) return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = Typecheck.Type.int } }, Typecheck.Type.int) };
            switch (t.kind) {
                .int, .bool, .float, .rawptr => return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = t } }, t) },
                .str, .@"struct", .@"enum" => {
                    const tmp = try b.addSlot(t);
                    const dst = try b.emit(.{ .slot_addr = tmp }, Typecheck.Type.int);
                    _ = try b.emit(.{ .copy = .{ .dst = dst, .src = addr, .ty = t } }, null);
                    return .{ .slot = tmp };
                },
                .unit => return .none,
                else => {
                    try b.note(n.main_token, "deref type unsupported in lower");
                    return .none;
                },
            }
        },
        else => {
            try b.note(n.main_token, "unary operator unsupported in lower");
            return .none;
        },
    }
}

/// Split the current block on `rhs == 0`: zero -> a no-return `.panic(reason)`
/// block, nonzero -> a fresh continuation the builder is left positioned in. Value-
/// free by design (no join/merge param) — the panic path never produces a value, so
/// the divide emitted afterward in `cont` is an ordinary SSA value dominating its
/// uses. Modeled on the `unwrap` trap-block; block ids are handed out in a fixed
/// per-call order so `--verify` re-lower stays byte-identical.
fn emitZeroGuard(b: *Builder, rhs: Ir.ValueId, ty: Typecheck.Type, reason: Ir.Terminator.PanicReason) error{OutOfMemory}!void {
    const zero = try b.emit(.{ .iconst = 0 }, ty);
    const is_zero = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = rhs, .rhs = zero } }, Typecheck.Type.@"bool");
    const cont = try b.addBlock();
    const panic_blk = try b.addBlock();
    b.setTerm(.{ .cond_br = .{ .cond = is_zero, .t = panic_blk, .f = cont } });
    b.switchTo(panic_blk);
    b.setTerm(.{ .panic = reason });
    b.switchTo(cont);
}

fn lowerBinary(b: *Builder, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Ir.Operand {
    _ = node_idx;
    const op = b.in.tokens[n.main_token].tag;
    switch (op) {
        .plus, .minus, .star, .slash => {
            // int stays an inline machine add/sub/mul/sdiv (bytes unchanged); a struct/enum
            // operand desugars to the resolved Add/Sub/Mul/Div witness via `lowerArithValue`
            //. The checker (T0028) has already proven a same-type conforming operand.
            const lt = b.in.node_types[(n.lhs).int()];
            if (isInlineArith(lt.kind)) {
                const lhs = operandValue(try lowerExpr(b, n.lhs));
                const rhs = operandValue(try lowerExpr(b, n.rhs));
                const bin: Ir.Bin = .{ .lhs = lhs, .rhs = rhs };
                // `/` gains a runtime zero-divisor guard; `+`/`-`/`*` stay byte-identical.
                if (op == .slash) {
                    try emitZeroGuard(b, rhs, lt, .div_by_zero);
                    const ir_op: Ir.Op = if (lt.isUnsignedInt()) Ir.Op{ .udiv = bin } else Ir.Op{ .sdiv = bin };
                    return .{ .value = try b.emit(ir_op, lt) };
                }
                const ir_op: Ir.Op = switch (op) {
                    .plus => .{ .add = bin },
                    .minus => .{ .sub = bin },
                    .star => .{ .mul = bin },
                    else => unreachable,
                };
                // `lt` is the operand width; the checker proved `eql(lt,rt)` and
                // returns it as the result type, so it is also the result width.
                return .{ .value = try b.emit(ir_op, lt) };
            }
            return try lowerArithValue(b, lt, n.lhs, n.rhs, op);
        },
        .lt, .lt_eq, .gt, .gt_eq => {
            // int/bool stay an inline `icmp` (bytes unchanged); str/struct/enum desugar to a
            // discriminant test on `Ord::cmp` via `lowerOrdValue`.
            const lt = b.in.node_types[(n.lhs).int()];
            if (isInlineOrd(lt.kind)) {
                const lhs = operandValue(try lowerExpr(b, n.lhs));
                const rhs = operandValue(try lowerExpr(b, n.rhs));
                const cc = condFromToken(op, lt.isUnsignedInt());
                return .{ .value = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.@"bool") };
            }
            return try lowerOrdValue(b, lt, n.lhs, n.rhs, op);
        },
        .eq_eq, .bang_eq => {
            // int/bool stay an inline `icmp` (bytes unchanged); str/unit/struct/enum
            // desugar to `Eq::eq` via `lowerEqValue`.
            const lt = b.in.node_types[(n.lhs).int()];
            if (isInlineEq(lt.kind)) {
                const lhs = operandValue(try lowerExpr(b, n.lhs));
                const rhs = operandValue(try lowerExpr(b, n.rhs));
                const cc = condFromToken(op, lt.isUnsignedInt());
                return .{ .value = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.@"bool") };
            }
            return try lowerEqValue(b, lt, n.lhs, n.rhs, op == .bang_eq);
        },
        .amp_amp, .pipe_pipe => return try lowerAndOrValue(b, n, op),
        .amp, .pipe, .caret, .lt_lt, .gt_gt => {
            const lt = b.in.node_types[(n.lhs).int()];
            const lhs = operandValue(try lowerExpr(b, n.lhs));
            const rhs = operandValue(try lowerExpr(b, n.rhs));
            const bin: Ir.Bin = .{ .lhs = lhs, .rhs = rhs };
            const ir_op: Ir.Op = switch (op) {
                .amp => .{ .band = bin },
                .pipe => .{ .bor = bin },
                .caret => .{ .bxor = bin },
                .lt_lt => .{ .shl = bin },
                .gt_gt => if (lt.isUnsignedInt()) Ir.Op{ .lshr = bin } else Ir.Op{ .ashr = bin },
                else => unreachable,
            };
            return .{ .value = try b.emit(ir_op, lt) };
        },
        .percent => {
            // Integer-only builtin like the bitwise arm above (no Div-style witness path),
            // so it trusts the checker (T "operands of '%' must be int") rather than
            // re-guarding the operand kind.
            const lt = b.in.node_types[(n.lhs).int()];
            const lhs = operandValue(try lowerExpr(b, n.lhs));
            const rhs = operandValue(try lowerExpr(b, n.rhs));
            try emitZeroGuard(b, rhs, lt, .rem_by_zero);
            const bin: Ir.Bin = .{ .lhs = lhs, .rhs = rhs };
            const ir_op: Ir.Op = if (lt.isUnsignedInt()) Ir.Op{ .umod = bin } else Ir.Op{ .smod = bin };
            return .{ .value = try b.emit(ir_op, lt) };
        },
        .plus_dot, .minus_dot, .star_dot, .slash_dot => {
            const lhs = operandValue(try lowerExpr(b, n.lhs));
            const rhs = operandValue(try lowerExpr(b, n.rhs));
            const bin: Ir.Bin = .{ .lhs = lhs, .rhs = rhs };
            // No zero-guard for `/.`: IEEE-754 division yields ±inf/NaN, never traps.
            const ir_op: Ir.Op = switch (op) {
                .plus_dot => .{ .fadd = bin },
                .minus_dot => .{ .fsub = bin },
                .star_dot => .{ .fmul = bin },
                .slash_dot => .{ .fdiv = bin },
                else => unreachable,
            };
            return .{ .value = try b.emit(ir_op, Typecheck.Type.float) };
        },
        .lt_dot, .gt_dot, .le_dot, .ge_dot => {
            const lhs = operandValue(try lowerExpr(b, n.lhs));
            const rhs = operandValue(try lowerExpr(b, n.rhs));
            const cc: Ir.FCond = switch (op) {
                .lt_dot => .lt,
                .gt_dot => .gt,
                .le_dot => .le,
                .ge_dot => .ge,
                else => unreachable,
            };
            return .{ .value = try b.emit(.{ .fcmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.@"bool") };
        },
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

/// True for the two `Eq` operand kinds that stay an inline `icmp` (int/bool). Every other
/// kind (unit/str/struct/enum) routes to `lowerEqValue`, so the int/bool emitted bytes are
/// unchanged BY CONSTRUCTION (the "int/bool == unchanged" acceptance criterion).
fn isInlineEq(k: Typecheck.Kind) bool {
    return k == .int or k == .bool;
}

/// The prelude `Ordering{lt,eq,gt}` variant DECL indices (registerPrelude order), i.e. the
/// tag `get_tag` reads. `<`/`>`/`<=`/`>=` desugar to a discriminant test against these.
pub const ord_lt: i64 = 0;
pub const ord_eq: i64 = 1;
pub const ord_gt: i64 = 2;

/// Fixed-seed constants for the structural `Hash` derive's fxhash mixer
/// `h := (rotl(h,5) ^ word) *% K` (see `hashMix`). Adding the logical shift + xor +
/// bitwise ops made the mixer a real fxhash step, replacing the old no-bitwise FNV
/// polynomial. Every LOCKED constraint still holds: a FIXED seed (never randomized/per-run),
/// deterministic, reproducible run-to-run AND byte-identical across `-jN`, and `Eq`-consistent
/// (the emitter walks the SAME field order `Eq` does). The constants MUST fit i64 (`< 2^63`):
/// the mixer wraps via the IR's `*%` (`opt/arith.zig` never traps). `hash_seed` is the pi
/// fractional word (shared: also the seed the builtin `.hash()` path uses); `str_hash_seed`
/// is a distinct word so a str field's byte fold does not alias the field seed; `hash_k` is
/// the fxhash odd multiplier, shared by BOTH mixer sites via `hashMix`.
pub const hash_seed: i64 = 0x243F6A8885A308D3;
const str_hash_seed: i64 = 0x2545F4914F6CDD1D;
pub const hash_k: i64 = 0x517cc1b727220a95;

/// The one fxhash mixing step, shared by BOTH `Hash` sites — this file's `hashStrAtPtr`
/// byte fold and `derive_emit`'s struct/enum field/tag fold: `h := (rotl(h,5) ^ word) *% K`.
/// The right rotate half is emitted as `.lshr` (LOGICAL): `h` is typed `int` (signed), so
/// `.ashr` would sign-fill a negative accumulator and diverge from fxhash. `K` is
/// re-materialized as a pure iconst at each call, so the emitter reads no shared mutable state
/// and a double-lower is byte-identical under `-jN`. All ops wrap (never trap).
pub fn hashMix(b: *Builder, h: Ir.ValueId, word: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const c5 = try b.emit(.{ .iconst = 5 }, int_ty);
    const c59 = try b.emit(.{ .iconst = 59 }, int_ty);
    const rot_lo = try b.emit(.{ .shl = .{ .lhs = h, .rhs = c5 } }, int_ty);
    const rot_hi = try b.emit(.{ .lshr = .{ .lhs = h, .rhs = c59 } }, int_ty);
    const rot = try b.emit(.{ .bor = .{ .lhs = rot_lo, .rhs = rot_hi } }, int_ty);
    const x = try b.emit(.{ .bxor = .{ .lhs = rot, .rhs = word } }, int_ty);
    const k = try b.emit(.{ .iconst = hash_k }, int_ty);
    return try b.emit(.{ .mul = .{ .lhs = x, .rhs = k } }, int_ty);
}

/// True for the two `Ord` operand kinds that stay an inline `icmp` (int signed cmp, bool
/// false<true). Every other kind (str/struct/enum) routes to `lowerOrdValue`, so the int/bool
/// emitted bytes are unchanged BY CONSTRUCTION (the "int comparisons unchanged" criterion).
fn isInlineOrd(k: Typecheck.Kind) bool {
    return k == .int or k == .bool;
}

/// True for the one arithmetic operand kind that stays an inline machine op (`int`). Every
/// other kind routes to `lowerArithValue`, so the int emitted bytes are unchanged BY
/// CONSTRUCTION (the "int `+`/`-`/`*`/`/` unchanged" acceptance criterion). str/bool
/// never reach lower for arithmetic — the checker rejects them (T0028).
fn isInlineArith(k: Typecheck.Kind) bool {
    return k == .int;
}

/// The slot an aggregate/str operand travels by (str/struct/enum always lower to `.slot`).
fn operandSlot(op: Ir.Operand) Ir.SlotId {
    return switch (op) {
        .slot => |s| s,
        else => Ir.none_slot,
    };
}

/// Lower `lhs == rhs` (or `!=` when `negate`) for a NON-inline operand kind
/// (unit/str/struct/enum), producing a bool value Operand. int/bool are never routed
/// here — they stay the inline `icmp` in `lowerBinary`/`genCond`, unchanged.
///   * unit  -> lower both operands for effect, then `bconst true` (the two `()` values are
///              always equal).
///   * str   -> a heap-free len-compare then a per-byte `load_byte` loop (see `lowerStrEq`).
///   * struct/enum -> resolve the `Eq` witness and emit `witness(self-by-value, rhs)` ->
///              bool, byte-identical to the `p.eq(q)` method form (see `lowerStructEq`).
/// `!=` wraps the resulting bool in a `bnot`.
fn lowerEqValue(b: *Builder, operand_ty: Typecheck.Type, lhs_node: Ast.Index, rhs_node: Ast.Index, negate: bool) error{OutOfMemory}!Ir.Operand {
    // A managed box compares by CELL IDENTITY: an 8-byte pointer `icmp`, not a structural
    // deref. Two independent `&41` are distinct cells (unequal); `r == r` is the same cell.
    if (isRefTy(b, operand_ty)) {
        const lv = operandValue(try lowerExpr(b, lhs_node));
        const rv = operandValue(try lowerExpr(b, rhs_node));
        const cc: Ir.Cond = if (negate) .ne else .eq;
        return .{ .value = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lv, .rhs = rv } }, Typecheck.Type.@"bool") };
    }
    // A struct/enum threads `negate` into `lowerStructEq` so the Ord-refinement `==` path
    // can emit `icmp ne` directly AND the eq-witness path stays byte-identical
    // (call then optional `bnot`, same value-id order). unit/str keep the uniform outer `bnot`.
    switch (operand_ty.kind) {
        .@"struct", .@"enum" => return try lowerStructEq(b, operand_ty, lhs_node, rhs_node, negate),
        .float => {
            const lhs = operandValue(try lowerExpr(b, lhs_node));
            const rhs = operandValue(try lowerExpr(b, rhs_node));
            const cc: Ir.FCond = if (negate) .ne else .eq;
            return .{ .value = try b.emit(.{ .fcmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.@"bool") };
        },
        else => {},
    }
    const raw: Ir.ValueId = switch (operand_ty.kind) {
        .unit => blk: {
            _ = try lowerExpr(b, lhs_node);
            _ = try lowerExpr(b, rhs_node);
            break :blk try b.emit(.{ .bconst = true }, Typecheck.Type.@"bool");
        },
        .str => try lowerStrEq(b, lhs_node, rhs_node),
        else => blk: {
            // Unreachable for a well-typed program (int/bool never routed here; any other
            // kind is a type error caught before lower). Note-and-drop to stay well-formed.
            try b.note(b.in.tree.nodes[(lhs_node).int()].main_token, "'==' operand type unsupported in lower");
            break :blk try b.emit(.{ .bconst = false }, Typecheck.Type.@"bool");
        },
    };
    if (negate) return .{ .value = try b.emit(.{ .bnot = raw }, Typecheck.Type.@"bool") };
    return .{ .value = raw };
}

/// Emit `==`/`!=` (`!=` when `negate`) for a struct/enum operand. If the type has
/// an explicit `Eq` witness, dispatch to it (byte-identical to `p.eq(q)`) then optionally
/// `bnot` — the eq-witness path, unchanged. If NOT (an Ord-only type, where the Ord-refinement
/// filled the `(Eq,T)` slot but added no `eq` method), lower `==` as `discriminant == ord_eq`
/// (`!=` as `discriminant != ord_eq`) via the `cmp` witness. Returns the final bool Operand.
fn lowerStructEq(b: *Builder, operand_ty: Typecheck.Type, lhs_node: Ast.Index, rhs_node: Ast.Index, negate: bool) error{OutOfMemory}!Ir.Operand {
    switch (Typecheck.resolveConformanceMethod(b.in.methods, operand_ty, "eq", b.in.prelude_ids.eq, null)) {
        .one => |m| {
            // A derived `Eq` / instance / plain fn all resolve through the shared
            // `witnessCallee` (derive checked first — a derive Method has `fn_id == 0`).
            const callee: Link.SymName = witnessCallee(b, m);
            const args = try b.gpa.alloc(Ir.Operand, 2);
            args[0] = try lowerExpr(b, lhs_node); // self, by value
            args[1] = try lowerExpr(b, rhs_node);
            const v = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, Typecheck.Type.@"bool");
            if (negate) return .{ .value = try b.emit(.{ .bnot = v }, Typecheck.Type.@"bool") };
            return .{ .value = v };
        },
        .none, .ambiguous => {
            // Ord-refinement `==`: no `eq` witness, but a `cmp` witness exists — `==`
            // is `cmp(a,b) == Ordering.eq`. The checker proved conformance (the refinement
            // filled `(Eq,T)`), so a `cmp` miss here is an internal invariant break.
            switch (Typecheck.resolveConformanceMethod(b.in.methods, operand_ty, "cmp", b.in.prelude_ids.ord, null)) {
                .one => {
                    const d = try lowerCmpDiscriminant(b, operand_ty, lhs_node, rhs_node);
                    const k = try b.emit(.{ .iconst = ord_eq }, Typecheck.Type.int);
                    const cc: Ir.Cond = if (negate) .ne else .eq;
                    return .{ .value = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = d, .rhs = k } }, Typecheck.Type.@"bool") };
                },
                .none, .ambiguous => {
                    try b.note(b.in.tree.nodes[(lhs_node).int()].main_token, "no unique 'Eq' or 'Ord' witness for '==' in lower");
                    const fv = try b.emit(.{ .bconst = false }, Typecheck.Type.@"bool");
                    if (negate) return .{ .value = try b.emit(.{ .bnot = fv }, Typecheck.Type.@"bool") };
                    return .{ .value = fv };
                },
            }
        },
    }
}

/// Lower `lhs <op> rhs` for a NON-inline `Ord` operand kind (str/struct/enum), producing a
/// bool value Operand. Computes the 3-way `cmp` discriminant, then tests it: `<` -> the
/// discriminant equals `ord_lt`; `>` -> equals `ord_gt`; `<=` -> NOT `ord_gt`; `>=` -> NOT
/// `ord_lt`. int/bool are never routed here — they stay the inline `icmp`, unchanged.
fn lowerOrdValue(b: *Builder, operand_ty: Typecheck.Type, lhs_node: Ast.Index, rhs_node: Ast.Index, op: TokenTag) error{OutOfMemory}!Ir.Operand {
    const d = try lowerCmpDiscriminant(b, operand_ty, lhs_node, rhs_node);
    const spec: struct { thr: i64, cc: Ir.Cond } = switch (op) {
        .lt => .{ .thr = ord_lt, .cc = .eq },
        .gt => .{ .thr = ord_gt, .cc = .eq },
        .lt_eq => .{ .thr = ord_gt, .cc = .ne },
        .gt_eq => .{ .thr = ord_lt, .cc = .ne },
        else => unreachable,
    };
    const k = try b.emit(.{ .iconst = spec.thr }, Typecheck.Type.int);
    return .{ .value = try b.emit(.{ .icmp = .{ .cc = spec.cc, .lhs = d, .rhs = k } }, Typecheck.Type.@"bool") };
}

/// The int 3-way `Ord` discriminant (0=lt/1=eq/2=gt) of `lhs`/`rhs`:
///   * str -> a heap-free lexicographic `load_byte` loop (`lowerStrCmp`), NO witness call.
///   * struct/enum -> the `cmp` witness call into a fresh ret_slot, then `get_tag` at
///     offset 0 (the proven match-dispatch idiom) reads the returned `Ordering`'s tag.
fn lowerCmpDiscriminant(b: *Builder, operand_ty: Typecheck.Type, lhs_node: Ast.Index, rhs_node: Ast.Index) error{OutOfMemory}!Ir.ValueId {
    switch (operand_ty.kind) {
        .str => return try lowerStrCmp(b, lhs_node, rhs_node),
        .@"struct", .@"enum" => {
            const m = switch (Typecheck.resolveConformanceMethod(b.in.methods, operand_ty, "cmp", b.in.prelude_ids.ord, null)) {
                .one => |mm| mm,
                else => {
                    // The checker proves exactly one `Ord` witness before lower; a miss here
                    // is an internal invariant break — note-and-drop rather than miscompile.
                    try b.note(b.in.tree.nodes[(lhs_node).int()].main_token, "no unique 'Ord' witness for comparison in lower");
                    return try b.emit(.{ .iconst = ord_eq }, Typecheck.Type.int);
                },
            };
            // The witness returns `Ordering`; its ret carries that enum type (the ret_slot ABI
            // + get_tag layout). A DERIVED `cmp` (`fn_id == 0`) reads it from the recipe,
            // a Mono instance from the instance, else the fn's own sig — all via
            // `witnessRet`/`witnessCallee` (derive-first), so a source-less Ord witness resolves.
            const ret_ty = witnessRet(b, m);
            const callee: Link.SymName = witnessCallee(b, m);
            const args = try b.gpa.alloc(Ir.Operand, 2);
            args[0] = try lowerExpr(b, lhs_node); // self, by value
            args[1] = try lowerExpr(b, rhs_node);
            const slot = try b.addSlot(ret_ty);
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = slot } }, null);
            const base = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            return try b.emit(.{ .get_tag = base }, Typecheck.Type.int);
        },
        else => {
            try b.note(b.in.tree.nodes[(lhs_node).int()].main_token, "comparison operand type unsupported in lower");
            return try b.emit(.{ .iconst = ord_eq }, Typecheck.Type.int);
        },
    }
}

/// Lower `lhs <op> rhs` for a NON-inline arithmetic operand kind (struct/enum), producing the
/// aggregate result as a `.slot` Operand. `+`/`-`/`*`/`/` map to the resolved
/// Add/Sub/Mul/Div witness, emitted as `witness(self-by-value, rhs) -> ret_slot`, byte-identical
/// to the general aggregate-return call path and to the `p.add(q)` method form. int is never
/// routed here (it stays the inline machine op in `lowerBinary`). The witness `ret_ty` comes off
/// its own Sig/instance (matching `lowerCmpDiscriminant`); T0024 forces `Out = Self`, so today it
/// equals `operand_ty`, but reading the sig keeps it correct for a future generic-impl instance.
fn lowerArithValue(b: *Builder, operand_ty: Typecheck.Type, lhs_node: Ast.Index, rhs_node: Ast.Index, op: TokenTag) error{OutOfMemory}!Ir.Operand {
    const method: []const u8 = switch (op) {
        .plus => "add",
        .minus => "sub",
        .star => "mul",
        .slash => "div",
        else => unreachable,
    };
    switch (Typecheck.resolveConformanceMethod(b.in.methods, operand_ty, method, Typecheck.witnessProtocolId(b.in.prelude_ids, method), null)) {
        .one => |m| {
            const ret_ty = if (m.instance) |ii| b.in.instances[ii].ret else b.in.sigs[m.fn_id].ret;
            const callee: Link.SymName = if (m.instance) |ii|
                .{ .kind = .user_fn, .name = b.in.instances[ii].name.? }
            else
                b.in.names[m.fn_id];
            const args = try b.gpa.alloc(Ir.Operand, 2);
            errdefer b.gpa.free(args);
            args[0] = try lowerExpr(b, lhs_node); // self, by value
            args[1] = try lowerExpr(b, rhs_node);
            const slot = try b.addSlot(ret_ty);
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = slot } }, null);
            return .{ .slot = slot };
        },
        .none, .ambiguous => {
            // The checker (T0028) proves exactly one arithmetic witness before lower; a miss
            // here is an internal invariant break — note-and-drop into a fresh slot rather than
            // miscompile, keeping the IR well-formed.
            try b.note(b.in.tree.nodes[(lhs_node).int()].main_token, "no unique arithmetic witness in lower");
            return .{ .slot = try b.addSlot(operand_ty) };
        },
    }
}

/// Heap-free lexicographic 3-way str comparison, extending `lowerStrEq`'s `load_byte`
/// idiom: walk both byte spans in lockstep in a slot-counter loop, yielding an int
/// discriminant (0=lt/1=eq/2=gt) through a join block param. A pure function of source
/// (slot-counter + br-arg joins), so it is `--verify`-stable. Zero-extended bytes (0..255)
/// make signed `lt`/`gt` correct. The first differing byte decides; if one span is a proper
/// prefix of the other, the shorter is less; equal spans compare equal.
fn lowerStrCmp(b: *Builder, lhs_node: Ast.Index, rhs_node: Ast.Index) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;

    const lhs_op = try lowerExpr(b, lhs_node);
    const rhs_op = try lowerExpr(b, rhs_node);
    const ls = operandSlot(lhs_op);
    const rs = operandSlot(rhs_op);
    if (ls == Ir.none_slot or rs == Ir.none_slot) {
        try b.note(b.in.tree.nodes[(lhs_node).int()].main_token, "str comparison operand is not a slot in lower");
        return try b.emit(.{ .iconst = ord_eq }, int_ty);
    }

    const lbase = try b.emit(.{ .slot_addr = ls }, int_ty);
    const rbase = try b.emit(.{ .slot_addr = rs }, int_ty);
    return strCmpAtPtrs(b, lbase, rbase);
}

/// Heap-free lexicographic 3-way str comparison given the ADDRESSES of two `{ptr@0, len@8}`
/// headers (factored out of `lowerStrCmp` mirroring `strEqAtPtrs`): the byte loop above,
/// yielding an int discriminant (0=lt/1=eq/2=gt) through a join block param. Pure of source, so
/// `--verify`-stable. Reused by the auto-derive Ord emitter for a `str` FIELD (base =
/// `field_addr(self, off)`), where there is no slot to name — only an address.
pub fn strCmpAtPtrs(b: *Builder, lbase: Ir.ValueId, rbase: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";

    // ptr@0 + len@8 of each {ptr,len} aggregate.
    const lp = try b.emit(.{ .load = .{ .addr = lbase, .ty = int_ty } }, int_ty);
    const llen_addr = try b.emit(.{ .field_addr = .{ .base = lbase, .off = 8, .ty = int_ty } }, int_ty);
    const ll = try b.emit(.{ .load = .{ .addr = llen_addr, .ty = int_ty } }, int_ty);
    const rp = try b.emit(.{ .load = .{ .addr = rbase, .ty = int_ty } }, int_ty);
    const rlen_addr = try b.emit(.{ .field_addr = .{ .base = rbase, .off = 8, .ty = int_ty } }, int_ty);
    const rl = try b.emit(.{ .load = .{ .addr = rlen_addr, .ty = int_ty } }, int_ty);

    // i := 0 in a slot (the loop induction var, mirroring `lowerStrEq`).
    const islot = try b.addSlot(int_ty);
    {
        const ia = try b.emit(.{ .slot_addr = islot }, int_ty);
        const zero = try b.emit(.{ .iconst = 0 }, int_ty);
        _ = try b.emit(.{ .store = .{ .addr = ia, .val = zero, .ty = int_ty } }, null);
    }

    const hdr = try b.addBlock();
    const hdr_r = try b.addBlock();
    const body = try b.addBlock();
    const body_gt = try b.addBlock();
    const inc = try b.addBlock();
    const lhs_done = try b.addBlock();
    const lt_out = try b.addBlock();
    const eq_out = try b.addBlock();
    const gt_out = try b.addBlock();
    const join = try b.addBlock();
    const merge = try b.addParam(join, int_ty);

    try brTo(b, hdr, .none);

    // hdr: lhs still has a byte at i ? -> hdr_r : lhs_done.
    b.switchTo(hdr);
    const ia_h = try b.emit(.{ .slot_addr = islot }, int_ty);
    const iv = try b.emit(.{ .load = .{ .addr = ia_h, .ty = int_ty } }, int_ty);
    const lhs_has = try b.emit(.{ .icmp = .{ .cc = .lt, .lhs = iv, .rhs = ll } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = lhs_has, .t = hdr_r, .f = lhs_done } });

    // hdr_r: rhs still has a byte at i ? -> body : gt_out (rhs exhausted first -> lhs greater).
    b.switchTo(hdr_r);
    const rhs_has = try b.emit(.{ .icmp = .{ .cc = .lt, .lhs = iv, .rhs = rl } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = rhs_has, .t = body, .f = gt_out } });

    // body: lb < rb ? -> lt_out : body_gt.
    b.switchTo(body);
    const lx = try b.emit(.{ .add = .{ .lhs = lp, .rhs = iv } }, int_ty);
    const lbyte = try b.emit(.{ .load_byte = lx }, int_ty);
    const rx = try b.emit(.{ .add = .{ .lhs = rp, .rhs = iv } }, int_ty);
    const rbyte = try b.emit(.{ .load_byte = rx }, int_ty);
    const lt_byte = try b.emit(.{ .icmp = .{ .cc = .lt, .lhs = lbyte, .rhs = rbyte } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = lt_byte, .t = lt_out, .f = body_gt } });

    // body_gt: lb > rb ? -> gt_out : inc (bytes equal, keep scanning).
    b.switchTo(body_gt);
    const gt_byte = try b.emit(.{ .icmp = .{ .cc = .gt, .lhs = lbyte, .rhs = rbyte } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = gt_byte, .t = gt_out, .f = inc } });

    // inc: i += 1; back-edge to hdr.
    b.switchTo(inc);
    {
        const ia = try b.emit(.{ .slot_addr = islot }, int_ty);
        const cur = try b.emit(.{ .load = .{ .addr = ia, .ty = int_ty } }, int_ty);
        const one = try b.emit(.{ .iconst = 1 }, int_ty);
        const next = try b.emit(.{ .add = .{ .lhs = cur, .rhs = one } }, int_ty);
        const ia2 = try b.emit(.{ .slot_addr = islot }, int_ty);
        _ = try b.emit(.{ .store = .{ .addr = ia2, .val = next, .ty = int_ty } }, null);
    }
    try brTo(b, hdr, .none);

    // lhs_done (lhs exhausted): rhs still has a byte ? -> lt_out (lhs shorter) : eq_out (equal).
    b.switchTo(lhs_done);
    const ia_d = try b.emit(.{ .slot_addr = islot }, int_ty);
    const iv_d = try b.emit(.{ .load = .{ .addr = ia_d, .ty = int_ty } }, int_ty);
    const rhs_left = try b.emit(.{ .icmp = .{ .cc = .lt, .lhs = iv_d, .rhs = rl } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = rhs_left, .t = lt_out, .f = eq_out } });

    // lt_out / eq_out / gt_out: deliver the discriminant to the join.
    b.switchTo(lt_out);
    const ltv = try b.emit(.{ .iconst = ord_lt }, int_ty);
    try brTo(b, join, .{ .value = ltv });

    b.switchTo(eq_out);
    const eqv = try b.emit(.{ .iconst = ord_eq }, int_ty);
    try brTo(b, join, .{ .value = eqv });

    b.switchTo(gt_out);
    const gtv = try b.emit(.{ .iconst = ord_gt }, int_ty);
    try brTo(b, join, .{ .value = gtv });

    b.switchTo(join);
    return merge;
}

/// Heap-free str equality: compare lengths, then bytes at `ptr + i` via `load_byte`
/// in a slot-counter loop, merging the bool result through a join block param. Built with
/// the exact `lowerFor` (slot counter) + `lowerAndOrValue` (br-arg block-param join) idioms
/// so it is a pure function of source (deterministic, `--verify`-stable). Returns the bool
/// result value (the join's merge param). An empty string compares equal to another empty
/// string (equal lengths -> `hdr` sees `i >= 0` immediately -> `eq_blk`).
fn lowerStrEq(b: *Builder, lhs_node: Ast.Index, rhs_node: Ast.Index) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";

    // Evaluate both str operands in the current block; each travels by slot.
    const lhs_op = try lowerExpr(b, lhs_node);
    const rhs_op = try lowerExpr(b, rhs_node);
    const ls = operandSlot(lhs_op);
    const rs = operandSlot(rhs_op);
    if (ls == Ir.none_slot or rs == Ir.none_slot) {
        try b.note(b.in.tree.nodes[(lhs_node).int()].main_token, "str '==' operand is not a slot in lower");
        return try b.emit(.{ .bconst = false }, bool_ty);
    }
    const lbase = try b.emit(.{ .slot_addr = ls }, int_ty);
    const rbase = try b.emit(.{ .slot_addr = rs }, int_ty);
    return strEqAtPtrs(b, lbase, rbase);
}

/// Heap-free str equality of two str aggregates given the ADDRESSES of their
/// `{ptr@0, len@8}` headers (factored out of `lowerStrEq`): compare
/// lengths, then bytes at `ptr + i` via `load_byte` in a slot-counter loop, merging the
/// bool through a join block param. Pure of source (slot-counter + br-arg joins), so
/// `--verify`-stable. Reused by the auto-derive emitter for a `str` FIELD (base =
/// `field_addr(self, off)`), where there is no slot to name — only an address.
pub fn strEqAtPtrs(b: *Builder, lbase: Ir.ValueId, rbase: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";

    // ptr@0 + len@8 of each {ptr,len} aggregate.
    const lp = try b.emit(.{ .load = .{ .addr = lbase, .ty = int_ty } }, int_ty);
    const llen_addr = try b.emit(.{ .field_addr = .{ .base = lbase, .off = 8, .ty = int_ty } }, int_ty);
    const ll = try b.emit(.{ .load = .{ .addr = llen_addr, .ty = int_ty } }, int_ty);
    const rp = try b.emit(.{ .load = .{ .addr = rbase, .ty = int_ty } }, int_ty);
    const rlen_addr = try b.emit(.{ .field_addr = .{ .base = rbase, .off = 8, .ty = int_ty } }, int_ty);
    const rl = try b.emit(.{ .load = .{ .addr = rlen_addr, .ty = int_ty } }, int_ty);
    const len_eq = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = ll, .rhs = rl } }, bool_ty);

    // i := 0 in a slot (the loop induction var, mirroring `lowerFor`).
    const islot = try b.addSlot(int_ty);
    {
        const ia = try b.emit(.{ .slot_addr = islot }, int_ty);
        const zero = try b.emit(.{ .iconst = 0 }, int_ty);
        _ = try b.emit(.{ .store = .{ .addr = ia, .val = zero, .ty = int_ty } }, null);
    }

    const hdr = try b.addBlock();
    const body = try b.addBlock();
    const inc = try b.addBlock();
    const eq_blk = try b.addBlock();
    const ne_blk = try b.addBlock();
    const join = try b.addBlock();
    const merge = try b.addParam(join, bool_ty);

    // Unequal lengths short-circuit to false; equal lengths enter the byte loop.
    b.setTerm(.{ .cond_br = .{ .cond = len_eq, .t = hdr, .f = ne_blk } });

    // hdr: i >= len ? all bytes matched (eq_blk) : compare byte i (body).
    b.switchTo(hdr);
    const ia_h = try b.emit(.{ .slot_addr = islot }, int_ty);
    const iv = try b.emit(.{ .load = .{ .addr = ia_h, .ty = int_ty } }, int_ty);
    const done = try b.emit(.{ .icmp = .{ .cc = .ge, .lhs = iv, .rhs = ll } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = done, .t = eq_blk, .f = body } });

    // body: load lp[i] and rp[i]; equal ? -> inc : ne_blk.
    b.switchTo(body);
    const lx = try b.emit(.{ .add = .{ .lhs = lp, .rhs = iv } }, int_ty);
    const lbyte = try b.emit(.{ .load_byte = lx }, int_ty);
    const rx = try b.emit(.{ .add = .{ .lhs = rp, .rhs = iv } }, int_ty);
    const rbyte = try b.emit(.{ .load_byte = rx }, int_ty);
    const byte_eq = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = lbyte, .rhs = rbyte } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = byte_eq, .t = inc, .f = ne_blk } });

    // inc: i += 1; back-edge to hdr.
    b.switchTo(inc);
    {
        const ia = try b.emit(.{ .slot_addr = islot }, int_ty);
        const cur = try b.emit(.{ .load = .{ .addr = ia, .ty = int_ty } }, int_ty);
        const one = try b.emit(.{ .iconst = 1 }, int_ty);
        const next = try b.emit(.{ .add = .{ .lhs = cur, .rhs = one } }, int_ty);
        const ia2 = try b.emit(.{ .slot_addr = islot }, int_ty);
        _ = try b.emit(.{ .store = .{ .addr = ia2, .val = next, .ty = int_ty } }, null);
    }
    try brTo(b, hdr, .none);

    // eq_blk / ne_blk: deliver the merge bool to the join.
    b.switchTo(eq_blk);
    const tv = try b.emit(.{ .bconst = true }, bool_ty);
    try brTo(b, join, .{ .value = tv });

    b.switchTo(ne_blk);
    const fv = try b.emit(.{ .bconst = false }, bool_ty);
    try brTo(b, join, .{ .value = fv });

    b.switchTo(join);
    return merge;
}

/// The emitted callee for a resolved conformance-witness `Method`: a derived unit's
/// synthetic name, a Mono instance's mangled name, else the fn's global
/// SymName. `derive` is checked FIRST (a derive Method has `fn_id == 0`, which would
/// otherwise mis-resolve to `names[0]`). Shared by `lowerStructEq` (top-level `==`) and
/// `structEqAtSlots` (an aggregate FIELD of a derived struct) so both pick the same
/// symbol lower's reloc + the fingerprint fold target.
pub fn witnessCallee(b: *Builder, m: Typecheck.Method) Link.SymName {
    if (m.derive) |di| return .{ .kind = .user_fn, .name = b.in.derives[di].name.? };
    if (m.instance) |ii| return .{ .kind = .user_fn, .name = b.in.instances[ii].name.? };
    return b.in.names[m.fn_id];
}

/// The ret type of a resolved conformance-witness `Method`: a source-less derive's recipe
/// `ret` (a derived `cmp`/`eq` witness has `fn_id == 0`, which would otherwise mis-read
/// `sigs[0].ret` and mis-size the ret_slot / `get_tag` layout), a Mono instance's ret, else
/// the fn's own sig ret. Shared by every witness CALL that must size a ret_slot from the
/// witness return type. `derive` is checked FIRST (mirroring `witnessCallee`).
pub fn witnessRet(b: *Builder, m: Typecheck.Method) Typecheck.Type {
    if (m.derive) |di| return b.in.derives[di].ret;
    if (m.instance) |ii| return b.in.instances[ii].ret;
    return b.in.sigs[m.fn_id].ret;
}

/// Heap-free per-byte hash of a str aggregate given the ADDRESS of its `{ptr@0, len@8}`
/// header: fold each byte through the fxhash mixer `h := (rotl(h,5) ^ byte) *% K` (see
/// `hashMix`) in a slot-counter loop, mirroring `strEqAtPtrs`'s `load_byte` walk (slot induction var +
/// slot accumulator). Pure of source (deterministic block/slot ids), so `--verify`-stable.
/// An empty string hashes to `str_hash_seed` (the loop runs zero times). Returns the int
/// hash value in the loop-exit block.
pub fn hashStrAtPtr(b: *Builder, base: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";

    // ptr@0 + len@8 of the {ptr,len} aggregate.
    const ptr = try b.emit(.{ .load = .{ .addr = base, .ty = int_ty } }, int_ty);
    const len_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = 8, .ty = int_ty } }, int_ty);
    const len = try b.emit(.{ .load = .{ .addr = len_addr, .ty = int_ty } }, int_ty);

    // h := str_hash_seed in a slot (the loop accumulator, mirroring the induction var).
    const hslot = try b.addSlot(int_ty);
    {
        const ha = try b.emit(.{ .slot_addr = hslot }, int_ty);
        const seed = try b.emit(.{ .iconst = str_hash_seed }, int_ty);
        _ = try b.emit(.{ .store = .{ .addr = ha, .val = seed, .ty = int_ty } }, null);
    }
    // i := 0 in a slot (mirrors `lowerFor` / `strEqAtPtrs`).
    const islot = try b.addSlot(int_ty);
    {
        const ia = try b.emit(.{ .slot_addr = islot }, int_ty);
        const zero = try b.emit(.{ .iconst = 0 }, int_ty);
        _ = try b.emit(.{ .store = .{ .addr = ia, .val = zero, .ty = int_ty } }, null);
    }

    const hdr = try b.addBlock();
    const body = try b.addBlock();
    const done = try b.addBlock();
    try brTo(b, hdr, .none);

    // hdr: i >= len ? done (all bytes folded) : fold byte i (body).
    b.switchTo(hdr);
    const ia_h = try b.emit(.{ .slot_addr = islot }, int_ty);
    const iv = try b.emit(.{ .load = .{ .addr = ia_h, .ty = int_ty } }, int_ty);
    const fin = try b.emit(.{ .icmp = .{ .cc = .ge, .lhs = iv, .rhs = len } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = fin, .t = done, .f = body } });

    // body: byte = load_byte(ptr + i); h = (rotl(h,5) ^ byte) *% K; i += 1; back-edge to hdr.
    b.switchTo(body);
    const bx = try b.emit(.{ .add = .{ .lhs = ptr, .rhs = iv } }, int_ty);
    const byte = try b.emit(.{ .load_byte = bx }, int_ty);
    const ha_b = try b.emit(.{ .slot_addr = hslot }, int_ty);
    const cur = try b.emit(.{ .load = .{ .addr = ha_b, .ty = int_ty } }, int_ty);
    const nh = try hashMix(b, cur, byte);
    const ha_s = try b.emit(.{ .slot_addr = hslot }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = ha_s, .val = nh, .ty = int_ty } }, null);
    const one = try b.emit(.{ .iconst = 1 }, int_ty);
    const next = try b.emit(.{ .add = .{ .lhs = iv, .rhs = one } }, int_ty);
    const ia_s = try b.emit(.{ .slot_addr = islot }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = ia_s, .val = next, .ty = int_ty } }, null);
    try brTo(b, hdr, .none);

    // done: the accumulated hash.
    b.switchTo(done);
    const ha_f = try b.emit(.{ .slot_addr = hslot }, int_ty);
    return try b.emit(.{ .load = .{ .addr = ha_f, .ty = int_ty } }, int_ty);
}

// The Display emitter and the `print(x)` dispatch share ONE raw write path: every
// literal (type/field/variant name + separators) and every `str` field is written by
// building a `str {ptr@0,len@8}` slot and calling the `print` builtin (write(1,ptr,len));
// an `int` calls the hand-asm `__display_int` builtin (stack-buffer decimal renderer); a
// `bool` inlines a `cond_br` over two literal writes. No allocator is ever referenced —
// only stack slots, cstring literals, and the `write` syscall.

/// The `print` builtin's stable symbol identity — a raw write-bytes primitive over a
/// `str {ptr,len}`. A comptime literal name is safe (codegen dupes callee names into
/// relocs; the Ir.Function only borrows it), matching `CodegenIr.lowerPrint`'s own sym.
const print_sym: Link.SymName = .{ .kind = .builtin, .name = "print" };
/// The `__display_int` builtin's stable symbol identity (the heap-free decimal renderer).
const display_int_sym: Link.SymName = .{ .kind = .builtin, .name = "__display_int" };
/// The `gc_alloc` builtin's stable symbol identity — returns a fresh zeroed cell pointer.
/// A `&x` box MINTS this call directly in lower; user source never names it, so this
/// bypasses the core-only allowlist by construction.
const gc_alloc_sym: Link.SymName = .{ .kind = .builtin, .name = "gc_alloc" };

/// Whether `t` is a managed box (`Ref[T]`/`gc_array[T]`): a struct carrying the reified
/// reference-family marker. A box is an 8-byte cell pointer — lowered as a scalar `int`,
/// NOT by its 8-byte struct field layout — so every scalar-value path guards on this.
pub fn isRefTy(b: *Builder, t: Typecheck.Type) bool {
    return t.kind == .@"struct" and t.struct_id < b.in.layouts.len and
        b.in.layouts[t.struct_id].native_family != .none;
}

/// Call the `print` builtin over the `str` slot `slot` — write its `{ptr,len}` bytes to
/// fd 1. The single shared raw write path (literals + `str` fields both route here).
pub fn emitPrintSlot(b: *Builder, slot: Ir.SlotId) error{OutOfMemory}!void {
    const args = try b.gpa.alloc(Ir.Operand, 1);
    errdefer b.gpa.free(args);
    args[0] = .{ .slot = slot };
    _ = try b.emit(.{ .call = .{ .callee = print_sym, .args = args, .ret_slot = Ir.none_slot } }, null);
}

/// Write a COMPILE-TIME byte string directly to fd 1: register it as a fn literal
/// (content-hash keyed, deduped), build a transient `str {ptr@0,len@8}` slot pointing at
/// it, and `print` those bytes. `bytes` is BORROWED (a layout name / a fixed separator);
/// it is duped into the owned literal table. An empty string is a no-op (no spurious call).
pub fn emitWriteLiteral(b: *Builder, bytes: []const u8) error{OutOfMemory}!void {
    if (bytes.len == 0) return;
    const int_ty = Typecheck.Type.int;
    const owned = try b.gpa.dupe(u8, bytes);
    const h = std.hash.Wyhash.hash(lit_seed, owned);
    try b.addLiteral(h, owned); // takes ownership of `owned` (frees a within-fn dup)
    const slot = try b.addSlot(Typecheck.Type.str);
    const base = try b.emit(.{ .slot_addr = slot }, int_ty);
    const p = try b.emit(.{ .cstr_ptr = h }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = base, .val = p, .ty = int_ty } }, null);
    const len_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = 8, .ty = int_ty } }, int_ty);
    const lenv = try b.emit(.{ .iconst = @intCast(bytes.len) }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = len_addr, .val = lenv, .ty = int_ty } }, null);
    try emitPrintSlot(b, slot);
}

/// Display an `int` VALUE by calling the hand-asm `__display_int` builtin. The value
/// travels in the first int-arg register; the builtin formats + writes it. Ret unit.
pub fn emitDisplayIntValue(b: *Builder, v: Ir.ValueId) error{OutOfMemory}!void {
    const args = try b.gpa.alloc(Ir.Operand, 1);
    errdefer b.gpa.free(args);
    args[0] = .{ .value = v };
    _ = try b.emit(.{ .call = .{ .callee = display_int_sym, .args = args, .ret_slot = Ir.none_slot } }, null);
}

/// Display a `bool` VALUE inline: `cond_br` on the value to a `true`/`false` literal
/// write, then join. Leaves the cursor at the join block so the caller keeps emitting.
pub fn emitDisplayBoolValue(b: *Builder, v: Ir.ValueId) error{OutOfMemory}!void {
    const t_blk = try b.addBlock();
    const f_blk = try b.addBlock();
    const join = try b.addBlock();
    b.setTerm(.{ .cond_br = .{ .cond = v, .t = t_blk, .f = f_blk } });
    b.switchTo(t_blk);
    try emitWriteLiteral(b, "true");
    if (!b.termSet()) try brTo(b, join, .none);
    b.switchTo(f_blk);
    try emitWriteLiteral(b, "false");
    if (!b.termSet()) try brTo(b, join, .none);
    b.switchTo(join);
}

/// Display an aggregate operand already MATERIALIZED into slot `slot`: resolve the
/// `display` witness and call `witness(slot) -> ()`, which writes the value's rendering to
/// fd 1. The slot-operand sibling of the top-level derive, so a nested aggregate FIELD
/// stays in lockstep with the callee's own derived unit. A miss is unreachable for a
/// conforming field (the synthesis barrier proved it) — note-and-drop rather than miscompile.
pub fn displayAtSlot(b: *Builder, ty: Typecheck.Type, slot: Ir.SlotId) error{OutOfMemory}!void {
    switch (Typecheck.resolveConformanceMethod(b.in.methods, ty, "display", b.in.prelude_ids.display, null)) {
        .one => |m| {
            const callee = witnessCallee(b, m);
            const args = try b.gpa.alloc(Ir.Operand, 1);
            errdefer b.gpa.free(args);
            args[0] = .{ .slot = slot };
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, null);
        },
        .none, .ambiguous => {
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "auto-derive Display: no display witness for an aggregate field in lower" });
            b.had_error = true;
        },
    }
}

fn shr(b: *Builder, v: Ir.ValueId, n: i64) error{OutOfMemory}!Ir.ValueId {
    const c = try b.emit(.{ .iconst = n }, Typecheck.Type.int);
    return b.emit(.{ .lshr = .{ .lhs = v, .rhs = c } }, Typecheck.Type.int);
}
fn andC(b: *Builder, v: Ir.ValueId, m: i64) error{OutOfMemory}!Ir.ValueId {
    const c = try b.emit(.{ .iconst = m }, Typecheck.Type.int);
    return b.emit(.{ .band = .{ .lhs = v, .rhs = c } }, Typecheck.Type.int);
}
fn orC(b: *Builder, v: Ir.ValueId, m: i64) error{OutOfMemory}!Ir.ValueId {
    const c = try b.emit(.{ .iconst = m }, Typecheck.Type.int);
    return b.emit(.{ .bor = .{ .lhs = v, .rhs = c } }, Typecheck.Type.int);
}
/// Fold bytes little-endian into one word: b0 | b1<<8 | b2<<16 | b3<<24.
fn packLE(b: *Builder, bytes: []const Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    var acc = bytes[0];
    var shift: i64 = 8;
    for (bytes[1..]) |y| {
        const sh = try b.emit(.{ .iconst = shift }, Typecheck.Type.int);
        const hi = try b.emit(.{ .shl = .{ .lhs = y, .rhs = sh } }, Typecheck.Type.int);
        acc = try b.emit(.{ .bor = .{ .lhs = acc, .rhs = hi } }, Typecheck.Type.int);
        shift += 8;
    }
    return acc;
}

/// Display a `char` VALUE (materialized in `slot`) by UTF-8-ENCODING its codepoint and
/// writing the <=4 bytes to fd 1 — the OVERRIDE body of char's derived Display witness
/// (`Display$display$s<id>`), replacing the structural `char(65)` tuple walk. Emitted ONCE
/// (in that shared unit) and CALLed from every char-display site; formerly inlined per site.
/// A 3-test band ladder
/// (cp<=0x7F / <=0x7FF / <=0xFFFF else 4-byte) packs the bytes LITTLE-ENDIAN into one word and
/// does ONE `.store` into an 8-byte scratch slot: `.store` is a 64-bit STR (no store_byte op),
/// so the high padding is written but never read — the `str` len is the true band width. The
/// codepoint is a proven-valid scalar (the decode/try_into gates: 0..0x10FFFF, no surrogates)
/// AND the char literal's 64-bit store zeroed the slot's high 4 bytes, so the loaded value has
/// bit63 clear: unsigned band tests need no validity branch. Pure IR (fixed block/value ids,
/// reads no map) → -jN- and O0≡O1-stable. Leaves the cursor at the join block.
pub fn lowerCharDisplay(b: *Builder, slot: Ir.SlotId) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";

    const cp_base = try b.emit(.{ .slot_addr = slot }, int_ty);
    const cp = try b.emit(.{ .load = .{ .addr = cp_base, .ty = Typecheck.Type.uint32 } }, int_ty);

    const byte_slot = try b.addSlot(int_ty); // 8-byte scratch for the packed LE word
    const b0_addr = try b.emit(.{ .slot_addr = byte_slot }, int_ty);

    const band1 = try b.addBlock();
    const test2 = try b.addBlock();
    const band2 = try b.addBlock();
    const test3 = try b.addBlock();
    const band3 = try b.addBlock();
    const band4 = try b.addBlock();
    const join = try b.addBlock();
    const len = try b.addParam(join, int_ty);

    // Unsigned band tests: a codepoint is an unsigned magnitude. The loaded value has bit63
    // clear, so .ule == .le here; .ule states the domain honestly and is width-robust.
    const c7f = try b.emit(.{ .iconst = 0x7F }, int_ty);
    const is1 = try b.emit(.{ .icmp = .{ .cc = .ule, .lhs = cp, .rhs = c7f } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = is1, .t = band1, .f = test2 } });

    // 1 byte: cp
    b.switchTo(band1);
    _ = try b.emit(.{ .store = .{ .addr = b0_addr, .val = cp, .ty = int_ty } }, null);
    try brTo(b, join, .{ .value = try b.emit(.{ .iconst = 1 }, int_ty) });

    b.switchTo(test2);
    const c7ff = try b.emit(.{ .iconst = 0x7FF }, int_ty);
    const is2 = try b.emit(.{ .icmp = .{ .cc = .ule, .lhs = cp, .rhs = c7ff } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = is2, .t = band2, .f = test3 } });

    // 2 bytes: 0xC0|(cp>>6), 0x80|(cp&0x3F)
    b.switchTo(band2);
    {
        const y0 = try orC(b, try shr(b, cp, 6), 0xC0);
        const y1 = try orC(b, try andC(b, cp, 0x3F), 0x80);
        _ = try b.emit(.{ .store = .{ .addr = b0_addr, .val = try packLE(b, &.{ y0, y1 }), .ty = int_ty } }, null);
        try brTo(b, join, .{ .value = try b.emit(.{ .iconst = 2 }, int_ty) });
    }

    b.switchTo(test3);
    const cffff = try b.emit(.{ .iconst = 0xFFFF }, int_ty);
    const is3 = try b.emit(.{ .icmp = .{ .cc = .ule, .lhs = cp, .rhs = cffff } }, bool_ty);
    b.setTerm(.{ .cond_br = .{ .cond = is3, .t = band3, .f = band4 } });

    // 3 bytes: 0xE0|(cp>>12), 0x80|((cp>>6)&0x3F), 0x80|(cp&0x3F)
    b.switchTo(band3);
    {
        const y0 = try orC(b, try shr(b, cp, 12), 0xE0);
        const y1 = try orC(b, try andC(b, try shr(b, cp, 6), 0x3F), 0x80);
        const y2 = try orC(b, try andC(b, cp, 0x3F), 0x80);
        _ = try b.emit(.{ .store = .{ .addr = b0_addr, .val = try packLE(b, &.{ y0, y1, y2 }), .ty = int_ty } }, null);
        try brTo(b, join, .{ .value = try b.emit(.{ .iconst = 3 }, int_ty) });
    }

    // 4 bytes: 0xF0|(cp>>18), 0x80|((cp>>12)&0x3F), 0x80|((cp>>6)&0x3F), 0x80|(cp&0x3F)
    b.switchTo(band4);
    {
        const y0 = try orC(b, try shr(b, cp, 18), 0xF0);
        const y1 = try orC(b, try andC(b, try shr(b, cp, 12), 0x3F), 0x80);
        const y2 = try orC(b, try andC(b, try shr(b, cp, 6), 0x3F), 0x80);
        const y3 = try orC(b, try andC(b, cp, 0x3F), 0x80);
        _ = try b.emit(.{ .store = .{ .addr = b0_addr, .val = try packLE(b, &.{ y0, y1, y2, y3 }), .ty = int_ty } }, null);
        try brTo(b, join, .{ .value = try b.emit(.{ .iconst = 4 }, int_ty) });
    }

    // join: build a runtime str{ptr=&byte_slot, len} and print exactly `len` bytes.
    b.switchTo(join);
    const str_slot = try b.addSlot(Typecheck.Type.str);
    const sbase = try b.emit(.{ .slot_addr = str_slot }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = sbase, .val = b0_addr, .ty = int_ty } }, null);
    const len_addr = try b.emit(.{ .field_addr = .{ .base = sbase, .off = 8, .ty = int_ty } }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = len_addr, .val = len, .ty = int_ty } }, null);
    try emitPrintSlot(b, str_slot);
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
    // A method call `recv.m(args)`: dispatch to the method's mangled symbol and
    // PREPEND the receiver as the `self` arg (arg 0). `self_recv` set ⟺ this is a
    // method call; the receiver expr is lowered as arg 0 below.
    var self_recv: Ast.Index = Ast.none;
    // A `mut self` method: pass the receiver's ADDRESS (a place) as arg 0 instead
    // of a by-value copy, so the callee mutates the caller's storage.
    var self_mut = false;
    if (methodGidOf(b, n)) |m| {
        // Method dispatch: callee = the method's mangled global symbol; the
        // receiver is prepended as `self` in the arg build below. TRIED FIRST so an
        // explicit-protocol-args method call `v.into[int]()` (a `type_app` over a
        // `field_access`) is not misread as a generic FUNCTION call by the `type_app`
        // branch below. The receiver is the field_access's lhs — reached through the
        // `type_app` for the explicit-args shape, else the callee (field_access) directly.
        // A method on a GENERIC-type instance: the resolver returns the
        // reified-dispatch entry carrying the mono `instance` index — dispatch to THAT
        // instance's mangled symbol (its own per-instance codegen unit), not the
        // never-lowered template's `names[fn_id]`. A SOURCE-LESS derive Method (
        // `fn_id == 0`) dispatches to its synthetic unit — a DIRECT `.hash()` call is
        // the first derive method reached here (Eq/Ord fire only via operators), so use the
        // shared `witnessCallee` (derive-first) rather than `names[m.fn_id]` (would be
        // `names[0]` — a wrong-symbol miscompile).
        callee = witnessCallee(b, m);
        self_recv = if (callee_node.tag == .type_app) b.in.tree.nodes[(callee_node.lhs).int()].lhs else callee_node.lhs;
        self_mut = m.mut_self;
    } else if (callee_node.tag == .type_app) {
        // `size_of[T]()` / `align_of[T]()`: the base resolves to a core-only builtin.
        // Fold to a compile-time ABI constant (no call, no reloc). Checked BEFORE the
        // generic-instance resolution below.
        {
            const bres = b.in.resolutions[(callee_node.lhs).int()];
            if (bres == .func and bres.func < b.in.names.len and b.in.names[bres.func].kind == .builtin) {
                const bn = b.in.names[bres.func].name;
                const bik = Intrinsic.lookup(bn);
                if (bik == .size_of or bik == .align_of) {
                    const tnodes = Ast.rangeSlice(b.in.tree, (callee_node.rhs).int());
                    const t = b.in.node_types[(tnodes[0]).int()];
                    const c: u32 = if (bik == .size_of)
                        Abi.typeSize(t, b.in.layouts, b.in.enum_layouts)
                    else
                        Abi.typeAlign(t, b.in.layouts, b.in.enum_layouts);
                    return .{ .value = try b.emit(.{ .iconst = @intCast(c) }, Typecheck.Type.int) };
                }
            }
        }
        // A generic call `id[int](..)`: the callee is a `type_app` whose base
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
        callee = .{ .kind = .user_fn, .name = b.in.instances[ii].name.? };
    } else if (builtinScalarEqCallee(b, n)) |ba| {
        // A builtin scalar `.eq()`. int/bool lower to an inline `icmp eq` (no
        // `.call`, no symbol/reloc — the recognizer is pure). str/unit route through the
        // SAME `lowerEqValue` the `==` operator uses (heap-free byte-compare / trivially
        // -true bconst), so `a.eq(b)` and `a == b` emit identical IR.
        const recv_ty = b.in.node_types[(ba.recv).int()];
        if (isInlineEq(recv_ty.kind)) {
            const lhs = operandValue(try lowerExpr(b, ba.recv));
            const rhs = operandValue(try lowerExpr(b, ba.arg));
            return .{ .value = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.@"bool") };
        }
        return try lowerEqValue(b, recv_ty, ba.recv, ba.arg, false);
    } else if (builtinScalarHashCallee(b, n)) |bh| {
        // A builtin scalar `.hash()` (completeness layer so `[T has Hash]` works at a
        // scalar T). No `.call`/symbol — like the scalar `.eq()`, the recognizer is pure:
        // int/bool hash to their own VALUE (identity — an int is its own hash, a bool is
        // 0/1); str folds its bytes via the SAME heap-free polynomial the derive uses
        // (`hashStrAtPtr`); unit hashes to the fixed seed (the sole `()` value).
        const recv_ty = b.in.node_types[(bh.recv).int()];
        switch (recv_ty.kind) {
            .int, .bool => return .{ .value = operandValue(try lowerExpr(b, bh.recv)) },
            .str => {
                const op = try lowerExpr(b, bh.recv);
                const s = operandSlot(op);
                if (s == Ir.none_slot) {
                    try b.note(callee_node.main_token, "str '.hash()' operand is not a slot in lower");
                    return .{ .value = try b.emit(.{ .iconst = 0 }, Typecheck.Type.int) };
                }
                const base = try b.emit(.{ .slot_addr = s }, Typecheck.Type.int);
                return .{ .value = try hashStrAtPtr(b, base) };
            },
            else => {
                _ = try lowerExpr(b, bh.recv); // unit: evaluate for effect, hash a constant
                return .{ .value = try b.emit(.{ .iconst = hash_seed }, Typecheck.Type.int) };
            },
        }
    } else if (optionResultMethodCallee(b, n)) |om| {
        // A native inherent method on a reified `Option`/`Result` instance: no
        // `.call`/symbol — inline the tag test / payload load per the reified layout.
        return try lowerOptionResultMethod(b, n, om);
    } else if (builtinConvCallee(b, n)) |cv| {
        // A target-directed `.into()`/`.try_into()` int conversion: no `.call`/symbol —
        // inline the mask/extend or the range-checked `Result` build.
        return try lowerConvMethod(b, node_idx, cv);
    } else {
        // The callee identifier resolves to a `.func` index into `names` (this also
        // covers the `print` builtin, whose name index points at the synthetic entry).
        const callee_res = b.in.resolutions[(n.lhs).int()];
        if (callee_res != .func) {
            try b.note(n.main_token, "call target unsupported in lower");
            return .none;
        }
        // The `print` builtin is a compiler-magic polymorphic dispatch by the single
        // arg's type: `str` keeps the raw write-bytes path (falls through below); `int`/
        // `bool` render inline (heap-free) and a struct/enum routes to its resolved `Display`
        // witness. The checker already required the arg to conform to `Display`, so a
        // struct/enum witness always resolves. Intercept BEFORE the plain-name resolution.
        if (callee_res.func < b.in.names.len) {
            const nm = b.in.names[callee_res.func];
            // Raw-pointer `store`/`load`: lower directly to the IR memory ops (no call,
            // no reloc). The checker fenced them in `unsafe { }` and typed the rawptr
            // operand; this milestone reads/writes a 64-bit int through the pointer.
            if (nm.kind == .builtin) if (Intrinsic.lookup(nm.name)) |ik| switch (ik) {
                .store => {
                    const sargs = Ast.rangeSlice(b.in.tree, (n.rhs).int());
                    const addr = operandValue(try lowerExpr(b, sargs[0]));
                    const val = operandValue(try lowerExpr(b, sargs[1]));
                    _ = try b.emit(.{ .store = .{ .addr = addr, .val = val, .ty = Typecheck.Type.int } }, null);
                    return .none;
                },
                .load => {
                    const largs = Ast.rangeSlice(b.in.tree, (n.rhs).int());
                    const addr = operandValue(try lowerExpr(b, largs[0]));
                    return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = Typecheck.Type.int } }, Typecheck.Type.int) };
                },
                else => {},
            };
            if (nm.kind == .builtin and std.mem.eql(u8, nm.name, "print")) {
                const parg = Ast.rangeSlice(b.in.tree, (n.rhs).int());
                if (parg.len == 1) {
                    const at = b.in.node_types[(parg[0]).int()];
                    switch (at.kind) {
                        .int => {
                            try emitDisplayIntValue(b, operandValue(try lowerExpr(b, parg[0])));
                            return .none;
                        },
                        .bool => {
                            try emitDisplayBoolValue(b, operandValue(try lowerExpr(b, parg[0])));
                            return .none;
                        },
                        .@"struct", .@"enum" => {
                            const op = try lowerExpr(b, parg[0]);
                            const slot = operandSlot(op);
                            if (slot == Ir.none_slot) {
                                try b.note(callee_node.main_token, "print of a struct/enum: arg is not a slot in lower");
                                return .none;
                            }
                            try displayAtSlot(b, at, slot);
                            return .none;
                        },
                        .unit => {
                            // `print(unit)` renders `()` — evaluate the arg for its effects
                            // (e.g. a unit-returning call), then write the literal. A unit
                            // operand must NEVER reach the raw `str` print path (no {ptr,len}).
                            _ = try lowerExpr(b, parg[0]);
                            try emitWriteLiteral(b, "()");
                            return .none;
                        },
                        else => {}, // str: the raw write-bytes builtin — fall through.
                    }
                }
            }
        }
        if (callee_node.tag == .identifier and callee_res.func < b.in.sigs.len and sigHasTypeVar(b.in.sigs[callee_res.func])) {
            // A bare inferred generic call `id(7)`: the plain-identifier callee
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
            callee = .{ .kind = .user_fn, .name = b.in.instances[ii].name.? };
        } else {
            callee = b.in.names[callee_res.func];
        }
    }
    const result_ty = b.in.node_types[(node_idx).int()];

    // Evaluate every arg left-to-right into an Operand (scalar→value, str→slot). For a
    // method call the receiver is the FIRST arg (`self`, by value — reusing the
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

    // A managed-box result is an 8-byte scalar cell pointer (returned in a register),
    // not an aggregate — read it as a scalar value, no ret_slot.
    if (isRefTy(b, result_ty)) {
        const v = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, Typecheck.Type.int);
        return .{ .value = v };
    }
    // Result placement: scalar → an Instr.result value; aggregate → a fresh
    // ret_slot; unit → neither. The ABI (reg vs sret) is decided in codegen.
    switch (result_ty.kind) {
        .int, .bool, .float, .rawptr => {
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
    // A native `Option`/`Result` method returning an enum payload (agg-T `unwrap`) ALSO
    // parses as `.call` over an unresolved field_access and has no `t.methods` entry, so
    // exclude it explicitly — else it misroutes as a variant construction. Scalar-T
    // is unaffected (its result is int/bool, so `ty.kind != .@"enum"` short-circuits).
    // A `.try_into()` returning `Result[T, ConvErr]` ALSO parses as `.call` over an
    // unresolved field_access with no `t.methods` entry and is not an
    // `optionResultMethodCallee`, so exclude it too — else it misroutes as a variant
    // construction (silent miscompile).
    return ty.kind == .@"enum" and b.in.tree.nodes[(n.lhs).int()].tag == .field_access and
        b.in.resolutions[(n.lhs).int()] != .func and methodGidOf(b, n) == null and
        optionResultMethodCallee(b, n) == null and builtinConvCallee(b, n) == null;
}

/// A tuple-struct constructor call `N(args)` in lower. A struct-typed `.call` over an
/// UNRESOLVED identifier callee is necessarily a validated tuple ctor (the checker
/// rejected record `N(..)`/arity/type errors and gated codegen). A struct-returning fn
/// is `.func`; a method has a field_access callee; a record uses `N{..}`; a value binding
/// shadowing the type name resolves `.local` and was already poisoned by the checker.
fn isTupleStructCtorCall(b: *Builder, n: Ast.Node, ty: Typecheck.Type) bool {
    return ty.kind == .@"struct" and b.in.tree.nodes[(n.lhs).int()].tag == .identifier and
        b.in.resolutions[(n.lhs).int()] == .unresolved;
}

/// Write a `Name(v0, v1, ..)` tuple-struct construction into `dst_ptr`: each positional
/// arg at `layout.offsets[i]` (declaration order). Positional twin of `lowerStructInitInto`.
fn lowerTupleStructInitInto(b: *Builder, node_idx: Ast.Index, dst_ptr: Ir.ValueId) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(node_idx).int()];
    const layout = b.in.layouts[b.in.node_types[(node_idx).int()].struct_id];
    const args = Ast.rangeSlice(b.in.tree, (n.rhs).int());
    std.debug.assert(args.len == layout.field_types.len); // guaranteed by the check-error gate
    for (args, 0..) |arg, i| {
        const faddr = try b.emit(.{ .field_addr = .{ .base = dst_ptr, .off = layout.offsets[i], .ty = layout.field_types[i] } }, Typecheck.Type.int);
        try lowerExprInto(b, arg, faddr, layout.field_types[i]);
    }
}

/// The `Method` a method call `recv.m(args)` dispatches to, or null when `n` is
/// not a method call. A method callee is a `field_access` NOT bound to a `.func`
/// (that is a qualified module call) whose receiver types to a concrete struct/enum
/// with a matching entry in the method table. A pure content-keyed lookup (no
/// hashmap/thread order), so it is identical at any `-jN`. Returns the whole
/// `Method` (not just `fn_id`) so the caller reads `mut_self` for the by-address
/// receiver ABI; `null`-semantics are unchanged for the ctor-call classifier.
fn methodGidOf(b: *Builder, n: Ast.Node) ?Typecheck.Method {
    // Two callee shapes are method dispatch: a bare `field_access` `v.m` (explicit args
    // null), and an explicit-protocol-args `type_app` over a `field_access` `v.m[int]`
    //. A qualified fn / qualified generic fn binds its field_access to `.func` and
    // is NOT a method — exclude both shapes on that.
    var cn = b.in.tree.nodes[(n.lhs).int()];
    var fa_idx = n.lhs;
    var explicit_args: ?[]const Typecheck.Type = null;
    var explicit_buf: [8]Typecheck.Type = undefined;
    if (cn.tag == .type_app) {
        const inner_idx = cn.lhs;
        if (b.in.tree.nodes[(inner_idx).int()].tag != .field_access) return null;
        if (b.in.resolutions[(inner_idx).int()] == .func) return null; // qualified generic fn
        const targ_nodes = Ast.rangeSlice(b.in.tree, (cn.rhs).int());
        if (targ_nodes.len > explicit_buf.len) return null; // defensive; a well-typed call is small
        for (targ_nodes, 0..) |tn, i| explicit_buf[i] = b.in.node_types[(tn).int()];
        explicit_args = explicit_buf[0..targ_nodes.len];
        fa_idx = inner_idx;
        cn = b.in.tree.nodes[(inner_idx).int()];
    } else {
        if (cn.tag != .field_access) return null;
        if (b.in.resolutions[(n.lhs).int()] == .func) return null;
    }
    const recv = b.in.node_types[(cn.lhs).int()];
    // Dispatch through the real Method path for any nominal OR builtin scalar receiver
    //: a user `impl int has P` registered a real `fn_id` on `recv = Type.int`, so
    // the resolver selects it. A builtin scalar `eq` has NO method entry (the recognizer
    // is pure) → this misses → `lowerCall`'s `builtinScalarEqCallee` branch fires. Reject
    // only the non-dispatchable kinds (invalid/never/type_var/app never reach lower).
    switch (recv.kind) {
        .@"struct", .@"enum", .int, .bool, .str, .unit => {},
        else => return null,
    }
    const member = b.in.tokens[cn.main_token].text(b.in.source);
    // The SAME multi-conformance disambiguation the checker + fingerprint use, so all
    // three select the identical witness (a divergence would be a miscompile or `-jN`
    // break). `.ambiguous`/`.none` -> not dispatchable here (the checker already erred).
    const m = switch (Typecheck.resolveConformanceMethod(b.in.methods, recv, member, null, explicit_args)) {
        .one => |mm| mm,
        else => return null,
    };
    // Defense in depth: a `mut self` method on a builtin scalar is rejected at check
    // time (T0022) because the by-address self ABI has no write-back path. Never
    // dispatch one here so a stray lower can't turn the receiver's slot address into
    // the callee's value slot; a well-typed program never reaches this.
    if (m.mut_self and recv.kind != .@"struct" and recv.kind != .@"enum") return null;
    return m;
}

/// A builtin scalar `eq` call `recv.eq(arg)`: the callee is a `field_access` NOT
/// bound to a `.func`, the receiver types to a scalar the recognizer accepts, and there
/// is exactly one arg (the checker already gated arity). Returns the receiver + arg
/// nodes so `lowerCall` can emit an inline `icmp eq` (no `.call`, no external symbol, so
/// nothing folds into a reloc — pure a-function-of-source, `--verify`-stable). Null when
/// `n` is not such a call (so `lowerCall` falls through to its normal resolution).
fn builtinScalarEqCallee(b: *Builder, n: Ast.Node) ?struct { recv: Ast.Index, arg: Ast.Index } {
    const cn = b.in.tree.nodes[(n.lhs).int()];
    if (cn.tag != .field_access or b.in.resolutions[(n.lhs).int()] == .func) return null;
    const member = b.in.tokens[cn.main_token].text(b.in.source);
    if (Typecheck.builtinScalarMethod(b.in.node_types[(cn.lhs).int()], member) == null) return null;
    const args = Ast.rangeSlice(b.in.tree, (n.rhs).int());
    if (args.len != 1) return null;
    return .{ .recv = cn.lhs, .arg = args[0] };
}

/// A builtin scalar `.hash()` call `recv.hash()`: the callee is a `field_access` NOT
/// bound to a `.func`, the member is `hash`, the receiver types to a scalar the recognizer
/// accepts, and there are zero args. Returns the receiver node so `lowerCall` can emit the
/// identity value / byte polynomial / constant inline (no `.call`, no external symbol —
/// pure a-function-of-source, `--verify`-stable). Null when `n` is not such a call (a
/// struct/enum `.hash()` was already dispatched to its derive unit by `methodGidOf`).
fn builtinScalarHashCallee(b: *Builder, n: Ast.Node) ?struct { recv: Ast.Index } {
    const cn = b.in.tree.nodes[(n.lhs).int()];
    if (cn.tag != .field_access or b.in.resolutions[(n.lhs).int()] == .func) return null;
    const member = b.in.tokens[cn.main_token].text(b.in.source);
    if (!std.mem.eql(u8, member, "hash")) return null;
    const bm = Typecheck.builtinScalarMethod(b.in.node_types[(cn.lhs).int()], member) orelse return null;
    if (bm.arity != 0) return null;
    if (Ast.rangeSlice(b.in.tree, (n.rhs).int()).len != 0) return null;
    return .{ .recv = cn.lhs };
}

/// A target-directed `.into()` / `.try_into()` conversion call on an integer receiver:
/// the callee is a `field_access` NOT bound to a `.func`, the receiver types to an
/// integer, the member is `into`/`try_into`, and there are zero args (the checker already
/// gated arity + the target). Returns the receiver node + member so `lowerCall` can emit
/// the inline mask/extend (`into`) or the range-checked `Result` build (`try_into`) — no
/// `.call`, no external symbol, so it is a pure function of source (`--verify`-stable).
/// Null when `n` is not such a call. A user `protocol Into` on a STRUCT never matches
/// (integer-receiver guard), so it stays on the normal method-dispatch path.
const ConvCall = struct { recv: Ast.Index, member: []const u8 };

fn builtinConvCallee(b: *Builder, n: Ast.Node) ?ConvCall {
    const cn = b.in.tree.nodes[(n.lhs).int()];
    if (cn.tag != .field_access or b.in.resolutions[(n.lhs).int()] == .func) return null;
    const recv_ty = b.in.node_types[(cn.lhs).int()];
    if (!recv_ty.isInteger() and !isCharTy(b, recv_ty) and recv_ty.kind != .float) return null;
    const member = b.in.tokens[cn.main_token].text(b.in.source);
    if (!std.mem.eql(u8, member, "into") and !std.mem.eql(u8, member, "try_into")) return null;
    if (Ast.rangeSlice(b.in.tree, (n.rhs).int()).len != 0) return null;
    return .{ .recv = cn.lhs, .member = member };
}

/// Whether `ty` is the compiler-provided `char` struct (the id the checker used, threaded
/// via `Inputs.char_struct`). Null (a prelude-less test caller) means no type is char.
pub fn isCharTy(b: *const Builder, ty: Typecheck.Type) bool {
    return Typecheck.isCharTy(ty, b.in.char_struct);
}

/// A native inherent method call on a reified `Option`/`Result` instance: the
/// callee is a `field_access` NOT bound to a `.func`, the receiver types to a reified
/// enum whose `native_family` is set (the robust per-instance key — a user `enum Option`
/// mangles to the same `Option$int` yet has `.none`), and the member is one of the 8
/// names. Returns the receiver node + the recognized op so `lowerCall` can inline the
/// tag test / payload load (no `.call`, no external symbol — pure a-function-of-source,
/// `--verify`-stable). Null when `n` is not such a call.
const OptResultCall = struct { recv: Ast.Index, op: Typecheck.OptResultMethod };

fn optionResultMethodCallee(b: *Builder, n: Ast.Node) ?OptResultCall {
    const cn = b.in.tree.nodes[(n.lhs).int()];
    if (cn.tag != .field_access or b.in.resolutions[(n.lhs).int()] == .func) return null;
    const recv_ty = b.in.node_types[(cn.lhs).int()];
    if (recv_ty.kind != .@"enum" or recv_ty.enum_id >= b.in.enum_layouts.len) return null;
    const fam = b.in.enum_layouts[recv_ty.enum_id].native_family;
    const member = b.in.tokens[cn.main_token].text(b.in.source);
    const op = Typecheck.optionResultMethod(fam, member) orelse return null;
    return .{ .recv = cn.lhs, .op = op };
}

/// Inline a native `Option`/`Result` method. The receiver is materialized into a
/// fresh slot (so a temporary construction receiver like `Option[int].some(40).unwrap()`
/// works, not just a local), giving a base ptr for `get_tag` + the payload load. Variant
/// 0 is the payload variant, 1 the absence/error (fixed by `registerPrelude`), so
/// `tag == 0` means present. `is_some`/`is_ok` = `tag == 0`; `is_none`/`is_err` =
/// `tag == 1`; `unwrap` cond-branches to a `.trap` on absence; `unwrap_or` merges the
/// payload / the default through a join block-arg. Scalar-payload T only (the accept
/// set is int/bool); aggregate-payload T is deferred (the classifiers below already
/// exclude native calls so it won't misroute as a variant construction).
fn lowerOptionResultMethod(b: *Builder, n: Ast.Node, om: OptResultCall) error{OutOfMemory}!Ir.Operand {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    const recv_ty = b.in.node_types[(om.recv).int()];
    const e = b.in.enum_layouts[recv_ty.enum_id];

    // The native path ships scalar-payload `unwrap`/`unwrap_or` only: a non-scalar payload would
    // truncate a str/aggregate to one 8-byte load, and an aggregate join block-arg
    // crashes codegen (unassigned value offset). The checker already rejects this with
    // T0018, so this is a defensive clean-fail should an aggregate reach `lower`.
    switch (om.op) {
        .unwrap, .unwrap_or => {
            const pty = e.variants[0].field_types[0];
            // A managed-box payload is an 8-byte scalar cell pointer, so it rides the
            // scalar unwrap path (an int-typed load + int block-arg) like int/bool.
            if (pty.kind != .int and pty.kind != .bool and !isRefTy(b, pty)) {
                try b.note(n.main_token, "unwrap on a non-scalar Option/Result payload is not yet supported");
                return .none;
            }
        },
        else => {},
    }

    const slot = try b.addSlot(recv_ty);
    const base = try b.emit(.{ .slot_addr = slot }, int_ty);
    try lowerExprInto(b, om.recv, base, recv_ty);
    const tag = try b.emit(.{ .get_tag = base }, int_ty);

    switch (om.op) {
        .is_tag0, .is_tag1 => {
            const k: i64 = if (om.op == .is_tag0) 0 else 1;
            const kv = try b.emit(.{ .iconst = k }, int_ty);
            return .{ .value = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = tag, .rhs = kv } }, bool_ty) };
        },
        .unwrap => {
            const payload_ty = e.variants[0].field_types[0];
            // A managed-box payload joins as an 8-byte scalar (int block-arg); a struct
            // block-arg would crash codegen (unassigned value offset).
            const merge_ty = if (isRefTy(b, payload_ty)) int_ty else payload_ty;
            const zero = try b.emit(.{ .iconst = 0 }, int_ty);
            const present = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = tag, .rhs = zero } }, bool_ty);
            const ok_blk = try b.addBlock();
            const trap_blk = try b.addBlock();
            const join = try b.addBlock();
            const merge = try b.addParam(join, merge_ty);
            b.setTerm(.{ .cond_br = .{ .cond = present, .t = ok_blk, .f = trap_blk } });
            b.switchTo(ok_blk);
            try brTo(b, join, .{ .value = try loadNativePayload(b, e, base) });
            b.switchTo(trap_blk);
            b.setTerm(.trap);
            b.switchTo(join);
            return .{ .value = merge };
        },
        .unwrap_or => {
            const payload_ty = e.variants[0].field_types[0];
            const merge_ty = if (isRefTy(b, payload_ty)) int_ty else payload_ty;
            const args = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const zero = try b.emit(.{ .iconst = 0 }, int_ty);
            const present = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = tag, .rhs = zero } }, bool_ty);
            const some_blk = try b.addBlock();
            const none_blk = try b.addBlock();
            const join = try b.addBlock();
            const merge = try b.addParam(join, merge_ty);
            b.setTerm(.{ .cond_br = .{ .cond = present, .t = some_blk, .f = none_blk } });
            b.switchTo(some_blk);
            try brTo(b, join, .{ .value = try loadNativePayload(b, e, base) });
            b.switchTo(none_blk);
            try brTo(b, join, .{ .value = operandValue(try lowerExpr(b, args[0])) });
            b.switchTo(join);
            return .{ .value = merge };
        },
    }
}

/// Re-canonicalize `v` to `ty`'s width via the existing `add v, 0` → `normalizeWidth`
/// idiom (uxt/sxt), reused by both conversion arms (`into`'s source-width widen and
/// `try_into`'s destination-width mask). No new IR op, so `--emit ir` goldens are stable.
pub fn recanonToWidth(b: *Builder, v: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!Ir.ValueId {
    const zero = try b.emit(.{ .iconst = 0 }, Typecheck.Type.int);
    return b.emit(.{ .add = .{ .lhs = v, .rhs = zero } }, ty);
}

/// Load a `char` receiver's codepoint — its single `uint32` field at offset 0 — as an int
/// value. A `char` is a struct, so `lowerExpr` yields it by slot; the load reads the whole
/// 8-byte field (high bits are 0, the field was stored width-normalized).
fn loadCharCodepoint(b: *Builder, recv: Ast.Index) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const op = try lowerExpr(b, recv);
    const s = operandSlot(op);
    if (s == Ir.none_slot) {
        try b.note(b.in.tree.nodes[(recv).int()].main_token, "char receiver is not a slot in lower");
        return try b.emit(.{ .iconst = 0 }, int_ty);
    }
    const base = try b.emit(.{ .slot_addr = s }, int_ty);
    return try b.emit(.{ .load = .{ .addr = base, .ty = Typecheck.Type.uint32 } }, Typecheck.Type.uint32);
}

/// A bool: whether int value `v` is a valid Unicode scalar (`0 <= v <= 0x10FFFF` and NOT a
/// UTF-16 surrogate `0xD800..=0xDFFF`). Composed from `icmp`s via 0/1 int arithmetic (no
/// bool-AND op), matching the derive emitter's multiply-accumulate idiom, so it stays
/// branch-free and `--verify`-stable.
pub fn validScalarValue(b: *Builder, v: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    const zero = try b.emit(.{ .iconst = 0 }, int_ty);
    const ge0 = try b.emit(.{ .icmp = .{ .cc = .ge, .lhs = v, .rhs = zero } }, bool_ty);
    const maxcp = try b.emit(.{ .iconst = 0x10FFFF }, int_ty);
    const le_max = try b.emit(.{ .icmp = .{ .cc = .le, .lhs = v, .rhs = maxcp } }, bool_ty);
    const in_range = try b.emit(.{ .mul = .{ .lhs = ge0, .rhs = le_max } }, int_ty);
    const sur_lo = try b.emit(.{ .iconst = 0xD800 }, int_ty);
    const ge_lo = try b.emit(.{ .icmp = .{ .cc = .ge, .lhs = v, .rhs = sur_lo } }, bool_ty);
    const sur_hi = try b.emit(.{ .iconst = 0xDFFF }, int_ty);
    const le_hi = try b.emit(.{ .icmp = .{ .cc = .le, .lhs = v, .rhs = sur_hi } }, bool_ty);
    const is_sur = try b.emit(.{ .mul = .{ .lhs = ge_lo, .rhs = le_hi } }, int_ty);
    const one = try b.emit(.{ .iconst = 1 }, int_ty);
    const not_sur = try b.emit(.{ .sub = .{ .lhs = one, .rhs = is_sur } }, int_ty);
    const valid_int = try b.emit(.{ .mul = .{ .lhs = in_range, .rhs = not_sur } }, int_ty);
    const zero2 = try b.emit(.{ .iconst = 0 }, int_ty);
    return try b.emit(.{ .icmp = .{ .cc = .ne, .lhs = valid_int, .rhs = zero2 } }, bool_ty);
}

/// Lower a target-directed conversion (`.into()` / `.try_into()`). `into` is lossless: an
/// int widen (`add v, 0` typed to the SOURCE width re-canonicalizes via uxt/sxt), char→int
/// (the codepoint field), or byte→char (construct `char{0: byte}`). `try_into` builds a
/// `Result[T, ConvErr]` into a fresh slot from a per-case validity predicate:
///   * int→int narrow: `Ok(masked)` iff `masked` (v truncated+re-extended to T) equals v —
///     and, cross-signedness, v is also non-negative read as signed;
///   * char→byte: the SAME narrow fits-check on the codepoint (unsigned) to `uint8`;
///   * int→char: `Ok(char{0: v})` iff v is a valid Unicode scalar (`validScalarValue`).
/// The reified layout of the call's own result enum drives every tag/offset — no prelude id.
fn lowerConvMethod(b: *Builder, node_idx: Ast.Index, cv: ConvCall) error{OutOfMemory}!Ir.Operand {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.@"bool";
    const recv_ty = b.in.node_types[(cv.recv).int()];
    const recv_is_char = isCharTy(b, recv_ty);

    if (std.mem.eql(u8, cv.member, "into")) {
        if (recv_is_char) {
            // char -> int: the codepoint field, re-canonicalized (uxt) to a clean value.
            const v = try loadCharCodepoint(b, cv.recv);
            return .{ .value = try recanonToWidth(b, v, Typecheck.Type.uint32) };
        }
        const into_ty = b.in.node_types[(node_idx).int()];
        if (into_ty.kind == .float) {
            // int -> float: lossless single scvtf. No Result, no witness, no slot.
            const v = operandValue(try lowerExpr(b, cv.recv));
            return .{ .value = try b.emit(.{ .scvtf = v }, Typecheck.Type.float) };
        }
        if (isCharTy(b, into_ty)) {
            // byte -> char: construct `char{0: byte}` (a byte is always a valid scalar).
            const v = operandValue(try lowerExpr(b, cv.recv));
            const slot = try b.addSlot(into_ty);
            const base = try b.emit(.{ .slot_addr = slot }, int_ty);
            _ = try b.emit(.{ .store = .{ .addr = base, .val = v, .ty = Typecheck.Type.uint32 } }, null);
            return .{ .slot = slot };
        }
        // int -> int widen: `add v, 0` typed to the SOURCE width re-canonicalizes.
        const v = operandValue(try lowerExpr(b, cv.recv));
        return .{ .value = try recanonToWidth(b, v, recv_ty) };
    }

    // try_into: build a `Result[T, ConvErr]` into a fresh slot.
    const v = if (recv_is_char) try loadCharCodepoint(b, cv.recv) else operandValue(try lowerExpr(b, cv.recv));
    const call_ty = b.in.node_types[(node_idx).int()];
    const e = b.in.enum_layouts[call_ty.enum_id];
    const dst_T = e.variants[0].field_types[0];
    const to_char = isCharTy(b, dst_T);

    // int→char / char→byte are heavy (~24 / ~12 SSA cells); inlining >10 / >20 per fn
    // overflowed the imm12 frame cap. Emit each as ONE shared witness and CALL it (O(1)/site).
    if (to_char) return convCallWitness(b, .conv_int_char, v, call_ty);
    if (recv_is_char) return convCallWitness(b, .conv_char_byte, v, call_ty);

    // float -> int: the shared program-global witness (O(1) frame/site — never inlined).
    // `v` is the raw f64 value cell; convCallWitness passes it as a float arg -> v0.
    if (recv_ty.kind == .float) return convCallWitness(b, .conv_float_int, v, call_ty);

    // int→int narrow: a per-(width,signedness) family — no single program-global
    // witness — so it stays inline (see derive_synth: only the 2 char cases become recipes).
    const slot = try b.addSlot(call_ty);
    const base = try b.emit(.{ .slot_addr = slot }, int_ty);
    const masked = try recanonToWidth(b, v, dst_T);
    const fits = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = masked, .rhs = v } }, bool_ty);
    const ok_blk = try b.addBlock();
    const err_blk = try b.addBlock();
    const join = try b.addBlock();
    if (recv_ty.isSigned() == dst_T.isSigned()) {
        b.setTerm(.{ .cond_br = .{ .cond = fits, .t = ok_blk, .f = err_blk } });
    } else {
        // Cross-signedness: `fits` alone misses the 64-bit int↔uint case (low bits coincide)
        // and a large unsigned whose low bits sign-extend negative. Also require v to be
        // non-negative read as signed.
        const sign_blk = try b.addBlock();
        b.setTerm(.{ .cond_br = .{ .cond = fits, .t = sign_blk, .f = err_blk } });
        b.switchTo(sign_blk);
        const z2 = try b.emit(.{ .iconst = 0 }, int_ty);
        const nn = try b.emit(.{ .icmp = .{ .cc = .ge, .lhs = v, .rhs = z2 } }, bool_ty);
        b.setTerm(.{ .cond_br = .{ .cond = nn, .t = ok_blk, .f = err_blk } });
    }
    try emitConvResultTail(b, e, base, ok_blk, err_blk, join, masked, dst_T);
    b.switchTo(join);
    return .{ .slot = slot };
}

/// Emit the shared `try_into` Result build tail: `ok_blk` stores tag 0 + `ok_value` (typed
/// `ok_store_ty`) into variant 0's payload; `err_blk` stores tag 1 + `ConvErr.out_of_range`
/// (tag 0) into variant 1's payload; both branch to `join` (the caller switches to it). The
/// enum `e` drives every tag/offset — the reified layout of the call's own Result — so the
/// int→int narrow inline and the two char witnesses build byte-identical Result values.
pub fn emitConvResultTail(
    b: *Builder,
    e: Typecheck.EnumLayout,
    base: Ir.ValueId,
    ok_blk: Ir.BlockId,
    err_blk: Ir.BlockId,
    join: Ir.BlockId,
    ok_value: Ir.ValueId,
    ok_store_ty: Typecheck.Type,
) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    b.switchTo(ok_blk);
    const ok_tag = try b.emit(.{ .iconst = 0 }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = base, .val = ok_tag, .ty = int_ty } }, null);
    const ok_off = e.payload_off + e.variants[0].offsets[0];
    const ok_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = ok_off, .ty = ok_store_ty } }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = ok_addr, .val = ok_value, .ty = ok_store_ty } }, null);
    try brTo(b, join, .none);

    b.switchTo(err_blk);
    const err_tag = try b.emit(.{ .iconst = 1 }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = base, .val = err_tag, .ty = int_ty } }, null);
    const err_off = e.payload_off + e.variants[1].offsets[0];
    const err_fty = e.variants[1].field_types[0];
    const err_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = err_off, .ty = err_fty } }, int_ty);
    const conv_err_tag = try b.emit(.{ .iconst = 0 }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = err_addr, .val = conv_err_tag, .ty = int_ty } }, null);
    try brTo(b, join, .none);
}

/// Resolve the shared fallible-char-conversion witness of `kind` by scanning the derive
/// table for its recipe (the synthesis barrier appended exactly one per used kind). Null
/// only on a mis-wired build (the checker gate guarantees a recipe at any real conv site).
fn convWitnessCallee(b: *Builder, kind: Derive.Kind) ?Link.SymName {
    for (b.in.derives) |d| if (d.kind == kind) return .{ .kind = .user_fn, .name = d.name.? };
    return null;
}

/// Lower a fallible char conversion as a CALL to its shared witness into a fresh Result
/// slot: `ret_slot` + the raw source scalar arg + the `bl`. O(1) frame per site (vs the
/// former ~24/~12-cell inline that overflowed the frame). `v` is passed RAW (no widen) —
/// exactly what the old inline consumed — so the verdicts are byte-identical.
fn convCallWitness(b: *Builder, kind: Derive.Kind, v: Ir.ValueId, call_ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const slot = try b.addSlot(call_ty);
    const callee = convWitnessCallee(b, kind) orelse {
        try b.diags.append(b.gpa, .{ .byte_offset = 0, .message = "conv witness missing in lower" });
        b.had_error = true;
        return .{ .slot = slot };
    };
    const args = try b.gpa.alloc(Ir.Operand, 1);
    errdefer b.gpa.free(args);
    args[0] = .{ .value = v };
    _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = slot } }, null);
    return .{ .slot = slot };
}

/// Load variant 0's (payload) single field from a reified `Option`/`Result` at `base`
///: absolute offset `payload_off + variant0.offsets[0]`, typed by its field type.
/// Scalar payload only; the accept set is int/bool.
fn loadNativePayload(b: *Builder, e: Typecheck.EnumLayout, base: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const payload_ty = e.variants[0].field_types[0];
    const off = e.payload_off + e.variants[0].offsets[0];
    const pa = try b.emit(.{ .field_addr = .{ .base = base, .off = off, .ty = payload_ty } }, int_ty);
    // A managed-box payload is an 8-byte scalar cell pointer: load it as `int` (a
    // struct-typed `.load` misfires in codegen).
    const load_ty = if (isRefTy(b, payload_ty)) int_ty else payload_ty;
    return try b.emit(.{ .load = .{ .addr = pa, .ty = load_ty } }, load_ty);
}

fn lowerExprInto(b: *Builder, expr: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(expr).int()];
    // A managed box produces an 8-byte scalar cell pointer straight into the destination.
    if (isRefTy(b, ty)) {
        const op = try lowerExpr(b, expr);
        if (!b.termSet()) _ = try b.emit(.{ .store = .{ .addr = dst_ptr, .val = operandValue(op), .ty = Typecheck.Type.int } }, null);
        return;
    }
    switch (ty.kind) {
        .int, .bool, .float => {
            const op = try lowerExpr(b, expr);
            // A diverging producer (e.g. a match arm body that `return`s) already set
            // the block terminator; a store into that dead tail is malformed IR.
            if (!b.termSet()) _ = try b.emit(.{ .store = .{ .addr = dst_ptr, .val = operandValue(op), .ty = ty } }, null);
            return;
        },
        .unit, .never => {
            // `never` means the expression diverges (e.g. a match arm whose body
            // `return`s). Lower it for its control-flow effect: it sets its own
            // terminator, so there is no value to produce into `dst_ptr`.
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
        .literal_char => try lowerCharLiteralInto(b, expr, dst_ptr),
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
        .tuple_field => try copyAggInto(b, expr, dst_ptr, ty),
        .call => {
            if (isTupleStructCtorCall(b, n, ty)) {
                try lowerTupleStructInitInto(b, expr, dst_ptr);
            } else if (isQualifiedVariantCtorCall(b, n, ty)) {
                try lowerEnumInitInto(b, expr, dst_ptr, ty);
            } else {
                try copyAggInto(b, expr, dst_ptr, ty);
            }
        },
        .if_stmt => try lowerIfValueInto(b, expr, dst_ptr, ty),
        .block => try lowerBlockValueInto(b, expr, dst_ptr, ty),
        .unsafe_block => try lowerBlockValueInto(b, n.lhs, dst_ptr, ty),
        .loop_expr => try lowerLoopValueInto(b, expr, dst_ptr, ty, null),
        .labeled => try lowerLabeledValueInto(b, expr, dst_ptr, ty),
        .match_expr => try lowerMatchInto(b, expr, dst_ptr, ty),
        .try_expr => try lowerTryInto(b, expr, dst_ptr, ty),
        .identifier => try copyAggInto(b, expr, dst_ptr, ty),
        // An arithmetic operator on struct/enum operands desugars to an
        // aggregate-returning Add/Sub/Mul/Div witness call; `lowerExpr` yields its
        // `.slot`, so copy those bytes into the destination like any other agg result.
        .binary => try copyAggInto(b, expr, dst_ptr, ty),
        // `*r` yielding an aggregate `T`: lowerUnary copies the cell into a temp slot,
        // so treat it like any other slot-valued aggregate result.
        .unary => try copyAggInto(b, expr, dst_ptr, ty),
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

/// The decoded codepoint of a char-literal token. The checker already validated the
/// content (T0036), so a decode error here is an internal invariant break — note-and-0
/// keeps lower total rather than miscompiling.
fn charLiteralCodepoint(b: *Builder, tok: u32) error{OutOfMemory}!u32 {
    return switch (Literal.decodeChar(b.in.tokens[tok].text(b.in.source))) {
        .ok => |cp| cp,
        else => blk: {
            try b.note(tok, "malformed char literal reached lower");
            break :blk 0;
        },
    };
}

/// Write a char literal `'x'` into `dst_ptr`: store the decoded codepoint into `char`'s
/// single `uint32` field (offset 0). `char` occupies an 8-byte slot; codegen lowers every
/// `.store` to a 64-bit `str` (it ignores the IR `.ty`) and the codepoint `iconst` has its
/// upper 32 bits clear, so the whole slot IS written with the high 4 bytes zeroed — a
/// load-bearing invariant that `loadCharCodepoint`, the derived Eq/Ord/Hash, and the
/// conversions rely on when they read the full 8-byte field (so it must not be narrowed
/// to a sub-word store).
fn lowerCharLiteralInto(b: *Builder, expr: Ast.Index, dst_ptr: Ir.ValueId) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(expr).int()];
    const cp = try charLiteralCodepoint(b, n.main_token);
    const v = try b.emit(.{ .iconst = @intCast(cp) }, Typecheck.Type.uint32);
    _ = try b.emit(.{ .store = .{ .addr = dst_ptr, .val = v, .ty = Typecheck.Type.uint32 } }, null);
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
    // A managed-box field reads as an 8-byte scalar cell pointer, not by aggregate copy.
    if (isRefTy(b, ty)) return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = Typecheck.Type.int } }, Typecheck.Type.int) };
    switch (ty.kind) {
        .int, .bool, .float => {
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
        .int, .bool, .float => {
            const v = operandValue(try lowerExpr(b, value));
            _ = try b.emit(.{ .store = .{ .addr = addr, .val = v, .ty = ty } }, null);
        },
        .str, .@"struct", .@"enum" => try lowerExprInto(b, value, addr, ty),
        else => try b.note(b.in.tree.nodes[(place).int()].main_token, "field store type unsupported in lower"),
    }
}

/// The address of a slot's CONTENTS. For an ordinary slot that is `slot_addr(slot)`.
/// For the `mut self` receiver slot the slot holds a POINTER to the caller's
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
        // A `*r` place: the address of the boxed `T` is the cell pointer, i.e. the Ref's
        // own scalar value. Works for a local Ref (loaded via `lowerIdentifier`) or any
        // Ref rvalue.
        .unary => return operandValue(try lowerExpr(b, n.lhs)),
        .field_access, .tuple_field => {
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
        .int, .bool, .float => {
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
    // A `never`-typed match: every arm diverged, so `join` is unreachable. Seal it as
    // unreachable so the caller's fall-through does not append a value-less `br` to the
    // fn exit (which carries a typed result param).
    if (ty.kind == .never) b.blocks.items[join].term_set = true;
}

/// A postfix `?` in VALUE context. Mirrors `lowerMatchValue`: materialize the
/// unwrapped payload via `lowerTryInto` — a scalar loads from a fresh slot; an
/// aggregate (str/struct/enum) yields `Operand.slot`. The residual (`none`/`err`)
/// arm early-returns from the enclosing fn inside `lowerTryInto`, so the value that
/// reaches here is always the happy-path payload.
fn lowerTryValue(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    switch (ty.kind) {
        .int, .bool, .float => {
            const slot = try b.addSlot(ty);
            const dst = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            try lowerTryInto(b, node_idx, dst, ty);
            const addr = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            const v = try b.emit(.{ .load = .{ .addr = addr, .ty = ty } }, ty);
            return .{ .value = v };
        },
        .str, .@"struct", .@"enum" => return try aggregateValue(b, node_idx, ty),
        .unit, .never => {
            // The tag-test and residual early-return still matter; only the happy-path
            // payload copy is a zero-size no-op. Mirror lowerMatchValue's .unit arm.
            const slot = try b.addSlot(Typecheck.Type.int);
            const dst = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            try lowerTryInto(b, node_idx, dst, ty);
            return .none;
        },
        else => {
            try b.note(b.in.tree.nodes[(node_idx).int()].main_token, "'?' payload type unsupported in lower");
            return .none;
        },
    }
}

/// Lower a postfix `?` writing the unwrapped payload into `dst_ptr`. A pure
/// control-flow desugar over the reified `Option`/`Result` operand:
///
///   spill operand → get_tag → cond_br(tag != 1 ? happy : residual)
///     residual: build the enclosing return-type residual (`none` = tag only;
///               `err` = tag + a copy of E) into a fresh `b.ret_type` slot, then
///               `br` to the exit (reusing the existing return edge).
///     happy:    extract variant-0's payload into `dst_ptr`; control continues.
///
/// Handles AGGREGATE payloads via field_addr + `copy` (NOT the scalar-only native
/// `unwrap`). Variant 0 is the payload (some/ok), variant 1 the residual (none/err),
/// fixed by `registerPrelude`.
fn lowerTryInto(b: *Builder, node_idx: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    const n = b.in.tree.nodes[(node_idx).int()];
    const op_ty = b.in.node_types[(n.lhs).int()];
    if (op_ty.kind != .@"enum") {
        try b.note(n.main_token, "'?' operand did not reify to an Option/Result enum");
        return;
    }

    // Spill the operand into a slot (its tag + payload are read by address).
    const op_op = try lowerExpr(b, n.lhs);
    const op_slot = try spillScrutinee(b, op_op, op_ty, n.main_token);
    if (op_slot == Ir.none_slot) return; // diagnostic already emitted

    const ol = b.in.enum_layouts[op_ty.enum_id];
    const op_base = try b.emit(.{ .slot_addr = op_slot }, int_ty);
    const tag = try b.emit(.{ .get_tag = op_base }, int_ty);
    const one = try b.emit(.{ .iconst = 1 }, int_ty);
    // tag != 1 → happy (tag 0 = some/ok); tag == 1 → residual (none/err).
    const is_happy = try b.emit(.{ .icmp = .{ .cc = .ne, .lhs = tag, .rhs = one } }, Typecheck.Type.@"bool");
    const happy_bb = try b.addBlock();
    const residual_bb = try b.addBlock();
    b.setTerm(.{ .cond_br = .{ .cond = is_happy, .t = happy_bb, .f = residual_bb } });

    // Residual: build the enclosing return-type residual and early-return it.
    b.switchTo(residual_bb);
    try buildResidual(b, op_base, ol, n.main_token);

    // Happy: extract variant-0's payload into the destination; control continues here.
    b.switchTo(happy_bb);
    const src = try addrAtOff(b, op_base, ol.payload_off + ol.variants[0].offsets[0], ty);
    try copyValueByType(b, dst_ptr, src, ty);
}

/// Build the enclosing return-type residual into a fresh `b.ret_type` slot and
/// early-return it via the exit block. `Option.none` is the tag alone;
/// `Result.err` is the tag plus a copy of the error payload from the operand's `err`
/// variant. The residual is built from the RETURN enum's OWN layout (`rl`) — the
/// operand and return enums may differ in size. When the error types MATCH, copying E
/// across the two layouts is a plain byte copy; when they DIFFER, the error is
/// WIDENED via the `From` witness `RetErr.from(opErr)`.
fn buildResidual(b: *Builder, op_base: Ir.ValueId, ol: Typecheck.EnumLayout, tok: u32) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    if (b.ret_type.kind != .@"enum") {
        try b.note(tok, "'?' enclosing return type did not reify to an Option/Result enum");
        return;
    }
    const rl = b.in.enum_layouts[b.ret_type.enum_id];
    const ret_slot = try b.addSlot(b.ret_type);
    const ret_base = try b.emit(.{ .slot_addr = ret_slot }, int_ty);

    // tag = 1 (none/err) at offset 0.
    const tagv = try b.emit(.{ .iconst = 1 }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = ret_base, .val = tagv, .ty = int_ty } }, null);

    // Result: place the error payload into the return's `err` variant. Option's residual
    // (`none`) is the tag alone. Two cases:
    //   * SAME error type -> a plain byte copy of E across the two (differently-sized)
    //     layouts. Src is sized by the operand's E, dst by the return's E — equal
    //     here, but sized separately so the widening branch below shares the same idiom.
    //   * DIFFERING error types -> WIDEN via `RetErr.from(opErr)`: materialize the
    //     operand's E into a slot, call the `From` witness `from(opErr) -> RetErr`, and copy
    //     its result into the return's err payload. The checker (typeOfTry) proved `RetErr has
    //     From[OpErr]` before this runs; a `.none`/`.ambiguous` resolution is an internal
    //     invariant break -> note-and-drop rather than miscompile.
    if (rl.native_family == .result) {
        const op_err = ol.variants[1].field_types[0];
        const ret_err = rl.variants[1].field_types[0];
        const src_off = ol.payload_off + ol.variants[1].offsets[0];
        const dst_off = rl.payload_off + rl.variants[1].offsets[0];
        if (Typecheck.Type.eql(op_err, ret_err)) {
            const src = try addrAtOff(b, op_base, src_off, op_err);
            const dst = try addrAtOff(b, ret_base, dst_off, ret_err);
            try copyValueByType(b, dst, src, ret_err);
        } else switch (Typecheck.resolveConformanceMethod(b.in.methods, ret_err, "from", b.in.prelude_ids.from, &.{op_err})) {
            .one => |m| {
                const src = try addrAtOff(b, op_base, src_off, op_err);
                // The `from` witness takes its `Src` arg by value: a scalar (int/bool) as a
                // loaded value, an aggregate (str/struct/enum) copied into a fresh slot —
                // mirroring `lowerExpr`'s scalar->value / aggregate->slot convention.
                const arg: Ir.Operand = switch (op_err.kind) {
                    .int, .bool => .{ .value = try b.emit(.{ .load = .{ .addr = src, .ty = op_err } }, op_err) },
                    else => blk: {
                        const arg_slot = try b.addSlot(op_err);
                        const arg_base = try b.emit(.{ .slot_addr = arg_slot }, int_ty);
                        try copyValueByType(b, arg_base, src, op_err);
                        break :blk .{ .slot = arg_slot };
                    },
                };
                const callee = witnessCallee(b, m);
                const from_ret = witnessRet(b, m);
                const args = try b.gpa.alloc(Ir.Operand, 1);
                args[0] = arg;
                const dst = try addrAtOff(b, ret_base, dst_off, ret_err);
                // Deliver the `from` result into the return's err payload: an aggregate ret
                // lands in a fresh ret_slot then copies out; a scalar ret is the call's value
                // then a store (mirroring `lowerCall`'s ret placement).
                switch (from_ret.kind) {
                    .int, .bool => {
                        const v = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, from_ret);
                        _ = try b.emit(.{ .store = .{ .addr = dst, .val = v, .ty = from_ret } }, null);
                    },
                    else => {
                        const from_slot = try b.addSlot(from_ret);
                        _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = from_slot } }, null);
                        const from_base = try b.emit(.{ .slot_addr = from_slot }, int_ty);
                        try copyValueByType(b, dst, from_base, ret_err);
                    },
                }
            },
            .none, .ambiguous => {
                try b.note(tok, "'?' error-widen: no unique From witness in lower");
            },
        }
    }
    try brTo(b, b.exit, .{ .slot = ret_slot });
}

/// Copy the value of `ty` at `src` into `dst`: a scalar load/store (int/bool), an
/// aggregate byte `copy` (str/struct/enum). Mirrors `bindLeaf`'s scalar-vs-aggregate
/// split; both sides are ptr values.
fn copyValueByType(b: *Builder, dst: Ir.ValueId, src: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    switch (ty.kind) {
        .int, .bool, .float => {
            const v = try b.emit(.{ .load = .{ .addr = src, .ty = ty } }, ty);
            _ = try b.emit(.{ .store = .{ .addr = dst, .val = v, .ty = ty } }, null);
        },
        else => _ = try b.emit(.{ .copy = .{ .dst = dst, .src = src, .ty = ty } }, null),
    }
}

/// A ptr value addressing [base + off]. `off == 0` collapses to the bare base ptr
/// (no zero-offset field_addr). The ptr-value sibling of `slotFieldAddr`.
fn addrAtOff(b: *Builder, base: Ir.ValueId, off: u32, ty: Typecheck.Type) error{OutOfMemory}!Ir.ValueId {
    if (off == 0) return base;
    return try b.emit(.{ .field_addr = .{ .base = base, .off = off, .ty = ty } }, Typecheck.Type.int);
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
        .int, .bool, .float => {
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
        .field_access, .tuple_field => isLocalRootedPlace(b, n.lhs),
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
                .lt, .lt_eq, .gt, .gt_eq => {
                    // int/bool stay an inline `icmp` then cond_br (bytes unchanged);
                    // str/struct/enum desugar via `lowerOrdValue`, then cond_br on the
                    // produced bool (its current block is the desugar's tail).
                    const lt = b.in.node_types[(n.lhs).int()];
                    if (isInlineOrd(lt.kind)) {
                        const lhs = operandValue(try lowerExpr(b, n.lhs));
                        const rhs = operandValue(try lowerExpr(b, n.rhs));
                        const cc = condFromToken(op, lt.isUnsignedInt());
                        const c = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.@"bool");
                        b.setTerm(.{ .cond_br = .{ .cond = c, .t = true_bb, .f = false_bb } });
                    } else {
                        const v = operandValue(try lowerOrdValue(b, lt, n.lhs, n.rhs, op));
                        b.setTerm(.{ .cond_br = .{ .cond = v, .t = true_bb, .f = false_bb } });
                    }
                },
                .eq_eq, .bang_eq => {
                    // int/bool stay an inline `icmp` then cond_br (bytes unchanged);
                    // str/unit/struct/enum desugar via `lowerEqValue`, then cond_br on
                    // the produced bool (its current block is the eq computation's tail).
                    const lt = b.in.node_types[(n.lhs).int()];
                    if (isInlineEq(lt.kind)) {
                        const lhs = operandValue(try lowerExpr(b, n.lhs));
                        const rhs = operandValue(try lowerExpr(b, n.rhs));
                        const cc = condFromToken(op, lt.isUnsignedInt());
                        const c = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.@"bool");
                        b.setTerm(.{ .cond_br = .{ .cond = c, .t = true_bb, .f = false_bb } });
                    } else {
                        const v = operandValue(try lowerEqValue(b, lt, n.lhs, n.rhs, op == .bang_eq));
                        b.setTerm(.{ .cond_br = .{ .cond = v, .t = true_bb, .f = false_bb } });
                    }
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
pub fn brTo(b: *Builder, dest: Ir.BlockId, arg: Ir.Operand) error{OutOfMemory}!void {
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
    // is taken on the correct layout, AND — for a monomorphized instance — it
    // is the SUBSTITUTED concrete type, so a ret spelled `T` (whose token would
    // otherwise fall to `typeFromRef`'s `int` default, miscompiling `id[bool]`/
    // `id[str]`) is resolved correctly. Byte-identical for a non-generic fn, whose
    // `s.ret` equals what `typeFromRef` would compute for the concrete spelling.
    if (in.sig) |s| {
        std.debug.assert(!s.ret.isTypeVar() and !s.ret.isApp()); // reified-away before lower
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
        std.debug.assert(!s.params[slot].isTypeVar() and !s.params[slot].isApp()); // reified-away before lower
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

/// Map a comparison token to its IR condition. `unsigned` (baked from the operand
/// type at lower-time) selects the unsigned magnitude conds so the choice survives
/// operand-rewriting opt passes; eq/ne are sign-agnostic.
fn condFromToken(tag: TokenTag, unsigned: bool) Ir.Cond {
    return switch (tag) {
        .lt => if (unsigned) .ult else .lt,
        .lt_eq => if (unsigned) .ule else .le,
        .gt => if (unsigned) .ugt else .gt,
        .gt_eq => if (unsigned) .uge else .ge,
        .eq_eq => .eq,
        .bang_eq => .ne,
        else => unreachable,
    };
}

/// Seed for the literal content hash. FROZEN: changing it re-hashes every literal,
/// so the `.cstr` reloc targets (the hashes) in existing cached FnCode blobs would
/// no longer line up with the literal table — cached blobs rot.
const lit_seed: u64 = 0x10c5_7e87;

/// Parse an int literal as written in source, for a PATTERN literal. Delegates to
/// `Literal.value` (base-0 decode, `_` stripped) so a `0x2A`/`0o52` pattern decodes to
/// the same value as the scrutinee it is matched against.
fn parseIntLit(text: []const u8) i64 {
    return Literal.value(text) orelse 0;
}

/// Decode a string-literal token into its runtime bytes via `Literal.decodeString`,
/// mapping the malformed-token kinds to their notes. Returns null on a bad token.
/// Caller owns the returned slice.
fn decodeStringLiteral(b: *Builder, tok: u32) error{OutOfMemory}!?[]u8 {
    const raw = b.in.tokens[tok].text(b.in.source);
    switch (try Literal.decodeString(b.gpa, raw)) {
        .ok => |bytes| return bytes,
        .malformed => try b.note(tok, "malformed string literal"),
        .dangling_backslash => try b.note(tok, "string literal ends with a dangling backslash"),
        .unknown_escape => try b.note(tok, "unknown escape in string literal"),
        .bad_hex_escape => try b.note(tok, "malformed '\\x'/'\\u' escape in string literal"),
        .bad_codepoint => try b.note(tok, "'\\u{...}' escape is not a Unicode scalar value"),
    }
    return null;
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
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
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
/// global fn id. Mirrors `Codegen.renderGraphIr`'s single-module path. The
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
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
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

/// Like `expectLoweredG`, but returns the rendered IR as a gpa-owned string so a test
/// can assert on SUBSTRINGS (used for the desugar shapes, where the exact id
/// numbering is not the point). Caller frees the returned slice.
fn renderLoweredG(gpa: std.mem.Allocator, src: []const u8, fn_name: []const u8) ![]u8 {
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

    const names = try gpa.alloc(Link.SymName, rr.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (rr.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

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
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
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
    return gpa.dupe(u8, w.buffered());
}

test "struct == lowers to a call to the Eq witness (no inline icmp)" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa,
        \\struct P { x: int }
        \\impl P has Eq { fn eq(self, o: P) -> bool { self.x == o.x } }
        \\fn cmp(p: P, q: P) -> bool { p == q }
        \\
    , "cmp");
    defer gpa.free(ir);
    // Dispatches to the witness call; the struct compare itself has no inline icmp.
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "eq$Eq(") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp") == null);
}

test "str == lowers to a heap-free len-compare + per-byte load_byte loop" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa,
        \\fn cmp(a: str, b: str) -> bool { a == b }
        \\
    , "cmp");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "load_byte") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp ge") != null); // the i >= len loop guard
    try testing.expect(std.mem.indexOf(u8, ir, "call") == null); // no witness fn, no heap
}

test "unit == folds to bconst true" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa,
        \\fn nothing() { return }
        \\fn cmp() -> bool { nothing() == nothing() }
        \\
    , "cmp");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "bconst true") != null);
}

test "!= wraps the Eq result in bnot" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa,
        \\struct P { x: int }
        \\impl P has Eq { fn eq(self, o: P) -> bool { self.x == o.x } }
        \\fn cmp(p: P, q: P) -> bool { p != q }
        \\
    , "cmp");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") != null);
}

test "int == stays a single inline icmp eq (regression pin: bytes unchanged)" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa,
        \\fn cmp(a: int, b: int) -> bool { a == b }
        \\
    , "cmp");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp eq") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "call") == null);
    try testing.expect(std.mem.indexOf(u8, ir, "load_byte") == null);
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") == null);
}

test "int != stays a single inline icmp ne (regression pin: no bnot)" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa,
        \\fn cmp(a: int, b: int) -> bool { a != b }
        \\
    , "cmp");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp ne") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") == null);
}

const ord_impl_src =
    \\struct P { x: int }
    \\impl P has Ord {
    \\ fn cmp(self, o: P) -> Ordering {
    \\  if self.x < o.x { Ordering.lt } else if self.x == o.x { Ordering.eq } else { Ordering.gt }
    \\ }
    \\}
    \\
;

test "struct `<` lowers to the cmp witness call + get_tag + `icmp eq` (no bnot)" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa, ord_impl_src ++ "fn use_lt(p: P, q: P) -> bool { p < q }\n", "use_lt");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null); // the cmp witness
    try testing.expect(std.mem.indexOf(u8, ir, "cmp$Ord(") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") != null); // read the Ordering tag
    try testing.expect(std.mem.indexOf(u8, ir, "icmp eq") != null); // discriminant == ord_lt(0)
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") == null);
    try testing.expect(std.mem.indexOf(u8, ir, "load_byte") == null); // struct, not str
}

test "struct `>=` lowers to the cmp witness call + get_tag + `icmp ne`" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa, ord_impl_src ++ "fn use_ge(p: P, q: P) -> bool { p >= q }\n", "use_ge");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp ne") != null); // discriminant != ord_lt(0)
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") == null);
}

test "int `<`/`<=`/`>`/`>=` stay a single inline icmp (regression pin: bytes unchanged)" {
    const gpa = testing.allocator;
    const cases = [_]struct { src: []const u8, want: []const u8 }{
        .{ .src = "fn f(a: int, b: int) -> bool { a < b }\n", .want = "icmp lt" },
        .{ .src = "fn f(a: int, b: int) -> bool { a <= b }\n", .want = "icmp le" },
        .{ .src = "fn f(a: int, b: int) -> bool { a > b }\n", .want = "icmp gt" },
        .{ .src = "fn f(a: int, b: int) -> bool { a >= b }\n", .want = "icmp ge" },
    };
    for (cases) |c| {
        const ir = try renderLoweredG(gpa, c.src, "f");
        defer gpa.free(ir);
        try testing.expect(std.mem.indexOf(u8, ir, c.want) != null);
        try testing.expect(std.mem.indexOf(u8, ir, "call") == null);
        try testing.expect(std.mem.indexOf(u8, ir, "load_byte") == null);
        try testing.expect(std.mem.indexOf(u8, ir, "get_tag") == null);
        try testing.expect(std.mem.indexOf(u8, ir, "bnot") == null);
    }
}

test "str `<` lowers to a heap-free lexicographic load_byte loop (no witness call)" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa, "fn use_lt(a: str, b: str) -> bool { a < b }\n", "use_lt");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "load_byte") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "call") == null); // no witness fn, no heap
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") == null); // str, not an enum witness
    try testing.expect(std.mem.indexOf(u8, ir, "icmp eq") != null); // discriminant == ord_lt(0)
}

test "bool `<` stays an inline icmp (false<true), no call/load_byte" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa, "fn f(a: bool, b: bool) -> bool { a < b }\n", "f");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp lt") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "call") == null);
    try testing.expect(std.mem.indexOf(u8, ir, "load_byte") == null);
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") == null);
}

test "Ord refines Eq — `==` on an Ord-only struct lowers via a cmp call + `icmp eq`" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa, ord_impl_src ++ "fn use_eq(p: P, q: P) -> bool { p == q }\n", "use_eq");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null); // the cmp witness (no `eq` witness)
    try testing.expect(std.mem.indexOf(u8, ir, "cmp$Ord(") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp eq") != null); // discriminant == ord_eq(1)
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") == null);
}

test "Ord refines Eq — `!=` on an Ord-only struct lowers via a cmp call + `icmp ne` (no bnot)" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa, ord_impl_src ++ "fn use_ne(p: P, q: P) -> bool { p != q }\n", "use_ne");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp ne") != null); // discriminant != ord_eq(1)
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") == null);
}

test "derivable struct `<` lowers to the derived cmp call + get_tag + `icmp eq`" {
    const gpa = testing.allocator;
    // No `impl P has Ord` — the checker records a derive request and the barrier synthesizes
    // `Ord$cmp$s0`; `p < q` resolves the DERIVED cmp witness (fn_id==0) and calls it.
    const ir = try renderLoweredG(gpa,
        \\struct P { x: int }
        \\fn use_lt(p: P, q: P) -> bool { p < q }
        \\
    , "use_lt");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "call @Ord$cmp$s0") != null); // the derived witness
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") != null); // read the Ordering tag
    try testing.expect(std.mem.indexOf(u8, ir, "icmp eq") != null); // discriminant == ord_lt(0)
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") == null);
}

test "`==` on a derivable-Ord struct lowers via the derived cmp call + `icmp eq`" {
    const gpa = testing.allocator;
    // `<` derives Ord, which fills `(Eq, P)`; `==` then routes through the same derived cmp
    // (no separate Eq witness) — the refinement path over a SOURCE-LESS witness.
    const ir = try renderLoweredG(gpa,
        \\struct P { x: int }
        \\fn use_both(p: P, q: P) -> int {
        \\ if p < q { return 1 }
        \\ if p == q { return 2 }
        \\ return 0
        \\}
        \\
    , "use_both");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "call @Ord$cmp$s0") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp eq") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "Eq$eq$") == null); // no separate Eq unit referenced
}

const add_impl_src =
    \\struct V2 { x: int, y: int }
    \\impl V2 has Add { fn add(self, o: V2) -> V2 { V2{ x: self.x + o.x, y: self.y + o.y } } }
    \\impl V2 has Sub { fn sub(self, o: V2) -> V2 { V2{ x: self.x - o.x, y: self.y - o.y } } }
    \\impl V2 has Mul { fn mul(self, o: V2) -> V2 { V2{ x: self.x * o.x, y: self.y * o.y } } }
    \\impl V2 has Div { fn div(self, o: V2) -> V2 { V2{ x: self.x / o.x, y: self.y / o.y } } }
    \\
;

test "struct `+` lowers to a call to the Add witness (aggregate return, no inline add)" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa, add_impl_src ++ "fn use_add(p: V2, q: V2) -> V2 { p + q }\n", "use_add");
    defer gpa.free(ir);
    // Dispatches to the `add` witness; its aggregate result lands in a ret_slot. The
    // top-level `+` itself must NOT emit an inline machine `add` (only the witness BODY
    // does, but that body is a different fn — `use_add` here holds just the call).
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "add$Add(") != null);
    try testing.expect(std.mem.indexOf(u8, ir, " -> s") != null); // aggregate ret_slot
    try testing.expect(std.mem.indexOf(u8, ir, "add %") == null); // no inline machine add
}

test "struct `-`/`*`/`/` dispatch to the Sub/Mul/Div witness (sibling protocols)" {
    const gpa = testing.allocator;
    const cases = [_]struct { use: []const u8, method: []const u8, inline_op: []const u8 }{
        .{ .use = "fn f(p: V2, q: V2) -> V2 { p - q }\n", .method = "sub$Sub(", .inline_op = "sub %" },
        .{ .use = "fn f(p: V2, q: V2) -> V2 { p * q }\n", .method = "mul$Mul(", .inline_op = "mul %" },
        .{ .use = "fn f(p: V2, q: V2) -> V2 { p / q }\n", .method = "div$Div(", .inline_op = "sdiv %" },
    };
    for (cases) |c| {
        const src = try std.fmt.allocPrint(gpa, "{s}{s}", .{ add_impl_src, c.use });
        defer gpa.free(src);
        const ir = try renderLoweredG(gpa, src, "f");
        defer gpa.free(ir);
        try testing.expect(std.mem.indexOf(u8, ir, "call @") != null);
        try testing.expect(std.mem.indexOf(u8, ir, c.method) != null);
        try testing.expect(std.mem.indexOf(u8, ir, c.inline_op) == null); // no inline machine op
    }
}

test "int `+`/`-`/`*`/`/` stay a single inline op (regression pin: bytes unchanged, no call)" {
    const gpa = testing.allocator;
    const cases = [_]struct { src: []const u8, want: []const u8 }{
        .{ .src = "fn f(a: int, b: int) -> int { a + b }\n", .want = "add %" },
        .{ .src = "fn f(a: int, b: int) -> int { a - b }\n", .want = "sub %" },
        .{ .src = "fn f(a: int, b: int) -> int { a * b }\n", .want = "mul %" },
        .{ .src = "fn f(a: int, b: int) -> int { a / b }\n", .want = "sdiv %" },
    };
    for (cases) |c| {
        const ir = try renderLoweredG(gpa, c.src, "f");
        defer gpa.free(ir);
        try testing.expect(std.mem.indexOf(u8, ir, c.want) != null);
        try testing.expect(std.mem.indexOf(u8, ir, "call") == null);
    }
}

const from_impl_src =
    \\enum SmallErr { bad }
    \\enum BigErr { small, other }
    \\impl BigErr has From[SmallErr] { fn from(s: SmallErr) -> BigErr { BigErr.small } }
    \\fn inner() -> Result[int, SmallErr] { return Result[int, SmallErr].err(SmallErr.bad) }
    \\fn outer() -> Result[int, BigErr] {
    \\ v := inner()?
    \\ return Result.ok(v)
    \\}
    \\fn same(r: Result[int, BigErr]) -> Result[int, BigErr] {
    \\ v := r?
    \\ return Result.ok(v + 1)
    \\}
    \\fn main() -> int { return 0 }
    \\
;

test "a `?` that WIDENS the error emits a call to the From witness in the residual" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa, from_impl_src, "outer");
    defer gpa.free(ir);
    // The err residual materializes the operand error, calls `BigErr.from(smallErr)`, and
    // copies the widened error into the return's err payload.
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "from") != null);
}

test "a `?` on a SAME-error-type Result stays a plain copy (no From witness call)" {
    const gpa = testing.allocator;
    const ir = try renderLoweredG(gpa, from_impl_src, "same");
    defer gpa.free(ir);
    // The identity path: the err payload is copied unchanged across the two layouts —
    // no `from` witness is resolved or called.
    try testing.expect(std.mem.indexOf(u8, ir, "from") == null);
}

test "`?` on a unit-payload Result still emits the residual early-return (no codegen diagnostic)" {
    const gpa = testing.allocator;
    // renderLoweredG asserts zero diagnostics; before the fix lowerTryValue had no `.unit`
    // arm and dropped a note here, so a checker-accepted program aborted at codegen.
    const ir = try renderLoweredG(gpa,
        \\fn inner() -> Result[(), int] { return Result[(), int].err(7) }
        \\fn outer() -> Result[int, int] {
        \\ inner()?
        \\ return Result.ok(0)
        \\}
        \\fn main() -> int { return 0 }
        \\
    , "outer");
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "cond_br") != null);
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
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
    };
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    var func = try lowerFn(gpa, in, fn_decl, names[0], true, &diags);
    defer func.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), diags.items.len);
    try testing.expect(func.blocks.len >= 5); // entry + exit + header/body/done
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
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
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
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
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

test "a mut-self method body loads self through the pointer slot (s0:int)" {
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

test "the caller passes the receiver place ADDRESS as arg 0 (a scalar value)" {
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

test "whole-self value read copies the pointee into a fresh temp" {
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
