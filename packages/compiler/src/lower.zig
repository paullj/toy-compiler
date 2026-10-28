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
//! (literal/identifier/var_decl/assign/arg/result) since it
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
const Walk = @import("ast/Walk.zig");
const Token = @import("ast/Token.zig").Token;
const TokenTag = @import("ast/Token.zig").Tag;
const Resolve = @import("resolve.zig");
const symbols_res = @import("symbols/Resolution.zig");
const Typecheck = @import("types.zig");
const Callee = @import("symbols/Callee.zig").Callee;
const Closure = @import("symbols/Closure.zig");
const Spec = @import("symbols/Spec.zig");
const LayoutEngine = @import("layout/Engine.zig");
const Link = @import("link/Link.zig");
const Ir = @import("ir/Ir.zig");
const Abi = @import("codegen/abi/Abi.zig");
const Sig = @import("symbols/Sig.zig").Sig;
const Mono = @import("symbols/Mono.zig");
const Derive = @import("symbols/Derive.zig");
const Intrinsic = @import("symbols/Intrinsic.zig");
const Modes = @import("symbols/Modes.zig");
const Body = @import("symbols/Body.zig");
const StdNames = @import("symbols/StdNames.zig");
const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;
const Literal = @import("types/literal.zig");
const closure = @import("lower/closure.zig");
const places = @import("lower/places.zig");

/// The read-only front-end inputs one unit's lowering needs: its body's tables and the
/// program-wide ones. A codegen job's `Engine.Frozen` carries them whole.
pub const Inputs = struct {
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const Resolve.Resolution,
    /// This body's node types and per-call method selections (symbols/Callee.zig).
    nodes: Body.Nodes,
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
    /// The monomorphization instance table, which a generic call's `.instance` callee
    /// indexes. Empty for a program with no generics.
    instances: []const Mono.Instance = &.{},
    /// Program-wide callee signatures by global fn id.
    sigs: []const Sig = &.{},
    /// The program-wide method table. `callees` selects a method call's entry by index;
    /// the operator/derive/`?`-widen witness sites resolve theirs here by protocol.
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
    /// The function-value tables. NO default (the `methods` discipline): a unit that
    /// names a fn as a value must find its thunk, so every build site threads this,
    /// passing `ClosureInputs.none` only for a unit that reads no AST.
    closure: ClosureInputs,

    pub fn bodyView(in: *const Inputs) Body.BodyView {
        return .{ .tree = in.tree, .resolutions = in.resolutions, .nodes = in.nodes };
    }

    pub fn programView(in: *const Inputs) Body.ProgramView {
        return .{ .sigs = in.sigs, .methods = in.methods, .instances = in.instances, .names = in.names, .layouts = in.layouts };
    }
};

/// The closure tables one lowering reads, borrowed from the typecheck result.
pub const ClosureInputs = struct {
    /// The fn-value thunks, sorted by `(kind, key)`.
    adapters: []const Closure.Adapter,
    /// The lambda units, sorted by `(owner, node)`.
    lambda_units: []const Closure.LambdaUnit,
    /// The body being lowered, as the owner of the lambdas written in it (a lambda unit
    /// lowers under its owner's). Null for a unit with no source body.
    owner: ?Closure.Owner = null,
    /// How this lowering holds the owner's captured locals, in the owner and in every
    /// lambda unit of it.
    storage: Closure.Storage = .{},
    /// The specialized copies generic calls may go to; empty means every call takes the
    /// erased instance.
    specs: Spec.Table = .{},
    /// A spec unit's own targets, ascending by param: a call of one of these params jumps
    /// straight to its code. Empty for every other body.
    spec_targets: []const Spec.Target = &.{},
    /// A lambda whose unit has a `frame_env` keeps its env in a frame slot of that struct
    /// type instead of a GC cell.
    frame_envs: bool = false,

    pub const none: ClosureInputs = .{ .adapters = &.{}, .lambda_units = &.{} };
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

    /// Resolver local → where it lives (`LocalHome`). Bound lazily the first time a local is
    /// touched, so slot ids track source order.
    homes: std.AutoHashMapUnmanaged(u32, places.LocalHome) = .empty,
    /// Set when a param other than `self` is `mut`, or when a lambda unit captures one by
    /// address. Two `addr` homes can then name one caller storage (`flip(q, q)`), so an
    /// aggregate store through one builds the whole value aside before writing any of it.
    /// A capture of a local kept in the owner's frame is never staged, as its box is not at
    /// -O0.
    staged_places: bool = false,
    /// Resolver locals a call's later argument may write before the call reads them: each
    /// local a call in this unit passes as a `mut` argument or a `mut self` receiver, and
    /// each local the owner keeps in its frame (a closure an argument calls may write it, as
    /// through its box at -O0). A plain aggregate local is read by slot at the call
    /// instruction, after a later argument's call has run, so a by-value read of one of
    /// these copies it on the spot.
    copy_on_read_roots: std.AutoHashMapUnmanaged(u32, void) = .empty,

    /// The enclosing loop/labeled-block contexts (innermost last).
    loops: std.ArrayList(LoopCtx) = .empty,
    /// A node already evaluated, which `lowerExpr` yields instead of lowering it again: the
    /// element `xs[i] op= v` read once, so the operator arms reused for it see a place
    /// whose handle and index ran once, and the element copy a field store `xs[i].f = v`
    /// changes, so the place chain addresses that copy.
    prelowered: ?Prelowered = null,

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
                .call_indirect => |c| b.gpa.free(c.args),
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
        b.homes.deinit(b.gpa);
        b.copy_on_read_roots.deinit(b.gpa);
        b.loops.deinit(b.gpa);
    }

    /// Whether resolver local `li` lives in a GC cell in this lowering.
    pub fn isBoxedLocal(b: *const Builder, li: u32) bool {
        return b.in.closure.storage.boxed.len != 0 and b.in.closure.storage.hold(li) == .cell;
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
        // Once a divergent sub-expression (a never call, `return`, ...) has terminated
        // the block, the rest of the enclosing construct is dead. Dropping its ops here
        // keeps every caller (a store of the never value's `.none` operand, say) from
        // needing its own guard. Like `setTerm`, free what we were handed.
        if (b.blocks.items[b.cur].term_set) {
            switch (op) {
                .call => |c| b.gpa.free(c.args),
                .call_indirect => |c| b.gpa.free(c.args),
                else => {},
            }
            return result;
        }
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

    pub const Prelowered = struct { node: Ast.Index, op: Ir.Operand };

    /// Have `lowerExpr` yield `op` for `node`, returning the outer entry for the caller to
    /// restore. One element store can run inside the value of another (`xs[0] += f(ys[1])`
    /// with a `mut` param, or an `if` block holding `ps[0].x = 1`), so entries nest.
    pub fn pushPrelowered(b: *Builder, node: Ast.Index, op: Ir.Operand) ?Prelowered {
        const outer = b.prelowered;
        b.prelowered = .{ .node = node, .op = op };
        return outer;
    }

    pub fn termSet(b: *Builder) bool {
        return b.blocks.items[b.cur].term_set;
    }

    /// Switch the cursor to `bid` (the block subsequent code appends to).
    pub fn switchTo(b: *Builder, bid: Ir.BlockId) void {
        b.cur = bid;
    }

    pub fn note(b: *Builder, tok: u32, msg: []const u8) error{OutOfMemory}!void {
        b.had_error = true;
        try b.diags.append(b.gpa, .{ .byte_offset = b.in.tokens[tok].start, .end = b.in.tokens[tok].end, .message = msg });
    }
};

/// How a value-`break` to this context delivers its value — the two merge shapes
/// are mutually exclusive, so a tagged union makes the illegal combinations
/// unrepresentable (and `lowerBreak` switches it exhaustively).
const Merge = union(enum) {
    /// Non-value loop/block: a `break` carries no value.
    none,
    /// SCALAR (or managed-box) value: a value-break's value rides POSITIONALLY as the
    /// join block's br arg, so the join param id need not be stored here.
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
    std.debug.assert(in.nodes.callees.len == in.nodes.types.len);
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
    for (proto.params, 0..) |pn, i| {
        // A `mut` param (`mut self` included) arrives as the caller's place address, so
        // its slot is typed `int` (an 8B pointer), NOT the Sig's by-value type that
        // `paramType` would give: the existing scalar-param ABI then carries the address in
        // one GPR, with no Abi/Codegen change, and `homeAddr` loads through the slot at
        // every use. Each param is decided on its own token, so a static method's `mut`
        // first param is a place param, not a receiver.
        const is_place = Ast.isMutParam(in.tree, in.tokens, pn);
        const pty = if (is_place) Typecheck.Type.int else paramType(in, proto, @intCast(i));
        const sid = try b.addSlot(pty);
        if (is_place and !(i == 0 and Ast.isSelfParam(in.tree, in.tokens, in.source, pn)))
            b.staged_places = true;
        try params.append(gpa, sid);
        // resolve assigns local slot indices in declaration order, params FIRST
        // (it declares each param but does not write `.local` onto the param node).
        // So a body identifier referencing param `i` resolves to `.local(i)`. Bind
        // local index `i` → this param's IR slot.
        try b.homes.put(gpa, @intCast(i), if (is_place) .{ .addr = sid } else .{ .slot = sid });
    }

    const blocks = try openBodyBlocks(&b, ret_type);
    if (b.in.closure.storage.boxed.len != 0) {
        const lis = try gpa.alloc(u32, params.items.len);
        defer gpa.free(lis);
        for (lis, 0..) |*li, i| li.* = @intCast(i);
        try places.boxParams(&b, lis, params.items, decl.main_token);
    }
    try places.noteMutArgRoots(&b, decl.rhs);
    try places.noteUnboxedRoots(&b);
    try lowerBodyInto(&b, decl.rhs, ret_type, blocks.exit);
    return try finishFn(&b, gpa, sym, &params, blocks.entry, blocks.exit);
}

pub const BodyBlocks = struct { entry: Ir.BlockId, exit: Ir.BlockId };

/// Open the entry block (b0, left current) and the single EXIT block, whose one param is
/// the return value (none for a unit function).
pub fn openBodyBlocks(b: *Builder, ret_type: Typecheck.Type) error{OutOfMemory}!BodyBlocks {
    const entry = try b.addBlock();
    b.switchTo(entry);
    const exit = try b.addBlock();
    b.exit = exit;
    if (ret_type.kind != .unit and ret_type.kind != .never) {
        // A managed box returns as an 8-byte scalar cell pointer (in a register), NOT via
        // the struct reg-pair/sret ABI, so the exit param is `int`. The body value flows
        // in as a scalar (every Ref producer yields `.value`), and the call site reads it
        // as a scalar too.
        const ret_param_ty: Typecheck.Type = switch (passKind(b, ret_type)) {
            .box => box_repr_ty,
            else => ret_type,
        };
        b.ret_param = try b.addParam(exit, ret_param_ty);
    }
    // The exit block returns its param (or unit).
    b.blocks.items[exit].term = if (b.ret_param == Ir.none_value)
        .{ .ret = .none }
    else
        .{ .ret = .{ .value = b.ret_param } };
    b.blocks.items[exit].term_set = true;
    return .{ .entry = entry, .exit = exit };
}

/// Lower `body` from the current block into `exit`.
pub fn lowerBodyInto(b: *Builder, body: Ast.Index, ret_type: Typecheck.Type, exit: Ir.BlockId) error{OutOfMemory}!void {
    // Lower the body. A non-unit fn's body is a VALUE block (implicit return):
    // its trailing expression is the return value, branched to the exit. A unit fn
    // lowers for effect and falls through with no value. If the body diverges
    // (every path returns), the trailing fall-through edge is simply not emitted.
    const body_is_value = (b.ret_param != Ir.none_value);
    if (body_is_value) {
        const v = try lowerBlockValue(b, body, ret_type);
        if (!b.termSet()) try brTo(b, exit, v);
    } else {
        try lowerBlockStmts(b, body);
        if (!b.termSet()) try brTo(b, exit, .none);
    }
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
    b.homes.deinit(gpa);
    b.copy_on_read_roots.deinit(gpa);
    b.loops.deinit(gpa);
    b.blocks = .empty;
    b.homes = .empty;
    b.copy_on_read_roots = .empty;
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
            const ty = b.in.nodes.typeOf(stmt_idx);
            const home = if (places.boxedBinding(b, stmt_idx)) |li| try places.bindBox(b, li, ty) else try places.localHome(b, stmt_idx, ty);
            try storeInto(b, home, .{ .expr = stmt.lhs }, ty);
        },
        .assign => {
            const target = b.in.tree.nodes[(stmt.lhs).int()];
            const place_ty = b.in.nodes.typeOf(stmt.lhs);
            const op = b.in.tokens[stmt.main_token].tag;
            const value: StoreValue = if (op.compoundBase()) |base|
                .{ .compound = .{ .op_tok = stmt.main_token, .op = base, .place = stmt.lhs, .rhs = stmt.rhs } }
            else
                .{ .expr = stmt.rhs };
            if (target.tag == .field_access or target.tag == .tuple_field) {
                try lowerFieldStore(b, stmt.lhs, value, place_ty);
                return;
            }
            if (target.tag == .index) {
                try lowerIndexStore(b, stmt.lhs, value);
                return;
            }
            const home = try places.bindingHome(b, stmt.lhs, place_ty);
            if (isAggregatePass(b, place_ty) and places.stagesStore(b, stmt.lhs, value)) {
                try places.storeStaged(b, value, try places.homeAddr(b, home), place_ty);
                return;
            }
            try storeInto(b, home, value, place_ty);
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
        .for_in_stmt => try lowerForIn(b, stmt_idx, null),
        .for_in2_stmt => try lowerForIn2(b, stmt_idx, null),
        .labeled => try lowerLabeledStmt(b, stmt_idx),
        .local_fn => try closure.lowerLocalFn(b, stmt_idx),
        .break_stmt => try lowerBreak(b, stmt_idx),
        .continue_stmt => try lowerContinue(b, stmt_idx),
        // A poison leaf must never reach lower: a tainted tree is gated out before
        // codegen, and it is not produced anywhere yet.
        .error_node => unreachable,
        else => try b.note(stmt.main_token, "statement unsupported in lower"),
    }
}

/// The value a store writes: an expression, or the `place op v` of a compound assignment,
/// which has no node of its own. Both go through the same store paths, so `x op= v` lowers
/// exactly as `x = x op v` does.
pub const StoreValue = union(enum) {
    expr: Ast.Index,
    compound: struct { op_tok: u32, op: TokenTag, place: Ast.Index, rhs: Ast.Index },

    fn token(v: StoreValue, b: *const Builder) u32 {
        return switch (v) {
            .expr => |e| b.in.tree.nodes[e.int()].main_token,
            .compound => |c| c.op_tok,
        };
    }
};

fn lowerStoreValue(b: *Builder, v: StoreValue) error{OutOfMemory}!Ir.Operand {
    return switch (v) {
        .expr => |e| try lowerExpr(b, e),
        // The place is read as an rvalue before `rhs` runs, as `place op rhs` reads it.
        .compound => |c| try lowerBinOp(b, c.op_tok, c.op, c.place, c.rhs),
    };
}

/// `lowerExprInto` for a `StoreValue`. A compound value is a binary operator's result, so
/// it takes `lowerExprInto`'s `.binary` path.
pub fn lowerStoreValueInto(b: *Builder, v: StoreValue, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const c = switch (v) {
        .expr => |e| return lowerExprInto(b, e, dst_ptr, ty),
        .compound => |c| c,
    };
    try storeOperandInto(b, try lowerBinOp(b, c.op_tok, c.op, c.place, c.rhs), dst_ptr, ty, c.op_tok);
}

/// Store an already-lowered `op` of type `ty` into `dst_ptr`: a scalar is stored, an
/// aggregate byte-copied from its slot.
fn storeOperandInto(b: *Builder, op: Ir.Operand, dst_ptr: Ir.ValueId, ty: Typecheck.Type, tok: u32) error{OutOfMemory}!void {
    switch (passKind(b, ty)) {
        // A diverging producer (e.g. a match arm body that `return`s) already set the
        // block terminator; a store into that dead tail is malformed IR.
        .box => if (!b.termSet()) {
            _ = try b.emit(.{ .store = .{ .addr = dst_ptr, .val = operandValue(op), .ty = Typecheck.Type.int } }, null);
        },
        .int, .bool, .float => if (!b.termSet()) {
            _ = try b.emit(.{ .store = .{ .addr = dst_ptr, .val = operandValue(op), .ty = ty } }, null);
        },
        .str, .@"struct", .@"enum" => {
            const src_ptr = try operandPtr(b, op);
            if (src_ptr == Ir.none_value) return; // a diagnostic was already emitted
            _ = try b.emit(.{ .copy = .{ .dst = dst_ptr, .src = src_ptr, .ty = ty } }, null);
        },
        // `never` means the producer diverged and set its own terminator, so there is no
        // value to store.
        .unit, .never => {},
        .rawptr, .invalid => try b.note(tok, "produce-into-slot type unsupported in lower"),
    }
}

/// Store `value` into the local at `home`. A scalar emits a store; an aggregate
/// (str/struct/enum) is PRODUCED INTO the storage directly via `lowerExprInto` (the
/// produce-into-slot path: construct/copy/match write straight to the destination,
/// never value-then-copy).
fn storeInto(b: *Builder, home: places.LocalHome, value: StoreValue, ty: Typecheck.Type) error{OutOfMemory}!void {
    // A managed box is a struct type but travels as an 8-byte scalar cell pointer, not an
    // aggregate slot, so store it like a scalar rather than routing to `lowerExprInto`.
    switch (passKind(b, ty)) {
        .box => {
            const v = try lowerStoreValue(b, value);
            const addr = try places.homeAddr(b, home);
            _ = try b.emit(.{ .store = .{ .addr = addr, .val = operandValue(v), .ty = Typecheck.Type.int } }, null);
        },
        .int, .bool, .float, .rawptr => {
            const v = try lowerStoreValue(b, value);
            const addr = try places.homeAddr(b, home);
            _ = try b.emit(.{ .store = .{ .addr = addr, .val = operandValue(v), .ty = ty } }, null);
        },
        .str, .@"struct", .@"enum" => {
            const dst = try places.homeAddr(b, home);
            try lowerStoreValueInto(b, value, dst, ty);
        },
        .unit => {
            _ = try lowerStoreValue(b, value); // for effect
        },
        .never => _ = try lowerStoreValue(b, value),
        .invalid => try b.note(value.token(b), "aggregate var/assign unsupported in lower"),
    }
}

/// Lower an expression in VALUE context: a scalar yields `Operand.value`; a str
/// (16-byte aggregate) yields `Operand.slot`; unit yields `.none`.
pub fn lowerExpr(b: *Builder, node_idx: Ast.Index) error{OutOfMemory}!Ir.Operand {
    if (node_idx == Ast.none) return .none;
    if (b.prelowered) |p| if (p.node == node_idx) return p.op;
    const n = b.in.tree.nodes[(node_idx).int()];
    const ty = b.in.nodes.typeOf(node_idx);
    // A check-time `type_var` / composite `App` / fn type is substituted/reified to a
    // concrete kind BEFORE lowering; if one reaches here, the mono tail missed a
    // node_types slot — trip loudly in Debug/ReleaseSafe rather than miscompile.
    std.debug.assert(ty.kind != .type_var and ty.kind != .app and ty.kind != .func);
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
            return .{ .value = try b.emit(.{ .bconst = t }, Typecheck.Type.bool) };
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
        .binary => return try lowerBinary(b, n),
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
        // An empty list literal `[]` is sugar for the type's associated `new()`.
        .empty_list => return .{ .slot = try lowerEmptyListSlot(b, node_idx, ty) },
        // A non-empty `[e0, ..]` builds a `Vec[V]` via `new()` + one `push` per element.
        .list_literal => return .{ .slot = try lowerListLiteralSlot(b, node_idx, ty) },
        .index => return try lowerIndex(b, node_idx, ty),
        // A value-headed `type_app` is a value index the parser could not tell from a
        // turbofish (`xs[i].f`); the checker typed it as an index, so lower it as one. A
        // real type-app is consumed by call/struct_init/field_access and never reaches
        // lowerExpr standalone.
        .type_app => {
            if (!appHeadIsValue(b, node_idx)) {
                try b.note(n.main_token, "expression unsupported in lower");
                return .none;
            }
            const targs = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            return try lowerIndexCore(b, n.lhs, targs[0], n.main_token, ty);
        },
        // Its body is its own unit; here it is only a value.
        .lambda => return try closure.lowerLambdaValue(b, node_idx, ty),
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
    if (closure.fnValueRef(b, node_idx)) |gid| return try closure.lowerFnValue(b, node_idx, gid, ty);
    const home = try places.bindingHome(b, node_idx, ty);
    // A managed box reads as an 8-byte scalar cell pointer (its own value), not by slot.
    switch (passKind(b, ty)) {
        .box => {
            const addr = try places.homeAddr(b, home);
            return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = Typecheck.Type.int } }, box_repr_ty) };
        },
        .int, .bool, .float, .rawptr => {
            const addr = try places.homeAddr(b, home);
            const v = try b.emit(.{ .load = .{ .addr = addr, .ty = ty } }, ty);
            return .{ .value = v };
        },
        .str, .@"struct", .@"enum" => {
            // A whole-value read of a local held through an address slot (`return p` or
            // passing `p` by value where `p` is a `mut` param or `mut self`; any read of a
            // boxed local): the slot holds a POINTER, not the value, so copy the pointee
            // into a fresh temp and yield that. An ordinary aggregate local is passed by
            // slot directly (no copy).
            if (home != .slot or places.readsCopyOnReadRoot(b, node_idx)) {
                const tmp = try b.addSlot(ty);
                const dst = try b.emit(.{ .slot_addr = tmp }, Typecheck.Type.int);
                _ = try b.emit(.{ .copy = .{ .dst = dst, .src = try places.homeAddr(b, home), .ty = ty } }, null);
                return .{ .slot = tmp };
            }
            return .{ .slot = home.slot }; // aggregate: pass by slot, no load
        },
        .unit => return .none,
        .never, .invalid => {
            try b.note(b.in.tree.nodes[(node_idx).int()].main_token, "identifier type unsupported in lower");
            return .none;
        },
    }
}

/// `Ref.new(v)`: box `v` into a fresh `gc_alloc`'d cell; the `Ref[T]` value IS the 8-byte
/// cell pointer.
fn lowerRefNew(b: *Builder, operand: Ast.Index, tok: u32) error{OutOfMemory}!Ir.Operand {
    const t = b.in.nodes.typeOf(operand);
    const cellptr = try emitGcAlloc(b, Abi.typeSize(t, b.in.layouts, b.in.enum_layouts));
    switch (passKind(b, t)) {
        .box => {
            const v = try lowerExpr(b, operand);
            _ = try b.emit(.{ .store = .{ .addr = cellptr, .val = operandValue(v), .ty = Typecheck.Type.int } }, null);
        },
        .int, .bool, .float, .rawptr => {
            const v = try lowerExpr(b, operand);
            _ = try b.emit(.{ .store = .{ .addr = cellptr, .val = operandValue(v), .ty = t } }, null);
        },
        .str, .@"struct", .@"enum" => try lowerExprInto(b, operand, cellptr, t),
        .unit => _ = try lowerExpr(b, operand),
        .never, .invalid => try b.note(tok, "boxed type unsupported in lower"),
    }
    return .{ .value = cellptr };
}

/// Read the `T` a `Ref[T]` cell at `addr` holds (offset 0).
fn lowerRefLoad(b: *Builder, addr: Ir.ValueId, t: Typecheck.Type, tok: u32) error{OutOfMemory}!Ir.Operand {
    switch (passKind(b, t)) {
        .box => return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = Typecheck.Type.int } }, box_repr_ty) },
        .int, .bool, .float, .rawptr => return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = t } }, t) },
        .str, .@"struct", .@"enum" => {
            const tmp = try b.addSlot(t);
            const dst = try b.emit(.{ .slot_addr = tmp }, Typecheck.Type.int);
            _ = try b.emit(.{ .copy = .{ .dst = dst, .src = addr, .ty = t } }, null);
            return .{ .slot = tmp };
        },
        .unit => return .none,
        .never, .invalid => {
            try b.note(tok, "deref type unsupported in lower");
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
            return .{ .value = try b.emit(.{ .neg = operandValue(v) }, b.in.nodes.typeOf(node_idx)) };
        },
        .bang => {
            const v = try lowerExpr(b, n.lhs);
            return .{ .value = try b.emit(.{ .bnot = operandValue(v) }, Typecheck.Type.bool) };
        },
        .tilde => {
            const v = try lowerExpr(b, n.lhs);
            return .{ .value = try b.emit(.{ .bcompl = operandValue(v) }, b.in.nodes.typeOf(node_idx)) };
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
    const is_zero = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = rhs, .rhs = zero } }, Typecheck.Type.bool);
    const cont = try b.addBlock();
    const panic_blk = try b.addBlock();
    b.setTerm(.{ .cond_br = .{ .cond = is_zero, .t = panic_blk, .f = cont } });
    b.switchTo(panic_blk);
    b.setTerm(.{ .panic = reason });
    b.switchTo(cont);
}

fn lowerBinary(b: *Builder, n: Ast.Node) error{OutOfMemory}!Ir.Operand {
    return lowerBinOp(b, n.main_token, b.in.tokens[n.main_token].tag, n.lhs, n.rhs);
}

/// `lhs op rhs` as a value. `op` is passed apart from `op_tok` (the token a lowering
/// note blames) so a compound `place op= v` can lower through the same arms.
fn lowerBinOp(b: *Builder, op_tok: u32, op: TokenTag, lhs_node: Ast.Index, rhs_node: Ast.Index) error{OutOfMemory}!Ir.Operand {
    switch (op) {
        .plus, .minus, .star, .slash => {
            // int stays an inline machine add/sub/mul/sdiv (bytes unchanged); a struct/enum
            // operand desugars to the resolved Add/Sub/Mul/Div witness via `lowerArithValue`
            //. The checker (T0028) has already proven a same-type conforming operand.
            const lt = b.in.nodes.typeOf(lhs_node);
            if (isInlineArith(lt.kind)) {
                const lhs = operandValue(try lowerExpr(b, lhs_node));
                const rhs = operandValue(try lowerExpr(b, rhs_node));
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
            return try lowerArithValue(b, lt, lhs_node, rhs_node, op);
        },
        .lt, .lt_eq, .gt, .gt_eq => {
            // int/bool stay an inline `icmp` (bytes unchanged); str/struct/enum desugar to a
            // discriminant test on `Ord::cmp` via `lowerOrdValue`.
            const lt = b.in.nodes.typeOf(lhs_node);
            if (isInlineOrd(lt.kind)) {
                const lhs = operandValue(try lowerExpr(b, lhs_node));
                const rhs = operandValue(try lowerExpr(b, rhs_node));
                const cc = condFromToken(op, lt.isUnsignedInt());
                return .{ .value = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.bool) };
            }
            return try lowerOrdValue(b, lt, lhs_node, rhs_node, op);
        },
        .eq_eq, .bang_eq => {
            // int/bool stay an inline `icmp` (bytes unchanged); str/unit/struct/enum
            // desugar to `Eq::eq` via `lowerEqValue`.
            const lt = b.in.nodes.typeOf(lhs_node);
            if (isInlineEq(lt.kind)) {
                const lhs = operandValue(try lowerExpr(b, lhs_node));
                const rhs = operandValue(try lowerExpr(b, rhs_node));
                const cc = condFromToken(op, lt.isUnsignedInt());
                return .{ .value = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.bool) };
            }
            return try lowerEqValue(b, lt, lhs_node, rhs_node, op == .bang_eq);
        },
        .amp_amp, .pipe_pipe => return try lowerAndOrValue(b, lhs_node, rhs_node, op),
        .amp, .pipe, .caret, .lt_lt, .gt_gt => {
            const lt = b.in.nodes.typeOf(lhs_node);
            const lhs = operandValue(try lowerExpr(b, lhs_node));
            const rhs = operandValue(try lowerExpr(b, rhs_node));
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
            const lt = b.in.nodes.typeOf(lhs_node);
            const lhs = operandValue(try lowerExpr(b, lhs_node));
            const rhs = operandValue(try lowerExpr(b, rhs_node));
            try emitZeroGuard(b, rhs, lt, .rem_by_zero);
            const bin: Ir.Bin = .{ .lhs = lhs, .rhs = rhs };
            const ir_op: Ir.Op = if (lt.isUnsignedInt()) Ir.Op{ .umod = bin } else Ir.Op{ .smod = bin };
            return .{ .value = try b.emit(ir_op, lt) };
        },
        .plus_dot, .minus_dot, .star_dot, .slash_dot => {
            const lhs = operandValue(try lowerExpr(b, lhs_node));
            const rhs = operandValue(try lowerExpr(b, rhs_node));
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
            const lhs = operandValue(try lowerExpr(b, lhs_node));
            const rhs = operandValue(try lowerExpr(b, rhs_node));
            const cc: Ir.FCond = switch (op) {
                .lt_dot => .lt,
                .gt_dot => .gt,
                .le_dot => .le,
                .ge_dot => .ge,
                else => unreachable,
            };
            return .{ .value = try b.emit(.{ .fcmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.bool) };
        },
        else => {
            try b.note(op_tok, "binary operator unsupported in lower");
            return .none;
        },
    }
}

/// `a && b` / `a || b` as a VALUE: short-circuit via blocks. The merge is a bool
/// block param on the join. For `&&`: a-false stores `false` and branches to join;
/// a-true falls through to evaluate b, which branches to join with its own value.
/// For `||`: a-true stores `true`; a-false evaluates b.
fn lowerAndOrValue(b: *Builder, lhs_node: Ast.Index, rhs_node: Ast.Index, op: TokenTag) error{OutOfMemory}!Ir.Operand {
    const eval_b = try b.addBlock();
    const short = try b.addBlock(); // the short-circuit constant block
    const join = try b.addBlock();
    const merge = try b.addParam(join, Typecheck.Type.bool);

    // Condition on `a`: for &&, a-true → eval_b, a-false → short. For ||, swap.
    if (op == .amp_amp) {
        try genCond(b, lhs_node, eval_b, short);
    } else {
        try genCond(b, lhs_node, short, eval_b);
    }

    // short block: the short-circuit constant (&&→false, ||→true) → join(const).
    b.switchTo(short);
    const c = try b.emit(.{ .bconst = (op == .pipe_pipe) }, Typecheck.Type.bool);
    try brTo(b, join, .{ .value = c });

    // eval_b block: evaluate b, branch to join with its value.
    b.switchTo(eval_b);
    const bv = try lowerExpr(b, rhs_node);
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
    if (LayoutEngine.isFnStruct(operand_ty, b.in.layouts)) return try closure.lowerFnIdentityEq(b, lhs_node, rhs_node, negate);
    const pk = passKind(b, operand_ty);
    // A managed box compares by CELL IDENTITY: an 8-byte pointer `icmp`, not a structural
    // deref. Two independent `Ref.new(41)` are distinct cells (unequal); `r == r` is the
    // same cell.
    // A struct/enum threads `negate` into `lowerStructEq` so the Ord-refinement `==` path
    // can emit `icmp ne` directly AND the eq-witness path stays byte-identical
    // (call then optional `bnot`, same value-id order). unit/str keep the uniform outer `bnot`.
    switch (pk) {
        .box => {
            const lv = operandValue(try lowerExpr(b, lhs_node));
            const rv = operandValue(try lowerExpr(b, rhs_node));
            const cc: Ir.Cond = if (negate) .ne else .eq;
            return .{ .value = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lv, .rhs = rv } }, Typecheck.Type.bool) };
        },
        .@"struct", .@"enum" => return try lowerStructEq(b, operand_ty, lhs_node, rhs_node, negate),
        .float => {
            const lhs = operandValue(try lowerExpr(b, lhs_node));
            const rhs = operandValue(try lowerExpr(b, rhs_node));
            const cc: Ir.FCond = if (negate) .ne else .eq;
            return .{ .value = try b.emit(.{ .fcmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.bool) };
        },
        else => {},
    }
    const raw: Ir.ValueId = switch (pk) {
        .unit => blk: {
            _ = try lowerExpr(b, lhs_node);
            _ = try lowerExpr(b, rhs_node);
            break :blk try b.emit(.{ .bconst = true }, Typecheck.Type.bool);
        },
        .str => try lowerStrEq(b, lhs_node, rhs_node),
        else => blk: {
            // Unreachable for a well-typed program (int/bool never routed here; any other
            // kind is a type error caught before lower). Note-and-drop to stay well-formed.
            try b.note(b.in.tree.nodes[(lhs_node).int()].main_token, "'==' operand type unsupported in lower");
            break :blk try b.emit(.{ .bconst = false }, Typecheck.Type.bool);
        },
    };
    if (negate) return .{ .value = try b.emit(.{ .bnot = raw }, Typecheck.Type.bool) };
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
            const v = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, Typecheck.Type.bool);
            if (negate) return .{ .value = try b.emit(.{ .bnot = v }, Typecheck.Type.bool) };
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
                    return .{ .value = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = d, .rhs = k } }, Typecheck.Type.bool) };
                },
                .none, .ambiguous => {
                    try b.note(b.in.tree.nodes[(lhs_node).int()].main_token, "no unique 'Eq' or 'Ord' witness for '==' in lower");
                    const fv = try b.emit(.{ .bconst = false }, Typecheck.Type.bool);
                    if (negate) return .{ .value = try b.emit(.{ .bnot = fv }, Typecheck.Type.bool) };
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
    return .{ .value = try b.emit(.{ .icmp = .{ .cc = spec.cc, .lhs = d, .rhs = k } }, Typecheck.Type.bool) };
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
    const method = Walk.opMethods(op)[0];
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
    const bool_ty = Typecheck.Type.bool;

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
    const bool_ty = Typecheck.Type.bool;

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
    const bool_ty = Typecheck.Type.bool;

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
    return Typecheck.witnessSym(m, b.in.derives, b.in.instances, b.in.names).?;
}

/// The ret type of a resolved conformance-witness `Method`: a source-less derive's recipe
/// `ret` (a derived `cmp`/`eq` witness has `fn_id == 0`, which would otherwise mis-read
/// `sigs[0].ret` and mis-size the ret_slot / `get_tag` layout), a Mono instance's ret, else
/// the fn's own sig ret. Shared by every witness CALL that must size a ret_slot from the
/// witness return type. `derive` is checked FIRST (mirroring `witnessCallee`).
pub fn witnessRet(b: *Builder, m: Typecheck.Method) Typecheck.Type {
    return Typecheck.witnessSig(m, b.in.derives, b.in.instances, b.in.sigs).?.ret;
}

/// Heap-free per-byte hash of a str aggregate given the ADDRESS of its `{ptr@0, len@8}`
/// header: fold each byte through the fxhash mixer `h := (rotl(h,5) ^ byte) *% K` (see
/// `hashMix`) in a slot-counter loop, mirroring `strEqAtPtrs`'s `load_byte` walk (slot induction var +
/// slot accumulator). Pure of source (deterministic block/slot ids), so `--verify`-stable.
/// An empty string hashes to `str_hash_seed` (the loop runs zero times). Returns the int
/// hash value in the loop-exit block.
pub fn hashStrAtPtr(b: *Builder, base: Ir.ValueId) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.bool;

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

/// The `__int_to_str` builtin's stable symbol identity — renders an i64 into a fresh
/// `gc_alloc`'d packed byte buffer and returns the str reg-pair {ptr,len} (x0:x1).
const int_to_str_sym: Link.SymName = .{ .kind = .builtin, .name = "__int_to_str" };

/// The `__uint_to_str` builtin's stable symbol identity — `__int_to_str`'s unsigned twin
/// for the 64-bit unsigned types, whose top bit would otherwise render as a sign.
const uint_to_str_sym: Link.SymName = .{ .kind = .builtin, .name = "__uint_to_str" };

/// The `__str_concat` builtin's stable symbol identity — joins two str values into a fresh
/// `gc_alloc`'d packed byte buffer and returns the str reg-pair {ptr,len} (x0:x1).
const str_concat_sym: Link.SymName = .{ .kind = .builtin, .name = "__str_concat" };

/// The slot ADDRESS of a str operand's `{ptr@0,len@8}`. The note-and-zero fallback keeps a
/// mis-lowered operand from becoming a wild pointer (a well-typed program never hits it).
fn strBase(b: *Builder, node: Ast.Index) error{OutOfMemory}!Ir.ValueId {
    const op = try lowerExpr(b, node);
    const slot = operandSlot(op);
    if (slot == Ir.none_slot) {
        try b.note(b.in.tree.nodes[(node).int()].main_token, "str method receiver is not a slot in lower");
        return try b.emit(.{ .iconst = 0 }, Typecheck.Type.int);
    }
    return try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
}

/// `lhs.concat(rhs)`: hand the two operands' `{ptr,len}` to the out-of-line `__str_concat`
/// builtin, which `gc_alloc`s exactly `len_lhs+len_rhs` packed bytes and copies both in.
/// The four scalar args are read from the operands' frame slots BEFORE the call — a raw
/// heap pointer is never held across the call (its sole safepoint), while the operand str
/// VALUES stay slot-rooted so the conservative frame scan keeps their buffers marked across
/// any collection the alloc triggers. The builtin returns the str reg-pair, which
/// `placeResult` stores x0->slot@0, x1->slot@8. Out-of-lining (vs an inline byte-copy loop)
/// keeps a concat's cost off the caller's spill-all frame, so concats do not stack up toward
/// `FrameTooLarge`.
fn lowerStrConcat(b: *Builder, node_idx: Ast.Index, lhs: Ast.Index, rhs: Ast.Index) error{OutOfMemory}!Ir.Operand {
    const int_ty = Typecheck.Type.int;
    const res_ty = b.in.nodes.typeOf(node_idx);

    const lb = try strBase(b, lhs);
    const rb = try strBase(b, rhs);
    const lp = try b.emit(.{ .load = .{ .addr = lb, .ty = int_ty } }, int_ty);
    const ll_addr = try b.emit(.{ .field_addr = .{ .base = lb, .off = 8, .ty = int_ty } }, int_ty);
    const ll = try b.emit(.{ .load = .{ .addr = ll_addr, .ty = int_ty } }, int_ty);
    const rp = try b.emit(.{ .load = .{ .addr = rb, .ty = int_ty } }, int_ty);
    const rl_addr = try b.emit(.{ .field_addr = .{ .base = rb, .off = 8, .ty = int_ty } }, int_ty);
    const rl = try b.emit(.{ .load = .{ .addr = rl_addr, .ty = int_ty } }, int_ty);

    const slot = try b.addSlot(res_ty);
    const args = try b.gpa.alloc(Ir.Operand, 4);
    errdefer b.gpa.free(args);
    args[0] = .{ .value = lp };
    args[1] = .{ .value = ll };
    args[2] = .{ .value = rp };
    args[3] = .{ .value = rl };
    _ = try b.emit(.{ .call = .{ .callee = str_concat_sym, .args = args, .ret_slot = slot } }, null);
    return .{ .slot = slot };
}

/// Materialize a transient `str {ptr@0,len@8}` slot over a COMPILE-TIME byte string: the
/// literal is content-hash keyed + deduped, and its `cstr_ptr` points into constant data
/// (never a gc cell). The seed/piece source for the display accumulator (`appendLiteral`).
pub fn emitStrLiteralSlot(b: *Builder, bytes: []const u8) error{OutOfMemory}!Ir.SlotId {
    const int_ty = Typecheck.Type.int;
    const owned = try b.gpa.dupe(u8, bytes);
    const h = std.hash.Wyhash.hash(lit_seed, owned);
    try b.addLiteral(h, owned);
    const slot = try b.addSlot(Typecheck.Type.str);
    const base = try b.emit(.{ .slot_addr = slot }, int_ty);
    const p = try b.emit(.{ .cstr_ptr = h }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = base, .val = p, .ty = int_ty } }, null);
    const len_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = 8, .ty = int_ty } }, int_ty);
    const lenv = try b.emit(.{ .iconst = @intCast(bytes.len) }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = len_addr, .val = lenv, .ty = int_ty } }, null);
    return slot;
}

/// Append `{add_ptr, add_len}` to the growable `str` accumulator slot `acc`, IN PLACE:
/// load `acc`'s current `{ptr,len}`, hand all four scalars to the out-of-line
/// `__str_concat` builtin, and write its fresh `gc_alloc`'d result back into `acc` (the
/// `ret_slot` IS `acc`). Generalizes `lowerStrConcat`'s tail into a reused-slot loop step
/// so a structural render costs ONE 16-byte str slot for the whole accumulator (not one
/// live temp per field — the frame-capacity fix). Reading the old `acc` as input and
/// writing the new `acc` as output does not alias: `__str_concat` reads+saves its four
/// scalar args before its `gc_alloc`, then returns a fresh buffer that `placeResult` stores
/// into `acc` only AFTER the call. The old `acc` buffer + the `add` buffer stay slot-rooted
/// across the collection the alloc may trigger.
fn appendRaw(b: *Builder, acc: Ir.SlotId, add_ptr: Ir.ValueId, add_len: Ir.ValueId) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    const abase = try b.emit(.{ .slot_addr = acc }, int_ty);
    const accp = try b.emit(.{ .load = .{ .addr = abase, .ty = int_ty } }, int_ty);
    const accl_addr = try b.emit(.{ .field_addr = .{ .base = abase, .off = 8, .ty = int_ty } }, int_ty);
    const accl = try b.emit(.{ .load = .{ .addr = accl_addr, .ty = int_ty } }, int_ty);
    const args = try b.gpa.alloc(Ir.Operand, 4);
    errdefer b.gpa.free(args);
    args[0] = .{ .value = accp };
    args[1] = .{ .value = accl };
    args[2] = .{ .value = add_ptr };
    args[3] = .{ .value = add_len };
    _ = try b.emit(.{ .call = .{ .callee = str_concat_sym, .args = args, .ret_slot = acc } }, null);
}

/// Append a COMPILE-TIME byte string to `acc`. The literal's `cstr_ptr` points into
/// constant data (never a gc cell), so it needs no rooting across the concat's alloc. An
/// empty string is a no-op. Fusing every fixed run (a field's `", name: "` separators) into
/// ONE interned literal per field halves the concats a struct render costs.
pub fn appendLiteral(b: *Builder, acc: Ir.SlotId, bytes: []const u8) error{OutOfMemory}!void {
    if (bytes.len == 0) return;
    const int_ty = Typecheck.Type.int;
    const owned = try b.gpa.dupe(u8, bytes);
    const h = std.hash.Wyhash.hash(lit_seed, owned);
    try b.addLiteral(h, owned);
    const p = try b.emit(.{ .cstr_ptr = h }, int_ty);
    const l = try b.emit(.{ .iconst = @intCast(bytes.len) }, int_ty);
    try appendRaw(b, acc, p, l);
}

/// Append the `str` slot `piece`'s `{ptr,len}` to `acc`. `piece` stays slot-rooted, so its
/// buffer survives the concat's collection.
pub fn appendPiece(b: *Builder, acc: Ir.SlotId, piece: Ir.SlotId) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    const pbase = try b.emit(.{ .slot_addr = piece }, int_ty);
    const pp = try b.emit(.{ .load = .{ .addr = pbase, .ty = int_ty } }, int_ty);
    const pl_addr = try b.emit(.{ .field_addr = .{ .base = pbase, .off = 8, .ty = int_ty } }, int_ty);
    const pl = try b.emit(.{ .load = .{ .addr = pl_addr, .ty = int_ty } }, int_ty);
    try appendRaw(b, acc, pp, pl);
}

/// The SOLE producer of a value's `str` rendering — the authority `to_string` (and the
/// derive `display` emitters) consume. `op` is `ty`'s already-lowered operand. Returns a `str` slot:
///   * str    -> the operand's own slot (identity, no alloc).
///   * int    -> `__int_to_str` into a fresh cell (i64::MIN-correct never-negate render);
///     a 64-bit unsigned int goes to `__uint_to_str` instead.
///   * bool   -> `cond_br` selecting the const-data `"true"`/`"false"` literal.
///   * unit   -> the const-data `"()"` literal (no alloc).
///   * anything else (a struct/enum incl `char`, a `float`) -> resolve the `display`
///     witness and call it -> a fresh str; a custom `impl Display` (std's for float) and a
///     structural derive land here identically, which is what makes `to_string` universal.
pub fn displayToSlot(b: *Builder, ty: Typecheck.Type, op: Ir.Operand) error{OutOfMemory}!Ir.SlotId {
    // Switch on `ty.kind` (NOT `passKind`): a `Ref`/`gc_array` box is a struct type that
    // renders STRUCTURALLY (`Ref$int{0: <cell>}`) through its derived display witness, so it
    // must reach the struct arm — `passKind` would divert it to `.box` and mis-drop it.
    switch (ty.kind) {
        .str => {
            const s = operandSlot(op);
            if (s != Ir.none_slot) return s;
            // A str VALUE operand never reaches here: `to_string(str)` returns the receiver
            // operand directly. Note-and-empty rather than miscompile keeps lower total for
            // the unreachable-for-well-typed case.
            try b.diags.append(b.gpa, .{ .byte_offset = 0, .end = 0, .message = "display: str operand is not a slot in lower" });
            b.had_error = true;
            return try emitStrLiteralSlot(b, "");
        },
        .int => {
            const v = operandValue(op);
            const args = try b.gpa.alloc(Ir.Operand, 1);
            errdefer b.gpa.free(args);
            args[0] = .{ .value = v };
            const slot = try b.addSlot(Typecheck.Type.str);
            const callee = if (ty.isUnsignedInt() and ty.intBits() == 64) uint_to_str_sym else int_to_str_sym;
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = slot } }, null);
            return slot;
        },
        .bool => {
            const v = operandValue(op);
            const res = try b.addSlot(Typecheck.Type.str);
            const t_blk = try b.addBlock();
            const f_blk = try b.addBlock();
            const join = try b.addBlock();
            b.setTerm(.{ .cond_br = .{ .cond = v, .t = t_blk, .f = f_blk } });
            b.switchTo(t_blk);
            try copyStrSlot(b, res, try emitStrLiteralSlot(b, "true"));
            try brTo(b, join, .none);
            b.switchTo(f_blk);
            try copyStrSlot(b, res, try emitStrLiteralSlot(b, "false"));
            try brTo(b, join, .none);
            b.switchTo(join);
            return res;
        },
        .unit => return try emitStrLiteralSlot(b, "()"),
        else => switch (Typecheck.resolveConformanceMethod(b.in.methods, ty, "display", b.in.prelude_ids.display, null)) {
            .one => |m| {
                const callee = witnessCallee(b, m);
                const args = try b.gpa.alloc(Ir.Operand, 1);
                errdefer b.gpa.free(args);
                args[0] = op; // self, by slot
                const slot = try b.addSlot(Typecheck.Type.str);
                _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = slot } }, null);
                return slot;
            },
            .none, .ambiguous => {
                try b.diags.append(b.gpa, .{ .byte_offset = 0, .end = 0, .message = "display: no display witness in lower" });
                b.had_error = true;
                return try emitStrLiteralSlot(b, "");
            },
        },
    }
}

/// Copy a source str slot's two words into `dst`. Used by `bool.to_string` to funnel both
/// literal arms into one result slot.
fn copyStrSlot(b: *Builder, dst: Ir.SlotId, src: Ir.SlotId) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    const dbase = try b.emit(.{ .slot_addr = dst }, int_ty);
    const sbase = try b.emit(.{ .slot_addr = src }, int_ty);
    _ = try b.emit(.{ .copy = .{ .dst = dbase, .src = sbase, .ty = Typecheck.Type.str } }, null);
}

/// The `gc_alloc` builtin's stable symbol identity — returns a fresh zeroed cell pointer.
/// A `Ref.new(x)` box (the `ref_new` intrinsic) or a captured-and-assigned local MINTS this
/// call directly in lower; user source never names it, so this bypasses the core-only
/// allowlist by construction.
const gc_alloc_sym: Link.SymName = .{ .kind = .builtin, .name = "gc_alloc" };

/// A fresh zeroed `gc_alloc(size)` cell pointer, typed as a managed box so the precise
/// root map sees it.
pub fn emitGcAlloc(b: *Builder, size: u32) error{OutOfMemory}!Ir.ValueId {
    const size_v = try b.emit(.{ .iconst = @intCast(size) }, Typecheck.Type.int);
    // Scope the `args` errdefer to the alloc→emit window: once emit succeeds the
    // block instruction owns `args`, so a later OOM must NOT re-free it here.
    const args = try b.gpa.alloc(Ir.Operand, 1);
    errdefer b.gpa.free(args);
    args[0] = .{ .value = size_v };
    return try b.emit(.{ .call = .{ .callee = gc_alloc_sym, .args = args, .ret_slot = Ir.none_slot } }, box_repr_ty);
}

/// The IR value type a managed-box PRODUCER records. A box is erased to a scalar
/// cell pointer, but typing it `.int` hides it from the precise root map (an integer
/// word and a live heap handle look identical). `.rawptr` is ABI-identical to `.int`
/// (scalar, 8 bytes, integer class — byte-neutral through codegen) yet marks the value
/// cell as managed, so the root projector emits it and `verifyRefsSpilled` covers it.
/// Box STORES keep their `.int` store-width (never read by the projector).
pub const box_repr_ty: Typecheck.Type = Typecheck.Type.rawptr;

/// The `gc_mark` builtin's stable symbol identity — marks one managed cell. A derived
/// `trace(obj)` unit `bl`s this per managed field; user source never names it.
pub const gc_mark_sym: Link.SymName = .{ .kind = .builtin, .name = "gc_mark" };

/// The `gc_mark_leaf` builtin's stable symbol identity — sets one cell's mark bit without
/// pushing it (the cell survives but is never scanned). An erased `str` trace unit `bl`s it
/// on the heap buffer; container traces `bl` it on their leaf-marked backings.
pub const gc_mark_leaf_sym: Link.SymName = .{ .kind = .builtin, .name = "gc_mark_leaf" };

/// Whether `t` is a managed box (`Ref[T]`/`gc_array[T]`): a struct carrying the reified
/// reference-family marker. A box is an 8-byte cell pointer — lowered as a scalar `int`,
/// NOT by its 8-byte struct field layout — so every scalar-value path guards on this.
/// The post-reify predicate over the layout SNAPSHOT; derive synthesis runs the same
/// authority over the live struct table (`Typecheck.isRefStruct`).
pub fn isRefTy(b: *Builder, t: Typecheck.Type) bool {
    return Typecheck.isRefStruct(t, b.in.layouts);
}

/// How a reified value TRAVELS through lowering, with the managed-box case lifted out
/// AHEAD of `.@"struct"`. A `Ref`/`gc_array` handle IS a struct type, so a bare
/// `ty.kind` switch would misroute it into the by-slot aggregate arm; classifying here
/// puts the 8-byte-cell-pointer (`.box`) case first, ONCE. Every value path that would
/// otherwise re-derive `if (isRefTy(...)) ...int... else switch (ty.kind)` switches on
/// this instead, so the box-before-struct ordering lives in a single authority. The
/// remaining tags mirror `ty.kind` (post-reify `type_var`/`app`/`func` fold into `.invalid`).
const PassKind = enum { box, int, bool, float, rawptr, str, @"struct", @"enum", unit, never, invalid };

pub fn passKind(b: *Builder, t: Typecheck.Type) PassKind {
    if (isRefTy(b, t)) return .box;
    return switch (t.kind) {
        .int => .int,
        .bool => .bool,
        .float => .float,
        .rawptr => .rawptr,
        .str => .str,
        .@"struct" => .@"struct",
        .@"enum" => .@"enum",
        .unit => .unit,
        .never => .never,
        .invalid, .type_var, .app, .func => .invalid,
    };
}

fn isAggregatePass(b: *Builder, t: Typecheck.Type) bool {
    return switch (passKind(b, t)) {
        .str, .@"struct", .@"enum" => true,
        else => false,
    };
}

/// The operand a value of `ty` held in `slot` is passed as: a scalar is loaded, an
/// aggregate travels by its slot. For a source-less unit forwarding its params.
pub fn slotOperand(b: *Builder, slot: Ir.SlotId, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const int_ty = Typecheck.Type.int;
    switch (passKind(b, ty)) {
        .box => {
            const addr = try b.emit(.{ .slot_addr = slot }, int_ty);
            return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = int_ty } }, box_repr_ty) };
        },
        .int, .bool, .float, .rawptr => {
            const addr = try b.emit(.{ .slot_addr = slot }, int_ty);
            return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = ty } }, ty) };
        },
        .str, .@"struct", .@"enum" => return .{ .slot = slot },
        .unit, .never, .invalid => return .none,
    }
}

/// The operand for a value of type `ty` stored at `addr`: a scalar or box word loaded,
/// an aggregate copied into a fresh slot.
pub fn addrOperand(b: *Builder, addr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    switch (passKind(b, ty)) {
        .box => return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = Typecheck.Type.int } }, box_repr_ty) },
        .int, .bool, .float, .rawptr => return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = ty } }, ty) },
        .str, .@"struct", .@"enum" => {
            const slot = try b.addSlot(ty);
            const dst = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            _ = try b.emit(.{ .copy = .{ .dst = dst, .src = addr, .ty = ty } }, null);
            return .{ .slot = slot };
        },
        .unit, .never, .invalid => return .none,
    }
}

/// Whether `t` is a by-slot aggregate for lowering (str/struct/enum), EXCLUDING a
/// managed-box handle (`Ref`/`gc_array`), which is a scalar 8-byte cell pointer.
fn isAggTy(b: *Builder, t: Typecheck.Type) bool {
    return switch (passKind(b, t)) {
        .str, .@"struct", .@"enum" => true,
        else => false,
    };
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

/// Render a `char` VALUE (materialized in `slot`) to a fresh HEAP `str` by UTF-8-ENCODING
/// its codepoint — the OVERRIDE body of char's derived Display witness
/// (`Display$display$s<id>`), replacing the structural `char(65)` tuple walk. Emitted ONCE
/// (in that shared unit) and CALLed from every char-display site; formerly inlined per site.
/// A 3-test band ladder
/// (cp<=0x7F / <=0x7FF / <=0xFFFF else 4-byte) packs the bytes LITTLE-ENDIAN into one word and
/// does ONE `.store` into an 8-byte scratch slot: `.store` is a 64-bit STR (no store_byte op),
/// so the high padding is written but never read — the `str` len is the true band width. The
/// codepoint is a proven-valid scalar (the decode/try_into gates: 0..0x10FFFF, no surrogates)
/// AND the char literal's 64-bit store zeroed the slot's high 4 bytes, so the loaded value has
/// bit63 clear: unsigned band tests need no validity branch. Pure IR (fixed block/value ids,
/// reads no map) → -jN- and O0≡O1-stable. The packed bytes are copied into a fresh gc cell
/// (the stack scratch dies on return; the caller — `to_string` — keeps the value):
/// the smallest size class is 16 bytes, so the 8-byte packed store is in-bounds while only
/// `len` bytes are ever read. Returns the built `str` slot in the join block.
pub fn lowerCharDisplay(b: *Builder, slot: Ir.SlotId) error{OutOfMemory}!Ir.SlotId {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.bool;

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

    // join: copy the packed word into a fresh gc cell and build a heap str{cell, len}.
    b.switchTo(join);
    const alloc_args = try b.gpa.alloc(Ir.Operand, 1);
    errdefer b.gpa.free(alloc_args);
    alloc_args[0] = .{ .value = len };
    const cell = try b.emit(.{ .call = .{ .callee = gc_alloc_sym, .args = alloc_args, .ret_slot = Ir.none_slot } }, box_repr_ty);
    const jb = try b.emit(.{ .slot_addr = byte_slot }, int_ty);
    const word = try b.emit(.{ .load = .{ .addr = jb, .ty = int_ty } }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = cell, .val = word, .ty = int_ty } }, null);
    const str_slot = try b.addSlot(Typecheck.Type.str);
    const sbase = try b.emit(.{ .slot_addr = str_slot }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = sbase, .val = cell, .ty = int_ty } }, null);
    const len_addr = try b.emit(.{ .field_addr = .{ .base = sbase, .off = 8, .ty = int_ty } }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = len_addr, .val = len, .ty = int_ty } }, null);
    return str_slot;
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
    switch (calleeOf(b, n)) {
        .method => |mi| {
            // A generic-type instance method dispatches to its instance's own symbol, and a
            // source-less derive (`fn_id == 0`) to its synthetic unit; `witnessCallee` knows
            // both (`names[m.fn_id]` would be `names[0]`, a wrong-symbol miscompile).
            const m = b.in.methods[mi];
            callee = witnessCallee(b, m);
            if (b.in.closure.specs.unitAt(b.in.closure.owner, node_idx)) |u| callee = .{ .kind = .user_fn, .name = u.name };
            // An associated function (`Vec[int].new()`) takes no `self`: its receiver is a type.
            self_recv = if (m.is_static) Ast.none else b.in.tree.nodes[Callee.memberNode(b.in.tree, n).?.int()].lhs;
            self_mut = m.mut_self;
        },
        .builtin => |k| return lowerBuiltinMethod(b, node_idx, n, k),
        .value => return closure.lowerValueCall(b, node_idx, n),
        .fn_ref => {
            // A call's slot never carries one: a direct callee is not typed as a value.
            try b.note(n.main_token, "function reference in callee position");
            return .none;
        },
        .bound_method => {
            // `recv.m(..)` is a method call; only a value-position `recv.m` binds.
            try b.note(n.main_token, "bound method in callee position");
            return .none;
        },
        .protocol => {
            // Only bounded-template spans carry one, and templates are never lowered. A note,
            // not `unreachable`, which ReleaseFast would turn into UB.
            try b.note(n.main_token, "protocol callee in lower");
            return .none;
        },
        .instance => |ii| {
            const name = if (b.in.closure.specs.unitAt(b.in.closure.owner, node_idx)) |u| u.name else b.in.instances[ii].name.?;
            callee = .{ .kind = .user_fn, .name = name };
        },
        .direct => |gid| {
            // The memory intrinsics (`store`/`load`/`offset`/..., the `call_hash`/`call_eq`
            // indirect-dispatch helpers and the `ref_*` cell ops) have no symbol of their own,
            // so intercept them BEFORE the plain-name resolution below looks for one.
            const nm = b.in.names[gid];
            if (nm.kind == .builtin) if (Intrinsic.lookup(nm.name)) |ik| if (try lowerIntrinsicCall(b, node_idx, n, ik)) |op| return op;
            callee = nm;
        },
        // A miss (arity/conflict/unbound/unminted) is an internal invariant break — Pass C
        // already gated it — so note-and-drop.
        .generic => {
            try b.note(callee_node.main_token, "unresolved generic instance in lower");
            return .none;
        },
        .none => return lowerTypeIntrinsicCall(b, n),
    }
    const result_ty = b.in.nodes.typeOf(node_idx);

    // Evaluate every arg left-to-right into an Operand (scalar→value, str→slot; a `mut`
    // param's arg is its place address). For a method call the receiver is the FIRST arg
    // (`self`, by value — reusing the struct-arg reg-pair/sret path — or its address for
    // `mut self`), followed by the source args in order.
    const arg_nodes = Ast.rangeSlice(b.in.tree, (n.rhs).int());
    const self_n: usize = if (self_recv != Ast.none) 1 else 0;
    const args = try b.gpa.alloc(Ir.Operand, arg_nodes.len + self_n);
    errdefer b.gpa.free(args);
    // A `mut` place rooted at a Vec element passes the address of a copy, written back
    // once the call returns.
    var wbs: std.ArrayList(places.ElementPlace) = .empty;
    defer wbs.deinit(b.gpa);
    if (self_recv != Ast.none) {
        if (self_mut) {
            // Pass the receiver's place ADDRESS (an int value) as `self`. The receiver
            // was proven a mutable place by the body checker (T0019); a non-place here
            // is an internal invariant break — note-and-drop rather than miscompile.
            const addr = try places.lowerMutPlaceAddr(b, self_recv, &wbs);
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
    const arg_mut = Modes.argMask(b.in.bodyView(), b.in.programView(), node_idx);
    for (arg_nodes, 0..) |arg, i| {
        if (Modes.isMut(arg_mut, i)) {
            // A `mut` param takes the argument's place ADDRESS, like a `mut self` receiver.
            const addr = try places.lowerMutPlaceAddr(b, arg, &wbs);
            if (addr == Ir.none_value) {
                try b.note(callee_node.main_token, "mut argument is not a place in lower");
                b.gpa.free(args);
                return .none;
            }
            args[self_n + i] = .{ .value = addr };
        } else args[self_n + i] = try lowerExpr(b, arg);
    }

    const result = try emitCall(b, .{ .direct = callee }, args, result_ty, n.main_token);
    try places.writeBacks(b, wbs.items);
    return result;
}

/// A call the checker selected no callee for: one of the compile-time intrinsics spelled
/// with type-args, which name no code; any other cannot lower.
fn lowerTypeIntrinsicCall(b: *Builder, n: Ast.Node) error{OutOfMemory}!Ir.Operand {
    const callee_node = b.in.tree.nodes[(n.lhs).int()];
    const bres: Resolve.Resolution = if (callee_node.tag == .type_app) b.in.resolutions[(callee_node.lhs).int()] else .unresolved;
    if (bres != .func) {
        try b.note(n.main_token, "call target unsupported in lower");
        return .none;
    }
    if (bres.func < b.in.names.len and b.in.names[bres.func].kind == .builtin) if (Intrinsic.lookup(b.in.names[bres.func].name)) |bik| switch (bik) {
        // `size_of[T]()` / `align_of[T]()`: a compile-time ABI constant (no call, no reloc).
        .size_of, .align_of => {
            const tnodes = Ast.rangeSlice(b.in.tree, (callee_node.rhs).int());
            const t = b.in.nodes.typeOf(tnodes[0]);
            const c: u32 = if (bik == .size_of)
                Abi.typeSize(t, b.in.layouts, b.in.enum_layouts)
            else
                Abi.typeAlign(t, b.in.layouts, b.in.enum_layouts);
            return .{ .value = try b.emit(.{ .iconst = @intCast(c) }, Typecheck.Type.int) };
        },
        // `gc_array[T](p)` wraps a raw cell pointer into an 8-byte handle: a checker-only
        // type change, so lower yields the pointer operand directly.
        .gc_array => {
            const vargs = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            return .{ .value = operandValue(try lowerExpr(b, vargs[0])) };
        },
        // `descriptor_of[T]()`: the address of T's descriptor entry, through the
        // reserved-sentinel `cstr_ptr` path (NOT folded to a constant — the address is a
        // link-time-resolved reloc, exactly like a string literal ptr).
        .descriptor_of => {
            const tnodes = Ast.rangeSlice(b.in.tree, (callee_node.rhs).int());
            const t = b.in.nodes.typeOf(tnodes[0]);
            return .{ .value = try b.emit(.{ .cstr_ptr = try Link.descHash(b.gpa, t) }, Typecheck.Type.rawptr) };
        },
        else => {},
    };
    try b.note(callee_node.main_token, "unresolved generic instance in lower");
    return .none;
}

/// A call of intrinsic `ik` lowered in place, or null for one that is called by its own
/// symbol (`panic`).
fn lowerIntrinsicCall(b: *Builder, node_idx: Ast.Index, n: Ast.Node, ik: Intrinsic.Kind) error{OutOfMemory}!?Ir.Operand {
    switch (ik) {
        .store => {
            const sargs = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const vty = b.in.nodes.typeOf(sargs[1]);
            const addr = operandValue(try lowerExpr(b, sargs[0]));
            // A zero-sized value (`()`) occupies no bytes: evaluate the value
            // expr for its effects, emit no store (the erased Map's value column
            // when V = ()). Also prevents an OOB write — the cell stride is
            // size_of[K]+0, so offset(cell, size_of[K]) points past the cell.
            if (vty.kind == .unit) {
                _ = try lowerExpr(b, sargs[1]);
                return .none;
            }
            if (isAggTy(b, vty)) {
                const src = try operandPtr(b, try lowerExpr(b, sargs[1]));
                if (src != Ir.none_value) _ = try b.emit(.{ .copy = .{ .dst = addr, .src = src, .ty = vty } }, null);
                return .none;
            }
            const val = operandValue(try lowerExpr(b, sargs[1]));
            _ = try b.emit(.{ .store = .{ .addr = addr, .val = val, .ty = vty } }, null);
            return .none;
        },
        .load => {
            const largs = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const addr = operandValue(try lowerExpr(b, largs[0]));
            const lty = b.in.nodes.typeOf(node_idx);
            // A load typed `()` yields the zero-sized unit value (`.none`): the
            // address was already evaluated for effect; read nothing (mp_val_at /
            // mp_grow when V = ()).
            if (lty.kind == .unit) return .none;
            if (isAggTy(b, lty)) {
                const slot = try b.addSlot(lty);
                const dst = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
                _ = try b.emit(.{ .copy = .{ .dst = dst, .src = addr, .ty = lty } }, null);
                return .{ .slot = slot };
            }
            return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = lty } }, lty) };
        },
        .offset => {
            // `offset(base, i)` is a plain 64-bit pointer add: the base's isRefTy
            // value already lowers to its 8-byte cell pointer, so no unwrap needed.
            const oargs = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const base = operandValue(try lowerExpr(b, oargs[0]));
            const idx = operandValue(try lowerExpr(b, oargs[1]));
            return .{ .value = try b.emit(.{ .add = .{ .lhs = base, .rhs = idx } }, Typecheck.Type.rawptr) };
        },
        .call_hash => {
            const a = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const fp = operandValue(try lowerExpr(b, a[0]));
            const kp = operandValue(try lowerExpr(b, a[1]));
            const args = try b.gpa.alloc(Ir.Operand, 1);
            args[0] = .{ .value = kp };
            return .{ .value = try b.emit(.{ .call_indirect = .{ .target = fp, .args = args, .ret_slot = Ir.none_slot } }, Typecheck.Type.int) };
        },
        .call_eq => {
            const a = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const fp = operandValue(try lowerExpr(b, a[0]));
            const ap = operandValue(try lowerExpr(b, a[1]));
            const bp = operandValue(try lowerExpr(b, a[2]));
            const args = try b.gpa.alloc(Ir.Operand, 2);
            args[0] = .{ .value = ap };
            args[1] = .{ .value = bp };
            return .{ .value = try b.emit(.{ .call_indirect = .{ .target = fp, .args = args, .ret_slot = Ir.none_slot } }, Typecheck.Type.bool) };
        },
        .store_byte => {
            const sargs = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const addr = operandValue(try lowerExpr(b, sargs[0]));
            const val = operandValue(try lowerExpr(b, sargs[1]));
            _ = try b.emit(.{ .store_byte = .{ .addr = addr, .val = val } }, null);
            return .none;
        },
        .load_byte => {
            const largs = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const addr = operandValue(try lowerExpr(b, largs[0]));
            return .{ .value = try b.emit(.{ .load_byte = addr }, Typecheck.Type.int) };
        },
        .ref_new => return try lowerRefNew(b, Ast.rangeSlice(b.in.tree, (n.rhs).int())[0], n.main_token),
        .ref_get => {
            const a = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const addr = operandValue(try lowerExpr(b, a[0]));
            return try lowerRefLoad(b, addr, b.in.nodes.typeOf(node_idx), n.main_token);
        },
        .ref_set => {
            const a = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            // The cell is evaluated before the value, as every place store is.
            const addr = operandValue(try lowerExpr(b, a[0]));
            const vty = b.in.nodes.typeOf(a[1]);
            if (vty.kind == .never) {
                _ = try lowerExpr(b, a[1]);
            } else try storeThrough(b, addr, .{ .expr = a[1] }, vty, n.main_token);
            return .none;
        },
        else => return null,
    }
}

/// What a call instruction invokes: a symbol (`bl`), or a computed code address (`blr`).
pub const CallTarget = union(enum) { direct: Link.SymName, indirect: Ir.ValueId };

fn callOp(target: CallTarget, args: []Ir.Operand, ret_slot: Ir.SlotId) Ir.Op {
    return switch (target) {
        .direct => |callee| .{ .call = .{ .callee = callee, .args = args, .ret_slot = ret_slot } },
        .indirect => |t| .{ .call_indirect = .{ .target = t, .args = args, .ret_slot = ret_slot } },
    };
}

/// Emit the call of `target` with the already-lowered `args` (ownership passes to the
/// instruction) and place its `result_ty` result. `tok` anchors the unsupported-type note.
pub fn emitCall(b: *Builder, target: CallTarget, args: []Ir.Operand, result_ty: Typecheck.Type, tok: u32) error{OutOfMemory}!Ir.Operand {
    // A managed-box result is an 8-byte scalar cell pointer (returned in a register),
    // not an aggregate — read it as a scalar value, no ret_slot.
    // Result placement: scalar → an Instr.result value; aggregate → a fresh
    // ret_slot; unit → neither. The ABI (reg vs sret) is decided in codegen.
    switch (passKind(b, result_ty)) {
        .box => {
            const v = try b.emit(callOp(target, args, Ir.none_slot), box_repr_ty);
            return .{ .value = v };
        },
        .int, .bool, .float, .rawptr => {
            const v = try b.emit(callOp(target, args, Ir.none_slot), result_ty);
            return .{ .value = v };
        },
        .str, .@"struct", .@"enum" => {
            // Aggregate result lands in a fresh slot; the ABI (reg-pair vs sret)
            // is decided in codegen from `ret_slot`.
            const slot = try b.addSlot(result_ty);
            _ = try b.emit(callOp(target, args, slot), null);
            return .{ .slot = slot };
        },
        .unit => {
            _ = try b.emit(callOp(target, args, Ir.none_slot), null);
            return .none;
        },
        .never => {
            // The call does not return, so nothing may follow it in this block: an
            // enclosing arm or fn tail would otherwise branch a value-less edge into a
            // join that carries a typed param.
            _ = try b.emit(callOp(target, args, Ir.none_slot), null);
            b.setTerm(.@"unreachable");
            return .none;
        },
        .invalid => {
            const slot = try b.addSlot(result_ty);
            _ = try b.emit(callOp(target, args, slot), null);
            try b.note(tok, "call result type unsupported in lower");
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
    // A method or builtin returning an enum has this same call shape; only the checker's
    // Callee tells the two apart.
    return ty.kind == .@"enum" and b.in.tree.nodes[(n.lhs).int()].tag == .field_access and
        b.in.resolutions[(n.lhs).int()] != .func and calleeOf(b, n) == .none;
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
    const layout = b.in.layouts[b.in.nodes.typeOf(node_idx).struct_id];
    const args = Ast.rangeSlice(b.in.tree, (n.rhs).int());
    std.debug.assert(args.len == layout.field_types.len); // guaranteed by the check-error gate
    for (args, 0..) |arg, i| {
        const faddr = try b.emit(.{ .field_addr = .{ .base = dst_ptr, .off = layout.offsets[i], .ty = layout.field_types[i] } }, Typecheck.Type.int);
        try lowerExprInto(b, arg, faddr, layout.field_types[i]);
    }
}

// noinline keeps the builtin arms' locals out of the frame of `lowerCall`, which the
// lowerer re-enters once per nested call.
/// A builtin method call the checker recorded as `k`: no `.call`/symbol, each lowers
/// inline to machine ops. The string builtins are immutable/functional: `concat` and an
/// int/bool `to_string` produce a fresh `str`, a `str` `to_string` shares the receiver,
/// `len`/`byte_at` read the value.
noinline fn lowerBuiltinMethod(b: *Builder, node_idx: Ast.Index, n: Ast.Node, k: Callee.Builtin) error{OutOfMemory}!Ir.Operand {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.bool;
    const fa_node = b.in.tree.nodes[Callee.memberNode(b.in.tree, n).?.int()];
    const recv = fa_node.lhs;
    const args = Ast.rangeSlice(b.in.tree, (n.rhs).int());
    switch (k) {
        .scalar_eq => {
            // int/bool lower to an inline `icmp eq`. str/unit route through the SAME
            // `lowerEqValue` the `==` operator uses (heap-free byte-compare / trivially
            // -true bconst), so `a.eq(b)` and `a == b` emit identical IR.
            std.debug.assert(args.len == 1);
            const recv_ty = b.in.nodes.typeOf(recv);
            if (isInlineEq(recv_ty.kind)) {
                const lhs = operandValue(try lowerExpr(b, recv));
                const rhs = operandValue(try lowerExpr(b, args[0]));
                return .{ .value = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.bool) };
            }
            return try lowerEqValue(b, recv_ty, recv, args[0], false);
        },
        .scalar_hash => {
            // The completeness layer so `[T has Hash]` works at a scalar T: int/bool hash
            // to their own VALUE (an int is its own hash, a bool is 0/1); str folds its
            // bytes via the SAME heap-free polynomial the derive uses (`hashStrAtPtr`);
            // unit hashes to the fixed seed (the sole `()` value).
            const recv_ty = b.in.nodes.typeOf(recv);
            switch (recv_ty.kind) {
                .int, .bool => return .{ .value = operandValue(try lowerExpr(b, recv)) },
                .str => {
                    const op = try lowerExpr(b, recv);
                    const s = operandSlot(op);
                    if (s == Ir.none_slot) {
                        try b.note(fa_node.main_token, "str '.hash()' operand is not a slot in lower");
                        return .{ .value = try b.emit(.{ .iconst = 0 }, Typecheck.Type.int) };
                    }
                    const base = try b.emit(.{ .slot_addr = s }, Typecheck.Type.int);
                    return .{ .value = try hashStrAtPtr(b, base) };
                },
                else => {
                    _ = try lowerExpr(b, recv); // unit: evaluate for effect, hash a constant
                    return .{ .value = try b.emit(.{ .iconst = hash_seed }, Typecheck.Type.int) };
                },
            }
        },
        .str_concat => return try lowerStrConcat(b, node_idx, recv, args[0]),
        .str_len => {
            const base = try strBase(b, recv);
            const len_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = 8, .ty = int_ty } }, int_ty);
            return .{ .value = try b.emit(.{ .load = .{ .addr = len_addr, .ty = int_ty } }, int_ty) };
        },
        .str_byte_at => {
            const base = try strBase(b, recv);
            const i = operandValue(try lowerExpr(b, args[0]));
            const len_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = 8, .ty = int_ty } }, int_ty);
            const len = try b.emit(.{ .load = .{ .addr = len_addr, .ty = int_ty } }, int_ty);
            // Two guards to a shared panic block (emitZeroGuard-style): i < 0, then i >= len.
            const zero = try b.emit(.{ .iconst = 0 }, int_ty);
            const neg = try b.emit(.{ .icmp = .{ .cc = .lt, .lhs = i, .rhs = zero } }, bool_ty);
            const chk2 = try b.addBlock();
            const panic_blk = try b.addBlock();
            b.setTerm(.{ .cond_br = .{ .cond = neg, .t = panic_blk, .f = chk2 } });
            b.switchTo(chk2);
            const over = try b.emit(.{ .icmp = .{ .cc = .ge, .lhs = i, .rhs = len } }, bool_ty);
            const cont = try b.addBlock();
            b.setTerm(.{ .cond_br = .{ .cond = over, .t = panic_blk, .f = cont } });
            b.switchTo(panic_blk);
            b.setTerm(.{ .panic = .index_oob });
            b.switchTo(cont);
            const ptr = try b.emit(.{ .load = .{ .addr = base, .ty = int_ty } }, int_ty);
            const bx = try b.emit(.{ .add = .{ .lhs = ptr, .rhs = i } }, int_ty);
            return .{ .value = try b.emit(.{ .load_byte = bx }, int_ty) };
        },
        .scalar_to_string => {
            // The `displayToSlot` renderer. `str` is identity (share the immutable receiver,
            // no alloc); int/bool/unit build a fresh str (via `__int_to_str` or
            // `__uint_to_str` / the literal cond / the const-data `"()"`).
            const recv_ty = b.in.nodes.typeOf(recv);
            if (recv_ty.kind == .str) return try lowerExpr(b, recv);
            const op = try lowerExpr(b, recv);
            return .{ .slot = try displayToSlot(b, recv_ty, op) };
        },
        .display_to_string => {
            // A struct/enum (incl `char`) `.to_string()`: render through the `displayToSlot`
            // authority. A custom `impl Display` and a structural derive land in the same
            // witness call, which is what makes `to_string` universal.
            const recv_ty = b.in.nodes.typeOf(recv);
            const op = try lowerExpr(b, recv);
            return .{ .slot = try displayToSlot(b, recv_ty, op) };
        },
        .opt_is_tag0, .opt_is_tag1, .opt_unwrap, .opt_unwrap_or => return try lowerOptionResultMethod(b, n, .{ .recv = recv, .op = Typecheck.builtinOptResult(k).? }),
        .conv_into => return try lowerConvMethod(b, node_idx, .{ .recv = recv, .into = true }),
        .conv_try_into => return try lowerConvMethod(b, node_idx, .{ .recv = recv, .into = false }),
    }
}

/// `into` picks the lossless `into` over the fallible `try_into`.
const ConvCall = struct { recv: Ast.Index, into: bool };

/// Whether `ty` is the compiler-provided `char` struct (the id the checker used, threaded
/// via `Inputs.char_struct`). Null (a prelude-less test caller) means no type is char.
pub fn isCharTy(b: *const Builder, ty: Typecheck.Type) bool {
    return Typecheck.isCharTy(ty, b.in.char_struct);
}

const OptResultCall = struct { recv: Ast.Index, op: Typecheck.OptResultMethod };

/// The checker's selection for call `n`, read from its slot.
fn calleeOf(b: *const Builder, n: Ast.Node) Callee.View {
    const slot = Callee.callSlot(b.in.tree, n) orelse return .none;
    return b.in.nodes.callee(slot).view();
}

/// Inline a native `Option`/`Result` method. The receiver is materialized into a
/// fresh slot (so a temporary construction receiver like `Option[int].some(40).unwrap()`
/// works, not just a local), giving a base ptr for `get_tag` + the payload load. Variant
/// 0 is the payload variant, 1 the absence/error (fixed by `registerPrelude`), so
/// `tag == 0` means present. `is_some`/`is_ok` = `tag == 0`; `is_none`/`is_err` =
/// `tag == 1`; `unwrap` cond-branches to a `.trap` on absence; `unwrap_or` merges the
/// payload / the default through a join block-arg. Scalar-payload T only (the accept
/// set is int/bool); aggregate-payload T is deferred (its builtin Callee already keeps it
/// from misrouting as a variant construction).
fn lowerOptionResultMethod(b: *Builder, n: Ast.Node, om: OptResultCall) error{OutOfMemory}!Ir.Operand {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.bool;
    const recv_ty = b.in.nodes.typeOf(om.recv);
    const e = b.in.enum_layouts[recv_ty.enum_id];

    // The native path ships scalar-payload `unwrap`/`unwrap_or` only: a non-scalar payload would
    // truncate a str/aggregate to one 8-byte load, and an aggregate join block-arg
    // crashes codegen (unassigned value offset). The checker already rejects this with
    // T0018, so this is a defensive clean-fail should an aggregate reach `lower`.
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
            const payload_ty = Typecheck.somePayload(e);
            const zero = try b.emit(.{ .iconst = 0 }, int_ty);
            const present = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = tag, .rhs = zero } }, bool_ty);
            const ok_blk = try b.addBlock();
            const trap_blk = try b.addBlock();
            const join = try b.addBlock();
            // An AGGREGATE payload (str/struct/enum) cannot ride a join block-arg (a
            // struct phi crashes codegen); route it through a result slot the some-arm
            // copies into, an argless join, then yield the slot — mirroring lowerTryInto.
            // A scalar/ref payload keeps the block-arg join, so its IR is byte-identical.
            if (isAggTy(b, payload_ty)) {
                const res = try aggSlot(b, payload_ty);
                b.setTerm(.{ .cond_br = .{ .cond = present, .t = ok_blk, .f = trap_blk } });
                b.switchTo(ok_blk);
                const src = try addrAtOff(b, base, Typecheck.payloadAddr(e, 0), payload_ty);
                try copyValueByType(b, res.addr, src, payload_ty);
                try brTo(b, join, .none);
                b.switchTo(trap_blk);
                b.setTerm(.trap);
                b.switchTo(join);
                return .{ .slot = res.slot };
            }
            const merge_ty: Typecheck.Type = switch (passKind(b, payload_ty)) {
                .box => box_repr_ty,
                else => payload_ty,
            };
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
            const payload_ty = Typecheck.somePayload(e);
            const args = Ast.rangeSlice(b.in.tree, (n.rhs).int());
            const zero = try b.emit(.{ .iconst = 0 }, int_ty);
            const present = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = tag, .rhs = zero } }, bool_ty);
            const some_blk = try b.addBlock();
            const none_blk = try b.addBlock();
            const join = try b.addBlock();
            if (isAggTy(b, payload_ty)) {
                const res = try aggSlot(b, payload_ty);
                b.setTerm(.{ .cond_br = .{ .cond = present, .t = some_blk, .f = none_blk } });
                b.switchTo(some_blk);
                const src = try addrAtOff(b, base, Typecheck.payloadAddr(e, 0), payload_ty);
                try copyValueByType(b, res.addr, src, payload_ty);
                try brTo(b, join, .none);
                b.switchTo(none_blk);
                try lowerExprInto(b, args[0], res.addr, payload_ty);
                try brTo(b, join, .none);
                b.switchTo(join);
                return .{ .slot = res.slot };
            }
            const merge_ty: Typecheck.Type = switch (passKind(b, payload_ty)) {
                .box => box_repr_ty,
                else => payload_ty,
            };
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
    const bool_ty = Typecheck.Type.bool;
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
    const bool_ty = Typecheck.Type.bool;
    const recv_ty = b.in.nodes.typeOf(cv.recv);
    const recv_is_char = isCharTy(b, recv_ty);

    if (cv.into) {
        if (recv_is_char) {
            // char -> int: the codepoint field, re-canonicalized (uxt) to a clean value.
            const v = try loadCharCodepoint(b, cv.recv);
            return .{ .value = try recanonToWidth(b, v, Typecheck.Type.uint32) };
        }
        const into_ty = b.in.nodes.typeOf(node_idx);
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
    const call_ty = b.in.nodes.typeOf(node_idx);
    const e = b.in.enum_layouts[call_ty.enum_id];
    const dst_T = Typecheck.somePayload(e);

    // int→char / char→byte are heavy (~24 / ~12 SSA cells); inlining >10 / >20 per fn
    // overflowed the imm12 frame cap. Emit each as ONE shared witness and CALL it (O(1)/site).
    // float -> int is one too (never inlined); `v` is then the raw f64 value cell, which
    // convCallWitness passes as a float arg -> v0.
    if (convWitnessKind(&b.in, recv_ty, call_ty)) |kind| return convCallWitness(b, kind, v, call_ty);

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
    const ok_off = Typecheck.payloadAddr(e, 0);
    const ok_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = ok_off, .ty = ok_store_ty } }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = ok_addr, .val = ok_value, .ty = ok_store_ty } }, null);
    try brTo(b, join, .none);

    b.switchTo(err_blk);
    const err_tag = try b.emit(.{ .iconst = 1 }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = base, .val = err_tag, .ty = int_ty } }, null);
    const err_off = Typecheck.payloadAddr(e, 1);
    const err_fty = Typecheck.errPayload(e);
    const err_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = err_off, .ty = err_fty } }, int_ty);
    const conv_err_tag = try b.emit(.{ .iconst = 0 }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = err_addr, .val = conv_err_tag, .ty = int_ty } }, null);
    try brTo(b, join, .none);
}

/// Resolve the shared fallible-char-conversion witness of `kind` by scanning the derive
/// table for its recipe (the synthesis barrier appended exactly one per used kind). Null
/// only on a mis-wired build (the checker gate guarantees a recipe at any real conv site).
fn convWitnessCallee(b: *Builder, kind: Derive.Kind) ?Link.SymName {
    const di = convWitness(&b.in, kind) orelse return null;
    return .{ .kind = .user_fn, .name = b.in.derives[di].name.? };
}

/// The shared witness a `try_into` from `recv_ty` to the `Result` `call_ty` calls, or null
/// when the conversion is computed inline (an int narrow). The fingerprint folds the same
/// witness ([C2]).
pub fn convWitnessKind(in: *const Inputs, recv_ty: Typecheck.Type, call_ty: Typecheck.Type) ?Derive.Kind {
    const dst_T = Typecheck.somePayload(in.enum_layouts[call_ty.enum_id]);
    if (Typecheck.isCharTy(dst_T, in.char_struct)) return .conv_int_char;
    if (Typecheck.isCharTy(recv_ty, in.char_struct)) return .conv_char_byte;
    if (recv_ty.kind == .float) return .conv_float_int;
    return null;
}

/// The derive index of the program-global witness of `kind`.
pub fn convWitness(in: *const Inputs, kind: Derive.Kind) ?usize {
    for (in.derives, 0..) |d, i| if (d.kind == kind) return i;
    return null;
}

/// Lower a fallible char conversion as a CALL to its shared witness into a fresh Result
/// slot: `ret_slot` + the raw source scalar arg + the `bl`. O(1) frame per site (vs the
/// former ~24/~12-cell inline that overflowed the frame). `v` is passed RAW (no widen) —
/// exactly what the old inline consumed — so the verdicts are byte-identical.
fn convCallWitness(b: *Builder, kind: Derive.Kind, v: Ir.ValueId, call_ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const slot = try b.addSlot(call_ty);
    const callee = convWitnessCallee(b, kind) orelse {
        try b.diags.append(b.gpa, .{ .byte_offset = 0, .end = 0, .message = "conv witness missing in lower" });
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
    const payload_ty = Typecheck.somePayload(e);
    const off = Typecheck.payloadAddr(e, 0);
    const pa = try b.emit(.{ .field_addr = .{ .base = base, .off = off, .ty = payload_ty } }, int_ty);
    // A managed-box payload is an 8-byte scalar cell pointer: load it at `int` WIDTH (a
    // struct-typed `.load` misfires in codegen) but record the produced VALUE as a managed
    // box (`box_repr_ty`), so the precise root map projects it. The width choice is
    // byte-neutral; only the value's cell type changes.
    const is_box = passKind(b, payload_ty) == .box;
    const load_width: Typecheck.Type = if (is_box) int_ty else payload_ty;
    const result_ty: Typecheck.Type = if (is_box) box_repr_ty else payload_ty;
    return try b.emit(.{ .load = .{ .addr = pa, .ty = load_width } }, result_ty);
}

pub fn lowerExprInto(b: *Builder, expr: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(expr).int()];
    // A managed box produces an 8-byte scalar cell pointer straight into the destination.
    switch (passKind(b, ty)) {
        .box, .int, .bool, .float, .unit, .never => return storeOperandInto(b, try lowerExpr(b, expr), dst_ptr, ty, n.main_token),
        .str, .@"struct", .@"enum" => {},
        .rawptr, .invalid => {
            try b.note(n.main_token, "produce-into-slot type unsupported in lower");
            return;
        },
    }

    // Aggregate: dispatch on the producing construct.
    switch (n.tag) {
        .literal_string => try storeStrLiteralInto(b, expr, dst_ptr),
        .literal_char => try lowerCharLiteralInto(b, expr, dst_ptr),
        .struct_init => try lowerStructInitInto(b, expr, dst_ptr),
        // The checker typed a module-headed `lib.S { .. }` as the struct it names.
        .enum_init_struct => if (ty.kind == .@"struct")
            try lowerStructInitInto(b, expr, dst_ptr)
        else
            try lowerEnumInitInto(b, expr, dst_ptr, ty),
        .enum_init_unit, .enum_init_tuple => try lowerEnumInitInto(b, expr, dst_ptr, ty),
        // qualified `N.V` (field_access) / `N.V(args)` (call) that typecheck
        // classified as enum-valued are variant constructions, NOT field reads /
        // real calls. Detect by enum type + shape.
        .field_access => {
            if (ty.kind == .@"enum" and symbols_res.isQualifiedVariant(b.in.tree, b.in.resolutions, expr)) {
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
        .identifier, .lambda => try copyAggInto(b, expr, dst_ptr, ty),
        // An empty list literal `[]`: construct via `new()` into a temp, then copy the
        // handle into the destination (mirrors a `.new()` call routed through copyAggInto).
        .empty_list => {
            const slot = try lowerEmptyListSlot(b, expr, ty);
            const src = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            _ = try b.emit(.{ .copy = .{ .dst = dst_ptr, .src = src, .ty = ty } }, null);
        },
        // A non-empty list literal: build into a temp then copy the handle out (mirrors
        // the `.empty_list` into-dest arm).
        .list_literal => {
            const slot = try lowerListLiteralSlot(b, expr, ty);
            const src = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
            _ = try b.emit(.{ .copy = .{ .dst = dst_ptr, .src = src, .ty = ty } }, null);
        },
        // An aggregate `Vec[V]` element read: lowerIndex yields its `.slot`, copy out.
        // A value-headed `type_app` is the same element read the parser could not tell
        // from a turbofish.
        .index, .type_app => try copyAggInto(b, expr, dst_ptr, ty),
        // An arithmetic operator on struct/enum operands desugars to an
        // aggregate-returning Add/Sub/Mul/Div witness call; `lowerExpr` yields its
        // `.slot`, so copy those bytes into the destination like any other agg result.
        .binary => try copyAggInto(b, expr, dst_ptr, ty),
        else => try b.note(n.main_token, "aggregate expression unsupported in lower"),
    }
}

/// Whether `lowerExprInto` writes `expr`'s aggregate value into the destination piece by
/// piece while it still evaluates parts of it, rather than finishing the value elsewhere
/// and copying it in whole.
pub fn producesInPlace(b: *Builder, expr: Ast.Index) bool {
    const n = b.in.tree.nodes[expr.int()];
    return switch (n.tag) {
        .struct_init, .enum_init_tuple, .enum_init_struct => true,
        .if_stmt, .block, .unsafe_block, .loop_expr, .labeled, .match_expr => true,
        .call => blk: {
            const ty = b.in.nodes.typeOf(expr);
            break :blk isTupleStructCtorCall(b, n, ty) or isQualifiedVariantCtorCall(b, n, ty);
        },
        else => false,
    };
}

/// An aggregate value yielded by an identifier / field-access / call: lower it to
/// an Operand.slot, then byte-copy that slot into `dst_ptr`.
fn copyAggInto(b: *Builder, expr: Ast.Index, dst_ptr: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    try storeOperandInto(b, try lowerExpr(b, expr), dst_ptr, ty, b.in.tree.nodes[expr.int()].main_token);
}

/// The ptr VALUE backing an aggregate Operand (its slot address). For a scalar/
/// none Operand this is a lowering bug → `none_value`.
pub fn operandPtr(b: *Builder, op: Ir.Operand) error{OutOfMemory}!Ir.ValueId {
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

/// Lower an annotated empty-list literal `[]` (type `ty`, a `Vec[T]`) into a fresh temp
/// slot by calling the type's associated `new()` constructor. This is the SAME mangled
/// symbol `Vec[T].new()` dispatches to, so `[]` is pure sugar for `.new()`.
fn lowerEmptyListSlot(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.SlotId {
    const slot = try b.addSlot(ty);
    const dm = comptime StdNames.desugarMethods(.empty_list);
    comptime std.debug.assert(dm.len == 1); // dm[0]: new
    const m = Typecheck.findMethod(b.in.methods, ty, dm[0].name) orelse {
        try b.note(b.in.tree.nodes[(node_idx).int()].main_token, "empty-list '[]' constructor unresolved in lower");
        return slot;
    };
    const callee = witnessCallee(b, m);
    const args = try b.gpa.alloc(Ir.Operand, 0);
    errdefer b.gpa.free(args);
    _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = slot } }, null);
    return slot;
}

/// Lower a non-empty list literal `[e0, ..]` (type `ty`, a `Vec[V]`) into a fresh temp
/// slot: call the type's associated `new()`, then one `push(self, ei)` per element.
/// `push` takes `self` by value (the 8-byte handle), so the same slot is passed as the
/// receiver each time — the header never moves, so every push targets the shared
/// backing. An aggregate element rides `push`'s arg by slot (its `store` copies
/// `size_of[V]` bytes), a scalar by value.
fn lowerListLiteralSlot(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.SlotId {
    const n = b.in.tree.nodes[(node_idx).int()];
    const slot = try b.addSlot(ty);
    const dm = comptime StdNames.desugarMethods(.list_literal);
    comptime std.debug.assert(dm.len == 2); // dm[0]: new, dm[1]: push
    const nm = Typecheck.findMethod(b.in.methods, ty, dm[0].name) orelse {
        try b.note(n.main_token, "list-literal 'new' constructor unresolved in lower");
        return slot;
    };
    {
        const new_args = try b.gpa.alloc(Ir.Operand, 0);
        errdefer b.gpa.free(new_args);
        _ = try b.emit(.{ .call = .{ .callee = witnessCallee(b, nm), .args = new_args, .ret_slot = slot } }, null);
    }
    const pm = Typecheck.findMethod(b.in.methods, ty, dm[1].name) orelse {
        try b.note(n.main_token, "list-literal 'push' unresolved in lower");
        return slot;
    };
    const push_callee = witnessCallee(b, pm);
    for (Ast.rangeSlice(b.in.tree, (n.rhs).int())) |ei| {
        const args = try b.gpa.alloc(Ir.Operand, 2);
        errdefer b.gpa.free(args);
        args[0] = .{ .slot = slot };
        args[1] = try lowerExpr(b, ei);
        _ = try b.emit(.{ .call = .{ .callee = push_callee, .args = args, .ret_slot = Ir.none_slot } }, null);
    }
    return slot;
}

/// The `core/mem` element primitive instance `prim[V]` a `Vec[V]` index desugars to, or
/// null when the primitive or its instance is absent (an internal invariant break —
/// scanCalls enqueues the instance whenever an index node types concrete). The
/// fingerprint folds the same instance ([C2]).
pub fn elemPrimInstance(in: *const Inputs, prim: StdNames.ElemPrim, elem_ty: Typecheck.Type) ?usize {
    var gid: ?u32 = null;
    for (in.names, 0..) |sn, i| {
        if (sn.kind == .user_fn and std.mem.eql(u8, sn.name, prim.name())) {
            gid = @intCast(i);
            break;
        }
    }
    const g = gid orelse return null;
    const targs = [_]Typecheck.Type{elem_ty};
    return Mono.find(in.instances, g, &targs);
}

fn elemPrimCallee(b: *Builder, prim: StdNames.ElemPrim, elem_ty: Typecheck.Type) ?Link.SymName {
    const ii = elemPrimInstance(&b.in, prim, elem_ty) orelse return null;
    return .{ .kind = .user_fn, .name = b.in.instances[ii].name.? };
}

/// A `Vec[V]` element named for a store: its handle and its index, each evaluated once.
pub const ElemRef = struct {
    handle: Ir.ValueId,
    idx: Ir.ValueId,
    tok: u32,
    ty: Typecheck.Type,
};

/// Evaluate the handle and then the index of the element `place` (an `.index`, or a
/// value-headed `type_app`) that a store writes, or null when the receiver has no address.
pub fn lowerElemRef(b: *Builder, place: Ast.Index) error{OutOfMemory}!?ElemRef {
    const n = b.in.tree.nodes[place.int()];
    const idx = if (n.tag == .type_app) Ast.rangeSlice(b.in.tree, n.rhs.int())[0] else n.rhs;
    const base = try operandPtr(b, try lowerExpr(b, n.lhs));
    if (base == Ir.none_value) return null;
    // The code that runs before the write may drop the Vec's last other reference (it can
    // rebind `xs`) and collect, so the handle is typed as a managed box: the root map then
    // keeps the Vec alive.
    const handle = try b.emit(.{ .load = .{ .addr = base, .ty = Typecheck.Type.int } }, box_repr_ty);
    const idx_val = operandValue(try lowerExpr(b, idx));
    return .{ .handle = handle, .idx = idx_val, .tok = n.main_token, .ty = b.in.nodes.typeOf(place) };
}

/// Read the element `r` names through `mem.ga_at[V]`.
pub fn emitElemRefRead(b: *Builder, r: ElemRef) error{OutOfMemory}!Ir.Operand {
    return emitElemRead(b, r.handle, r.idx, r.tok, r.ty);
}

/// Store `val` to the element `r` names through `mem.ga_set_at[V]`, which checks the
/// bounds again.
pub fn emitElemRefStore(b: *Builder, r: ElemRef, val: Ir.Operand) error{OutOfMemory}!void {
    const callee = elemPrimCallee(b, .set_at, r.ty) orelse {
        try b.note(r.tok, "unresolved 'ga_set_at' index primitive in lower");
        return;
    };
    const args = try b.gpa.alloc(Ir.Operand, 3);
    errdefer b.gpa.free(args);
    args[0] = .{ .value = r.handle };
    args[1] = .{ .value = r.idx };
    args[2] = val;
    _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, null);
}

/// Lower `recv[i] = v` as `mem.ga_set_at[V](recv.h, i, v)`: the handle, the index and the
/// value in source order, then the checked store, as `recv.set(i, v)` evaluates. A compound
/// `recv[i] op= v` reads the element through `mem.ga_at[V]` after the index and before `v`,
/// so the handle and index are evaluated once and an out-of-bounds index panics before `v`.
fn lowerIndexStore(b: *Builder, place: Ast.Index, value: StoreValue) error{OutOfMemory}!void {
    const r = try lowerElemRef(b, place) orelse return;
    const val = switch (value) {
        .expr => |e| try lowerExpr(b, e),
        .compound => blk: {
            const outer = b.pushPrelowered(place, try emitElemRefRead(b, r));
            defer b.prelowered = outer;
            break :blk try lowerStoreValue(b, value);
        },
    };
    try emitElemRefStore(b, r, val);
}

/// Lower a value index `recv[i]` reading a `Vec[V]` element. The receiver's single field
/// `h` (offset 0) is the 8-byte `gc_array[V]` handle, passed to `mem.ga_at[V]` along with
/// the index — one call to the bounds-checked element-read primitive that owns the header
/// offsets and stride (see `core/mem`). The result is placed by how V travels: an
/// aggregate into a fresh slot, a scalar/box in a register.
fn lowerIndex(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const n = b.in.tree.nodes[(node_idx).int()];
    return try lowerIndexCore(b, n.lhs, n.rhs, n.main_token, ty);
}

/// A bracketed-postfix `type_app` whose head chain's LEFTMOST identifier resolves to a
/// value is a value index the parser could not tell from a turbofish; mirrors
/// `BodyChecker.appHeadIsValue` via the shared authority so lower reinterprets exactly
/// the nodes the checker typed as an index.
pub fn appHeadIsValue(b: *Builder, node_idx: Ast.Index) bool {
    const n = b.in.tree.nodes[(node_idx).int()];
    return n.tag == .type_app and symbols_res.leftmostHeadIsValue(b.in.tree, b.in.resolutions, n.lhs);
}

fn lowerIndexCore(b: *Builder, recv: Ast.Index, idx: Ast.Index, main_token: u32, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const int_ty = Typecheck.Type.int;
    const recv_op = try lowerExpr(b, recv);
    const base = try operandPtr(b, recv_op);
    if (base == Ir.none_value) return .none;
    const handle = try b.emit(.{ .load = .{ .addr = base, .ty = int_ty } }, int_ty);
    const idx_val = operandValue(try lowerExpr(b, idx));
    return emitElemRead(b, handle, idx_val, main_token, ty);
}

/// Call `mem.ga_at[V](handle, idx)`, placing the element by how V travels: an aggregate
/// into a fresh slot, a scalar/box in a register.
fn emitElemRead(b: *Builder, handle: Ir.ValueId, idx_val: Ir.ValueId, main_token: u32, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const callee = elemPrimCallee(b, .at, ty) orelse {
        try b.note(main_token, "unresolved 'ga_at' index primitive in lower");
        return .none;
    };
    const args = try b.gpa.alloc(Ir.Operand, 2);
    errdefer b.gpa.free(args);
    args[0] = .{ .value = handle };
    args[1] = .{ .value = idx_val };
    switch (passKind(b, ty)) {
        .box => return .{ .value = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, box_repr_ty) },
        .int, .bool, .float, .rawptr => return .{ .value = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = Ir.none_slot } }, ty) },
        .str, .@"struct", .@"enum" => {
            const slot = try b.addSlot(ty);
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = slot } }, null);
            return .{ .slot = slot };
        },
        .unit, .never, .invalid => {
            const slot = try b.addSlot(ty);
            _ = try b.emit(.{ .call = .{ .callee = callee, .args = args, .ret_slot = slot } }, null);
            try b.note(main_token, "index result type unsupported in lower");
            return .{ .slot = slot };
        },
    }
}

/// Materialize an aggregate-producing expression into a fresh temp slot and return
/// Operand.slot. Used in value context (an arg, a nested field initializer source,
/// a scrutinee spill).
pub fn aggregateValue(b: *Builder, expr: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
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
    if (closure.fnValueRef(b, node_idx)) |gid| return try closure.lowerFnValue(b, node_idx, gid, ty);
    if (closure.boundMethodRef(b, node_idx)) |mi| return try closure.lowerBoundValue(b, node_idx, mi, ty);
    // A function-typed member that is no field of its base is a bound method the seal
    // could not resolve; reading it as a field would read the base itself.
    if (LayoutEngine.isFnStruct(ty, b.in.layouts) and !closure.namesField(b, node_idx)) {
        try b.note(b.in.tree.nodes[(node_idx).int()].main_token, "bound method unresolved in lower");
        return .none;
    }
    // A qualified enum unit-variant `N.V` reaches here as field_access but is a
    // construction, not a field read.
    if (ty.kind == .@"enum" and symbols_res.isQualifiedVariant(b.in.tree, b.in.resolutions, node_idx)) {
        return try aggregateValue(b, node_idx, ty);
    }
    const addr = try places.lowerPlaceAddr(b, node_idx);
    if (addr == Ir.none_value) return .none;
    // A managed-box field reads as an 8-byte scalar cell pointer, not by aggregate copy.
    switch (passKind(b, ty)) {
        .box => return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = Typecheck.Type.int } }, box_repr_ty) },
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
        .rawptr, .never, .invalid => {
            try b.note(b.in.tree.nodes[(node_idx).int()].main_token, "field access type unsupported in lower");
            return .none;
        },
    }
}

/// Store into a `field_access` place (`p.x = v`, `p.a.b = v`). The place address is
/// a field_addr chain rooted at a local slot; scalar → store, aggregate → produce
/// into the place address.
fn lowerFieldStore(b: *Builder, place: Ast.Index, value: StoreValue, ty: Typecheck.Type) error{OutOfMemory}!void {
    if (places.elementRoot(b, place)) |root| return lowerElementFieldStore(b, place, root, value, ty);
    const addr = try places.lowerPlaceAddr(b, place);
    if (addr == Ir.none_value) {
        try b.note(b.in.tree.nodes[(place).int()].main_token, "field store target unsupported in lower");
        return;
    }
    if (isAggregatePass(b, ty) and places.stagesStore(b, place, value)) return places.storeStaged(b, value, addr, ty);
    try storeThrough(b, addr, value, ty, b.in.tree.nodes[(place).int()].main_token);
}

/// Store into a field of a `Vec` element (`ps[i].f = v`, `ps[i].q.a op= v`) by
/// read-modify-write: the handle and the index, then (for a compound store) the old field
/// through a read of the element, then the value, then copy the element in,
/// store the field into the copy and write the copy back. No user code runs between the
/// copy-in and the write-back, so a write the value makes to the element is kept, and a
/// value that grows the Vec is copied in from the new backing.
fn lowerElementFieldStore(b: *Builder, place: Ast.Index, root: Ast.Index, value: StoreValue, ty: Typecheck.Type) error{OutOfMemory}!void {
    const tok = b.in.tree.nodes[place.int()].main_token;
    var ep = try places.lowerElementPlace(b, root) orelse return;
    const v = switch (value) {
        .expr => |e| try lowerExpr(b, e),
        // The old field is read from its own copy of the element, before the value runs,
        // so `copyIn` below still sees every write the value makes (D54).
        .compound => blk: {
            const outer = b.pushPrelowered(root, try emitElemRefRead(b, ep.ref));
            defer b.prelowered = outer;
            break :blk try lowerStoreValue(b, value);
        },
    };
    if (b.termSet()) return;
    const addr = try places.copyIn(b, &ep, place);
    if (addr == Ir.none_value) {
        try b.note(tok, "field store target unsupported in lower");
        return;
    }
    try storeOperandInto(b, v, addr, ty, tok);
    try places.writeBack(b, ep);
}

/// Store `value` (of type `ty`) into the storage at `addr`.
fn storeThrough(b: *Builder, addr: Ir.ValueId, value: StoreValue, ty: Typecheck.Type, tok: u32) error{OutOfMemory}!void {
    switch (ty.kind) {
        .int, .bool, .float => {
            const v = operandValue(try lowerStoreValue(b, value));
            _ = try b.emit(.{ .store = .{ .addr = addr, .val = v, .ty = ty } }, null);
        },
        .str, .@"struct", .@"enum" => try lowerStoreValueInto(b, value, addr, ty),
        // A zero-sized place (`()`): evaluate the RHS for effect, store nothing.
        .unit => _ = try lowerStoreValue(b, value),
        else => try b.note(tok, "field store type unsupported in lower"),
    }
}

/// Write a `Name { field: value, ... }` construction into the destination ptr.
/// Each field initializer writes at `layout.offsets[idx]` (mapped by name, so a
/// reordered initializer list is fine), scalars via store, aggregates recursing
/// through `lowerExprInto` at a field_addr.
fn lowerStructInitInto(b: *Builder, node_idx: Ast.Index, dst_ptr: Ir.ValueId) error{OutOfMemory}!void {
    const n = b.in.tree.nodes[(node_idx).int()];
    const id = b.in.nodes.typeOf(node_idx).struct_id;
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

/// A managed box is a struct type that travels as its 8-byte cell pointer, so a value
/// produced into `slot` reads back like a scalar: the struct arm's `aggregateValue` would
/// produce it through `lowerExprInto`, which lowers a box by value, straight back here.
fn loadBoxSlot(b: *Builder, slot: Ir.SlotId) error{OutOfMemory}!Ir.Operand {
    const addr = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int);
    return .{ .value = try b.emit(.{ .load = .{ .addr = addr, .ty = Typecheck.Type.int } }, box_repr_ty) };
}

/// A `match` in VALUE context yielding a SCALAR (or str/struct/enum). Materialize
/// the result via `lowerMatchInto` into a fresh slot, then yield: a scalar loads
/// from the slot to a value; an aggregate yields Operand.slot.
fn lowerMatchValue(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    if (passKind(b, ty) == .box) {
        const slot = try b.addSlot(ty);
        try lowerMatchInto(b, node_idx, try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int), ty);
        return try loadBoxSlot(b, slot);
    }
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
    const scrut_ty = b.in.nodes.typeOf(n.lhs);

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
    if (passKind(b, ty) == .box) {
        const slot = try b.addSlot(ty);
        try lowerTryInto(b, node_idx, try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int), ty);
        return try loadBoxSlot(b, slot);
    }
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
    const op_ty = b.in.nodes.typeOf(n.lhs);
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
    const is_happy = try b.emit(.{ .icmp = .{ .cc = .ne, .lhs = tag, .rhs = one } }, Typecheck.Type.bool);
    const happy_bb = try b.addBlock();
    const residual_bb = try b.addBlock();
    b.setTerm(.{ .cond_br = .{ .cond = is_happy, .t = happy_bb, .f = residual_bb } });

    // Residual: build the enclosing return-type residual and early-return it.
    b.switchTo(residual_bb);
    try buildResidual(b, op_base, ol, n.main_token);

    // Happy: extract variant-0's payload into the destination; control continues here.
    b.switchTo(happy_bb);
    const src = try addrAtOff(b, op_base, Typecheck.payloadAddr(ol, 0), ty);
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
        const op_err = Typecheck.errPayload(ol);
        const ret_err = Typecheck.errPayload(rl);
        const src_off = Typecheck.payloadAddr(ol, 1);
        const dst_off = Typecheck.payloadAddr(rl, 1);
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

/// A fresh temp slot for an aggregate result, paired with its base address. The
/// `Option`/`Result` aggregate-payload arms take the slot BEFORE the branch (its addr
/// must dominate both arms — `unwrap_or` writes the default into it from the none arm),
/// then copy the payload in the happy arm, so they hold the pieces apart rather than
/// calling `materializeAgg`.
const AggSlot = struct { slot: Ir.SlotId, addr: Ir.ValueId };

fn aggSlot(b: *Builder, ty: Typecheck.Type) error{OutOfMemory}!AggSlot {
    const slot = try b.addSlot(ty);
    return .{ .slot = slot, .addr = try b.emit(.{ .slot_addr = slot }, Typecheck.Type.int) };
}

/// Copy an aggregate value at `src` into a fresh temp slot, returning it. The single
/// "result slot + copy" primitive for when the source address is already in hand: a value
/// that cannot ride a join block-arg (a struct phi crashes codegen) travels through a slot
/// instead — an index read yields its element this way.
fn materializeAgg(b: *Builder, ty: Typecheck.Type, src: Ir.ValueId) error{OutOfMemory}!Ir.SlotId {
    const res = try aggSlot(b, ty);
    try copyValueByType(b, res.addr, src, ty);
    return res.slot;
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
            const c = try b.emit(.{ .icmp = .{ .cc = .ne, .lhs = cur, .rhs = litv } }, Typecheck.Type.bool);
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
            const c = try b.emit(.{ .icmp = .{ .cc = .ne, .lhs = tagv, .rhs = want } }, Typecheck.Type.bool);
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
    const dst_home = if (places.boxedBinding(b, bind_idx)) |li| try places.bindBox(b, li, ty) else try places.localHome(b, bind_idx, ty);
    const src = try slotFieldAddr(b, base, off, ty);
    const dst = try places.homeAddr(b, dst_home);
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
/// carried as a block param on `join`. Typecheck guarantees an else. An aggregate
/// result is produced into a temp slot instead (see `paramOperand`).
fn lowerIfValue(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    if (isAggregatePass(b, ty)) return try aggregateValue(b, node_idx, ty);
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
    return paramOperand(b, ty, merge);
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
    const node = b.in.tree.nodes[(stmt_idx).int()];
    return switch (node.tag) {
        .expr_stmt => node.lhs,
        // An else-less `if` produces no value; it lowers as an effect statement,
        // so it is not this block's trailing value expression.
        .if_stmt => if (Ast.ifHeaderAt(b.in.tree, (node.rhs).int()).else_node != Ast.none) stmt_idx else null,
        .match_expr, .loop_expr, .labeled, .block => stmt_idx,
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
    // A boxed `i` needs a fresh cell per iteration (a closure keeps the one it captured), so
    // the induction slot is then a hidden counter: each iteration copies it into a new cell,
    // and `inc` copies the cell back first, so assigning `i` still steers the loop.
    const boxed_li = places.boxedBinding(b, stmt_idx);
    const islot = if (boxed_li != null) try b.addSlot(Typecheck.Type.int) else (try places.localHome(b, stmt_idx, Typecheck.Type.int)).slot;

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
    const cmp = try b.emit(.{ .icmp = .{ .cc = .ge, .lhs = iv, .rhs = hi } }, Typecheck.Type.bool);
    b.setTerm(.{ .cond_br = .{ .cond = cmp, .t = done, .f = body } });

    b.switchTo(body);
    if (boxed_li) |li| {
        const cell = try places.homeAddr(b, try places.bindBox(b, li, Typecheck.Type.int));
        const ca = try b.emit(.{ .slot_addr = islot }, Typecheck.Type.int);
        const cur = try b.emit(.{ .load = .{ .addr = ca, .ty = Typecheck.Type.int } }, Typecheck.Type.int);
        _ = try b.emit(.{ .store = .{ .addr = cell, .val = cur, .ty = Typecheck.Type.int } }, null);
    }
    try b.loops.append(b.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .break_bb = done, .continue_bb = inc, .merge = .none });
    try lowerBlockStmts(b, stmt.lhs);
    _ = b.loops.pop();
    if (!b.termSet()) try brTo(b, inc, .none);

    b.switchTo(inc);
    if (boxed_li) |li| {
        const cell = try places.homeAddr(b, b.homes.get(li).?);
        const cur = try b.emit(.{ .load = .{ .addr = cell, .ty = Typecheck.Type.int } }, Typecheck.Type.int);
        const ca = try b.emit(.{ .slot_addr = islot }, Typecheck.Type.int);
        _ = try b.emit(.{ .store = .{ .addr = ca, .val = cur, .ty = Typecheck.Type.int } }, null);
    }
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

/// `iter()` on a for-in's receiver `recv`, its result into `it_slot`. The receiver goes as
/// `iter` declares it: by value, or for `mut self` the place's address (the checker made it
/// a place, T0019). False when that fails to lower (a note is out).
fn emitForIter(b: *Builder, recv: Ast.Index, iter_m: Typecheck.Method, it_slot: Ir.SlotId) error{OutOfMemory}!bool {
    var wbs: std.ArrayList(places.ElementPlace) = .empty;
    defer wbs.deinit(b.gpa);
    const self0: Ir.Operand = if (iter_m.mut_self) blk: {
        const addr = try places.lowerMutPlaceAddr(b, recv, &wbs);
        if (addr == Ir.none_value) {
            try b.note(b.in.tree.nodes[recv.int()].main_token, "for-in: mut-self 'iter' receiver is not a place in lower");
            return false;
        }
        break :blk .{ .value = addr };
    } else try lowerExpr(b, recv);
    const args = try b.gpa.alloc(Ir.Operand, 1);
    errdefer b.gpa.free(args);
    args[0] = self0;
    _ = try b.emit(.{ .call = .{ .callee = witnessCallee(b, iter_m), .args = args, .ret_slot = it_slot } }, null);
    try places.writeBacks(b, wbs.items);
    return true;
}

/// `next()` on the iterator in `it_slot`, its `Option` into `opt_slot`. The iterator goes
/// as `next` declares it: `Iterator.next` declares `mut self` and T0076 holds every witness
/// to that, so it is the slot's address, and a by-value `next` would get the slot's value.
fn emitForNext(b: *Builder, next_m: Typecheck.Method, it_slot: Ir.SlotId, opt_slot: Ir.SlotId) error{OutOfMemory}!void {
    const args = try b.gpa.alloc(Ir.Operand, 1);
    errdefer b.gpa.free(args);
    args[0] = if (next_m.mut_self) .{ .value = try b.emit(.{ .slot_addr = it_slot }, Typecheck.Type.int) } else .{ .slot = it_slot };
    _ = try b.emit(.{ .call = .{ .callee = witnessCallee(b, next_m), .args = args, .ret_slot = opt_slot } }, null);
}

/// What a `for` loop over a value of type `recv_ty` calls: `iter` on it, then `next` on
/// the iterator `iter` returns, with that iterator's type and `next`'s `Option`. The
/// fingerprint folds these same methods, since no call slot names them ([C2]).
pub const ForInCalls = struct { iter: Typecheck.Method, next: Typecheck.Method, iter_ty: Typecheck.Type, opt_ty: Typecheck.Type };

pub const ForInError = error{ NoIter, IterNotInstance, NoNext, NextNotInstance };

pub fn forInCalls(in: *const Inputs, recv_ty: Typecheck.Type) ForInError!ForInCalls {
    const dm = comptime StdNames.desugarMethods(.for_in_stmt);
    comptime std.debug.assert(dm.len == 2); // dm[0]: iter, dm[1]: next
    comptime std.debug.assert(std.mem.eql(u8, StdNames.desugarMethods(.for_in2_stmt)[0].name, dm[0].name) and
        std.mem.eql(u8, StdNames.desugarMethods(.for_in2_stmt)[1].name, dm[1].name));
    const iter_m = Typecheck.findMethod(in.methods, recv_ty, dm[0].name) orelse return error.NoIter;
    const ii = iter_m.instance orelse return error.IterNotInstance;
    const iter_ty = in.instances[ii].ret;
    const next_m = Typecheck.findMethod(in.methods, iter_ty, dm[1].name) orelse return error.NoNext;
    const ni = next_m.instance orelse return error.NextNotInstance;
    return .{ .iter = iter_m, .next = next_m, .iter_ty = iter_ty, .opt_ty = in.instances[ni].ret };
}

fn forInFailure(e: ForInError, comptime what: []const u8) []const u8 {
    return switch (e) {
        error.NoIter => what ++ ": no reified 'iter' method in lower",
        error.IterNotInstance => what ++ ": 'iter' is not a reified instance in lower",
        error.NoNext => what ++ ": no reified 'next' method in lower",
        error.NextNotInstance => what ++ ": 'next' is not a reified instance in lower",
    };
}

/// `for x in xs { body }`: the structural iterator desugar, composing the existing
/// loop + Option-tag primitives (never touches `lowerFor`, so range-for stays
/// byte-identical). Lowers to:
///   `it := xs.iter()`                         (a fresh iterator in a mutable slot)
///   header: `opt := it.next()`; branch on the tag  (some -> body, none -> done)
///   body:   copy Some's payload into `x`; run the loop body; back-edge to header
/// Each receiver goes as its method declares it (`emitForIter`, `emitForNext`).
/// break -> done, continue -> header.
fn lowerForIn(b: *Builder, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
    const stmt = b.in.tree.nodes[(stmt_idx).int()];
    const int_ty = Typecheck.Type.int;

    const calls = forInCalls(&b.in, b.in.nodes.typeOf(stmt.rhs)) catch |e| return b.note(stmt.main_token, forInFailure(e, "for-in"));
    const next_m = calls.next;
    const it_slot = try b.addSlot(calls.iter_ty);
    if (!try emitForIter(b, stmt.rhs, calls.iter, it_slot)) return;

    const opt_ty = calls.opt_ty;
    if (opt_ty.kind != .@"enum" or opt_ty.enum_id >= b.in.enum_layouts.len) {
        try b.note(stmt.main_token, "for-in: 'next' did not reify to an Option enum in lower");
        return;
    }
    const ol = b.in.enum_layouts[opt_ty.enum_id];
    const item_ty = Typecheck.somePayload(ol);
    const opt_slot = try b.addSlot(opt_ty);
    // A boxed `x` gets a fresh cell per iteration instead of a slot.
    const boxed_x = places.boxedBinding(b, stmt_idx);
    const xslot = if (boxed_x != null) Ir.none_slot else (try places.localHome(b, stmt_idx, item_ty)).slot;

    const header = try b.addBlock();
    const body = try b.addBlock();
    const done = try b.addBlock();

    try brTo(b, header, .none);
    b.switchTo(header);
    {
        try emitForNext(b, next_m, it_slot, opt_slot);
        const base = try b.emit(.{ .slot_addr = opt_slot }, int_ty);
        const tag = try b.emit(.{ .get_tag = base }, int_ty);
        const zero = try b.emit(.{ .iconst = 0 }, int_ty);
        const is_some = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = tag, .rhs = zero } }, Typecheck.Type.bool);
        b.setTerm(.{ .cond_br = .{ .cond = is_some, .t = body, .f = done } }); // variant 0 = some
    }

    b.switchTo(body);
    {
        const cell: ?places.LocalHome = if (boxed_x) |li| try places.bindBox(b, li, item_ty) else null;
        const base = try b.emit(.{ .slot_addr = opt_slot }, int_ty);
        const src = try addrAtOff(b, base, Typecheck.payloadAddr(ol, 0), item_ty);
        const dst = if (cell) |c| try places.homeAddr(b, c) else try b.emit(.{ .slot_addr = xslot }, int_ty);
        try copyValueByType(b, dst, src, item_ty); // AGGREGATE-safe payload copy
    }
    try b.loops.append(b.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .break_bb = done, .continue_bb = header, .merge = .none });
    try lowerBlockStmts(b, stmt.lhs);
    _ = b.loops.pop();
    if (!b.termSet()) try brTo(b, header, .none);

    b.switchTo(done);
}

/// `for k, v in m { body }`: the two-binding iterator desugar. Identical machinery to
/// `lowerForIn` (iter into a slot, header `opt := it.next()`, tag branch) except the
/// Some-payload — a reified `Entry[K, V]` struct — is split into TWO slots: field 0 into
/// the first binding, field 1 into the second. Reading `entries[0..len)` in the iterator
/// yields INSERTION order, so the loop observes it directly.
fn lowerForIn2(b: *Builder, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
    const stmt = b.in.tree.nodes[(stmt_idx).int()];
    const h = Ast.forIn2HeaderAt(b.in.tree, stmt.lhs.int());
    const int_ty = Typecheck.Type.int;

    const calls = forInCalls(&b.in, b.in.nodes.typeOf(stmt.rhs)) catch |e| return b.note(stmt.main_token, forInFailure(e, "for-in2"));
    const next_m = calls.next;
    const it_slot = try b.addSlot(calls.iter_ty);
    if (!try emitForIter(b, stmt.rhs, calls.iter, it_slot)) return;

    const opt_ty = calls.opt_ty;
    if (opt_ty.kind != .@"enum" or opt_ty.enum_id >= b.in.enum_layouts.len) {
        try b.note(stmt.main_token, "for-in2: 'next' did not reify to an Option enum in lower");
        return;
    }
    const ol = b.in.enum_layouts[opt_ty.enum_id];
    const item_ty = Typecheck.somePayload(ol); // the reified Entry[K,V] struct
    const opt_slot = try b.addSlot(opt_ty);
    const elayout = b.in.layouts[item_ty.struct_id];
    // A boxed binding gets a fresh cell per iteration instead of a slot.
    const boxed_k = places.boxedBinding(b, stmt_idx);
    const boxed_v = places.boxedBinding(b, h.val_leaf);
    const kslot = if (boxed_k != null) Ir.none_slot else (try places.localHome(b, stmt_idx, elayout.field_types[0])).slot; // first binding
    const vslot = if (boxed_v != null) Ir.none_slot else (try places.localHome(b, h.val_leaf, elayout.field_types[1])).slot; // second binding

    const header = try b.addBlock();
    const body = try b.addBlock();
    const done = try b.addBlock();

    try brTo(b, header, .none);
    b.switchTo(header);
    {
        try emitForNext(b, next_m, it_slot, opt_slot);
        const base = try b.emit(.{ .slot_addr = opt_slot }, int_ty);
        const tag = try b.emit(.{ .get_tag = base }, int_ty);
        const zero = try b.emit(.{ .iconst = 0 }, int_ty);
        const is_some = try b.emit(.{ .icmp = .{ .cc = .eq, .lhs = tag, .rhs = zero } }, Typecheck.Type.bool);
        b.setTerm(.{ .cond_br = .{ .cond = is_some, .t = body, .f = done } }); // variant 0 = some
    }

    b.switchTo(body);
    {
        const kcell: ?places.LocalHome = if (boxed_k) |li| try places.bindBox(b, li, elayout.field_types[0]) else null;
        const vcell: ?places.LocalHome = if (boxed_v) |li| try places.bindBox(b, li, elayout.field_types[1]) else null;
        const base = try b.emit(.{ .slot_addr = opt_slot }, int_ty);
        const payload = try addrAtOff(b, base, Typecheck.payloadAddr(ol, 0), item_ty);
        const kaddr = try addrAtOff(b, payload, elayout.offsets[0], elayout.field_types[0]);
        const kdst = if (kcell) |c| try places.homeAddr(b, c) else try b.emit(.{ .slot_addr = kslot }, int_ty);
        try copyValueByType(b, kdst, kaddr, elayout.field_types[0]);
        const vaddr = try addrAtOff(b, payload, elayout.offsets[1], elayout.field_types[1]);
        const vdst = if (vcell) |c| try places.homeAddr(b, c) else try b.emit(.{ .slot_addr = vslot }, int_ty);
        try copyValueByType(b, vdst, vaddr, elayout.field_types[1]);
    }
    try b.loops.append(b.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .break_bb = done, .continue_bb = header, .merge = .none });
    try lowerBlockStmts(b, h.body);
    _ = b.loops.pop();
    if (!b.termSet()) try brTo(b, header, .none);

    b.switchTo(done);
}

/// VALUE `loop { body }`: infinite loop yielding via `break <expr>`. `top` is the
/// back-edge target; `exit` carries the merge param. A break-less (`never`) loop
/// never reaches `exit` → its body just back-edges; the exit stays unreachable. An
/// aggregate result is produced into a temp slot instead (see `paramOperand`).
fn lowerLoopValue(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type, label: ?[]const u8) error{OutOfMemory}!Ir.Operand {
    if (isAggregatePass(b, ty)) {
        const res = try aggSlot(b, ty);
        try lowerLoopValueInto(b, node_idx, res.addr, ty, label);
        return .{ .slot = res.slot };
    }
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
    return paramOperand(b, ty, merge);
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
        .for_in_stmt => try lowerForIn(b, n.lhs, label),
        .for_in2_stmt => try lowerForIn2(b, n.lhs, label),
        .loop_expr => _ = try lowerLoopValue(b, n.lhs, b.in.nodes.typeOf(n.lhs), label),
        .block => _ = try lowerLabeledBlock(b, n.lhs, b.in.nodes.typeOf(node_idx), label),
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
        .for_in_stmt => {
            try lowerForIn(b, n.lhs, label);
            return .none;
        },
        .for_in2_stmt => {
            try lowerForIn2(b, n.lhs, label);
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
/// the value. The exit's merge param is the block's value; an aggregate result is
/// produced into a temp slot instead (see `paramOperand`).
fn lowerLabeledBlock(b: *Builder, block_idx: Ast.Index, ty: Typecheck.Type, label: []const u8) error{OutOfMemory}!Ir.Operand {
    if (isAggregatePass(b, ty)) {
        const res = try aggSlot(b, ty);
        try lowerLabeledBlockInto(b, block_idx, res.addr, ty, label);
        return .{ .slot = res.slot };
    }
    const exit = try b.addBlock();
    const is_value = ty.kind != .unit and ty.kind != .never;
    const merge: Ir.ValueId = if (is_value) try b.addParam(exit, ty) else Ir.none_value;

    try b.loops.append(b.gpa, .{ .kind = .labeled_block, .label = label, .construct_node = block_idx, .break_bb = exit, .continue_bb = Ir.none_block, .merge = if (is_value) .scalar else .none });
    const tv = try lowerBlockValue(b, block_idx, ty);
    _ = b.loops.pop();
    if (!b.termSet()) try brTo(b, exit, tv); // fall-through value

    b.switchTo(exit);
    return paramOperand(b, ty, merge);
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
                    const lt = b.in.nodes.typeOf(n.lhs);
                    if (isInlineOrd(lt.kind)) {
                        const lhs = operandValue(try lowerExpr(b, n.lhs));
                        const rhs = operandValue(try lowerExpr(b, n.rhs));
                        const cc = condFromToken(op, lt.isUnsignedInt());
                        const c = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.bool);
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
                    const lt = b.in.nodes.typeOf(n.lhs);
                    if (isInlineEq(lt.kind)) {
                        const lhs = operandValue(try lowerExpr(b, n.lhs));
                        const rhs = operandValue(try lowerExpr(b, n.rhs));
                        const cc = condFromToken(op, lt.isUnsignedInt());
                        const c = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = lhs, .rhs = rhs } }, Typecheck.Type.bool);
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
/// param value is `none_value`), else `.value` of the param. Never an aggregate
/// (str/struct/enum, so also a fn pair): the param's frame cell holds its bytes but has
/// no address in the IR, while the consumers that read an aggregate in place (a value
/// call, fn identity `==`, a field read, a `match` scrutinee) take the address of an
/// `Operand.slot`; fed a `.value`, they fail to lower or read garbage. So an aggregate
/// if/loop/labeled-block value is produced into a temp slot, as `match` and `?` are.
fn paramOperand(b: *Builder, ty: Typecheck.Type, merge: Ir.ValueId) Ir.Operand {
    std.debug.assert(!isAggregatePass(b, ty));
    if (merge == Ir.none_value) return .none;
    return .{ .value = merge };
}

/// Extract the `value` of an Operand for an op that needs a scalar value id. A
/// `.none`/`.slot` here is a lowering bug for scalar context; default to
/// `none_value` so we never index garbage (a diagnostic was already emitted).
pub fn operandValue(op: Ir.Operand) Ir.ValueId {
    return switch (op) {
        .value => |v| v,
        else => Ir.none_value,
    };
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
        std.debug.assert(!s.ret.isTypeVar() and !s.ret.isApp() and !s.ret.isFunc()); // reified-away before lower
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
        std.debug.assert(!s.params[slot].isTypeVar() and !s.params[slot].isApp() and !s.params[slot].isFunc()); // reified-away before lower
        return s.params[slot];
    };
    if (in.nodes.has(param_node)) {
        const t = in.nodes.typeOf(param_node);
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
    if (std.mem.eql(u8, name, "bool")) return Typecheck.Type.bool;
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

    const names = try gpa.alloc(Link.SymName, fn_nodes.items.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (fn_nodes.items, 0..) |idx, i| {
        const nm = tokens[tree.nodes[(idx).int()].main_token].text(src);
        names[i] = .{ .kind = .user_fn, .name = try gpa.dupe(u8, nm) };
    }

    const in = Inputs{
        .tree = .{ .nodes = tree.nodes, .extra = tree.extra },
        .tokens = tokens,
        .source = src,
        .resolutions = rr.resolutions[0],
        .nodes = .{ .types = tc.node_types[0], .callees = tc.callees[0] },
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = names,
        .methods = tc.methods,
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
        .closure = .{ .adapters = tc.adapters, .lambda_units = tc.lambda_units },
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

/// How `renderUnit` lowers: frame storage on (as any opt pass turns it on, here with no
/// pass rewriting the IR), and generic calls with a known fn argument sent to spec units.
const UnitOptions = struct { frame: bool = false, specialize: bool = false };

/// Check `src` as a one-module program, lower every unit through the driver's own path
/// (`Codegen.renderIr` at -O0), and return the rendered IR of the unit named `name`, or of
/// the first whose last dotted segment is `name` (a method `main.P.bump` is `bump`).
/// gpa-owned.
fn renderUnit(gpa: std.mem.Allocator, src: []const u8, name: []const u8, o: UnitOptions) ![]u8 {
    const Codegen = @import("driver/Codegen.zig");
    const tokens = try @import("lex.zig").tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try @import("parse.zig").expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    var g = try Graph.single(gpa, "main", "", src, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinit(gpa);
    var res = try ResolveGraph.resolveGraph(gpa, &g);
    defer res.deinit(gpa);
    var tc = try TypecheckGraph.checkGraph(gpa, &g, &res, null, 0);
    defer tc.deinit(gpa);
    const ir = switch (try Codegen.renderIr(gpa, &g, &res, &tc, .{ .opt = .O0, .specialize = o.specialize, .frame = o.frame })) {
        .ok => |text| text,
        .err => |e| {
            std.debug.print("lowering failed: {s}\n", .{e.message});
            return error.TestUnexpectedResult;
        },
    };
    defer gpa.free(ir);
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, ir, at, "fn ")) |start| {
        const open = std.mem.indexOfScalarPos(u8, ir, start, '(') orelse break;
        const end = (std.mem.indexOfPos(u8, ir, start, "\n}\n") orelse break) + 3;
        const sym = ir[start + 3 .. open];
        if (std.mem.eql(u8, sym, name) or (std.mem.endsWith(u8, sym, name) and sym.len > name.len and sym[sym.len - name.len - 1] == '.'))
            return gpa.dupe(u8, ir[start..end]);
        at = end;
    }
    return error.TestUnexpectedResult;
}

/// `renderUnit` with default options, compared to `want`.
fn expectUnit(src: []const u8, name: []const u8, want: []const u8) !void {
    const ir = try renderUnit(testing.allocator, src, name, .{});
    defer testing.allocator.free(ir);
    try testing.expectEqualStrings(want, ir);
}

test "a lambda value is fn_addr of <owner>$l0 with a null env" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn f() -> int {
        \\ g := fn(x: int) -> int { x + 1 }
        \\ return g(1)
        \\}
        \\
    , "f", .{});
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "fn_addr @main.f$l0") != null);
    // The env word is a null rawptr: `iconst 0` stored at +8 as rawptr.
    try testing.expect(std.mem.indexOf(u8, ir, "iconst 0\n  store %") != null);
    try testing.expect(std.mem.indexOf(u8, ir, " : rawptr\n") != null);
    // The lambda's body is not lowered into its owner.
    try testing.expect(std.mem.indexOf(u8, ir, "add ") == null);
    try testing.expect(std.mem.indexOf(u8, ir, "call_indirect") != null);
}

test "a '_' capture lowers like the lambda it means; a hole pipe is a direct call" {
    const gpa = testing.allocator;
    {
        const ir = try renderUnit(gpa,
            \\fn add(a: int, b: int) -> int { a + b }
            \\fn f() -> int {
            \\ g := add(1, _)
            \\ return g(2)
            \\}
            \\
        , "f", .{});
        defer gpa.free(ir);
        try testing.expect(std.mem.indexOf(u8, ir, "fn_addr @main.f$l0") != null);
        try testing.expect(std.mem.indexOf(u8, ir, "iconst 0\n  store %") != null);
        try testing.expect(std.mem.indexOf(u8, ir, "call @gc_alloc(") == null);
        try testing.expect(std.mem.indexOf(u8, ir, "call @main.add(") == null);
    }
    {
        const ir = try renderUnit(gpa,
            \\fn add(a: int, b: int) -> int { a + b }
            \\fn f() -> int {
            \\ a := 5
            \\ g := add(a, _)
            \\ return g(2)
            \\}
            \\
        , "f", .{});
        defer gpa.free(ir);
        try testing.expect(std.mem.indexOf(u8, ir, "fn_addr @main.f$l0") != null);
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, ir, "call @gc_alloc("));
        // An 8-byte env holding `a`, by value.
        try testing.expect(std.mem.indexOf(u8, ir,
            \\  %4 = iconst 8
            \\  %5 = call @gc_alloc(%4)
            \\  %6 = field_addr %5, 0 : int
            \\  %7 = slot_addr s0
            \\  %8 = load %7 : int
            \\  store %6, %8 : int
            \\
        ) != null);
    }
    {
        const ir = try renderUnit(gpa,
            \\fn add(a: int, b: int) -> int { a + b }
            \\fn f() -> int {
            \\ x := 3
            \\ return x |> add(1, _)
            \\}
            \\
        , "f", .{});
        defer gpa.free(ir);
        try testing.expect(std.mem.indexOf(u8, ir, "fn_addr") == null);
        try testing.expect(std.mem.indexOf(u8, ir, "call @main.add(") != null);
    }
}

test "two lambdas and a nested one get ranks 0, 1, 2 in pre-order" {
    const gpa = testing.allocator;
    const src =
        \\fn f() -> int {
        \\ a := fn() -> int { 1 }
        \\ b := fn() -> fn() -> int { fn() { 3 } }
        \\ return a() + b()()
        \\}
        \\
    ;
    const ir = try renderUnit(gpa, src, "f", .{});
    defer gpa.free(ir);
    const l0 = std.mem.indexOf(u8, ir, "fn_addr @main.f$l0").?;
    const l1 = std.mem.indexOf(u8, ir, "fn_addr @main.f$l1").?;
    try testing.expect(l0 < l1);
    // The nested lambda is created by its parent's unit, not by `f`.
    try testing.expect(std.mem.indexOf(u8, ir, "main.f$l2") == null);
    const outer = try renderUnit(gpa, src, "main.f$l1", .{});
    defer gpa.free(outer);
    try testing.expect(std.mem.indexOf(u8, outer, "fn_addr @main.f$l2") != null);
}

test "lowerLambda binds env as slot 0 and params by resolver slot" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn f() -> int {
        \\ a := 5
        \\ g := fn(x: int, y: bool) -> int { if y { x } else { 0 } }
        \\ return a + g(1, true)
        \\}
        \\
    , "main.f$l0", .{});
    defer gpa.free(ir);
    try testing.expectEqualStrings(
        \\fn main.f$l0(s0, s1, s2) -> int {
        \\  slots: s0:rawptr s1:int s2:bool
        \\b0:
        \\  %2 = slot_addr s2
        \\  %3 = load %2 : bool
        \\  cond_br %3, b2, b3
        \\b1(%0:int):
        \\  ret %0
        \\b2:
        \\  %4 = slot_addr s1
        \\  %5 = load %4 : int
        \\  br b4(%5)
        \\b3:
        \\  %6 = iconst 0
        \\  br b4(%6)
        \\b4(%1:int):
        \\  br b1(%1)
        \\}
        \\
    , ir);
}

test "a capturing lambda allocates its env once and stores each capture at its offset" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn f() -> int {
        \\ n := 5
        \\ s := "ab"
        \\ g := fn(k: int) -> int { k + n + s.len() }
        \\ return g(1)
        \\}
        \\
    , "f", .{});
    defer gpa.free(ir);
    // One 24-byte cell, allocated before it is filled: n at +0 by value, s at +8 by copy;
    // then the cell is the pair's env word.
    try testing.expect(std.mem.indexOf(u8, ir,
        \\  %8 = iconst 24
        \\  %9 = call @gc_alloc(%8)
        \\  %10 = field_addr %9, 0 : int
        \\  %11 = slot_addr s0
        \\  %12 = load %11 : int
        \\  store %10, %12 : int
        \\  %13 = field_addr %9, 8 : str
        \\  %14 = slot_addr s1
        \\  copy %13 <- %14 : str
        \\  %15 = slot_addr s3
        \\  %16 = fn_addr @main.f$l0
        \\  store %15, %16 : int
        \\  %17 = field_addr %15, 8 : rawptr
        \\  store %17, %9 : rawptr
        \\
    ) != null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, ir, "call @gc_alloc("));
}

test "a lambda prologue copies captures into locals" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn f() -> int {
        \\ n := 5
        \\ s := "ab"
        \\ g := fn(k: int) -> int { k + n + s.len() }
        \\ return g(1)
        \\}
        \\
    , "main.f$l0", .{});
    defer gpa.free(ir);
    try testing.expectEqualStrings(
        \\fn main.f$l0(s0, s1) -> int {
        \\  slots: s0:rawptr s1:int s2:int s3:str
        \\b0:
        \\  %1 = slot_addr s0
        \\  %2 = load %1 : rawptr
        \\  %3 = field_addr %2, 0 : int
        \\  %4 = slot_addr s2
        \\  %5 = load %3 : int
        \\  store %4, %5 : int
        \\  %6 = field_addr %2, 8 : str
        \\  %7 = slot_addr s3
        \\  copy %7 <- %6 : str
        \\  %8 = slot_addr s1
        \\  %9 = load %8 : int
        \\  %10 = slot_addr s2
        \\  %11 = load %10 : int
        \\  %12 = add %9, %11
        \\  %13 = slot_addr s3
        \\  %14 = field_addr %13, 8 : int
        \\  %15 = load %14 : int
        \\  %16 = add %12, %15
        \\  br b1(%16)
        \\b1(%0:int):
        \\  ret %0
        \\}
        \\
    , ir);
}

test "a boxed local's reads and writes go through its cell" {
    try expectUnit(
        \\fn f() -> int {
        \\ n := 1
        \\ g := fn() { n = n + 1 }
        \\ g()
        \\ n = n + 10
        \\ return n
        \\}
        \\
    , "f",
        // `n` lives in an 8-byte cell held by the rawptr slot s0: its declaration allocates
        // the cell first, the env stores the cell pointer (not the value), and every read
        // and write of `n` loads the pointer.
        \\fn main.f() -> int {
        \\  slots: s0:rawptr s1:fn$unit s2:fn$unit
        \\b0:
        \\  %1 = iconst 8
        \\  %2 = call @gc_alloc(%1)
        \\  %3 = slot_addr s0
        \\  store %3, %2 : rawptr
        \\  %4 = iconst 1
        \\  %5 = slot_addr s0
        \\  %6 = load %5 : rawptr
        \\  store %6, %4 : int
        \\  %7 = slot_addr s1
        \\  %8 = iconst 8
        \\  %9 = call @gc_alloc(%8)
        \\  %10 = field_addr %9, 0 : rawptr
        \\  %11 = slot_addr s0
        \\  %12 = load %11 : rawptr
        \\  store %10, %12 : rawptr
        \\  %13 = slot_addr s2
        \\  %14 = fn_addr @main.f$l0
        \\  store %13, %14 : int
        \\  %15 = field_addr %13, 8 : rawptr
        \\  store %15, %9 : rawptr
        \\  %16 = slot_addr s2
        \\  copy %7 <- %16 : fn$unit
        \\  %17 = slot_addr s1
        \\  %18 = load %17 : int
        \\  %19 = field_addr %17, 8 : rawptr
        \\  %20 = load %19 : rawptr
        \\  call_indirect %18(%20)
        \\  %21 = slot_addr s0
        \\  %22 = load %21 : rawptr
        \\  %23 = load %22 : int
        \\  %24 = iconst 10
        \\  %25 = add %23, %24
        \\  %26 = slot_addr s0
        \\  %27 = load %26 : rawptr
        \\  store %27, %25 : int
        \\  %28 = slot_addr s0
        \\  %29 = load %28 : rawptr
        \\  %30 = load %29 : int
        \\  br b1(%30)
        \\b1(%0:int):
        \\  ret %0
        \\}
        \\
    );
    // The lambda binds the capture as an address slot holding the pointer from its env.
    const gpa = testing.allocator;
    const lam = try renderUnit(gpa,
        \\fn f() -> int {
        \\ n := 1
        \\ g := fn() { n = n + 1 }
        \\ g()
        \\ n = n + 10
        \\ return n
        \\}
        \\
    , "main.f$l0", .{});
    defer gpa.free(lam);
    try testing.expectEqualStrings(
        \\fn main.f$l0(s0) -> unit {
        \\  slots: s0:rawptr s1:rawptr
        \\b0:
        \\  %0 = slot_addr s0
        \\  %1 = load %0 : rawptr
        \\  %2 = field_addr %1, 0 : rawptr
        \\  %3 = load %2 : rawptr
        \\  %4 = slot_addr s1
        \\  store %4, %3 : rawptr
        \\  %5 = slot_addr s1
        \\  %6 = load %5 : rawptr
        \\  %7 = load %6 : int
        \\  %8 = iconst 1
        \\  %9 = add %7, %8
        \\  %10 = slot_addr s1
        \\  %11 = load %10 : rawptr
        \\  store %11, %9 : int
        \\  br b1
        \\b1:
        \\  ret
        \\}
        \\
    , lam);
}

test "a boxed for variable gets a fresh box per iteration and copies back before inc" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn f() -> int {
        \\ s := 0
        \\ for i in 0..3 {
        \\  g := fn() { i = i + 1 }
        \\  g()
        \\  s = s + i
        \\ }
        \\ return s
        \\}
        \\
    , "f", .{});
    defer gpa.free(ir);
    // s1 is the hidden counter the header tests; the body starts by allocating a new cell
    // for `i` (s2) and copying the counter into it.
    try testing.expect(std.mem.indexOf(u8, ir,
        \\  %7 = load %6 : int
        \\  %8 = icmp ge %7, %5
        \\  cond_br %8, b5, b3
        \\b3:
        \\  %9 = iconst 8
        \\  %10 = call @gc_alloc(%9)
        \\  %11 = slot_addr s2
        \\  store %11, %10 : rawptr
        \\  %12 = slot_addr s2
        \\  %13 = load %12 : rawptr
        \\  %14 = slot_addr s1
        \\  %15 = load %14 : int
        \\  store %13, %15 : int
        \\
    ) != null);
    // `inc` copies the cell back into the counter before stepping it, so an assignment to
    // `i` in the body still steers the loop.
    try testing.expect(std.mem.indexOf(u8, ir,
        \\b4:
        \\  %37 = slot_addr s2
        \\  %38 = load %37 : rawptr
        \\  %39 = load %38 : int
        \\  %40 = slot_addr s1
        \\  store %40, %39 : int
        \\  %41 = slot_addr s1
        \\  %42 = load %41 : int
        \\  %43 = iconst 1
        \\  %44 = add %42, %43
        \\
    ) != null);
    // One cell per iteration plus one env per iteration, both in the body.
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "call @gc_alloc("));
}

test "a read-only capture is not boxed" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn f() -> int {
        \\ n := 1
        \\ g := fn() -> int { n }
        \\ return g()
        \\}
        \\
    , "f", .{});
    defer gpa.free(ir);
    // One allocation, the env; `n` is a plain int slot whose value is copied in.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, ir, "call @gc_alloc("));
    try testing.expect(std.mem.indexOf(u8, ir, "slots: s0:int ") != null);
    try testing.expect(std.mem.indexOf(u8, ir,
        \\  %6 = field_addr %5, 0 : int
        \\  %7 = slot_addr s0
        \\  %8 = load %7 : int
        \\  store %6, %8 : int
        \\
    ) != null);
}

test "mut self receiver lowering is unchanged" {
    // A `mut self` method that also has a boxed local: `self` stays the int pointer slot
    // s0 (loaded as int), `n` the rawptr cell slot s2 (loaded as rawptr).
    try expectUnit(
        \\struct P { x: int, y: int }
        \\impl P {
        \\ fn step(mut self, d: int) -> int {
        \\  n := d
        \\  f := fn() { n = n + 1 }
        \\  f()
        \\  self.x = self.x + n
        \\  self = P { x: self.x, y: n }
        \\  self.y
        \\ }
        \\}
        \\fn main() -> int {
        \\ p := P { x: 1, y: 2 }
        \\ return p.step(3) + p.x
        \\}
        \\
    , "step",
        \\fn main.P.step(s0, s1) -> int {
        \\  slots: s0:int s1:int s2:rawptr s3:fn$unit s4:fn$unit s5:P
        \\b0:
        \\  %1 = iconst 8
        \\  %2 = call @gc_alloc(%1)
        \\  %3 = slot_addr s2
        \\  store %3, %2 : rawptr
        \\  %4 = slot_addr s1
        \\  %5 = load %4 : int
        \\  %6 = slot_addr s2
        \\  %7 = load %6 : rawptr
        \\  store %7, %5 : int
        \\  %8 = slot_addr s3
        \\  %9 = iconst 8
        \\  %10 = call @gc_alloc(%9)
        \\  %11 = field_addr %10, 0 : rawptr
        \\  %12 = slot_addr s2
        \\  %13 = load %12 : rawptr
        \\  store %11, %13 : rawptr
        \\  %14 = slot_addr s4
        \\  %15 = fn_addr @main.P.step$l0
        \\  store %14, %15 : int
        \\  %16 = field_addr %14, 8 : rawptr
        \\  store %16, %10 : rawptr
        \\  %17 = slot_addr s4
        \\  copy %8 <- %17 : fn$unit
        \\  %18 = slot_addr s3
        \\  %19 = load %18 : int
        \\  %20 = field_addr %18, 8 : rawptr
        \\  %21 = load %20 : rawptr
        \\  call_indirect %19(%21)
        \\  %22 = slot_addr s0
        \\  %23 = load %22 : int
        \\  %24 = field_addr %23, 0 : int
        \\  %25 = slot_addr s0
        \\  %26 = load %25 : int
        \\  %27 = field_addr %26, 0 : int
        \\  %28 = load %27 : int
        \\  %29 = slot_addr s2
        \\  %30 = load %29 : rawptr
        \\  %31 = load %30 : int
        \\  %32 = add %28, %31
        \\  store %24, %32 : int
        \\  %33 = slot_addr s0
        \\  %34 = load %33 : int
        \\  %35 = slot_addr s5
        \\  %36 = field_addr %35, 0 : int
        \\  %37 = slot_addr s0
        \\  %38 = load %37 : int
        \\  %39 = field_addr %38, 0 : int
        \\  %40 = load %39 : int
        \\  store %36, %40 : int
        \\  %41 = field_addr %35, 8 : int
        \\  %42 = slot_addr s2
        \\  %43 = load %42 : rawptr
        \\  %44 = load %43 : int
        \\  store %41, %44 : int
        \\  %45 = slot_addr s5
        \\  copy %34 <- %45 : P
        \\  %46 = slot_addr s0
        \\  %47 = load %46 : int
        \\  %48 = field_addr %47, 8 : int
        \\  %49 = load %48 : int
        \\  br b1(%49)
        \\b1(%0:int):
        \\  ret %0
        \\}
        \\
    );
}

test "every binding form boxes" {
    const gpa = testing.allocator;
    const Case = struct { src: []const u8, unit: []const u8, lambda: bool = false, cells: usize };
    // Each boxed binding reaches its cell with no lower note (an unconverted slot access
    // is one). `cells` counts the allocations in the lowered unit: boxes plus envs.
    const cases = [_]Case{
        // `:=`
        .{ .src = "fn f() -> int {\n n := 1\n g := fn() { n = 2 }\n g()\n n\n}\n", .unit = "f", .cells = 2 },
        // a fn param
        .{ .src = "fn f(n: int) -> int {\n g := fn() { n = n + 1 }\n g()\n n\n}\n", .unit = "f", .cells = 2 },
        // a struct param, stored to through a field
        .{ .src = "struct P { x: int }\nfn f(p: P) -> int {\n g := fn() { p.x = 3 }\n g()\n p.x\n}\n", .unit = "f", .cells = 2 },
        // a lambda param, boxed in the lambda's own unit
        .{ .src = "fn f() -> int {\n h := fn(n: int) -> int {\n  g := fn() { n = n + 1 }\n  g()\n  n\n }\n h(1)\n}\n", .unit = "main.f$l0", .lambda = true, .cells = 2 },
        // a range `for` variable
        .{ .src = "fn f() -> int {\n s := 0\n for i in 0..3 {\n  g := fn() { i = i + 1 }\n  g()\n  s = s + i\n }\n s\n}\n", .unit = "f", .cells = 2 },
        // a match binding
        .{ .src = "fn f(o: Option[int]) -> int {\n match o {\n  .some(v) -> {\n   g := fn() { v = v + 1 }\n   g()\n   v\n  },\n  .none -> 0,\n }\n}\n", .unit = "f", .cells = 2 },
        // an or-pattern binding: each alternative points the one address slot at a new cell
        .{ .src = "enum E { A(int), B(int) }\nfn f(e: E) -> int {\n match e {\n  .A(k) | .B(k) -> {\n   g := fn() { k = k + 1 }\n   g()\n   k\n  },\n }\n}\n", .unit = "f", .cells = 3 },
        // a capture a lambda passes through to a nested one: the outer env carries the pointer
        .{ .src = "fn f() -> int {\n n := 1\n h := fn() -> fn() -> int { fn() { n = n + 1\n n } }\n h()()\n}\n", .unit = "main.f$l0", .lambda = true, .cells = 1 },
        // a `mut self` call on the captured local
        .{ .src = "struct C { n: int }\nimpl C { fn bump(mut self) { self.n = self.n + 1 } }\nfn f() -> int {\n c := C { n: 0 }\n g := fn() { c.bump() }\n g()\n c.bump()\n c.n\n}\n", .unit = "f", .cells = 2 },
    };
    for (cases) |case| {
        const src = try std.mem.concat(gpa, u8, &.{ case.src, "fn main() -> int { return 0 }\n" });
        defer gpa.free(src);
        const ir = if (case.lambda) try renderUnit(gpa, src, case.unit, .{}) else try renderUnit(gpa, src, case.unit, .{});
        defer gpa.free(ir);
        try testing.expectEqual(case.cells, std.mem.count(u8, ir, "call @gc_alloc("));
        try testing.expect(std.mem.indexOf(u8, ir, ":rawptr") != null);
    }
}

test "a fn value lowers to fn_addr of its thunk and a null env" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn g(x: int) -> int { return x }
        \\fn f() -> int {
        \\ h := g
        \\ return h(1)
        \\}
        \\
    , "f", .{});
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "fn_addr @main.g$fnval\n") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "iconst 0\n") != null);
    try testing.expect(std.mem.indexOf(u8, ir, ", 8 : rawptr\n") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "call @") == null);
}

test "a value call passes env as argument 0" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn apply(f: fn(int, bool) -> int, x: int) -> int { return f(x, true) }
        \\
    , "apply", .{});
    defer gpa.free(ir);
    // The code is the pair's word at +0 and the env its rawptr word at +8, both loaded
    // before the args are evaluated.
    try testing.expect(std.mem.indexOf(u8, ir,
        \\  %1 = slot_addr s0
        \\  %2 = load %1 : int
        \\  %3 = field_addr %1, 8 : rawptr
        \\  %4 = load %3 : rawptr
        \\
    ) != null);
    try testing.expect(std.mem.indexOf(u8, ir, "%8 = call_indirect %2(%4, %6, %7)\n") != null);
}

test "fn identity == compares both words, no Eq call" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn same(a: fn(int) -> int, b: fn(int) -> int) -> bool { return a == b }
        \\fn diff(a: fn(int) -> int, b: fn(int) -> int) -> bool { return a != b }
        \\
    , "same", .{});
    defer gpa.free(ir);
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, ir, "bxor"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, ir, "bor"));
    try testing.expect(std.mem.indexOf(u8, ir, "icmp eq") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "call") == null);
    const ne = try renderUnit(gpa,
        \\fn diff(a: fn(int) -> int, b: fn(int) -> int) -> bool { return a != b }
        \\
    , "diff", .{});
    defer gpa.free(ne);
    try testing.expect(std.mem.indexOf(u8, ne, "icmp ne") != null);
    try testing.expect(std.mem.indexOf(u8, ne, "call") == null);
}

test "struct == lowers to a call to the Eq witness (no inline icmp)" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\struct P { x: int }
        \\impl P has Eq { fn eq(self, o: P) -> bool { self.x == o.x } }
        \\fn cmp(p: P, q: P) -> bool { p == q }
        \\
    , "cmp", .{});
    defer gpa.free(ir);
    // Dispatches to the witness call; the struct compare itself has no inline icmp.
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "eq$Eq(") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp") == null);
}

test "str == lowers to a heap-free len-compare + per-byte load_byte loop" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn cmp(a: str, b: str) -> bool { a == b }
        \\
    , "cmp", .{});
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "load_byte") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp ge") != null); // the i >= len loop guard
    try testing.expect(std.mem.indexOf(u8, ir, "call") == null); // no witness fn, no heap
}

test "unit == folds to bconst true" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn nothing() { return }
        \\fn cmp() -> bool { nothing() == nothing() }
        \\
    , "cmp", .{});
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "bconst true") != null);
}

test "!= wraps the Eq result in bnot" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\struct P { x: int }
        \\impl P has Eq { fn eq(self, o: P) -> bool { self.x == o.x } }
        \\fn cmp(p: P, q: P) -> bool { p != q }
        \\
    , "cmp", .{});
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") != null);
}

test "int == stays a single inline icmp eq (regression pin: bytes unchanged)" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn cmp(a: int, b: int) -> bool { a == b }
        \\
    , "cmp", .{});
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp eq") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "call") == null);
    try testing.expect(std.mem.indexOf(u8, ir, "load_byte") == null);
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") == null);
}

test "int != stays a single inline icmp ne (regression pin: no bnot)" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\fn cmp(a: int, b: int) -> bool { a != b }
        \\
    , "cmp", .{});
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
    const ir = try renderUnit(gpa, ord_impl_src ++ "fn use_lt(p: P, q: P) -> bool { p < q }\n", "use_lt", .{});
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
    const ir = try renderUnit(gpa, ord_impl_src ++ "fn use_ge(p: P, q: P) -> bool { p >= q }\n", "use_ge", .{});
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
        const ir = try renderUnit(gpa, c.src, "f", .{});
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
    const ir = try renderUnit(gpa, "fn use_lt(a: str, b: str) -> bool { a < b }\n", "use_lt", .{});
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "load_byte") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "call") == null); // no witness fn, no heap
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") == null); // str, not an enum witness
    try testing.expect(std.mem.indexOf(u8, ir, "icmp eq") != null); // discriminant == ord_lt(0)
}

test "bool `<` stays an inline icmp (false<true), no call/load_byte" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa, "fn f(a: bool, b: bool) -> bool { a < b }\n", "f", .{});
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp lt") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "call") == null);
    try testing.expect(std.mem.indexOf(u8, ir, "load_byte") == null);
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") == null);
}

test "Ord refines Eq — `==` on an Ord-only struct lowers via a cmp call + `icmp eq`" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa, ord_impl_src ++ "fn use_eq(p: P, q: P) -> bool { p == q }\n", "use_eq", .{});
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null); // the cmp witness (no `eq` witness)
    try testing.expect(std.mem.indexOf(u8, ir, "cmp$Ord(") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "get_tag") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "icmp eq") != null); // discriminant == ord_eq(1)
    try testing.expect(std.mem.indexOf(u8, ir, "bnot") == null);
}

test "Ord refines Eq — `!=` on an Ord-only struct lowers via a cmp call + `icmp ne` (no bnot)" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa, ord_impl_src ++ "fn use_ne(p: P, q: P) -> bool { p != q }\n", "use_ne", .{});
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
    const ir = try renderUnit(gpa,
        \\struct P { x: int }
        \\fn use_lt(p: P, q: P) -> bool { p < q }
        \\
    , "use_lt", .{});
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
    const ir = try renderUnit(gpa,
        \\struct P { x: int }
        \\fn use_both(p: P, q: P) -> int {
        \\ if p < q { return 1 }
        \\ if p == q { return 2 }
        \\ return 0
        \\}
        \\
    , "use_both", .{});
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
    const ir = try renderUnit(gpa, add_impl_src ++ "fn use_add(p: V2, q: V2) -> V2 { p + q }\n", "use_add", .{});
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
        const ir = try renderUnit(gpa, src, "f", .{});
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
        const ir = try renderUnit(gpa, c.src, "f", .{});
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
    const ir = try renderUnit(gpa, from_impl_src, "outer", .{});
    defer gpa.free(ir);
    // The err residual materializes the operand error, calls `BigErr.from(smallErr)`, and
    // copies the widened error into the return's err payload.
    try testing.expect(std.mem.indexOf(u8, ir, "call @") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "from") != null);
}

test "a `?` on a SAME-error-type Result stays a plain copy (no From witness call)" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa, from_impl_src, "same", .{});
    defer gpa.free(ir);
    // The identity path: the err payload is copied unchanged across the two layouts —
    // no `from` witness is resolved or called.
    try testing.expect(std.mem.indexOf(u8, ir, "from") == null);
}

test "`?` on a unit-payload Result still emits the residual early-return (no codegen diagnostic)" {
    const gpa = testing.allocator;
    // renderUnit fails on any lowering diagnostic; before the fix lowerTryValue had no `.unit`
    // arm and dropped a note here, so a checker-accepted program aborted at codegen.
    const ir = try renderUnit(gpa,
        \\fn inner() -> Result[(), int] { return Result[(), int].err(7) }
        \\fn outer() -> Result[int, int] {
        \\ inner()?
        \\ return Result.ok(0)
        \\}
        \\fn main() -> int { return 0 }
        \\
    , "outer", .{});
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
        .nodes = .{ .types = tc.node_types[0], .callees = tc.callees[0] },
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = &names,
        .methods = tc.methods,
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
        .closure = .{ .adapters = tc.adapters, .lambda_units = tc.lambda_units },
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
        .nodes = .{ .types = tc.node_types[0], .callees = tc.callees[0] },
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = &names,
        .methods = tc.methods,
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
        .closure = .{ .adapters = tc.adapters, .lambda_units = tc.lambda_units },
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
        .nodes = .{ .types = tc.node_types[0], .callees = tc.callees[0] },
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = &names,
        .methods = tc.methods,
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
        .closure = .{ .adapters = tc.adapters, .lambda_units = tc.lambda_units },
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

test "lower-aggregates: value-if returning a struct produces both arms into one temp slot" {
    try expectLowered(
        "struct V { a: int, b: int }\n" ++
            "fn f(c: int) -> V {\n if c == 1 { V { a: 10, b: 20 } } else { V { a: 1, b: 1 } }\n}\n",
        "f",
        "fn f(s0) -> V {\n" ++
            "  slots: s0:int s1:V\n" ++
            "b0:\n" ++
            "  %1 = slot_addr s1\n" ++
            "  %2 = slot_addr s0\n" ++
            "  %3 = load %2 : int\n" ++
            "  %4 = iconst 1\n" ++
            "  %5 = icmp eq %3, %4\n" ++
            "  cond_br %5, b2, b3\n" ++
            "b1(%0:V):\n" ++
            "  ret %0\n" ++
            "b2:\n" ++
            "  %6 = field_addr %1, 0 : int\n" ++
            "  %7 = iconst 10\n" ++
            "  store %6, %7 : int\n" ++
            "  %8 = field_addr %1, 8 : int\n" ++
            "  %9 = iconst 20\n" ++
            "  store %8, %9 : int\n" ++
            "  br b4\n" ++
            "b3:\n" ++
            "  %10 = field_addr %1, 0 : int\n" ++
            "  %11 = iconst 1\n" ++
            "  store %10, %11 : int\n" ++
            "  %12 = field_addr %1, 8 : int\n" ++
            "  %13 = iconst 1\n" ++
            "  store %12, %13 : int\n" ++
            "  br b4\n" ++
            "b4:\n" ++
            "  br b1(s1)\n" ++
            "}\n",
    );
}

test "lower-aggregates: value-loop break of a struct produces into a temp slot" {
    try expectLowered(
        "struct V { a: int, b: int }\nfn f() -> V {\n loop { break V { a: 1, b: 2 } }\n}\n",
        "f",
        "fn f() -> V {\n" ++
            "  slots: s0:V\n" ++
            "b0:\n" ++
            "  %1 = slot_addr s0\n" ++
            "  br b2\n" ++
            "b1(%0:V):\n" ++
            "  ret %0\n" ++
            "b2:\n" ++
            "  %2 = field_addr %1, 0 : int\n" ++
            "  %3 = iconst 1\n" ++
            "  store %2, %3 : int\n" ++
            "  %4 = field_addr %1, 8 : int\n" ++
            "  %5 = iconst 2\n" ++
            "  store %4, %5 : int\n" ++
            "  br b3\n" ++
            "b3:\n" ++
            "  br b1(s0)\n" ++
            "}\n",
    );
}

test "lower-aggregates: labeled-block break of a struct produces into a temp slot" {
    try expectLowered(
        "struct V { a: int, b: int }\nfn f() -> V {\n @blk { break @blk V { a: 1, b: 2 } }\n}\n",
        "f",
        "fn f() -> V {\n" ++
            "  slots: s0:V\n" ++
            "b0:\n" ++
            "  %1 = slot_addr s0\n" ++
            "  %2 = field_addr %1, 0 : int\n" ++
            "  %3 = iconst 1\n" ++
            "  store %2, %3 : int\n" ++
            "  %4 = field_addr %1, 8 : int\n" ++
            "  %5 = iconst 2\n" ++
            "  store %4, %5 : int\n" ++
            "  br b2\n" ++
            "b1(%0:V):\n" ++
            "  ret %0\n" ++
            "b2:\n" ++
            "  br b1(s0)\n" ++
            "}\n",
    );
}

test "lower-aggregates: a value call on an if reads the fn pair from its temp slot" {
    // A join block param has no address to load the pair's code and env from; the temp
    // slot both arms copy into does.
    try expectUnit(
        \\fn inc(x: int) -> int { x + 1 }
        \\fn dec(x: int) -> int { x - 1 }
        \\fn f(c: bool) -> int { (if c { inc } else { dec })(41) }
        \\
    , "f",
        \\fn main.f(s0) -> int {
        \\  slots: s0:bool s1:fn$int$int s2:fn$int$int s3:fn$int$int
        \\b0:
        \\  %1 = slot_addr s1
        \\  %2 = slot_addr s0
        \\  %3 = load %2 : bool
        \\  cond_br %3, b2, b3
        \\b1(%0:int):
        \\  ret %0
        \\b2:
        \\  %4 = slot_addr s2
        \\  %5 = fn_addr @main.inc$fnval
        \\  store %4, %5 : int
        \\  %6 = field_addr %4, 8 : rawptr
        \\  %7 = iconst 0
        \\  store %6, %7 : rawptr
        \\  %8 = slot_addr s2
        \\  copy %1 <- %8 : fn$int$int
        \\  br b4
        \\b3:
        \\  %9 = slot_addr s3
        \\  %10 = fn_addr @main.dec$fnval
        \\  store %9, %10 : int
        \\  %11 = field_addr %9, 8 : rawptr
        \\  %12 = iconst 0
        \\  store %11, %12 : rawptr
        \\  %13 = slot_addr s3
        \\  copy %1 <- %13 : fn$int$int
        \\  br b4
        \\b4:
        \\  %14 = slot_addr s1
        \\  %15 = load %14 : int
        \\  %16 = field_addr %14, 8 : rawptr
        \\  %17 = load %16 : rawptr
        \\  %18 = iconst 41
        \\  %19 = call_indirect %15(%17, %18)
        \\  br b1(%19)
        \\}
        \\
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

test "for passes each receiver as its method declares it (#29)" {
    const gpa = testing.allocator;
    const src =
        \\protocol Iterator[Item] { fn next(mut self) -> Option[Item] }
        \\struct Up[T] { i: int }
        \\impl Up[T] has Iterator[int] {
        \\ fn next(mut self) -> Option[int] {
        \\  self.i = self.i - 1
        \\  if self.i < 0 { return Option[int].none }
        \\  return Option.some(self.i)
        \\ }
        \\}
        \\impl Up[T] { fn iter(mut self) -> Up[T] { self } }
        \\struct Down[T] { i: int }
        \\impl Down[T] has Iterator[int] {
        \\ fn next(mut self) -> Option[int] {
        \\  self.i = self.i - 1
        \\  if self.i < 0 { return Option[int].none }
        \\  return Option.some(self.i)
        \\ }
        \\}
        \\impl Down[T] { fn iter(self) -> Down[T] { self } }
        \\fn main() -> int {
        \\ u := Up[int] { i: 3 }
        \\ d := Down[int] { i: 3 }
        \\ t := 0
        \\ for x in u { t = t + x }
        \\ for x in d { t = t + x }
        \\ return t
        \\}
        \\
    ;
    const ir = try renderUnit(gpa, src, "main", .{});
    defer gpa.free(ir);
    // `iter(mut self)` gets the local's address, `iter(self)` the local's slot.
    try testing.expect(std.mem.indexOf(u8, ir, "  %9 = slot_addr s0\n  call @main.Up.iter$int(%9 -> s3)\n") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "  call @main.Down.iter$int(s1 -> s6)\n") != null);
    // `next(mut self)` gets the iterator slot's address each time round.
    try testing.expect(std.mem.indexOf(u8, ir, "  %10 = slot_addr s3\n  call @main.Up.next$Iterator$int$int(%10 -> s4)\n") != null);
    try testing.expect(std.mem.indexOf(u8, ir, "  %25 = slot_addr s6\n  call @main.Down.next$Iterator$int$int(%25 -> s7)\n") != null);
}

test "a mut-self method body loads self through the pointer slot (s0:int)" {
    try expectUnit(
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

test "a mut param loads and stores through its int slot" {
    try expectUnit(
        "fn bump(mut n: int) { n = n + 1 }\n" ++
            "fn main() -> int { c := 1\n bump(c)\n return c }\n",
        "bump",
        "fn main.bump(s0) -> unit {\n" ++
            "  slots: s0:int\n" ++
            "b0:\n" ++
            "  %0 = slot_addr s0\n" ++
            "  %1 = load %0 : int\n" ++
            "  %2 = load %1 : int\n" ++
            "  %3 = iconst 1\n" ++
            "  %4 = add %2, %3\n" ++
            "  %5 = slot_addr s0\n" ++
            "  %6 = load %5 : int\n" ++
            "  store %6, %4 : int\n" ++
            "  br b1\n" ++
            "b1:\n" ++
            "  ret\n" ++
            "}\n",
    );
}

test "a mut argument passes the place address" {
    try expectUnit(
        "struct P { x: int, y: int }\n" ++
            "fn bump(mut n: int) { n = n + 1 }\n" ++
            "fn main() -> int { c := 1\n p := P{ x: 2, y: 3 }\n bump(c)\n bump(p.y)\n return c + p.y }\n",
        "main",
        "fn main() -> int {\n" ++
            "  slots: s0:int s1:P\n" ++
            "b0:\n" ++
            "  %1 = iconst 1\n" ++
            "  %2 = slot_addr s0\n" ++
            "  store %2, %1 : int\n" ++
            "  %3 = slot_addr s1\n" ++
            "  %4 = field_addr %3, 0 : int\n" ++
            "  %5 = iconst 2\n" ++
            "  store %4, %5 : int\n" ++
            "  %6 = field_addr %3, 8 : int\n" ++
            "  %7 = iconst 3\n" ++
            "  store %6, %7 : int\n" ++
            "  %8 = slot_addr s0\n" ++
            "  call @main.bump(%8)\n" ++
            "  %9 = slot_addr s1\n" ++
            "  %10 = field_addr %9, 8 : int\n" ++
            "  call @main.bump(%10)\n" ++
            "  %11 = slot_addr s0\n" ++
            "  %12 = load %11 : int\n" ++
            "  %13 = slot_addr s1\n" ++
            "  %14 = field_addr %13, 8 : int\n" ++
            "  %15 = load %14 : int\n" ++
            "  %16 = add %12, %15\n" ++
            "  br b1(%16)\n" ++
            "b1(%0:int):\n" ++
            "  ret %0\n" ++
            "}\n",
    );
}

test "a forwarded mut param passes the address it holds" {
    try expectUnit(
        "fn bump(mut n: int) { n = n + 1 }\n" ++
            "fn twice(mut n: int) { bump(n)\n bump(n) }\n" ++
            "fn main() -> int { c := 1\n twice(c)\n return c }\n",
        "twice",
        "fn main.twice(s0) -> unit {\n" ++
            "  slots: s0:int\n" ++
            "b0:\n" ++
            "  %0 = slot_addr s0\n" ++
            "  %1 = load %0 : int\n" ++
            "  call @main.bump(%1)\n" ++
            "  %2 = slot_addr s0\n" ++
            "  %3 = load %2 : int\n" ++
            "  call @main.bump(%3)\n" ++
            "  br b1\n" ++
            "b1:\n" ++
            "  ret\n" ++
            "}\n",
    );
}

test "a whole assignment to a mut aggregate param builds the value aside, then copies it to the caller's storage" {
    try expectUnit(
        "struct P { x: int, y: int }\n" ++
            "fn set(mut p: P) { p = P{ x: 5, y: 6 } }\n" ++
            "fn main() -> int { q := P{ x: 1, y: 2 }\n set(q)\n return q.x }\n",
        "set",
        "fn main.set(s0) -> unit {\n" ++
            "  slots: s0:int s1:P\n" ++
            "b0:\n" ++
            "  %0 = slot_addr s0\n" ++
            "  %1 = load %0 : int\n" ++
            "  %2 = slot_addr s1\n" ++
            "  %3 = field_addr %2, 0 : int\n" ++
            "  %4 = iconst 5\n" ++
            "  store %3, %4 : int\n" ++
            "  %5 = field_addr %2, 8 : int\n" ++
            "  %6 = iconst 6\n" ++
            "  store %5, %6 : int\n" ++
            "  %7 = slot_addr s1\n" ++
            "  copy %1 <- %7 : P\n" ++
            "  br b1\n" ++
            "b1:\n" ++
            "  ret\n" ++
            "}\n",
    );
}

test "a mut self method with no other mut param still produces a whole assignment in place" {
    try expectUnit(
        "struct P { x: int, y: int }\n" ++
            "impl P { fn set(mut self) { self = P{ x: 5, y: 6 } } }\n" ++
            "fn main() -> int { q := P{ x: 1, y: 2 }\n q.set()\n return q.x }\n",
        "set",
        "fn main.P.set(s0) -> unit {\n" ++
            "  slots: s0:int\n" ++
            "b0:\n" ++
            "  %0 = slot_addr s0\n" ++
            "  %1 = load %0 : int\n" ++
            "  %2 = field_addr %1, 0 : int\n" ++
            "  %3 = iconst 5\n" ++
            "  store %2, %3 : int\n" ++
            "  %4 = field_addr %1, 8 : int\n" ++
            "  %5 = iconst 6\n" ++
            "  store %4, %5 : int\n" ++
            "  br b1\n" ++
            "b1:\n" ++
            "  ret\n" ++
            "}\n",
    );
}

/// How many `copy` instructions of the type spelled `ty` the rendered unit `ir` has.
fn copiesOf(ir: []const u8, ty: []const u8) usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, ir, '\n');
    while (it.next()) |line| {
        const t = std.mem.trim(u8, line, " ");
        if (!std.mem.startsWith(u8, t, "copy ") or !std.mem.endsWith(u8, t, ty)) continue;
        if (std.mem.endsWith(u8, t[0 .. t.len - ty.len], " : ")) n += 1;
    }
    return n;
}

test "an aggregate store whose value reads its destination builds the value aside, then copies it (#26)" {
    const gpa = testing.allocator;
    const src =
        \\struct P { x: int, y: int }
        \\struct O { p: P }
        \\impl P {
        \\ fn swap(mut self) { self = P { x: self.y, y: self.x } }
        \\ fn reset(mut self) { self = P { x: 0, y: 0 } }
        \\}
        \\fn local() -> int {
        \\ p := P { x: 1, y: 2 }
        \\ p = P { x: p.y, y: p.x }
        \\ p.x
        \\}
        \\fn field() -> int {
        \\ o := O { p: P { x: 1, y: 2 } }
        \\ o.p = P { x: o.p.y, y: o.p.x }
        \\ o.p.x
        \\}
        \\fn ifv(c: bool) -> int {
        \\ p := P { x: 1, y: 2 }
        \\ p = if c { P { x: p.y, y: p.x } } else { P { x: 0, y: 0 } }
        \\ p.x
        \\}
        \\fn other() -> int {
        \\ p := P { x: 1, y: 2 }
        \\ q := P { x: 3, y: 4 }
        \\ p = P { x: q.y, y: q.x }
        \\ p.x
        \\}
        \\fn main() -> int {
        \\ p := P { x: 1, y: 2 }
        \\ p.swap()
        \\ p.reset()
        \\ return local() + field() + ifv(true) + other()
        \\}
        \\
    ;
    const Case = struct { unit: []const u8, ty: []const u8, copies: usize };
    for ([_]Case{
        .{ .unit = "local", .ty = "P", .copies = 1 },
        .{ .unit = "swap", .ty = "P", .copies = 1 },
        .{ .unit = "field", .ty = "P", .copies = 1 },
        .{ .unit = "ifv", .ty = "P", .copies = 1 },
        // A value that never reads its destination is still produced in place.
        .{ .unit = "reset", .ty = "P", .copies = 0 },
        .{ .unit = "other", .ty = "P", .copies = 0 },
    }) |c| {
        const ir = try renderUnit(gpa, src, c.unit, .{});
        defer gpa.free(ir);
        testing.expectEqual(c.copies, copiesOf(ir, c.ty)) catch |e| {
            std.debug.print("unit {s}:\n{s}\n", .{ c.unit, ir });
            return e;
        };
    }
}

test "a store into a place a closure or a caller may hold stages a value that runs code (#26)" {
    const gpa = testing.allocator;
    const src =
        \\struct P { x: int, y: int }
        \\fn id(k: int) -> int { k }
        \\impl P {
        \\ fn set(mut self, g: fn() -> int) { self = P { x: 5, y: g() } }
        \\ fn put(mut self, k: int) { self = P { x: 5, y: k + 1 } }
        \\}
        \\fn boxed() -> int {
        \\ p := P { x: 1, y: 2 }
        \\ g := fn() { p.x }
        \\ p = P { x: 5, y: g() }
        \\ p = P { x: 6, y: 7 }
        \\ p.y
        \\}
        \\fn plain() -> int {
        \\ p := P { x: 1, y: 2 }
        \\ p = P { x: 5, y: id(3) }
        \\ p.y
        \\}
        \\fn main() -> int {
        \\ p := P { x: 1, y: 2 }
        \\ p.set(fn() { 3 })
        \\ p.put(4)
        \\ return boxed() + plain()
        \\}
        \\
    ;
    const Case = struct { unit: []const u8, frame: bool, copies: usize };
    for ([_]Case{
        // Only the store whose value calls code is staged.
        .{ .unit = "boxed", .frame = false, .copies = 1 },
        .{ .unit = "boxed", .frame = true, .copies = 1 },
        .{ .unit = "set", .frame = false, .copies = 1 },
        .{ .unit = "put", .frame = false, .copies = 0 },
        // Nothing but the local itself can reach an unshared local.
        .{ .unit = "plain", .frame = false, .copies = 0 },
    }) |c| {
        const ir = try renderUnit(gpa, src, c.unit, .{ .frame = c.frame });
        defer gpa.free(ir);
        testing.expectEqual(c.copies, copiesOf(ir, "P")) catch |e| {
            std.debug.print("unit {s} (frame {}):\n{s}\n", .{ c.unit, c.frame, ir });
            return e;
        };
    }
}

test "a by-value read of a local some call passes as a mut argument copies it on the spot" {
    try expectUnit(
        "struct Q { a: int, b: int }\n" ++
            "fn touch(mut p: Q) -> int { p.a = 50\n 0 }\n" ++
            "fn hq(v: Q, k: int) -> int { v.a + k }\n" ++
            "fn main() -> int { q := Q{ a: 1, b: 2 }\n return hq(q, touch(q)) }\n",
        "main",
        "fn main() -> int {\n" ++
            "  slots: s0:Q s1:Q\n" ++
            "b0:\n" ++
            "  %1 = slot_addr s0\n" ++
            "  %2 = field_addr %1, 0 : int\n" ++
            "  %3 = iconst 1\n" ++
            "  store %2, %3 : int\n" ++
            "  %4 = field_addr %1, 8 : int\n" ++
            "  %5 = iconst 2\n" ++
            "  store %4, %5 : int\n" ++
            "  %6 = slot_addr s1\n" ++
            "  %7 = slot_addr s0\n" ++
            "  copy %6 <- %7 : Q\n" ++
            "  %8 = slot_addr s0\n" ++
            "  %9 = call @main.touch(%8)\n" ++
            "  %10 = call @main.hq(s1, %9)\n" ++
            "  br b1(%10)\n" ++
            "b1(%0:int):\n" ++
            "  ret %0\n" ++
            "}\n",
    );
}

test "a by-value read of a local some call passes as a mut self receiver copies it on the spot (#27)" {
    try expectUnit(
        "struct Q { a: int, b: int }\n" ++
            "impl Q { fn touch(mut self) -> int { self.a = 50\n 0 } }\n" ++
            "fn hq(v: Q, k: int) -> int { v.a + k }\n" ++
            "fn main() -> int { q := Q{ a: 1, b: 2 }\n return hq(q, q.touch()) }\n",
        "main",
        "fn main() -> int {\n" ++
            "  slots: s0:Q s1:Q\n" ++
            "b0:\n" ++
            "  %1 = slot_addr s0\n" ++
            "  %2 = field_addr %1, 0 : int\n" ++
            "  %3 = iconst 1\n" ++
            "  store %2, %3 : int\n" ++
            "  %4 = field_addr %1, 8 : int\n" ++
            "  %5 = iconst 2\n" ++
            "  store %4, %5 : int\n" ++
            "  %6 = slot_addr s1\n" ++
            "  %7 = slot_addr s0\n" ++
            "  copy %6 <- %7 : Q\n" ++
            "  %8 = slot_addr s0\n" ++
            "  %9 = call @main.Q.touch(%8)\n" ++
            "  %10 = call @main.hq(s1, %9)\n" ++
            "  br b1(%10)\n" ++
            "b1(%0:int):\n" ++
            "  ret %0\n" ++
            "}\n",
    );
}

test "the caller passes the receiver place ADDRESS as arg 0 (a scalar value)" {
    try expectUnit(
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
    try expectUnit(
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

/// `fn main` of `src` lowered with the Callee slot of its `member` call replaced by
/// `slot`, keeping the IR text and every lower diagnostic message.
const OverriddenLowering = struct {
    ir: []u8,
    notes: [][]const u8,

    fn deinit(self: OverriddenLowering, gpa: std.mem.Allocator) void {
        gpa.free(self.ir);
        gpa.free(self.notes);
    }
};

fn lowerMainWithCallee(gpa: std.mem.Allocator, src: []const u8, member: []const u8, slot: Callee) !OverriddenLowering {
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
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    const names = try gpa.alloc(Link.SymName, rr.fns.len);
    defer gpa.free(names);
    var main_gid: ?usize = null;
    for (rr.fns, 0..) |gf, i| {
        names[i] = .{ .kind = if (gf.decl_node == Ast.none) .builtin else .user_fn, .name = gf.name };
        if (gf.decl_node != Ast.none and std.mem.eql(u8, gf.name, "main")) main_gid = i;
    }
    const gid = main_gid orelse return error.TestExpectedMain;

    const cs = try gpa.dupe(Callee, tc.callees[0]);
    defer gpa.free(cs);
    const t: Ast.Tree = .{ .nodes = tree.nodes, .extra = tree.extra };
    const fa = for (tree.nodes) |n| {
        if (n.tag != .call) continue;
        const f = Callee.memberNode(t, n) orelse continue;
        if (std.mem.eql(u8, tokens[tree.nodes[f.int()].main_token].text(src), member)) break f;
    } else return error.TestExpectedMethodCall;
    cs[fa.int()] = slot;

    const in = Inputs{
        .tree = t,
        .tokens = tokens,
        .source = src,
        .resolutions = rr.resolutions[0],
        .nodes = .{ .types = tc.node_types[0], .callees = cs },
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = names,
        .sig = tc.sigs[gid],
        .instances = tc.instances,
        .sigs = tc.sigs,
        .methods = tc.methods,
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
        .closure = .{ .adapters = tc.adapters, .lambda_units = tc.lambda_units },
    };
    var diags: std.ArrayList(Diagnostic) = .empty;
    defer diags.deinit(gpa);
    var func = try lowerFn(gpa, in, rr.fns[gid].decl_node, names[gid], true, &diags);
    defer func.deinit(gpa);

    var buf: [8192]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try Ir.render(&w, &func, in.layouts, in.enum_layouts);
    const ir = try gpa.dupe(u8, w.buffered());
    errdefer gpa.free(ir);
    const notes = try gpa.alloc([]const u8, diags.items.len);
    for (diags.items, notes) |d, *msg| msg.* = d.message;
    return .{ .ir = ir, .notes = notes };
}

const acc_plus_src =
    \\struct Acc { n: int }
    \\impl Acc { fn plus(self, a: int) -> int { self.n + a } }
    \\fn main() -> int {
    \\  acc := Acc { n: 1 }
    \\  return acc.plus(1)
    \\}
    \\
;

test "lowerCall dispatches through the Callee table" {
    const gpa = testing.allocator;
    const Lexer = @import("lex.zig");
    const Parser = @import("parse.zig");
    const tokens = try Lexer.tokenize(gpa, acc_plus_src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, acc_plus_src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    var fe = try frontEnd(gpa, tokens, .{ .nodes = tree.nodes, .extra = tree.extra }, acc_plus_src);
    defer fe.deinit(gpa);
    const picked = for (tree.nodes) |n| {
        if (n.tag != .call) continue;
        const fa = Callee.memberNode(.{ .nodes = tree.nodes, .extra = tree.extra }, n) orelse continue;
        break fe.typecheck.callees[0][fa.int()];
    } else return error.TestExpectedMethodCall;
    try testing.expect(picked.view() == .method);

    const dispatched = try lowerMainWithCallee(gpa, acc_plus_src, "plus", picked);
    defer dispatched.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), dispatched.notes.len);
    try testing.expect(std.mem.indexOf(u8, dispatched.ir, "call @main.Acc.plus(") != null);

    const blanked = try lowerMainWithCallee(gpa, acc_plus_src, "plus", .{});
    defer blanked.deinit(gpa);
    try testing.expect(std.mem.indexOf(u8, blanked.ir, "call @main.Acc.plus(") == null);
    for (blanked.notes) |msg| {
        if (std.mem.eql(u8, msg, "call target unsupported in lower")) break;
    } else return error.TestExpectedUnsupportedNote;
}

test "a protocol callee reaching lower is a note, not UB" {
    const gpa = testing.allocator;
    const lowered = try lowerMainWithCallee(gpa, acc_plus_src, "plus", Callee.protocolMethod(0, 0, 0));
    defer lowered.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), lowered.notes.len);
    try testing.expectEqualStrings("protocol callee in lower", lowered.notes[0]);
}

test "a local fn's prologue materializes its self value" {
    const gpa = testing.allocator;
    const src =
        \\fn f() -> int {
        \\ fn down(n: int) -> int { if n <= 0 { 0 } else { down(n - 1) } }
        \\ return down(3)
        \\}
        \\
    ;
    const unit_ir = try renderUnit(gpa, src, "main.f$l0", .{});
    defer gpa.free(unit_ir);
    // `down` is s2: the pair of the unit's own code and the env it was called with, so the
    // recursive call is an ordinary value call.
    try testing.expectEqualStrings(
        \\fn main.f$l0(s0, s1) -> int {
        \\  slots: s0:rawptr s1:int s2:fn$int$int
        \\b0:
        \\  %1 = slot_addr s0
        \\  %2 = load %1 : rawptr
        \\  %3 = slot_addr s2
        \\  %4 = fn_addr @main.f$l0
        \\  store %3, %4 : int
        \\  %5 = field_addr %3, 8 : rawptr
        \\  store %5, %2 : rawptr
        \\  %7 = slot_addr s1
        \\  %8 = load %7 : int
        \\  %9 = iconst 0
        \\  %10 = icmp le %8, %9
        \\  cond_br %10, b2, b3
        \\b1(%0:int):
        \\  ret %0
        \\b2:
        \\  %11 = iconst 0
        \\  br b4(%11)
        \\b3:
        \\  %12 = slot_addr s2
        \\  %13 = load %12 : int
        \\  %14 = field_addr %12, 8 : rawptr
        \\  %15 = load %14 : rawptr
        \\  %16 = slot_addr s1
        \\  %17 = load %16 : int
        \\  %18 = iconst 1
        \\  %19 = sub %17, %18
        \\  %20 = call_indirect %13(%15, %19)
        \\  br b4(%20)
        \\b4(%6:int):
        \\  br b1(%6)
        \\}
        \\
    , unit_ir);
    // The owner makes the closure straight into the name's home slot, s0.
    const owner_ir = try renderUnit(gpa, src, "f", .{});
    defer gpa.free(owner_ir);
    try testing.expectEqualStrings(
        \\fn main.f() -> int {
        \\  slots: s0:fn$int$int
        \\b0:
        \\  %1 = slot_addr s0
        \\  %2 = fn_addr @main.f$l0
        \\  store %1, %2 : int
        \\  %3 = field_addr %1, 8 : rawptr
        \\  %4 = iconst 0
        \\  store %3, %4 : rawptr
        \\  %5 = slot_addr s0
        \\  %6 = load %5 : int
        \\  %7 = field_addr %5, 8 : rawptr
        \\  %8 = load %7 : rawptr
        \\  %9 = iconst 3
        \\  %10 = call_indirect %6(%8, %9)
        \\  br b1(%10)
        \\b1(%0:int):
        \\  ret %0
        \\}
        \\
    , owner_ir);
}

test "a bound method creates {adapter, env} and the adapter calls the witness" {
    const gpa = testing.allocator;
    const src =
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, x: int) -> int { self.n + x } }
        \\fn f() -> int {
        \\ acc := Acc { n: 1 }
        \\ g := acc.plus
        \\ return g(2)
        \\}
        \\
    ;
    const ir = try renderUnit(gpa, src, "f", .{});
    defer gpa.free(ir);
    // The env is a fresh cell holding a copy of `acc`, made once; the pair's code is the
    // adapter, which `adapter_emit` lowers to a call of `main.Acc.plus` with that copy.
    try testing.expectEqualStrings(
        \\fn main.f() -> int {
        \\  slots: s0:Acc s1:fn$int$int s2:fn$int$int
        \\b0:
        \\  %1 = slot_addr s0
        \\  %2 = field_addr %1, 0 : int
        \\  %3 = iconst 1
        \\  store %2, %3 : int
        \\  %4 = slot_addr s1
        \\  %5 = iconst 8
        \\  %6 = call @gc_alloc(%5)
        \\  %7 = slot_addr s0
        \\  copy %6 <- %7 : Acc
        \\  %8 = slot_addr s2
        \\  %9 = fn_addr @main.Acc.plus$bound
        \\  store %8, %9 : int
        \\  %10 = field_addr %8, 8 : rawptr
        \\  store %10, %6 : rawptr
        \\  %11 = slot_addr s2
        \\  copy %4 <- %11 : fn$int$int
        \\  %12 = slot_addr s1
        \\  %13 = load %12 : int
        \\  %14 = field_addr %12, 8 : rawptr
        \\  %15 = load %14 : rawptr
        \\  %16 = iconst 2
        \\  %17 = call_indirect %13(%15, %16)
        \\  br b1(%17)
        \\b1(%0:int):
        \\  ret %0
        \\}
        \\
    , ir);
}

test "a bound mut self method passes the box pointer" {
    const gpa = testing.allocator;
    const src =
        \\struct C { n: int }
        \\impl C { fn bump(mut self) { self.n = self.n + 1 } }
        \\fn f() -> int {
        \\ c := C { n: 0 }
        \\ inc := c.bump
        \\ inc()
        \\ return c.n
        \\}
        \\
    ;
    const ir = try renderUnit(gpa, src, "f", .{});
    defer gpa.free(ir);
    // `c` lives in a cell (s0 holds its pointer), and the env holds that pointer, so the
    // adapter's `bump` and the later `c.n` reach the same storage.
    try testing.expectEqualStrings(
        \\fn main.f() -> int {
        \\  slots: s0:rawptr s1:fn$unit s2:fn$unit
        \\b0:
        \\  %1 = iconst 8
        \\  %2 = call @gc_alloc(%1)
        \\  %3 = slot_addr s0
        \\  store %3, %2 : rawptr
        \\  %4 = slot_addr s0
        \\  %5 = load %4 : rawptr
        \\  %6 = field_addr %5, 0 : int
        \\  %7 = iconst 0
        \\  store %6, %7 : int
        \\  %8 = slot_addr s1
        \\  %9 = slot_addr s0
        \\  %10 = load %9 : rawptr
        \\  %11 = iconst 8
        \\  %12 = call @gc_alloc(%11)
        \\  store %12, %10 : rawptr
        \\  %13 = slot_addr s2
        \\  %14 = fn_addr @main.C.bump$boundbox
        \\  store %13, %14 : int
        \\  %15 = field_addr %13, 8 : rawptr
        \\  store %15, %12 : rawptr
        \\  %16 = slot_addr s2
        \\  copy %8 <- %16 : fn$unit
        \\  %17 = slot_addr s1
        \\  %18 = load %17 : int
        \\  %19 = field_addr %17, 8 : rawptr
        \\  %20 = load %19 : rawptr
        \\  call_indirect %18(%20)
        \\  %21 = slot_addr s0
        \\  %22 = load %21 : rawptr
        \\  %23 = field_addr %22, 0 : int
        \\  %24 = load %23 : int
        \\  br b1(%24)
        \\b1(%0:int):
        \\  ret %0
        \\}
        \\
    , ir);
}

/// `both`, called from `main` with `args`; a local holding a fn is no spec target.
fn specSrc(comptime args: []const u8) []const u8 {
    return
    \\fn add(x: int, y: int) -> int { x + y }
    \\fn both[T](f: fn(T, T) -> T, g: fn(T, T) -> T, x: T) -> T { g(f(x, x), x) }
    \\fn main() -> int {
    \\    h := add
    ++ "\n    both(" ++ args ++ ")\n}\n";
}

test "a spec unit calls its named target directly" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa, specSrc("h, add, 1"), "main.both$int$sp1$main.add", .{ .specialize = true });
    defer gpa.free(ir);
    // `g` (param 1) is a direct call without an env; `f` (param 0, no target) stays a value
    // call through its pair.
    try testing.expectEqualStrings(
        \\fn main.both$int$sp1$main.add(s0, s1, s2) -> int {
        \\  slots: s0:fn$int$int$int s1:fn$int$int$int s2:int
        \\b0:
        \\  %1 = slot_addr s0
        \\  %2 = load %1 : int
        \\  %3 = field_addr %1, 8 : rawptr
        \\  %4 = load %3 : rawptr
        \\  %5 = slot_addr s2
        \\  %6 = load %5 : int
        \\  %7 = slot_addr s2
        \\  %8 = load %7 : int
        \\  %9 = call_indirect %2(%4, %6, %8)
        \\  %10 = slot_addr s2
        \\  %11 = load %10 : int
        \\  %12 = call @main.add(%9, %11)
        \\  br b1(%12)
        \\b1(%0:int):
        \\  ret %0
        \\}
        \\
    , ir);
}

test "a lambda target receives the env from the pair" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa, specSrc("fn(x: int, y: int) -> int { x + y }, h, 1"), "main.both$int$sp0$main$l0", .{ .specialize = true });
    defer gpa.free(ir);
    // The env word at +8 of `f`'s pair is the lambda unit's hidden first arg; there is no
    // code load and no indirect call for `f`.
    try testing.expectEqualStrings(
        \\fn main.both$int$sp0$main$l0(s0, s1, s2) -> int {
        \\  slots: s0:fn$int$int$int s1:fn$int$int$int s2:int
        \\b0:
        \\  %1 = slot_addr s1
        \\  %2 = load %1 : int
        \\  %3 = field_addr %1, 8 : rawptr
        \\  %4 = load %3 : rawptr
        \\  %5 = slot_addr s0
        \\  %6 = field_addr %5, 8 : rawptr
        \\  %7 = load %6 : rawptr
        \\  %8 = slot_addr s2
        \\  %9 = load %8 : int
        \\  %10 = slot_addr s2
        \\  %11 = load %10 : int
        \\  %12 = call @main$l0(%7, %9, %11)
        \\  %13 = slot_addr s2
        \\  %14 = load %13 : int
        \\  %15 = call_indirect %2(%4, %12, %14)
        \\  br b1(%15)
        \\b1(%0:int):
        \\  ret %0
        \\}
        \\
    , ir);
}

test "with frame storage a non-escaping env is a frame slot and its mutated capture is unboxed" {
    const gpa = testing.allocator;
    const src =
        \\fn run(f: fn()) { f() }
        \\fn main() -> int {
        \\ t := 0
        \\ run(fn() { t = t + 1 })
        \\ return t
        \\}
        \\
    ;
    const heap = try renderUnit(gpa, src, "main", .{ .frame = false });
    defer gpa.free(heap);
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, heap, "call @gc_alloc"));
    const frame = try renderUnit(gpa, src, "main", .{ .frame = true });
    defer gpa.free(frame);
    try testing.expect(std.mem.indexOf(u8, frame, "call @gc_alloc") == null);
    try testing.expect(std.mem.indexOf(u8, frame, "s0:int s1:env$main$l0 s2:fn$unit") != null);
    // The env word is t's address, stored as an int so it is never a root.
    try testing.expect(std.mem.indexOf(u8, frame, "%3 = slot_addr s1\n  %4 = field_addr %3, 0 : int\n  %5 = slot_addr s0\n  store %4, %5 : int") != null);
}

test "with frame storage a by-value read of an unboxed aggregate copies it first" {
    const gpa = testing.allocator;
    const src =
        \\struct P { x: int, y: int }
        \\fn show(p: P, z: int) -> int { p.x + z }
        \\fn main() -> int {
        \\ p := P { x: 1, y: 2 }
        \\ poke := fn() -> int {
        \\  p.x = 9
        \\  0
        \\ }
        \\ return show(p, poke())
        \\}
        \\
    ;
    const frame = try renderUnit(gpa, src, "main", .{ .frame = true });
    defer gpa.free(frame);
    try testing.expect(std.mem.indexOf(u8, frame, "call @gc_alloc") == null);
    try testing.expect(std.mem.indexOf(u8, frame, "copy ") != null);
}

test "with frame storage an unboxed capture is an int place slot in the lambda whose aggregate stores are not staged" {
    const gpa = testing.allocator;
    // The `mut` param turns staging on for the lambda; only its store is staged, as `p`'s
    // value reads neither `p` nor code that could.
    const src =
        \\struct P { x: int, y: int }
        \\fn run(f: fn()) { f() }
        \\fn f(mut q: P) -> int {
        \\ p := P { x: 1, y: 2 }
        \\ run(fn() {
        \\  p = P { x: 3, y: 4 }
        \\  q = P { x: q.y, y: q.x }
        \\ })
        \\ return p.x
        \\}
        \\fn main() -> int {
        \\ c := P { x: 3, y: 4 }
        \\ return f(c)
        \\}
        \\
    ;
    const cell = try renderUnit(gpa, src, "main.f$l0", .{ .frame = false });
    defer gpa.free(cell);
    try testing.expect(std.mem.indexOf(u8, cell, "slots: s0:rawptr s1:int s2:rawptr") != null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, cell, "copy "));
    const addr = try renderUnit(gpa, src, "main.f$l0", .{ .frame = true });
    defer gpa.free(addr);
    try testing.expect(std.mem.indexOf(u8, addr, "slots: s0:rawptr s1:int s2:int") != null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, addr, "copy "));
}

test "a .ref capture is an int place slot in the lambda, and a non-self .ref stages aggregate stores like its owner" {
    const gpa = testing.allocator;
    const src =
        \\struct P { x: int, y: int }
        \\fn run(f: fn()) { f() }
        \\impl P {
        \\ fn flip(mut self) {
        \\  run(fn() { self = P { x: 3, y: 4 } })
        \\ }
        \\}
        \\fn swap(mut a: P) {
        \\ run(fn() { a = P { x: a.y, y: a.x } })
        \\}
        \\fn swap2(mut a: P) {
        \\ a = P { x: a.y, y: a.x }
        \\}
        \\fn main() -> int {
        \\ p := P { x: 1, y: 2 }
        \\ p.flip()
        \\ swap(p)
        \\ swap2(p)
        \\ return p.x
        \\}
        \\
    ;
    // The env word is the caller's address, read as an int into a place slot; the new
    // value is built whole in a temp and then copied through that address.
    const lam = try renderUnit(gpa, src, "main.swap$l0", .{});
    defer gpa.free(lam);
    try testing.expect(std.mem.indexOf(u8, lam, "slots: s0:rawptr s1:int s2:P\n") != null);
    try testing.expect(std.mem.indexOf(u8, lam, "%2 = field_addr %1, 0 : int\n  %3 = load %2 : int\n  %4 = slot_addr s1\n  store %4, %3 : int\n") != null);
    try testing.expect(std.mem.indexOf(u8, lam, "copy %6 <- %18 : P\n") != null);
    const own = try renderUnit(gpa, src, "swap2", .{});
    defer gpa.free(own);
    try testing.expect(std.mem.indexOf(u8, own, "slots: s0:int s1:P\n") != null);
    try testing.expect(std.mem.indexOf(u8, own, "copy %1 <- %13 : P\n") != null);
    // A `mut self` owner writes its receiver in place when the value does not read it, so
    // its lambda does too.
    const self_lam = try renderUnit(gpa, src, "main.P.flip$l0", .{});
    defer gpa.free(self_lam);
    try testing.expect(std.mem.indexOf(u8, self_lam, "slots: s0:rawptr s1:int\n") != null);
    try testing.expect(std.mem.indexOf(u8, self_lam, "copy ") == null);
}

test "a bound value on a mut param stores the place address in an 8-byte heap env" {
    const gpa = testing.allocator;
    const ir = try renderUnit(gpa,
        \\struct P { x: int }
        \\impl P { fn get(self) -> int { self.x } }
        \\fn f(mut c: P) -> int {
        \\ g := c.get
        \\ return g()
        \\}
        \\fn main() -> int {
        \\ c := P { x: 1 }
        \\ return f(c)
        \\}
        \\
    , "f", .{});
    defer gpa.free(ir);
    try testing.expect(std.mem.indexOf(u8, ir,
        \\  %2 = iconst 8
        \\  %3 = call @gc_alloc(%2)
        \\  %4 = slot_addr s0
        \\  %5 = load %4 : int
        \\  store %3, %5 : int
        \\  %6 = slot_addr s2
        \\  %7 = fn_addr @main.P.get$boundref
        \\
    ) != null);
}

test "Escape.callTarget names the unit lowerCall calls" {
    const gpa = testing.allocator;
    const Escape = @import("symbols/Escape.zig");
    const Lexer = @import("lex.zig");
    const Parser = @import("parse.zig");
    const src =
        \\struct Vec[T] { x: T }
        \\struct Bag[T] { x: T }
        \\struct C { n: int }
        \\impl C {
        \\ fn run(self, f: fn(int)) { f(self.n) }
        \\}
        \\impl Bag[T] {
        \\ fn each(self, f: fn(T)) { f(self.x) }
        \\ fn make(x: T, f: fn(T)) -> Bag[T] {
        \\  f(x)
        \\  Bag { x: x }
        \\ }
        \\}
        \\fn run(f: fn(int)) { f(2) }
        \\fn each[T](xs: Vec[T], f: fn(T)) { f(xs.x) }
        \\fn apply[T](f: fn(T) -> T, x: T) -> T { f(x) }
        \\fn main() -> int {
        \\ t := 0
        \\ run(fn(v: int) { t = t + v })
        \\ each(Vec { x: 3 }, fn(v: int) { t = t + v })
        \\ a := apply[int](fn(v: int) -> int { v + t }, 4)
        \\ c := C { n: 5 }
        \\ c.run(fn(v: int) { t = t + v })
        \\ bag := Bag.make(6, fn(v: int) { t = t + v })
        \\ bag.each(fn(v: int) { t = t + v })
        \\ return t + a
        \\}
        \\
    ;
    const ir = try renderUnit(gpa, src, "main", .{});
    defer gpa.free(ir);

    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    const t: Ast.Tree = .{ .nodes = tree.nodes, .extra = tree.extra };
    var fe = try frontEnd(gpa, tokens, t, src);
    defer fe.deinit(gpa);
    const rr = fe.resolve;
    const tc = fe.typecheck;
    const body: Body.BodyView = .{ .tree = t, .resolutions = rr.resolutions[0], .nodes = .{ .types = tc.node_types[0], .callees = tc.callees[0] } };
    const prog: Body.ProgramView = .{ .sigs = tc.sigs, .methods = tc.methods, .instances = tc.instances, .layouts = tc.layouts };

    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(gpa);
    for (tree.nodes, 0..) |n, k| {
        if (n.tag != .call or n.lhs == Ast.none) continue;
        const passes_fn = for (Ast.rangeSlice(t, n.rhs.int())) |a| {
            if (LayoutEngine.isFnStruct(tc.node_types[0][a.int()], tc.layouts)) break true;
        } else false;
        if (!passes_fn) continue;
        const target = Escape.callTarget(body, prog, tc.bodies, Ast.Index.from(@intCast(k))) orelse return error.TestExpectedTarget;
        const sym = tc.bodies.items[target.unit].sig.name;
        var buf: [128]u8 = undefined;
        const call = try std.fmt.bufPrint(&buf, "call @{s}(", .{sym});
        if (std.mem.indexOf(u8, ir, call) == null) {
            std.debug.print("callTarget names {s}, lower calls something else:\n{s}\n", .{ sym, ir });
            return error.TestUnexpectedResult;
        }
        try got.print(gpa, "{s}/{d} ", .{ sym, target.self_n });
    }
    try testing.expectEqualStrings("main.run/0 main.each$int/0 main.apply$int/0 main.C.run/1 main.Bag.make$int/0 main.Bag.each$int/1 ", got.items);
}
