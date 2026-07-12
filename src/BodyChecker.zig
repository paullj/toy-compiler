const std = @import("std");
const Ast = @import("ast/Ast.zig");
const Token = @import("ast/Token.zig").Token;
const TokenTag = @import("ast/Token.zig").Tag;
const Resolution = @import("symbols/Resolution.zig").Resolution;
const DiagnosticSink = @import("diagnostics/Sink.zig");
const LayoutEngine = @import("layout/Engine.zig");
const Type = @import("layout/Type.zig").Type;
const StructSym = LayoutEngine.StructSym;
const VariantSym = LayoutEngine.VariantSym;
const EnumSym = LayoutEngine.EnumSym;

// Pass A (`Typecheck`) owns the whole-program model + the shared type-reference
// helpers (`refs`) + the frozen `Model`/`FnSym`/`LoopCtx` descriptors; Pass C reads
// them here. The import is mutual (types.zig constructs a `BodyChecker` per fn), which
// Zig resolves lazily — there is no by-value type cycle (`model` is a pointer).
const Typecheck = @import("types.zig");
const Conform = Typecheck.conform;
const Composite = @import("symbols/Composite.zig");
const Infer = @import("symbols/Infer.zig");
const Intrinsic = @import("symbols/Intrinsic.zig");
const conform = @import("types/conform.zig");
const ControlFlow = @import("ControlFlow.zig");
const PatternChecker = @import("PatternChecker.zig");
const Literal = @import("types/literal.zig");
const Model = Typecheck.Model;
const FnSym = Typecheck.FnSym;
const LoopCtx = Typecheck.LoopCtx;
const GraphCtx = Typecheck.GraphCtx;
const refs = Typecheck.refs;

/// The form a construction NODE supplies (independent of the declared variant).
const InitForm = enum { unit, tuple, @"struct" };

/// Substitute a template type through a concrete type-arg tuple: a
/// `type_var(ord)` becomes `targs[ord]`; an `App(ctor, [pat..])` recursively
/// substitutes each arg and re-interns (so a template's field pattern `Box[T]`
/// grounds to `Box[int]`); any concrete type passes through. Needs the intern table,
/// hence the `bc` receiver. The sibling of `Typecheck.substType`.
pub fn substTy(bc: *BodyChecker, ty: Type, targs: []const Type) Type {
    if (ty.isTypeVar()) {
        const ord = ty.typeVarOrd();
        return if (ord < targs.len) targs[ord] else Type.invalid;
    }
    if (ty.isApp()) {
        const e = bc.composite.at(ty.appIdx());
        var buf: [8]Type = undefined;
        const sub: []Type = if (e.args.len <= buf.len) buf[0..e.args.len] else (bc.gpa.alloc(Type, e.args.len) catch return Type.invalid);
        defer if (e.args.len > buf.len) bc.gpa.free(sub);
        for (e.args, 0..) |a, i| sub[i] = substTy(bc, a, targs);
        const idx = bc.composite.intern(bc.gpa, e.ctor, sub, e.ctor_is_enum) catch return Type.invalid;
        return Type.app(idx);
    }
    return ty;
}

/// True when `ty` is a concrete value type usable as a monomorphization type-arg. This
/// predicate admits a ground `App` (a generic-struct instance used as a type-arg). Mirrors
/// `Typecheck.isConcreteValue`.
fn isConcreteValue(ty: Type) bool {
    return switch (ty.kind) {
        .int, .bool, .str, .@"struct", .@"enum", .app => true,
        else => false,
    };
}

/// A human name for an Option/Result family, for the `?` mismatch diagnostic.
fn familyName(fam: LayoutEngine.NativeEnumFamily) []const u8 {
    return switch (fam) {
        .option => "Option",
        .result => "Result",
        .none => "a non-Option/Result",
    };
}

/// Per-function checking context over a FROZEN `*const Model`. Holds the per-fn
/// scratch (slot_types/cur_ret/loop_stack/expected) and the cursor (tree/tokens/
/// source/resolutions/node_types/graph_mod) — all set once at construction from
/// the fn's owning module, never swapped (gphSelect's save/restore is gone on this
/// path). Diagnostics go to a LOCAL `sink` (scoped to the fn's module); the per-fn
/// body region (`checkBodies`) merges it into the shared sink in fn-id order AFTER
/// the body walk, so parallel Pass C never touches shared mutable diag state.
/// `node_types` aliases the program-wide array but each BodyChecker writes ONLY its
/// own fn's node span (disjoint by construction).
pub const BodyChecker = struct {
    model: *const Model,
    gpa: std.mem.Allocator,

    // Cursor — set at construction from the fn's owning module; never swapped.
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const Resolution,
    node_types: []Type,
    graph_mod: u32,

    // Per-fn scratch.
    slot_types: std.ArrayList(Type) = .empty,
    cur_ret: Type = .unit,
    loop_stack: std.ArrayList(LoopCtx) = .empty,
    expected: ?Type = null,
    /// Whether the checker is inside an `unsafe { .. }` block. Gates the raw-pointer
    /// `store`/`load` ops — a use outside `unsafe` is a T0038 error.
    in_unsafe: bool = false,

    // Local diagnostic sink (merged into the shared result by the Pass-C driver).
    // Its scope is set once in `bodyCheckerFor` to the fn's owning module (graph)
    // or left NO_SCOPE (single-file).
    sink: DiagnosticSink,

    gph_fn_names: ?[]const []const u8 = null,

    /// The shared composite (`App`) intern table, borrowed from the owning
    /// `Typecheck` (stable heap address). Generic-struct construction/field-access
    /// forms + reads back `App`s here; the mono tail reifies them away afterward.
    composite: *Composite,

    /// Per-instance substitution, set ONLY by the monomorphization tail's
    /// per-instance re-check (`Typecheck.recheck`). When set, a generic-parameter
    /// type-ref name inside the body resolves through it (via `genericParamType`) to
    /// the concrete arg; null on every normal Pass-C body walk, which is therefore
    /// byte-identical. Borrowed for the duration of one re-check.
    subst: ?Subst = null,

    /// The receiver type when checking an inherent method's body, set by
    /// `bodyCheckerFor` from the method's `FnSym.self_type`; null for a non-method.
    /// Consumed by the `Self` type-ref hook (`selfType`).
    cur_self_type: ?Type = null,

    /// The per-generic-param bound protocol ids (`[T has P]`), indexed by type-var
    /// ordinal — `bound_protocols[ord]` is the protocol bounding param `ord`, or null if
    /// unbounded. Set by `bodyCheckerFor` from `FnSym.generic_bounds`; empty (inert) for
    /// a non-generic/unbounded fn. Consumed ONLY by the bound-as-axiom `type_var`-receiver
    /// method dispatch in `typeOfCall` (a bounded template's body check); in a per-instance
    /// re-check the receiver is grounded, so that branch never fires.
    bound_protocols: []const ?u32 = &.{},

    /// The per-generic-param bound protocol type-args (`[T has P[args]]`), parallel to
    /// `bound_protocols`. `bound_protocol_args[ord]` grounds the bound protocol's own
    /// type-params in the bound-as-axiom `type_var`-receiver dispatch (so `v.into[int]()`
    /// on a bounded `T` types the protocol method with `U == int`). Empty (inert) for a
    /// non-generic/unbounded fn. Set by `bodyCheckerFor` from `FnSym.generic_bound_args`.
    bound_protocol_args: []const []const Type = &.{},

    /// The ordered generic-param NAMES, set by `bodyCheckerFor` from
    /// `FnSym.generic_params`; empty for a non-generic fn. Read ONLY by the innermost-
    /// failure diagnostic so an abstract `App`'s `type_var` culprit renders as its param
    /// name (`T`) rather than the opaque `type_var` tag. Borrowed source slices.
    gph_generic_params: []const []const u8 = &.{},

    /// Structural derive requests this fn's body recorded: a `==`/`!=` on a
    /// derivable type with no impl. Moved out into the `BodyResult` after the walk;
    /// merged fn-id-ordered by `checkBodies`. THREAD-LOCAL (one list per BodyChecker),
    /// so the parallel Pass C never shares mutable derive state.
    derive_reqs: std.ArrayList(Typecheck.DeriveReq) = .empty,

    /// The concrete `Result[char/byte, ConvErr]` a `try_into` in this fn typed — the
    /// checker-side capture that gates + seeds the shared fallible-char-conversion witness
    /// (see `types.derive_synth`). `null` until such a conversion is dispatched; moved into
    /// the `BodyResult` after the walk. POD, so no teardown.
    conv_int_char_result: ?Type = null,
    conv_char_byte_result: ?Type = null,
    conv_float_int_result: ?Type = null,

    /// The recursive `conforms` query's memo, keyed by `(protocol, kind, type-id)`.
    /// THREAD-LOCAL (one map per BodyChecker) so the query is race-free under the
    /// parallel body fan-out; the answer is stable (layouts/conformances frozen).
    conforms_memo: std.AutoHashMapUnmanaged(u64, bool) = .empty,

    /// The ordered generic-param names + the concrete args they bind to, in
    /// generic-param order (`names[i]` binds `types[i]`).
    pub const Subst = struct { names: []const []const u8, types: []const Type };

    pub fn deinit(bc: *BodyChecker) void {
        bc.slot_types.deinit(bc.gpa);
        bc.loop_stack.deinit(bc.gpa);
        bc.sink.deinit();
        bc.derive_reqs.deinit(bc.gpa);
        bc.conforms_memo.deinit(bc.gpa);
    }

    /// Walk this fn's body: rebuild the slot table, type the body block against the
    /// declared return type, and enforce definite-return. The cursor + scratch live
    /// on `bc`; the immutable fn/struct/enum tables are read through `bc.model`.
    pub fn checkBody(bc: *BodyChecker, fid: u32, f: FnSym) !void {
        _ = fid;
        const decl = bc.tree.nodes[(f.decl_node).int()];
        const proto = Ast.protoAt(bc.tree, (decl.lhs).int());

        // Rebuild the per-function slot→type table. Parameters get slots 0..N first
        // (the resolver declares them first), then `:=` locals as we encounter them.
        bc.slot_types.clearRetainingCapacity();
        for (f.params) |pty| try bc.slot_types.append(bc.gpa, pty);
        bc.cur_ret = f.ret;

        const want_value = (f.ret.kind != .unit and f.ret.kind != .invalid);
        const body_ty = try bc.typeOfBlockExpected(decl.rhs, want_value, if (want_value) f.ret else null);

        // A non-unit function must produce a value on every path: either all paths
        // structurally return, OR the body's trailing expression has the declared
        // type. `assignable` folds poison/never/eql exactly as before.
        if (want_value and !bc.blockReturns(decl.rhs) and !Type.assignable(f.ret, body_ty)) {
            try bc.sink.emitFmt(bc.byteOf(decl.main_token), "function '{s}' must return {s} but may fall off the end", .{ bc.nameText(decl.main_token), bc.typeName(f.ret) });
        }
        _ = proto;
    }

    // The control-flow / divergence / break family lives in `ControlFlow.zig` as
    // pure functions over a small read-only `Ctx` (the tree + resolutions + enum
    // table + tokens). These thin wrappers build a `Ctx` from `bc` and delegate,
    // so the mutually-recursive AST walks (which drift when a new node is added)
    // are defined and unit-tested in one place.

    pub fn cflow(bc: *const BodyChecker) ControlFlow.Ctx {
        return ControlFlow.Ctx.fromChecker(bc);
    }

    fn blockReturns(bc: *const BodyChecker, block_idx: Ast.Index) bool {
        return ControlFlow.blockReturns(bc.cflow(), block_idx);
    }

    fn stmtReturns(bc: *const BodyChecker, stmt_idx: Ast.Index) bool {
        return ControlFlow.stmtReturns(bc.cflow(), stmt_idx);
    }

    fn blockDiverges(bc: *const BodyChecker, block_idx: Ast.Index) bool {
        return ControlFlow.blockDiverges(bc.cflow(), block_idx);
    }

    fn stmtDiverges(bc: *const BodyChecker, stmt_idx: Ast.Index) bool {
        return ControlFlow.stmtDiverges(bc.cflow(), stmt_idx);
    }

    fn blockHasBreak(bc: *const BodyChecker, block_idx: Ast.Index, target: Ast.Index) bool {
        return ControlFlow.blockHasBreak(bc.cflow(), block_idx, target);
    }

    fn typeOfBlockExpected(bc: *BodyChecker, block_idx: Ast.Index, want_value: bool, exp: ?Type) error{OutOfMemory}!Type {
        const save = bc.expected;
        bc.expected = exp;
        defer bc.expected = save;
        return bc.checkBlock(block_idx, want_value);
    }

    fn checkBlock(bc: *BodyChecker, block_idx: Ast.Index, want_value: bool) error{OutOfMemory}!Type {
        const stmts = Ast.rangeSlice(bc.tree, (bc.tree.nodes[(block_idx).int()].lhs).int());
        if (stmts.len == 0) {
            bc.node_types[(block_idx).int()] = .unit;
            return .unit;
        }
        // The expected type (a one-shot for an inferred `.V`) applies ONLY to this
        // block's trailing value expression, never the non-final statements. Capture
        // it, clear it for the non-final walk, then restore for the trailing expr.
        const block_expected = bc.expected;
        bc.expected = null;
        for (stmts[0 .. stmts.len - 1]) |s| try bc.checkStmt(s); // effect only
        const last = stmts[stmts.len - 1];
        const last_n = bc.tree.nodes[(last).int()];
        var bt: Type = .unit;
        if (last_n.tag == .expr_stmt) {
            bt = try bc.typeOfExpected(last_n.lhs, if (want_value) block_expected else null);
            bc.node_types[(last).int()] = bt; // the expr_stmt carries the value type
        } else if (want_value and (last_n.tag == .if_stmt or last_n.tag == .block)) {
            bt = try bc.typeOfExpected(last, block_expected); // value context: validates + types + memoizes
        } else {
            try bc.checkStmt(last); // statement context (incl. trailing else-less if)
        }
        bc.node_types[(block_idx).int()] = bt;
        return bt;
    }

    fn checkStmt(bc: *BodyChecker, stmt_idx: Ast.Index) error{OutOfMemory}!void {
        const stmt = bc.tree.nodes[(stmt_idx).int()];
        switch (stmt.tag) {
            .var_decl => {
                // `x: T = e` (rhs is the type ref) binds x:T and checks the
                // initializer against it; `x := e` (rhs none) infers x from e.
                const ty: Type = if (stmt.rhs != Ast.none) blk: {
                    const declared = bc.typeFromNode(stmt.rhs);
                    const got = try bc.typeOfExpected(stmt.lhs, declared);
                    if (!Type.assignable(declared, got)) {
                        try bc.sink.emitFmt(bc.byteOf(stmt.main_token), "cannot bind {s} to '{s}' of type {s}", .{ bc.typeName(got), bc.nameText(stmt.main_token), bc.typeName(declared) });
                    }
                    break :blk declared;
                } else try bc.typeOf(stmt.lhs);
                bc.node_types[(stmt_idx).int()] = ty;
                // Record the new local's type at its slot. The resolver bound the
                // var_decl node to a `.local` slot; append/extend slot_types to fit.
                if (bc.resolutions[(stmt_idx).int()] == .local) {
                    const slot = bc.resolutions[(stmt_idx).int()].local;
                    try bc.setSlot(slot, ty);
                }
                if (ty.kind == .unit) {
                    try bc.sink.emitFmt(bc.byteOf(stmt.main_token), "cannot bind () to '{s}'", .{bc.nameText(stmt.main_token)});
                }
            },
            .assign => {
                const target = bc.tree.nodes[(stmt.lhs).int()];
                // Compute the place type FIRST so it can flow into the rhs as the
                // expected type (an inferred `.V` assigned to a known-typed place).
                const lhs: Type = switch (target.tag) {
                    .identifier => if (bc.resolutions[(stmt.lhs).int()] == .local)
                        bc.slotType(bc.resolutions[(stmt.lhs).int()].local)
                    else
                        .invalid,
                    // A `.`-rooted place or a `*r` deref place types as an expression
                    // (the `.unary` case types the boxed `T`, poisoning on a non-reference).
                    .field_access, .tuple_field, .unary => try bc.typeOf(stmt.lhs),
                    else => .invalid,
                };
                bc.node_types[(stmt.lhs).int()] = lhs;
                const rhs = try bc.typeOfExpected(stmt.rhs, if (lhs.kind == .invalid) null else lhs);
                if (!Type.assignable(lhs, rhs)) {
                    try bc.sink.emitFmt(bc.byteOf(target.main_token), "cannot assign {s} to variable of type {s}", .{ bc.typeName(rhs), bc.typeName(lhs) });
                }
            },
            .return_stmt => {
                const ty: Type = if (stmt.lhs == Ast.none) Type.unit else try bc.typeOfExpected(stmt.lhs, if (bc.cur_ret.kind == .invalid) null else bc.cur_ret);
                // Skip when either side is poison so an unknown return type (already
                // reported) doesn't trigger a spurious second diagnostic here. A
                // `never`-typed operand (e.g. `return x` where x binds a break-less
                // loop) is also fine: control never actually reaches the return, so
                // `never` unifies with any declared type — mirrors the trailing-expr
                // body check above.
                if (!Type.assignable(bc.cur_ret, ty)) {
                    try bc.sink.emitFmt(bc.byteOf(stmt.main_token), "return type {s} does not match declared {s}", .{ bc.typeName(ty), bc.typeName(bc.cur_ret) });
                }
            },
            .expr_stmt => _ = try bc.typeOf(stmt.lhs),
            .block => _ = try bc.checkBlock(stmt_idx, false),
            .if_stmt => {
                const ct = try bc.typeOf(stmt.lhs);
                if (ct.kind != .invalid and ct.kind != .bool)
                    try bc.sink.emit(bc.byteOf(bc.tree.nodes[(stmt.lhs).int()].main_token), "if condition must be bool");
                const h = Ast.ifHeaderAt(bc.tree, (stmt.rhs).int());
                _ = try bc.checkBlock(h.then_block, false);
                if (h.else_node != Ast.none) {
                    if (bc.tree.nodes[(h.else_node).int()].tag == .if_stmt)
                        try bc.checkStmt(h.else_node)
                    else
                        _ = try bc.checkBlock(h.else_node, false);
                }
            },
            .while_stmt => try bc.checkWhile(stmt_idx, null),
            .for_stmt => try bc.checkFor(stmt_idx, null),
            .for_in_stmt => try bc.checkForIn(stmt_idx, null),
            .labeled => _ = try bc.checkLabeled(stmt_idx, false),
            .break_stmt => {
                const ctx = bc.targetCtx(stmt_idx) orelse {
                    // No matching context: a bare break with an empty stack ("outside a
                    // loop"); a labeled break is reported by resolve as undefined.
                    if (bc.resolutions[(stmt_idx).int()] != .label)
                        try bc.sink.emit(bc.byteOf(stmt.main_token), "break outside of a loop");
                    if (stmt.lhs != Ast.none) _ = try bc.typeOf(stmt.lhs);
                    return;
                };
                if (stmt.lhs == Ast.none) {
                    ctx.saw_bare_break = true;
                    if (ctx.is_value) ctx.join = try bc.merge(stmt.main_token, ctx.join, Type.unit);
                } else {
                    const vt = try bc.typeOf(stmt.lhs);
                    if (!ctx.is_value) {
                        if (vt.kind != .invalid and vt.kind != .unit)
                            try bc.sink.emit(bc.byteOf(stmt.main_token), "cannot break with a value out of a while/for loop");
                    } else {
                        ctx.saw_value_break = true;
                        ctx.join = try bc.merge(stmt.main_token, ctx.join, vt);
                    }
                }
            },
            .continue_stmt => {
                const ctx = bc.targetCtx(stmt_idx) orelse {
                    if (bc.resolutions[(stmt_idx).int()] != .label)
                        try bc.sink.emit(bc.byteOf(stmt.main_token), "continue outside of a loop");
                    return;
                };
                // `continue` is meaningful only on a loop; a labeled bare block is not.
                if (ctx.kind == .labeled_block)
                    try bc.sink.emit(bc.byteOf(stmt.main_token), "cannot continue a labeled block (not a loop)");
            },
            // A poison leaf as a statement: already-diagnosed, no further check.
            .error_node => {},
            else => _ = try bc.typeOf(stmt_idx),
        }
    }

    fn targetCtx(bc: *BodyChecker, stmt_idx: Ast.Index) ?*LoopCtx {
        const items = bc.loop_stack.items;
        if (bc.resolutions[(stmt_idx).int()] == .label) {
            const target = bc.resolutions[(stmt_idx).int()].label;
            var i = items.len;
            while (i > 0) {
                i -= 1;
                if (items[i].construct_node == target) return &items[i];
            }
            return null;
        }
        var i = items.len;
        while (i > 0) {
            i -= 1;
            if (items[i].kind != .labeled_block) return &items[i];
        }
        return null;
    }

    fn checkWhile(bc: *BodyChecker, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
        const stmt = bc.tree.nodes[(stmt_idx).int()];
        const ct = try bc.typeOf(stmt.lhs);
        if (ct.kind != .invalid and ct.kind != .bool)
            try bc.sink.emit(bc.byteOf(bc.tree.nodes[(stmt.lhs).int()].main_token), "while condition must be bool");
        try bc.loop_stack.append(bc.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
        _ = try bc.checkBlock(stmt.rhs, false);
        _ = bc.loop_stack.pop();
    }

    fn checkFor(bc: *BodyChecker, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
        const stmt = bc.tree.nodes[(stmt_idx).int()];
        const h = Ast.forHeaderAt(bc.tree, (stmt.rhs).int());
        const lo = try bc.typeOf(h.lo);
        const hi = try bc.typeOf(h.hi);
        if (lo.kind != .invalid and lo.kind != .int)
            try bc.sink.emit(bc.byteOf(bc.tree.nodes[(h.lo).int()].main_token), "for range bounds must be int");
        if (hi.kind != .invalid and hi.kind != .int)
            try bc.sink.emit(bc.byteOf(bc.tree.nodes[(h.hi).int()].main_token), "for range bounds must be int");
        if (bc.resolutions[(stmt_idx).int()] == .local) try bc.setSlot(bc.resolutions[(stmt_idx).int()].local, Type.int);
        try bc.loop_stack.append(bc.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
        _ = try bc.checkBlock(stmt.lhs, false);
        _ = bc.loop_stack.pop();
    }

    /// `for x in xs { body }` over a reference-semantic container. The receiver must have
    /// an inherent `iter()` returning a type that conforms to `Iterator[Item]`; the loop
    /// var binds to `Item` (mirroring `checkFor`'s `setSlot(.., int)`). A receiver with no
    /// `iter()` / no Iterator conformance is a compile error naming `Iterator`. `node_types
    /// [stmt]` is set to the iterator type — load-bearing: `scanCalls`/reify/`lowerForIn`
    /// read it, exactly as `dispatchAppMethod` writes the receiver App onto the call node.
    fn checkForIn(bc: *BodyChecker, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
        const stmt = bc.tree.nodes[(stmt_idx).int()];
        const recv = try bc.typeOf(stmt.rhs);
        const iter_ty = bc.resolveIterType(recv) orelse blk: {
            // A non-iterable receiver: name the protocol so the error is greppable.
            if (recv.kind != .invalid)
                try bc.sink.emitFmtCode(.T0023, bc.byteOf(bc.tree.nodes[(stmt.rhs).int()].main_token), "type '{s}' does not conform to protocol 'Iterator'", .{bc.typeName(recv)});
            break :blk Type.invalid;
        };
        const item: Type = if (iter_ty.kind == .invalid)
            .invalid
        else
            try conform.iteratorItem(bc.model, bc.composite, bc.gpa, iter_ty) orelse blk: {
                try bc.sink.emitFmtCode(.T0023, bc.byteOf(bc.tree.nodes[(stmt.rhs).int()].main_token), "type '{s}' does not conform to protocol 'Iterator'", .{bc.typeName(recv)});
                break :blk Type.invalid;
            };
        bc.node_types[(stmt_idx).int()] = iter_ty;
        if (bc.resolutions[(stmt_idx).int()] == .local) try bc.setSlot(bc.resolutions[(stmt_idx).int()].local, item);
        try bc.loop_stack.append(bc.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
        _ = try bc.checkBlock(stmt.lhs, false);
        _ = bc.loop_stack.pop();
    }

    /// The return type of the inherent `iter()` method on `recv`, or null when `recv`
    /// has no such method. Only a generic-type (`App`) receiver is supported this
    /// milestone (Vec); the impl's type-params bind by matching the template's `Self`
    /// pattern against the receiver's args — the SAME `Infer.match` `dispatchAppMethod`
    /// runs. `.invalid` when `iter()` exists but is malformed (self-less / arity).
    fn resolveIterType(bc: *BodyChecker, recv: Type) ?Type {
        if (!recv.isApp()) return null;
        const e = bc.composite.at(recv.appIdx());
        const m = Typecheck.findGenericMethod(bc.model.templates, e.ctor, e.ctor_is_enum, "iter") orelse return null;
        const mf = bc.model.fns[m.fn_id];
        if (!m.has_self or mf.params.len != 1) return Type.invalid; // iter(self), no extra args
        const targs = (Typecheck.bindImplParams(bc.gpa, bc.composite, mf, e.args) catch return Type.invalid) orelse return Type.invalid;
        defer bc.gpa.free(targs);
        return substTy(bc, mf.ret, targs);
    }

    fn checkLabeled(bc: *BodyChecker, idx: Ast.Index, want_value: bool) error{OutOfMemory}!Type {
        const n = bc.tree.nodes[(idx).int()];
        const label = bc.nameText(n.main_token);
        const inner = bc.tree.nodes[(n.lhs).int()];
        const ty: Type = switch (inner.tag) {
            .block => try bc.typeOfLabeledBlock(n.lhs, label, want_value),
            .loop_expr => try bc.typeOfLoop(n.lhs, inner, label),
            .while_stmt => blk: {
                try bc.checkWhile(n.lhs, label);
                break :blk Type.unit;
            },
            .for_stmt => blk: {
                try bc.checkFor(n.lhs, label);
                break :blk Type.unit;
            },
            .for_in_stmt => blk: {
                try bc.checkForIn(n.lhs, label);
                break :blk Type.unit;
            },
            else => Type.invalid,
        };
        bc.node_types[(idx).int()] = ty;
        return ty;
    }

    fn typeOfLabeledBlock(bc: *BodyChecker, block_idx: Ast.Index, label: []const u8, want_value: bool) error{OutOfMemory}!Type {
        try bc.loop_stack.append(bc.gpa, .{ .kind = .labeled_block, .label = label, .construct_node = block_idx, .is_value = true, .join = .never, .saw_value_break = false, .saw_bare_break = false });
        const fall = try bc.checkBlock(block_idx, want_value);
        const ctx = bc.loop_stack.pop().?;
        // The trailing-expr value is unreachable iff the block's last statement
        // diverges; in that case the block's value comes entirely from its breaks.
        const ft: Type = if (bc.blockDiverges(block_idx)) Type.never else fall;
        return bc.merge(bc.tree.nodes[(block_idx).int()].main_token, ft, ctx.join);
    }

    pub fn typeOfExpected(bc: *BodyChecker, node_idx: Ast.Index, exp: ?Type) error{OutOfMemory}!Type {
        const save = bc.expected;
        bc.expected = exp;
        defer bc.expected = save;
        return bc.typeOf(node_idx);
    }

    /// Mint the poison type ONLY after THIS fn's local sink already reported a
    /// diagnostic (the rustc `span_delayed_bug`/`ErrorGuaranteed` analog). `bc.sink`
    /// is a PER-FN sink, so `count() > 0` proves the poison co-occurs with a
    /// user-facing error emitted by this very fn — a silent poison (invalid minted
    /// with no reported error) trips the assert in Debug/ReleaseSafe. It is a plain
    /// `return Type.invalid` in ReleaseFast, so release bytes are unchanged. Only the
    /// value-poison identifier sites (func/module/bare-struct-as-value) route through
    /// here; structural poison (Ast.none, error_node, operand-invalid propagation) is
    /// exempt because it carries no new error of its own.
    fn poison(bc: *const BodyChecker) Type {
        std.debug.assert(bc.sink.count() > 0);
        return Type.invalid;
    }

    /// Type a numeric literal token. Defaults to platform `int`, but adopts an
    /// `expected` integer WIDTH (`x: int8 = 100`, or a narrow-int match scrutinee) and
    /// range-checks the magnitude against it. Adopts the width even on a range error so
    /// the downstream assignability check stays silent — one T0034, never a paired
    /// mismatch. The single source shared by the expression `literal_number` arm and the
    /// PatternChecker numeric-pattern path (both must adopt + range-check identically).
    pub fn typeNumericLiteral(bc: *BodyChecker, main_token: u32, expected: ?Type) error{OutOfMemory}!Type {
        const raw = bc.tokens[main_token].text(bc.source);
        if (expected) |e| if (e.isInteger()) {
            if (!Literal.fitsWidth(raw, e, false))
                try bc.sink.emitFmtCode(.T0034, bc.byteOf(main_token), "literal out of range for type '{s}'", .{bc.typeName(e)});
            return e;
        };
        // Unannotated default is platform `int`: route the range verdict through the
        // SAME decode codegen uses (`Literal.value`), so a literal past 2^64-1 — which
        // no 64-bit `iconst` can represent — is caught here instead of escaping to a
        // codegen-time note. Adopt `int` on error, matching the annotated arm (no cascade).
        // Only the truly unannotated case: a non-integer `expected` (e.g. `s: str = <over-u64>`,
        // or an over-u64 pattern against a bool scrutinee) keeps relying on its own mismatch
        // diagnostic — emitting T0034 there too would recreate the paired report this avoids.
        if (expected == null and Literal.value(raw) == null)
            try bc.sink.emitFmtCode(.T0034, bc.byteOf(main_token), "literal out of range for type '{s}'", .{bc.typeName(Type.int)});
        return Type.int;
    }

    /// Type a char literal `'x'` to the compiler-provided `char` struct. The content is
    /// VALIDATED here (a `''`/`'ab'`/bad-escape/bad-`\u` is a clean T0036 at check time,
    /// never a lower-time crash); lower re-decodes the SAME token for the codepoint value.
    /// Returns the `char` type even on a content error (no cascade), mirroring the numeric
    /// range check. `.invalid` only if no prelude `char` exists (a prelude-less caller).
    pub fn typeCharLiteral(bc: *BodyChecker, main_token: u32) error{OutOfMemory}!Type {
        const raw = bc.tokens[main_token].text(bc.source);
        switch (Literal.decodeChar(raw)) {
            .ok => {},
            .empty => try bc.sink.emitCode(.T0036, bc.byteOf(main_token), "empty char literal"),
            .too_many => try bc.sink.emitCode(.T0036, bc.byteOf(main_token), "char literal must hold exactly one codepoint"),
            .dangling_backslash => try bc.sink.emitCode(.T0036, bc.byteOf(main_token), "char literal ends with a dangling backslash"),
            .unknown_escape => try bc.sink.emitCode(.T0036, bc.byteOf(main_token), "unknown escape in char literal"),
            .bad_hex_escape => try bc.sink.emitCode(.T0036, bc.byteOf(main_token), "malformed '\\x'/'\\u' escape in char literal"),
            .bad_codepoint => try bc.sink.emitCode(.T0036, bc.byteOf(main_token), "'\\u{...}' escape is not a Unicode scalar value"),
            .bad_utf8 => try bc.sink.emitCode(.T0036, bc.byteOf(main_token), "char literal is not valid UTF-8"),
            .malformed => try bc.sink.emitCode(.T0036, bc.byteOf(main_token), "malformed char literal"),
        }
        if (bc.model.prelude) |p| if (p.char_struct) |cid| return Type.structT(cid);
        return .invalid;
    }

    pub fn typeOf(bc: *BodyChecker, node_idx: Ast.Index) error{OutOfMemory}!Type {
        if (node_idx == Ast.none) return .invalid; // structural poison: no emit (exempt)
        const n = bc.tree.nodes[(node_idx).int()];
        const ty: Type = switch (n.tag) {
            .literal_number => try bc.typeNumericLiteral(n.main_token, bc.expected),
            .literal_float => Type.float,
            .literal_bool => Type.@"bool",
            .literal_string => Type.str,
            .literal_char => try bc.typeCharLiteral(n.main_token),
            .identifier => switch (bc.resolutions[(node_idx).int()]) {
                .local => |slot| bc.slotType(slot),
                .func => blk: {
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "function '{s}' is not a value", .{bc.nameText(n.main_token)});
                    break :blk bc.poison();
                },
                .unresolved => blk: {
                    // Resolve quietly skips a struct-named identifier (it expects this
                    // to be a struct type-name in a position it doesn't bind, or a
                    // positional `Point(...)` callee it diagnoses elsewhere). A BARE
                    // struct name used as a value (`q := P`, `P.x`) reaches here with
                    // no diagnostic — report it so it never escapes to codegen.
                    if (bc.activeStructMap().get(bc.nameText(n.main_token)) != null) {
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "type '{s}' is not a value", .{bc.nameText(n.main_token)});
                        break :blk bc.poison();
                    }
                    // Else: an ordinary undeclared name already reported by resolve
                    // (this branch emits nothing) — keep plain poison, no guard.
                    break :blk Type.invalid;
                },
                .label => Type.invalid, // never on an identifier node (break/continue only)
                .module => blk: {
                    // A bare imported-namespace name used as a value (`x := mod`):
                    // a module is not a value. (A `mod.member` access never reaches
                    // here — the receiver is consumed by typeOfFieldAccess/Call.)
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "module '{s}' is not a value", .{bc.nameText(n.main_token)});
                    break :blk bc.poison();
                },
            },
            .unary => blk: {
                const op = bc.tokens[n.main_token].tag;
                // The most-negative signed literal (`-128: int8`, `-9223372036854775808:
                // int`) has a magnitude one past the positive max, so range-checking the
                // bare operand first would spuriously reject it. When negating a literal
                // into a SIGNED width, range-check the magnitude against `maxMagnitude+1`
                // (the signed-min magnitude) and pin the operand's node type so lowering
                // (which parses the positive magnitude then negates + width-wraps) agrees.
                if (op == .minus and n.lhs != Ast.none and bc.tree.nodes[n.lhs.int()].tag == .literal_number) {
                    if (bc.expected) |e| if (e.isSigned()) {
                        const lit_tok = bc.tree.nodes[n.lhs.int()].main_token;
                        const raw = bc.tokens[lit_tok].text(bc.source);
                        if (!Literal.fitsWidth(raw, e, true))
                            try bc.sink.emitFmtCode(.T0034, bc.byteOf(lit_tok), "literal out of range for type '{s}'", .{bc.typeName(e)});
                        bc.node_types[n.lhs.int()] = e;
                        break :blk e;
                    };
                }
                const operand = try bc.typeOf(n.lhs);
                if (operand.kind == .invalid) break :blk Type.invalid; // poison propagation: no emit (exempt)
                switch (op) {
                    .minus => {
                        if (operand.isInteger()) break :blk operand;
                        try bc.sink.emit(bc.byteOf(n.main_token), "operand of '-' must be int");
                    },
                    .bang => {
                        if (operand.kind == .bool) break :blk Type.@"bool";
                        try bc.sink.emit(bc.byteOf(n.main_token), "operand of '!' must be bool");
                    },
                    .tilde => {
                        if (operand.isInteger()) break :blk operand;
                        try bc.sink.emit(bc.byteOf(n.main_token), "operand of '~' must be int");
                    },
                    // `&x` boxes `x` into a fresh managed cell, yielding `Ref[T]`. Reify
                    // runs later, so `Ref[T]` is an `.app` during body-check.
                    .amp => break :blk Type.app(bc.composite.intern(bc.gpa, bc.model.prelude.?.ref_struct.?, &.{operand}, false) catch break :blk Type.invalid),
                    // `*r` place-reads the boxed `T`. The operand must be a `Ref[T]` App.
                    .star => {
                        if (operand.isApp()) {
                            const e = bc.composite.at(operand.appIdx());
                            if (!e.ctor_is_enum and e.ctor == bc.model.prelude.?.ref_struct.?) break :blk e.args[0];
                        }
                        try bc.sink.emit(bc.byteOf(n.main_token), "operand of '*' must be a reference");
                    },
                    else => {},
                }
                break :blk Type.invalid;
            },
            .binary => blk: {
                // Type each operand with NO expected type (the only `typeOf` consumer of
                // `bc.expected` is the literal arm, so scoping it off here is safe), then
                // let a bare-literal operand adopt its typed sibling's width — so `a + 1`
                // (a: int8) adopts int8 and `a + 300` fires T0034 exactly once via the
                // sibling path, never also via the outer expected.
                const saved = bc.expected;
                bc.expected = null;
                var lt = try bc.typeOf(n.lhs);
                var rt = try bc.typeOf(n.rhs);
                bc.expected = saved;
                if (lt.kind == .invalid or rt.kind == .invalid) break :blk Type.invalid; // poison propagation: no emit (exempt)
                const l_lit = bc.tree.nodes[n.lhs.int()].tag == .literal_number;
                const r_lit = bc.tree.nodes[n.rhs.int()].tag == .literal_number;
                if (r_lit and !l_lit and lt.isInteger()) rt = try bc.typeOfExpected(n.rhs, lt);
                if (l_lit and !r_lit and rt.isInteger()) lt = try bc.typeOfExpected(n.lhs, rt);
                // Both operands bare literals (`c: int8 = 1 + 2`): no typed sibling to
                // adopt from, so re-type each under the OUTER expected width (running the
                // per-literal range check) — else both stay platform `int`, the homogeneous
                // arm yields `int`, and the annotated bind is wrongly rejected. A no-op for
                // a platform-`int` target, so `x: int = 1 + 2` is unaffected.
                if (l_lit and r_lit) if (saved) |e| if (e.isInteger()) {
                    lt = try bc.typeOfExpected(n.lhs, e);
                    rt = try bc.typeOfExpected(n.rhs, e);
                };
                const op = bc.tokens[n.main_token].tag;
                const op_text = bc.tokens[n.main_token].text(bc.source);
                switch (op) {
                    .plus, .minus, .star, .slash => {
                        // `+`/`-`/`*`/`/` desugar to the arithmetic protocols Add/Sub/Mul/Div
                        //. Builtin `int` stays inline (a single machine add/sub/mul/sdiv,
                        // no witness). Otherwise HOMOGENEOUS like `==`/`<`: require the same
                        // type FIRST, then type to the operand type (`Out = Self`) iff it
                        // conforms to the operator's protocol — a user struct/enum `impl T has
                        // Add`, or a `[T has Add]` bound in a generic body. str/bool have NO
                        // arithmetic impl (str concat allocates, deferred), so they fall to T0028.
                        if (lt.isInteger() and Type.eql(lt, rt)) break :blk lt;
                        const ap = bc.arithProtocol(op);
                        if (!Type.eql(lt, rt)) {
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must have the same type", .{op_text});
                        } else if (bc.conformsToArith(lt, ap.pid)) {
                            break :blk lt;
                        } else {
                            try bc.sink.emitFmtCode(.T0028, bc.byteOf(n.main_token), "'{s}' requires an '{s}' impl for type '{s}'", .{ op_text, ap.name, bc.typeName(lt) });
                        }
                    },
                    .lt, .lt_eq, .gt, .gt_eq => {
                        // `<`/`>`/`<=`/`>=` desugar to a discriminant test on `Ord::cmp`'s
                        // result. HOMOGENEOUS like `==`: require the same type FIRST,
                        // then type to bool iff the operand conforms to `Ord` — a concrete
                        // int/str/bool prelude conformance or user struct/enum `impl T has Ord`,
                        // or a `[T has Ord]` bound in a generic body.
                        if (!Type.eql(lt, rt)) {
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must have the same type", .{op_text});
                        } else if (try bc.conformsTo(lt, bc.model.preludeProtocols().ord, true)) {
                            break :blk Type.@"bool";
                        } else {
                            try bc.sink.emitFmtCode(.T0027, bc.byteOf(n.main_token), "'{s}' requires an 'Ord' impl for type '{s}'", .{ op_text, bc.nonConformingName(lt, bc.model.preludeProtocols().ord) });
                        }
                    },
                    .eq_eq, .bang_eq => {
                        // `==`/`!=` desugar to `Eq::eq`. HOMOGENEOUS: require the same
                        // type FIRST (so a cross-type compare stays "same type" even if both
                        // sides individually conform), then type to bool iff the operand
                        // conforms to `Eq` — a concrete int/bool/str/unit prelude conformance
                        // or user struct/enum impl, or a `[T has Eq]` bound in a generic body.
                        if (!Type.eql(lt, rt)) {
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must have the same type", .{op_text});
                        } else if (try bc.conformsTo(lt, bc.model.preludeProtocols().eq, true)) {
                            break :blk Type.@"bool";
                        } else if (try bc.deriveBlocker(lt, bc.model.preludeProtocols().eq)) |blocker| {
                            // A struct that would derive `Eq` but for one non-conforming
                            // field names that field (T0029). A payload enum / other type
                            // keeps the "no Eq impl" T0026 below.
                            try bc.sink.emitFmtCode(.T0029, bc.byteOf(n.main_token), "cannot derive 'Eq' for '{s}': field '{s}' of type '{s}' does not conform to 'Eq'", .{ bc.typeName(lt), blocker.name, bc.typeName(blocker.ty) });
                        } else {
                            try bc.sink.emitFmtCode(.T0026, bc.byteOf(n.main_token), "'{s}' requires an 'Eq' impl for type '{s}'", .{ op_text, bc.nonConformingName(lt, bc.model.preludeProtocols().eq) });
                        }
                    },
                    .amp_amp, .pipe_pipe => {
                        if (lt.kind == .bool and rt.kind == .bool) break :blk Type.@"bool";
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must be bool", .{op_text});
                    },
                    .amp, .pipe, .caret, .lt_lt, .gt_gt, .percent => {
                        if (lt.isInteger() and Type.eql(lt, rt)) break :blk lt;
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must be int", .{op_text});
                    },
                    .plus_dot, .minus_dot, .star_dot, .slash_dot => {
                        // The dotted operators are float-ONLY inline machine ops (no protocol
                        // desugar): the non-dotted `+`/`<` on a float falls to T0028/T0027.
                        if (lt.kind == .float and Type.eql(lt, rt)) break :blk lt;
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must both be float", .{op_text});
                    },
                    .lt_dot, .gt_dot, .le_dot, .ge_dot => {
                        if (lt.kind == .float and Type.eql(lt, rt)) break :blk Type.@"bool";
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must both be float", .{op_text});
                    },
                    else => {},
                }
                break :blk Type.invalid;
            },
            .empty_list => blk: {
                // An empty list literal `[]` is typed bidirectionally by its expected
                // type: a `Vec[T]` annotation makes it the type's associated `new()`
                // constructor. With no expected type (a bare `xs := []`) the element type
                // is uninferable — require an annotation.
                const exp = bc.expected orelse {
                    try bc.sink.emit(bc.byteOf(n.main_token), "an empty list literal '[]' requires a type annotation, e.g. 'xs: Vec[int] = []'");
                    break :blk Type.invalid;
                };
                if (!exp.isApp()) {
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "an empty list literal '[]' cannot produce '{s}'", .{bc.typeName(exp)});
                    break :blk Type.invalid;
                }
                const e = bc.composite.at(exp.appIdx());
                if (e.ctor_is_enum or Typecheck.findGenericMethod(bc.model.templates, e.ctor, false, "new") == null) {
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "an empty list literal '[]' cannot produce '{s}'", .{bc.typeName(exp)});
                    break :blk Type.invalid;
                }
                break :blk exp;
            },
            .list_literal => blk: {
                // A non-empty `[e0, ..]` builds a populated `Vec[V]`: V is inferred from
                // the first element (seeded by a `Vec[W]` annotation when present); every
                // remaining element must be assignable to V.
                const elems = Ast.rangeSlice(bc.tree, n.rhs.int());
                const vec_ctor = bc.activeStructMap().get("Vec") orelse {
                    for (elems) |ei| _ = try bc.typeOf(ei);
                    try bc.sink.emit(bc.byteOf(n.main_token), "a list literal requires 'Vec' in scope (try 'import std/vec')");
                    break :blk Type.invalid;
                };
                var seed: ?Type = null;
                if (bc.expected) |exp| if (exp.isApp()) {
                    const ce = bc.composite.at(exp.appIdx());
                    if (ce.ctor == vec_ctor and ce.args.len == 1) seed = ce.args[0];
                };
                const v_ty = try bc.typeOfExpected(elems[0], seed);
                const result = Type.app(bc.internApp(vec_ctor, &.{v_ty}, false) catch break :blk Type.invalid);
                for (elems[1..], 1..) |ei, k| {
                    const at = try bc.typeOfExpected(ei, v_ty);
                    if (!Type.assignable(v_ty, at))
                        try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(ei).int()].main_token), "list element {d}: expected '{s}', got '{s}'", .{ k, bc.typeName(v_ty), bc.typeName(at) });
                }
                if (bc.expected) |exp| if (!Type.eql(result, exp)) {
                    if (seed) |s|
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "list element type '{s}' does not match the annotation's '{s}'", .{ bc.typeName(v_ty), bc.typeName(s) })
                    else
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "a list literal of type '{s}' cannot produce '{s}'", .{ bc.typeName(result), bc.typeName(exp) });
                    break :blk Type.invalid;
                };
                break :blk result;
            },
            .index => blk: {
                // `recv[i]` reads a `Vec[V]` element; the result type is V.
                const rt = try bc.typeOf(n.lhs);
                const it = try bc.typeOfExpected(n.rhs, Type.int);
                const vec_ctor = bc.activeStructMap().get("Vec");
                if (!rt.isApp()) {
                    if (rt.kind != .invalid)
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "cannot index a value of type '{s}'", .{bc.typeName(rt)});
                    break :blk Type.invalid;
                }
                const e = bc.composite.at(rt.appIdx());
                if (e.ctor_is_enum or vec_ctor == null or e.ctor != vec_ctor.? or e.args.len != 1) {
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "cannot index a value of type '{s}'", .{bc.typeName(rt)});
                    break :blk Type.invalid;
                }
                if (it.kind != .int and it.kind != .invalid)
                    try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(n.rhs).int()].main_token), "an index must be 'int', got '{s}'", .{bc.typeName(it)});
                break :blk e.args[0];
            },
            .call => try bc.typeOfCall(node_idx, n),
            .struct_init => try bc.typeOfStructInit(node_idx, n),
            .field_access => try bc.typeOfFieldAccess(node_idx, n),
            .tuple_field => try bc.typeOfTupleField(n),
            .enum_init_unit, .enum_init_tuple, .enum_init_struct => try bc.typeOfEnumInit(node_idx, n),
            .match_expr => return PatternChecker.typeOfMatch(bc, node_idx, n), // sets node_types itself
            .try_expr => return bc.typeOfTry(node_idx, n), // sets node_types itself
            .literal_unit => Type.unit,
            .block => try bc.checkBlock(node_idx, true),
            .unsafe_block => blk: {
                const save = bc.in_unsafe;
                bc.in_unsafe = true;
                defer bc.in_unsafe = save;
                break :blk try bc.checkBlock(n.lhs, true);
            },
            .if_stmt => try bc.typeOfIf(node_idx, n),
            .loop_expr => return bc.typeOfLoop(node_idx, n, null), // sets node_types itself
            .labeled => return bc.checkLabeled(node_idx, true), // sets node_types itself
            // A poison leaf types as the poison `invalid` — already-diagnosed, so
            // no diagnostic here and no cascade (`assignable` absorbs `.invalid`).
            .error_node => Type.invalid, // already diagnosed in the parser (exempt)
            else => Type.invalid, // defensive unknown-kind fallthrough: no emit (exempt)
        };
        bc.node_types[(node_idx).int()] = ty;
        return ty;
    }

    fn typeOfStructInit(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        _ = node_idx;
        const lhs = bc.tree.nodes[(n.lhs).int()];
        // Resolve the constructed type: a plain identifier `P { .. }` names a
        // non-generic struct; a `type_app` lhs `Box[int] { .. }` names a generic-struct
        // INSTANCE — a composite `App`. `targs` are the concrete type-args the
        // template's field PATTERNS are substituted through (empty for a plain struct).
        var ctor_id: u32 = undefined;
        var targs: []const Type = &.{};
        var result: Type = undefined;
        var disp_name: []const u8 = undefined;
        if (lhs.tag == .type_app) {
            const app_ty = bc.typeFromNode(n.lhs); // App or invalid (already diagnosed)
            if (!app_ty.isApp()) {
                for (Ast.rangeSlice(bc.tree, (n.rhs).int())) |fi| _ = try bc.typeOf(bc.tree.nodes[(fi).int()].lhs);
                return .invalid;
            }
            const e = bc.composite.at(app_ty.appIdx());
            ctor_id = e.ctor;
            targs = e.args;
            result = app_ty;
            disp_name = bc.model.structs[ctor_id].name;
        } else {
            // A qualified struct-variant `N.V { ... }` arrives as enum_init_struct (the
            // parser upgrades a `field_access {`), not here — so the non-type_app lhs is
            // always a plain type-name identifier; no enum routing needed.
            disp_name = bc.nameText(lhs.main_token);
            const id = bc.activeStructMap().get(disp_name) orelse {
                for (Ast.rangeSlice(bc.tree, (n.rhs).int())) |fi| _ = try bc.typeOf(bc.tree.nodes[(fi).int()].lhs);
                try bc.sink.emitFmt(bc.byteOf(lhs.main_token), "unknown struct type '{s}'", .{disp_name});
                return .invalid;
            };
            // A generic struct constructed WITHOUT type args (`Box{ v: 1 }`) infers its
            // type-params by matching field VALUE types against the declared field
            // PATTERNS (the matcher over the interned `App`). The inferred `App` is
            // content-addressed, so it selects the SAME reified struct as an explicit
            // `Box[int]{..}` and dedups to one codegen unit.
            if (id < bc.model.structs.len and bc.model.structs[id].is_generic) {
                const gsym = bc.model.structs[id];
                const inits = Ast.rangeSlice(bc.tree, (n.rhs).int());
                // Type each value ONCE (source order); the shared field-check tail reuses
                // these via `pretyped`, so inner-value diagnostics aren't double-emitted.
                const pre = try bc.gpa.alloc(Type, inits.len);
                defer bc.gpa.free(pre);
                for (inits, 0..) |fi_idx, ii| pre[ii] = try bc.typeOf(bc.tree.nodes[(fi_idx).int()].lhs);
                // Align values to decl order + record each field's first supplier (for
                // conflict spans). A missing field's slot stays `.invalid`/`Ast.none`: the
                // matcher's never/invalid rule skips it and its span is never dereferenced.
                const aligned = try bc.gpa.alloc(Type, gsym.field_names.len);
                defer bc.gpa.free(aligned);
                @memset(aligned, Type.invalid);
                const supplier = try bc.gpa.alloc(Ast.Index, gsym.field_names.len);
                defer bc.gpa.free(supplier);
                @memset(supplier, Ast.none);
                for (inits, 0..) |fi_idx, ii| {
                    const fname = bc.nameText(bc.tree.nodes[(fi_idx).int()].main_token);
                    for (gsym.field_names, 0..) |dn, j| {
                        if (std.mem.eql(u8, dn, fname)) {
                            if (supplier[j] == Ast.none) {
                                supplier[j] = fi_idx;
                                aligned[j] = pre[ii];
                            }
                            break;
                        }
                    }
                }
                const n_gp: u32 = @intCast(gsym.generic_params.len);
                const out = try bc.gpa.alloc(Type, n_gp);
                defer bc.gpa.free(out);
                const bnd = try bc.gpa.alloc(bool, n_gp);
                defer bc.gpa.free(bnd);
                const fp = try bc.gpa.alloc(usize, n_gp);
                defer bc.gpa.free(fp);
                switch (Infer.match(n_gp, gsym.field_types, aligned, out, bnd, fp)) {
                    // A field-vs-field conflict is authoritative, reported at the two
                    // supplier spans — the target type never overrides it.
                    .conflict => |c| {
                        const later = bc.tree.nodes[(supplier[c.second_pos]).int()].main_token;
                        const earlier = bc.tree.nodes[(supplier[c.first_pos]).int()].main_token;
                        try bc.sink.emitFmtCodeRelated(.T0015, bc.byteOf(later), bc.byteOf(earlier), "conflicting types for type parameter '{s}': {s} vs {s}", .{ gsym.generic_params[c.ord], bc.typeName(aligned[c.first_pos]), bc.typeName(aligned[c.second_pos]) });
                        return .invalid;
                    },
                    // Target-fill any still-open param from the expected type
                    // (`p: Phantom[int] = Phantom{..}`); an arg-vs-expected disagreement
                    // surfaces as T0015. Byte-identical when no target exists.
                    .ok, .unbound => switch (try bc.reconcileTargetArgs(out, bnd, bc.expectedAppArgs(id, false), gsym.generic_params, lhs.main_token)) {
                        .ok => {
                            const app = Type.app(try bc.internApp(id, out, false));
                            return bc.checkStructFieldInits(n, gsym, out, disp_name, app, pre);
                        },
                        .err => return .invalid,
                        .unbound => |ord| {
                            try bc.sink.emitFmtCode(.T0016, bc.byteOf(lhs.main_token), "cannot infer type parameter '{s}' for '{s}'; add explicit type arguments, e.g. {s}[int]{{ .. }}", .{ gsym.generic_params[ord], disp_name, disp_name });
                            return .invalid;
                        },
                    },
                }
            }
            ctor_id = id;
            result = Type.structT(id);
        }
        const sym = bc.model.structs[ctor_id];
        return bc.checkStructFieldInits(n, sym, targs, disp_name, result, null);
    }

    /// The field-init check shared by all three struct-construction paths (non-generic
    /// plain, explicit `Box[int]{..}`, and the inferred `Box{..}`). When `pretyped`
    /// is non-null it supplies the already-computed value type per init (source order),
    /// so the inferred path — which must type each value ONCE to run inference — does not
    /// re-walk (and thus re-diagnose) the value expressions; `pretyped == null` types
    /// each value here and is byte-identical to the earlier tail.
    fn checkStructFieldInits(
        bc: *BodyChecker,
        n: Ast.Node,
        sym: StructSym,
        targs: []const Type,
        disp_name: []const u8,
        result: Type,
        pretyped: ?[]const Type,
    ) error{OutOfMemory}!Type {
        const inits = Ast.rangeSlice(bc.tree, (n.rhs).int());

        // Track which declared fields are supplied (for missing/duplicate checks).
        var seen = try bc.gpa.alloc(bool, sym.field_names.len);
        defer bc.gpa.free(seen);
        @memset(seen, false);

        for (inits, 0..) |fi_idx, ii| {
            const fi = bc.tree.nodes[(fi_idx).int()];
            const fname = bc.nameText(fi.main_token);
            const vt = if (pretyped) |pt| pt[ii] else try bc.typeOf(fi.lhs);
            // Find the declared field by name.
            var found: ?usize = null;
            for (sym.field_names, 0..) |dn, j| {
                if (std.mem.eql(u8, dn, fname)) {
                    found = j;
                    break;
                }
            }
            if (found) |j| {
                if (seen[j]) {
                    try bc.sink.emitFmt(bc.byteOf(fi.main_token), "duplicate field '{s}' in '{s}'", .{ fname, disp_name });
                }
                seen[j] = true;
                // Substitute the (possibly generic) declared field type through the
                // instance's type-args; a non-generic struct has `targs.len == 0`, so
                // `substTy` is the identity and this is byte-identical.
                const fty = substTy(bc, sym.field_types[j], targs);
                if (!Type.assignable(fty, vt)) {
                    try bc.sink.emitFmt(bc.byteOf(fi.main_token), "field '{s}': expected {s}, got {s}", .{ fname, bc.typeName(fty), bc.typeName(vt) });
                }
            } else {
                try bc.sink.emitFmt(bc.byteOf(fi.main_token), "unknown field '{s}' in '{s}'", .{ fname, disp_name });
            }
        }
        for (sym.field_names, 0..) |dn, j| {
            if (!seen[j]) try bc.sink.emitFmt(bc.byteOf(n.main_token), "missing field '{s}' in '{s}'", .{ dn, disp_name });
        }
        return result;
    }

    fn qualifiedEnumId(bc: *BodyChecker, node_idx: Ast.Index) ?u32 {
        const g = bc.model.graph;
        const n = bc.tree.nodes[(node_idx).int()];
        if (n.tag != .field_access) return null;
        const recv = bc.tree.nodes[(n.lhs).int()];
        if (recv.tag != .identifier) return null;
        if (bc.resolutions[(n.lhs).int()] != .module) return null;
        const recv_name = bc.nameText(recv.main_token);
        const target = g.namespaceOfIn(bc.graph_mod, recv_name) orelse return null;
        const member = bc.nameText(n.main_token);
        return g.mods[target].enum_ids.get(member);
    }

    fn typeOfFieldAccess(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        // A qualified UNIT-variant reference `N.V`: the receiver is an enum type-name
        // identifier (resolved quietly to .unresolved). Treat it as construction.
        const recv = bc.tree.nodes[(n.lhs).int()];
        if (recv.tag == .identifier and bc.activeEnumMap().get(bc.nameText(recv.main_token)) != null) {
            return bc.typeOfEnumInitQualified(node_idx, .unit, n.lhs, n.main_token, Ast.none);
        }
        // A 3-level cross-module unit-variant `mod.Enum.Variant`: the receiver of THIS
        // field_access is the inner `mod.Enum` field_access (graph mode). Resolve the
        // inner to a global enum id and treat this node as a unit-variant construction.
        if (bc.qualifiedEnumId(n.lhs)) |enum_id| {
            const ty = try bc.checkVariant(enum_id, n.main_token, .unit, Ast.none);
            bc.node_types[(node_idx).int()] = ty;
            return ty;
        }
        // An explicit generic-enum UNIT-variant `Opt[int].none`: the receiver is a
        // `type_app` resolving to an enum-`App`. Construct the unit variant through the
        // App's args and type the node as the `App` (reified to `enumT` in the mono tail).
        // Placed BEFORE the value `typeOf(n.lhs)` so a `type_app` receiver never mis-routes
        // into the struct-field-access path.
        if (recv.tag == .type_app) {
            const app_ty = bc.typeFromNode(n.lhs);
            if (app_ty.isApp() and bc.composite.at(app_ty.appIdx()).ctor_is_enum) {
                const e = bc.composite.at(app_ty.appIdx());
                const ty = try bc.checkVariantPayloads(e.ctor, n.main_token, .unit, Ast.none, e.args, null);
                bc.node_types[(node_idx).int()] = ty;
                return ty;
            }
        }
        const base = try bc.typeOf(n.lhs);
        if (base.kind == .invalid) return .invalid;
        // A field access over a generic-struct INSTANCE `b.v` where `b: Box[int]`:
        // the base is a composite `App`; resolve the field's DECLARED (pattern) type
        // through the App's type-args so `v` on `Box[int]` types as `int`.
        if (base.isApp()) {
            const e = bc.composite.at(base.appIdx());
            const sym = bc.model.structs[e.ctor];
            const fname = bc.nameText(n.main_token);
            for (sym.field_names, 0..) |dn, j| {
                if (std.mem.eql(u8, dn, fname)) return substTy(bc, sym.field_types[j], e.args);
            }
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "no field '{s}' in struct '{s}'", .{ fname, sym.name });
            return .invalid;
        }
        if (!base.isStruct()) {
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "cannot access field '{s}' of non-struct type {s}", .{ bc.nameText(n.main_token), bc.typeName(base) });
            return .invalid;
        }
        const sym = bc.model.structs[base.struct_id];
        const fname = bc.nameText(n.main_token);
        for (sym.field_names, 0..) |dn, j| {
            if (std.mem.eql(u8, dn, fname)) return sym.field_types[j];
        }
        try bc.sink.emitFmt(bc.byteOf(n.main_token), "no field '{s}' in struct '{s}'", .{ fname, sym.name });
        return .invalid;
    }

    fn typeOfTupleField(bc: *BodyChecker, n: Ast.Node) error{OutOfMemory}!Type {
        const base = try bc.typeOf(n.lhs);
        if (base.kind == .invalid) return .invalid; // no cascade on an already-poisoned base
        if (!base.isStruct() or !bc.model.structs[base.struct_id].is_tuple) {
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "'.{s}' positional access requires a tuple struct, got {s}", .{ bc.nameText(n.main_token), bc.typeName(base) });
            return .invalid;
        }
        const sym = bc.model.structs[base.struct_id];
        // Match the field-name STRING (the canonical "0".."63") exactly as lower does, so a
        // non-canonical spelling (`p.01`, `p.0_1`) is rejected here rather than type-checking
        // as index N while lower — which string-matches — resolves it to field 0.
        const name = bc.nameText(n.main_token);
        for (sym.field_names, sym.field_types) |fname, fty| {
            if (std.mem.eql(u8, fname, name)) return fty;
        }
        try bc.sink.emitFmt(bc.byteOf(n.main_token), "tuple struct '{s}' has no field .{s}", .{ sym.name, name });
        return .invalid;
    }

    fn typeOfEnumInit(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        _ = node_idx;
        const node_form: InitForm = switch (n.tag) {
            .enum_init_unit => .unit,
            .enum_init_tuple => .tuple,
            .enum_init_struct => .@"struct",
            else => unreachable,
        };
        const args: Ast.Index = if (n.tag == .enum_init_unit) Ast.none else n.rhs;
        // Resolve the enum id: qualified (lhs is the type-name identifier), an explicit
        // generic-enum instance (lhs is a `type_app`), or inferred (the one-shot
        // expected type must be an enum).
        var enum_id: u32 = undefined;
        if (n.lhs != Ast.none) {
            const lhs_node = bc.tree.nodes[(n.lhs).int()];
            // An explicit generic-enum variant `Shape[int].seg{ .. }` / `Either[int,bool].left(..)`
            // reaching here as enum_init_* (the parser upgraded a `type_app`-rooted
            // `field_access`): resolve the App and route through the payload checker with
            // the instance's type-args (which reifies to a concrete `enumT`).
            if (lhs_node.tag == .type_app) {
                const app_ty = bc.typeFromNode(n.lhs);
                if (app_ty.isApp() and bc.composite.at(app_ty.appIdx()).ctor_is_enum) {
                    const e = bc.composite.at(app_ty.appIdx());
                    return bc.checkVariantPayloads(e.ctor, n.main_token, node_form, args, e.args, null);
                }
                // invalid App (already diagnosed) or a struct-App: type args + poison.
                try bc.typeArgsForEffect(node_form, args);
                return .invalid;
            }
            const tname = bc.nameText(lhs_node.main_token);
            enum_id = bc.activeEnumMap().get(tname) orelse {
                try bc.typeArgsForEffect(node_form, args);
                try bc.sink.emitFmt(bc.byteOf(lhs_node.main_token), "'{s}' is not an enum type", .{tname});
                return .invalid;
            };
        } else {
            const exp = bc.expected orelse {
                try bc.typeArgsForEffect(node_form, args);
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "cannot infer the enum type for '.{s}' here", .{bc.nameText(n.main_token)});
                return .invalid;
            };
            // The expected type may be a plain `enumT` or a generic-enum instance `App`
            //. `checkVariant` infers the enum's type-params from the payload for the
            // generic case (a nullary `.none` under a generic expected is uninferable ->
            // T0016; target-typing it is deferred).
            enum_id = bc.scrutEnumId(exp) orelse {
                try bc.typeArgsForEffect(node_form, args);
                if (exp.kind != .invalid)
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "'.{s}' expects an enum type, but {s} was expected here", .{ bc.nameText(n.main_token), bc.typeName(exp) });
                return .invalid;
            };
        }
        return bc.checkVariant(enum_id, n.main_token, node_form, args);
    }

    fn typeOfEnumInitQualified(bc: *BodyChecker, node_idx: Ast.Index, node_form: InitForm, type_node: Ast.Index, vtok: u32, args: Ast.Index) error{OutOfMemory}!Type {
        const tname = bc.nameText(bc.tree.nodes[(type_node).int()].main_token);
        const enum_id = bc.activeEnumMap().get(tname) orelse return .invalid; // caller checked
        const ty = try bc.checkVariant(enum_id, vtok, node_form, args);
        bc.node_types[(node_idx).int()] = ty;
        return ty;
    }

    fn typeArgsForEffect(bc: *BodyChecker, node_form: InitForm, args: Ast.Index) error{OutOfMemory}!void {
        if (args == Ast.none) return;
        if (node_form == .@"struct") {
            for (Ast.rangeSlice(bc.tree, (args).int())) |fi| _ = try bc.typeOf(bc.tree.nodes[(fi).int()].lhs);
        } else {
            for (Ast.rangeSlice(bc.tree, (args).int())) |a| _ = try bc.typeOf(a);
        }
    }

    /// The entry every enum-construction site funnels through. For a non-generic enum it
    /// delegates straight to `checkVariantPayloads` with empty type-args (byte-identical
    /// to the earlier body). For a generic enum constructed WITHOUT explicit type args
    /// (`Wrap.w(5)`, `Opt.none`), it infers the enum's type-params by matching
    /// the payload VALUE types against the variant's declared payload PATTERNS, then
    /// delegates with the inferred args. A payload that binds only SOME params
    /// (`Either.left(x)`) or a nullary variant (`Opt.none`) is uninferable — routed to
    /// T0016 (target-type inference for those is deferred). Explicit
    /// `Either[int,bool].left(..)` never reaches here (it calls `checkVariantPayloads`
    /// directly with the App's args).
    fn checkVariant(bc: *BodyChecker, enum_id: u32, vtok: u32, node_form: InitForm, args: Ast.Index) error{OutOfMemory}!Type {
        const e = bc.model.enums[enum_id];
        if (!e.is_generic) return bc.checkVariantPayloads(enum_id, vtok, node_form, args, &.{}, null);

        const vname = bc.nameText(vtok);
        var vi: ?usize = null;
        for (e.variants, 0..) |v, i| {
            if (std.mem.eql(u8, v.name, vname)) {
                vi = i;
                break;
            }
        }
        const variant = if (vi) |i| e.variants[i] else {
            try bc.typeArgsForEffect(node_form, args);
            try bc.sink.emitFmt(bc.byteOf(vtok), "enum '{s}' has no variant '{s}'", .{ e.name, vname });
            return .invalid;
        };

        // Type each payload value ONCE (source order) so inference sees the value types;
        // pass them to `checkVariantPayloads` as `pretyped` so payload expressions are not
        // re-walked (and inner diagnostics not double-emitted).
        const arg_nodes: []const Ast.Index = if (args == Ast.none) &.{} else Ast.rangeSlice(bc.tree, (args).int());
        const pre = try bc.gpa.alloc(Type, arg_nodes.len);
        defer bc.gpa.free(pre);
        for (arg_nodes, 0..) |a_idx, ii| {
            pre[ii] = if (node_form == .@"struct") try bc.typeOf(bc.tree.nodes[(a_idx).int()].lhs) else try bc.typeOf(a_idx);
        }

        // Align payload value types to the variant's declared payload PATTERNS: positional
        // for a tuple, by-name for a struct. A slot with no supplier stays `.invalid`
        // (the matcher's never/invalid rule skips it). Only used when the payload form
        // matches the variant form; a form mismatch falls through to
        // `checkVariantPayloads` (which reports it) with empty inferred args.
        const want_struct = variant.form == .@"struct";
        const aligned = try bc.gpa.alloc(Type, variant.field_types.len);
        defer bc.gpa.free(aligned);
        @memset(aligned, Type.invalid);
        if (formMatches(node_form, variant.form)) {
            if (want_struct) {
                for (arg_nodes, 0..) |fi_idx, ii| {
                    const fname = bc.nameText(bc.tree.nodes[(fi_idx).int()].main_token);
                    for (variant.field_names, 0..) |dn, j| {
                        if (std.mem.eql(u8, dn, fname)) {
                            if (aligned[j].kind == .invalid) aligned[j] = pre[ii];
                            break;
                        }
                    }
                }
            } else {
                for (arg_nodes, 0..) |_, ii| {
                    if (ii < aligned.len) aligned[ii] = pre[ii];
                }
            }
        }

        const n_gp: u32 = @intCast(e.generic_params.len);
        const out = try bc.gpa.alloc(Type, n_gp);
        defer bc.gpa.free(out);
        const bnd = try bc.gpa.alloc(bool, n_gp);
        defer bc.gpa.free(bnd);
        const fp = try bc.gpa.alloc(usize, n_gp);
        defer bc.gpa.free(fp);
        switch (Infer.match(n_gp, variant.field_types, aligned, out, bnd, fp)) {
            // An arg-vs-arg conflict (two payload values disagree) is authoritative and
            // reported at the payload spans — the target type never overrides it.
            .conflict => |c| {
                try bc.sink.emitFmtCode(.T0015, bc.byteOf(vtok), "conflicting types for type parameter '{s}': {s} vs {s}", .{ e.generic_params[c.ord], bc.typeName(aligned[c.first_pos]), bc.typeName(aligned[c.second_pos]) });
                return .invalid;
            },
            // Both a full arg-bind (`.ok`) and a partial/nullary bind (`.unbound`) funnel
            // through the target-fill: the expected type (`x: Opt[int] = Opt.none`) pins
            // any still-open param, and an arg-vs-expected disagreement surfaces as T0015.
            .ok, .unbound => switch (try bc.reconcileTargetArgs(out, bnd, bc.expectedAppArgs(enum_id, true), e.generic_params, vtok)) {
                .ok => return bc.checkVariantPayloads(enum_id, vtok, node_form, args, out, pre),
                .err => return .invalid,
                .unbound => |ord| {
                    try bc.sink.emitFmtCode(.T0016, bc.byteOf(vtok), "cannot infer type parameter '{s}' for '{s}.{s}'; add explicit type arguments, e.g. {s}[int].{s}", .{ e.generic_params[ord], e.name, vname, e.name, vname });
                    return .invalid;
                },
            },
        }
    }

    /// Whether a construction node's form matches a variant's declared form.
    fn formMatches(node_form: InitForm, vform: LayoutEngine.VariantForm) bool {
        return switch (vform) {
            .unit => node_form == .unit,
            .tuple => node_form == .tuple,
            .@"struct" => node_form == .@"struct",
        };
    }

    /// Check one variant construction's payload against the variant's declared payload
    /// `targs` are the enum instance's concrete type-args — each
    /// declared payload pattern is substituted through them via `substTy` before the
    /// assignability check (empty for a non-generic enum ⇒ `substTy` is the identity ⇒
    /// byte-identical to the earlier body). Returns the enum-`App` for a generic enum (so
    /// the node types as `App`, later reified to `enumT`) or `enumT(enum_id)` for a
    /// non-generic one. `pretyped` (source order, aligned with the payload nodes) skips
    /// re-walking payload values the inference path already typed.
    fn checkVariantPayloads(bc: *BodyChecker, enum_id: u32, vtok: u32, node_form: InitForm, args: Ast.Index, targs: []const Type, pretyped: ?[]const Type) error{OutOfMemory}!Type {
        const e = bc.model.enums[enum_id];
        const vname = bc.nameText(vtok);
        // The result type: the enum-App for a generic instance (reified to `enumT` in the
        // mono tail), else the plain concrete enum.
        const result: Type = if (e.is_generic) Type.app(try bc.internApp(enum_id, targs, true)) else Type.enumT(enum_id);
        var vi: ?usize = null;
        for (e.variants, 0..) |v, i| {
            if (std.mem.eql(u8, v.name, vname)) {
                vi = i;
                break;
            }
        }
        const variant = if (vi) |i| e.variants[i] else {
            try bc.typeArgsForEffect(node_form, args);
            try bc.sink.emitFmt(bc.byteOf(vtok), "enum '{s}' has no variant '{s}'", .{ e.name, vname });
            return .invalid;
        };
        if (!formMatches(node_form, variant.form)) {
            try bc.typeArgsForEffect(node_form, args);
            try bc.sink.emitFmt(bc.byteOf(vtok), "variant '{s}.{s}' is constructed with the wrong form", .{ e.name, vname });
            return result;
        }
        switch (variant.form) {
            .unit => {},
            .tuple => {
                const elems = if (args == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (args).int());
                if (elems.len != variant.field_types.len) {
                    if (pretyped == null) {
                        for (elems) |a| _ = try bc.typeOf(a);
                    }
                    try bc.sink.emitFmt(bc.byteOf(vtok), "variant '{s}.{s}' expects {d} value(s), got {d}", .{ e.name, vname, variant.field_types.len, elems.len });
                    return result;
                }
                for (elems, variant.field_types, 0..) |a, fty_pat, i| {
                    const fty = substTy(bc, fty_pat, targs);
                    const at = if (pretyped) |pt| pt[i] else try bc.typeOfExpected(a, fty);
                    if (!Type.assignable(fty, at))
                        try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(a).int()].main_token), "variant '{s}.{s}': expected {s}, got {s}", .{ e.name, vname, bc.typeName(fty), bc.typeName(at) });
                }
            },
            .@"struct" => {
                const inits = if (args == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (args).int());
                var seen = try bc.gpa.alloc(bool, variant.field_names.len);
                defer bc.gpa.free(seen);
                @memset(seen, false);
                for (inits, 0..) |fi_idx, ii| {
                    const fi = bc.tree.nodes[(fi_idx).int()];
                    const fname = bc.nameText(fi.main_token);
                    var found: ?usize = null;
                    for (variant.field_names, 0..) |dn, j| {
                        if (std.mem.eql(u8, dn, fname)) {
                            found = j;
                            break;
                        }
                    }
                    if (found) |j| {
                        if (seen[j]) try bc.sink.emitFmt(bc.byteOf(fi.main_token), "duplicate field '{s}' in '{s}.{s}'", .{ fname, e.name, vname });
                        seen[j] = true;
                        const fty = substTy(bc, variant.field_types[j], targs);
                        const vt = if (pretyped) |pt| pt[ii] else try bc.typeOfExpected(fi.lhs, fty);
                        if (!Type.assignable(fty, vt))
                            try bc.sink.emitFmt(bc.byteOf(fi.main_token), "field '{s}': expected {s}, got {s}", .{ fname, bc.typeName(fty), bc.typeName(vt) });
                    } else {
                        if (pretyped == null) _ = try bc.typeOf(fi.lhs);
                        try bc.sink.emitFmt(bc.byteOf(fi.main_token), "unknown field '{s}' in '{s}.{s}'", .{ fname, e.name, vname });
                    }
                }
                for (variant.field_names, 0..) |dn, j| {
                    if (!seen[j]) try bc.sink.emitFmt(bc.byteOf(vtok), "missing field '{s}' in '{s}.{s}'", .{ dn, e.name, vname });
                }
            },
        }
        return result;
    }

    fn checkTupleStructInit(bc: *BodyChecker, n: Ast.Node, sid: u32) error{OutOfMemory}!Type {
        const sym = bc.model.structs[sid];
        const name_tok = bc.tree.nodes[(n.lhs).int()].main_token;
        const result = Type.structT(sid);
        const elems = if (n.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (n.rhs).int());
        if (elems.len != sym.field_types.len) {
            for (elems) |a| _ = try bc.typeOf(a); // surface inner arg errors first
            try bc.sink.emitFmt(bc.byteOf(name_tok), "tuple struct '{s}' expects {d} value(s), got {d}", .{ sym.name, sym.field_types.len, elems.len });
            return result;
        }
        for (elems, sym.field_types) |a, fty| {
            const at = try bc.typeOfExpected(a, fty); // expected type drives literal narrowing (integer widths)
            if (!Type.assignable(fty, at))
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(a).int()].main_token), "tuple struct '{s}': expected {s}, got {s}", .{ sym.name, bc.typeName(fty), bc.typeName(at) });
        }
        return result;
    }

    /// The GLOBAL enum id a match scrutinee / variant-pattern expected type refers to:
    /// a plain `enumT` yields its id directly; a generic-enum instance `App`
    /// (`Either[int,bool]`) yields its enum ctor id (coverage/exhaustiveness run over
    /// the TEMPLATE's variants — same names/arity as the reified instance; payload types
    /// are substituted through the App's args at bind sites). A struct-App or scalar is
    /// not an enum here.
    pub fn scrutEnumId(bc: *const BodyChecker, ty: Type) ?u32 {
        if (ty.kind == .@"enum") return ty.enum_id;
        if (ty.isApp()) {
            const e = bc.composite.at(ty.appIdx());
            if (e.ctor_is_enum) return e.ctor;
        }
        return null;
    }

    /// The target type-args when `bc.expected` is an `App` of exactly THIS ctor with
    /// matching enum/struct-ness (target typing). Anything else — a scalar, the
    /// wrong ctor, a plain `enumT`, or no expected type — yields null, so the
    /// `fillExpected` reconcile is a no-op and behavior is byte-identical.
    /// This ctor-gate is the sole guard against a stale/unrelated expected type wrongly
    /// filling; a same-ctor `App` always has `args.len == generic_params.len`, so the
    /// arity matches at the two construction sites.
    fn expectedAppArgs(bc: *const BodyChecker, ctor_id: u32, want_enum: bool) ?[]const Type {
        const exp = bc.expected orelse return null;
        if (!exp.isApp()) return null;
        const e = bc.composite.at(exp.appIdx());
        if (e.ctor != ctor_id or e.ctor_is_enum != want_enum) return null;
        return e.args;
    }

    const Reconciled = union(enum) { ok, err, unbound: u32 };

    /// The shared post-`match` target-fill reconcile for BOTH construction sites
    /// (enum via `checkVariant`, struct via `typeOfStructInit`). It emits the
    /// identically-worded T0015 arg-vs-expected conflict and the identical T0013
    /// concrete-value gate (both differ across the two sites only by `span_tok`).
    /// It returns `.unbound` WITHOUT emitting, so each site keeps its own
    /// byte-identical, site-specific T0016 message. On `.ok` the caller proceeds to
    /// the construction tail (`checkVariantPayloads` / `internApp` + field checks).
    fn reconcileTargetArgs(
        bc: *BodyChecker,
        out: []Type,
        bound: []bool,
        exp_args: ?[]const Type,
        param_names: []const []const u8,
        span_tok: u32,
    ) error{OutOfMemory}!Reconciled {
        switch (Infer.fillExpected(out, bound, exp_args)) {
            .conflict => |c| {
                try bc.sink.emitFmtCode(.T0015, bc.byteOf(span_tok), "conflicting types for type parameter '{s}': {s} inferred from the value, {s} from the expected type", .{ param_names[c.ord], bc.typeName(c.arg), bc.typeName(c.expected) });
                return .err;
            },
            .unbound => |u| return .{ .unbound = u.ord },
            .ok => {
                for (out) |ta| {
                    if (!isConcreteValue(ta)) {
                        try bc.sink.emitCode(.T0013, bc.byteOf(span_tok), "inferred type argument must be a concrete value type; add explicit type arguments");
                        return .err;
                    }
                }
                return .ok;
            },
        }
    }

    /// The `Option`/`Result` family of a type: `.option`/`.result` when it is an `App`
    /// over the prelude Option/Result template (an enum ctor), else `.none`. A user
    /// `enum Option` has a distinct ctor id, so this stays `.none` on it.
    fn optResultFamily(bc: *BodyChecker, ty: Type) LayoutEngine.NativeEnumFamily {
        if (!ty.isApp()) return .none;
        const e = bc.composite.at(ty.appIdx());
        if (!e.ctor_is_enum) return .none;
        return Typecheck.optResultFamilyOf(bc.model, e.ctor);
    }

    /// Type a postfix `?`. The operand must be an `Option`/`Result`, and the
    /// enclosing return type (`cur_ret`) must be a MATCHING one that can absorb the
    /// residual: the SAME family, and — for `Result` — the SAME error type. The
    /// `?`-expression's type is the operand's unwrapped payload (variant-0 type-arg).
    /// The family / error-type checks are kept SEPARATE from payload typing so a
    /// mismatch never masks (nor is masked by) a payload type — the payload is bound
    /// best-effort even on a diagnostic path to avoid a cascade at the `:=`/use site.
    fn typeOfTry(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        const ot = try bc.typeOf(n.lhs);
        if (ot.kind == .invalid) {
            bc.node_types[(node_idx).int()] = .invalid;
            return .invalid; // poison-absorb: the operand was already diagnosed
        }
        const op_fam = bc.optResultFamily(ot);
        const ret_fam = bc.optResultFamily(bc.cur_ret);
        const payload: Type = if (op_fam != .none) blk: {
            const args = bc.composite.at(ot.appIdx()).args;
            break :blk if (args.len >= 1) args[0] else Type.invalid;
        } else Type.invalid;

        if (op_fam == .none) {
            try bc.sink.emitFmtCode(.T0032, bc.byteOf(n.main_token), "'?' operand must be an Option or Result, got {s}", .{bc.typeName(ot)});
        } else if (ret_fam == .none) {
            try bc.sink.emitFmtCode(.T0032, bc.byteOf(n.main_token), "'?' requires the enclosing function to return an Option or Result, but it returns {s}", .{bc.typeName(bc.cur_ret)});
        } else if (op_fam != ret_fam) {
            try bc.sink.emitFmtCode(.T0033, bc.byteOf(n.main_token), "'?' on {s} in a function returning {s}", .{ familyName(op_fam), familyName(ret_fam) });
        } else if (op_fam == .result) {
            const op_args = bc.composite.at(ot.appIdx()).args;
            const ret_args = bc.composite.at(bc.cur_ret.appIdx()).args;
            if (op_args.len >= 2 and ret_args.len >= 2 and !Type.eql(op_args[1], ret_args[1])) {
                // Differing error types are allowed to WIDEN via the `From` protocol —
                // the `err(e)` residual is re-emitted as `RetErr.from(opErr)` iff `RetErr has
                // From[OpErr]`. The operand (source) error type disambiguates a multi-conformance
                // `From[Src]` on one target. With no such conformance (or no `From` protocol —
                // a prelude-less caller), keep the T0033 mismatch. The witness is resolved
                // in lower via `resolveConformanceMethod`, which selects the SAME conformance.
                const widened = if (bc.model.preludeProtocols().from) |from_id|
                    Conform.existence(bc.model, from_id, ret_args[1], &.{op_args[1]})
                else
                    false;
                if (!widened)
                    try bc.sink.emitFmtCode(.T0033, bc.byteOf(n.main_token), "'?' error type {s} does not match the enclosing Result error type {s}", .{ bc.typeName(op_args[1]), bc.typeName(ret_args[1]) });
            }
        }
        bc.node_types[(node_idx).int()] = payload;
        return payload;
    }

    fn typeOfIf(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        _ = node_idx;
        const ct = try bc.typeOf(n.lhs);
        if (ct.kind != .invalid and ct.kind != .bool)
            try bc.sink.emit(bc.byteOf(bc.tree.nodes[(n.lhs).int()].main_token), "if condition must be bool");
        const h = Ast.ifHeaderAt(bc.tree, (n.rhs).int());
        if (h.else_node == Ast.none) {
            try bc.sink.emit(bc.byteOf(n.main_token), "value-if requires else");
            _ = try bc.checkBlock(h.then_block, false); // validate the arm anyway
            return .invalid;
        }
        const then_ty0 = try bc.checkBlock(h.then_block, true);
        const then_ty: Type = if (bc.blockDiverges(h.then_block)) Type.never else then_ty0;
        var else_ty: Type = undefined;
        if (bc.tree.nodes[(h.else_node).int()].tag == .if_stmt) {
            const e0 = try bc.typeOfIf(h.else_node, bc.tree.nodes[(h.else_node).int()]); // else-if ladder
            else_ty = if (bc.stmtDiverges(h.else_node)) Type.never else e0;
        } else {
            const e0 = try bc.checkBlock(h.else_node, true);
            else_ty = if (bc.blockDiverges(h.else_node)) Type.never else e0;
        }
        return bc.merge(n.main_token, then_ty, else_ty);
    }

    fn typeOfLoop(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, label: ?[]const u8) error{OutOfMemory}!Type {
        try bc.loop_stack.append(bc.gpa, .{ .kind = .loop, .label = label, .construct_node = node_idx, .is_value = true, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
        _ = try bc.checkBlock(n.lhs, false); // body in statement ctx; breaks fill join
        const ctx = bc.loop_stack.pop().?;
        const ty: Type = if (!ctx.saw_value_break and !ctx.saw_bare_break) Type.never else ctx.join;
        bc.node_types[(node_idx).int()] = ty;
        return ty;
    }

    pub fn merge(bc: *BodyChecker, at: u32, a: Type, b: Type) error{OutOfMemory}!Type {
        if (a.kind == .invalid or b.kind == .invalid) return .invalid; // poison absorbs
        if (a.kind == .never) return b; // covers never+never → never
        if (b.kind == .never) return a;
        if (Type.eql(a, b)) return a; // agreement → one phi type
        try bc.sink.emitFmt(bc.byteOf(at), "branches yield different types ({s} vs {s})", .{ bc.typeName(a), bc.typeName(b) });
        return .invalid; // mismatch, no coercion
    }

    /// Type-check a resolved method call `recv.m(args)` against the selected witness `m`
    ///: the `mut self` place gate, arity, and per-arg assignability, typing the call
    /// node as the method's return. `fa` is the `field_access` (`main_token` = member,
    /// `lhs` = receiver). Shared by the bare-dispatch path and the explicit-protocol-args
    /// (`v.m[int]()`) path so both type identically. `args` is `n`'s value-arg range.
    fn dispatchMethod(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, fa: Ast.Node, recv_ty: Type, member: []const u8, m: Typecheck.Method) error{OutOfMemory}!Type {
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        const mf = bc.model.fns[m.fn_id];
        if (m.mut_self) {
            if (recv_ty.kind != .@"struct" and recv_ty.kind != .@"enum") {
                try bc.sink.emitFmtCode(.T0022, bc.byteOf(fa.main_token), "mutating method '{s}' is not supported on the builtin type '{s}'; `mut self` is only allowed on struct and enum receivers", .{ member, bc.typeName(recv_ty) });
            } else if (!bc.isMutablePlace(fa.lhs)) {
                try bc.sink.emitFmtCode(.T0019, bc.byteOf(bc.tree.nodes[(fa.lhs).int()].main_token), "cannot call mutating method '{s}' on a temporary; the receiver must be a mutable variable (a local or a field of one)", .{member});
            }
        }
        const self_off: usize = @min(mf.params.len, 1);
        const want = mf.params.len - self_off;
        if (args.len != want) {
            for (args) |a| _ = try bc.typeOf(a);
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ want, args.len });
            bc.node_types[(node_idx).int()] = mf.ret;
            return mf.ret;
        }
        for (args, mf.params[self_off..], 0..) |a, pty, i| {
            const at = try bc.typeOfExpected(a, if (pty.kind == .invalid) null else pty);
            if (!Type.assignable(pty, at)) {
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(a).int()].main_token), "argument {d}: expected {s}, got {s}", .{ i + 1, bc.typeName(pty), bc.typeName(at) });
            }
        }
        bc.node_types[(node_idx).int()] = mf.ret;
        return mf.ret;
    }

    /// Emit the T0025 ambiguity diagnostic: the receiver conforms to one generic
    /// protocol MULTIPLE times and the use site gave no disambiguating type-args. Names the
    /// conflicting conformances in table (source fn-id) order — deterministic at any `-jN`.
    fn emitAmbiguousConformance(bc: *BodyChecker, at_tok: u32, recv: Type, member: []const u8) error{OutOfMemory}!void {
        var list: std.ArrayList(Typecheck.Method) = .empty;
        defer list.deinit(bc.gpa);
        try Typecheck.conformancesFor(bc.model.methods, recv, member, bc.gpa, &list);
        var names: std.ArrayList(u8) = .empty;
        defer names.deinit(bc.gpa);
        for (list.items, 0..) |m, i| {
            if (i != 0) try names.appendSlice(bc.gpa, " and ");
            try names.appendSlice(bc.gpa, bc.model.protocols[m.protocol_id.?].name);
            if (m.protocol_args.len > 0) {
                try names.append(bc.gpa, '[');
                for (m.protocol_args, 0..) |pa, j| {
                    if (j != 0) try names.appendSlice(bc.gpa, ", ");
                    try names.appendSlice(bc.gpa, bc.typeName(pa));
                }
                try names.append(bc.gpa, ']');
            }
        }
        try bc.sink.emitFmtCode(.T0025, bc.byteOf(at_tok), "ambiguous conformance: '{s}' on '{s}' matches multiple conformances ({s}); add explicit protocol type arguments, e.g. .{s}[int]()", .{ member, bc.typeName(recv), names.items, member });
    }

    /// Type an explicit-protocol-args method call `v.m[int](args)`: the callee is a
    /// `type_app` over a `field_access`. The type-arg node_types are written (so lower /
    /// AstWalk read them back to select the SAME witness), then the receiver is dispatched:
    /// a concrete struct/enum/scalar disambiguates the witness by the explicit args; a
    /// `type_var` (bound-as-axiom, checking a bounded template) grounds the bound protocol's
    /// params with the explicit args.
    fn typeOfExplicitMethodCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node) error{OutOfMemory}!Type {
        const fa = bc.tree.nodes[(callee.lhs).int()]; // the field_access `v.m`
        const member = bc.nameText(fa.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        const targ_nodes = Ast.rangeSlice(bc.tree, (callee.rhs).int());
        const explicit = try bc.gpa.alloc(Type, targ_nodes.len);
        defer bc.gpa.free(explicit);
        for (targ_nodes, 0..) |tn, i| {
            const ty = bc.typeFromNode(tn); // subst-aware inside an instance re-check
            bc.node_types[(tn).int()] = ty;
            explicit[i] = ty;
        }
        const recv_ty = try bc.typeOf(fa.lhs); // also populates node_types[recv] for lower
        if (recv_ty.kind == .invalid) return .invalid;
        if (recv_ty.kind == .@"struct" or recv_ty.kind == .@"enum" or recv_ty.isScalar())
        {
            switch (Typecheck.resolveConformanceMethod(bc.model.methods, recv_ty, member, null, explicit)) {
                .one => |m| return try bc.dispatchMethod(node_idx, n, fa, recv_ty, member, m),
                else => {
                    for (args) |a| _ = try bc.typeOf(a);
                    try bc.sink.emitFmtCode(.T0018, bc.byteOf(fa.main_token), "no method '{s}' on type '{s}' for the given protocol type arguments", .{ member, bc.typeName(recv_ty) });
                    return .invalid;
                },
            }
        }
        if (recv_ty.isTypeVar()) {
            // Bound-as-axiom with explicit protocol args (checking a bounded template): the
            // only protocol callable on `T` is its declared bound; ground the resolved
            // method with the explicit args (`Self` -> the `type_var` receiver).
            const ord = recv_ty.typeVarOrd();
            const pid_opt: ?u32 = if (ord < bc.bound_protocols.len) bc.bound_protocols[ord] else null;
            if (pid_opt) |pid| {
                const p = bc.model.protocols[pid];
                for (p.methods, 0..) |mn, k| if (std.mem.eql(u8, mn, member)) {
                    const psig = p.method_params[k];
                    const self_off: usize = @min(psig.len, 1);
                    const want = psig.len - self_off;
                    if (args.len != want) {
                        for (args) |a| _ = try bc.typeOf(a);
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ want, args.len });
                    } else for (args, psig[self_off..], 0..) |a, pty, i| {
                        const wt = Typecheck.groundProtoType(pty, recv_ty, explicit);
                        const at = try bc.typeOfExpected(a, if (wt.kind == .invalid) null else wt);
                        if (!Type.assignable(wt, at))
                            try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(a).int()].main_token), "argument {d}: expected {s}, got {s}", .{ i + 1, bc.typeName(wt), bc.typeName(at) });
                    }
                    const ret = Typecheck.groundProtoType(p.method_rets[k], recv_ty, explicit);
                    bc.node_types[(node_idx).int()] = ret;
                    return ret;
                };
                for (args) |a| _ = try bc.typeOf(a);
                try bc.sink.emitFmtCode(.T0018, bc.byteOf(fa.main_token), "no method '{s}' on type parameter bounded by protocol '{s}'", .{ member, p.name });
                return .invalid;
            }
            for (args) |a| _ = try bc.typeOf(a);
            try bc.sink.emitFmtCode(.T0018, bc.byteOf(fa.main_token), "no method '{s}' on unbounded type parameter", .{member});
            return .invalid;
        }
        for (args) |a| _ = try bc.typeOf(a);
        try bc.sink.emitFmtCode(.T0018, bc.byteOf(fa.main_token), "no method '{s}' on type '{s}'", .{ member, bc.typeName(recv_ty) });
        return .invalid;
    }

    // A qualified tuple-variant construction `N.V(args)` arrives as a `.call` whose
    // callee is a `field_access` over an enum type-name / type_app / qualified
    // namespace. Route it to the matching enum-init checker; return null when the
    // receiver is a value (or a struct-`App`), i.e. a method call the caller dispatches.
    // The type_app-not-App drain returns a non-null `.invalid` (NOT null) so caller
    // method dispatch does not re-run `typeOf` on the args and double-diagnose.
    fn tryEnumConstruction(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node) error{OutOfMemory}!?Type {
        const recv = bc.tree.nodes[(callee.lhs).int()];
        if (recv.tag == .identifier and bc.activeEnumMap().get(bc.nameText(recv.main_token)) != null) {
            return try bc.typeOfEnumInitQualified(node_idx, .tuple, callee.lhs, callee.main_token, n.rhs);
        }
        // An explicit generic-enum tuple-variant construction `Either[int,bool].left(42)`
        //: the callee field_access's receiver is a `type_app` resolving to an
        // enum-`App`. Substitute the variant payload patterns through the App's args
        // and type the node as the `App` (reified to `enumT` in the mono tail).
        if (recv.tag == .type_app) {
            const app_ty = bc.typeFromNode(callee.lhs);
            if (app_ty.isApp() and bc.composite.at(app_ty.appIdx()).ctor_is_enum) {
                const e = bc.composite.at(app_ty.appIdx());
                const ty = try bc.checkVariantPayloads(e.ctor, callee.main_token, .tuple, n.rhs, e.args, null);
                bc.node_types[(node_idx).int()] = ty;
                return ty;
            }
            if (!app_ty.isApp()) {
                // typeFromNode already diagnosed (unknown/arity/not-generic); type the
                // args for effect + poison so nothing cascades.
                for (Ast.rangeSlice(bc.tree, (n.rhs).int())) |arg| _ = try bc.typeOf(arg);
                return Type.invalid;
            }
            // A struct-`App` receiver (`Box[int].m(..)`) is a method call; fall through.
        }
        // A cross-module tuple-variant `mod.Enum.Variant(args)` (graph mode): the
        // callee field_access's receiver is the inner `mod.Enum`.
        if (bc.qualifiedEnumId(callee.lhs)) |enum_id| {
            const ty = try bc.checkVariant(enum_id, callee.main_token, .tuple, n.rhs);
            bc.node_types[(node_idx).int()] = ty;
            return ty;
        }
        return null;
    }

    /// An ASSOCIATED-function call `Vec[int].new(args)`: a `field_access` callee whose
    /// receiver is a generic-STRUCT `type_app`, dispatching to a self-less method in the
    /// type's inherent `impl` (no `self` arg). Returns the (substituted) return type, or
    /// null when this is not an associated call (an instance-method receiver, an enum, a
    /// module — all handled by their own paths). Two error gates:
    ///   * a `type_app` receiver whose member IS an instance method -> T0018;
    ///   * a bare `Vec.new()` (no turbofish) naming a generic struct's associated fn ->
    ///     the turbofish-required T0016 (the element type cannot be inferred).
    fn tryAssociatedCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node) error{OutOfMemory}!?Type {
        const recv = bc.tree.nodes[(callee.lhs).int()];
        const member = bc.nameText(callee.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        if (recv.tag == .type_app) {
            const app_ty = bc.typeFromNode(callee.lhs);
            if (!app_ty.isApp()) return null; // typeFromNode already diagnosed
            const e = bc.composite.at(app_ty.appIdx());
            if (e.ctor_is_enum) return null; // an enum type_app is variant construction
            const m = Typecheck.findGenericMethod(bc.model.templates, e.ctor, false, member) orelse return null;
            if (m.has_self) {
                for (args) |a| _ = try bc.typeOf(a);
                try bc.sink.emitFmtCode(.T0018, bc.byteOf(callee.main_token), "'{s}' is an instance method, not an associated function; call it on a value", .{member});
                return .invalid;
            }
            // Load-bearing: the reified receiver flows to scanCalls / reify / lower off
            // `node_types[type_app]`.
            bc.node_types[(callee.lhs).int()] = app_ty;
            const mf = bc.model.fns[m.fn_id];
            const targs_opt = try Typecheck.bindImplParams(bc.gpa, bc.composite, mf, e.args);
            defer if (targs_opt) |ta| bc.gpa.free(ta);
            const bound_ok = targs_opt != null;
            if (args.len != mf.params.len) {
                for (args) |a| _ = try bc.typeOf(a);
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ mf.params.len, args.len });
                if (!bound_ok) return .invalid;
                const ret = substTy(bc, mf.ret, targs_opt.?);
                bc.node_types[(node_idx).int()] = ret;
                return ret;
            }
            if (!bound_ok) {
                for (args) |a| _ = try bc.typeOfExpected(a, null);
                return .invalid;
            }
            const targs = targs_opt.?;
            for (args, mf.params, 0..) |a, pty, i| {
                const want_ty = substTy(bc, pty, targs);
                const at = try bc.typeOfExpected(a, if (want_ty.kind == .invalid) null else want_ty);
                if (!Type.assignable(want_ty, at))
                    try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(a).int()].main_token), "argument {d}: expected {s}, got {s}", .{ i + 1, bc.typeName(want_ty), bc.typeName(at) });
            }
            const ret = substTy(bc, mf.ret, targs);
            bc.node_types[(node_idx).int()] = ret;
            return ret;
        }
        // A bare `Vec.new()` (no turbofish) naming a generic struct with an associated
        // `new`: the element type is uninferable, so require explicit type arguments.
        if (recv.tag == .identifier and bc.resolutions[(callee.lhs).int()] != .module) {
            const name = bc.nameText(recv.main_token);
            if (bc.activeStructMap().get(name)) |sid| {
                if (sid < bc.structSyms().len and bc.structSyms()[sid].is_generic and
                    Typecheck.findGenericMethod(bc.model.templates, sid, false, member) != null)
                {
                    for (args) |a| _ = try bc.typeOf(a);
                    try bc.sink.emitFmtCode(.T0016, bc.byteOf(recv.main_token), "cannot infer type parameter for '{s}.{s}'; add explicit type arguments, e.g. {s}[int].{s}", .{ name, member, name, member });
                    return .invalid;
                }
            }
        }
        return null;
    }

    fn typeOfCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        const callee = bc.tree.nodes[(n.lhs).int()];
        if (callee.tag == .field_access) {
            if (try bc.tryEnumConstruction(node_idx, n, callee)) |t| return t;
            if (try bc.tryAssociatedCall(node_idx, n, callee)) |t| return t;
            // A method call `recv.m(args)` on a VALUE receiver. The enum-variant /
            // qualified-call cases above fire only for an enum type-name / type_app /
            // qualified-namespace receiver; a field_access callee whose receiver is a
            // plain VALUE of a concrete struct/enum is method dispatch. A qualified fn
            // `mod.f()` binds the field_access itself to `.func` (handled below), so
            // guard on that; a namespace receiver is `.module` (also skipped).
            const recv_is_func = bc.resolutions[(n.lhs).int()] == .func;
            const recv_res = bc.resolutions[(callee.lhs).int()];
            if (!recv_is_func and recv_res != .module) {
                if (try bc.dispatchValueMethod(node_idx, n, callee)) |t| return t;
            }
        }
        // An explicit-protocol-args method call `v.into[int](..)`: the callee is a
        // `type_app` OVER a `field_access` NOT bound to a `.func` (a qualified generic fn
        // `mod.f[int]()` binds its field_access to `.func` and stays a generic FUNCTION
        // call). Intercept BEFORE the generic-function `type_app` path.
        if (callee.tag == .type_app and bc.tree.nodes[(callee.lhs).int()].tag == .field_access and
            bc.resolutions[(callee.lhs).int()] != .func)
            return bc.typeOfExplicitMethodCall(node_idx, n, callee);
        // The `size_of[T]()` / `align_of[T]()` intrinsics: a `type_app` whose base
        // identifier resolves to a core-only `.builtin`. They take one type-arg and no
        // value args and type as `int` (the constant is materialized in lower). Intercept
        // BEFORE the generic-function `type_app` path (which would reject the builtin as
        // non-generic).
        if (callee.tag == .type_app and bc.tree.nodes[(callee.lhs).int()].tag == .identifier) {
            const bres = bc.resolutions[(callee.lhs).int()];
            if (bres == .func and bc.model.fns[bres.func].kind == .builtin) {
                const bn = bc.nameText(bc.tree.nodes[(callee.lhs).int()].main_token);
                const bik = Intrinsic.lookup(bn);
                if (bik == .size_of or bik == .align_of) {
                    const targ_nodes = Ast.rangeSlice(bc.tree, (callee.rhs).int());
                    for (targ_nodes) |tn| bc.node_types[(tn).int()] = bc.typeFromNode(tn);
                    const val_args = Ast.rangeSlice(bc.tree, (n.rhs).int());
                    for (val_args) |arg| _ = try bc.typeOf(arg);
                    if (targ_nodes.len != 1)
                        try bc.sink.emitFmt(bc.byteOf(callee.main_token), "'{s}' expects exactly one type argument", .{bn});
                    if (val_args.len != 0)
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "'{s}' takes no value arguments", .{bn});
                    bc.node_types[(node_idx).int()] = Type.int;
                    return Type.int;
                }
                // `gc_array[T](p)`: wrap a raw cell pointer into a `gc_array[T]` handle (an
                // 8-byte reference over the prelude box template). The value change is
                // checker-only — lower passes the pointer operand straight through.
                if (bik == .gc_array) {
                    const targ_nodes = Ast.rangeSlice(bc.tree, (callee.rhs).int());
                    for (targ_nodes) |tn| bc.node_types[(tn).int()] = bc.typeFromNode(tn);
                    const val_args = Ast.rangeSlice(bc.tree, (n.rhs).int());
                    for (val_args) |arg| _ = try bc.typeOf(arg);
                    if (targ_nodes.len != 1) {
                        try bc.sink.emitFmt(bc.byteOf(callee.main_token), "'{s}' expects exactly one type argument", .{bn});
                        return .invalid;
                    }
                    if (val_args.len != 1)
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "'{s}' expects exactly one value argument", .{bn});
                    const elem = bc.node_types[(targ_nodes[0]).int()];
                    const box = bc.model.prelude.?.gc_array_struct.?;
                    const app = Type.app(bc.composite.intern(bc.gpa, box, &.{elem}, false) catch return .invalid);
                    bc.node_types[(node_idx).int()] = app;
                    return app;
                }
            }
        }
        // An explicit-args generic call `id[int](7)`: the callee is a `type_app`
        // whose base identifier carries the template's `.func`. A pure per-call
        // operation (substitute the params/ret through the explicit args, check the
        // value args) that writes only concrete types into `node_types`.
        if (callee.tag == .type_app) return bc.typeOfGenericCall(node_idx, n, callee);
        // A tuple-struct constructor `N(args)`: an UNRESOLVED identifier callee naming a
        // tuple struct is positional construction. A struct name binds nothing in
        // lookupName, so its callee stays `.unresolved`; gating on that (not merely
        // `!= .func`) means a `.local`/`.param` value shadowing the type name falls
        // through to typeOfDirectCall's "called value is not a function" instead of being
        // silently misrouted to construction. A record struct keeps the "use named
        // construction" reject in typeOfDirectCall; enum-variant construction has a
        // field_access callee (never here).
        if (callee.tag == .identifier and bc.resolutions[(n.lhs).int()] == .unresolved) {
            if (bc.activeStructMap().get(bc.nameText(callee.main_token))) |sid| {
                if (bc.model.structs[sid].is_tuple) return try bc.checkTupleStructInit(n, sid);
            }
        }
        return bc.typeOfDirectCall(node_idx, n);
    }

    /// The plain fn-callee tail of a call `f(args)`: the callee resolves directly
    /// to a function (the construction / value-method / explicit-method / generic
    /// dispatch cases in typeOfCall all declined). Recomputes the callee node.
    fn typeOfDirectCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        const callee = bc.tree.nodes[(n.lhs).int()];
        const callee_res = bc.resolutions[(n.lhs).int()];
        if (callee_res != .func) {
            // Type the args anyway so their own errors surface, then poison.
            for (Ast.rangeSlice(bc.tree, (n.rhs).int())) |arg| _ = try bc.typeOf(arg);
            if (callee_res == .local) {
                try bc.sink.emit(bc.byteOf(n.main_token), "called value is not a function");
            } else if (bc.tree.nodes[(n.lhs).int()].tag == .identifier) {
                // A struct-named callee `Point(1,2)` is positional construction, which
                // we reject — point at named construction instead.
                const cname = bc.nameText(bc.tree.nodes[(n.lhs).int()].main_token);
                if (bc.activeStructMap().get(cname) != null)
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "use named construction '{s} {{ ... }}', not '{s}(...)'", .{ cname, cname });
            } else if (callee_res == .unresolved and bc.tree.nodes[(n.lhs).int()].tag == .field_access) {
                // A qualified call `recv.member(...)` whose callee stayed `.unresolved`:
                // resolve neither bound it to a fn nor reported it (e.g. `recv` is a
                // top-level fn shadowing an import namespace, so the field-access value
                // path is taken and left unresolved). Emit a clean diagnostic at the
                // member token instead of silently poisoning — otherwise the call is
                // dropped and `-o` later crashes in codegen with no user error.
                const fa = bc.tree.nodes[(n.lhs).int()];
                const member = bc.nameText(fa.main_token);
                try bc.sink.emitFmt(bc.byteOf(fa.main_token), "cannot resolve member '{s}' to a callable function", .{member});
            }
            // Any remaining `.unresolved` was already reported by resolve.
            return .invalid;
        }
        const f = bc.model.fns[callee_res.func];
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        // The `print` builtin: a compiler-magic polymorphic builtin accepting ANY
        // `Display`-conforming argument (Q8 — NOT a monomorphized generic). Intercept BEFORE
        // the generic/arg-assignability check below (which previously rejected `print(42)` with
        // "expected str, got int"). Require exactly one arg, require it conform to `Display`
        // (recording a ground struct/enum derive req so the serial barrier synthesizes the
        // `display` unit), and type the call `.unit`. `print("..")` still works (str conforms);
        // a non-conforming arg is T0031, naming the blocking struct field where applicable.
        if (f.kind == .builtin) {
            // The core-only heap/raw-pointer intrinsics. Discriminated by the callee
            // token (a `FnSym` carries no name), typed BEFORE the print/panic 1-arg
            // path since their arities differ (0 for `gc_span_count`, 2 for `store`).
            if (callee.tag == .identifier) {
                const bn = bc.nameText(callee.main_token);
                if (Intrinsic.lookup(bn)) |ik| switch (ik) {
                    .gc_alloc => {
                        if (args.len != 1) {
                            for (args) |arg| _ = try bc.typeOf(arg);
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 1), args.len });
                        } else {
                            const st = try bc.typeOf(args[0]);
                            if (st.kind != .int and st.kind != .invalid)
                                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(args[0]).int()].main_token), "'gc_alloc' size must be an 'int', got '{s}'", .{bc.typeName(st)});
                        }
                        bc.node_types[(node_idx).int()] = Type.rawptr;
                        return Type.rawptr;
                    },
                    .gc_span_count => {
                        if (args.len != 0) {
                            for (args) |arg| _ = try bc.typeOf(arg);
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 0), args.len });
                        }
                        bc.node_types[(node_idx).int()] = Type.int;
                        return Type.int;
                    },
                    .store => {
                        if (args.len != 2) {
                            for (args) |arg| _ = try bc.typeOf(arg);
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 2), args.len });
                        } else {
                            const pt = try bc.typeOf(args[0]);
                            _ = try bc.typeOf(args[1]);
                            if (!bc.in_unsafe)
                                try bc.sink.emitFmtCode(.T0038, bc.byteOf(callee.main_token), "raw pointer 'store' requires an 'unsafe' block", .{});
                            if (pt.kind != .rawptr and pt.kind != .invalid)
                                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(args[0]).int()].main_token), "'store' expects a 'rawptr', got '{s}'", .{bc.typeName(pt)});
                        }
                        bc.node_types[(node_idx).int()] = Type.unit;
                        return Type.unit;
                    },
                    .load => {
                        if (args.len != 1) {
                            for (args) |arg| _ = try bc.typeOf(arg);
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 1), args.len });
                        } else {
                            const pt = try bc.typeOf(args[0]);
                            if (!bc.in_unsafe)
                                try bc.sink.emitFmtCode(.T0038, bc.byteOf(callee.main_token), "raw pointer 'load' requires an 'unsafe' block", .{});
                            if (pt.kind != .rawptr and pt.kind != .invalid)
                                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(args[0]).int()].main_token), "'load' expects a 'rawptr', got '{s}'", .{bc.typeName(pt)});
                        }
                        // A `load` used where a scalar value type is expected (a generic
                        // `ga_get[T]` returning `load(..)`) yields that element type, so a
                        // `bool` element round-trips as `bool`; every other context reads a
                        // 64-bit `int` (mem_selftest is unaffected — its loads are int-typed).
                        const lt: Type = if (bc.expected) |e|
                            (if (e.isScalar() or e.kind == .float or e.kind == .@"struct" or e.kind == .@"enum" or e.kind == .str) e else Type.int)
                        else
                            Type.int;
                        bc.node_types[(node_idx).int()] = lt;
                        return lt;
                    },
                    .offset => {
                        // Raw pointer arithmetic `offset(base, i)` -> `base + i` as a
                        // `rawptr`. `base` is a `rawptr`, an `int`, or a reference-family
                        // handle (a `gc_array[T]`, an 8-byte cell pointer); `i` is an int.
                        // NOT `unsafe`-gated (pure arithmetic) — the `store`/`load` it feeds
                        // stay gated.
                        if (args.len != 2) {
                            for (args) |arg| _ = try bc.typeOf(arg);
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 2), args.len });
                        } else {
                            const bt = try bc.typeOf(args[0]);
                            const it = try bc.typeOf(args[1]);
                            const base_ok = bt.kind == .rawptr or bt.kind == .int or bt.kind == .invalid or bc.isRefPayload(bt);
                            if (!base_ok)
                                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(args[0]).int()].main_token), "'offset' base must be a 'rawptr' or 'int', got '{s}'", .{bc.typeName(bt)});
                            if (it.kind != .int and it.kind != .invalid)
                                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(args[1]).int()].main_token), "'offset' index must be an 'int', got '{s}'", .{bc.typeName(it)});
                        }
                        bc.node_types[(node_idx).int()] = Type.rawptr;
                        return Type.rawptr;
                    },
                    .gc_array => {
                        // The bare-call form `gc_array(p)` is rejected; the wrapping form
                        // is `gc_array[T](p)`, intercepted in `typeOfCall` as a `type_app`.
                        for (args) |arg| _ = try bc.typeOf(arg);
                        try bc.sink.emit(bc.byteOf(n.main_token), "'gc_array' requires a type argument, e.g. gc_array[int](p)");
                        return .invalid;
                    },
                    // `size_of[T]()`/`align_of[T]()` are the type_app form handled above;
                    // a bare-call misuse falls through to the generic path unchanged.
                    .size_of, .align_of => {},
                };
            }
            if (args.len != 1) {
                for (args) |arg| _ = try bc.typeOf(arg);
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 1), args.len });
                bc.node_types[(node_idx).int()] = Type.unit;
                return Type.unit;
            }
            const at = try bc.typeOf(args[0]);
            const at_tok = bc.tree.nodes[(args[0]).int()].main_token;
            if (at.kind == .invalid) {
                // The arg already reported its own error; type the call unit, don't cascade.
                bc.node_types[(node_idx).int()] = Type.unit;
                return Type.unit;
            }
            // `panic(msg)`: the message must be a `str` (it lowers to a raw {ptr,len}
            // write; a non-str arg would be marshalled per its own ABI and misread).
            // Unlike `print` (any Display), panic is str-only. Discriminate by the callee
            // token — the Model fn carries no name. Type the call `.unit` regardless so a
            // bad arg reports exactly once without cascading.
            if (callee.tag == .identifier and std.mem.eql(u8, bc.nameText(callee.main_token), "panic")) {
                if (at.kind != .str)
                    try bc.sink.emitFmt(bc.byteOf(at_tok), "panic message must be a 'str', got '{s}'", .{bc.typeName(at)});
                bc.node_types[(node_idx).int()] = Type.unit;
                return Type.unit;
            }
            if (try bc.conformsTo(at, bc.model.preludeProtocols().display, true)) {
                bc.node_types[(node_idx).int()] = Type.unit;
                return Type.unit;
            }
            if (try bc.deriveBlocker(at, bc.model.preludeProtocols().display)) |blocker| {
                try bc.sink.emitFmtCode(.T0031, bc.byteOf(at_tok), "cannot 'print' a '{s}': field '{s}' of type '{s}' does not conform to 'Display'", .{ bc.typeName(at), blocker.name, bc.typeName(blocker.ty) });
            } else {
                // A `type_var` (a generic param without a `Display` bound) renders as its
                // source name (`T`), mirroring `nonConformingName`, not the opaque `type_var`.
                const nm = if (at.isTypeVar() and at.typeVarOrd() < bc.gph_generic_params.len) bc.gph_generic_params[at.typeVarOrd()] else bc.typeName(at);
                try bc.sink.emitFmtCode(.T0031, bc.byteOf(at_tok), "cannot 'print' a value of type '{s}': it does not conform to 'Display'", .{nm});
            }
            return .invalid;
        }
        // A bare (no-explicit-args) generic call `id(7)`: infer each type-arg by
        // one-sided structural matching of the template's param patterns against the
        // ground argument types, then reuse `applyGenericSig` to check args + type the
        // node. The matcher runs and is discarded here — no `type_var` is ever stored.
        // Explicit `id[int](7)` is handled above (typeOfGenericCall) and still overrides.
        if (f.isGeneric()) {
            // Type-args are inferred only for a PLAIN-IDENTIFIER callee `id(7)`. A bare
            // qualified generic call `mod.id(7)` (field_access callee) is NOT inferred:
            // the three post-typecheck consumers (scanCalls/lower/CallVisitor) key the
            // bare path on a plain identifier too, so accepting it here would type the
            // node concretely but mint NO instance (a `Mono.find` miss in lower). Require
            // explicit type args instead — the same clean reject as before.
            if (callee.tag != .identifier) {
                for (args) |arg| _ = try bc.typeOf(arg);
                try bc.sink.emitCode(.T0013, bc.byteOf(n.main_token), "generic call requires explicit type arguments, e.g. f[int](..)");
                return .invalid;
            }
            const arg_types = try bc.gpa.alloc(Type, args.len);
            defer bc.gpa.free(arg_types);
            for (args, 0..) |arg, i| arg_types[i] = try bc.typeOf(arg); // synth once (self-typing)
            if (args.len != f.params.len) {
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ f.params.len, args.len });
                return .invalid; // never leak f.ret (a type_var) on the error path
            }
            const n_gp: u32 = @intCast(f.generic_params.len);
            const out = try bc.gpa.alloc(Type, n_gp);
            defer bc.gpa.free(out);
            const bnd = try bc.gpa.alloc(bool, n_gp);
            defer bc.gpa.free(bnd);
            const fp = try bc.gpa.alloc(usize, n_gp);
            defer bc.gpa.free(fp);
            switch (Infer.match(n_gp, f.params, arg_types, out, bnd, fp)) {
                .conflict => |c| {
                    // Primary = the LATER arg (source order), related = the earlier one:
                    // a stable, order-independent (symmetric `Type.eql`) span pair that is
                    // reproducible at `-jN`. Emitted before the poison return.
                    const later = bc.tree.nodes[(args[c.second_pos]).int()].main_token;
                    const earlier = bc.tree.nodes[(args[c.first_pos]).int()].main_token;
                    try bc.sink.emitFmtCodeRelated(.T0015, bc.byteOf(later), bc.byteOf(earlier), "conflicting types for type parameter '{s}': {s} vs {s}", .{ f.generic_params[c.ord], bc.typeName(arg_types[c.first_pos]), bc.typeName(arg_types[c.second_pos]) });
                    return .invalid;
                },
                .unbound => |u| {
                    try bc.sink.emitFmtCode(.T0016, bc.byteOf(n.main_token), "cannot infer type parameter '{s}'; add explicit type arguments, e.g. f[int](..)", .{f.generic_params[u.ord]});
                    return .invalid;
                },
                .ok => {
                    // The inferred args must be monomorphizable value types — the SAME
                    // gate `scanCalls` applies, so Pass C and discovery agree on which
                    // bare calls become instances (a `unit`-inferred var, say, is rejected
                    // here rather than silently dropped by discovery → lower miss).
                    for (out) |ta| {
                        if (!isConcreteValue(ta)) {
                            try bc.sink.emitCode(.T0013, bc.byteOf(n.main_token), "inferred type argument must be a concrete value type; add explicit type arguments");
                            return .invalid;
                        }
                    }
                    return bc.applyGenericSig(node_idx, n, f, out, arg_types);
                },
            }
        }
        if (args.len != f.params.len) {
            for (args) |arg| _ = try bc.typeOf(arg);
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ f.params.len, args.len });
            return f.ret;
        }
        for (args, f.params, 0..) |arg, pty, i| {
            const at = try bc.typeOfExpected(arg, if (pty.kind == .invalid) null else pty);
            if (!Type.assignable(pty, at)) {
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(arg).int()].main_token), "argument {d}: expected {s}, got {s}", .{ i + 1, bc.typeName(pty), bc.typeName(at) });
            }
        }
        return f.ret;
    }

    fn dispatchValueMethod(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node) error{OutOfMemory}!?Type {
        const recv_ty = try bc.typeOf(callee.lhs); // also populates node_types[recv] for lower
        if (recv_ty.kind == .invalid) return Type.invalid; // receiver already errored → no cascade
        if (recv_ty.kind == .@"struct" or recv_ty.kind == .@"enum" or recv_ty.isScalar() or recv_ty.kind == .float)
        {
            return try bc.dispatchConcreteMethod(node_idx, n, callee, recv_ty);
        } else if (recv_ty.kind == .app) {
            return try bc.dispatchAppMethod(node_idx, n, callee, recv_ty);
        } else if (recv_ty.isTypeVar()) {
            return try bc.dispatchTypeVarMethod(node_idx, n, callee, recv_ty);
        }
        // No final else: an unhandled receiver kind (.never/.func) falls through to the
        // callee_res != .func path in typeOfCall.
        return null;
    }

    fn dispatchConcreteMethod(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node, recv_ty: Type) error{OutOfMemory}!Type {
        const member = bc.nameText(callee.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        // Resolve the witness via the shared multi-conformance resolver
        // (the SAME rule lower + AstWalk use). `.one` is byte-identical to the
        // `findMethod` for every existing program (inherent / single
        // conformance / prelude); `.ambiguous` (a doubly-conforming generic
        // protocol used with no type-args) is T0025 — never an arbitrary pick.
        switch (Typecheck.resolveConformanceMethod(bc.model.methods, recv_ty, member, null, null)) {
            .one => |m| {
                // An associated (self-less) function must be reached via `Type[..].fn()`,
                // never on a value receiver.
                if (m.is_static) {
                    for (args) |a| _ = try bc.typeOf(a);
                    try bc.sink.emitFmtCode(.T0018, bc.byteOf(callee.main_token), "'{s}' is an associated function; call it as '{s}.{s}(..)'", .{ member, bc.typeName(recv_ty), member });
                    return .invalid;
                }
                return try bc.dispatchMethod(node_idx, n, callee, recv_ty, member, m);
            },
            .ambiguous => {
                for (args) |a| _ = try bc.typeOf(a);
                try bc.emitAmbiguousConformance(callee.main_token, recv_ty, member);
                return .invalid;
            },
            .none => {},
        }
        // Target-directed `.into()` / `.try_into()` on an integer OR `char` receiver:
        // resolve the destination from `bc.expected`. `into` is lossless (int widen /
        // char↔int·byte); `try_into` is fallible (int narrow / int→char / char→byte),
        // returning a synthesized `Result[T, ConvErr]`. A user `into`/`try_into` method was
        // already selected above; the builtin recognizer only matches int/char, so a
        // struct receiver with no such method falls through to T0018 — never a wrong pick.
        const char_id = if (bc.model.prelude) |p| p.char_struct else null;
        const recv_is_char = Typecheck.isCharTy(recv_ty, char_id);
        if ((recv_ty.isInteger() or recv_is_char or recv_ty.kind == .float) and (std.mem.eql(u8, member, "into") or std.mem.eql(u8, member, "try_into"))) {
            if (bc.expected) |exp| {
                if (Typecheck.builtinConvMethod(recv_ty, exp, member, char_id)) |cm| {
                    if (args.len != 0) {
                        for (args) |a| _ = try bc.typeOf(a);
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 0), args.len });
                    }
                    const ret: Type = switch (cm.kind) {
                        .widen, .char_to_int, .byte_to_char, .int_to_float => cm.target,
                        .narrow, .int_to_char, .char_to_byte, .float_to_int => Type.app(try bc.internApp(
                            bc.model.prelude.?.result_enum.?,
                            &.{ cm.target, Type.enumT(bc.model.prelude.?.conv_err_enum.?) },
                            true,
                        )),
                    };
                    // Capture the concrete Result so the synthesis barrier can reify the
                    // fallible char conversions as shared witnesses (frame-overflow fix). The
                    // int→int `.narrow` family has no single program-global witness, so it is
                    // NOT captured — it stays inlined per site.
                    switch (cm.kind) {
                        .int_to_char => bc.conv_int_char_result = ret,
                        .char_to_byte => bc.conv_char_byte_result = ret,
                        .float_to_int => bc.conv_float_int_result = ret,
                        else => {},
                    }
                    bc.node_types[(node_idx).int()] = ret;
                    return ret;
                }
            }
        }
        // A direct `.hash()` on a struct/enum with NO explicit impl: the
        // structural `Hash` derive trigger. Hash has no operator, so (unlike
        // `==`/`Eq`) this is the ONLY firing site — a direct method call. Mirrors
        // the `==` operator's `conformsTo`/`deriveBlocker` split: an all-`Hash`-
        // fields aggregate records the derive + types the call `int`; a struct with
        // a non-conforming field names it (T0030); a payload enum with a non-
        // conforming payload has no single nameable field, so it falls through to
        // T0018. Scalars fall through to `builtinScalarMethod` below.
        if (std.mem.eql(u8, member, "hash") and bc.model.preludeProtocols().hash != null and
            (recv_ty.kind == .@"struct" or recv_ty.kind == .@"enum"))
        {
            if (args.len != 0) {
                for (args) |a| _ = try bc.typeOf(a);
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 0), args.len });
                bc.node_types[(node_idx).int()] = Type.int;
                return Type.int;
            }
            if (try bc.conformsTo(recv_ty, bc.model.preludeProtocols().hash, true)) {
                bc.node_types[(node_idx).int()] = Type.int;
                return Type.int;
            }
            if (try bc.deriveBlocker(recv_ty, bc.model.preludeProtocols().hash)) |blocker| {
                try bc.sink.emitFmtCode(.T0030, bc.byteOf(callee.main_token), "cannot derive 'Hash' for '{s}': field '{s}' of type '{s}' does not conform to 'Hash'", .{ bc.typeName(recv_ty), blocker.name, bc.typeName(blocker.ty) });
                return .invalid;
            }
            // No structural derive and no nameable blocker: fall through to T0018.
        }
        // A builtin scalar protocol method: `n.eq(m)` on int/bool, or
        // `n.hash()` on any scalar. Not a `t.methods` entry (the recognizer is
        // pure), so `findMethod` misses; recognize it here, check arity (`eq` = 1
        // non-self arg, `hash` = 0) + each arg assignable to `Self` (the
        // homogeneous receiver), and type the call to the method's return. A user
        // `impl int has P` was already handled above (its real `fn_id` is in the
        // method table), so this only fires for the builtins.
        if (Typecheck.builtinScalarMethod(recv_ty, member)) |bm| {
            if (args.len != bm.arity) {
                for (args) |a| _ = try bc.typeOf(a);
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ bm.arity, args.len });
            } else for (args) |a| {
                const at = try bc.typeOfExpected(a, recv_ty);
                if (!Type.assignable(recv_ty, at))
                    try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(a).int()].main_token), "argument 1: expected {s}, got {s}", .{ bc.typeName(recv_ty), bc.typeName(at) });
            }
            bc.node_types[(node_idx).int()] = bm.ret;
            return bm.ret;
        }
        // A concrete struct/enum/scalar value with no such method.
        for (args) |a| _ = try bc.typeOf(a);
        try bc.sink.emitFmtCode(.T0018, bc.byteOf(callee.main_token), "no method '{s}' on type '{s}'", .{ member, bc.typeName(recv_ty) });
        return .invalid;
    }

    fn dispatchAppMethod(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node, recv_ty: Type) error{OutOfMemory}!Type {
        // A method call on a generic-type instance `b.get()`: the
        // receiver is an `App` (`Box[int]`; reify runs later in the mono
        // tail). Its ctor selects the `impl Box[T]` method TEMPLATE and the
        // impl's type-params bind by matching the template's `Self`
        // pattern-args against the receiver App's concrete args — the SAME
        // `Infer.match` scanCalls/lower run. Only the impl's params bind; a
        // method's OWN `[U]` generics are NOT inferred (deferred).
        const member = bc.nameText(callee.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        const e = bc.composite.at(recv_ty.appIdx());
        if (Typecheck.findGenericMethod(bc.model.templates, e.ctor, e.ctor_is_enum, member)) |m| {
            // An associated (self-less) function is not callable on a value receiver —
            // it must be reached via `Type[args].fn()`.
            if (!m.has_self) {
                for (args) |a| _ = try bc.typeOf(a);
                try bc.sink.emitFmtCode(.T0018, bc.byteOf(callee.main_token), "'{s}' is an associated function; call it as '{s}[..].{s}(..)'", .{ member, bc.typeName(recv_ty), member });
                return .invalid;
            }
            const mf = bc.model.fns[m.fn_id];
            const n_gp: u32 = @intCast(mf.generic_params.len);
            const targs = try bc.gpa.alloc(Type, n_gp);
            defer bc.gpa.free(targs);
            const bnd = try bc.gpa.alloc(bool, n_gp);
            defer bc.gpa.free(bnd);
            const fp = try bc.gpa.alloc(usize, n_gp);
            defer bc.gpa.free(fp);
            const pat: []const Type = if (mf.self_type.isApp()) bc.composite.at(mf.self_type.appIdx()).args else &.{};
            var bound_ok = pat.len == e.args.len;
            if (bound_ok) switch (Infer.match(n_gp, pat, e.args, targs, bnd, fp)) {
                .ok => {},
                else => bound_ok = false, // an unbound impl param: poison, no cascade
            };
            // A `mut self` method mutates the receiver in place, so it may
            // only be called on a mutable place (reuse the mut-self gate + T0019).
            if (m.mut_self and !bc.isMutablePlace(callee.lhs)) {
                try bc.sink.emitFmtCode(.T0019, bc.byteOf(bc.tree.nodes[(callee.lhs).int()].main_token), "cannot call mutating method '{s}' on a temporary; the receiver must be a mutable variable (a local or a field of one)", .{member});
            }
            const self_off: usize = @min(mf.params.len, 1);
            const want = mf.params.len - self_off;
            if (args.len != want) {
                for (args) |a| _ = try bc.typeOf(a);
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ want, args.len });
                if (!bound_ok) return .invalid;
                const ret = substTy(bc, mf.ret, targs);
                bc.node_types[(node_idx).int()] = ret;
                return ret;
            }
            if (!bound_ok) {
                for (args) |a| _ = try bc.typeOfExpected(a, null);
                return .invalid;
            }
            for (args, mf.params[self_off..], 0..) |a, pty, i| {
                const want_ty = substTy(bc, pty, targs);
                const at = try bc.typeOfExpected(a, if (want_ty.kind == .invalid) null else want_ty);
                if (!Type.assignable(want_ty, at)) {
                    try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(a).int()].main_token), "argument {d}: expected {s}, got {s}", .{ i + 1, bc.typeName(want_ty), bc.typeName(at) });
                }
            }
            const ret = substTy(bc, mf.ret, targs);
            bc.node_types[(node_idx).int()] = ret;
            return ret;
        }
        // A compiler-provided inherent method on the prelude `Option`/`Result`
        // enums: recognized by ctor id, NOT a `t.methods` entry (source-
        // less, like the builtin scalar `eq`/`hash`), so `findGenericMethod`
        // misses above. `is_some`/`is_none`/`is_ok`/`is_err` -> bool (arity 0);
        // `unwrap` -> the payload type-arg (arity 0); `unwrap_or` -> the payload
        // type-arg (arity 1, default assignable to it). The receiver is an `App`
        // during Pass C (`e.args[0]` is the payload T); `lower` inlines the tag
        // test / payload load. Gated on `ctor_is_enum` (struct/enum ids share a
        // numeric space). An unknown member falls through to T0018 below.
        if (e.ctor_is_enum) {
            const fam = Typecheck.optResultFamilyOf(bc.model, e.ctor);
            if (Typecheck.optionResultMethod(fam, member)) |op| {
                const t_ty: Type = if (e.args.len >= 1) e.args[0] else .invalid;
                // Predicates are tag-only, so safe for any payload. `unwrap`/
                // `unwrap_or` carry the payload as ONE scalar value through
                // `lower`, so only recognize them for a scalar (int/bool)
                // payload; a str/struct/enum payload falls through to the T0018
                // below — a clean rejection, since aggregate-payload unwrap is
                // deferred (a scalar load would truncate a fat/aggregate
                // value, and an aggregate join block-arg crashes codegen).
                const native_ok = switch (op) {
                    .is_tag0, .is_tag1 => true,
                    // A managed-box payload (`Ref[T]`, an `.app` during Pass C) is an
                    // 8-byte scalar cell pointer, so it rides the scalar unwrap path.
                    .unwrap, .unwrap_or => t_ty.kind != .invalid,
                };
                if (native_ok) switch (op) {
                    .is_tag0, .is_tag1 => {
                        if (args.len != 0) {
                            for (args) |a| _ = try bc.typeOf(a);
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 0), args.len });
                        }
                        bc.node_types[(node_idx).int()] = Type.bool;
                        return Type.bool;
                    },
                    .unwrap => {
                        if (args.len != 0) {
                            for (args) |a| _ = try bc.typeOf(a);
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 0), args.len });
                        }
                        bc.node_types[(node_idx).int()] = t_ty;
                        return t_ty;
                    },
                    .unwrap_or => {
                        if (args.len != 1) {
                            for (args) |a| _ = try bc.typeOfExpected(a, null);
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 1), args.len });
                        } else {
                            const at = try bc.typeOfExpected(args[0], if (t_ty.kind == .invalid) null else t_ty);
                            if (!Type.assignable(t_ty, at))
                                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(args[0]).int()].main_token), "argument 1: expected {s}, got {s}", .{ bc.typeName(t_ty), bc.typeName(at) });
                        }
                        bc.node_types[(node_idx).int()] = t_ty;
                        return t_ty;
                    },
                };
            }
        }
        // A generic-type value with no such method.
        for (args) |a| _ = try bc.typeOf(a);
        try bc.sink.emitFmtCode(.T0018, bc.byteOf(callee.main_token), "no method '{s}' on type '{s}'", .{ member, bc.typeName(recv_ty) });
        return .invalid;
    }

    fn dispatchTypeVarMethod(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node, recv_ty: Type) error{OutOfMemory}!Type {
        // A method call `v.m(..)` on a `type_var` receiver (bound-as-axiom):
        // this only happens while checking a BOUNDED generic template's body.
        // ONLY the methods of the param's declared bound `[T has P]` are
        // callable; resolve `m` against the bound protocol's decoded signature
        // (Self -> the bounded type_var). A non-protocol/unbounded method call
        // is T0018. (In a per-instance re-check the receiver is grounded, so
        // this branch never fires — the struct/scalar branch above does.)
        const member = bc.nameText(callee.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        const ord = recv_ty.typeVarOrd();
        const pid_opt: ?u32 = if (ord < bc.bound_protocols.len) bc.bound_protocols[ord] else null;
        if (pid_opt) |pid| {
            const p = bc.model.protocols[pid];
            // The bound's protocol type-args (`[T has Into[int]]` -> `[int]`)
            // ground the protocol's OWN generic params (`tv(1..)`); `Self`
            // (`tv(0)`) grounds to the bounded `type_var` receiver itself.
            const pargs: []const Type = if (ord < bc.bound_protocol_args.len) bc.bound_protocol_args[ord] else &.{};
            var mi: ?usize = null;
            for (p.methods, 0..) |mn, k| if (std.mem.eql(u8, mn, member)) {
                mi = k;
                break;
            };
            if (mi) |k| {
                const psig = p.method_params[k]; // [self, ...]
                const self_off: usize = @min(psig.len, 1);
                const want = psig.len - self_off;
                if (args.len != want) {
                    for (args) |a| _ = try bc.typeOf(a);
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ want, args.len });
                } else for (args, psig[self_off..], 0..) |a, pty, i| {
                    const wt = Typecheck.groundProtoType(pty, recv_ty, pargs);
                    const at = try bc.typeOfExpected(a, if (wt.kind == .invalid) null else wt);
                    if (!Type.assignable(wt, at))
                        try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(a).int()].main_token), "argument {d}: expected {s}, got {s}", .{ i + 1, bc.typeName(wt), bc.typeName(at) });
                }
                const ret = Typecheck.groundProtoType(p.method_rets[k], recv_ty, pargs);
                bc.node_types[(node_idx).int()] = ret;
                return ret;
            }
            for (args) |a| _ = try bc.typeOf(a);
            try bc.sink.emitFmtCode(.T0018, bc.byteOf(callee.main_token), "no method '{s}' on type parameter bounded by protocol '{s}'", .{ member, p.name });
            return .invalid;
        }
        for (args) |a| _ = try bc.typeOf(a);
        try bc.sink.emitFmtCode(.T0018, bc.byteOf(callee.main_token), "no method '{s}' on unbounded type parameter", .{member});
        return .invalid;
    }

    /// Type an explicit-args generic call `id[A,B](..)`. The callee `type_app`'s base
    /// identifier resolves to the generic template; the explicit type-args resolve to
    /// concrete `Type`s (written into `node_types` so the mono tail can read them back
    /// to discover the instance), the value args are checked against the SUBSTITUTED
    /// param types, and the call node types as the substituted return. Every type this
    /// writes is concrete: a `type_var` never escapes (the args are concrete value
    /// types, and inside an instance re-check `bc.subst` grounds a generic-param
    /// type-arg to concrete before it is stored).
    fn typeOfGenericCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node) error{OutOfMemory}!Type {
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        const base_res = bc.resolutions[(callee.lhs).int()];
        if (base_res != .func) {
            for (args) |arg| _ = try bc.typeOf(arg);
            // The base of a `[..]` callee that is not a fn: nothing else takes type
            // args (no generic types in value position), so it is a plain
            // not-a-function.
            try bc.sink.emit(bc.byteOf(n.main_token), "called value is not a function");
            return .invalid;
        }
        const f = bc.model.fns[base_res.func];
        const targ_nodes = Ast.rangeSlice(bc.tree, (callee.rhs).int());
        if (!f.isGeneric()) {
            for (targ_nodes) |tn| bc.node_types[(tn).int()] = bc.typeFromNode(tn);
            for (args) |arg| _ = try bc.typeOf(arg);
            try bc.sink.emitFmt(bc.byteOf(callee.main_token), "'{s}' is not generic; drop the type arguments", .{bc.nameText(bc.tree.nodes[(callee.lhs).int()].main_token)});
            return f.ret;
        }
        if (targ_nodes.len != f.generic_params.len) {
            for (targ_nodes) |tn| bc.node_types[(tn).int()] = bc.typeFromNode(tn);
            for (args) |arg| _ = try bc.typeOf(arg);
            try bc.sink.emitFmt(bc.byteOf(callee.main_token), "expected {d} type argument(s), got {d}", .{ f.generic_params.len, targ_nodes.len });
            return .invalid;
        }
        const targs = try bc.gpa.alloc(Type, targ_nodes.len);
        defer bc.gpa.free(targs);
        var all_concrete = true;
        for (targ_nodes, 0..) |tn, i| {
            const ty = bc.typeFromNode(tn); // subst-aware inside an instance re-check
            bc.node_types[(tn).int()] = ty;
            targs[i] = ty;
            if (ty.kind == .invalid) {
                all_concrete = false; // an unknown type already reported by typeFromNode
            } else if (!isConcreteValue(ty)) {
                // `type_var`/`unit` are not monomorphizable type-args. A ground `App`
                // (`Box[int]`) now IS — it reifies to a concrete struct in the mono
                // tail — so `isConcreteValue` admits it and it is stored + reified later.
                try bc.sink.emitCode(.T0013, bc.byteOf(bc.tree.nodes[(tn).int()].main_token), "type argument must be a concrete value type");
                all_concrete = false;
            }
        }
        // Check the value args against the substituted params regardless (surface arg
        // errors), but only produce the concrete return type when every type-arg is
        // sound — else poison so no `type_var`/half-substituted type is stored.
        if (args.len != f.params.len) {
            for (args) |arg| _ = try bc.typeOf(arg);
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ f.params.len, args.len });
            if (!all_concrete) return .invalid;
            const ret = substTy(bc, f.ret, targs);
            bc.node_types[(node_idx).int()] = ret;
            return ret;
        }
        if (!all_concrete) {
            for (args) |arg| _ = try bc.typeOfExpected(arg, null);
            return .invalid;
        }
        return bc.applyGenericSig(node_idx, n, f, targs, null);
    }

    /// The shared tail of both generic-call paths (explicit `id[int](7)` and inferred
    /// `id(7)`): check each value arg against the SUBSTITUTED param, type the call node
    /// as the substituted return, and return it. `targs` is the concrete type-arg tuple
    /// (explicit args, or the inferred args). `pretyped` is the inferred path's
    /// already-synthesized arg types — passing them avoids re-walking the args (which
    /// would double-emit inner-arg diagnostics, a diag-count nondeterminism); `null`
    /// re-types each arg in check mode against its substituted param, byte-identical to
    /// the explicit loop.
    fn applyGenericSig(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, f: FnSym, targs: []const Type, pretyped: ?[]const Type) error{OutOfMemory}!Type {
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        for (args, f.params, 0..) |arg, pty, i| {
            const want = substTy(bc, pty, targs);
            const at = if (pretyped) |pt| pt[i] else try bc.typeOfExpected(arg, if (want.kind == .invalid) null else want);
            if (!Type.assignable(want, at)) {
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(arg).int()].main_token), "argument {d}: expected {s}, got {s}", .{ i + 1, bc.typeName(want), bc.typeName(at) });
            }
        }
        const ret = substTy(bc, f.ret, targs);
        bc.node_types[(node_idx).int()] = ret;
        return ret;
    }

    /// Whether `t` conforms to the derivable prelude protocol `pid_opt` — the shared
    /// predicate behind the `==`/`!=` (`Eq`), `<`/`>`/`<=`/`>=` (`Ord`),
    /// `.hash()` (`Hash`), and `print(x)` (`Display`) operator/trigger typings.
    /// A concrete type resolves via the frozen conformance table (`Conform.existence` covers
    /// the int/bool/str/unit prelude conformances AND every user `impl T has P`); a
    /// `type_var` in a bounded generic body conforms as-axiom when its declared bound IS
    /// `pid`. A prelude-less caller (`pid_opt == null`) denies conformance, so the operator
    /// emits its T002x rather than miscompiling.
    /// After those misses, a struct (or enum) whose fields all conform with NO explicit impl
    /// conforms STRUCTURALLY. When `record_derive`, record the derive request (so the serial
    /// synthesis barrier emits the source-less witness) and return true. `Conform.classify`
    /// tries the direct (existence ∨ axiom) tier first, keeping an explicit/refinement type
    /// off the structural path (no double-fire). Fallible (`Conform.structural` + the request
    /// record allocate) and takes `*BodyChecker` (records into the thread-local `derive_reqs`).
    fn conformsTo(bc: *BodyChecker, t: Type, pid_opt: ?u32, record_derive: bool) error{OutOfMemory}!bool {
        const pid = pid_opt orelse return false;
        switch (try Conform.classify(bc.model, bc.composite, bc.bound_protocols, &bc.conforms_memo, bc.gpa, t, pid)) {
            .none => return false,
            .direct => return true,
            .structural => {
                // Record a derive request ONLY for a GROUND operand: an abstract `App`
                // (a `type_var` inside, e.g. `Box[T]` in a bounded template's definition
                // check) is accepted-but-not-recorded — its concrete instance re-check
                // records the ground `Box[int]`, which reify+synthesize can mint a witness for.
                if (record_derive and bc.isGround(t)) try bc.recordDeriveReq(pid, t);
                return true;
            },
        }
    }

    /// Whether `t` is fully ground: no `type_var` anywhere. A struct/enum id and any
    /// scalar are ground; a `type_var` is not; an `App` is ground iff every arg is. Gates
    /// `recordDeriveReq` so the synthesis barrier only ever sees a concrete (reifiable) type.
    fn isGround(bc: *BodyChecker, t: Type) bool {
        if (t.isTypeVar()) return false;
        if (t.isApp()) {
            const e = bc.composite.at(t.appIdx());
            for (e.args) |a| if (!bc.isGround(a)) return false;
            return true;
        }
        return true;
    }

    /// Whether `t` is a managed box (`Ref[T]`/`gc_array[T]`) — a struct-`App` over a
    /// prelude box template — during Pass C (before reify). Such a payload is an 8-byte
    /// scalar cell pointer, so it rides the scalar native-Option/Result unwrap path.
    fn isRefPayload(bc: *BodyChecker, t: Type) bool {
        if (!t.isApp()) return false;
        const e = bc.composite.at(t.appIdx());
        if (e.ctor_is_enum) return false;
        return if (bc.model.prelude) |p| p.refFamily(e.ctor) != .none else false;
    }

    /// Record a structural derive request once per (protocol, type) in this fn (a fn may
    /// `==` a type repeatedly; the synthesis barrier dedups across fns too).
    fn recordDeriveReq(bc: *BodyChecker, pid: u32, t: Type) error{OutOfMemory}!void {
        for (bc.derive_reqs.items) |r| if (r.protocol_id == pid and Type.eql(r.conform_ty, t)) return;
        try bc.derive_reqs.append(bc.gpa, .{ .protocol_id = pid, .conform_ty = t });
    }

    /// The first struct field that blocks a structural derive of `pid_opt` (Eq T0029,
    /// Hash T0030, Display T0031), for the message that names it; null when `t` is
    /// not a struct, has no such protocol, or every field conforms. Struct-only: a payload
    /// enum with a non-conforming payload has no single nameable field, so it falls through
    /// to the generic "does not conform" message instead.
    fn deriveBlocker(bc: *BodyChecker, t: Type, pid_opt: ?u32) error{OutOfMemory}!?Typecheck.NonConformingField {
        if (t.kind != .@"struct") return null;
        const pid = pid_opt orelse return null;
        return Conform.firstNonConformingField(bc.model.structs, bc.model.enums, bc.model.conformances, t, pid, &bc.conforms_memo, bc.gpa, bc.composite, bc.bound_protocols);
    }

    /// Map an arithmetic operator token to its prelude protocol id (from the frozen Model)
    /// + display name. One switch keeps operator → protocol a single source of truth
    /// so the checker and the T0028 message can never drift. A prelude-less caller leaves
    /// `pid` null (denied by `conformsToArith`); the `else` is unreachable for the four
    /// arithmetic tokens this is only called with.
    fn arithProtocol(bc: *const BodyChecker, op: TokenTag) struct { pid: ?u32, name: []const u8 } {
        return switch (op) {
            .plus => .{ .pid = bc.model.preludeProtocols().add, .name = "Add" },
            .minus => .{ .pid = bc.model.preludeProtocols().sub, .name = "Sub" },
            .star => .{ .pid = bc.model.preludeProtocols().mul, .name = "Mul" },
            .slash => .{ .pid = bc.model.preludeProtocols().div, .name = "Div" },
            else => .{ .pid = null, .name = "Add" },
        };
    }

    /// Whether `t` conforms to the arithmetic protocol `pid_opt` — the predicate the
    /// `+`/`-`/`*`/`/` operator typing uses for non-int operands. Calls `Conform.direct`
    /// (existence ∨ axiom) and NOT `Conform.classify`: a concrete type resolves via the
    /// frozen conformance table (`Conform.existence` covers the builtin `int` conformance AND
    /// every user `impl T has Add`); a `type_var` in a bounded generic body conforms as-axiom
    /// when its declared bound IS the operator's protocol. Arithmetic protocols are NOT
    /// structurally derivable, so there is deliberately no structural tier here. A null id
    /// (prelude-less caller) denies conformance, so the operator emits T0028 rather than
    /// miscompiling.
    fn conformsToArith(bc: *const BodyChecker, t: Type, pid_opt: ?u32) bool {
        const pid = pid_opt orelse return false;
        return Conform.direct(bc.model, t, pid, bc.bound_protocols);
    }

    /// The name a T0026/T0027 message uses for a non-conforming operand: a plain
    /// type renders normally, but an `App` (`Box[T]`/`Box[int]`) recurses to the DEEPEST
    /// field/payload that actually fails `pid` and names IT — a `type_var` culprit as its
    /// generic-param name (`T`), so the error points at the real cause instead of the
    /// opaque outer ctor. Best-effort on the already-failed error path.
    fn nonConformingName(bc: *BodyChecker, lt: Type, pid_opt: ?u32) []const u8 {
        const pid = pid_opt orelse return bc.typeName(lt);
        if (lt.kind != .app) return bc.typeName(lt);
        const culprit = bc.deepestNonConforming(lt, pid);
        if (culprit.isTypeVar()) {
            const ord = culprit.typeVarOrd();
            if (ord < bc.gph_generic_params.len) return bc.gph_generic_params[ord];
            return bc.typeName(culprit);
        }
        return bc.typeName(culprit);
    }

    /// Walk `t` to the innermost type that fails to conform to `pid`: an `App`
    /// substitutes its ctor's field/payload patterns and descends into the first that
    /// fails; a struct/enum descends into its first non-conforming field/payload; a leaf
    /// (`type_var`/scalar) is returned as-is. Terminates via the `seen` App-index guard:
    /// the type graph is cyclic (a recursive template re-interns to one App index), so
    /// re-entry on that index is cut. Degrades to `t` on any allocation failure.
    fn deepestNonConforming(bc: *BodyChecker, t: Type, pid: u32) Type {
        var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
        defer seen.deinit(bc.gpa);
        return bc.deepestNonConformingGuarded(t, pid, &seen);
    }

    fn deepestNonConformingGuarded(bc: *BodyChecker, t: Type, pid: u32, seen: *std.AutoHashMapUnmanaged(u32, void)) Type {
        if (t.isApp()) {
            // A self-referential generic (`next: Node[T]`) re-interns to the same App
            // index; a plain non-conforming struct can put the recursive field before
            // the failing one, so guard re-entry to keep the walk terminating.
            const ai = t.appIdx();
            if ((seen.getOrPut(bc.gpa, ai) catch return t).found_existing) return t;
            const e = bc.composite.at(ai);
            if (e.ctor_is_enum) {
                if (e.ctor < bc.model.enums.len) {
                    for (bc.model.enums[e.ctor].variants) |v| for (v.field_types) |ft| {
                        const sub = Conform.substPattern(bc.composite, bc.gpa, ft, e.args) catch return t;
                        if (!bc.conformsQuiet(sub, pid)) return bc.deepestNonConformingGuarded(sub, pid, seen);
                    };
                }
            } else {
                if (e.ctor < bc.model.structs.len) {
                    for (bc.model.structs[e.ctor].field_types) |ft| {
                        const sub = Conform.substPattern(bc.composite, bc.gpa, ft, e.args) catch return t;
                        if (!bc.conformsQuiet(sub, pid)) return bc.deepestNonConformingGuarded(sub, pid, seen);
                    }
                }
            }
            return t;
        }
        if (t.kind == .@"struct" and t.struct_id < bc.model.structs.len) {
            for (bc.model.structs[t.struct_id].field_types) |ft|
                if (!bc.conformsQuiet(ft, pid)) return bc.deepestNonConformingGuarded(ft, pid, seen);
        } else if (t.kind == .@"enum" and t.enum_id < bc.model.enums.len) {
            for (bc.model.enums[t.enum_id].variants) |v| for (v.field_types) |ft|
                if (!bc.conformsQuiet(ft, pid)) return bc.deepestNonConformingGuarded(ft, pid, seen);
        }
        return t;
    }

    fn conformsQuiet(bc: *BodyChecker, t: Type, pid: u32) bool {
        return Conform.structural(bc.model.structs, bc.model.enums, bc.model.conformances, t, pid, &bc.conforms_memo, bc.gpa, bc.composite, bc.bound_protocols) catch true;
    }

    pub fn typeName(bc: *const BodyChecker, ty: Type) []const u8 {
        return refs.typeName(bc, ty);
    }

    pub fn graphCtx(bc: *const BodyChecker) *GraphCtx {
        return bc.model.graph;
    }
    pub fn structSyms(bc: *const BodyChecker) []const StructSym {
        return bc.model.structs;
    }
    pub fn enumSyms(bc: *const BodyChecker) []const EnumSym {
        return bc.model.enums;
    }

    fn slotType(bc: *const BodyChecker, slot: u32) Type {
        return if (slot < bc.slot_types.items.len) bc.slot_types.items[slot] else .invalid;
    }

    pub fn setSlot(bc: *BodyChecker, slot: u32, ty: Type) !void {
        while (bc.slot_types.items.len <= slot) try bc.slot_types.append(bc.gpa, .invalid);
        bc.slot_types.items[slot] = ty;
    }

    pub fn nameText(bc: *const BodyChecker, tok: u32) []const u8 {
        return refs.nameText(bc, tok);
    }

    pub fn byteOf(bc: *const BodyChecker, tok: u32) u32 {
        return refs.byteOf(bc, tok);
    }

    /// Whether `node_idx` names a mutable, addressable place: an identifier bound to
    /// a `.local` (all locals — incl. the `self` receiver — are mutable), or a
    /// `field_access` chain rooted at one. A temporary/rvalue (a construction, a call
    /// result, a literal) is NOT a place. Gates a `mut self` method call (T0019).
    fn isMutablePlace(bc: *const BodyChecker, node_idx: Ast.Index) bool {
        const n = bc.tree.nodes[(node_idx).int()];
        return switch (n.tag) {
            .identifier => bc.resolutions[(node_idx).int()] == .local,
            .field_access, .tuple_field => bc.isMutablePlace(n.lhs),
            else => false,
        };
    }

    fn typeFromNode(bc: *BodyChecker, type_node: Ast.Index) Type {
        return refs.typeFromNode(bc, type_node);
    }

    /// Map a type-ref NAME to its concrete substitution when re-checking a generic
    /// instance (`subst` set by the mono tail), else null. The Pass-C sibling of
    /// `Typecheck.genericParamType`; `refs.typeFromNode` calls whichever the active
    /// checker exposes. No `subst` (every normal body walk) => null => the ordinary
    /// unknown-type path, so non-generic checking is byte-identical.
    pub fn genericParamType(bc: *const BodyChecker, name: []const u8) ?Type {
        const s = bc.subst orelse return null;
        for (s.names, 0..) |gp, i| {
            if (std.mem.eql(u8, gp, name)) return if (i < s.types.len) s.types[i] else Type.invalid;
        }
        return null;
    }

    /// The receiver type when checking an inherent method's body, so a `Self`
    /// type-ref in a body annotation resolves to it via `refs.typeFromNode`. Null
    /// outside a method (byte-identical to non-method checking).
    pub fn selfType(bc: *const BodyChecker) ?Type {
        return bc.cur_self_type;
    }

    /// Intern a composite `App(ctor, args)`. The shared `refs` type-application
    /// resolver calls this via the `anytype` cursor.
    pub fn internApp(bc: *BodyChecker, ctor: u32, args: []const Type, ctor_is_enum: bool) !u32 {
        return bc.composite.intern(bc.gpa, ctor, args, ctor_is_enum);
    }

    fn typeFromQualified(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) Type {
        return refs.typeFromQualified(bc, node_idx, n);
    }

    pub fn activeStructMap(bc: *const BodyChecker) *const std.StringHashMapUnmanaged(u32) {
        return &bc.model.graph.mods[bc.graph_mod].struct_ids;
    }

    pub fn activeEnumMap(bc: *const BodyChecker) *const std.StringHashMapUnmanaged(u32) {
        return &bc.model.graph.mods[bc.graph_mod].enum_ids;
    }

    pub fn activeAliasMap(bc: *const BodyChecker) *const std.StringHashMapUnmanaged(Type) {
        return &bc.model.graph.mods[bc.graph_mod].alias_ids;
    }
};
