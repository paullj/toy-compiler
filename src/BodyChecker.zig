const std = @import("std");
const Ast = @import("ast/Ast.zig");
const Token = @import("ast/Token.zig").Token;
const TokenTag = @import("ast/Token.zig").Tag;
const Resolution = @import("symbols/Resolution.zig").Resolution;
const DiagnosticSink = @import("diagnostics/Sink.zig");
const LayoutEngine = @import("layout/Engine.zig");
const Type = LayoutEngine.Type;
const StructSym = LayoutEngine.StructSym;
const VariantSym = LayoutEngine.VariantSym;
const EnumSym = LayoutEngine.EnumSym;

// Pass A (`Typecheck`) owns the whole-program model + the shared type-reference
// helpers (`refs`) + the frozen `Model`/`FnSym`/`LoopCtx` descriptors; Pass C reads
// them here. The import is mutual (types.zig constructs a `BodyChecker` per fn), which
// Zig resolves lazily — there is no by-value type cycle (`model` is a pointer).
const Typecheck = @import("types.zig");
const Composite = @import("symbols/Composite.zig");
const Infer = @import("symbols/Infer.zig");
const ControlFlow = @import("ControlFlow.zig");
const Model = Typecheck.Model;
const FnSym = Typecheck.FnSym;
const LoopCtx = Typecheck.LoopCtx;
const GraphCtx = Typecheck.GraphCtx;
const refs = Typecheck.refs;

/// The form a construction NODE supplies (independent of the declared variant).
const InitForm = enum { unit, tuple, @"struct" };

/// Substitute a template type through a concrete type-arg tuple (M2/M4): a
/// `type_var(ord)` becomes `targs[ord]`; an `App(ctor, [pat..])` recursively
/// substitutes each arg and re-interns (so a template's field pattern `Box[T]`
/// grounds to `Box[int]`); any concrete type passes through. Needs the intern table,
/// hence the `bc` receiver. The sibling of `Typecheck.substType`.
fn substTy(bc: *BodyChecker, ty: Type, targs: []const Type) Type {
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

/// True when `ty` is a concrete value type usable as a monomorphization type-arg. M4
/// admits a ground `App` (a generic-struct instance used as a type-arg). Mirrors
/// `Typecheck.isConcreteValue`.
fn isConcreteValue(ty: Type) bool {
    return switch (ty.kind) {
        .int, .bool, .str, .@"struct", .@"enum", .app => true,
        else => false,
    };
}

/// Coverage state for a match, by scrutinee kind. Enum: a per-variant seen bitmap.
/// Bool: which of true/false a literal arm has covered. Int: nothing (an infinite
/// domain — exhaustiveness only via `_`).
const Cov = union(enum) {
    @"enum": []bool,
    @"bool": *BoolCov,
    int,
};
const BoolCov = struct { t: bool = false, f: bool = false };

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

    // Local diagnostic sink (merged into the shared result by the Pass-C driver).
    // Its scope is set once in `bodyCheckerFor` to the fn's owning module (graph)
    // or left NO_SCOPE (single-file).
    sink: DiagnosticSink,

    gph_fn_names: ?[]const []const u8 = null,

    /// The shared composite (`App`) intern table (M4), borrowed from the owning
    /// `Typecheck` (stable heap address). Generic-struct construction/field-access
    /// forms + reads back `App`s here; the mono tail reifies them away afterward.
    composite: *Composite,

    /// Per-instance substitution, set ONLY by the monomorphization tail's
    /// per-instance re-check (`Typecheck.recheck`). When set, a generic-parameter
    /// type-ref name inside the body resolves through it (via `genericParamType`) to
    /// the concrete arg; null on every normal Pass-C body walk, which is therefore
    /// byte-identical. Borrowed for the duration of one re-check.
    subst: ?Subst = null,

    /// The receiver type when checking an inherent method's body (M8), set by
    /// `bodyCheckerFor` from the method's `FnSym.self_type`; null for a non-method.
    /// Consumed by the `Self` type-ref hook (`selfType`).
    cur_self_type: ?Type = null,

    /// The per-generic-param bound protocol ids (M13, `[T has P]`), indexed by type-var
    /// ordinal — `bound_protocols[ord]` is the protocol bounding param `ord`, or null if
    /// unbounded. Set by `bodyCheckerFor` from `FnSym.generic_bounds`; empty (inert) for
    /// a non-generic/unbounded fn. Consumed ONLY by the bound-as-axiom `type_var`-receiver
    /// method dispatch in `typeOfCall` (a bounded template's body check); in a per-instance
    /// re-check the receiver is grounded, so that branch never fires.
    bound_protocols: []const ?u32 = &.{},

    /// The per-generic-param bound protocol type-args (M14, `[T has P[args]]`), parallel to
    /// `bound_protocols`. `bound_protocol_args[ord]` grounds the bound protocol's own
    /// type-params in the bound-as-axiom `type_var`-receiver dispatch (so `v.into[int]()`
    /// on a bounded `T` types the protocol method with `U == int`). Empty (inert) for a
    /// non-generic/unbounded fn. Set by `bodyCheckerFor` from `FnSym.generic_bound_args`.
    bound_protocol_args: []const []const Type = &.{},

    /// Structural derive requests this fn's body recorded (M18): a `==`/`!=` on a
    /// derivable type with no impl. Moved out into the `BodyResult` after the walk;
    /// merged fn-id-ordered by `checkBodies`. THREAD-LOCAL (one list per BodyChecker),
    /// so the parallel Pass C never shares mutable derive state.
    derive_reqs: std.ArrayList(Typecheck.DeriveReq) = .empty,

    /// The recursive `conforms` query's memo (M18), keyed by `(protocol, kind, type-id)`.
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

    fn cflow(bc: *const BodyChecker) ControlFlow.Ctx {
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
                    .field_access => try bc.typeOf(stmt.lhs),
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

    fn typeOfExpected(bc: *BodyChecker, node_idx: Ast.Index, exp: ?Type) error{OutOfMemory}!Type {
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

    fn typeOf(bc: *BodyChecker, node_idx: Ast.Index) error{OutOfMemory}!Type {
        if (node_idx == Ast.none) return .invalid; // structural poison: no emit (exempt)
        const n = bc.tree.nodes[(node_idx).int()];
        const ty: Type = switch (n.tag) {
            .literal_number => Type.int,
            .literal_bool => Type.@"bool",
            .literal_string => Type.str,
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
                const operand = try bc.typeOf(n.lhs);
                if (operand.kind == .invalid) break :blk Type.invalid; // poison propagation: no emit (exempt)
                const op = bc.tokens[n.main_token].tag;
                switch (op) {
                    .minus => {
                        if (operand.kind == .int) break :blk Type.int;
                        try bc.sink.emit(bc.byteOf(n.main_token), "operand of '-' must be int");
                    },
                    .bang => {
                        if (operand.kind == .bool) break :blk Type.@"bool";
                        try bc.sink.emit(bc.byteOf(n.main_token), "operand of '!' must be bool");
                    },
                    else => {},
                }
                break :blk Type.invalid;
            },
            .binary => blk: {
                const lt = try bc.typeOf(n.lhs);
                const rt = try bc.typeOf(n.rhs);
                if (lt.kind == .invalid or rt.kind == .invalid) break :blk Type.invalid; // poison propagation: no emit (exempt)
                const op = bc.tokens[n.main_token].tag;
                const op_text = bc.tokens[n.main_token].text(bc.source);
                switch (op) {
                    .plus, .minus, .star, .slash => {
                        // `+`/`-`/`*`/`/` desugar to the arithmetic protocols Add/Sub/Mul/Div
                        // (M17). Builtin `int` stays inline (a single machine add/sub/mul/sdiv,
                        // no witness). Otherwise HOMOGENEOUS like `==`/`<`: require the same
                        // type FIRST, then type to the operand type (`Out = Self`) iff it
                        // conforms to the operator's protocol — a user struct/enum `impl T has
                        // Add`, or a `[T has Add]` bound in a generic body. str/bool have NO
                        // arithmetic impl (str concat allocates, deferred), so they fall to T0028.
                        if (lt.kind == .int and rt.kind == .int) break :blk Type.int;
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
                        // result (M16). HOMOGENEOUS like `==`: require the same type FIRST,
                        // then type to bool iff the operand conforms to `Ord` — a concrete
                        // int/str/bool prelude conformance or user struct/enum `impl T has Ord`,
                        // or a `[T has Ord]` bound in a generic body.
                        if (!Type.eql(lt, rt)) {
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must have the same type", .{op_text});
                        } else if (bc.conformsToOrd(lt)) {
                            break :blk Type.@"bool";
                        } else {
                            try bc.sink.emitFmtCode(.T0027, bc.byteOf(n.main_token), "'{s}' requires an 'Ord' impl for type '{s}'", .{ op_text, bc.typeName(lt) });
                        }
                    },
                    .eq_eq, .bang_eq => {
                        // `==`/`!=` desugar to `Eq::eq` (M15). HOMOGENEOUS: require the same
                        // type FIRST (so a cross-type compare stays "same type" even if both
                        // sides individually conform), then type to bool iff the operand
                        // conforms to `Eq` — a concrete int/bool/str/unit prelude conformance
                        // or user struct/enum impl, or a `[T has Eq]` bound in a generic body.
                        if (!Type.eql(lt, rt)) {
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must have the same type", .{op_text});
                        } else if (try bc.conformsToEq(lt)) {
                            break :blk Type.@"bool";
                        } else if (try bc.eqDeriveBlocker(lt)) |blocker| {
                            // M18: a struct that would derive `Eq` but for one non-conforming
                            // field names that field (T0029). A payload enum / other type
                            // keeps the M15 "no Eq impl" T0026 below.
                            try bc.sink.emitFmtCode(.T0029, bc.byteOf(n.main_token), "cannot derive 'Eq' for '{s}': field '{s}' of type '{s}' does not conform to 'Eq'", .{ bc.typeName(lt), blocker.name, bc.typeName(blocker.ty) });
                        } else {
                            try bc.sink.emitFmtCode(.T0026, bc.byteOf(n.main_token), "'{s}' requires an 'Eq' impl for type '{s}'", .{ op_text, bc.typeName(lt) });
                        }
                    },
                    .amp_amp, .pipe_pipe => {
                        if (lt.kind == .bool and rt.kind == .bool) break :blk Type.@"bool";
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must be bool", .{op_text});
                    },
                    else => {},
                }
                break :blk Type.invalid;
            },
            .call => try bc.typeOfCall(node_idx, n),
            .struct_init => try bc.typeOfStructInit(node_idx, n),
            .field_access => try bc.typeOfFieldAccess(node_idx, n),
            .enum_init_unit, .enum_init_tuple, .enum_init_struct => try bc.typeOfEnumInit(node_idx, n),
            .match_expr => return bc.typeOfMatch(node_idx, n), // sets node_types itself
            .literal_unit => Type.unit,
            .block => try bc.checkBlock(node_idx, true),
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
        // INSTANCE (M4) — a composite `App`. `targs` are the concrete type-args the
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
            // type-params (M5) by matching field VALUE types against the declared field
            // PATTERNS (the M3 matcher over M4's `App`). The inferred `App` is
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
                    // supplier spans — M7's target type never overrides it.
                    .conflict => |c| {
                        const later = bc.tree.nodes[(supplier[c.second_pos]).int()].main_token;
                        const earlier = bc.tree.nodes[(supplier[c.first_pos]).int()].main_token;
                        try bc.sink.emitFmtCodeRelated(.T0015, bc.byteOf(later), bc.byteOf(earlier), "conflicting types for type parameter '{s}': {s} vs {s}", .{ gsym.generic_params[c.ord], bc.typeName(aligned[c.first_pos]), bc.typeName(aligned[c.second_pos]) });
                        return .invalid;
                    },
                    // Target-fill any still-open param from the expected type
                    // (`p: Phantom[int] = Phantom{..}`); an arg-vs-expected disagreement
                    // surfaces as T0015. Byte-identical to pre-M7 when no target exists.
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
    /// plain, explicit `Box[int]{..}`, and the M5-inferred `Box{..}`). When `pretyped`
    /// is non-null it supplies the already-computed value type per init (source order),
    /// so the inferred path — which must type each value ONCE to run inference — does not
    /// re-walk (and thus re-diagnose) the value expressions; `pretyped == null` types
    /// each value here and is byte-identical to the pre-M5 tail.
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
                // `substTy` is the identity and this is byte-identical to pre-M4.
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
        // An explicit generic-enum UNIT-variant `Opt[int].none` (M6): the receiver is a
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
        // A field access over a generic-struct INSTANCE `b.v` where `b: Box[int]` (M4):
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
        // generic-enum instance (lhs is a `type_app`, M6), or inferred (the one-shot
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
            // (M6). `checkVariant` infers the enum's type-params from the payload for the
            // generic case (a nullary `.none` under a generic expected is uninferable ->
            // T0016; target-typing it is M7).
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
    /// to the pre-M6 body). For a generic enum constructed WITHOUT explicit type args
    /// (`Wrap.w(5)`, `Opt.none`), it infers the enum's type-params (M5-style) by matching
    /// the payload VALUE types against the variant's declared payload PATTERNS, then
    /// delegates with the inferred args. A payload that binds only SOME params
    /// (`Either.left(x)`) or a nullary variant (`Opt.none`) is uninferable — routed to
    /// T0016 (the M6/M7 boundary: target-type inference for those is M7). Explicit
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
            // reported at the payload spans — M7's target type never overrides it.
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
    /// (M6-generalized). `targs` are the enum instance's concrete type-args — each
    /// declared payload pattern is substituted through them via `substTy` before the
    /// assignability check (empty for a non-generic enum ⇒ `substTy` is the identity ⇒
    /// byte-identical to the pre-M6 body). Returns the enum-`App` for a generic enum (so
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

    /// The GLOBAL enum id a match scrutinee / variant-pattern expected type refers to:
    /// a plain `enumT` yields its id directly; a generic-enum instance `App`
    /// (`Either[int,bool]`, M6) yields its enum ctor id (coverage/exhaustiveness run over
    /// the TEMPLATE's variants — same names/arity as the reified instance; payload types
    /// are substituted through the App's args at bind sites). A struct-App or scalar is
    /// not an enum here.
    fn scrutEnumId(bc: *const BodyChecker, ty: Type) ?u32 {
        if (ty.kind == .@"enum") return ty.enum_id;
        if (ty.isApp()) {
            const e = bc.composite.at(ty.appIdx());
            if (e.ctor_is_enum) return e.ctor;
        }
        return null;
    }

    /// The target type-args when `bc.expected` is an `App` of exactly THIS ctor with
    /// matching enum/struct-ness (M7 target typing). Anything else — a scalar, the
    /// wrong ctor, a plain `enumT`, or no expected type — yields null, so the
    /// `fillExpected` reconcile is a no-op and behavior is byte-identical to pre-M7.
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

    fn typeOfMatch(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        const st = try bc.typeOf(n.lhs);
        const arms = Ast.rangeSlice(bc.tree, (n.rhs).int());
        if (st.kind == .invalid) {
            // Poison-absorb: still walk arms (bodies may have their own errors) but
            // don't emit a scrutinee or exhaustiveness error.
            for (arms) |arm_idx| _ = try bc.typeOf(Ast.armHeaderAt(bc.tree, (bc.tree.nodes[(arm_idx).int()].rhs).int()).body);
            bc.node_types[(node_idx).int()] = .invalid;
            return .invalid;
        }
        // A generic-enum instance scrutinee is an `App` (reified to `enumT` in the mono
        // tail); its coverage runs over the template enum's variants (M6).
        const enum_id = bc.scrutEnumId(st);
        if (enum_id == null and st.kind != .int and st.kind != .bool) {
            for (arms) |arm_idx| _ = try bc.typeOf(Ast.armHeaderAt(bc.tree, (bc.tree.nodes[(arm_idx).int()].rhs).int()).body);
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "match scrutinee must be an enum, int, or bool, got {s}", .{bc.typeName(st)});
            bc.node_types[(node_idx).int()] = .invalid;
            return .invalid;
        }

        var seen: []bool = &.{};
        var bool_cov: BoolCov = .{};
        var cov: Cov = if (enum_id) |eid| blk: {
            seen = try bc.gpa.alloc(bool, bc.model.enums[eid].variants.len);
            @memset(seen, false);
            break :blk .{ .@"enum" = seen };
        } else switch (st.kind) {
            .bool => .{ .@"bool" = &bool_cov },
            else => .int,
        };
        defer if (enum_id != null) bc.gpa.free(seen);

        var has_wildcard = false;
        var result: Type = Type.never;
        for (arms) |arm_idx| {
            const arm = bc.tree.nodes[(arm_idx).int()];
            const h = Ast.armHeaderAt(bc.tree, (arm.rhs).int());
            const guarded = h.guard != Ast.none;
            try bc.checkPattern(arm.lhs, st, &cov, &has_wildcard, !guarded);
            if (guarded) {
                const gt = try bc.typeOf(h.guard);
                if (gt.kind != .invalid and gt.kind != .bool)
                    try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(h.guard).int()].main_token), "match guard must be bool, got {s}", .{bc.typeName(gt)});
            }
            const body_ty0 = try bc.typeOfExpected(h.body, bc.expected);
            const body_ty: Type = if (bc.armDiverges(h.body)) Type.never else body_ty0;
            result = try bc.merge(n.main_token, result, body_ty);
        }
        if (!has_wildcard) switch (cov) {
            .@"enum" => |sv| {
                const e = bc.model.enums[enum_id.?];
                for (e.variants, 0..) |v, i| {
                    if (!sv[i]) try bc.sink.emitFmt(bc.byteOf(n.main_token), "non-exhaustive match: missing variant '{s}'", .{v.name});
                }
            },
            .bool => |bcov| if (!(bcov.t and bcov.f))
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "non-exhaustive match: bool requires both true and false (or '_')", .{}),
            .int => try bc.sink.emitFmt(bc.byteOf(n.main_token), "non-exhaustive match: int match requires '_'", .{}),
        };
        bc.node_types[(node_idx).int()] = result;
        return result;
    }

    // Pattern irrefutability lives in `ControlFlow.zig` too (pure walks over the
    // tree + enum table); the pattern checker below delegates through these thin
    // wrappers. `variantPayloadIrrefutable` is the one that takes a resolved
    // `VariantSym` directly (the caller already has it).

    fn irrefutable(bc: *const BodyChecker, pat_idx: Ast.Index, ty: Type) bool {
        return ControlFlow.irrefutable(bc.cflow(), pat_idx, ty);
    }

    fn variantPayloadIrrefutable(bc: *const BodyChecker, pat_idx: Ast.Index, variant: VariantSym) bool {
        return ControlFlow.variantPayloadIrrefutable(bc.cflow(), pat_idx, variant);
    }

    fn armDiverges(bc: *const BodyChecker, node_idx: Ast.Index) bool {
        return ControlFlow.armDiverges(bc.cflow(), node_idx);
    }

    fn checkPattern(bc: *BodyChecker, pat_idx: Ast.Index, expected: Type, cov: *Cov, has_wildcard: *bool, count_cov: bool) error{OutOfMemory}!void {
        const pat = bc.tree.nodes[(pat_idx).int()];
        switch (pat.tag) {
            .pattern_wildcard => if (count_cov) {
                has_wildcard.* = true;
            },
            .pattern_binding => {
                // Record the type this binding matched AGAINST on its own node. The
                // binding's slot is SHARED across or-pattern alternatives (Resolve), so
                // the slot type is overwritten and can't reveal a `.A(x) | .B(x)` type
                // divergence; the per-node matched type can (read by `collectBindings`).
                bc.node_types[(pat_idx).int()] = expected;
                if (pat.rhs == Ast.none) {
                    // Bind-whole: type by value; a top-level bare binding is irrefutable.
                    if (bc.resolutions[(pat_idx).int()] == .local) try bc.setSlot(bc.resolutions[(pat_idx).int()].local, expected);
                    if (count_cov) has_wildcard.* = true;
                } else {
                    try bc.checkPattern(pat.rhs, expected, cov, has_wildcard, count_cov);
                }
            },
            .pattern_literal => {
                const lt: Type = if (bc.tokens[pat.main_token].tag == .number) Type.int else Type.@"bool";
                if (expected.kind != .invalid and !Type.eql(lt, expected))
                    try bc.sink.emitFmt(bc.byteOf(pat.main_token), "literal pattern type {s} does not match scrutinee {s}", .{ bc.typeName(lt), bc.typeName(expected) });
                // A bool literal records its case toward coverage; int never covers.
                if (count_cov) switch (cov.*) {
                    .bool => |bcov| {
                        if (lt.kind == .bool) {
                            if (std.mem.eql(u8, bc.nameText(pat.main_token), "true")) bcov.t = true else bcov.f = true;
                        }
                    },
                    else => {},
                };
            },
            .pattern_or => {
                for (Ast.rangeSlice(bc.tree, (pat.lhs).int())) |a| try bc.checkPattern(a, expected, cov, has_wildcard, count_cov);
                try bc.checkOrBindings(pat_idx);
            },
            .pattern_variant => try bc.checkVariantPattern(pat_idx, expected, cov, has_wildcard, count_cov),
            else => {},
        }
    }

    fn checkVariantPattern(bc: *BodyChecker, pat_idx: Ast.Index, expected: Type, cov: *Cov, has_wildcard: *bool, count_cov: bool) error{OutOfMemory}!void {
        const pat = bc.tree.nodes[(pat_idx).int()];
        // A generic-enum instance scrutinee is an `App` (M6); its enum id + instance
        // type-args come off the composite entry. A plain `enumT` has no type-args, so
        // `substTy(..., &.{})` below is the identity — byte-identical to pre-M6.
        const enum_id = bc.scrutEnumId(expected) orelse {
            if (expected.kind != .invalid)
                try bc.sink.emitFmt(bc.byteOf(pat.main_token), "variant pattern on a non-enum scrutinee {s}", .{bc.typeName(expected)});
            return;
        };
        const targs: []const Type = if (expected.isApp()) bc.composite.at(expected.appIdx()).args else &.{};
        const e = bc.model.enums[enum_id];
        // A qualified `N.V` pattern: the type-name must name the scrutinee enum.
        if (pat.lhs != Ast.none) {
            const tname = bc.nameText(bc.tree.nodes[(pat.lhs).int()].main_token);
            if (bc.activeEnumMap().get(tname)) |qid| {
                if (qid != enum_id)
                    try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(pat.lhs).int()].main_token), "pattern enum '{s}' does not match scrutinee '{s}'", .{ tname, e.name });
            } else {
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(pat.lhs).int()].main_token), "'{s}' is not an enum type", .{tname});
            }
        }
        const vname = bc.nameText(pat.main_token);
        var vi: ?usize = null;
        for (e.variants, 0..) |v, i| {
            if (std.mem.eql(u8, v.name, vname)) {
                vi = i;
                break;
            }
        }
        const variant = if (vi) |i| blk: {
            // Cover variant i only when this arm counts AND variant i's payload is fully
            // matched (`.V(_)`/`.V(x)` cover it; `.V(0)` or `.V(.W(x))` over a multi-variant
            // inner enum do not — caught by the type-aware payload check). For a generic
            // instance the payload types must be SUBSTITUTED through `targs` first so the
            // irrefutability decision sees concrete types (a bind/wildcard is irrefutable
            // regardless, so `Either[int,bool]`'s `.left(n)`/`.right(_)` cover correctly).
            if (count_cov and try bc.variantPayloadIrrefutableSubst(pat_idx, e.variants[i], targs)) cov.@"enum"[i] = true;
            break :blk e.variants[i];
        } else {
            try bc.sink.emitFmt(bc.byteOf(pat.main_token), "enum '{s}' has no variant '{s}'", .{ e.name, vname });
            return;
        };
        const binders = if (pat.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (pat.rhs).int());
        switch (variant.form) {
            .unit => {
                if (binders.len != 0)
                    try bc.sink.emitFmt(bc.byteOf(pat.main_token), "unit variant '{s}.{s}' binds no payload", .{ e.name, vname });
            },
            .tuple => {
                if (binders.len != variant.field_types.len) {
                    try bc.sink.emitFmt(bc.byteOf(pat.main_token), "variant '{s}.{s}' binds {d} value(s), got {d}", .{ e.name, vname, variant.field_types.len, binders.len });
                    return;
                }
                for (binders, variant.field_types) |b_idx, fty_pat| {
                    try bc.checkPattern(b_idx, substTy(bc, fty_pat, targs), cov, has_wildcard, false);
                }
            },
            .@"struct" => {
                for (binders) |b_idx| {
                    const b = bc.tree.nodes[(b_idx).int()];
                    // A struct binding's SOURCE field name is the rename source (lhs),
                    // or the bound name itself when punning.
                    const src_name = if (b.lhs != Ast.none) bc.nameText(bc.tree.nodes[(b.lhs).int()].main_token) else bc.nameText(b.main_token);
                    var fty: Type = .invalid;
                    var found = false;
                    for (variant.field_names, 0..) |dn, j| {
                        if (std.mem.eql(u8, dn, src_name)) {
                            fty = substTy(bc, variant.field_types[j], targs);
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        try bc.sink.emitFmt(bc.byteOf(b.main_token), "no field '{s}' in '{s}.{s}'", .{ src_name, e.name, vname });
                    }
                    // The carrier IS a pattern_binding: if it has a sub-pattern, match
                    // the field against it; else bind the whole field by value.
                    try bc.checkPattern(b_idx, fty, cov, has_wildcard, false);
                }
            },
        }
    }

    /// App-aware `irrefutable` (M7). A non-`App` type delegates verbatim to
    /// `bc.irrefutable` (= `ControlFlow.irrefutable`), so every scalar / plain-`enumT` /
    /// `structT` payload is byte-identical to pre-M7. For an `App` payload (a substituted
    /// generic-enum/struct instance — the case `ControlFlow` cannot resolve because its
    /// `Ctx` has no composite table) we handle the pattern here: a nested variant/or
    /// pattern resolves the enum via `scrutEnumId` + the App's own type-args and recurses.
    /// Terminates: it descends the FINITE sub-pattern tree, not the (possibly cyclic) type.
    fn irrefutableTy(bc: *BodyChecker, pat_idx: Ast.Index, ty: Type) error{OutOfMemory}!bool {
        if (!ty.isApp()) return bc.irrefutable(pat_idx, ty);
        const pat = bc.tree.nodes[(pat_idx).int()];
        return switch (pat.tag) {
            .pattern_wildcard => true,
            .pattern_binding => pat.rhs == Ast.none or try bc.irrefutableTy(pat.rhs, ty),
            .pattern_literal => false,
            .pattern_or => try bc.orCoversTyApp(pat_idx, ty),
            .pattern_variant => blk: {
                // A single-variant enum instance is the only variant pattern that can be
                // total (the tag test cannot fail); its payload must then be irrefutable.
                const eid = bc.scrutEnumId(ty) orelse break :blk false;
                const e = bc.model.enums[eid];
                if (e.variants.len != 1) break :blk false;
                break :blk try bc.variantPayloadIrrefutableSubst(pat_idx, e.variants[0], bc.composite.at(ty.appIdx()).args);
            },
            else => false,
        };
    }

    /// `ControlFlow.orCoversType` mirrored, App-aware (payloads substituted through the
    /// instance's `targs` before the per-variant irrefutability check).
    fn orCoversTyApp(bc: *BodyChecker, or_idx: Ast.Index, ty: Type) error{OutOfMemory}!bool {
        const alts = Ast.rangeSlice(bc.tree, (bc.tree.nodes[(or_idx).int()].lhs).int());
        for (alts) |a| if (try bc.irrefutableTy(a, ty)) return true;
        const eid = bc.scrutEnumId(ty) orelse return false;
        const e = bc.model.enums[eid];
        const targs = bc.composite.at(ty.appIdx()).args;
        var seen = [_]bool{false} ** 64; // matches ControlFlow's cap
        if (e.variants.len > seen.len) return false;
        for (alts) |a| {
            const ap = bc.tree.nodes[(a).int()];
            if (ap.tag != .pattern_variant) continue;
            const vname = bc.nameText(ap.main_token);
            for (e.variants, 0..) |v, i| {
                if (std.mem.eql(u8, v.name, vname) and try bc.variantPayloadIrrefutableSubst(a, v, targs)) seen[i] = true;
            }
        }
        for (e.variants, 0..) |_, i| if (!seen[i]) return false;
        return true;
    }

    /// `variantPayloadIrrefutable` over a variant whose payload types have been
    /// SUBSTITUTED through the enum instance's `targs` (M6). For `targs.len == 0` (a
    /// non-generic / plain-`enumT` scrutinee) this delegates to the borrowed-variant
    /// call, byte-identical to pre-M6. For a generic instance each binder is checked via
    /// the App-aware `irrefutableTy` against its substituted field type (M7) — this is
    /// what makes a nested variant/or pattern over a substituted generic-enum-instance
    /// (`App`) payload compute correctly rather than always-refutable.
    fn variantPayloadIrrefutableSubst(bc: *BodyChecker, pat_idx: Ast.Index, variant: VariantSym, targs: []const Type) error{OutOfMemory}!bool {
        if (targs.len == 0) return bc.variantPayloadIrrefutable(pat_idx, variant);
        const pat = bc.tree.nodes[(pat_idx).int()];
        const binders = if (pat.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (pat.rhs).int());
        switch (variant.form) {
            .unit => return binders.len == 0,
            .tuple => {
                if (binders.len != variant.field_types.len) return false;
                for (binders, variant.field_types) |b, fpat| {
                    if (!try bc.irrefutableTy(b, substTy(bc, fpat, targs))) return false;
                }
                return true;
            },
            .@"struct" => {
                for (binders) |b_idx| {
                    const b = bc.tree.nodes[(b_idx).int()];
                    const src = if (b.lhs != Ast.none) bc.nameText(bc.tree.nodes[(b.lhs).int()].main_token) else bc.nameText(b.main_token);
                    var fty: Type = .invalid;
                    for (variant.field_names, 0..) |dn, j| if (std.mem.eql(u8, dn, src)) {
                        fty = substTy(bc, variant.field_types[j], targs);
                        break;
                    };
                    if (!try bc.irrefutableTy(b_idx, fty)) return false;
                }
                return true;
            },
        }
    }

    fn checkOrBindings(bc: *BodyChecker, or_idx: Ast.Index) error{OutOfMemory}!void {
        const alts = Ast.rangeSlice(bc.tree, (bc.tree.nodes[(or_idx).int()].lhs).int());
        if (alts.len < 2) return;
        var first_map: std.StringHashMapUnmanaged(Type) = .empty;
        defer first_map.deinit(bc.gpa);
        try bc.collectBindings(alts[0], &first_map);
        var ok = true;
        for (alts[1..]) |alt| {
            var m: std.StringHashMapUnmanaged(Type) = .empty;
            defer m.deinit(bc.gpa);
            try bc.collectBindings(alt, &m);
            if (m.count() != first_map.count()) {
                ok = false;
            } else {
                var it = m.iterator();
                while (it.next()) |entry| {
                    const want = first_map.get(entry.key_ptr.*) orelse {
                        ok = false;
                        break;
                    };
                    if (!Type.eql(want, entry.value_ptr.*)) {
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok) break;
        }
        if (!ok)
            try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(or_idx).int()].main_token), "or-pattern alternatives must bind the same names and types", .{});
    }

    fn collectBindings(bc: *BodyChecker, pat_idx: Ast.Index, out: *std.StringHashMapUnmanaged(Type)) error{OutOfMemory}!void {
        const pat = bc.tree.nodes[(pat_idx).int()];
        switch (pat.tag) {
            .pattern_binding => {
                const name = bc.nameText(pat.main_token);
                // The PER-NODE matched type (set in checkPattern), NOT the shared slot
                // type — so two alternatives binding the same name at different field
                // types are seen as different and rejected.
                const ty: Type = bc.node_types[(pat_idx).int()];
                try out.put(bc.gpa, name, ty);
                if (pat.rhs != Ast.none) try bc.collectBindings(pat.rhs, out);
            },
            .pattern_variant => if (pat.rhs != Ast.none)
                for (Ast.rangeSlice(bc.tree, (pat.rhs).int())) |c| try bc.collectBindings(c, out),
            .pattern_or => for (Ast.rangeSlice(bc.tree, (pat.lhs).int())) |a| try bc.collectBindings(a, out),
            else => {},
        }
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

    fn merge(bc: *BodyChecker, at: u32, a: Type, b: Type) error{OutOfMemory}!Type {
        if (a.kind == .invalid or b.kind == .invalid) return .invalid; // poison absorbs
        if (a.kind == .never) return b; // covers never+never → never
        if (b.kind == .never) return a;
        if (Type.eql(a, b)) return a; // agreement → one phi type
        try bc.sink.emitFmt(bc.byteOf(at), "branches yield different types ({s} vs {s})", .{ bc.typeName(a), bc.typeName(b) });
        return .invalid; // mismatch, no coercion
    }

    /// Type-check a resolved method call `recv.m(args)` against the selected witness `m`
    /// (M8+): the `mut self` place gate, arity, and per-arg assignability, typing the call
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
        // params[0] is the synthesized `self`; value args match params[1..].
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

    /// Emit the T0025 ambiguity diagnostic (M14): the receiver conforms to one generic
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

    /// Type an explicit-protocol-args method call `v.m[int](args)` (M14): the callee is a
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
        if (recv_ty.kind == .@"struct" or recv_ty.kind == .@"enum" or
            recv_ty.kind == .int or recv_ty.kind == .bool or recv_ty.kind == .str or recv_ty.kind == .unit)
        {
            switch (Typecheck.resolveConformanceMethod(bc.model.methods, recv_ty, member, explicit)) {
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

    fn typeOfCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        // A qualified tuple-variant construction `N.V(args)` arrives as a `.call`
        // whose callee is a `field_access` over an enum type-name identifier. Route
        // it to the enum-init checker (treating `n` as a tuple construction).
        const callee = bc.tree.nodes[(n.lhs).int()];
        if (callee.tag == .field_access) {
            const recv = bc.tree.nodes[(callee.lhs).int()];
            if (recv.tag == .identifier and bc.activeEnumMap().get(bc.nameText(recv.main_token)) != null) {
                return bc.typeOfEnumInitQualified(node_idx, .tuple, callee.lhs, callee.main_token, n.rhs);
            }
            // An explicit generic-enum tuple-variant construction `Either[int,bool].left(42)`
            // (M6): the callee field_access's receiver is a `type_app` resolving to an
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
                    return .invalid;
                }
                // A struct-`App` receiver (`Box[int].m(..)`) is a method call — M8+; fall
                // through to the not-a-function path below.
            }
            // A cross-module tuple-variant `mod.Enum.Variant(args)` (graph mode): the
            // callee field_access's receiver is the inner `mod.Enum`.
            if (bc.qualifiedEnumId(callee.lhs)) |enum_id| {
                const ty = try bc.checkVariant(enum_id, callee.main_token, .tuple, n.rhs);
                bc.node_types[(node_idx).int()] = ty;
                return ty;
            }
            // A method call `recv.m(args)` on a VALUE receiver (M8). The enum-variant /
            // qualified-call cases above fire only for an enum type-name / type_app /
            // qualified-namespace receiver; a field_access callee whose receiver is a
            // plain VALUE of a concrete struct/enum is method dispatch. A qualified fn
            // `mod.f()` binds the field_access itself to `.func` (handled below), so
            // guard on that; a namespace receiver is `.module` (also skipped).
            const recv_is_func = bc.resolutions[(n.lhs).int()] == .func;
            const recv_res = bc.resolutions[(callee.lhs).int()];
            if (!recv_is_func and recv_res != .module) {
                const recv_ty = try bc.typeOf(callee.lhs); // also populates node_types[recv] for lower
                if (recv_ty.kind == .invalid) return .invalid; // receiver already errored → no cascade
                if (recv_ty.kind == .@"struct" or recv_ty.kind == .@"enum" or
                    recv_ty.kind == .int or recv_ty.kind == .bool or recv_ty.kind == .str or recv_ty.kind == .unit)
                {
                    const member = bc.nameText(callee.main_token);
                    const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
                    // M14: resolve the witness via the shared multi-conformance resolver
                    // (the SAME rule lower + AstWalk use). `.one` is byte-identical to the
                    // pre-M14 `findMethod` for every existing program (inherent / single
                    // conformance / prelude); `.ambiguous` (a doubly-conforming generic
                    // protocol used with no type-args) is T0025 — never an arbitrary pick.
                    switch (Typecheck.resolveConformanceMethod(bc.model.methods, recv_ty, member, null)) {
                        .one => |m| return try bc.dispatchMethod(node_idx, n, callee, recv_ty, member, m),
                        .ambiguous => {
                            for (args) |a| _ = try bc.typeOf(a);
                            try bc.emitAmbiguousConformance(callee.main_token, recv_ty, member);
                            return .invalid;
                        },
                        .none => {},
                    }
                    // A builtin scalar protocol method (M12): `n.eq(m)` on int/bool. Not a
                    // `t.methods` entry (the recognizer is pure), so `findMethod` misses;
                    // recognize it here, check arity==1 + the arg is assignable to `Self`
                    // (the homogeneous `Eq` receiver), and type the call to the method's
                    // return. A user `impl int has P` was already handled above (its real
                    // `fn_id` is in the method table), so this only fires for the builtins.
                    if (Typecheck.builtinScalarMethod(recv_ty, member)) |bm| {
                        if (args.len != 1) {
                            for (args) |a| _ = try bc.typeOf(a);
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ @as(usize, 1), args.len });
                        } else {
                            const at = try bc.typeOfExpected(args[0], recv_ty);
                            if (!Type.assignable(recv_ty, at))
                                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(args[0]).int()].main_token), "argument 1: expected {s}, got {s}", .{ bc.typeName(recv_ty), bc.typeName(at) });
                        }
                        bc.node_types[(node_idx).int()] = bm.ret;
                        return bm.ret;
                    }
                    // A concrete struct/enum/scalar value with no such method (M8/M12).
                    for (args) |a| _ = try bc.typeOf(a);
                    try bc.sink.emitFmtCode(.T0018, bc.byteOf(callee.main_token), "no method '{s}' on type '{s}'", .{ member, bc.typeName(recv_ty) });
                    return .invalid;
                } else if (recv_ty.kind == .app) {
                    // A method call on a generic-type instance `b.get()` (M10): the
                    // receiver is an `App` (`Box[int]`; reify runs later in the mono
                    // tail). Its ctor selects the `impl Box[T]` method TEMPLATE and the
                    // impl's type-params bind by matching the template's `Self`
                    // pattern-args against the receiver App's concrete args — the SAME
                    // `Infer.match` scanCalls/lower run. Only the impl's params bind; a
                    // method's OWN `[U]` generics are NOT inferred (deferred).
                    const member = bc.nameText(callee.main_token);
                    const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
                    const e = bc.composite.at(recv_ty.appIdx());
                    if (Typecheck.findGenericMethod(bc.model.methods, e.ctor, e.ctor_is_enum, member)) |m| {
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
                        // only be called on a mutable place (reuse the M9 gate + T0019).
                        if (m.mut_self and !bc.isMutablePlace(callee.lhs)) {
                            try bc.sink.emitFmtCode(.T0019, bc.byteOf(bc.tree.nodes[(callee.lhs).int()].main_token), "cannot call mutating method '{s}' on a temporary; the receiver must be a mutable variable (a local or a field of one)", .{member});
                        }
                        // params[0] is the synthesized `self`; value args match params[1..].
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
                    // A generic-type value with no such method (M10).
                    for (args) |a| _ = try bc.typeOf(a);
                    try bc.sink.emitFmtCode(.T0018, bc.byteOf(callee.main_token), "no method '{s}' on type '{s}'", .{ member, bc.typeName(recv_ty) });
                    return .invalid;
                } else if (recv_ty.isTypeVar()) {
                    // A method call `v.m(..)` on a `type_var` receiver (M13 bound-as-axiom):
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
                        // M14: the bound's protocol type-args (`[T has Into[int]]` -> `[int]`)
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
            }
        }
        // An explicit-protocol-args method call `v.into[int](..)` (M14): the callee is a
        // `type_app` OVER a `field_access` NOT bound to a `.func` (a qualified generic fn
        // `mod.f[int]()` binds its field_access to `.func` and stays a generic FUNCTION
        // call). Intercept BEFORE the generic-function `type_app` path.
        if (callee.tag == .type_app and bc.tree.nodes[(callee.lhs).int()].tag == .field_access and
            bc.resolutions[(callee.lhs).int()] != .func)
            return bc.typeOfExplicitMethodCall(node_idx, n, callee);
        // An explicit-args generic call `id[int](7)` (M2): the callee is a `type_app`
        // whose base identifier carries the template's `.func`. A pure per-call
        // operation (substitute the params/ret through the explicit args, check the
        // value args) that writes only concrete types into `node_types`.
        if (callee.tag == .type_app) return bc.typeOfGenericCall(node_idx, n, callee);
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
        // A bare (no-explicit-args) generic call `id(7)` (M3): infer each type-arg by
        // one-sided structural matching of the template's param patterns against the
        // ground argument types, then reuse `applyGenericSig` to check args + type the
        // node. The matcher runs and is discarded here — no `type_var` is ever stored.
        // Explicit `id[int](7)` is handled above (typeOfGenericCall) and still overrides.
        if (f.isGeneric()) {
            // M3 infers type-args only for a PLAIN-IDENTIFIER callee `id(7)`. A bare
            // qualified generic call `mod.id(7)` (field_access callee) is NOT inferred:
            // the three post-typecheck consumers (scanCalls/lower/CallVisitor) key the
            // bare path on a plain identifier too, so accepting it here would type the
            // node concretely but mint NO instance (a `Mono.find` miss in lower). Require
            // explicit type args instead — the same clean reject as pre-M3.
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
            // args in M2 (no generic types in value position), so it is a plain
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
                // (`Box[int]`) now IS (M4) — it reifies to a concrete struct in the mono
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
    /// (explicit args, or the M3-inferred args). `pretyped` is the inferred path's
    /// already-synthesized arg types — passing them avoids re-walking the args (which
    /// would double-emit inner-arg diagnostics, a diag-count nondeterminism); `null`
    /// re-types each arg in check mode against its substituted param, byte-identical to
    /// the M2 explicit loop.
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

    /// Whether `t` conforms to the prelude `Eq` protocol (M15) — the predicate the
    /// `==`/`!=` operator typing uses. A concrete type resolves via the frozen
    /// conformance table (`findConformance` covers the int/bool/str/unit prelude
    /// conformances AND every user `impl T has Eq`); a `type_var` in a bounded generic
    /// body conforms as-axiom when its declared bound IS `Eq`. A prelude-less caller
    /// (`eq_protocol_id == null`) denies conformance, so the operator emits T0026 rather
    /// than miscompiling.
    /// M18: after the `findConformance`/bound-axiom misses, an all-`Eq`-fields struct (or
    /// empty-payload enum) with NO explicit impl conforms STRUCTURALLY. Record the derive
    /// request (so the serial synthesis barrier emits the source-less unit) and return
    /// true so `==`/`!=` types bool. `findConformance`-first keeps an explicit/Ord-
    /// refinement type off this path (no double-fire). Now fallible (`conforms` + the
    /// request record allocate) and takes `*BodyChecker` (records into the thread-local
    /// `derive_reqs`).
    fn conformsToEq(bc: *BodyChecker, t: Type) error{OutOfMemory}!bool {
        const eq_pid = bc.model.eq_protocol_id orelse return false;
        if (Typecheck.findConformance(bc.model, eq_pid, t, &.{})) return true;
        if (t.isTypeVar()) {
            const ord = t.typeVarOrd();
            if (ord >= bc.bound_protocols.len) return false;
            return (bc.bound_protocols[ord] orelse return false) == eq_pid;
        }
        switch (t.kind) {
            .@"struct", .@"enum" => {},
            else => return false,
        }
        if (try Typecheck.conforms(bc.model.structs, bc.model.enums, bc.model.conformances, t, eq_pid, &bc.conforms_memo, bc.gpa)) {
            try bc.recordDeriveReq(eq_pid, t);
            return true;
        }
        return false;
    }

    /// Record a structural derive request once per (protocol, type) in this fn (a fn may
    /// `==` a type repeatedly; the synthesis barrier dedups across fns too).
    fn recordDeriveReq(bc: *BodyChecker, pid: u32, t: Type) error{OutOfMemory}!void {
        for (bc.derive_reqs.items) |r| if (r.protocol_id == pid and Type.eql(r.conform_ty, t)) return;
        try bc.derive_reqs.append(bc.gpa, .{ .protocol_id = pid, .conform_ty = t });
    }

    /// The first struct field that blocks a structural `Eq` derive (M18), for the T0029
    /// message; null when `t` is not a struct, has no eq protocol, or every field conforms.
    fn eqDeriveBlocker(bc: *BodyChecker, t: Type) error{OutOfMemory}!?Typecheck.NonConformingField {
        if (t.kind != .@"struct") return null;
        const eq_pid = bc.model.eq_protocol_id orelse return null;
        return Typecheck.firstNonConformingField(bc.model.structs, bc.model.enums, bc.model.conformances, t, eq_pid, &bc.conforms_memo, bc.gpa);
    }

    /// Whether `t` conforms to the prelude `Ord` protocol (M16) — the predicate the
    /// `<`/`>`/`<=`/`>=` operator typing uses. Mirrors `conformsToEq`: a concrete type
    /// resolves via the frozen conformance table (`findConformance` covers the int/str/bool
    /// prelude conformances AND every user `impl T has Ord`); a `type_var` in a bounded
    /// generic body conforms as-axiom when its declared bound IS `Ord`. A prelude-less caller
    /// (`ord_protocol_id == null`) denies conformance, so the operator emits T0027.
    fn conformsToOrd(bc: *const BodyChecker, t: Type) bool {
        const ord_pid = bc.model.ord_protocol_id orelse return false;
        if (Typecheck.findConformance(bc.model, ord_pid, t, &.{})) return true;
        if (t.isTypeVar()) {
            const ord = t.typeVarOrd();
            if (ord >= bc.bound_protocols.len) return false;
            return (bc.bound_protocols[ord] orelse return false) == ord_pid;
        }
        return false;
    }

    /// Map an arithmetic operator token to its prelude protocol id (from the frozen Model)
    /// + display name (M17). One switch keeps operator → protocol a single source of truth
    /// so the checker and the T0028 message can never drift. A prelude-less caller leaves
    /// `pid` null (denied by `conformsToArith`); the `else` is unreachable for the four
    /// arithmetic tokens this is only called with.
    fn arithProtocol(bc: *const BodyChecker, op: TokenTag) struct { pid: ?u32, name: []const u8 } {
        return switch (op) {
            .plus => .{ .pid = bc.model.add_protocol_id, .name = "Add" },
            .minus => .{ .pid = bc.model.sub_protocol_id, .name = "Sub" },
            .star => .{ .pid = bc.model.mul_protocol_id, .name = "Mul" },
            .slash => .{ .pid = bc.model.div_protocol_id, .name = "Div" },
            else => .{ .pid = null, .name = "Add" },
        };
    }

    /// Whether `t` conforms to the arithmetic protocol `pid_opt` (M17) — the predicate the
    /// `+`/`-`/`*`/`/` operator typing uses for non-int operands. Mirrors `conformsToEq`/
    /// `conformsToOrd`: a concrete type resolves via the frozen conformance table
    /// (`findConformance` covers the builtin `int` conformance AND every user `impl T has
    /// Add`); a `type_var` in a bounded generic body conforms as-axiom when its declared
    /// bound IS the operator's protocol. A null id (prelude-less caller) denies conformance,
    /// so the operator emits T0028 rather than miscompiling.
    fn conformsToArith(bc: *const BodyChecker, t: Type, pid_opt: ?u32) bool {
        const pid = pid_opt orelse return false;
        if (Typecheck.findConformance(bc.model, pid, t, &.{})) return true;
        if (t.isTypeVar()) {
            const ord = t.typeVarOrd();
            if (ord >= bc.bound_protocols.len) return false;
            return (bc.bound_protocols[ord] orelse return false) == pid;
        }
        return false;
    }

    fn typeName(bc: *const BodyChecker, ty: Type) []const u8 {
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

    fn setSlot(bc: *BodyChecker, slot: u32, ty: Type) !void {
        while (bc.slot_types.items.len <= slot) try bc.slot_types.append(bc.gpa, .invalid);
        bc.slot_types.items[slot] = ty;
    }

    fn nameText(bc: *const BodyChecker, tok: u32) []const u8 {
        return refs.nameText(bc, tok);
    }

    fn byteOf(bc: *const BodyChecker, tok: u32) u32 {
        return refs.byteOf(bc, tok);
    }

    /// Whether `node_idx` names a mutable, addressable place: an identifier bound to
    /// a `.local` (all locals — incl. the `self` receiver — are mutable), or a
    /// `field_access` chain rooted at one. A temporary/rvalue (a construction, a call
    /// result, a literal) is NOT a place. Gates a `mut self` method call (M9, T0019).
    fn isMutablePlace(bc: *const BodyChecker, node_idx: Ast.Index) bool {
        const n = bc.tree.nodes[(node_idx).int()];
        return switch (n.tag) {
            .identifier => bc.resolutions[(node_idx).int()] == .local,
            .field_access => bc.isMutablePlace(n.lhs),
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

    /// The receiver type when checking an inherent method's body (M8), so a `Self`
    /// type-ref in a body annotation resolves to it via `refs.typeFromNode`. Null
    /// outside a method (byte-identical to pre-M8 non-method checking).
    pub fn selfType(bc: *const BodyChecker) ?Type {
        return bc.cur_self_type;
    }

    /// Intern a composite `App(ctor, args)` (M4). The shared `refs` type-application
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
};
