const std = @import("std");
const Ast = @import("ast/Ast.zig");
const Token = @import("ast/Token.zig").Token;
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
const ControlFlow = @import("ControlFlow.zig");
const Model = Typecheck.Model;
const FnSym = Typecheck.FnSym;
const LoopCtx = Typecheck.LoopCtx;
const GraphCtx = Typecheck.GraphCtx;
const refs = Typecheck.refs;

/// The form a construction NODE supplies (independent of the declared variant).
const InitForm = enum { unit, tuple, @"struct" };

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

    pub fn deinit(bc: *BodyChecker) void {
        bc.slot_types.deinit(bc.gpa);
        bc.loop_stack.deinit(bc.gpa);
        bc.sink.deinit();
    }

    // ---- BodyChecker methods (the per-fn body-walk relations) --------------

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
                        if (lt.kind == .int and rt.kind == .int) break :blk Type.int;
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must be int", .{op_text});
                    },
                    .lt, .lt_eq, .gt, .gt_eq => {
                        if (lt.kind == .int and rt.kind == .int) break :blk Type.@"bool";
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must be int", .{op_text});
                    },
                    .eq_eq, .bang_eq => {
                        if (Type.eql(lt, rt) and (lt.kind == .int or lt.kind == .bool)) break :blk Type.@"bool";
                        if (Type.eql(lt, rt) and lt.kind == .str) {
                            // Same type, but str comparison isn't supported — say so,
                            // rather than the misleading "must have the same type".
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "'{s}' on str is unsupported", .{op_text});
                        } else {
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must have the same type", .{op_text});
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
        // A qualified struct-variant `N.V { ... }` would arrive as enum_init_struct
        // (the parser upgrades a `field_access {`), not here — so struct_init's lhs is
        // always a plain type-name identifier; no enum routing needed.
        _ = node_idx;
        const name = bc.nameText(bc.tree.nodes[(n.lhs).int()].main_token);
        const id = bc.activeStructMap().get(name) orelse {
            for (Ast.rangeSlice(bc.tree, (n.rhs).int())) |fi| _ = try bc.typeOf(bc.tree.nodes[(fi).int()].lhs);
            try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(n.lhs).int()].main_token), "unknown struct type '{s}'", .{name});
            return .invalid;
        };
        const sym = bc.model.structs[id];
        const inits = Ast.rangeSlice(bc.tree, (n.rhs).int());
    
        // Track which declared fields are supplied (for missing/duplicate checks).
        var seen = try bc.gpa.alloc(bool, sym.field_names.len);
        defer bc.gpa.free(seen);
        @memset(seen, false);
    
        for (inits) |fi_idx| {
            const fi = bc.tree.nodes[(fi_idx).int()];
            const fname = bc.nameText(fi.main_token);
            const vt = try bc.typeOf(fi.lhs);
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
                    try bc.sink.emitFmt(bc.byteOf(fi.main_token), "duplicate field '{s}' in '{s}'", .{ fname, name });
                }
                seen[j] = true;
                const fty = sym.field_types[j];
                if (!Type.assignable(fty, vt)) {
                    try bc.sink.emitFmt(bc.byteOf(fi.main_token), "field '{s}': expected {s}, got {s}", .{ fname, bc.typeName(fty), bc.typeName(vt) });
                }
            } else {
                try bc.sink.emitFmt(bc.byteOf(fi.main_token), "unknown field '{s}' in '{s}'", .{ fname, name });
            }
        }
        for (sym.field_names, 0..) |dn, j| {
            if (!seen[j]) try bc.sink.emitFmt(bc.byteOf(n.main_token), "missing field '{s}' in '{s}'", .{ dn, name });
        }
        return Type.structT(id);
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
        const base = try bc.typeOf(n.lhs);
        if (base.kind == .invalid) return .invalid;
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
        // Resolve the enum id: qualified (lhs is the type-name identifier) or inferred
        // (the one-shot expected type must be an enum).
        var enum_id: u32 = undefined;
        if (n.lhs != Ast.none) {
            const tname = bc.nameText(bc.tree.nodes[(n.lhs).int()].main_token);
            enum_id = bc.activeEnumMap().get(tname) orelse {
                try bc.typeArgsForEffect(node_form, args);
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(n.lhs).int()].main_token), "'{s}' is not an enum type", .{tname});
                return .invalid;
            };
        } else {
            const exp = bc.expected orelse {
                try bc.typeArgsForEffect(node_form, args);
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "cannot infer the enum type for '.{s}' here", .{bc.nameText(n.main_token)});
                return .invalid;
            };
            if (!exp.isEnum()) {
                try bc.typeArgsForEffect(node_form, args);
                if (exp.kind != .invalid)
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "'.{s}' expects an enum type, but {s} was expected here", .{ bc.nameText(n.main_token), bc.typeName(exp) });
                return .invalid;
            }
            enum_id = exp.enum_id;
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

    fn checkVariant(bc: *BodyChecker, enum_id: u32, vtok: u32, node_form: InitForm, args: Ast.Index) error{OutOfMemory}!Type {
        const e = bc.model.enums[enum_id];
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
        const want_form: InitForm = switch (variant.form) {
            .unit => .unit,
            .tuple => .tuple,
            .@"struct" => .@"struct",
        };
        if (node_form != want_form) {
            try bc.typeArgsForEffect(node_form, args);
            try bc.sink.emitFmt(bc.byteOf(vtok), "variant '{s}.{s}' is constructed with the wrong form", .{ e.name, vname });
            return Type.enumT(enum_id);
        }
        switch (variant.form) {
            .unit => {},
            .tuple => {
                const elems = if (args == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (args).int());
                if (elems.len != variant.field_types.len) {
                    for (elems) |a| _ = try bc.typeOf(a);
                    try bc.sink.emitFmt(bc.byteOf(vtok), "variant '{s}.{s}' expects {d} value(s), got {d}", .{ e.name, vname, variant.field_types.len, elems.len });
                    return Type.enumT(enum_id);
                }
                for (elems, variant.field_types) |a, fty| {
                    const at = try bc.typeOfExpected(a, fty);
                    if (!Type.assignable(fty, at))
                        try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[(a).int()].main_token), "variant '{s}.{s}': expected {s}, got {s}", .{ e.name, vname, bc.typeName(fty), bc.typeName(at) });
                }
            },
            .@"struct" => {
                const inits = if (args == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (args).int());
                var seen = try bc.gpa.alloc(bool, variant.field_names.len);
                defer bc.gpa.free(seen);
                @memset(seen, false);
                for (inits) |fi_idx| {
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
                        const fty = variant.field_types[j];
                        const vt = try bc.typeOfExpected(fi.lhs, fty);
                        if (!Type.assignable(fty, vt))
                            try bc.sink.emitFmt(bc.byteOf(fi.main_token), "field '{s}': expected {s}, got {s}", .{ fname, bc.typeName(fty), bc.typeName(vt) });
                    } else {
                        _ = try bc.typeOf(fi.lhs);
                        try bc.sink.emitFmt(bc.byteOf(fi.main_token), "unknown field '{s}' in '{s}.{s}'", .{ fname, e.name, vname });
                    }
                }
                for (variant.field_names, 0..) |dn, j| {
                    if (!seen[j]) try bc.sink.emitFmt(bc.byteOf(vtok), "missing field '{s}' in '{s}.{s}'", .{ dn, e.name, vname });
                }
            },
        }
        return Type.enumT(enum_id);
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
        if (st.kind != .@"enum" and st.kind != .int and st.kind != .bool) {
            for (arms) |arm_idx| _ = try bc.typeOf(Ast.armHeaderAt(bc.tree, (bc.tree.nodes[(arm_idx).int()].rhs).int()).body);
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "match scrutinee must be an enum, int, or bool, got {s}", .{bc.typeName(st)});
            bc.node_types[(node_idx).int()] = .invalid;
            return .invalid;
        }

        var seen: []bool = &.{};
        var bool_cov: BoolCov = .{};
        var cov: Cov = switch (st.kind) {
            .@"enum" => blk: {
                seen = try bc.gpa.alloc(bool, bc.model.enums[st.enum_id].variants.len);
                @memset(seen, false);
                break :blk .{ .@"enum" = seen };
            },
            .bool => .{ .@"bool" = &bool_cov },
            else => .int,
        };
        defer if (st.kind == .@"enum") bc.gpa.free(seen);
    
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
                const e = bc.model.enums[st.enum_id];
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
        if (expected.kind != .@"enum") {
            if (expected.kind != .invalid)
                try bc.sink.emitFmt(bc.byteOf(pat.main_token), "variant pattern on a non-enum scrutinee {s}", .{bc.typeName(expected)});
            return;
        }
        const enum_id = expected.enum_id;
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
            // inner enum do not — caught by the type-aware payload check).
            if (count_cov and bc.variantPayloadIrrefutable(pat_idx, e.variants[i])) cov.@"enum"[i] = true;
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
                for (binders, variant.field_types) |b_idx, fty| {
                    try bc.checkPattern(b_idx, fty, cov, has_wildcard, false);
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
                            fty = variant.field_types[j];
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
            // A cross-module tuple-variant `mod.Enum.Variant(args)` (graph mode): the
            // callee field_access's receiver is the inner `mod.Enum`.
            if (bc.qualifiedEnumId(callee.lhs)) |enum_id| {
                const ty = try bc.checkVariant(enum_id, callee.main_token, .tuple, n.rhs);
                bc.node_types[(node_idx).int()] = ty;
                return ty;
            }
        }
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

    fn typeFromNode(bc: *BodyChecker, type_node: Ast.Index) Type {
        return refs.typeFromNode(bc, type_node);
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
