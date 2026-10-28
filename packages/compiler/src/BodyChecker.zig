const std = @import("std");
const builtin = @import("builtin");
const Ast = @import("ast/Ast.zig");
const Token = @import("ast/Token.zig").Token;
const TokenTag = @import("ast/Token.zig").Tag;
const Resolution = @import("symbols/Resolution.zig").Resolution;
const symbols_res = @import("symbols/Resolution.zig");
const Callee = @import("symbols/Callee.zig").Callee;
const Closure = @import("symbols/Closure.zig");
const Modes = @import("symbols/Modes.zig");
const DiagnosticSink = @import("diagnostics/Sink.zig");
const Span = @import("diagnostics/model.zig").Span;
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
const StdNames = @import("symbols/StdNames.zig");
const conform = @import("types/conform.zig");
const ControlFlow = @import("ControlFlow.zig");
const PatternChecker = @import("PatternChecker.zig");
const LambdaChecker = @import("LambdaChecker.zig");
const CallInfer = @import("types/call_infer.zig");
const Literal = @import("types/literal.zig");
const nearmiss = @import("diagnostics/nearmiss.zig");
const Model = Typecheck.Model;
const FnSym = Typecheck.FnSym;
const LoopCtx = Typecheck.LoopCtx;
const GraphCtx = Typecheck.GraphCtx;
const refs = Typecheck.refs;

/// The form a construction NODE supplies (independent of the declared variant).
const InitForm = enum { unit, tuple, @"struct" };

/// Substitute a template type through a concrete type-arg tuple: a
/// `type_var(ord)` becomes `targs[ord]`; an `App(ctor, [pat..])` or a fn type
/// recursively substitutes each arg and re-interns (so a template's field pattern
/// `Box[T]` grounds to `Box[int]`); any concrete type passes through. Needs the intern
/// table, hence the `bc` receiver. The sibling of `Typecheck.substType`.
pub fn substTy(bc: *BodyChecker, ty: Type, targs: []const Type) Type {
    if (ty.isTypeVar()) {
        const ord = ty.typeVarOrd();
        return if (ord < targs.len) targs[ord] else Type.invalid;
    }
    return refs.mapComposite(bc, ty, struct {
        bc: *BodyChecker,
        targs: []const Type,
        pub fn map(m: @This(), x: Type) Type {
            return substTy(m.bc, x, m.targs);
        }
    }{ .bc = bc, .targs = targs });
}

/// True when `ty` is a concrete value type usable as a monomorphization type-arg. This
/// predicate admits a ground `App` (a generic-struct instance used as a type-arg) and
/// `unit` (a concrete zero-sized type-arg, as in `Map[T, ()]`) and a function type.
/// Mirrors `Typecheck.isConcreteValue`.
pub fn isConcreteValue(ty: Type) bool {
    return switch (ty.kind) {
        .int, .bool, .float, .str, .@"struct", .@"enum", .app, .unit, .func => true,
        else => false,
    };
}

/// Near-miss candidate iterators (`next() ?[]const u8`) for the "did you mean" enrichment
/// of the field / enum-variant / method not-found diagnostics. `suggest` copies the
/// iterator by value and its strict-unique-winner rule makes the result order-independent.
const FieldNameIter = struct {
    names: []const []const u8,
    i: usize = 0,
    pub fn next(self: *FieldNameIter) ?[]const u8 {
        if (self.i >= self.names.len) return null;
        defer self.i += 1;
        return self.names[self.i];
    }
};

const VariantNameIter = struct {
    variants: []const LayoutEngine.VariantSym,
    i: usize = 0,
    pub fn next(self: *VariantNameIter) ?[]const u8 {
        if (self.i >= self.variants.len) return null;
        defer self.i += 1;
        return self.variants[self.i].name;
    }
};

/// Yields the names of every method callable on `recv`: the concrete-receiver methods
/// (`recv` type-equals `m.recv` — the same filter `findMethod`/`resolveConformanceMethod`
/// dispatch on) PLUS, when `recv` is a generic instance (`App`), the template methods of
/// its ctor (which live in `templates`, keyed by `recv_ctor` — a plain `Vec[int]` has NO
/// entry in `methods`, so without this a method typo on Vec/Map/Set would get no hint).
const MethodNameIter = struct {
    methods: []const Typecheck.Method,
    recv: Type,
    templates: []const Typecheck.TemplateMethod = &.{},
    app_ctor: u32 = 0,
    app_is_enum: bool = false,
    app: bool = false,
    i: usize = 0,
    ti: usize = 0,
    pub fn next(self: *MethodNameIter) ?[]const u8 {
        while (self.i < self.methods.len) {
            const m = self.methods[self.i];
            self.i += 1;
            if (Type.eql(m.recv, self.recv)) return m.name;
        }
        if (self.app) while (self.ti < self.templates.len) {
            const t = self.templates[self.ti];
            self.ti += 1;
            if (t.recv_ctor == self.app_ctor and t.recv_is_enum == self.app_is_enum) return t.name;
        };
        return null;
    }
};

/// A human name for an Option/Result family, for the `?` mismatch diagnostic.
fn familyName(fam: LayoutEngine.NativeEnumFamily) []const u8 {
    return switch (fam) {
        .option => "Option",
        .result => "Result",
        .none => "a non-Option/Result",
    };
}

/// Reachability + must-use lint DECISIONS as pure functions over a read-only cursor,
/// mirroring `ControlFlow`. Kept beside the divergence walks they consume; the emitting
/// `BodyChecker.warn*` methods are thin shims that build a `ControlFlow.Ctx`, call these, and render
/// the diagnostic — so the decision logic is unit-tested here without a `DiagnosticSink`.
const Lint = struct {
    /// Index into `stmts` of the first statement made unreachable by a preceding always-
    /// diverging statement, or null. (Emit side: W0005, at most once per block.)
    fn firstUnreachable(ctx: ControlFlow.Ctx, stmts: []const Ast.Index) ?usize {
        if (stmts.len < 2) return null;
        for (stmts[0 .. stmts.len - 1], 1..) |s, next| {
            if (divergesForReach(ctx, s)) return next;
        }
        return null;
    }

    /// Whether `stmt_idx` always transfers control away, so a following statement in the
    /// same block cannot run. `ControlFlow.stmtDiverges` is the shared authority; it is
    /// NOT modified here. The `never` clause additionally catches a `var_decl` bound to a
    /// diverging value (`x := loop {}`) and a never-typed call statement (`panic(..)`),
    /// which `stmtDiverges` leaves `false`. A non-final `expr_stmt` carries no type of its
    /// own, so its expression's type is read.
    fn divergesForReach(ctx: ControlFlow.Ctx, stmt_idx: Ast.Index) bool {
        if (ControlFlow.stmtDiverges(ctx, stmt_idx)) return true;
        const s = ctx.tree.nodes[stmt_idx.int()];
        const typed = if (s.tag == .expr_stmt) s.lhs else stmt_idx;
        return ctx.node_types[typed.int()].kind == .never;
    }

    /// The literal-bool condition token to warn on (a bare `true`/`false`), or null.
    /// (Emit side: W0007.) Reads only the tree, so it takes it directly.
    fn constIfCond(tree: Ast.Tree, cond: Ast.Index) ?u32 {
        const node = tree.nodes[cond.int()];
        if (node.tag != .literal_bool) return null;
        return node.main_token;
    }

    /// The must-use family name to warn about for a dropped expression-statement value,
    /// or null to stay silent. Split from the intern-table lookup (`optResultFamily`) so
    /// the poison-suppression + family gate is testable without a `Composite`. A poison
    /// return type checks the body in statement context, so a trailing value the author
    /// meant to RETURN reaches the drop check as a "dropped" value; suppress it to avoid
    /// stacking a spurious drop warning on the already-reported type error.
    fn droppedMustUseName(cur_ret: Type, fam: LayoutEngine.NativeEnumFamily) ?[]const u8 {
        if (cur_ret.kind == .invalid) return null;
        if (fam == .none) return null;
        return familyName(fam);
    }

    const Uncalled = enum { value, lambda, capture };

    /// Which kind of uncalled function a dropped expression-statement value is, or null.
    /// Only forms that do nothing once their value is dropped count, so the statement is
    /// almost surely a call missing its `(..)`. A call that returns a function has run
    /// its effects, so its dropped result is like any other dropped value. Suppressed under
    /// a poison return type for the reason `droppedMustUseName` gives.
    fn uncalledFn(tree: Ast.Tree, tokens: []const Token, cur_ret: Type, expr: Ast.Index, vt: Type) ?Uncalled {
        if (Ast.isCaptureLambda(tree, tokens, expr)) return if (vt.kind == .invalid) null else .capture;
        if (!vt.isFunc() or cur_ret.kind == .invalid) return null;
        return switch (tree.nodes[expr.int()].tag) {
            .identifier, .field_access, .tuple_field, .index => .value,
            .lambda => .lambda,
            else => null,
        };
    }
};

/// A type expected of the node being typed. `soft` when it reached a call's receiver from
/// the call, so it only hints: a literal that cannot produce it takes its own type instead
/// of an error.
pub const Expected = struct { ty: Type, soft: bool = false };

/// The state of one body being checked, which a lambda or local fn body replaces whole
/// while it is checked (`enterBody`/`leaveBody`).
pub const BodyCtx = struct {
    cur_ret: Type = .unit,
    expected: ?Expected = null,
    /// The innermost `_` capture's hole while its body is checked: the one node whose
    /// typing with a known param type gives the capture its param type. `none` in any
    /// other body, so it also tells whether the innermost body is a capture.
    hole: Ast.Index = Ast.none,
    /// While a generic call types its waiting args: the callee's type-param names and which
    /// of them are still open, so an expected fn type that holds an open one names it.
    open_names: []const []const u8 = &.{},
    open_vars: []const bool = &.{},
    /// The `loop_stack` depth at the innermost enclosing lambda or local fn: a jump may
    /// target only the contexts at or above it. 0 outside every one.
    loop_floor: usize = 0,
    /// Whether that innermost one is a local fn, which T0051 names.
    floor_is_local_fn: bool = false,
    /// Inside a lambda whose return type is neither annotated nor expected: each
    /// `return` joins its operand into `ret_join` instead of checking it against `cur_ret`.
    ret_infer: bool = false,
    ret_join: Type = .never,
    /// Inside a lambda whose return type is taken from the fn type expected for it rather
    /// than written: that fn type, which a mismatched `return` names, since the lambda
    /// declared nothing.
    ret_expected: ?Type = null,
};

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
    /// The method-call selections, parallel to `node_types` and written under the same
    /// own-span-only rule. Written whenever a selection is made, whether or not the args
    /// then check, so an argument error keeps the pick.
    callees: []Callee,
    graph_mod: u32,

    // Per-fn scratch.
    slot_types: std.ArrayList(Type) = .empty,
    loop_stack: std.ArrayList(LoopCtx) = .empty,
    /// The innermost body being checked: the fn's own, or a lambda's or local fn's in it.
    body: BodyCtx = .{},
    /// Whether the checker is inside an `unsafe { .. }` block. Gates the raw-pointer
    /// `store`/`load` ops — a use outside `unsafe` is a T0038 error.
    in_unsafe: bool = false,
    /// The argument a generic call is typing. A lambda there is typed after the call's
    /// other args, against what they bind, so an untyped param it still cannot type names
    /// the generic call.
    generic_arg: Ast.Index = Ast.none,
    /// An argument of a call already rejected with its own error: an unannotated lambda
    /// there is no second error.
    quiet_arg: Ast.Index = Ast.none,
    /// This fn's lambdas as the resolver recorded them, or null when it has none.
    closures: ?*const Closure.FnClosures = null,
    /// The local slots a call writes through: the root of a `mut self` receiver place or
    /// of a `mut` argument. A captured one must be boxed like an assigned one.
    mutated_roots: std.ArrayList(u32) = .empty,
    /// The locals a bound method value captures as its receiver (`f := acc.plus`).
    bound_roots: std.ArrayList(u32) = .empty,
    /// The locals an assignment writes (`x = ..`, or a place store rooted at `x`).
    assigned_roots: std.ArrayList(u32) = .empty,
    /// The assignment target being typed, or none: the member it names is a field, never a
    /// bound method, so `acc.plus = f` stays a missing field, and an index it names is a
    /// store. A node, not a flag, so the rest of the place (a call's arguments, a lambda's
    /// body) still binds.
    place_target: Ast.Index = Ast.none,
    /// This body's boxed locals, ascending: filled by `classifyCaptures` at the end of the
    /// walk and moved out by the caller.
    boxed: std.ArrayList(u32) = .empty,

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

    /// True ONLY for a bounded generic template's one-shot bound-as-axiom body check
    /// (`bodyUnit`), where `subst` maps each param name to its own ordinal `type_var`
    /// rather than a concrete type. In that mode a nested generic call spelled with the
    /// template's params (`mp_find[K, V]`) legitimately passes `type_var` type-args that
    /// ground at the OUTER template's instantiation, so the "concrete value type-arg"
    /// gate must admit a `type_var`. Never set on a per-instance re-check (subst is
    /// concrete there) or a non-generic body, so those paths stay byte-identical.
    abstract_template: bool = false,

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

    /// A bounded template's instance re-check only: the callees its bound-as-axiom check
    /// recorded on the module's nodes (borrowed), where a call on a bounded param holds
    /// the bound's protocol it resolved through (see `boundConformance`). Empty elsewhere.
    axiom_callees: []const Callee = &.{},

    /// The ordered generic-param NAMES, set by `bodyCheckerFor` from
    /// `FnSym.generic_params`; empty for a non-generic fn. Read ONLY by the innermost-
    /// failure diagnostic so an abstract `App`'s `type_var` culprit renders as its param
    /// name (`T`) rather than the opaque `type_var` tag. Borrowed source slices.
    gph_generic_params: []const []const u8 = &.{},

    /// Set while a hint is written as its template spells it (`templateSpelling`).
    spell_as_template: bool = false,
    /// Each T0016/T0025 this instance re-check reported whose text names the instance's
    /// own types, with the text as the template spells it (owned), for the mono tail to
    /// unify the reports one site gets from several instances (`unifyInstanceHints`).
    template_spellings: std.ArrayList(TemplateSpelling) = .empty,

    /// Structural derive requests this fn's body recorded: a `==`/`!=` on a
    /// derivable type with no impl. Moved out into the `BodyResult` after the walk;
    /// merged fn-id-ordered by `checkBodies`. THREAD-LOCAL (one list per BodyChecker),
    /// so the parallel Pass C never shares mutable derive state.
    derive_reqs: std.ArrayList(Typecheck.DeriveReq) = .empty,

    /// The types this fn's body demanded a descriptor for (`descriptor_of[T]()`). Moved
    /// into the `BodyResult` after the walk; merged fn-id-ordered by `checkBodies`.
    /// THREAD-LOCAL, mirroring `derive_reqs` (no shared mutable state under parallel Pass C).
    descriptor_reqs: std.ArrayList(Type) = .empty,

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

    /// What `typeUnderHint` chose for a node that erred under its hint, while an
    /// outermost `typeUnderHint` runs: an outer retry types its nested receivers again, so
    /// without it each level of nesting would double the work.
    hint_choices: std.AutoHashMapUnmanaged(Ast.Index, HintChoice) = .empty,
    hint_depth: u32 = 0,

    /// The ordered generic-param names + the concrete args they bind to, in
    /// generic-param order (`names[i]` binds `types[i]`).
    pub const Subst = struct { names: []const []const u8, types: []const Type };

    pub const HintChoice = struct { hint: Type, keep: bool };

    /// `message` is the reported text (the sink's own buffer); `template` the same text with
    /// each instance type arg written as its param (owned).
    pub const TemplateSpelling = struct { message: []const u8, template: []u8 };

    pub fn deinit(bc: *BodyChecker) void {
        bc.slot_types.deinit(bc.gpa);
        bc.loop_stack.deinit(bc.gpa);
        bc.mutated_roots.deinit(bc.gpa);
        bc.bound_roots.deinit(bc.gpa);
        bc.assigned_roots.deinit(bc.gpa);
        bc.boxed.deinit(bc.gpa);
        bc.sink.deinit();
        bc.derive_reqs.deinit(bc.gpa);
        bc.descriptor_reqs.deinit(bc.gpa);
        bc.conforms_memo.deinit(bc.gpa);
        bc.hint_choices.deinit(bc.gpa);
        for (bc.template_spellings.items) |ts| bc.gpa.free(ts.template);
        bc.template_spellings.deinit(bc.gpa);
    }

    /// Walk this fn's body: rebuild the slot table, type the body block against the
    /// declared return type, and enforce definite-return. The cursor + scratch live
    /// on `bc`; the immutable fn/struct/enum tables are read through `bc.model`.
    pub fn checkBody(bc: *BodyChecker, fid: u32, f: FnSym) !void {
        _ = fid;
        const decl = bc.tree.nodes[(f.decl_node).int()];

        // Rebuild the per-function slot→type table. Parameters get slots 0..N first
        // (the resolver declares them first), then `:=` locals as we encounter them.
        bc.slot_types.clearRetainingCapacity();
        for (f.params) |pty| try bc.slot_types.append(bc.gpa, pty);
        bc.body.cur_ret = f.ret;

        const want_value = (f.ret.kind != .unit and f.ret.kind != .invalid);
        const body_ty = try bc.typeOfBlockExpected(decl.rhs, want_value, if (want_value) f.ret else null);

        // A non-unit function must produce a value on every path: either all paths
        // structurally return, OR the body's trailing expression has the declared
        // type. `assignable` folds poison/never/eql exactly as before.
        if (want_value and !bc.blockReturns(decl.rhs) and !Type.assignable(f.ret, body_ty)) {
            if (!try bc.reportTailMismatch(decl.rhs, body_ty, f.ret))
                try bc.sink.report(.{ .span = bc.tokSpan(decl.main_token), .code = .T0043 }, "function '{s}' must return {s} but may fall off the end", .{ bc.nameText(decl.main_token), bc.typeName(f.ret) });
        }
        try LambdaChecker.classifyCaptures(bc, f.mut_mask);
    }

    /// The expression whose value a block yields, or null when it ends in a statement.
    pub fn blockTail(bc: *const BodyChecker, block_idx: Ast.Index) ?Ast.Index {
        const stmts = Ast.rangeSlice(bc.tree, (bc.tree.nodes[(block_idx).int()].lhs).int());
        if (stmts.len == 0) return null;
        const last = stmts[stmts.len - 1];
        const n = bc.tree.nodes[last.int()];
        return switch (n.tag) {
            .expr_stmt => n.lhs,
            .block, .if_stmt => last,
            else => null,
        };
    }

    /// A body that ends in a value of a type its declared return type does not take is
    /// T0042 at that value. Reporting it as a missing return (T0043) would send the user
    /// looking for a path with no value. False when the body ends with no value, which is
    /// the caller's T0043.
    pub fn reportTailMismatch(bc: *BodyChecker, block_idx: Ast.Index, body_ty: Type, ret: Type) error{OutOfMemory}!bool {
        if (body_ty.kind == .unit or body_ty.kind == .never or body_ty.kind == .invalid) return false;
        const tail = bc.blockTail(block_idx) orelse return false;
        try bc.sink.report(.{ .span = bc.spanOf(tail), .code = .T0042 }, "value {s} does not match declared return type {s}", .{ bc.typeName(body_ty), bc.typeName(ret) });
        return true;
    }

    /// The type expected of the node being typed, if any.
    pub fn expectedTy(bc: *const BodyChecker) ?Type {
        return if (bc.body.expected) |e| e.ty else null;
    }

    /// Whether this fn reported an ERROR since sink position `mark`. Warnings do not
    /// count: a T0050 they suppressed would let a silent poison reach codegen.
    pub fn errorsSince(bc: *const BodyChecker, mark: usize) bool {
        for (bc.sink.items()[mark..]) |d| if (d.severity == .err) return true;
        return false;
    }

    fn errorCountSince(bc: *const BodyChecker, mark: usize) usize {
        var n: usize = 0;
        for (bc.sink.items()[mark..]) |d| n += @intFromBool(d.severity == .err);
        return n;
    }

    // The control-flow / divergence / break family lives in `ControlFlow.zig` as
    // pure functions over a small read-only `Ctx` (the tree + resolutions + enum
    // table + tokens). These thin wrappers build a `Ctx` from `bc` and delegate,
    // so the mutually-recursive AST walks (which drift when a new node is added)
    // are defined and unit-tested in one place.

    pub fn cflow(bc: *const BodyChecker) ControlFlow.Ctx {
        return ControlFlow.Ctx.fromChecker(bc);
    }

    pub fn blockReturns(bc: *const BodyChecker, block_idx: Ast.Index) bool {
        return ControlFlow.blockReturns(bc.cflow(), block_idx);
    }

    fn stmtReturns(bc: *const BodyChecker, stmt_idx: Ast.Index) bool {
        return ControlFlow.stmtReturns(bc.cflow(), stmt_idx);
    }

    pub fn blockDiverges(bc: *const BodyChecker, block_idx: Ast.Index) bool {
        return ControlFlow.blockDiverges(bc.cflow(), block_idx);
    }

    fn stmtDiverges(bc: *const BodyChecker, stmt_idx: Ast.Index) bool {
        return ControlFlow.stmtDiverges(bc.cflow(), stmt_idx);
    }

    fn blockHasBreak(bc: *const BodyChecker, block_idx: Ast.Index, target: Ast.Index) bool {
        return ControlFlow.blockHasBreak(bc.cflow(), block_idx, target);
    }

    pub fn typeOfBlockExpected(bc: *BodyChecker, block_idx: Ast.Index, want_value: bool, exp: ?Type) error{OutOfMemory}!Type {
        const save = bc.body.expected;
        bc.body.expected = if (exp) |t| .{ .ty = t } else null;
        defer bc.body.expected = save;
        return bc.checkBlock(block_idx, want_value);
    }

    fn checkBlock(bc: *BodyChecker, block_idx: Ast.Index, want_value: bool) error{OutOfMemory}!Type {
        const stmts = Ast.rangeSlice(bc.tree, (bc.tree.nodes[(block_idx).int()].lhs).int());
        if (stmts.len == 0) {
            bc.node_types[(block_idx).int()] = .unit;
            return .unit;
        }
        // The expected type applies ONLY to this block's trailing value expression, never
        // the non-final statements. It is restored on return because the caller may check
        // further arms against it (an `if`'s else, the next match arm).
        const block_expected = bc.body.expected;
        bc.body.expected = null;
        defer bc.body.expected = block_expected;
        for (stmts[0 .. stmts.len - 1]) |s| {
            try bc.checkStmt(s); // effect only
            const sn = bc.tree.nodes[(s).int()];
            if (sn.tag == .expr_stmt)
                try bc.warnDroppedValue(s, bc.node_types[(sn.lhs).int()]);
        }
        const last = stmts[stmts.len - 1];
        const last_n = bc.tree.nodes[(last).int()];
        var bt: Type = .unit;
        if (last_n.tag == .expr_stmt) {
            bt = try bc.typeUnder(last_n.lhs, if (want_value) block_expected else null);
            bc.node_types[(last).int()] = bt; // the expr_stmt carries the value type
            if (!want_value) try bc.warnDroppedValue(last, bt);
        } else if (want_value and last_n.tag == .block) {
            bt = try bc.typeUnder(last, block_expected); // value context: validates + types + memoizes
        } else if (want_value and last_n.tag == .if_stmt and
            Ast.ifHeaderAt(bc.tree, (last_n.rhs).int()).else_node != Ast.none)
        {
            // A value-if needs both arms to yield a value; an else-less trailing
            // `if` produces nothing, so it is an effect statement, not this
            // block's value.
            bt = try bc.typeUnder(last, block_expected); // value context: validates + types + memoizes
        } else {
            try bc.checkStmt(last); // statement context (else-less trailing if / want_value=false)
        }
        bc.node_types[(block_idx).int()] = bt;
        try bc.warnUnreachable(stmts);
        return bt;
    }

    /// Emit W0005 on the first statement made unreachable by a preceding always-
    /// diverging statement in the same block. Runs after the block is typed so the
    /// divergence walk can read a match scrutinee's memoized type. Read-only: it only
    /// emits a diagnostic, so it never perturbs the definite-return / fall-off logic
    /// nor the emitted machine code. At most one W0005 per block (no cascade).
    fn warnUnreachable(bc: *BodyChecker, stmts: []const Ast.Index) error{OutOfMemory}!void {
        const dead = Lint.firstUnreachable(bc.cflow(), stmts) orelse return;
        try bc.sink.report(.{ .span = bc.spanOf(stmts[dead]), .code = .W0005 }, "unreachable code; the previous statement always diverges", .{});
    }

    /// Emit W0012 when a discarded expression statement's value is a function it never
    /// calls, else W0008 when it is an `Option`/`Result`. Each message is the same for
    /// every monomorphization instance (W0008 names only the family; W0012 spells the
    /// source), so the per-instance re-checks of a generic fn yield one diagnostic that
    /// the sink dedups on (scope, offset, code, message). Diagnostics-only — it reads the
    /// already-memoized value type and never perturbs typing, the value-if path, or the
    /// emitted machine code.
    fn warnDroppedValue(bc: *BodyChecker, stmt_idx: Ast.Index, vt: Type) error{OutOfMemory}!void {
        const expr = bc.tree.nodes[stmt_idx.int()].lhs;
        if (Lint.uncalledFn(bc.tree, bc.tokens, bc.body.cur_ret, expr, vt)) |form| return bc.warnUncalled(stmt_idx, expr, vt, form);
        const name = Lint.droppedMustUseName(bc.body.cur_ret, bc.optResultFamily(vt)) orelse return;
        try bc.sink.report(.{ .span = bc.spanOf(stmt_idx), .code = .W0008 }, "unused {s} value; handle it, bind it with ':=', or propagate with '?'", .{name});
    }

    /// W0012 for a function value an expression statement drops. Since function values
    /// and bound methods became expressions, `c.bump` (meant as `c.bump()`) is no longer an
    /// error but a statement that does nothing, so say so and spell the call.
    fn warnUncalled(bc: *BodyChecker, stmt_idx: Ast.Index, expr: Ast.Index, vt: Type, form: Lint.Uncalled) error{OutOfMemory}!void {
        const at: DiagnosticSink.Report = .{ .span = bc.spanOf(stmt_idx), .code = .W0012 };
        switch (form) {
            // Once that statement read a `_` local (`show(_)`), so the message names the rename.
            .capture => try bc.sink.report(at, "unused '_' capture; the '_' makes the call a function, which is never called; to pass a local named '_', rename it", .{}),
            .lambda => try bc.sink.report(at, "lambda is never called, so its body never runs; bind it with ':=' and call it, or write its statements without the lambda", .{}),
            .value => {
                const args = if (Composite.fnParts(bc.composite.at(vt.funcIdx())).params.len == 0) "()" else "(..)";
                const span = bc.spanOf(expr);
                const text = bc.source[span.start..span.end];
                // A multi-line or commented receiver chain would quote badly; name the parens alone.
                if (std.mem.indexOfAny(u8, text, "\n#") == null)
                    try bc.sink.report(at, "function value is never called; did you mean '{s}{s}'?", .{ text, args })
                else
                    try bc.sink.report(at, "function value is never called; add '{s}' to call it", .{args});
            },
        }
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
                        try bc.sink.report(.{ .span = bc.spanOf(stmt.lhs), .code = .T0041 }, "cannot bind {s} to '{s}' of type {s}", .{ bc.typeName(got), bc.nameText(stmt.main_token), bc.typeName(declared) });
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
                    // A `.`-rooted place and an element type as expressions; an index's
                    // own errors (`m[k]` on a Map is T0057) know they are on a store.
                    .field_access, .tuple_field, .index => blk: {
                        const was = bc.place_target;
                        bc.place_target = stmt.lhs;
                        defer bc.place_target = was;
                        break :blk try bc.typeOf(stmt.lhs);
                    },
                    else => .invalid,
                };
                bc.node_types[(stmt.lhs).int()] = lhs;
                try LambdaChecker.noteAssignedRoot(bc, stmt.lhs);
                const rhs = if (bc.tokens[stmt.main_token].tag.compoundBase()) |base|
                    try bc.compoundValue(stmt_idx, base, lhs)
                else
                    try bc.typeOfExpected(stmt.rhs, if (lhs.kind == .invalid) null else lhs);
                if (!Type.assignable(lhs, rhs)) {
                    const what = if (target.tag == .index) "an element" else "variable";
                    try bc.sink.report(.{ .span = bc.spanOf(stmt.rhs), .code = .T0041 }, "cannot assign {s} to {s} of type {s}", .{ bc.typeName(rhs), what, bc.typeName(lhs) });
                }
                switch (target.tag) {
                    .index => bc.noteElementWrite(stmt.lhs, bc.tokens[stmt.main_token].tag.compoundBase() != null),
                    .field_access, .tuple_field => if (bc.rootIsTemporary(stmt.lhs)) {
                        if (lhs.kind != .invalid)
                            try bc.sink.report(.{ .span = bc.spanOf(stmt.lhs), .code = .T0075 }, "cannot assign to a field of a temporary value, such as a call result; it is a copy, so the write would be lost", .{});
                    } else bc.noteElementWrite(stmt.lhs, true),
                    else => {},
                }
            },
            .return_stmt => {
                if (bc.body.ret_infer) {
                    const rt: Type = if (stmt.lhs == Ast.none) Type.unit else try bc.typeOfExpected(stmt.lhs, null);
                    return LambdaChecker.joinLambdaReturn(bc, if (stmt.lhs == Ast.none) stmt_idx else stmt.lhs, rt);
                }
                const ty: Type = if (stmt.lhs == Ast.none) Type.unit else try bc.typeOfExpected(stmt.lhs, if (bc.body.cur_ret.kind == .invalid) null else bc.body.cur_ret);
                // Skip when either side is poison so an unknown return type (already
                // reported) doesn't trigger a spurious second diagnostic here. A
                // `never`-typed operand (e.g. `return x` where x binds a break-less
                // loop) is also fine: control never actually reaches the return, so
                // `never` unifies with any declared type — mirrors the trailing-expr
                // body check above.
                if (!Type.assignable(bc.body.cur_ret, ty)) {
                    const at: DiagnosticSink.Report = .{ .span = bc.spanOf(if (stmt.lhs == Ast.none) stmt_idx else stmt.lhs), .code = .T0042 };
                    if (bc.body.ret_expected) |e|
                        try bc.sink.report(at, "return type {s} does not match expected {s} (the lambda is expected to be {s})", .{ bc.typeName(ty), bc.typeName(bc.body.cur_ret), bc.typeName(e) })
                    else
                        try bc.sink.report(at, "return type {s} does not match declared {s}", .{ bc.typeName(ty), bc.typeName(bc.body.cur_ret) });
                }
            },
            .expr_stmt => _ = try bc.typeOf(stmt.lhs),
            .block => _ = try bc.checkBlock(stmt_idx, false),
            .if_stmt => {
                const ct = try bc.typeOf(stmt.lhs);
                if (ct.kind != .invalid and ct.kind != .bool)
                    try bc.sink.report(.{ .span = bc.spanOf(stmt.lhs), .code = .T0045 }, "if condition must be bool", .{});
                try bc.warnConstIfCond(stmt.lhs);
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
            .for_in2_stmt => try bc.checkForIn2(stmt_idx, null),
            .labeled => _ = try bc.checkLabeled(stmt_idx, false),
            .local_fn => try LambdaChecker.checkLocalFn(bc, stmt_idx, stmt),
            .break_stmt => {
                const ctx = bc.targetCtx(stmt_idx) orelse {
                    // No matching context: a bare break with an empty stack ("outside a
                    // loop"); a labeled break is reported by resolve as undefined.
                    if (LambdaChecker.jumpLeavesLambda(bc, stmt_idx))
                        try bc.sink.report(.{ .span = bc.spanOf(stmt_idx), .code = .T0051 }, "cannot break a loop outside this {s}", .{LambdaChecker.jumpFloorName(bc)})
                    else if (bc.resolutions[(stmt_idx).int()] != .label)
                        try bc.sink.report(.{ .span = bc.spanOf(stmt_idx), .code = .T0060 }, "break outside of a loop", .{});
                    if (stmt.lhs != Ast.none) _ = try bc.typeOf(stmt.lhs);
                    return;
                };
                if (stmt.lhs == Ast.none) {
                    ctx.saw_bare_break = true;
                    if (ctx.is_value) ctx.join = try bc.merge(stmt.main_token, ctx.join, Type.unit);
                } else {
                    const vt = try bc.typeUnder(stmt.lhs, ctx.expected);
                    if (!ctx.is_value) {
                        if (vt.kind != .invalid and vt.kind != .unit)
                            try bc.sink.report(.{ .span = bc.spanOf(stmt_idx), .code = .T0060 }, "cannot break with a value out of a while/for loop", .{});
                    } else {
                        ctx.saw_value_break = true;
                        ctx.join = try bc.merge(stmt.main_token, ctx.join, vt);
                    }
                }
            },
            .continue_stmt => {
                const ctx = bc.targetCtx(stmt_idx) orelse {
                    if (LambdaChecker.jumpLeavesLambda(bc, stmt_idx))
                        try bc.sink.report(.{ .span = bc.spanOf(stmt_idx), .code = .T0051 }, "cannot continue a loop outside this {s}", .{LambdaChecker.jumpFloorName(bc)})
                    else if (bc.resolutions[(stmt_idx).int()] != .label)
                        try bc.sink.report(.{ .span = bc.spanOf(stmt_idx), .code = .T0060 }, "continue outside of a loop", .{});
                    return;
                };
                // `continue` is meaningful only on a loop; a labeled bare block is not.
                if (ctx.kind == .labeled_block)
                    try bc.sink.report(.{ .span = bc.spanOf(stmt_idx), .code = .T0060 }, "cannot continue a labeled block (not a loop)", .{});
            },
            // A poison leaf as a statement: already-diagnosed, no further check.
            .error_node => {},
            else => _ = try bc.typeOf(stmt_idx),
        }
    }

    fn targetCtx(bc: *BodyChecker, stmt_idx: Ast.Index) ?*LoopCtx {
        const items = bc.loop_stack.items[bc.body.loop_floor..];
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
            try bc.sink.report(.{ .span = bc.spanOf(stmt.lhs), .code = .T0045 }, "while condition must be bool", .{});
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
            try bc.sink.report(.{ .span = bc.spanOf(h.lo), .code = .T0044 }, "for range bounds must be int", .{});
        if (hi.kind != .invalid and hi.kind != .int)
            try bc.sink.report(.{ .span = bc.spanOf(h.hi), .code = .T0044 }, "for range bounds must be int", .{});
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
        const iter_ty = try bc.resolveIterType(recv, stmt.rhs) orelse blk: {
            // A non-iterable receiver: name the protocol so the error is greppable.
            if (recv.kind != .invalid)
                try bc.sink.report(.{ .span = bc.spanOf(stmt.rhs), .code = .T0023 }, "type '{s}' does not conform to protocol 'Iterator'", .{bc.typeName(recv)});
            break :blk Type.invalid;
        };
        const item: Type = if (iter_ty.kind == .invalid)
            .invalid
        else
            try conform.iteratorItem(bc.model, bc.composite, bc.gpa, iter_ty) orelse blk: {
                try bc.sink.report(.{ .span = bc.spanOf(stmt.rhs), .code = .T0023 }, "type '{s}' does not conform to protocol 'Iterator'", .{bc.typeName(recv)});
                break :blk Type.invalid;
            };
        bc.node_types[(stmt_idx).int()] = iter_ty;
        if (iter_ty.kind != .invalid and bc.iterTakesMutSelf(recv)) try bc.checkMutReceiver(stmt.rhs, StdNames.method_iter);
        if (bc.resolutions[(stmt_idx).int()] == .local) try bc.setSlot(bc.resolutions[(stmt_idx).int()].local, item);
        try bc.loop_stack.append(bc.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
        _ = try bc.checkBlock(stmt.lhs, false);
        _ = bc.loop_stack.pop();
    }

    /// `for k, v in m { body }`: the receiver's `iter()` must yield `Iterator[Entry[K,V]]`.
    /// `k` binds to the Entry's first field type, `v` to its second (the field-order coupling
    /// on `struct Entry[K,V] { key, val }`). `node_types[stmt]` = the iterator type (read by
    /// scanCalls/reify/lower), exactly as `checkForIn`.
    fn checkForIn2(bc: *BodyChecker, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
        const stmt = bc.tree.nodes[stmt_idx.int()];
        const h = Ast.forIn2HeaderAt(bc.tree, stmt.lhs.int());
        const recv = try bc.typeOf(stmt.rhs);
        const iter_ty = try bc.resolveIterType(recv, stmt.rhs) orelse blk: {
            if (recv.kind != .invalid)
                try bc.sink.report(.{ .span = bc.spanOf(stmt.rhs), .code = .T0023 }, "type '{s}' does not conform to protocol 'Iterator'", .{bc.typeName(recv)});
            break :blk Type.invalid;
        };
        const item: Type = if (iter_ty.kind == .invalid) .invalid else try conform.iteratorItem(bc.model, bc.composite, bc.gpa, iter_ty) orelse blk: {
            try bc.sink.report(.{ .span = bc.spanOf(stmt.rhs), .code = .T0023 }, "type '{s}' does not conform to protocol 'Iterator'", .{bc.typeName(recv)});
            break :blk Type.invalid;
        };
        var kty: Type = .invalid;
        var vty: Type = .invalid;
        if (item.isApp()) {
            const e = bc.composite.at(item.appIdx());
            const entry_ctor = bc.activeStructMap().get(StdNames.entry_struct);
            if (!e.ctor_is_enum and entry_ctor != null and e.ctor == entry_ctor.? and e.args.len == 2) {
                kty = e.args[0];
                vty = e.args[1];
            } else {
                try bc.sink.report(.{ .span = bc.spanOf(stmt.rhs), .code = .T0023 }, "'for k, v' requires an iterator over 'Entry' pairs", .{});
            }
        } else if (item.kind != .invalid) {
            try bc.sink.report(.{ .span = bc.spanOf(stmt.rhs), .code = .T0023 }, "'for k, v' requires an iterator over 'Entry' pairs", .{});
        }
        bc.node_types[stmt_idx.int()] = iter_ty;
        if (iter_ty.kind != .invalid and bc.iterTakesMutSelf(recv)) try bc.checkMutReceiver(stmt.rhs, StdNames.method_iter);
        if (bc.resolutions[stmt_idx.int()] == .local) try bc.setSlot(bc.resolutions[stmt_idx.int()].local, kty);
        if (bc.resolutions[h.val_leaf.int()] == .local) try bc.setSlot(bc.resolutions[h.val_leaf.int()].local, vty);
        try bc.loop_stack.append(bc.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
        _ = try bc.checkBlock(h.body, false);
        _ = bc.loop_stack.pop();
    }

    /// The return type of the inherent `iter()` method on `recv`, or null when `recv`
    /// has no such method. Only a generic-type (`App`) receiver is supported this
    /// milestone (Vec); the impl's type-params bind by matching the template's `Self`
    /// pattern against the receiver's args — the SAME `Infer.match` `dispatchAppMethod`
    /// runs. `.invalid` when `iter()` exists but is malformed (self-less / arity), or is
    /// ambiguous (T0025 at `iterable`).
    fn resolveIterType(bc: *BodyChecker, recv: Type, iterable: Ast.Index) error{OutOfMemory}!?Type {
        if (!recv.isApp()) return null;
        const e = bc.composite.at(recv.appIdx());
        // A desugar resolves at the program's site wherever it is written, as lower and
        // discovery resolve it (`StdNames.desugarMethods`).
        const m = switch (Typecheck.pickGenericMethod(bc.model.templates, e.ctor, e.ctor_is_enum, StdNames.method_iter, .program)) {
            .one => |m| m,
            .ambiguous => {
                try bc.emitAmbiguousConformance(bc.spanOf(iterable), recv, StdNames.method_iter, null);
                return Type.invalid;
            },
            .none => return null,
        };
        const mf = bc.model.fns[m.fn_id];
        if (!m.has_self or mf.params.len != 1) return Type.invalid; // iter(self), no extra args
        const targs = (Typecheck.bindImplParams(bc.gpa, bc.composite, mf, e.args) catch return Type.invalid) orelse return Type.invalid;
        defer bc.gpa.free(targs);
        return substTy(bc, mf.ret, targs);
    }

    /// Whether `recv`'s `iter()` takes `mut self`: `for` then passes the receiver's
    /// address, so it must be a place, and the loop writes it.
    fn iterTakesMutSelf(bc: *BodyChecker, recv: Type) bool {
        if (!recv.isApp()) return false;
        const e = bc.composite.at(recv.appIdx());
        const m = Typecheck.findGenericMethod(bc.model.templates, e.ctor, e.ctor_is_enum, StdNames.method_iter, .program) orelse return false;
        return m.mut_self;
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
            .for_in2_stmt => blk: {
                try bc.checkForIn2(n.lhs, label);
                break :blk Type.unit;
            },
            else => Type.invalid,
        };
        bc.node_types[(idx).int()] = ty;
        return ty;
    }

    fn typeOfLabeledBlock(bc: *BodyChecker, block_idx: Ast.Index, label: []const u8, want_value: bool) error{OutOfMemory}!Type {
        try bc.loop_stack.append(bc.gpa, .{ .kind = .labeled_block, .label = label, .construct_node = block_idx, .is_value = true, .join = .never, .saw_value_break = false, .saw_bare_break = false, .expected = if (want_value) bc.body.expected else null });
        const fall = try bc.checkBlock(block_idx, want_value);
        const ctx = bc.loop_stack.pop().?;
        // The trailing-expr value is unreachable iff the block's last statement
        // diverges; in that case the block's value comes entirely from its breaks.
        const ft: Type = if (bc.blockDiverges(block_idx)) Type.never else fall;
        return bc.merge(bc.tree.nodes[(block_idx).int()].main_token, ft, ctx.join);
    }

    pub fn typeOfExpected(bc: *BodyChecker, node_idx: Ast.Index, exp: ?Type) error{OutOfMemory}!Type {
        return bc.typeUnder(node_idx, if (exp) |t| .{ .ty = t } else null);
    }

    /// `typeOfExpected` keeping whether the expected type is only a hint, for a node that
    /// yields its construct's value (an arm, a break, a block's tail): a hint a receiver
    /// passes on stays one.
    pub fn typeUnder(bc: *BodyChecker, node_idx: Ast.Index, exp: ?Expected) error{OutOfMemory}!Type {
        if (exp) |want| if (node_idx != Ast.none and node_idx == bc.body.hole) try LambdaChecker.adoptHole(bc, want.ty);
        const save = bc.body.expected;
        bc.body.expected = exp;
        defer bc.body.expected = save;
        return bc.typeOf(node_idx);
    }

    /// The expected type of a part of a generic construction or call (a field value, a
    /// payload, an arg): its declared pattern `pat` outright when that holds no type
    /// parameter, else through `targs`, the type args the parent's own expected type
    /// binds. When that gives no type the part is typed on its own, never against the
    /// parent's type (`partHint`).
    pub fn partWant(bc: *BodyChecker, pat: Type, targs: ?[]const Type) ?Expected {
        const soft = if (bc.body.expected) |e| e.soft else false;
        if (bc.isGround(pat)) return .{ .ty = pat, .soft = soft };
        if (targs) |ta| {
            const w = substTy(bc, pat, ta);
            if (w.kind != .invalid and bc.givesType(w)) return .{ .ty = w, .soft = soft };
        }
        return bc.partHint();
    }

    /// What a part with no type of its own takes from its parent: nothing, unless the
    /// parent holds only a receiver's hint, which never errors and so reaches the parts
    /// as itself (`x: int8 = Box { v: 1 }.get()` makes the `1` an `int8`).
    pub fn partHint(bc: *const BodyChecker) ?Expected {
        const e = bc.body.expected orelse return null;
        return if (e.soft) e else null;
    }

    /// A method call's receiver. The call's expected type is the result's, not the
    /// receiver's, yet a receiver that is itself a literal used to take it (`x: int8 =
    /// Box { v: 1 }.get()`), and programs rely on that; so it reaches the receiver only as
    /// a hint (`typeUnderHint`).
    fn typeOfReceiver(bc: *BodyChecker, recv: Ast.Index) error{OutOfMemory}!Type {
        return bc.typeUnderHint(recv, if (bc.body.expected) |e| e.ty else null);
    }

    /// Type `node_idx` under `hint` as a soft expected type, which never turns into an
    /// error of its own: when the node errs under it, it is typed again with none and
    /// the typing with fewer errors stands. Each arm of a value `if` or `match` decides
    /// alone whether it takes a hint (`1` fits an `int8`, `300` does not), so only the
    /// whole value can tell that its arms took it in different ways; an error under the
    /// hint is that signal. An error of the node's own (`1 + true` in one arm) errs either
    /// way, and dropping the hint would only add the errors of arms that needed it. A
    /// hint is no type the node must have, so it never fills a capture's hole
    /// (`typeUnder`).
    pub fn typeUnderHint(bc: *BodyChecker, node_idx: Ast.Index, hint: ?Type) error{OutOfMemory}!Type {
        const save = bc.body.expected;
        defer bc.body.expected = save;
        bc.body.expected = null;
        const h = hint orelse return bc.typeOf(node_idx);
        bc.hint_depth += 1;
        defer {
            bc.hint_depth -= 1;
            if (bc.hint_depth == 0) bc.hint_choices.clearRetainingCapacity();
        }
        const hinted: Expected = .{ .ty = h, .soft = true };
        if (bc.hint_choices.get(node_idx)) |c| if (c.hint.eql(h)) {
            bc.body.expected = if (c.keep) hinted else null;
            return bc.typeOf(node_idx);
        };
        const mark = bc.sink.count();
        bc.body.expected = hinted;
        const t = try bc.typeOf(node_idx);
        const hinted_errs = bc.errorCountSince(mark);
        if (hinted_errs == 0) return t;
        bc.sink.truncate(mark);
        bc.body.expected = null;
        const bare = try bc.typeOf(node_idx);
        const keep = bc.errorCountSince(mark) > hinted_errs;
        try bc.hint_choices.put(bc.gpa, node_idx, .{ .hint = h, .keep = keep });
        if (!keep) return bare;
        bc.sink.truncate(mark);
        bc.body.expected = hinted;
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
    pub fn poison(bc: *const BodyChecker) Type {
        std.debug.assert(bc.sink.count() > 0);
        return Type.invalid;
    }

    /// Type a numeric literal token. Defaults to platform `int`, but adopts an
    /// `expected` integer WIDTH (`x: int8 = 100`, or a narrow-int match scrutinee) and
    /// range-checks the magnitude against it. Adopts the width even on a range error so
    /// the downstream assignability check stays silent — one T0034, never a paired
    /// mismatch. The single source shared by the expression `literal_number` arm and the
    /// PatternChecker numeric-pattern path (both must adopt + range-check identically).
    /// The width a numeric literal adopts: the expected type, unless that is only a hint
    /// the literal does not fit, which leaves the literal its own type and no error.
    fn literalExpected(bc: *const BodyChecker, main_token: u32) ?Type {
        const e = bc.body.expected orelse return null;
        if (e.soft and e.ty.isInteger() and !Literal.fitsWidth(bc.tokens[main_token].text(bc.source), e.ty, false)) return null;
        return e.ty;
    }

    pub fn typeNumericLiteral(bc: *BodyChecker, main_token: u32, expected: ?Type) error{OutOfMemory}!Type {
        const raw = bc.tokens[main_token].text(bc.source);
        if (expected) |e| if (e.isInteger()) {
            if (!Literal.fitsWidth(raw, e, false))
                try bc.sink.report(.{ .span = bc.tokSpan(main_token), .code = .T0034 }, "literal out of range for type '{s}'", .{bc.typeName(e)});
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
            try bc.sink.report(.{ .span = bc.tokSpan(main_token), .code = .T0034 }, "literal out of range for type '{s}'", .{bc.typeName(Type.int)});
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
            .empty => try bc.sink.report(.{ .span = bc.tokSpan(main_token), .code = .T0036 }, "empty char literal", .{}),
            .too_many => try bc.sink.report(.{ .span = bc.tokSpan(main_token), .code = .T0036 }, "char literal must hold exactly one codepoint; use double quotes for a string", .{}),
            .dangling_backslash => try bc.sink.report(.{ .span = bc.tokSpan(main_token), .code = .T0036 }, "char literal ends with a dangling backslash", .{}),
            .unknown_escape => try bc.sink.report(.{ .span = bc.tokSpan(main_token), .code = .T0036 }, "unknown escape in char literal", .{}),
            .bad_hex_escape => try bc.sink.report(.{ .span = bc.tokSpan(main_token), .code = .T0036 }, "malformed '\\x'/'\\u' escape in char literal", .{}),
            .bad_codepoint => try bc.sink.report(.{ .span = bc.tokSpan(main_token), .code = .T0036 }, "'\\u{{...}}' escape is not a Unicode scalar value", .{}),
            .bad_utf8 => try bc.sink.report(.{ .span = bc.tokSpan(main_token), .code = .T0036 }, "char literal is not valid UTF-8", .{}),
            .malformed => try bc.sink.report(.{ .span = bc.tokSpan(main_token), .code = .T0036 }, "malformed char literal", .{}),
        }
        if (bc.model.prelude) |p| if (p.char_struct) |cid| return Type.structT(cid);
        return .invalid;
    }

    /// A bracketed-postfix `type_app` whose head chain's LEFTMOST identifier resolves
    /// to a value is a value index the parser could not distinguish from a turbofish
    /// (`b.items[i].f` looks exactly like `mod.Box[int].new()`). A real type-app's
    /// leftmost is a struct/enum name, generic fn, or module — never `.local` — so
    /// this reinterpretation never touches one. `b.items[i]` (leftmost `b` -> .local)
    /// is an index; `mod.Box[int]` (leftmost `mod` -> .module) stays a turbofish.
    fn appHeadIsValue(bc: *const BodyChecker, node_idx: Ast.Index) bool {
        const n = bc.tree.nodes[(node_idx).int()];
        return n.tag == .type_app and symbols_res.leftmostHeadIsValue(bc.tree, bc.resolutions, n.lhs);
    }

    /// `recv[idx]` reads a `Vec[V]` element; the result type is V. Shared by the `.index`
    /// arm and the value-headed `.type_app` reinterpretation.
    fn typeOfIndexParts(bc: *BodyChecker, site: Ast.Index, recv: Ast.Index, idx: Ast.Index) error{OutOfMemory}!Type {
        if (bc.typeNameAt(idx)) |tname| return bc.typeArgsOnValue(site, recv, idx, tname);
        const rt = try bc.typeOf(recv);
        const it = try bc.typeOfExpected(idx, Type.int);
        const vec_ctor = bc.activeStructMap().get(StdNames.vec_struct);
        if (!rt.isApp()) {
            if (rt.kind != .invalid)
                try bc.sink.report(.{ .span = bc.spanOf(site), .code = .T0057 }, "cannot index a value of type '{s}'", .{bc.typeName(rt)});
            return Type.invalid;
        }
        const e = bc.composite.at(rt.appIdx());
        // A miss is a normal outcome for a hashed lookup, so `Map` deliberately has no
        // index operator (a lookup returns `Option`); reject `m[k]` by name before the
        // generic reject, exactly as `Vec` is recognized by name.
        if (bc.activeStructMap().get(StdNames.map_struct)) |map_ctor| {
            if (!e.ctor_is_enum and e.ctor == map_ctor) {
                const use = if (site == bc.place_target) ".set(k, v)" else ".get";
                try bc.sink.report(.{ .span = bc.spanOf(site), .code = .T0057 }, "Map has no index operator; use {s}", .{use});
                return Type.invalid;
            }
        }
        // Only std's `Vec` has the element primitives the sugar lowers to; a user `struct
        // Vec` in scope is just another type.
        if (e.ctor_is_enum or vec_ctor == null or e.ctor != vec_ctor.? or e.args.len != 1 or !bc.isStdVec(e)) {
            // A `Vec` that reached this module through another one (`m.keys()`) still has
            // its methods, but the sugar names `Vec` as this module sees it.
            if (vec_ctor == null and bc.isStdVec(e))
                try bc.sink.report(.{ .span = bc.spanOf(site), .code = .T0066 }, "indexing a 'Vec' needs 'Vec' in scope; " ++ StdNames.add_import, .{"vec"})
            else
                try bc.sink.report(.{ .span = bc.spanOf(site), .code = .T0057 }, "cannot index a value of type '{s}'", .{bc.typeName(rt)});
            return Type.invalid;
        }
        if (it.kind != .int and it.kind != .invalid)
            try bc.sink.report(.{ .span = bc.spanOf(idx), .code = .T0044 }, "an index must be 'int', got '{s}'", .{bc.typeName(it)});
        return e.args[0];
    }

    /// Whether composite `e` is std/vec's `Vec`.
    fn isStdVec(bc: *const BodyChecker, e: Composite.Entry) bool {
        if (e.ctor_is_enum or e.ctor >= bc.model.structs.len) return false;
        const sym = bc.model.structs[e.ctor];
        return std.mem.eql(u8, sym.name, StdNames.vec_struct) and bc.model.graph.mods[sym.mod].bundled;
    }

    /// The type a bracket subscript names, or null when it is a value: a builtin type
    /// name, or a struct, enum or alias of this module, that no local shadows (the
    /// resolver leaves such a subscript unresolved for this check).
    fn typeNameAt(bc: *const BodyChecker, idx: Ast.Index) ?[]const u8 {
        const n = bc.tree.nodes[idx.int()];
        if (n.tag != .identifier or bc.resolutions[idx.int()] != .unresolved) return null;
        const name = bc.nameText(n.main_token);
        if (Typecheck.type_names.has(name) or bc.activeStructMap().contains(name) or
            bc.activeEnumMap().contains(name) or bc.activeAliasMap().contains(name)) return name;
        return null;
    }

    /// `f[int]` where `f` is no generic fn: only a generic fn's call takes type args, and
    /// a function value or a plain fn is not one. A generic fn named as a value is T0049
    /// at its name.
    fn typeArgsOnValue(bc: *BodyChecker, site: Ast.Index, recv: Ast.Index, idx: Ast.Index, tname: []const u8) error{OutOfMemory}!Type {
        const rt = try bc.typeOf(recv);
        if (rt.kind == .invalid) return .invalid;
        if (rt.isFunc()) {
            const sp = bc.spanOf(recv);
            try bc.sink.report(.{ .span = bc.typeArgsSpan(site), .code = .T0054 }, "'{s}' is not generic; drop the type arguments", .{bc.source[sp.start..sp.end]});
        } else {
            try bc.sink.report(.{ .span = bc.spanOf(idx), .code = .T0056 }, "type '{s}' is not a value", .{tname});
        }
        return .invalid;
    }

    /// The struct or enum a type-named node names: a bare `K` the resolver left
    /// unresolved, or a qualified `lib.K`. Both null when the node is neither.
    const TypeIds = struct { s: ?u32 = null, e: ?u32 = null };
    fn typeIdsNamed(bc: *const BodyChecker, idx: Ast.Index) TypeIds {
        const r = bc.tree.nodes[idx.int()];
        switch (r.tag) {
            .identifier => {
                if (bc.resolutions[idx.int()] != .unresolved) return .{};
                const name = bc.nameText(r.main_token);
                return .{ .s = bc.activeStructMap().get(name), .e = bc.activeEnumMap().get(name) };
            },
            .field_access => {
                const inner = bc.tree.nodes[r.lhs.int()];
                if (inner.tag != .identifier or bc.resolutions[r.lhs.int()] != .module) return .{};
                const g = bc.model.graph;
                const target = g.namespaceOfIn(bc.graph_mod, bc.nameText(inner.main_token)) orelse return .{};
                const name = bc.nameText(r.main_token);
                return .{ .s = g.mods[target].struct_ids.get(name), .e = g.mods[target].enum_ids.get(name) };
            },
            else => return .{},
        }
    }

    /// T0056 at `idx`, a struct or enum name used as a value, an enum's with a hint that
    /// names a variant; false when `idx` names neither.
    fn reportTypeNotValue(bc: *BodyChecker, idx: Ast.Index) error{OutOfMemory}!bool {
        const ids = bc.typeIdsNamed(idx);
        const sp = bc.spanOf(idx);
        const name = bc.source[sp.start..sp.end];
        const at: DiagnosticSink.Report = .{ .span = sp, .code = .T0056 };
        // A struct gets no hint: its name as a value is as often a mistaken associated
        // call (`Set.from(..)`) as a missing construction, and a std struct or another
        // module's is not built with `S { .. }` at all.
        if (ids.s != null) {
            try bc.sink.report(at, "type '{s}' is not a value", .{name});
            return true;
        }
        const eid = ids.e orelse return false;
        const variants = bc.model.enums[eid].variants;
        if (variants.len == 0) {
            try bc.sink.report(at, "type '{s}' is not a value", .{name});
            return true;
        }
        const shape = switch (variants[0].form) {
            .unit => "",
            .tuple => "(..)",
            .@"struct" => " { .. }",
        };
        try bc.sink.report(at, "type '{s}' is not a value; name one of its variants, as in '{s}.{s}{s}'", .{ name, name, variants[0].name, shape });
        return true;
    }

    pub fn typeOf(bc: *BodyChecker, node_idx: Ast.Index) error{OutOfMemory}!Type {
        if (node_idx == Ast.none) return .invalid; // structural poison: no emit (exempt)
        const n = bc.tree.nodes[(node_idx).int()];
        const ty: Type = switch (n.tag) {
            .literal_number => try bc.typeNumericLiteral(n.main_token, bc.literalExpected(n.main_token)),
            .literal_float => Type.float,
            .literal_bool => Type.bool,
            .literal_string => Type.str,
            .literal_char => try bc.typeCharLiteral(n.main_token),
            .identifier => switch (bc.resolutions[(node_idx).int()]) {
                .local => |slot| bc.slotType(slot),
                .func => |gid| try bc.typeOfFnValue(node_idx, gid, n.main_token),
                .unresolved => blk: {
                    // Resolve quietly skips a struct- or enum-named identifier (a type
                    // name in a position it doesn't bind: a constructor, a variant or an
                    // associated call). Used BARE as a value (`q := P`, `id(Shape)`) it
                    // reaches here with no diagnostic, so report it here or it reaches
                    // lowering.
                    if (try bc.reportTypeNotValue(node_idx)) break :blk bc.poison();
                    // Else: an ordinary undeclared name already reported by resolve
                    // (this branch emits nothing) — keep plain poison, no guard.
                    break :blk Type.invalid;
                },
                .label => Type.invalid, // never on an identifier node (break/continue only)
                .module => blk: {
                    // A bare imported-namespace name used as a value (`x := mod`):
                    // a module is not a value. (A `mod.member` access never reaches
                    // here — the receiver is consumed by typeOfFieldAccess/Call.)
                    try bc.sink.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0056 }, "module '{s}' is not a value", .{bc.nameText(n.main_token)});
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
                    if (bc.expectedTy()) |e| if (e.isSigned()) {
                        const lit_tok = bc.tree.nodes[n.lhs.int()].main_token;
                        const raw = bc.tokens[lit_tok].text(bc.source);
                        if (!Literal.fitsWidth(raw, e, true))
                            try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0034 }, "literal out of range for type '{s}'", .{bc.typeName(e)});
                        bc.node_types[n.lhs.int()] = e;
                        break :blk e;
                    };
                }
                const operand = try bc.typeOf(n.lhs);
                if (operand.kind == .invalid) break :blk Type.invalid; // poison propagation: no emit (exempt)
                switch (op) {
                    .minus => {
                        if (operand.isInteger()) break :blk operand;
                        try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0044 }, "operand of '-' must be int", .{});
                    },
                    .bang => {
                        if (operand.kind == .bool) break :blk Type.bool;
                        try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0044 }, "operand of '!' must be bool", .{});
                    },
                    .tilde => {
                        if (operand.isInteger()) break :blk operand;
                        try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0044 }, "operand of '~' must be int", .{});
                    },
                    else => {},
                }
                break :blk Type.invalid;
            },
            .binary => blk: {
                // Type each operand with NO expected type (the only `typeOf` consumer of
                // `bc.body.expected` is the literal arm, so scoping it off here is safe), then
                // let a bare-literal operand adopt its typed sibling's width — so `a + 1`
                // (a: int8) adopts int8 and `a + 300` fires T0034 exactly once via the
                // sibling path, never also via the outer expected.
                const saved = bc.body.expected;
                bc.body.expected = null;
                var lt = try bc.typeOf(n.lhs);
                var rt = try bc.typeOf(n.rhs);
                bc.body.expected = saved;
                if (lt.kind == .invalid or rt.kind == .invalid) break :blk Type.invalid; // poison propagation: no emit (exempt)
                const l_lit = bc.tree.nodes[n.lhs.int()].tag == .literal_number;
                const r_lit = bc.tree.nodes[n.rhs.int()].tag == .literal_number;
                if (!l_lit) rt = try bc.adoptLiteralWidth(n.rhs, rt, lt);
                if (!r_lit) lt = try bc.adoptLiteralWidth(n.lhs, lt, rt);
                // Both operands bare literals (`c: int8 = 1 + 2`): no typed sibling to
                // adopt from, so re-type each under the OUTER expected width (running the
                // per-literal range check) — else both stay platform `int`, the homogeneous
                // arm yields `int`, and the annotated bind is wrongly rejected. A no-op for
                // a platform-`int` target, so `x: int = 1 + 2` is unaffected.
                if (l_lit and r_lit) if (saved) |e| if (e.ty.isInteger()) {
                    lt = try bc.typeOfExpected(n.lhs, e.ty);
                    rt = try bc.typeOfExpected(n.rhs, e.ty);
                };
                const op_tok = bc.tokens[n.main_token];
                break :blk try bc.binaryResult(node_idx, .{ .node = node_idx }, op_tok.tag, op_tok.text(bc.source), lt, rt);
            },
            .empty_list => blk: {
                // An empty list literal `[]` is typed bidirectionally by its expected
                // type: a `Vec[T]` annotation makes it the type's associated `new()`
                // constructor. With no expected type (a bare `xs := []`) the element type
                // is uninferable — require an annotation.
                const exp = bc.expectedTy() orelse {
                    try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0016 }, "an empty list literal '[]' requires a type annotation, e.g. 'xs: Vec[int] = []'", .{});
                    break :blk Type.invalid;
                };
                if (!exp.isApp()) {
                    try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0065 }, "an empty list literal '[]' cannot produce '{s}'", .{bc.typeName(exp)});
                    break :blk Type.invalid;
                }
                const e = bc.composite.at(exp.appIdx());
                if (e.ctor_is_enum or Typecheck.findGenericMethod(bc.model.templates, e.ctor, false, StdNames.method_new, .program) == null) {
                    try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0065 }, "an empty list literal '[]' cannot produce '{s}'", .{bc.typeName(exp)});
                    break :blk Type.invalid;
                }
                break :blk exp;
            },
            .list_literal => blk: {
                // A non-empty `[e0, ..]` builds a populated `Vec[V]`: V is inferred from
                // the first element (seeded by a `Vec[W]` annotation when present); every
                // remaining element must be assignable to V.
                const elems = Ast.rangeSlice(bc.tree, n.rhs.int());
                const vec_ctor = bc.activeStructMap().get(StdNames.vec_struct) orelse {
                    for (elems) |ei| _ = try bc.typeOf(ei);
                    try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0066 }, "a list literal needs 'Vec' in scope; " ++ StdNames.add_import, .{"vec"});
                    break :blk Type.invalid;
                };
                var seed: ?Type = null;
                if (bc.expectedTy()) |exp| if (exp.isApp()) {
                    const ce = bc.composite.at(exp.appIdx());
                    if (ce.ctor == vec_ctor and ce.args.len == 1) seed = ce.args[0];
                };
                const v_ty = try bc.typeOfExpected(elems[0], seed);
                const result = Type.app(bc.internApp(vec_ctor, &.{v_ty}, false) catch break :blk Type.invalid);
                for (elems[1..], 1..) |ei, k| {
                    const at = try bc.typeOfExpected(ei, v_ty);
                    if (!Type.assignable(v_ty, at))
                        try bc.sink.report(.{ .span = bc.spanOf(ei), .code = .T0065 }, "list element {d}: expected '{s}', got '{s}'", .{ k, bc.typeName(v_ty), bc.typeName(at) });
                }
                if (bc.body.expected) |exp| if (!Type.eql(result, exp.ty) and !exp.soft) {
                    if (seed) |s|
                        try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0065 }, "list element type '{s}' does not match the annotation's '{s}'", .{ bc.typeName(v_ty), bc.typeName(s) })
                    else
                        try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0065 }, "a list literal of type '{s}' cannot produce '{s}'", .{ bc.typeName(result), bc.typeName(exp.ty) });
                    break :blk Type.invalid;
                };
                break :blk result;
            },
            .index => try bc.typeOfIndexParts(node_idx, n.lhs, n.rhs),
            // A `type_app` reaches `typeOf` as a standalone expr only when the parser
            // could not tell a value index `xs[i]` from a turbofish and the head turned
            // out to resolve to a value; reinterpret it as an index. A real type-app is
            // consumed by call/struct_init/field_access and never lands here.
            .type_app => blk: {
                if (!bc.appHeadIsValue(node_idx)) break :blk Type.invalid;
                const targs = Ast.rangeSlice(bc.tree, (n.rhs).int());
                if (targs.len > 0) if (bc.typeNameAt(targs[0])) |tname| break :blk try bc.typeArgsOnValue(node_idx, n.lhs, targs[0], tname);
                if (targs.len != 1) {
                    try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0057 }, "cannot index a value with {d} subscripts", .{targs.len});
                    break :blk Type.invalid;
                }
                break :blk try bc.typeOfIndexParts(node_idx, n.lhs, targs[0]);
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
            .lambda => return LambdaChecker.typeOfLambda(bc, node_idx, n), // sets node_types itself
            // A poison leaf types as the poison `invalid` — already-diagnosed, so
            // no diagnostic here and no cascade (`assignable` absorbs `.invalid`).
            .error_node => Type.invalid, // already diagnosed in the parser (exempt)
            else => Type.invalid, // defensive unknown-kind fallthrough: no emit (exempt)
        };
        bc.node_types[(node_idx).int()] = ty;
        return ty;
    }

    fn typeOfStructInit(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        const lhs = bc.tree.nodes[(n.lhs).int()];
        // A `type_app` lhs `Box[int] { .. }` names a generic-struct INSTANCE — a
        // composite `App` whose type-args the template's field PATTERNS are substituted
        // through. Otherwise the lhs is a plain type-name identifier: a qualified
        // `N.V { ... }` arrives as enum_init_struct (the parser upgrades a
        // `field_access {`), not here.
        if (lhs.tag == .type_app and !bc.appHeadIsValue(n.lhs)) {
            const app_ty = bc.typeFromNode(n.lhs); // App or invalid (already diagnosed)
            if (!app_ty.isApp()) {
                for (Ast.rangeSlice(bc.tree, (n.rhs).int())) |fi| _ = try bc.typeUnder(bc.tree.nodes[(fi).int()].lhs, bc.partHint());
                return .invalid;
            }
            const e = bc.composite.at(app_ty.appIdx());
            const sym = bc.model.structs[e.ctor];
            return bc.checkStructFieldInits(node_idx, n, sym, e.args, sym.name, app_ty, null);
        }
        const id = bc.activeStructMap().get(bc.nameText(lhs.main_token)) orelse {
            for (Ast.rangeSlice(bc.tree, (n.rhs).int())) |fi| _ = try bc.typeUnder(bc.tree.nodes[(fi).int()].lhs, bc.partHint());
            try bc.sink.report(.{ .span = bc.tokSpan(lhs.main_token), .code = .T0001 }, "unknown struct type '{s}'", .{bc.nameText(lhs.main_token)});
            return .invalid;
        };
        return bc.typeOfNamedStructInit(node_idx, n, id, lhs.main_token);
    }

    /// A struct literal with no type args, of struct `id` named at `name_tok`: a bare
    /// `P { .. }` or a module-qualified `lib.P { .. }`.
    fn typeOfNamedStructInit(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, id: u32, name_tok: u32) error{OutOfMemory}!Type {
        const disp_name = bc.nameText(name_tok);
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
            // Align values to decl order + record each field's first supplier (for
            // conflict spans). A missing field's slot stays `.invalid`/`Ast.none`: the
            // matcher's never/invalid rule skips it and its span is never dereferenced.
            const aligned = try bc.gpa.alloc(Type, gsym.field_names.len);
            defer bc.gpa.free(aligned);
            @memset(aligned, Type.invalid);
            const supplier = try bc.gpa.alloc(Ast.Index, gsym.field_names.len);
            defer bc.gpa.free(supplier);
            @memset(supplier, Ast.none);
            const exp_args = bc.expectedAppArgs(id, false);
            for (inits, 0..) |fi_idx, ii| {
                const fi = bc.tree.nodes[(fi_idx).int()];
                const j = fieldIndex(gsym.field_names, bc.nameText(fi.main_token));
                pre[ii] = try bc.typeUnder(fi.lhs, if (j) |k| bc.partWant(gsym.field_types[k], exp_args) else bc.partHint());
                if (j) |k| if (supplier[k] == Ast.none) {
                    supplier[k] = fi_idx;
                    aligned[k] = pre[ii];
                };
            }
            const n_gp: u32 = @intCast(gsym.generic_params.len);
            const out = try bc.gpa.alloc(Type, n_gp);
            defer bc.gpa.free(out);
            const bnd = try bc.gpa.alloc(bool, n_gp);
            defer bc.gpa.free(bnd);
            const fp = try bc.gpa.alloc(usize, n_gp);
            defer bc.gpa.free(fp);
            switch (Infer.match(n_gp, gsym.field_types, aligned, out, bnd, fp, bc.composite.decomposer())) {
                // A field-vs-field conflict is authoritative, reported at the two
                // supplier spans — the target type never overrides it.
                .conflict => |c| {
                    // Name the clashing LEAF types (`c.prev`/`c.cur`) — a container
                    // field clash names `int` vs `bool`, not the container ctor. An
                    // intra-field clash (`Both[T,T]`) shares one supplier span, so
                    // collapse to a single caret rather than double-underlining it.
                    if (c.first_pos == c.second_pos) {
                        try bc.sink.report(.{ .span = bc.spanOf(supplier[c.second_pos]), .code = .T0015 }, "conflicting types for type parameter '{s}': {s} vs {s}", .{ gsym.generic_params[c.ord], bc.typeName(c.prev), bc.typeName(c.cur) });
                    } else {
                        const earlier = bc.tree.nodes[(supplier[c.first_pos]).int()].main_token;
                        try bc.sink.report(.{ .span = bc.spanOf(supplier[c.second_pos]), .code = .T0015, .related = bc.tokSpan(earlier) }, "conflicting types for type parameter '{s}': {s} vs {s}", .{ gsym.generic_params[c.ord], bc.typeName(c.prev), bc.typeName(c.cur) });
                    }
                    return .invalid;
                },
                // Target-fill any still-open param from the expected type
                // (`p: Phantom[int] = Phantom{..}`); an arg-vs-expected disagreement
                // surfaces as T0015. Byte-identical when no target exists.
                .ok, .unbound => switch (try bc.reconcileTargetArgs(out, bnd, exp_args, gsym.generic_params, name_tok)) {
                    .ok => {
                        const app = Type.app(try bc.internApp(id, out, false));
                        return bc.checkStructFieldInits(node_idx, n, gsym, out, disp_name, app, pre);
                    },
                    .err => return .invalid,
                    .unbound => |ord| {
                        for (out, bnd) |*o, b| if (!b) {
                            o.* = .invalid;
                        };
                        try bc.reportUninferable(bc.tokSpan(name_tok), " '{s}' for '{s}'", .{ gsym.generic_params[ord], disp_name }, disp_name, out, &.{"{ .. }"}, null);
                        return .invalid;
                    },
                },
            }
        }
        return bc.checkStructFieldInits(node_idx, n, bc.model.structs[id], &.{}, disp_name, Type.structT(id), null);
    }

    /// The field-init check shared by all three struct-construction paths (non-generic
    /// plain, explicit `Box[int]{..}`, and the inferred `Box{..}`). When `pretyped`
    /// is non-null it supplies the already-computed value type per init (source order),
    /// so the inferred path — which must type each value ONCE to run inference — does not
    /// re-walk (and thus re-diagnose) the value expressions; `pretyped == null` types
    /// each value here and is byte-identical to the earlier tail.
    fn checkStructFieldInits(
        bc: *BodyChecker,
        node_idx: Ast.Index,
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
            // Find the declared field by name.
            var found: ?usize = null;
            for (sym.field_names, 0..) |dn, j| {
                if (std.mem.eql(u8, dn, fname)) {
                    found = j;
                    break;
                }
            }
            // Substitute the (possibly generic) declared field type through the
            // instance's type-args; a non-generic struct has `targs.len == 0`, so
            // `substTy` is the identity.
            const want: ?Type = if (found) |j| substTy(bc, sym.field_types[j], targs) else null;
            // A field value is typed against its field, never against the struct's own
            // expected type (`s: S = S { tags: ["y"] }`). A fn-typed field hands its type down
            // even when it holds a type var: a lambda's params come from it.
            const field_want: ?Type = if (want) |w| (if (w.isFunc() or (w.kind != .invalid and bc.givesType(w))) w else null) else null;
            const vt = if (pretyped) |pt| pt[ii] else if (field_want) |w| try bc.typeOfExpected(fi.lhs, w) else try bc.typeUnder(fi.lhs, bc.partHint());
            if (found) |j| {
                if (seen[j]) {
                    try bc.sink.report(.{ .span = bc.tokSpan(fi.main_token), .code = .T0064 }, "duplicate field '{s}' in '{s}'", .{ fname, disp_name });
                }
                seen[j] = true;
                const fty = want.?;
                if (!Type.assignable(fty, vt)) {
                    try bc.sink.report(.{ .span = bc.spanOf(fi.lhs), .code = .T0041 }, "field '{s}': expected {s}, got {s}", .{ fname, bc.typeName(fty), bc.typeName(vt) });
                }
            } else {
                try bc.sink.report(.{ .span = bc.tokSpan(fi.main_token), .code = .T0046 }, "unknown field '{s}' in '{s}'", .{ fname, disp_name });
            }
        }
        for (sym.field_names, 0..) |dn, j| {
            if (!seen[j]) try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0047 }, "missing field '{s}' in '{s}'", .{ dn, disp_name });
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
        // A qualified `mod.f` in value position: the resolver bound the access itself.
        switch (bc.resolutions[(node_idx).int()]) {
            .func => |gid| return bc.typeOfFnValue(node_idx, gid, n.main_token),
            else => {},
        }
        if (try bc.reportTypeNotValue(node_idx)) return bc.poison();
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
            const ty = try bc.checkVariant(node_idx, enum_id, n.main_token, .unit, Ast.none);
            bc.node_types[(node_idx).int()] = ty;
            return ty;
        }
        // An explicit generic-enum UNIT-variant `Opt[int].none`: the receiver is a
        // `type_app` resolving to an enum-`App`. Construct the unit variant through the
        // App's args and type the node as the `App` (reified to `enumT` in the mono tail).
        // Placed BEFORE the value `typeOf(n.lhs)` so a `type_app` receiver never mis-routes
        // into the struct-field-access path.
        if (recv.tag == .type_app and !bc.appHeadIsValue(n.lhs)) {
            const app_ty = bc.typeFromNode(n.lhs);
            if (app_ty.isApp() and bc.composite.at(app_ty.appIdx()).ctor_is_enum) {
                const e = bc.composite.at(app_ty.appIdx());
                const ty = try bc.checkVariantPayloads(node_idx, e.ctor, n.main_token, .unit, Ast.none, e.args, null);
                bc.node_types[(node_idx).int()] = ty;
                return ty;
            }
            // `Box[int]` names a type, not a value: an associated fn of it is reached by a
            // call only.
            if (app_ty.isApp()) {
                const e = bc.composite.at(app_ty.appIdx());
                const member = bc.nameText(n.main_token);
                if (Typecheck.findGenericMethod(bc.model.templates, e.ctor, false, member, bc.methodSite())) |m| if (!m.has_self)
                    return try bc.reportNotAValue(n, member, "an associated function");
                try bc.sink.report(.{ .span = bc.spanOf(n.lhs), .code = .T0056 }, "type '{s}' is not a value", .{bc.typeName(app_ty)});
                return bc.poison();
            }
        }
        const base = try bc.typeOf(n.lhs);
        if (base.kind == .invalid) return .invalid;
        // A field access over a generic-struct INSTANCE `b.v` where `b: Box[int]`:
        // the base is a composite `App`; resolve the field's DECLARED (pattern) type
        // through the App's type-args so `v` on `Box[int]` types as `int`.
        if (base.isApp()) {
            const e = bc.composite.at(base.appIdx());
            if (e.ctor_is_enum) if (try bc.tryBoundMethod(node_idx, n, base)) |t| return t;
            const sym = bc.model.structs[e.ctor];
            const fname = bc.nameText(n.main_token);
            for (sym.field_names, 0..) |dn, j| {
                if (std.mem.eql(u8, dn, fname)) return substTy(bc, sym.field_types[j], e.args);
            }
            if (try bc.tryBoundMethod(node_idx, n, base)) |t| return t;
            if (bc.rejectedMethod(base, fname)) return .invalid;
            try bc.emitNoField(bc.tokSpan(n.main_token), fname, sym.name, sym.field_names);
            return .invalid;
        }
        if (!base.isStruct()) {
            if (try bc.tryBoundMethod(node_idx, n, base)) |t| return t;
            try bc.sink.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0046 }, "cannot access field '{s}' of non-struct type {s}", .{ bc.nameText(n.main_token), bc.typeName(base) });
            return .invalid;
        }
        const sym = bc.model.structs[base.struct_id];
        const fname = bc.nameText(n.main_token);
        for (sym.field_names, 0..) |dn, j| {
            if (std.mem.eql(u8, dn, fname)) return sym.field_types[j];
        }
        if (try bc.tryBoundMethod(node_idx, n, base)) |t| return t;
        try bc.emitNoField(bc.tokSpan(n.main_token), fname, sym.name, sym.field_names);
        return .invalid;
    }

    /// `recv.m` in value position where `m` is no field: a bound method, the closure that
    /// captures the receiver and calls `recv.m(..)`. It types as the method's fn type
    /// without `self`, and its selection sits on the `field_access` itself. Null when `m`
    /// names no method, so the caller reports the missing field. A method the compiler
    /// expands at each call (a builtin, a native `Option`/`Result` method, a protocol
    /// method of a type parameter) has no code a value could point at.
    fn tryBoundMethod(bc: *BodyChecker, node_idx: Ast.Index, fa: Ast.Node, recv_ty: Type) error{OutOfMemory}!?Type {
        if (node_idx == bc.place_target) return null;
        const member = bc.nameText(fa.main_token);
        if (recv_ty.kind == .@"struct" or recv_ty.kind == .@"enum" or recv_ty.isScalar() or recv_ty.kind == .float) {
            switch (Typecheck.pickMethodIndex(bc.model.methods, recv_ty, member, null, null, bc.methodSite())) {
                .one => |mi| {
                    const m = bc.model.methods[mi];
                    if (m.is_static) return try bc.reportStaticBound(fa, member);
                    const mf = bc.model.fns[m.fn_id];
                    if (mf.isGeneric()) return try bc.reportNotAValue(fa, member, "a generic method");
                    if (mf.mut_mask >> 1 != 0) return try bc.reportNotAValue(fa, member, "a method with 'mut' parameters");
                    if (!try bc.bindReceiver(node_idx, fa, recv_ty, member, m.mut_self)) return bc.poison();
                    const ty = refs.internFn(bc, mf.params[@min(mf.params.len, 1)..], mf.ret);
                    if (ty.kind == .invalid) return error.OutOfMemory;
                    bc.setCalleeAt(node_idx, Callee.boundMethod(mi));
                    return ty;
                },
                .ambiguous => {
                    try bc.emitAmbiguousConformance(bc.tokSpan(fa.main_token), recv_ty, member, fa.lhs);
                    return bc.poison();
                },
                .none => {},
            }
            if (bc.isBuiltinMethod(recv_ty, member)) return try bc.reportNotAValue(fa, member, "a built-in method");
            return null;
        }
        if (recv_ty.isApp()) {
            const e = bc.composite.at(recv_ty.appIdx());
            const picked: ?Typecheck.TemplateMethod = switch (Typecheck.pickGenericMethod(bc.model.templates, e.ctor, e.ctor_is_enum, member, bc.methodSite())) {
                .one => |m| m,
                .ambiguous => {
                    try bc.emitAmbiguousConformance(bc.tokSpan(fa.main_token), recv_ty, member, fa.lhs);
                    return bc.poison();
                },
                .none => null,
            };
            if (picked) |m| {
                if (!m.has_self) return try bc.reportStaticBound(fa, member);
                const mf = bc.model.fns[m.fn_id];
                // A method's own `[U]` is fixed per call, which a value could not do.
                if (Typecheck.hasOwnTypeParams(bc.composite, mf)) return try bc.reportNotAValue(fa, member, "a generic method");
                const targs = (try Typecheck.bindImplParams(bc.gpa, bc.composite, mf, e.args)) orelse
                    return try bc.reportNotAValue(fa, member, "a generic method");
                defer bc.gpa.free(targs);
                if (mf.mut_mask >> 1 != 0) return try bc.reportNotAValue(fa, member, "a method with 'mut' parameters");
                if (!try bc.bindReceiver(node_idx, fa, recv_ty, member, m.mut_self)) return bc.poison();
                const ps = mf.params[@min(mf.params.len, 1)..];
                const sub = try bc.gpa.alloc(Type, ps.len);
                defer bc.gpa.free(sub);
                for (ps, sub) |p, *o| o.* = substTy(bc, p, targs);
                const ty = refs.internFn(bc, sub, substTy(bc, mf.ret, targs));
                if (ty.kind == .invalid) return error.OutOfMemory;
                bc.setCalleeAt(node_idx, Callee.pending_bound);
                return ty;
            }
            const native = e.ctor_is_enum and Typecheck.optionResultMethod(Typecheck.optResultFamilyOf(bc.model, e.ctor), member) != null;
            const display = std.mem.eql(u8, member, "to_string") and bc.model.preludeProtocols().display != null;
            if (native or display) return try bc.reportNotAValue(fa, member, "a built-in method");
            return null;
        }
        if (recv_ty.isTypeVar()) {
            const ord = recv_ty.typeVarOrd();
            const pid = (if (ord < bc.bound_protocols.len) bc.bound_protocols[ord] else null) orelse return null;
            for (bc.model.protocols[pid].methods) |mn| if (std.mem.eql(u8, mn, member))
                return try bc.reportNotAValue(fa, member, "a protocol method of a type parameter");
            if (bc.model.preludeProtocols().display) |d| if (d == pid and std.mem.eql(u8, member, "to_string"))
                return try bc.reportNotAValue(fa, member, "a built-in method");
            return null;
        }
        return null;
    }

    /// Whether `member` on concrete `recv` is a method the compiler provides with no
    /// `methods` row: a scalar or `str` builtin, a target-directed conversion, or the
    /// derivable `hash`/`to_string` of a struct or enum.
    fn isBuiltinMethod(bc: *const BodyChecker, recv: Type, member: []const u8) bool {
        const char_id = if (bc.model.prelude) |p| p.char_struct else null;
        if (Typecheck.isBuiltinTypeMethod(recv, member, char_id)) return true;
        if (recv.kind != .@"struct" and recv.kind != .@"enum") return false;
        const pp = bc.model.preludeProtocols();
        return (std.mem.eql(u8, member, "hash") and pp.hash != null) or (std.mem.eql(u8, member, "to_string") and pp.display != null);
    }

    fn reportNotAValue(bc: *BodyChecker, fa: Ast.Node, member: []const u8, what: []const u8) error{OutOfMemory}!Type {
        try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0049 }, "'{s}' is {s} and cannot be used as a value; wrap the call in a lambda", .{ member, what });
        return bc.poison();
    }

    fn reportStaticBound(bc: *BodyChecker, fa: Ast.Node, member: []const u8) error{OutOfMemory}!Type {
        try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0018 }, "'{s}' is an associated function, not a method; it has no receiver to bind", .{member});
        return bc.poison();
    }

    /// Check the receiver of bound method `member` and record it as captured. A `mut self`
    /// method writes its receiver through the closure, so the receiver must be a local
    /// variable, which then lives in a box; a copy of a field or of a temporary would drop
    /// every write. Returns false after reporting.
    fn bindReceiver(bc: *BodyChecker, node_idx: Ast.Index, fa: Ast.Node, recv_ty: Type, member: []const u8, mut_self: bool) error{OutOfMemory}!bool {
        const recv = fa.lhs;
        const local = Closure.boundRecvLocal(bc.tree, bc.resolutions, node_idx);
        if (mut_self) {
            if (recv_ty.kind != .@"struct" and recv_ty.kind != .@"enum" and !recv_ty.isApp()) {
                try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0022 }, "mutating method '{s}' is not supported on the builtin type '{s}'; `mut self` is only allowed on struct and enum receivers", .{ member, bc.typeName(recv_ty) });
                return false;
            }
            if (!bc.isVariablePlace(recv)) {
                const what = if (bc.elementRoot(recv) != null) "a Vec element" else "a temporary";
                try bc.sink.report(.{ .span = bc.spanOf(recv), .code = .T0019 }, "cannot bind mutating method '{s}' to {s}; the receiver must be a local variable", .{ member, what });
                return false;
            }
            const slot = local orelse {
                try bc.sink.report(.{ .span = bc.spanOf(recv), .code = .T0072 }, "binding the 'mut self' method '{s}' needs a local receiver, not a field; bind it on a local variable", .{member});
                return false;
            };
            try bc.mutated_roots.append(bc.gpa, slot);
        }
        if (local) |slot| try bc.bound_roots.append(bc.gpa, slot);
        return true;
    }

    fn typeOfTupleField(bc: *BodyChecker, n: Ast.Node) error{OutOfMemory}!Type {
        const base = try bc.typeOf(n.lhs);
        if (base.kind == .invalid) return .invalid; // no cascade on an already-poisoned base
        if (!base.isStruct() or !bc.model.structs[base.struct_id].is_tuple) {
            try bc.sink.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0046 }, "'.{s}' positional access requires a tuple struct, got {s}", .{ bc.nameText(n.main_token), bc.typeName(base) });
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
        try bc.sink.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0046 }, "tuple struct '{s}' has no field .{s}", .{ sym.name, name });
        return .invalid;
    }

    fn typeOfEnumInit(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
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
                    return bc.checkVariantPayloads(node_idx, e.ctor, n.main_token, node_form, args, e.args, null);
                }
                // invalid App (already diagnosed) or a struct-App: type args + poison.
                try bc.typeArgsForEffect(node_form, args);
                return .invalid;
            }
            // The parser cannot tell a module head from an enum, so `lib.Pt { .. }`
            // lands here too; the resolver bound its head to the module.
            if (lhs_node.tag == .identifier and bc.resolutions[n.lhs.int()] == .module)
                return bc.typeOfModuleStructInit(node_idx, n, node_form, args);
            const found = switch (lhs_node.tag) {
                .field_access => bc.qualifiedEnumId(n.lhs),
                else => bc.activeEnumMap().get(bc.nameText(lhs_node.main_token)),
            };
            enum_id = found orelse {
                try bc.typeArgsForEffect(node_form, args);
                const sp = bc.spanOf(n.lhs);
                try bc.sink.report(.{ .span = sp, .code = .T0062 }, "'{s}' is not an enum type", .{bc.source[sp.start..sp.end]});
                return .invalid;
            };
        } else {
            const exp = bc.expectedTy() orelse {
                try bc.typeArgsForEffect(node_form, args);
                try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0062 }, "cannot infer the enum type for '.{s}' here", .{bc.nameText(n.main_token)});
                return .invalid;
            };
            // The expected type may be a plain `enumT` or a generic-enum instance `App`
            //. `checkVariant` infers the enum's type-params from the payload for the
            // generic case (a nullary `.none` under a generic expected is uninferable ->
            // T0016; target-typing it is deferred).
            enum_id = bc.scrutEnumId(exp) orelse {
                try bc.typeArgsForEffect(node_form, args);
                if (exp.kind != .invalid)
                    try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0062 }, "'.{s}' expects an enum type, but {s} was expected here", .{ bc.nameText(n.main_token), bc.typeName(exp) });
                return .invalid;
            };
        }
        return bc.checkVariant(node_idx, enum_id, n.main_token, node_form, args);
    }

    /// `mod.S { .. }`: a literal of the struct `S` of module `mod`, reached as a
    /// variant construction whose head the resolver bound to that module.
    fn typeOfModuleStructInit(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, node_form: InitForm, args: Ast.Index) error{OutOfMemory}!Type {
        if (node_form == .@"struct") {
            const target = bc.resolutions[n.lhs.int()].module;
            if (bc.model.graph.mods[target].struct_ids.get(bc.nameText(n.main_token))) |id|
                return bc.typeOfNamedStructInit(node_idx, n, id, n.main_token);
        }
        try bc.typeArgsForEffect(node_form, args);
        const sp: Span = .{ .start = bc.spanOf(n.lhs).start, .end = bc.tokSpan(n.main_token).end };
        try bc.sink.report(.{ .span = sp, .code = .T0001 }, "unknown struct type '{s}'", .{bc.source[sp.start..sp.end]});
        return .invalid;
    }

    fn typeOfEnumInitQualified(bc: *BodyChecker, node_idx: Ast.Index, node_form: InitForm, type_node: Ast.Index, vtok: u32, args: Ast.Index) error{OutOfMemory}!Type {
        const tname = bc.nameText(bc.tree.nodes[(type_node).int()].main_token);
        const enum_id = bc.activeEnumMap().get(tname) orelse return .invalid; // caller checked
        const ty = try bc.checkVariant(node_idx, enum_id, vtok, node_form, args);
        bc.node_types[(node_idx).int()] = ty;
        return ty;
    }

    fn typeArgsForEffect(bc: *BodyChecker, node_form: InitForm, args: Ast.Index) error{OutOfMemory}!void {
        if (args == Ast.none) return;
        if (node_form == .@"struct") {
            for (Ast.rangeSlice(bc.tree, (args).int())) |fi| _ = try bc.typeUnder(bc.tree.nodes[(fi).int()].lhs, bc.partHint());
        } else {
            for (Ast.rangeSlice(bc.tree, (args).int())) |a| _ = try bc.typeUnder(a, bc.partHint());
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
    fn checkVariant(bc: *BodyChecker, site: Ast.Index, enum_id: u32, vtok: u32, node_form: InitForm, args: Ast.Index) error{OutOfMemory}!Type {
        const e = bc.model.enums[enum_id];
        if (!e.is_generic) return bc.checkVariantPayloads(site, enum_id, vtok, node_form, args, &.{}, null);

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
            try bc.emitNoVariant(bc.tokSpan(vtok), e.name, vname, e.variants);
            return .invalid;
        };

        // Type each payload value ONCE (source order) so inference sees the value types;
        // pass them to `checkVariantPayloads` as `pretyped` so payload expressions are not
        // re-walked (and inner diagnostics not double-emitted).
        const arg_nodes: []const Ast.Index = if (args == Ast.none) &.{} else Ast.rangeSlice(bc.tree, (args).int());
        const pre = try bc.gpa.alloc(Type, arg_nodes.len);
        defer bc.gpa.free(pre);
        const exp_args = bc.expectedAppArgs(enum_id, true);
        const fits = formMatches(node_form, variant.form);
        for (arg_nodes, 0..) |a_idx, ii| {
            if (node_form == .@"struct") {
                const fi = bc.tree.nodes[(a_idx).int()];
                const j = if (fits) fieldIndex(variant.field_names, bc.nameText(fi.main_token)) else null;
                pre[ii] = try bc.typeUnder(fi.lhs, if (j) |k| bc.partWant(variant.field_types[k], exp_args) else bc.partHint());
            } else {
                const pat: ?Type = if (fits and ii < variant.field_types.len) variant.field_types[ii] else null;
                pre[ii] = try bc.typeUnder(a_idx, if (pat) |t| bc.partWant(t, exp_args) else bc.partHint());
            }
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
        const supplier = try bc.gpa.alloc(Ast.Index, variant.field_types.len);
        defer bc.gpa.free(supplier);
        @memset(supplier, site);
        if (fits) {
            if (want_struct) {
                for (arg_nodes, 0..) |fi_idx, ii| {
                    const fname = bc.nameText(bc.tree.nodes[(fi_idx).int()].main_token);
                    for (variant.field_names, 0..) |dn, j| {
                        if (std.mem.eql(u8, dn, fname)) {
                            if (aligned[j].kind == .invalid) {
                                aligned[j] = pre[ii];
                                supplier[j] = fi_idx;
                            }
                            break;
                        }
                    }
                }
            } else {
                for (arg_nodes, 0..) |a_idx, ii| {
                    if (ii < aligned.len) {
                        aligned[ii] = pre[ii];
                        supplier[ii] = a_idx;
                    }
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
        switch (Infer.match(n_gp, variant.field_types, aligned, out, bnd, fp, bc.composite.decomposer())) {
            // An arg-vs-arg conflict (two payload values disagree) is authoritative and
            // reported at the payload spans — the target type never overrides it.
            .conflict => |c| {
                // Name the clashing LEAF types (`c.prev`/`c.cur`), not the container ctor.
                try bc.sink.report(.{ .span = bc.spanOf(supplier[c.second_pos]), .code = .T0015 }, "conflicting types for type parameter '{s}': {s} vs {s}", .{ e.generic_params[c.ord], bc.typeName(c.prev), bc.typeName(c.cur) });
                return .invalid;
            },
            // Both a full arg-bind (`.ok`) and a partial/nullary bind (`.unbound`) funnel
            // through the target-fill: the expected type (`x: Opt[int] = Opt.none`) pins
            // any still-open param, and an arg-vs-expected disagreement surfaces as T0015.
            .ok, .unbound => switch (try bc.reconcileTargetArgs(out, bnd, exp_args, e.generic_params, vtok)) {
                .ok => return bc.checkVariantPayloads(site, enum_id, vtok, node_form, args, out, pre),
                .err => return .invalid,
                .unbound => |ord| {
                    for (out, bnd) |*o, b| if (!b) {
                        o.* = .invalid;
                    };
                    try bc.reportUninferable(bc.tokSpan(vtok), " '{s}' for '{s}.{s}'", .{ e.generic_params[ord], e.name, vname }, e.name, out, &.{ ".", vname }, null);
                    return .invalid;
                },
            },
        }
    }

    fn fieldIndex(names: []const []const u8, name: []const u8) ?usize {
        for (names, 0..) |dn, j| if (std.mem.eql(u8, dn, name)) return j;
        return null;
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
    /// re-walking payload values the inference path already typed. Not routed through
    /// `checkCall`: a payload is "values" of a variant, not arguments of a callee, and the
    /// struct form matches fields by name.
    fn checkVariantPayloads(bc: *BodyChecker, site: Ast.Index, enum_id: u32, vtok: u32, node_form: InitForm, args: Ast.Index, targs: []const Type, pretyped: ?[]const Type) error{OutOfMemory}!Type {
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
            try bc.emitNoVariant(bc.tokSpan(vtok), e.name, vname, e.variants);
            return .invalid;
        };
        if (!formMatches(node_form, variant.form)) {
            try bc.typeArgsForEffect(node_form, args);
            try bc.sink.report(.{ .span = bc.spanOf(site), .code = .T0063 }, "variant '{s}.{s}' is constructed with the wrong form", .{ e.name, vname });
            return result;
        }
        switch (variant.form) {
            .unit => {},
            .tuple => {
                const elems = if (args == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (args).int());
                if (elems.len != variant.field_types.len) {
                    if (pretyped == null) {
                        for (elems) |a| _ = try bc.typeUnder(a, bc.partHint());
                    }
                    try bc.sink.report(.{ .span = bc.countSpan(site), .code = .T0039 }, "variant '{s}.{s}' expects {d} value(s), got {d}{s}", .{ e.name, vname, variant.field_types.len, elems.len, bc.pipedArityHint(site) });
                    return result;
                }
                for (elems, variant.field_types, 0..) |a, fty_pat, i| {
                    const fty = substTy(bc, fty_pat, targs);
                    const at = if (pretyped) |pt| pt[i] else try bc.typeOfExpected(a, fty);
                    if (!Type.assignable(fty, at)) {
                        var hint: std.ArrayList(u8) = .empty;
                        defer hint.deinit(bc.gpa);
                        try bc.writeCaptureHint(&hint, a, fty, at);
                        try bc.sink.report(.{ .span = bc.spanOf(a), .code = .T0040 }, "variant '{s}.{s}': expected {s}, got {s}{s}", .{ e.name, vname, bc.typeName(fty), bc.typeName(at), hint.items });
                    }
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
                        if (seen[j]) try bc.sink.report(.{ .span = bc.tokSpan(fi.main_token), .code = .T0064 }, "duplicate field '{s}' in '{s}.{s}'", .{ fname, e.name, vname });
                        seen[j] = true;
                        const fty = substTy(bc, variant.field_types[j], targs);
                        const vt = if (pretyped) |pt| pt[ii] else try bc.typeOfExpected(fi.lhs, fty);
                        if (!Type.assignable(fty, vt))
                            try bc.sink.report(.{ .span = bc.spanOf(fi.lhs), .code = .T0041 }, "field '{s}': expected {s}, got {s}", .{ fname, bc.typeName(fty), bc.typeName(vt) });
                    } else {
                        if (pretyped == null) _ = try bc.typeUnder(fi.lhs, bc.partHint());
                        try bc.sink.report(.{ .span = bc.tokSpan(fi.main_token), .code = .T0046 }, "unknown field '{s}' in '{s}.{s}'", .{ fname, e.name, vname });
                    }
                }
                for (variant.field_names, 0..) |dn, j| {
                    if (!seen[j]) try bc.sink.report(.{ .span = bc.spanOf(site), .code = .T0047 }, "missing field '{s}' in '{s}.{s}'", .{ dn, e.name, vname });
                }
            },
        }
        return result;
    }

    /// `Ref` is registered as a one-field tuple struct only to give the cell a layout; its
    /// field is the runtime cell pointer, so positional construction would let lower build
    /// a box out of a bare value (it recursed until the stack overflowed).
    fn rejectRefTupleCall(bc: *BodyChecker, call: Ast.Index, n: Ast.Node, sid: u32) error{OutOfMemory}!Type {
        if (n.rhs != Ast.none) try bc.drainArgs(Ast.rangeSlice(bc.tree, (n.rhs).int()));
        const has_new = Typecheck.findGenericMethod(bc.model.templates, sid, false, "new", bc.methodSite()) != null;
        var hb: [64]u8 = undefined;
        try bc.sink.report(.{ .span = bc.spanOf(call), .code = .T0063 }, "'Ref' is built with 'Ref.new(v)', not 'Ref(...)'{s}", .{importHint(&hb, if (has_new) null else StdNames.familyModule(.ref))});
        return .invalid;
    }

    // Not routed through `checkCall` for the same reason as `checkVariantPayloads`: a
    // construction's elements are "values" of the struct, with its own wording.
    fn checkTupleStructInit(bc: *BodyChecker, call: Ast.Index, n: Ast.Node, sid: u32) error{OutOfMemory}!Type {
        const sym = bc.model.structs[sid];
        const result = Type.structT(sid);
        const elems = if (n.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, (n.rhs).int());
        if (elems.len != sym.field_types.len) {
            for (elems) |a| _ = try bc.typeUnder(a, bc.partHint()); // surface inner arg errors first
            try bc.sink.report(.{ .span = bc.argsSpan(call), .code = .T0039 }, "tuple struct '{s}' expects {d} value(s), got {d}{s}", .{ sym.name, sym.field_types.len, elems.len, bc.pipedArityHint(call) });
            return result;
        }
        for (elems, sym.field_types) |a, fty| {
            const at = try bc.typeOfExpected(a, fty); // expected type drives literal narrowing (integer widths)
            if (!Type.assignable(fty, at)) {
                var hint: std.ArrayList(u8) = .empty;
                defer hint.deinit(bc.gpa);
                try bc.writeCaptureHint(&hint, a, fty, at);
                try bc.sink.report(.{ .span = bc.spanOf(a), .code = .T0040 }, "tuple struct '{s}': expected {s}, got {s}{s}", .{ sym.name, bc.typeName(fty), bc.typeName(at), hint.items });
            }
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

    /// The target type-args when the expected type is an `App` of exactly THIS ctor with
    /// matching enum/struct-ness (target typing). Anything else — a scalar, the
    /// wrong ctor, a plain `enumT`, or no expected type — yields null, so the
    /// `fillExpected` reconcile is a no-op and behavior is byte-identical.
    /// This ctor-gate is the sole guard against a stale/unrelated expected type wrongly
    /// filling; a same-ctor `App` always has `args.len == generic_params.len`, so the
    /// arity matches at the two construction sites.
    fn expectedAppArgs(bc: *const BodyChecker, ctor_id: u32, want_enum: bool) ?[]const Type {
        const exp = bc.expectedTy() orelse return null;
        if (!exp.isApp()) return null;
        const e = bc.composite.at(exp.appIdx());
        if (e.ctor != ctor_id or e.ctor_is_enum != want_enum) return null;
        return e.args;
    }

    const Reconciled = union(enum) { ok, err, unbound: u32 };

    fn targetConflicts(out: []const Type, bound: []const bool, exp_args: ?[]const Type) bool {
        const ea = exp_args orelse return false;
        if (ea.len != out.len) return false;
        for (out, bound, ea) |o, b, e| if (b and !Type.eql(o, e)) return true;
        return false;
    }

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
        const soft = if (bc.body.expected) |e| e.soft else false;
        const ea = if (soft and targetConflicts(out, bound, exp_args)) null else exp_args;
        switch (Infer.fillExpected(out, bound, ea)) {
            // Unlike the leaf-carrying `match` conflict, this compares the WHOLE
            // per-ordinal type-args (no structural descent) — a container clash here
            // could differ at several depths at once (`Map[int,bool]` vs `Map[str,char]`
            // has two differing leaves), so there is no single leaf to name. Spelling an
            // `App`'s type-args is a change to the shared `typeName` accessor, out of
            // scope for this leaf-carry fix; `arg`/`expected` render as-is.
            .conflict => |c| {
                try bc.sink.report(.{ .span = bc.tokSpan(span_tok), .code = .T0015 }, "conflicting types for type parameter '{s}': {s} inferred from the value, {s} from the expected type", .{ param_names[c.ord], bc.typeName(c.arg), bc.typeName(c.expected) });
                return .err;
            },
            .unbound => |u| return .{ .unbound = u.ord },
            .ok => {
                for (out) |ta| {
                    // A bounded template's own param inferred here grounds at the outer
                    // instantiation; admit it (the instance re-check re-gates concretely).
                    if (bc.abstract_template and ta.isTypeVar()) continue;
                    if (!isConcreteValue(ta)) {
                        try bc.sink.report(.{ .span = bc.tokSpan(span_tok), .code = .T0013 }, "inferred type argument must be a concrete value type; add explicit type arguments", .{});
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
        const ret_fam = bc.optResultFamily(bc.body.cur_ret);
        const payload: Type = if (op_fam != .none) blk: {
            const args = bc.composite.at(ot.appIdx()).args;
            break :blk if (args.len >= 1) args[0] else Type.invalid;
        } else Type.invalid;

        if (op_fam == .none) {
            try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0032 }, "'?' operand must be an Option or Result, got {s}", .{bc.typeName(ot)});
        } else if (bc.body.ret_infer and bc.body.hole != Ast.none) {
            // A capture has no place for a `-> R`, so "annotate" cannot be followed.
            try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0032 }, "'?' in a '_' capture exits the capture, so it needs an expected fn type returning Option[..] or Result[..]; or write the lambda out with its '-> Option[..]'", .{});
        } else if (bc.body.ret_infer) {
            try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0032 }, "'?' in a lambda needs its return type; annotate -> Option[..] or Result[..]", .{});
        } else if (ret_fam == .none) {
            try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0032 }, "'?' requires the enclosing function to return an Option or Result, but it returns {s}", .{bc.typeName(bc.body.cur_ret)});
        } else if (op_fam != ret_fam) {
            try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0033 }, "'?' on {s} in a function returning {s}", .{ familyName(op_fam), familyName(ret_fam) });
        } else if (op_fam == .result) {
            const op_args = bc.composite.at(ot.appIdx()).args;
            const ret_args = bc.composite.at(bc.body.cur_ret.appIdx()).args;
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
                    try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0033 }, "'?' error type {s} does not match the enclosing Result error type {s}", .{ bc.typeName(op_args[1]), bc.typeName(ret_args[1]) });
            }
        }
        bc.node_types[(node_idx).int()] = payload;
        return payload;
    }

    fn typeOfIf(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        _ = node_idx;
        // The expected type is the arms', not the condition's, which types as it does in a
        // statement `if`.
        const ct = try bc.typeOfExpected(n.lhs, null);
        if (ct.kind != .invalid and ct.kind != .bool)
            try bc.sink.report(.{ .span = bc.spanOf(n.lhs), .code = .T0045 }, "if condition must be bool", .{});
        try bc.warnConstIfCond(n.lhs);
        const h = Ast.ifHeaderAt(bc.tree, (n.rhs).int());
        if (h.else_node == Ast.none) {
            try bc.sink.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0059 }, "value-if requires else", .{});
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

    // A bare `true`/`false` literal for a condition means one branch is dead — leftover
    // debug code or a mistake. Diagnostics-only; nothing here reaches lowering. The
    // message argument is the source literal (constant per site, never instance-varying),
    // so per-generic-instance rechecks dedup on scope+offset+code+message.
    fn warnConstIfCond(bc: *BodyChecker, cond: Ast.Index) error{OutOfMemory}!void {
        const tok = Lint.constIfCond(bc.tree, cond) orelse return;
        try bc.sink.report(.{ .span = bc.tokSpan(tok), .code = .W0007 }, "if condition is always {s}", .{bc.nameText(tok)});
    }

    fn typeOfLoop(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, label: ?[]const u8) error{OutOfMemory}!Type {
        try bc.loop_stack.append(bc.gpa, .{ .kind = .loop, .label = label, .construct_node = node_idx, .is_value = true, .join = Type.never, .saw_value_break = false, .saw_bare_break = false, .expected = bc.body.expected });
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
        try bc.sink.report(.{ .span = bc.tokSpan(at), .code = .T0058 }, "branches yield different types ({s} vs {s})", .{ bc.typeName(a), bc.typeName(b) });
        return .invalid; // mismatch, no coercion
    }

    // The FnSym.decl_node doc mandates keying builtin-ness off `.kind`, not the sentinel.
    // A cross-module callee's decl offset would render against THIS module's source (the
    // sink scope == graph_mod), so no label there — the message still carries name+signature.
    fn calleeDeclSite(bc: *const BodyChecker, f: FnSym) ?Span {
        if (f.kind == .builtin or f.mod != bc.graph_mod) return null;
        return bc.tokSpan(bc.tree.nodes[(f.decl_node).int()].main_token);
    }

    /// The source extent of the subtree at `node`: what a diagnostic about the whole
    /// expression or statement underlines.
    pub fn spanOf(bc: *const BodyChecker, node: Ast.Index) Span {
        return refs.spanOf(bc, node);
    }

    pub fn tokSpan(bc: *const BodyChecker, tok: u32) Span {
        return refs.tokSpan(bc, tok);
    }

    /// A call's argument list `(…)`: from the call's `(` to its end. For a piped call
    /// that is from the `|>`: the step `|> f(a)` is what the user wrote wrong, and a
    /// span from the lhs would cover the whole chain before it.
    fn argsSpan(bc: *const BodyChecker, call: Ast.Index) Span {
        return .{ .start = bc.byteOf(bc.tree.nodes[call.int()].main_token), .end = bc.spanOf(call).end };
    }

    /// A `type_app`'s type-argument list `[…]`: from its `[` to its end.
    fn typeArgsSpan(bc: *const BodyChecker, app: Ast.Index) Span {
        return .{ .start = bc.byteOf(bc.tree.nodes[app.int()].main_token), .end = bc.spanOf(app).end };
    }

    // The piped value is invisible in the written argument list, so without these a
    // count or position that includes it reads as off by one. Empty for a plain call.
    // A hole pipe `x |> f(a, _)` spells out where the piped value goes, so its count
    // needs no hint and its argument hint names the `_`'s position.
    fn pipedArityHint(bc: *const BodyChecker, call: Ast.Index) []const u8 {
        return if (Ast.pipePrepends(bc.tree, bc.tokens, bc.source, call)) " (the piped value is argument 1)" else "";
    }

    pub fn pipedArgHint(bc: *const BodyChecker, call: Ast.Index, idx: usize) []const u8 {
        const at = Ast.pipedArgIndex(bc.tree, bc.tokens, call) orelse return "";
        return if (idx == at + 1) " (the piped value)" else "";
    }

    /// Where a count error on a construction or call points: its argument list for a
    /// piped call (the failing `|> ..` step, see `argsSpan`), else the whole node.
    fn countSpan(bc: *const BodyChecker, site: Ast.Index) Span {
        return if (Ast.isPipedCall(bc.tree, bc.tokens, site)) bc.argsSpan(site) else bc.spanOf(site);
    }

    /// How a declared param becomes the wanted arg type. Applied lazily per arg, so a
    /// call needs no substituted-signature buffer.
    const ParamMap = union(enum) {
        id,
        subst: []const Type,
        ground: struct { self_ty: Type, pargs: []const Type },
        /// The type-args failed to bind (already diagnosed): args are typed with no
        /// expectation and never judged, and the raw params are rendered.
        unbound,
    };

    /// The shape a call's value args are checked against (see `checkCall`).
    pub const CallSig = struct {
        /// The non-self params: they drive the count, the wanted types, and the
        /// rendered signature.
        params: []const Type,
        /// Null drops the "; 'f' takes (..)" suffix.
        name: ?[]const u8 = null,
        decl: ?Span = null,
        /// The callee's own generic-param names, so an unsubstituted `type_var` param
        /// renders as `T` rather than its kind tag.
        generic_names: []const []const u8 = &.{},
        map: ParamMap = .id,
        /// The inferred-generic path's already-synthesized arg types; re-walking the
        /// args would double-emit their inner diagnostics. That path checks the count
        /// itself first, so only matching-count, bound calls carry it.
        pretyped: ?[]const Type = null,
        /// Bit i: value arg i binds a `mut` param.
        mut_mask: u64 = 0,

        pub fn substituted(s: CallSig, targs: ?[]const Type) CallSig {
            var out = s;
            out.map = if (targs) |ta| .{ .subst = ta } else .unbound;
            return out;
        }
    };

    pub fn fnSig(bc: *const BodyChecker, f: FnSym, name: []const u8) CallSig {
        return .{ .params = f.params, .name = name, .decl = bc.calleeDeclSite(f), .generic_names = f.generic_params, .mut_mask = f.mut_mask };
    }

    pub fn methodSig(bc: *const BodyChecker, f: FnSym, name: []const u8) CallSig {
        return .{ .params = f.params[@min(f.params.len, 1)..], .name = name, .decl = bc.calleeDeclSite(f), .generic_names = f.generic_params, .mut_mask = f.mut_mask >> 1 };
    }

    fn paramWant(bc: *BodyChecker, sig: CallSig, p: Type) Type {
        return switch (sig.map) {
            .id, .unbound => p,
            .subst => |ta| substTy(bc, p, ta),
            .ground => |g| Typecheck.groundProtoType(bc, p, g.self_ty, g.pargs),
        };
    }

    /// Type a failed call's args only for their own diagnostics. With no expectation:
    /// the call's expected type belongs to its result, so passing it down would judge a
    /// literal arg against it (`x: int8 = f(300)` would add a false out-of-range error).
    /// Every caller has rejected the call (or its callee was already poison), which is why
    /// nothing types a lambda arg's params: a T0050 there would blame the lambda for it.
    pub fn drainArgs(bc: *BodyChecker, args: []const Ast.Index) error{OutOfMemory}!void {
        const was = bc.quiet_arg;
        defer bc.quiet_arg = was;
        for (args) |a| {
            bc.quiet_arg = a;
            _ = try bc.typeOfExpected(a, null);
        }
    }

    /// A `mut self` call writes through its receiver, which must be a mutable place.
    fn checkMutReceiver(bc: *BodyChecker, recv: Ast.Index, member: []const u8) error{OutOfMemory}!void {
        try LambdaChecker.noteMutatedRoot(bc, recv);
        if (!bc.isMutablePlace(recv)) {
            try bc.sink.report(.{ .span = bc.spanOf(recv), .code = .T0019 }, "cannot call mutating method '{s}' on a temporary; the receiver must be a mutable place (a local, a Vec element, or a field of either)", .{member});
        } else bc.noteElementWrite(recv, true);
    }

    /// Check `args` against `sig`: arity (typing every arg for effect first on a
    /// mismatch, unless pretyped), then each arg's assignability to its wanted type.
    /// Returns false on an arity mismatch.
    pub fn checkCall(bc: *BodyChecker, call: Ast.Index, args: []const Ast.Index, sig: CallSig) error{OutOfMemory}!bool {
        if (args.len != sig.params.len) {
            std.debug.assert(sig.pretyped == null);
            try bc.drainArgs(args);
            try bc.reportArity(call, args.len, sig);
            return false;
        }
        for (args, sig.params, 0..) |a, p, i| {
            if (sig.map == .unbound) {
                std.debug.assert(sig.pretyped == null);
                _ = try bc.typeOfExpected(a, null);
                continue;
            }
            const want = bc.paramWant(sig, p);
            const got = if (sig.pretyped) |pt| pt[i] else try bc.typeOfExpected(a, if (want.kind == .invalid) null else want);
            if (!Type.assignable(want, got)) {
                var suffix: std.ArrayList(u8) = .empty;
                defer suffix.deinit(bc.gpa);
                try bc.writeSigSuffix(&suffix, sig);
                try bc.writeCaptureHint(&suffix, a, want, got);
                try bc.sink.report(.{ .span = bc.spanOf(a), .code = .T0040, .related = sig.decl }, "argument {d}{s}: expected {s}, got {s}{s}", .{ i + 1, bc.pipedArgHint(call, i + 1), bc.typeName(want), bc.typeName(got), suffix.items });
            }
            if (Modes.isMut(sig.mut_mask, i)) try bc.checkMutArg(call, a, i, sig);
        }
        return true;
    }

    /// The hint for a `_` capture `a` passed where `want`, a non-function, is wanted: the
    /// `_` captures only its own call, so the user most likely meant the outer call's.
    fn writeCaptureHint(bc: *BodyChecker, out: *std.ArrayList(u8), a: Ast.Index, want: Type, got: Type) error{OutOfMemory}!void {
        if (!Ast.isCaptureLambda(bc.tree, bc.tokens, a) or want.isFunc() or !got.isFunc()) return;
        const parts = Composite.fnParts(bc.composite.at(got.funcIdx()));
        if (parts.params.len != 1) return;
        try out.print(bc.gpa, "; its '_' makes it a function, so to pass the call's result, write the lambda around the outer call, as in fn(v: {s}) {{ f(g(v)) }}", .{bc.typeName(parts.params[0])});
    }

    /// A `mut` parameter writes through its argument, so the argument must name a mutable
    /// place, and the local it is rooted at counts as mutated (a captured one is boxed).
    fn checkMutArg(bc: *BodyChecker, call: Ast.Index, arg: Ast.Index, i: usize, sig: CallSig) error{OutOfMemory}!void {
        const callee = sig.name orelse "the function";
        if (bc.body.hole != Ast.none and arg == bc.body.hole) {
            try bc.sink.report(.{ .span = bc.spanOf(arg), .code = .T0019, .related = sig.decl }, "argument {d} is a capture's '_', a copy of the value it is called with; '{s}' takes it as 'mut', so its change would be lost", .{ i + 1, callee });
        } else if (!bc.isMutablePlace(arg)) {
            try bc.sink.report(.{ .span = bc.spanOf(arg), .code = .T0019, .related = sig.decl }, "argument {d}{s} must be a mutable place (a local, a Vec element, or a field of either); '{s}' takes it as 'mut'", .{ i + 1, bc.pipedArgHint(call, i + 1), callee });
        } else if (bc.placeHopsThroughMethod(arg)) {
            try bc.sink.report(.{ .span = bc.spanOf(arg), .code = .T0019, .related = sig.decl }, "argument {d}{s} is a method value, not a place; '{s}' takes it as 'mut'", .{ i + 1, bc.pipedArgHint(call, i + 1), callee });
        } else if (bc.namesLocalFn(arg)) {
            try bc.sink.report(.{ .span = bc.spanOf(arg), .code = .T0019, .related = sig.decl }, "argument {d} names the local function '{s}', which a 'mut' parameter could reassign", .{ i + 1, bc.nameText(bc.tree.nodes[arg.int()].main_token) });
        } else {
            try LambdaChecker.noteMutatedRoot(bc, arg);
            bc.noteElementWrite(arg, true);
        }
    }

    /// Whether a `.field_access` hop of the place chain `arg` names no field of its base: a
    /// bound method (`p.get`) is a fresh closure, with no storage a `mut` param could
    /// write through.
    fn placeHopsThroughMethod(bc: *const BodyChecker, arg: Ast.Index) bool {
        var cur = arg;
        while (true) {
            const n = bc.tree.nodes[cur.int()];
            switch (n.tag) {
                .field_access => if (!bc.namesStructField(cur)) return true,
                .tuple_field => {},
                else => return false,
            }
            cur = n.lhs;
        }
    }

    fn namesStructField(bc: *const BodyChecker, fa: Ast.Index) bool {
        const n = bc.tree.nodes[fa.int()];
        const base = bc.node_types[n.lhs.int()];
        const sid = if (base.isApp()) blk: {
            const e = bc.composite.at(base.appIdx());
            if (e.ctor_is_enum) return false;
            break :blk e.ctor;
        } else if (base.isStruct()) base.struct_id else return false;
        if (sid >= bc.structSyms().len) return false;
        const fname = bc.nameText(n.main_token);
        for (bc.structSyms()[sid].field_names) |dn| if (std.mem.eql(u8, dn, fname)) return true;
        return false;
    }

    /// Whether `arg` names a local fn, whose binding is never reassigned (R0007) and never
    /// boxed.
    fn namesLocalFn(bc: *const BodyChecker, arg: Ast.Index) bool {
        if (bc.tree.nodes[arg.int()].tag != .identifier) return false;
        const slot = switch (bc.resolutions[arg.int()]) {
            .local => |s| s,
            else => return false,
        };
        const fc = bc.closures orelse return false;
        for (fc.boundaries) |b| if (b.self_slot) |ss| if (ss == slot) return true;
        return false;
    }

    /// Arity only, for the intrinsics whose per-arg rules are bespoke.
    fn checkArity(bc: *BodyChecker, call: Ast.Index, args: []const Ast.Index, want: usize) error{OutOfMemory}!bool {
        if (args.len == want) return true;
        try bc.drainArgs(args);
        try bc.emitArity(call, want, args.len, null, "");
        return false;
    }

    /// Emit only: the args are already typed.
    pub fn reportArity(bc: *BodyChecker, call: Ast.Index, got: usize, sig: CallSig) error{OutOfMemory}!void {
        var suffix: std.ArrayList(u8) = .empty;
        defer suffix.deinit(bc.gpa);
        try bc.writeSigSuffix(&suffix, sig);
        try bc.emitArity(call, sig.params.len, got, sig.decl, suffix.items);
    }

    pub fn emitArity(bc: *BodyChecker, call: Ast.Index, want: usize, got: usize, related: ?Span, suffix: []const u8) error{OutOfMemory}!void {
        try bc.sink.report(.{ .span = bc.argsSpan(call), .code = .T0039, .related = related }, "expected {d} argument(s), got {d}{s}{s}", .{ want, got, bc.pipedArityHint(call), suffix });
    }

    fn writeSigSuffix(bc: *BodyChecker, out: *std.ArrayList(u8), sig: CallSig) error{OutOfMemory}!void {
        try bc.writeSigSuffixOpen(out, sig, null, null);
    }

    /// `writeSigSuffix` for a generic call whose type args are only partly bound: each
    /// param is substituted through `targs`, and a var still open in `open` prints as its
    /// declared name.
    pub fn writeSigSuffixOpen(bc: *BodyChecker, out: *std.ArrayList(u8), sig: CallSig, targs: ?[]const Type, open: ?[]const bool) error{OutOfMemory}!void {
        const name = sig.name orelse return;
        try out.appendSlice(bc.gpa, "; '");
        try out.appendSlice(bc.gpa, name);
        try out.appendSlice(bc.gpa, "' takes (");
        for (sig.params, 0..) |p, i| {
            if (i != 0) try out.appendSlice(bc.gpa, ", ");
            if (Modes.isMut(sig.mut_mask, i)) try out.appendSlice(bc.gpa, "mut ");
            if (targs) |ta| {
                try bc.writeTypeNamed(out, substTy(bc, p, ta), sig.generic_names, open, bc.openOrdBase());
                continue;
            }
            // A mapped param's leftover type_vars belong to the caller's scope, not the
            // callee's, so only a raw param takes the callee's names.
            const raw = sig.map == .id or sig.map == .unbound;
            if (raw) try bc.writeTypeNamed(out, p, sig.generic_names, null, 0) else try out.appendSlice(bc.gpa, bc.typeName(bc.paramWant(sig, p)));
        }
        try out.append(bc.gpa, ')');
    }

    /// `t` as `typeName` spells it, except that a type var `base + ord` with a name in
    /// `names` (and, when `open` is given, still open there) prints as that name, at any
    /// depth. A fn type's label is fixed when the type is interned, so a fn type is
    /// spelled from its parts here.
    pub fn writeTypeNamed(bc: *BodyChecker, out: *std.ArrayList(u8), t: Type, names: []const []const u8, open: ?[]const bool, base: u32) error{OutOfMemory}!void {
        if (t.isTypeVar() and t.typeVarOrd() >= base) {
            const ord = t.typeVarOrd() - base;
            const is_open = if (open) |o| ord < o.len and o[ord] else true;
            if (ord < names.len and is_open) return out.appendSlice(bc.gpa, names[ord]);
        }
        if (!t.isFunc()) return out.appendSlice(bc.gpa, bc.typeName(t));
        const parts = Composite.fnParts(bc.composite.at(t.funcIdx()));
        try out.appendSlice(bc.gpa, "fn(");
        for (parts.params, 0..) |fp, i| {
            if (i != 0) try out.appendSlice(bc.gpa, ", ");
            if (fp.kind == .unit) try out.appendSlice(bc.gpa, "()") else try bc.writeTypeNamed(out, fp, names, open, base);
        }
        try out.append(bc.gpa, ')');
        if (parts.ret.kind == .unit) return;
        try out.appendSlice(bc.gpa, " -> ");
        try bc.writeTypeNamed(out, parts.ret, names, open, base);
    }

    pub fn setCallee(bc: *BodyChecker, call: Ast.Node, c: Callee) void {
        // A bounded template's receiver types live in a discarded scratch, so the seal
        // could never resolve a pending slot there.
        if (c.isPending() and bc.abstract_template) return;
        const fa = Callee.memberNode(bc.tree, call) orelse return;
        const slot = &bc.callees[fa.int()];
        if (builtin.mode == .Debug) std.debug.assert(slot.tag == .none or std.meta.eql(slot.*, c));
        slot.* = c;
    }

    /// Write `c` at `slot` itself: a value call's `callSlot`, or a fn or method named as a
    /// value.
    fn setCalleeAt(bc: *BodyChecker, slot: Ast.Index, c: Callee) void {
        // A bounded template is never lowered, and its instances record their own refs.
        if (bc.abstract_template and (c.tag == .fn_ref or c.tag == .bound_method or c.isPending())) return;
        const cell = &bc.callees[slot.int()];
        if (builtin.mode == .Debug) std.debug.assert(cell.tag == .none or std.meta.eql(cell.*, c));
        cell.* = c;
    }

    /// A named fn in value position (`add`, `mod.f`): its fn type, recording the
    /// reference on the name's own node. A builtin (compiler-expanded), an extern (a
    /// foreign symbol) or a generic template (only instances exist) has no code a value
    /// could point at.
    fn typeOfFnValue(bc: *BodyChecker, node_idx: Ast.Index, gid: u32, tok: u32) error{OutOfMemory}!Type {
        const f = bc.model.fns[gid];
        const what: ?[]const u8 = if (f.kind == .builtin)
            "a builtin"
        else if (f.kind == .import)
            "an extern function"
        else if (f.isGeneric())
            "a generic function"
        else if (f.mut_mask != 0)
            "a function with 'mut' parameters"
        else
            null;
        if (what) |w| {
            try bc.sink.report(.{ .span = bc.tokSpan(tok), .code = .T0049 }, "'{s}' is {s} and cannot be used as a value", .{ bc.nameText(tok), w });
            return bc.poison();
        }
        const ty = refs.internFn(bc, f.params, f.ret);
        if (ty.kind == .invalid) return error.OutOfMemory;
        bc.setCalleeAt(node_idx, Callee.fnRef(gid));
        return ty;
    }

    /// A call of the function value typed `fn_ty` (already typed on the callee node):
    /// the args check against its params, so their expected types flow in as for a
    /// direct call, and the call types as its return.
    fn typeOfValueCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, fn_ty: Type) error{OutOfMemory}!Type {
        const parts = Composite.fnParts(bc.composite.at(fn_ty.funcIdx()));
        if (Callee.callSlot(bc.tree, n)) |slot| bc.setCalleeAt(slot, Callee.value_call);
        _ = try bc.checkCall(node_idx, Ast.rangeSlice(bc.tree, (n.rhs).int()), .{ .params = parts.params, .name = bc.valueCalleeName(n.lhs) });
        bc.node_types[(node_idx).int()] = parts.ret;
        return parts.ret;
    }

    /// How a value call's errors name its callee in "'f' takes (..)": as written, when it
    /// is a name or a field chain (`f`, `h.run`); null for any other expression, which
    /// would quote badly.
    fn valueCalleeName(bc: *const BodyChecker, callee: Ast.Index) ?[]const u8 {
        var cur = callee;
        while (true) {
            const cn = bc.tree.nodes[cur.int()];
            switch (cn.tag) {
                .identifier => break,
                .field_access, .tuple_field => cur = cn.lhs,
                else => return null,
            }
        }
        const sp = bc.spanOf(callee);
        const text = bc.source[sp.start..sp.end];
        if (std.mem.indexOfAny(u8, text, "\n#") != null) return null;
        return text;
    }

    /// A call whose callee is an arbitrary expression (`get()(1)`, `fs[0](1)`): type the
    /// callee, then call it if it is a function value.
    fn typeOfExprCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        const ct = try bc.typeOfExpected(n.lhs, null);
        if (ct.isFunc()) return bc.typeOfValueCall(node_idx, n, ct);
        try bc.drainArgs(Ast.rangeSlice(bc.tree, (n.rhs).int()));
        if (ct.kind == .invalid) return .invalid;
        try bc.sink.report(.{ .span = bc.spanOf(n.lhs), .code = .T0055 }, "called value is not a function", .{});
        return .invalid;
    }

    /// `recv.member(..)` where `member` names no method but a fn-typed field of the
    /// struct receiver: a call of the field's value. `field_ty` is the field's type,
    /// already substituted through an App receiver's args.
    fn typeOfFieldCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, field_ty: Type) error{OutOfMemory}!Type {
        bc.node_types[(n.lhs).int()] = field_ty;
        return bc.typeOfValueCall(node_idx, n, field_ty);
    }

    /// The fn-typed field `member` of struct `sid`, substituted through `targs`, or null.
    fn fnField(bc: *BodyChecker, sid: u32, member: []const u8, targs: []const Type) ?Type {
        const sym = bc.model.structs[sid];
        for (sym.field_names, sym.field_types) |fname, fty| {
            if (!std.mem.eql(u8, fname, member)) continue;
            const t = if (targs.len > 0) substTy(bc, fty, targs) else fty;
            return if (t.isFunc()) t else null;
        }
        return null;
    }

    /// Type-check a resolved method call `recv.m(args)` against the selected witness `m`
    ///: the `mut self` place gate, arity, and per-arg assignability, typing the call
    /// node as the method's return. `fa` is the `field_access` (`main_token` = member,
    /// `lhs` = receiver). Shared by the bare-dispatch path and the explicit-protocol-args
    /// (`v.m[int]()`) path so both type identically. `args` is `n`'s value-arg range.
    fn dispatchMethod(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, fa: Ast.Node, recv_ty: Type, member: []const u8, mi: u32) error{OutOfMemory}!Type {
        bc.setCallee(n, Callee.method(mi));
        const m = bc.model.methods[mi];
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        const mf = bc.model.fns[m.fn_id];
        // Its params would bind per call, but a non-generic type's method rows have no
        // instances for the seal to pick from.
        if (mf.isGeneric()) {
            try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0013 }, "'{s}' declares its own type parameters, which a method of the non-generic type '{s}' cannot have yet", .{ member, bc.typeName(recv_ty) });
            try bc.drainArgs(args);
            return .invalid;
        }
        if (m.mut_self) {
            if (recv_ty.kind != .@"struct" and recv_ty.kind != .@"enum") {
                try LambdaChecker.noteMutatedRoot(bc, fa.lhs);
                try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0022 }, "mutating method '{s}' is not supported on the builtin type '{s}'; `mut self` is only allowed on struct and enum receivers", .{ member, bc.typeName(recv_ty) });
            } else try bc.checkMutReceiver(fa.lhs, member);
        }
        _ = try bc.checkCall(node_idx, args, if (m.is_static) bc.fnSig(mf, member) else bc.methodSig(mf, member));
        bc.node_types[(node_idx).int()] = mf.ret;
        return mf.ret;
    }

    const BoundConformance = struct { pid: u32, args: ?[]const Type };

    /// The conformance a call takes in a bounded template's instance re-check when the
    /// bound-as-axiom check resolved it through a param's bound (`[T has Show]`,
    /// `t.render()`): that bound's protocol, and its args grounded through the instance
    /// (null when two bounds of the protocol on this type disagree). The receiver's own
    /// methods would offer every conformance of the instance type, which is ambiguous
    /// once it has a second `render`. Null for any other call. `buf` holds the args.
    fn boundConformance(bc: *BodyChecker, call: Ast.Node, recv: Type, buf: *std.ArrayList(Type)) error{OutOfMemory}!?BoundConformance {
        const fa = Callee.memberNode(bc.tree, call) orelse return null;
        if (fa.int() >= bc.axiom_callees.len) return null;
        const pid = switch (bc.axiom_callees[fa.int()].view()) {
            .protocol => |p| p.protocol,
            else => return null,
        };
        const s = bc.subst orelse return .{ .pid = pid, .args = null };
        var found = false;
        for (bc.bound_protocols, 0..) |bp, ord| {
            if ((bp orelse continue) != pid or ord >= s.types.len or !Type.eql(s.types[ord], recv)) continue;
            const pargs: []const Type = if (ord < bc.bound_protocol_args.len) bc.bound_protocol_args[ord] else &.{};
            if (!found) {
                for (pargs) |pa| try buf.append(bc.gpa, substTy(bc, pa, s.types));
                found = true;
                continue;
            }
            if (pargs.len != buf.items.len) return .{ .pid = pid, .args = null };
            for (pargs, buf.items) |pa, have| if (!Type.eql(substTy(bc, pa, s.types), have)) return .{ .pid = pid, .args = null };
        }
        return .{ .pid = pid, .args = if (found) buf.items else null };
    }

    /// The method template of a generic receiver that a call resolved through a bound
    /// takes (`boundConformance`): the one witnessing that bound's conformance, when the
    /// receiver's ctor has several same-named conformance methods. Null for any other call.
    fn boundTemplate(bc: *BodyChecker, call: Ast.Node, recv: Type, member: []const u8) error{OutOfMemory}!?Typecheck.TemplateMethod {
        var buf: std.ArrayList(Type) = .empty;
        defer buf.deinit(bc.gpa);
        const b = (try bc.boundConformance(call, recv, &buf)) orelse return null;
        const want = b.args orelse return null;
        const e = bc.composite.at(recv.appIdx());
        const site = bc.methodSite();
        for (bc.model.templates) |tm| {
            if (tm.recv_ctor != e.ctor or tm.recv_is_enum != e.ctor_is_enum or !std.mem.eql(u8, tm.name, member)) continue;
            if (site == .bundled and tm.shadows) continue;
            if ((tm.protocol_id orelse continue) != b.pid or tm.protocol_args.len != want.len) continue;
            const same = for (tm.protocol_args, want) |pa, w| {
                if (!Type.eql(try conform.substPattern(bc.composite, bc.gpa, pa, e.args), w)) break false;
            } else true;
            if (same) return tm;
        }
        return null;
    }

    /// Emit the T0025 ambiguity diagnostic: the receiver conforms more than once with a
    /// method `member` and the use site gave no disambiguating type-args. Names the
    /// conflicting conformances in table (source fn-id) order — deterministic at any `-jN`.
    /// A generic receiver's are its ctor's templates, their arg patterns grounded through
    /// the receiver's args.
    fn emitAmbiguousConformance(bc: *BodyChecker, at: Span, recv: Type, member: []const u8, value_recv: ?Ast.Index) error{OutOfMemory}!void {
        var confs: std.ArrayList(AmbiguousConformance) = .empty;
        defer confs.deinit(bc.gpa);
        var arg_buf: std.ArrayList(Type) = .empty;
        defer arg_buf.deinit(bc.gpa);
        if (recv.isApp()) {
            const e = bc.composite.at(recv.appIdx());
            const site = bc.methodSite();
            for (bc.model.templates) |tm| {
                if (tm.recv_ctor != e.ctor or tm.recv_is_enum != e.ctor_is_enum or !std.mem.eql(u8, tm.name, member)) continue;
                if (site == .bundled and tm.shadows) continue;
                const pid = tm.protocol_id orelse continue;
                const start = arg_buf.items.len;
                for (tm.protocol_args) |pa| try arg_buf.append(bc.gpa, try conform.substPattern(bc.composite, bc.gpa, pa, e.args));
                try confs.append(bc.gpa, .{ .pid = pid, .args_start = start, .args_len = arg_buf.items.len - start });
            }
        } else {
            var list: std.ArrayList(Typecheck.Method) = .empty;
            defer list.deinit(bc.gpa);
            try Typecheck.conformancesFor(bc.model.methods, recv, member, bc.gpa, &list);
            for (list.items) |m| {
                const start = arg_buf.items.len;
                try arg_buf.appendSlice(bc.gpa, m.protocol_args);
                try confs.append(bc.gpa, .{ .pid = m.protocol_id.?, .args_start = start, .args_len = m.protocol_args.len });
            }
        }
        const Msg = struct {
            recv: Type,
            member: []const u8,
            confs: []const AmbiguousConformance,
            buf: []const Type,
            value_recv: ?Ast.Index,

            fn write(m: @This(), b: *BodyChecker, out: *std.ArrayList(u8)) error{OutOfMemory}!void {
                try out.print(b.gpa, "ambiguous conformance: '{s}' on '{s}' matches multiple conformances (", .{ m.member, b.typeName(m.recv) });
                for (m.confs, 0..) |c, i| {
                    if (i != 0) try out.appendSlice(b.gpa, " and ");
                    try b.writeConformance(out, c, m.buf);
                }
                try out.appendSlice(b.gpa, "); ");
                try b.writeAmbiguityHint(out, m.recv, m.member, m.confs, m.buf, m.value_recv);
            }
        };
        try bc.reportHinted(.{ .span = at, .code = .T0025 }, Msg{ .recv = recv, .member = member, .confs = confs.items, .buf = arg_buf.items, .value_recv = value_recv });
    }

    const AmbiguousConformance = struct {
        pid: u32,
        args_start: usize,
        args_len: usize,

        fn args(c: AmbiguousConformance, buf: []const Type) []const Type {
            return buf[c.args_start..][0..c.args_len];
        }
    };

    /// `Into[int]`, or `Show` for a protocol without args.
    fn writeConformance(bc: *BodyChecker, out: *std.ArrayList(u8), c: AmbiguousConformance, buf: []const Type) error{OutOfMemory}!void {
        try out.appendSlice(bc.gpa, bc.model.protocols[c.pid].name);
        try bc.writeTypeArgsSource(out, c.args(buf), &.{});
    }

    /// How to choose among `confs`, in text that compiles where it is written. Explicit
    /// args name a conformance whose args no other one shares; a method value takes none,
    /// so at `value_recv.m` they go in a lambda's call. Any other is reached through a fn
    /// bounded by its protocol, since no call syntax names a protocol.
    fn writeAmbiguityHint(bc: *BodyChecker, out: *std.ArrayList(u8), recv: Type, member: []const u8, confs: []const AmbiguousConformance, buf: []const Type, value_recv: ?Ast.Index) error{OutOfMemory}!void {
        var named: usize = 0;
        var unnamed: ?AmbiguousConformance = null;
        for (confs, 0..) |c, i| {
            if (namedByArgs(confs, buf, i)) named += 1 else if (unnamed == null) unnamed = c;
        }
        if (named > 0) {
            const recv_text: ?[]const u8 = if (value_recv) |vr| bc.oneLineText(vr) else null;
            if (value_recv != null) {
                try out.appendSlice(bc.gpa, "a method value takes no protocol type arguments, so call the method in a lambda");
                if (recv_text != null) try out.appendSlice(bc.gpa, ", e.g. ");
            } else try out.appendSlice(bc.gpa, "add explicit protocol type arguments, e.g. ");
            var k: usize = 0;
            if (value_recv == null or recv_text != null) for (confs, 0..) |c, i| {
                if (!namedByArgs(confs, buf, i)) continue;
                if (k != 0) try out.appendSlice(bc.gpa, if (k + 1 == named) " or " else ", ");
                k += 1;
                if (recv_text) |rt| {
                    try out.append(bc.gpa, '\'');
                    try bc.writeLambdaCall(out, rt, recv, member, c, buf);
                    try out.append(bc.gpa, '\'');
                    continue;
                }
                try out.append(bc.gpa, '.');
                try out.appendSlice(bc.gpa, member);
                try bc.writeTypeArgsSource(out, c.args(buf), &.{});
                try out.appendSlice(bc.gpa, "()");
            };
        }
        const c = unnamed orelse return;
        if (named > 0) try out.appendSlice(bc.gpa, ", or ");
        try out.appendSlice(bc.gpa, "call it through a fn bounded by '");
        try bc.writeConformance(out, c, buf);
        try out.appendSlice(bc.gpa, "', e.g. '");
        try bc.writeBoundedFn(out, member, c, buf);
        try out.append(bc.gpa, '\'');
    }

    fn namedByArgs(confs: []const AmbiguousConformance, buf: []const Type, i: usize) bool {
        const a = confs[i].args(buf);
        if (a.len == 0) return false;
        for (confs, 0..) |d, j| if (j != i and Typecheck.eqlTypeVec(a, d.args(buf))) return false;
        return true;
    }

    /// `fn render_via[T has Show](x: T, a0: int) -> int { x.render(a0) }`: a fn whose bound
    /// makes `x.render(..)` take that conformance's witness.
    fn writeBoundedFn(bc: *BodyChecker, out: *std.ArrayList(u8), member: []const u8, c: AmbiguousConformance, buf: []const Type) error{OutOfMemory}!void {
        const p = bc.model.protocols[c.pid];
        const k = for (p.methods, 0..) |mn, i| {
            if (std.mem.eql(u8, mn, member)) break i;
        } else return;
        // Self is type var 0 and stays one, spelled `T`; the protocol's own params ground.
        const ground = try bc.gpa.alloc(Type, c.args_len + 1);
        defer bc.gpa.free(ground);
        ground[0] = Type.typeVar(0);
        @memcpy(ground[1..], c.args(buf));
        const self_name = [_][]const u8{"T"};
        try out.print(bc.gpa, "fn {s}_via[T has ", .{member});
        try bc.writeConformance(out, c, buf);
        try out.appendSlice(bc.gpa, "](");
        if (p.requiresMutSelf(k)) try out.appendSlice(bc.gpa, "mut ");
        try out.appendSlice(bc.gpa, "x: T");
        const params = p.method_params[k];
        for (params[@min(params.len, 1)..], 0..) |pt, i| {
            try out.print(bc.gpa, ", a{d}: ", .{i});
            try bc.writeTypeSource(out, substTy(bc, pt, ground), &self_name);
        }
        try out.append(bc.gpa, ')');
        const ret = substTy(bc, p.method_rets[k], ground);
        if (ret.kind != .unit) {
            try out.appendSlice(bc.gpa, " -> ");
            try bc.writeTypeSource(out, ret, &self_name);
        }
        try out.print(bc.gpa, " {{ x.{s}(", .{member});
        for (0..params.len -| 1) |i| try out.print(bc.gpa, "{s}a{d}", .{ if (i == 0) "" else ", ", i });
        try out.appendSlice(bc.gpa, ") }");
    }

    /// `fn(a0: int) { p.tag[int](a0) }`: a lambda that calls `member` on `recv_text` through
    /// conformance `c`, taking the method's other params.
    fn writeLambdaCall(bc: *BodyChecker, out: *std.ArrayList(u8), recv_text: []const u8, recv: Type, member: []const u8, c: AmbiguousConformance, buf: []const Type) error{OutOfMemory}!void {
        const p = bc.model.protocols[c.pid];
        const k = for (p.methods, 0..) |mn, i| {
            if (std.mem.eql(u8, mn, member)) break i;
        } else return;
        const ground = try bc.gpa.alloc(Type, c.args_len + 1);
        defer bc.gpa.free(ground);
        ground[0] = recv;
        @memcpy(ground[1..], c.args(buf));
        const params = p.method_params[k];
        try out.appendSlice(bc.gpa, "fn(");
        for (params[@min(params.len, 1)..], 0..) |pt, i| {
            try out.print(bc.gpa, "{s}a{d}: ", .{ if (i == 0) "" else ", ", i });
            try bc.writeTypeSource(out, substTy(bc, pt, ground), &.{});
        }
        try out.print(bc.gpa, ") {{ {s}.{s}", .{ recv_text, member });
        try bc.writeTypeArgsSource(out, c.args(buf), &.{});
        try out.append(bc.gpa, '(');
        for (0..params.len -| 1) |i| try out.print(bc.gpa, "{s}a{d}", .{ if (i == 0) "" else ", ", i });
        try out.appendSlice(bc.gpa, ") }");
    }

    /// Node `idx` as written, or null when it spans lines.
    fn oneLineText(bc: *const BodyChecker, idx: Ast.Index) ?[]const u8 {
        const sp = bc.spanOf(idx);
        const text = bc.source[sp.start..sp.end];
        return if (std.mem.indexOfScalar(u8, text, '\n') == null) text else null;
    }

    /// A callee as written (`pick`, `lib.pick`), or its own name when it spans lines.
    fn calleeText(bc: *const BodyChecker, callee: Ast.Index) []const u8 {
        const sp = bc.spanOf(callee);
        const text = bc.source[sp.start..sp.end];
        return if (std.mem.indexOfScalar(u8, text, '\n') == null) text else bc.nameText(bc.tree.nodes[callee.int()].main_token);
    }

    /// T0016, with a hint that names the callee with explicit type args, `name[A, B]`:
    /// each arg already known spelled out and `int` for each open one (`.invalid`), so
    /// the hint where the call names its callee gets past T0001 and T0016. `rest` spells
    /// the rest of the expression (`.ok`, or `(..)` standing for the args), and
    /// `then_args`, when given, the type args after it (an associated fn's own).
    pub fn reportUninferable(bc: *BodyChecker, at: Span, comptime what: []const u8, what_args: anytype, name: []const u8, args: []const Type, rest: []const []const u8, then_args: ?[]const Type) error{OutOfMemory}!void {
        const Hint = struct {
            what_args: @TypeOf(what_args),
            name: []const u8,
            args: []const Type,
            rest: []const []const u8,
            then_args: ?[]const Type,

            fn write(h: @This(), b: *BodyChecker, out: *std.ArrayList(u8)) error{OutOfMemory}!void {
                try out.print(b.gpa, "cannot infer type parameter" ++ what ++ "; add explicit type arguments, e.g. ", h.what_args);
                try out.appendSlice(b.gpa, h.name);
                try b.writeHintArgs(out, h.args);
                for (h.rest) |r| try out.appendSlice(b.gpa, r);
                if (h.then_args) |ta| try b.writeHintArgs(out, ta);
            }
        };
        try bc.reportHinted(.{ .span = at, .code = .T0016 }, Hint{ .what_args = what_args, .name = name, .args = args, .rest = rest, .then_args = then_args });
    }

    /// `[A, B]`, with `int` for an open (`.invalid`) arg.
    fn writeHintArgs(bc: *BodyChecker, out: *std.ArrayList(u8), args: []const Type) error{OutOfMemory}!void {
        try out.append(bc.gpa, '[');
        for (args, 0..) |a, i| {
            if (i != 0) try out.appendSlice(bc.gpa, ", ");
            if (a.kind == .invalid) try out.appendSlice(bc.gpa, "int") else try bc.writeTypeSource(out, a, bc.gph_generic_params);
        }
        try out.append(bc.gpa, ']');
    }

    /// Report the message `hint.write` builds. In an instance re-check its text may name
    /// the instance's types, so it is also written as the template spells it, for
    /// `Typecheck.unifyInstanceHints`.
    fn reportHinted(bc: *BodyChecker, r: DiagnosticSink.Report, hint: anytype) error{OutOfMemory}!void {
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(bc.gpa);
        try hint.write(bc, &text);
        try bc.sink.report(r, "{s}", .{text.items});
        if (bc.subst == null or bc.abstract_template) return;
        var tmpl: std.ArrayList(u8) = .empty;
        defer tmpl.deinit(bc.gpa);
        bc.spell_as_template = true;
        defer bc.spell_as_template = false;
        try hint.write(bc, &tmpl);
        if (std.mem.eql(u8, tmpl.items, text.items)) return;
        const reported = bc.sink.items()[bc.sink.count() - 1].message;
        try bc.template_spellings.ensureUnusedCapacity(bc.gpa, 1);
        bc.template_spellings.appendAssumeCapacity(.{ .message = reported, .template = try tmpl.toOwnedSlice(bc.gpa) });
    }

    /// In an instance re-check, the template param whose arg `t` is, when exactly one is:
    /// spelled with it, a hint reads alike in every instance whose args differ there.
    fn templateSpelling(bc: *const BodyChecker, t: Type) ?[]const u8 {
        if (!bc.spell_as_template) return null;
        const s = bc.subst orelse return null;
        var found: ?[]const u8 = null;
        for (s.names, s.types) |name, ty| if (Type.eql(ty, t)) {
            if (found != null) return null;
            found = name;
        };
        return found;
    }

    /// `t` as source spells it, generic args included, so a hint that names it compiles
    /// where it is pasted. A type var `ord` with a name in `names` prints as that name.
    pub fn writeTypeSource(bc: *BodyChecker, out: *std.ArrayList(u8), t: Type, names: []const []const u8) error{OutOfMemory}!void {
        if (bc.templateSpelling(t)) |name| return out.appendSlice(bc.gpa, name);
        if (t.kind == .unit) return out.appendSlice(bc.gpa, "()");
        if (t.isApp()) {
            try out.appendSlice(bc.gpa, bc.typeName(t));
            return bc.writeTypeArgsSource(out, bc.composite.at(t.appIdx()).args, names);
        }
        if (!t.isFunc()) return bc.writeTypeNamed(out, t, names, null, 0);
        const parts = Composite.fnParts(bc.composite.at(t.funcIdx()));
        try out.appendSlice(bc.gpa, "fn(");
        for (parts.params, 0..) |fp, i| {
            if (i != 0) try out.appendSlice(bc.gpa, ", ");
            try bc.writeTypeSource(out, fp, names);
        }
        try out.append(bc.gpa, ')');
        if (parts.ret.kind == .unit) return;
        try out.appendSlice(bc.gpa, " -> ");
        try bc.writeTypeSource(out, parts.ret, names);
    }

    /// `[A, B]` after a name, or nothing for no args.
    fn writeTypeArgsSource(bc: *BodyChecker, out: *std.ArrayList(u8), args: []const Type, names: []const []const u8) error{OutOfMemory}!void {
        if (args.len == 0) return;
        try out.append(bc.gpa, '[');
        for (args, 0..) |a, j| {
            if (j != 0) try out.appendSlice(bc.gpa, ", ");
            try bc.writeTypeSource(out, a, names);
        }
        try out.append(bc.gpa, ']');
    }

    /// Type an explicit-protocol-args method call `v.m[int](args)`: the callee is a
    /// `type_app` over a `field_access`. The type-arg node_types are written (the touched-type
    /// fold reads every typed node), then the receiver is dispatched and its pick recorded:
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
        if (try bc.explicitCallOnType(node_idx, n, callee, explicit)) |t| return t;
        const recv_ty = try bc.typeOfReceiver(fa.lhs); // also populates node_types[recv] for lower
        if (recv_ty.kind == .invalid) return .invalid;
        if (recv_ty.kind == .@"struct" or recv_ty.kind == .@"enum" or recv_ty.isScalar()) {
            var bound_args: std.ArrayList(Type) = .empty;
            defer bound_args.deinit(bc.gpa);
            const bound = try bc.boundConformance(n, recv_ty, &bound_args);
            switch (Typecheck.pickMethodIndex(bc.model.methods, recv_ty, member, if (bound) |b| b.pid else null, explicit, bc.methodSite())) {
                .one => |mi| return try bc.dispatchMethod(node_idx, n, fa, recv_ty, member, mi),
                else => return bc.reportNoExplicitMethod(n, recv_ty),
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
                    bc.setCallee(n, Callee.protocolMethod(pid, @intCast(k)));
                    const psig = p.method_params[k];
                    // No callee FnSym exists for a bound-as-axiom protocol method, so there is
                    // no decl to point at.
                    _ = try bc.checkCall(node_idx, args, .{
                        .params = psig[@min(psig.len, 1)..],
                        .name = member,
                        .map = .{ .ground = .{ .self_ty = recv_ty, .pargs = explicit } },
                    });
                    const ret = Typecheck.groundProtoType(bc, p.method_rets[k], recv_ty, explicit);
                    bc.node_types[(node_idx).int()] = ret;
                    return ret;
                };
                try bc.drainArgs(args);
                try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0018 }, "no method '{s}' on type parameter bounded by protocol '{s}'", .{ member, p.name });
                return .invalid;
            }
            try bc.drainArgs(args);
            try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0018 }, "no method '{s}' on unbounded type parameter", .{member});
            return .invalid;
        }
        if (recv_ty.isApp()) {
            const e = bc.composite.at(recv_ty.appIdx());
            switch (try Typecheck.pickExplicitTemplate(bc.gpa, bc.composite, bc.model.templates, bc.model.fns, e.ctor, e.ctor_is_enum, e.args, member, bc.methodSite(), explicit)) {
                .ok => |p| {
                    defer bc.gpa.free(p.targs);
                    return bc.checkExplicitTemplateCall(node_idx, n, callee, recv_ty, p.m, p.targs, p.own, false);
                },
                .arity => |a| return bc.reportExplicitArity(n, member, a.want, explicit.len),
                .no_conformance => return bc.reportNoExplicitMethod(n, recv_ty),
                .none => {},
            }
        }
        try bc.drainArgs(args);
        try bc.emitNoMethod(bc.tokSpan(fa.main_token), member, recv_ty);
        return .invalid;
    }

    /// An explicit-args call whose receiver names a type (`Box[int].with[bool](..)`,
    /// `Box.with[bool](..)`, `K.make[int]()`): an associated function's. Null when the
    /// receiver is a value.
    fn explicitCallOnType(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node, explicit: []const Type) error{OutOfMemory}!?Type {
        const fa = bc.tree.nodes[(callee.lhs).int()];
        const recv = bc.tree.nodes[(fa.lhs).int()];
        const member = bc.nameText(fa.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        if (recv.tag == .type_app and !bc.appHeadIsValue(fa.lhs)) {
            const app_ty = bc.typeFromNode(fa.lhs);
            if (!app_ty.isApp()) {
                try bc.drainArgs(args);
                return .invalid; // typeFromNode reported it
            }
            // Load-bearing: discovery and the seal read the receiver off this node.
            bc.node_types[(fa.lhs).int()] = app_ty;
            const e = bc.composite.at(app_ty.appIdx());
            switch (try Typecheck.pickExplicitTemplate(bc.gpa, bc.composite, bc.model.templates, bc.model.fns, e.ctor, e.ctor_is_enum, e.args, member, bc.methodSite(), explicit)) {
                .ok => |p| {
                    defer bc.gpa.free(p.targs);
                    return try bc.checkExplicitTemplateCall(node_idx, n, callee, app_ty, p.m, p.targs, p.own, true);
                },
                .arity => |a| return try bc.reportExplicitArity(n, member, a.want, explicit.len),
                .no_conformance => return try bc.reportNoExplicitMethod(n, app_ty),
                .none => {
                    try bc.drainArgs(args);
                    try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0018 }, "'{s}' has no associated function '{s}'", .{ bc.typeName(app_ty), member });
                    return .invalid;
                },
            }
        }
        // A bare generic type: its own args come from nowhere once the method's are given.
        if (recv.tag == .identifier and bc.resolutions[(fa.lhs).int()] == .unresolved) {
            if (bc.activeStructMap().get(bc.nameText(recv.main_token))) |sid| {
                if (sid < bc.structSyms().len and bc.structSyms()[sid].is_generic) {
                    if (Typecheck.findGenericMethod(bc.model.templates, sid, false, member, bc.methodSite()) != null) {
                        try bc.drainArgs(args);
                        try CallInfer.reportUninferableAssociated(bc, recv, member, sid, null, &.{}, null);
                        return .invalid;
                    }
                }
            }
        }
        // A non-generic type's associated fn or variant takes no type args: neither has
        // type params of its own, and a protocol declares no associated fn to pick.
        if (bc.concreteTypeNamed(fa.lhs)) |ty| {
            try bc.drainArgs(args);
            const is_variant = ty.kind == .@"enum" and for (bc.model.enums[ty.enum_id].variants) |v| {
                if (std.mem.eql(u8, v.name, member)) break true;
            } else false;
            if (is_variant or Typecheck.pickMethodIndex(bc.model.methods, ty, member, null, null, bc.methodSite()) != .none)
                try bc.sink.report(.{ .span = bc.typeArgsSpan(n.lhs), .code = .T0054 }, "'{s}' is not generic; drop the type arguments", .{member})
            else
                try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0018 }, "'{s}' has no associated function '{s}'", .{ bc.typeName(ty), member });
            return .invalid;
        }
        return null;
    }

    fn reportExplicitArity(bc: *BodyChecker, n: Ast.Node, member: []const u8, want: usize, got: usize) error{OutOfMemory}!Type {
        try bc.drainArgs(Ast.rangeSlice(bc.tree, (n.rhs).int()));
        try bc.sink.report(.{ .span = bc.typeArgsSpan(n.lhs), .code = .T0054 }, "'{s}' takes {d} type argument(s) of its own, got {d}", .{ member, want, got });
        return .invalid;
    }

    fn reportNoExplicitMethod(bc: *BodyChecker, n: Ast.Node, recv_ty: Type) error{OutOfMemory}!Type {
        try bc.drainArgs(Ast.rangeSlice(bc.tree, (n.rhs).int()));
        const fa = bc.tree.nodes[bc.tree.nodes[(n.lhs).int()].lhs.int()];
        if (bc.rejectedMethod(recv_ty, bc.nameText(fa.main_token))) return .invalid;
        try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0018 }, "no method '{s}' on type '{s}' for the given protocol type arguments", .{ bc.nameText(fa.main_token), bc.typeName(recv_ty) });
        return .invalid;
    }

    /// `b.m[A..](args)` on a generic-type receiver (`on_type`: `Box[int].m[A..](args)`),
    /// where the explicit args picked the template `m` and, with the receiver, every one
    /// of its type params (`targs`). The call node carries the pick for the seal
    /// (`Callee.pendingTemplate`), since the arg types alone cannot tell
    /// `b.none[int]()` from `b.none[bool]()`.
    fn checkExplicitTemplateCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node, recv_ty: Type, m: Typecheck.TemplateMethod, targs: []const Type, own: bool, on_type: bool) error{OutOfMemory}!Type {
        const fa = bc.tree.nodes[(callee.lhs).int()];
        const member = bc.nameText(fa.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        if (on_type and m.has_self) {
            try bc.drainArgs(args);
            try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0018 }, "'{s}' is an instance method, not an associated function; call it on a value", .{member});
            return .invalid;
        }
        if (!on_type and !m.has_self) {
            try bc.drainArgs(args);
            const e = bc.composite.at(recv_ty.appIdx());
            const has_field = !e.ctor_is_enum and bc.fnField(e.ctor, member, e.args) != null;
            try bc.reportAssociatedOnValue(fa, bc.typeName(recv_ty), "[..]", has_field);
            return .invalid;
        }
        if (own) {
            const targ_nodes = Ast.rangeSlice(bc.tree, (callee.rhs).int());
            var all_concrete = true;
            for (targ_nodes, targs[targs.len - targ_nodes.len ..]) |tn, ty| {
                if (ty.kind == .invalid) {
                    all_concrete = false;
                } else if (!isConcreteValue(ty) and !(bc.abstract_template and ty.isTypeVar())) {
                    try bc.sink.report(.{ .span = bc.spanOf(tn), .code = .T0013 }, "type argument must be a concrete value type", .{});
                    all_concrete = false;
                }
            }
            if (!all_concrete) {
                try bc.drainArgs(args);
                return .invalid;
            }
        }
        const mf = bc.model.fns[m.fn_id];
        if (m.mut_self) try bc.checkMutReceiver(fa.lhs, member);
        bc.setCallee(n, Callee.pendingCall(node_idx));
        bc.setCalleeAt(node_idx, Callee.pendingTemplate(m.fn_id, own));
        const sig = if (on_type) bc.fnSig(mf, member) else bc.methodSig(mf, member);
        _ = try bc.checkCall(node_idx, args, sig.substituted(targs));
        return bc.substRet(node_idx, mf, targs);
    }

    /// A method pipe's piped value is its receiver, so it must be a value, as for any
    /// pipe. Unchecked, `io |> .print(s)` or `Shape |> .circle(1)` would pass as the
    /// qualified call it desugars to.
    fn rejectPipedNamespace(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!?Type {
        if (!Ast.isPipedCall(bc.tree, bc.tokens, node_idx) or Ast.pipedArgIndex(bc.tree, bc.tokens, node_idx) != null) return null;
        const callee = bc.tree.nodes[n.lhs.int()];
        const fa = bc.tree.nodes[(if (callee.tag == .type_app) callee.lhs else n.lhs).int()];
        if (fa.tag != .field_access) return null;
        const what = bc.namespaceKind(fa.lhs) orelse return null;
        try bc.drainArgs(Ast.rangeSlice(bc.tree, n.rhs.int()));
        const sp = bc.spanOf(fa.lhs);
        const name = bc.source[sp.start..sp.end];
        try bc.sink.report(.{ .span = sp, .code = .T0056 }, "{s} '{s}' is not a value, so it cannot be piped; use '{s}.{s}' directly", .{ what, name, name, bc.nameText(fa.main_token) });
        return bc.poison();
    }

    /// "module" or "type" when `recv` names one rather than a value, else null.
    fn namespaceKind(bc: *BodyChecker, recv: Ast.Index) ?[]const u8 {
        const r = bc.tree.nodes[recv.int()];
        switch (r.tag) {
            .identifier => switch (bc.resolutions[recv.int()]) {
                .module => return "module",
                .unresolved => {
                    const name = bc.nameText(r.main_token);
                    if (bc.activeStructMap().get(name) != null or bc.activeEnumMap().get(name) != null) return "type";
                    return null;
                },
                else => return null,
            },
            .type_app => return if (bc.appHeadIsValue(recv)) null else "type",
            .field_access => {
                const inner = bc.tree.nodes[r.lhs.int()];
                if (inner.tag != .identifier or bc.resolutions[r.lhs.int()] != .module) return null;
                const g = bc.model.graph;
                const target = g.namespaceOfIn(bc.graph_mod, bc.nameText(inner.main_token)) orelse return null;
                const member = bc.nameText(r.main_token);
                if (g.mods[target].struct_ids.get(member) != null or g.mods[target].enum_ids.get(member) != null) return "type";
                return null;
            },
            else => return null,
        }
    }

    // A qualified tuple-variant construction `N.V(args)` arrives as a `.call` whose
    // callee is a `field_access` over an enum type-name / type_app / qualified
    // namespace. Route it to the matching enum-init checker; return null when the
    // receiver is a value (or a struct-`App`), i.e. a method call the caller dispatches.
    // The type_app-not-App drain returns a non-null `.invalid` (NOT null) so caller
    // method dispatch does not re-run `typeOf` on the args and double-diagnose.
    fn tryEnumConstruction(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node) error{OutOfMemory}!?Type {
        const recv = bc.tree.nodes[(callee.lhs).int()];
        if (recv.tag == .identifier) if (bc.activeEnumMap().get(bc.nameText(recv.main_token))) |enum_id| {
            if (bc.isAssociatedFnOfEnum(enum_id, callee.main_token)) return null;
            return try bc.typeOfEnumInitQualified(node_idx, .tuple, callee.lhs, callee.main_token, n.rhs);
        };
        // An explicit generic-enum tuple-variant construction `Either[int,bool].left(42)`
        //: the callee field_access's receiver is a `type_app` resolving to an
        // enum-`App`. Substitute the variant payload patterns through the App's args
        // and type the node as the `App` (reified to `enumT` in the mono tail).
        if (recv.tag == .type_app and !bc.appHeadIsValue(callee.lhs)) {
            const app_ty = bc.typeFromNode(callee.lhs);
            if (app_ty.isApp() and bc.composite.at(app_ty.appIdx()).ctor_is_enum) {
                const e = bc.composite.at(app_ty.appIdx());
                const ty = try bc.checkVariantPayloads(node_idx, e.ctor, callee.main_token, .tuple, n.rhs, e.args, null);
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
            if (bc.isAssociatedFnOfEnum(enum_id, callee.main_token)) return null;
            const ty = try bc.checkVariant(node_idx, enum_id, callee.main_token, .tuple, n.rhs);
            bc.node_types[(node_idx).int()] = ty;
            return ty;
        }
        return null;
    }

    /// `E.m(..)` names an associated fn rather than a variant: a non-generic enum `E` with
    /// no variant `m` and a method `m` (a variant keeps the name when both exist).
    fn isAssociatedFnOfEnum(bc: *BodyChecker, enum_id: u32, member_tok: u32) bool {
        const e = bc.model.enums[enum_id];
        if (e.is_generic) return false;
        const member = bc.nameText(member_tok);
        for (e.variants) |v| if (std.mem.eql(u8, v.name, member)) return false;
        return Typecheck.pickMethodIndex(bc.model.methods, Type.enumT(enum_id), member, null, null, bc.methodSite()) != .none;
    }

    /// The non-generic struct or enum a call's receiver names as a type (`K` in `K.make()`,
    /// or a qualified `lib.K`), or null when the receiver is a value, a module or a generic
    /// type.
    fn concreteTypeNamed(bc: *BodyChecker, recv_idx: Ast.Index) ?Type {
        const ids = bc.typeIdsNamed(recv_idx);
        if (ids.s) |sid| return if (sid < bc.structSyms().len and !bc.structSyms()[sid].is_generic) Type.structT(sid) else null;
        if (ids.e) |eid| return if (!bc.model.enums[eid].is_generic) Type.enumT(eid) else null;
        return null;
    }

    /// `K.make(args)` on a non-generic struct or enum `K`: a self-less fn of the type's
    /// inherent impl. Recorded as its method callee, which lower calls with no receiver,
    /// as it does the generic `Box[int].new()`.
    fn checkConcreteAssociatedCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node, ty: Type) error{OutOfMemory}!Type {
        const member = bc.nameText(callee.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        switch (Typecheck.pickMethodIndex(bc.model.methods, ty, member, null, null, bc.methodSite())) {
            .one => |mi| {
                if (bc.model.methods[mi].is_static) return bc.dispatchMethod(node_idx, n, callee, ty, member, mi);
                try bc.drainArgs(args);
                try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0018 }, "'{s}' is an instance method, not an associated function; call it on a value", .{member});
            },
            .ambiguous => {
                try bc.drainArgs(args);
                try bc.emitAmbiguousConformance(bc.tokSpan(callee.main_token), ty, member, null);
            },
            .none => {
                try bc.drainArgs(args);
                try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0018 }, "'{s}' has no associated function '{s}'", .{ bc.typeName(ty), member });
            },
        }
        return .invalid;
    }

    /// An ASSOCIATED-function call `Vec[int].new(args)`: a `field_access` callee whose
    /// receiver is a generic-STRUCT `type_app`, dispatching to a self-less method in the
    /// type's inherent `impl` (no `self` arg). Returns the (substituted) return type, or
    /// null when this is not an associated call (an instance-method receiver, an enum, a
    /// module — all handled by their own paths). Two error gates:
    ///   * a `type_app` receiver whose member IS an instance method -> T0018;
    ///   * a bare `Vec.new()` (no turbofish) naming a generic struct's associated fn whose
    ///     type params its value args cannot bind -> the turbofish-required T0016.
    fn tryAssociatedCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node) error{OutOfMemory}!?Type {
        const recv = bc.tree.nodes[(callee.lhs).int()];
        const member = bc.nameText(callee.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        if (recv.tag == .type_app and !bc.appHeadIsValue(callee.lhs)) {
            const app_ty = bc.typeFromNode(callee.lhs);
            if (!app_ty.isApp()) return null; // typeFromNode already diagnosed
            const e = bc.composite.at(app_ty.appIdx());
            if (e.ctor_is_enum) return null; // an enum type_app is variant construction
            const m = Typecheck.findGenericMethod(bc.model.templates, e.ctor, false, member, bc.methodSite()) orelse {
                // An instance has no methods of its own (`impl Box[int]` is rejected), so a
                // member no template declares names nothing.
                try bc.drainArgs(args);
                if (bc.rejectedOnType(e.ctor, e.ctor_is_enum, member)) return .invalid;
                const name = bc.nameText(bc.tree.nodes[recv.lhs.int()].main_token);
                const ref_struct = if (bc.model.prelude) |pre| pre.ref_struct else null;
                var hb: [64]u8 = undefined;
                const hint = importHint(&hb, if (ref_struct == e.ctor) bc.stdMethodModule(.ref, member) else null);
                try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0018 }, "'{s}' has no associated function '{s}'{s}", .{ name, member, hint });
                return .invalid;
            };
            if (m.has_self) {
                try bc.drainArgs(args);
                try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0018 }, "'{s}' is an instance method, not an associated function; call it on a value", .{member});
                return .invalid;
            }
            // Load-bearing: the reified receiver flows to scanCalls / reify / sealCallees off
            // `node_types[type_app]`.
            bc.node_types[(callee.lhs).int()] = app_ty;
            const mf = bc.model.fns[m.fn_id];
            if (Typecheck.hasOwnTypeParams(bc.composite, mf))
                return try CallInfer.checkOwnGenericCall(bc, node_idx, n, callee, app_ty, mf, member, true);
            bc.setCallee(n, Callee.pending);
            const targs_opt = try Typecheck.bindImplParams(bc.gpa, bc.composite, mf, e.args);
            defer if (targs_opt) |ta| bc.gpa.free(ta);
            _ = try bc.checkCall(node_idx, args, bc.fnSig(mf, member).substituted(targs_opt));
            return bc.substRet(node_idx, mf, targs_opt orelse return .invalid);
        }
        if (bc.concreteTypeNamed(callee.lhs)) |ty| return try bc.checkConcreteAssociatedCall(node_idx, n, callee, ty);
        // A bare `Box.new(5)` (no turbofish) naming a generic struct with an associated fn:
        // its type params come from the value args, or the call needs explicit ones.
        if (recv.tag == .identifier and bc.resolutions[(callee.lhs).int()] != .module) {
            const name = bc.nameText(recv.main_token);
            if (bc.activeStructMap().get(name)) |sid| {
                if (sid < bc.structSyms().len and bc.structSyms()[sid].is_generic) {
                    if (Typecheck.findGenericMethod(bc.model.templates, sid, false, member, bc.methodSite())) |m| {
                        if (try CallInfer.inferAssociatedCall(bc, node_idx, n, callee, m, args)) |t| return t;
                        try bc.drainArgs(args);
                        try CallInfer.reportUninferableAssociated(bc, recv, member, sid, null, &.{}, null);
                        return .invalid;
                    }
                    if (bc.rejectedOnType(sid, false, member)) {
                        try bc.drainArgs(args);
                        return .invalid;
                    }
                    // `Ref` is a prelude type, but its API is library code: name the import
                    // when no module of the program brought it in.
                    if (bc.model.prelude) |pre| if (pre.ref_struct) |rs| if (sid == rs) {
                        try bc.drainArgs(args);
                        var hb: [64]u8 = undefined;
                        try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0018 }, "'{s}' has no associated function '{s}'{s}", .{ name, member, importHint(&hb, bc.stdMethodModule(.ref, member)) });
                        return .invalid;
                    };
                }
            }
        }
        return null;
    }

    /// The std module to import for a missing method `member` of the type whose ctor or
    /// kind is `family`, or null when importing one would not provide it. `Ref`'s every
    /// method comes from std/ref, so while no module declares one, any member names it.
    fn stdMethodModule(bc: *const BodyChecker, family: StdNames.MethodFamily, member: []const u8) ?[]const u8 {
        if (StdNames.methodModule(family, member)) |m| return m;
        if (family != .ref) return null;
        const rs = (bc.model.prelude orelse return null).ref_struct orelse return null;
        for (bc.model.templates) |tm| if (tm.recv_ctor == rs and !tm.recv_is_enum) return null;
        return StdNames.familyModule(.ref);
    }

    /// The family of `recv_ty` whose methods partly live in a std module (`StdNames`).
    fn methodFamily(bc: *const BodyChecker, recv_ty: Type) ?StdNames.MethodFamily {
        if (recv_ty.kind == .str) return .str;
        if (!recv_ty.isApp()) return null;
        const e = bc.composite.at(recv_ty.appIdx());
        if (e.ctor_is_enum) return switch (Typecheck.optResultFamilyOf(bc.model, e.ctor)) {
            .none => null,
            .option => .option,
            .result => .result,
        };
        if (bc.model.prelude) |p| if (p.ref_struct == e.ctor) return .ref;
        return null;
    }

    /// "; add 'import std/<module>'" into `buf` for `stdMethodModule`, else "".
    fn importHint(buf: []u8, module: ?[]const u8) []const u8 {
        const m = module orelse return "";
        return std.fmt.bufPrint(buf, "; " ++ StdNames.add_import, .{m}) catch "";
    }

    fn typeOfCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        if (try bc.rejectPipedNamespace(node_idx, n)) |t| return t;
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
        // A value index `fs[i](x)` the parser read as a turbofish: its head is a local, so
        // it is a function value fetched from a container, not a generic instance.
        if (callee.tag == .type_app and bc.tree.nodes[(callee.lhs).int()].tag == .identifier and bc.appHeadIsValue(n.lhs))
            return bc.typeOfExprCall(node_idx, n);
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
                        try bc.sink.report(.{ .span = bc.typeArgsSpan(n.lhs), .code = .T0054 }, "'{s}' expects exactly one type argument", .{bn});
                    if (val_args.len != 0)
                        try bc.sink.report(.{ .span = bc.argsSpan(node_idx), .code = .T0039 }, "'{s}' takes no value arguments", .{bn});
                    bc.node_types[(node_idx).int()] = Type.int;
                    return Type.int;
                }
                // `descriptor_of[T]()`: yields a `rawptr` to `T`'s descriptor entry (the
                // constant is materialized in lower). Records the demand so the synthesis
                // barrier plans the descriptor + erased units; an aggregate/ground key also
                // forces the natural Hash+Eq witnesses the erased struct/enum walk delegates
                // to (and their transitive per-field witnesses).
                if (bik == .descriptor_of) {
                    const targ_nodes = Ast.rangeSlice(bc.tree, (callee.rhs).int());
                    for (targ_nodes) |tn| bc.node_types[(tn).int()] = bc.typeFromNode(tn);
                    const val_args = Ast.rangeSlice(bc.tree, (n.rhs).int());
                    for (val_args) |arg| _ = try bc.typeOf(arg);
                    if (targ_nodes.len != 1)
                        try bc.sink.report(.{ .span = bc.typeArgsSpan(n.lhs), .code = .T0054 }, "'descriptor_of' expects exactly one type argument", .{});
                    if (val_args.len != 0)
                        try bc.sink.report(.{ .span = bc.argsSpan(node_idx), .code = .T0039 }, "'descriptor_of' takes no value arguments", .{});
                    if (targ_nodes.len == 1) {
                        const t = bc.node_types[(targ_nodes[0]).int()];
                        // Only a ground type is recorded here: a `type_var` from a generic
                        // template body would reify to a phantom entry (and, being scalar-ranked,
                        // sort-tie with other type-vars → `-jN` table divergence). Concrete
                        // instances are recorded when `recheck` re-runs the body with the arg
                        // substituted.
                        //
                        // The descriptor's erased hash/eq witnesses are seeded from
                        // `descriptor_reqs` in the synthesis barrier, gated on the SAME
                        // `conform.hashable` verdict the erased-unit gate uses — so an
                        // explicit-impl key supplies its own witnesses (no derived recipe) and a
                        // Ref-tainted / non-hashable key records nothing (no phantom recipe, no
                        // mislocated T0030).
                        if (bc.isGround(t)) try bc.recordDescriptorReq(t);
                    }
                    bc.node_types[(node_idx).int()] = Type.rawptr;
                    return Type.rawptr;
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
                        try bc.sink.report(.{ .span = bc.typeArgsSpan(n.lhs), .code = .T0054 }, "'{s}' expects exactly one type argument", .{bn});
                        return .invalid;
                    }
                    if (val_args.len != 1)
                        try bc.sink.report(.{ .span = bc.argsSpan(node_idx), .code = .T0039 }, "'{s}' expects exactly one value argument", .{bn});
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
                if (bc.model.structs[sid].native_family == .ref) return try bc.rejectRefTupleCall(node_idx, n, sid);
                if (bc.model.structs[sid].is_tuple) return try bc.checkTupleStructInit(node_idx, n, sid);
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
        if (callee_res == .local) {
            const st = bc.slotType(callee_res.local);
            if (st.isFunc()) {
                bc.node_types[(n.lhs).int()] = st;
                return bc.typeOfValueCall(node_idx, n, st);
            }
            // A local poisoned by an earlier error may well hold a function.
            if (st.kind == .invalid) {
                try bc.drainArgs(Ast.rangeSlice(bc.tree, (n.rhs).int()));
                return .invalid;
            }
        }
        if (callee_res != .func and callee.tag != .identifier and callee.tag != .field_access)
            return bc.typeOfExprCall(node_idx, n);
        if (callee_res != .func) {
            // Type the args anyway so their own errors surface, then poison.
            for (Ast.rangeSlice(bc.tree, (n.rhs).int())) |arg| _ = try bc.typeOf(arg);
            if (callee_res == .local) {
                try bc.sink.report(.{ .span = bc.spanOf(n.lhs), .code = .T0055 }, "called value is not a function", .{});
            } else if (bc.tree.nodes[(n.lhs).int()].tag == .identifier) {
                // A struct-named callee `Point(1,2)` is positional construction, which
                // we reject — point at named construction instead.
                const cname = bc.nameText(bc.tree.nodes[(n.lhs).int()].main_token);
                if (bc.activeStructMap().get(cname) != null)
                    try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0063 }, "use named construction '{s} {{ ... }}', not '{s}(...)'", .{ cname, cname })
                    // A module or an enum binds no fn, and resolve reports neither, so the call
                    // would otherwise pass check and fail in lower (`5 |> io` is `io(5)`).
                else if (callee_res == .module)
                    try bc.sink.report(.{ .span = bc.spanOf(n.lhs), .code = .T0055 }, "module '{s}' is not a function; call one of its functions, as in '{s}.f(..)'", .{ cname, cname })
                else if (bc.activeEnumMap().get(cname) != null)
                    try bc.sink.report(.{ .span = bc.spanOf(n.lhs), .code = .T0055 }, "enum '{s}' is not a function; construct one of its variants, as in '{s}.v(..)'", .{ cname, cname });
            } else if (callee_res == .unresolved and bc.tree.nodes[(n.lhs).int()].tag == .field_access) {
                // A qualified call `recv.member(...)` whose callee stayed `.unresolved`:
                // resolve neither bound it to a fn nor reported it (e.g. `recv` is a
                // top-level fn shadowing an import namespace, so the field-access value
                // path is taken and left unresolved). Emit a clean diagnostic at the
                // member token instead of silently poisoning — otherwise the call is
                // dropped and `-o` later crashes in codegen with no user error.
                const fa = bc.tree.nodes[(n.lhs).int()];
                const member = bc.nameText(fa.main_token);
                try bc.sink.report(.{ .span = bc.tokSpan(fa.main_token), .code = .T0055 }, "cannot resolve member '{s}' to a callable function", .{member});
            }
            // Any remaining `.unresolved` was already reported by resolve.
            return .invalid;
        }
        const f = bc.model.fns[callee_res.func];
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        // The bodyless builtins: the bundled-library intrinsics, then the global
        // `panic(str)`. Intercept BEFORE the generic/arg-assignability check below
        // (a builtin carries only a placeholder sig — its arg-checking is token-based here).
        if (f.kind == .builtin) {
            // The bundled-library intrinsics: the core-only heap/raw-pointer ops and the
            // std-visible `ref_*` cell ops. Discriminated by the callee
            // token (a `FnSym` carries no name), typed BEFORE the panic 1-arg
            // path since their arities differ (0 for `gc_span_count`, 2 for `store`).
            if (callee.tag == .identifier) {
                const bn = bc.nameText(callee.main_token);
                if (Intrinsic.lookup(bn)) |ik| switch (ik) {
                    .gc_alloc => {
                        if (try bc.checkArity(node_idx, args, 1)) {
                            const st = try bc.typeOf(args[0]);
                            if (st.kind != .int and st.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[0]), .code = .T0040 }, "'gc_alloc' size must be an 'int', got '{s}'", .{bc.typeName(st)});
                        }
                        bc.node_types[(node_idx).int()] = Type.rawptr;
                        return Type.rawptr;
                    },
                    .gc_span_count => {
                        _ = try bc.checkArity(node_idx, args, 0);
                        bc.node_types[(node_idx).int()] = Type.int;
                        return Type.int;
                    },
                    .gc_collect => {
                        _ = try bc.checkArity(node_idx, args, 0);
                        bc.node_types[(node_idx).int()] = Type.unit;
                        return Type.unit;
                    },
                    .gc_stats => {
                        _ = try bc.checkArity(node_idx, args, 0);
                        bc.node_types[(node_idx).int()] = Type.int;
                        return Type.int;
                    },
                    .text_base => {
                        _ = try bc.checkArity(node_idx, args, 0);
                        bc.node_types[(node_idx).int()] = Type.rawptr;
                        return Type.rawptr;
                    },
                    .call_hash => {
                        if (try bc.checkArity(node_idx, args, 2)) {
                            const p0 = try bc.typeOf(args[0]);
                            const p1 = try bc.typeOf(args[1]);
                            if (!bc.in_unsafe)
                                try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0038 }, "raw pointer 'call_hash' requires an 'unsafe' block", .{});
                            if (p0.kind != .rawptr and p0.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[0]), .code = .T0040 }, "'call_hash' expects a 'rawptr', got '{s}'", .{bc.typeName(p0)});
                            if (p1.kind != .rawptr and p1.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[1]), .code = .T0040 }, "'call_hash' expects a 'rawptr', got '{s}'", .{bc.typeName(p1)});
                        }
                        bc.node_types[(node_idx).int()] = Type.int;
                        return Type.int;
                    },
                    .call_eq => {
                        if (try bc.checkArity(node_idx, args, 3)) {
                            const p0 = try bc.typeOf(args[0]);
                            const p1 = try bc.typeOf(args[1]);
                            const p2 = try bc.typeOf(args[2]);
                            if (!bc.in_unsafe)
                                try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0038 }, "raw pointer 'call_eq' requires an 'unsafe' block", .{});
                            if (p0.kind != .rawptr and p0.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[0]), .code = .T0040 }, "'call_eq' expects a 'rawptr', got '{s}'", .{bc.typeName(p0)});
                            if (p1.kind != .rawptr and p1.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[1]), .code = .T0040 }, "'call_eq' expects a 'rawptr', got '{s}'", .{bc.typeName(p1)});
                            if (p2.kind != .rawptr and p2.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[2]), .code = .T0040 }, "'call_eq' expects a 'rawptr', got '{s}'", .{bc.typeName(p2)});
                        }
                        bc.node_types[(node_idx).int()] = Type.bool;
                        return Type.bool;
                    },
                    .store => {
                        if (try bc.checkArity(node_idx, args, 2)) {
                            const pt = try bc.typeOf(args[0]);
                            _ = try bc.typeOf(args[1]);
                            if (!bc.in_unsafe)
                                try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0038 }, "raw pointer 'store' requires an 'unsafe' block", .{});
                            if (pt.kind != .rawptr and pt.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[0]), .code = .T0040 }, "'store' expects a 'rawptr', got '{s}'", .{bc.typeName(pt)});
                        }
                        bc.node_types[(node_idx).int()] = Type.unit;
                        return Type.unit;
                    },
                    .load => {
                        if (try bc.checkArity(node_idx, args, 1)) {
                            const pt = try bc.typeOf(args[0]);
                            if (!bc.in_unsafe)
                                try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0038 }, "raw pointer 'load' requires an 'unsafe' block", .{});
                            if (pt.kind != .rawptr and pt.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[0]), .code = .T0040 }, "'load' expects a 'rawptr', got '{s}'", .{bc.typeName(pt)});
                        }
                        // A `load` used where a scalar value type is expected (a generic
                        // `ga_get[T]` returning `load(..)`) yields that element type, so a
                        // `bool` element round-trips as `bool`; every other context reads a
                        // 64-bit `int` (mem_selftest is unaffected — its loads are int-typed).
                        // A generic-instance element type (`Vec[int]`, `Option[int]`) or a fn
                        // type is still check-time during the pre-reify instance recheck, so
                        // accept `.app`/`.func` too —
                        // reify later rewrites the load's node type to the concrete
                        // `structT`/`enumT`. (A ref-family element already rides `isRefPayload`; a
                        // scalar/`str`/float/struct/enum element rides its own kind.)
                        const lt: Type = if (bc.expectedTy()) |e|
                            (if (e.isScalar() or e.kind == .float or e.kind == .@"struct" or e.kind == .@"enum" or e.kind == .str or e.kind == .app or e.kind == .func or bc.isRefPayload(e)) e else Type.int)
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
                        if (try bc.checkArity(node_idx, args, 2)) {
                            const bt = try bc.typeOf(args[0]);
                            const it = try bc.typeOf(args[1]);
                            const base_ok = bt.kind == .rawptr or bt.kind == .int or bt.kind == .invalid or bc.isRefPayload(bt);
                            if (!base_ok)
                                try bc.sink.report(.{ .span = bc.spanOf(args[0]), .code = .T0040 }, "'offset' base must be a 'rawptr' or 'int', got '{s}'", .{bc.typeName(bt)});
                            if (it.kind != .int and it.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[1]), .code = .T0040 }, "'offset' index must be an 'int', got '{s}'", .{bc.typeName(it)});
                        }
                        bc.node_types[(node_idx).int()] = Type.rawptr;
                        return Type.rawptr;
                    },
                    .gc_array => {
                        // The bare-call form `gc_array(p)` is rejected; the wrapping form
                        // is `gc_array[T](p)`, intercepted in `typeOfCall` as a `type_app`.
                        try bc.drainArgs(args);
                        try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0054 }, "'gc_array' requires a type argument, e.g. gc_array[int](p)", .{});
                        return .invalid;
                    },
                    .ref_new => {
                        if (!try bc.checkArity(node_idx, args, 1)) return .invalid;
                        const operand = try bc.typeOfExpected(args[0], null);
                        if (operand.kind == .invalid) return .invalid; // poison propagation: no emit (exempt)
                        // Reify runs later, so `Ref[T]` is an `.app` during body-check.
                        const t = Type.app(bc.composite.intern(bc.gpa, bc.model.prelude.?.ref_struct.?, &.{operand}, false) catch return error.OutOfMemory);
                        bc.node_types[(node_idx).int()] = t;
                        return t;
                    },
                    .ref_get, .ref_set => {
                        if (!try bc.checkArity(node_idx, args, if (ik == .ref_get) 1 else 2)) return .invalid;
                        const rt = try bc.typeOfExpected(args[0], null);
                        const payload = bc.refPayload(rt);
                        if (payload.kind == .invalid and rt.kind != .invalid)
                            try bc.sink.report(.{ .span = bc.spanOf(args[0]), .code = .T0044 }, "'{s}' expects a 'Ref', got '{s}'", .{ bn, bc.typeName(rt) });
                        if (ik == .ref_get) {
                            bc.node_types[(node_idx).int()] = payload;
                            return payload;
                        }
                        const vt = try bc.typeOfExpected(args[1], if (payload.kind == .invalid) null else payload);
                        if (payload.kind != .invalid and !Type.assignable(payload, vt)) {
                            var hint: std.ArrayList(u8) = .empty;
                            defer hint.deinit(bc.gpa);
                            try bc.writeCaptureHint(&hint, args[1], payload, vt);
                            try bc.sink.report(.{ .span = bc.spanOf(args[1]), .code = .T0040 }, "argument 2: expected {s}, got {s}{s}", .{ bc.typeName(payload), bc.typeName(vt), hint.items });
                        }
                        bc.node_types[(node_idx).int()] = Type.unit;
                        return Type.unit;
                    },
                    .store_byte => {
                        if (try bc.checkArity(node_idx, args, 2)) {
                            const pt = try bc.typeOf(args[0]);
                            _ = try bc.typeOf(args[1]);
                            if (!bc.in_unsafe)
                                try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0038 }, "raw pointer 'store_byte' requires an 'unsafe' block", .{});
                            if (pt.kind != .rawptr and pt.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[0]), .code = .T0040 }, "'store_byte' expects a 'rawptr', got '{s}'", .{bc.typeName(pt)});
                        }
                        bc.node_types[(node_idx).int()] = Type.unit;
                        return Type.unit;
                    },
                    .load_byte => {
                        if (try bc.checkArity(node_idx, args, 1)) {
                            const pt = try bc.typeOf(args[0]);
                            if (!bc.in_unsafe)
                                try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0038 }, "raw pointer 'load_byte' requires an 'unsafe' block", .{});
                            if (pt.kind != .rawptr and pt.kind != .invalid)
                                try bc.sink.report(.{ .span = bc.spanOf(args[0]), .code = .T0040 }, "'load_byte' expects a 'rawptr', got '{s}'", .{bc.typeName(pt)});
                        }
                        bc.node_types[(node_idx).int()] = Type.int;
                        return Type.int;
                    },
                    // `size_of[T]()`/`align_of[T]()`/`descriptor_of[T]()` are the type_app
                    // form handled above; a bare-call misuse names no type, so it can never
                    // produce a size/align/descriptor — reject with the type-arg hint
                    // (mirroring `.gc_array`), never fall through to the generic path.
                    .size_of, .align_of, .descriptor_of => {
                        try bc.drainArgs(args);
                        try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0054 }, "'{s}' requires a type argument, e.g. {s}[int]()", .{ bn, bn });
                        return .invalid;
                    },
                };
            }
            // `panic(msg)`: the message must be a `str` (it lowers to a raw {ptr,len}
            // write; a non-str arg would be marshalled per its own ABI and misread).
            // Discriminate by the callee token — the Model fn carries no name. The call has
            // its seeded `never` return on every path, a bad arity or arg included: it still
            // cannot return, so the enclosing fn or arm must not also report a missing value.
            if (callee.tag == .identifier and std.mem.eql(u8, bc.nameText(callee.main_token), "panic")) {
                if (try bc.checkArity(node_idx, args, 1)) {
                    const at = try bc.typeOf(args[0]);
                    if (at.kind != .str and at.kind != .invalid and at.kind != .never)
                        try bc.sink.report(.{ .span = bc.spanOf(args[0]), .code = .T0040 }, "panic message must be a 'str', got '{s}'", .{bc.typeName(at)});
                }
                std.debug.assert(f.ret.kind == .never);
                bc.node_types[(node_idx).int()] = f.ret;
                return f.ret;
            }
            // No other 1-arg builtin exists post the intrinsic switch above; a `.func` here
            // is not a builtin call the checker recognizes. Reject rather than fall into the
            // generic path against a placeholder sig.
            return .invalid;
        }
        // A bare (no-explicit-args) generic call `id(7)`: infer each type-arg by
        // one-sided structural matching of the template's param patterns against the
        // ground argument types, then check the args + type the node. The matcher runs and
        // is discarded here — no `type_var` is ever stored.
        // Explicit `id[int](7)` is handled above (typeOfGenericCall) and still overrides.
        if (f.isGeneric()) {
            // Type-args are inferred for a PLAIN-IDENTIFIER callee `id(7)` OR a qualified
            // module-member callee `mod.id(7)`. A qualified pub fn binds `.func(gid)` onto
            // its field_access (resolveModuleMember), so this arm was reached with the SAME
            // gid a plain callee carries (the `!= .func` guard in typeOfDirectCall already
            // passed), hence a qualified module generic fn. The post-typecheck consumers key
            // both shapes off `resolutions[call.lhs]`, so the inferred instance is found at
            // lower. Any other callee (e.g. a value-index `xs[i]()`) cannot infer — require
            // explicit type args.
            if (callee.tag != .identifier and callee.tag != .field_access) {
                try bc.drainArgs(args);
                try bc.sink.report(.{ .span = bc.spanOf(n.lhs), .code = .T0013 }, "generic call requires explicit type arguments, e.g. f[int](..)", .{});
                return .invalid;
            }
            const sig = bc.fnSig(f, bc.nameText(callee.main_token));
            if (args.len != f.params.len) {
                for (args) |arg| _ = try CallInfer.typeOfGenericArg(bc, arg, bc.partHint());
                try bc.reportArity(node_idx, args.len, sig);
                return .invalid; // never leak f.ret (a type_var) on the error path
            }
            const arg_types = try bc.gpa.alloc(Type, args.len);
            defer bc.gpa.free(arg_types);
            const out = try bc.gpa.alloc(Type, f.generic_params.len);
            defer bc.gpa.free(out);
            switch (try CallInfer.inferArgs(bc, .{ .call = node_idx, .sig = sig }, args, f.params, f.generic_params, null, .{ .ret = f.ret }, arg_types, out)) {
                .failed => return .invalid,
                .unbound => |ord| {
                    try bc.reportUninferable(bc.spanOf(node_idx), " '{s}'", .{f.generic_params[ord]}, bc.calleeText(n.lhs), out, &.{"(..)"}, null);
                    return .invalid;
                },
                .ok => {
                    if (!try CallInfer.concreteTargs(bc, node_idx, out)) return .invalid;
                    var inferred = sig.substituted(out);
                    inferred.pretyped = arg_types;
                    _ = try bc.checkCall(node_idx, args, inferred);
                    return bc.substRet(node_idx, f, out);
                },
            }
        }
        _ = try bc.checkCall(node_idx, args, bc.fnSig(f, bc.nameText(callee.main_token)));
        return f.ret;
    }

    fn dispatchValueMethod(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node) error{OutOfMemory}!?Type {
        const recv_ty = try bc.typeOfReceiver(callee.lhs); // also populates node_types[recv] for lower
        if (recv_ty.kind == .invalid) return Type.invalid; // receiver already errored → no cascade
        if (recv_ty.kind == .@"struct" or recv_ty.kind == .@"enum" or recv_ty.isScalar() or recv_ty.kind == .float) {
            return try bc.dispatchConcreteMethod(node_idx, n, callee, recv_ty);
        } else if (recv_ty.kind == .app) {
            return try bc.dispatchAppMethod(node_idx, n, callee, recv_ty);
        } else if (recv_ty.isTypeVar()) {
            return try bc.dispatchTypeVarMethod(node_idx, n, callee, recv_ty);
        } else if (recv_ty.isFunc()) {
            try bc.drainArgs(Ast.rangeSlice(bc.tree, (n.rhs).int()));
            try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0018 }, "function values have no methods ('{s}' on '{s}')", .{ bc.nameText(callee.main_token), bc.typeName(recv_ty) });
            return Type.invalid;
        }
        // No final else: an unhandled receiver kind (.never) falls through to the
        // callee_res != .func path in typeOfCall.
        return null;
    }

    fn dispatchConcreteMethod(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node, recv_ty: Type) error{OutOfMemory}!Type {
        const member = bc.nameText(callee.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        var bound_args: std.ArrayList(Type) = .empty;
        defer bound_args.deinit(bc.gpa);
        const bound = try bc.boundConformance(n, recv_ty, &bound_args);
        // `.one` is byte-identical to `findMethod` for every existing program (inherent /
        // single conformance / prelude); `.ambiguous` (a doubly-conforming generic
        // protocol used with no type-args) is T0025 — never an arbitrary pick.
        switch (Typecheck.pickMethodIndex(bc.model.methods, recv_ty, member, if (bound) |b| b.pid else null, if (bound) |b| b.args else null, bc.methodSite())) {
            .one => |mi| {
                // An associated (self-less) function must be reached via `Type[..].fn()`,
                // never on a value receiver.
                if (bc.model.methods[mi].is_static) {
                    try bc.drainArgs(args);
                    const has_field = recv_ty.kind == .@"struct" and bc.fnField(recv_ty.struct_id, member, &.{}) != null;
                    try bc.reportAssociatedOnValue(callee, bc.typeName(recv_ty), "", has_field);
                    return .invalid;
                }
                return try bc.dispatchMethod(node_idx, n, callee, recv_ty, member, mi);
            },
            .ambiguous => {
                try bc.drainArgs(args);
                try bc.emitAmbiguousConformance(bc.tokSpan(callee.main_token), recv_ty, member, null);
                return .invalid;
            },
            .none => {},
        }
        // Target-directed `.into()` / `.try_into()` on an integer OR `char` receiver:
        // resolve the destination from the expected type. `into` is lossless (int widen /
        // char↔int·byte); `try_into` is fallible (int narrow / int→char / char→byte),
        // returning a synthesized `Result[T, ConvErr]`. A user `into`/`try_into` method was
        // already selected above; the builtin recognizer only matches int/char, so a
        // struct receiver with no such method falls through to T0018 — never a wrong pick.
        const char_id = if (bc.model.prelude) |p| p.char_struct else null;
        const recv_is_char = Typecheck.isCharTy(recv_ty, char_id);
        if ((recv_ty.isInteger() or recv_is_char or recv_ty.kind == .float) and (std.mem.eql(u8, member, "into") or std.mem.eql(u8, member, "try_into"))) {
            if (bc.expectedTy()) |exp| {
                if (Typecheck.builtinConvMethod(recv_ty, exp, member, char_id)) |cm| {
                    bc.setCallee(n, Callee.builtinMethod(if (std.mem.eql(u8, member, "into")) .conv_into else .conv_try_into));
                    _ = try bc.checkCall(node_idx, args, .{ .params = &.{}, .name = member });
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
            if (!try bc.checkCall(node_idx, args, .{ .params = &.{}, .name = member })) {
                if (try bc.conformsTo(recv_ty, bc.model.preludeProtocols().hash, node_idx)) bc.setCallee(n, Callee.pending);
                bc.node_types[(node_idx).int()] = Type.int;
                return Type.int;
            }
            if (try bc.conformsTo(recv_ty, bc.model.preludeProtocols().hash, node_idx)) {
                bc.setCallee(n, Callee.pending);
                bc.node_types[(node_idx).int()] = Type.int;
                return Type.int;
            }
            if (try bc.deriveBlocker(recv_ty, bc.model.preludeProtocols().hash)) |blocker| {
                try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0030 }, "cannot derive 'Hash' for '{s}': field '{s}' of type '{s}' does not conform to 'Hash'", .{ bc.typeName(recv_ty), blocker.name, bc.typeName(blocker.ty) });
                return .invalid;
            }
            // No structural derive and no nameable blocker: fall through to T0018.
        }
        // A direct `.to_string()` with NO explicit `to_string` fn on any type the builtin
        // scalar list leaves out: a struct, an enum, or a scalar such as `float` whose
        // Display is an `impl` in std. One gate for all of them, the shared
        // `conformsTo(display)`, so a custom `impl Display` (a `.direct` conformance) reuses
        // its witness and a structural derive is recorded only for an aggregate. A struct
        // with a non-Display field names it (T0031); anything else that does not conform
        // falls through to T0018.
        if (std.mem.eql(u8, member, "to_string") and bc.model.preludeProtocols().display != null and
            Typecheck.builtinStrMethod(recv_ty, member) == null)
        {
            if (!try bc.checkCall(node_idx, args, .{ .params = &.{}, .name = member })) {
                bc.setCallee(n, Callee.builtinMethod(.display_to_string));
                bc.node_types[(node_idx).int()] = Type.str;
                return Type.str;
            }
            if (try bc.conformsTo(recv_ty, bc.model.preludeProtocols().display, node_idx)) {
                bc.setCallee(n, Callee.builtinMethod(.display_to_string));
                bc.node_types[(node_idx).int()] = Type.str;
                return Type.str;
            }
            if (try bc.deriveBlocker(recv_ty, bc.model.preludeProtocols().display)) |blocker| {
                var buf: [64]u8 = undefined;
                const field_ty = bc.typeName(blocker.ty);
                const hint = importHint(&buf, Typecheck.scalarImplModule(bc.model, field_ty, bc.model.preludeProtocols().display.?));
                try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0031 }, "cannot 'to_string' a '{s}': field '{s}' of type '{s}' does not conform to 'Display'{s}", .{ bc.typeName(recv_ty), blocker.name, field_ty, hint });
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
            bc.setCallee(n, Callee.builtinMethod(bm.builtin));
            const self_param = [_]Type{recv_ty};
            _ = try bc.checkCall(node_idx, args, .{ .params = self_param[0..bm.arity], .name = member });
            bc.node_types[(node_idx).int()] = bm.ret;
            return bm.ret;
        }
        // The string builtin methods (`concat`/`len`/`byte_at`/`to_string`) — kept out of
        // `builtinScalarMethod`, whose params are all `Self` (`byte_at` takes an int).
        if (Typecheck.builtinStrMethod(recv_ty, member)) |bm| {
            bc.setCallee(n, Callee.builtinMethod(bm.builtin));
            _ = try bc.checkCall(node_idx, args, .{ .params = bm.params, .name = member });
            bc.node_types[(node_idx).int()] = bm.ret;
            return bm.ret;
        }
        // No method: a fn-typed field of that name is called as a value (`h.run(4)`).
        if (recv_ty.kind == .@"struct") if (bc.fnField(recv_ty.struct_id, member, &.{})) |fty|
            return bc.typeOfFieldCall(node_idx, n, fty);
        // A concrete struct/enum/scalar value with no such method.
        try bc.drainArgs(args);
        try bc.emitNoMethod(bc.tokSpan(callee.main_token), member, recv_ty);
        return .invalid;
    }

    /// T0018 at `recv.m(..)` where `m` is an associated fn. A fn-typed field `m` beside it
    /// is not called instead (a call selects the method first); the message only says how
    /// to reach each, because "call it as 'T.m(..)'" alone misleads someone after the field.
    fn reportAssociatedOnValue(bc: *BodyChecker, callee: Ast.Node, type_name: []const u8, targs: []const u8, has_field: bool) error{OutOfMemory}!void {
        const member = bc.nameText(callee.main_token);
        const at: DiagnosticSink.Report = .{ .span = bc.tokSpan(callee.main_token), .code = .T0018 };
        if (!has_field) return bc.sink.report(at, "'{s}' is an associated function; call it as '{s}{s}.{s}(..)'", .{ member, type_name, targs, member });
        const rs = bc.spanOf(callee.lhs);
        const text = bc.source[rs.start..rs.end];
        const recv = if (std.mem.indexOfScalar(u8, text, '\n') == null) text else "x";
        try bc.sink.report(at, "'{s}' is both an associated function of '{s}' and a field of fn type: '{s}{s}.{s}(..)' calls the function and '{s}.{s}' reads the field", .{ member, type_name, type_name, targs, member, recv, member });
    }

    fn dispatchAppMethod(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node, recv_ty: Type) error{OutOfMemory}!Type {
        // A method call on a generic-type instance `b.get()`: the
        // receiver is an `App` (`Box[int]`; reify runs later in the mono
        // tail). Its ctor selects the `impl Box[T]` method TEMPLATE and the
        // impl's type-params bind by matching the template's `Self`
        // pattern-args against the receiver App's concrete args — the SAME
        // `Infer.match` scanCalls/lower run.
        const member = bc.nameText(callee.main_token);
        const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
        const e = bc.composite.at(recv_ty.appIdx());
        var by_bound = false;
        const picked: ?Typecheck.TemplateMethod = switch (Typecheck.pickGenericMethod(bc.model.templates, e.ctor, e.ctor_is_enum, member, bc.methodSite())) {
            .one => |m| m,
            .ambiguous => if (try bc.boundTemplate(n, recv_ty, member)) |m| blk: {
                by_bound = true;
                break :blk m;
            } else {
                try bc.drainArgs(args);
                try bc.emitAmbiguousConformance(bc.tokSpan(callee.main_token), recv_ty, member, null);
                return .invalid;
            },
            .none => null,
        };
        if (picked) |m| {
            // An associated (self-less) function is not callable on a value receiver —
            // it must be reached via `Type[args].fn()`.
            if (!m.has_self) {
                try bc.drainArgs(args);
                const has_field = !e.ctor_is_enum and bc.fnField(e.ctor, member, e.args) != null;
                try bc.reportAssociatedOnValue(callee, bc.typeName(recv_ty), "[..]", has_field);
                return .invalid;
            }
            const mf = bc.model.fns[m.fn_id];
            if (m.mut_self) try bc.checkMutReceiver(callee.lhs, member);
            if (Typecheck.hasOwnTypeParams(bc.composite, mf)) return CallInfer.checkOwnGenericCall(bc, node_idx, n, callee, recv_ty, mf, member, false);
            const n_gp: u32 = @intCast(mf.generic_params.len);
            const targs = try bc.gpa.alloc(Type, n_gp);
            defer bc.gpa.free(targs);
            const bnd = try bc.gpa.alloc(bool, n_gp);
            defer bc.gpa.free(bnd);
            const fp = try bc.gpa.alloc(usize, n_gp);
            defer bc.gpa.free(fp);
            const pat: []const Type = if (mf.self_type.isApp()) bc.composite.at(mf.self_type.appIdx()).args else &.{};
            var bound_ok = pat.len == e.args.len;
            if (bound_ok) switch (Infer.match(n_gp, pat, e.args, targs, bnd, fp, bc.composite.decomposer())) {
                .ok => {},
                else => bound_ok = false, // an unbound impl param: poison, no cascade
            };
            if (bound_ok and by_bound) {
                // The receiver's rows share this name; only the template tells them apart.
                bc.setCallee(n, Callee.pendingCall(node_idx));
                bc.setCalleeAt(node_idx, Callee.pendingTemplate(m.fn_id, false));
            } else if (bound_ok) bc.setCallee(n, Callee.pending);
            _ = try bc.checkCall(node_idx, args, bc.methodSig(mf, member).substituted(if (bound_ok) targs else null));
            if (!bound_ok) return .invalid;
            return bc.substRet(node_idx, mf, targs);
        }
        // A compiler-provided inherent method on the prelude `Option`/`Result`
        // enums: recognized by ctor id, NOT a `t.methods` entry (source-
        // less, like the builtin scalar `eq`/`hash`), so `pickGenericMethod`
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
                if (native_ok) {
                    bc.setCallee(n, Callee.builtinMethod(Typecheck.optResultBuiltin(op)));
                    switch (op) {
                        .is_tag0, .is_tag1 => {
                            _ = try bc.checkCall(node_idx, args, .{ .params = &.{}, .name = member });
                            bc.node_types[(node_idx).int()] = Type.bool;
                            return Type.bool;
                        },
                        .unwrap => {
                            _ = try bc.checkCall(node_idx, args, .{ .params = &.{}, .name = member });
                            bc.node_types[(node_idx).int()] = t_ty;
                            return t_ty;
                        },
                        .unwrap_or => {
                            const payload = [_]Type{t_ty};
                            _ = try bc.checkCall(node_idx, args, .{ .params = &payload, .name = member });
                            bc.node_types[(node_idx).int()] = t_ty;
                            return t_ty;
                        },
                    }
                }
            }
        }
        // The universal `to_string` on a reified generic instance (e.g. `Option[int]`):
        // it reifies to a concrete struct/enum whose structural `Display` derive fires
        // through the SAME `conformsTo(display)` gate a plain struct/enum uses. Mirrors the
        // struct/enum `to_string` arm so `io.print(Option[int])` — whose body is
        // `x.to_string()` at `x: Option[int]` — types `str` and records the derive.
        if (std.mem.eql(u8, member, "to_string") and bc.model.preludeProtocols().display != null) {
            if (!try bc.checkCall(node_idx, args, .{ .params = &.{}, .name = member })) {
                bc.setCallee(n, Callee.builtinMethod(.display_to_string));
                bc.node_types[(node_idx).int()] = Type.str;
                return Type.str;
            }
            if (try bc.conformsTo(recv_ty, bc.model.preludeProtocols().display, node_idx)) {
                bc.setCallee(n, Callee.builtinMethod(.display_to_string));
                bc.node_types[(node_idx).int()] = Type.str;
                return Type.str;
            }
            if (try bc.deriveBlocker(recv_ty, bc.model.preludeProtocols().display)) |blocker| {
                var buf: [64]u8 = undefined;
                const field_ty = bc.typeName(blocker.ty);
                const hint = importHint(&buf, Typecheck.scalarImplModule(bc.model, field_ty, bc.model.preludeProtocols().display.?));
                try bc.sink.report(.{ .span = bc.spanOf(node_idx), .code = .T0031 }, "cannot 'to_string' a '{s}': field '{s}' of type '{s}' does not conform to 'Display'{s}", .{ bc.typeName(recv_ty), blocker.name, field_ty, hint });
                return .invalid;
            }
            if (try bc.unmetGenericBound(recv_ty, bc.model.preludeProtocols().display)) |u| {
                try bc.drainArgs(args);
                try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0018 }, "no method 'to_string' on type '{s}'; its generic impl of 'Display' needs type argument '{s}' to conform to '{s}'", .{ bc.typeName(recv_ty), bc.typeName(u.arg), bc.model.protocols[u.pid].name });
                return .invalid;
            }
        }
        // No method: a fn-typed field of that name, its type substituted through the args.
        if (!e.ctor_is_enum) if (bc.fnField(e.ctor, member, e.args)) |fty|
            return bc.typeOfFieldCall(node_idx, n, fty);
        // A generic-type value with no such method.
        try bc.drainArgs(args);
        try bc.emitNoMethod(bc.tokSpan(callee.main_token), member, recv_ty);
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
            // `to_string` is the universal Display renderer, callable on ANY Display-conforming
            // receiver — including a `type_var` bounded by Display (as-axiom in the template
            // body). It is not a Display protocol *method*, so the bound-method search below
            // would miss it; recognize it here so `x.to_string()` in a `[T has Display]` body
            // types `str`. The per-instance re-check grounds the receiver, so the concrete
            // struct/enum/scalar `to_string` path (with its derive/witness) handles lowering.
            if (std.mem.eql(u8, member, "to_string") and bc.model.preludeProtocols().display != null and
                pid == bc.model.preludeProtocols().display.?)
            {
                bc.setCallee(n, Callee.builtinMethod(.display_to_string));
                _ = try bc.checkCall(node_idx, args, .{ .params = &.{}, .name = member });
                bc.node_types[(node_idx).int()] = Type.str;
                return Type.str;
            }
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
                bc.setCallee(n, Callee.protocolMethod(pid, @intCast(k)));
                const psig = p.method_params[k]; // [self, ...]
                _ = try bc.checkCall(node_idx, args, .{
                    .params = psig[@min(psig.len, 1)..],
                    .name = member,
                    .map = .{ .ground = .{ .self_ty = recv_ty, .pargs = pargs } },
                });
                const ret = Typecheck.groundProtoType(bc, p.method_rets[k], recv_ty, pargs);
                bc.node_types[(node_idx).int()] = ret;
                return ret;
            }
            try bc.drainArgs(args);
            try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0018 }, "no method '{s}' on type parameter bounded by protocol '{s}'", .{ member, p.name });
            return .invalid;
        }
        try bc.drainArgs(args);
        try bc.sink.report(.{ .span = bc.tokSpan(callee.main_token), .code = .T0018 }, "no method '{s}' on unbounded type parameter", .{member});
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
            try bc.drainArgs(args);
            // The base of a `[..]` callee that is not a fn: nothing else takes type
            // args (no generic types in value position), so it is a plain
            // not-a-function.
            try bc.sink.report(.{ .span = bc.spanOf(n.lhs), .code = .T0055 }, "called value is not a function", .{});
            return .invalid;
        }
        const f = bc.model.fns[base_res.func];
        const targ_nodes = Ast.rangeSlice(bc.tree, (callee.rhs).int());
        if (!f.isGeneric()) {
            for (targ_nodes) |tn| bc.node_types[(tn).int()] = bc.typeFromNode(tn);
            try bc.drainArgs(args);
            try bc.sink.report(.{ .span = bc.typeArgsSpan(n.lhs), .code = .T0054 }, "'{s}' is not generic; drop the type arguments", .{bc.nameText(bc.tree.nodes[(callee.lhs).int()].main_token)});
            return f.ret;
        }
        if (targ_nodes.len != f.generic_params.len) {
            for (targ_nodes) |tn| bc.node_types[(tn).int()] = bc.typeFromNode(tn);
            try bc.drainArgs(args);
            try bc.sink.report(.{ .span = bc.typeArgsSpan(n.lhs), .code = .T0054 }, "expected {d} type argument(s), got {d}", .{ f.generic_params.len, targ_nodes.len });
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
            } else if (bc.abstract_template and ty.isTypeVar()) {
                // A bounded template's own param passed to a nested generic call: it
                // grounds at the outer template's instantiation, so it is admitted here
                // (the reified instance re-check substitutes a concrete type and re-gates).
                // Left `all_concrete` so the call types to its substituted (type_var)
                // return — matching the enclosing template's own type_var signature.
            } else if (!isConcreteValue(ty)) {
                // A `type_var` is not a monomorphizable type-arg. A ground `App`
                // (`Box[int]`) IS — it reifies to a concrete struct in the mono tail — and
                // `unit` IS (a zero-sized `()` type-arg); `isConcreteValue` admits both.
                try bc.sink.report(.{ .span = bc.spanOf(tn), .code = .T0013 }, "type argument must be a concrete value type", .{});
                all_concrete = false;
            }
        }
        // Check the value args against the substituted params regardless (surface arg
        // errors), but only produce the concrete return type when every type-arg is
        // sound — else poison so no `type_var`/half-substituted type is stored.
        const name = bc.nameText(bc.tree.nodes[(callee.lhs).int()].main_token);
        _ = try bc.checkCall(node_idx, args, bc.fnSig(f, name).substituted(if (all_concrete) targs else null));
        if (!all_concrete) return .invalid;
        return bc.substRet(node_idx, f, targs);
    }

    /// Type the call node as `f`'s return substituted through `targs`: the tail of every
    /// generic call path (explicit, inferred, associated, generic-impl method).
    pub fn substRet(bc: *BodyChecker, node_idx: Ast.Index, f: FnSym, targs: []const Type) Type {
        const ret = substTy(bc, f.ret, targs);
        bc.node_types[(node_idx).int()] = ret;
        return ret;
    }

    /// Whether `t` conforms to the derivable prelude protocol `pid_opt` — the shared
    /// predicate behind the `==`/`!=` (`Eq`), `<`/`>`/`<=`/`>=` (`Ord`),
    /// `.hash()` (`Hash`), and `to_string`/`x.to_string()` (`Display`) operator/trigger typings.
    /// A concrete type resolves via the frozen conformance table (`Conform.existence` covers
    /// the int/bool/str/unit prelude conformances AND every user `impl T has P`); a
    /// `type_var` in a bounded generic body conforms as-axiom when its declared bound IS
    /// `pid`. A prelude-less caller (`pid_opt == null`) denies conformance, so the operator
    /// emits its T002x rather than miscompiling.
    /// After those misses, a struct (or enum) whose fields all conform with NO explicit impl
    /// conforms STRUCTURALLY. With a `record_at` expression, record the derive request (so the serial
    /// synthesis barrier emits the source-less witness) and return true. `Conform.classify`
    /// tries the direct (existence ∨ axiom) tier first, keeping an explicit/refinement type
    /// off the structural path (no double-fire). Fallible (`Conform.structural` + the request
    /// record allocate) and takes `*BodyChecker` (records into the thread-local `derive_reqs`).
    fn conformsTo(bc: *BodyChecker, t: Type, pid_opt: ?u32, record_at: ?Ast.Index) error{OutOfMemory}!bool {
        const pid = pid_opt orelse return false;
        switch (try Conform.classify(bc.model, bc.composite, bc.bound_protocols, &bc.conforms_memo, bc.gpa, t, pid)) {
            .none => return false,
            .direct => return true,
            // An explicit generic conformance: its witness is an instance of the impl's
            // method, which the mono tail instantiates from this same request.
            .generic, .structural => {
                // Record a derive request ONLY for a GROUND operand: an abstract `App`
                // (a `type_var` inside, e.g. `Box[T]` in a bounded template's definition
                // check) is accepted-but-not-recorded — its concrete instance re-check
                // records the ground `Box[int]`, which reify+synthesize can mint a witness for.
                if (record_at) |site| if (bc.isGround(t)) try bc.recordDeriveReq(pid, t, site);
                return true;
            },
        }
    }

    /// The impl-param bound that keeps `t`'s explicit generic impl of `pid_opt` from
    /// applying, if one does: an error naming it beats suggesting an impl that exists.
    fn unmetGenericBound(bc: *BodyChecker, t: Type, pid_opt: ?u32) error{OutOfMemory}!?Conform.UnmetBound {
        const pid = pid_opt orelse return null;
        if (!t.isApp()) return null;
        const tb: Conform.Tables = .{ .structs = bc.model.structs, .enums = bc.model.enums, .conformances = bc.model.conformances, .gen = .of(bc.model), .composite = bc.composite, .gpa = bc.gpa, .bound_protocols = bc.bound_protocols };
        return switch (try Conform.generic(tb, &bc.conforms_memo, t, pid, &.{})) {
            .none => null,
            .met => |m| {
                m.deinit(bc.gpa);
                return null;
            },
            .unmet => |u| u,
        };
    }

    /// Whether `t` is fully ground: no `type_var` anywhere. A struct/enum id and any
    /// scalar are ground; a `type_var` is not; an `App` or a fn type is ground iff every
    /// arg is. Gates `recordDeriveReq` so the synthesis barrier only ever sees a concrete
    /// (reifiable) type.
    pub fn isGround(bc: *BodyChecker, t: Type) bool {
        if (t.isTypeVar()) return false;
        if (t.isApp() or t.isFunc()) {
            const e = bc.composite.at(t.struct_id);
            for (e.args) |a| if (!bc.isGround(a)) return false;
            return true;
        }
        return true;
    }

    /// Whether `t` can serve as an expected type (for a lambda param, a `_` capture's param,
    /// a struct field's value, or a generic method arg once the receiver binds): a ground type,
    /// or in a bounded template's abstract check one that also holds the template's own
    /// type vars, which are as known there as any concrete type. A var of a generic call
    /// that is still open takes an ordinal past them (`openOrdBase`), so it gives none.
    pub fn givesType(bc: *BodyChecker, t: Type) bool {
        if (t.isTypeVar()) return bc.abstract_template and t.typeVarOrd() < bc.openOrdBase();
        if (t.isApp() or t.isFunc()) {
            const e = bc.composite.at(t.struct_id);
            for (e.args) |a| if (!bc.givesType(a)) return false;
            return true;
        }
        return true;
    }

    /// Where a generic call's still-open var `ord` sits in the type a waiting arg is
    /// checked against: at `openOrdBase() + ord`. In a bounded template's abstract check
    /// that is past the template's own vars, which a bound var may stand for, so the two
    /// never share an ordinal.
    pub fn openOrdBase(bc: *const BodyChecker) u32 {
        return if (bc.abstract_template) @intCast(bc.subst.?.types.len) else 0;
    }

    /// The `T` of a prelude `Ref[T]` App, or `.invalid` for any other type.
    fn refPayload(bc: *BodyChecker, t: Type) Type {
        if (!t.isApp()) return .invalid;
        const rs = (bc.model.prelude orelse return .invalid).ref_struct orelse return .invalid;
        const e = bc.composite.at(t.appIdx());
        if (!e.ctor_is_enum and e.ctor == rs) return e.args[0];
        return .invalid;
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
    fn recordDeriveReq(bc: *BodyChecker, pid: u32, t: Type, site: Ast.Index) error{OutOfMemory}!void {
        for (bc.derive_reqs.items) |r| if (r.protocol_id == pid and Type.eql(r.conform_ty, t)) return;
        try bc.derive_reqs.append(bc.gpa, .{
            .protocol_id = pid,
            .conform_ty = t,
            .site = .{ .scope = bc.sink.cur_scope, .node = site },
        });
    }

    /// Record a descriptor request once per type in this fn (a fn may `descriptor_of[T]`
    /// repeatedly; the synthesis barrier dedups across fns too).
    fn recordDescriptorReq(bc: *BodyChecker, t: Type) error{OutOfMemory}!void {
        for (bc.descriptor_reqs.items) |r| if (Type.eql(r, t)) return;
        try bc.descriptor_reqs.append(bc.gpa, t);
    }

    /// The first struct field that blocks a structural derive of `pid_opt` (Eq T0029,
    /// Hash T0030, Display T0031), for the message that names it; null when `t` is
    /// not a struct, has no such protocol, or every field conforms. Struct-only: a payload
    /// enum with a non-conforming payload has no single nameable field, so it falls through
    /// to the generic "does not conform" message instead.
    fn deriveBlocker(bc: *BodyChecker, t: Type, pid_opt: ?u32) error{OutOfMemory}!?Typecheck.NonConformingField {
        if (t.kind != .@"struct") return null;
        const pid = pid_opt orelse return null;
        return Conform.firstNonConformingField(bc.model.structs, bc.model.enums, bc.model.conformances, .of(bc.model), t, pid, &bc.conforms_memo, bc.gpa, bc.composite, bc.bound_protocols);
    }

    /// Where an operator error underlines, resolved only when one is reported: a
    /// node's span walks its whole subtree, too costly to pay on every operator.
    const Blame = union(enum) { node: Ast.Index, token: u32 };

    fn blameSpan(bc: *const BodyChecker, blame: Blame) Span {
        return switch (blame) {
            .node => |n| bc.spanOf(n),
            .token => |t| bc.tokSpan(t),
        };
    }

    /// The result type of the binary operator `op` over operands already typed `lt`
    /// and `rt`, reporting at `blame`. `site` is where a structural derive request is
    /// recorded. Shared by `a op b` and the compound `place op= v`, so the two forms
    /// type and diagnose alike.
    fn binaryResult(bc: *BodyChecker, site: Ast.Index, blame: Blame, op: TokenTag, op_text: []const u8, lt: Type, rt: Type) error{OutOfMemory}!Type {
        switch (op) {
            .plus, .minus, .star, .slash => {
                // `+`/`-`/`*`/`/` desugar to the arithmetic protocols Add/Sub/Mul/Div
                //. Builtin `int` stays inline (a single machine add/sub/mul/sdiv,
                // no witness). Otherwise HOMOGENEOUS like `==`/`<`: require the same
                // type FIRST, then type to the operand type (`Out = Self`) iff it
                // conforms to the operator's protocol — a user struct/enum `impl T has
                // Add`, or a `[T has Add]` bound in a generic body. str/bool have NO
                // arithmetic impl (str concat allocates, deferred), so they fall to T0028.
                if (lt.isInteger() and Type.eql(lt, rt)) return lt;
                const ap = bc.arithProtocol(op);
                if (!Type.eql(lt, rt)) {
                    try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0044 }, "operands of '{s}' must have the same type", .{op_text});
                } else if (bc.conformsToArith(lt, ap.pid)) {
                    return lt;
                } else if (lt.kind == .float) {
                    // Every arithmetic operator is one character, and its float twin puts
                    // a `.` after it: `+` is `+.`, `+=` is `+.=`.
                    try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0028 }, "'{s}' requires an '{s}' impl for type '{s}'; use '{s}.{s}' for float arithmetic", .{ op_text, ap.name, bc.typeName(lt), op_text[0..1], op_text[1..] });
                } else {
                    try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0028 }, "'{s}' requires an '{s}' impl for type '{s}'", .{ op_text, ap.name, bc.typeName(lt) });
                }
            },
            .lt, .lt_eq, .gt, .gt_eq => {
                // `<`/`>`/`<=`/`>=` desugar to a discriminant test on `Ord::cmp`'s
                // result. HOMOGENEOUS like `==`: require the same type FIRST,
                // then type to bool iff the operand conforms to `Ord` — a concrete
                // int/str/bool prelude conformance or user struct/enum `impl T has Ord`,
                // or a `[T has Ord]` bound in a generic body.
                if (!Type.eql(lt, rt)) {
                    try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0044 }, "operands of '{s}' must have the same type", .{op_text});
                } else if (try bc.conformsTo(lt, bc.model.preludeProtocols().ord, site)) {
                    return Type.bool;
                } else if (try bc.unmetGenericBound(lt, bc.model.preludeProtocols().ord)) |u| {
                    try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0027 }, "'{s}' requires an 'Ord' impl for type '{s}'; its generic impl needs type argument '{s}' to conform to '{s}'", .{ op_text, bc.typeName(lt), bc.typeName(u.arg), bc.model.protocols[u.pid].name });
                } else {
                    try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0027 }, "'{s}' requires an 'Ord' impl for type '{s}'", .{ op_text, bc.nonConformingName(lt, bc.model.preludeProtocols().ord) });
                }
            },
            .eq_eq, .bang_eq => {
                // `==`/`!=` desugar to `Eq::eq`. HOMOGENEOUS: require the same
                // type FIRST (so a cross-type compare stays "same type" even if both
                // sides individually conform), then type to bool iff the operand
                // conforms to `Eq` — a concrete int/bool/str/unit prelude conformance
                // or user struct/enum impl, or a `[T has Eq]` bound in a generic body.
                if (!Type.eql(lt, rt)) {
                    try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0044 }, "operands of '{s}' must have the same type", .{op_text});
                } else if (lt.isFunc()) {
                    // Function values compare by identity (code and env), not `Eq`.
                    return Type.bool;
                } else if (try bc.conformsTo(lt, bc.model.preludeProtocols().eq, site)) {
                    return Type.bool;
                } else if (try bc.unmetGenericBound(lt, bc.model.preludeProtocols().eq)) |u| {
                    try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0026 }, "'{s}' requires an 'Eq' impl for type '{s}'; its generic impl needs type argument '{s}' to conform to '{s}'", .{ op_text, bc.typeName(lt), bc.typeName(u.arg), bc.model.protocols[u.pid].name });
                } else if (try bc.deriveBlocker(lt, bc.model.preludeProtocols().eq)) |blocker| {
                    // A struct that would derive `Eq` but for one non-conforming
                    // field names that field (T0029). A payload enum / other type
                    // keeps the "no Eq impl" T0026 below.
                    try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0029 }, "cannot derive 'Eq' for '{s}': field '{s}' of type '{s}' does not conform to 'Eq'", .{ bc.typeName(lt), blocker.name, bc.typeName(blocker.ty) });
                } else {
                    try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0026 }, "'{s}' requires an 'Eq' impl for type '{s}'", .{ op_text, bc.nonConformingName(lt, bc.model.preludeProtocols().eq) });
                }
            },
            .amp_amp, .pipe_pipe => {
                if (lt.kind == .bool and rt.kind == .bool) return Type.bool;
                try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0044 }, "operands of '{s}' must be bool", .{op_text});
            },
            .amp, .pipe, .caret, .lt_lt, .gt_gt, .percent => {
                if (lt.isInteger() and Type.eql(lt, rt)) return lt;
                try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0044 }, "operands of '{s}' must be int", .{op_text});
            },
            .plus_dot, .minus_dot, .star_dot, .slash_dot => {
                // The dotted operators are float-ONLY inline machine ops (no protocol
                // desugar): the non-dotted `+`/`<` on a float falls to T0028/T0027.
                if (lt.kind == .float and Type.eql(lt, rt)) return lt;
                try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0044 }, "operands of '{s}' must both be float", .{op_text});
            },
            .lt_dot, .gt_dot, .le_dot, .ge_dot => {
                if (lt.kind == .float and Type.eql(lt, rt)) return Type.bool;
                try bc.sink.report(.{ .span = bc.blameSpan(blame), .code = .T0044 }, "operands of '{s}' must both be float", .{op_text});
            },
            else => {},
        }
        return Type.invalid;
    }

    /// The type of binary operand `operand` (typed `t` with no expected type): a bare
    /// integer literal beside an operand of integer type `sibling` takes that width.
    fn adoptLiteralWidth(bc: *BodyChecker, operand: Ast.Index, t: Type, sibling: Type) error{OutOfMemory}!Type {
        if (bc.tree.nodes[operand.int()].tag != .literal_number or !sibling.isInteger()) return t;
        return bc.typeOfExpected(operand, sibling);
    }

    /// The type `place op= v` stores, `place op v`, given the place type `lt`. `v` is typed
    /// as a binary operand is, with no expected type, so a bare literal adopts the place's
    /// width and anything else must already match it; errors name the compound operator.
    fn compoundValue(bc: *BodyChecker, stmt_idx: Ast.Index, base: TokenTag, lt: Type) error{OutOfMemory}!Type {
        const stmt = bc.tree.nodes[stmt_idx.int()];
        var rt = try bc.typeOfExpected(stmt.rhs, null);
        if (lt.kind == .invalid or rt.kind == .invalid) return Type.invalid;
        rt = try bc.adoptLiteralWidth(stmt.rhs, rt, lt);
        const op_tok = bc.tokens[stmt.main_token];
        return bc.binaryResult(stmt_idx, .{ .token = stmt.main_token }, base, op_tok.text(bc.source), lt, rt);
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
        return Conform.structural(bc.model.structs, bc.model.enums, bc.model.conformances, .of(bc.model), t, pid, &bc.conforms_memo, bc.gpa, bc.composite, bc.bound_protocols) catch true;
    }

    pub fn typeName(bc: *const BodyChecker, ty: Type) []const u8 {
        // A bounded template's abstract check holds its own params as type vars, which
        // read as the template spells them.
        if (bc.abstract_template and ty.isTypeVar() and ty.typeVarOrd() < bc.subst.?.names.len)
            return bc.subst.?.names[ty.typeVarOrd()];
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

    pub fn slotType(bc: *const BodyChecker, slot: u32) Type {
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

    /// Whether `node_idx` names a mutable place a `mut self` receiver or a `mut` argument
    /// can write (T0019): a variable place, or a field chain rooted at a Vec element, which
    /// `lower` copies in before the call and writes back after it.
    fn isMutablePlace(bc: *const BodyChecker, node_idx: Ast.Index) bool {
        return bc.isVariablePlace(node_idx) or bc.elementRoot(node_idx) != null;
    }

    /// Whether `node_idx` names a variable: an identifier bound to a `.local` (all locals —
    /// incl. the `self` receiver — are mutable), or a field chain rooted at one. A bound
    /// `mut self` method keeps its receiver by reference past the binding, which a Vec
    /// element's copy-in and write-back cannot span, so it needs one (D52).
    fn isVariablePlace(bc: *const BodyChecker, node_idx: Ast.Index) bool {
        const n = bc.tree.nodes[(node_idx).int()];
        return switch (n.tag) {
            .identifier => bc.resolutions[(node_idx).int()] == .local,
            .field_access, .tuple_field => bc.isVariablePlace(n.lhs),
            else => false,
        };
    }

    fn elementRoot(bc: *const BodyChecker, node_idx: Ast.Index) ?Ast.Index {
        return symbols_res.elementRoot(bc.tree, bc.resolutions, node_idx);
    }

    /// Mark the element a written place is rooted at, so discovery and the fingerprint
    /// enqueue the element primitives `lower` calls: `ga_set_at` for a store, and `ga_at`
    /// as well when `reads` (the element is read, changed and written back).
    fn noteElementWrite(bc: *BodyChecker, place: Ast.Index, reads: bool) void {
        if (bc.abstract_template) return;
        if (bc.elementRoot(place)) |root| bc.setCalleeAt(root, if (reads) Callee.elem_write else Callee.elem_store);
    }

    /// Whether a `.field`/`.tuple_field` store chain bottoms out at a temporary (`p.get()`,
    /// `o?`), which has no home to write back to, so a store through it would write the
    /// copy and silently drop the write. A name is a place, and `lower` copies a Vec
    /// element in and writes it back. A type-named head (`Vec[int].x`) is not a value
    /// either, but its own error stands.
    fn rootIsTemporary(bc: *const BodyChecker, node_idx: Ast.Index) bool {
        var cur = node_idx;
        while (true) {
            const n = bc.tree.nodes[cur.int()];
            switch (n.tag) {
                .field_access, .tuple_field => cur = n.lhs,
                .identifier, .index, .type_app => return false,
                else => return true,
            }
        }
    }

    pub fn typeFromNode(bc: *BodyChecker, type_node: Ast.Index) Type {
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

    /// Emit "no field '{s}' in struct '{s}'", enriched with a "did you mean" hint when a
    /// declared field name is a close, unambiguous typo of `fname`.
    fn emitNoField(bc: *BodyChecker, at: Span, fname: []const u8, sym_name: []const u8, names: []const []const u8) error{OutOfMemory}!void {
        if (nearmiss.suggest(fname, FieldNameIter{ .names = names })) |cand|
            try bc.sink.report(.{ .span = at, .code = .T0046 }, "no field '{s}' in struct '{s}'; did you mean '{s}'?", .{ fname, sym_name, cand })
        else
            try bc.sink.report(.{ .span = at, .code = .T0046 }, "no field '{s}' in struct '{s}'", .{ fname, sym_name });
    }

    /// Emit "enum '{s}' has no variant '{s}'", enriched with a "did you mean" hint when a
    /// declared variant name is a close, unambiguous typo of `vname`. `pub` so the pattern
    /// checker reuses it for a typo'd variant in a match pattern (same message + hint).
    pub fn emitNoVariant(bc: *BodyChecker, at: Span, enum_name: []const u8, vname: []const u8, variants: []const LayoutEngine.VariantSym) error{OutOfMemory}!void {
        if (nearmiss.suggest(vname, VariantNameIter{ .variants = variants })) |cand|
            try bc.sink.report(.{ .span = at, .code = .T0061 }, "enum '{s}' has no variant '{s}'; did you mean '{s}'?", .{ enum_name, vname, cand })
        else
            try bc.sink.report(.{ .span = at, .code = .T0061 }, "enum '{s}' has no variant '{s}'", .{ enum_name, vname });
    }

    /// Emit T0018 "no method '{s}' on type '{s}'", enriched with a "did you mean" hint when
    /// a method callable on the receiver is a close, unambiguous typo of `member`. The
    /// candidate set is the concrete-receiver methods plus, for a generic-instance (`App`)
    /// receiver, its ctor's template methods (so `xs.pushh(1)` on a Vec suggests `push`).
    fn emitNoMethod(bc: *BodyChecker, at: Span, member: []const u8, recv_ty: Type) error{OutOfMemory}!void {
        if (bc.rejectedMethod(recv_ty, member)) return;
        if (bc.methodFamily(recv_ty)) |fam| if (bc.stdMethodModule(fam, member)) |module|
            return bc.sink.report(.{ .span = at, .code = .T0018 }, "no method '{s}' on type '{s}'; " ++ StdNames.add_import, .{ member, bc.typeName(recv_ty), module });
        if (!recv_ty.isApp() and recv_ty.kind != .@"struct" and recv_ty.kind != .@"enum") {
            if (Typecheck.scalarMethodModule(bc.model, bc.typeName(recv_ty), member)) |module|
                return bc.sink.report(.{ .span = at, .code = .T0018 }, "no method '{s}' on type '{s}'; its impl is in std/{s}: " ++ StdNames.add_import, .{ member, bc.typeName(recv_ty), module, module });
        }
        var iter = MethodNameIter{ .methods = bc.model.methods, .recv = recv_ty };
        if (recv_ty.isApp()) {
            const e = bc.composite.at(recv_ty.appIdx());
            iter.app = true;
            iter.templates = bc.model.templates;
            iter.app_ctor = e.ctor;
            iter.app_is_enum = e.ctor_is_enum;
        }
        if (nearmiss.suggest(member, iter)) |cand|
            try bc.sink.report(.{ .span = at, .code = .T0018 }, "no method '{s}' on type '{s}'; did you mean '{s}'?", .{ member, bc.typeName(recv_ty), cand })
        else
            try bc.sink.report(.{ .span = at, .code = .T0018 }, "no method '{s}' on type '{s}'", .{ member, bc.typeName(recv_ty) });
    }

    /// Whether `member` on `recv_ty` is a method of an impl T0013 rejected. That rejection
    /// is the error, so a use of the method reports nothing more.
    fn rejectedMethod(bc: *const BodyChecker, recv_ty: Type, member: []const u8) bool {
        if (Typecheck.rejectedProtocolMethod(bc.model.rejected_impls, bc.model.protocols, bc.composite, recv_ty, member)) return true;
        if (!recv_ty.isApp()) return false;
        const e = bc.composite.at(recv_ty.appIdx());
        return bc.rejectedOnType(e.ctor, e.ctor_is_enum, member);
    }

    /// `rejectedMethod` for a call through the generic type itself (`Box.make()`).
    fn rejectedOnType(bc: *const BodyChecker, ctor: u32, is_enum: bool, member: []const u8) bool {
        for (bc.model.rejected_methods) |r| {
            if (r.recv_ctor == ctor and r.recv_is_enum == is_enum and std.mem.eql(u8, r.name, member)) return true;
        }
        return false;
    }

    /// Where this body's method calls are written (see `Typecheck.Site`).
    pub fn methodSite(bc: *const BodyChecker) Typecheck.Site {
        return Typecheck.Site.of(bc.model.graph.mods[bc.graph_mod].bundled);
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

// The flow-lint decision logic lives in the `Lint` namespace as pure functions over a
// read-only `ControlFlow.Ctx`, so it is unit-tested here on hand-built node graphs (the same
// idiom `ControlFlow` uses) without standing up a whole `BodyChecker` + `DiagnosticSink`.
// The `Builder` is reused from `ControlFlow` — the module whose walks `Lint` consumes.
const testing = std.testing;
const Builder = @import("ControlFlow.zig").Builder;

test "Lint.divergesForReach unions control-flow divergence and never-typed statements" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    // loop { return } — diverges via ControlFlow.stmtDiverges.
    const ret_loop = try b.loopOf(try b.block(&.{try b.ret()}));
    // A `var_decl` bound to a diverging value is `never`-typed (`x := loop {}`).
    const never_stmt = try b.add(.{ .tag = .var_decl, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    b.node_types.items[never_stmt.int()] = Type.never;
    // `panic(..)` as a non-final statement: only its call is typed, the expr_stmt is not.
    const callee = try b.add(.{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const call = try b.add(.{ .tag = .call, .main_token = 0, .lhs = callee, .rhs = Ast.none });
    b.node_types.items[call.int()] = Type.never;
    const panic_stmt = try b.add(.{ .tag = .expr_stmt, .main_token = 0, .lhs = call, .rhs = Ast.none });
    // A plain declaration and a unit call statement fall through.
    const plain = try b.add(.{ .tag = .var_decl, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const unit_call = try b.add(.{ .tag = .call, .main_token = 0, .lhs = callee, .rhs = Ast.none });
    b.node_types.items[unit_call.int()] = Type.unit;
    const unit_stmt = try b.add(.{ .tag = .expr_stmt, .main_token = 0, .lhs = unit_call, .rhs = Ast.none });

    const c = b.ctx();
    try testing.expect(Lint.divergesForReach(c, ret_loop));
    try testing.expect(Lint.divergesForReach(c, never_stmt));
    try testing.expect(Lint.divergesForReach(c, panic_stmt));
    try testing.expect(!Lint.divergesForReach(c, plain));
    try testing.expect(!Lint.divergesForReach(c, unit_stmt));
}

test "Lint.firstUnreachable returns the first dead index, or null when nothing follows a divergence" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    const ret_loop = try b.loopOf(try b.block(&.{try b.ret()}));
    const p1 = try b.add(.{ .tag = .var_decl, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const p2 = try b.add(.{ .tag = .var_decl, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });

    const c = b.ctx();
    // [diverge, dead, dead] → first unreachable at index 1 (reported once).
    try testing.expectEqual(@as(?usize, 1), Lint.firstUnreachable(c, &.{ ret_loop, p1, p2 }));
    // Nothing follows the divergence, or there is no divergence → null.
    try testing.expectEqual(@as(?usize, null), Lint.firstUnreachable(c, &.{ p1, ret_loop }));
    try testing.expectEqual(@as(?usize, null), Lint.firstUnreachable(c, &.{p1}));
    try testing.expectEqual(@as(?usize, null), Lint.firstUnreachable(c, &.{}));
}

test "Lint.constIfCond returns the token only for a bare literal-bool condition" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    const lit = try b.add(.{ .tag = .literal_bool, .main_token = 7, .lhs = Ast.none, .rhs = Ast.none });
    const id = try b.add(.{ .tag = .identifier, .main_token = 3, .lhs = Ast.none, .rhs = Ast.none });
    const tree: Ast.Tree = .{ .nodes = b.nodes.items, .extra = b.extra.items };

    try testing.expectEqual(@as(?u32, 7), Lint.constIfCond(tree, lit));
    try testing.expectEqual(@as(?u32, null), Lint.constIfCond(tree, id));
}

test "Lint.droppedMustUseName suppresses poison and non-must-use, else names the family" {
    // A poison return type suppresses the drop warning (avoids stacking on a real error).
    try testing.expect(Lint.droppedMustUseName(.invalid, .option) == null);
    // A non-Option/Result value is never a must-use.
    try testing.expect(Lint.droppedMustUseName(.unit, .none) == null);
    try testing.expectEqualStrings("Option", Lint.droppedMustUseName(.unit, .option).?);
    try testing.expectEqualStrings("Result", Lint.droppedMustUseName(.unit, .result).?);
}

test "Lint.uncalledFn flags effect-free fn-typed forms and captures, never a call or poison" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    // token 0 is `fn`, token 1 a `_` identifier: a capture lambda sits on the `_`.
    const toks = [_]Token{ .{ .tag = .kw_fn, .start = 0, .end = 2 }, .{ .tag = .identifier, .start = 3, .end = 4 } };
    const id = try b.add(.{ .tag = .identifier, .main_token = 1, .lhs = Ast.none, .rhs = Ast.none });
    const field = try b.add(.{ .tag = .field_access, .main_token = 1, .lhs = id, .rhs = Ast.none });
    const index = try b.add(.{ .tag = .index, .main_token = 1, .lhs = id, .rhs = id });
    const call = try b.add(.{ .tag = .call, .main_token = 1, .lhs = id, .rhs = Ast.none });
    const lam = try b.add(.{ .tag = .lambda, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const cap = try b.add(.{ .tag = .lambda, .main_token = 1, .lhs = Ast.none, .rhs = Ast.none });
    const tree: Ast.Tree = .{ .nodes = b.nodes.items, .extra = b.extra.items };
    const f = Type.func(0);

    try testing.expectEqual(@as(?Lint.Uncalled, .value), Lint.uncalledFn(tree, &toks, .unit, id, f));
    try testing.expectEqual(@as(?Lint.Uncalled, .value), Lint.uncalledFn(tree, &toks, .unit, field, f));
    try testing.expectEqual(@as(?Lint.Uncalled, .value), Lint.uncalledFn(tree, &toks, .unit, index, f));
    try testing.expectEqual(@as(?Lint.Uncalled, .lambda), Lint.uncalledFn(tree, &toks, .unit, lam, f));
    try testing.expectEqual(@as(?Lint.Uncalled, .capture), Lint.uncalledFn(tree, &toks, .unit, cap, f));
    // A call ran its effects; a non-fn value is not a function; poison stays quiet.
    try testing.expectEqual(@as(?Lint.Uncalled, null), Lint.uncalledFn(tree, &toks, .unit, call, f));
    try testing.expectEqual(@as(?Lint.Uncalled, null), Lint.uncalledFn(tree, &toks, .unit, id, .int));
    try testing.expectEqual(@as(?Lint.Uncalled, null), Lint.uncalledFn(tree, &toks, .invalid, id, f));
    try testing.expectEqual(@as(?Lint.Uncalled, null), Lint.uncalledFn(tree, &toks, .unit, cap, .invalid));
}

test "a checker diagnostic spans the node it is about, not just its first token" {
    try Typecheck.expectSpans(&.{
        .{ .src = "fn f() {\n x: int = 1 == 2\n return\n}\n", .key = "", .want = "1 == 2" },
        .{ .src = "fn g(a: int) -> int { return a }\nfn f() -> int { return g(1, 2) }\n", .key = "", .want = "(1, 2)" },
        .{ .src = "fn g(a: int) -> int { return a }\nfn f() -> int { return g(1 == 1) }\n", .key = "", .want = "1 == 1" },
        .{ .src = "fn f() -> int { return 1 + true }\n", .key = "", .want = "1 + true" },
        .{ .src = "fn f() -> int { return (1 == 1) }\n", .key = "", .want = "1 == 1" },
        .{ .src = "fn f() -> int { return (1 + 2) * true }\n", .key = "", .want = "(1 + 2) * true" },
        .{ .src = "fn f() -> int {\n return 1\n 2 + 3\n}\n", .key = "unreachable code", .want = "2 + 3" },
        .{ .src = "fn f() {\n for i in 0..(1 == 1) { }\n}\n", .key = "for range bounds", .want = "1 == 1" },
        .{ .src = "fn f(a: int) -> int { return a[0] }\n", .key = "cannot index", .want = "a[0]" },
        .{ .src = "struct P { x: int }\nfn f(p: P) -> P { return p + p }\n", .key = "'+' requires", .want = "p + p" },
        .{ .src = "struct P { x: int }\nfn f() -> P { return P { x: 1 == 1 } }\n", .key = "field 'x'", .want = "1 == 1" },
        .{ .src = "struct T(int, int)\nfn f() -> T { return T(1) }\n", .key = "tuple struct 'T' expects", .want = "(1)" },
        .{ .src = "enum E { a(int), b }\nfn f() -> E { return E.a(1, 2) }\n", .key = "variant 'E.a' expects", .want = "E.a(1, 2)" },
        .{ .src = "fn id[U](a: U, b: U) -> U { return a }\nfn f() -> int { return id(1, 1 == 1) }\n", .key = "conflicting types", .want = "1 == 1" },
        .{ .src = "fn g(a: int) -> int { return a }\nfn f() -> int { return g[int](1) }\n", .key = "'g' is not generic", .want = "[int]" },
        .{ .src = "struct P { x: int }\nfn f() -> P { return P(1) }\n", .key = "use named construction", .want = "P(1)" },
        .{ .src = "fn f(a: int) -> int { return a(1) }\n", .key = "called value is not a function", .want = "a" },
        .{ .src = "fn g() -> int { return 1 }\nfn f() -> int { return g()(1) }\n", .key = "called value is not a function", .want = "g()" },
        .{ .src = "fn f() {\n x := panic\n}\n", .key = "cannot be used as a value", .want = "panic" },
        .{ .src = "fn id[T](x: T) -> T { return x }\nfn f() {\n x := id\n}\n", .key = "cannot be used as a value", .want = "id" },
        .{ .src = "struct P { x: int }\nimpl P { fn bump(mut self, d: int) { self.x = self.x + d } }\nfn mk() -> P { return P { x: 0 } }\nfn f() { mk().bump(1) }\n", .key = "cannot call mutating method", .want = "mk()" },
        .{ .src = "fn f() {\n break\n}\n", .key = "break outside", .want = "break" },
    });
}

test "every arm of a value if, match, loop or labeled block takes the one expected type" {
    const shape = "enum Shape { circle(int), square(int) }\n";
    const cases = [_][]const u8{
        "fn f(v: int) -> Option[str] { if v > 3 { Option.some(\"big\") } else { Option.none } }\n",
        "fn f(v: int) -> Option[str] { if v > 3 { Option.none } else { Option.some(\"big\") } }\n",
        "fn f(v: int) -> int8 { if v > 3 { 1 } else { 2 } }\n",
        "fn f(v: int, w: int8) -> int8 { if v > 3 { 1 } else { w } }\n",
        "fn f(v: int, w: int8) -> int8 { if v > 3 { w } else { 1 } }\n",
        shape ++ "fn f(v: int) -> Shape { if v > 3 { .circle(1) } else { .square(2) } }\n",
        shape ++ "fn f(v: int) -> Shape { if v > 3 { Shape.circle(1) } else { .square(2) } }\n",
        shape ++ "fn f(v: int) -> Shape { if v > 3 { .circle(1) } else { Shape.square(2) } }\n",
        "fn f(v: int) -> Option[str] {\n x: Option[str] = if v > 3 { Option.some(\"big\") } else { Option.none }\n x\n}\n",
        "fn f(v: int) -> Option[str] {\n x: Option[str] = if v > 3 { Option.none } else { Option.some(\"big\") }\n x\n}\n",
        "fn f(v: int) -> Option[str] { return if v > 3 { Option.some(\"big\") } else { Option.none } }\n",
        "fn f(v: int) -> Option[str] { return if v > 3 { Option.none } else { Option.some(\"big\") } }\n",
        "fn f(v: int) -> Option[str] { if v > 3 { Option.none } else if v > 1 { Option.some(\"mid\") } else { Option.none } }\n",
        "fn f(v: int) -> Option[str] { if v > 3 { Option.some(\"big\") } else if v > 1 { Option.none } else { Option.none } }\n",
        "fn f(v: int) -> Option[str] { if v > 3 { Option.none } else if v > 1 { Option.none } else { Option.some(\"x\") } }\n",
        "fn f(v: int) -> int8 { if v > 3 { 1 } else if v > 1 { 2 } else { 3 } }\n",
        "fn f(v: int) -> Option[str] { match v { 1 -> Option.some(\"y\"), _ -> if v > 3 { Option.some(\"z\") } else { Option.none } } }\n",
        "fn f(v: int) -> Option[str] { if (if v > 3 { true } else { false }) { Option.none } else { Option.some(\"x\") } }\n",
        "fn f(v: int) -> Option[str] { match (if v > 1 { 1 } else { 0 }) { 1 -> Option.none, _ -> Option.some(\"x\") } }\n",
        "fn f(v: int) -> Option[str] { match v { n if (if n > 3 { true } else { false }) -> Option.some(\"y\"), _ -> Option.none } }\n",
        "fn f(v: int) -> Option[str] {\n loop {\n  if v > 3 { break Option.none }\n  break Option.some(\"x\")\n }\n}\n",
        "fn f(v: int) -> Option[str] {\n @l loop {\n  if v > 3 { break @l Option.none }\n  break @l Option.some(\"x\")\n }\n}\n",
        "fn f(v: int) -> Option[str] {\n @o loop {\n  loop { break @o Option.none }\n }\n}\n",
        "fn f(v: int) -> Option[str] {\n @b {\n  if v > 3 { break @b Option.none }\n  Option.some(\"x\")\n }\n}\n",
        "fn f(v: int) -> int8 { loop { if v > 3 { break 1 } else { break 2 } } }\n",
        "fn f(v: int) -> int8 {\n x: int8 = @b {\n  if v > 3 { break @b 1 }\n  2\n }\n x\n}\n",
    };
    for (cases) |src| try Typecheck.expectDiagCodes(src, &.{});
}

test "an arm checked against the expected type still reports its own mismatch" {
    try Typecheck.expectDiagCodes("fn f(v: int) -> int8 { if v > 3 { 1 } else { 300 } }\n", &.{.T0034});
    try Typecheck.expectDiagCodes("fn f(v: int) -> int8 { if v > 3 { 300 } else { 1 } }\n", &.{.T0034});
    try Typecheck.expectDiagCodes("fn f(v: int) -> int8 { if v > 3 { 1 } else if v > 1 { 2 } else { 300 } }\n", &.{.T0034});
    try Typecheck.expectDiagCodes("fn f(v: int) -> Option[str] { if v > 3 { Option.some(\"a\") } else { Option.some(5) } }\n", &.{.T0015});
    try Typecheck.expectDiagCodes("fn f(v: int) -> Option[str] { if v > 3 { Option.none } else if v > 1 { Option.none } else { Option.some(5) } }\n", &.{.T0015});
    try Typecheck.expectDiagCodes("fn f(v: int) -> int8 { if v > 3 { 1 } else { \"s\" } }\n", &.{.T0058});
    try Typecheck.expectDiagCodes("fn f(v: int) -> int8 { loop { if v > 3 { break 1 } else { break 300 } } }\n", &.{.T0034});
}

test "a value if's condition and a match guard take no expected type from the value" {
    try Typecheck.expectDiagCodes("fn f(v: int) -> int8 { if 300 { 1 } else { 2 } }\n", &.{.T0045});
    try Typecheck.expectDiagCodes("fn f(v: int) -> int8 { if v > 1 { 1 } else if 300 { 2 } else { 3 } }\n", &.{.T0045});
    try Typecheck.expectDiagCodes("fn f(v: int) -> int8 { match v { _ if 300 -> 1, _ -> 2 } }\n", &.{.T0045});
}

test "a receiver's expected type stays a hint through every arm, break and scrutinee" {
    const small = "impl int { fn small(self) -> int8 { if self > 100 { return 1 }\n return 0 } }\n";
    const cases = [_][]const u8{
        small ++ "fn f() -> int8 { 300.small() }\n",
        small ++ "fn f() -> int8 { (loop { break 300 }).small() }\n",
        small ++ "fn f() -> int8 { (@b { break @b 300 }).small() }\n",
        "fn f() -> int8 { match 300 { 300 -> 1, _ -> 2 } }\n",
    };
    for (cases) |src| try Typecheck.expectDiagCodes(src, &.{});
    // A hard expected type still rejects what cannot produce it.
    try Typecheck.expectDiagCodes("fn f() -> int8 { loop { break 300 } }\n", &.{.T0034});
    try Typecheck.expectDiagCodes("fn f() -> int8 { @b { break @b 300 } }\n", &.{.T0034});
}
