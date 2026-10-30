const std = @import("std");
const Ast = @import("ast/Ast.zig");
const Closure = @import("symbols/Closure.zig");
const Modes = @import("symbols/Modes.zig");
const symbols_res = @import("symbols/Resolution.zig");
const Composite = @import("symbols/Composite.zig");
const Type = @import("layout/Type.zig").Type;

// The lambda and local fn subsystem, carved out of `BodyChecker` as free functions over a
// `*BodyChecker` (the `PatternChecker` pattern): typing a lambda or local fn body, the
// per-body state it swaps in, the jumps that may not leave it, and how each captured
// local is held.
const BC = @import("BodyChecker.zig");
const Typecheck = @import("types.zig");
const BodyChecker = BC.BodyChecker;
const refs = Typecheck.refs;

/// `reportTailMismatch` for a lambda whose return type came from the type expected for
/// it (`exp`), not from a `-> R`: the message spells the type the lambda really has,
/// since nothing it wrote declared the one it is held to.
fn reportExpectedTailMismatch(bc: *BodyChecker, block_idx: Ast.Index, params: []const Type, body_ty: Type, exp: Type) error{OutOfMemory}!bool {
    if (body_ty.kind == .unit or body_ty.kind == .never or body_ty.kind == .invalid) return false;
    const tail = bc.blockTail(block_idx) orelse return false;
    const parts = Composite.fnParts(bc.composite.at(exp.funcIdx()));
    const real: LambdaSig = .{ .bc = bc, .params = params, .expected = parts.params, .ret = body_ty };
    try bc.report(.{ .span = bc.spanOf(tail), .code = .T0042 }, "lambda has type {f}, but {f} is expected: its value is {f}, not {f}", .{ real, bc.fmtType(exp), bc.fmtType(body_ty), bc.fmtType(parts.ret) });
    return true;
}

/// A lambda's signature as a `{f}` argument, so a report written as its template spells
/// it spells these types that way too.
const LambdaSig = struct {
    bc: *const BodyChecker,
    params: []const Type,
    expected: []const Type,
    ret: Type,

    pub fn format(s: LambdaSig, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.writeAll("fn(");
        for (s.params, 0..) |p, i| {
            if (i != 0) try w.writeAll(", ");
            // An open `_` param has no type of its own yet; the expected one stands in.
            const shown = if (p.kind == .invalid and i < s.expected.len) s.expected[i] else p;
            try w.print("{f}", .{s.bc.fmtType(shown).nested()});
        }
        try w.print(") -> {f}", .{s.bc.fmtType(s.ret)});
    }
};

/// Decide how each captured local is held. A closure copies a capture when it is made,
/// which is sound only for a local that nothing assigns; a captured local that is
/// assigned (`x = ..`, a place store rooted at it, a `mut self` call on it, a `mut`
/// argument rooted at it) is boxed instead, into `bc.boxed`. A bound method value
/// captures its local receiver the same way. A `mut` param's slot (`mut self`
/// included) holds the caller's address, not a value a cell could hold, so it is never
/// boxed: the closure holds the address, which is legal only if the closure never
/// outlives the call. That depends on other fns' bodies, so the serial escape pass
/// decides it (T0052), not this per-body check. A bounded template's abstract check
/// skips this: each instance re-check decides for itself.
pub fn classifyCaptures(bc: *BodyChecker, place_params: u64) error{OutOfMemory}!void {
    if (bc.abstract_template) return;
    if (bc.closures == null and bc.bound_roots.items.len == 0) return;
    // Params are resolver slots 0..n-1, so a param's slot is its index, and a mask bit
    // is set only below the param count.
    std.mem.sort(u32, bc.mutated_roots.items, {}, std.sort.asc(u32));
    const assigned = sortedSet(bc.assigned_roots.items);
    var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer seen.deinit(bc.gpa);
    if (bc.closures) |fc| {
        // Boundaries are in node order, where a nested lambda precedes its parent; the
        // rank is source (pre-) order, so the first boundary by rank holds a slot's first use.
        const by_rank = try bc.gpa.alloc(u32, fc.boundaries.len);
        defer bc.gpa.free(by_rank);
        for (fc.boundaries, 0..) |b, i| by_rank[b.rank] = @intCast(i);
        for (by_rank) |bi| for (fc.boundaries[bi].captures) |c| {
            if (Modes.isMut(place_params, c.slot)) continue;
            if ((try seen.getOrPut(bc.gpa, c.slot)).found_existing) continue;
            if (Closure.contains(assigned, c.slot) or Closure.contains(bc.mutated_roots.items, c.slot)) {
                try bc.boxed.append(bc.gpa, c.slot);
            }
        };
    }
    for (bc.bound_roots.items) |slot| {
        if (Modes.isMut(place_params, slot)) continue;
        if ((try seen.getOrPut(bc.gpa, slot)).found_existing) continue;
        if (Closure.contains(assigned, slot) or Closure.contains(bc.mutated_roots.items, slot)) {
            try bc.boxed.append(bc.gpa, slot);
        }
    }
    std.mem.sort(u32, bc.boxed.items, {}, std.sort.asc(u32));
}

/// Sort `items` in place and drop repeats, returning the deduped prefix.
fn sortedSet(items: []u32) []const u32 {
    std.mem.sort(u32, items, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (items) |x| {
        if (n > 0 and items[n - 1] == x) continue;
        items[n] = x;
        n += 1;
    }
    return items[0..n];
}

/// Record the local an assignment to `place` writes.
pub fn noteAssignedRoot(bc: *BodyChecker, place: Ast.Index) error{OutOfMemory}!void {
    if (symbols_res.placeRoot(bc.tree, bc.resolutions, place)) |slot| try bc.assigned_roots.append(bc.gpa, slot);
}

/// Record the local a call writes through: the root of a `mut self` receiver place or
/// of a `mut` argument.
pub fn noteMutatedRoot(bc: *BodyChecker, place: Ast.Index) error{OutOfMemory}!void {
    if (symbols_res.placeRoot(bc.tree, bc.resolutions, place)) |slot| try bc.mutated_roots.append(bc.gpa, slot);
}

/// `fn(params) -> R { body }`: a value of type `fn(P..) -> R`. A param's type is its
/// annotation, else the expected fn type's; the return type is the annotation, else the
/// expected fn type's, else the join of the body's value and its `return`s. `return`
/// and `?` exit the lambda, and no jump leaves it. A part of the expected type that
/// still holds a generic callee's open type variable provides nothing, except a return
/// type such as `Option[U]`, which each returned value is checked against for the args
/// it knows; a bounded template's own type vars provide themselves (`givesType`). A `_`
/// capture's param with no expected type stays open until its hole's call position
/// types it (`adoptHole`), and is T0050 only when the body reported no error.
pub fn typeOfLambda(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    bc.lambda_depth += 1;
    defer {
        bc.lambda_depth -= 1;
        if (bc.lambda_depth == 0) bc.lambda_choices.clearRetainingCapacity();
    }
    const proto = Ast.protoAt(bc.tree, n.lhs.int());
    const capture = Ast.isCaptureLambda(bc.tree, bc.tokens, node_idx);
    var open = false;
    var exp_fn: ?Type = null;
    var exp_params: []const Type = &.{};
    var exp_ret: ?Type = null;
    var partial_ret: ?Type = null;
    var arity_err = false;
    if (bc.expectedTy()) |e| if (e.isFunc()) {
        const parts = Composite.fnParts(bc.composite.at(e.funcIdx()));
        if (parts.params.len == proto.params.len) {
            exp_fn = e;
            exp_params = parts.params;
            if (bc.givesType(parts.ret))
                exp_ret = parts.ret
            else if (!parts.ret.isTypeVar())
                partial_ret = parts.ret;
        } else {
            const shown = bc.fmtTypeNamed(e, bc.body.open_names, bc.body.open_vars, bc.openOrdBase());
            if (capture)
                try bc.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0039 }, "a '_' capture takes 1 parameter, but the expected type {f} takes {d}", .{ shown, parts.params.len })
            else
                try bc.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0039 }, "lambda takes {d} parameter(s), but the expected type {f} takes {d}", .{ proto.params.len, shown, parts.params.len });
            arity_err = true;
        }
    };
    const params = try bc.gpa.alloc(Type, proto.params.len);
    defer bc.gpa.free(params);
    var poisoned = arity_err;
    for (proto.params, 0..) |p, i| {
        const pn = bc.tree.nodes[p.int()];
        const ty: Type = if (pn.lhs != Ast.none)
            bc.typeFromNode(pn.lhs)
        else if (i < exp_params.len and bc.givesType(exp_params[i]))
            exp_params[i]
        else if (capture and !arity_err) blk: {
            open = true;
            break :blk Type.invalid;
        } else blk: {
            // After an arity error no expected type lines up with the params.
            if (!arity_err and node_idx != bc.quiet_arg) {
                const hint = if (i < exp_params.len or node_idx == bc.generic_arg) " (no other argument of the generic call fixes its type)" else "";
                try bc.sink.report(.{ .span = bc.tokSpan(pn.main_token), .code = .T0050 }, "cannot infer the type of lambda parameter '{s}'; annotate it{s}", .{ bc.nameText(pn.main_token), hint });
            }
            // A quiet arg's call reports its own error, possibly only once its args
            // are typed, so this sink may not hold it yet.
            break :blk if (node_idx == bc.quiet_arg and !arity_err) Type.invalid else bc.poison();
        };
        if (ty.kind == .invalid and !open) poisoned = true;
        params[i] = ty;
        bc.node_types[p.int()] = ty;
        if (bc.resolutions[p.int()] == .local) try bc.setSlot(bc.resolutions[p.int()].local, ty);
        // An annotated param the expected type does not have means the lambda does not
        // fit that type at all, so its return type comes from its own body: the
        // mismatch then names the type the lambda really has.
        if (pn.lhs != Ast.none and i < exp_params.len and bc.givesType(exp_params[i]) and
            ty.kind != .invalid and !Type.eql(ty, exp_params[i]))
        {
            exp_ret = null;
            partial_ret = null;
        }
    }

    const frame = enterBody(bc, false);
    defer leaveBody(bc, frame);
    if (capture) bc.body.hole = Ast.captureHole(bc.tree, node_idx);
    const fresh = bc.body;
    const mark = bc.sink.count();

    const annotated = proto.ret_type != Ast.none;
    const declared: ?Type = if (annotated) refs.retTypeFromNode(bc, proto.ret_type) else exp_ret;
    var ret: Type = undefined;
    if (declared) |r| {
        ret = try checkDeclaredRet(bc, n, params, r, annotated, exp_fn);
    } else if (recallFix(bc, node_idx, exp_fn)) |chosen| {
        ret = if (chosen) |fix|
            try checkDeclaredRet(bc, n, params, fix, false, refs.internFn(bc, exp_params, fix))
        else
            try inferRet(bc, n, partial_ret);
    } else {
        ret = try inferRet(bc, n, partial_ret);
        if (bc.body.ret_fix) |fix| if (bc.errorsSince(mark)) {
            // A value checked before the one that fixed the open vars could not use them
            // (`Option.none` in `if c { return Option.none }`), so the body is checked
            // again against what they fixed, as `inferArgs` types the args that wait after
            // the rest. The second check stands only when it errs less.
            const errs = bc.errorCountSince(mark);
            const fixed_fn = refs.internFn(bc, exp_params, fix);
            bc.sink.truncate(mark);
            bc.body = fresh;
            const fixed = try checkDeclaredRet(bc, n, params, fix, false, fixed_fn);
            const keep = bc.errorCountSince(mark) < errs;
            try bc.lambda_choices.put(bc.gpa, node_idx, .{ .exp_fn = exp_fn, .fix = if (keep) fix else null });
            if (keep) {
                ret = fixed;
            } else {
                bc.sink.truncate(mark);
                bc.body = fresh;
                ret = try inferRet(bc, n, partial_ret);
            }
        };
    }
    if (open) {
        const p = proto.params[0];
        const t: Type = if (bc.resolutions[p.int()] == .local) bc.slotType(bc.resolutions[p.int()].local) else .invalid;
        if (t.kind == .invalid) {
            poisoned = true;
            if (!bc.errorsSince(mark) and node_idx != bc.quiet_arg)
                try bc.sink.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0050 }, "cannot infer the type of '_'; nothing in the call gives it one; write the type arguments, as in f[int](a, _), or a typed lambda, as in fn(v: int) {{ .. }}", .{});
        } else {
            params[0] = t;
            bc.node_types[p.int()] = t;
        }
    }
    if (ret.kind == .invalid) poisoned = true;
    const ty: Type = if (poisoned) .invalid else refs.internFn(bc, params, ret);
    bc.node_types[node_idx.int()] = ty;
    return ty;
}

/// Check a lambda body against its return type `r`, written (`annotated`) or taken from
/// `exp_fn`, the fn type expected for the lambda.
fn checkDeclaredRet(bc: *BodyChecker, n: Ast.Node, params: []const Type, r: Type, annotated: bool, exp_fn: ?Type) error{OutOfMemory}!Type {
    bc.body.cur_ret = r;
    bc.body.ret_infer = false;
    if (!annotated) bc.body.ret_expected = exp_fn;
    const want_value = r.kind != .unit and r.kind != .invalid;
    const body_ty = try bc.typeOfBlockExpected(n.rhs, want_value, if (want_value) r else null);
    if (want_value and !bc.blockReturns(n.rhs) and !Type.assignable(r, body_ty)) {
        if (annotated) {
            if (!try bc.reportTailMismatch(n.rhs, body_ty, r))
                try bc.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0043 }, "lambda must return {f} but may fall off the end", .{bc.fmtType(r)});
        } else if (try reportExpectedTailMismatch(bc, n.rhs, params, body_ty, exp_fn.?)) {
            // Poisoned so the call or binding that expected the type does not report
            // the same mismatch again.
            return .invalid;
        } else {
            try bc.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0043 }, "lambda must return {f} but may fall off the end", .{bc.fmtType(r)});
        }
    }
    return r;
}

/// Infer a lambda's return type from its body: the join of its value and its `return`s,
/// each checked against `partial` when the expected fn type gives one.
fn inferRet(bc: *BodyChecker, n: Ast.Node, partial: ?Type) error{OutOfMemory}!Type {
    bc.body.cur_ret = Type.unit;
    bc.body.ret_infer = true;
    bc.body.ret_join = .never;
    bc.body.ret_partial = partial;
    const body_ty = try bc.typeOfBlockExpected(n.rhs, true, partial);
    const tail: Type = if (bc.blockDiverges(n.rhs)) .never else body_ty;
    return joinLambdaTail(bc, n, tail);
}

/// The return type a re-check chose for the lambda at `node_idx` when it was last checked
/// against the same expected fn type: the type its returns fixed, or null for the type
/// they join to. Unset when it was never re-checked there.
fn recallFix(bc: *BodyChecker, node_idx: Ast.Index, exp_fn: ?Type) ??Type {
    const c = bc.lambda_choices.get(node_idx) orelse return null;
    const same = if (c.exp_fn) |a| (if (exp_fn) |b| a.eql(b) else false) else exp_fn == null;
    return if (same) c.fix else null;
}

/// `fn name(params) -> R { body }` in a block: a value of type `fn(P..) -> R` (a missing
/// `-> R` is unit), bound to `name` before the body is checked so the body can call
/// itself. The body checks like an annotated lambda's: `return` exits the local fn and
/// no jump leaves it.
pub fn checkLocalFn(bc: *BodyChecker, idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!void {
    const proto = Ast.protoAt(bc.tree, n.lhs.int());
    const params = try bc.gpa.alloc(Type, proto.params.len);
    defer bc.gpa.free(params);
    var poisoned = false;
    for (proto.params, params) |p, *pty| {
        const ty = bc.typeFromNode(bc.tree.nodes[p.int()].lhs);
        if (ty.kind == .invalid) poisoned = true;
        pty.* = ty;
        bc.node_types[p.int()] = ty;
        if (bc.resolutions[p.int()] == .local) try bc.setSlot(bc.resolutions[p.int()].local, ty);
    }
    const ret = refs.retTypeFromNode(bc, proto.ret_type);
    if (ret.kind == .invalid) poisoned = true;
    const ty: Type = if (poisoned) .invalid else refs.internFn(bc, params, ret);
    bc.node_types[idx.int()] = ty;
    if (bc.resolutions[idx.int()] == .local) try bc.setSlot(bc.resolutions[idx.int()].local, ty);

    const frame = enterBody(bc, true);
    defer leaveBody(bc, frame);
    bc.body.cur_ret = ret;
    bc.body.ret_infer = false;
    const want_value = ret.kind != .unit and ret.kind != .invalid;
    const body_ty = try bc.typeOfBlockExpected(n.rhs, want_value, if (want_value) ret else null);
    if (want_value and !bc.blockReturns(n.rhs) and !Type.assignable(ret, body_ty)) {
        if (!try bc.reportTailMismatch(n.rhs, body_ty, ret))
            try bc.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0043 }, "function '{s}' must return {f} but may fall off the end", .{ bc.nameText(n.main_token), bc.fmtType(ret) });
    }
}

/// Enter a lambda or local fn body: no expected type flows in and no jump leaves it.
/// The caller sets the return state; `leaveBody` restores the enclosing body's.
fn enterBody(bc: *BodyChecker, is_local_fn: bool) BC.BodyCtx {
    const saved = bc.body;
    bc.body = .{ .loop_floor = bc.loop_stack.items.len, .floor_is_local_fn = is_local_fn };
    return saved;
}

fn leaveBody(bc: *BodyChecker, saved: BC.BodyCtx) void {
    bc.body = saved;
}

/// Give the open capture param the type an argument loop wants at its hole. Only
/// argument loops pass the hole a want, so `want` is the callee's param type; never
/// adopt from the expected type, which inside the capture body is unrelated to the hole.
pub fn adoptHole(bc: *BodyChecker, want: Type) error{OutOfMemory}!void {
    const r = bc.resolutions[bc.body.hole.int()];
    if (r != .local) return;
    if (bc.slotType(r.local).kind != .invalid) return;
    if (want.kind == .invalid or !bc.givesType(want)) return;
    try bc.setSlot(r.local, want);
}

/// Join a `return` operand into the inferred lambda return type.
pub fn joinLambdaReturn(bc: *BodyChecker, at: Ast.Index, ty: Type) error{OutOfMemory}!void {
    const j = bc.body.ret_join;
    if (j.kind == .invalid or ty.kind == .invalid) {
        bc.body.ret_join = .invalid;
    } else if (j.kind == .never) {
        bc.body.ret_join = ty;
    } else if (ty.kind != .never and !Type.eql(j, ty)) {
        try bc.report(.{ .span = bc.spanOf(at), .code = .T0042 }, "return type {f} does not match the lambda's earlier return type {f}", .{ bc.fmtType(ty), bc.fmtType(j) });
        bc.body.ret_join = .invalid;
    }
}

/// The inferred return type: the `return`s' join with the body's trailing value (`never`
/// when the body diverges). A lambda that never yields a value returns `()`.
fn joinLambdaTail(bc: *BodyChecker, n: Ast.Node, tail: Type) error{OutOfMemory}!Type {
    const j = bc.body.ret_join;
    if (j.kind == .invalid or tail.kind == .invalid) return .invalid;
    if (j.kind == .never) return if (tail.kind == .never) Type.unit else tail;
    if (tail.kind == .never or Type.eql(j, tail)) return j;
    if (tail.kind == .unit) {
        try bc.report(.{ .span = bc.tokSpan(n.main_token), .code = .T0043 }, "lambda must return {f} but may fall off the end", .{bc.fmtType(j)});
    } else {
        const at = if (bc.blockTail(n.rhs)) |t| bc.spanOf(t) else bc.tokSpan(n.main_token);
        try bc.report(.{ .span = at, .code = .T0042 }, "lambda's value {f} does not match its return type {f}", .{ bc.fmtType(tail), bc.fmtType(j) });
    }
    return .invalid;
}

pub fn jumpFloorName(bc: *const BodyChecker) []const u8 {
    if (bc.body.floor_is_local_fn) return "local function";
    return if (bc.body.hole != Ast.none) "'_' capture" else "lambda";
}

/// A `break`/`continue` at `stmt_idx` whose target lies outside the innermost lambda:
/// its label names a construct below `loop_floor`, or (bare) a loop is open below it.
pub fn jumpLeavesLambda(bc: *const BodyChecker, stmt_idx: Ast.Index) bool {
    if (bc.body.loop_floor == 0) return false;
    const outer = bc.loop_stack.items[0..bc.body.loop_floor];
    if (bc.resolutions[(stmt_idx).int()] == .label) {
        const target = bc.resolutions[(stmt_idx).int()].label;
        for (outer) |c| if (c.construct_node == target) return true;
        return false;
    }
    for (outer) |c| if (c.kind != .labeled_block) return true;
    return false;
}

test "in a bounded template a '_' or an untyped lambda param takes the template's own type var" {
    const add = "protocol Add {\n fn add(self, o: Self) -> Self\n}\nfn twice[T](f: fn(T) -> T, x: T) -> T { f(f(x)) }\n";
    const clean = [_][]const u8{
        add ++ "fn go[T has Add](t: T) -> T { twice(t.add(_), t) }\n",
        add ++ "fn go[T has Add](t: T) -> T { twice(fn(o) { t.add(o) }, t) }\n",
        add ++ "fn go[T has Add](t: T) -> T {\n f := t.add(_)\n f(t)\n}\n",
        add ++ "fn go[T has Add](t: T) -> T {\n f: fn(T) -> T = fn(o) { t.add(o) }\n f(t)\n}\n",
        // `A` is still open, at the ordinal of the template's `K` that `B` is bound to.
        add ++ "fn ap[A, B](f: fn(B) -> A, b: B) -> A { f(b) }\nfn go[K has Add, V](k: K, v: V) -> K { ap(fn(o) { k.add(o) }, k) }\n",
        add ++ "fn ap[A, B](f: fn(B) -> A, b: B) -> A { f(b) }\nfn go[K has Add, V](k: K, v: V) -> K { ap(k.add(_), k) }\n",
    };
    for (clean) |src| try Typecheck.expectDiagCodes(src, &.{});
    try Typecheck.expectDiagCodes(add ++ "fn go[T has Add](t: T) -> T {\n f := fn(o) { t.add(o) }\n f(t)\n}\n", &.{.T0050});
    // `A` is open at the ordinal of the template's `K`, which `B` is bound to: `o` gets neither.
    try Typecheck.expectDiagCodes(add ++ "fn ap[A, B](f: fn(A) -> B, b: B) -> B { b }\nfn go[K has Add, V](k: K, v: V) -> K { ap(fn(o) { k }, k) }\n", &.{.T0050});
}

const then_impls =
    \\impl Option[T] {
    \\ fn then[U](self, f: fn(T) -> Option[U]) -> Option[U] {
    \\  match self {
    \\   .some(v) -> f(v),
    \\   .none -> Option[U].none,
    \\  }
    \\ }
    \\}
    \\impl Result[T, E] {
    \\ fn then[U](self, f: fn(T) -> Result[U, E]) -> Result[U, E] {
    \\  match self {
    \\   .ok(v) -> f(v),
    \\   .err(e) -> Result.err(e),
    \\  }
    \\ }
    \\}
    \\
;

test "a then lambda's constructors take what its expected return type knows" {
    const clean = [_][]const u8{
        then_impls ++ "fn go(o: Option[int]) -> bool {\n r := o.then(fn(v) { if v > 3 { Option.some(\"big\") } else { Option.none } })\n r.is_some()\n}\n",
        then_impls ++ "fn go(o: Option[int]) -> bool {\n r := o.then(fn(v) { if v > 3 { Option.none } else { Option.some(\"big\") } })\n r.is_some()\n}\n",
        then_impls ++ "fn go(o: Option[int]) -> bool {\n r := o.then(fn(v) {\n  if v > 3 { return Option.none }\n  Option.some(\"big\")\n })\n r.is_some()\n}\n",
        then_impls ++ "fn go(o: Option[int]) -> bool {\n r := o.then(fn(v) { match v { 4 -> Option.none, _ -> Option.some(v) } })\n r.is_some()\n}\n",
        then_impls ++ "fn go(o: Option[int]) -> bool {\n r := o.then(fn(v) { if v > 3 { .none } else { .some(v) } })\n r.is_some()\n}\n",
        then_impls ++ "fn go(x: Result[int, str]) -> bool {\n r := x.then(fn(v) { Result.ok(v + 1) })\n r.is_ok()\n}\n",
        then_impls ++ "fn go(x: Result[int, str]) -> bool {\n r := x.then(fn(v) { if v > 0 { Result.err(\"neg\") } else { Result.ok(true) } })\n r.is_ok()\n}\n",
        then_impls ++ "fn go(o: Option[int]) -> bool {\n r := o.then(fn(v) -> Option[str] { if v > 3 { Option.none } else { Option.some(\"big\") } })\n r.is_some()\n}\n",
    };
    for (clean) |src| try Typecheck.expectDiagCodes(src, &.{});
}

test "a then lambda whose returns fix no open var still asks for type args, naming them all" {
    try Typecheck.expectOneDiag(then_impls ++ "fn go(o: Option[int]) -> int {\n _r := o.then(fn(_v) { Option.none })\n 0\n}\n", .T0016, "cannot infer type parameter 'T' for 'Option.none'; add explicit type arguments, e.g. Option[int].none");
    try Typecheck.expectOneDiag(then_impls ++ "fn go(r: Result[int, str]) -> int {\n _s := r.then(fn(_v) { Result.err(\"x\") })\n 0\n}\n", .T0016, "cannot infer type parameter 'T' for 'Result.err'; add explicit type arguments, e.g. Result[int, str].err");
    // Two returns that disagree are a branch mismatch, not an error at the later one.
    try Typecheck.expectDiagCodes(then_impls ++ "fn go(o: Option[int]) -> int {\n _r := o.then(fn(v) { if v > 3 { Option.some(1) } else { Option.some(\"x\") } })\n 0\n}\n", &.{.T0058});
}

test "an if value's expected type reaches its else arm and an else-if ladder" {
    const clean = [_][]const u8{
        "fn f(b: bool) -> Option[str] { if b { Option.some(\"x\") } else { Option.none } }\n",
        "fn f(b: bool) -> Option[str] { if b { Option.none } else { Option.some(\"x\") } }\n",
        "fn f(n: int) -> Option[str] { if n > 1 { Option.some(\"x\") } else if n > 0 { Option.none } else { Option.none } }\n",
        "fn f(n: int) -> Option[str] { return if n > 1 { Option.none } else if n > 0 { Option.none } else { Option.some(\"x\") } }\n",
    };
    for (clean) |src| try Typecheck.expectDiagCodes(src, &.{});
}

fn nestedThen(gpa: std.mem.Allocator, depth: usize, inner: []const u8) ![]u8 {
    var e: std.ArrayList(u8) = .empty;
    defer e.deinit(gpa);
    try e.appendSlice(gpa, inner);
    var i = depth;
    while (i > 0) : (i -= 1) {
        const wrapped = try std.fmt.allocPrint(gpa, "Option.some(v{d}).then(fn(v{d}) {{ if v{d} < 100 {{ {s} }} else {{ Option.none }} }})", .{ i - 1, i, i, e.items });
        defer gpa.free(wrapped);
        e.clearRetainingCapacity();
        try e.appendSlice(gpa, wrapped);
    }
    return std.fmt.allocPrint(gpa, "{s}fn go(v0: int) -> bool {{\n r := {s}\n r.is_some()\n}}\n", .{ then_impls, e.items });
}

test "nested then lambdas check in time linear in their depth, erring or not" {
    const gpa = std.testing.allocator;
    const clean = try nestedThen(gpa, 24, "Option.some(v0 + 1)");
    defer gpa.free(clean);
    try Typecheck.expectDiagCodes(clean, &.{});
    const erring = try nestedThen(gpa, 24, "_e := v0 + true\n Option.some(v0)");
    defer gpa.free(erring);
    try Typecheck.expectDiagCodes(erring, &.{.T0044});
}
