//! Generic-call inference, carved out of `BodyChecker` as free functions over a
//! `*BodyChecker` (the `PatternChecker` pattern): typing a generic call's args in the order
//! that binds its type params, filling what the args leave open, and reporting what
//! cannot be inferred.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Callee = @import("../symbols/Callee.zig").Callee;
const Composite = @import("../symbols/Composite.zig");
const Infer = @import("../symbols/Infer.zig");
const Type = @import("../layout/Type.zig").Type;
const BC = @import("../BodyChecker.zig");
const Typecheck = @import("../types.zig");
const LambdaChecker = @import("../LambdaChecker.zig");
const BodyChecker = BC.BodyChecker;
const FnSym = Typecheck.FnSym;
const refs = Typecheck.refs;

/// Name the clashing LEAF types (`c.prev`/`c.cur`), not the whole arg: for a container
/// conflict the arg is `Vec[int]`/`Vec[bool]` but the real clash is `int` vs `bool`.
/// Spans still point at the two source args. Primary = the LATER arg (source order),
/// related = the earlier one: a stable, order-independent (symmetric `Type.eql`) span
/// pair that is reproducible at `-jN`. An intra-arg clash (`Map[T,T]` vs
/// `Map[int,bool]`) has `first_pos == second_pos`, so both spans are the ONE source arg —
/// collapse to a single caret rather than double-underlining one token.
fn reportInferConflict(bc: *BodyChecker, args: []const Ast.Index, generic_params: []const []const u8, c: @FieldType(Infer.Outcome, "conflict")) error{OutOfMemory}!void {
    if (c.first_pos == c.second_pos) {
        try bc.sink.report(.{ .span = bc.spanOf(args[c.second_pos]), .code = .T0015 }, "conflicting types for type parameter '{s}': {s} vs {s}", .{ generic_params[c.ord], bc.typeName(c.prev), bc.typeName(c.cur) });
    } else {
        try bc.sink.report(.{ .span = bc.spanOf(args[c.second_pos]), .code = .T0015, .related = bc.spanOf(args[c.first_pos]) }, "conflicting types for type parameter '{s}': {s} vs {s}", .{ generic_params[c.ord], bc.typeName(c.prev), bc.typeName(c.cur) });
    }
}

/// How `inferArgs` types an argument that does not wait for the other arguments.
const ArgTyping = union(enum) {
    /// Against its param through what the call's expected type binds when matched
    /// against the callee's return type (this pattern): `x: int8 = id(5)` gives `5` an
    /// `int8`, and `n: int = first([5, 6])` gives the list a `Vec[int]`.
    ret: Type,
    /// With no expectation from the call, as an associated call types its args: an
    /// arg whose param holds no type parameter still has that type outright.
    none,
    /// Against its param with the bindings so far in, when that is ground: a method
    /// call, whose receiver binds the impl's params first.
    receiver,
};

/// `.failed`: an error is already reported. `.unbound`: the ordinal no arg binds, for
/// the caller to report in its own words, with what the args did bind in the type args
/// and `.invalid` for each open one.
const InferResult = union(enum) { ok, failed, unbound: u32 };

/// Whether `arg` takes its type from what a generic call's other args bind: a lambda
/// with an unannotated param (a `_` capture included), or the hole of the capture
/// being checked.
fn waitsForTypeArgs(bc: *const BodyChecker, arg: Ast.Index) bool {
    if (bc.body.hole != Ast.none and arg == bc.body.hole) return true;
    const n = bc.tree.nodes[arg.int()];
    if (n.tag != .lambda) return false;
    for (Ast.protoAt(bc.tree, n.lhs.int()).params) |p| if (bc.tree.nodes[p.int()].lhs == Ast.none) return true;
    return false;
}

/// The partly bound param `want` as a waiting arg's expected type, only when it gives
/// the arg a type (`givesType`): the whole type for the hole, a fn type's param for one
/// of a lambda's unannotated params. Otherwise null, and the arg takes what the call's
/// expected type gives its param, so an expected fn type on the call still types it
/// (`g: fn(int) -> int = id(fn(x) { x + 1 })`). A fn type of another arity is passed
/// on, so the lambda reports the count rather than an untypable param.
fn helpfulWant(bc: *BodyChecker, arg: Ast.Index, want: Type) ?Type {
    if (want.kind == .invalid) return null;
    if (bc.body.hole != Ast.none and arg == bc.body.hole) return if (bc.givesType(want)) want else null;
    if (!want.isFunc()) return null;
    const parts = Composite.fnParts(bc.composite.at(want.funcIdx()));
    const proto = Ast.protoAt(bc.tree, bc.tree.nodes[arg.int()].lhs.int());
    if (parts.params.len != proto.params.len) return want;
    for (proto.params, parts.params) |p, wp| {
        if (bc.tree.nodes[p.int()].lhs == Ast.none and bc.givesType(wp)) return want;
    }
    return null;
}

pub fn typeOfGenericArg(bc: *BodyChecker, arg: Ast.Index, want: ?BC.Expected) error{OutOfMemory}!Type {
    const was = bc.generic_arg;
    bc.generic_arg = arg;
    defer bc.generic_arg = was;
    return bc.typeUnder(arg, want);
}

/// The callee's type args as the call's expected type binds them through `ret`, an
/// open one at `openOrdBase`; null when there is no expected type or it cannot be the
/// return type. Owned by the caller.
fn retBindings(bc: *BodyChecker, ret: Type, n_gp: u32) error{OutOfMemory}!?[]Type {
    const exp = bc.expectedTy() orelse return null;
    const targs = try bc.gpa.alloc(Type, n_gp);
    errdefer bc.gpa.free(targs);
    const bnd = try bc.gpa.alloc(bool, n_gp);
    defer bc.gpa.free(bnd);
    const fp = try bc.gpa.alloc(usize, n_gp);
    defer bc.gpa.free(fp);
    if (Infer.match(n_gp, &.{ret}, &.{exp}, targs, bnd, fp, bc.composite.decomposer()) == .conflict) {
        bc.gpa.free(targs);
        return null;
    }
    fillOpen(targs, bnd, bc.openOrdBase());
    return targs;
}

/// An open var substitutes to itself at `base + ord`, clear of the vars a bound one may
/// stand for (`BodyChecker.openOrdBase`), which `typeOfLambda` and `adoptHole` treat as
/// giving no type and a diagnostic names apart from them.
fn fillOpen(targs: []Type, bound: []const bool, base: u32) void {
    for (targs, bound, 0..) |*t, b, ord| {
        if (!b) t.* = Type.typeVar(base + @as(u32, @intCast(ord)));
    }
}

/// The inferred type args must be monomorphizable value types — the SAME gate
/// `scanCalls` applies, so Pass C and discovery agree on which calls become instances
/// (a `unit`-inferred var, say, is rejected here rather than silently dropped by
/// discovery → lower miss).
pub fn concreteTargs(bc: *BodyChecker, call: Ast.Index, targs: []const Type) error{OutOfMemory}!bool {
    for (targs) |ta| {
        // A bounded template's own param inferred here grounds at the outer
        // instantiation; admit it (the instance re-check re-gates concretely).
        if (bc.abstract_template and ta.isTypeVar()) continue;
        if (!BC.isConcreteValue(ta)) {
            try bc.sink.report(.{ .span = bc.spanOf(call), .code = .T0013 }, "inferred type argument must be a concrete value type; add explicit type arguments", .{});
            return false;
        }
    }
    return true;
}

/// The call `inferArgs` serves: what an argument that cannot fit its param is reported
/// against. `nodes[first..]` are the call's value args.
const InferSite = struct { call: Ast.Index, sig: BodyChecker.CallSig };

/// Infer a generic callee's type args from a call, typing each of `nodes` once into
/// `arg_types`. `nodes` line up with `params`, the receiver first when `recv` is set.
/// The receiver binds first, then every arg that does not wait (`waitsForTypeArgs`);
/// then each waiting arg in source order is typed against its param with the bindings
/// so far in, and the args bind again, so a later lambda sees what an earlier one's
/// return fixed. One left-to-right `Infer.match` over all args decides, as discovery's.
/// A var left unbound after an argument error is no second error, nor is it when an
/// argument's shape cannot fit its param: that argument is the error.
pub fn inferArgs(bc: *BodyChecker, site: InferSite, nodes: []const Ast.Index, params: []const Type, names: []const []const u8, recv: ?Type, typing: ArgTyping, arg_types: []Type, targs: []Type) error{OutOfMemory}!InferResult {
    const n_gp: u32 = @intCast(targs.len);
    bc.infer_depth += 1;
    defer {
        bc.infer_depth -= 1;
        if (bc.infer_depth == 0) bc.retype_choices.clearRetainingCapacity();
    }
    const bnd = try bc.gpa.alloc(bool, n_gp);
    defer bc.gpa.free(bnd);
    const fp = try bc.gpa.alloc(usize, n_gp);
    defer bc.gpa.free(fp);
    @memset(bnd, false);
    const dec = bc.composite.decomposer();
    const first: usize = @intFromBool(recv != null);
    if (recv) |r| {
        arg_types[0] = r;
        _ = Infer.match(n_gp, params[0..1], arg_types[0..1], targs, bnd, fp, dec);
    }
    fillOpen(targs, bnd, bc.openOrdBase());
    const mark = bc.sink.count();
    const exp_targs: ?[]Type = switch (typing) {
        .ret => |r| try retBindings(bc, r, n_gp),
        else => null,
    };
    defer if (exp_targs) |t| bc.gpa.free(t);
    var waiting = false;
    for (nodes[first..], first..) |a, i| {
        if (waitsForTypeArgs(bc, a)) {
            arg_types[i] = .invalid; // binds nothing until it is typed
            waiting = true;
            continue;
        }
        const want: ?BC.Expected = if (typing == .receiver) blk: {
            const w = BC.substTy(bc, params[i], targs);
            break :blk if (w.kind != .invalid and bc.givesType(w)) .{ .ty = w } else null;
        } else bc.partWant(params[i], exp_targs);
        if (try recallRetype(bc, a, params[0 .. i + 1], arg_types[0..i], n_gp, want)) |chosen| {
            arg_types[i] = try typeOfGenericArg(bc, a, chosen);
            continue;
        }
        const arg_mark = bc.sink.count();
        arg_types[i] = try typeOfGenericArg(bc, a, want);
        if (bc.errorsSince(arg_mark)) arg_types[i] = try retypeFromEarlier(bc, a, params[0 .. i + 1], arg_types[0..i], n_gp, want, arg_mark, arg_types[i]);
    }
    var res = Infer.match(n_gp, params, arg_types, targs, bnd, fp, dec);
    if (waiting) {
        const open = try bc.gpa.alloc(bool, n_gp);
        defer bc.gpa.free(open);
        const save_names = bc.body.open_names;
        const save_vars = bc.body.open_vars;
        defer {
            bc.body.open_names = save_names;
            bc.body.open_vars = save_vars;
        }
        bc.body.open_names = names;
        bc.body.open_vars = open;
        for (nodes[first..], first..) |a, i| {
            if (!waitsForTypeArgs(bc, a)) continue;
            const fallback = bc.partWant(params[i], exp_targs);
            if (res == .conflict) {
                arg_types[i] = try typeOfGenericArg(bc, a, fallback);
                continue;
            }
            fillOpen(targs, bnd, bc.openOrdBase());
            for (open, bnd) |*o, b| o.* = !b;
            const helpful: ?BC.Expected = if (helpfulWant(bc, a, BC.substTy(bc, params[i], targs))) |w| .{ .ty = w } else null;
            arg_types[i] = try typeOfGenericArg(bc, a, helpful orelse fallback);
            res = Infer.match(n_gp, params, arg_types, targs, bnd, fp, dec);
        }
    }
    switch (res) {
        .conflict => |c| {
            // A var the receiver fixed (`T` of `Vec[int]` in `xs.map(..)`) is no type
            // parameter the call infers, so an arg that disagrees with it is a plain
            // argument mismatch, not a conflict between two args.
            if (recv != null and c.first_pos == 0) {
                @memset(bnd, false);
                _ = Infer.match(n_gp, params[0..1], arg_types[0..1], targs, bnd, fp, dec);
                fillOpen(targs, bnd, bc.openOrdBase());
                if (try reportMisfit(bc, site, nodes, first, params, arg_types, targs, bnd)) return .failed;
            }
            try reportInferConflict(bc, nodes, names, c);
            return .failed;
        },
        .unbound => |u| {
            if (bc.errorsSince(mark)) return .failed;
            // An `.invalid` arg was diagnosed where it went wrong (a resolve error, say);
            // the hole being typed is invalid only because it waits for this call.
            for (nodes[first..], arg_types[first..]) |a, at| if (at.kind == .invalid and a != bc.body.hole) return .failed;
            fillOpen(targs, bnd, bc.openOrdBase());
            if (try reportMisfit(bc, site, nodes, first, params, arg_types, targs, bnd)) return .failed;
            for (targs, bnd) |*t, b| if (!b) {
                t.* = .invalid;
            };
            return .{ .unbound = u.ord };
        },
        .ok => return .ok,
    }
}

/// An arg that errs under the want it was given is typed again against its param (the
/// last of `params`) with what the earlier args bind, and keeps that typing when it errs
/// less: `Option.none`, `.none` and `[]` take their type only from their slot, which
/// `pair(t, Option.none)` fixes through `t`. Only an erring arg is typed again, and only
/// earlier args count, so an arg that types itself (`f(x8, 5)`) binds as before and
/// check and discovery bind the same type args.
fn retypeFromEarlier(bc: *BodyChecker, arg: Ast.Index, params: []const Type, earlier: []const Type, n_gp: u32, want: ?BC.Expected, mark: usize, got: Type) error{OutOfMemory}!Type {
    const w = (try earlierWant(bc, params, earlier, n_gp)) orelse return got;
    if (want) |e| if (e.ty.eql(w)) return got;
    const errs = bc.errorCountSince(mark);
    bc.sink.truncate(mark);
    const t = try typeOfGenericArg(bc, arg, .{ .ty = w });
    const keep = bc.errorCountSince(mark) < errs;
    try bc.retype_choices.put(bc.gpa, arg, .{ .want = want, .from_earlier = w, .keep = keep });
    if (keep) return t;
    bc.sink.truncate(mark);
    return typeOfGenericArg(bc, arg, want);
}

/// The want `retypeFromEarlier` chose for `arg` when it was last typed in the same
/// context (its want and what the earlier args bind), so it is typed once under it.
fn recallRetype(bc: *BodyChecker, arg: Ast.Index, params: []const Type, earlier: []const Type, n_gp: u32, want: ?BC.Expected) error{OutOfMemory}!??BC.Expected {
    const c = bc.retype_choices.get(arg) orelse return null;
    if (!sameWant(c.want, want)) return null;
    const w = (try earlierWant(bc, params, earlier, n_gp)) orelse return null;
    if (!w.eql(c.from_earlier)) return null;
    return if (c.keep) .{ .ty = w } else want;
}

fn sameWant(a: ?BC.Expected, b: ?BC.Expected) bool {
    const x = a orelse return b == null;
    const y = b orelse return false;
    return x.soft == y.soft and x.ty.eql(y.ty);
}

/// The last of `params` with what `earlier` (the args before it) binds, when that is a
/// type an arg can take from its slot.
fn earlierWant(bc: *BodyChecker, params: []const Type, earlier: []const Type, n_gp: u32) error{OutOfMemory}!?Type {
    const targs = try bc.gpa.alloc(Type, n_gp);
    defer bc.gpa.free(targs);
    const bnd = try bc.gpa.alloc(bool, n_gp);
    defer bc.gpa.free(bnd);
    const fp = try bc.gpa.alloc(usize, n_gp);
    defer bc.gpa.free(fp);
    const i = earlier.len;
    if (Infer.match(n_gp, params[0..i], earlier, targs, bnd, fp, bc.composite.decomposer()) == .conflict) return null;
    fillOpen(targs, bnd, bc.openOrdBase());
    const w = BC.substTy(bc, params[i], targs);
    if (w.kind == .invalid or !bc.givesType(w)) return null;
    return w;
}

/// Whether `got` can stand where `want` is wanted once `want`'s type vars are bound: a
/// type var fits anything, a fn or a generic type fits part by part, and anything else
/// by assignability.
fn fitsShape(bc: *BodyChecker, want: Type, got: Type) bool {
    if (got.kind == .invalid or got.kind == .never or want.isTypeVar()) return true;
    if (want.isFunc() or want.isApp()) {
        if (want.isFunc() != got.isFunc() or want.isApp() != got.isApp()) return false;
        const w = bc.composite.at(want.struct_id);
        const g = bc.composite.at(got.struct_id);
        if (want.isApp() and (w.ctor != g.ctor or w.ctor_is_enum != g.ctor_is_enum)) return false;
        if (w.args.len != g.args.len) return false;
        for (w.args, g.args) |wa, ga| if (!fitsShape(bc, wa, ga)) return false;
        return true;
    }
    return Type.assignable(want, got);
}

/// Report the first value arg whose type cannot fit its partly bound param, the reason
/// a var stayed unbound (a lambda of the wrong shape binds nothing). False when every
/// arg fits, so the var is genuinely uninferable.
fn reportMisfit(bc: *BodyChecker, site: InferSite, nodes: []const Ast.Index, first: usize, params: []const Type, arg_types: []const Type, targs: []const Type, bnd: []const bool) error{OutOfMemory}!bool {
    const open = try bc.gpa.alloc(bool, bnd.len);
    defer bc.gpa.free(open);
    for (open, bnd) |*o, b| o.* = !b;
    for (nodes[first..], first..) |a, i| {
        const want = BC.substTy(bc, params[i], targs);
        if (fitsShape(bc, want, arg_types[i])) continue;
        const pos = i - first + 1;
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(bc.gpa);
        try text.print(bc.gpa, "argument {d}{s}: expected ", .{ pos, bc.pipedArgHint(site.call, pos) });
        try bc.writeTypeNamed(&text, want, site.sig.generic_names, open, bc.openOrdBase());
        try text.print(bc.gpa, ", got {s}", .{bc.typeName(arg_types[i])});
        try bc.writeSigSuffixOpen(&text, site.sig, targs, open);
        try bc.sink.report(.{ .span = bc.spanOf(a), .code = .T0040, .related = site.sig.decl }, "{s}", .{text.items});
        return true;
    }
    return false;
}

/// A call of a method with its own type params (`fn map[U](self, f: fn(T) -> U)`), or
/// of such an associated fn through a turbofish receiver (`is_static`): the receiver
/// binds the impl's params and the args the method's own. Each instantiation has its
/// own row on one receiver, so the pending slot carries the call for the seal to pick;
/// a call that picks none leaves the template for the LSP to show.
pub fn checkOwnGenericCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node, recv_ty: Type, mf: FnSym, member: []const u8, is_static: bool) error{OutOfMemory}!Type {
    const args = Ast.rangeSlice(bc.tree, (n.rhs).int());
    const sig = if (is_static) bc.fnSig(mf, member) else bc.methodSig(mf, member);
    bc.setCallee(n, Callee.pendingCall(node_idx));
    const params = try bc.gpa.alloc(Type, mf.params.len + @intFromBool(is_static));
    defer bc.gpa.free(params);
    if (is_static) {
        params[0] = mf.self_type;
        @memcpy(params[1..], mf.params);
    } else @memcpy(params, mf.params);
    const targs = try bc.gpa.alloc(Type, mf.generic_params.len);
    defer bc.gpa.free(targs);
    if (args.len != sig.params.len) {
        try reportOwnGenericArity(bc, node_idx, args.len, sig, params[0], recv_ty, targs);
        try bc.drainArgs(args);
        return .invalid;
    }
    const nodes = try bc.gpa.alloc(Ast.Index, args.len + 1);
    defer bc.gpa.free(nodes);
    nodes[0] = callee.lhs;
    @memcpy(nodes[1..], args);
    const arg_types = try bc.gpa.alloc(Type, nodes.len);
    defer bc.gpa.free(arg_types);
    switch (try inferArgs(bc, .{ .call = node_idx, .sig = sig }, nodes, params, mf.generic_params, recv_ty, .receiver, arg_types, targs)) {
        .failed => return .invalid,
        .unbound => |ord| {
            const n_impl = Typecheck.implParamCount(bc.composite, mf);
            try bc.reportUninferable(bc.spanOf(node_idx), " '{s}' of method '{s}' from its arguments", .{ mf.generic_params[ord], member }, member, targs[n_impl..], &.{"(..)"}, null);
            return .invalid;
        },
        .ok => {},
    }
    if (!try concreteTargs(bc, node_idx, targs)) return .invalid;
    var s = sig.substituted(targs);
    s.pretyped = arg_types[1..];
    _ = try bc.checkCall(node_idx, args, s);
    return bc.substRet(node_idx, mf, targs);
}

/// The arity error of an own-generic call, its signature shown as the receiver binds
/// it: `'map' takes (fn(int) -> U)`, not the template's raw vars.
fn reportOwnGenericArity(bc: *BodyChecker, call: Ast.Index, got: usize, sig: BodyChecker.CallSig, self_pat: Type, recv_ty: Type, targs: []Type) error{OutOfMemory}!void {
    const n_gp: u32 = @intCast(targs.len);
    const bnd = try bc.gpa.alloc(bool, n_gp);
    defer bc.gpa.free(bnd);
    const fp = try bc.gpa.alloc(usize, n_gp);
    defer bc.gpa.free(fp);
    _ = Infer.match(n_gp, &.{self_pat}, &.{recv_ty}, targs, bnd, fp, bc.composite.decomposer());
    fillOpen(targs, bnd, bc.openOrdBase());
    const open = try bc.gpa.alloc(bool, n_gp);
    defer bc.gpa.free(open);
    for (open, bnd) |*o, b| o.* = !b;
    var suffix: std.ArrayList(u8) = .empty;
    defer suffix.deinit(bc.gpa);
    try bc.writeSigSuffixOpen(&suffix, sig, targs, open);
    try bc.emitArity(call, sig.params.len, got, sig.decl, suffix.items);
}

/// T0016 for `Pair.make(..)` on the bare generic struct `sid`. When the call was inferred,
/// `mf` is the fn, `fn_args` what the call bound of its type params (`.invalid` when
/// open) and `open` the one no arg binds. The hint names the struct's args through the
/// impl's pattern, then the fn's own args when it has any, so it gets past T0016 for both.
pub fn reportUninferableAssociated(bc: *BodyChecker, recv: Ast.Node, member: []const u8, sid: u32, mf: ?FnSym, fn_args: []const Type, open: ?u32) error{OutOfMemory}!void {
    const name = bc.nameText(recv.main_token);
    const n = bc.structSyms()[sid].generic_params.len;
    const args = try bc.gpa.alloc(Type, n);
    defer bc.gpa.free(args);
    @memset(args, .invalid);
    var own: ?[]const Type = null;
    if (mf) |f| {
        if (f.self_type.isApp()) {
            const pat = bc.composite.at(f.self_type.appIdx()).args;
            for (args[0..@min(n, pat.len)], pat[0..@min(n, pat.len)]) |*a, p| {
                if (p.isTypeVar() and p.typeVarOrd() < fn_args.len) a.* = fn_args[p.typeVarOrd()];
            }
        }
        const n_impl = Typecheck.implParamCount(bc.composite, f);
        if (f.generic_params.len > n_impl) own = fn_args[n_impl..];
    }
    if (open) |ord|
        try bc.reportUninferable(bc.tokSpan(recv.main_token), " '{s}' for '{s}.{s}'", .{ mf.?.generic_params[ord], name, member }, name, args, &.{ ".", member }, own)
    else
        try bc.reportUninferable(bc.tokSpan(recv.main_token), " for '{s}.{s}'", .{ name, member }, name, args, &.{ ".", member }, own);
}

/// Type a bare `Box.new(5)`: bind the impl's type params from the value args alone,
/// then check the call as `Box[int].new(5)` would be. Returns null, with no arg typed,
/// when the member takes `self`.
pub fn inferAssociatedCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, callee: Ast.Node, m: Typecheck.TemplateMethod, args: []const Ast.Index) error{OutOfMemory}!?Type {
    if (m.has_self) return null;
    const mf = bc.model.fns[m.fn_id];
    if (!mf.self_type.isApp()) return null;
    // As for a generic fn: the count is checked before inference, which would blame a
    // missing type argument instead.
    if (args.len != mf.params.len) {
        try bc.drainArgs(args);
        try bc.reportArity(node_idx, args.len, bc.fnSig(mf, bc.nameText(callee.main_token)));
        return .invalid;
    }
    const arg_types = try bc.gpa.alloc(Type, args.len);
    defer bc.gpa.free(arg_types);
    const out = try bc.gpa.alloc(Type, mf.generic_params.len);
    defer bc.gpa.free(out);
    const recv = bc.tree.nodes[(callee.lhs).int()];
    switch (try inferArgs(bc, .{ .call = node_idx, .sig = bc.fnSig(mf, bc.nameText(callee.main_token)) }, args, mf.params, mf.generic_params, null, .none, arg_types, out)) {
        .failed => return .invalid,
        .unbound => |ord| {
            try reportUninferableAssociated(bc, recv, bc.nameText(callee.main_token), bc.composite.at(mf.self_type.appIdx()).ctor, mf, out, ord);
            return .invalid;
        },
        .ok => {},
    }
    if (!try concreteTargs(bc, node_idx, out)) return .invalid;
    const pat = bc.composite.at(mf.self_type.appIdx()).args;
    const app_args = try bc.gpa.alloc(Type, pat.len);
    defer bc.gpa.free(app_args);
    for (pat, app_args) |p, *o| o.* = BC.substTy(bc, p, out);
    const sid = bc.composite.at(mf.self_type.appIdx()).ctor;
    // Load-bearing, as on the `type_app` path: scanCalls / reify / sealCallees read the
    // receiver off `node_types[callee.lhs]` whatever the node's tag.
    bc.node_types[(callee.lhs).int()] = Type.app(bc.composite.intern(bc.gpa, sid, app_args, false) catch return error.OutOfMemory);
    bc.setCallee(n, if (Typecheck.hasOwnTypeParams(bc.composite, mf)) Callee.pendingCall(node_idx) else Callee.pending);
    var sig = bc.fnSig(mf, bc.nameText(callee.main_token)).substituted(out);
    sig.pretyped = arg_types;
    _ = try bc.checkCall(node_idx, args, sig);
    return bc.substRet(node_idx, mf, out);
}
