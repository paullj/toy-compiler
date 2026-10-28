//! The whole-program conformance-coherence phase, extracted from the `Typecheck`
//! mega-struct as free functions over `*Typecheck` (Zig has no struct-field privacy, so
//! these read the checker's fields directly). `checkCoherence` is the serial pre-Pass-C
//! barrier that enforces one-impl-per-(protocol, receiver) and completeness; `deriveEqFromOrd`
//! is the refinement pass that runs right after it. Both are driven once from `runGraph`.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Typecheck = @import("../types.zig");
const Type = Typecheck.Type;
const Conformance = Typecheck.Conformance;
const FnSym = Typecheck.FnSym;
const GraphModuleInput = Typecheck.GraphModuleInput;
const Span = @import("../diagnostics/model.zig").Span;
const Infer = @import("../symbols/Infer.zig");

const testing = std.testing;

/// Whole-program conformance coherence. Walks every `impl .. has ..` in
/// module-then-decl order (SERIAL, before Pass C) and enforces:
///   * exactly one impl per (protocol, receiver type-ctor) — a duplicate (INCLUDING a
///     retroactive one in a SIBLING module: no orphan rule) is T0020 at the second impl,
///     or at the program's impl when the other is in bundled std;
///   * `has P` names a declared protocol AND provides every method P requires — an
///     undeclared protocol OR a missing method is T0021.
/// Accepted conformances are recorded onto `t.conformances`. The
/// `seen` set is `getOrPut`-only (never iterated), so its hash/thread order cannot leak
/// into the emit stream; emit order is module-id then source order → `-jN`-stable.
pub fn checkCoherence(t: *Typecheck, mods: []const GraphModuleInput) !void {
    // `seen` maps a serialized coherence key (see `writeCoherenceKey`) to the impl that
    // claimed it (null for a prelude conformance or a generic impl); it owns its keys
    // (dup'd on insert), freed on return.
    var seen: std.StringHashMapUnmanaged(?ImplAt) = .empty;
    defer {
        var it = seen.keyIterator();
        while (it.next()) |k| t.gpa.free(k.*);
        seen.deinit(t.gpa);
    }
    var keybuf: std.ArrayList(u8) = .empty;
    defer keybuf.deinit(t.gpa);

    // Seed `seen` from the builtin conformances already pre-registered by the prelude
    // BEFORE walking user impls, so a duplicate user `impl int has Eq` collides
    // (T0020). At this point `t.conformances` holds EXACTLY the prelude entries (the
    // only other appender is this fn's accept path below), so iterating its slice in
    // insertion order is a pure function of source — `-jN`-stable.
    for (t.conformances.items) |c| {
        try writeCoherenceKey(t.gpa, &keybuf, c.protocol, c.recv, c.protocol_args);
        const gop = try seen.getOrPut(t.gpa, keybuf.items);
        if (!gop.found_existing) {
            gop.key_ptr.* = try t.gpa.dupe(u8, keybuf.items);
            gop.value_ptr.* = null;
        }
    }

    for (mods, 0..) |_, mi| {
        const mod: u32 = @intCast(mi);
        _ = t.gphSelect(mod);
        if (t.tree.nodes.len == 0) continue;
        const prog = t.tree.nodes[Ast.root(t.tree.nodes).int()];
        if (prog.tag != .program) continue;

        for (Ast.rangeSlice(t.tree, prog.lhs.int())) |decl_idx| {
            const decl = t.tree.nodes[decl_idx.int()];
            if (decl.tag != .impl_has_decl) continue;

            // Resolve the protocol. An undeclared protocol ref -> T0021 (no coherence
            // key to form; nothing more to check for this impl).
            const proto_ref = Ast.implProtocol(t.tree, decl).?;
            const pid = t.protocolIdFromNode(proto_ref) orelse {
                // A generic ref `P[int]` is a `type_app` whose main token is its `[`.
                const ref_tok = t.tree.nodes[Ast.protocolRefBase(t.tree, proto_ref).int()].main_token;
                try t.sink.report(.{ .span = t.tokSpan(ref_tok), .code = .T0021 }, "'{s}' is not a declared protocol", .{t.nameText(ref_tok)});
                continue;
            };

            const provided = Ast.implMethods(t.tree, decl);
            // T0013 rejected the receiver pattern: the impl conforms to nothing, so it has
            // no completeness or signature to get wrong.
            if (t.implRejected(mod, decl.lhs)) continue;

            // Completeness: every method the protocol requires must be provided.
            for (t.protocols.items[pid].methods) |req| {
                var found = false;
                for (provided) |mnode| {
                    if (std.mem.eql(u8, t.nameText(t.tree.nodes[mnode.int()].main_token), req)) {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    try t.sink.report(.{ .span = implHeaderSpan(t, decl, proto_ref), .code = .T0021 }, "impl of protocol '{s}' for '{s}' is missing method '{s}'", .{ t.protocols.items[pid].name, t.nameText(decl.main_token), req });
            }

            const prot = t.protocols.items[pid];

            // A GENERIC conforming receiver `impl Ctor[T] has P[..]`: its methods
            // register as generic-impl TEMPLATEs (never `t.methods`), so record the
            // conformance in the separate template table keyed by the receiver ctor and
            // monomorphize the T0024 signature check per the receiver's own type-params.
            // `receiverTypeFromNode` returns null for a `type_app`, so the concrete path
            // below never handles this shape.
            if (t.tree.nodes[decl.lhs.int()].tag == .type_app) {
                try recordGenericConformance(t, mod, decl_idx, proto_ref, pid, prot, provided);
                continue;
            }

            // Coherence: reject a duplicate (protocol, receiver-ctor). The receiver was
            // already resolved (and any error emitted) in Phase A, so re-resolve
            // silently; an unresolved receiver simply forms no coherence key.
            const recv = t.receiverTypeFromNode(decl.lhs) orelse continue;

            // decode + validate this impl's protocol type-args `impl P has Into[int]`.
            // Arity must match the protocol's generic-param count; each arg must be a
            // concrete non-composite value type (a composite `App` arg is out of scope —
            // its check-time index is run-order-dependent, which would break the coherence
            // key + fp determinism). A bad-args impl forms no key + records no conformance
            // (so a later valid sibling still registers cleanly).
            const arg_nodes = Ast.protocolRefArgs(t.tree, proto_ref);
            if (arg_nodes.len != prot.generic_params.len) {
                try t.sink.report(.{ .span = t.spanOf(proto_ref), .code = .T0054 }, "protocol '{s}' expects {d} type argument(s), got {d}", .{ prot.name, prot.generic_params.len, arg_nodes.len });
                continue;
            }
            var pargs_buf: std.ArrayList(Type) = .empty;
            defer pargs_buf.deinit(t.gpa);
            var pargs_ok = true;
            var rejected = false;
            for (arg_nodes) |an| {
                const aty = t.typeFromNode(an);
                switch (aty.kind) {
                    .int, .bool, .str, .unit, .@"struct", .@"enum" => {},
                    .invalid => pargs_ok = false, // typeFromNode already emitted T0001
                    else => {
                        try t.sink.report(.{ .span = t.spanOf(an), .code = .T0013 }, "a generic-protocol argument must be a concrete non-composite value type (composite protocol args are not yet supported)", .{});
                        pargs_ok = false;
                        rejected = true;
                    },
                }
                try pargs_buf.append(t.gpa, aty);
            }
            if (rejected) try t.noteRejectedImpl(mod, decl.lhs, recv);
            if (!pargs_ok) continue;
            const pargs = pargs_buf.items;

            // Signature compatibility (T0024): a conforming impl method's signature
            // must MATCH the protocol's declared signature, with `Self` grounded to the
            // receiver and each protocol type-param grounded to `pargs` — the soundness
            // prerequisite for bound-as-axiom checking (a bound body types `v.m()` against
            // the protocol sig, so a mismatched impl would miscompile). Builtin-scalar
            // conformances (`int`/`bool has Eq`) are pre-registered by the prelude, not
            // `impl_has_decl` source decls, so this loop never visits them (no method node
            // to check) — correct by construction.
            for (prot.methods, 0..) |req, j| {
                // Resolve the method THIS impl block provides for `req` — never
                // `findMethod`'s whole-program first match: for a shared imported
                // receiver type an overlapping sibling impl in ANOTHER module (T0020)
                // registers a `Method` with the same `(recv, name)`, and its
                // `FnSym.decl_node` indexes that foreign module's tree — dereferencing it
                // against the currently-active `t.tree` here (a different, possibly
                // shorter node array) crashes or misattributes the diagnostic. The
                // provided method nodes are in `t.tree`; the FnSym is keyed by
                // `(mod, decl_node)`, which uniquely identifies this block's method.
                var method_node: Ast.Index = Ast.none;
                for (provided) |pn| {
                    if (std.mem.eql(u8, t.nameText(t.tree.nodes[pn.int()].main_token), req)) {
                        method_node = pn;
                        break;
                    }
                }
                if (method_node == Ast.none) continue; // missing method: already T0021 above
                var maybe_fn: ?FnSym = null;
                var impl_gid: u32 = 0;
                for (t.fns.items, 0..) |f, fi| {
                    if (f.mod == mod and f.decl_node == method_node) {
                        maybe_fn = f;
                        impl_gid = @intCast(fi);
                        break;
                    }
                }
                const impl_fn = maybe_fn orelse continue;
                const want_params = prot.method_params[j];
                const want_ret = prot.method_rets[j];
                var mismatch = impl_fn.params.len != want_params.len;
                if (!mismatch) {
                    for (want_params, impl_fn.params) |wp, ip| {
                        if (!Type.eql(Typecheck.groundProtoType(t, wp, recv, pargs), ip)) {
                            mismatch = true;
                            break;
                        }
                    }
                    if (!mismatch and !Type.eql(Typecheck.groundProtoType(t, want_ret, recv, pargs), impl_fn.ret)) mismatch = true;
                }
                if (mismatch)
                    try t.sink.report(.{ .span = sigSpan(t, method_node), .code = .T0024 }, "impl method '{s}' has a signature incompatible with protocol '{s}'", .{ req, prot.name });
                try checkReceiverMode(t, prot, j, method_node);
                // STAMP this witness `Method` entry with the conformance's
                // `(protocol_id, protocol_args)` so the multi-conformance resolver can pick
                // it by args. Runs BEFORE `buildModel` (the Model aliases `t.methods.items`
                // AFTER this), and mutates in place (no realloc), so the stamp is
                // snapshot-safe. Each entry gets its OWN owned dupe (freed per-entry).
                for (t.methods.items) |*mth| {
                    if (mth.fn_id == impl_gid) {
                        mth.protocol_id = pid;
                        mth.protocol_args = try t.gpa.dupe(Type, pargs);
                        break;
                    }
                }
            }

            try writeCoherenceKey(t.gpa, &keybuf, pid, recv, pargs);
            const gop = try seen.getOrPut(t.gpa, keybuf.items);
            const here: ImplAt = .{ .mod = mod, .decl = decl_idx };
            if (gop.found_existing) {
                _ = try reportOverlap(t, pid, gop.value_ptr.*, here, "");
            } else {
                gop.key_ptr.* = try t.gpa.dupe(u8, keybuf.items);
                gop.value_ptr.* = here;
                try t.conformances.append(t.gpa, .{ .protocol = pid, .recv = recv, .protocol_args = try t.gpa.dupe(Type, pargs) });
            }
        }
    }
}

/// Record a GENERIC conformance `impl Ctor[T] has P[..]` into `t.template_conformances`
/// and validate its method signatures (T0024) monomorphized per the receiver's own
/// type-params. Its witnesses are the impl's templates, stamped below; each reached
/// instance gets its concrete row in `monomorphize`. Overlap detection compares the
/// receiver CTOR int (the App index is interning-order dependent — would break `-jN`).
fn recordGenericConformance(
    t: *Typecheck,
    mod: u32,
    decl_idx: Ast.Index,
    proto_ref: Ast.Index,
    pid: u32,
    prot: Typecheck.ProtocolSym,
    provided: []const Ast.Index,
) !void {
    const decl = t.tree.nodes[decl_idx.int()];
    // No head: the receiver did not resolve, which Phase A reported.
    const head = t.implHead(mod, decl.lhs) orelse return;
    const recv_app = head.recv;
    const e = t.composite.at(recv_app.appIdx());
    const gnames = head.gnames;

    const arg_nodes = Ast.protocolRefArgs(t.tree, proto_ref);
    if (arg_nodes.len != prot.generic_params.len) {
        try t.sink.report(.{ .span = t.spanOf(proto_ref), .code = .T0054 }, "protocol '{s}' expects {d} type argument(s), got {d}", .{ prot.name, prot.generic_params.len, arg_nodes.len });
        return;
    }
    // The protocol-arg PATTERN: an arg naming an impl type-param becomes `type_var(ord)`
    // (grounded through the receiver App at the use site); any other arg is a concrete
    // type via `typeFromNode`. Put the impl's params in scope for the decode so a NESTED
    // type-app arg spelling them (`Iterator[Entry[K, V]]`) resolves K/V to their type_vars
    // rather than T0001; a bare-identifier gname arg is mapped directly (same type_var),
    // and a concrete arg still decodes concretely.
    const prev_gp = t.cur_generic_params;
    t.cur_generic_params = gnames;
    defer t.cur_generic_params = prev_gp;
    var pargs_buf: std.ArrayList(Type) = .empty;
    defer pargs_buf.deinit(t.gpa);
    for (arg_nodes) |an| {
        const anode = t.tree.nodes[an.int()];
        var mapped: ?Type = null;
        if (anode.tag == .identifier) {
            const nm = t.nameText(anode.main_token);
            for (gnames, 0..) |gn, k| if (std.mem.eql(u8, gn, nm)) {
                mapped = Type.typeVar(@intCast(k));
                break;
            };
        }
        try pargs_buf.append(t.gpa, mapped orelse t.typeFromNode(an));
    }
    const pargs = pargs_buf.items;

    // T0024: each protocol method's signature — `Self` grounded to the receiver App and
    // each protocol type-param grounded to the pattern, DEEPLY (so `Option[Item]`'s inner
    // `type_var` grounds inside the App) — must match the impl method's decoded sig.
    for (prot.methods, 0..) |req, j| {
        var method_node: Ast.Index = Ast.none;
        for (provided) |pn| {
            if (std.mem.eql(u8, t.nameText(t.tree.nodes[pn.int()].main_token), req)) {
                method_node = pn;
                break;
            }
        }
        if (method_node == Ast.none) continue; // missing method: already T0021
        var maybe_fn: ?FnSym = null;
        for (t.fns.items) |f| {
            if (f.mod == mod and f.decl_node == method_node) {
                maybe_fn = f;
                break;
            }
        }
        const impl_fn = maybe_fn orelse continue;
        const want_params = prot.method_params[j];
        const want_ret = prot.method_rets[j];
        var mismatch = impl_fn.params.len != want_params.len;
        if (!mismatch) {
            for (want_params, impl_fn.params) |wp, ip| {
                if (!Type.eql(Typecheck.groundProtoType(t, wp, recv_app, pargs), ip)) {
                    mismatch = true;
                    break;
                }
            }
            if (!mismatch and !Type.eql(Typecheck.groundProtoType(t, want_ret, recv_app, pargs), impl_fn.ret)) mismatch = true;
        }
        if (mismatch)
            try t.sink.report(.{ .span = sigSpan(t, method_node), .code = .T0024 }, "impl method '{s}' has a signature incompatible with protocol '{s}'", .{ req, prot.name });
        try checkReceiverMode(t, prot, j, method_node);
    }

    // Every impl pattern is the receiver's own params in order, so `type_var(k)` names
    // the same receiver arg in both impls and one substitution grounds both patterns.
    // Two raw patterns that differ (`Conv[T]`, `Conv[int]`) can still both apply at one
    // instance (`Box[int]`), where no call could choose between them.
    const n_vars: u32 = @intCast(e.args.len);
    const here: ImplAt = .{ .mod = mod, .decl = decl_idx };
    var replaces: ?*Typecheck.TemplateConformance = null;
    for (t.template_conformances.items) |*tc| {
        if (tc.protocol_id != pid or tc.recv_ctor != e.ctor or tc.recv_is_enum != e.ctor_is_enum) continue;
        if (!try patternsOverlap(t.gpa, @max(n_vars, tc.n_params), tc.protocol_args, pargs, t.composite.decomposer())) continue;
        const loser = try reportOverlap(t, pid, .{ .mod = tc.impl_mod, .decl = tc.impl_decl }, here, ": an earlier impl's protocol arguments can equal these at one instance");
        if (loser.mod == here.mod and loser.decl == here.decl) return;
        replaces = tc;
        break;
    }
    const owned_pargs = try t.gpa.dupe(Type, pargs);
    const row: Typecheck.TemplateConformance = .{
        .protocol_id = pid,
        .recv_ctor = e.ctor,
        .recv_is_enum = e.ctor_is_enum,
        .protocol_args = owned_pargs,
        .recv = recv_app,
        .n_params = n_vars,
        .bounds = head.bounds,
        .bound_args = head.bound_args,
        .impl_mod = mod,
        .impl_decl = decl_idx,
    };
    if (replaces) |tc| {
        // The earlier impl lost (a program impl overlapping std's) and its templates are
        // gone, so its row would conform with no witness.
        if (tc.protocol_args.len > 0) t.gpa.free(@constCast(tc.protocol_args));
        tc.* = row;
    } else t.template_conformances.append(t.gpa, row) catch |err| {
        t.gpa.free(owned_pargs);
        return err;
    };
    // Stamp each witness template with its conformance, so explicit protocol args on a
    // call (`b.into[int]()`) pick among the receiver's conformances. In place, before
    // the Model snapshot, as for a concrete conformance's `Method` row.
    for (provided) |pn| {
        for (t.fns.items, 0..) |f, fi| {
            if (f.mod != mod or f.decl_node != pn) continue;
            for (t.templates.items) |*tm| if (tm.fn_id == fi) {
                tm.protocol_id = pid;
                tm.protocol_args = owned_pargs;
            };
            break;
        }
    }
}

/// Whether one substitution of the receiver's `n_vars` params makes the protocol-arg
/// patterns `a` and `b` equal: two-sided unification over one shared variable space.
/// An `invalid` arg was already reported, so it overlaps nothing.
fn patternsOverlap(gpa: std.mem.Allocator, n_vars: u32, a: []const Type, b: []const Type, dec: Infer.Decomposer) !bool {
    if (a.len != b.len) return false;
    const sub = try gpa.alloc(?Type, n_vars);
    defer gpa.free(sub);
    @memset(sub, null);
    for (a, b) |x, y| if (!unifyPattern(sub, x, y, dec)) return false;
    return true;
}

fn resolvePattern(sub: []const ?Type, ty: Type) Type {
    var cur = ty;
    while (cur.isTypeVar() and cur.typeVarOrd() < sub.len) {
        cur = sub[cur.typeVarOrd()] orelse break;
    }
    return cur;
}

fn unifyPattern(sub: []?Type, x0: Type, y0: Type, dec: Infer.Decomposer) bool {
    const x = resolvePattern(sub, x0);
    const y = resolvePattern(sub, y0);
    if (x.kind == .invalid or y.kind == .invalid) return false;
    if (x.isTypeVar() and y.isTypeVar() and x.typeVarOrd() == y.typeVarOrd()) return true;
    if (x.isTypeVar()) return bindPattern(sub, x.typeVarOrd(), y, dec);
    if (y.isTypeVar()) return bindPattern(sub, y.typeVarOrd(), x, dec);
    const xe = dec.func(dec.ctx, x) orelse return Type.eql(x, y);
    const ye = dec.func(dec.ctx, y) orelse return false;
    if (xe.is_fn != ye.is_fn or xe.ctor != ye.ctor or xe.is_enum != ye.is_enum or xe.args.len != ye.args.len) return false;
    for (xe.args, ye.args) |xa, ya| if (!unifyPattern(sub, xa, ya, dec)) return false;
    return true;
}

fn bindPattern(sub: []?Type, ord: u32, ty: Type, dec: Infer.Decomposer) bool {
    if (ord >= sub.len) return false;
    // A var bound to a type that contains it has only infinite solutions.
    if (occursIn(sub, ord, ty, dec)) return false;
    sub[ord] = ty;
    return true;
}

fn occursIn(sub: []const ?Type, ord: u32, ty0: Type, dec: Infer.Decomposer) bool {
    const ty = resolvePattern(sub, ty0);
    if (ty.isTypeVar()) return ty.typeVarOrd() == ord;
    const e = dec.func(dec.ctx, ty) orelse return false;
    for (e.args) |arg| if (occursIn(sub, ord, arg, dec)) return true;
    return false;
}

/// T0076: the conformance method at `fn_idx` must take its receiver the way protocol
/// method `j` declares it, `self` or `mut self`. The two pass the receiver differently
/// (a copy, or the caller's address), and a desugar that calls the witness (`==`, `+`,
/// `io.println`, a `Map` key's hash, `for`'s `next`) has only the protocol to go by.
fn checkReceiverMode(t: *Typecheck, prot: Typecheck.ProtocolSym, j: usize, fn_idx: Ast.Index) !void {
    const proto = Ast.protoAt(t.tree, t.tree.nodes[fn_idx.int()].lhs.int());
    if (proto.params.len == 0 or !Ast.isSelfParam(t.tree, t.tokens, t.source, proto.params[0])) return;
    const want = prot.requiresMutSelf(j);
    const self_tok = t.tree.nodes[proto.params[0].int()].main_token;
    const has_mut = Ast.isMutParam(t.tree, t.tokens, proto.params[0]);
    if (has_mut == want) return;
    const span: Span = .{ .start = t.byteOf(if (has_mut) self_tok - 1 else self_tok), .end = t.tokens[self_tok].end };
    const name = t.nameText(t.tree.nodes[fn_idx.int()].main_token);
    if (want)
        try t.sink.report(.{ .span = span, .code = .T0076 }, "impl method '{s}' takes 'self', but protocol '{s}' declares 'mut self'; write 'mut self'", .{ name, prot.name })
    else
        try t.sink.report(.{ .span = span, .code = .T0076 }, "impl method '{s}' takes 'mut self', but protocol '{s}' declares 'self'; write 'self'", .{ name, prot.name });
}

/// `impl T has P`: the header of a conformance block, short of its methods. The decl
/// node's main token is the receiver's last segment, so walk back to `impl`.
/// An `impl .. has ..` decl: its module and its node there.
const ImplAt = struct { mod: u32, decl: Ast.Index };

/// T0020 for the impl `second`, which repeats the conformance `first` claimed. When one of
/// the two is in bundled std the program's impl is the one to remove, so the error sits on
/// it and names std's module, whichever came first in module order. The losing impl's
/// method rows and templates are dropped, so its calls neither shadow std's (W0011) nor
/// look ambiguous between two witnesses (T0025): this error is the only one the overlap
/// gives. Returns the loser.
fn reportOverlap(t: *Typecheck, pid: u32, first: ?ImplAt, second: ImplAt, detail: []const u8) !ImplAt {
    const mods = t.graph.mods;
    var loser = second;
    var std_mod: ?u32 = null;
    if (first) |f| {
        if (mods[f.mod].bundled and !mods[second.mod].bundled) std_mod = f.mod;
        if (!mods[f.mod].bundled and mods[second.mod].bundled) {
            loser = f;
            std_mod = second.mod;
        }
    }
    const prev = t.gphSelect(loser.mod);
    defer _ = t.gphSelect(prev);
    const decl = t.tree.nodes[loser.decl.int()];
    const proto_ref = Ast.implProtocol(t.tree, decl).?;
    const span = implHeaderSpan(t, decl, proto_ref);
    const pname = t.protocols.items[pid].name;
    if (std_mod) |sm|
        try t.sink.report(.{ .span = span, .code = .T0020 }, "overlapping impl of protocol '{s}' for type '{s}': {s} already provides one; remove this impl", .{ pname, t.nameText(decl.main_token), mods[sm].path })
    else
        try t.sink.report(.{ .span = span, .code = .T0020 }, "overlapping impl of protocol '{s}' for type '{s}'{s}", .{ pname, t.nameText(decl.main_token), detail });
    for (Ast.implMethods(t.tree, decl)) |pn| {
        const fid = for (t.fns.items, 0..) |f, fi| {
            if (f.mod == loser.mod and f.decl_node == pn) break fi;
        } else continue;
        var i: usize = 0;
        while (i < t.methods.items.len) {
            const m = t.methods.items[i];
            if (m.fn_id != fid) {
                i += 1;
                continue;
            }
            if (m.protocol_args.len > 0) t.gpa.free(@constCast(m.protocol_args));
            _ = t.methods.orderedRemove(i);
        }
        // A template borrows its `protocol_args` from its conformance row.
        var j: usize = 0;
        while (j < t.templates.items.len) {
            if (t.templates.items[j].fn_id == fid) _ = t.templates.orderedRemove(j) else j += 1;
        }
    }
    return loser;
}

fn implHeaderSpan(t: *const Typecheck, decl: Ast.Node, proto_ref: Ast.Index) Span {
    var tok = decl.main_token;
    while (tok > 0 and t.tokens[tok].tag != .kw_impl) tok -= 1;
    return .{ .start = t.byteOf(tok), .end = t.spanOf(proto_ref).end };
}

/// `name[..](..) -> R`: a method's signature, short of its body. Measured from the
/// tokens, not the param nodes: a synthesized `self` param's type-ref is the impl's
/// receiver, far above the method.
fn sigSpan(t: *const Typecheck, fn_idx: Ast.Index) Span {
    const name_tok = t.tree.nodes[fn_idx.int()].main_token;
    const proto = Ast.protoAt(t.tree, t.tree.nodes[fn_idx.int()].lhs.int());
    if (proto.ret_type != Ast.none) return .{ .start = t.byteOf(name_tok), .end = t.spanOf(proto.ret_type).end };
    var tok = name_tok + 1;
    var depth: u32 = 0;
    while (tok < t.tokens.len) : (tok += 1) switch (t.tokens[tok].tag) {
        .l_paren, .l_bracket => depth += 1,
        .r_paren, .r_bracket => {
            depth -|= 1;
            if (depth == 0 and t.tokens[tok].tag == .r_paren) break;
        },
        .l_brace, .eof => {
            tok -= 1;
            break;
        },
        else => {},
    };
    return .{ .start = t.byteOf(name_tok), .end = t.tokens[@min(tok, t.tokens.len - 1)].end };
}

/// Whether `conf` already records a `(pid, recv)` conformance (any protocol_args).
fn conformanceExists(conf: []const Conformance, pid: u32, recv: Type) bool {
    for (conf) |c| if (c.protocol == pid and Type.eql(c.recv, recv)) return true;
    return false;
}

/// Whether `types` already contains `recv` (by `Type.eql`).
fn containsType(types: []const Type, recv: Type) bool {
    for (types) |ty| if (Type.eql(ty, recv)) return true;
    return false;
}

/// The receivers that need an `(Eq, recv)` refinement: each `Ord` receiver in
/// insertion order, deduped, skipping any that already carries an explicit/prelude
/// `(Eq, recv)`. PURE (no `Typecheck` state) so the exactly-one-entry invariant is
/// unit-testable directly on literal conformance lists — the reason we don't widen
/// `GraphResult` to expose the conformance table.
fn ordEqRefinementReceivers(gpa: std.mem.Allocator, conf: []const Conformance, ord_pid: u32, eq_pid: u32, out: *std.ArrayList(Type)) !void {
    for (conf) |c| {
        if (c.protocol != ord_pid) continue; // `Ord` is homogeneous — protocol_args empty
        if (conformanceExists(conf, eq_pid, c.recv)) continue; // explicit/prelude Eq wins
        if (containsType(out.items, c.recv)) continue; // one refinement per receiver
        try out.append(gpa, c.recv);
    }
}

/// Ord-refines-Eq: append exactly one `(Eq, recv)` conformance per `Ord` receiver
/// lacking an existing `Eq` entry. Scans a STABLE prefix of `t.conformances` then appends,
/// so a freshly-appended refinement never seeds another (idempotent, insertion-ordered).
pub fn deriveEqFromOrd(t: *Typecheck) !void {
    const pre = t.prelude orelse return;
    const ord_pid = pre.protocols.ord orelse return;
    const eq_pid = pre.protocols.eq orelse return;
    const prefix = t.conformances.items.len;
    var add: std.ArrayList(Type) = .empty;
    defer add.deinit(t.gpa);
    try ordEqRefinementReceivers(t.gpa, t.conformances.items[0..prefix], ord_pid, eq_pid, &add);
    for (add.items) |recv| try t.conformances.append(t.gpa, .{ .protocol = eq_pid, .recv = recv });

    // The same refinement for a generic `impl Box[T] has Ord`, with the impl's bounds.
    const gprefix = t.template_conformances.items.len;
    for (0..gprefix) |i| {
        const row = t.template_conformances.items[i];
        if (row.protocol_id != ord_pid) continue;
        const has_eq = for (t.template_conformances.items) |o| {
            if (o.protocol_id == eq_pid and o.recv_ctor == row.recv_ctor and o.recv_is_enum == row.recv_is_enum) break true;
        } else false;
        if (has_eq) continue;
        var eq_row = row;
        eq_row.protocol_id = eq_pid;
        eq_row.protocol_args = &.{};
        eq_row.refines = ord_pid;
        try t.template_conformances.append(t.gpa, eq_row);
    }
}

/// Append a little-endian u32 to a byte buffer (coherence-key serialization helper).
fn appendKeyU32(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try buf.appendSlice(gpa, b[0..]);
}

/// Append one `Type`'s identity (kind byte + nominal id + the int descriptor) to a byte
/// buffer. The per-kind id and the int-only descriptor rule both route through
/// `Type.nominalId`/`Type.carriesIntDesc`, so this key can never drift from the flat
/// `appendKeyBytes`/`Mono` serializers the way it once did (the dropped-int_desc bug).
/// `pub` so the cross-serializer keystone test in `tests/driver.zig` can pin the parity.
pub fn appendKeyType(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), ty: Type) !void {
    try buf.append(gpa, @intFromEnum(ty.kind));
    try appendKeyU32(gpa, buf, ty.nominalId());
    if (Type.carriesIntDesc(ty.kind)) try buf.append(gpa, @as(u8, @bitCast(ty.int_desc)));
}

/// Serialize a `(protocol, receiver-type, protocol-args)` conformance key into `buf`.
/// Variable arity forces a byte key (an `AutoHashMap` struct can't hold a slice),
/// so `checkCoherence`'s `seen` is a `StringHashMap` over these bytes — the SAME
/// serialized-key discipline `Mono.writeKey` uses for instances. Folding the protocol
/// args means `impl P has Into[int]` and `impl P has Into[bool]` DON'T collide (distinct
/// bytes); likewise distinct int widths/signs (`int8` vs `uint8`) via `appendKeyType`'s
/// int_desc byte, but two identical `Into[int]` still do (T0020). Internal to `checkCoherence`
/// (never persisted / fingerprinted), so the layout is free to change.
fn writeCoherenceKey(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), pid: u32, recv: Type, protocol_args: []const Type) !void {
    buf.clearRetainingCapacity();
    try appendKeyU32(gpa, buf, pid);
    try appendKeyType(gpa, buf, recv);
    try appendKeyU32(gpa, buf, @intCast(protocol_args.len));
    for (protocol_args) |a| try appendKeyType(gpa, buf, a);
}

test "ordEqRefinementReceivers registers exactly one (Eq,T) per Ord recv; explicit Eq wins" {
    const gpa = testing.allocator;
    const P = Type.structT(0);
    const Q = Type.structT(1);
    const ord_pid: u32 = 1;
    const eq_pid: u32 = 0;

    // [(Ord,P)] -> exactly one refinement (P).
    {
        const conf = [_]Conformance{.{ .protocol = ord_pid, .recv = P }};
        var out: std.ArrayList(Type) = .empty;
        defer out.deinit(gpa);
        try ordEqRefinementReceivers(gpa, &conf, ord_pid, eq_pid, &out);
        try testing.expectEqual(@as(usize, 1), out.items.len);
        try testing.expect(Type.eql(out.items[0], P));
    }
    // [(Ord,P),(Eq,P)] -> zero (an explicit/prelude Eq is authoritative).
    {
        const conf = [_]Conformance{ .{ .protocol = ord_pid, .recv = P }, .{ .protocol = eq_pid, .recv = P } };
        var out: std.ArrayList(Type) = .empty;
        defer out.deinit(gpa);
        try ordEqRefinementReceivers(gpa, &conf, ord_pid, eq_pid, &out);
        try testing.expectEqual(@as(usize, 0), out.items.len);
    }
    // A duplicate (Ord,P) still refines exactly once (deduped per receiver).
    {
        const conf = [_]Conformance{ .{ .protocol = ord_pid, .recv = P }, .{ .protocol = ord_pid, .recv = P } };
        var out: std.ArrayList(Type) = .empty;
        defer out.deinit(gpa);
        try ordEqRefinementReceivers(gpa, &conf, ord_pid, eq_pid, &out);
        try testing.expectEqual(@as(usize, 1), out.items.len);
    }
    // Two distinct Ord receivers -> two refinements, in insertion order (deterministic).
    {
        const conf = [_]Conformance{ .{ .protocol = ord_pid, .recv = P }, .{ .protocol = ord_pid, .recv = Q } };
        var out: std.ArrayList(Type) = .empty;
        defer out.deinit(gpa);
        try ordEqRefinementReceivers(gpa, &conf, ord_pid, eq_pid, &out);
        try testing.expectEqual(@as(usize, 2), out.items.len);
        try testing.expect(Type.eql(out.items[0], P));
        try testing.expect(Type.eql(out.items[1], Q));
    }
}

test "coherence diagnostics span the impl or signature at fault" {
    try Typecheck.expectSpans(&.{
        .{ .src = "protocol Show { fn show(self) -> int }\nstruct A { x: int }\nimpl A has Show { fn other(self) -> int { return 1 } }\n", .key = "missing method", .want = "impl A has Show" },
        .{ .src = "protocol Show { fn show(self) -> int }\nstruct A { x: int }\nimpl A has Show { fn show(self, _y: int) -> bool { return true } }\n", .key = "incompatible with protocol", .want = "show(self, _y: int) -> bool" },
        .{ .src = "struct A { x: int }\nimpl A has Nope[int] { fn f(self) -> int { return 1 } }\n", .key = "is not a declared protocol", .want = "Nope" },
        .{ .src = "protocol Conv[U] { fn conv(self) -> U }\nstruct A { x: int }\nimpl A has Conv[int, bool] { fn conv(self) -> int { return 1 } }\n", .key = "protocol 'Conv' expects", .want = "Conv[int, bool]" },
        .{ .src = "struct A { x: int }\nimpl A has Eq { fn eq(mut self, o: A) -> bool { self.x == o.x } }\n", .key = "declares 'self'", .want = "mut self" },
        .{ .src = "protocol Step { fn step(mut self) -> int }\nstruct A { x: int }\nimpl A has Step { fn step(self) -> int { self.x } }\n", .key = "declares 'mut self'", .want = "self" },
        .{ .src = "protocol Step[U] { fn step(mut self) -> U }\nstruct B[T] { x: int }\nimpl B[T] has Step[int] { fn step(self) -> int { self.x } }\n", .key = "declares 'mut self'", .want = "self" },
    });
}

test "a conformance method's receiver mode must be its protocol's (T0076)" {
    const Case = struct { src: []const u8, t0076: usize };
    for ([_]Case{
        // The prelude protocols declare `self`.
        .{ .src = "struct A { x: int }\nimpl A has Eq { fn eq(mut self, o: A) -> bool { self.x == o.x } }\nimpl A has Display { fn display(mut self) -> str { \"a\" } }\n", .t0076 = 2 },
        .{ .src = "struct A { x: int }\nimpl A has Eq { fn eq(self, o: A) -> bool { self.x == o.x } }\n", .t0076 = 0 },
        // A user protocol's `mut self`, on a plain and a generic receiver.
        .{ .src = "protocol Step { fn step(mut self) -> int }\nstruct A { x: int }\nimpl A has Step { fn step(mut self) -> int { self.x } }\n", .t0076 = 0 },
        .{ .src = "protocol Step { fn step(mut self) -> int }\nstruct A { x: int }\nimpl A has Step { fn step(self) -> int { self.x } }\n", .t0076 = 1 },
        .{ .src = "protocol Step { fn step(mut self) -> int }\nstruct B[T] { x: int }\nimpl B[T] has Step { fn step(mut self) -> int { self.x } }\n", .t0076 = 0 },
        .{ .src = "protocol Step { fn step(self) -> int }\nstruct B[T] { x: int }\nimpl B[T] has Step { fn step(mut self) -> int { self.x } }\n", .t0076 = 1 },
    }) |c| try Typecheck.expectCodeCount(c.src, .T0076, c.t0076);
}

test "generic conformances whose protocol args meet at one instance overlap (T0020)" {
    const Case = struct { src: []const u8, t0020: usize };
    const conv = "protocol Conv[X] { fn conv(self) -> X }\n";
    const conv2 = "protocol Conv2[X, Y] { fn conv2(self) -> int }\n";
    for ([_]Case{
        .{ .src = conv ++ "struct Box[T] { v: T }\nimpl Box[T] has Conv[T] { fn conv(self) -> T { self.v } }\nimpl Box[T] has Conv[int] { fn conv(self) -> int { 99 } }\n", .t0020 = 1 },
        .{ .src = conv ++ "struct Box[T] { v: T }\nimpl Box[T] has Conv[int] { fn conv(self) -> int { 99 } }\nimpl Box[T] has Conv[T] { fn conv(self) -> T { self.v } }\n", .t0020 = 1 },
        .{ .src = conv ++ "struct Box[T] { v: T }\nimpl Box[T] has Conv[int] { fn conv(self) -> int { 99 } }\nimpl Box[T] has Conv[bool] { fn conv(self) -> bool { true } }\n", .t0020 = 0 },
        .{ .src = conv ++ "struct Pair[A, B] { a: A, b: B }\nimpl Pair[A, B] has Conv[A] { fn conv(self) -> A { self.a } }\nimpl Pair[A, B] has Conv[B] { fn conv(self) -> B { self.b } }\n", .t0020 = 1 },
        .{ .src = conv2 ++ "struct Pair[A, B] { a: A, b: B }\nimpl Pair[A, B] has Conv2[A, int] { fn conv2(self) -> int { 1 } }\nimpl Pair[A, B] has Conv2[bool, B] { fn conv2(self) -> int { 2 } }\n", .t0020 = 1 },
        .{ .src = conv2 ++ "struct Pair[A, B] { a: A, b: B }\nimpl Pair[A, B] has Conv2[A, int] { fn conv2(self) -> int { 1 } }\nimpl Pair[A, B] has Conv2[A, bool] { fn conv2(self) -> int { 2 } }\n", .t0020 = 0 },
        .{ .src = conv2 ++ "struct Box[T] { v: T }\nimpl Box[T] has Conv2[Option[T], T] { fn conv2(self) -> int { 1 } }\nimpl Box[T] has Conv2[Option[int], int] { fn conv2(self) -> int { 2 } }\n", .t0020 = 1 },
        // `T` binds to `bool` inside `Option`, so the second arg asks `int` == `bool`.
        .{ .src = conv2 ++ "struct Box[T] { v: T }\nimpl Box[T] has Conv2[Option[T], int] { fn conv2(self) -> int { 1 } }\nimpl Box[T] has Conv2[Option[bool], T] { fn conv2(self) -> int { 2 } }\n", .t0020 = 0 },
        .{ .src = conv ++ "struct Box[T] { v: T }\nimpl Box[T] has Conv[Option[T]] { fn conv(self) -> Option[T] { .some(self.v) } }\nimpl Box[T] has Conv[Result[T, int]] { fn conv(self) -> Result[T, int] { .ok(self.v) } }\n", .t0020 = 0 },
        // `T` against `Option[T]` would need an infinite type, so no instance has both.
        .{ .src = conv ++ "struct Box[T] { v: T }\nimpl Box[T] has Conv[T] { fn conv(self) -> T { self.v } }\nimpl Box[T] has Conv[Option[T]] { fn conv(self) -> Option[T] { .some(self.v) } }\n", .t0020 = 0 },
    }) |c| try Typecheck.expectCodeCount(c.src, .T0020, c.t0020);
}
