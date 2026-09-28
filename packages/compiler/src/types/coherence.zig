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

const testing = std.testing;

/// Whole-program conformance coherence. Walks every `impl .. has ..` in
/// module-then-decl order (SERIAL, before Pass C) and enforces:
///   * exactly one impl per (protocol, receiver type-ctor) — a duplicate (INCLUDING a
///     retroactive one in a SIBLING module: no orphan rule) is T0020 at the second impl;
///   * `has P` names a declared protocol AND provides every method P requires — an
///     undeclared protocol OR a missing method is T0021.
/// Accepted conformances are recorded onto `t.conformances`. The
/// `seen` set is `getOrPut`-only (never iterated), so its hash/thread order cannot leak
/// into the emit stream; emit order is module-id then source order → `-jN`-stable.
pub fn checkCoherence(t: *Typecheck, mods: []const GraphModuleInput) !void {
    // `seen` maps a serialized coherence key (see `writeCoherenceKey`) to presence; it
    // owns its keys (dup'd on insert), freed on return.
    var seen: std.StringHashMapUnmanaged(void) = .empty;
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
        if (!gop.found_existing) gop.key_ptr.* = try t.gpa.dupe(u8, keybuf.items);
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

            // Completeness: every method the protocol requires must be provided.
            const provided = Ast.implMethods(t.tree, decl);
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
                try recordGenericConformance(t, &seen, &keybuf, mod, decl, proto_ref, pid, prot, provided);
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
                try t.sink.report(.{ .span = t.spanOf(proto_ref) }, "protocol '{s}' expects {d} type argument(s), got {d}", .{ prot.name, prot.generic_params.len, arg_nodes.len });
                continue;
            }
            var pargs_buf: std.ArrayList(Type) = .empty;
            defer pargs_buf.deinit(t.gpa);
            var pargs_ok = true;
            for (arg_nodes) |an| {
                const aty = t.typeFromNode(an);
                switch (aty.kind) {
                    .int, .bool, .str, .unit, .@"struct", .@"enum" => {},
                    .invalid => pargs_ok = false, // typeFromNode already emitted T0001
                    else => {
                        try t.sink.report(.{ .span = t.spanOf(an) }, "a generic-protocol argument must be a concrete non-composite value type (composite protocol args are not yet supported)", .{});
                        pargs_ok = false;
                    },
                }
                try pargs_buf.append(t.gpa, aty);
            }
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
                        if (!Type.eql(Typecheck.groundProtoTypeDeep(t, wp, recv, pargs), ip)) {
                            mismatch = true;
                            break;
                        }
                    }
                    if (!mismatch and !Type.eql(Typecheck.groundProtoTypeDeep(t, want_ret, recv, pargs), impl_fn.ret)) mismatch = true;
                }
                if (mismatch)
                    try t.sink.report(.{ .span = sigSpan(t, method_node), .code = .T0024 }, "impl method '{s}' has a signature incompatible with protocol '{s}'", .{ req, prot.name });
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
            if (gop.found_existing) {
                try t.sink.report(.{ .span = implHeaderSpan(t, decl, proto_ref), .code = .T0020 }, "overlapping impl of protocol '{s}' for type '{s}'", .{ t.protocols.items[pid].name, t.nameText(decl.main_token) });
            } else {
                gop.key_ptr.* = try t.gpa.dupe(u8, keybuf.items);
                try t.conformances.append(t.gpa, .{ .protocol = pid, .recv = recv, .protocol_args = try t.gpa.dupe(Type, pargs) });
            }
        }
    }
}

/// Record a GENERIC conformance `impl Ctor[T] has P[..]` into `t.template_conformances`
/// (invisible to every existing conformance query) and validate its method signatures
/// (T0024) monomorphized per the receiver's own type-params. The impl's methods live in
/// `t.templates`, so there is no `t.methods` witness to stamp (the `for x in xs` desugar
/// dispatches structurally on the reified receiver). Duplicate detection keys on the
/// receiver CTOR int (the App index is interning-order dependent — would break `-jN`).
fn recordGenericConformance(
    t: *Typecheck,
    seen: *std.StringHashMapUnmanaged(void),
    keybuf: *std.ArrayList(u8),
    mod: u32,
    decl: Ast.Node,
    proto_ref: Ast.Index,
    pid: u32,
    prot: Typecheck.ProtocolSym,
    provided: []const Ast.Index,
) !void {
    // A representative impl-method FnSym carries the impl's generic-param NAMES and the
    // receiver App (the `Self` pattern) — shared by every method, so the first that
    // resolved in Phase A suffices; none means every method errored (nothing to record).
    var rep: ?FnSym = null;
    outer: for (provided) |pn| {
        for (t.fns.items) |f| {
            if (f.mod == mod and f.decl_node == pn) {
                rep = f;
                break :outer;
            }
        }
    }
    const impl_fn0 = rep orelse return;
    if (!impl_fn0.self_type.isApp()) return;
    const recv_app = impl_fn0.self_type;
    const e = t.composite.at(recv_app.appIdx());
    const gnames = impl_fn0.generic_params;

    const arg_nodes = Ast.protocolRefArgs(t.tree, proto_ref);
    if (arg_nodes.len != prot.generic_params.len) {
        try t.sink.report(.{ .span = t.spanOf(proto_ref) }, "protocol '{s}' expects {d} type argument(s), got {d}", .{ prot.name, prot.generic_params.len, arg_nodes.len });
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
                if (!Type.eql(Typecheck.groundProtoTypeDeep(t, wp, recv_app, pargs), ip)) {
                    mismatch = true;
                    break;
                }
            }
            if (!mismatch and !Type.eql(Typecheck.groundProtoTypeDeep(t, want_ret, recv_app, pargs), impl_fn.ret)) mismatch = true;
        }
        if (mismatch)
            try t.sink.report(.{ .span = sigSpan(t, method_node), .code = .T0024 }, "impl method '{s}' has a signature incompatible with protocol '{s}'", .{ req, prot.name });
    }

    // Duplicate detection: a 'G'-tagged key so it never aliases a concrete
    // `writeCoherenceKey`; keyed on the receiver CTOR (deterministic at any `-jN`).
    keybuf.clearRetainingCapacity();
    try keybuf.append(t.gpa, 'G');
    try appendKeyU32(t.gpa, keybuf, pid);
    try appendKeyU32(t.gpa, keybuf, e.ctor);
    try keybuf.append(t.gpa, @intFromBool(e.ctor_is_enum));
    for (pargs) |a| try appendKeyType(t.gpa, keybuf, a);
    const gop = try seen.getOrPut(t.gpa, keybuf.items);
    if (gop.found_existing) {
        try t.sink.report(.{ .span = implHeaderSpan(t, decl, proto_ref), .code = .T0020 }, "overlapping impl of protocol '{s}' for type '{s}'", .{ prot.name, t.nameText(decl.main_token) });
        return;
    }
    gop.key_ptr.* = try t.gpa.dupe(u8, keybuf.items);
    try t.template_conformances.append(t.gpa, .{ .protocol_id = pid, .recv_ctor = e.ctor, .recv_is_enum = e.ctor_is_enum, .protocol_args = try t.gpa.dupe(Type, pargs) });
}

/// `impl T has P`: the header of a conformance block, short of its methods. The decl
/// node's main token is the receiver's last segment, so walk back to `impl`.
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
