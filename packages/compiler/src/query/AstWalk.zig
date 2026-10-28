//! The fingerprint's visitors over the canonical walk (`ast/Walk.zig`). Three consumers
//! ride the walk's one event stream and so CANNOT drift:
//!
//!   * `HashVisitor` folds the structural body fingerprint (tag bytes + leaf text
//!     + arity/optionality sentinels) — byte-identical to what `Fingerprint`
//!     folded when it owned its own walk.
//!   * `CallVisitor` records each callee's `Sig` at a `.callee` event (reading
//!     `frozen.resolutions` at the callee identifier, or the checker's selection in
//!     `frozen.nodes` for a method call), in body-walk order.
//!   * `TouchedVisitor` records each touched type's layout at a `.touch` event
//!     (reading `frozen.nodes`) plus the owning fn's param/return types at a
//!     `.type_ref` event.

const std = @import("std");
const Token = @import("../ast/Token.zig").Token;
const TokenTag = @import("../ast/Token.zig").Tag;
const Ast = @import("../ast/Ast.zig");
const Walk = @import("../ast/Walk.zig");
const Typecheck = @import("../types.zig");
const Mono = @import("../symbols/Mono.zig");
const Callee = @import("../symbols/Callee.zig").Callee;
const Spec = @import("../symbols/Spec.zig");
const Body = @import("../symbols/Body.zig");
const lower = @import("../lower.zig");
const StdNames = @import("../symbols/StdNames.zig");
const symbols_res = @import("../symbols/Resolution.zig");

pub const Sig = @import("../symbols/Sig.zig").Sig;
const Fingerprint = @import("Fingerprint.zig");
pub const TouchedType = Fingerprint.TouchedType;

/// Fixed sentinel params for the builtin scalar `eq` fold. FILE-SCOPE (not a stack
/// temporary) because `Fingerprint.fingerprint` reads each folded `Sig.params` AFTER the
/// call walk returns — a `&.{recv,recv}` local would dangle. `eq` is homogeneous, so the
/// params mirror the receiver Kind; choosing per-kind (int vs bool) is strictly more
/// forward-stable than a single kind-agnostic array at zero cost, and — once chosen —
/// must stay stable or warm caches invalidate.
const eq_params_int = [_]Typecheck.Type{ Typecheck.Type.int, Typecheck.Type.int };
const eq_params_bool = [_]Typecheck.Type{ Typecheck.Type.bool, Typecheck.Type.bool };
// The builtin scalar `hash` sentinel params: 1-ary (`hash(self) -> int`), so a single
// receiver-typed element — distinct from the 2-ary `eq` sentinel above by both count + name.
const hash_params_int = [_]Typecheck.Type{Typecheck.Type.int};
const hash_params_bool = [_]Typecheck.Type{Typecheck.Type.bool};

pub fn updateU32(h: *std.hash.Wyhash, v: u32) void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    h.update(&buf);
}

/// Length-prefixed leaf text, so `"ab"+"c"` cannot collide with `"a"+"bc"`.
pub fn updateLeaf(h: *std.hash.Wyhash, text: []const u8) void {
    updateU32(h, @intCast(text.len));
    h.update(text);
}

/// Folds the structural body fingerprint. The byte stream is identical to the
/// walk `Fingerprint` owned before this module: `enter` -> the tag byte;
/// `leaf`/`raw_leaf` -> length-prefixed text; `count` -> u32 LE; `flag` ->
/// `@intFromBool` as one byte. `touch`/`callee`/`type_ref` are IGNORED (the
/// type-ref node is already folded via its own `enter` recursion, so the hash is
/// byte-identical to today).
pub const HashVisitor = struct {
    h: *std.hash.Wyhash,

    pub fn on(self: *HashVisitor, ev: Walk.Event) void {
        switch (ev) {
            .enter => |e| self.h.update(&[_]u8{@intFromEnum(e.tag)}),
            .leaf, .raw_leaf => |t| updateLeaf(self.h, t),
            .count => |c| updateU32(self.h, c),
            .flag => |f| self.h.update(&[_]u8{@intFromBool(f)}),
            .touch, .callee, .type_ref, .operator, .try_operator => {},
        }
    }
};

/// Records each callee's identity-and-signature in body-walk order, so the
/// fingerprint's (b) component lines up positionally with this stream. It reads the
/// unit's own lowering inputs, so the identity it folds is the one lower relocates to.
pub const CallVisitor = struct {
    const Self = @This();
    gpa: std.mem.Allocator,
    frozen: *const lower.Inputs,
    out: *std.ArrayList(Sig),
    /// The OWNING fn's typecheck Sig, threaded by `walkCalls` (mirroring `TouchedVisitor`).
    /// Its `.ret` is the enclosing fn's reified return enum — the source of the target
    /// error type a widening `?` converts INTO. Defaulted `null` so the ~7 test
    /// CallVisitor constructions that don't fold a `?` widen stay unchanged; a `.try_operator`
    /// with no threaded sig folds nothing (`orelse return`).
    fn_sig: ?Sig = null,

    pub fn on(self: *Self, ev: Walk.Event) error{OutOfMemory}!void {
        switch (ev) {
            // A named fn or a method taken as a value: its adapter is the reloc target
            // and calls the fn or the method's witness, so the caller depends on that
            // identity and signature exactly as a direct call does.
            .enter => |e| {
                const fr = self.frozen;
                try self.foldDesugar(e.idx, e.tag);
                if (!fr.nodes.has(e.idx)) return;
                switch (fr.nodes.callee(e.idx).view()) {
                    .fn_ref => |gid| if (gid < fr.names.len and gid < fr.sigs.len) {
                        const nm = fr.names[gid];
                        const sig = fr.sigs[gid];
                        try self.out.append(self.gpa, .{ .kind = nm.kind, .name = nm.name, .params = sig.params, .ret = sig.ret });
                    },
                    .bound_method => |mi| if (mi < fr.methods.len) try self.foldWitness(fr.methods[mi]),
                    .none, .method, .protocol, .builtin, .value, .direct, .instance, .generic => {},
                }
            },
            .callee => |c| try self.foldCallee(c.call),
            .operator => |e| {
                // A desugaring operator on a struct/enum operand desugars to that type's
                // witness — `==`/`!=` to `Eq::eq` (or, for an Ord-refinement operand with no
                // `eq`, its `Ord::cmp`), a comparison to `Ord::cmp`, arithmetic to its
                // Add/Sub/Mul/Div method. Fold the FIRST resolving witness so the caller
                // tracks the SAME identity the desugared call's reloc targets, resolved by
                // the operator's own protocol as lower's witness sites resolve it. A scalar
                // operand (int/bool/str/unit) inlines to a machine op with NO symbol, so there
                // is nothing to fold. A pre-typecheck view (lhs out of range or an invalid
                // recv) folds nothing.
                const bn = self.frozen.tree.nodes[e.idx.int()];
                if (!self.frozen.nodes.has(bn.lhs)) return;
                const recv = self.frozen.nodes.typeOf(bn.lhs);
                if (recv.kind != .@"struct" and recv.kind != .@"enum") return;
                for (Walk.opMethods(Walk.operatorOf(self.frozen.tokens[bn.main_token].tag))) |method| {
                    const pid = Typecheck.witnessProtocolId(self.frozen.prelude_ids, method);
                    switch (Typecheck.resolveConformanceMethod(self.frozen.methods, recv, method, pid, null)) {
                        .one => |m| return self.foldWitness(m),
                        .none, .ambiguous => {},
                    }
                }
            },
            .try_operator => |e| {
                // A WIDENING Result `?` re-emits `err(e)` as `RetErr.from(e)`; fold that
                // `From` witness so the enclosing fn's codegen key depends on the resolved
                // conformance — via the SAME resolver+args lower's buildResidual uses
                // (lower.zig:3063), tracking the SAME witness the widen's reloc targets. Only
                // fires for a Result operand whose error type DIFFERS from the enclosing return's:
                // the identity case (op_err==ret_err), an Option `?`, and every non-`?` fn fold
                // NOTHING. Every index is guarded — a pre-typecheck / empty-table view (no threaded
                // sig or empty tables) folds nothing, mirroring the eq/ord/arith guards above.
                const fs = self.fn_sig orelse return;
                if (fs.ret.kind != .@"enum" or fs.ret.enum_id >= self.frozen.enum_layouts.len) return;
                const tn = self.frozen.tree.nodes[e.idx.int()];
                if (!self.frozen.nodes.has(tn.lhs)) return;
                const op = self.frozen.nodes.typeOf(tn.lhs);
                if (op.kind != .@"enum" or op.enum_id >= self.frozen.enum_layouts.len) return;
                const ol = self.frozen.enum_layouts[op.enum_id];
                const rl = self.frozen.enum_layouts[fs.ret.enum_id];
                if (ol.native_family != .result or rl.native_family != .result) return;
                if (ol.variants.len < 2 or rl.variants.len < 2) return;
                if (ol.variants[1].field_types.len < 1 or rl.variants[1].field_types.len < 1) return;
                const op_err = Typecheck.errPayload(ol);
                const ret_err = Typecheck.errPayload(rl);
                if (Typecheck.Type.eql(op_err, ret_err)) return;
                switch (Typecheck.resolveConformanceMethod(self.frozen.methods, ret_err, "from", self.frozen.prelude_ids.from, &.{op_err})) {
                    .one => |m| try self.foldWitness(m),
                    .none, .ambiguous => {},
                }
            },
            else => {},
        }
    }

    /// Fold what a sugar node calls, which no call slot names: `for`'s `iter` and `next`, a
    /// list literal's `new` and `push`, an index's element primitive instance. Lower resolves each
    /// through the same helpers, so the fold names what the code relocates to ([C2]).
    fn foldDesugar(self: *Self, idx: Ast.Index, tag: Ast.Node.Tag) error{OutOfMemory}!void {
        const fr = self.frozen;
        if (!fr.nodes.has(idx)) return;
        const n = fr.tree.nodes[idx.int()];
        switch (tag) {
            .for_in_stmt, .for_in2_stmt => {
                const calls = lower.forInCalls(fr, fr.nodes.typeOf(n.rhs)) catch return;
                try self.foldWitness(calls.iter);
                try self.foldWitness(calls.next);
            },
            .empty_list, .list_literal => for (StdNames.desugarMethods(tag)) |dm| {
                if (Typecheck.findMethod(fr.methods, fr.nodes.typeOf(idx), dm.name)) |m| try self.foldWitness(m);
            },
            .index, .type_app => if (tag == .index or (fr.resolutions.len == fr.tree.nodes.len and symbols_res.leftmostHeadIsValue(fr.tree, fr.resolutions, n.lhs))) {
                for (StdNames.ElemPrim.of(fr.nodes.callee(idx))) |prim| try self.foldElemPrim(idx, prim);
            },
            else => {},
        }
    }

    fn foldElemPrim(self: *Self, idx: Ast.Index, prim: StdNames.ElemPrim) error{OutOfMemory}!void {
        const fr = self.frozen;
        const ii = lower.elemPrimInstance(fr, prim, fr.nodes.typeOf(idx)) orelse return;
        const inst = fr.instances[ii];
        try self.out.append(self.gpa, .{ .kind = .user_fn, .name = inst.name.?, .params = inst.params, .ret = inst.ret, .mut_mask = templateMask(fr.sigs, inst) });
    }

    /// Fold call `call`'s callee identity and signature from the checker's table: the fn
    /// a direct call names, the instance (or its spec unit) a generic call runs, or the
    /// method a member call selects. A generic call with no instance here folds nothing,
    /// as does a turbofish intrinsic, which names no code.
    fn foldCallee(self: *Self, call: Ast.Index) error{OutOfMemory}!void {
        const fr = self.frozen;
        const slot = Callee.callSlot(fr.tree, fr.tree.nodes[call.int()]) orelse return;
        if (!fr.nodes.has(slot)) return;
        switch (fr.nodes.callee(slot).view()) {
            // `gid` indexes BOTH `names` (the SymName{kind,name} the reloc target carries)
            // and `sigs`. Fold the full identity so a builtin<->user_fn shadow switch flips
            // the caller's hash.
            .direct => |gid| if (gid < fr.sigs.len and gid < fr.names.len) {
                const nm = fr.names[gid];
                const sig = fr.sigs[gid];
                try self.out.append(self.gpa, .{ .kind = nm.kind, .name = nm.name, .params = sig.params, .ret = sig.ret, .mut_mask = sig.mut_mask });
            },
            .instance => |ii| if (ii < fr.instances.len) try self.foldInstance(call, ii),
            .method, .builtin => try self.foldCalleeTable(call),
            .none, .generic, .protocol, .value, .fn_ref, .bound_method => {},
        }
    }

    /// Fold call `call_idx`'s method callee from the checker's Callee table.
    fn foldCalleeTable(self: *Self, call_idx: Ast.Index) error{OutOfMemory}!void {
        const fr = self.frozen;
        const fa = Callee.memberNode(fr.tree, fr.tree.nodes[call_idx.int()]) orelse return;
        if (!fr.nodes.has(fa)) return;
        const recv_idx = fr.tree.nodes[fa.int()].lhs;
        const recv: Typecheck.Type = if (fr.nodes.has(recv_idx)) fr.nodes.typeOf(recv_idx) else .{ .kind = .invalid };
        switch (fr.nodes.callee(fa).view()) {
            // A value call's code is not known statically; `.fn_ref` and `.bound_method`
            // never sit on a call's slot (they fold at their own node's `.enter`).
            .none, .protocol, .value, .fn_ref, .bound_method, .direct, .instance, .generic => {},
            .method => |i| if (i < fr.methods.len) {
                // A specialized call relocs to its spec unit, which has the instance's sig.
                if (fr.methods[i].instance) |ii| if (ii < fr.instances.len) {
                    if (fr.closure.specs.unitAt(fr.closure.owner, call_idx)) |u| {
                        const inst = fr.instances[ii];
                        return self.out.append(self.gpa, .{ .kind = .user_fn, .name = u.name, .params = inst.params, .ret = inst.ret, .mut_mask = templateMask(fr.sigs, inst) });
                    }
                };
                try self.foldWitness(fr.methods[i]);
            },
            .builtin => |k| switch (k) {
                // `to_string` relocs to the receiver's Display witness (there is no
                // `to_string` symbol), so that witness is what callers depend on.
                .display_to_string => switch (Typecheck.resolveConformanceMethod(fr.methods, recv, "display", fr.prelude_ids.display, null)) {
                    .one => |m| try self.foldWitness(m),
                    .none, .ambiguous => {},
                },
                .scalar_eq => {
                    const params: []const Typecheck.Type = if (recv.kind == .bool) &eq_params_bool else &eq_params_int;
                    try self.out.append(self.gpa, .{ .kind = .builtin, .name = "eq", .params = params, .ret = Typecheck.Type.bool });
                },
                .scalar_hash => {
                    const params: []const Typecheck.Type = if (recv.kind == .bool) &hash_params_bool else &hash_params_int;
                    try self.out.append(self.gpa, .{ .kind = .builtin, .name = "hash", .params = params, .ret = Typecheck.Type.int });
                },
                // A `try_into` to or from `char`, or from `float`, calls a shared witness.
                .conv_try_into => {
                    const call_ty: Typecheck.Type = if (fr.nodes.has(call_idx)) fr.nodes.typeOf(call_idx) else .{ .kind = .invalid };
                    if (call_ty.kind != .@"enum" or call_ty.enum_id >= fr.enum_layouts.len) return;
                    const kind = lower.convWitnessKind(fr, recv, call_ty) orelse return;
                    const d = fr.derives[lower.convWitness(fr, kind) orelse return];
                    try self.out.append(self.gpa, .{ .kind = .user_fn, .name = d.name.?, .params = d.params, .ret = d.ret });
                },
                .str_concat, .str_len, .str_byte_at, .scalar_to_string, .conv_into, .opt_is_tag0, .opt_is_tag1, .opt_unwrap, .opt_unwrap_or => {},
            },
        }
    }

    /// Fold a resolved method witness's identity+sig. A GENERIC-type instance
    /// method carries the mono `instance` index -> fold the INSTANCE identity
    /// (the real reloc target), not the never-lowered template's `names/sigs[fn_id]`.
    fn foldWitness(self: *Self, m: Typecheck.Method) error{OutOfMemory}!void {
        // A source-less derive folds its synthetic unit, so a nested-field override
        // that flips the unit's key flips the caller's too. A partial view folds nothing.
        const fr = self.frozen;
        const sym = Typecheck.witnessSym(m, fr.derives, fr.instances, fr.names) orelse return;
        const sig = Typecheck.witnessSig(m, fr.derives, fr.instances, fr.sigs) orelse return;
        try self.out.append(self.gpa, .{ .kind = sym.kind, .name = sym.name, .params = sig.params, .ret = sig.ret, .mut_mask = sig.mut_mask });
    }

    /// Fold instance `ii`'s identity, or the spec unit's name when call `call` goes to one
    /// (the instance's sig either way), tracking the SAME `Instance` the reloc targets.
    fn foldInstance(self: *Self, call: Ast.Index, ii: u32) error{OutOfMemory}!void {
        const fr = self.frozen;
        const inst = fr.instances[ii];
        const name = if (fr.closure.specs.unitAt(fr.closure.owner, call)) |u| u.name else inst.name.?;
        try self.out.append(self.gpa, .{ .kind = .user_fn, .name = name, .params = inst.params, .ret = inst.ret, .mut_mask = templateMask(fr.sigs, inst) });
    }
};

/// An instance's modes: its template's, since modes are syntactic.
fn templateMask(sigs: []const Sig, inst: Mono.Instance) u64 {
    return if (inst.template_gid < sigs.len) sigs[inst.template_gid].mut_mask else 0;
}

/// Records each touched type's layout in body-walk order: at a `.touch` event the
/// node's type in `frozen.nodes` (when it has a slot); at a `.type_ref` event the
/// OWNING fn's sig param/return type (falling back to a bare-name re-scan when no
/// sig is threaded). Owns the layout slices it appends (free via
/// `freeTouched`).
pub const TouchedVisitor = struct {
    const Self = @This();
    gpa: std.mem.Allocator,
    frozen: *const lower.Inputs,
    fn_sig: ?Sig,
    out: *std.ArrayList(TouchedType),

    pub fn on(self: *Self, ev: Walk.Event) error{OutOfMemory}!void {
        switch (ev) {
            .touch => |t| {
                if (self.frozen.nodes.has(t.idx)) try appendTouched(self.gpa, self.frozen.layouts, self.frozen.enum_layouts, self.frozen.nodes.typeOf(t.idx), self.out);
            },
            .type_ref => |r| {
                // Prefer the typecheck-resolved sig type (carries the right
                // GLOBAL id, incl. a cross-module qualified `mod.Type`); fall
                // back to the bare-name re-resolution when no sig is threaded.
                const ty = if (self.fn_sig) |s| (if (r.is_ret)
                    s.ret
                else if (r.ordinal < s.params.len)
                    s.params[r.ordinal]
                else
                    typeRefToType(self.frozen, r.idx)) else typeRefToType(self.frozen, r.idx);
                try appendTouched(self.gpa, self.frozen.layouts, self.frozen.enum_layouts, ty, self.out);
            },
            else => {},
        }
    }
};

/// Resolve a DECLARED type-ref node (a param/return type annotation) to a Type by
/// name, using the struct/enum tables. Builtins map to their scalar/str kinds; a
/// struct/enum name resolves to its id; anything else (incl. `()`) -> unit.
/// Mirrors `lower`'s type-ref resolution so the fingerprint folds the SAME layout
/// codegen will use, independent of node_types (which Typecheck never sets here).
pub fn typeRefToType(frozen: *const lower.Inputs, type_node: Ast.Index) Typecheck.Type {
    const n = frozen.tree.nodes[type_node.int()];
    if (n.tag == .literal_unit) return Typecheck.Type.unit;
    const name = frozen.tokens[n.main_token].text(frozen.source);
    if (std.mem.eql(u8, name, "int")) return Typecheck.Type.int;
    if (std.mem.eql(u8, name, "bool")) return Typecheck.Type.bool;
    if (std.mem.eql(u8, name, "str")) return Typecheck.Type.str;
    for (frozen.layouts, 0..) |l, id| {
        if (std.mem.eql(u8, l.name, name)) return Typecheck.Type.structT(@intCast(id));
    }
    for (frozen.enum_layouts, 0..) |e, id| {
        if (std.mem.eql(u8, e.name, name)) return Typecheck.Type.enumT(@intCast(id));
    }
    return Typecheck.Type.unit;
}

/// Append a `TouchedType` for `ty`, building the struct/enum layout descriptor.
pub fn appendTouched(gpa: std.mem.Allocator, layouts: []const Typecheck.Layout, enum_layouts: []const Typecheck.EnumLayout, ty: Typecheck.Type, out: *std.ArrayList(TouchedType)) !void {
    if (ty.kind == .@"struct") {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(gpa);
        try structLayoutBytes(gpa, layouts, ty.struct_id, &buf);
        try out.append(gpa, .{ .kind = .@"struct", .layout = try buf.toOwnedSlice(gpa) });
        return;
    }
    if (ty.kind == .@"enum") {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(gpa);
        try enumLayoutBytes(gpa, layouts, enum_layouts, ty.enum_id, &buf);
        try out.append(gpa, .{ .kind = .@"enum", .layout = try buf.toOwnedSlice(gpa) });
        return;
    }
    // Only an `.int` carries a meaningful descriptor; every other kind folds the stable
    // default byte (single-sourced via `carriesIntDesc`, so this can't drift from the
    // other identity serializers' int-only rule).
    const int_desc: u8 = if (Typecheck.Type.carriesIntDesc(ty.kind))
        @bitCast(ty.int_desc)
    else
        @bitCast(Typecheck.IntDesc{});
    try out.append(gpa, .{ .kind = ty.kind, .int_desc = int_desc });
}

/// Collect the signatures of every function `idx` calls, in body walk order (the
/// order `Fingerprint.fingerprint`'s (b) component expects) by riding the one AST
/// walk here — so there is no second hand-mirrored walk to drift from the fold.
///
/// `fn_sig` threads in the OWNING fn's signature so the `?`-widen fold can read the
/// enclosing fn's reified return-Result error type — the target a widening `?` converts
/// INTO via `From`. It is the fn's typecheck Sig when known; null elsewhere (a `null` sig
/// folds no `?` witness, so those fingerprints stay byte-identical).
pub fn walkCalls(gpa: std.mem.Allocator, frozen: *const lower.Inputs, idx: Ast.Index, fn_sig: ?Sig, out: *std.ArrayList(Sig)) !void {
    var v: CallVisitor = .{ .gpa = gpa, .frozen = frozen, .fn_sig = fn_sig, .out = out };
    try Walk.walk(.{ .tree = frozen.tree, .tokens = frozen.tokens, .source = frozen.source }, idx, &v);
}

/// Collect the types `idx` touches (its node types under the subtree, in walk order),
/// each as a `TouchedType` carrying — for an aggregate — an index-free layout descriptor
/// so a field-layout edit flips every using fn's hash. Feeds the fingerprint's (c)
/// component. Caller frees each `layout` slice (see `freeTouched`).
///
/// `fn_sig` threads in the OWNING fn's signature so the fn_decl case folds the
/// ABI-correct param/return types. Its `params`/`ret` carry the GLOBAL struct/enum ids the
/// typechecker resolved — including a CROSS-MODULE qualified type-ref `b: rect.Rect` which
/// a bare-name scan would otherwise mis-resolve to the FIRST same-named type in the
/// program-wide layout table. Folding the sig types makes a pub-type LAYOUT edit reach
/// EXACTLY the importers that name it.
pub fn walkTouchedSig(gpa: std.mem.Allocator, frozen: *const lower.Inputs, idx: Ast.Index, fn_sig: ?Sig, out: *std.ArrayList(TouchedType)) error{OutOfMemory}!void {
    var v: TouchedVisitor = .{ .gpa = gpa, .frozen = frozen, .fn_sig = fn_sig, .out = out };
    try Walk.walk(.{ .tree = frozen.tree, .tokens = frozen.tokens, .source = frozen.source }, idx, &v);
}

/// Free the layout slices owned by a `walkTouchedSig` result.
pub fn freeTouched(gpa: std.mem.Allocator, items: []const TouchedType) void {
    for (items) |t| if (t.layout.len > 0) gpa.free(t.layout);
}

/// Build the ordered `TouchedType` list for a monomorphized instance's concrete
/// type-args, each carrying its full index-free layout descriptor via the same
/// `appendTouched` the (c) touched fold uses. Fed into `fingerprint`'s (d) component so
/// `id[int]` and `id[Point]` diverge and a struct-layout edit to a type-arg invalidates
/// exactly the dependent instance. Empty for a non-generic fn. Caller frees via `freeTouched`.
pub fn walkTypeArgs(gpa: std.mem.Allocator, frozen: *const lower.Inputs, args: []const @import("../layout/Type.zig").Type, out: *std.ArrayList(TouchedType)) error{OutOfMemory}!void {
    for (args) |a| try appendTouched(gpa, frozen.layouts, frozen.enum_layouts, a, out);
}

/// Build the ordered `Fingerprint.ResolvedConformance` list for a monomorphized
/// instance's resolved `[T has P]` bounds, fed into `fingerprint`'s (e) component. Each
/// entry's `conform` is the conforming type's index-free layout descriptor built by the
/// SAME `appendTouched` the type-arg (d) fold uses; `protocol_name` and `witness_syms` are
/// borrowed from the `Mono.ResolvedConformance` (both outlive codegen). Empty for a
/// non-bounded instance. Caller frees via `freeConformances`.
pub fn walkConformances(gpa: std.mem.Allocator, frozen: *const lower.Inputs, conformances: []const Mono.ResolvedConformance, out: *std.ArrayList(Fingerprint.ResolvedConformance)) error{OutOfMemory}!void {
    for (conformances) |rc| {
        var tmp: std.ArrayList(TouchedType) = .empty;
        defer tmp.deinit(gpa);
        try appendTouched(gpa, frozen.layouts, frozen.enum_layouts, rc.conform_ty, &tmp);
        // The conformance's protocol type-args, each as an index-free layout descriptor
        // built by the SAME `appendTouched` -> the (e) fold distinguishes `Into[int]` from
        // `Into[bool]`. OWNED (freed via `freeConformances`).
        var pargs: std.ArrayList(TouchedType) = .empty;
        errdefer {
            freeTouched(gpa, pargs.items);
            pargs.deinit(gpa);
        }
        for (rc.protocol_args) |pa| try appendTouched(gpa, frozen.layouts, frozen.enum_layouts, pa, &pargs);
        // Reserve the out slot BEFORE detaching pargs so the append can't fail once
        // ownership of pargs_owned + tmp.items[0].layout has left the errdefer's cover.
        try out.ensureUnusedCapacity(gpa, 1);
        const pargs_owned = try pargs.toOwnedSlice(gpa);
        out.appendAssumeCapacity(.{ .protocol_name = rc.protocol_name, .conform = tmp.items[0], .witness_syms = rc.witness_syms, .protocol_args = pargs_owned });
    }
}

/// Free the conform-layout + protocol-arg-layout slices owned by a `walkConformances`
/// result (the outer list is caller-owned; `protocol_name`/`witness_syms` are borrowed).
pub fn freeConformances(gpa: std.mem.Allocator, items: []const Fingerprint.ResolvedConformance) void {
    for (items) |c| {
        if (c.conform.layout.len > 0) gpa.free(c.conform.layout);
        freeTouched(gpa, c.protocol_args);
        if (c.protocol_args.len > 0) gpa.free(@constCast(c.protocol_args));
    }
}

/// Index-free struct layout descriptor: name + per-field (name, kind, offset),
/// recursing nested structs, + size + align. Editing any of these flips the bytes.
pub fn structLayoutBytes(gpa: std.mem.Allocator, layouts: []const Typecheck.Layout, id: u32, buf: *std.ArrayList(u8)) !void {
    const l = layouts[id];
    try buf.appendSlice(gpa, l.name);
    try buf.append(gpa, 0);
    for (l.field_names, l.field_types, l.offsets) |fn_, fty, off| {
        try buf.appendSlice(gpa, fn_);
        try buf.append(gpa, 0);
        try buf.append(gpa, @intFromEnum(fty.kind));
        var ob: [4]u8 = undefined;
        std.mem.writeInt(u32, &ob, off, .little);
        try buf.appendSlice(gpa, &ob);
        if (fty.kind == .@"struct") try structLayoutBytes(gpa, layouts, fty.struct_id, buf);
    }
    var sz: [8]u8 = undefined;
    std.mem.writeInt(u32, sz[0..4], l.size, .little);
    std.mem.writeInt(u32, sz[4..8], l.@"align", .little);
    try buf.appendSlice(gpa, &sz);
}

/// Index-free enum layout descriptor: name + tag_size + payload_off + per-variant
/// (name + form byte + per payload field (name + kind + payload-local offset,
/// recursing nested struct/enum)) + size + align. Editing any variant/payload
/// flips the bytes, recompiling every using fn (cache soundness).
pub fn enumLayoutBytes(gpa: std.mem.Allocator, layouts: []const Typecheck.Layout, enum_layouts: []const Typecheck.EnumLayout, id: u32, buf: *std.ArrayList(u8)) !void {
    const e = enum_layouts[id];
    try buf.appendSlice(gpa, e.name);
    try buf.append(gpa, 0);
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], e.tag_size, .little);
    std.mem.writeInt(u32, hdr[4..8], e.payload_off, .little);
    try buf.appendSlice(gpa, &hdr);
    for (e.variants) |v| {
        try buf.appendSlice(gpa, v.name);
        try buf.append(gpa, 0);
        try buf.append(gpa, @intFromEnum(v.form));
        if (v.form == .@"struct") {
            for (v.field_names, v.field_types, v.offsets) |fn_, fty, off| {
                try buf.appendSlice(gpa, fn_);
                try buf.append(gpa, 0);
                try buf.append(gpa, @intFromEnum(fty.kind));
                var ob: [4]u8 = undefined;
                std.mem.writeInt(u32, &ob, off, .little);
                try buf.appendSlice(gpa, &ob);
                if (fty.kind == .@"struct") try structLayoutBytes(gpa, layouts, fty.struct_id, buf);
                if (fty.kind == .@"enum") try enumLayoutBytes(gpa, layouts, enum_layouts, fty.enum_id, buf);
            }
        } else {
            for (v.field_types, v.offsets) |fty, off| {
                try buf.append(gpa, @intFromEnum(fty.kind));
                var ob: [4]u8 = undefined;
                std.mem.writeInt(u32, &ob, off, .little);
                try buf.appendSlice(gpa, &ob);
                if (fty.kind == .@"struct") try structLayoutBytes(gpa, layouts, fty.struct_id, buf);
                if (fty.kind == .@"enum") try enumLayoutBytes(gpa, layouts, enum_layouts, fty.enum_id, buf);
            }
        }
    }
    var sz: [8]u8 = undefined;
    std.mem.writeInt(u32, sz[0..4], e.size, .little);
    std.mem.writeInt(u32, sz[4..8], e.@"align", .little);
    try buf.appendSlice(gpa, &sz);
}

const testing = std.testing;
const Lexer = @import("../lex.zig");
const Parser = @import("../parse.zig");

const Built = struct {
    tokens: []Token,
    tree: Ast.Tree,
    source: []const u8,

    fn deinit(self: *Built, gpa: std.mem.Allocator) void {
        gpa.free(self.tokens);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
    }

    fn fnDecl(self: *const Built, idx: usize) Ast.Index {
        const prog = self.tree.nodes[Ast.root(self.tree.nodes).int()];
        return Ast.rangeSlice(self.tree, prog.lhs.int())[idx];
    }

    fn src(self: *const Built) Walk.Source {
        return .{ .tree = self.tree, .tokens = self.tokens, .source = self.source };
    }
};

fn build(gpa: std.mem.Allocator, source: []const u8) !Built {
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, source);
    return .{ .tokens = tokens, .tree = tree, .source = source };
}

/// The position-specific events the drift guard pins, paired with the node `idx`
/// each fires at. Recording the FULL ordered stream (not just `enter`) is what
/// makes the guard catch a mis-placed `.touch`/`.callee`/`.type_ref` dispatch —
/// not only a forked traversal order.
const StreamStep = struct {
    kind: enum { enter, touch, callee, type_ref, operator, try_operator },
    idx: Ast.Index,
};

/// Records the ordered `(kind, idx)` event stream — the load-bearing thing all
/// three real consumers must agree on. Wrapped around each real consumer in the
/// drift guard so any divergence (forked order OR a moved dispatch) is a failing
/// test, not a comment. `leaf`/`count`/`flag` are pure fold bytes with no node
/// identity, so they are not recorded here (the Fingerprint hash tests pin those).
const StreamRecorder = struct {
    out: *std.ArrayList(StreamStep),
    gpa: std.mem.Allocator,
    pub fn on(self: *StreamRecorder, ev: Walk.Event) error{OutOfMemory}!void {
        switch (ev) {
            .enter => |e| try self.out.append(self.gpa, .{ .kind = .enter, .idx = e.idx }),
            .touch => |t| try self.out.append(self.gpa, .{ .kind = .touch, .idx = t.idx }),
            .callee => |c| try self.out.append(self.gpa, .{ .kind = .callee, .idx = c.idx }),
            .type_ref => |r| try self.out.append(self.gpa, .{ .kind = .type_ref, .idx = r.idx }),
            .operator => |e| try self.out.append(self.gpa, .{ .kind = .operator, .idx = e.idx }),
            .try_operator => |e| try self.out.append(self.gpa, .{ .kind = .try_operator, .idx = e.idx }),
            .leaf, .raw_leaf, .count, .flag => {},
        }
    }
};

/// Lowering inputs over a real parse with every table empty except `with`'s, so the
/// boundary tests can drive CallVisitor/TouchedVisitor. Empty resolution/type tables make
/// the consumers no-op on appends while STILL walking every node, so the enter-order parity
/// is what a test isolates.
fn fakeInputs(tree: Ast.Tree, tokens: []const Token, source: []const u8, with: FakeTables) lower.Inputs {
    return .{
        .tree = tree,
        .tokens = tokens,
        .source = source,
        .resolutions = with.resolutions,
        .nodes = .{ .types = with.node_types, .callees = with.callees },
        .layouts = with.layouts,
        .enum_layouts = with.enum_layouts,
        .names = with.names,
        .sigs = with.sigs,
        .instances = with.instances,
        .methods = with.methods,
        .derives = &.{},
        .prelude_ids = with.prelude_ids,
        .closure = with.closure,
    };
}

const FakeTables = struct {
    resolutions: []const @import("../symbols/Resolution.zig").Resolution = &.{},
    node_types: []const Typecheck.Type = &.{},
    callees: []const Callee = &.{},
    layouts: []const Typecheck.Layout = &.{},
    enum_layouts: []const Typecheck.EnumLayout = &.{},
    names: []const @import("../link/Link.zig").SymName = &.{},
    sigs: []const Sig = &.{},
    instances: []const Mono.Instance = &.{},
    methods: []const Typecheck.Method = &.{},
    prelude_ids: Typecheck.PreludeProtocolIds = .{},
    closure: lower.ClosureInputs = .none,
};

test "[DRIFT GUARD] all three consumers observe the SAME event stream + dispatch positions" {
    const gpa = testing.allocator;
    // One fn exercising every shape: nested calls, binary a-b, struct + enum init,
    // a match with guard + or-pattern + sub-pattern, a labeled loop with break @l,
    // and a struct param + struct return. The match arm patterns are the load-
    // bearing `collect=false` case (their subtrees must NOT emit `.touch`).
    var b = try build(gpa,
        \\struct P { x: int }
        \\enum E { A, C(int) }
        \\fn g(p: P) -> P { return p }
        \\fn f(p: P, e: E) -> P {
        \\  q := g(g(p))
        \\  d := q.x - p.x
        \\  d += 1
        \\  s := P { x: d }
        \\  eqp := q == s
        \\  m := match e { .A | .C(0) -> 1, .C(r) -> r }
        \\  @l loop { if m > 0 { break @l 1 } else { break @l 2 } }
        \\  return s
        \\}
    );
    defer b.deinit(gpa);
    const decl = b.fnDecl(3); // f (decls: struct P=0, enum E=1, fn g=2, fn f=3)

    // Size the frozen reads to the tree (production always has one entry per node),
    // so CallVisitor/TouchedVisitor read in-bounds. All unresolved/invalid, so no
    // appends fire — the test isolates the EVENT stream, not collection content.
    const resolutions = try gpa.alloc(@import("../symbols/Resolution.zig").Resolution, b.tree.nodes.len);
    defer gpa.free(resolutions);
    @memset(resolutions, .unresolved);
    const node_types = try gpa.alloc(Typecheck.Type, b.tree.nodes.len);
    defer gpa.free(node_types);
    @memset(node_types, Typecheck.Type{ .kind = .invalid });
    const frozen = fakeInputs(b.tree, b.tokens, b.source, .{ .resolutions = resolutions, .node_types = node_types });

    var hash_stream: std.ArrayList(StreamStep) = .empty;
    defer hash_stream.deinit(gpa);
    var call_stream: std.ArrayList(StreamStep) = .empty;
    defer call_stream.deinit(gpa);
    var touched_stream: std.ArrayList(StreamStep) = .empty;
    defer touched_stream.deinit(gpa);

    // A combined visitor: the real consumer PLUS a full-stream recorder, so the
    // exact production walk (callee/touch/type_ref dispatch included) is recorded.
    const HashProbe = struct {
        rec: StreamRecorder,
        h: HashVisitor,
        pub fn on(self: *@This(), ev: Walk.Event) error{OutOfMemory}!void {
            try self.rec.on(ev);
            self.h.on(ev);
        }
    };
    const CallProbe = struct {
        rec: StreamRecorder,
        c: CallVisitor,
        pub fn on(self: *@This(), ev: Walk.Event) error{OutOfMemory}!void {
            try self.rec.on(ev);
            try self.c.on(ev);
        }
    };
    const TouchedProbe = struct {
        rec: StreamRecorder,
        t: TouchedVisitor,
        pub fn on(self: *@This(), ev: Walk.Event) error{OutOfMemory}!void {
            try self.rec.on(ev);
            try self.t.on(ev);
        }
    };

    var h = std.hash.Wyhash.init(0);
    var hp = HashProbe{ .rec = .{ .out = &hash_stream, .gpa = gpa }, .h = .{ .h = &h } };
    try Walk.walk(b.src(), decl, &hp);

    var sigs: std.ArrayList(Sig) = .empty;
    defer sigs.deinit(gpa);
    var cp = CallProbe{ .rec = .{ .out = &call_stream, .gpa = gpa }, .c = .{ .gpa = gpa, .frozen = &frozen, .out = &sigs } };
    try Walk.walk(b.src(), decl, &cp);

    var touched: std.ArrayList(TouchedType) = .empty;
    defer touched.deinit(gpa);
    var tp = TouchedProbe{ .rec = .{ .out = &touched_stream, .gpa = gpa }, .t = .{ .gpa = gpa, .frozen = &frozen, .fn_sig = null, .out = &touched } };
    try Walk.walk(b.src(), decl, &tp);

    // The three consumers ride ONE walk, so their FULL event streams (enter +
    // touch + callee + type_ref, by kind AND idx) are identical. Comparing the
    // whole stream — not just `enter` — means a future edit that forks the order
    // OR moves a `.touch`/`.callee`/`.type_ref` dispatch for one consumer fails.
    try testing.expectEqualSlices(StreamStep, hash_stream.items, call_stream.items);
    try testing.expectEqualSlices(StreamStep, hash_stream.items, touched_stream.items);

    // Below: pin the dispatch POSITIONS so a mis-placed event is caught, not just a
    // divergence between consumers. One stream suffices (all three are equal).
    const stream = hash_stream.items;

    // Sanity: a non-trivial shape, and every kind actually fired.
    try testing.expect(stream.len > 40);
    var saw_enter = false;
    var saw_touch = false;
    var saw_callee = false;
    var saw_type_ref = false;
    // The fixture exercises all three operator categories: `q == s` (eq), `m > 0` (ord),
    // `q.x - p.x` (arith) — all now one `.operator` event, distinguished by token tag.
    var saw_eq_op = false;
    var saw_ord_op = false;
    var saw_arith_op = false;
    for (stream) |st| switch (st.kind) {
        .enter => saw_enter = true,
        .touch => saw_touch = true,
        .callee => saw_callee = true,
        .type_ref => saw_type_ref = true,
        .operator => switch (Walk.operatorOf(b.tokens[b.tree.nodes[st.idx.int()].main_token].tag)) {
            .eq_eq, .bang_eq => saw_eq_op = true,
            .lt, .lt_eq, .gt, .gt_eq => saw_ord_op = true,
            .plus, .minus, .star, .slash => saw_arith_op = true,
            else => {},
        },
        .try_operator => {}, // this fixture has no `?`, so it never fires (proved by the placement loop below being vacuous)
    };
    try testing.expect(saw_enter and saw_touch and saw_callee and saw_type_ref and saw_eq_op and saw_ord_op and saw_arith_op);

    // (1) NO ORPHAN DISPATCH: every touch/callee/type_ref idx is a node that was
    // entered. A dispatch at a node outside the walked subtree would fail here.
    for (stream) |st| {
        if (st.kind == .enter) continue;
        var entered = false;
        for (stream) |e| {
            if (e.kind == .enter and e.idx == st.idx) {
                entered = true;
                break;
            }
        }
        try testing.expect(entered);
    }

    // (2) SUPPRESSION INVARIANT (`collect=false`): a `match_arm` pattern subtree
    // must NOT emit `.touch`, even though every pattern node IS entered. This is
    // the load-bearing thing `collect` controls; a future edit that forgets to
    // turn collection off would add a pattern node to the touch set and fail.
    var pattern_entered: u32 = 0;
    for (stream) |st| {
        if (st.kind != .enter) continue;
        if (isPatternTag(b.tree.nodes[st.idx.int()].tag)) {
            pattern_entered += 1;
            // This pattern node was entered; assert it is NOT in the touch set.
            for (stream) |t| {
                if (t.kind == .touch) try testing.expect(t.idx != st.idx);
            }
        }
    }
    // The fixture's match arms have an or-pattern, a literal sub-pattern, and a
    // binding — several pattern nodes — so suppression is genuinely exercised.
    try testing.expect(pattern_entered >= 3);

    // (3) `.callee` PLACEMENT: every `.callee` idx is the `lhs` (callee subtree) of
    // some entered `.call` node. A callee recorded at the wrong node fails here.
    for (stream) |st| {
        if (st.kind != .callee) continue;
        var matched = false;
        for (stream) |e| {
            if (e.kind == .enter and b.tree.nodes[e.idx.int()].tag == .call and b.tree.nodes[e.idx.int()].lhs == st.idx) {
                matched = true;
                break;
            }
        }
        try testing.expect(matched);
    }

    // (4) `.type_ref` PLACEMENT: every `.type_ref` idx is a param/return type node
    // of the fn_decl's proto. A type_ref emitted at the wrong position fails here.
    const proto = Ast.protoAt(b.tree, b.tree.nodes[decl.int()].lhs.int());
    for (stream) |st| {
        if (st.kind != .type_ref) continue;
        var is_proto_type = st.idx == proto.ret_type;
        for (proto.params) |p| {
            if (b.tree.nodes[p.int()].lhs == st.idx) is_proto_type = true;
        }
        try testing.expect(is_proto_type);
    }

    // (5) `.operator` PLACEMENT: every `.operator` idx is an entered `.binary` or compound
    // `.assign` node whose operator has a candidate desugaring method. A signal at the wrong node — or
    // fired for a non-desugaring binary — fails here.
    var saw_compound_op = false;
    for (stream) |st| {
        if (st.kind != .operator) continue;
        const bnode = b.tree.nodes[st.idx.int()];
        try testing.expect(bnode.tag == .binary or bnode.tag == .assign);
        if (bnode.tag == .assign) saw_compound_op = true;
        try testing.expect(Walk.opMethods(Walk.operatorOf(b.tokens[bnode.main_token].tag)).len > 0);
    }
    try testing.expect(saw_compound_op);
}

/// True for the pattern-subtree tags the touched walk descends with `collect=false`
/// (so they are entered + folded by the hash, but never emit `.touch`).
fn isPatternTag(tag: Ast.Node.Tag) bool {
    return switch (tag) {
        .pattern_variant, .pattern_wildcard, .pattern_binding, .pattern_literal, .pattern_or => true,
        else => false,
    };
}

test "[LEAK GUARD] TouchedVisitor layout slices round-trip through freeTouched" {
    const gpa = testing.allocator;
    // A fn touching a struct AND an enum so appendTouched allocates layout slices;
    // testing.allocator fails the test on any leak.
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    const layouts = [_]Typecheck.Layout{.{
        .name = "P",
        .field_names = @constCast(&[_][]const u8{"x"}),
        .field_types = @constCast(&[_]Typecheck.Type{Typecheck.Type.int}),
        .offsets = @constCast(&[_]u32{0}),
        .size = 8,
        .@"align" = 8,
    }};
    const variants = [_]Typecheck.VariantLayout{.{
        .name = "A",
        .form = .unit,
        .field_names = &.{},
        .field_types = &.{},
        .offsets = &.{},
    }};
    const enum_layouts = [_]Typecheck.EnumLayout{.{
        .name = "E",
        .tag_size = 1,
        .payload_off = 8,
        .variants = @constCast(&variants),
        .size = 8,
        .@"align" = 8,
    }};

    var touched: std.ArrayList(TouchedType) = .empty;
    defer touched.deinit(gpa);
    defer freeTouched(gpa, touched.items);

    const frozen = fakeInputs(b.tree, b.tokens, b.source, .{ .layouts = &layouts, .enum_layouts = &enum_layouts });
    // A sig naming the struct param and enum return drives both layout builders
    // through the type_ref path.
    const sig = Sig{ .kind = .user_fn, .name = "f", .params = &.{Typecheck.Type.structT(0)}, .ret = Typecheck.Type.enumT(0) };
    var tv = TouchedVisitor{ .gpa = gpa, .frozen = &frozen, .fn_sig = sig, .out = &touched };
    try Walk.walk(b.src(), decl, &tv);

    // At least the struct param and enum return were collected with owned layouts.
    var saw_struct = false;
    var saw_enum = false;
    for (touched.items) |t| {
        if (t.kind == .@"struct") saw_struct = true;
        if (t.kind == .@"enum") saw_enum = true;
    }
    try testing.expect(saw_struct);
    try testing.expect(saw_enum);
}

test "[CROSS-MODULE TYPE-REF] threaded fn_sig folds the sig's type, not a bare-name first-match" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    // Two same-named structs in the layout table; the threaded sig points at the
    // SECOND. typeRefToType's bare-name scan would pick the FIRST.
    const first = Typecheck.Layout{
        .name = "Rect",
        .field_names = @constCast(&[_][]const u8{"a"}),
        .field_types = @constCast(&[_]Typecheck.Type{Typecheck.Type.int}),
        .offsets = @constCast(&[_]u32{0}),
        .size = 8,
        .@"align" = 8,
    };
    const second = Typecheck.Layout{
        .name = "Rect",
        .field_names = @constCast(&[_][]const u8{ "w", "h" }),
        .field_types = @constCast(&[_]Typecheck.Type{ Typecheck.Type.int, Typecheck.Type.int }),
        .offsets = @constCast(&[_]u32{ 0, 8 }),
        .size = 16,
        .@"align" = 8,
    };
    const layouts = [_]Typecheck.Layout{ first, second };

    // Expected bytes: the SECOND struct's layout.
    var expect_buf: std.ArrayList(u8) = .empty;
    defer expect_buf.deinit(gpa);
    const fr = fakeInputs(b.tree, b.tokens, b.source, .{ .layouts = &layouts });
    try structLayoutBytes(gpa, fr.layouts, 1, &expect_buf);

    var touched: std.ArrayList(TouchedType) = .empty;
    defer touched.deinit(gpa);
    defer for (touched.items) |t| {
        if (t.layout.len > 0) gpa.free(t.layout);
    };
    // The param's bare type-ref node names "int" (so a bare-name scan can't find
    // Rect at all); the sig says param 0 is Rect#1. Folding the sig is what reaches
    // the right cross-module type.
    const sig = Sig{ .kind = .user_fn, .name = "f", .params = &.{Typecheck.Type.structT(1)}, .ret = Typecheck.Type.int };
    var tv = TouchedVisitor{ .gpa = gpa, .frozen = &fr, .fn_sig = sig, .out = &touched };
    try Walk.walk(b.src(), decl, &tv);

    var found: ?[]const u8 = null;
    for (touched.items) |t| {
        if (t.kind == .@"struct") found = t.layout;
    }
    try testing.expect(found != null);
    try testing.expectEqualSlices(u8, expect_buf.items, found.?);
}

test "a fn_type annotation folds arity and ret presence" {
    const gpa = testing.allocator;
    const sources = [_][]const u8{
        "fn f(g: fn(int) -> int) {}\n",
        "fn f(g: fn(int)) {}\n",
        "fn f(g: fn(int, int) -> int) {}\n",
        "fn f(g: fn(int) -> bool) {}\n",
        "fn f(g: fn() -> int) {}\n",
    };
    var fps: [sources.len]u64 = undefined;
    for (sources, 0..) |src, i| {
        var b = try build(gpa, src);
        defer b.deinit(gpa);
        var h = std.hash.Wyhash.init(0);
        var hv = HashVisitor{ .h = &h };
        try Walk.walk(b.src(), b.fnDecl(0), &hv);
        fps[i] = h.final();
    }
    for (fps, 0..) |a, i| for (fps[i + 1 ..]) |c| try testing.expect(a != c);
}

test "a param's own mut changes the fn's body hash" {
    const gpa = testing.allocator;
    const pairs = [_][2][]const u8{
        .{ "fn f(x: int) { x = 1 }\n", "fn f(mut x: int) { x = 1 }\n" },
        .{ "struct P { x: int }\nimpl P { fn m(self) { } }\n", "struct P { x: int }\nimpl P { fn m(mut self) { } }\n" },
    };
    for (pairs) |pair| {
        var fps: [2]u64 = undefined;
        for (pair, &fps) |src, *out| {
            var b = try build(gpa, src);
            defer b.deinit(gpa);
            const decl = for (b.tree.nodes, 0..) |n, i| {
                if (n.tag == .fn_decl) break Ast.Index.from(@intCast(i));
            } else return error.TestExpectedFn;
            var h = std.hash.Wyhash.init(0);
            var hv = HashVisitor{ .h = &h };
            try Walk.walk(b.src(), decl, &hv);
            out.* = h.final();
        }
        try testing.expect(fps[0] != fps[1]);
    }
}

test "an assignment's operator changes the fn's body hash" {
    const gpa = testing.allocator;
    const srcs = [_][]const u8{
        "fn f(mut x: int) { x = 1 }\n",
        "fn f(mut x: int) { x += 1 }\n",
        "fn f(mut x: int) { x -= 1 }\n",
    };
    var fps: [srcs.len]u64 = undefined;
    for (srcs, &fps) |src, *out| {
        var b = try build(gpa, src);
        defer b.deinit(gpa);
        const decl = for (b.tree.nodes, 0..) |n, i| {
            if (n.tag == .fn_decl) break Ast.Index.from(@intCast(i));
        } else return error.TestExpectedFn;
        var h = std.hash.Wyhash.init(0);
        var hv = HashVisitor{ .h = &h };
        try Walk.walk(b.src(), decl, &hv);
        out.* = h.final();
    }
    try testing.expect(fps[0] != fps[1] and fps[1] != fps[2] and fps[0] != fps[2]);
}

test "a lambda folds param-type and ret presence" {
    const gpa = testing.allocator;
    const sources = [_][]const u8{
        "fn f() { g := fn(x) { x } }\n",
        "fn f() { g := fn(x: int) { x } }\n",
        "fn f() { g := fn(x: int) -> int { x } }\n",
        "fn f() { g := fn(x, y) { x } }\n",
        "fn f() { g := fn() { x } }\n",
    };
    var fps: [sources.len]u64 = undefined;
    for (sources, 0..) |src, i| {
        var b = try build(gpa, src);
        defer b.deinit(gpa);
        var h = std.hash.Wyhash.init(0);
        var hv = HashVisitor{ .h = &h };
        try Walk.walk(b.src(), b.fnDecl(0), &hv);
        fps[i] = h.final();
    }
    for (fps, 0..) |a, i| for (fps[i + 1 ..]) |c| try testing.expect(a != c);
}

test "an owner fingerprint includes its lambda body" {
    const gpa = testing.allocator;
    const sources = [_][]const u8{
        "fn f() -> int { g := fn(x: int) -> int { x + 1 }\n return g(1) }\n",
        "fn f() -> int { g := fn(x: int) -> int { x + 2 }\n return g(1) }\n",
        "fn f() -> int { g := fn(x: int) -> int { x - 1 }\n return g(1) }\n",
    };
    var fps: [sources.len]u64 = undefined;
    for (sources, 0..) |src, i| {
        var b = try build(gpa, src);
        defer b.deinit(gpa);
        var h = std.hash.Wyhash.init(0);
        var hv = HashVisitor{ .h = &h };
        try Walk.walk(b.src(), b.fnDecl(0), &hv);
        fps[i] = h.final();
    }
    for (fps, 0..) |a, i| for (fps[i + 1 ..]) |c| try testing.expect(a != c);
}

test "a lambda Walk.walk emits no type_ref" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(a: int) -> int { g := fn(x: int, y: bool) -> int { x }\n return g(a, true) }\n");
    defer b.deinit(gpa);
    var stream: std.ArrayList(StreamStep) = .empty;
    defer stream.deinit(gpa);
    var rec = StreamRecorder{ .out = &stream, .gpa = gpa };
    try Walk.walk(b.src(), b.fnDecl(0), &rec);
    var type_refs: usize = 0;
    var lambdas: usize = 0;
    for (stream.items) |st| {
        if (st.kind == .type_ref) type_refs += 1;
        if (st.kind == .enter and b.tree.nodes[st.idx.int()].tag == .lambda) lambdas += 1;
    }
    try testing.expectEqual(@as(usize, 1), lambdas);
    // Only the owner's own `a: int` and `-> int`.
    try testing.expectEqual(@as(usize, 2), type_refs);
}

test "a local fn folds its name, param types, ret and body into its owner" {
    const gpa = testing.allocator;
    const sources = [_][]const u8{
        "fn f() { fn g(x: int) -> int { x } }\n",
        "fn f() { fn h(x: int) -> int { x } }\n",
        "fn f() { fn g(x: bool) -> int { x } }\n",
        "fn f() { fn g(x: int) { x } }\n",
        "fn f() { fn g(x: int, y: int) -> int { x } }\n",
        "fn f() { fn g(x: int) -> int { x + 1 } }\n",
        "fn f() { g := fn(x: int) -> int { x } }\n",
    };
    var fps: [sources.len]u64 = undefined;
    for (sources, 0..) |src, i| {
        var b = try build(gpa, src);
        defer b.deinit(gpa);
        var h = std.hash.Wyhash.init(0);
        var hv = HashVisitor{ .h = &h };
        try Walk.walk(b.src(), b.fnDecl(0), &hv);
        fps[i] = h.final();
    }
    for (fps, 0..) |a, i| for (fps[i + 1 ..]) |c| try testing.expect(a != c);
}

test "a local fn Walk.walk emits no type_ref" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(a: int) -> int {\n fn g(x: int, y: bool) -> int { x }\n return g(a, true)\n}\n");
    defer b.deinit(gpa);
    var stream: std.ArrayList(StreamStep) = .empty;
    defer stream.deinit(gpa);
    var rec = StreamRecorder{ .out = &stream, .gpa = gpa };
    try Walk.walk(b.src(), b.fnDecl(0), &rec);
    var type_refs: usize = 0;
    var local_fns: usize = 0;
    for (stream.items) |st| {
        if (st.kind == .type_ref) type_refs += 1;
        if (st.kind == .enter and b.tree.nodes[st.idx.int()].tag == .local_fn) local_fns += 1;
    }
    try testing.expectEqual(@as(usize, 1), local_fns);
    // Only the owner's own `a: int` and `-> int`.
    try testing.expectEqual(@as(usize, 2), type_refs);
}

test "the .operator event is IGNORED by the hash (int == fingerprint unchanged)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(a: int, b: int) -> bool { a == b }\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    // Hash 1: the real walk, which emits `.operator` after the operands.
    var h1 = std.hash.Wyhash.init(0);
    var hv = HashVisitor{ .h = &h1 };
    try Walk.walk(b.src(), decl, &hv);

    // Hash 2: the SAME walk fed to a visitor that DROPS `.operator` before folding —
    // i.e. the hash as if the operator event had never been introduced. Byte-identity proves
    // the operator event never enters the fingerprint byte-stream, so every existing
    // int/bool/str/unit `==` fp is preserved and warm caches never churn.
    var h2 = std.hash.Wyhash.init(0);
    const Filter = struct {
        h: HashVisitor,
        pub fn on(self: *@This(), ev: Walk.Event) void {
            switch (ev) {
                .operator => {},
                else => self.h.on(ev),
            }
        }
    };
    var fv = Filter{ .h = .{ .h = &h2 } };
    try Walk.walk(b.src(), decl, &fv);

    try testing.expectEqual(h1.final(), h2.final());
}

test "the .operator event is IGNORED by the hash (int < fingerprint unchanged)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(a: int, b: int) -> bool { a < b }\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    // Hash 1: the real walk, which emits `.operator` after the operands.
    var h1 = std.hash.Wyhash.init(0);
    var hv = HashVisitor{ .h = &h1 };
    try Walk.walk(b.src(), decl, &hv);

    // Hash 2: the SAME walk with `.operator` dropped before folding — the hash as if the
    // operator event had never been introduced. Byte-identity proves the operator event never
    // enters the fingerprint byte-stream, so every existing int/str/bool comparison fp is
    // preserved and warm caches never churn.
    var h2 = std.hash.Wyhash.init(0);
    const Filter = struct {
        h: HashVisitor,
        pub fn on(self: *@This(), ev: Walk.Event) void {
            switch (ev) {
                .operator => {},
                else => self.h.on(ev),
            }
        }
    };
    var fv = Filter{ .h = .{ .h = &h2 } };
    try Walk.walk(b.src(), decl, &fv);

    try testing.expectEqual(h1.final(), h2.final());
}

test "the .operator event is IGNORED by the hash (int + fingerprint unchanged)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(a: int, b: int) -> int { a + b }\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    // Hash 1: the real walk, which emits `.operator` after the operands.
    var h1 = std.hash.Wyhash.init(0);
    var hv = HashVisitor{ .h = &h1 };
    try Walk.walk(b.src(), decl, &hv);

    // Hash 2: the SAME walk with `.operator` dropped before folding — the hash as if the
    // operator event had never been introduced. Byte-identity proves the operator event never
    // enters the fingerprint byte-stream, so every existing int arithmetic fp is preserved
    // and warm caches never churn.
    var h2 = std.hash.Wyhash.init(0);
    const Filter = struct {
        h: HashVisitor,
        pub fn on(self: *@This(), ev: Walk.Event) void {
            switch (ev) {
                .operator => {},
                else => self.h.on(ev),
            }
        }
    };
    var fv = Filter{ .h = .{ .h = &h2 } };
    try Walk.walk(b.src(), decl, &fv);

    try testing.expectEqual(h1.final(), h2.final());
}

test "`p == q` folds the SAME Eq witness Sig as `p.eq(q)` (fingerprint agreement)" {
    const gpa = testing.allocator;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const Link = @import("../link/Link.zig");

    const src =
        \\struct P { x: int }
        \\impl P has Eq { fn eq(self, o: P) -> bool { self.x == o.x } }
        \\fn use_op(p: P, q: P) -> bool { p == q }
        \\fn use_method(p: P, q: P) -> bool { p.eq(q) }
        \\
    ;
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
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
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    const names = try gpa.alloc(Link.SymName, res.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (res.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

    const frozen = fakeInputs(tree, tokens, src, .{ .resolutions = res.resolutions[0], .node_types = tc.node_types[0], .callees = tc.callees[0], .layouts = tc.layouts, .enum_layouts = tc.enum_layouts, .names = names, .sigs = tc.sigs, .instances = tc.instances, .methods = tc.methods, .prelude_ids = tc.prelude_ids });

    // The only TOP-LEVEL fn_decls are use_op then use_method (the impl's `eq` is nested
    // inside the impl_has_decl, not a program child), so they arrive in source order.
    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    var op_decl: Ast.Index = Ast.none;
    var meth_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, prog.lhs.int())) |idx| {
        if (tree.nodes[idx.int()].tag != .fn_decl) continue;
        if (op_decl == Ast.none) op_decl = idx else meth_decl = idx;
    }
    try testing.expect(op_decl != Ast.none and meth_decl != Ast.none);

    var op_sigs: std.ArrayList(Sig) = .empty;
    defer op_sigs.deinit(gpa);
    var vop = CallVisitor{ .gpa = gpa, .frozen = &frozen, .out = &op_sigs };
    try Walk.walk(.{ .tree = tree, .tokens = tokens, .source = src }, op_decl, &vop);

    var meth_sigs: std.ArrayList(Sig) = .empty;
    defer meth_sigs.deinit(gpa);
    var vm = CallVisitor{ .gpa = gpa, .frozen = &frozen, .out = &meth_sigs };
    try Walk.walk(.{ .tree = tree, .tokens = tokens, .source = src }, meth_decl, &vm);

    // Both fold exactly one witness Sig, and it is the SAME one: `p == q` desugars to the
    // same `Eq::eq` `p.eq(q)` dispatches to, so the fingerprints track one identity.
    try testing.expectEqual(@as(usize, 1), op_sigs.items.len);
    try testing.expectEqual(@as(usize, 1), meth_sigs.items.len);
    const a = op_sigs.items[0];
    const m = meth_sigs.items[0];
    try testing.expectEqual(a.kind, m.kind);
    try testing.expectEqualStrings(a.name, m.name);
    try testing.expectEqual(a.ret.kind, m.ret.kind);
    try testing.expectEqual(a.params.len, m.params.len);
}

test "`a < b` folds the SAME Ord::cmp witness Sig as `a.cmp(b)` (incremental soundness)" {
    const gpa = testing.allocator;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const Link = @import("../link/Link.zig");

    const src =
        \\struct P { x: int }
        \\impl P has Ord {
        \\ fn cmp(self, o: P) -> Ordering {
        \\  if self.x < o.x { Ordering.lt } else if self.x == o.x { Ordering.eq } else { Ordering.gt }
        \\ }
        \\}
        \\fn use_op(a: P, b: P) -> bool { a < b }
        \\fn use_method(a: P, b: P) -> Ordering { a.cmp(b) }
        \\
    ;
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
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
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    const names = try gpa.alloc(Link.SymName, res.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (res.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

    const frozen = fakeInputs(tree, tokens, src, .{ .resolutions = res.resolutions[0], .node_types = tc.node_types[0], .callees = tc.callees[0], .layouts = tc.layouts, .enum_layouts = tc.enum_layouts, .names = names, .sigs = tc.sigs, .instances = tc.instances, .methods = tc.methods, .prelude_ids = tc.prelude_ids });

    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    var op_decl: Ast.Index = Ast.none;
    var meth_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, prog.lhs.int())) |idx| {
        if (tree.nodes[idx.int()].tag != .fn_decl) continue;
        if (op_decl == Ast.none) op_decl = idx else meth_decl = idx;
    }
    try testing.expect(op_decl != Ast.none and meth_decl != Ast.none);

    var op_sigs: std.ArrayList(Sig) = .empty;
    defer op_sigs.deinit(gpa);
    var vop = CallVisitor{ .gpa = gpa, .frozen = &frozen, .out = &op_sigs };
    try Walk.walk(.{ .tree = tree, .tokens = tokens, .source = src }, op_decl, &vop);

    var meth_sigs: std.ArrayList(Sig) = .empty;
    defer meth_sigs.deinit(gpa);
    var vm = CallVisitor{ .gpa = gpa, .frozen = &frozen, .out = &meth_sigs };
    try Walk.walk(.{ .tree = tree, .tokens = tokens, .source = src }, meth_decl, &vm);

    // `a < b` folds the SAME `Ord::cmp` witness `a.cmp(b)` dispatches to, so editing the `cmp`
    // body invalidates a `<` caller. Each folds the cmp witness exactly once.
    try testing.expectEqual(@as(usize, 1), op_sigs.items.len);
    try testing.expectEqual(@as(usize, 1), meth_sigs.items.len);
    const a = op_sigs.items[0];
    const m = meth_sigs.items[0];
    try testing.expectEqual(a.kind, m.kind);
    try testing.expectEqualStrings(a.name, m.name);
    try testing.expectEqual(a.ret.kind, m.ret.kind);
    try testing.expectEqual(a.params.len, m.params.len);
}

test "`p + q` folds the SAME Add witness Sig as `p.add(q)` (incremental soundness)" {
    const gpa = testing.allocator;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const Link = @import("../link/Link.zig");

    const src =
        \\struct V2 { x: int, y: int }
        \\impl V2 has Add { fn add(self, o: V2) -> V2 { V2{ x: self.x + o.x, y: self.y + o.y } } }
        \\fn use_op(p: V2, q: V2) -> V2 { p + q }
        \\fn use_method(p: V2, q: V2) -> V2 { p.add(q) }
        \\
    ;
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
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
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    const names = try gpa.alloc(Link.SymName, res.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (res.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

    const frozen = fakeInputs(tree, tokens, src, .{ .resolutions = res.resolutions[0], .node_types = tc.node_types[0], .callees = tc.callees[0], .layouts = tc.layouts, .enum_layouts = tc.enum_layouts, .names = names, .sigs = tc.sigs, .instances = tc.instances, .methods = tc.methods, .prelude_ids = tc.prelude_ids });

    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    var op_decl: Ast.Index = Ast.none;
    var meth_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, prog.lhs.int())) |idx| {
        if (tree.nodes[idx.int()].tag != .fn_decl) continue;
        if (op_decl == Ast.none) op_decl = idx else meth_decl = idx;
    }
    try testing.expect(op_decl != Ast.none and meth_decl != Ast.none);

    var op_sigs: std.ArrayList(Sig) = .empty;
    defer op_sigs.deinit(gpa);
    var vop = CallVisitor{ .gpa = gpa, .frozen = &frozen, .out = &op_sigs };
    try Walk.walk(.{ .tree = tree, .tokens = tokens, .source = src }, op_decl, &vop);

    var meth_sigs: std.ArrayList(Sig) = .empty;
    defer meth_sigs.deinit(gpa);
    var vm = CallVisitor{ .gpa = gpa, .frozen = &frozen, .out = &meth_sigs };
    try Walk.walk(.{ .tree = tree, .tokens = tokens, .source = src }, meth_decl, &vm);

    // `p + q` folds the SAME `Add::add` witness `p.add(q)` dispatches to, so editing the `add`
    // body invalidates a `+` caller (the stale-cache-soundness guarantee). Each folds once.
    try testing.expectEqual(@as(usize, 1), op_sigs.items.len);
    try testing.expectEqual(@as(usize, 1), meth_sigs.items.len);
    const a = op_sigs.items[0];
    const m = meth_sigs.items[0];
    try testing.expectEqual(a.kind, m.kind);
    try testing.expectEqualStrings(a.name, m.name);
    try testing.expectEqual(a.ret.kind, m.ret.kind);
    try testing.expectEqual(a.params.len, m.params.len);
}

test "`p += q` folds the SAME Add witness Sig as `p + q` (incremental soundness)" {
    const gpa = testing.allocator;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const Link = @import("../link/Link.zig");

    const src =
        \\struct V2 { x: int, y: int }
        \\impl V2 has Add { fn add(self, o: V2) -> V2 { V2{ x: self.x + o.x, y: self.y + o.y } } }
        \\fn use_op(p: V2, q: V2) -> V2 { p + q }
        \\fn use_compound(mut p: V2, q: V2) { p += q }
        \\
    ;
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
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
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    const names = try gpa.alloc(Link.SymName, res.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (res.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

    const frozen = fakeInputs(tree, tokens, src, .{ .resolutions = res.resolutions[0], .node_types = tc.node_types[0], .callees = tc.callees[0], .layouts = tc.layouts, .enum_layouts = tc.enum_layouts, .names = names, .sigs = tc.sigs, .instances = tc.instances, .methods = tc.methods, .prelude_ids = tc.prelude_ids });

    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    var op_decl: Ast.Index = Ast.none;
    var compound_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, prog.lhs.int())) |idx| {
        if (tree.nodes[idx.int()].tag != .fn_decl) continue;
        if (op_decl == Ast.none) op_decl = idx else compound_decl = idx;
    }
    try testing.expect(op_decl != Ast.none and compound_decl != Ast.none);

    var op_sigs: std.ArrayList(Sig) = .empty;
    defer op_sigs.deinit(gpa);
    var vop = CallVisitor{ .gpa = gpa, .frozen = &frozen, .out = &op_sigs };
    try Walk.walk(.{ .tree = tree, .tokens = tokens, .source = src }, op_decl, &vop);

    var compound_sigs: std.ArrayList(Sig) = .empty;
    defer compound_sigs.deinit(gpa);
    var vm = CallVisitor{ .gpa = gpa, .frozen = &frozen, .out = &compound_sigs };
    try Walk.walk(.{ .tree = tree, .tokens = tokens, .source = src }, compound_decl, &vm);

    // `p += q` folds the SAME witness as `p + q`, so editing `add` invalidates it too.
    try testing.expectEqual(@as(usize, 1), op_sigs.items.len);
    try testing.expectEqual(@as(usize, 1), compound_sigs.items.len);
    const a = op_sigs.items[0];
    const m = compound_sigs.items[0];
    try testing.expectEqual(a.kind, m.kind);
    try testing.expectEqualStrings(a.name, m.name);
    try testing.expectEqual(a.ret.kind, m.ret.kind);
    try testing.expectEqual(a.params.len, m.params.len);
}

test "the .try_operator event is IGNORED by the hash (a `?` fingerprint is unchanged)" {
    const gpa = testing.allocator;
    var b = try build(gpa,
        \\fn f(r: Result[int, int]) -> Result[int, int] {
        \\  v := r?
        \\  return Result.ok(v)
        \\}
    );
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    // Hash 1: the real walk, which emits `.try_operator` after the `?` operand.
    var h1 = std.hash.Wyhash.init(0);
    var hv = HashVisitor{ .h = &h1 };
    try Walk.walk(b.src(), decl, &hv);

    // Hash 2: the SAME walk with `.try_operator` dropped before folding — the hash as if the
    // event had never been introduced. Byte-identity proves the operator event never
    // enters the fingerprint byte-stream, so every existing `?` fp (Option `?`, same-error
    // Result `?`, and this widening one) is preserved and warm caches never churn.
    var h2 = std.hash.Wyhash.init(0);
    const Filter = struct {
        h: HashVisitor,
        pub fn on(self: *@This(), ev: Walk.Event) void {
            switch (ev) {
                .try_operator => {},
                else => self.h.on(ev),
            }
        }
    };
    var fv = Filter{ .h = .{ .h = &h2 } };
    try Walk.walk(b.src(), decl, &fv);

    try testing.expectEqual(h1.final(), h2.final());
}

test "a WIDENING `?` folds the resolved `From` witness; the identity `?` folds nothing" {
    const gpa = testing.allocator;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const Link = @import("../link/Link.zig");

    const src =
        \\enum SmallErr { bad }
        \\enum BigErr { small, other }
        \\impl BigErr has From[SmallErr] { fn from(s: SmallErr) -> BigErr { BigErr.small } }
        \\fn inner() -> Result[int, SmallErr] { return Result[int, SmallErr].err(SmallErr.bad) }
        \\fn outer() -> Result[int, BigErr] {
        \\  v := inner()?
        \\  return Result.ok(v)
        \\}
        \\fn same_err(r: Result[int, BigErr]) -> Result[int, BigErr] {
        \\  v := r?
        \\  return Result.ok(v)
        \\}
        \\
    ;
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
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
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    const names = try gpa.alloc(Link.SymName, res.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (res.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

    const frozen = fakeInputs(tree, tokens, src, .{ .resolutions = res.resolutions[0], .node_types = tc.node_types[0], .callees = tc.callees[0], .layouts = tc.layouts, .enum_layouts = tc.enum_layouts, .names = names, .sigs = tc.sigs, .instances = tc.instances, .methods = tc.methods, .prelude_ids = tc.prelude_ids });

    // Look fns up by NAME (the impl + inner shift source positions), not source order.
    var outer_decl: Ast.Index = Ast.none;
    var outer_sig: ?Sig = null;
    var same_decl: Ast.Index = Ast.none;
    var same_sig: ?Sig = null;
    var inner_sig: ?Sig = null;
    for (res.fns, 0..) |gf, i| {
        // Names are module-qualified (`main.outer`); match the `.<fn>` suffix.
        if (std.mem.endsWith(u8, gf.name, ".outer")) {
            outer_decl = gf.decl_node;
            outer_sig = tc.sigs[i];
        } else if (std.mem.endsWith(u8, gf.name, ".same_err")) {
            same_decl = gf.decl_node;
            same_sig = tc.sigs[i];
        } else if (std.mem.endsWith(u8, gf.name, ".inner")) {
            inner_sig = tc.sigs[i];
        }
    }
    try testing.expect(outer_decl != Ast.none and same_decl != Ast.none and inner_sig != null);

    // The witness the widen must fold, resolved the SAME way lower's buildResidual does:
    // BigErr's `From[SmallErr]::from`. The reified error types come from the `err` variant
    // (variants[1]) of each fn's reified return-Result enum — exactly what the fold reads.
    const rl = tc.enum_layouts[outer_sig.?.ret.enum_id];
    const ol = tc.enum_layouts[inner_sig.?.ret.enum_id];
    const ret_err = Typecheck.errPayload(rl);
    const op_err = Typecheck.errPayload(ol);
    const pick = Typecheck.resolveConformanceMethod(tc.methods, ret_err, "from", tc.prelude_ids.from, &.{op_err});
    try testing.expect(pick == .one);
    const want_name = if (pick.one.instance) |ii| tc.instances[ii].name.? else names[pick.one.fn_id].name;

    // outer's `?` WIDENS (SmallErr -> BigErr): its fold stream contains the `from` witness.
    var outer_sigs: std.ArrayList(Sig) = .empty;
    defer outer_sigs.deinit(gpa);
    var vo = CallVisitor{ .gpa = gpa, .frozen = &frozen, .fn_sig = outer_sig, .out = &outer_sigs };
    try Walk.walk(.{ .tree = tree, .tokens = tokens, .source = src }, outer_decl, &vo);

    // same_err's `?` is the IDENTITY case (BigErr == BigErr): it folds NO `from` witness, so
    // its (b) callee-sig stream is byte-identical to a `?`-free fn (warm cache preserved).
    var same_sigs: std.ArrayList(Sig) = .empty;
    defer same_sigs.deinit(gpa);
    var vs = CallVisitor{ .gpa = gpa, .frozen = &frozen, .fn_sig = same_sig, .out = &same_sigs };
    try Walk.walk(.{ .tree = tree, .tokens = tokens, .source = src }, same_decl, &vs);

    var outer_has = false;
    for (outer_sigs.items) |s| {
        if (std.mem.eql(u8, s.name, want_name)) outer_has = true;
    }
    var same_has = false;
    for (same_sigs.items) |s| {
        if (std.mem.eql(u8, s.name, want_name)) same_has = true;
    }
    try testing.expect(outer_has);
    try testing.expect(!same_has);
}

test "nodeSpan covers a subtree from its first token through the brackets it closes" {
    const gpa = std.testing.allocator;
    const src_text = "fn f(a: int, b: int) -> int { return a }\nfn main() -> int {\n    x: int = f(1, [2, 0]) + 3\n    return x\n}\n";
    var b = try build(gpa, src_text);
    defer b.deinit(gpa);
    const src: Walk.Source = .{ .tree = b.tree, .tokens = b.tokens, .source = b.source };
    var saw = [_]bool{false} ** 3;
    for (b.tree.nodes, 0..) |n, i| {
        const sp = Walk.nodeSpan(src, Ast.Index.from(@intCast(i)));
        const text = src_text[sp.start..sp.end];
        switch (n.tag) {
            .call => {
                try std.testing.expectEqualStrings("f(1, [2, 0])", text);
                saw[0] = true;
            },
            .binary => {
                try std.testing.expectEqualStrings("f(1, [2, 0]) + 3", text);
                saw[1] = true;
            },
            .var_decl => {
                try std.testing.expectEqualStrings("x: int = f(1, [2, 0]) + 3", text);
                saw[2] = true;
            },
            else => {},
        }
    }
    try std.testing.expect(saw[0] and saw[1] and saw[2]);
}

test "nodeSpan reaches a closing bracket on a later line" {
    const gpa = std.testing.allocator;
    const src_text = "fn g(a: int, b: int) -> int { return a }\nfn f() -> int {\n    return g(1,\n        2,\n    ) + 3\n}\n";
    var b = try build(gpa, src_text);
    defer b.deinit(gpa);
    const src: Walk.Source = .{ .tree = b.tree, .tokens = b.tokens, .source = b.source };
    for (b.tree.nodes, 0..) |n, i| {
        if (n.tag != .binary) continue;
        const sp = Walk.nodeSpan(src, Ast.Index.from(@intCast(i)));
        try std.testing.expectEqualStrings("g(1,\n        2,\n    ) + 3", src_text[sp.start..sp.end]);
        return;
    }
    return error.NoBinary;
}

test "nodeSpan of a bodyless signature ends at the signature" {
    const gpa = std.testing.allocator;
    const src_text = "protocol One { fn one(self) -> int }\nprotocol Two {\n    fn two(self) -> int\n}\nextern fn ext(a: int) -> int\nfn main() -> int { 0 }\n";
    var b = try build(gpa, src_text);
    defer b.deinit(gpa);
    const src: Walk.Source = .{ .tree = b.tree, .tokens = b.tokens, .source = b.source };
    var got: std.ArrayList([]const u8) = .empty;
    defer got.deinit(gpa);
    for (b.tree.nodes, 0..) |n, i| {
        if (n.tag != .fn_decl and n.tag != .extern_fn_decl) continue;
        const sp = Walk.nodeSpan(src, Ast.Index.from(@intCast(i)));
        try got.append(gpa, src_text[sp.start..sp.end]);
    }
    try std.testing.expectEqual(@as(usize, 4), got.items.len);
    try std.testing.expectEqualStrings("fn one(self) -> int", got.items[0]);
    try std.testing.expectEqualStrings("fn two(self) -> int", got.items[1]);
    try std.testing.expectEqualStrings("extern fn ext(a: int) -> int", got.items[2]);
    try std.testing.expectEqualStrings("fn main() -> int { 0 }", got.items[3]);
}

test "nodeSpan of a local fn runs from its `fn` to its closing brace" {
    const gpa = std.testing.allocator;
    const src_text = "fn main() -> int {\n    fn g(n: int) -> int {\n        n\n    }\n    return g(1)\n}\n";
    var b = try build(gpa, src_text);
    defer b.deinit(gpa);
    const src: Walk.Source = .{ .tree = b.tree, .tokens = b.tokens, .source = b.source };
    for (b.tree.nodes, 0..) |n, i| {
        if (n.tag != .local_fn) continue;
        const sp = Walk.nodeSpan(src, Ast.Index.from(@intCast(i)));
        try std.testing.expectEqualStrings("fn g(n: int) -> int {\n        n\n    }", src_text[sp.start..sp.end]);
        return;
    }
    return error.NoLocalFn;
}

test "nodeSpan starts an inferred variant at its dot" {
    const gpa = std.testing.allocator;
    const src_text = "enum E { a(int), b }\nfn f() -> E { return .a(1) }\n";
    var b = try build(gpa, src_text);
    defer b.deinit(gpa);
    const src: Walk.Source = .{ .tree = b.tree, .tokens = b.tokens, .source = b.source };
    for (b.tree.nodes, 0..) |n, i| {
        if (n.tag != .enum_init_tuple) continue;
        const sp = Walk.nodeSpan(src, Ast.Index.from(@intCast(i)));
        try std.testing.expectEqualStrings(".a(1)", src_text[sp.start..sp.end]);
        return;
    }
    return error.NoVariantInit;
}

/// One module resolved and typechecked, with `names` in global fn-id order as
/// `renderGraphIr` builds it, so a `CallVisitor` folds the checker's real callees.
/// Heap-pinned because `res` and `tc` are built against `&g`.
const Checked = struct {
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const Link = @import("../link/Link.zig");

    src: []const u8,
    tokens: []Token,
    tree: Ast.Tree,
    g: Graph.Graph,
    res: ResolveGraph.GraphResult,
    tc: TypecheckGraph.GraphResult,
    names: []Link.SymName,

    fn create(gpa: std.mem.Allocator, src: []const u8) !*Checked {
        const self = try gpa.create(Checked);
        errdefer gpa.destroy(self);
        self.src = src;
        self.tokens = try Lexer.tokenize(gpa, src);
        errdefer gpa.free(self.tokens);
        self.tree = try Parser.expectTree(gpa, self.tokens, src);
        errdefer {
            gpa.free(self.tree.nodes);
            gpa.free(self.tree.extra);
        }
        self.g = try Graph.single(gpa, "main", "", src, self.tokens, self.tree.nodes, self.tree.extra, self.tree.pub_bits);
        errdefer self.g.deinit(gpa);
        self.res = try ResolveGraph.resolveGraph(gpa, &self.g);
        errdefer self.res.deinit(gpa);
        self.tc = try TypecheckGraph.checkGraph(gpa, &self.g, &self.res, null, 0);
        errdefer self.tc.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), self.tc.diags.len);
        self.names = try gpa.alloc(Link.SymName, self.res.fns.len);
        for (self.res.fns, 0..) |gf, i| {
            const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
            self.names[i] = .{ .kind = kind, .name = gf.name };
        }
        return self;
    }

    fn destroy(self: *Checked, gpa: std.mem.Allocator) void {
        gpa.free(self.names);
        self.tc.deinit(gpa);
        self.res.deinit(gpa);
        self.g.deinit(gpa);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
        gpa.free(self.tokens);
        gpa.destroy(self);
    }

    /// The member node of the only method call spelled `member`.
    fn memberOf(self: *const Checked, member: []const u8) !Ast.Index {
        for (self.tree.nodes) |n| {
            if (n.tag != .call) continue;
            const fa = Callee.memberNode(self.tree, n) orelse continue;
            if (std.mem.eql(u8, self.tokens[self.tree.nodes[fa.int()].main_token].text(self.src), member)) return fa;
        }
        return error.TestExpectedMethodCall;
    }

    /// Fold the callee Sigs of top-level fn `fn_name` against `callees`.
    fn fold(self: *const Checked, gpa: std.mem.Allocator, callees: []const Callee, fn_name: []const u8, out: *std.ArrayList(Sig)) !void {
        return self.foldWith(gpa, callees, fn_name, .none, out);
    }

    /// `fold` under the given closure tables.
    fn foldWith(self: *const Checked, gpa: std.mem.Allocator, callees: []const Callee, fn_name: []const u8, closure: lower.ClosureInputs, out: *std.ArrayList(Sig)) !void {
        const frozen = fakeInputs(self.tree, self.tokens, self.src, .{ .resolutions = self.res.resolutions[0], .node_types = self.tc.node_types[0], .callees = callees, .layouts = self.tc.layouts, .enum_layouts = self.tc.enum_layouts, .names = self.names, .sigs = self.tc.sigs, .instances = self.tc.instances, .methods = self.tc.methods, .prelude_ids = self.tc.prelude_ids, .closure = closure });
        const prog = self.tree.nodes[Ast.root(self.tree.nodes).int()];
        for (Ast.rangeSlice(self.tree, prog.lhs.int())) |idx| {
            const n = self.tree.nodes[idx.int()];
            if (n.tag != .fn_decl or !std.mem.eql(u8, self.tokens[n.main_token].text(self.src), fn_name)) continue;
            var v = CallVisitor{ .gpa = gpa, .frozen = &frozen, .out = out };
            return Walk.walk(.{ .tree = self.tree, .tokens = self.tokens, .source = self.src }, idx, &v);
        }
        return error.TestExpectedFn;
    }
};

test "method callee folds come from the Callee table" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\struct P { x: int }
        \\impl P has Eq { fn eq(self, o: P) -> bool { self.x == o.x } }
        \\fn use_method(p: P, q: P) -> bool { p.eq(q) }
        \\
    );
    defer fx.destroy(gpa);
    const mi = switch (fx.tc.callees[0][(try fx.memberOf("eq")).int()].view()) {
        .method => |i| i,
        else => return error.TestExpectedMethodCallee,
    };
    const m = fx.tc.methods[mi];

    var sigs: std.ArrayList(Sig) = .empty;
    defer sigs.deinit(gpa);
    try fx.fold(gpa, fx.tc.callees[0], "use_method", &sigs);
    try testing.expectEqual(@as(usize, 1), sigs.items.len);
    try testing.expectEqual(fx.names[m.fn_id].kind, sigs.items[0].kind);
    try testing.expectEqualStrings(fx.names[m.fn_id].name, sigs.items[0].name);
    try testing.expect(Typecheck.eqlTypeVec(fx.tc.sigs[m.fn_id].params, sigs.items[0].params));
    try testing.expect(Typecheck.Type.eql(fx.tc.sigs[m.fn_id].ret, sigs.items[0].ret));

    const blank = try gpa.alloc(Callee, fx.tc.callees[0].len);
    defer gpa.free(blank);
    @memset(blank, .{});
    sigs.clearRetainingCapacity();
    try fx.fold(gpa, blank, "use_method", &sigs);
    try testing.expectEqual(@as(usize, 0), sigs.items.len);
}

test "a user to_string method folds its own witness, not Display" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\struct P { x: int }
        \\impl P { fn to_string(self) -> str { "p" } }
        \\impl P has Display { fn display(self) -> str { "d" } }
        \\fn use_it(p: P) -> str { p.to_string() }
        \\
    );
    defer fx.destroy(gpa);
    const mi = switch (fx.tc.callees[0][(try fx.memberOf("to_string")).int()].view()) {
        .method => |i| i,
        else => return error.TestExpectedMethodCallee,
    };
    try testing.expectEqualStrings("to_string", fx.tc.methods[mi].name);

    var sigs: std.ArrayList(Sig) = .empty;
    defer sigs.deinit(gpa);
    try fx.fold(gpa, fx.tc.callees[0], "use_it", &sigs);
    try testing.expectEqual(@as(usize, 1), sigs.items.len);
    try testing.expectEqualStrings(fx.names[fx.tc.methods[mi].fn_id].name, sigs.items[0].name);
    try testing.expect(std.mem.endsWith(u8, sigs.items[0].name, ".to_string"));
}

test "builtin eq and hash sentinels keep their receiver-kind params" {
    const gpa = testing.allocator;
    const T = Typecheck.Type;
    const fx = try Checked.create(gpa,
        \\fn int_eq() -> bool { (3).eq(3) }
        \\fn bool_eq() -> bool { true.eq(false) }
        \\fn str_hash() -> int { "s".hash() }
        \\fn unit_hash() -> int { ().hash() }
        \\
    );
    defer fx.destroy(gpa);
    const cases = [_]struct { f: []const u8, name: []const u8, params: []const T, ret: T }{
        .{ .f = "int_eq", .name = "eq", .params = &.{ T.int, T.int }, .ret = T.bool },
        .{ .f = "bool_eq", .name = "eq", .params = &.{ T.bool, T.bool }, .ret = T.bool },
        .{ .f = "str_hash", .name = "hash", .params = &.{T.int}, .ret = T.int },
        .{ .f = "unit_hash", .name = "hash", .params = &.{T.int}, .ret = T.int },
    };
    for (cases) |c| {
        var sigs: std.ArrayList(Sig) = .empty;
        defer sigs.deinit(gpa);
        try fx.fold(gpa, fx.tc.callees[0], c.f, &sigs);
        try testing.expectEqual(@as(usize, 1), sigs.items.len);
        try testing.expectEqual(@import("../link/Link.zig").SymKind.builtin, sigs.items[0].kind);
        try testing.expectEqualStrings(c.name, sigs.items[0].name);
        try testing.expect(Typecheck.eqlTypeVec(c.params, sigs.items[0].params));
        try testing.expect(T.eql(c.ret, sigs.items[0].ret));
    }
}

/// The full codegen fingerprint of top-level fn `fn_name` in `fx`, with its callee Sigs
/// folded from the Callee table (no touched types, type-args or conformances).
fn fnFingerprint(fx: *const Checked, gpa: std.mem.Allocator, fn_name: []const u8) !u64 {
    var sigs: std.ArrayList(Sig) = .empty;
    defer sigs.deinit(gpa);
    try fx.fold(gpa, fx.tc.callees[0], fn_name, &sigs);
    const prog = fx.tree.nodes[Ast.root(fx.tree.nodes).int()];
    for (Ast.rangeSlice(fx.tree, prog.lhs.int())) |idx| {
        const n = fx.tree.nodes[idx.int()];
        if (n.tag != .fn_decl or !std.mem.eql(u8, fx.tokens[n.main_token].text(fx.src), fn_name)) continue;
        return Fingerprint.fingerprint(fx.tree, fx.tokens, fx.src, idx, sigs.items, &.{}, &.{}, &.{});
    }
    return error.TestExpectedFn;
}

test "a fn value reference folds the referenced fn's symbol and sig; a body change does not flip it" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\fn g(x: int) -> int { x + 1 }
        \\fn use_it() -> int {
        \\    h := g
        \\    h(2)
        \\}
        \\
    );
    defer fx.destroy(gpa);
    var sigs: std.ArrayList(Sig) = .empty;
    defer sigs.deinit(gpa);
    try fx.fold(gpa, fx.tc.callees[0], "use_it", &sigs);
    // Exactly the reference: the value call `h(2)` folds nothing (its code is dynamic).
    try testing.expectEqual(@as(usize, 1), sigs.items.len);
    try testing.expectEqualStrings("main.g", sigs.items[0].name);
    try testing.expectEqual(@as(usize, 1), sigs.items[0].params.len);
    try testing.expectEqual(Typecheck.Kind.int, sigs.items[0].ret.kind);

    // Without the table the reference folds nothing, so the fold is what the key rests on.
    const blank = try gpa.alloc(Callee, fx.tc.callees[0].len);
    defer gpa.free(blank);
    @memset(blank, .{});
    sigs.clearRetainingCapacity();
    try fx.fold(gpa, blank, "use_it", &sigs);
    try testing.expectEqual(@as(usize, 0), sigs.items.len);

    // `use_it`'s own text is fixed; only `g` changes: its body must not flip the key,
    // its signature must.
    const srcs = [_][]const u8{
        "fn g(x: int) -> int { x + 1 }\nfn use_it() -> int {\n    _h := g\n    0\n}\n",
        "fn g(x: int) -> int { x + 100 }\nfn use_it() -> int {\n    _h := g\n    0\n}\n",
        "fn g(_x: bool) -> int { 1 }\nfn use_it() -> int {\n    _h := g\n    0\n}\n",
    };
    var fps: [srcs.len]u64 = undefined;
    for (srcs, &fps) |src, *fp| {
        const c = try Checked.create(gpa, src);
        defer c.destroy(gpa);
        fp.* = try fnFingerprint(c, gpa, "use_it");
    }
    try testing.expectEqual(fps[0], fps[1]);
    try testing.expect(fps[0] != fps[2]);
}

test "a direct call folds exactly once" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\fn g(x: int) -> int { x + 1 }
        \\fn use_it() -> int { g(2) }
        \\
    );
    defer fx.destroy(gpa);
    var sigs: std.ArrayList(Sig) = .empty;
    defer sigs.deinit(gpa);
    try fx.fold(gpa, fx.tc.callees[0], "use_it", &sigs);
    try testing.expectEqual(@as(usize, 1), sigs.items.len);
    try testing.expectEqualStrings("main.g", sigs.items[0].name);
}

test "a bound method value folds its witness" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, x: int) -> int { self.n + x } }
        \\fn use_it(acc: Acc) -> int {
        \\    f := acc.plus
        \\    f(2)
        \\}
        \\
    );
    defer fx.destroy(gpa);
    var sigs: std.ArrayList(Sig) = .empty;
    defer sigs.deinit(gpa);
    try fx.fold(gpa, fx.tc.callees[0], "use_it", &sigs);
    // The value's adapter calls `Acc.plus`; the value call `f(2)` folds nothing.
    try testing.expectEqual(@as(usize, 1), sigs.items.len);
    try testing.expectEqualStrings("main.Acc.plus", sigs.items[0].name);
    try testing.expectEqual(@as(usize, 2), sigs.items[0].params.len);

    const blank = try gpa.alloc(Callee, fx.tc.callees[0].len);
    defer gpa.free(blank);
    @memset(blank, .{});
    sigs.clearRetainingCapacity();
    try fx.fold(gpa, blank, "use_it", &sigs);
    try testing.expectEqual(@as(usize, 0), sigs.items.len);

    // A body edit of the method keeps `use_it`'s key; a signature edit flips it.
    const srcs = [_][]const u8{
        "struct Acc { n: int }\nimpl Acc { fn plus(self, x: int) -> int { self.n + x } }\nfn use_it(acc: Acc) -> int {\n    _f := acc.plus\n    0\n}\n",
        "struct Acc { n: int }\nimpl Acc { fn plus(self, x: int) -> int { self.n * x } }\nfn use_it(acc: Acc) -> int {\n    _f := acc.plus\n    0\n}\n",
        "struct Acc { n: int }\nimpl Acc { fn plus(self, _x: bool) -> int { self.n } }\nfn use_it(acc: Acc) -> int {\n    _f := acc.plus\n    0\n}\n",
    };
    var fps: [srcs.len]u64 = undefined;
    for (srcs, &fps) |src, *fp| {
        const c = try Checked.create(gpa, src);
        defer c.destroy(gpa);
        fp.* = try fnFingerprint(c, gpa, "use_it");
    }
    try testing.expectEqual(fps[0], fps[1]);
    try testing.expect(fps[0] != fps[2]);
}

test "a specialized call site folds the spec unit's name" {
    const gpa = testing.allocator;
    const fx = try Checked.create(gpa,
        \\fn inc(x: int) -> int { x + 1 }
        \\fn apply[T](f: fn(T) -> T, x: T) -> T { f(x) }
        \\fn use_it() -> int { apply(inc, 1) + apply(fn(x: int) { x }, 2) }
        \\
    );
    defer fx.destroy(gpa);
    const inst_i = for (fx.tc.instances, 0..) |inst, i| {
        if (std.mem.eql(u8, inst.name.?, "main.apply$int")) break i;
    } else return error.TestExpectedInstance;
    const inst = fx.tc.instances[inst_i];
    const inc: Spec.Target = .{ .param = 0, .kind = .named, .sym = .{ .kind = .user_fn, .name = "main.inc" } };
    const units = [_]Spec.Unit{.{ .inst = @intCast(inst_i), .name = "main.apply$int$sp0$main.inc", .targets = &.{inc} }};
    const masks = [_]Spec.TemplateMask{.{ .gid = inst.template_gid, .eligible = 1 }};
    const owner = for (fx.res.fns, 0..) |gf, i| {
        if (std.mem.eql(u8, gf.name, "main.use_it")) break fx.tc.bodies.ofFn(@intCast(i)).?;
    } else return error.TestExpectedFn;
    // The first `apply` call goes to the unit; the lambda's call has none.
    const first_call = for (fx.tree.nodes, 0..) |n, i| {
        if (n.tag == .call and fx.tree.nodes[n.lhs.int()].tag == .identifier and
            std.mem.eql(u8, fx.tokens[fx.tree.nodes[n.lhs.int()].main_token].text(fx.src), "apply")) break Ast.Index.from(@intCast(i));
    } else return error.TestExpectedCall;
    const calls = [_]Spec.Call{.{ .owner = owner, .call = first_call, .unit = 0 }};
    const closure: lower.ClosureInputs = .{ .adapters = &.{}, .specs = .{ .units = &units, .masks = &masks, .calls = &calls }, .lambda_units = fx.tc.lambda_units, .owner = owner };

    var sigs: std.ArrayList(Sig) = .empty;
    defer sigs.deinit(gpa);
    try fx.foldWith(gpa, fx.tc.callees[0], "use_it", closure, &sigs);
    // The specialized call (a call folds before its args), `inc` as a value, then the
    // lambda's call: its unit is not in the table (past the cap, say), so it folds the
    // erased instance.
    try testing.expectEqual(@as(usize, 3), sigs.items.len);
    try testing.expectEqualStrings("main.apply$int$sp0$main.inc", sigs.items[0].name);
    try testing.expectEqualStrings("main.inc", sigs.items[1].name);
    try testing.expectEqualStrings("main.apply$int", sigs.items[2].name);
    // Either way the call folds the instance's signature.
    for ([_]Sig{ sigs.items[0], sigs.items[2] }) |sg| {
        try testing.expect(Typecheck.eqlTypeVec(inst.params, sg.params));
        try testing.expect(Typecheck.Type.eql(inst.ret, sg.ret));
    }

    // With no table the same call folds the erased instance's name.
    sigs.clearRetainingCapacity();
    try fx.fold(gpa, fx.tc.callees[0], "use_it", &sigs);
    try testing.expectEqualStrings("main.apply$int", sigs.items[0].name);
}
