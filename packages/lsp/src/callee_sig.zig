//! The signature of the method a `recv.m(..)` call selected, read from the checker's
//! Callee table. Hover, definition and signature help all go through `of`, so the three
//! features agree on which method a call invokes.
//!
//! KNOWN LIMITATIONS:
//!   * No selection is recorded, so `of` is null, for a method call inside an unbounded
//!     generic fn or impl body (`self.get()` in `impl Box[T]`): such a body is checked only
//!     per instance, never on its own.
//!   * The same holds inside a bounded template for a call whose method row exists only
//!     once instances are made (a generic type's method such as `xs.len()` with
//!     `xs: Vec[int]`, an associated fn, a derived `hash`): the template's receiver types
//!     are scratch, so its pick is never resolved.
//!   * A generic protocol's method on a bounded receiver keeps the protocol's own params
//!     (`U`, not `int`, for `T has Scale[int]`): the Callee names the protocol method, not
//!     the bound's arguments.

const std = @import("std");
const toyc = @import("toy_compiler");

const Ast = toyc.Ast;
const Callee = toyc.Callee;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Type = toyc.Typecheck.Type;

pub const CalleeSig = struct {
    /// The member as spelled at the call site.
    name: []const u8,
    /// The receiver comes first iff `has_self`, so `has_self` implies a non-empty `params`.
    params: []const Type,
    /// May be shorter than `params` (a source-less method has no names to show).
    param_names: []const []const u8,
    ret: Type,
    has_self: bool,
    /// Bit i: `params[i]` is `mut` (`Modes`).
    mut_mask: u64 = 0,
    /// The names of the `type_var`s in `params`/`ret`: a protocol method is written over
    /// `Self` and the protocol's own generic params.
    var_names: []const []const u8 = &.{},
    /// The source `fn_decl` the call reaches, when one exists.
    decl: ?Decl = null,

    pub const Decl = struct { module: u32, node: u32 };
};

/// The selected method of the call whose member node is `fa` in module `mod`, or null
/// when `fa` records no selection (a field read, a variant, a `mod.f`, a failed dispatch).
/// Borrowed type/name slices point into the compiler results; `a` owns only what is built.
pub fn of(
    a: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const TypecheckGraph.GraphResult,
    mod: u32,
    fa: u32,
) !?CalleeSig {
    var sig = (try select(a, graph, res, tc, mod, fa)) orelse return null;
    // Callers slice the receiver off; a broken buffer must not make that slice panic.
    sig.has_self = sig.has_self and sig.params.len > 0;
    return sig;
}

fn select(
    a: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const TypecheckGraph.GraphResult,
    mod: u32,
    fa: u32,
) !?CalleeSig {
    if (mod >= tc.callees.len or fa >= tc.callees[mod].len) return null;
    const m = &graph.modules[mod];
    const fa_node = m.nodes[fa];
    const name = m.tokens[fa_node.main_token].text(m.source);
    switch (tc.callees[mod][fa].view()) {
        // A generic method whose call did not check (`xs.map(` mid-typing) picked no
        // instance; its template still names what the user is calling.
        .none => {
            const gid = tc.callees[mod][fa].templateFn() orelse return null;
            if (gid >= res.fns.len or gid >= tc.sigs.len) return null;
            const gf = res.fns[gid];
            if (gf.decl_node == Ast.none) return null;
            const sig = tc.sigs[gid];
            const names = try declParamNames(a, graph, gf.module, gf.decl_node);
            const has_self = names.len > 0 and std.mem.eql(u8, names[0], "self");
            return .{ .name = name, .params = sig.params, .param_names = names, .ret = sig.ret, .has_self = has_self, .mut_mask = sig.mut_mask, .decl = .{ .module = gf.module, .node = gf.decl_node.int() } };
        },
        // A function value's signature is its type, not a declaration's.
        .value, .fn_ref, .bound_method => return null,
        .method => |i| {
            if (i >= tc.methods.len) return null;
            const meth = tc.methods[i];
            // A derive's `fn_id` is 0, so it must be read before any fn-table lookup.
            if (meth.derive) |di| {
                if (di >= tc.derives.len) return null;
                const d = tc.derives[di];
                return .{ .name = name, .params = d.params, .param_names = selfNames(d.params.len), .ret = d.ret, .has_self = true };
            }
            if (meth.fn_id >= res.fns.len) return null;
            const gf = res.fns[meth.fn_id];
            const names = try declParamNames(a, graph, gf.module, gf.decl_node);
            const decl: ?CalleeSig.Decl = if (gf.decl_node == Ast.none) null else .{ .module = gf.module, .node = gf.decl_node.int() };
            if (meth.fn_id >= tc.sigs.len) return null;
            const sig = tc.sigs[meth.fn_id];
            // A generic template's own sig is over type_vars; the instance is concrete. A
            // reified row's `fn_id` is its template's, whose modes hold for every instance.
            if (meth.instance) |ii| {
                if (ii >= tc.instances.len) return null;
                const inst = tc.instances[ii];
                return .{ .name = name, .params = inst.params, .param_names = names, .ret = inst.ret, .has_self = !meth.is_static, .mut_mask = sig.mut_mask, .decl = decl };
            }
            return .{ .name = name, .params = sig.params, .param_names = names, .ret = sig.ret, .has_self = !meth.is_static, .mut_mask = sig.mut_mask, .decl = decl };
        },
        .protocol => |p| {
            if (p.protocol >= tc.protocols.len) return null;
            const proto = tc.protocols[p.protocol];
            if (p.method >= proto.method_params.len or p.method >= proto.method_rets.len) return null;
            const params = proto.method_params[p.method];
            const var_names = try a.alloc([]const u8, proto.generic_params.len + 1);
            var_names[0] = "Self";
            @memcpy(var_names[1..], proto.generic_params);
            var sig: CalleeSig = .{
                .name = name,
                .params = params,
                .param_names = selfNames(params.len),
                .ret = proto.method_rets[p.method],
                .has_self = true,
                .var_names = var_names,
            };
            // A prelude protocol is registered natively and has no source decl.
            if (proto.decl_node == Ast.none) return sig;
            const pm = &graph.modules[proto.mod];
            const sig_nodes = Ast.rangeSlice(pm.tree(), pm.nodes[proto.decl_node.int()].lhs.int());
            // The protocol's method index is its position among the decl's signatures.
            if (p.method >= sig_nodes.len) return sig;
            const node = sig_nodes[p.method];
            sig.param_names = try declParamNames(a, graph, proto.mod, node);
            sig.decl = .{ .module = proto.mod, .node = node.int() };
            return sig;
        },
        .builtin => |k| {
            const nts = tc.node_types[mod];
            const recv: Type = if (fa_node.lhs != Ast.none and fa_node.lhs.int() < nts.len) nts[fa_node.lhs.int()] else .invalid;
            const call_ty = callType(m, nts, fa);
            // Kept here rather than as a compiler table: a new `Builtin` breaks this switch
            // at compile time, and the compiler grows no LSP-only API.
            const shape: struct { params: []const Type, ret: Type, names: []const []const u8 } = switch (k) {
                .scalar_eq => .{ .params = try a.dupe(Type, &.{ recv, recv }), .ret = Type.bool, .names = &.{ "self", "other" } },
                .scalar_hash => .{ .params = try a.dupe(Type, &.{recv}), .ret = Type.int, .names = &.{"self"} },
                .str_concat => .{ .params = &.{ Type.str, Type.str }, .ret = Type.str, .names = &.{ "self", "other" } },
                .str_len => .{ .params = &.{Type.str}, .ret = Type.int, .names = &.{"self"} },
                .str_byte_at => .{ .params = &.{ Type.str, Type.int }, .ret = Type.int, .names = &.{ "self", "i" } },
                .scalar_to_string, .display_to_string => .{ .params = try a.dupe(Type, &.{recv}), .ret = Type.str, .names = &.{"self"} },
                .conv_into, .conv_try_into => .{ .params = try a.dupe(Type, &.{recv}), .ret = call_ty, .names = &.{"self"} },
                .opt_is_tag0, .opt_is_tag1 => .{ .params = try a.dupe(Type, &.{recv}), .ret = Type.bool, .names = &.{"self"} },
                .opt_unwrap => .{ .params = try a.dupe(Type, &.{recv}), .ret = call_ty, .names = &.{"self"} },
                .opt_unwrap_or => .{ .params = try a.dupe(Type, &.{ recv, call_ty }), .ret = call_ty, .names = &.{ "self", "default" } },
            };
            return .{ .name = name, .params = shape.params, .param_names = shape.names, .ret = shape.ret, .has_self = true };
        },
    }
}

/// Whether the method `fa` calls is best shown as its declaration spells it: a user
/// protocol's method or a generic method's template, whose checked types are opaque once
/// typecheck is done.
pub fn spelledFromDecl(tc: *const TypecheckGraph.GraphResult, mod: u32, fa: u32) bool {
    const c = tc.callees[mod][fa];
    return c.view() == .protocol or c.templateFn() != null;
}

/// The names of a source-less method's params (a derive, a prelude protocol's method): a
/// binary one (`eq`, `cmp`) compares against `other`.
fn selfNames(n: usize) []const []const u8 {
    return if (n == 2) &.{ "self", "other" } else &.{"self"};
}

/// The type of the call whose member node is `fa`: a builtin's result depends on context
/// (`into`'s target, `unwrap`'s payload), and the checker recorded it on the call.
fn callType(m: *const Graph.Module, nts: []const Type, fa: u32) Type {
    for (m.nodes, 0..) |n, c| {
        const member = Callee.memberNode(m.tree(), n) orelse continue;
        if (member.int() == fa) return if (c < nts.len) nts[c] else .invalid;
    }
    return .invalid;
}

/// Parameter names from the `fn_decl` `decl_node` in module `mod`; empty when there is no
/// source decl.
pub fn declParamNames(a: std.mem.Allocator, graph: *const Graph.Graph, mod: u32, decl_node: Ast.Index) ![]const []const u8 {
    if (decl_node == Ast.none) return &.{};
    const tm = &graph.modules[mod];
    const proto = Ast.protoAt(tm.tree(), tm.nodes[decl_node.int()].lhs.int());
    const out = try a.alloc([]const u8, proto.params.len);
    for (proto.params, out) |p, *pname| pname.* = tm.tokens[tm.nodes[p.int()].main_token].text(tm.source);
    return out;
}

const testing = std.testing;
const Documents = @import("Documents.zig");
const Workspace = @import("Workspace.zig");
const fixture = @import("test_util.zig").method_fixture;

test "of: every recorded callee in the fixture yields a signature; a non-method field access yields null" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    var graph = try ws.discover(gpa, "file:///p/main.toy", fixture);
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);
    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, ws.io, 0);
    defer tc.deinit(gpa);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const entry = graph.entry_index;
    const m = graph.entry();
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(gpa);
    var kinds = std.EnumSet(std.meta.Tag(Callee.View)){};
    for (tc.callees[entry], 0..) |c, fa| {
        const v = c.view();
        if (v == .none) continue;
        const sig = (try of(a, &graph, &res, &tc, entry, @intCast(fa))) orelse return error.RecordedCalleeWithoutSig;
        try testing.expectEqualStrings(m.tokens[m.nodes[fa].main_token].text(m.source), sig.name);
        try testing.expect(sig.params.len >= @intFromBool(sig.has_self));
        try seen.put(gpa, sig.name, {});
        kinds.insert(std.meta.activeTag(v));
    }
    for ([_][]const u8{ "plus", "make", "get", "dbl", "concat", "eq", "unwrap", "unwrap_or", "hash", "len" }) |want| {
        if (!seen.contains(want)) {
            std.debug.print("no callee recorded for '{s}'\n", .{want});
            return error.MissingCallee;
        }
    }
    for ([_]std.meta.Tag(Callee.View){ .method, .protocol, .builtin }) |t| try testing.expect(kinds.contains(t));

    // `Kind.plus` and `Option.some` are field accesses that select no method.
    for (m.nodes, 0..) |n, i| {
        if (n.tag != .field_access) continue;
        const text = m.tokens[n.main_token].text(m.source);
        const lhs = m.nodes[n.lhs.int()];
        const recv = m.tokens[lhs.main_token].text(m.source);
        if ((std.mem.eql(u8, text, "plus") and std.mem.eql(u8, recv, "Kind")) or std.mem.eql(u8, text, "some")) {
            try testing.expect((try of(a, &graph, &res, &tc, entry, @intCast(i))) == null);
        }
    }
}
