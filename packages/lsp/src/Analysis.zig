//! One check of one document version, which every feature renders from: hover,
//! definition, signature help, completion, inlay hints and diagnostics read it instead of
//! each running discover -> resolve -> typecheck, so an edit costs one check. The server
//! keeps the latest analysis per document and drops it when that document, or one it
//! reads, changes.
//!
//! It owns the compiler results and indexes them once: the nodes each token heads, the
//! innermost fn, local fn or lambda around each token, and each declaration's global fn
//! id. A query's answer borrows the analysis, so a feature renders it into its request's
//! arena before the analysis can be replaced.
//!
//! Selection is token-keyed because the AST stores only a `main_token` per node: the
//! nodes "at" a cursor are the ones whose main token is the token under it. A query is
//! only valid on a checked analysis (`checked != null`); a discovery that fails (a module
//! that does not parse, a missing import) leaves only diagnostics.

const std = @import("std");
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const position = @import("position.zig");
const places = @import("places.zig");
const render = @import("render.zig");
const publish = @import("diagnostics.zig");
const Workspace = @import("Workspace.zig");

const Ast = toyc.Ast;
const Callee = toyc.Callee;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Type = toyc.Typecheck.Type;
const SourceMap = toyc.term.render.SourceMap;
const Span = toyc.Walk.Span;

const Analysis = @This();

gpa: std.mem.Allocator,
/// Backs the indexes, the copies below and every memoized answer.
arena: std.heap.ArenaAllocator,
uri: []const u8,
version: i64,
/// The text the analysis checked.
text: []const u8,
encoding: position.Encoding,
graph: Graph.Graph,
/// The entry module's lines.
sm: SourceMap,
checked: ?Checked = null,
/// What a publish sends for this version.
diagnostics: []const protocol.LspDiagnostic = &.{},
/// `places.mutArgs`, computed on first use.
mut_args: ?[]const places.MutArg = null,

pub const Checked = struct {
    res: ResolveGraph.GraphResult,
    tc: TypecheckGraph.GraphResult,
    /// Token `t` heads nodes `tok_nodes[tok_first[t]..tok_first[t + 1]]`, in node order.
    tok_first: []const u32,
    tok_nodes: []const u32,
    /// The innermost fn with a body, local fn or lambda whose span holds token `t`, else
    /// `no_node`.
    scope_of: []const u32,
    /// The top-level fn or method whose body holds token `t`, else `no_node`.
    fn_of: []const u32,
    /// Each local fn's and lambda's innermost enclosing scope.
    scope_parent: std.AutoHashMapUnmanaged(u32, u32),
    /// Each fn, local fn and lambda node's span.
    spans: std.AutoHashMapUnmanaged(u32, Span),
    /// The entry module's fn decl nodes, by global fn id.
    gid_of: std.AutoHashMapUnmanaged(u32, u32),
    /// Each member node that heads a call (`recv.m(..)`, `recv.m[T](..)`), by its call.
    call_of: std.AutoHashMapUnmanaged(u32, u32),
    /// Each node that explicit type args apply to (`id` in `id[int]`), by the `type_app`.
    type_app_of: std.AutoHashMapUnmanaged(u32, u32),
    /// The entry's generic fns and generic types' methods (see `isTemplate`).
    templates: std.AutoHashMapUnmanaged(u32, void),
};

const no_node = std.math.maxInt(u32);

/// Check `text` as the document `uri` at `version`. Only a hard I/O or OOM fault errors;
/// a compile problem is the analysis' diagnostics.
pub fn init(gpa: std.mem.Allocator, ws: Workspace, uri: []const u8, text: []const u8, version: i64) !*Analysis {
    const an = try gpa.create(Analysis);
    errdefer gpa.destroy(an);
    an.* = .{
        .gpa = gpa,
        .arena = .init(gpa),
        .uri = &.{},
        .version = version,
        .text = &.{},
        .encoding = ws.encoding,
        .graph = undefined,
        .sm = undefined,
    };
    errdefer an.arena.deinit();
    const a = an.arena.allocator();
    an.uri = try a.dupe(u8, uri);
    an.text = try a.dupe(u8, text);

    an.graph = try ws.discover(gpa, uri, text);
    errdefer an.graph.deinit(gpa);
    const m = an.graph.entry();
    an.sm = try SourceMap.init(gpa, m.file, m.source);
    errdefer an.sm.deinit(gpa);

    if (an.graph.err == null) {
        var resolved = try ResolveGraph.resolveGraph(gpa, &an.graph);
        errdefer resolved.deinit(gpa);
        // Even with resolve errors: mid-edit, the rest of the buffer still types.
        var typed = try TypecheckGraph.checkGraph(gpa, &an.graph, &resolved, ws.io, 0);
        errdefer typed.deinit(gpa);
        an.checked = try index(a, &an.graph, resolved, typed);
    }
    errdefer if (an.checked) |*c| {
        c.tc.deinit(gpa);
        c.res.deinit(gpa);
    };
    an.diagnostics = try publish.collect(a, an);
    return an;
}

pub fn deinit(an: *Analysis) void {
    const gpa = an.gpa;
    if (an.checked) |*c| {
        c.tc.deinit(gpa);
        c.res.deinit(gpa);
    }
    an.sm.deinit(gpa);
    an.graph.deinit(gpa);
    an.arena.deinit();
    gpa.destroy(an);
}

fn index(a: std.mem.Allocator, graph: *const Graph.Graph, resolved: ResolveGraph.GraphResult, typed: TypecheckGraph.GraphResult) !Checked {
    const m = graph.entry();
    const tree = m.tree();
    const ntok = m.tokens.len;

    const tok_first = try a.alloc(u32, ntok + 1);
    @memset(tok_first, 0);
    for (m.nodes) |n| if (n.main_token < ntok) {
        tok_first[n.main_token + 1] += 1;
    };
    for (1..tok_first.len) |t| tok_first[t] += tok_first[t - 1];
    const tok_nodes = try a.alloc(u32, tok_first[ntok]);
    const fill = try a.dupe(u32, tok_first[0..ntok]);
    for (m.nodes, 0..) |n, i| if (n.main_token < ntok) {
        tok_nodes[fill[n.main_token]] = @intCast(i);
        fill[n.main_token] += 1;
    };

    var spans: std.AutoHashMapUnmanaged(u32, Span) = .empty;
    const Scope = struct { node: u32, span: Span, top: bool };
    var scopes: std.ArrayList(Scope) = .empty;
    for (m.nodes, 0..) |n, i| {
        const top = switch (n.tag) {
            .fn_decl => true,
            .local_fn, .lambda => false,
            else => continue,
        };
        if (n.rhs == Ast.none) continue; // a protocol's or an extern's signature: no body
        // A `_` capture's body is the call it is written in: it holds no scope of its own.
        if (Ast.isCaptureLambda(tree, m.tokens, Ast.Index.from(@intCast(i)))) continue;
        const sp = toyc.Walk.nodeSpan(.{ .tree = tree, .tokens = m.tokens, .source = m.source }, Ast.Index.from(@intCast(i)));
        try spans.put(a, @intCast(i), sp);
        try scopes.append(a, .{ .node = @intCast(i), .span = sp, .top = top });
    }
    // Wider spans first, so a nested scope overwrites the one around it.
    std.mem.sort(Scope, scopes.items, {}, struct {
        fn wider(_: void, x: Scope, y: Scope) bool {
            return x.span.end - x.span.start > y.span.end - y.span.start;
        }
    }.wider);
    const scope_of = try a.alloc(u32, ntok);
    const fn_of = try a.alloc(u32, ntok);
    @memset(scope_of, no_node);
    @memset(fn_of, no_node);
    var scope_parent: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    for (scopes.items) |s| {
        var t = firstTokenAt(m.tokens, s.span.start);
        // The scope painted here before this one is the one around it.
        if (!s.top and t < ntok and scope_of[t] != no_node) try scope_parent.put(a, s.node, scope_of[t]);
        while (t < ntok and m.tokens[t].start < s.span.end) : (t += 1) {
            scope_of[t] = s.node;
            if (s.top) fn_of[t] = s.node;
        }
    }

    var gid_of: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    for (resolved.fns, 0..) |gf, g| {
        if (gf.module != graph.entry_index or gf.decl_node == Ast.none) continue;
        try gid_of.put(a, gf.decl_node.int(), @intCast(g));
    }
    var call_of: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    var type_app_of: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    for (m.nodes, 0..) |n, c| {
        if (n.tag == .type_app and n.lhs != Ast.none) try type_app_of.put(a, n.lhs.int(), @intCast(c));
        const member = Callee.memberNode(tree, n) orelse continue;
        try call_of.put(a, member.int(), @intCast(c));
    }
    var templates: std.AutoHashMapUnmanaged(u32, void) = .empty;
    for (m.nodes, 0..) |n, i| {
        if (n.tag == .fn_decl and Ast.protoAt(tree, n.lhs.int()).generic_params.len > 0) try templates.put(a, @intCast(i), {});
    }
    for (typed.templates) |t| {
        const gf = resolved.fns[t.fn_id];
        if (gf.module == graph.entry_index and gf.decl_node != Ast.none) try templates.put(a, gf.decl_node.int(), {});
    }

    return .{
        .res = resolved,
        .tc = typed,
        .tok_first = tok_first,
        .tok_nodes = tok_nodes,
        .scope_of = scope_of,
        .fn_of = fn_of,
        .scope_parent = scope_parent,
        .spans = spans,
        .gid_of = gid_of,
        .call_of = call_of,
        .type_app_of = type_app_of,
        .templates = templates,
    };
}

/// The first token starting at or after `off`.
fn firstTokenAt(tokens: []const toyc.Token, off: u32) usize {
    var lo: usize = 0;
    var hi: usize = tokens.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (tokens[mid].start < off) lo = mid + 1 else hi = mid;
    }
    return lo;
}

/// Whether a change to the document at `path` can change this analysis: its graph read
/// that file, or its discovery failed (an opened document can satisfy a missing import).
/// A change to the analysis' own document is the caller's to see.
pub fn reads(an: *const Analysis, path: []const u8) bool {
    if (an.graph.err != null) return true;
    for (an.graph.modules, 0..) |m, i| {
        if (i != an.entry() and std.mem.eql(u8, m.file, path)) return true;
    }
    return false;
}

pub fn module(an: *const Analysis) *const Graph.Module {
    return an.graph.entry();
}

pub fn entry(an: *const Analysis) u32 {
    return an.graph.entry_index;
}

pub fn res(an: *const Analysis) *const ResolveGraph.GraphResult {
    return &an.checked.?.res;
}

pub fn tc(an: *const Analysis) *const TypecheckGraph.GraphResult {
    return &an.checked.?.tc;
}

pub fn resolutions(an: *const Analysis) []const ResolveGraph.Resolution {
    return an.res().resolutions[an.entry()];
}

pub fn nodeTypes(an: *const Analysis) []const Type {
    return an.tc().node_types[an.entry()];
}

pub fn callees(an: *const Analysis) []const Callee {
    return an.tc().callees[an.entry()];
}

/// The byte offset of an LSP position in the document: null past its last line, the
/// line's end past its last character.
pub fn offsetAt(an: *const Analysis, line: u32, character: u32) ?u32 {
    return position.positionToOffset(&an.sm, line, character, an.encoding);
}

/// The token under an LSP position, or null over a gap, a comment or past the end.
pub fn tokenAt(an: *const Analysis, line: u32, character: u32) ?u32 {
    const off = an.offsetAt(line, character) orelse return null;
    return position.tokenAt(an.module().tokens, off);
}

pub fn positionOf(an: *const Analysis, off: u32) protocol.Position {
    return position.offsetToPosition(&an.sm, off, an.encoding);
}

pub fn rangeOf(an: *const Analysis, start: u32, end: u32) protocol.Range {
    return position.rangeOf(&an.sm, start, end, an.encoding);
}

/// The nodes token `tok` heads, in node order.
pub fn nodesAt(an: *const Analysis, tok: u32) []const u32 {
    const c = &an.checked.?;
    if (tok + 1 >= c.tok_first.len) return &.{};
    return c.tok_nodes[c.tok_first[tok]..c.tok_first[tok + 1]];
}

/// The innermost fn, local fn or lambda whose source holds token `tok`.
pub fn scopeOf(an: *const Analysis, tok: u32) ?u32 {
    const c = &an.checked.?;
    if (tok >= c.scope_of.len or c.scope_of[tok] == no_node) return null;
    return c.scope_of[tok];
}

/// The scope around the local fn or lambda `scope`.
pub fn scopeParent(an: *const Analysis, scope: u32) ?u32 {
    return an.checked.?.scope_parent.get(scope);
}

/// The top-level fn or method whose source holds token `tok`.
pub fn fnOf(an: *const Analysis, tok: u32) ?u32 {
    const c = &an.checked.?;
    if (tok >= c.fn_of.len or c.fn_of[tok] == no_node) return null;
    return c.fn_of[tok];
}

/// The global fn id of the entry module's fn decl `decl`.
pub fn gidOf(an: *const Analysis, decl: u32) ?u32 {
    return an.checked.?.gid_of.get(decl);
}

/// Whether `fn_node`, a top-level fn or method, is a generic template: a generic fn or a
/// generic type's method. Its body is checked only per instance (a bounded one also over
/// scratch types), so what its own table selects is no concrete selection.
pub fn isTemplate(an: *const Analysis, fn_node: u32) bool {
    return an.checked.?.templates.contains(fn_node);
}

/// Whether node `node` sits in a generic template's body.
pub fn inTemplate(an: *const Analysis, node: u32) bool {
    const f = an.fnOf(an.module().nodes[node].main_token) orelse return false;
    return an.isTemplate(f);
}

/// The slot the checker records the call of `callee` on: the `type_app` it heads when
/// given explicit type args (`id[int](..)`), else its own node.
pub fn callSlotOf(an: *const Analysis, callee: u32) u32 {
    return an.checked.?.type_app_of.get(callee) orelse callee;
}

/// What a free fn's call at `slot` runs.
pub const FreeCallee = union(enum) {
    /// A non-generic fn: its global id.
    direct: u32,
    /// A generic fn's instance.
    instance: u32,
    /// A generic fn whose instance is not known here: shown as declared.
    generic: u32,
};

/// The free fn the call whose slot is `slot` runs, from its Callee; null for any other
/// call. In a generic template's body, which only its instances check, a generic fn shows
/// as declared (its types follow the template's params, and the body's own instance pick
/// was made over no concrete types), and a non-generic one as every instance's.
pub fn freeCallee(an: *const Analysis, slot: u32) ?FreeCallee {
    const t = an.tc();
    const own = an.callees()[slot];
    if (an.inTemplate(slot)) {
        var agreed: ?Callee = null;
        var it = an.instancesOf(an.fnOf(an.module().nodes[slot].main_token));
        const same = while (it.next()) |inst| {
            const ic = inst.nodes().callee(Ast.Index.from(slot));
            if (ic.view() == .none) continue;
            if (agreed) |p| if (!std.meta.eql(p, ic)) break false;
            agreed = ic;
        } else true;
        if (same) if (agreed) |c| if (c.view() == .direct) return .{ .direct = c.view().direct };
        // A fn the resolver names but no instance agrees on: as declared.
        const gid: u32 = switch (own.view()) {
            .direct, .generic => |g| g,
            .instance => |ii| if (ii < t.instances.len) t.instances[ii].template_gid else return null,
            else => an.headFn(slot) orelse return null,
        };
        return if (gid < t.sigs.len and t.sigs[gid].isGenericTemplate()) .{ .generic = gid } else .{ .direct = gid };
    }
    return freeOf(t, own);
}

fn freeOf(t: *const TypecheckGraph.GraphResult, c: Callee) ?FreeCallee {
    return switch (c.view()) {
        .direct => |g| .{ .direct = g },
        .generic => |g| .{ .generic = g },
        .instance => |ii| if (ii < t.instances.len) .{ .instance = ii } else null,
        else => null,
    };
}

/// The fn the resolver says the call at `slot` names: its own, or the head of its
/// explicit type args.
fn headFn(an: *const Analysis, slot: u32) ?u32 {
    const n = an.module().nodes[slot];
    const head = if (n.tag == .type_app) n.lhs else Ast.Index.from(slot);
    if (head == Ast.none) return null;
    const rs = an.resolutions();
    if (head.int() >= rs.len) return null;
    return switch (rs[head.int()]) {
        .func => |g| g,
        else => null,
    };
}

/// Node `node`'s source extent (see `Walk.nodeSpan`).
pub fn span(an: *const Analysis, node: u32) Span {
    if (an.checked.?.spans.get(node)) |sp| return sp;
    const m = an.module();
    return toyc.Walk.nodeSpan(.{ .tree = m.tree(), .tokens = m.tokens, .source = m.source }, Ast.Index.from(node));
}

/// Node `node`'s checked type, when it has one worth showing.
pub fn typeAt(an: *const Analysis, node: u32) ?Type {
    const nts = an.nodeTypes();
    if (node >= nts.len) return null;
    return if (render.isRenderable(nts[node])) nts[node] else null;
}

/// What a token names, as every feature reads it: a named fn, then a member that may
/// select a method, then a local. Func-first: a `mod.fn` access and its callee share a
/// token, and the callable is meant.
pub const Symbol = union(enum) {
    /// A named fn: its global id.
    func: u32,
    /// A `recv.m` access that is no `mod.f`: the member a method call or a bound method
    /// value selects through (see `calleeAt`), or a field read.
    member: u32,
    /// A local, at a binding or a use: its slot and the node naming it.
    local: struct { slot: u32, node: u32 },
};

pub fn symbolAt(an: *const Analysis, tok: u32) ?Symbol {
    const rs = an.resolutions();
    const at = an.nodesAt(tok);
    for (at) |i| if (i < rs.len and rs[i] == .func) return .{ .func = rs[i].func };
    for (at) |i| if (an.module().nodes[i].tag == .field_access) return .{ .member = i };
    for (at) |i| if (i < rs.len and rs[i] == .local) return .{ .local = .{ .slot = rs[i].local, .node = i } };
    return null;
}

/// The call that member node `fa` heads (`recv.m(..)`, `recv.m[T](..)`), if any.
pub fn callOfMember(an: *const Analysis, fa: u32) ?u32 {
    return an.checked.?.call_of.get(fa);
}

/// Where a declaration is: its module and its name token there.
pub const Site = struct { module: u32, tok: u32 };

/// The declaration the symbol at `tok` names: a fn's (in whichever module declares it),
/// a local's binding, the method a call or a bound method value selects, or the type of
/// that name. Null for a builtin, a derive, a bundled declaration (no file a client can
/// open) and a token that names nothing.
pub fn declOf(an: *const Analysis, a: std.mem.Allocator, tok: u32) !?Site {
    if (an.symbolAt(tok)) |sym| switch (sym) {
        .func => |f| {
            const fns = an.res().fns;
            if (f >= fns.len) return null;
            return an.sourceSite(fns[f].module, fns[f].decl_node);
        },
        .local => |l| {
            const t = an.localDeclToken(tok, l.slot) orelse return null;
            return .{ .module = an.entry(), .tok = t };
        },
        // A derive, a builtin or a prelude protocol method has no source to jump to.
        .member => |fa| if (try an.calleeAt(a, fa)) |sig| {
            const decl = sig.decl orelse return null;
            return an.sourceSite(decl.module, Ast.Index.from(decl.node));
        },
    };
    // A type name is left unresolved by the resolver.
    const m = an.module();
    if (m.tokens[tok].tag != .identifier) return null;
    const td = an.typeDecl(m.tokens[tok].text(m.source)) orelse return null;
    return an.sourceSite(td.module, Ast.Index.from(td.node));
}

/// A type's declaration: its module and its decl node.
pub const TypeDecl = struct { module: u32, node: u32 };

/// The declaration of the type named `name` (a struct, tuple struct, enum or protocol), in
/// whichever module declares it; the entry module wins a name clash. The resolver leaves a
/// type name unresolved, so the name is all there is to go on.
pub fn typeDecl(an: *const Analysis, name: []const u8) ?TypeDecl {
    var found: ?TypeDecl = null;
    for (an.graph.modules, 0..) |*tm, mi| {
        if (tm.nodes.len == 0) continue;
        const prog = tm.nodes[Ast.root(tm.nodes).int()];
        if (prog.tag != .program) continue;
        for (Ast.rangeSlice(tm.tree(), prog.lhs.int())) |d| {
            const decl = tm.nodes[d.int()];
            switch (decl.tag) {
                .struct_decl, .tuple_struct_decl, .enum_decl, .protocol_decl => {},
                else => continue,
            }
            if (!std.mem.eql(u8, tm.tokens[decl.main_token].text(tm.source), name)) continue;
            if (found == null or mi == an.entry()) found = .{ .module = @intCast(mi), .node = d.int() };
        }
    }
    return found;
}

/// The name token of `decl` in `mod`, or null when there is no openable source: a seeded
/// builtin has no decl, and the embedded std has no file a client can open.
fn sourceSite(an: *const Analysis, mod: u32, decl: Ast.Index) ?Site {
    if (decl == Ast.none) return null;
    const tm = &an.graph.modules[mod];
    if (tm.bundled) return null;
    return .{ .module = mod, .tok = tm.nodes[decl.int()].main_token };
}

/// The binding token of the local in `slot` that `use_tok` names, within its top-level
/// fn. A top-level fn's param carries no resolution, so it is found by position; every
/// other binder (a `:=`, a loop key, a local fn, a lambda's param) by the slot it carries.
fn localDeclToken(an: *const Analysis, use_tok: u32, slot: u32) ?u32 {
    const m = an.module();
    const fn_node = an.fnOf(use_tok) orelse return null;
    const proto = Ast.protoAt(m.tree(), m.nodes[fn_node].lhs.int());
    if (slot < proto.params.len) return m.nodes[proto.params[slot].int()].main_token;
    const rs = an.resolutions();
    // The binder-tag filter is load-bearing: a use also carries the slot. (Accepted gap: a
    // `for k, v in` value binds on an `.identifier` leaf, so a use of `v` finds nothing.)
    for (m.nodes, 0..) |n, i| {
        switch (n.tag) {
            .var_decl, .for_stmt, .for_in_stmt, .for_in2_stmt, .local_fn, .param => {},
            else => continue,
        }
        if (i >= rs.len or an.fnOf(n.main_token) != fn_node) continue;
        if (rs[i] == .local and rs[i].local == slot) return n.main_token;
    }
    return null;
}

/// The signature of the method a `recv.m(..)` call selected, or a `recv.m` bound method
/// value binds, read from the checker's Callee table, so hover, definition and signature
/// help agree on which method a call invokes. In a generic template's body, which only its
/// instances check, it is read from the instances' tables (see `instanceSelect`).
///
/// KNOWN LIMITATIONS:
///   * A method call in a generic body that no instance reaches has no selection: the
///     checker checks an unbounded body only per instance, and a bounded one over scratch
///     types that resolve no generic type's method.
///   * A generic protocol's method on a bounded receiver keeps the protocol's own params
///     (`U`, not `int`, for `T has Scale[int]`): the Callee names the protocol method, not
///     the bound's arguments.
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
    /// How it is shown: by its checked types, or, where those are opaque once typecheck is
    /// done, as its declaration spells it.
    spell: Spell = .checked,

    pub const Decl = struct { module: u32, node: u32 };

    pub const Spell = enum {
        checked,
        /// A user protocol's method signature, as written.
        protocol,
        /// A generic method's header, as written (`fn map(self: Vec[T], f: fn(T) -> U)`).
        template,
    };
};

/// The selected method of the call whose member node is `fa`, or the method the bound
/// method value `fa` binds; null when `fa` records no method (a field read, a fn-typed
/// field's call, a variant, a `mod.f`, a failed dispatch). Borrowed type and name slices
/// point into the analysis; `a` owns only what is built.
pub fn calleeAt(an: *const Analysis, a: std.mem.Allocator, fa: u32) !?CalleeSig {
    var sig = (try an.selectCallee(a, fa)) orelse return null;
    // Callers slice the receiver off; a broken buffer must not make that slice panic.
    sig.has_self = sig.has_self and sig.params.len > 0;
    return sig;
}

fn selectCallee(an: *const Analysis, a: std.mem.Allocator, fa: u32) !?CalleeSig {
    const cs = an.callees();
    if (fa >= cs.len) return null;
    const c = cs[fa];
    // A bounded receiver's protocol method is the template body's own answer, the same in
    // every instance; anything else its instances know better.
    if (c.view() != .protocol and c.templateFn() == null and an.inTemplate(fa)) {
        if (try an.instanceSelect(a, fa)) |sig| return sig;
    }
    return an.selectFrom(a, fa, c, .{ .types = an.nodeTypes(), .callees = an.callees() });
}

/// The method at member `fa`, in a generic template's body, as the template's instances
/// selected it. A pick that reaches a generic method shows that method as declared: its
/// types follow the template's own params, which no one instance shows. Any other pick is
/// shown as made when every instance made it on the same receiver type (a concrete type's
/// method, a builtin). Null when no instance reaches the body or the instances reach
/// different methods.
fn instanceSelect(an: *const Analysis, a: std.mem.Allocator, fa: u32) !?CalleeSig {
    const recv = an.module().nodes[fa].lhs;
    var first: ?*const toyc.Mono.Instance = null;
    var same = true;
    var reach: ?u32 = null;
    var one_reach = true;
    var it = an.instancesOf(an.fnOf(an.module().nodes[fa].main_token));
    const fa_i = Ast.Index.from(fa);
    while (it.next()) |inst| {
        const c = inst.nodes().callee(fa_i);
        if (c.view() == .none) continue;
        if (first) |f| {
            if (!std.meta.eql(f.nodes().callee(fa_i), c)) same = false;
            if (recv != Ast.none and !Type.eql(f.nodes().typeOf(recv), inst.nodes().typeOf(recv))) same = false;
        } else first = inst;
        const g = an.declaredFn(c);
        if (g == null or (reach != null and reach.? != g.?)) one_reach = false;
        if (reach == null) reach = g;
    }
    const f = first orelse return null;
    if (same and !an.reifiesGeneric(f.nodes().callee(fa_i))) return an.selectFrom(a, fa, f.nodes().callee(fa_i), f.nodes());
    if (!one_reach) return null;
    return an.templateSig(a, fa, reach orelse return null);
}

/// Whether method pick `c` is a generic method's instance.
fn reifiesGeneric(an: *const Analysis, c: Callee) bool {
    const mi = switch (c.view()) {
        .method, .bound_method => |mi| mi,
        else => return false,
    };
    return mi < an.tc().methods.len and an.tc().methods[mi].instance != null;
}

/// The fn a method pick runs as its source declares it: the generic method an instance's
/// row reifies, else the row's own fn. Null for a derive and any pick that is no method.
fn declaredFn(an: *const Analysis, c: Callee) ?u32 {
    const t = an.tc();
    const mi = switch (c.view()) {
        .method, .bound_method => |mi| mi,
        else => return null,
    };
    if (mi >= t.methods.len) return null;
    const meth = t.methods[mi];
    if (meth.derive != null) return null;
    if (meth.instance) |ii| return if (ii < t.instances.len) t.instances[ii].template_gid else null;
    return meth.fn_id;
}

/// Generic method `gid`, called through member `fa`, as its declaration spells it.
fn templateSig(an: *const Analysis, a: std.mem.Allocator, fa: u32, gid: u32) !?CalleeSig {
    const t = an.tc();
    const r = an.res();
    if (gid >= r.fns.len or gid >= t.sigs.len) return null;
    const gf = r.fns[gid];
    if (gf.decl_node == Ast.none) return null;
    const m = an.module();
    const sig = t.sigs[gid];
    const names = try an.declParamNames(a, gf.module, gf.decl_node);
    const has_self = names.len > 0 and std.mem.eql(u8, names[0], "self");
    return .{
        .name = m.tokens[m.nodes[fa].main_token].text(m.source),
        .params = sig.params,
        .param_names = names,
        .ret = sig.ret,
        .has_self = has_self,
        .mut_mask = sig.mut_mask,
        .decl = .{ .module = gf.module, .node = gf.decl_node.int() },
        .spell = .template,
    };
}

/// What callee `c` at member `fa` selects, reading node types from `nts` (the module's, or
/// an instance's).
fn selectFrom(an: *const Analysis, a: std.mem.Allocator, fa: u32, c: Callee, nodes: toyc.Body.Nodes) !?CalleeSig {
    const t = an.tc();
    const r = an.res();
    const m = an.module();
    const fa_node = m.nodes[fa];
    const name = m.tokens[fa_node.main_token].text(m.source);
    switch (c.view()) {
        // A generic method whose call did not check (`xs.map(` mid-typing) picked no
        // instance; its template still names what the user is calling.
        .none => return an.templateSig(a, fa, c.templateFn() orelse return null),
        // A function value's signature is its type, not a declaration's; a free fn call's
        // slot is not a member call's.
        .value, .fn_ref, .direct, .instance, .generic => return null,
        // A bound method value is typed as its method minus `self`, and the method's own
        // row says the rest: names, decl and modes.
        .method, .bound_method => |i| {
            if (i >= t.methods.len) return null;
            const meth = t.methods[i];
            // The derive, instance or fn the call lowers to, as the compiler picks it.
            const sig = toyc.Typecheck.witnessSig(meth, t.derives, t.instances, t.sigs) orelse return null;
            // A derive's `fn_id` is 0: it has no declaration to name its params.
            if (meth.derive != null) return .{ .name = name, .params = sig.params, .param_names = selfNames(sig.params.len), .ret = sig.ret, .has_self = true };
            if (meth.fn_id >= r.fns.len) return null;
            const gf = r.fns[meth.fn_id];
            const names = try an.declParamNames(a, gf.module, gf.decl_node);
            const decl: ?CalleeSig.Decl = if (gf.decl_node == Ast.none) null else .{ .module = gf.module, .node = gf.decl_node.int() };
            return .{ .name = name, .params = sig.params, .param_names = names, .ret = sig.ret, .has_self = !meth.is_static, .mut_mask = sig.mut_mask, .decl = decl };
        },
        .protocol => |p| {
            if (p.protocol >= t.protocols.len) return null;
            const proto = t.protocols[p.protocol];
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
            const pm = &an.graph.modules[proto.mod];
            const sig_nodes = Ast.rangeSlice(pm.tree(), pm.nodes[proto.decl_node.int()].lhs.int());
            // The protocol's method index is its position among the decl's signatures.
            if (p.method >= sig_nodes.len) return sig;
            const node = sig_nodes[p.method];
            sig.param_names = try an.declParamNames(a, proto.mod, node);
            sig.decl = .{ .module = proto.mod, .node = node.int() };
            sig.spell = .protocol;
            return sig;
        },
        .builtin => |k| {
            const recv: Type = if (fa_node.lhs != Ast.none and nodes.has(fa_node.lhs)) nodes.typeOf(fa_node.lhs) else .invalid;
            const call_ty: Type = if (an.callOfMember(fa)) |call| (if (nodes.has(Ast.Index.from(call))) nodes.typeOf(Ast.Index.from(call)) else .invalid) else .invalid;
            // Kept here rather than as a compiler table: a new `Builtin` breaks this switch
            // at compile time, and the compiler grows no LSP-only API.
            const shape: struct { params: []const Type, ret: Type, names: []const []const u8 } = switch (k) {
                .scalar_eq => .{ .params = try a.dupe(Type, &.{ recv, recv }), .ret = Type.bool, .names = &.{ "self", "other" } },
                .scalar_hash => .{ .params = try a.dupe(Type, &.{recv}), .ret = Type.int, .names = &.{"self"} },
                .scalar_cmp => .{ .params = try a.dupe(Type, &.{ recv, recv }), .ret = call_ty, .names = &.{ "self", "other" } },
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

/// The names of a source-less method's params (a derive, a prelude protocol's method): a
/// binary one (`eq`, `cmp`) compares against `other`.
fn selfNames(n: usize) []const []const u8 {
    return if (n == 2) &.{ "self", "other" } else &.{"self"};
}

/// Parameter names from the `fn_decl` `decl` in module `mod`; empty when there is no
/// source decl.
pub fn declParamNames(an: *const Analysis, a: std.mem.Allocator, mod: u32, decl: Ast.Index) ![]const []const u8 {
    if (decl == Ast.none) return &.{};
    const tm = &an.graph.modules[mod];
    const proto = Ast.protoAt(tm.tree(), tm.nodes[decl.int()].lhs.int());
    const out = try a.alloc([]const u8, proto.params.len);
    for (proto.params, out) |p, *pname| pname.* = tm.tokens[tm.nodes[p.int()].main_token].text(tm.source);
    return out;
}

/// The instances of the generic fn `fn_node` (none for a fn that is no template): each
/// rechecked its body, so its tables cover the nodes the template's own table leaves
/// empty.
pub fn instancesOf(an: *const Analysis, fn_node: ?u32) InstanceIterator {
    return .{ .an = an, .fn_node = fn_node orelse no_node };
}

pub const InstanceIterator = struct {
    an: *const Analysis,
    fn_node: u32,
    i: usize = 0,

    pub fn next(it: *InstanceIterator) ?*const toyc.Mono.Instance {
        const insts = it.an.tc().instances;
        while (it.i < insts.len) {
            const inst = &insts[it.i];
            it.i += 1;
            if (inst.mod == it.an.entry() and inst.decl_node.int() == it.fn_node) return inst;
        }
        return null;
    }
};

/// Whether member node `fa` binds a method as a value (`g := acc.plus`): in its body as
/// checked, or in an instance of the generic fn around it.
pub fn bindsMethod(an: *const Analysis, fa: u32) bool {
    if (an.callees()[fa].view() == .bound_method) return true;
    var it = an.instancesOf(an.fnOf(an.module().nodes[fa].main_token));
    while (it.next()) |inst| if (inst.nodes().callee(Ast.Index.from(fa)).view() == .bound_method) return true;
    return false;
}

/// The receiver type of the bound method value `fa`, when every check of it agrees on
/// one: its body's, or each instance's of the generic fn around it.
pub fn boundReceiver(an: *const Analysis, fa: u32) ?Type {
    const recv = an.module().nodes[fa].lhs;
    if (recv == Ast.none) return null;
    if (an.callees()[fa].view() == .bound_method) return an.typeAt(recv.int());
    var agreed: ?Type = null;
    var it = an.instancesOf(an.fnOf(an.module().nodes[fa].main_token));
    while (it.next()) |inst| {
        if (inst.nodes().callee(Ast.Index.from(fa)).view() != .bound_method) continue;
        const t = inst.nodes().typeOf(recv);
        if (!render.isRenderable(t)) return null;
        if (agreed) |prev| if (!Type.eql(prev, t)) return null;
        agreed = t;
    }
    return agreed;
}

/// Every argument a `mut` param takes and every `mut self` receiver (see `places`).
pub fn mutArgs(an: *Analysis) ![]const places.MutArg {
    if (an.mut_args) |ma| return ma;
    const ma = try places.mutArgs(an, an.arena.allocator());
    an.mut_args = ma;
    return ma;
}

const testing = std.testing;
const test_util = @import("test_util.zig");

test "nodesAt lists a token's nodes in node order; scopeOf and fnOf find the innermost and the top-level scope" {
    const gpa = testing.allocator;
    const src =
        \\struct S { a: int }
        \\fn apply(f: fn(int) -> int, v: int) -> int { f(v) }
        \\fn main() -> int {
        \\    base := 1
        \\    fn step(k: int) -> int { k + base }
        \\    return apply(fn(x) { x + step(base) }, 2)
        \\}
        \\
    ;
    const an = try test_util.analyze(gpa, src);
    defer an.deinit();
    const m = an.module();
    const tokAt = struct {
        fn f(x: *const Analysis, needle: []const u8) u32 {
            const off: u32 = @intCast(std.mem.indexOf(u8, x.module().source, needle).?);
            return position.tokenAt(x.module().tokens, off).?;
        }
    }.f;

    for (m.nodes, 0..) |n, i| {
        const at = an.nodesAt(n.main_token);
        try testing.expect(std.mem.indexOfScalar(u32, at, @intCast(i)) != null);
        for (at[1..], at[0..at.len -| 1]) |later, earlier| try testing.expect(earlier < later);
    }

    const main_fn = an.fnOf(tokAt(an, "base := 1")).?;
    try testing.expectEqualStrings("main", m.tokens[m.nodes[main_fn].main_token].text(m.source));
    try testing.expectEqual(main_fn, an.scopeOf(tokAt(an, "base := 1")).?);
    // A local fn's body and a lambda's body are their own scopes inside `main`.
    const k_use = tokAt(an, "k + base");
    try testing.expectEqual(toyc.Ast.Node.Tag.local_fn, m.nodes[an.scopeOf(k_use).?].tag);
    try testing.expectEqual(main_fn, an.fnOf(k_use).?);
    const x_use = tokAt(an, "x + step");
    try testing.expectEqual(toyc.Ast.Node.Tag.lambda, m.nodes[an.scopeOf(x_use).?].tag);
    try testing.expectEqual(main_fn, an.fnOf(x_use).?);
    // A declaration's keyword is in its span; a token outside every fn has none.
    try testing.expectEqual(main_fn, an.fnOf(tokAt(an, "fn main")).?);
    try testing.expect(an.fnOf(tokAt(an, "a: int")) == null);
    try testing.expect(an.scopeOf(tokAt(an, "a: int")) == null);
    try testing.expectEqual(an.gidOf(main_fn).?, for (an.res().fns, 0..) |gf, g| {
        if (gf.decl_node.int() == main_fn and gf.module == an.entry()) break g;
    } else unreachable);
}

test "calleeAt: every recorded callee in the method fixture yields a signature; a non-method field access yields null" {
    const gpa = testing.allocator;
    const an = try test_util.analyze(gpa, test_util.method_fixture);
    defer an.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const m = an.module();
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(gpa);
    var kinds = std.EnumSet(std.meta.Tag(Callee.View)){};
    for (an.callees(), 0..) |c, fa| {
        const v = c.view();
        switch (v) {
            // A free fn call's slot is no member call's.
            .none, .direct, .instance, .generic => continue,
            else => {},
        }
        const sig = (try an.calleeAt(a, @intCast(fa))) orelse return error.RecordedCalleeWithoutSig;
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
            try testing.expect((try an.calleeAt(a, @intCast(i))) == null);
        }
    }
}

test "calleeAt: a bound method value yields its method's signature; a value call and a fn name yield null" {
    const gpa = testing.allocator;
    const src =
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
        \\struct Box[T] { v: T }
        \\impl Box[T] { fn get(self) -> T { self.v } }
        \\fn add(a: int, b: int) -> int { a + b }
        \\fn main() -> int {
        \\    acc := Acc { n: 1 }
        \\    g := acc.plus
        \\    bg := Box { v: 2 }.get
        \\    f := add
        \\    return g(1) + bg() + f(1, 2)
        \\}
        \\
    ;
    const an = try test_util.analyze(gpa, src);
    defer an.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    var bound: usize = 0;
    var values: usize = 0;
    for (an.callees(), 0..) |c, n| switch (c.view()) {
        .bound_method => {
            const sig = (try an.calleeAt(arena.allocator(), @intCast(n))) orelse return error.BoundMethodWithoutSig;
            try testing.expect(sig.has_self and sig.decl != null);
            try testing.expectEqualStrings("self", sig.param_names[0]);
            bound += 1;
        },
        .value, .fn_ref => {
            try testing.expect((try an.calleeAt(arena.allocator(), @intCast(n))) == null);
            values += 1;
        },
        else => {},
    };
    try testing.expectEqual(@as(usize, 2), bound);
    try testing.expect(values >= 4);
}

test "a discovery that fails leaves only diagnostics" {
    const gpa = testing.allocator;
    const an = try test_util.analyze(gpa, "fn main( -> int {\n");
    defer an.deinit();
    try testing.expect(an.checked == null);
    try testing.expect(an.diagnostics.len > 0);
}

test "hover, definition and signature help agree on what each token of the method fixtures is" {
    for ([_]struct { src: []const u8, calls: usize }{
        // plus (twice), make, get, concat, eq, unwrap, unwrap_or, hash, len, dbl (twice), twice.
        .{ .src = test_util.method_fixture, .calls = 13 },
        // get (three times), plus, twice, use_box, with_acc (twice).
        .{ .src = test_util.generic_body_fixture, .calls = 8 },
    }) |fx| try expectAgreement(fx.src, fx.calls);
}

fn expectAgreement(src: []const u8, want_calls: usize) !void {
    const gpa = testing.allocator;
    const hover = @import("hover.zig");
    const definition = @import("definition.zig");
    const signature = @import("signature.zig");
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const an = try test_util.analyzeIn(gpa, ws, "file:///p/main.toy", src);
    defer an.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const m = an.module();

    var calls: usize = 0;
    for (m.tokens, 0..) |t, ti| {
        if (t.tag == .newline or t.tag == .eof) continue;
        const tok: u32 = @intCast(ti);
        const name = t.text(m.source);
        const p = an.positionOf(t.start);
        const h = try hover.at(a, an, p.line, p.character);
        const d = try definition.at(a, an, p.line, p.character);

        // A use and its declaration spell one name, and the declaration is the one
        // `declOf` names.
        if (d) |loc| {
            const site = (try an.declOf(a, tok)).?;
            try testing.expectEqual(an.entry(), site.module);
            const decl = m.tokens[site.tok];
            try testing.expectEqual(an.rangeOf(decl.start, decl.end), loc.range);
            try testing.expectEqualStrings(name, decl.text(m.source));
        }

        // A callee that selects a callable is shown as that callable by all three.
        if (t.tag != .identifier or ti + 1 >= m.tokens.len or m.tokens[ti + 1].tag != .l_paren) continue;
        const callable = switch (an.symbolAt(tok) orelse continue) {
            .func => true,
            .member => |fa| an.callOfMember(fa) != null and try an.calleeAt(a, fa) != null,
            .local => false,
        };
        if (!callable) continue;
        calls += 1;
        const head = try std.fmt.allocPrint(a, "fn {s}(", .{name});
        const at_paren = an.positionOf(m.tokens[ti + 1].end);
        const s = (try signature.at(a, gpa, ws, an, at_paren.line, at_paren.character)) orelse {
            std.debug.print("no signature help for '{s}'\n", .{name});
            return error.NoSignatureHelp;
        };
        const hv = h orelse {
            std.debug.print("no hover for '{s}'\n", .{name});
            return error.NoHover;
        };
        try testing.expect(std.mem.startsWith(u8, hv, "```toy\n"));
        testing.expect(std.mem.startsWith(u8, hv["```toy\n".len..], head) and std.mem.startsWith(u8, s.signatures[0].label, head)) catch |e| {
            std.debug.print("'{s}': hover {s} / signature {s}\n", .{ name, hv, s.signatures[0].label });
            return e;
        };
        // A callable with a declaration jumps to a fn of that name.
        if (d) |loc| {
            const decl_tok = position.tokenAt(m.tokens, an.offsetAt(loc.range.start.line, loc.range.start.character).?).?;
            const is_fn = for (an.nodesAt(decl_tok)) |i| {
                if (m.nodes[i].tag == .fn_decl) break true;
            } else false;
            try testing.expect(is_fn);
        }
    }
    try testing.expectEqual(want_calls, calls);
}

test "a bound method value in a generic fn: its instances say it binds, and agree or not on its receiver" {
    const gpa = testing.allocator;
    const src =
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
        \\struct Box[T] { v: T }
        \\impl Box[T] { fn get(self) -> T { self.v } }
        \\fn f[T](x: T) -> int {
        \\    acc := Acc { n: 1 }
        \\    g := acc.plus
        \\    b := Box { v: x }
        \\    h := b.get
        \\    return g(1)
        \\}
        \\fn main() -> int { return f(1) + f("s") }
        \\
    ;
    const an = try test_util.analyze(gpa, src);
    defer an.deinit();
    const m = an.module();
    const memberAt = struct {
        fn f(x: *const Analysis, needle: []const u8) u32 {
            const off: u32 = @intCast(std.mem.indexOf(u8, x.module().source, needle).?);
            const tok = position.tokenAt(x.module().tokens, off).?;
            return x.symbolAt(tok).?.member;
        }
    }.f;
    const plus = memberAt(an, "plus\n");
    const get = memberAt(an, "get\n");
    // The template's own table records neither: only its instances check its body.
    try testing.expect(an.callees()[plus].view() != .bound_method);
    try testing.expect(an.bindsMethod(plus) and an.bindsMethod(get));
    const recv = an.boundReceiver(plus).?;
    try testing.expectEqualStrings("Acc", @import("render.zig").bareName(an.tc().layouts[recv.struct_id].name));
    // `Box[int]` in one instance, `Box[str]` in the other.
    try testing.expect(an.boundReceiver(get) == null);
    try testing.expect(!an.bindsMethod(memberAt(an, "n + k")));
    _ = m;
}
