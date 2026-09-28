//! textDocument/completion: the candidates at a cursor position.
//!
//! Three contexts are served: SCOPE (a bare identifier position → in-scope locals /
//! params / visible top-level fns + types + keywords), MEMBER (`recv.` → the receiver
//! type's fields + methods), and MODULE (`imported_ns.` → that module's pub members).
//! The active context returns ONLY its own candidates — a member/module context never
//! falls back to scope, so a `p.` never leaks globals.
//!
//! Unlike hover, completion must survive a BROKEN buffer: the user is mid-edit, so a
//! bare `recv.` does not parse. Detection therefore runs over a fault-tolerant raw LEX
//! of the buffer (never the parse tree): the token immediately before the cursor decides
//! the context. For a trailing `.` the buffer is REPAIRED — a probe identifier is spliced
//! at the cursor so `recv.` becomes `recv.<probe>`, which parses to a `field_access` whose
//! receiver we then type. The receiver's type is read from the (fault-tolerant) typecheck
//! result even when the rest of the document has errors.
//!
//! KNOWN LIMITATIONS (tractable approximation, not bugs):
//!   * A buffer with a parse error ANYWHERE OTHER than the cursor's own trailing `.`
//!     re-triggers discovery's whole-tree discard → an empty list. Single-edit buffers
//!     (the overwhelmingly common live case) are served.
//!   * A scalar/`str` receiver (`x: int`, `"s".`) and a bare TYPE-NAME / static receiver
//!     (`Counter.`) resolve to no field/method set → a valid empty list. A member context
//!     needs a struct/enum VALUE receiver.
//!   * The resolver keeps no positional lexical-scope table, so SCOPE is an approximation:
//!     the enclosing fn's params + its `:=` locals (bracketed by source position between
//!     adjacent fn decls) + all visible top-level decls + the language keywords. A cursor
//!     inside a fn signature may misattribute the enclosing fn.
//!   * Scalar / `str` / generic-template methods are not surfaced (only concrete
//!     `methods`-table entries are).

const std = @import("std");
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const position = @import("position.zig");
const render = @import("render.zig");
const Workspace = @import("Workspace.zig");

const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Token = toyc.Token;
const Ast = toyc.Ast;
const Type = toyc.Typecheck.Type;
const SourceMap = toyc.term.render.SourceMap;

const kind = protocol.completion_kind;

/// The probe identifier spliced after a trailing `.` so the buffer parses. Long +
/// underscored so it can never collide with a real toy identifier the user typed.
const probe = "__toy_completion_probe__";

/// Owns the arena backing `items`. `items` is empty (never null) and stays valid until
/// `deinit`; the compiler results it was rendered from have already been torn down.
pub const Completions = struct {
    arena: std.heap.ArenaAllocator,
    items: []const protocol.CompletionItem,

    pub fn deinit(self: *Completions) void {
        self.arena.deinit();
    }
};

const Context = enum { none, scope, member };

/// The detected completion context, decided purely from the tokens before the cursor.
/// `member`/`module` are NOT distinguished here — that needs the receiver's RESOLUTION
/// (a namespace vs a value), read after the buffer is compiled.
const Detected = struct {
    context: Context,
    /// A trailing `.` under the cursor: splice `probe` at the cursor so the access parses.
    repair: bool = false,
    /// Member with an already-typed (partial) field name after the `.`: the source START
    /// offset of that field-name token, used to locate the existing `field_access`.
    partial_start: u32 = 0,
};

/// The completion candidates at (`line`, `character`) in `source`. Never errors on a
/// compile problem — only a hard I/O / OOM fault propagates. An empty list is valid.
pub fn completionsAt(
    gpa: std.mem.Allocator,
    ws: Workspace,
    source: []const u8,
    line: u32,
    character: u32,
    doc_uri: []const u8,
) !Completions {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();

    // Detect the context from a fault-tolerant raw lex — the parse tree may not exist yet.
    const toks = try toyc.Lexer.tokenize(gpa, source);
    defer gpa.free(toks);

    var sm = try SourceMap.init(gpa, "c", source);
    defer sm.deinit(gpa);

    const off = position.positionToOffset(&sm, line, character, ws.encoding) orelse return empty(arena);
    const det = classify(toks, off);
    if (det.context == .none) return empty(arena);

    // Repair a trailing `.` so the access parses; otherwise compile the buffer as-is.
    const repaired = if (det.repair)
        try std.mem.concat(gpa, u8, &.{ source[0..off], probe, source[off..] })
    else
        source;
    defer if (det.repair) gpa.free(repaired);

    var graph = try ws.discover(gpa, doc_uri, repaired);
    defer graph.deinit(gpa);
    // A parse error other than the repaired cursor discards the tree — nothing to offer.
    if (graph.err != null) return empty(arena);

    // Unlike hover, we do NOT bail on resolve/type errors: an unknown probe member or a
    // mistyped elsewhere is expected mid-edit. The receiver still resolves and types.
    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, ws.io, 0);
    defer tc.deinit(gpa);

    const m = graph.entry();
    const entry = graph.entry_index;
    const a = arena.allocator();

    var col: Collector = .{ .a = a };

    switch (det.context) {
        .none => {},
        .scope => try collectScope(&col, m, entry, &res, &tc, off),
        .member => {
            const recv = locateReceiver(m, det) orelse return empty(arena);
            try collectMember(&col, graph.modules, entry, &res, &tc, recv);
        },
    }

    return .{ .arena = arena, .items = try col.items.toOwnedSlice(a) };
}

fn empty(arena: std.heap.ArenaAllocator) Completions {
    return .{ .arena = arena, .items = &.{} };
}

/// Accumulates candidates into the result arena, de-duping by `(label, kind)`. Every
/// label/detail is arena-`dupe`d here because the sources (method/sig/layout names, token
/// text) are BORROWED from the compiler results, which are torn down before we return.
const Collector = struct {
    a: std.mem.Allocator,
    items: std.ArrayList(protocol.CompletionItem) = .empty,

    fn add(c: *Collector, label: []const u8, k: u8, detail: ?[]const u8) !void {
        for (c.items.items) |it| {
            if (it.kind == k and std.mem.eql(u8, it.label, label)) return;
        }
        try c.items.append(c.a, .{
            .label = try c.a.dupe(u8, label),
            .kind = k,
            .detail = if (detail) |d| try c.a.dupe(u8, d) else null,
        });
    }
};

/// The token whose `start` is the largest one strictly less than `off`, or null. Binary
/// search over the start-sorted stream (`< off`, so a token STARTING at the cursor — the
/// char being typed — is not "before" it).
fn lastBefore(tokens: []const Token, off: u32) ?usize {
    var lo: usize = 0;
    var hi: usize = tokens.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (tokens[mid].start < off) lo = mid + 1 else hi = mid;
    }
    return if (lo == 0) null else lo - 1;
}

/// Walk back over the lexer's zero-width `.newline` terminators to the previous REAL
/// token, so a cursor on the line after `recv` still sees the `.`/identifier before it.
fn skipNewlineBack(tokens: []const Token, from: usize) ?usize {
    var i = from;
    while (tokens[i].tag == .newline) {
        if (i == 0) return null;
        i -= 1;
    }
    return i;
}

/// Decide the context from the tokens before `off`:
///   `.dot` before                     → member (repair: splice a probe at the cursor)
///   `.identifier` preceded by `.dot`   → member (partial field name already typed)
///   `.identifier` otherwise            → scope
///   anything else                      → none (no meaningful context)
fn classify(tokens: []const Token, off: u32) Detected {
    const p0 = lastBefore(tokens, off) orelse return .{ .context = .none };
    const p = skipNewlineBack(tokens, p0) orelse return .{ .context = .none };
    switch (tokens[p].tag) {
        .dot => return .{ .context = .member, .repair = true },
        .identifier => {
            const q0 = lastBefore(tokens, tokens[p].start) orelse return .{ .context = .scope };
            const q = skipNewlineBack(tokens, q0) orelse return .{ .context = .scope };
            if (tokens[q].tag == .dot)
                return .{ .context = .member, .partial_start = tokens[p].start };
            return .{ .context = .scope };
        },
        else => return .{ .context = .none },
    }
}

/// Locate the receiver NODE index of the target `field_access`. For a repaired buffer the
/// probe field name is matched by text prefix (robust to the byte-offset shift the splice
/// caused); for a partial name the field-name token is matched by source START offset
/// (robust to any node/token index reshuffle between our raw lex and the compile).
fn locateReceiver(m: *const Graph.Module, det: Detected) ?u32 {
    for (m.nodes) |n| {
        if (n.tag != .field_access) continue;
        const t = m.tokens[n.main_token].text(m.source);
        if (det.repair) {
            if (std.mem.startsWith(u8, t, probe)) return n.lhs.int();
        } else if (m.tokens[n.main_token].start == det.partial_start) {
            return n.lhs.int();
        }
    }
    return null;
}

/// Member/module enumeration. Decides module vs value by the receiver's RESOLUTION FIRST
/// (a `.module` receiver must not be misread as a value), then hard-returns without a
/// scope fallback so unrelated globals never leak into a `recv.` completion.
fn collectMember(
    col: *Collector,
    mods: []const Graph.Module,
    entry: u32,
    res: *const ResolveGraph.GraphResult,
    tc: *const TypecheckGraph.GraphResult,
    recv: u32,
) !void {
    const resolutions = res.resolutions[entry];
    if (recv < resolutions.len and resolutions[recv] == .module) {
        try collectModule(col, mods, res, resolutions[recv].module);
        return;
    }

    const node_types = tc.node_types[entry];
    if (recv >= node_types.len) return;
    const rt = node_types[recv];
    if (rt.kind != .@"struct" and rt.kind != .@"enum") return;

    if (rt.kind == .@"struct" and rt.struct_id < tc.layouts.len) {
        const layout = tc.layouts[rt.struct_id];
        for (layout.field_names, layout.field_types) |fname, fty| {
            var buf: std.ArrayList(u8) = .empty;
            try render.renderType(col.a, &buf, fty, tc.layouts, tc.enum_layouts);
            try col.add(render.bareName(fname), kind.field, buf.items);
        }
    }

    for (tc.methods) |mth| {
        if (!Type.eql(mth.recv, rt)) continue;
        var detail: ?[]const u8 = null;
        var buf: std.ArrayList(u8) = .empty;
        if (mth.fn_id < tc.sigs.len) {
            try render.renderSig(col.a, &buf, tc.sigs[mth.fn_id], tc.layouts, tc.enum_layouts);
            detail = buf.items;
        }
        try col.add(render.bareName(mth.name), kind.method, detail);
    }
}

/// A module's pub members: its pub fns (from the global fn table) plus its pub types
/// (walked off that module's own program range). No detail on the fns — a generic std
/// signature (`print[T has Display]`) renders as `fn print(?) -> ()`, which is noise.
fn collectModule(col: *Collector, mods: []const Graph.Module, res: *const ResolveGraph.GraphResult, mod_id: u32) !void {
    for (res.fns) |gf| {
        if (gf.module != mod_id or !gf.is_pub) continue;
        try col.add(render.bareName(gf.name), kind.function, null);
    }
    if (mod_id >= mods.len) return;
    const tmod = &mods[mod_id];
    if (tmod.nodes.len == 0) return;
    const tree = tmod.tree();
    const prog = tmod.nodes[Ast.root(tmod.nodes).int()];
    if (prog.tag != .program) return;
    for (Ast.rangeSlice(tree, prog.lhs.int())) |decl_idx| {
        if (!tree.isPub(decl_idx)) continue;
        const decl = tmod.nodes[decl_idx.int()];
        const name = tmod.tokens[decl.main_token].text(tmod.source);
        switch (decl.tag) {
            .struct_decl, .tuple_struct_decl => try col.add(name, kind.@"struct", null),
            .enum_decl => try col.add(name, kind.@"enum", null),
            else => {},
        }
    }
}

/// Scope enumeration (documented approximation): keywords, visible top-level fns/types,
/// and the enclosing fn's params + `:=` locals.
fn collectScope(
    col: *Collector,
    m: *const Graph.Module,
    entry: u32,
    res: *const ResolveGraph.GraphResult,
    tc: *const TypecheckGraph.GraphResult,
    off: u32,
) !void {
    for (toyc.token.keywords.keys()) |kw| try col.add(kw, kind.keyword, null);

    // Top-level user fns of the entry module. `recv_type == none` EXCLUDES inherent
    // methods (which also live in `res.fns`, module == entry) so a method never leaks
    // into plain scope as a free function.
    for (res.fns, 0..) |gf, i| {
        if (gf.kind != .user_fn or gf.module != entry or gf.recv_type != Ast.none) continue;
        var detail: ?[]const u8 = null;
        var buf: std.ArrayList(u8) = .empty;
        if (i < tc.sigs.len) {
            try render.renderSig(col.a, &buf, tc.sigs[i], tc.layouts, tc.enum_layouts);
            detail = buf.items;
        }
        try col.add(render.bareName(gf.name), kind.function, detail);
    }

    // Top-level type decls.
    const tree = m.tree();
    if (m.nodes.len != 0) {
        const prog = m.nodes[Ast.root(m.nodes).int()];
        if (prog.tag == .program) {
            for (Ast.rangeSlice(tree, prog.lhs.int())) |decl_idx| {
                const decl = m.nodes[decl_idx.int()];
                const name = m.tokens[decl.main_token].text(m.source);
                switch (decl.tag) {
                    .struct_decl, .tuple_struct_decl => try col.add(name, kind.@"struct", null),
                    .enum_decl => try col.add(name, kind.@"enum", null),
                    else => {},
                }
            }
        }
    }

    try collectEnclosing(col, m, entry, res, tc, off);
}

/// Add the enclosing fn's params + `:=` locals. The resolver keeps no positional scope,
/// so the enclosing fn is the last `fn_decl` starting at/before the cursor and its locals
/// are bracketed to `[fn.start, next_fn.start)`. Impl methods are interleaved `fn_decl`s,
/// handled by the next-fn bound.
fn collectEnclosing(
    col: *Collector,
    m: *const Graph.Module,
    entry: u32,
    res: *const ResolveGraph.GraphResult,
    tc: *const TypecheckGraph.GraphResult,
    off: u32,
) !void {
    var encl: ?u32 = null;
    var encl_start: u32 = 0;
    var next_start: u32 = std.math.maxInt(u32);
    for (m.nodes, 0..) |n, i| {
        if (n.tag != .fn_decl) continue;
        const start = m.tokens[n.main_token].start;
        if (start <= off and (encl == null or start >= encl_start)) {
            encl = @intCast(i);
            encl_start = start;
        }
    }
    if (encl == null) return;
    for (m.nodes) |n| {
        if (n.tag != .fn_decl) continue;
        const start = m.tokens[n.main_token].start;
        if (start > encl_start and start < next_start) next_start = start;
    }

    const fn_node = encl.?;
    const decl = m.nodes[fn_node];
    const proto = Ast.protoAt(m.tree(), decl.lhs.int());

    // Map the enclosing fn decl node → its global fn id, to read param types from the sig
    // (a param's own `node_types` slot is `.invalid`).
    var fid: ?u32 = null;
    for (res.fns, 0..) |gf, i| {
        if (gf.module == entry and gf.decl_node == Ast.Index.from(fn_node)) {
            fid = @intCast(i);
            break;
        }
    }

    for (proto.params, 0..) |param_idx, k| {
        const param = m.nodes[param_idx.int()];
        const name = m.tokens[param.main_token].text(m.source);
        var detail: ?[]const u8 = null;
        var buf: std.ArrayList(u8) = .empty;
        if (fid) |f| if (f < tc.sigs.len and k < tc.sigs[f].params.len) {
            try render.renderType(col.a, &buf, tc.sigs[f].params[k], tc.layouts, tc.enum_layouts);
            detail = buf.items;
        };
        try col.add(name, kind.variable, detail);
    }

    const node_types = tc.node_types[entry];
    for (m.nodes, 0..) |n, i| {
        if (n.tag != .var_decl) continue;
        const start = m.tokens[n.main_token].start;
        if (start < encl_start or start >= next_start) continue;
        const name = m.tokens[n.main_token].text(m.source);
        var detail: ?[]const u8 = null;
        var buf: std.ArrayList(u8) = .empty;
        if (i < node_types.len and isRenderable(node_types[i])) {
            try render.renderType(col.a, &buf, node_types[i], tc.layouts, tc.enum_layouts);
            detail = buf.items;
        }
        try col.add(name, kind.variable, detail);
    }
}

fn isRenderable(t: Type) bool {
    return t.kind != .invalid and t.kind != .type_var and t.kind != .app;
}

const testing = std.testing;

fn tok(tag: toyc.Tag, start: u32, end: u32) Token {
    return .{ .tag = tag, .start = start, .end = end };
}

test "classify: trailing dot is a member context needing repair" {
    // `foo.` — foo[0,3) dot[3,4) eof[4,4). Cursor at 4 (after the dot).
    const toks = [_]Token{ tok(.identifier, 0, 3), tok(.dot, 3, 4), tok(.eof, 4, 4) };
    const d = classify(&toks, 4);
    try testing.expectEqual(Context.member, d.context);
    try testing.expect(d.repair);
}

test "classify: partial field name is a member context located by start" {
    // `foo.ba` — foo[0,3) dot[3,4) ba[4,6) eof. Cursor at 6 (after `ba`).
    const toks = [_]Token{ tok(.identifier, 0, 3), tok(.dot, 3, 4), tok(.identifier, 4, 6), tok(.eof, 6, 6) };
    const d = classify(&toks, 6);
    try testing.expectEqual(Context.member, d.context);
    try testing.expect(!d.repair);
    try testing.expectEqual(@as(u32, 4), d.partial_start);
}

test "classify: a bare identifier is a scope context" {
    // `he` — he[0,2) eof. Cursor at 2.
    const toks = [_]Token{ tok(.identifier, 0, 2), tok(.eof, 2, 2) };
    const d = classify(&toks, 2);
    try testing.expectEqual(Context.scope, d.context);
}

test "classify: a non-name/non-dot preceding token yields no context" {
    // `(` — l_paren[0,1) eof. Cursor at 1.
    const toks = [_]Token{ tok(.l_paren, 0, 1), tok(.eof, 1, 1) };
    try testing.expectEqual(Context.none, classify(&toks, 1).context);
    // Cursor at 0 (nothing before) is also none.
    try testing.expectEqual(Context.none, classify(&toks, 0).context);
}

test "classify: newline terminators are skipped back to the real token" {
    // `foo.` then a newline, cursor on the next line — the dot is still the real preceding.
    const toks = [_]Token{ tok(.identifier, 0, 3), tok(.dot, 3, 4), tok(.newline, 4, 4), tok(.eof, 8, 8) };
    const d = classify(&toks, 6);
    try testing.expectEqual(Context.member, d.context);
    try testing.expect(d.repair);
}
