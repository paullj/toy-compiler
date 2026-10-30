//! textDocument/completion: the candidates at a cursor position.
//!
//! Three contexts are served: SCOPE (a bare identifier position → in-scope locals /
//! params / visible top-level fns + types + keywords), MEMBER (`recv.` → the receiver
//! type's fields + methods; only methods after `x |> .`), and MODULE (`imported_ns.` →
//! that module's pub members).
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
//!     the params of the top-level fn around the cursor + every `:=` local and local fn in
//!     it (wherever declared) + all visible top-level decls + the language keywords. Outside
//!     any fn's source, only the top-level decls and keywords.
//!   * Scalar / `str` builtin methods are not surfaced; a generic type's methods show as
//!     declared (`fn map(self: Vec[T], f: fn(T) -> U) -> Vec[U]`), not over the receiver's
//!     args.

const std = @import("std");
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const position = @import("position.zig");
const render = @import("render.zig");
const Analysis = @import("Analysis.zig");
const Workspace = @import("Workspace.zig");

const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const Token = toyc.Token;
const Ast = toyc.Ast;
const Type = toyc.Typecheck.Type;

const kind = protocol.completion_kind;

/// The probe identifier spliced after a trailing `.` so the buffer parses. Long +
/// underscored so it can never collide with a real toy identifier the user typed.
const probe = "__toy_completion_probe__";

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
    /// A `x |> .m(..)` step: only a method can follow its `.`.
    methods_only: bool = false,
};

/// The completion candidates at (`line`, `character`) in `doc`'s document, allocated in
/// `a`. A repaired buffer is checked with `gpa` in `ws` and torn down before this returns.
/// An empty list is valid.
pub fn at(a: std.mem.Allocator, gpa: std.mem.Allocator, ws: Workspace, doc: *Analysis, line: u32, character: u32) ![]const protocol.CompletionItem {
    const source = doc.text;
    // Detect the context from a fault-tolerant raw lex — the parse tree may not exist yet.
    const toks = try toyc.Lexer.tokenize(a, source);
    const off = doc.offsetAt(line, character) orelse return &.{};
    const det = classify(toks, off);
    if (det.context == .none) return &.{};

    // Repair a trailing `.` so the access parses; otherwise read the document as it is.
    if (!det.repair) return collect(a, doc, det, toks, off);
    const repaired = try std.mem.concat(a, u8, &.{ source[0..off], probe, source[off..] });
    const an = try Analysis.init(gpa, ws, doc.uri, repaired, doc.version);
    defer an.deinit();
    return collect(a, an, det, toks, off);
}

fn collect(a: std.mem.Allocator, an: *Analysis, det: Detected, toks: []const Token, off: u32) ![]const protocol.CompletionItem {
    // A parse error other than the repaired cursor discards the tree — nothing to offer.
    // Unlike hover, resolve and type errors do not bail: an unknown probe member or a
    // mistyped line elsewhere is expected mid-edit, and the receiver still resolves and types.
    if (an.checked == null) return &.{};
    var col: Collector = .{ .a = a };
    switch (det.context) {
        .none => {},
        .scope => try collectScope(&col, an, toks, off),
        .member => {
            const recv = locateReceiver(an.module(), det) orelse return &.{};
            try collectMember(&col, an, recv, det.methods_only);
        },
    }
    return col.items.toOwnedSlice(a);
}

/// Accumulates candidates into the result arena, de-duping by `(label, kind)`. Every
/// label/detail is arena-`dupe`d here because the sources (method/sig/layout names, token
/// text) are BORROWED from the compiler results, which are torn down before we return.
const Collector = struct {
    a: std.mem.Allocator,
    items: std.ArrayList(protocol.CompletionItem) = .empty,

    fn add(c: *Collector, label: []const u8, k: u8, detail: ?[]const u8) !void {
        return c.addSorted(label, k, detail, null);
    }

    fn addSorted(c: *Collector, label: []const u8, k: u8, detail: ?[]const u8, sort_text: ?[]const u8) !void {
        for (c.items.items) |it| {
            if (it.kind == k and std.mem.eql(u8, it.label, label)) return;
        }
        try c.items.append(c.a, .{
            .label = try c.a.dupe(u8, label),
            .kind = k,
            .detail = if (detail) |d| try c.a.dupe(u8, d) else null,
            .sortText = if (sort_text) |t| try c.a.dupe(u8, t) else null,
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
        .dot => return .{ .context = .member, .repair = true, .methods_only = pipedAt(tokens, p) },
        .identifier => {
            const q0 = lastBefore(tokens, tokens[p].start) orelse return .{ .context = .scope };
            const q = skipNewlineBack(tokens, q0) orelse return .{ .context = .scope };
            if (tokens[q].tag == .dot)
                return .{ .context = .member, .partial_start = tokens[p].start, .methods_only = pipedAt(tokens, q) };
            return .{ .context = .scope };
        },
        else => return .{ .context = .none },
    }
}

/// Whether the `.` at `dot` follows a `|>`.
fn pipedAt(tokens: []const Token, dot: usize) bool {
    if (dot == 0) return false;
    const p = skipNewlineBack(tokens, dot - 1) orelse return false;
    return tokens[p].tag == .pipe_gt;
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
fn collectMember(col: *Collector, an: *const Analysis, recv: u32, methods_only: bool) !void {
    const res = an.res();
    const tc = an.tc();
    const resolutions = an.resolutions();
    if (recv < resolutions.len and resolutions[recv] == .module) {
        // A module is no value, so it cannot be piped.
        if (!methods_only) try collectModule(col, an.graph.modules, res, resolutions[recv].module);
        return;
    }

    const node_types = an.nodeTypes();
    if (recv >= node_types.len) return;
    const rt = node_types[recv];
    if (rt.kind != .@"struct" and rt.kind != .@"enum") return;

    // A field private to another module is still offered, so the user learns it exists,
    // but marked and sorted after the rest: T0078 rejects its use.
    if (!methods_only and rt.kind == .@"struct" and rt.struct_id < tc.layouts.len) {
        const layout = tc.layouts[rt.struct_id];
        for ([_]bool{ false, true }) |want_hidden| {
            for (layout.field_names, layout.field_types, 0..) |fname, fty, j| {
                const f = an.fieldOf(rt.struct_id, j);
                const hidden = if (f) |fd| !an.fieldVisible(fd) else false;
                if (hidden != want_hidden) continue;
                var buf: std.ArrayList(u8) = .empty;
                try render.renderType(col.a, &buf, fty, tc.layouts, tc.enum_layouts);
                const label = render.bareName(fname);
                if (!hidden) {
                    try col.add(label, kind.field, buf.items);
                    continue;
                }
                try buf.print(col.a, " (private to module {s})", .{an.graph.modules[f.?.module].path});
                // `~` orders after every identifier, as a client sorts by sortText, else label.
                try col.addSorted(label, kind.field, buf.items, try std.mem.concat(col.a, u8, &.{ "~", label }));
            }
        }
    }

    // A generic type's methods are its templates, shown as declared: an instance row
    // exists only for a method some body already calls on these args.
    const template = templateOf(tc, rt);
    if (template) |tm| for (tc.templates) |t| {
        if (t.recv_ctor != tm.ctor or t.recv_is_enum != tm.is_enum or !t.has_self or t.fn_id >= res.fns.len) continue;
        const gf = res.fns[t.fn_id];
        if (gf.decl_node == Ast.none) continue;
        var buf: std.ArrayList(u8) = .empty;
        try render.appendFnHeader(col.a, &buf, &an.graph.modules[gf.module], gf.decl_node.int(), t.name);
        try col.add(t.name, kind.method, buf.items);
    };

    for (tc.methods) |mth| {
        if (!Type.eql(mth.recv, rt) or mth.is_static) continue;
        if (template != null and mth.instance != null) continue;
        var detail: ?[]const u8 = null;
        var buf: std.ArrayList(u8) = .empty;
        // The fn, instance or derive the method runs; its row names it as the user does.
        if (toyc.Typecheck.witnessSig(mth, tc.derives, tc.instances, tc.sigs)) |sig| {
            try render.renderSig(col.a, &buf, .{ .name = mth.name, .params = sig.params, .ret = sig.ret, .mut_mask = sig.mut_mask }, tc.layouts, tc.enum_layouts);
            detail = buf.items;
        }
        try col.add(render.bareName(mth.name), kind.method, detail);
    }
}

/// The key of `rt`'s method templates: a non-generic type itself when it has some, else
/// the generic type `rt` is an instance of, or null for a type that is no instance.
fn templateOf(tc: *const toyc.TypecheckGraph.GraphResult, rt: Type) ?struct { ctor: u32, is_enum: bool } {
    for (tc.templates) |t| if (t.recv_ctor == rt.nominalId() and t.recv_is_enum == (rt.kind == .@"enum")) return .{ .ctor = t.recv_ctor, .is_enum = t.recv_is_enum };
    // A method some body calls on `rt` names its template exactly.
    for (tc.methods) |mth| {
        if (!Type.eql(mth.recv, rt)) continue;
        const ii = mth.instance orelse continue;
        if (ii >= tc.instances.len) continue;
        for (tc.templates) |t| if (t.fn_id == tc.instances[ii].template_gid) return .{ .ctor = t.recv_ctor, .is_enum = t.recv_is_enum };
    }
    // Else by name: an instance is named `Template$args`, and only an unambiguous name
    // counts.
    const is_enum = rt.kind == .@"enum";
    if (if (is_enum) rt.enum_id >= tc.enum_layouts.len else rt.struct_id >= tc.layouts.len) return null;
    const name = render.bareName(if (is_enum) tc.enum_layouts[rt.enum_id].name else tc.layouts[rt.struct_id].name);
    const base = name[0 .. std.mem.indexOfScalar(u8, name, '$') orelse return null];
    var found: ?u32 = null;
    for (tc.templates) |t| {
        if (t.recv_is_enum != is_enum or t.recv_ctor >= (if (is_enum) tc.enum_layouts.len else tc.layouts.len)) continue;
        const tname = if (is_enum) tc.enum_layouts[t.recv_ctor].name else tc.layouts[t.recv_ctor].name;
        if (!std.mem.eql(u8, render.bareName(tname), base)) continue;
        if (found) |f| if (f != t.recv_ctor) return null;
        found = t.recv_ctor;
    }
    return .{ .ctor = found orelse return null, .is_enum = is_enum };
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
fn collectScope(col: *Collector, an: *const Analysis, toks: []const Token, off: u32) !void {
    const m = an.module();
    const res = an.res();
    const tc = an.tc();
    for (toyc.token.keywords.keys()) |kw| try col.add(kw, kind.keyword, null);

    // Top-level user fns of the entry module. `recv_type == none` EXCLUDES inherent
    // methods (which also live in `res.fns`, module == entry) so a method never leaks
    // into plain scope as a free function.
    for (res.fns, 0..) |gf, i| {
        if (gf.kind != .user_fn or gf.module != an.entry() or gf.recv_type != Ast.none) continue;
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

    // The name being typed is the token before the cursor; the raw lex and the checked
    // buffer agree on it, since a scope context is never repaired.
    const typed = lastBefore(toks, off) orelse return;
    try collectEnclosing(col, an, @intCast(typed));
}

/// Add the params of each lambda and local fn around the cursor, then the enclosing fn's
/// params + `:=` locals + local fns. The resolver keeps no positional scope, so every local
/// of the fn is offered, wherever it is declared.
fn collectEnclosing(col: *Collector, an: *const Analysis, typed: u32) !void {
    const m = an.module();
    const tc = an.tc();
    var scope = an.scopeOf(typed);
    while (scope) |sc| : (scope = an.scopeParent(sc)) {
        if (m.nodes[sc].tag == .fn_decl) break;
        // A closure's param types are its own fn type's: its params' slots stay untyped.
        const fs = render.fnSigOf(an.nodeTypes()[sc], tc.layouts);
        for (Ast.protoAt(m.tree(), m.nodes[sc].lhs.int()).params, 0..) |p, k| {
            var detail: ?[]const u8 = null;
            var buf: std.ArrayList(u8) = .empty;
            if (fs) |f| if (k < f.params.len and render.isRenderable(f.params[k])) {
                try render.renderType(col.a, &buf, f.params[k], tc.layouts, tc.enum_layouts);
                detail = buf.items;
            };
            try col.add(m.tokens[m.nodes[p.int()].main_token].text(m.source), kind.variable, detail);
        }
    }
    const fn_node = an.fnOf(typed) orelse return;
    const decl = m.nodes[fn_node];
    const proto = Ast.protoAt(m.tree(), decl.lhs.int());

    // Param types come from the sig: a param's own `node_types` slot is `.invalid`.
    const fid = an.gidOf(fn_node);
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

    for (m.nodes, 0..) |n, i| {
        const k = switch (n.tag) {
            .var_decl => kind.variable,
            .local_fn => kind.function,
            else => continue,
        };
        if (an.fnOf(n.main_token) != fn_node) continue;
        const name = m.tokens[n.main_token].text(m.source);
        var detail: ?[]const u8 = null;
        var buf: std.ArrayList(u8) = .empty;
        if (an.typeAt(@intCast(i))) |t| {
            try render.renderType(col.a, &buf, t, tc.layouts, tc.enum_layouts);
            detail = buf.items;
        }
        try col.add(name, k, detail);
    }
}

const testing = std.testing;

/// A test's candidates: their analysis and the answer, torn down together.
const Completions = struct {
    an: *Analysis,
    arena: std.heap.ArenaAllocator,
    items: []const protocol.CompletionItem,

    fn deinit(c: *Completions) void {
        c.arena.deinit();
        c.an.deinit();
    }
};

/// The candidates at (`line`, `character`) in `source`, checked as the document `doc_uri`
/// in `ws`.
fn completionsAt(gpa: std.mem.Allocator, ws: Workspace, source: []const u8, line: u32, character: u32, doc_uri: []const u8) !Completions {
    const an = try @import("test_util.zig").analyzeIn(gpa, ws, doc_uri, source);
    errdefer an.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const items = try at(arena.allocator(), gpa, ws, an, line, character);
    return .{ .an = an, .arena = arena, .items = items };
}

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

test "completion: a local fn before the cursor keeps the enclosing fn's params and locals" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\fn helper(x: int) -> int { return x }
        \\fn main(arg: int) -> int {
        \\    base := 1
        \\    fn step(k: int) -> int { k + base }
        \\    count := step(arg)
        \\    return count
        \\}
        \\
    ;
    const p = @import("test_util.zig").posAfterLast(src, "count");
    var c = try completionsAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy");
    defer c.deinit();
    const Want = struct { label: []const u8, kind: u8, detail: ?[]const u8 };
    const wants = [_]Want{
        .{ .label = "arg", .kind = kind.variable, .detail = "int" },
        .{ .label = "base", .kind = kind.variable, .detail = "int" },
        .{ .label = "count", .kind = kind.variable, .detail = "int" },
        .{ .label = "step", .kind = kind.function, .detail = "fn(int) -> int" },
        .{ .label = "helper", .kind = kind.function, .detail = null },
    };
    for (wants) |w| {
        const it = for (c.items) |it| {
            if (std.mem.eql(u8, it.label, w.label)) break it;
        } else {
            std.debug.print("no completion '{s}'\n", .{w.label});
            return error.MissingCompletion;
        };
        try testing.expectEqual(w.kind, it.kind);
        if (w.detail) |d| try testing.expectEqualStrings(d, it.detail.?);
    }
    // Another fn's param and a local fn's own param are not in scope at the cursor.
    for (c.items) |it| {
        try testing.expect(!std.mem.eql(u8, it.label, "x"));
        try testing.expect(!std.mem.eql(u8, it.label, "k"));
    }
}

/// The candidates at `@` in `src`, whose `@` is dropped, with no other document open.
fn completeAt(gpa: std.mem.Allocator, marked: []const u8) !Completions {
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const cut = std.mem.indexOfScalar(u8, marked, '@').?;
    const src = try std.mem.concat(gpa, u8, &.{ marked[0..cut], marked[cut + 1 ..] });
    defer gpa.free(src);
    const p = @import("test_util.zig").posAfterLast(src, marked[0..cut]);
    return completionsAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy");
}

fn detailOf(items: []const protocol.CompletionItem, label: []const u8) ?[]const u8 {
    for (items) |it| if (std.mem.eql(u8, it.label, label)) return it.detail orelse "";
    return null;
}

const gaps_fixture =
    \\import std/vec
    \\struct Acc { n: int }
    \\impl Acc {
    \\    fn plus(self, k: int) -> int { self.n + k }
    \\    fn zero() -> Acc { Acc { n: 0 } }
    \\}
    \\protocol Area { fn area(self) -> int }
    \\struct Sq { side: int }
    \\impl Sq has Area { fn area(self) -> int { self.side * self.side } }
    \\fn apply(f: fn(int, str) -> int) -> int { f(1, "s") }
    \\fn main() -> int {
    \\    acc := Acc { n: 1 }
    \\    xs := [1, 2]
    \\    sq := Sq { side: 2 }
    \\
;

test "completion inside a lambda or a local fn offers its params, and none outside it" {
    const gpa = testing.allocator;
    const cases = [_]struct { tail: []const u8, has: []const []const u8, not: []const []const u8 }{
        .{ .tail = "    k := apply(fn(a, b) { a@ })\n    return k\n}\n", .has = &.{ "a", "b", "acc" }, .not = &.{"z"} },
        .{ .tail = "    k := apply(fn(a, b) {\n        fn inner(z: int) -> int { z@ }\n        inner(a)\n    })\n    return k\n}\n", .has = &.{ "z", "a", "b", "inner" }, .not = &.{} },
        .{ .tail = "    k := apply(fn(a, b) { a })\n    return k@\n}\n", .has = &.{"acc"}, .not = &.{ "a", "b" } },
    };
    for (cases) |c| {
        const src = try std.mem.concat(gpa, u8, &.{ gaps_fixture, c.tail });
        defer gpa.free(src);
        var got = try completeAt(gpa, src);
        defer got.deinit();
        for (c.has) |l| if (detailOf(got.items, l) == null) {
            std.debug.print("no '{s}' in '{s}'\n", .{ l, c.tail });
            return error.MissingCompletion;
        };
        for (c.not) |l| try testing.expect(detailOf(got.items, l) == null);
    }
    const src = try std.mem.concat(gpa, u8, &.{ gaps_fixture, cases[0].tail });
    defer gpa.free(src);
    var got = try completeAt(gpa, src);
    defer got.deinit();
    try testing.expectEqualStrings("int", detailOf(got.items, "a").?);
    try testing.expectEqualStrings("str", detailOf(got.items, "b").?);
}

test "completion after a method pipe offers methods only" {
    const gpa = testing.allocator;
    for ([_][]const u8{ "    k := acc |> .@\n    return 0\n}\n", "    k := acc\n        |> .pl@\n    return 0\n}\n" }) |tail| {
        const src = try std.mem.concat(gpa, u8, &.{ gaps_fixture, tail });
        defer gpa.free(src);
        var got = try completeAt(gpa, src);
        defer got.deinit();
        try testing.expectEqualStrings("fn plus(Acc, int) -> int", detailOf(got.items, "plus").?);
        try testing.expect(detailOf(got.items, "n") == null);
    }
}

test "completion offers a generic std type's methods as declared, and no associated fn on a value" {
    const gpa = testing.allocator;
    const src = try std.mem.concat(gpa, u8, &.{ gaps_fixture, "    k := xs.@\n    return 0\n}\n" });
    defer gpa.free(src);
    var got = try completeAt(gpa, src);
    defer got.deinit();
    const want = [_]struct { label: []const u8, detail: []const u8 }{
        .{ .label = "map", .detail = "fn map(self: Vec[T], f: fn(T) -> U) -> Vec[U]" },
        .{ .label = "filter", .detail = "fn filter(self: Vec[T], f: fn(T) -> bool) -> Vec[T]" },
        .{ .label = "fold", .detail = "fn fold(self: Vec[T], init: A, f: fn(A, T) -> A) -> A" },
        .{ .label = "each", .detail = "fn each(self: Vec[T], f: fn(T)) -> ()" },
        // Called by the list literal, so an instance row exists; it still shows as declared.
        .{ .label = "push", .detail = "fn push(self: Vec[T], x: T) -> ()" },
        .{ .label = "sort", .detail = "fn sort(self: Vec[T]) -> ()" },
    };
    for (want) |w| try testing.expectEqualStrings(w.detail, detailOf(got.items, w.label) orelse {
        std.debug.print("no '{s}'\n", .{w.label});
        return error.MissingCompletion;
    });
    try testing.expect(detailOf(got.items, "new") == null);

    const src2 = try std.mem.concat(gpa, u8, &.{ gaps_fixture, "    k := acc.@\n    return 0\n}\n" });
    defer gpa.free(src2);
    var got2 = try completeAt(gpa, src2);
    defer got2.deinit();
    try testing.expect(detailOf(got2.items, "plus") != null);
    try testing.expect(detailOf(got2.items, "zero") == null);
}

test "completion lists a non-generic type's own-generic method once, as declared" {
    const gpa = testing.allocator;
    var got = try completeAt(gpa,
        \\struct K { v: int }
        \\impl K {
        \\    fn map[U](self, f: fn(int) -> U) -> U { f(self.v) }
        \\    fn with[U](v: int, _u: U) -> K { K { v: v } }
        \\    fn get(self) -> int { self.v }
        \\}
        \\fn main() -> int {
        \\    k := K.with(1, true)
        \\    _a := k.map(fn(x) { x + 1 })
        \\    _b := k.map(fn(x) { x > 1 })
        \\    n := k.@
        \\    return 0
        \\}
        \\
    );
    defer got.deinit();
    var maps: usize = 0;
    for (got.items) |it| maps += @intFromBool(std.mem.eql(u8, it.label, "map"));
    try testing.expectEqual(@as(usize, 1), maps);
    try testing.expectEqualStrings("fn map(self: K, f: fn(int) -> U) -> U", detailOf(got.items, "map").?);
    try testing.expectEqualStrings("fn get(K) -> int", detailOf(got.items, "get").?);
    try testing.expect(detailOf(got.items, "with") == null);

    // With no call, no instance row names the template.
    var bare = try completeAt(gpa,
        \\struct K { v: int }
        \\impl K {
        \\    fn map[U](self, f: fn(int) -> U) -> U { f(self.v) }
        \\}
        \\fn main() -> int {
        \\    k := K { v: 1 }
        \\    n := k.@
        \\    return 0
        \\}
        \\
    );
    defer bare.deinit();
    try testing.expectEqualStrings("fn map(self: K, f: fn(int) -> U) -> U", detailOf(bare.items, "map").?);
}

test "completion names a protocol witness as written" {
    const gpa = testing.allocator;
    const src = try std.mem.concat(gpa, u8, &.{ gaps_fixture, "    k := sq.@\n    return 0\n}\n" });
    defer gpa.free(src);
    var got = try completeAt(gpa, src);
    defer got.deinit();
    try testing.expectEqualStrings("fn area(Sq) -> int", detailOf(got.items, "area").?);
}

test "completion outside a struct's module lists its private fields after its pub ones, marked private" {
    const gpa = testing.allocator;
    const tu = @import("test_util.zig");
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws = try tu.privacyWorkspace(gpa, &docs);
    const Want = struct { label: []const u8, detail: []const u8, last: bool };
    const cases = [_]struct { recv: []const u8, fields: []const Want }{
        .{ .recv = "p", .fields = &.{
            .{ .label = "x", .detail = "int", .last = false },
            .{ .label = "y", .detail = "int (private to module lib)", .last = true },
        } },
        // In its own module every field is visible: declaration order, no mark.
        .{ .recv = "q", .fields = &.{
            .{ .label = "b", .detail = "int", .last = false },
            .{ .label = "a", .detail = "int", .last = false },
        } },
    };
    const ret = std.mem.indexOf(u8, tu.privacy_main, "    return").?;
    for (cases) |c| {
        const line = try std.fmt.allocPrint(gpa, "    k := {s}.", .{c.recv});
        defer gpa.free(line);
        const src = try std.mem.concat(gpa, u8, &.{ tu.privacy_main[0..ret], line, "\n", tu.privacy_main[ret..] });
        defer gpa.free(src);
        const p = tu.posAfterLast(src, line);
        var got = try completionsAt(gpa, ws, src, p.line, p.character, tu.privacy_main_uri);
        defer got.deinit();
        var fields: std.ArrayList(protocol.CompletionItem) = .empty;
        defer fields.deinit(gpa);
        for (got.items) |it| if (it.kind == kind.field) try fields.append(gpa, it);
        try testing.expectEqual(c.fields.len, fields.items.len);
        for (c.fields, fields.items) |w, it| {
            try testing.expectEqualStrings(w.label, it.label);
            try testing.expectEqualStrings(w.detail, it.detail.?);
            if (!w.last) {
                try testing.expect(it.sortText == null);
                continue;
            }
            // A client orders by sortText, else by label: this one sorts after every other.
            for (got.items) |o| if (o.kind != it.kind or !std.mem.eql(u8, o.label, it.label))
                try testing.expect(std.mem.order(u8, o.sortText orelse o.label, it.sortText.?) == .lt);
        }
    }
}
