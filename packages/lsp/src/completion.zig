//! textDocument/completion: the candidates at a cursor position.
//!
//! Five contexts are served: SCOPE (a bare identifier position → in-scope locals /
//! params / visible top-level fns + types + keywords), MEMBER (`recv.` → the receiver
//! type's fields + methods; only methods after `x |> .`), MODULE (`imported_ns.` →
//! that module's pub members, re-exports included), IMPORT PATH (`import a/` → the next
//! path segment: a module or a folder in any search root, or the bundled std) and IMPORT
//! ITEM (`import a/b.{` → what module `a/b` exports).
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
const lsp_uri = @import("uri.zig");
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
    if (importSite(toks, source, off)) |site| return switch (site) {
        .path => |prefix| collectModulePaths(a, ws, doc.uri, try std.mem.join(a, "/", prefix.slice())),
        .item => |path| collectImportItems(a, gpa, ws, doc.uri, try std.mem.join(a, "/", path.slice())),
        .alias => &.{},
    };
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
        if (!methods_only) try collectExports(col, &an.graph, resolutions[recv].module);
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

/// What module `mod` exports, as an importer names it: its pub items, and the names its
/// `pub import`s bind (a namespace, or a selected item followed to its declaration).
fn collectExports(col: *Collector, g: *const Graph.Graph, mod: u32) !void {
    const m = &g.modules[mod];
    if (m.nodes.len == 0) return;
    const tree = m.tree();
    const prog = m.nodes[Ast.root(m.nodes).int()];
    if (prog.tag != .program) return;
    for (Ast.rangeSlice(tree, prog.lhs.int())) |d| {
        if (!tree.isPub(d)) continue;
        const decl = m.nodes[d.int()];
        switch (decl.tag) {
            .import_decl => {
                const bound = Ast.importAliasTok(decl).unwrap() orelse Ast.TokIndex.from(decl.main_token);
                try col.add(m.tokens[bound.int()].text(m.source), kind.module, null);
            },
            .import_select_decl => for (Ast.importItems(tree, decl)) |item| {
                const label = m.tokens[item.bound().int()].text(m.source);
                const ex = g.exportOf(mod, label) orelse continue;
                try addItem(col, g, ex.mod, ex.decl, label);
            },
            else => try addItem(col, g, mod, d, m.tokens[decl.main_token].text(m.source)),
        }
    }
}

/// Item `decl` of module `mod` (a namespace when none) under the name `label`. A fn shows
/// its header as written: a generic std fn's checked types would render as `?`.
fn addItem(col: *Collector, g: *const Graph.Graph, mod: u32, decl: Ast.Index, label: []const u8) !void {
    if (decl == Ast.none) return col.add(label, kind.module, null);
    const m = &g.modules[mod];
    switch (m.nodes[decl.int()].tag) {
        .fn_decl => {
            var buf: std.ArrayList(u8) = .empty;
            try render.appendFnHeader(col.a, &buf, m, decl.int(), label);
            try col.add(label, kind.function, buf.items);
        },
        .extern_fn_decl => try col.add(label, kind.function, null),
        .struct_decl, .tuple_struct_decl, .type_alias_decl => try col.add(label, kind.@"struct", null),
        .enum_decl => try col.add(label, kind.@"enum", null),
        .protocol_decl => try col.add(label, kind.interface, null),
        else => {},
    }
}

/// Where the cursor sits in an import: typing a path (`import a/|`, the complete segments
/// before it), in a selection list (`import a/b.{x, |`, the module's path), or naming an
/// alias (`{x as |`), which is a new name and has nothing to offer.
const ImportSite = union(enum) {
    path: Segments,
    item: Segments,
    alias,
};

const Segments = struct {
    buf: [16][]const u8 = undefined,
    len: usize = 0,

    fn slice(s: *const Segments) []const []const u8 {
        return s.buf[0..s.len];
    }

    /// Prepend: the tokens are walked back from the cursor.
    fn push(s: *Segments, seg: []const u8) bool {
        if (s.len == s.buf.len) return false;
        std.mem.copyBackwards([]const u8, s.buf[1 .. s.len + 1], s.buf[0..s.len]);
        s.buf[0] = seg;
        s.len += 1;
        return true;
    }
};

/// The import site at `off`, read off the raw lex, or null outside an import.
fn importSite(tokens: []const Token, source: []const u8, off: u32) ?ImportSite {
    var p = lastBefore(tokens, off) orelse return null;
    // A name being typed is what the candidate replaces; its context is what precedes it.
    if (tokens[p].tag == .identifier and tokens[p].end >= off) p = std.math.sub(usize, p, 1) catch return null;
    if (tokens[p].tag == .kw_as) {
        return if (selectionPath(tokens, source, p) != null) .alias else null;
    }
    if (importPath(tokens, source, p, .kw_import)) |segs| return .{ .path = segs };
    if (selectionPath(tokens, source, p)) |segs| return .{ .item = segs };
    return null;
}

/// The path segments of `import a/b/` ending at token `last` (a `/`, or `import` itself for
/// an empty path), when the token before the path is `head`.
fn importPath(tokens: []const Token, source: []const u8, last: usize, head: toyc.Tag) ?Segments {
    var segs: Segments = .{};
    var i = last;
    while (true) {
        if (tokens[i].tag == head) return segs;
        if (tokens[i].tag != .slash or i < 2 or tokens[i - 1].tag != .identifier) return null;
        if (!segs.push(tokens[i - 1].text(source))) return null;
        i -= 2;
    }
}

/// The module path of the selection list `import a/b.{..` that token `last` sits in.
fn selectionPath(tokens: []const Token, source: []const u8, last: usize) ?Segments {
    var i = last;
    while (true) : (i -= 1) {
        switch (tokens[i].tag) {
            .identifier, .comma, .kw_as, .newline => {},
            .l_brace => break,
            else => return null,
        }
        if (i == 0) return null;
    }
    if (i < 3 or tokens[i - 1].tag != .dot or tokens[i - 2].tag != .identifier) return null;
    // The last segment has no `/` after it: read it, then the rest as a path.
    const segs = if (tokens[i - 3].tag == .kw_import) Segments{} else if (tokens[i - 3].tag == .slash)
        importPath(tokens, source, i - 3, .kw_import) orelse return null
    else
        return null;
    var out = segs;
    if (out.len == out.buf.len) return null;
    out.buf[out.len] = tokens[i - 2].text(source);
    out.len += 1;
    return out;
}

/// What module `path` exports, read from a graph that imports only it, so the list is
/// served while the buffer's own selection list does not parse yet.
fn collectImportItems(a: std.mem.Allocator, gpa: std.mem.Allocator, ws: Workspace, uri: []const u8, path: []const u8) ![]const protocol.CompletionItem {
    // `core/` is private to the bundle (R0013), a resolve error discovery does not raise.
    if (std.mem.startsWith(u8, path, "core/")) return &.{};
    const src = try std.fmt.allocPrint(a, "import {s}\n", .{path});
    var g = try ws.discover(gpa, uri, src);
    defer g.deinit(gpa);
    if (g.err != null) return &.{};
    var col: Collector = .{ .a = a };
    for (g.modules, 0..) |m, i| if (std.mem.eql(u8, m.path, path)) {
        try collectExports(&col, &g, @intCast(i));
        break;
    };
    return col.items.toOwnedSlice(a);
}

/// The path segments that can follow `prefix` (`/`-joined, empty at the start) in an
/// import: from the bundled std, then from every search root, its open documents and, with
/// a disk, its files. A segment that names a module (`x.toy`, or a folder with `mod.toy`)
/// is a module; a folder that holds modules, itself or one folder down, is a folder.
fn collectModulePaths(a: std.mem.Allocator, ws: Workspace, uri: []const u8, prefix: []const u8) ![]const protocol.CompletionItem {
    var col: Collector = .{ .a = a };
    var bundled = Graph.bundledPaths();
    // `core/` is private to the bundle (R0013): a user module cannot import it.
    while (bundled.next()) |p| if (!std.mem.startsWith(u8, p, "core/")) try addModulePath(&col, p, prefix);

    // A uri with no file has no folder to root imports at (see `Workspace.discover`).
    const doc_path = (try lsp_uri.uriToPath(a, uri)) orelse return col.items.toOwnedSlice(a);
    var roots: std.ArrayList([]const u8) = .empty;
    try roots.append(a, ws.roots.root orelse std.fs.path.dirname(doc_path) orelse ".");
    try roots.appendSlice(a, ws.roots.include);

    for (roots.items) |unresolved| {
        // A configured root may be spelled `dir/.`; paths are compared as text below.
        const root = try std.fs.path.resolve(a, &.{unresolved});
        var docs = ws.docs.map.keyIterator();
        while (docs.next()) |k| {
            const path = (try lsp_uri.uriToPath(a, k.*)) orelse continue;
            if (std.mem.eql(u8, path, doc_path)) continue;
            const rel = relativeTo(root, path) orelse continue;
            if (!std.mem.endsWith(u8, rel, Graph.ext)) continue;
            try addModulePath(&col, moduleOf(rel[0 .. rel.len - Graph.ext.len]), prefix);
        }
        if (ws.disk) try listDir(&col, ws, a, root, prefix, doc_path);
    }
    return col.items.toOwnedSlice(a);
}

/// `path` relative to the folder `root`, or null outside it.
fn relativeTo(root: []const u8, path: []const u8) ?[]const u8 {
    const r = std.mem.trimEnd(u8, root, "/");
    if (!std.mem.startsWith(u8, path, r) or path.len <= r.len + 1 or path[r.len] != '/') return null;
    return path[r.len + 1 ..];
}

/// The module a root-relative file stem is: `util/mod` is the folder module `util`.
fn moduleOf(stem: []const u8) []const u8 {
    const tail = "/" ++ Graph.folder_entry;
    if (std.mem.endsWith(u8, stem, tail)) return stem[0 .. stem.len - tail.len];
    return stem;
}

/// The segment of module `path` after `prefix`, if `path` lies under it.
fn addModulePath(col: *Collector, path: []const u8, prefix: []const u8) !void {
    const rest = if (prefix.len == 0) path else blk: {
        if (!std.mem.startsWith(u8, path, prefix) or path.len <= prefix.len or path[prefix.len] != '/') return;
        break :blk path[prefix.len + 1 ..];
    };
    const end = std.mem.indexOfScalar(u8, rest, '/');
    try addSegment(col, rest[0 .. end orelse rest.len], end == null);
}

/// One path segment; a name that is a module and a folder of modules too shows once, as
/// the module.
fn addSegment(col: *Collector, seg: []const u8, is_module: bool) !void {
    // A folder module's entry file is imported by its folder's path, but a folder named
    // `mod` can sit mid-path.
    if (!isSegment(seg) or (is_module and std.mem.eql(u8, seg, Graph.folder_entry))) return;
    for (col.items.items) |*it| if (std.mem.eql(u8, it.label, seg)) {
        if (is_module) it.kind = kind.module;
        return;
    };
    try col.add(seg, if (is_module) kind.module else kind.folder, null);
}

/// Whether `seg` can be spelled as an import path segment: an identifier, no keyword.
fn isSegment(seg: []const u8) bool {
    if (seg.len == 0 or std.ascii.isDigit(seg[0])) return false;
    for (seg) |c| if (!std.ascii.isAlphanumeric(c) and c != '_') return false;
    return toyc.token.keywords.get(seg) == null;
}

/// The modules and folders on disk in `root`/`prefix`.
fn listDir(col: *Collector, ws: Workspace, a: std.mem.Allocator, root: []const u8, prefix: []const u8, doc_path: []const u8) !void {
    const base = if (prefix.len == 0) root else try std.fs.path.join(a, &.{ root, prefix });
    var dir = std.Io.Dir.cwd().openDir(ws.io, base, .{ .iterate = true }) catch return;
    defer dir.close(ws.io);
    var it = dir.iterate();
    while (it.next(ws.io) catch return) |e| switch (e.kind) {
        .file => {
            if (!std.mem.endsWith(u8, e.name, Graph.ext)) continue;
            if (std.mem.eql(u8, try std.fs.path.join(a, &.{ base, e.name }), doc_path)) continue;
            try addSegment(col, e.name[0 .. e.name.len - Graph.ext.len], true);
        },
        .directory => {
            if (!isSegment(e.name)) continue;
            const entry = try std.fmt.allocPrint(a, "{s}/{s}{s}", .{ e.name, Graph.folder_entry, Graph.ext });
            const folder_module = !std.mem.eql(u8, e.name, Graph.folder_entry) and
                if (dir.access(ws.io, entry, .{})) true else |_| false;
            if (folder_module or holdsModules(ws.io, dir, e.name, 1)) try addSegment(col, e.name, folder_module);
        },
        else => {},
    };
}

/// Whether folder `name` in `parent` holds a module, directly or up to `depth` folders
/// down. Bounded, since a project root also holds folders such as `node_modules`.
fn holdsModules(io: std.Io, parent: std.Io.Dir, name: []const u8, depth: u8) bool {
    var dir = parent.openDir(io, name, .{ .iterate = true }) catch return false;
    defer dir.close(io);
    var it = dir.iterate();
    while (it.next(io) catch return false) |e| switch (e.kind) {
        .file => if (std.mem.endsWith(u8, e.name, Graph.ext)) return true,
        .directory => if (depth > 0 and isSegment(e.name) and holdsModules(io, dir, e.name, depth - 1)) return true,
        else => {},
    };
    return false;
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

/// The candidates at `@` in `marked` (the `@` dropped), checked as the entry `uri` in `ws`.
fn completeIn(gpa: std.mem.Allocator, ws: Workspace, uri: []const u8, marked: []const u8) !Completions {
    const cut = std.mem.indexOfScalar(u8, marked, '@').?;
    const src = try std.mem.concat(gpa, u8, &.{ marked[0..cut], marked[cut + 1 ..] });
    defer gpa.free(src);
    const p = @import("test_util.zig").posAt(src, cut);
    return completionsAt(gpa, ws, src, p.line, p.character, uri);
}

fn kindOf(items: []const protocol.CompletionItem, label: []const u8) ?u8 {
    for (items) |it| if (std.mem.eql(u8, it.label, label)) return it.kind;
    return null;
}

test "completion in a selection list offers what the module exports, re-exports included" {
    const gpa = testing.allocator;
    const tu = @import("test_util.zig");
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws = try tu.reexportWorkspace(gpa, &docs);
    const Want = struct { label: []const u8, kind: u8 };
    const cases = [_]struct { src: []const u8, has: []const Want, not: []const []const u8 }{
        .{
            .src = "import util.{@}\nfn main() -> int { 0 }\n",
            .has = &.{ .{ .label = "square", .kind = kind.function }, .{ .label = "Pt", .kind = kind.@"struct" }, .{ .label = "math", .kind = kind.module }, .{ .label = "side", .kind = kind.function } },
            .not = &.{ "helper", "sq", "hidden", "fn" },
        },
        .{
            .src = "import util.{side, Pt as Q, sq@}\nfn main() -> int { 0 }\n",
            .has = &.{.{ .label = "square", .kind = kind.function }},
            .not = &.{"helper"},
        },
        .{
            .src = "import util/math.{\n    s@\n}\n",
            .has = &.{ .{ .label = "sq", .kind = kind.function }, .{ .label = "Pt", .kind = kind.@"struct" } },
            .not = &.{"hidden"},
        },
        .{ .src = "import util.{square as @}\n", .has = &.{}, .not = &.{ "square", "Pt", "fn" } },
        .{ .src = "import nowhere.{@}\n", .has = &.{}, .not = &.{"fn"} },
        .{ .src = "import core/ffi.{@}\n", .has = &.{}, .not = &.{ "abs", "sqrt", "fn" } },
    };
    for (cases) |c| {
        var got = try completeIn(gpa, ws, tu.reexport_main_uri, c.src);
        defer got.deinit();
        for (c.has) |w| testing.expectEqual(@as(?u8, w.kind), kindOf(got.items, w.label)) catch |e| {
            std.debug.print("'{s}' in {s}\n", .{ w.label, c.src });
            return e;
        };
        for (c.not) |l| testing.expect(kindOf(got.items, l) == null) catch |e| {
            std.debug.print("unexpected '{s}' in {s}\n", .{ l, c.src });
            return e;
        };
    }
    var sq = try completeIn(gpa, ws, tu.reexport_main_uri, "import util.{@}\n");
    defer sq.deinit();
    try testing.expectEqualStrings("fn square(x: int) -> int", detailOf(sq.items, "square").?);
}

test "completion after a module namespace lists its re-exports too" {
    const gpa = testing.allocator;
    const tu = @import("test_util.zig");
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws = try tu.reexportWorkspace(gpa, &docs);
    var got = try completeIn(gpa, ws, tu.reexport_main_uri, "import util\nfn main() -> int {\n    x := util.@\n    return 0\n}\n");
    defer got.deinit();
    for ([_][]const u8{ "square", "Pt", "math", "side" }) |l| if (kindOf(got.items, l) == null) {
        std.debug.print("no '{s}'\n", .{l});
        return error.MissingCompletion;
    };
    try testing.expect(kindOf(got.items, "helper") == null);
    try testing.expect(kindOf(got.items, "sq") == null);
}

test "completion of an import path offers the next segment from every root and the std" {
    const gpa = testing.allocator;
    const tu = @import("test_util.zig");
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    var ws = try tu.reexportWorkspace(gpa, &docs);
    try docs.put(gpa, "file:///rx/lib/extra.toy", "pub fn one() -> int { 1 }\n", 1);
    ws.roots = .{ .include = &.{"/rx/lib"} };
    const Want = struct { label: []const u8, kind: u8 };
    const cases = [_]struct { src: []const u8, has: []const Want, not: []const []const u8 }{
        .{
            .src = "import @\n",
            .has = &.{ .{ .label = "util", .kind = kind.module }, .{ .label = "std", .kind = kind.folder }, .{ .label = "extra", .kind = kind.module }, .{ .label = "lib", .kind = kind.folder } },
            .not = &.{ "core", "main", "fn" },
        },
        .{ .src = "import u@\n", .has = &.{.{ .label = "util", .kind = kind.module }}, .not = &.{} },
        .{ .src = "import util/@\n", .has = &.{.{ .label = "math", .kind = kind.module }}, .not = &.{ "mod", "util", "std" } },
        .{ .src = "import std/@\n", .has = &.{ .{ .label = "io", .kind = kind.module }, .{ .label = "vec", .kind = kind.module } }, .not = &.{"util"} },
        .{ .src = "pub import util/ma@\n", .has = &.{.{ .label = "math", .kind = kind.module }}, .not = &.{} },
        .{ .src = "import util\n@\n", .has = &.{}, .not = &.{"math"} },
    };
    for (cases) |c| {
        var got = try completeIn(gpa, ws, tu.reexport_main_uri, c.src);
        defer got.deinit();
        for (c.has) |w| testing.expectEqual(@as(?u8, w.kind), kindOf(got.items, w.label)) catch |e| {
            std.debug.print("'{s}' in {s}\n", .{ w.label, c.src });
            return e;
        };
        for (c.not) |l| testing.expect(kindOf(got.items, l) == null) catch |e| {
            std.debug.print("unexpected '{s}' in {s}\n", .{ l, c.src });
            return e;
        };
    }
}

test "completion of an import path lists modules and folders on disk, not the entry itself" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const Io = std.Io;
    const dir = ".lsp-import-paths-test";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    for ([_][]const u8{ "util/mod.toy", "util/math.toy", "pkg/a.toy", ".toy/x.toy", "main.toy", "other.toy", "docs/notes.md", "deep/inner/y.toy", "a/mod/x.toy" }) |f| {
        const path = try std.fs.path.join(gpa, &.{ dir, f });
        defer gpa.free(path);
        try Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(path).?);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "pub fn v() -> int { 1 }\n" });
    }
    try Io.Dir.cwd().createDirPath(io, dir ++ "/empty");
    const abs = try Io.Dir.cwd().realPathFileAlloc(io, dir, gpa);
    defer gpa.free(abs);
    const uri = try std.fmt.allocPrint(gpa, "file://{s}/main.toy", .{abs});
    defer gpa.free(uri);
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = io, .docs = &docs };

    var top = try completeIn(gpa, ws, uri, "import @\n");
    defer top.deinit();
    try testing.expectEqual(@as(?u8, kind.module), kindOf(top.items, "util"));
    try testing.expectEqual(@as(?u8, kind.folder), kindOf(top.items, "pkg"));
    try testing.expectEqual(@as(?u8, kind.module), kindOf(top.items, "other"));
    try testing.expect(kindOf(top.items, "main") == null);
    try testing.expect(kindOf(top.items, ".toy") == null);
    try testing.expect(kindOf(top.items, "empty") == null);
    try testing.expect(kindOf(top.items, "docs") == null);
    try testing.expectEqual(@as(?u8, kind.folder), kindOf(top.items, "deep"));

    var a_dir = try completeIn(gpa, ws, uri, "import a/@\n");
    defer a_dir.deinit();
    try testing.expectEqual(@as(?u8, kind.folder), kindOf(a_dir.items, "mod"));

    // A root spelled with `.` still names the entry's own folder.
    for ([_][]const u8{ "/.", "/./" }) |suffix| {
        const dotted = try std.fmt.allocPrint(gpa, "{s}{s}", .{ abs, suffix });
        defer gpa.free(dotted);
        var rooted = ws;
        rooted.roots = .{ .root = dotted };
        var got = try completeIn(gpa, rooted, uri, "import @\n");
        defer got.deinit();
        try testing.expect(kindOf(got.items, "main") == null);
        try testing.expectEqual(@as(?u8, kind.module), kindOf(got.items, "other"));
    }

    var sub = try completeIn(gpa, ws, uri, "import util/@\n");
    defer sub.deinit();
    try testing.expectEqual(@as(?u8, kind.module), kindOf(sub.items, "math"));
    try testing.expect(kindOf(sub.items, "mod") == null);
}

test "importSite: path, selection list and alias positions" {
    const gpa = testing.allocator;
    const cases = [_]struct { src: []const u8, want: ?std.meta.Tag(ImportSite), path: []const u8 = "" }{
        .{ .src = "import ", .want = .path },
        .{ .src = "import a/b/", .want = .path, .path = "a/b" },
        .{ .src = "import a/b/c", .want = .path, .path = "a/b" },
        .{ .src = "import a/b.{", .want = .item, .path = "a/b" },
        .{ .src = "import a.{x, y as z,\n  w", .want = .item, .path = "a" },
        .{ .src = "import a.{x as ", .want = .alias },
        .{ .src = "import a.{x as y", .want = .alias },
        .{ .src = "import a as ", .want = null },
        .{ .src = "import a ", .want = null },
        .{ .src = "fn f() { S {\n  a", .want = null },
        .{ .src = "x", .want = null },
    };
    for (cases) |c| {
        const toks = try toyc.Lexer.tokenize(gpa, c.src);
        defer gpa.free(toks);
        const got = importSite(toks, c.src, @intCast(c.src.len));
        testing.expectEqual(c.want, if (got) |g| std.meta.activeTag(g) else null) catch |e| {
            std.debug.print("importSite of '{s}'\n", .{c.src});
            return e;
        };
        const segs = switch (got orelse continue) {
            .path, .item => |s| s,
            .alias => continue,
        };
        const joined = try std.mem.join(gpa, "/", segs.slice());
        defer gpa.free(joined);
        try testing.expectEqualStrings(c.path, joined);
    }
}
