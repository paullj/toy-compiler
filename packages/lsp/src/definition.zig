//! textDocument/definition: from a USE of a symbol, the source range of its DECLARATION.
//!
//! Checks through the same `Workspace` as hover, but needs only discover→resolve: a
//! declaration is found from `resolutions` + the merged fn table + the AST, never from a
//! `Type`, so typecheck is skipped and UNRELATED resolve errors are tolerated (a valid
//! symbol still navigates in a partially-broken buffer). Only a missing tree (`graph.err`)
//! aborts — there is then nothing to resolve against.
//!
//! Returns a `Def { uri, range }`. The uri is EITHER the open-doc uri (`doc_uri`, for a
//! within-file target) OR an imported module's `file://` path (cross-file target, an open
//! document or a file on disk). The check roots sibling imports at the open doc's directory,
//! so `import a/b` resolves to the real project files.
//!
//! CROSS-FILE go-to-definition is supported for imported PUB fns: a `mod.fn` field-access
//! resolves to a global `.func` carrying its owning module + decl node, which maps to that
//! module's real `file` and thus a real `file://` uri. Cross-file TYPES are out of scope: the
//! resolver leaves an imported type name `.unresolved`, so there is no owning-module handle to
//! follow — a type use navigates only within the entry module.
//!
//! A broken import (`graph.err != null`, e.g. a typo'd import path) disables go-to-def for the
//! WHOLE buffer — there is then no sound tree to resolve against. Pre-existing behavior; once
//! real imports resolve, a genuinely-broken import still zeroes out def.

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const hover = @import("hover.zig");
const lsp_uri = @import("uri.zig");
const Workspace = @import("Workspace.zig");
const Documents = @import("Documents.zig");

const Driver = toyc.Driver;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const Ast = toyc.Ast;
const Token = toyc.Token;
const SourceMap = toyc.term.render.SourceMap;

/// A resolved definition: the target's `file://` uri (always owned) plus the decl range in
/// that file's coordinates.
pub const Def = struct {
    uri: []u8,
    range: protocol.Range,
};

/// The definition for the symbol used at (`line`, `character`) in `source`, or null when
/// there is nothing to navigate to (a gap, an out-of-range or unresolved position, a bundled
/// or builtin target with no openable file, or a buffer that does not parse). `doc_uri` is the
/// open document's real uri: it both roots sibling-import resolution at the real project dir
/// and is returned verbatim for a within-file target. Never errors on a compile problem —
/// only a hard I/O / OOM fault propagates.
pub fn definitionAt(
    gpa: std.mem.Allocator,
    ws: Workspace,
    source: []const u8,
    line: u32,
    character: u32,
    doc_uri: []const u8,
) !?Def {
    var graph = try ws.discover(gpa, doc_uri, source);
    defer graph.deinit(gpa);
    if (graph.err != null) return null; // broken import -> no sound tree to resolve

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);

    const entry = graph.entry_index;
    const m = graph.entry();

    var sm = try SourceMap.init(gpa, m.file, m.source);
    defer sm.deinit(gpa);

    const off = hover.positionToOffset(&sm, line, character) orelse return null;
    const use_tok = hover.tokenAt(m.tokens, off) orelse return null;
    const site = declSiteFor(&graph, entry, &res, use_tok) orelse return null;

    if (site.module == entry) {
        // Within-file: the request's uri verbatim (m.file is its decoded path).
        return .{ .uri = try gpa.dupe(u8, doc_uri), .range = rangeOfToken(&sm, m.tokens, site.tok) };
    }

    const tm = &graph.modules[site.module];
    // A bundled module's `file` is the relative `<bundled>/…` marker, which no client can
    // open; every real module file is absolute.
    if (!std.fs.path.isAbsolute(tm.file)) return null;
    var tsm = try SourceMap.init(gpa, tm.file, tm.source);
    defer tsm.deinit(gpa);
    return .{ .uri = try lsp_uri.pathToUri(gpa, tm.file), .range = rangeOfToken(&tsm, tm.tokens, site.tok) };
}

/// A decl site: the module it lives in and its name token within that module.
const Site = struct { module: u32, tok: u32 };

/// The decl site for the symbol whose use is `use_tok`, tried func → local → type. Func-first
/// mirrors hover: a `mod.fn` field-access and its callee identifier share a token, and the
/// callable's decl is the intended target. A local shadowing a type name also wins by this
/// order. A `.func` may live in ANOTHER module (an imported pub fn) — its owning module + decl
/// node come from the global fn table.
fn declSiteFor(graph: *const Graph.Graph, entry: u32, res: *const ResolveGraph.GraphResult, use_tok: u32) ?Site {
    const m = &graph.modules[entry];
    const resolutions = res.resolutions[entry];
    var func_id: ?u32 = null;
    var local_slot: ?u32 = null;
    for (m.nodes, 0..) |n, i| {
        if (n.main_token != use_tok or i >= resolutions.len) continue;
        switch (resolutions[i]) {
            .func => |f| if (func_id == null) {
                func_id = f;
            },
            .local => |s| if (local_slot == null) {
                local_slot = s;
            },
            else => {},
        }
    }
    if (func_id) |f| {
        if (f >= res.fns.len) return null;
        const gf = res.fns[f];
        if (gf.decl_node == Ast.none) return null; // seeded builtin: no source decl
        const tm = &graph.modules[gf.module];
        if (tm.bundled) return null; // embedded stdlib: no openable file
        return .{ .module = gf.module, .tok = tm.nodes[gf.decl_node.int()].main_token };
    }
    if (local_slot) |s| {
        const t = localDeclToken(m, entry, res, use_tok, s) orelse return null;
        return .{ .module = entry, .tok = t };
    }
    const t = typeDeclToken(m, use_tok) orelse return null; // a type name is left .unresolved
    return .{ .module = entry, .tok = t };
}

/// The binding token for the local in `slot`, within the fn enclosing `use_tok`. The
/// resolver keeps no positional scope table, so the enclosing fn is the last `fn_decl`
/// starting at/before the use, bracketed to `[fn.start, next_fn.start)` (ported from
/// completion's `collectEnclosing`). A param is located POSITIONALLY (params are declared in
/// order and never carry a resolution); a `:=` / for-key binding is found by scanning the
/// binder-tagged nodes that carry this slot.
fn localDeclToken(m: *const Graph.Module, entry: u32, res: *const ResolveGraph.GraphResult, use_tok: u32, slot: u32) ?u32 {
    const resolutions = res.resolutions[entry];
    const use_off = m.tokens[use_tok].start;

    var encl: ?u32 = null;
    var encl_start: u32 = 0;
    var next_start: u32 = std.math.maxInt(u32);
    for (m.nodes, 0..) |n, i| {
        if (n.tag != .fn_decl) continue;
        const s = m.tokens[n.main_token].start;
        if (s <= use_off and (encl == null or s >= encl_start)) {
            encl = @intCast(i);
            encl_start = s;
        }
    }
    if (encl == null) return null;
    for (m.nodes) |n| {
        if (n.tag != .fn_decl) continue;
        const s = m.tokens[n.main_token].start;
        if (s > encl_start and s < next_start) next_start = s;
    }

    const decl = m.nodes[encl.?];
    const proto = Ast.protoAt(m.tree(), decl.lhs.int());
    if (slot < proto.params.len) return m.nodes[proto.params[slot].int()].main_token;

    // A `:=` / for-KEY binder carrying this slot. The binder-tag filter is load-bearing:
    // identifier USE nodes also carry `.local` with the same slot, so an unfiltered scan
    // would match a use, not the binding. (Accepted gap: a `for k, v in` VALUE var binds on
    // an `.identifier` val_leaf, not a binder-tagged node, so a use of `v` finds nothing.)
    for (m.nodes, 0..) |n, i| {
        switch (n.tag) {
            .var_decl, .for_stmt, .for_in_stmt, .for_in2_stmt => {},
            else => continue,
        }
        if (i >= resolutions.len) continue;
        const s = m.tokens[n.main_token].start;
        if (s < encl_start or s >= next_start) continue;
        if (resolutions[i] == .local and resolutions[i].local == slot) return n.main_token;
    }
    return null;
}

/// The name token of the entry module's top-level type decl named like `use_tok`. A struct-
/// literal head / bare type name is left `.unresolved` by the resolver, so func/local miss
/// and this name fallback fires. `main_token` on a `*_decl` IS its name token.
fn typeDeclToken(m: *const Graph.Module, use_tok: u32) ?u32 {
    if (m.tokens[use_tok].tag != .identifier) return null;
    if (m.nodes.len == 0) return null;
    const name = m.tokens[use_tok].text(m.source);
    const prog = m.nodes[Ast.root(m.nodes).int()];
    if (prog.tag != .program) return null;
    for (Ast.rangeSlice(m.tree(), prog.lhs.int())) |decl_idx| {
        const decl = m.nodes[decl_idx.int()];
        switch (decl.tag) {
            .struct_decl, .tuple_struct_decl, .enum_decl => if (std.mem.eql(u8, m.tokens[decl.main_token].text(m.source), name)) return decl.main_token,
            else => {},
        }
    }
    return null;
}

/// The 0-based LSP range of token `t`. `SourceMap.lineCol` yields 1-based line + 1-based
/// byte column (per `diagnostics.pointRange`), so both drop by one; the token's exclusive
/// end offset maps straight to the LSP exclusive end character.
fn rangeOfToken(sm: *const SourceMap, tokens: []const Token, t: u32) protocol.Range {
    const s = sm.lineCol(tokens[t].start);
    const e = sm.lineCol(tokens[t].end);
    return .{
        .start = .{ .line = @intCast(s.line - 1), .character = @intCast(s.col - 1) },
        .end = .{ .line = @intCast(e.line - 1), .character = @intCast(e.col - 1) },
    };
}

const testing = std.testing;

test "rangeOfToken maps a token's [start,end) to a 0-based exclusive range" {
    const gpa = testing.allocator;
    const src = "fn add(a: int) -> int { return a }";
    var sm = try SourceMap.init(gpa, "t", src);
    defer sm.deinit(gpa);

    // The `add` name token spans bytes [3, 6).
    const add = std.mem.indexOf(u8, src, "add").?;
    const toks = [_]Token{.{ .tag = .identifier, .start = @intCast(add), .end = @intCast(add + 3) }};
    const r = rangeOfToken(&sm, &toks, 0);
    try testing.expectEqual(@as(u32, 0), r.start.line);
    try testing.expectEqual(@as(u32, 3), r.start.character);
    try testing.expectEqual(@as(u32, 0), r.end.line);
    try testing.expectEqual(@as(u32, 6), r.end.character); // exclusive end
}

/// The 0-based (line, character) of the first byte of `needle` in `src`.
fn posOf(src: []const u8, needle: []const u8) protocol.Position {
    const idx = std.mem.indexOf(u8, src, needle).?;
    var line: u32 = 0;
    var col: u32 = 0;
    for (src[0..idx]) |ch| {
        if (ch == '\n') {
            line += 1;
            col = 0;
        } else col += 1;
    }
    return .{ .line = line, .character = col };
}

test "definition: cross-file jump to an imported pub fn; within-file stays local; imports resolve" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // A REAL on-disk two-file project. Only helper.toy must come from disk: the entry is
    // served from its buffer, and `doc_uri` roots imports at the real dir.
    const dir_name = ".lsp-xfile-test";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const helper_src = "pub fn foo() -> int { return 1 }\n"; // MUST be pub for cross-file resolution
    const main_src =
        "import helper\n" ++
        "\n" ++
        "fn main() -> int {\n" ++
        "    x := helper.foo()\n" ++
        "    return x\n" ++
        "}\n";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir_name ++ "/helper.toy", .data = helper_src });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir_name ++ "/main.toy", .data = main_src });

    // Root at the realpath so it is ABSOLUTE (definitionAt refuses a relative tm.file) and
    // canonical (macOS reaches the test dir via a /tmp -> /private/tmp symlink). tm.file is
    // built raw from the root string we pass, so want_uri derives from the SAME abs_dir.
    const abs_dir_z = try Io.Dir.cwd().realPathFileAlloc(io, dir_name, gpa);
    defer gpa.free(abs_dir_z);
    const abs_dir = try gpa.dupe(u8, abs_dir_z);
    defer gpa.free(abs_dir);
    const main_path = try std.fs.path.join(gpa, &.{ abs_dir, "main.toy" });
    defer gpa.free(main_path);
    const doc_uri = try std.fmt.allocPrint(gpa, "file://{s}", .{main_path});
    defer gpa.free(doc_uri);
    const helper_path = try std.fs.path.join(gpa, &.{ abs_dir, "helper.toy" });
    defer gpa.free(helper_path);
    const want_uri = try std.fmt.allocPrint(gpa, "file://{s}", .{helper_path});
    defer gpa.free(want_uri);

    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = io, .docs = &docs };

    // (1) Cross-file: the `foo` use in `helper.foo()` -> helper.toy's real uri + `foo` decl.
    {
        const use = posOf(main_src, "foo()");
        const d = (try definitionAt(gpa, ws, main_src, use.line, use.character, doc_uri)) orelse
            return error.CrossFileReturnedNull; // non-vacuous: pub fn MUST resolve cross-file
        defer gpa.free(d.uri);
        try testing.expectEqualStrings(want_uri, d.uri);
        try testing.expectEqual(@as(u32, 0), d.range.start.line);
        const foo_col: u32 = @intCast(std.mem.indexOf(u8, helper_src, "foo").?);
        try testing.expectEqual(foo_col, d.range.start.character);
        try testing.expectEqual(foo_col + 3, d.range.end.character);
    }

    // (2) Within-file: the `x` use in `return x` -> the OPEN doc uri.
    {
        const rx = posOf(main_src, "return x");
        const use_char: u32 = rx.character + @as(u32, @intCast("return ".len));
        const d = (try definitionAt(gpa, ws, main_src, rx.line, use_char, doc_uri)) orelse
            return error.WithinFileReturnedNull;
        defer gpa.free(d.uri);
        try testing.expectEqualStrings(doc_uri, d.uri);
    }
}

test "definition: cross-file jump into another OPEN document, disk off" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const helper_src = "pub fn foo() -> int { return 1 }\n";
    try docs.put(gpa, "file:///blocks/helper.toy", helper_src, 1);
    const main_src = "import helper\nfn main() -> int { return helper.foo() }\n";

    const ws: Workspace = .{ .io = Io.failing, .docs = &docs, .disk = false };
    const use = posOf(main_src, "foo()");
    const d = (try definitionAt(gpa, ws, main_src, use.line, use.character, "file:///blocks/main.toy")) orelse
        return error.CrossFileReturnedNull;
    defer gpa.free(d.uri);
    try testing.expectEqualStrings("file:///blocks/helper.toy", d.uri);
    const foo_col: u32 = @intCast(std.mem.indexOf(u8, helper_src, "foo").?);
    try testing.expectEqual(foo_col, d.range.start.character);
}
