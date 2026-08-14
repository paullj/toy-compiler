//! textDocument/definition: from a USE of a symbol, the source range of its DECLARATION.
//!
//! Reuses hover's scratch round-trip + `probe = null` contract (see `hover.zig` /
//! `diagnostics.zig` for why the round-trip is sound), but needs only discover→resolve: a
//! declaration is found from `resolutions` + the merged fn table + the AST, never from a
//! `Type`, so typecheck is skipped and UNRELATED resolve errors are tolerated (a valid
//! symbol still navigates in a partially-broken buffer). Only a missing tree (`graph.err`)
//! aborts — there is then nothing to resolve against.
//!
//! Returns a POD `Range` in the ENTRY document's own coordinates, NOT a `Location`: the
//! caller supplies the open-doc uri from the dispatch layer, so the helper never handles a
//! uri and therefore cannot leak the internal scratch path.
//!
//! WITHIN-FILE ONLY. A use of an imported symbol returns null: the buffer is checked from a
//! scratch directory, so a sibling `import a/b` does not resolve to the real project file,
//! and a cross-file decl would have to map back to that file's real uri — a scratch-approach
//! redesign, deferred to its own milestone.

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const hover = @import("hover.zig");

const Driver = toyc.Driver;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const Ast = toyc.Ast;
const Token = toyc.Token;
const SourceMap = toyc.term.render.SourceMap;

/// The declaration range for the symbol used at (`line`, `character`) in `source`, in the
/// entry document's coordinates, or null when there is nothing to navigate to (a gap, an
/// out-of-range or unresolved position, an imported symbol, or a buffer that does not
/// parse). Never errors on a compile problem — only a hard I/O / OOM fault propagates.
pub fn definitionAt(
    gpa: std.mem.Allocator,
    io: Io,
    scratch_path: []const u8,
    source: []const u8,
    line: u32,
    character: u32,
) !?protocol.Range {
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = scratch_path, .data = source });

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);

    var graph = try Graph.discover(gpa, io, cache, "native", scratch_path, null);
    defer graph.deinit(gpa);
    if (graph.err != null) return null; // no tree -> nothing to resolve

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);

    const m = graph.entry();
    const entry = graph.entry_index;

    var sm = try SourceMap.init(gpa, m.file, m.source);
    defer sm.deinit(gpa);

    const off = hover.positionToOffset(&sm, line, character) orelse return null;
    const use_tok = hover.tokenAt(m.tokens, off) orelse return null;
    const decl_tok = declTokenFor(m, entry, &res, use_tok) orelse return null;
    return rangeOfToken(&sm, m.tokens, decl_tok);
}

/// The declaration NAME token for the symbol whose use is `use_tok`, tried func → local →
/// type. Func-first mirrors hover: a `mod.fn` field-access and its callee identifier share a
/// token, and the callable's decl is the intended target. A local shadowing a type name also
/// wins by this order.
fn declTokenFor(m: *const Graph.Module, entry: u32, res: *const ResolveGraph.GraphResult, use_tok: u32) ?u32 {
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
        if (gf.module != entry) return null; // cross-file: deferred (checked BEFORE indexing m.nodes)
        return m.nodes[gf.decl_node.int()].main_token; // fn_decl.main_token IS the name token
    }
    if (local_slot) |s| return localDeclToken(m, entry, res, use_tok, s);
    return typeDeclToken(m, use_tok); // a type name is left .unresolved -> name fallback
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
