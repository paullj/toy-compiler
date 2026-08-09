//! textDocument/hover: the type (or signature) at a byte offset.
//!
//! Reuses `diagnostics.checkBuffer`'s scratch round-trip + `probe = null` contract (see
//! that file's header for why the round-trip is sound), but instead of mapping diagnostics
//! it RETAINS the graph/resolve/typecheck results to read `node_types` / `sigs` /
//! `resolutions`, renders the answer into an owned arena, THEN tears the compiler results
//! down. The render must precede those deinits: a `Sig.name` and a `Layout.name` are
//! BORROWED from the resolve/graph tables, so the arena copy is what outlives them.
//!
//! Selection is token-keyed because the AST stores only a `main_token` per node — no span,
//! no subtree-exit marker. The innermost node covering an offset is therefore "the node
//! whose `main_token` is the token under the cursor". A `{` / `(` / keyword token thus
//! hovers its enclosing node's type by design (a `block` -> `()`, a `call`'s `(` -> the
//! call's result); a whitespace/comment GAP carries no token and yields null.

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");

const Driver = toyc.Driver;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Decide = toyc.Decide;
const Token = toyc.Token;
const Type = toyc.Typecheck.Type;
const Layout = toyc.Typecheck.Layout;
const EnumLayout = toyc.Typecheck.EnumLayout;
const SourceMap = toyc.term.render.SourceMap;

/// Owns the arena backing `value`. `kind` is a static string. `value` stays valid until
/// `deinit`; the compiler results it was rendered from have already been torn down.
pub const Hover = struct {
    arena: std.heap.ArenaAllocator,
    value: []const u8,
    kind: []const u8,

    pub fn deinit(self: *Hover) void {
        self.arena.deinit();
    }
};

/// The type/signature at (`line`, `character`) in `source`, or null when there is nothing
/// meaningful under the cursor (a gap, an out-of-range or poisoned position, or a document
/// that does not parse/resolve). Never errors on a compile problem — only a hard I/O / OOM
/// fault propagates.
pub fn hoverAt(
    gpa: std.mem.Allocator,
    io: Io,
    scratch_path: []const u8,
    source: []const u8,
    line: u32,
    character: u32,
) !?Hover {
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = scratch_path, .data = source });

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);

    var graph = try Graph.discover(gpa, io, cache, "native", scratch_path, null);
    defer graph.deinit(gpa);
    if (graph.err != null) return null;

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    if (Decide.resolveHasError(res.diags)) return null;

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, io, 0);
    defer tc.deinit(gpa);

    const m = graph.entry();
    const entry = graph.entry_index;

    var sm = try SourceMap.init(gpa, m.file, m.source);
    defer sm.deinit(gpa);

    const off = positionToOffset(&sm, line, character) orelse return null;
    const tok = tokenAt(m.tokens, off) orelse return null;

    const resolutions = res.resolutions[entry];
    const node_types = tc.node_types[entry];

    var fid: ?u32 = null;
    var ty: ?Type = null;
    for (m.nodes, 0..) |n, i| {
        if (n.main_token != tok) continue;
        // func-first: a token shared by an identifier and its enclosing node (e.g. an
        // `assign`'s name token) should prefer the callable's signature over the bare
        // identifier's slot type.
        if (fid == null and resolutions[i] == .func and resolutions[i].func < tc.sigs.len) fid = resolutions[i].func;
        if (ty == null and isRenderable(node_types[i])) ty = node_types[i];
    }
    if (fid == null and ty == null) return null;

    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var buf: std.ArrayList(u8) = .empty;
    if (fid) |f| {
        try renderSig(a, &buf, tc.sigs[f], tc.layouts, tc.enum_layouts);
    } else {
        try renderType(a, &buf, ty.?, tc.layouts, tc.enum_layouts);
    }
    return .{ .arena = arena, .value = buf.items, .kind = "plaintext" };
}

/// A type worth showing. The poison/check-only kinds (`invalid`, `type_var`, `app`) never
/// reach a hover answer — a slot left at the `@memset(.invalid)` default is "no info".
fn isRenderable(t: Type) bool {
    return t.kind != .invalid and t.kind != .type_var and t.kind != .app;
}

/// Exact inverse of `diagnostics.pointRange` (`off = line_starts[line] + character`). An
/// out-of-range line yields null; a character past the line's end clamps to the end (which
/// `tokenAt` then reads as a gap -> null). u64 arithmetic so the `line + 1` / `line + 2`
/// index used to bracket the line cannot overflow.
pub fn positionToOffset(sm: *const SourceMap, line: u32, character: u32) ?u32 {
    const lc = sm.lineCount();
    if (line >= lc) return null;
    const ls: u64 = sm.lineStart(@as(usize, line) + 1);
    const le: u64 = if (@as(usize, line) + 1 < lc) sm.lineStart(@as(usize, line) + 2) else sm.bytes.len;
    const want: u64 = ls + character;
    return @intCast(if (want >= le) le else want);
}

/// The token whose half-open `[start, end)` contains `off`, or null for a gap / EOF. Binary
/// search over the start-sorted, non-overlapping token stream.
pub fn tokenAt(tokens: []const Token, off: u32) ?u32 {
    var lo: usize = 0;
    var hi: usize = tokens.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (tokens[mid].start <= off) lo = mid + 1 else hi = mid;
    }
    if (lo == 0) return null;
    const i = lo - 1;
    return if (off < tokens[i].end) @intCast(i) else null;
}

/// Render a type as the compiler names it elsewhere. `layouts`/`enum_layouts` supply the
/// nominal name; the append copies that borrowed name into the caller's arena.
/// A symbol name minus its leading module qualifier. The LSP checks the buffer through an
/// internal scratch file, so the module segment is an artifact the user never wrote — hover
/// shows the bare declared name. Toy identifiers contain no '.', so the last '.' is the
/// module separator.
pub fn bareName(name: []const u8) []const u8 {
    return if (std.mem.lastIndexOfScalar(u8, name, '.')) |dot| name[dot + 1 ..] else name;
}

pub fn renderType(a: std.mem.Allocator, buf: *std.ArrayList(u8), t: Type, layouts: []const Layout, enum_layouts: []const EnumLayout) !void {
    switch (t.kind) {
        .int => try buf.appendSlice(a, t.intName()),
        .bool => try buf.appendSlice(a, "bool"),
        .str => try buf.appendSlice(a, "str"),
        .float => try buf.appendSlice(a, "float"),
        .unit => try buf.appendSlice(a, "()"),
        .rawptr => try buf.appendSlice(a, "rawptr"),
        .never => try buf.appendSlice(a, "never"),
        .@"struct" => try buf.appendSlice(a, if (t.struct_id < layouts.len) bareName(layouts[t.struct_id].name) else "struct"),
        .@"enum" => try buf.appendSlice(a, if (t.enum_id < enum_layouts.len) bareName(enum_layouts[t.enum_id].name) else "enum"),
        // Filtered out by `isRenderable` before we get here; kept total for the compiler.
        .invalid, .type_var, .app => try buf.appendSlice(a, "?"),
    }
}

/// `fn name(P0, P1, ...) -> R`. `sig` is duck-typed (`Typecheck.Sig` is not re-exported)
/// so this stays decoupled from the symbol module's surface.
pub fn renderSig(a: std.mem.Allocator, buf: *std.ArrayList(u8), sig: anytype, layouts: []const Layout, enum_layouts: []const EnumLayout) !void {
    try buf.appendSlice(a, "fn ");
    try buf.appendSlice(a, bareName(sig.name));
    try buf.append(a, '(');
    for (sig.params, 0..) |p, i| {
        if (i != 0) try buf.appendSlice(a, ", ");
        try renderType(a, buf, p, layouts, enum_layouts);
    }
    try buf.appendSlice(a, ") -> ");
    try renderType(a, buf, sig.ret, layouts, enum_layouts);
}

const testing = std.testing;

test "tokenAt: hit, gap, past-last-end, empty" {
    const toks = [_]Token{
        .{ .tag = .identifier, .start = 0, .end = 3 },
        .{ .tag = .identifier, .start = 5, .end = 8 },
    };
    try testing.expectEqual(@as(?u32, 0), tokenAt(&toks, 0));
    try testing.expectEqual(@as(?u32, 0), tokenAt(&toks, 2));
    try testing.expectEqual(@as(?u32, null), tokenAt(&toks, 3)); // gap: end is exclusive
    try testing.expectEqual(@as(?u32, null), tokenAt(&toks, 4)); // whitespace gap
    try testing.expectEqual(@as(?u32, 1), tokenAt(&toks, 7));
    try testing.expectEqual(@as(?u32, null), tokenAt(&toks, 8)); // past last end
    try testing.expectEqual(@as(?u32, null), tokenAt(&.{}, 0)); // empty stream
}

test "positionToOffset: basis, OOB line, char past EOL clamps" {
    const gpa = testing.allocator;
    const src = "ab\ncde\n";
    var sm = try SourceMap.init(gpa, "t", src);
    defer sm.deinit(gpa);

    try testing.expectEqual(@as(?u32, 0), positionToOffset(&sm, 0, 0));
    try testing.expectEqual(@as(?u32, 1), positionToOffset(&sm, 0, 1));
    try testing.expectEqual(@as(?u32, 3), positionToOffset(&sm, 1, 0)); // start of "cde"
    try testing.expectEqual(@as(?u32, 5), positionToOffset(&sm, 1, 2));
    // A char past the line end clamps to the line's end (the trailing '\n' index).
    try testing.expectEqual(@as(?u32, 3), positionToOffset(&sm, 0, 50));
    // OOB line -> null.
    try testing.expectEqual(@as(?u32, null), positionToOffset(&sm, 99, 0));
}

test "positionToOffset is the exact inverse of lineCol" {
    const gpa = testing.allocator;
    const src = "fn f() -> int {\n  x := 1\n  return x\n}\n";
    var sm = try SourceMap.init(gpa, "t", src);
    defer sm.deinit(gpa);

    var off: u32 = 0;
    while (off <= src.len) : (off += 1) {
        const lc = sm.lineCol(off);
        const round = positionToOffset(&sm, @intCast(lc.line - 1), @intCast(lc.col - 1));
        try testing.expectEqual(@as(?u32, off), round);
    }
}
