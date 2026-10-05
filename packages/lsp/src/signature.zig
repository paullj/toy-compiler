//! textDocument/signatureHelp: when the cursor sits inside a call's argument list, the
//! callee's signature plus the ACTIVE PARAMETER index.
//!
//! Checks through the same `Workspace` as hover, but detection runs FIRST over a
//! fault-tolerant raw LEX of the buffer — the enclosing call is found from the token stream,
//! so an outside-any-call cursor is rejected before any compile. Like completion, this must survive a broken buffer: signature help
//! fires mid-type (`add(1, ` with no closing paren), so an unterminated call is REPAIRED by
//! appending its missing closers at EOF (offset-preserving) and resolve/type errors are
//! tolerated — an arity error on the repaired call must not suppress the declared signature.
//!
//! A free fn's signature is read from `resolutions` -> `sigs` (the same path hover uses for
//! a callee), rendered through `render.bareName` so the module qualifier never leaks. A
//! method call's comes from the checker's Callee table (`callee_sig`), shown without its
//! `self`: the receiver is left of the `.`, never inside the parens. Param labels are
//! [start,end) byte spans into the rendered label.
//!
//! KNOWN LIMITATIONS (documented, not bugs):
//!   * A turbofish call `id[int](` or `v.m[int](` — the token before `(` is `]`, not an
//!     identifier — yields no help. Free fns and module-qualified free fns (`io.f(`) are
//!     correct.
//!   * A builtin method on a concrete receiver inside a bounded template (`(3).eq(`) shows
//!     no receiver type: the template's node types are scratch, so its param renders `?`.
//!   * A method call inside a generic body can yield no help at all, and a generic
//!     protocol's method shows the protocol's own params: see `callee_sig`'s limitations.
//!   * A bare pipe target `x |> f` has no `(`, so it yields no help of its own.
//!   * A parse error OTHER than the appended closers (or a stray unmatched closer) re-triggers
//!     discovery's whole-tree discard -> null (the single-edit repair limit, as in completion).
//!   * Signature help may fire inside a fn DECLARATION's own param parens (`(` preceded by the
//!     decl name); harmless.

const std = @import("std");
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const position = @import("position.zig");
const render = @import("render.zig");
const callee_sig = @import("callee_sig.zig");
const Workspace = @import("Workspace.zig");

const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Token = toyc.Token;
const Type = toyc.Typecheck.Type;
const Layout = toyc.Typecheck.Layout;
const EnumLayout = toyc.Typecheck.EnumLayout;
const SourceMap = toyc.term.render.SourceMap;

/// Owns the arena backing `help`. The compiler results it was rendered from have already
/// been torn down by the time this is returned.
pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    help: protocol.SignatureHelp,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
    }
};

const Enclosing = struct { callee_start: u32, active: u32 };

/// The innermost enclosing call the cursor is inside, or null. A call's `(` is never
/// preceded by a statement-terminator newline, so the IMMEDIATELY preceding token decides
/// `is_call` — no newline-skip (that would misread `foo⏎(x)`, two statements, as a call).
/// An identifier before `(` covers `add(` and the member callee of `recv.f(`. Turbofish
/// `id[int](` (preceding token `]`) and grouping / `if` / `while` parens are excluded. Only
/// tokens starting strictly before `off` drive the stack, so a `(` exactly at the cursor is
/// "not yet entered" and a cursor just before it is outside.
fn enclosingCall(gpa: std.mem.Allocator, toks: []const Token, off: u32) !?Enclosing {
    const Frame = struct { is_call: bool, callee_start: u32, lparen_i: usize };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(gpa);
    for (toks, 0..) |t, i| {
        if (t.start >= off) break;
        switch (t.tag) {
            .l_paren => {
                const is_call = i > 0 and toks[i - 1].tag == .identifier;
                try stack.append(gpa, .{
                    .is_call = is_call,
                    .callee_start = if (is_call) toks[i - 1].start else 0,
                    .lparen_i = i,
                });
            },
            .l_bracket, .l_brace => try stack.append(gpa, .{ .is_call = false, .callee_start = 0, .lparen_i = i }),
            // Pop on any closer regardless of type: a mismatched closer in a broken prefix
            // also fails to re-parse (whole-tree discard -> null), so no extra handling pays.
            .r_paren, .r_bracket, .r_brace => if (stack.items.len > 0) {
                _ = stack.pop();
            },
            else => {},
        }
    }
    var k = stack.items.len;
    while (k > 0) {
        k -= 1;
        if (stack.items[k].is_call) {
            const fr = stack.items[k];
            return .{ .callee_start = fr.callee_start, .active = activeParam(toks, fr.lparen_i, off) };
        }
    }
    return null;
}

/// The count of top-level argument separators between the call `(` and `off`. Commas nested
/// in inner ()/[]/{} are at depth > 0 and excluded; an unbalanced inner prefix only drives
/// depth negative, which never over-counts.
fn activeParam(toks: []const Token, lparen_i: usize, off: u32) u32 {
    var depth: i32 = 0;
    var commas: u32 = 0;
    var i = lparen_i + 1;
    while (i < toks.len and toks[i].start < off) : (i += 1) switch (toks[i].tag) {
        .l_paren, .l_bracket, .l_brace => depth += 1,
        .r_paren, .r_bracket, .r_brace => depth -= 1,
        .comma => if (depth == 0) {
            commas += 1;
        },
        else => {},
    };
    return commas;
}

/// Append-only EOF balance. Returns `source` unchanged (same ptr) when already balanced (the
/// common complete-call case -> no copy); appends the unclosed openers' matching closers
/// innermost-first at EOF (offset-preserving, so every original token start stays valid in
/// the recompiled stream); null on an extra closer (unrepairable single-edit).
fn repair(gpa: std.mem.Allocator, source: []const u8, toks: []const Token) !?[]const u8 {
    var stack: std.ArrayList(u8) = .empty;
    defer stack.deinit(gpa);
    for (toks) |t| switch (t.tag) {
        .l_paren => try stack.append(gpa, ')'),
        .l_bracket => try stack.append(gpa, ']'),
        .l_brace => try stack.append(gpa, '}'),
        .r_paren, .r_bracket, .r_brace => {
            if (stack.items.len == 0) return null;
            _ = stack.pop();
        },
        else => {},
    };
    if (stack.items.len == 0) return source;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, source);
    var i = stack.items.len;
    while (i > 0) : (i -= 1) try out.append(gpa, stack.items[i - 1]);
    return try out.toOwnedSlice(gpa);
}

/// Mirrors `render.renderSig` byte-for-byte (for empty `var_names`) while capturing each
/// param's [start,end) span, so a param label always points at the RIGHT occurrence even when
/// two params share a type name. Type/name bytes are borrowed off the compiler tables; the
/// appends copy them into `a`.
fn renderLabel(
    a: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    name: []const u8,
    params: []const Type,
    mut_mask: u64,
    ret: Type,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
    var_names: []const []const u8,
    ranges: *std.ArrayList([2]u32),
) !void {
    try buf.appendSlice(a, "fn ");
    try buf.appendSlice(a, name);
    try buf.append(a, '(');
    for (params, 0..) |p, i| {
        if (i != 0) try buf.appendSlice(a, ", ");
        const s: u32 = @intCast(buf.items.len);
        if (toyc.Modes.isMut(mut_mask, i)) try buf.appendSlice(a, "mut ");
        try render.renderTypeNamed(a, buf, p, layouts, enum_layouts, var_names);
        try ranges.append(a, .{ s, @intCast(buf.items.len) });
    }
    try buf.appendSlice(a, ") -> ");
    try render.renderTypeNamed(a, buf, ret, layouts, enum_layouts, var_names);
}

/// A user protocol method's label spelled from its declaration, as hover shows it: a
/// protocol's generic types (`Option[int]`) are opaque once typecheck is done, so the
/// checked types would render `?`.
fn renderDeclLabel(
    a: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    pm: *const toyc.Graph.Module,
    decl: u32,
    name: []const u8,
    has_self: bool,
    ranges: *std.ArrayList([2]u32),
) !void {
    const proto = toyc.Ast.protoAt(pm.tree(), pm.nodes[decl].lhs.int());
    try buf.appendSlice(a, "fn ");
    try buf.appendSlice(a, name);
    try buf.append(a, '(');
    const params = proto.params[@min(@intFromBool(has_self), proto.params.len)..];
    for (params, 0..) |p, i| {
        if (i != 0) try buf.appendSlice(a, ", ");
        const s: u32 = @intCast(buf.items.len);
        if (toyc.Ast.isMutParam(pm.tree(), pm.tokens, p)) try buf.appendSlice(a, "mut ");
        try appendTypeText(a, buf, pm, pm.nodes[p.int()].lhs);
        try ranges.append(a, .{ s, @intCast(buf.items.len) });
    }
    try buf.appendSlice(a, ") -> ");
    if (proto.ret_type == toyc.Ast.none) try buf.appendSlice(a, "()") else try appendTypeText(a, buf, pm, proto.ret_type);
}

fn appendTypeText(a: std.mem.Allocator, buf: *std.ArrayList(u8), pm: *const toyc.Graph.Module, ty: toyc.Ast.Index) !void {
    const sp = toyc.AstWalk.nodeSpan(.{ .tree = pm.tree(), .tokens = pm.tokens, .source = pm.source }, ty);
    try render.appendCollapsed(a, buf, pm.source[sp.start..sp.end]);
}

/// The signature help at (`line`, `character`) in `source`, or null when the cursor is not
/// inside a resolvable call. Never errors on a compile problem — only a hard I/O / OOM fault
/// propagates.
pub fn signatureHelpAt(
    gpa: std.mem.Allocator,
    ws: Workspace,
    source: []const u8,
    line: u32,
    character: u32,
    doc_uri: []const u8,
) !?Result {
    const toks = try toyc.Lexer.tokenize(gpa, source);
    defer gpa.free(toks);

    var sm = try SourceMap.init(gpa, "s", source);
    defer sm.deinit(gpa);

    const off = position.positionToOffset(&sm, line, character, ws.encoding) orelse return null;
    const enc = (try enclosingCall(gpa, toks, off)) orelse return null;

    const repaired = (try repair(gpa, source, toks)) orelse return null;
    defer if (repaired.ptr != source.ptr) gpa.free(repaired);

    var graph = try ws.discover(gpa, doc_uri, repaired);
    defer graph.deinit(gpa);
    if (graph.err != null) return null;

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    // UNCONDITIONAL: a repaired arity error must not suppress the declared signature.
    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, ws.io, 0);
    defer tc.deinit(gpa);

    const entry = graph.entry_index;
    const m = graph.entry();
    const resolutions = res.resolutions[entry];

    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    // The callee node keys on its identifier token's source start (robust to node/token
    // reshuffle): a free fn's resolves to `.func`, a method's member node records a Callee.
    // The call node's own main_token is the `(` (or the `|>` of a piped call), a different
    // start, so the two never collide.
    const Hit = union(enum) { func: u32, method: callee_sig.CalleeSig };
    const hit: ?struct { what: Hit, node: usize } = for (m.nodes, 0..) |n, i| {
        if (m.tokens[n.main_token].start != enc.callee_start or i >= resolutions.len) continue;
        if (resolutions[i] == .func and resolutions[i].func < tc.sigs.len) {
            break .{ .what = .{ .func = resolutions[i].func }, .node = i };
        }
        if (n.tag == .field_access and resolutions[i] != .func) {
            if (try callee_sig.of(a, &graph, &res, &tc, entry, @intCast(i))) |ms| break .{ .what = .{ .method = ms }, .node = i };
        }
    } else null;
    const callee = hit orelse {
        arena.deinit();
        return null;
    };

    // In `x |> f(a, ..)` the piped `x` is argument 0 but sits left of the `(`, so the comma
    // count is one short. The tree, not the tokens, says whether the call is piped: a token
    // look-back from the callee would miss an `m.f` callee. A value written at a `_` is
    // already counted.
    var active = enc.active;
    for (m.nodes, 0..) |n, ci| {
        if (n.tag != .call or n.lhs.int() != callee.node) continue;
        if (toyc.Ast.pipePrepends(m.tree(), m.tokens, m.source, toyc.Ast.Index.from(@intCast(ci)))) active += 1;
        break;
    }

    var label: std.ArrayList(u8) = .empty;
    var ranges: std.ArrayList([2]u32) = .empty;
    switch (callee.what) {
        .func => |fid| {
            const sig = tc.sigs[fid];
            try renderLabel(a, &label, render.bareName(sig.name), sig.params, sig.mut_mask, sig.ret, tc.layouts, tc.enum_layouts, &.{}, &ranges);
        },
        .method => |ms| if (ms.decl != null and callee_sig.spelledFromDecl(&tc, entry, @intCast(callee.node)))
            try renderDeclLabel(a, &label, &graph.modules[ms.decl.?.module], ms.decl.?.node, ms.name, ms.has_self, &ranges)
        else
            try renderLabel(a, &label, ms.name, ms.params[@intFromBool(ms.has_self)..], ms.mut_mask >> @intFromBool(ms.has_self), ms.ret, tc.layouts, tc.enum_layouts, ms.var_names, &ranges),
    }

    const params = try a.alloc(protocol.ParameterInformation, ranges.items.len);
    for (ranges.items, 0..) |r, i| params[i] = .{ .label = r };

    if (ranges.items.len == 0) {
        active = 0;
    } else if (active > ranges.items.len - 1) {
        active = @intCast(ranges.items.len - 1);
    }

    const sigs = try a.alloc(protocol.SignatureInformation, 1);
    sigs[0] = .{ .label = label.items, .parameters = params };

    return .{ .arena = arena, .help = .{ .signatures = sigs, .activeSignature = 0, .activeParameter = active } };
}

const testing = std.testing;
const test_util = @import("test_util.zig");

fn tok(tag: toyc.Tag, start: u32, end: u32) Token {
    return .{ .tag = tag, .start = start, .end = end };
}

test "activeParam: counts only top-level commas" {
    // `add(1, 2)` — add[0,3) ([3,4) 1[4,5) ,[5,6) 2[7,8) )[8,9)
    const toks = [_]Token{
        tok(.identifier, 0, 3), tok(.l_paren, 3, 4), tok(.number, 4, 5),
        tok(.comma, 5, 6),      tok(.number, 7, 8),  tok(.r_paren, 8, 9),
    };
    try testing.expectEqual(@as(u32, 0), activeParam(&toks, 1, 5)); // before the comma
    try testing.expectEqual(@as(u32, 1), activeParam(&toks, 1, 7)); // after the comma
}

test "activeParam: a comma nested in inner parens does not count" {
    // `f(g(1,2), 3)` — f( g( 1 , 2 ) , 3 )
    const toks = [_]Token{
        tok(.identifier, 0, 1), tok(.l_paren, 1, 2),   tok(.identifier, 2, 3),
        tok(.l_paren, 3, 4),    tok(.number, 4, 5),    tok(.comma, 5, 6),
        tok(.number, 6, 7),     tok(.r_paren, 7, 8),   tok(.comma, 8, 9),
        tok(.number, 10, 11),   tok(.r_paren, 11, 12),
    };
    // Cursor past the OUTER comma (offset 10): the inner `,` at depth 1 is excluded.
    try testing.expectEqual(@as(u32, 1), activeParam(&toks, 1, 10));
}

test "activeParam: a bracket comma does not count" {
    // `f(a[i,j], 3)`
    const toks = [_]Token{
        tok(.identifier, 0, 1), tok(.l_paren, 1, 2),    tok(.identifier, 2, 3),
        tok(.l_bracket, 3, 4),  tok(.identifier, 4, 5), tok(.comma, 5, 6),
        tok(.identifier, 6, 7), tok(.r_bracket, 7, 8),  tok(.comma, 8, 9),
        tok(.number, 10, 11),   tok(.r_paren, 11, 12),
    };
    try testing.expectEqual(@as(u32, 1), activeParam(&toks, 1, 10));
}

test "enclosingCall: a bare call is entered, callee_start is the identifier" {
    // `add(1)` — add[0,3) ([3,4) 1[4,5) )[5,6)
    const toks = [_]Token{ tok(.identifier, 0, 3), tok(.l_paren, 3, 4), tok(.number, 4, 5), tok(.r_paren, 5, 6) };
    const enc = (try enclosingCall(testing.allocator, &toks, 4)).?;
    try testing.expectEqual(@as(u32, 0), enc.callee_start);
    try testing.expectEqual(@as(u32, 0), enc.active);
}

test "enclosingCall: the INNERMOST call wins for a nested call" {
    // `f(g(1))` — f( g( 1 ) ) with g's identifier at [2,3).
    const toks = [_]Token{
        tok(.identifier, 0, 1), tok(.l_paren, 1, 2), tok(.identifier, 2, 3),
        tok(.l_paren, 3, 4),    tok(.number, 4, 5),  tok(.r_paren, 5, 6),
        tok(.r_paren, 6, 7),
    };
    const enc = (try enclosingCall(testing.allocator, &toks, 5)).?; // inside the inner call
    try testing.expectEqual(@as(u32, 2), enc.callee_start); // g, not f
}

test "enclosingCall: a grouping paren is not a call" {
    // `x := (a)` — x[0,1) :=[2,4) ([5,6) a[6,7) )[7,8)
    const toks = [_]Token{
        tok(.identifier, 0, 1), tok(.colon_eq, 2, 4), tok(.l_paren, 5, 6),
        tok(.identifier, 6, 7), tok(.r_paren, 7, 8),
    };
    try testing.expectEqual(@as(?Enclosing, null), try enclosingCall(testing.allocator, &toks, 7));
}

test "enclosingCall: a newline before the paren is not a call" {
    // `foo⏎(x)` — foo[0,3) newline[3,3] ([4,5) x[5,6) )[6,7). Two statements, not a call.
    const toks = [_]Token{
        tok(.identifier, 0, 3), tok(.newline, 3, 3), tok(.l_paren, 4, 5),
        tok(.identifier, 5, 6), tok(.r_paren, 6, 7),
    };
    try testing.expectEqual(@as(?Enclosing, null), try enclosingCall(testing.allocator, &toks, 5));
}

test "enclosingCall: an unterminated call at EOF still resolves with the active param" {
    // `add(1, ` — add[0,3) ([3,4) 1[4,5) ,[5,6) eof[7,7)
    const toks = [_]Token{
        tok(.identifier, 0, 3), tok(.l_paren, 3, 4), tok(.number, 4, 5),
        tok(.comma, 5, 6),      tok(.eof, 7, 7),
    };
    const enc = (try enclosingCall(testing.allocator, &toks, 7)).?;
    try testing.expectEqual(@as(u32, 0), enc.callee_start);
    try testing.expectEqual(@as(u32, 1), enc.active);
}

test "enclosingCall: no call yields null" {
    const toks = [_]Token{ tok(.identifier, 0, 6), tok(.identifier, 7, 8), tok(.eof, 8, 8) };
    try testing.expectEqual(@as(?Enclosing, null), try enclosingCall(testing.allocator, &toks, 8));
}

test "repair: a balanced source is returned unchanged (same pointer)" {
    const gpa = testing.allocator;
    const src = "fn main() -> int { return 0 }";
    const toks = try toyc.Lexer.tokenize(gpa, src);
    defer gpa.free(toks);
    const out = (try repair(gpa, src, toks)).?;
    try testing.expect(out.ptr == src.ptr);
}

test "repair: an unterminated call appends the missing closers at EOF" {
    const gpa = testing.allocator;
    const src = "fn f() -> int { g(1, ";
    const toks = try toyc.Lexer.tokenize(gpa, src);
    defer gpa.free(toks);
    const out = (try repair(gpa, src, toks)).?;
    defer gpa.free(out);
    try testing.expect(out.ptr != src.ptr);
    // Two open parens + one open brace, closed innermost-first: `)` then `}`.
    try testing.expect(std.mem.startsWith(u8, out, src));
    try testing.expectEqualStrings(")}", out[src.len..]);
}

test "repair: an extra closer is unrepairable" {
    const gpa = testing.allocator;
    const src = ")";
    const toks = try toyc.Lexer.tokenize(gpa, src);
    defer gpa.free(toks);
    try testing.expectEqual(@as(?[]const u8, null), try repair(gpa, src, toks));
}

test "renderLabel: matches renderSig and spans the correct param occurrences" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const int = Type.int;
    const Sig = struct { name: []const u8, params: []const Type, ret: Type, mut_mask: u64 = 0 };
    const sig: Sig = .{ .name = "m.add", .params = &.{ int, int }, .ret = int };

    const no_layouts: []const Layout = &.{};
    const no_enums: []const EnumLayout = &.{};

    var label: std.ArrayList(u8) = .empty;
    var ranges: std.ArrayList([2]u32) = .empty;
    try renderLabel(a, &label, render.bareName(sig.name), sig.params, sig.mut_mask, sig.ret, no_layouts, no_enums, &.{}, &ranges);

    var expect: std.ArrayList(u8) = .empty;
    try render.renderSig(a, &expect, sig, no_layouts, no_enums);
    try testing.expectEqualStrings(expect.items, label.items);
    try testing.expectEqualStrings("fn add(int, int) -> int", label.items);

    // Two params; the SECOND `int` span is the later occurrence — proves the offset label and
    // that `bareName` dropped the `m.` qualifier (label starts `fn add`, not `fn m.add`).
    try testing.expectEqual(@as(usize, 2), ranges.items.len);
    try testing.expectEqual([2]u32{ 7, 10 }, ranges.items[0]);
    try testing.expectEqual([2]u32{ 12, 15 }, ranges.items[1]);
}

test "renderLabel: a zero-param sig has no ranges" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const int = Type.int;
    const Sig = struct { name: []const u8, params: []const Type, ret: Type, mut_mask: u64 = 0 };
    const sig: Sig = .{ .name = "foo", .params = &.{}, .ret = int };

    const no_layouts: []const Layout = &.{};
    const no_enums: []const EnumLayout = &.{};

    var label: std.ArrayList(u8) = .empty;
    var ranges: std.ArrayList([2]u32) = .empty;
    try renderLabel(a, &label, render.bareName(sig.name), sig.params, sig.mut_mask, sig.ret, no_layouts, no_enums, &.{}, &ranges);
    try testing.expectEqualStrings("fn foo() -> int", label.items);
    try testing.expectEqual(@as(usize, 0), ranges.items.len);
}

test "signatureHelpAt: a piped call's active param counts the piped value" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn add3(a: int, b: int, c: int) -> int { return a + b + c }
        \\fn main() -> int {
        \\    h := 5 |> add(2, _)
        \\    k := 5 |> add(_, 2)
        \\    x := 1 |> add(2)
        \\    y := 1 |> add3(2, 3)
        \\    z := add(1, 2) |> add(3)
        \\    return x + y + z + h + k
        \\}
        \\
    ;
    const cases = [_]struct { after: []const u8, label: []const u8, active: u32 }{
        .{ .after = "h := 5 |> add(2, ", .label = "fn add(int, int) -> int", .active = 1 },
        .{ .after = "h := 5 |> add(", .label = "fn add(int, int) -> int", .active = 0 },
        .{ .after = "k := 5 |> add(_, ", .label = "fn add(int, int) -> int", .active = 1 },
        .{ .after = "k := 5 |> add(", .label = "fn add(int, int) -> int", .active = 0 },
        .{ .after = "1 |> add(", .label = "fn add(int, int) -> int", .active = 1 },
        .{ .after = "1 |> add3(2, ", .label = "fn add3(int, int, int) -> int", .active = 2 },
        .{ .after = "z := add(1, ", .label = "fn add(int, int) -> int", .active = 1 },
        .{ .after = "z := add(", .label = "fn add(int, int) -> int", .active = 0 },
        .{ .after = "|> add(", .label = "fn add(int, int) -> int", .active = 1 },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no signature help after '{s}'\n", .{c.after});
            return error.NoSignatureHelp;
        };
        defer r.deinit();
        try testing.expectEqualStrings(c.label, r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
    }
}

test "signatureHelpAt: a piped module-qualified call counts the piped value" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    try docs.put(gpa, "file:///p/helper.toy", "pub fn add(a: int, b: int, c: int) -> int { return a + b + c }\n", 1);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src = "import helper\nfn main() -> int {\n    return 1 |> helper.add(2, 3)\n}\n";
    const cases = [_]struct { after: []const u8, active: u32 }{
        .{ .after = "helper.add(", .active = 1 },
        .{ .after = "helper.add(2, ", .active = 2 },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse
            return error.NoSignatureHelp;
        defer r.deinit();
        try testing.expectEqualStrings("fn add(int, int, int) -> int", r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
    }
}

test "signatureHelpAt: method calls show the selected method, piped or not" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src = test_util.method_fixture;
    const plus = "fn plus(int, int) -> int";
    const cases = [_]struct { after: []const u8, label: []const u8, active: u32 }{
        .{ .after = "x := acc.plus(", .label = plus, .active = 0 },
        .{ .after = "acc.plus(1, ", .label = plus, .active = 1 },
        .{ .after = "1 |> acc.plus(", .label = plus, .active = 1 },
        .{ .after = "Box[int].make(", .label = "fn make(int) -> Box[int]", .active = 0 },
        .{ .after = "b.get(", .label = "fn get() -> int", .active = 0 },
        .{ .after = "v.dbl(", .label = "fn dbl() -> int", .active = 0 },
        .{ .after = "\"hi\".concat(", .label = "fn concat(str) -> str", .active = 0 },
        .{ .after = "(3).eq(", .label = "fn eq(int) -> bool", .active = 0 },
        .{ .after = "o.unwrap_or(", .label = "fn unwrap_or(int) -> int", .active = 0 },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no signature help after '{s}'\n", .{c.after});
            return error.NoSignatureHelp;
        };
        defer r.deinit();
        try testing.expectEqualStrings(c.label, r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
    }
}

test "signatureHelpAt: a generic method's own type params show while its call is incomplete" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const pre =
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\    fn map[U](self, f: fn(T) -> U) -> Box[U] { Box { v: f(self.v) } }
        \\    fn fold[A](self, init: A, f: fn(A, T) -> A) -> A { f(init, self.v) }
        \\}
        \\fn main() -> int {
        \\    b := Box { v: 2 }
        \\
    ;
    const cases = [_]struct { tail: []const u8, label: []const u8, active: u32 }{
        .{ .tail = "    k := b.map(", .label = "fn map(fn(T) -> U) -> Box[U]", .active = 0 },
        .{ .tail = "    k := b.fold(0, ", .label = "fn fold(A, fn(A, T) -> A) -> A", .active = 1 },
        .{ .tail = "    k := b.map(fn(s) { s.nope() }", .label = "fn map(fn(T) -> U) -> Box[U]", .active = 0 },
    };
    for (cases) |c| {
        const src = try std.mem.concat(gpa, u8, &.{ pre, c.tail });
        defer gpa.free(src);
        const p = test_util.posAfterLast(src, c.tail);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no signature help after '{s}'\n", .{c.tail});
            return error.NoSignatureHelp;
        };
        defer r.deinit();
        try testing.expectEqualStrings(c.label, r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
    }
}

test "signatureHelpAt: a method pipe's piped value is the receiver, not an argument" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct Acc { n: int }
        \\impl Acc {
        \\    fn plus(self, a: int, b: int) -> int { return self.n + a + b }
        \\}
        \\fn main() -> int {
        \\    acc := Acc { n: 1 }
        \\    y := 1 |> acc.plus(2)
        \\    z := acc |> .plus(1, 2)
        \\    return y + z
        \\}
        \\
    ;
    const cases = [_]struct { after: []const u8, active: u32 }{
        .{ .after = "|> .plus(", .active = 0 },
        .{ .after = "|> .plus(1, ", .active = 1 },
        .{ .after = "1 |> acc.plus(", .active = 1 },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no signature help after '{s}'\n", .{c.after});
            return error.NoSignatureHelp;
        };
        defer r.deinit();
        try testing.expectEqualStrings("fn plus(int, int) -> int", r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
    }
}

test "signatureHelpAt: a user protocol's method is labelled from its declaration" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\protocol Finder { fn find(self, k: Option[int], n: int) -> Option[int] }
        \\struct P { x: int }
        \\impl P has Finder { fn find(self, k: Option[int], n: int) -> Option[int] { k } }
        \\fn look[T has Finder](v: T) -> int { return v.find(Option.some(1), 2).unwrap_or(0) }
        \\fn main() -> int { return look(P { x: 1 }) }
        \\
    ;
    const cases = [_]struct { after: []const u8, active: u32 }{
        .{ .after = "v.find(", .active = 0 },
        .{ .after = "v.find(Option.some(1), ", .active = 1 },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse
            return error.NoSignatureHelp;
        defer r.deinit();
        const label = "fn find(Option[int], int) -> Option[int]";
        try testing.expectEqualStrings(label, r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
        const params = r.help.signatures[0].parameters;
        try testing.expectEqual(@as(usize, 2), params.len);
        try testing.expectEqualStrings("Option[int]", label[params[0].label[0]..params[0].label[1]]);
        try testing.expectEqualStrings("int", label[params[1].label[0]..params[1].label[1]]);
    }
}

test "signatureHelpAt: an unterminated method call keeps its selection" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    // The repaired `acc.plus(1, )` is an arity error; the selection survives it.
    const src =
        \\struct Acc { n: int }
        \\impl Acc {
        \\    fn plus(self, a: int, b: int) -> int { return self.n + a + b }
        \\}
        \\fn main() -> int {
        \\    acc := Acc { n: 1 }
        \\
    ++ "    x := acc.plus(1, ";
    const p = test_util.posAfterLast(src, "acc.plus(1, ");
    var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse
        return error.NoSignatureHelp;
    defer r.deinit();
    try testing.expectEqualStrings("fn plus(int, int) -> int", r.help.signatures[0].label);
    try testing.expectEqual(@as(u32, 1), r.help.activeParameter);
}

test "signatureHelpAt: a mut param's label says mut" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct P { x: int }
        \\impl P {
        \\    fn bump(mut self, mut d: int) { self.x = self.x + d }
        \\}
        \\fn inc(mut n: int, k: int) { n = n + k }
        \\fn main() -> int {
        \\    c := 1
        \\    inc(c, 2)
        \\    p := P { x: 1 }
        \\    p.bump(c)
        \\    return c + p.x
        \\}
        \\
    ;
    const cases = [_]struct { after: []const u8, label: []const u8 }{
        .{ .after = "inc(", .label = "fn inc(mut int, int) -> ()" },
        .{ .after = "p.bump(", .label = "fn bump(mut int) -> ()" },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no signature help after '{s}'\n", .{c.after});
            return error.NoSignatureHelp;
        };
        defer r.deinit();
        try testing.expectEqualStrings(c.label, r.help.signatures[0].label);
    }
}
