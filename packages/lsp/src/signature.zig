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
//! The callee's signature is read from `resolutions` -> `sigs` (the same path hover uses for
//! a callee), rendered through `render.bareName` so the module qualifier never leaks; param
//! labels are [start,end) byte spans into the rendered label.
//!
//! KNOWN LIMITATIONS (documented, not bugs):
//!   * A turbofish call `id[int](` — the token before `(` is `]`, not an identifier — yields
//!     no help. Free fns and module-qualified free fns (`io.f(`) are correct.
//!   * An instance-method call `recv.m(` renders the full signature including the `self`
//!     param as param 0, so `activeParameter` is off by one for methods. Only free / module-
//!     qualified free fns are exact.
//!   * A parse error OTHER than the appended closers (or a stray unmatched closer) re-triggers
//!     discovery's whole-tree discard -> null (the single-edit repair limit, as in completion).
//!   * Signature help may fire inside a fn DECLARATION's own param parens (`(` preceded by the
//!     decl name); harmless.

const std = @import("std");
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const position = @import("position.zig");
const render = @import("render.zig");
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

/// Mirrors `render.renderSig` byte-for-byte while capturing each param's [start,end) span, so
/// a param label always points at the RIGHT occurrence even when two params share a type
/// name. Type/name bytes are borrowed off the compiler tables; the appends copy them into `a`.
fn renderLabel(
    a: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    sig: anytype,
    layouts: anytype,
    enum_layouts: anytype,
    ranges: *std.ArrayList([2]u32),
) !void {
    try buf.appendSlice(a, "fn ");
    try buf.appendSlice(a, render.bareName(sig.name));
    try buf.append(a, '(');
    for (sig.params, 0..) |p, i| {
        if (i != 0) try buf.appendSlice(a, ", ");
        const s: u32 = @intCast(buf.items.len);
        try render.renderType(a, buf, p, layouts, enum_layouts);
        try ranges.append(a, .{ s, @intCast(buf.items.len) });
    }
    try buf.appendSlice(a, ") -> ");
    try render.renderType(a, buf, sig.ret, layouts, enum_layouts);
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

    const off = position.positionToOffset(&sm, line, character) orelse return null;
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

    const m = graph.entry();
    const resolutions = res.resolutions[graph.entry_index];

    // The callee node keys on its identifier token's source start (robust to node/token
    // reshuffle). Its resolution is `.func`; the call node's own main_token is the `(`, a
    // different start, so the two never collide.
    var fid: ?u32 = null;
    for (m.nodes, 0..) |n, i| {
        if (m.tokens[n.main_token].start != enc.callee_start) continue;
        if (i < resolutions.len and resolutions[i] == .func and resolutions[i].func < tc.sigs.len) {
            fid = resolutions[i].func;
            break;
        }
    }
    const f = fid orelse return null;
    const sig = tc.sigs[f];

    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    var label: std.ArrayList(u8) = .empty;
    var ranges: std.ArrayList([2]u32) = .empty;
    try renderLabel(a, &label, sig, tc.layouts, tc.enum_layouts, &ranges);

    const params = try a.alloc(protocol.ParameterInformation, ranges.items.len);
    for (ranges.items, 0..) |r, i| params[i] = .{ .label = r };

    var active = enc.active;
    if (sig.params.len == 0) {
        active = 0;
    } else if (active > sig.params.len - 1) {
        active = @intCast(sig.params.len - 1);
    }

    const sigs = try a.alloc(protocol.SignatureInformation, 1);
    sigs[0] = .{ .label = label.items, .parameters = params };

    return .{ .arena = arena, .help = .{ .signatures = sigs, .activeSignature = 0, .activeParameter = active } };
}

const testing = std.testing;

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
        tok(.identifier, 0, 1), tok(.l_paren, 1, 2),  tok(.identifier, 2, 3),
        tok(.l_paren, 3, 4),    tok(.number, 4, 5),   tok(.comma, 5, 6),
        tok(.number, 6, 7),     tok(.r_paren, 7, 8),  tok(.comma, 8, 9),
        tok(.number, 10, 11),   tok(.r_paren, 11, 12),
    };
    // Cursor past the OUTER comma (offset 10): the inner `,` at depth 1 is excluded.
    try testing.expectEqual(@as(u32, 1), activeParam(&toks, 1, 10));
}

test "activeParam: a bracket comma does not count" {
    // `f(a[i,j], 3)`
    const toks = [_]Token{
        tok(.identifier, 0, 1), tok(.l_paren, 1, 2),   tok(.identifier, 2, 3),
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
    const Sig = struct { name: []const u8, params: []const Type, ret: Type };
    const sig: Sig = .{ .name = "m.add", .params = &.{ int, int }, .ret = int };

    const no_layouts: []const Layout = &.{};
    const no_enums: []const EnumLayout = &.{};

    var label: std.ArrayList(u8) = .empty;
    var ranges: std.ArrayList([2]u32) = .empty;
    try renderLabel(a, &label, sig, no_layouts, no_enums, &ranges);

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
    const Sig = struct { name: []const u8, params: []const Type, ret: Type };
    const sig: Sig = .{ .name = "foo", .params = &.{}, .ret = int };

    const no_layouts: []const Layout = &.{};
    const no_enums: []const EnumLayout = &.{};

    var label: std.ArrayList(u8) = .empty;
    var ranges: std.ArrayList([2]u32) = .empty;
    try renderLabel(a, &label, sig, no_layouts, no_enums, &ranges);
    try testing.expectEqualStrings("fn foo() -> int", label.items);
    try testing.expectEqual(@as(usize, 0), ranges.items.len);
}
