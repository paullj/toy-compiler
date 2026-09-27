//! Bridge from the compiler's check front-end to LSP diagnostics. The graph sequence
//! mirrors `toy check`: discover -> resolveGraph -> (if resolve is clean) checkGraph, then
//! stop before lower/codegen. Diagnostics belonging to an IMPORTED module are dropped from
//! this publish — they carry byte offsets into their own source and belong to their own URIs.

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const hover = @import("hover.zig");
const Workspace = @import("Workspace.zig");
const Documents = @import("Documents.zig");

const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Decide = toyc.Decide;
const Sink = toyc.DiagnosticSink;
const Diagnostic = Sink.Diagnostic;
const codes = toyc.diagnostics.codes;
const model = toyc.diagnostics.model;
const SourceMap = toyc.term.render.SourceMap;
const Token = toyc.Token;

/// Owns the arena backing `items` (their message/related strings). `items` stays valid
/// until `deinit`; the compiler graph it was mapped from has already been torn down.
pub const Mapped = struct {
    arena: std.heap.ArenaAllocator,
    items: []const protocol.LspDiagnostic,

    pub fn deinit(self: *Mapped) void {
        self.arena.deinit();
    }
};

/// Check `source` (an open document's buffer) and return the mapped LSP diagnostics. `uri` is only used as the URI of any related-location. A
/// hard I/O / OOM failure propagates; a compile problem is reported as diagnostics, never
/// an error.
pub fn checkBuffer(
    gpa: std.mem.Allocator,
    ws: Workspace,
    uri: []const u8,
    source: []const u8,
) !Mapped {
    var graph = try ws.discover(gpa, uri, source);
    defer graph.deinit(gpa);

    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    var sm = try SourceMap.init(gpa, graph.entry().file, graph.entry().source);
    defer sm.deinit(gpa);
    const src: Source = .{ .sm = &sm, .tokens = graph.entry().tokens };

    var out: std.ArrayList(protocol.LspDiagnostic) = .empty;

    if (graph.err) |ge| {
        // A tainted parse in the ENTRY carries its full coded diagnostic list; map it.
        const in_entry = ge.module == null or ge.module == graph.entry_index;
        if (ge.kind == .parse and in_entry) {
            try mapInto(a, src, &graph, uri, &out, ge.parse_diags);
        } else if (in_entry) {
            // Any other structural error that belongs to the entry and carries an offset
            // (an unresolvable/mis-cased import token) becomes one diagnostic there.
            if (ge.byte_offset) |off| {
                try out.append(a, try mapStructural(a, src, off, ge.message));
            }
        }
        // A structural error owned by an imported module belongs to its own URI: dropped.
        return .{ .arena = arena, .items = out.items };
    }

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);

    var tc: ?TypecheckGraph.GraphResult = null;
    defer if (tc) |*t| t.deinit(gpa);
    if (!Decide.resolveHasError(res.diags)) {
        tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, ws.io, 0);
    }
    const type_diags: []const Diagnostic = if (tc) |t| t.diags else &.{};

    // Merge resolve + type diagnostics and re-sort on the canonical key so they surface in
    // source order across stages (a resolve warning and a type error interleave by line).
    const combined = try gpa.alloc(Diagnostic, res.diags.len + type_diags.len);
    defer gpa.free(combined);
    @memcpy(combined[0..res.diags.len], res.diags);
    @memcpy(combined[res.diags.len..], type_diags);
    Sink.sortSlice(combined);

    try mapInto(a, src, &graph, uri, &out, combined);
    return .{ .arena = arena, .items = out.items };
}

/// The entry module's text and tokens: what a diagnostic's byte offset is mapped against.
const Source = struct {
    sm: *const SourceMap,
    tokens: []const Token,
};

/// Append every diagnostic that belongs to the ENTRY module (unscoped, or the entry's
/// scope) to `out`, mapped against `src`. Imported-module diagnostics are skipped.
fn mapInto(
    a: std.mem.Allocator,
    src: Source,
    graph: *const Graph.Graph,
    uri: []const u8,
    out: *std.ArrayList(protocol.LspDiagnostic),
    diags: []const Diagnostic,
) !void {
    for (diags) |d| {
        if (d.scope != Sink.NO_SCOPE and d.scope != graph.entry_index) continue;
        try out.append(a, try mapOne(a, src, uri, d));
    }
}

fn mapOne(a: std.mem.Allocator, src: Source, uri: []const u8, d: Diagnostic) !protocol.LspDiagnostic {
    var related: ?[]const protocol.Related = null;
    if (d.related != Sink.NO_RELATED) {
        const arr = try a.alloc(protocol.Related, 1);
        arr[0] = .{
            .location = .{ .uri = try a.dupe(u8, uri), .range = tokenRange(src, d.related) },
            .message = codes.relatedLabel(d.code),
        };
        related = arr;
    }
    return .{
        .range = tokenRange(src, d.byte_offset),
        .severity = lspSeverity(d.severity),
        // A static registry string; safe to reference after the graph is freed.
        .code = codes.str(d.code),
        .message = try a.dupe(u8, d.message),
        .relatedInformation = related,
    };
}

fn mapStructural(a: std.mem.Allocator, src: Source, off: u32, message: []const u8) !protocol.LspDiagnostic {
    return .{
        .range = tokenRange(src, off),
        .severity = protocol.severity.err,
        .message = try a.dupe(u8, message),
    };
}

/// The range of the token at `off`. A diagnostic records only the offset where its token
/// starts; an editor needs an extent to underline and to hover. An opening `(` / `[` is a
/// call's or index's argument list (e.g. an arity error), so its range runs to the matching
/// close. Off any token (a gap, or EOF for an unterminated construct) the range is empty.
///
/// `SourceMap.lineCol` is 1-based line + 1-based BYTE column; LSP wants 0-based line +
/// 0-based character. Byte column equals the UTF-16 character for ASCII; a utf-16-only
/// client on genuinely multibyte source would need a byte->utf-16 pass over `sm.lineText`
/// here — out of scope while we advertise utf-8.
fn tokenRange(src: Source, off: u32) protocol.Range {
    const end = if (hover.tokenAt(src.tokens, off)) |t| src.tokens[matchingClose(src.tokens, t)].end else off;
    return .{ .start = position(src.sm, off), .end = position(src.sm, end) };
}

/// The token closing the group `open` opens, or `open` itself when it opens none (or the
/// group is unterminated).
fn matchingClose(tokens: []const Token, open: u32) u32 {
    const close: toyc.Tag = switch (tokens[open].tag) {
        .l_paren => .r_paren,
        .l_bracket => .r_bracket,
        else => return open,
    };
    const tag = tokens[open].tag;
    var depth: u32 = 0;
    for (tokens[open..], open..) |t, i| {
        if (t.tag == tag) depth += 1;
        if (t.tag == close) {
            depth -= 1;
            if (depth == 0) return @intCast(i);
        }
    }
    return open;
}

fn position(sm: *const SourceMap, off: u32) protocol.Position {
    const lc = sm.lineCol(off);
    return .{ .line = @intCast(lc.line - 1), .character = @intCast(lc.col - 1) };
}

fn lspSeverity(s: model.Severity) u8 {
    return switch (s) {
        .err => protocol.severity.err,
        .warning => protocol.severity.warning,
        .note => protocol.severity.information,
        .help => protocol.severity.hint,
    };
}

const testing = std.testing;

test "checkBuffer maps an arity mismatch to a T0039 on the erroring line" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const src = try Io.Dir.cwd().readFileAlloc(io, "tests/corpora/diagnostics/arity_mismatch.toy", gpa, .unlimited);
    defer gpa.free(src);

    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = io, .docs = &docs, .disk = false };
    var mapped = try checkBuffer(gpa, ws, "file:///doc.toy", src);
    defer mapped.deinit();

    var found = false;
    for (mapped.items) |d| {
        if (d.code) |c| if (std.mem.eql(u8, c, "T0039")) {
            found = true;
            // `add(1)` sits on source line 4 (0-based 3) in the fixture; the range spans
            // exactly its argument list `(1)`.
            try testing.expectEqual(@as(u32, 3), d.range.start.line);
            try testing.expectEqual(d.range.start.line, d.range.end.line);
            try testing.expectEqual(d.range.start.character + 3, d.range.end.character);
            try testing.expectEqualStrings("defined here", d.relatedInformation.?[0].message);
        };
    }
    try testing.expect(found);
}
