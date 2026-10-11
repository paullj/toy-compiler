//! Bridge from the compiler's check front-end to LSP diagnostics. Reads an `Analysis`,
//! mirroring `toy check`: a discovery error alone, else resolve's diagnostics and, when
//! resolve is clean, typecheck's. Diagnostics belonging to an IMPORTED module are dropped
//! from this publish: they carry byte offsets into their own source and belong to their
//! own URIs.

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const position = @import("position.zig");
const Analysis = @import("Analysis.zig");

const Graph = toyc.Graph;
const Decide = toyc.Decide;
const Sink = toyc.DiagnosticSink;
const Diagnostic = Sink.Diagnostic;
const codes = toyc.diagnostics.codes;
const model = toyc.diagnostics.model;
const SourceMap = toyc.term.render.SourceMap;

/// The LSP diagnostics of `an`'s document, allocated in `a` (strings included). A compile
/// problem is reported as diagnostics, never an error.
pub fn collect(a: std.mem.Allocator, an: *const Analysis) ![]const protocol.LspDiagnostic {
    const graph = &an.graph;
    const sm = &an.sm;
    var out: std.ArrayList(protocol.LspDiagnostic) = .empty;

    if (graph.err) |ge| {
        // A tainted parse in the ENTRY carries its full coded diagnostic list; map it.
        const in_entry = ge.module == null or ge.module == graph.entry_index;
        if (ge.kind == .parse and in_entry) {
            try mapInto(a, sm, an.encoding, graph, an.uri, &out, ge.parse_diags);
        } else if (in_entry) {
            // Any other structural error that belongs to the entry and carries an offset
            // (an unresolvable/mis-cased import token) becomes one diagnostic there.
            if (ge.span) |span| {
                try out.append(a, .{
                    .range = position.rangeOf(sm, span.start, span.end, an.encoding),
                    .severity = protocol.severity.err,
                    .code = codes.str(ge.code()),
                    .message = try a.dupe(u8, ge.message),
                });
            }
        }
        // A structural error owned by an imported module belongs to its own URI: dropped.
        return out.items;
    }

    const res = an.res();
    const type_diags: []const Diagnostic = if (Decide.resolveHasError(res.diags)) &.{} else an.tc().diags;

    // Merge resolve + type diagnostics and re-sort on the canonical key so they surface in
    // source order across stages (a resolve warning and a type error interleave by line).
    const combined = try a.alloc(Diagnostic, res.diags.len + type_diags.len);
    @memcpy(combined[0..res.diags.len], res.diags);
    @memcpy(combined[res.diags.len..], type_diags);
    Sink.sortSlice(combined);

    try mapInto(a, sm, an.encoding, graph, an.uri, &out, combined);
    return out.items;
}

/// Append every diagnostic that belongs to the ENTRY module (unscoped, or the entry's
/// scope) to `out`, mapped against `sm`. Imported-module diagnostics are skipped.
fn mapInto(
    a: std.mem.Allocator,
    sm: *const SourceMap,
    enc: position.Encoding,
    graph: *const Graph.Graph,
    uri: []const u8,
    out: *std.ArrayList(protocol.LspDiagnostic),
    diags: []const Diagnostic,
) !void {
    for (diags) |d| {
        if (d.scope != Sink.NO_SCOPE and d.scope != graph.entry_index) continue;
        try out.append(a, try mapOne(a, sm, enc, uri, d));
    }
}

fn mapOne(a: std.mem.Allocator, sm: *const SourceMap, enc: position.Encoding, uri: []const u8, d: Diagnostic) !protocol.LspDiagnostic {
    var related: ?[]const protocol.Related = null;
    if (d.relatedSpan()) |rs| {
        const arr = try a.alloc(protocol.Related, 1);
        arr[0] = .{
            .location = .{ .uri = try a.dupe(u8, uri), .range = position.rangeOf(sm, rs.start, rs.end, enc) },
            .message = codes.relatedLabel(d.code),
        };
        related = arr;
    }
    return .{
        .range = position.rangeOf(sm, d.span().start, d.span().end, enc),
        .severity = lspSeverity(d.severity),
        // A static registry string; safe to reference after the graph is freed.
        .code = codes.str(d.code),
        .message = try a.dupe(u8, d.message),
        .relatedInformation = related,
    };
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
const Documents = @import("Documents.zig");
const Workspace = @import("Workspace.zig");

/// The diagnostics a publish of `src` (as `file:///doc.toy`) sends.
const Checked = struct {
    an: *Analysis,
    items: []const protocol.LspDiagnostic,

    fn deinit(c: *Checked) void {
        c.an.deinit();
    }
};

fn checkBuffer(gpa: std.mem.Allocator, src: []const u8) !Checked {
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const an = try @import("test_util.zig").analyzeIn(gpa, .{ .io = Io.failing, .docs = &docs, .disk = false }, "file:///doc.toy", src);
    return .{ .an = an, .items = an.diagnostics };
}

test "checkBuffer maps an arity mismatch to a T0039 on the erroring line" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const src = try Io.Dir.cwd().readFileAlloc(io, "tests/corpora/diagnostics/arity_mismatch.toy", gpa, .unlimited);
    defer gpa.free(src);

    var mapped = try checkBuffer(gpa, src);
    defer mapped.deinit();

    var found = false;
    for (mapped.items) |d| {
        if (d.code) |c| if (std.mem.eql(u8, c, "T0039")) {
            found = true;
            // `add(1)` sits on source line 4 (0-based 3) in the fixture; the compiler spans
            // exactly its argument list `(1)`.
            try testing.expectEqual(@as(u32, 3), d.range.start.line);
            try testing.expectEqual(d.range.start.line, d.range.end.line);
            try testing.expectEqual(d.range.start.character + 3, d.range.end.character);
            try testing.expectEqualStrings("defined here", d.relatedInformation.?[0].message);
        };
    }
    try testing.expect(found);
}

test "checkBuffer maps a piped T0039 to the |> step and each P0014 to its annotated line" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    {
        const src = "fn add(a: int, b: int) -> int { return a + b }\nfn main() -> int {\n    x := 1 |> add\n    return x\n}\n";
        var mapped = try checkBuffer(gpa, src);
        defer mapped.deinit();
        try testing.expectEqual(@as(usize, 1), mapped.items.len);
        const d = mapped.items[0];
        try testing.expectEqualStrings("T0039", d.code.?);
        const line = "    x := 1 |> add";
        try testing.expectEqual(@as(u32, 2), d.range.start.line);
        try testing.expectEqual(@as(u32, @intCast(std.mem.indexOf(u8, line, "|>").?)), d.range.start.character);
        try testing.expectEqual(@as(u32, @intCast(line.len)), d.range.end.character);
    }
    {
        const src = try Io.Dir.cwd().readFileAlloc(io, "tests/corpora/diagnostics/pipe_target.toy", gpa, .unlimited);
        defer gpa.free(src);
        var mapped = try checkBuffer(gpa, src);
        defer mapped.deinit();
        // Every P0014 lands on a line the fixture annotates with it, and each annotation
        // gets one, so a range that drifts to another line fails.
        var lines: std.ArrayList([]const u8) = .empty;
        defer lines.deinit(gpa);
        var it = std.mem.splitScalar(u8, src, '\n');
        while (it.next()) |l| try lines.append(gpa, l);
        var p0014: usize = 0;
        for (mapped.items) |d| {
            const c = d.code orelse continue;
            if (!std.mem.eql(u8, c, "P0014")) continue;
            p0014 += 1;
            try testing.expect(std.mem.indexOf(u8, lines.items[d.range.start.line], "#~ ERROR [P0014]") != null);
        }
        try testing.expectEqual(std.mem.count(u8, src, "#~ ERROR [P0014]"), p0014);
    }
}

test "checkBuffer codes a missing import and a call of a non-function" {
    const gpa = testing.allocator;
    const cases = [_]struct { src: []const u8, code: []const u8, line: u32 }{
        .{ .src = "import no_such_module\nfn main() -> int { return 0 }\n", .code = "R0003", .line = 0 },
        .{ .src = "fn main() -> int {\n    x := 1\n    return x(2)\n}\n", .code = "T0055", .line = 2 },
    };
    for (cases) |cs| {
        var mapped = try checkBuffer(gpa, cs.src);
        defer mapped.deinit();
        try testing.expectEqual(@as(usize, 1), mapped.items.len);
        try testing.expectEqualStrings(cs.code, mapped.items[0].code.?);
        try testing.expectEqual(cs.line, mapped.items[0].range.start.line);
    }
}

test "checkBuffer publishes a std-shadowing W0011 as a warning on its annotated line" {
    try expectWarningsOnAnnotatedLines("tests/corpora/diagnostics/shadows_std_method.toy", "W0011");
}

test "checkBuffer publishes an uncalled-function W0012 as a warning on its annotated line" {
    try expectWarningsOnAnnotatedLines("tests/corpora/diagnostics/uncalled_function.toy", "W0012");
}

test "checkBuffer publishes a fn-field and method name clash W0013 as a warning on its annotated line" {
    try expectWarningsOnAnnotatedLines("tests/corpora/diagnostics/fn_field_method_clash.toy", "W0013");
}

/// Every `code` diagnostic of the fixture at `path` is a warning on a line annotated
/// `#~ WARN [code]`, and each annotation gets one.
fn expectWarningsOnAnnotatedLines(path: []const u8, code: []const u8) !void {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const src = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(src);

    var mapped = try checkBuffer(gpa, src);
    defer mapped.deinit();

    var buf: [32]u8 = undefined;
    const marker = try std.fmt.bufPrint(&buf, "#~ WARN [{s}]", .{code});
    var lines: std.ArrayList([]const u8) = .empty;
    defer lines.deinit(gpa);
    var it = std.mem.splitScalar(u8, src, '\n');
    while (it.next()) |l| try lines.append(gpa, l);
    var seen: usize = 0;
    for (mapped.items) |d| {
        const c = d.code orelse continue;
        if (!std.mem.eql(u8, c, code)) continue;
        seen += 1;
        try testing.expectEqual(protocol.severity.warning, d.severity);
        try testing.expect(std.mem.indexOf(u8, lines.items[d.range.start.line], marker) != null);
    }
    try testing.expectEqual(std.mem.count(u8, src, marker), seen);
}
