//! The `toy check` emitter: run the front-end (lex -> parse -> resolve -> typecheck,
//! never codegen) over the input files, collect EVERY diagnostic (unlike a `-o` build,
//! which early-bails on the first tainted stage), apply the late severity config,
//! and render either the human pretty form (reusing the shared Renderer) or a stable
//! line-delimited NDJSON stream.
//!
//! Split out of `main.zig` so the aggregation + NDJSON serializer are library-visible
//! (and unit-testable) rather than trapped in the CLI-only exe module: `test-bin` never
//! analyzes exe entrypoints, so any logic that must be covered lives here.
//!
//! - The NDJSON line schema is fixed and stable (keys emitted in a constant order, no
//!   trailing whitespace, RFC-8259 string escaping): `{"code","level","byte","byte_end",
//!   "line","col","message","file","rendered","labels"}`. `byte`..`byte_end` is the
//!   primary span. `code` is null for an uncoded
//!   diagnostic. `line`/`col` are 1-based, computed from the owning file's SourceMap.
//!   `file` is the diagnostic's source path (the module path in graph mode, else the
//!   input file). `rendered` is the PLAIN (uncolored) pretty single-diagnostic render
//!   string, JSON-escaped (embedded newlines become `\n`). `labels` is an array of
//!   SECONDARY labels: a duplicate-definition carries its "previously defined here"
//!   label as `{message,file,line,col,byte_start,byte_end,is_primary:false}`.
//! - Both emitters gate on the SAME effective severity as the human renderer
//!   (`SevCfg.resolve`), so `--ignore`d diagnostics vanish from both forms and the
//!   error/warning counts match what the summary line prints.

// This file is INSIDE the `toy_compiler` library module (root.zig re-exports it as
// `Check`), so it reaches its peers by RELATIVE path — never `@import("toy_compiler")`,
// which is only visible to the sibling exe module. Matches Driver.zig's convention.
const std = @import("std");
const Io = std.Io;
const Driver = @import("Driver.zig");
const Graph = @import("Graph.zig");
const Diagnostic = @import("../diagnostics/Sink.zig").Diagnostic;
const NO_SCOPE = @import("../diagnostics/Sink.zig").NO_SCOPE;
const codes = @import("../diagnostics/codes.zig");
const model = @import("../diagnostics/model.zig");
const SevCfg = @import("../diagnostics/severity_config.zig");
const Rr = struct {
    const SourceMap = @import("../term/render/SourceMap.zig");
    const Diagnostic = @import("../term/render/Diagnostic.zig");
    const Renderer = @import("../term/render/Renderer.zig");
};
const Style = @import("../term/Style.zig");

/// The error/warning tally over the VISIBLE (non-`--ignore`d) diagnostics, by
/// effective severity. `notes`/`help` never count toward either.
pub const Counts = struct {
    errors: usize = 0,
    warnings: usize = 0,

    /// True when at least one diagnostic resolved to an error, so the process should
    /// exit non-zero (unless `--exit-zero` overrides).
    pub fn hasErrors(self: Counts) bool {
        return self.errors > 0;
    }
};

/// The effective-severity band for a diagnostic under `cfg`: null when `--ignore`d
/// (so both emitters drop it and it counts toward neither tally).
fn effective(d: Diagnostic, cfg: SevCfg.SeverityConfig) ?model.Severity {
    return SevCfg.resolve(d.code, d.severity, cfg);
}

/// Add one diagnostic's effective severity to `counts` (skipping ignored / note / help).
fn tally(d: Diagnostic, cfg: SevCfg.SeverityConfig, counts: *Counts) void {
    const s = effective(d, cfg) orelse return;
    switch (s) {
        .err => counts.errors += 1,
        .warning => counts.warnings += 1,
        else => {},
    }
}

/// The severity word emitted in the NDJSON `level` field (a lower-case tag matching the
/// `model.Severity` enum) — kept as a distinct fn so the wire spelling can never drift
/// from the render-side naming by accident.
fn levelWord(s: model.Severity) []const u8 {
    return switch (s) {
        .err => "error",
        .warning => "warning",
        .note => "note",
        .help => "help",
    };
}

/// Write `s` as a JSON string BODY (no surrounding quotes), reusing std's escaper.
/// Default options leave bytes >= 0x20 (incl. UTF-8) raw and escape `"`/`\`/control
/// bytes deterministically, so golden NDJSON diffs stay byte-stable.
fn writeJsonStr(out: *Io.Writer, s: []const u8) !void {
    try std.json.Stringify.encodeJsonStringChars(s, .{}, out);
}

/// Serialize ONE diagnostic as a single NDJSON line (trailing '\n') against a prepared
/// SourceMap for its owning file `file`. `eff` is the already-resolved effective
/// severity. Key order is FIXED: code, level, byte, line, col, message, file, rendered,
/// labels. `rendered` is the PLAIN (uncolored, ascii-caret) pretty single-diagnostic
/// render, JSON-escaped (embedded newlines/control bytes escaped). `labels` is the
/// diagnostic's secondary labels (empty `[]` when none); a duplicate-definition's
/// "previously defined here" label carries its own line/col/byte span, `is_primary:false`.
fn writeNdjsonLine(out: *Io.Writer, gpa: std.mem.Allocator, sm: *const Rr.SourceMap, file: []const u8, d: Diagnostic, eff: model.Severity) !void {
    const lc = sm.lineCol(d.byte_offset);
    try out.writeAll("{\"code\":");
    if (codes.str(d.code)) |cs| {
        try out.writeByte('"');
        try writeJsonStr(out, cs);
        try out.writeByte('"');
    } else {
        try out.writeAll("null");
    }
    try out.writeAll(",\"level\":\"");
    try writeJsonStr(out, levelWord(eff));
    try out.print("\",\"byte\":{d},\"byte_end\":{d},\"line\":{d},\"col\":{d},\"message\":\"", .{ d.byte_offset, d.span().end, lc.line, lc.col });
    try writeJsonStr(out, d.message);
    try out.writeAll("\",\"file\":\"");
    try writeJsonStr(out, file);

    // `rendered`: the PLAIN pretty single-diagnostic render (color .none, ascii carets),
    // captured into a temp buffer then JSON-escaped so embedded newlines/carets survive
    // as a single valid JSON string.
    var sec_buf: [1]Rr.Diagnostic.Label = undefined;
    const rich = model.richFromPod(d, eff, codes.str(d.code), codes.relatedLabel(d.code), &sec_buf);
    var aw: Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try Rr.Renderer.render(rich, sm, &aw.writer, .{ .color = Style.ColorLevel.none, .unicode = false });
    try out.writeAll("\",\"rendered\":\"");
    try writeJsonStr(out, aw.writer.buffered());

    // `labels`: the secondary labels (currently only the duplicate's "previously defined
    // here"). Each carries its own line/col computed off the SAME file's SourceMap (the
    // related offset is in the same scope) plus the byte span and `is_primary:false`.
    try out.writeAll("\",\"labels\":[");
    for (rich.secondary, 0..) |lbl, i| {
        if (i != 0) try out.writeByte(',');
        const llc = sm.lineCol(lbl.span.start);
        try out.writeAll("{\"message\":\"");
        try writeJsonStr(out, lbl.message);
        try out.writeAll("\",\"file\":\"");
        try writeJsonStr(out, file);
        try out.print("\",\"line\":{d},\"col\":{d},\"byte_start\":{d},\"byte_end\":{d},\"is_primary\":false}}", .{ llc.line, llc.col, lbl.span.start, lbl.span.end });
    }
    try out.writeAll("]}\n");
}

/// The per-file diagnostic sets a `check` run reports, in stable stage order:
/// parse diagnostics first (the tainted-parse `diags`), then resolve, then typecheck.
/// A caller iterates these so the human + NDJSON forms surface the SAME set.
/// (`Parser.Diagnostic` IS the shared `Diagnostic` POD, so `r.diags` is passed through
/// verbatim — no reconstruction.)
fn forEachDiag(
    r: Driver.FileResult,
    ctx: anytype,
    comptime f: fn (@TypeOf(ctx), Diagnostic) anyerror!void,
) anyerror!void {
    for (r.diags) |d| try f(ctx, d);
    if (r.resolve) |res| for (res.diags) |d| try f(ctx, d);
    if (r.typecheck) |tc| for (tc.diags) |d| try f(ctx, d);
}

/// Emit every diagnostic across `results` as NDJSON (one JSON object per line), applying
/// `cfg` so `--ignore`d ones vanish. Returns the visible error/warning tally. A file's
/// diagnostics render against its own SourceMap (built once per file).
pub fn emitNdjson(out: *Io.Writer, gpa: std.mem.Allocator, results: []const Driver.FileResult, cfg: SevCfg.SeverityConfig) !Counts {
    var counts: Counts = .{};
    for (results) |r| {
        // Build a SourceMap once per file; every diagnostic on the file uses it.
        var sm = try Rr.SourceMap.init(gpa, r.path, r.source);
        defer sm.deinit(gpa);
        const Ctx = struct {
            out: *Io.Writer,
            gpa: std.mem.Allocator,
            sm: *const Rr.SourceMap,
            file: []const u8,
            cfg: SevCfg.SeverityConfig,
            counts: *Counts,
            fn emit(c: @This(), d: Diagnostic) anyerror!void {
                const eff = SevCfg.resolve(d.code, d.severity, c.cfg) orelse return;
                try writeNdjsonLine(c.out, c.gpa, c.sm, c.file, d, eff);
                switch (eff) {
                    .err => c.counts.errors += 1,
                    .warning => c.counts.warnings += 1,
                    else => {},
                }
            }
        };
        try forEachDiag(r, Ctx{ .out = out, .gpa = gpa, .sm = &sm, .file = r.path, .cfg = cfg, .counts = &counts }, Ctx.emit);
    }
    return counts;
}

/// True when ANY result is a hard front-end/IO failure: `err` is set but the file
/// produced NO diagnostics across parse/resolve/typecheck (e.g. the file was missing or
/// unreadable, so it never reached a stage that could emit a compile diagnostic). Both
/// the human and NDJSON emitters gate exit 2 on this — sharing the predicate here means
/// the two forms cannot drift (a missing file must never look "checked clean" to an
/// NDJSON consumer). A file that DID emit diagnostics has `err` set to a stage-error
/// sentinel (ParseError/ResolveError/TypeError); those are compile failures (exit 1),
/// not hard errors, so the diagnostic-count guard excludes them.
pub fn anyHardError(results: []const Driver.FileResult) bool {
    for (results) |r| {
        const has_diags = r.diags.len != 0 or
            (if (r.resolve) |res| res.diags.len != 0 else false) or
            (if (r.typecheck) |tc| tc.diags.len != 0 else false);
        if (r.err != null and !has_diags) return true;
    }
    return false;
}

/// Count the visible error/warning tally across `results` under `cfg` WITHOUT rendering
/// (used by the human emitter, which renders via the shared Renderer path and just needs
/// the summary counts).
pub fn tallyAll(results: []const Driver.FileResult, cfg: SevCfg.SeverityConfig) Counts {
    var counts: Counts = .{};
    for (results) |r| {
        for (r.diags) |d| tally(d, cfg, &counts);
        if (r.resolve) |res| for (res.diags) |d| tally(d, cfg, &counts);
        if (r.typecheck) |tc| for (tc.diags) |d| tally(d, cfg, &counts);
    }
    return counts;
}

// The CLI `toy check` routes through the SAME graph front-end `build` uses
// (`Graph.discover` -> `resolveGraph` -> `checkGraph`), so imports are followed
// and a valid multi-module program reports ZERO diagnostics. The graph carries a
// FLAT diagnostic slice (resolve + typecheck), each tagged with its owning module
// via `Diagnostic.scope`; these helpers tally + serialize that slice, mapping each
// diagnostic's `file` to its OWNING module's path (`graph.modules[scope].path`),
// not always the entry. The NDJSON wire schema is byte-identical to the per-file
// form (`writeNdjsonLine` is shared), so a single-module graph emits exactly what
// the old single-file path did.

/// The module a diagnostic's scope points at: the owning module's on-disk file path,
/// so the NDJSON `file` field and the human render both attribute a diagnostic from
/// an imported module to THAT module. `NO_SCOPE` (single-file) or an out-of-range id
/// falls back to the entry module (matching `DiagRender.moduleAt`).
fn moduleFileFor(g: *const Graph.Graph, scope: u32) []const u8 {
    if (scope != NO_SCOPE and scope < g.modules.len) return g.modules[scope].file;
    return g.entry().file;
}

/// The owning module's SOURCE for a diagnostic's scope (same fallback as `moduleFileFor`).
fn moduleSrcFor(g: *const Graph.Graph, scope: u32) []const u8 {
    if (scope != NO_SCOPE and scope < g.modules.len) return g.modules[scope].source;
    return g.entry().source;
}

/// Count the visible error/warning tally across a graph's FLAT diagnostic slice under
/// `cfg` (the multi-module analogue of `tallyAll`). `--ignore`d and note/help
/// diagnostics count toward neither.
pub fn tallyGraph(diags: []const Diagnostic, cfg: SevCfg.SeverityConfig) Counts {
    var counts: Counts = .{};
    for (diags) |d| tally(d, cfg, &counts);
    return counts;
}

/// Emit every diagnostic in a graph's FLAT slice as NDJSON (one JSON object per line),
/// applying `cfg` so `--ignore`d ones vanish. Each diagnostic renders against its
/// OWNING module's SourceMap (built lazily, one per scope; the slice is pre-sorted by
/// (scope, offset) so same-scope diagnostics are contiguous and the map is reused).
/// The `file` field is the owning module's path. Returns the visible tally.
pub fn emitNdjsonGraph(out: *Io.Writer, gpa: std.mem.Allocator, g: *const Graph.Graph, diags: []const Diagnostic, cfg: SevCfg.SeverityConfig) !Counts {
    var counts: Counts = .{};
    var cache: Rr.SourceMap.ScopeCache = .{};
    defer cache.deinit(gpa);
    for (diags) |d| {
        const eff = SevCfg.resolve(d.code, d.severity, cfg) orelse continue; // --ignore
        const file = moduleFileFor(g, d.scope);
        const sm = try cache.get(gpa, d.scope, file, moduleSrcFor(g, d.scope));
        try writeNdjsonLine(out, gpa, sm, file, d, eff);
        switch (eff) {
            .err => counts.errors += 1,
            .warning => counts.warnings += 1,
            else => {},
        }
    }
    return counts;
}

const testing = std.testing;

test "writeJsonStr escapes quotes, backslashes, and control bytes" {
    var buf: [256]u8 = undefined;
    var w = Io.Writer.fixed(&buf);
    try writeJsonStr(&w, "a\"b\\c\nd\te");
    try testing.expectEqualStrings("a\\\"b\\\\c\\nd\\te", w.buffered());
}

test "writeJsonStr uses \\u00XX for other control bytes" {
    var buf: [64]u8 = undefined;
    var w = Io.Writer.fixed(&buf);
    try writeJsonStr(&w, &[_]u8{0x01});
    try testing.expectEqualStrings("\\u0001", w.buffered());
}

test "writeNdjsonLine emits fixed key order with 1-based line/col, null code, file, rendered, empty labels" {
    const src = "fn main() {\n  x\n}\n";
    var sm = try Rr.SourceMap.init(testing.allocator, "t.toy", src);
    defer sm.deinit(testing.allocator);
    var aw: Io.Writer.Allocating = .init(testing.allocator);
    defer aw.deinit();
    // offset of the 'x' on line 2: after "fn main() {\n  " == 14
    const off: u32 = 14;
    try writeNdjsonLine(&aw.writer, testing.allocator, &sm, "t.toy", .{ .byte_offset = off, .end = off, .message = "boom", .code = .none, .severity = .err }, .err);
    const got = aw.writer.buffered();
    // Prefix through the shared fields is stable and byte-checkable.
    try testing.expect(std.mem.startsWith(u8, got, "{\"code\":null,\"level\":\"error\",\"byte\":14,\"byte_end\":14,\"line\":2,\"col\":3,\"message\":\"boom\",\"file\":\"t.toy\",\"rendered\":\""));
    // No related location => empty labels array; the line ends with `,"labels":[]}\n`.
    try testing.expect(std.mem.endsWith(u8, got, ",\"labels\":[]}\n"));
    // Every embedded newline in the rendered snippet is escaped (no raw '\n' before the
    // trailing terminator), so the whole record stays one NDJSON line.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, got, "\n"));
}

test "writeNdjsonLine emits a secondary label for a related location (duplicate-definition shape)" {
    const src = "fn f() {}\nfn f() {}\n";
    var sm = try Rr.SourceMap.init(testing.allocator, "dup.toy", src);
    defer sm.deinit(testing.allocator);
    var aw: Io.Writer.Allocating = .init(testing.allocator);
    defer aw.deinit();
    // primary at the second `f` (13..14), related at the first `f` (3..4).
    try writeNdjsonLine(&aw.writer, testing.allocator, &sm, "dup.toy", .{ .byte_offset = 13, .end = 14, .message = "duplicate function 'f'", .code = .R0002, .severity = .err, .related = 3, .related_end = 4 }, .err);
    const got = aw.writer.buffered();
    try testing.expect(std.mem.indexOf(u8, got, "\"labels\":[{\"message\":\"previously defined here\",\"file\":\"dup.toy\"") != null);
    try testing.expect(std.mem.indexOf(u8, got, "\"is_primary\":false") != null);
    try testing.expect(std.mem.indexOf(u8, got, "\"byte_start\":3") != null);
    // Still exactly one NDJSON line.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, got, "\n"));
}

test "levelWord matches severity spelling" {
    try testing.expectEqualStrings("error", levelWord(.err));
    try testing.expectEqualStrings("warning", levelWord(.warning));
    try testing.expectEqualStrings("note", levelWord(.note));
    try testing.expectEqualStrings("help", levelWord(.help));
}

test "anyHardError: err with no diagnostics is hard; err with diagnostics is not" {
    // A missing/unreadable file: `err` set, zero diagnostics -> hard error (exit 2).
    const hard = [_]Driver.FileResult{.{ .path = "x", .err = error.FileNotFound }};
    try testing.expect(anyHardError(&hard));

    // A compile diagnostic: `err` set to a stage sentinel BUT diagnostics present ->
    // NOT a hard error (that path is exit 1). One parse diagnostic is enough.
    const dummy = [_]Diagnostic{.{ .byte_offset = 0, .end = 0, .message = "boom", .code = .none, .severity = .err }};
    const soft = [_]Driver.FileResult{.{ .path = "x", .err = error.ParseError, .diags = &dummy }};
    try testing.expect(!anyHardError(&soft));

    // A clean file: no err, no diagnostics -> not hard.
    const clean = [_]Driver.FileResult{.{ .path = "x" }};
    try testing.expect(!anyHardError(&clean));

    // Mixed: any hard-error file trips the gate.
    const mixed = [_]Driver.FileResult{ .{ .path = "ok" }, .{ .path = "bad", .err = error.FileNotFound } };
    try testing.expect(anyHardError(&mixed));
}
