//! The `toy check` emitter: run the front-end (lex -> parse -> resolve -> typecheck,
//! never codegen) over the input files, collect EVERY diagnostic (unlike a `-o` build,
//! which early-bails on the first tainted stage), apply the late C3 severity config,
//! and render either the human pretty form (reusing the shared Renderer) or a stable
//! line-delimited NDJSON stream.
//!
//! Split out of `main.zig` so the aggregation + NDJSON serializer are library-visible
//! (and unit-testable) rather than trapped in the CLI-only exe module: `test-bin` never
//! analyzes exe entrypoints, so any logic that must be covered lives here.
//!
//! - The NDJSON line schema is fixed and stable (keys emitted in a constant order, no
//!   trailing whitespace, RFC-8259 string escaping): `{"code","level","byte","line",
//!   "col","message"}`. `code` is null for an uncoded diagnostic. `line`/`col` are
//!   1-based, computed from the owning file's SourceMap.
//! - Both emitters gate on the SAME effective severity as the human renderer
//!   (`SevCfg.resolve`), so `--ignore`d diagnostics vanish from both forms and the
//!   error/warning counts match what the summary line prints.

// This file is INSIDE the `toy_compiler` library module (root.zig re-exports it as
// `Check`), so it reaches its peers by RELATIVE path — never `@import("toy_compiler")`,
// which is only visible to the sibling exe module. Matches Driver.zig's convention.
const std = @import("std");
const Io = std.Io;
const Driver = @import("Driver.zig");
const Diagnostic = @import("../diagnostics/Sink.zig").Diagnostic;
const codes = @import("../diagnostics/codes.zig");
const model = @import("../diagnostics/model.zig");
const SevCfg = @import("../diagnostics/severity_config.zig");
const Rr = struct {
    const SourceMap = @import("../term/render/SourceMap.zig");
};

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

/// Write `s` as a JSON string BODY (no surrounding quotes) with RFC-8259 escaping:
/// `"` and `\` are backslash-escaped, control bytes below 0x20 use the short escapes
/// where defined (`\n`,`\t`,`\r`,`\b`,`\f`) else `\u00XX`. Deterministic byte-for-byte
/// so golden NDJSON diffs are stable.
fn writeJsonStr(out: *Io.Writer, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try out.writeAll("\\\""),
            '\\' => try out.writeAll("\\\\"),
            '\n' => try out.writeAll("\\n"),
            '\t' => try out.writeAll("\\t"),
            '\r' => try out.writeAll("\\r"),
            0x08 => try out.writeAll("\\b"),
            0x0c => try out.writeAll("\\f"),
            else => if (c < 0x20) {
                try out.print("\\u{x:0>4}", .{c});
            } else {
                try out.writeByte(c);
            },
        }
    }
}

/// Serialize ONE diagnostic as a single NDJSON line (trailing '\n') against a prepared
/// SourceMap for its owning file. `eff` is the already-resolved effective severity.
/// Key order is FIXED: code, level, byte, line, col, message.
fn writeNdjsonLine(out: *Io.Writer, sm: *const Rr.SourceMap, d: Diagnostic, eff: model.Severity) !void {
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
    try out.print("\",\"byte\":{d},\"line\":{d},\"col\":{d},\"message\":\"", .{ d.byte_offset, lc.line, lc.col });
    try writeJsonStr(out, d.message);
    try out.writeAll("\"}\n");
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
            sm: *const Rr.SourceMap,
            cfg: SevCfg.SeverityConfig,
            counts: *Counts,
            fn emit(c: @This(), d: Diagnostic) anyerror!void {
                const eff = SevCfg.resolve(d.code, d.severity, c.cfg) orelse return;
                try writeNdjsonLine(c.out, c.sm, d, eff);
                switch (eff) {
                    .err => c.counts.errors += 1,
                    .warning => c.counts.warnings += 1,
                    else => {},
                }
            }
        };
        try forEachDiag(r, Ctx{ .out = out, .sm = &sm, .cfg = cfg, .counts = &counts }, Ctx.emit);
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

// --- tests -----------------------------------------------------------------

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

test "writeNdjsonLine emits fixed key order with 1-based line/col and null code" {
    const src = "fn main() {\n  x\n}\n";
    var sm = try Rr.SourceMap.init(testing.allocator, "t.toy", src);
    defer sm.deinit(testing.allocator);
    var buf: [256]u8 = undefined;
    var w = Io.Writer.fixed(&buf);
    // offset of the 'x' on line 2: after "fn main() {\n  " == 14
    const off: u32 = 14;
    try writeNdjsonLine(&w, &sm, .{ .byte_offset = off, .message = "boom", .code = .none, .severity = .err }, .err);
    try testing.expectEqualStrings(
        "{\"code\":null,\"level\":\"error\",\"byte\":14,\"line\":2,\"col\":3,\"message\":\"boom\"}\n",
        w.buffered(),
    );
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
    const dummy = [_]Diagnostic{.{ .byte_offset = 0, .message = "boom", .code = .none, .severity = .err }};
    const soft = [_]Driver.FileResult{.{ .path = "x", .err = error.ParseError, .diags = &dummy }};
    try testing.expect(!anyHardError(&soft));

    // A clean file: no err, no diagnostics -> not hard.
    const clean = [_]Driver.FileResult{.{ .path = "x" }};
    try testing.expect(!anyHardError(&clean));

    // Mixed: any hard-error file trips the gate.
    const mixed = [_]Driver.FileResult{ .{ .path = "ok" }, .{ .path = "bad", .err = error.FileNotFound } };
    try testing.expect(anyHardError(&mixed));
}
