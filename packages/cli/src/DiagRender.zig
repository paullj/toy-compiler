//! Diagnostics rendering: turn sink diagnostics / graph errors into terminal bytes
//! via SourceMap + render.Diagnostic + the pretty Renderer (snippet + caret). Owns
//! the shared status palette (below) — the one hue set both this and Report use.
//!
//! Location fallback: the Renderer needs a Span to draw a snippet, so a Graph.Error
//! or EmitError with no `span` (or no loaded source) renders through
//! `renderPlainError` as the plain no-location `path: error: msg (detail)` shape.

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");
const codes = toyc.diagnostics.codes;
const SevCfg = toyc.diagnostics.severity_config;
const Graph = toyc.Graph;
const Codegen = toyc.DriverCodegen;
const cli = @import("cli/Cli.zig");
const term = toyc.term;
const Style = term.Style;
const Rr = term.render;

// Status palette. `err`/`faint` are sourced from `Theme` (the single source of the
// diagnostic hues) so the status table and the pretty snippet can never drift; every
// use goes through `Style.styled` (zero bytes at `.none`, so `.none` == plain).
pub const sty_err: Style.Style = Rr.Theme.plain.style(.err); // bright red 9, bold — the header word hue
pub const sty_ok: Style.Style = .{ .fg = .{ .ansi = 10 } }; // bright green — success, not a severity (no Theme equivalent)
pub const sty_head: Style.Style = .{ .bold = true };
pub const sty_faint: Style.Style = Rr.Theme.plain.secondaryStyle(); // dim — matches secondary labels

/// Renderer options for a colour level: ASCII carets (unicode off) keep output
/// byte-stable and gate-safe.
fn renderOpts(level: Style.ColorLevel) Rr.Renderer.RenderOpts {
    return .{ .color = level, .unicode = false };
}

/// RENDER-TIME cap: at most this many diagnostics are drawn per batch; the rest
/// collapse into one "... and N more" line. This is OUTPUT-ONLY — the collected /
/// returned diagnostic set stays complete and uncapped, so incremental/cached
/// fingerprints stay stable. A constant now; trivially a field/flag later.
pub const DIAG_CAP: usize = 100;

/// Emit the trailing "... and {total - shown} more" summary (dim) when a batch was
/// capped. No-op when nothing was dropped. Borrows no diagnostic strings; the count
/// is formatted into a stack buffer, so this never allocates.
pub fn renderCapSummary(out: *Io.Writer, level: Style.ColorLevel, total: usize, shown: usize) !void {
    if (total <= shown) return;
    var buf: [48]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, "... and {d} more", .{total - shown}) catch "... and more";
    try sty_faint.styled(out, level, line);
    try out.writeByte('\n');
}

/// Emit the diagnostic-count summary line "N error(s), M warning(s)" (skipping a zero
/// component); no-op when both are zero. The counts are already severity-config-resolved
/// by the caller (ignored diagnostics excluded). The `error(s)` count reads err-red when
/// non-zero, else the whole line is dim. Never allocates.
pub fn renderDiagSummary(out: *Io.Writer, level: Style.ColorLevel, n_err: usize, n_warn: usize) !void {
    if (n_err == 0 and n_warn == 0) return;
    var buf: [64]u8 = undefined;
    const line = if (n_err != 0 and n_warn != 0)
        std.fmt.bufPrint(&buf, "{d} error(s), {d} warning(s)", .{ n_err, n_warn }) catch "errors"
    else if (n_err != 0)
        std.fmt.bufPrint(&buf, "{d} error(s)", .{n_err}) catch "errors"
    else
        std.fmt.bufPrint(&buf, "{d} warning(s)", .{n_warn}) catch "warnings";
    const style = if (n_err != 0) sty_err else sty_faint;
    try style.styled(out, level, line);
    try out.writeByte('\n');
}

/// Count VISIBLE (non-`--ignore`d) diagnostics under `cfg`, so the "... and N more" cap
/// summary excludes suppressed ones. Empty `cfg` == identity (all visible).
fn countVisible(diags: []const toyc.DiagnosticSink.Diagnostic, cfg: SevCfg.SeverityConfig) usize {
    var n: usize = 0;
    for (diags) |d| {
        if (SevCfg.resolve(d.code, d.severity, cfg) != null) n += 1;
    }
    return n;
}

/// Render a resolve/type sink diagnostic against a prepared SourceMap. Severity is the
/// LATE override: `SevCfg.resolve` reads the POD default and applies `cfg` (empty ==
/// identity); a `null` result means the code was `--ignore`'d, so nothing is written and
/// `false` is returned (for cap/summary accounting). The POD is never rewritten, so the
/// cached blob stays rule-independent.
pub fn renderSinkDiag(out: *Io.Writer, level: Style.ColorLevel, sm: *const Rr.SourceMap, d: toyc.DiagnosticSink.Diagnostic, cfg: SevCfg.SeverityConfig) !bool {
    const eff = SevCfg.resolve(d.code, d.severity, cfg) orelse return false; // .ignore -> zero bytes
    var sec_buf: [1]Rr.Diagnostic.Label = undefined;
    const rich = Rr.Diagnostic.richFromPod(d, eff, codes.str(d.code), codes.relatedLabel(d.code), &sec_buf);
    try Rr.Renderer.render(rich, sm, out, renderOpts(level));
    return true;
}

/// Render a LOCATED error: with an offset, build a one-off SourceMap over `name`/`src`
/// and a primary label at `[off, off)` carrying `message`; a non-empty `detail`
/// becomes a `= note:` footer. With no offset, fall back to `renderPlainError`
/// (detail inline as `(detail)`), preserving the old no-location shape.
fn renderLocated(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, name: []const u8, src: []const u8, span: ?Rr.Diagnostic.Span, message: []const u8, detail: []const u8) !void {
    const primary = span orelse return renderPlainError(out, level, name, message, detail);
    var sm = try Rr.SourceMap.init(gpa, name, src);
    defer sm.deinit(gpa);
    var notes_buf: [1]Rr.Diagnostic.Note = undefined;
    const notes: []const Rr.Diagnostic.Note = if (detail.len != 0) blk: {
        notes_buf[0] = .{ .kind = .note, .message = detail };
        break :blk notes_buf[0..1];
    } else &.{};
    const d = Rr.Diagnostic.Diagnostic{
        .severity = .err,
        .message = message,
        .primary = .{ .kind = .primary, .span = primary, .message = message },
        .notes = notes,
    };
    try Rr.Renderer.render(d, &sm, out, renderOpts(level));
}

/// The LOCATION-LESS fallback (no offset, or the arg-error path). Emits the old
/// `path: error: msg (detail)` shape with only the `error` word styled at `sty_err`
/// (bright red 9 bold — matching the Renderer's own header word hue). Gate-safe: at
/// `.none` `styled` writes only "error".
fn renderPlainError(out: *Io.Writer, level: Style.ColorLevel, path: []const u8, message: []const u8, detail: []const u8) !void {
    try out.print("{s}: ", .{path});
    try sty_err.styled(out, level, "error");
    try out.print(": {s}", .{message});
    if (detail.len != 0) try out.print(" ({s})", .{detail});
    try out.writeByte('\n');
}

/// The module a diagnostic's id points at. `NO_SCOPE` (single-file) and any
/// out-of-range id fall back to the entry module, so a foreign or stale module id
/// carried on a diagnostic can never index out of bounds once multi-module
/// diagnostics land. Assumes `g.modules` is non-empty (every build has modules).
fn moduleAt(g: *const Graph.Graph, id: u32) *const Graph.Module {
    return if (id < g.modules.len) &g.modules[id] else g.entry();
}

/// Render a batch of sink diagnostics (resolve/type) against their owning modules,
/// REUSING one SourceMap per scope: the diagnostics are sorted by (scope, offset) so
/// same-scope diagnostics are contiguous — we build a map per scope and keep it while
/// the scope holds. Scope `NO_SCOPE` (single-file) picks the entry module.
pub fn renderScopedDiags(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, g: *const Graph.Graph, diags: []const toyc.DiagnosticSink.Diagnostic, cfg: SevCfg.SeverityConfig) !void {
    // Apply the severity config before the cap so "... and N more" excludes --ignore'd
    // diagnostics; the passed-in `diags` slice is NEVER truncated (render-only).
    const visible = countVisible(diags, cfg);
    var cache: Rr.SourceMap.ScopeCache = .{};
    defer cache.deinit(gpa);
    var drawn: usize = 0;
    for (diags) |d| {
        if (SevCfg.resolve(d.code, d.severity, cfg) == null) continue; // --ignore: skip
        if (drawn == DIAG_CAP) break;
        const m = moduleAt(g, d.scope); // NO_SCOPE / out-of-range -> entry
        const sm = try cache.get(gpa, d.scope, m.path, m.source);
        _ = try renderSinkDiag(out, level, sm, d, cfg);
        drawn += 1;
    }
    try renderCapSummary(out, level, visible, drawn);
}

/// `toy check`: render EVERY diagnostic on a single file against ONE SourceMap over the
/// file, applying the severity config and the render cap. The caller passes a single
/// slice already sorted on the canonical (scope, byte_offset, code, message) key, so
/// parse/resolve/typecheck diagnostics interleave in SOURCE order rather than by stage.
/// Unlike the inspection table's `printFailure`, this never prints a summary row — the
/// `check` action reports diagnostics only, then a program-wide `N error(s), M warning(s)`
/// summary; the caller tallies the visible counts separately for the exit gate + summary.
pub fn renderFileDiags(
    gpa: std.mem.Allocator,
    out: *Io.Writer,
    level: Style.ColorLevel,
    path: []const u8,
    source: []const u8,
    diags: []const toyc.DiagnosticSink.Diagnostic,
    cfg: SevCfg.SeverityConfig,
) !void {
    if (diags.len == 0) return;
    const visible = countVisible(diags, cfg);
    var sm = try Rr.SourceMap.init(gpa, path, source);
    defer sm.deinit(gpa);
    var drawn: usize = 0;
    for (diags) |d| {
        if (drawn == DIAG_CAP) break;
        if (try renderSinkDiag(out, level, &sm, d, cfg)) drawn += 1; // false == --ignore'd
    }
    try renderCapSummary(out, level, visible, drawn);
}

/// Render a graph-discovery structural error against the owning module's source (or
/// the entry path when no module loaded). With a location, snippet + `= note: detail`;
/// otherwise the plain `path: error: msg (detail)` fallback.
pub fn renderGraphError(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, g: *const Graph.Graph, e: Graph.Error, cfg: SevCfg.SeverityConfig) !void {
    // A tainted parse carries the module's FULL coded diagnostic list; render those
    // (codes + carets) so `-o` matches `check`'s per-file output. Other kinds
    // (missing/escape/cycle) carry no parse_diags and fall through unchanged.
    if (e.kind == .parse and e.parse_diags.len > 0) {
        const m = moduleAt(g, e.module orelse g.entry_index);
        var sm = try Rr.SourceMap.init(gpa, m.path, m.source);
        defer sm.deinit(gpa);
        for (e.parse_diags) |d| _ = try renderSinkDiag(out, level, &sm, d, cfg);
        return;
    }
    const name = if (e.module) |m| moduleAt(g, m).path else if (g.modules.len > 0) g.entry().path else "<entry>";
    const has_src = e.module != null or g.modules.len > 0;
    if (e.span != null and has_src) {
        const src: []const u8 = if (e.module) |m| moduleAt(g, m).source else g.entry().source;
        try renderLocated(gpa, out, level, name, src, e.span, e.message, e.detail);
    } else {
        try renderPlainError(out, level, name, e.message, e.detail);
    }
}

/// Render a code-emission `EmitError` from a graph build against its owning module
/// (or the entry module when `module` is null). EmitError carries no detail.
pub fn renderGraphEmit(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, g: *const Graph.Graph, e: Codegen.EmitError) !void {
    const m = if (e.module) |mi| moduleAt(g, mi) else g.entry();
    if (e.span) |span| {
        try renderLocated(gpa, out, level, m.path, m.source, span, e.message, "");
    } else {
        try renderPlainError(out, level, m.path, e.message, "");
    }
}

/// Format + print every accumulated `cli.Sink` error (the parser never prints —
/// this REPLACES the old `argError` for grammar/parse failures). One styled
/// `error: <detail>` line per error; a trailing `Try 'toy --help' ...` hint. Every
/// string in a Sink.Error is BORROWED from argv, so this must run before argv frees.
pub fn printCliErrors(out: *Io.Writer, level: Style.ColorLevel, errs: []const cli.Sink.Error) !void {
    for (errs) |e| {
        try sty_err.styled(out, level, "error");
        try out.writeAll(": ");
        switch (e.kind) {
            .unknown_flag => try out.print("unknown flag: {s}", .{e.arg}),
            .missing_value => try out.print("{s} requires a value ({s})", .{ e.arg, e.expected }),
            // Enum bad_value carries the choice list; list it as `(one of: a|b|c)` and,
            // when the input was close to a choice, add a faint `did you mean 'X'?` note.
            // Non-enum values keep the `(expected int)`-style wording, same reordered shape.
            .bad_value => {
                if (e.choices.len != 0) {
                    try out.print("invalid value '{s}' for {s} (one of: {s})", .{ e.got, e.arg, e.choices });
                    if (e.suggestion.len != 0) {
                        try out.writeByte('\n');
                        var buf: [96]u8 = undefined;
                        const s = std.fmt.bufPrint(&buf, "did you mean '{s}'?", .{e.suggestion}) catch "did you mean";
                        try sty_faint.styled(out, level, s);
                    }
                } else {
                    try out.print("invalid value '{s}' for {s} (expected {s})", .{ e.got, e.arg, e.expected });
                }
            },
            .missing_required => try out.print("missing required argument: {s}", .{e.arg}),
            .unexpected_arg => try out.print("unexpected argument: {s}", .{e.arg}),
            .conflict => try out.print("{s} conflicts with {s}", .{ e.arg, e.where }),
            .unmet_requirement => try out.print("{s} requires {s}", .{ e.arg, e.where }),
        }
        try out.writeByte('\n');
    }
    try sty_faint.styled(out, level, "Try 'toy --help' for more information.");
    try out.writeByte('\n');
}
