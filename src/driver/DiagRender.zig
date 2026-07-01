//! Diagnostics rendering: turn sink diagnostics / graph errors into terminal bytes
//! via SourceMap + render.Diagnostic + the pretty Renderer (snippet + caret). Owns
//! the shared status palette (below) — the one hue set both this and Report use.
//!
//! Framework gap (worked around, not patched): the Renderer needs a Span to snippet,
//! so a Graph.Error / EmitError with no `byte_offset` (or no loaded source) falls
//! back to `renderPlainError` (`path: error: msg (detail)`), preserving the old
//! no-location shape.

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");
const Graph = toyc.Graph;
const Codegen = toyc.DriverCodegen;
const cli = toyc.cli;
const term = toyc.term;
const Style = term.Style;
const Rr = term.render;

// Status palette. Bright-ansi indices so `Color.downgrade` is the identity at ansi16/ansi256; every
// use goes through `Style.styled` (zero bytes at `.none`, so `.none` == plain).
// `err` matches the Renderer's own header word hue (bright red 9, bold).
pub const sty_err: Style.Style = .{ .fg = .{ .ansi = 9 }, .bold = true }; // Theme.plain.style(.err)
pub const sty_ok: Style.Style = .{ .fg = .{ .ansi = 10 } }; // bright green
pub const sty_head: Style.Style = .{ .bold = true };
pub const sty_faint: Style.Style = .{ .dim = true };

/// Renderer options for a colour level: ASCII carets (unicode off) keep output
/// byte-stable and gate-safe.
pub fn renderOpts(level: Style.ColorLevel) Rr.Renderer.RenderOpts {
    return .{ .color = level, .unicode = false };
}

/// Render a resolve/type sink diagnostic against a prepared SourceMap: a zero-width
/// primary `.err` at `d.byte_offset` (via `fromSink`), snippet + caret.
pub fn renderSinkDiag(out: *Io.Writer, level: Style.ColorLevel, sm: *const Rr.SourceMap, d: toyc.DiagnosticSink.Diagnostic) !void {
    try Rr.Renderer.render(Rr.Diagnostic.fromSink(d), sm, out, renderOpts(level));
}

/// Render a LOCATED error: with an offset, build a one-off SourceMap over `name`/`src`
/// and a primary label at `[off, off)` carrying `message`; a non-empty `detail`
/// becomes a `= note:` footer. With no offset, fall back to `renderPlainError`
/// (detail inline as `(detail)`), preserving the old no-location shape.
pub fn renderLocated(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, name: []const u8, src: []const u8, byte_offset: ?u32, message: []const u8, detail: []const u8) !void {
    const off = byte_offset orelse return renderPlainError(out, level, name, message, detail);
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
        .primary = .{ .kind = .primary, .span = .{ .start = off, .end = off }, .message = message },
        .notes = notes,
    };
    try Rr.Renderer.render(d, &sm, out, renderOpts(level));
}

/// The LOCATION-LESS fallback (no offset, or the arg-error path). Emits the old
/// `path: error: msg (detail)` shape with only the `error` word styled at `sty_err`
/// (bright red 9 bold — matching the Renderer's own header word hue). Gate-safe: at
/// `.none` `styled` writes only "error".
pub fn renderPlainError(out: *Io.Writer, level: Style.ColorLevel, path: []const u8, message: []const u8, detail: []const u8) !void {
    try out.print("{s}: ", .{path});
    try sty_err.styled(out, level, "error");
    try out.print(": {s}", .{message});
    if (detail.len != 0) try out.print(" ({s})", .{detail});
    try out.writeByte('\n');
}

/// Render a batch of sink diagnostics (resolve/type) against their owning modules,
/// REUSING one SourceMap per scope: the diagnostics are sorted by (scope, offset) so
/// same-scope diagnostics are contiguous — we build a map per scope and keep it while
/// the scope holds. Scope `NO_SCOPE` (single-file) picks the entry module.
pub fn renderScopedDiags(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, g: *const Graph.Graph, diags: []const toyc.DiagnosticSink.Diagnostic) !void {
    var cached_scope: ?u32 = null;
    var sm: Rr.SourceMap = undefined;
    defer if (cached_scope != null) sm.deinit(gpa);
    for (diags) |d| {
        if (cached_scope == null or cached_scope.? != d.scope) {
            if (cached_scope != null) sm.deinit(gpa);
            const m = if (d.scope == toyc.DiagnosticSink.NO_SCOPE) g.entry() else &g.modules[d.scope];
            // Clear `cached_scope` before the `try init`: `sm` was just deinit'd, so a
            // failing init must not let the `defer` fire on the freed `sm` (double-free).
            // Reassign after init.
            cached_scope = null;
            sm = try Rr.SourceMap.init(gpa, m.path, m.source); // Module.source []u8 coerces to []const u8
            cached_scope = d.scope;
        }
        try renderSinkDiag(out, level, &sm, d);
    }
}

/// Render a graph-discovery structural error against the owning module's source (or
/// the entry path when no module loaded). With a location, snippet + `= note: detail`;
/// otherwise the plain `path: error: msg (detail)` fallback.
pub fn renderGraphError(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, g: *const Graph.Graph, e: Graph.Error) !void {
    const name = if (e.module) |m| g.modules[m].path else if (g.modules.len > 0) g.entry().path else "<entry>";
    const has_src = e.module != null or g.modules.len > 0;
    if (e.byte_offset != null and has_src) {
        const src: []const u8 = if (e.module) |m| g.modules[m].source else g.entry().source;
        try renderLocated(gpa, out, level, name, src, e.byte_offset, e.message, e.detail);
    } else {
        try renderPlainError(out, level, name, e.message, e.detail);
    }
}

/// Render a code-emission `EmitError` from a graph build against its owning module
/// (or the entry module when `module` is null). EmitError carries no detail.
pub fn renderGraphEmit(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, g: *const Graph.Graph, e: Codegen.EmitError) !void {
    const m = if (e.module) |mi| &g.modules[mi] else g.entry();
    if (e.byte_offset) |off| {
        try renderLocated(gpa, out, level, m.path, m.source, off, e.message, "");
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
            .bad_value => try out.print("invalid value for {s}: '{s}' (expected {s})", .{ e.arg, e.got, e.expected }),
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
