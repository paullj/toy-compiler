//! M12 — the single-line pretty-diagnostic renderer: the payoff that COMPOSES
//! the four render leaves (SourceMap, Diagnostic + Theme, Style, width) into
//! rustc/ariadne-class output. Given a rich `Diagnostic`, one `SourceMap`, a
//! writer, and a plain `RenderOpts`, it prints a header, a source-location line,
//! a numbered-gutter snippet with carets/underlines aligned under the exact
//! terminal cells, and footer notes:
//!
//!     error[E0308]: mismatched types
//!      --> main.toy:3:13
//!       |
//!     3 |     let x = "hi" + 1
//!       |             ^^^^ expected integer, found string
//!       |
//!       = help: try converting the string to a number
//!
//! WHY a pure LAYOUT->EMIT seam. The load-bearing invariant of this whole
//! framework is ZERO-ESCAPE at `ColorLevel.none`: colour must change ONLY
//! styling, NEVER layout, so the plain bytes equal the coloured bytes with every
//! SGR run removed. We make that STRUCTURAL rather than test-only by splitting
//! the code in two disjoint categories:
//!   - LAYOUT (pure): `digits`, `startDisplayCol`, `runCellsClamped`, the sort —
//!     helpers that NEVER take/read `opts.color`. They decide spacing, glyph
//!     counts, and text identically at every colour level.
//!   - EMIT (writer + colour): every plain layout byte (spaces, `|`, `-->`,
//!     digits, `:`, source text, `=`, plain messages) goes through raw
//!     `writeAll`/`writeByte`/`splatByteAll`/`print`, which CANNOT emit 0x1b.
//!     The ONLY escape-producing calls are `Style.styled` and `styledGlyphRun`
//!     (a `sgrInto` + loop + `closeInto` wrapper); both write ZERO bytes at
//!     `.none`. The gutter (pipe/number/prefix) is INTENTIONALLY UNSTYLED, so
//!     the whole gutter is structurally escape-free.
//! Therefore stripping every `\x1b[...m` run from a coloured render yields the
//! `.none` render byte-for-byte, and line count / glyph runs are colour-invariant
//! by construction.
//!
//! DETERMINISM. Labels are ordered by a STABLE TOTAL order — (start display
//! column asc, primary-before-secondary, original index asc) — via an in-place
//! insertion sort over a small fixed-cap index array (no allocation, no sorted
//! copy). The same `Diagnostic` therefore renders byte-identical every run.
//!
//! SCOPE — STRICTLY SINGLE-LINE (multi-line rails are M13, the NEXT milestone):
//!   - M12 draws ONLY the primary's start line. Any label whose start line is
//!     not the primary's line is DROPPED (M13 rails will render off-line
//!     labels).
//!   - Every label is rendered on the START line of its span. If a label's span
//!     crosses a '\n', its underline is CLAMPED to the end of the start line (no
//!     cross-line rails, no runaway). M13 replaces this clamp with proper rails.
//!   - ONE marker row PER label, in sorted order, each drawing only its own
//!     caret/underline + message. NO shared/merged underline rows and NO
//!     cross-label vertical connector bars (those are M13).
//!   - `Label.source` and `Diagnostic.scope` are IGNORED uniformly: a single
//!     `SourceMap` is passed in and every span indexes it. Multi-source routing
//!     is a future (M13+) concern. Message wrapping is out of scope (messages
//!     are single-line, unbounded).
//!
//! LAYERING. Imports ONLY `std` plus the four render leaves it composes. It does
//! NOT import `Terminal` (the renderer⊥Terminal rule — colour arrives as a
//! `ColorLevel` in `opts`, never by sniffing a tty), `Progress`, `ansi`
//! directly (reached only transitively through `Style`), or anything under
//! `cli/*`.

const std = @import("std");
const SourceMap = @import("SourceMap.zig");
const Diagnostic = @import("Diagnostic.zig");
const Theme = @import("Theme.zig");
const Style = @import("../Style.zig");
const width = @import("../width.zig");

/// Deterministic upper bound on the labels a single diagnostic renders. The
/// compiler emits very few, so this is generous; beyond it the tail is silently
/// truncated (primary kept at index 0), which keeps byte-stability at the cap.
const MAX_LABELS: usize = 32;

/// Plain, tty-independent knobs. `color` is threaded to every `Style.styled` /
/// `styledGlyphRun` call (never sniffed); `unicode` picks the glyph set via
/// `Theme.forUnicode`; `tab_width` feeds `SourceMap.displayCol` so tab-indented
/// carets land under the right cell.
pub const RenderOpts = struct {
    color: Style.ColorLevel = .none,
    unicode: bool = false,
    tab_width: usize = SourceMap.default_tab_width,
};

/// A label plus its original position in the diagnostic (index 0 == primary,
/// 1.. == secondary[i-1]). `order` is the third, always-distinct sort key that
/// makes the total order stable.
const LabelRef = struct {
    lbl: Diagnostic.Label,
    order: u8,
};

/// Render `d` against the single `SourceMap` `sm`. Derives the theme from
/// `opts.unicode` internally and threads `opts.color` to every styled span.
/// Allocation-free: all working state is stack-only. See the module doc for the
/// output shape, the LAYOUT/EMIT seam, and the single-line scope.
pub fn render(d: Diagnostic.Diagnostic, sm: *const SourceMap, w: *std.Io.Writer, opts: RenderOpts) std.Io.Writer.Error!void {
    const theme = Theme.forUnicode(opts.unicode);

    // ---- LAYOUT (pure; never reads opts.color) -----------------------------
    // The one source line M12 draws is the primary's start line; `gw` (gutter
    // width) is its decimal digit count, computed once and reused for every row.
    const primary_line = sm.lineCol(d.primary.span.start).line;
    const gw = digits(primary_line);

    // Collect labels into a fixed-cap index array: primary first (order 0), then
    // each secondary. Deterministic truncation past MAX_LABELS keeps the primary.
    var refs: [MAX_LABELS]LabelRef = undefined;
    var n: usize = 0;
    refs[n] = .{ .lbl = d.primary, .order = 0 };
    n += 1;
    for (d.secondary, 0..) |s, i| {
        if (n >= MAX_LABELS) {
            std.debug.assert(false); // more labels than the cap (debug only; release truncates)
            break;
        }
        refs[n] = .{ .lbl = s, .order = @intCast(i + 1) };
        n += 1;
    }

    // Drop labels whose start line != primary_line: M12 draws only the primary's
    // line; M13 rails will render off-line labels. Compaction preserves the
    // remaining refs' relative order so the subsequent sort stays stable.
    var kept: usize = 0;
    var r: usize = 0;
    while (r < n) : (r += 1) {
        if (sm.lineCol(refs[r].lbl.span.start).line == primary_line) {
            refs[kept] = refs[r];
            kept += 1;
        }
    }

    // Sort the kept refs in place by the stable total order (start display col,
    // primary-first, original index). Insertion sort: O(n^2), n <= 32, no alloc.
    var i: usize = 1;
    while (i < kept) : (i += 1) {
        const key = refs[i];
        var j: usize = i;
        while (j > 0 and lessLabel(sm, opts.tab_width, key, refs[j - 1])) : (j -= 1) {
            refs[j] = refs[j - 1];
        }
        refs[j] = key;
    }

    // ---- EMIT --------------------------------------------------------------
    // 1. HEADER: styled severity word, then plain "[code]" (omitted when null),
    //    then plain ": message". Nothing between word and '['; nothing between
    //    ']' and ':'; exactly one space after ':'.
    try theme.style(d.severity).styled(w, opts.color, theme.word(d.severity));
    if (d.code) |code| {
        try w.writeByte('[');
        try w.writeAll(code);
        try w.writeByte(']');
    }
    try w.writeAll(": ");
    try w.writeAll(d.message);
    try w.writeByte('\n');

    // 2. LOCATION: " --> name:line:col" — entirely plain, single leading space.
    //    The column here is the BYTE column (lineCol().col), matching the
    //    compiler's existing file:line:col convention; every OTHER column in this
    //    renderer is a DISPLAY column.
    const loc = sm.lineCol(d.primary.span.start);
    try w.writeAll(" --> ");
    try w.writeAll(sm.name);
    try w.print(":{d}:{d}\n", .{ loc.line, loc.col });

    // 3. SNIPPET: a top separator, the source row (source text NEVER styled),
    //    then one marker row per kept label in sorted order.
    try emptyPrefix(w, gw);
    try w.writeByte('\n');

    try numberedPrefix(w, gw, primary_line);
    try w.writeByte(' ');
    try w.writeAll(sm.lineText(d.primary.span.start));
    try w.writeByte('\n');

    var k: usize = 0;
    while (k < kept) : (k += 1) {
        try emitMarkerRow(w, theme, opts, gw, d.severity, refs[k].lbl, sm, primary_line);
    }

    // 4. FOOTER (only when notes exist): the closing "  |" separator is the FIRST
    //    footer row (a note-less diagnostic ends at its last marker row — no
    //    dangling separator). Then one "= <word>: <message>" line per note.
    if (d.notes.len != 0) {
        try emptyPrefix(w, gw);
        try w.writeByte('\n');
        for (d.notes) |note| {
            // The '=' sits in the PIPE column: gw spaces + one space + '='
            // aligns it under the '|' of the gutter rows above. This footer
            // gutter has NO pipe, so it is not `emptyPrefix`.
            try w.splatByteAll(' ', gw);
            try w.writeByte(' ');
            try w.writeAll(theme.glyphs.note_bullet);
            try w.writeByte(' ');
            const nsev = noteSeverity(note.kind);
            try theme.style(nsev).styled(w, opts.color, theme.word(nsev));
            try w.writeAll(": ");
            try w.writeAll(note.message);
            try w.writeByte('\n');
        }
    }
}

/// Decimal digit count of `n`, min 1 (line numbers are 1-based, so `n >= 1`).
/// A manual loop, matching the house style of hand-rolled math in the leaves.
fn digits(n: usize) usize {
    var v = n;
    var d: usize = 1;
    while (v >= 10) : (v /= 10) d += 1;
    return d;
}

/// A numbered gutter prefix: right-align `line` in a `gw`-wide field, then
/// `<line> |`. The prefix has NO trailing space — content rows prepend their own
/// single space, which keeps goldens free of trailing-whitespace ambiguity.
fn numberedPrefix(w: *std.Io.Writer, gw: usize, line: usize) std.Io.Writer.Error!void {
    try w.splatByteAll(' ', gw - digits(line));
    try w.print("{d} |", .{line});
}

/// An empty gutter prefix (separator / marker rows): `gw` spaces then ` |`. Like
/// `numberedPrefix`, no trailing space.
fn emptyPrefix(w: *std.Io.Writer, gw: usize) std.Io.Writer.Error!void {
    try w.splatByteAll(' ', gw);
    try w.writeAll(" |");
}

/// Emit a styled run of `n` copies of `glyph`. THE single escape-emitting seam
/// for glyph runs: one `sgrInto`, `n` raw `writeAll`s, one `closeInto`. At
/// `.none` `sgrInto`/`closeInto` write zero bytes, so the run is the bare glyphs.
/// The glyph is repeated by COUNT (not by byte length) so a multi-byte unicode
/// glyph ('─' = 3 bytes / 1 cell) yields exactly `n` cells, not `3n`.
fn styledGlyphRun(s: Style.Style, w: *std.Io.Writer, level: Style.ColorLevel, glyph: []const u8, n: usize) std.Io.Writer.Error!void {
    try s.sgrInto(w, level);
    var i: usize = 0;
    while (i < n) : (i += 1) try w.writeAll(glyph);
    try s.closeInto(w, level);
}

/// Bridge a `NoteKind` to the `Severity` that `Theme` keys presentation off (the
/// M11-deferred mapping, done inline here rather than editing Theme):
/// `.note` -> `.note`, `.help` -> `.help`. Gives the footer note-line its word
/// and style.
fn noteSeverity(k: Diagnostic.NoteKind) Diagnostic.Severity {
    return switch (k) {
        .note => .note,
        .help => .help,
    };
}

/// The stable total order over labels for the marker rows (PURE — no colour):
///   1. start DISPLAY column ascending (leftmost marker first);
///   2. tie -> primary before secondary;
///   3. tie -> original index ascending (`order`).
/// Total + stable, so the same diagnostic renders byte-identical every run.
fn lessLabel(sm: *const SourceMap, tab_width: usize, a: LabelRef, b: LabelRef) bool {
    const ca = startDisplayCol(sm, a.lbl, tab_width);
    const cb = startDisplayCol(sm, b.lbl, tab_width);
    if (ca != cb) return ca < cb;
    // Primary sorts before secondary. LabelKind is an enum with primary=0, so a
    // straight tag comparison works, but be explicit for clarity.
    const pa = a.lbl.kind == .primary;
    const pb = b.lbl.kind == .primary;
    if (pa != pb) return pa; // pa true (primary) => a comes first
    return a.order < b.order;
}

/// 1-based DISPLAY column of a label's span start (PURE). The caret's leading
/// padding is `this - 1`; using the DISPLAY column (not the byte column) is what
/// lands the marker under the exact terminal cell on tab/CJK/emoji lines.
fn startDisplayCol(sm: *const SourceMap, lbl: Diagnostic.Label, tab_width: usize) usize {
    return sm.displayCol(lbl.span.start, tab_width);
}

/// Display-cell width of a label's underline run, with the single-line CLAMP
/// (PURE). A span crossing a '\n' is clamped to the end of its start line — no
/// cross-line rails, no runaway. M13 replaces this clamp with proper rails.
/// A zero-width span yields 1 (a single caret cell); otherwise the clamped span's
/// DISPLAY width, floored at 1.
///
/// The width is a DIFFERENCE OF DISPLAY COLUMNS, `displayCol(eff_end) -
/// displayCol(start)`, NOT `width.displayWidth(spanText)`. That distinction is
/// load-bearing for a span containing an INTERIOR tab: width.zig reports TAB as
/// 0 cells (it is a C0 control), so `displayWidth("a\tb")` = 2, under-running the
/// on-screen 5 cells (a=col1, tab expands to the col-4 stop, b=col5). displayCol
/// walks the line prefix with elastic tab-stop expansion, so its difference is
/// the true on-screen cell span and the underline lands under the highlighted
/// text. (Leading-indent tabs BEFORE the span start are handled by the pad, which
/// is already displayCol-based; this fixes tabs INSIDE the span.)
fn runCellsClamped(sm: *const SourceMap, lbl: Diagnostic.Label, primary_line: usize, tab_width: usize) usize {
    if (lbl.span.isZeroWidth()) return 1;
    // The visible text of the start line, and its exclusive end offset. lineText
    // is CRLF-stripped, so lineStart + lineText.len is the end of the VISIBLE
    // text (never a trailing CR/LF) — do NOT walk to the next '\n'.
    const line_start_off = sm.lineStart(primary_line);
    const line_text = sm.lineText(lbl.span.start);
    const line_end_off = line_start_off + @as(u32, @intCast(line_text.len));
    const eff_end = @min(lbl.span.end, line_end_off); // clamp cross-line span
    // Tab-aware span width via the difference of 1-based display columns; the
    // per-column offset cancels, so this is the exact on-screen cell count.
    const cells = sm.displayCol(eff_end, tab_width) - sm.displayCol(lbl.span.start, tab_width);
    return @max(cells, 1);
}

/// Emit one marker row for `lbl`: `emptyPrefix` + ' ' + pad-to-column + styled
/// glyph run + (optional) styled message. The glyph run and message share the
/// label's style — a PRIMARY label uses the diagnostic's `severity` hue (so the
/// caret matches the header word), a SECONDARY label the dim `secondaryStyle`.
/// This matches rustc's look and keeps the styled-span set minimal (still
/// zero-escape at `.none`). An empty message ends the row right after the glyph
/// run — no trailing space.
fn emitMarkerRow(
    w: *std.Io.Writer,
    theme: Theme,
    opts: RenderOpts,
    gw: usize,
    severity: Diagnostic.Severity,
    lbl: Diagnostic.Label,
    sm: *const SourceMap,
    primary_line: usize,
) std.Io.Writer.Error!void {
    // Gutter (unstyled), then the content row's own leading space.
    try emptyPrefix(w, gw);
    try w.writeByte(' ');

    // Pad to the start DISPLAY column (col - 1 cells before the marker). Because
    // displayCol walks the line prefix in cells (tabs->stops, CJK/emoji=2,
    // combining=0), this lands the marker under the exact terminal cell with no
    // special-casing here.
    const start_dcol = startDisplayCol(sm, lbl, opts.tab_width);
    try w.splatByteAll(' ', start_dcol - 1);

    // Glyph + style. Use a single caret for a zero-width point OR a one-cell run
    // (crisp caret-vs-underline rule); otherwise an underline as wide as the
    // span's display cells. All theme glyphs are 1 display cell, so repeating by
    // count yields exactly `run_cells` cells.
    const run_cells = runCellsClamped(sm, lbl, primary_line, opts.tab_width);
    const use_caret = lbl.span.isZeroWidth() or run_cells <= 1;
    const s = if (lbl.kind == .primary) theme.style(severity) else theme.secondaryStyle();

    if (use_caret) {
        try styledGlyphRun(s, w, opts.color, theme.caret(), 1);
    } else {
        try styledGlyphRun(s, w, opts.color, theme.underline(lbl.kind), run_cells);
    }

    // Message (styled like its glyph run). Empty message => bare underline, row
    // ends right after the glyph run with no trailing space.
    if (lbl.message.len != 0) {
        try w.writeByte(' ');
        try s.styled(w, opts.color, lbl.message);
    }
    try w.writeByte('\n');
}

// ---- tests -----------------------------------------------------------------

const testing = std.testing;

// Build a SourceMap over a literal for a test. Caller defers deinit.
fn mapOver(name: []const u8, bytes: []const u8) SourceMap {
    return SourceMap.init(testing.allocator, name, bytes) catch unreachable;
}

// Render into a fixed buffer and return the written slice (aliased into `buf`).
fn renderInto(buf: []u8, d: Diagnostic.Diagnostic, sm: *const SourceMap, opts: RenderOpts) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    render(d, sm, &w, opts) catch unreachable;
    return w.buffered();
}

// Strip every \x1b[...m SGR run from `in` into `out`, returning the plain slice.
// Used by the color-invariance test to prove coloured == plain minus SGR.
fn stripSgr(out: []u8, in: []const u8) []const u8 {
    var oi: usize = 0;
    var i: usize = 0;
    while (i < in.len) {
        if (in[i] == 0x1b and i + 1 < in.len and in[i + 1] == '[') {
            i += 2;
            while (i < in.len and in[i] != 'm') i += 1;
            if (i < in.len) i += 1; // consume the 'm'
            continue;
        }
        out[oi] = in[i];
        oi += 1;
        i += 1;
    }
    return out[0..oi];
}

// The canonical E0308 diagnostic used by several goldens. Line 3 is
// `    let x = "hi" + 1` (4 leading spaces); the `"hi"` span is bytes 12..16
// within line 3, all ASCII, so byte col == display col == 13.
const e0308_src = "let a = 1\nlet b = 2\n    let x = \"hi\" + 1\n";
// Byte offsets of the '"' and one past the closing '"' within the whole source.
// Line 3 starts after two "...\n" lines: 10 + 10 = 20. The 4 spaces + `let x = `
// = 4 + 8 = 12 bytes, so '"' is at 20 + 12 = 32, closing-quote-end at 36.
const e0308_quote_start: u32 = 32;
const e0308_quote_end: u32 = 36;

test "T1 zero-width primary caret: exactly one '^' with full frozen golden" {
    var sm = mapOver("z.toy", "let x = 1\n");
    defer sm.deinit(sm.gpa);
    // Zero-width primary at byte 4 ('x'), display col 5.
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "undefined name",
        .primary = .{ .kind = .primary, .span = .{ .start = 4, .end = 4 }, .message = "not found" },
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        \\error: undefined name
        \\ --> z.toy:1:5
        \\  |
        \\1 | let x = 1
        \\  |     ^ not found
        \\
    ,
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T2 wide underline primary: E0308 minus notes, '^^^^' == displayWidth 4" {
    var sm = mapOver("main.toy", e0308_src);
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "mismatched types",
        .primary = .{ .kind = .primary, .span = .{ .start = e0308_quote_start, .end = e0308_quote_end }, .message = "expected integer, found string" },
        .code = "E0308",
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        \\error[E0308]: mismatched types
        \\ --> main.toy:3:13
        \\  |
        \\3 |     let x = "hi" + 1
        \\  |             ^^^^ expected integer, found string
        \\
    ,
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T2b full E0308 golden with help note (the North Star, byte-for-byte)" {
    var sm = mapOver("main.toy", e0308_src);
    defer sm.deinit(sm.gpa);
    const notes = [_]Diagnostic.Note{
        .{ .kind = .help, .message = "try converting the string to a number" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "mismatched types",
        .primary = .{ .kind = .primary, .span = .{ .start = e0308_quote_start, .end = e0308_quote_end }, .message = "expected integer, found string" },
        .notes = &notes,
        .code = "E0308",
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        \\error[E0308]: mismatched types
        \\ --> main.toy:3:13
        \\  |
        \\3 |     let x = "hi" + 1
        \\  |             ^^^^ expected integer, found string
        \\  |
        \\  = help: try converting the string to a number
        \\
    ,
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T3 code omitted when null: header has no brackets" {
    var sm = mapOver("s.toy", "abcde\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "boom",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 2 }, .message = "here" },
    };
    var buf: [256]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{});
    // No '[' in the header line (nor anywhere: source has none).
    try testing.expect(std.mem.indexOfScalar(u8, out, '[') == null);
    try testing.expectEqualStrings(
        \\error: boom
        \\ --> s.toy:1:1
        \\  |
        \\1 | abcde
        \\  | ^^ here
        \\
    , out);
}

test "T4 multiple labels one line: two marker rows in (col, primary-first) order" {
    // "    let x = 12": primary on `x` (byte 8, display col 9, 1 cell -> caret);
    // secondary on `12` (bytes 12..14, display col 13, 2 cells -> '--' underline).
    var sm = mapOver("m.toy", "    let x = 12\n");
    defer sm.deinit(sm.gpa);
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 12, .end = 14 }, .message = "operand here" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "type error",
        .primary = .{ .kind = .primary, .span = .{ .start = 8, .end = 9 }, .message = "this binding" },
        .secondary = &secondary,
    };
    var buf: [512]u8 = undefined;
    // primary at display col 9 (pad 8) -> one-cell caret; secondary at display
    // col 13 (pad 12) -> two-cell '--' underline (secondary glyph). Row order is
    // (col asc): primary (col 9) first, then secondary (col 13).
    try testing.expectEqualStrings(
        \\error: type error
        \\ --> m.toy:1:9
        \\  |
        \\1 |     let x = 12
        \\  |         ^ this binding
        \\  |             -- operand here
        \\
    ,
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T5 primary-before-secondary tie-break at the same start display col" {
    // Primary and secondary share the SAME start (col 1). Primary row must be first.
    var sm = mapOver("t.toy", "xy\n");
    defer sm.deinit(sm.gpa);
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 0, .end = 2 }, .message = "context" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "clash",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 2 }, .message = "primary" },
        .secondary = &secondary,
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        \\error: clash
        \\ --> t.toy:1:1
        \\  |
        \\1 | xy
        \\  | ^^ primary
        \\  | -- context
        \\
    ,
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T6 notes (note + help): trailing separator then '= note:' / '= help:'" {
    var sm = mapOver("n.toy", "value\n");
    defer sm.deinit(sm.gpa);
    const notes = [_]Diagnostic.Note{
        .{ .kind = .note, .message = "first defined here" },
        .{ .kind = .help, .message = "rename it" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "duplicate",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 5 }, .message = "" },
        .notes = &notes,
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        \\error: duplicate
        \\ --> n.toy:1:1
        \\  |
        \\1 | value
        \\  | ^^^^^
        \\  |
        \\  = note: first defined here
        \\  = help: rename it
        \\
    ,
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T7 CJK line: caret aligns from displayCol, run cells from displayWidth" {
    // "世 x" : 世 (2 cells, 3 bytes), space, 'x'. Underline the 'x' word.
    // bytes: 世=0..3, ' '=3, 'x'=4. Primary span [4,5) -> display col 4 (2+1+1),
    // width 1 -> caret. Wait: prefix "世 " = 2 + 1 = 3 cells, so 'x' is display col 4.
    var sm = mapOver("c.toy", "\u{4E16} x\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "cjk",
        .primary = .{ .kind = .primary, .span = .{ .start = 4, .end = 5 }, .message = "here" },
    };
    var buf: [512]u8 = undefined;
    // pad = display col 4 - 1 = 3 spaces before the caret.
    try testing.expectEqualStrings(
        "error: cjk\n" ++
            " --> c.toy:1:5\n" ++ // byte col: 世=3 bytes, ' '=1 -> 'x' at byte 4, col 5
            "  |\n" ++
            "1 | \u{4E16} x\n" ++
            "  |    ^ here\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T7b CJK wide underline: two CJK chars -> 4-cell underline" {
    // "世界x": 世界 = 4 cells / 6 bytes. Underline span [0,6) -> 4 cells.
    var sm = mapOver("c2.toy", "\u{4E16}\u{754C}x\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "wide",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 6 }, .message = "pair" },
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        "error: wide\n" ++
            " --> c2.toy:1:1\n" ++
            "  |\n" ++
            "1 | \u{4E16}\u{754C}x\n" ++
            "  | ^^^^ pair\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T8 emoji line: 🎉 (2 cells) in prefix shifts pad by 2" {
    // "🎉x=1": 🎉 = 2 cells / 4 bytes. Underline 'x' at byte 4, display col 3.
    var sm = mapOver("e.toy", "\u{1F389}x=1\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "emoji",
        .primary = .{ .kind = .primary, .span = .{ .start = 4, .end = 5 }, .message = "x" },
    };
    var buf: [512]u8 = undefined;
    // display col 3 -> pad 2 before the caret. byte col: 🎉=4 bytes -> 'x' col 5.
    try testing.expectEqualStrings(
        "error: emoji\n" ++
            " --> e.toy:1:5\n" ++
            "  |\n" ++
            "1 | \u{1F389}x=1\n" ++
            "  |   ^ x\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T9 tab-indented source: pad includes tab expansion (tab_width=4)" {
    // "\tx = 1": leading tab expands to col 4 (0-based), so 'x' is display col 5.
    var sm = mapOver("tab.toy", "\tx = 1\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "tabbed",
        .primary = .{ .kind = .primary, .span = .{ .start = 1, .end = 2 }, .message = "x" },
    };
    var buf: [512]u8 = undefined;
    // display col 5 -> pad 4 spaces (the tab expands in the marker row too, since
    // pad is display-cell count). byte col: tab=1 byte -> 'x' at byte 1, col 2.
    try testing.expectEqualStrings(
        "error: tabbed\n" ++
            " --> tab.toy:1:2\n" ++
            "  |\n" ++
            "1 | \tx = 1\n" ++
            "  |     ^ x\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T9b interior tab in span: underline width is tab-aware (displayCol diff, not displayWidth)" {
    // "a\tb": 'a' (col 1), TAB expands to the col-4 stop, 'b' (col 5). A primary
    // span [0,3) covers a+TAB+b and occupies 5 on-screen cells. width.displayWidth
    // reports TAB as 0, so "a\tb" measures only 2 — that undersized the underline
    // before the fix. The tab-aware run width is displayCol(3)-displayCol(0) = 5,
    // so five carets sit under a..b. NOTE: run_cells > 1 => underline, and all the
    // theme glyphs are 1 cell so the caret glyph repeats to the correct 5 cells.
    var sm = mapOver("tin.toy", "a\tb\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "interior tab",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 3 }, .message = "spans a tab" },
    };
    var buf: [512]u8 = undefined;
    // pad = displayCol(0)-1 = 0; run = 5 cells ("^^^^^") under a, the tab's 3 fill
    // cells, and b. byte col: 'a' at byte 0 -> col 1.
    try testing.expectEqualStrings(
        "error: interior tab\n" ++
            " --> tin.toy:1:1\n" ++
            "  |\n" ++
            "1 | a\tb\n" ++
            "  | ^^^^^ spans a tab\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T10 combining mark before span: base+mark = 1 cell, pad accounts mark as 0" {
    // "e\u{0301}x": 'e' (1 cell) + combining acute (0 cells) + 'x'. 'x' at byte 3.
    // display col of 'x' = 1(e) + 0(mark) + 1 = col 2 -> pad 1.
    var sm = mapOver("cm.toy", "e\u{0301}x\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "combining",
        .primary = .{ .kind = .primary, .span = .{ .start = 3, .end = 4 }, .message = "x" },
    };
    var buf: [512]u8 = undefined;
    // display col of 'x' = e(1) + mark(0) + 1 = 2 -> pad 1 (the mark adds NO
    // cell). byte col: e=1 byte, mark=2 bytes -> 'x' at byte 3, byte col 4.
    try testing.expectEqualStrings(
        "error: combining\n" ++
            " --> cm.toy:1:4\n" ++
            "  |\n" ++
            "1 | e\u{0301}x\n" ++
            "  |  ^ x\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T11 PLAIN zero-escape: a rich diagnostic at .none has no 0x1b byte" {
    var sm = mapOver("main.toy", e0308_src);
    defer sm.deinit(sm.gpa);
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = e0308_quote_start + 5, .end = e0308_quote_start + 6 }, .message = "operand" },
    };
    const notes = [_]Diagnostic.Note{
        .{ .kind = .note, .message = "n" },
        .{ .kind = .help, .message = "h" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "mismatched types",
        .primary = .{ .kind = .primary, .span = .{ .start = e0308_quote_start, .end = e0308_quote_end }, .message = "expected integer, found string" },
        .secondary = &secondary,
        .notes = &notes,
        .code = "E0308",
    };
    var buf: [1024]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{ .color = .none });
    try testing.expect(std.mem.indexOfScalar(u8, out, 0x1b) == null);
}

test "T12 forced color: .ansi16 has escapes, layout matches plain, strip == plain" {
    var sm = mapOver("main.toy", e0308_src);
    defer sm.deinit(sm.gpa);
    const notes = [_]Diagnostic.Note{
        .{ .kind = .help, .message = "try converting the string to a number" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "mismatched types",
        .primary = .{ .kind = .primary, .span = .{ .start = e0308_quote_start, .end = e0308_quote_end }, .message = "expected integer, found string" },
        .notes = &notes,
        .code = "E0308",
    };
    var plain_buf: [1024]u8 = undefined;
    var color_buf: [1024]u8 = undefined;
    const plain = renderInto(&plain_buf, d, &sm, .{ .color = .none });
    const colored = renderInto(&color_buf, d, &sm, .{ .color = .ansi16 });

    // Colored has escapes; plain does not.
    try testing.expect(std.mem.indexOfScalar(u8, colored, 0x1b) != null);
    try testing.expect(std.mem.indexOfScalar(u8, plain, 0x1b) == null);
    // Identical line count (layout is colour-invariant).
    try testing.expectEqual(
        std.mem.count(u8, plain, "\n"),
        std.mem.count(u8, colored, "\n"),
    );
    // Stripping every SGR run from the colored output yields the plain bytes.
    var strip_buf: [1024]u8 = undefined;
    try testing.expectEqualStrings(plain, stripSgr(&strip_buf, colored));
}

test "T13 byte-identical across runs: same diagnostic renders the same bytes" {
    var sm = mapOver("m.toy", "    let x = 1\n");
    defer sm.deinit(sm.gpa);
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 12, .end = 13 }, .message = "b" },
        .{ .kind = .secondary, .span = .{ .start = 4, .end = 7 }, .message = "a" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .warning,
        .message = "m",
        .primary = .{ .kind = .primary, .span = .{ .start = 8, .end = 9 }, .message = "p" },
        .secondary = &secondary,
    };
    var b1: [512]u8 = undefined;
    var b2: [512]u8 = undefined;
    const a = renderInto(&b1, d, &sm, .{});
    // Copy out before reusing buffers (renderInto aliases the arg buffer).
    var hold: [512]u8 = undefined;
    @memcpy(hold[0..a.len], a);
    const first = hold[0..a.len];
    const b = renderInto(&b2, d, &sm, .{});
    try testing.expectEqualStrings(first, b);
}

test "T14 cross-line span clamp: underline stops at end of the start line" {
    // "abc\ndef": primary span [0,7) starts on line 1, ends on line 2. Clamp the
    // underline to the end of line 1 ("abc" = 3 cells), no rail, no runaway.
    var sm = mapOver("x.toy", "abc\ndef\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "spanned",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 7 }, .message = "crosses" },
    };
    var buf: [512]u8 = undefined;
    // Only line 1 is drawn; underline is 3 cells ("abc").
    try testing.expectEqualStrings(
        \\error: spanned
        \\ --> x.toy:1:1
        \\  |
        \\1 | abc
        \\  | ^^^ crosses
        \\
    ,
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T15 one-cell non-zero span renders a caret, not an underline" {
    var sm = mapOver("o.toy", "ab\n");
    defer sm.deinit(sm.gpa);
    // Span [0,1): one byte, one cell, NON-zero-width -> use_caret via run_cells<=1.
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "one",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 1 }, .message = "c" },
    };
    var buf: [256]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{});
    // The marker row is a single '^' (caret), not "^" repeated as an underline.
    try testing.expect(std.mem.indexOf(u8, out, "| ^ c\n") != null);
    // And there is no "^^" run anywhere.
    try testing.expect(std.mem.indexOf(u8, out, "^^") == null);
}

test "T16 unicode theme glyphs: underline '─', secondary '┄', note bullet '═'" {
    var sm = mapOver("u.toy", "abcd\n");
    defer sm.deinit(sm.gpa);
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 2, .end = 4 }, .message = "s" },
    };
    const notes = [_]Diagnostic.Note{
        .{ .kind = .note, .message = "n" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "u",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 2 }, .message = "p" },
        .secondary = &secondary,
        .notes = &notes,
    };
    var buf: [512]u8 = undefined;
    // Primary underline is '─' x2, secondary '┄' x2, note bullet '═'.
    try testing.expectEqualStrings(
        "error: u\n" ++
            " --> u.toy:1:1\n" ++
            "  |\n" ++
            "1 | abcd\n" ++
            "  | \u{2500}\u{2500} p\n" ++
            "  |   \u{2504}\u{2504} s\n" ++
            "  |\n" ++
            "  \u{2550} note: n\n",
        renderInto(&buf, d, &sm, .{ .unicode = true }),
    );

    // Layout invariant under glyph set: the unicode primary run is the SAME cell
    // count as the plain run ('^^' -> '──', both 2 cells).
    var plain_buf: [512]u8 = undefined;
    const plain = renderInto(&plain_buf, d, &sm, .{ .unicode = false });
    // Both have the same number of newlines (identical row structure).
    try testing.expectEqual(
        std.mem.count(u8, plain, "\n"),
        std.mem.count(u8, renderInto(&buf, d, &sm, .{ .unicode = true }), "\n"),
    );
}

test "T17 gutter width scales for a 2-digit line number" {
    // 11 lines so the primary lands on line 12 (>= 10) -> gw = 2.
    const src = "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nlet x = 1\n";
    var sm = mapOver("g.toy", src);
    defer sm.deinit(sm.gpa);
    // Byte offset of 'x' on line 12: sum of the first 11 lines' lengths + 4.
    // l1..l9 = "lN\n" = 3 bytes each (9*3 = 27); l10,l11 = "lNN\n" = 4 each (8);
    // line 12 starts at 35; "let x = " is 8 bytes -> 'x' at 39.
    const x_off: u32 = 35 + 4; // "let " = 4 bytes -> 'x' at 39
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "scale",
        .primary = .{ .kind = .primary, .span = .{ .start = x_off, .end = x_off + 1 }, .message = "x" },
    };
    var buf: [512]u8 = undefined;
    // Numbered prefix "12 |" (gw=2), empty prefix "   |" (2 spaces + " |").
    try testing.expectEqualStrings(
        "error: scale\n" ++
            " --> g.toy:12:5\n" ++
            "   |\n" ++
            "12 | let x = 1\n" ++
            "   |     ^ x\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T18 empty message = bare marker: row ends right after the glyph run" {
    var sm = mapOver("b.toy", "abcd\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "bare",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 3 }, .message = "" },
    };
    var buf: [256]u8 = undefined;
    // The marker row ends exactly after "^^^" with a newline, no trailing space.
    try testing.expectEqualStrings(
        \\error: bare
        \\ --> b.toy:1:1
        \\  |
        \\1 | abcd
        \\  | ^^^
        \\
    ,
        renderInto(&buf, d, &sm, .{}),
    );
}

test "off-primary-line labels are dropped (M12 draws only the primary line)" {
    // Primary on line 1; a secondary on line 2 must NOT appear.
    var sm = mapOver("d.toy", "abc\ndef\n");
    defer sm.deinit(sm.gpa);
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 4, .end = 7 }, .message = "off-line" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "drop",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 3 }, .message = "here" },
        .secondary = &secondary,
    };
    var buf: [512]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{});
    try testing.expect(std.mem.indexOf(u8, out, "off-line") == null);
    // Exactly one marker row (the primary): count lines starting with "  | ^".
    try testing.expectEqualStrings(
        \\error: drop
        \\ --> d.toy:1:1
        \\  |
        \\1 | abc
        \\  | ^^^ here
        \\
    , out);
}

test "warning severity word is used in the header" {
    var sm = mapOver("w.toy", "abc\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .warning,
        .message = "careful",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 1 }, .message = "" },
    };
    var buf: [256]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{});
    try testing.expect(std.mem.startsWith(u8, out, "warning: careful\n"));
}

test "fromSink round-trips: a zero-width sink diagnostic renders one caret" {
    var sm = mapOver("f.toy", "abcdef\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.fromSink(.{ .byte_offset = 2, .message = "boom" });
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings(
        \\error: boom
        \\ --> f.toy:1:3
        \\  |
        \\1 | abcdef
        \\  |   ^ boom
        \\
    ,
        renderInto(&buf, d, &sm, .{}),
    );
}
