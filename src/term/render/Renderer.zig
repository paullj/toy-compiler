//! Multi-line pretty-diagnostic renderer: draws the snippet, carets/underlines,
//! footer notes, and a left rail joining a multi-line span's start line to its end
//! line (rustc/ariadne style). Single source, allocation-free (stack-only).
//! - Zero-escape at `ColorLevel.none`: colour changes styling only, never layout,
//!   so stripping every SGR run from a coloured render yields the `.none` render
//!   byte-for-byte. Structural, not test-only: LAYOUT helpers never read
//!   `opts.color`; EMIT sends plain bytes through raw writes (can't emit 0x1b) and
//!   escapes only via `Style.styled`/`styledGlyphRun` (zero bytes at `.none`). The
//!   gutter '|' stays ASCII even in unicode mode; only the rail region uses │╭╰─,
//!   which is why the rail is a distinct column from the gutter pipe.
//! - Rail geometry: a connector row draws all `R` rail cells (like a source row),
//!   so the content origin is at display offset `R` on every row; a source char at
//!   display col `dcol` sits at offset `R + (dcol-1)`, and a connector's run+caret
//!   reach it with `K = dcol-1` regardless of the span's rail column. All
//!   horizontal distances are display-column differences (never byte counts), so
//!   connectors align on tab/CJK/emoji lines.
//! - Overlap crossing (simplification): with only 4 rail glyphs an outer connector
//!   crossing a still-open inner rail overwrites it with `rail_horizontal` (no `┼`).
//!   Aligned + readable, never corrupt; only staggered (non-nested) overlaps hit it.
//! - Determinism: markers sort by (start display col, primary-first, order); rails
//!   are greedy-colored over a separate index array (never permuting the marker
//!   sort). Same diagnostic renders byte-identical every run.
//! - Scope: single source (`Label.source`/`scope` ignored); no message wrapping;
//!   rail depth capped at `MAX_MULTILINE` (overflow demotes to a clamped underline).
//! - Imports only `std` + the four render leaves; colour arrives as a `ColorLevel`,
//!   never by sniffing a tty.

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

/// Rail-column cap (overlap depth). Up to `MAX_MULTILINE` overlapping spans render
/// faithfully via greedy coloring; extra spans past the cap are re-tagged
/// single-line and rendered as a clamped underline (a debug assert flags overflow).
const MAX_MULTILINE: usize = 4;

/// The elision-row glyph (file-local: no reserved Theme glyph exists). Sits in the
/// content area (after ` |` + space), so the '|' column stays straight. Both are
/// plain bytes (gate-safe); chosen by `opts.unicode` in emit.
const ELISION_PLAIN: []const u8 = "...";
const ELISION_UNICODE: []const u8 = "\u{22EE}"; // ⋮ vertical ellipsis, 1 cell

/// Tty-independent knobs. `color` is threaded to every styled call (never sniffed);
/// `unicode` picks the glyph set; `tab_width` feeds `SourceMap.displayCol`.
pub const RenderOpts = struct {
    color: Style.ColorLevel = .none,
    unicode: bool = false,
    tab_width: usize = SourceMap.default_tab_width,
};

/// A label plus its original position in the diagnostic (index 0 == primary,
/// 1.. == secondary[i-1]). `order` is the distinct third sort key for stability.
/// `multi` marks a multi-line span (set during layout); `rail` is its assigned
/// column when multi. A single-line label (or a demoted-past-cap span) takes the
/// marker/clamp path.
const LabelRef = struct {
    lbl: Diagnostic.Label,
    order: u8,
    multi: bool = false,
    rail: usize = 0,
};

/// A resolved multi-line span (built during layout): 1-based line + display-column
/// endpoints, the sort `order`, and the greedily assigned `rail` column.
const MultiSpan = struct {
    start_line: usize,
    end_line: usize,
    start_dcol: usize,
    end_dcol: usize,
    order: u8,
    rail: usize,
};

/// Render `d` against the single `SourceMap` `sm`. Derives the theme from
/// `opts.unicode`, threads `opts.color` to every styled span. Allocation-free.
/// See the module doc for the output shape, the layout/emit seam, and the rail.
pub fn render(d: Diagnostic.Diagnostic, sm: *const SourceMap, w: *std.Io.Writer, opts: RenderOpts) std.Io.Writer.Error!void {
    const theme = Theme.forUnicode(opts.unicode);

    // LAYOUT (pure; never reads opts.color).

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

    // Classify each ref as single- or multi-line, and gather the multi-line spans
    // into `multis` (capped at MAX_MULTILINE, in original order). A ref whose span
    // crosses a '\n' but overflows the cap is re-tagged single-line so it takes
    // the M12 start-line-clamped path (documented simplification; asserted).
    var multis: [MAX_MULTILINE]MultiSpan = undefined;
    var m: usize = 0;
    for (refs[0..n]) |*ref| {
        if (classifyMulti(sm, ref.lbl, opts.tab_width)) |ms| {
            if (m >= MAX_MULTILINE) {
                std.debug.assert(false); // more overlapping multi-line spans than the cap
                ref.multi = false; // overflow => demote to clamped single-line
                continue;
            }
            var this = ms;
            this.order = ref.order;
            multis[m] = this;
            ref.multi = true;
            m += 1;
        } else {
            ref.multi = false;
        }
    }

    // Assign rail columns via deterministic greedy interval coloring; R == depth.
    const R = assignRails(multis[0..m]);
    // Copy each span's assigned column back onto its ref so the marker/connector
    // emit can find the column from either the ref list or the multis list.
    for (refs[0..n]) |*ref| {
        if (!ref.multi) continue;
        for (multis[0..m]) |ms| {
            if (ms.order == ref.order) {
                ref.rail = ms.rail;
                break;
            }
        }
    }

    // Line bounds across BOTH endpoints of EVERY ref (a multi-line span's END
    // line can be the deepest), so `gw` covers the deepest rendered line and the
    // rendered-line walk spans every labelled/interior line.
    var min_line: usize = std.math.maxInt(usize);
    var max_line: usize = 0;
    for (refs[0..n]) |ref| {
        const sl = sm.lineCol(ref.lbl.span.start).line;
        const el = sm.lineCol(ref.lbl.span.end).line;
        if (sl < min_line) min_line = sl;
        if (el > max_line) max_line = el;
    }
    const gw = digits(max_line);

    // Sort the refs in place by the stable total order (start display col,
    // primary-first, original index). Insertion sort: O(n^2), n <= 32, no alloc.
    // This is INDEPENDENT of the rail-column coloring above (which used a copy),
    // so the marker order and the rail order never interfere.
    var i: usize = 1;
    while (i < n) : (i += 1) {
        const key = refs[i];
        var j: usize = i;
        while (j > 0 and lessLabel(sm, opts.tab_width, key, refs[j - 1])) : (j -= 1) {
            refs[j] = refs[j - 1];
        }
        refs[j] = key;
    }

    // EMIT.
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

    // 3. TOP SEPARATOR: the rail region is trailing-trimmed to blank (nothing is
    //    open above the first rendered line), so at any R this is M12's "  |".
    try emitSeparator(w, gw, R, multis[0..m], min_line, .top);

    // 4. LINE LOOP: walk every source line in [min_line, max_line], rendering a
    //    line iff it carries a single-line label OR lies inside an open rail;
    //    collapse a gap between non-adjacent rendered lines with one elision row.
    var prev_rendered: usize = 0;
    var line = min_line;
    while (line <= max_line) : (line += 1) {
        const inside = R > 0 and anyRailOpen(multis[0..m], line);
        const has_single = anySingleOnLine(sm, refs[0..n], line);
        if (!(inside or has_single)) continue;

        if (prev_rendered != 0 and line != prev_rendered + 1) {
            // A gap can only occur where no rail is open (every interior line of a
            // span is rendered), so the elision row never needs a rail cell.
            std.debug.assert(!anyRailOpen(multis[0..m], line - 1));
            try emitElisionRow(w, opts, gw);
        }
        prev_rendered = line;

        try emitLineBlock(w, theme, opts, gw, R, d.severity, sm, line, multis[0..m], refs[0..n]);
    }

    // 5. FOOTER (only when notes exist): the closing "  |" separator is the FIRST
    //    footer row (a note-less diagnostic ends at its last snippet row — no
    //    dangling separator). Then one "= <word>: <message>" line per note. The
    //    footer never carries a rail region (locked: the rail is snippet-only).
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

/// Decimal digit count of `n`, min 1 (line numbers are 1-based).
fn digits(n: usize) usize {
    var v = n;
    var d: usize = 1;
    while (v >= 10) : (v /= 10) d += 1;
    return d;
}

/// Classify a label's span: returns a `MultiSpan` iff start and end resolve to
/// different source lines, else `null` (single-line, incl. every zero-width span).
/// Captures the display columns so the connector emit never re-derives them;
/// `order` left 0 (the caller stamps the ref's order).
fn classifyMulti(sm: *const SourceMap, lbl: Diagnostic.Label, tab_width: usize) ?MultiSpan {
    const start_line = sm.lineCol(lbl.span.start).line;
    const end_line = sm.lineCol(lbl.span.end).line;
    if (end_line <= start_line) return null;
    return .{
        .start_line = start_line,
        .end_line = end_line,
        .start_dcol = sm.displayCol(lbl.span.start, tab_width),
        .end_dcol = sm.displayCol(lbl.span.end, tab_width),
        .order = 0,
        .rail = 0,
    };
}

/// Assign each multi-line span a rail column via deterministic greedy interval
/// coloring; returns the rail depth `R` (max column + 1, or 0 when empty). Sorts a
/// copy-index array (never permuting the caller's order beyond the `.rail` write),
/// then gives each span the lowest column whose last span ended before it starts.
/// Earlier-start / later-end spans take lower columns, so nested spans land on
/// higher columns (column 0 == outermost). Ties break by `order` (primary first).
fn assignRails(multis: []MultiSpan) usize {
    const m = multis.len;
    if (m == 0) return 0;

    // Index array sorted by the coloring order; insertion sort, no alloc.
    var idx: [MAX_MULTILINE]usize = undefined;
    for (0..m) |k| idx[k] = k;
    var i: usize = 1;
    while (i < m) : (i += 1) {
        const key = idx[i];
        var j: usize = i;
        while (j > 0 and lessMulti(multis[key], multis[idx[j - 1]])) : (j -= 1) {
            idx[j] = idx[j - 1];
        }
        idx[j] = key;
    }

    // Greedy coloring: `col_last_end[c]` is the end_line of the last span placed
    // in column c (0 == unused sentinel; line numbers are >= 1).
    var col_last_end: [MAX_MULTILINE]usize = [_]usize{0} ** MAX_MULTILINE;
    var depth: usize = 0;
    for (idx[0..m]) |k| {
        var c: usize = 0;
        while (c < MAX_MULTILINE) : (c += 1) {
            if (col_last_end[c] < multis[k].start_line) break;
        }
        std.debug.assert(c < MAX_MULTILINE); // guaranteed: <= m <= MAX_MULTILINE spans
        multis[k].rail = c;
        col_last_end[c] = multis[k].end_line;
        if (c + 1 > depth) depth = c + 1;
    }
    return depth;
}

/// The coloring order over multi-line spans: (start_line asc, end_line desc,
/// order asc). Outer spans sort first so they claim lower columns; `order` breaks ties.
fn lessMulti(a: MultiSpan, b: MultiSpan) bool {
    if (a.start_line != b.start_line) return a.start_line < b.start_line;
    if (a.end_line != b.end_line) return a.end_line > b.end_line;
    return a.order < b.order;
}

/// True iff any multi-line span is open across `line` (start_line <= line <=
/// end_line). Decides both whether a line renders and whether a source-row rail
/// cell is `rail_vertical` vs a space.
fn anyRailOpen(multis: []const MultiSpan, line: usize) bool {
    for (multis) |ms| {
        if (ms.start_line <= line and line <= ms.end_line) return true;
    }
    return false;
}

/// True iff any single-line ref starts on `line` (multi-line refs excluded — they
/// render via the rail-open predicate). Forces a single-line label's own line to
/// render even with no rail open.
fn anySingleOnLine(sm: *const SourceMap, refs: []const LabelRef, line: usize) bool {
    for (refs) |ref| {
        if (ref.multi) continue;
        if (sm.lineCol(ref.lbl.span.start).line == line) return true;
    }
    return false;
}

/// A numbered gutter prefix: `line` right-aligned in a `gw`-wide field, then
/// `<line> |`. No trailing space — content rows prepend their own single space.
fn numberedPrefix(w: *std.Io.Writer, gw: usize, line: usize) std.Io.Writer.Error!void {
    try w.splatByteAll(' ', gw - digits(line));
    try w.print("{d} |", .{line});
}

/// An empty gutter prefix (separator / marker / connector rows): `gw` spaces then
/// ` |`. No trailing space.
fn emptyPrefix(w: *std.Io.Writer, gw: usize) std.Io.Writer.Error!void {
    try w.splatByteAll(' ', gw);
    try w.writeAll(" |");
}

/// Emit a styled run of `n` copies of `glyph` — the single escape-emitting seam
/// for glyph runs (bare glyphs at `.none`). Repeats by count, not byte length, so
/// a multi-byte glyph ('─' = 3 bytes / 1 cell) yields `n` cells, not `3n`.
fn styledGlyphRun(s: Style.Style, w: *std.Io.Writer, level: Style.ColorLevel, glyph: []const u8, n: usize) std.Io.Writer.Error!void {
    try s.sgrInto(w, level);
    var i: usize = 0;
    while (i < n) : (i += 1) try w.writeAll(glyph);
    try s.closeInto(w, level);
}

/// The style for a ref's rail cells / marker run: primary uses the severity hue
/// (matching the header word), secondary the dim `secondaryStyle`. Centralised so
/// the source-row rail, connectors, and markers all agree.
fn refStyle(theme: Theme, severity: Diagnostic.Severity, kind: Diagnostic.LabelKind) Style.Style {
    return if (kind == .primary) theme.style(severity) else theme.secondaryStyle();
}

/// The rail style for the multi-line span owning rail column `c`: matches the
/// span's `order` back to a ref to recover its kind. If no span owns `c` (a blank
/// cell), the returned style is irrelevant (a space is emitted raw).
fn railStyleForColumn(theme: Theme, severity: Diagnostic.Severity, multis: []const MultiSpan, refs: []const LabelRef, c: usize) Style.Style {
    for (multis) |ms| {
        if (ms.rail != c) continue;
        for (refs) |ref| {
            if (ref.multi and ref.order == ms.order) return refStyle(theme, severity, ref.lbl.kind);
        }
    }
    return refStyle(theme, severity, .primary);
}

/// Bridge a `NoteKind` to the `Severity` that `Theme` keys presentation off:
/// `.note` -> `.note`, `.help` -> `.help`.
fn noteSeverity(k: Diagnostic.NoteKind) Diagnostic.Severity {
    return switch (k) {
        .note => .note,
        .help => .help,
    };
}

/// The stable total order over labels for the marker rows: (start display col asc,
/// primary before secondary, order asc). Total + stable => byte-identical every run.
fn lessLabel(sm: *const SourceMap, tab_width: usize, a: LabelRef, b: LabelRef) bool {
    const ca = startDisplayCol(sm, a.lbl, tab_width);
    const cb = startDisplayCol(sm, b.lbl, tab_width);
    if (ca != cb) return ca < cb;
    const pa = a.lbl.kind == .primary;
    const pb = b.lbl.kind == .primary;
    if (pa != pb) return pa; // pa true (primary) => a comes first
    return a.order < b.order;
}

/// 1-based display column of a label's span start (caret pad is `this - 1`). The
/// display column, not the byte column, is what lands the marker under the exact
/// terminal cell on tab/CJK/emoji lines.
fn startDisplayCol(sm: *const SourceMap, lbl: Diagnostic.Label, tab_width: usize) usize {
    return sm.displayCol(lbl.span.start, tab_width);
}

/// Display-cell width of a single-line label's underline run, clamped to the start
/// line (a cross-line span, only reachable when demoted past the rail cap, doesn't
/// run away). Zero-width -> 1; else the clamped span's display width, floored at 1.
/// The width is a difference of display columns, NOT `width.displayWidth(spanText)`:
/// for an interior tab, width.zig reports TAB as 0 cells so `displayWidth("a\tb")`
/// = 2, under-running the on-screen 5; the displayCol difference is the true span.
fn runCellsClamped(sm: *const SourceMap, lbl: Diagnostic.Label, line: usize, tab_width: usize) usize {
    if (lbl.span.isZeroWidth()) return 1;
    const line_start_off = sm.lineStart(line);
    const line_text = sm.lineText(lbl.span.start);
    const line_end_off = line_start_off + @as(u32, @intCast(line_text.len));
    const eff_end = @min(lbl.span.end, line_end_off); // clamp cross-line span
    const cells = sm.displayCol(eff_end, tab_width) - sm.displayCol(lbl.span.start, tab_width);
    return @max(cells, 1);
}

/// Emit the top separator row: `emptyPrefix` then the rail region trailing-trimmed
/// (cells up to the last open one; all-blank emits nothing, so the row ends at `|`).
/// The top separator sits above the first rendered line where nothing is open, so
/// it is always all-blank; the trim logic keeps it correct if that ever changes.
fn emitSeparator(w: *std.Io.Writer, gw: usize, R: usize, multis: []const MultiSpan, line: usize, kind: enum { top }) std.Io.Writer.Error!void {
    _ = kind;
    try emptyPrefix(w, gw);
    if (R > 0) {
        // Find the last non-blank (open) cell; a cell is open iff a span occupies
        // that column and is open across `line`. Above the first rendered line
        // nothing is open, so `last` stays 0 and nothing is drawn.
        var last: usize = 0; // 1-based count of cells to draw
        var c: usize = 0;
        while (c < R) : (c += 1) {
            if (railCellOpen(multis, c, line)) last = c + 1;
        }
        // Trailing-trim: draw exactly `last` cells; blank cells are raw spaces.
        // (Reachable only if a future caller passes a `line` with an open rail;
        // the top separator's `line` is min_line, above which nothing is open.)
        var k: usize = 0;
        while (k < last) : (k += 1) {
            try w.writeByte(' ');
        }
    }
    try w.writeByte('\n');
}

/// True iff rail column `c` is drawn on a source row for `line`: a span owns `c`
/// and start_line < line <= end_line. On a span's start line the cell is a space
/// (the corner descends on the connector below), so it is not open here.
fn railCellOpen(multis: []const MultiSpan, c: usize, line: usize) bool {
    for (multis) |ms| {
        if (ms.rail == c and ms.start_line < line and line <= ms.end_line) return true;
    }
    return false;
}

/// Emit the R-cell rail region for a source row at `line`: each cell is
/// `rail_vertical` (styled) when open, else a raw space. Nothing at `R == 0`.
fn emitRailRegion(w: *std.Io.Writer, theme: Theme, opts: RenderOpts, severity: Diagnostic.Severity, R: usize, multis: []const MultiSpan, refs: []const LabelRef, line: usize) std.Io.Writer.Error!void {
    if (R == 0) return;
    var c: usize = 0;
    while (c < R) : (c += 1) {
        if (railCellOpen(multis, c, line)) {
            const s = railStyleForColumn(theme, severity, multis, refs, c);
            try styledGlyphRun(s, w, opts.color, theme.glyphs.rail_vertical, 1);
        } else {
            try w.writeByte(' ');
        }
    }
}

/// Emit the R-cell rail region for a connector row of the span owning column
/// `own`. Draws all `R` cells (like a source row) so the content origin stays at
/// offset `R`, which is why the caret lands at display offset `R + (dcol-1)` for
/// every `own`. Cells: `c < own` open-state (`rail_vertical` or space); `c == own`
/// the corner `own_glyph`; `c > own` `rail_horizontal` (the run crosses them —
/// overwriting a still-open inner rail, the documented overlap simplification).
/// Nothing at `R == 0`.
fn emitConnectorRail(w: *std.Io.Writer, theme: Theme, opts: RenderOpts, severity: Diagnostic.Severity, R: usize, multis: []const MultiSpan, refs: []const LabelRef, line: usize, own: usize, own_glyph: []const u8, own_style: Style.Style) std.Io.Writer.Error!void {
    if (R == 0) return;
    var c: usize = 0;
    while (c < R) : (c += 1) {
        if (c == own) {
            try styledGlyphRun(own_style, w, opts.color, own_glyph, 1);
        } else if (c > own) {
            // The connector's horizontal run crosses this cell (see doc above).
            try styledGlyphRun(own_style, w, opts.color, theme.glyphs.rail_horizontal, 1);
        } else if (railCellOpen(multis, c, line)) {
            const s = railStyleForColumn(theme, severity, multis, refs, c);
            try styledGlyphRun(s, w, opts.color, theme.glyphs.rail_vertical, 1);
        } else {
            try w.writeByte(' ');
        }
    }
}

/// Emit a source row for `line`: `numberedPrefix` + ' ' + rail region + the
/// borrowed (CRLF-stripped) line text. The source text is never styled.
fn emitSourceLine(w: *std.Io.Writer, theme: Theme, opts: RenderOpts, gw: usize, R: usize, severity: Diagnostic.Severity, sm: *const SourceMap, line: usize, multis: []const MultiSpan, refs: []const LabelRef) std.Io.Writer.Error!void {
    try numberedPrefix(w, gw, line);
    try w.writeByte(' ');
    try emitRailRegion(w, theme, opts, severity, R, multis, refs, line);
    try w.writeAll(sm.lineText(sm.lineStart(line)));
    try w.writeByte('\n');
}

/// Emit the start-connector row for multi-line span `s` (after its start-line
/// source row): the R-cell connector rail (corner `rail_top` at `s.rail`) + a
/// `rail_horizontal` run of `K = start_dcol - 1` + a caret at the start column. No
/// message (it rides the end connector). K floors at 0.
fn emitStartConnector(w: *std.Io.Writer, theme: Theme, opts: RenderOpts, gw: usize, R: usize, severity: Diagnostic.Severity, s: MultiSpan, multis: []const MultiSpan, refs: []const LabelRef, kind: Diagnostic.LabelKind) std.Io.Writer.Error!void {
    const style = refStyle(theme, severity, kind);
    try emptyPrefix(w, gw);
    try w.writeByte(' ');
    // Use the span's own start line so lower cells reflect their open-state there.
    try emitConnectorRail(w, theme, opts, severity, R, multis, refs, s.start_line, s.rail, theme.glyphs.rail_top, style);
    // Horizontal run to the caret. K = start_dcol - 1 (>= 0).
    const K = if (s.start_dcol > 0) s.start_dcol - 1 else 0;
    try styledGlyphRun(style, w, opts.color, theme.glyphs.rail_horizontal, K);
    try styledGlyphRun(style, w, opts.color, theme.caret(), 1);
    try w.writeByte('\n');
}

/// Emit the end-connector row for multi-line span `s` (after its end-line source
/// row): the R-cell connector rail (corner `rail_bottom` at `s.rail`) + a
/// `rail_horizontal` run of `K' = end_dcol - 1` + a caret at the exclusive end
/// column (one cell past the last highlighted char) + the label message. K' floors at 0.
fn emitEndConnector(w: *std.Io.Writer, theme: Theme, opts: RenderOpts, gw: usize, R: usize, severity: Diagnostic.Severity, s: MultiSpan, lbl: Diagnostic.Label, multis: []const MultiSpan, refs: []const LabelRef) std.Io.Writer.Error!void {
    const style = refStyle(theme, severity, lbl.kind);
    try emptyPrefix(w, gw);
    try w.writeByte(' ');
    try emitConnectorRail(w, theme, opts, severity, R, multis, refs, s.end_line, s.rail, theme.glyphs.rail_bottom, style);
    const K = if (s.end_dcol > 0) s.end_dcol - 1 else 0;
    try styledGlyphRun(style, w, opts.color, theme.glyphs.rail_horizontal, K);
    try styledGlyphRun(style, w, opts.color, theme.caret(), 1);
    if (lbl.message.len != 0) {
        try w.writeByte(' ');
        try style.styled(w, opts.color, lbl.message);
    }
    try w.writeByte('\n');
}

/// Emit one single-line marker row for `lbl` on `line`: `emptyPrefix` + ' ' + rail
/// region (so the marker sits after the rail) + pad-to-column + styled glyph run +
/// optional message. A one-cell/zero-width span draws a caret; a wider span an
/// underline as wide as its display cells.
fn emitMarkerRow(
    w: *std.Io.Writer,
    theme: Theme,
    opts: RenderOpts,
    gw: usize,
    R: usize,
    severity: Diagnostic.Severity,
    lbl: Diagnostic.Label,
    sm: *const SourceMap,
    line: usize,
    multis: []const MultiSpan,
    refs: []const LabelRef,
) std.Io.Writer.Error!void {
    // Gutter (unstyled), then the content row's own leading space, then the rail.
    try emptyPrefix(w, gw);
    try w.writeByte(' ');
    try emitRailRegion(w, theme, opts, severity, R, multis, refs, line);

    // Pad to the start DISPLAY column (col - 1 cells before the marker).
    const start_dcol = startDisplayCol(sm, lbl, opts.tab_width);
    try w.splatByteAll(' ', start_dcol - 1);

    // Glyph + style. Single caret for a zero-width point OR a one-cell run;
    // otherwise an underline as wide as the span's display cells.
    const run_cells = runCellsClamped(sm, lbl, line, opts.tab_width);
    const use_caret = lbl.span.isZeroWidth() or run_cells <= 1;
    const s = refStyle(theme, severity, lbl.kind);

    if (use_caret) {
        try styledGlyphRun(s, w, opts.color, theme.caret(), 1);
    } else {
        try styledGlyphRun(s, w, opts.color, theme.underline(lbl.kind), run_cells);
    }

    if (lbl.message.len != 0) {
        try w.writeByte(' ');
        try s.styled(w, opts.color, lbl.message);
    }
    try w.writeByte('\n');
}

/// Emit an elision row: `emptyPrefix` + ' ' + the elision glyph. The rail region
/// is always blank here (a gap only falls where no rail is open, asserted by the
/// caller), so the glyph sits after the post-prefix space and '|' stays straight.
fn emitElisionRow(w: *std.Io.Writer, opts: RenderOpts, gw: usize) std.Io.Writer.Error!void {
    try emptyPrefix(w, gw);
    try w.writeByte(' ');
    try w.writeAll(if (opts.unicode) ELISION_UNICODE else ELISION_PLAIN);
    try w.writeByte('\n');
}

/// Emit the full block of rows for one rendered `line`, in a deterministic order:
/// source row, then start connectors (rail column asc), then single-line marker
/// rows (`lessLabel` order), then end connectors (rail column asc). This keeps a
/// span's start caret above its open rail and its end caret + message below.
fn emitLineBlock(w: *std.Io.Writer, theme: Theme, opts: RenderOpts, gw: usize, R: usize, severity: Diagnostic.Severity, sm: *const SourceMap, line: usize, multis: []const MultiSpan, refs: []const LabelRef) std.Io.Writer.Error!void {
    // 1. source row.
    try emitSourceLine(w, theme, opts, gw, R, severity, sm, line, multis, refs);

    // 2. start connectors, by rail column ascending. Scan columns 0..R so the
    //    output order is deterministic without sorting the spans.
    var col: usize = 0;
    while (col < R) : (col += 1) {
        for (multis) |ms| {
            if (ms.rail == col and ms.start_line == line) {
                try emitStartConnector(w, theme, opts, gw, R, severity, ms, multis, refs, multiKind(refs, ms));
            }
        }
    }

    // 3. single-line marker rows, in the globally sorted `refs` order.
    for (refs) |ref| {
        if (ref.multi) continue;
        if (sm.lineCol(ref.lbl.span.start).line != line) continue;
        try emitMarkerRow(w, theme, opts, gw, R, severity, ref.lbl, sm, line, multis, refs);
    }

    // 4. end connectors, by rail column ascending.
    col = 0;
    while (col < R) : (col += 1) {
        for (multis) |ms| {
            if (ms.rail == col and ms.end_line == line) {
                try emitEndConnector(w, theme, opts, gw, R, severity, ms, multiLabel(refs, ms), multis, refs);
            }
        }
    }
}

/// Recover the `LabelKind` of the ref owning span `ms` (matched by `order`).
/// Defaults `.primary` if not found (unreachable — every MultiSpan came from a ref).
fn multiKind(refs: []const LabelRef, ms: MultiSpan) Diagnostic.LabelKind {
    for (refs) |ref| {
        if (ref.multi and ref.order == ms.order) return ref.lbl.kind;
    }
    return .primary;
}

/// Recover the full `Label` of the ref owning span `ms` (matched by `order`), so
/// the end connector can print its message. Defaults to a bare primary label
/// if not found (unreachable).
fn multiLabel(refs: []const LabelRef, ms: MultiSpan) Diagnostic.Label {
    for (refs) |ref| {
        if (ref.multi and ref.order == ms.order) return ref.lbl;
    }
    return .{ .kind = .primary, .span = .{ .start = 0, .end = 0 } };
}

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
const e0308_quote_start: u32 = 32;
const e0308_quote_end: u32 = 36;

test "T1 zero-width primary caret: exactly one '^' with full frozen golden" {
    var sm = mapOver("z.toy", "let x = 1\n");
    defer sm.deinit(sm.gpa);
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
    var sm = mapOver("c.toy", "\u{4E16} x\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "cjk",
        .primary = .{ .kind = .primary, .span = .{ .start = 4, .end = 5 }, .message = "here" },
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        "error: cjk\n" ++
            " --> c.toy:1:5\n" ++
            "  |\n" ++
            "1 | \u{4E16} x\n" ++
            "  |    ^ here\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T7b CJK wide underline: two CJK chars -> 4-cell underline" {
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
    var sm = mapOver("e.toy", "\u{1F389}x=1\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "emoji",
        .primary = .{ .kind = .primary, .span = .{ .start = 4, .end = 5 }, .message = "x" },
    };
    var buf: [512]u8 = undefined;
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
    var sm = mapOver("tab.toy", "\tx = 1\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "tabbed",
        .primary = .{ .kind = .primary, .span = .{ .start = 1, .end = 2 }, .message = "x" },
    };
    var buf: [512]u8 = undefined;
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
    var sm = mapOver("tin.toy", "a\tb\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "interior tab",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 3 }, .message = "spans a tab" },
    };
    var buf: [512]u8 = undefined;
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
    var sm = mapOver("cm.toy", "e\u{0301}x\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "combining",
        .primary = .{ .kind = .primary, .span = .{ .start = 3, .end = 4 }, .message = "x" },
    };
    var buf: [512]u8 = undefined;
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

    try testing.expect(std.mem.indexOfScalar(u8, colored, 0x1b) != null);
    try testing.expect(std.mem.indexOfScalar(u8, plain, 0x1b) == null);
    try testing.expectEqual(
        std.mem.count(u8, plain, "\n"),
        std.mem.count(u8, colored, "\n"),
    );
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
    var hold: [512]u8 = undefined;
    @memcpy(hold[0..a.len], a);
    const first = hold[0..a.len];
    const b = renderInto(&b2, d, &sm, .{});
    try testing.expectEqualStrings(first, b);
}

test "T14 cross-line span renders as a rail (M13 replaces the M12 clamp)" {
    // "abc\ndef": primary span [0,7) starts on line 1, ends on line 2 (byte 7 is
    // the '\n' terminating line 2? no: bytes 0..3 'abc', 3 '\n', 4..7 'def', 7
    // '\n'. lineCol(7) -> line 2, col 4 (one past 'def'). So this is a genuine
    // multi-line span => R=1 => rail. start_dcol=1 (K=0, immediate caret),
    // end_dcol=4 (K'=3). Line 1 source + start connector, line 2 source + end
    // connector with the "crosses" message.
    var sm = mapOver("x.toy", "abc\ndef\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "spanned",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 7 }, .message = "crosses" },
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        \\error: spanned
        \\ --> x.toy:1:1
        \\  |
        \\1 |  abc
        \\  | /^
        \\2 | |def
        \\  | \---^ crosses
        \\
    ,
        renderInto(&buf, d, &sm, .{}),
    );
}

test "T15 one-cell non-zero span renders a caret, not an underline" {
    var sm = mapOver("o.toy", "ab\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "one",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 1 }, .message = "c" },
    };
    var buf: [256]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{});
    try testing.expect(std.mem.indexOf(u8, out, "| ^ c\n") != null);
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

    var plain_buf: [512]u8 = undefined;
    const plain = renderInto(&plain_buf, d, &sm, .{ .unicode = false });
    try testing.expectEqual(
        std.mem.count(u8, plain, "\n"),
        std.mem.count(u8, renderInto(&buf, d, &sm, .{ .unicode = true }), "\n"),
    );
}

test "T17 gutter width scales for a 2-digit line number" {
    const src = "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nlet x = 1\n";
    var sm = mapOver("g.toy", src);
    defer sm.deinit(sm.gpa);
    const x_off: u32 = 35 + 4;
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "scale",
        .primary = .{ .kind = .primary, .span = .{ .start = x_off, .end = x_off + 1 }, .message = "x" },
    };
    var buf: [512]u8 = undefined;
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

test "M13: off-line single-line label renders on its own line (no rail)" {
    // Primary on line 1 (single-line), a secondary on line 2 (also single-line).
    // Both start_line==end_line => R=0 => NO rail. M13 renders line 2 on its own
    // (M12 dropped it). Lines 1,2 adjacent => no elision.
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
    // The off-line label now DOES appear.
    try testing.expect(std.mem.indexOf(u8, out, "off-line") != null);
    // No rail cell (R=0): no '|' beyond the two gutter pipes per row.
    try testing.expectEqualStrings(
        \\error: drop
        \\ --> d.toy:1:1
        \\  |
        \\1 | abc
        \\  | ^^^ here
        \\2 | def
        \\  | --- off-line
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

test "a zero-width primary renders one caret (the sink-diagnostic render shape)" {
    var sm = mapOver("f.toy", "abcdef\n");
    defer sm.deinit(sm.gpa);
    // The shape `DiagRender.renderSinkDiag` builds inline from a POD: a zero-width
    // primary at the byte offset, no code (=> no bracket), .err severity.
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "boom",
        .primary = .{ .kind = .primary, .span = .{ .start = 2, .end = 2 }, .message = "boom" },
    };
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

// The canonical multi-line block diagnostic. Source "fn f() {\n    a\n}\n":
//   line_starts = [0, 9, 15, 17]; '{' at byte 7 (line 1, byteCol 8, displayCol 8);
//   '}' at byte 15 (line 3), span.end = 16 (displayCol on line 3 = 2). So the
//   primary span {7,16} is multi-line (start line 1, end line 3) => R=1.
//   start_dcol=8 (K=7), end_dcol=2 (K'=1), gw=digits(3)=1, location 1:8.
const blk_src = "fn f() {\n    a\n}\n";

test "M13-1 single multi-line span PLAIN: rail top/vertical/bottom, connectors + message" {
    var sm = mapOver("blk.toy", blk_src);
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "unterminated block",
        .primary = .{ .kind = .primary, .span = .{ .start = 7, .end = 16 }, .message = "unterminated block" },
    };
    var buf: [512]u8 = undefined;
    // Start-line source-row rail cell is a SPACE (corner descends below), so the
    // rail column is: start-conn '/', L2/L3 '|', end-conn '\'. All at output col
    // index 4. Start caret at col 8 ('{'), end caret at exclusive col 2 (past '}').
    try testing.expectEqualStrings(
        "error: unterminated block\n" ++
            " --> blk.toy:1:8\n" ++
            "  |\n" ++
            "1 |  fn f() {\n" ++
            "  | /-------^\n" ++
            "2 | |    a\n" ++
            "3 | |}\n" ++
            "  | \\-^ unterminated block\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "M13-2 single multi-line span UNICODE: rail ╭ │ ╰ ─, gutter '|' stays ASCII" {
    var sm = mapOver("blk.toy", blk_src);
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "unterminated block",
        .primary = .{ .kind = .primary, .span = .{ .start = 7, .end = 16 }, .message = "unterminated block" },
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        "error: unterminated block\n" ++
            " --> blk.toy:1:8\n" ++
            "  |\n" ++
            "1 |  fn f() {\n" ++
            "  | \u{256D}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}^\n" ++
            "2 | \u{2502}    a\n" ++
            "3 | \u{2502}}\n" ++
            "  | \u{2570}\u{2500}^ unterminated block\n",
        renderInto(&buf, d, &sm, .{ .unicode = true }),
    );
}

test "M13-3 span across 4 lines PLAIN: intervening rail_vertical rows, no connector" {
    // "a\nb\nc\nd\n": line_starts [0,2,4,6,8]. Primary {0,7}: start line 1 (col 1),
    // end byte 7 -> line 4 ('d' at 6, byte 7 is one past 'd' -> col 2). Lines 2,3
    // are intervening: source rows with '|' and NO connector row.
    var sm = mapOver("four.toy", "a\nb\nc\nd\n");
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "big span",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 7 }, .message = "whole thing" },
    };
    var buf: [512]u8 = undefined;
    // start_dcol=1 (K=0 -> immediate caret), end_dcol=2 (K'=1).
    try testing.expectEqualStrings(
        "error: big span\n" ++
            " --> four.toy:1:1\n" ++
            "  |\n" ++
            "1 |  a\n" ++
            "  | /^\n" ++
            "2 | |b\n" ++
            "3 | |c\n" ++
            "4 | |d\n" ++
            "  | \\-^ whole thing\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "M13-4 message rides the END connector, not the START" {
    var sm = mapOver("blk.toy", blk_src);
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "hdr",
        .primary = .{ .kind = .primary, .span = .{ .start = 7, .end = 16 }, .message = "MSG_HERE" },
    };
    var buf: [512]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{});
    // The message appears exactly once, on the end-connector line (which begins
    // with the rail_bottom '\'), never on the start-connector line ('/').
    const start_line = "  | /-------^\n";
    const end_line = "  | \\-^ MSG_HERE\n";
    try testing.expect(std.mem.indexOf(u8, out, start_line) != null);
    try testing.expect(std.mem.indexOf(u8, out, end_line) != null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, "MSG_HERE"));
}

test "M13-5 single-line secondary on an interior line sits AFTER the rail_vertical" {
    // Multi-line primary {0,7} over "a\nb\nc\nd\n" opens the rail on lines 1..4.
    // A single-line secondary on line 2 ('b', byte 2, span {2,3}) must render its
    // caret AFTER the open rail cell.
    var sm = mapOver("mix.toy", "a\nb\nc\nd\n");
    defer sm.deinit(sm.gpa);
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 2, .end = 3 }, .message = "bee" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "mixed",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 7 }, .message = "span" },
        .secondary = &secondary,
    };
    var buf: [512]u8 = undefined;
    // Line 2 source "2 | |b" then a marker row "  | |^ bee": the open rail '|'
    // comes first, then 'b' at content display col 1 (pad 0), so the caret sits
    // directly after the rail cell. The secondary marker uses '^' (1-cell).
    try testing.expectEqualStrings(
        "error: mixed\n" ++
            " --> mix.toy:1:1\n" ++
            "  |\n" ++
            "1 |  a\n" ++
            "  | /^\n" ++
            "2 | |b\n" ++
            "  | |^ bee\n" ++
            "3 | |c\n" ++
            "4 | |d\n" ++
            "  | \\-^ span\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "M13-6 gutter width from a 2-digit max line: rail + '|' stay straight" {
    // Source with lines up to a 2-digit end line. 8 short lines, then a span from
    // line 8 to line 12. line_starts: lines 1..9 are "N\n" (2 bytes) for N=1..8 =>
    // wait, build explicitly. Use "l1\n"..."l12\n" style so we get to line 12.
    const src = "a\nb\nc\nd\ne\nf\ng\nSTART x\ni\nj\nk\nEND\n";
    var sm = mapOver("wide.toy", src);
    defer sm.deinit(sm.gpa);
    // line_starts: a\n=0, b=2, c=4, d=6, e=8, f=10, g=12, START x=14 (line 8),
    // i=22, j=24, k=26, END=28 (line 12). span from byte 14 (line 8, 'S', col 1)
    // to byte 30 (inside "END" on line 12). displayCol(30): line 12 starts at 28,
    // 30-28=2 chars 'EN' -> col 3.
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "wide gutter",
        .primary = .{ .kind = .primary, .span = .{ .start = 14, .end = 30 }, .message = "far" },
    };
    var buf: [1024]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{});
    // gw=digits(12)=2. Numbered rows "N |" right-aligned in 2, empty " |" as
    // "   |". The elision collapses lines 9..11 (no rail? no — the rail is open
    // on 8..12, so EVERY line renders, no elision). Verify the 2-wide gutter and
    // straight rail.
    try testing.expectEqualStrings(
        "error: wide gutter\n" ++
            " --> wide.toy:8:1\n" ++
            "   |\n" ++
            " 8 |  START x\n" ++
            "   | /^\n" ++
            " 9 | |i\n" ++
            "10 | |j\n" ++
            "11 | |k\n" ++
            "12 | |END\n" ++
            "   | \\--^ far\n",
        out,
    );
}

test "M13-7 elision PLAIN: two single-line labels far apart => one '...' row" {
    // Two SINGLE-line labels on lines 1 and 5, no multi-line span => R=0, no rail.
    // Lines 2,3,4 are not labelled and no rail is open => elide with one "...".
    var sm = mapOver("el.toy", "one\ntwo\nthree\nfour\nfive\n");
    defer sm.deinit(sm.gpa);
    // line_starts: 0, 4, 8, 14, 19, 24. Primary {0,3} on line 1; secondary on
    // line 5: "five" at byte 19, span {19,23}.
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 19, .end = 23 }, .message = "last" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "gap",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 3 }, .message = "first" },
        .secondary = &secondary,
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        "error: gap\n" ++
            " --> el.toy:1:1\n" ++
            "  |\n" ++
            "1 | one\n" ++
            "  | ^^^ first\n" ++
            "  | ...\n" ++
            "5 | five\n" ++
            "  | ---- last\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "M13-8 elision UNICODE: same shape with '⋮'" {
    var sm = mapOver("el.toy", "one\ntwo\nthree\nfour\nfive\n");
    defer sm.deinit(sm.gpa);
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 19, .end = 23 }, .message = "last" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "gap",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 3 }, .message = "first" },
        .secondary = &secondary,
    };
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        "error: gap\n" ++
            " --> el.toy:1:1\n" ++
            "  |\n" ++
            "1 | one\n" ++
            "  | \u{2500}\u{2500}\u{2500} first\n" ++
            "  | \u{22EE}\n" ++
            "5 | five\n" ++
            "  | \u{2504}\u{2504}\u{2504}\u{2504} last\n",
        renderInto(&buf, d, &sm, .{ .unicode = true }),
    );
}

test "M13-9 two overlapping multi-line spans PLAIN: R=2, nested columns, no corruption" {
    // Source of 6 lines. Outer primary span lines 1->5, inner secondary span
    // lines 2->4. Greedy coloring: outer (start 1) gets column 0, inner (start 2,
    // last_end col0 = 5 >= 2) gets column 1 => R=2. Outer=primary (severity),
    // inner=secondary (dim).
    var sm = mapOver("ov.toy", "aa\nbb\ncc\ndd\nee\nff\n");
    defer sm.deinit(sm.gpa);
    // line_starts: 0,3,6,9,12,15,18. Outer primary {0,13}: start line 1 col 1,
    // end byte 13 -> line 5 ('e' at 12, byte 13 col 2). Inner secondary {3,10}:
    // start line 2 col 1, end byte 10 -> line 4 ('d' at 9, byte 10 col 2).
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 3, .end = 10 }, .message = "inner" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "nested",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 13 }, .message = "outer" },
        .secondary = &secondary,
    };
    var buf: [1024]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{});
    // Determinism + structural sanity: no 0x1b at .none, and the two rails render
    // (column 0 outer, column 1 inner). Assert the exact byte-for-byte layout.
    // R=2. Source rows have 2 rail cells. Outer opens on line 1 (start conn col0),
    // inner opens on line 2 (start conn col1). Inner ends line 4 (end conn col1),
    // outer ends line 5 (end conn col0).
    // Rail cell states per source row (cell0 outer, cell1 inner):
    //  L1: outer start-line => cell0 space; inner not yet => cell1 space.
    //  L2: outer open (1<2<=5) => cell0 '|'; inner start-line => cell1 space.
    //  L3: outer '|'; inner open (2<3<=4) => cell1 '|'.
    //  L4: outer '|'; inner open => cell1 '|'.
    //  L5: outer open (1<5<=5) => cell0 '|'; inner past end => cell1 space.
    // Outer own=0, inner own=1, R=2. Connector rows draw all R rail cells then
    // K=dcol-1 horizontals + caret. Outer start: cell0 '/', cell1 '-' (crossed),
    // K=0 => "/-^", caret under first 'a' (offset 2 = R). Inner start: cell0 '|'
    // (outer open on L2), cell1 '/', K=0 => "|/^", caret under 'b'. Inner end:
    // cell0 '|', cell1 '\', K=1 => "|\-^", caret at exclusive col 2 (offset 3).
    // Outer end: cell0 '\', cell1 '-' (crossed), K=1 => "\--^", caret at offset 3.
    try testing.expectEqualStrings(
        "error: nested\n" ++
            " --> ov.toy:1:1\n" ++
            "  |\n" ++
            "1 |   aa\n" ++
            "  | /-^\n" ++
            "2 | | bb\n" ++
            "  | |/^\n" ++
            "3 | ||cc\n" ++
            "4 | ||dd\n" ++
            "  | |\\-^ inner\n" ++
            "5 | | ee\n" ++
            "  | \\--^ outer\n",
        out,
    );
    try testing.expect(std.mem.indexOfScalar(u8, out, 0x1b) == null);
}

test "M13-9b staggered (non-nested) overlap: crossing stays aligned, not corrupt" {
    // The case greedy interval coloring is specifically built for: two multi-line
    // spans that OVERLAP but do NOT nest (outer ends WHILE the inner is still
    // open). Outer primary lines 1->4 (col 0), spanB secondary lines 3->6 (col 1).
    // On line 4 the outer END connector crosses spanB's still-open col-1 rail: the
    // documented simplification overwrites spanB's '|' with rail_horizontal on THAT
    // connector row only. Output stays column-aligned (the locked "must not
    // corrupt" bar); every caret still lands via displayCol.
    var sm = mapOver("ov.toy", "aa\nbb\ncc\ndd\nee\nff\n");
    defer sm.deinit(sm.gpa);
    // line_starts: 0,3,6,9,12,15,18. Outer primary {0,10}: start line 1 col 1, end
    // byte 10 -> line 4 ('d' at 9, col 2). spanB secondary {6,17}: start line 3 col
    // 1, end byte 17 -> line 6 ('f' at 16, col 3). Greedy: outer col0 (end 4),
    // spanB col1 (col0 last_end 4 >= 3) => R=2.
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 6, .end = 17 }, .message = "spanB" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "staggered",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 10 }, .message = "outer" },
        .secondary = &secondary,
    };
    var buf: [1024]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{});
    // L4 source row shows BOTH rails "||dd" (spanB open, outer open through its end
    // line). The outer END connector "\--^ outer" replaces spanB's col-1 '|' with
    // '-' as the horizontal run crosses it — aligned, readable, never corrupt.
    // spanB's rail resumes as '|' on L5/L6 source rows (cell0 blank once the outer
    // has closed). Outer end caret at exclusive col 2 (offset 3); spanB end caret
    // at exclusive col 3 (offset 4).
    try testing.expectEqualStrings(
        "error: staggered\n" ++
            " --> ov.toy:1:1\n" ++
            "  |\n" ++
            "1 |   aa\n" ++
            "  | /-^\n" ++
            "2 | | bb\n" ++
            "3 | | cc\n" ++
            "  | |/^\n" ++
            "4 | ||dd\n" ++
            "  | \\--^ outer\n" ++
            "5 |  |ee\n" ++
            "6 |  |ff\n" ++
            "  |  \\--^ spanB\n",
        out,
    );
    try testing.expect(std.mem.indexOfScalar(u8, out, 0x1b) == null);
}

test "M13-10 tab/CJK start line: connector K + caret land via displayCol" {
    // Line 1 begins with a tab, then a multi-line span starts after it. The start
    // connector's horizontal run must use displayCol (tab expands to col 5), not
    // the byte offset. "\tXY\nZ\n": tab (col 1->stop 4), X at display col 5.
    var sm = mapOver("tabml.toy", "\tXY\nZ\n");
    defer sm.deinit(sm.gpa);
    // line_starts: 0, 4, 6. Span from byte 1 ('X', line 1, displayCol 5) to byte
    // 5 ('Z' end -> line 2 col 2). start_dcol=5 (K=4), end_dcol=2 (K'=1).
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "tabbed span",
        .primary = .{ .kind = .primary, .span = .{ .start = 1, .end = 5 }, .message = "t" },
    };
    var buf: [512]u8 = undefined;
    // Start connector: rail_top + 4 horizontals + caret (caret under 'X' at
    // display col 5). Source row rail cell is a space on the start line.
    try testing.expectEqualStrings(
        "error: tabbed span\n" ++
            " --> tabml.toy:1:2\n" ++
            "  |\n" ++
            "1 |  \tXY\n" ++
            "  | /----^\n" ++
            "2 | |Z\n" ++
            "  | \\-^ t\n",
        renderInto(&buf, d, &sm, .{}),
    );
}

test "M13-11 PLAIN zero-escape at .none for a railed diagnostic (generalizes T11)" {
    var sm = mapOver("blk.toy", blk_src);
    defer sm.deinit(sm.gpa);
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 13, .end = 14 }, .message = "a" },
    };
    const notes = [_]Diagnostic.Note{
        .{ .kind = .note, .message = "n" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "railed",
        .primary = .{ .kind = .primary, .span = .{ .start = 7, .end = 16 }, .message = "block" },
        .secondary = &secondary,
        .notes = &notes,
    };
    var buf: [1024]u8 = undefined;
    const out = renderInto(&buf, d, &sm, .{ .color = .none });
    try testing.expect(std.mem.indexOfScalar(u8, out, 0x1b) == null);
}

test "M13-12 color strip-equality for a railed diagnostic (generalizes T12)" {
    var sm = mapOver("blk.toy", blk_src);
    defer sm.deinit(sm.gpa);
    const d = Diagnostic.Diagnostic{
        .severity = .err,
        .message = "railed",
        .primary = .{ .kind = .primary, .span = .{ .start = 7, .end = 16 }, .message = "unterminated block" },
    };
    var plain_buf: [1024]u8 = undefined;
    var color_buf: [1024]u8 = undefined;
    const plain = renderInto(&plain_buf, d, &sm, .{ .color = .none });
    const colored = renderInto(&color_buf, d, &sm, .{ .color = .ansi16 });
    try testing.expect(std.mem.indexOfScalar(u8, colored, 0x1b) != null);
    try testing.expectEqual(
        std.mem.count(u8, plain, "\n"),
        std.mem.count(u8, colored, "\n"),
    );
    var strip_buf: [1024]u8 = undefined;
    try testing.expectEqualStrings(plain, stripSgr(&strip_buf, colored));
}

test "M13-13 byte-identical across runs for a railed + overlapping diagnostic (generalizes T13)" {
    var sm = mapOver("ov.toy", "aa\nbb\ncc\ndd\nee\nff\n");
    defer sm.deinit(sm.gpa);
    const secondary = [_]Diagnostic.Label{
        .{ .kind = .secondary, .span = .{ .start = 3, .end = 10 }, .message = "inner" },
    };
    const d = Diagnostic.Diagnostic{
        .severity = .warning,
        .message = "nested",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 13 }, .message = "outer" },
        .secondary = &secondary,
    };
    var b1: [1024]u8 = undefined;
    var b2: [1024]u8 = undefined;
    const a = renderInto(&b1, d, &sm, .{});
    var hold: [1024]u8 = undefined;
    @memcpy(hold[0..a.len], a);
    const first = hold[0..a.len];
    const b = renderInto(&b2, d, &sm, .{});
    try testing.expectEqualStrings(first, b);
}
