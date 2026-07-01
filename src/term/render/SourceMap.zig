//! Source registry: maps byte offsets to line/column and extracts source lines
//! (tab-aware) for diagnostic snippets. This is the FIRST link of the
//! diagnostics rendering chain — the typed data a pretty single-line renderer
//! (M12) needs to point a caret under the exact terminal cell. It does the
//! math; it never prints. A pure borrow-only leaf: bytes/offsets in,
//! line/col/text out; no I/O, no writer.
//!
//! WHY an eager `line_starts:[]u32` index (not a per-query forward scan): a
//! diagnostic BATCH resolves many offsets against ONE source. A forward scan
//! re-walks from byte 0 for every offset (O(n) per query, O(n*d) per batch),
//! whereas one O(n) scan at `init` plus O(log lines) binary search per query is
//! strictly cheaper and — being immutable after init — makes every query a
//! `const` pure fn with no lazy state and no races. It is also exactly the data
//! structure M12 needs to slice a RANGE of lines (`lineStart(n)`) without
//! re-deriving offsets. Memory is bounded at `#lines * 4` bytes.
//!
//! BYTE column vs DISPLAY column — kept as separately-named fns on purpose:
//!   - `byteCol` counts BYTES since the last '\n' (1-based). It is byte-for-byte
//!     identical to the driver's `lineCol` (src/driver/main.zig:983), which
//!     iterates bytes and bumps `col` per byte — for ALL input, UTF-8 included,
//!     not merely ASCII. This is the offset a lexer/parser reports.
//!   - `displayCol` counts terminal CELLS since the last '\n' (1-based), via M9
//!     `width.displayWidth` with tab expansion. This is the caret column M12
//!     aligns under a glyph.
//! Feeding a display column where a byte column is expected (or vice versa)
//! silently mis-aligns a caret, so the two never share a name and a parity test
//! pins `byteCol` to the driver convention on both ASCII and multi-byte sources.
//!
//! TAB model — elastic tab stops: a '\t' in the DISPLAY path advances the column
//! to the next multiple of `tab_width` (the classic terminal behaviour), because
//! `width.displayWidthCp` reports 0 for TAB (it is a C0 control), so expansion is
//! genuinely this module's job. Expansion lives ONLY in the display path; the
//! byte path treats a tab as the single byte it is.
//!
//! LIMITS (documented on purpose, not bugs):
//!   - Lifetime: `name` and `bytes` are BORROWED. The caller owns them and they
//!     MUST outlive the `Source`. `deinit` frees ONLY the owned `line_starts`.
//!   - Only '\n' splits lines. A CRLF "\r\n" is handled (the trailing '\r' is
//!     stripped from `lineText`), but a bare old-Mac '\r' is NOT a line break —
//!     it stays inside the line's text. Matching width.zig, this is a stated
//!     limit, not a silent misbehaviour.
//!   - ZWJ over-count: display widths come from width.zig, which measures ZWJ
//!     emoji sequences codepoint-by-codepoint and thus over-counts vs. a modern
//!     terminal. Inherited here so caret columns never UNDER-shoot the glyph.

const std = @import("std");
const unicode = std.unicode;
const width = @import("../width.zig");

const Source = @This();

/// Stored so a caller can `src.deinit(src.gpa)` without re-threading the
/// allocator; `deinit` still takes `gpa` explicitly per house style.
gpa: std.mem.Allocator,
/// BORROWED display name (e.g. "main.toy"). Never freed; must outlive `Source`.
name: []const u8,
/// BORROWED source bytes. Never freed; must outlive `Source`.
bytes: []const u8,
/// OWNED. `line_starts[0] == 0` always; `line_starts[i]` is the byte offset of
/// the first char of line `i+1`, i.e. one past the i-th '\n'. Its length is the
/// line count == (number of '\n') + 1. Sorted ascending, so a binary search
/// maps an offset to its line.
line_starts: []u32,

/// A resolved position. Both fields are 1-based. `col` is BYTE-based (bytes
/// since the last '\n', +1) — superset-compatible with driver/main.zig:983.
pub const LineCol = struct { line: usize, col: usize };

/// A half-open byte range `[start, end)` into `bytes`. `start == end` is a
/// legal ZERO-WIDTH point: a caret sitting between two chars, or at EOF.
pub const Span = struct { start: u32, end: u32 };

/// The tab stop used when a caller does not supply one. Four cells is a common
/// editor default and keeps snippet carets aligned with typical source layout.
pub const default_tab_width: usize = 4;

/// Build the line index with a single O(n) scan over `bytes`. `name` and
/// `bytes` are borrowed (see the module LIMITS) — this makes no copy of either.
/// The only heap allocation is `line_starts`.
pub fn init(gpa: std.mem.Allocator, name: []const u8, bytes: []const u8) std.mem.Allocator.Error!Source {
    // Offsets are u32 everywhere (Diagnostic.byte_offset), so a source larger
    // than u32 can't be addressed. Catch it at the boundary rather than let an
    // @intCast truncate silently below.
    std.debug.assert(bytes.len <= std.math.maxInt(u32));

    var list: std.ArrayList(u32) = .empty;
    // If any append or the final toOwnedSlice fails, free the partial list so
    // init is leak-free on the error path (testing.allocator would catch a miss).
    errdefer list.deinit(gpa);

    // Line 1 always starts at byte 0, even for an empty source (which is one
    // empty line). Every '\n' opens the next line at the byte just past it.
    try list.append(gpa, 0);
    for (bytes, 0..) |c, i| {
        if (c == '\n') try list.append(gpa, @intCast(i + 1)); // i < bytes.len <= maxInt(u32), so i+1 fits u32
    }

    return .{
        .gpa = gpa,
        .name = name,
        .bytes = bytes,
        .line_starts = try list.toOwnedSlice(gpa),
    };
}

/// Frees the owned `line_starts` (the only allocation). `name`/`bytes` are
/// borrowed and left untouched. `gpa` is taken explicitly per house style and
/// MUST be the same allocator passed to `init`.
pub fn deinit(self: *Source, gpa: std.mem.Allocator) void {
    gpa.free(self.line_starts);
    self.* = undefined;
}

/// Number of lines == `line_starts.len`. Always >= 1 (an empty source is one
/// empty line).
pub fn lineCount(self: *const Source) usize {
    return self.line_starts.len;
}

/// 0-based line index containing `offset` (clamped to `bytes.len` for EOF). This
/// is the largest `i` with `line_starts[i] <= off`, i.e. `upperBound(off) - 1`.
/// An offset landing EXACTLY on a line start (one past a '\n') maps to that NEW
/// line — the classic off-by-one this fn is written to get right.
///
/// Hand-rolled binary search in the width.zig `inRanges` house style (keeps the
/// leaf dependency-light — no std.sort import). Invariant: `line_starts[0] == 0`
/// and it is sorted ascending, so a match always exists and `hi` never underflows.
pub fn lineIndex(self: *const Source, offset: u32) usize {
    const off = @min(offset, castLen(self.bytes.len));
    var lo: usize = 0;
    var hi: usize = self.line_starts.len; // half-open; upperBound(off) lands here
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (self.line_starts[mid] <= off) {
            lo = mid + 1; // mid is a candidate; look right for a later start <= off
        } else {
            hi = mid;
        }
    }
    // `lo` is now upperBound(off) (count of starts <= off). Since starts[0]==0<=off
    // always, lo >= 1, so lo-1 is the answer and never underflows.
    return lo - 1;
}

/// Resolve `offset` to a 1-based line + 1-based BYTE column. Byte-for-byte
/// identical to driver/main.zig:983 `lineCol` for ALL input (ASCII and UTF-8),
/// because that fn counts bytes and so does this one. Out-of-range offsets clamp
/// to `bytes.len` exactly like the driver (`@min(offset, len)`).
pub fn lineCol(self: *const Source, offset: u32) LineCol {
    const off = @min(offset, castLen(self.bytes.len));
    const i = self.lineIndex(off);
    // Subtraction stays in u32 (both are u32 offsets, and line_starts[i] <= off
    // so it never underflows); widen to usize BEFORE the `+ 1` so the increment
    // can't overflow u32 at an EOF offset on a ~4GiB single-line source.
    const col: usize = @as(usize, off - self.line_starts[i]) + 1;
    return .{ .line = i + 1, .col = col };
}

/// Convenience alias for the BYTE column of `offset` (== `lineCol(offset).col`).
/// Named distinctly from `displayCol` so a caller can never confuse a byte
/// column (for arithmetic on offsets) with a display column (for caret layout).
pub fn byteCol(self: *const Source, offset: u32) usize {
    return self.lineCol(offset).col;
}

/// 1-based DISPLAY column of `offset`: how many terminal cells precede it on its
/// line, +1. Tabs expand to the next `tab_width` multiple; wide glyphs count 2;
/// combining marks count 0 (all via M9 `width`). `tab_width == 0` is coerced to
/// 1 so the tab-stop math can never divide by zero or stall.
pub fn displayCol(self: *const Source, offset: u32, tab_width: usize) usize {
    const off = @min(offset, castLen(self.bytes.len));
    const i = self.lineIndex(off);
    const prefix = self.bytes[self.line_starts[i]..off];
    return 1 + displayWidthExpandingTabs(prefix, tab_width);
}

/// BORROWED slice of the whole line containing `offset`, with the trailing '\n'
/// and (for CRLF) the preceding '\r' stripped so a snippet renders clean. The
/// last line has no trailing '\n' to strip. No allocation — this is a sub-slice
/// of `bytes`.
pub fn lineText(self: *const Source, offset: u32) []const u8 {
    const off = @min(offset, castLen(self.bytes.len));
    const i = self.lineIndex(off);
    const start = self.line_starts[i];
    // Line ends at the next line's start (just past its '\n'), or EOF for the
    // last line. `end_excl` is the exclusive end of the line's TEXT.
    var end_excl: usize = if (i + 1 < self.line_starts.len) self.line_starts[i + 1] else self.bytes.len;
    // Strip the '\n' that opened the next line...
    const stripped_lf = end_excl > start and self.bytes[end_excl - 1] == '\n';
    if (stripped_lf) end_excl -= 1;
    // ...then a '\r' immediately before it (CRLF). Bare '\r' is not a line
    // break (see LIMITS), so we only strip it when it directly preceded a '\n'
    // — i.e. only when the '\n' strip above actually fired. A lone trailing '\r'
    // (last byte of the whole source, no '\n' after it) is kept in the text.
    if (stripped_lf and end_excl > start and self.bytes[end_excl - 1] == '\r') end_excl -= 1;
    return self.bytes[start..end_excl];
}

/// Byte offset where a 1-based line begins (== `line_starts[line-1]`). Asserts
/// `1 <= line_1based <= lineCount()`; lets M12 slice a range of lines directly.
pub fn lineStart(self: *const Source, line_1based: usize) u32 {
    std.debug.assert(line_1based >= 1 and line_1based <= self.line_starts.len);
    return self.line_starts[line_1based - 1];
}

/// BORROWED bytes of `span`, clamping BOTH ends to `bytes.len` so an
/// out-of-range span never reads out of bounds. A zero-width span yields "".
/// Does NOT clip to a single line — a multi-line span returns the '\n's too.
pub fn spanText(self: *const Source, span: Span) []const u8 {
    const len = castLen(self.bytes.len);
    const start = @min(span.start, len);
    const end = @max(start, @min(span.end, len)); // guard against end < start
    return self.bytes[start..end];
}

/// PRIVATE. 0-based DISPLAY width of `prefix`, expanding tabs to elastic stops.
/// Walks by UTF-8 codepoint: a '\t' advances the running column to the next
/// multiple of `tab_width`; anything else adds `width.displayWidthCp(cp)`.
///
/// Invalid bytes degrade EXACTLY as width.zig does — an invalid start byte or a
/// truncated/ill-formed sequence counts as width 1 and advances one byte — so
/// display columns and the widths width.zig reserves stay consistent, and the
/// walk never stalls on corrupt input.
fn displayWidthExpandingTabs(prefix: []const u8, tab_width: usize) usize {
    const tw = if (tab_width == 0) 1 else tab_width; // avoid divide-by-zero / stall
    var col: usize = 0;
    var i: usize = 0;
    while (i < prefix.len) {
        // Fast path for TAB before decoding: it is a single byte and needs the
        // elastic-stop math, which width.displayWidthCp (returning 0) can't do.
        if (prefix[i] == '\t') {
            col = (col / tw + 1) * tw; // advance to the next multiple of tw
            i += 1;
            continue;
        }
        const len = unicode.utf8ByteSequenceLength(prefix[i]) catch {
            col += 1;
            i += 1;
            continue;
        };
        if (i + len > prefix.len) {
            col += 1;
            i += 1;
            continue;
        }
        const cp = unicode.utf8Decode(prefix[i .. i + len]) catch {
            col += 1;
            i += 1;
            continue;
        };
        col += width.displayWidthCp(cp);
        i += len;
    }
    return col;
}

/// `bytes.len` as a u32. `init` asserts `bytes.len <= maxInt(u32)`, so this is
/// lossless for any `Source` that was constructed successfully; centralised so
/// the clamp sites read clearly.
fn castLen(len: usize) u32 {
    return @intCast(len);
}

// ---- tests -----------------------------------------------------------------

const testing = std.testing;

// The byte-counting reference: a verbatim copy of driver/main.zig:983 `lineCol`.
// Parity tests assert `Source.lineCol` equals this at every offset, pinning the
// superset-compatibility contract so a future refactor can't silently drift.
fn refLineCol(source: []const u8, offset: u32) LineCol {
    var line: usize = 1;
    var col: usize = 1;
    const end = @min(offset, source.len);
    for (source[0..end]) |c| {
        if (c == '\n') {
            line += 1;
            col = 1;
        } else col += 1;
    }
    return .{ .line = line, .col = col };
}

test "init on empty source: one empty line, clean queries, leak-free" {
    var src = try Source.init(testing.allocator, "empty.toy", "");
    defer src.deinit(src.gpa);

    try testing.expectEqual(@as(usize, 1), src.lineCount());
    try testing.expectEqualSlices(u32, &[_]u32{0}, src.line_starts);
    try testing.expectEqual(LineCol{ .line = 1, .col = 1 }, src.lineCol(0));
    try testing.expectEqualStrings("", src.lineText(0));
    // Offset past EOF clamps to len (== 0) and stays valid.
    try testing.expectEqual(LineCol{ .line = 1, .col = 1 }, src.lineCol(99));
}

test "single line, no trailing newline" {
    var src = try Source.init(testing.allocator, "s.toy", "hello");
    defer src.deinit(src.gpa);

    try testing.expectEqual(@as(usize, 1), src.lineCount());
    try testing.expectEqual(LineCol{ .line = 1, .col = 1 }, src.lineCol(0));
    // Offset AT EOF (== len) is a legal caret one past the last char.
    try testing.expectEqual(LineCol{ .line = 1, .col = 6 }, src.lineCol(5));
    try testing.expectEqualStrings("hello", src.lineText(0));
    try testing.expectEqualStrings("hello", src.lineText(5)); // EOF still on line 1
}

test "multi-line literal: line starts, mid-line, across a newline" {
    // "ab\ncde\nf": '\n' at bytes 2 and 6 -> line starts [0, 3, 7].
    var src = try Source.init(testing.allocator, "m.toy", "ab\ncde\nf");
    defer src.deinit(src.gpa);

    try testing.expectEqualSlices(u32, &[_]u32{ 0, 3, 7 }, src.line_starts);
    // Line start of line 2 (offset just past the first '\n').
    try testing.expectEqual(LineCol{ .line = 2, .col = 1 }, src.lineCol(3));
    // Mid-line on line 2.
    try testing.expectEqual(LineCol{ .line = 2, .col = 2 }, src.lineCol(4));
    // Across the '\n': 'b' is the last char of line 1...
    try testing.expectEqual(LineCol{ .line = 1, .col = 3 }, src.lineCol(2));
    // ...and the very next offset opens line 2.
    try testing.expectEqual(LineCol{ .line = 2, .col = 1 }, src.lineCol(3));
    try testing.expectEqualStrings("ab", src.lineText(0));
    try testing.expectEqualStrings("cde", src.lineText(3));
    try testing.expectEqualStrings("f", src.lineText(7));
}

test "EOF span clamps to last line; offset > len == offset == len" {
    // "ab\ncde": last line "cde" starts at byte 3, len == 6.
    var src = try Source.init(testing.allocator, "e.toy", "ab\ncde");
    defer src.deinit(src.gpa);

    // off == len: line 2, col == trailing-line byte length + 1 == 4.
    try testing.expectEqual(LineCol{ .line = 2, .col = 4 }, src.lineCol(6));
    // off > len clamps to len and gives the identical result.
    try testing.expectEqual(src.lineCol(6), src.lineCol(1000));
}

test "byteCol parity with driver lineCol on ASCII, every offset" {
    const s = "let x = 1\nlet y = 2\n\nreturn x + y";
    var src = try Source.init(testing.allocator, "p.toy", s);
    defer src.deinit(src.gpa);

    var off: u32 = 0;
    while (off <= s.len) : (off += 1) {
        const got = src.lineCol(off);
        const want = refLineCol(s, off);
        try testing.expectEqual(want.line, got.line);
        try testing.expectEqual(want.col, got.col);
        // byteCol is the same value as lineCol().col.
        try testing.expectEqual(want.col, src.byteCol(off));
    }
}

test "byteCol parity with driver lineCol on multi-byte UTF-8, every offset (GRAFT)" {
    // Mix wide (世), emoji (🎉), and a combining acute so a codepoint-counting
    // impl would diverge — proving byteCol is BYTE-based, matching the driver.
    const s = "\u{4E16}a\n\u{1F389}\ne\u{0301}x";
    var src = try Source.init(testing.allocator, "u.toy", s);
    defer src.deinit(src.gpa);

    var off: u32 = 0;
    while (off <= s.len) : (off += 1) {
        const got = src.lineCol(off);
        const want = refLineCol(s, off);
        try testing.expectEqual(want.line, got.line);
        try testing.expectEqual(want.col, got.col);
    }
}

test "tab expansion (elastic stops) in the display path only" {
    // "\tx" tab_width=4: '\t' -> col 4 (0-based), so 'x' at display col 5.
    {
        var src = try Source.init(testing.allocator, "t.toy", "\tx");
        defer src.deinit(src.gpa);
        try testing.expectEqual(@as(usize, 5), src.displayCol(1, 4));
        try testing.expectEqual(@as(usize, 2), src.byteCol(1)); // a tab is 1 byte
    }
    // "a\tb" tab_width=4: 'a'=1, tab jumps to next multiple of 4 (col 4),
    // so 'b' sits at display col 5.
    {
        var src = try Source.init(testing.allocator, "t.toy", "a\tb");
        defer src.deinit(src.gpa);
        try testing.expectEqual(@as(usize, 5), src.displayCol(2, 4));
    }
    // "\t\t" tab_width=8: first tab -> 8, second -> 16, so offset 2 is col 17.
    {
        var src = try Source.init(testing.allocator, "t.toy", "\t\t");
        defer src.deinit(src.gpa);
        try testing.expectEqual(@as(usize, 17), src.displayCol(2, 8));
    }
}

test "tab_width edge: 1 collapses, 0 coerced to 1 (no crash)" {
    var src = try Source.init(testing.allocator, "t.toy", "\tx");
    defer src.deinit(src.gpa);
    // tab_width 1: tab advances one cell, so 'x' is at display col 2.
    try testing.expectEqual(@as(usize, 2), src.displayCol(1, 1));
    // tab_width 0 is coerced to 1: same deterministic result, no divide-by-zero.
    try testing.expectEqual(@as(usize, 2), src.displayCol(1, 0));
}

test "CJK display column: wide char occupies two cells" {
    // "世x": 世 is 3 bytes / 2 cells; 'x' at byte offset 3.
    var src = try Source.init(testing.allocator, "c.toy", "\u{4E16}x");
    defer src.deinit(src.gpa);
    try testing.expectEqual(@as(usize, 4), src.byteCol(3)); // 3 bytes + 1
    try testing.expectEqual(@as(usize, 3), src.displayCol(3, default_tab_width)); // 2 cells + 1
}

test "emoji display column: 4-byte / 2-cell glyph" {
    // "🎉x": 🎉 is 4 bytes / 2 cells; 'x' at byte offset 4.
    var src = try Source.init(testing.allocator, "e.toy", "\u{1F389}x");
    defer src.deinit(src.gpa);
    try testing.expectEqual(@as(usize, 5), src.byteCol(4)); // 4 bytes + 1
    try testing.expectEqual(@as(usize, 3), src.displayCol(4, default_tab_width)); // 2 cells + 1
}

test "combining mark display column: base + mark occupy one cell" {
    // "e\u{0301}x": 'e' (1 byte/1 cell) + combining acute (2 bytes/0 cells) + 'x'.
    // 'x' at byte offset 3.
    var src = try Source.init(testing.allocator, "m.toy", "e\u{0301}x");
    defer src.deinit(src.gpa);
    try testing.expectEqual(@as(usize, 4), src.byteCol(3)); // 3 bytes + 1
    // Display: 'e' is 1 cell, the combining mark 0 cells, so the prefix is 1 cell
    // wide and 'x' sits at display col 2 — the mark does NOT advance the caret.
    try testing.expectEqual(@as(usize, 2), src.displayCol(3, default_tab_width)); // e=1, mark=0, +1
}

test "zero-width span: start == end yields empty text at a valid caret point" {
    var src = try Source.init(testing.allocator, "z.toy", "abcdef");
    defer src.deinit(src.gpa);
    try testing.expectEqualStrings("", src.spanText(.{ .start = 2, .end = 2 }));
    // The point is a valid caret: line/col queries at it are well-formed.
    try testing.expectEqual(LineCol{ .line = 1, .col = 3 }, src.lineCol(2));
    try testing.expectEqual(@as(usize, 3), src.displayCol(2, default_tab_width));
}

test "binary-search boundaries: exact line-start lands on the NEW line" {
    // Line starts at [0, 10, 20] via '\n' at bytes 9 and 19.
    const s = "aaaaaaaaa\nbbbbbbbbb\nccccc"; // 9, 1, 9, 1, 5
    var src = try Source.init(testing.allocator, "b.toy", s);
    defer src.deinit(src.gpa);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 10, 20 }, src.line_starts);

    try testing.expectEqual(@as(usize, 0), src.lineIndex(9)); // last byte of line 1
    try testing.expectEqual(@as(usize, 1), src.lineIndex(10)); // exact start of line 2
    try testing.expectEqual(@as(usize, 1), src.lineIndex(19)); // last byte of line 2
    try testing.expectEqual(@as(usize, 2), src.lineIndex(20)); // exact start of line 3
    // Past EOF clamps to the last line.
    try testing.expectEqual(@as(usize, 2), src.lineIndex(@intCast(s.len)));
    try testing.expectEqual(@as(usize, 2), src.lineIndex(9999));
}

test "binary-search: single-line source always resolves to line index 0" {
    var src = try Source.init(testing.allocator, "one.toy", "single line here");
    defer src.deinit(src.gpa);
    try testing.expectEqual(@as(usize, 0), src.lineIndex(0));
    try testing.expectEqual(@as(usize, 0), src.lineIndex(7));
    try testing.expectEqual(@as(usize, 0), src.lineIndex(9999)); // clamps, still line 0
}

test "lineText strips trailing newline and CRLF" {
    // CRLF: first line "a\r\n" -> text "a".
    {
        var src = try Source.init(testing.allocator, "crlf.toy", "a\r\nb");
        defer src.deinit(src.gpa);
        try testing.expectEqualStrings("a", src.lineText(0));
        try testing.expectEqualStrings("b", src.lineText(3)); // last line, whole
    }
    // LF only.
    {
        var src = try Source.init(testing.allocator, "lf.toy", "a\nb");
        defer src.deinit(src.gpa);
        try testing.expectEqualStrings("a", src.lineText(0));
        try testing.expectEqualStrings("b", src.lineText(2));
    }
    // Bare trailing '\r' (last byte of the whole source, no '\n' after it) is
    // NOT a line break (see LIMITS) and MUST stay in the text — the '\r' strip
    // only fires when it directly preceded a stripped '\n'.
    {
        var src = try Source.init(testing.allocator, "cr.toy", "ab\r");
        defer src.deinit(src.gpa);
        try testing.expectEqualSlices(u32, &[_]u32{0}, src.line_starts); // no '\n' -> one line
        try testing.expectEqualStrings("ab\r", src.lineText(0)); // CR kept, per contract
    }
    // Bare '\r' in the MIDDLE of a line likewise stays; only the real '\n' at the
    // line end is stripped.
    {
        var src = try Source.init(testing.allocator, "cr2.toy", "a\rb\nc");
        defer src.deinit(src.gpa);
        try testing.expectEqualStrings("a\rb", src.lineText(0)); // mid-line CR kept, '\n' stripped
        try testing.expectEqualStrings("c", src.lineText(4));
    }
}

test "empty middle line" {
    // "a\n\nb": '\n' at bytes 1 and 2 -> line starts [0, 2, 3].
    var src = try Source.init(testing.allocator, "mid.toy", "a\n\nb");
    defer src.deinit(src.gpa);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 2, 3 }, src.line_starts);
    try testing.expectEqual(LineCol{ .line = 2, .col = 1 }, src.lineCol(2));
    try testing.expectEqualStrings("", src.lineText(2)); // the empty middle line
    try testing.expectEqual(LineCol{ .line = 3, .col = 1 }, src.lineCol(3));
}

test "spanText clamps out-of-range end; lineStart over valid range" {
    var src = try Source.init(testing.allocator, "sp.toy", "abcdef"); // len 6
    defer src.deinit(src.gpa);
    // End past len clamps to len — no OOB, returns bytes[1..6].
    try testing.expectEqualStrings("bcdef", src.spanText(.{ .start = 1, .end = 999 }));
    // Both ends past len -> empty (clamped equal).
    try testing.expectEqualStrings("", src.spanText(.{ .start = 100, .end = 200 }));
    // A malformed span with end < start clamps to empty rather than panicking.
    try testing.expectEqualStrings("", src.spanText(.{ .start = 4, .end = 2 }));

    // lineStart over the valid 1-based range.
    try testing.expectEqual(@as(u32, 0), src.lineStart(1));
}

test "spanText spans a newline (does not clip to a single line)" {
    var src = try Source.init(testing.allocator, "x.toy", "ab\ncd");
    defer src.deinit(src.gpa);
    // A span crossing the '\n' returns the newline byte too.
    try testing.expectEqualStrings("b\nc", src.spanText(.{ .start = 1, .end = 4 }));
}

test "source ending in a newline has a trailing empty line" {
    // "ab\n": '\n' at byte 2 -> line starts [0, 3]. Line 2 is the empty line at EOF.
    var src = try Source.init(testing.allocator, "nl.toy", "ab\n");
    defer src.deinit(src.gpa);
    try testing.expectEqualSlices(u32, &[_]u32{ 0, 3 }, src.line_starts);
    try testing.expectEqual(@as(usize, 2), src.lineCount());
    // EOF (== len == 3) resolves to the trailing empty line 2.
    try testing.expectEqual(LineCol{ .line = 2, .col = 1 }, src.lineCol(3));
    try testing.expectEqualStrings("", src.lineText(3));
}

test "displayWidthExpandingTabs mirrors width.zig invalid-byte degradation" {
    // Invalid start byte (lone continuation 0x80) -> width 1, advance 1; then 'a'.
    try testing.expectEqual(@as(usize, 2), displayWidthExpandingTabs("\x80a", 4));
    // Truncated 3-byte lead (0xE4 at end of buffer) -> width 1, advance 1.
    try testing.expectEqual(@as(usize, 1), displayWidthExpandingTabs("\xE4", 4));
    // A tab interleaved with valid text still expands correctly around a bad byte.
    // "\t" -> 4, then invalid 0x80 -> +1 == 5.
    try testing.expectEqual(@as(usize, 5), displayWidthExpandingTabs("\t\x80", 4));
}

test "borrowed name and bytes are referenced, not copied" {
    const name = "borrow.toy";
    const bytes = "one\ntwo";
    var src = try Source.init(testing.allocator, name, bytes);
    defer src.deinit(src.gpa);
    // Identity, not just equality: the Source points at the caller's buffers.
    try testing.expectEqual(name.ptr, src.name.ptr);
    try testing.expectEqual(bytes.ptr, src.bytes.ptr);
    // lineText returns a sub-slice of the SAME backing buffer.
    const lt = src.lineText(0);
    try testing.expectEqual(bytes.ptr, lt.ptr);
}
