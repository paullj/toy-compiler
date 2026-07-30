//! Terminal display width of Unicode text: how many cells a codepoint or UTF-8
//! string occupies (0 for combining/zero-width/controls, 2 for East-Asian Wide +
//! Fullwidth + emoji, else 1). A pure leaf; no I/O, no allocation.
//! - The range tables cover the common wide/combining blocks, not the full UCD; a
//!   rare codepoint outside them defaults to width 1.
//! - No grapheme/ZWJ segmentation: a ZWJ emoji sequence is measured per codepoint,
//!   so it over-counts. Conservative direction — never under-reserve space.
//! - Tab (U+0009) is a C0 control and reports 0; the caller expands tabs via its
//!   own `tab_width` before measuring.

const std = @import("std");
const unicode = std.unicode;

// A half-open codepoint interval [lo, hi]. Tables are sorted by `lo` and
// non-overlapping so membership is a binary search.
const Range = struct { lo: u21, hi: u21 };

// The binary search below is only correct when a table is sorted by `lo` with
// no overlaps. Prove it at compile time so a mis-ordered edit fails the build
// rather than silently returning wrong widths.
fn assertSorted(comptime ranges: []const Range) void {
    comptime {
        var prev: u21 = 0;
        for (ranges, 0..) |r, idx| {
            if (r.lo > r.hi) @compileError("width range has lo > hi");
            if (idx != 0 and r.lo <= prev) @compileError("width ranges must be strictly ascending and non-overlapping");
            prev = r.hi;
        }
    }
}

fn inRanges(cp: u21, ranges: []const Range) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = ranges[mid];
        if (cp < r.lo) {
            hi = mid;
        } else if (cp > r.hi) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

// Zero-width: combining marks (Mn/Me), format/join controls (Cf), and explicit
// zero-width characters. Not exhaustive; covers the common blocks.
const zero_width = [_]Range{
    .{ .lo = 0x0300, .hi = 0x036F }, // combining diacritical marks
    .{ .lo = 0x0483, .hi = 0x0489 }, // Cyrillic combining
    .{ .lo = 0x0591, .hi = 0x05BD }, // Hebrew points
    .{ .lo = 0x0610, .hi = 0x061A }, // Arabic marks
    .{ .lo = 0x064B, .hi = 0x065F }, // Arabic marks
    .{ .lo = 0x0670, .hi = 0x0670 }, // Arabic superscript alef
    .{ .lo = 0x06D6, .hi = 0x06DC }, // Arabic small high marks
    .{ .lo = 0x06DF, .hi = 0x06E4 },
    .{ .lo = 0x06E7, .hi = 0x06E8 },
    .{ .lo = 0x06EA, .hi = 0x06ED },
    .{ .lo = 0x0711, .hi = 0x0711 }, // Syriac
    .{ .lo = 0x0730, .hi = 0x074A },
    .{ .lo = 0x0E31, .hi = 0x0E31 }, // Thai
    .{ .lo = 0x0E34, .hi = 0x0E3A },
    .{ .lo = 0x0E47, .hi = 0x0E4E },
    .{ .lo = 0x1DC0, .hi = 0x1DFF }, // combining diacritical marks supplement
    .{ .lo = 0x200B, .hi = 0x200F }, // ZWSP, ZWNJ, ZWJ, LRM, RLM
    .{ .lo = 0x202A, .hi = 0x202E }, // bidi embedding/override controls
    .{ .lo = 0x2060, .hi = 0x2064 }, // word joiner, invisible operators
    .{ .lo = 0x20D0, .hi = 0x20F0 }, // combining marks for symbols
    .{ .lo = 0xFE00, .hi = 0xFE0F }, // variation selectors
    .{ .lo = 0xFEFF, .hi = 0xFEFF }, // BOM / ZWNBSP
    .{ .lo = 0xE0100, .hi = 0xE01EF }, // variation selectors supplement
};

// Width-2: East-Asian Wide + Fullwidth ranges plus common emoji blocks.
// Not exhaustive; covers the common CJK/emoji blocks.
const wide = [_]Range{
    .{ .lo = 0x1100, .hi = 0x115F }, // Hangul Jamo (leading)
    .{ .lo = 0x2600, .hi = 0x27BF }, // misc symbols + dingbats (common emoji)
    .{ .lo = 0x2E80, .hi = 0x303E }, // CJK radicals, Kangxi, CJK symbols/punctuation
    .{ .lo = 0x3041, .hi = 0x33FF }, // Hiragana, Katakana, CJK compat, etc.
    .{ .lo = 0x3400, .hi = 0x4DBF }, // CJK Unified Ext A
    .{ .lo = 0x4E00, .hi = 0x9FFF }, // CJK Unified Ideographs
    .{ .lo = 0xA000, .hi = 0xA4CF }, // Yi
    .{ .lo = 0xAC00, .hi = 0xD7A3 }, // Hangul syllables
    .{ .lo = 0xF900, .hi = 0xFAFF }, // CJK compatibility ideographs
    .{ .lo = 0xFE30, .hi = 0xFE4F }, // CJK compatibility forms
    .{ .lo = 0xFF01, .hi = 0xFF60 }, // Fullwidth forms
    .{ .lo = 0xFFE0, .hi = 0xFFE6 }, // Fullwidth signs
    .{ .lo = 0x1F1E6, .hi = 0x1F1FF }, // regional indicators (flags)
    .{ .lo = 0x1F300, .hi = 0x1FAFF }, // emoji (misc symbols, pictographs, ...)
    .{ .lo = 0x20000, .hi = 0x3FFFD }, // CJK Unified Ext B..
};

/// Display width of a single codepoint, in terminal cells.
///   0 — combining marks, zero-width/format controls, and C0/C1 controls
///   2 — East-Asian Wide, Fullwidth, and common emoji
///   1 — everything else (the default for codepoints outside the tables)
pub fn displayWidthCp(cp: u21) u8 {
    comptime assertSorted(&zero_width);
    comptime assertSorted(&wide);
    // C0 controls (incl. tab, newline) and DEL + C1 controls render nothing.
    if (cp < 0x20 or (cp >= 0x7F and cp <= 0x9F)) return 0;
    if (inRanges(cp, &zero_width)) return 0;
    if (inRanges(cp, &wide)) return 2;
    return 1;
}

/// Sum of `displayWidthCp` over a UTF-8 string. Invalid bytes are handled
/// deterministically: an invalid start byte or a truncated/ill-formed sequence
/// counts as width 1 and advances one byte, so measurement never stalls and a
/// corrupt run degrades gracefully instead of throwing.
pub fn displayWidth(bytes: []const u8) usize {
    var total: usize = 0;
    var i: usize = 0;
    while (i < bytes.len) {
        const len = unicode.utf8ByteSequenceLength(bytes[i]) catch {
            total += 1;
            i += 1;
            continue;
        };
        if (i + len > bytes.len) {
            total += 1;
            i += 1;
            continue;
        }
        const cp = unicode.utf8Decode(bytes[i .. i + len]) catch {
            total += 1;
            i += 1;
            continue;
        };
        total += displayWidthCp(cp);
        i += len;
    }
    return total;
}

test "displayWidthCp ascii is 1" {
    try std.testing.expectEqual(@as(u8, 1), displayWidthCp('a'));
}

test "displayWidthCp CJK is 2" {
    try std.testing.expectEqual(@as(u8, 2), displayWidthCp(0x3042)); // あ
    try std.testing.expectEqual(@as(u8, 2), displayWidthCp(0x4E16)); // 世
}

test "displayWidthCp emoji is 2" {
    try std.testing.expectEqual(@as(u8, 2), displayWidthCp(0x1F389)); // 🎉
}

test "displayWidthCp combining mark is 0" {
    try std.testing.expectEqual(@as(u8, 0), displayWidthCp(0x0301)); // combining acute
}

test "displayWidthCp controls are 0" {
    try std.testing.expectEqual(@as(u8, 0), displayWidthCp(0x09)); // tab: caller expands
    try std.testing.expectEqual(@as(u8, 0), displayWidthCp(0x0A)); // newline
    try std.testing.expectEqual(@as(u8, 0), displayWidthCp(0x7F)); // DEL
}

test "displayWidthCp zero-width and variation selector are 0" {
    try std.testing.expectEqual(@as(u8, 0), displayWidthCp(0x200B)); // ZWSP
    try std.testing.expectEqual(@as(u8, 0), displayWidthCp(0x200D)); // ZWJ
    try std.testing.expectEqual(@as(u8, 0), displayWidthCp(0xFE0F)); // variation selector-16
}

test "displayWidth mixed string sums correctly" {
    // "aあ🎉" = 1 + 2 + 2 = 5
    try std.testing.expectEqual(@as(usize, 5), displayWidth("a\u{3042}\u{1F389}"));
    // A base letter plus a combining mark occupies one cell: 'e' + U+0301 = 1 + 0.
    try std.testing.expectEqual(@as(usize, 1), displayWidth("e\u{0301}"));
}

test "displayWidth plain ascii" {
    try std.testing.expectEqual(@as(usize, 5), displayWidth("hello"));
}

test "displayWidth invalid byte degrades to width 1 and continues" {
    // A lone continuation byte (0x80) is an invalid start: width 1, advance 1.
    // The trailing 'a' still counts, proving measurement does not stall.
    try std.testing.expectEqual(@as(usize, 2), displayWidth("\x80a"));
}

test "displayWidth truncated multibyte sequence degrades to width 1" {
    // 0xE4 announces a 3-byte sequence but the buffer ends: width 1, advance 1.
    try std.testing.expectEqual(@as(usize, 1), displayWidth("\xE4"));
}

test "displayWidth ZWJ emoji sequence over-counts (documented conservative limit)" {
    // Family "👨‍👩‍👧" = man + ZWJ + woman + ZWJ + girl. A modern terminal renders
    // one 2-cell glyph, but without grapheme segmentation we sum per codepoint:
    // 2 (man) + 0 (ZWJ) + 2 (woman) + 0 (ZWJ) + 2 (girl) = 6. Conservative:
    // we over-reserve, never under-reserve.
    try std.testing.expectEqual(@as(usize, 6), displayWidth("\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}"));
}

// The "noop" guarantee: arbitrary UTF-8 (emoji, CJK, RTL) flows through the
// term styling path byte-for-byte. width.zig computes widths; it must never be
// mistaken for a filter that touches the bytes the writer emits.
test "Style.styled passes arbitrary UTF-8 through byte-for-byte" {
    const Style = @import("Style.zig");
    const input = "\u{1F389}\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}\u{202B}RTL\u{202C}";

    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    try (Style.Style{}).styled(&fw, .none, input);
    try std.testing.expectEqualStrings(input, fw.buffered());

    // A plain writeAll of the same bytes is byte-identical to the styled .none
    // path, confirming styling added nothing and corrupted nothing.
    var buf2: [256]u8 = undefined;
    var fw2 = std.Io.Writer.fixed(&buf2);
    try fw2.writeAll(input);
    try std.testing.expectEqualStrings(fw.buffered(), fw2.buffered());
}
