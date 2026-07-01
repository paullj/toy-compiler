//! Severity-to-presentation mapping: for each `Severity`, the user-facing word
//! and a `Style` (fg color + attributes), plus two glyph sets — pure 7-bit
//! ASCII and box-drawing unicode — for carets, underlines, notes, and the M13
//! rail. This is the presentation TABLE; it hands back VALUES only.
//!
//! WHY a data-only theme. `Theme` NEVER emits, NEVER holds a writer, and NEVER
//! calls `Style.styled`/`sgrInto`/`closeInto`. It returns `Style.Style` values
//! and glyph `[]const u8` slices; M12 does all emission, passing the terminal's
//! `ColorLevel` to `Style.styled`, which writes ZERO bytes at `.none`. So
//! gate-safety (no escape codes on a dumb terminal) is automatic and lives in
//! `Style`, not here — the `.none` gate-safety test below documents that.
//!
//! PALETTE (documented on purpose). Severity colors are BARE `.ansi` indices,
//! chosen so `Color.downgrade` is the identity at every `ColorLevel` (the
//! codebase's downgrade leaves `.ansi` untouched for `.ansi16`/`.ansi256`); no
//! rounding, and zero bytes at `.none`. `.rgb`/`.indexed` would round on
//! downgrade and are deliberately avoided.
//!   error   = bright red    (ansi 9),  bold
//!   warning = bright yellow (ansi 11), bold
//!   note    = bright cyan   (ansi 14), bold
//!   help    = bright green  (ansi 10), bold
//! Secondary labels use a severity-INDEPENDENT `{ .dim = true }` so they recede
//! behind the primary regardless of the diagnostic's severity.
//!
//! GLYPH SPLIT. The caret and underline glyphs are shared across severities
//! (only the COLOR varies by severity, via `style`); the severity distinction
//! is carried entirely by `Style`, not by glyph shape. The caret stays `"^"` in
//! BOTH sets — it is guaranteed a single display cell, so M12's caret-column
//! math never desyncs; a fancy caret risks width-2 rendering. The M13 rail
//! glyphs are reserved NOW so `Theme` never reshapes when multi-line lands.
//!
//! LAYERING. Imports ONLY `std`, `../Style.zig` (for `Style.Style`/`Style.Color`
//! VALUES), and the sibling `Diagnostic.zig` (for `Severity`/`LabelKind` — a
//! single source of truth, so the two files cannot drift). It does NOT import
//! `SourceMap`, `ansi`, `Terminal`, `Progress`, `Renderer`, or anything under
//! `cli/*`, and takes NO writer.

const std = @import("std");
// Note: `Style` is the FILE namespace; the struct is `Style.Style`, the color is
// `Style.Color`. Returns/literals here are `Style.Style{...}`, never bare `Style`.
const Style = @import("../Style.zig");
// Sibling render types — single source of truth for the enums, so Theme's
// severity keys and this file's `Diagnostic.Severity` can never drift apart.
const Diagnostic = @import("Diagnostic.zig");
const Severity = Diagnostic.Severity;

const Theme = @This();

/// Whether this theme draws with unicode box-drawing glyphs (`true`) or pure
/// ASCII (`false`). Selects which glyph set is active; `style`/`word` are
/// glyph-set-independent. Spelled `is_unicode` (not `unicode`) so it does not
/// collide with the `pub const unicode` preset in this same namespace.
is_unicode: bool,
/// The active glyph set (carets, underlines, and the reserved M13 rail).
glyphs: Glyphs,

/// One glyph set. Carets/underlines are shared across severities (color, not
/// shape, encodes severity). The rail glyphs are reserved for M13's multi-line
/// spans so this struct never reshapes later.
pub const Glyphs = struct {
    /// Primary point / zero-width caret. Always a single display cell.
    caret: []const u8,
    /// Underline run under a wider primary span.
    underline_primary: []const u8,
    /// Underline run under a secondary span (shape may equal primary; the
    /// receding effect comes from `secondaryStyle`, not the glyph).
    underline_secondary: []const u8,
    // --- M13 rail set (reserved now; unused by M12) ---
    rail_vertical: []const u8,
    rail_top: []const u8,
    rail_bottom: []const u8,
    rail_horizontal: []const u8,
    note_bullet: []const u8,
};

/// Pure 7-bit ASCII glyph set (safe on any terminal / when unicode is off).
pub const plain_glyphs = Glyphs{
    .caret = "^",
    .underline_primary = "^",
    .underline_secondary = "-",
    .rail_vertical = "|",
    .rail_top = "/",
    .rail_bottom = "\\",
    .rail_horizontal = "-",
    .note_bullet = "=",
};

/// Box-drawing unicode glyph set (written with `\u{...}` escapes so this source
/// stays 7-bit ASCII and the byte content is unambiguous). The caret stays `"^"`
/// to keep a guaranteed 1-cell width.
pub const unicode_glyphs = Glyphs{
    .caret = "^",
    .underline_primary = "\u{2500}", // ─ box drawings light horizontal
    .underline_secondary = "\u{2504}", // ┄ light triple dash horizontal
    .rail_vertical = "\u{2502}", // │
    .rail_top = "\u{256D}", // ╭
    .rail_bottom = "\u{2570}", // ╰
    .rail_horizontal = "\u{2500}", // ─
    .note_bullet = "\u{2550}", // ═ double horizontal
};

/// The plain-ASCII preset.
pub const plain = Theme{ .is_unicode = false, .glyphs = plain_glyphs };
/// The unicode box-drawing preset.
pub const unicode = Theme{ .is_unicode = true, .glyphs = unicode_glyphs };

/// Pick a preset from a capability flag (a driver passes the terminal's unicode
/// support).
pub fn forUnicode(want_unicode: bool) Theme {
    return if (want_unicode) unicode else plain;
}

/// The word + style for a severity, kept together so the exhaustive `switch`
/// (not an array) is the single place that must handle a new `Severity` — adding
/// one is a compile error here.
const SevInfo = struct {
    word: []const u8,
    style: Style.Style,
};

fn info(sev: Severity) SevInfo {
    return switch (sev) {
        .err => .{ .word = "error", .style = .{ .fg = .{ .ansi = 9 }, .bold = true } }, // bright red
        .warning => .{ .word = "warning", .style = .{ .fg = .{ .ansi = 11 }, .bold = true } }, // bright yellow
        .note => .{ .word = "note", .style = .{ .fg = .{ .ansi = 14 }, .bold = true } }, // bright cyan
        .help => .{ .word = "help", .style = .{ .fg = .{ .ansi = 10 }, .bold = true } }, // bright green
    };
}

/// The style (fg color + attributes) for a severity's header/primary. Takes
/// `self` for future per-theme variance; today it is glyph-set-independent.
pub fn style(self: Theme, sev: Severity) Style.Style {
    _ = self;
    return info(sev).style;
}

/// The user-facing word for a severity ("error", "warning", "note", "help").
/// The word "error" lives HERE, which is why the enum tag can be `err`.
pub fn word(self: Theme, sev: Severity) []const u8 {
    _ = self;
    return info(sev).word;
}

/// The style for a secondary label: severity-independent and dim, so secondary
/// context recedes behind the primary regardless of the diagnostic's severity.
pub fn secondaryStyle(self: Theme) Style.Style {
    _ = self;
    return .{ .dim = true };
}

/// The caret / zero-width point glyph in the active set.
pub fn caret(self: Theme) []const u8 {
    return self.glyphs.caret;
}

/// The underline run glyph for a label kind in the active set.
pub fn underline(self: Theme, kind: Diagnostic.LabelKind) []const u8 {
    return switch (kind) {
        .primary => self.glyphs.underline_primary,
        .secondary => self.glyphs.underline_secondary,
    };
}

// ---- tests -----------------------------------------------------------------

const testing = std.testing;

test "severity styles are bold with the documented ansi hue" {
    try testing.expectEqual(Style.Style{ .fg = .{ .ansi = 9 }, .bold = true }, plain.style(.err));
    try testing.expectEqual(Style.Style{ .fg = .{ .ansi = 11 }, .bold = true }, plain.style(.warning));
    try testing.expectEqual(Style.Style{ .fg = .{ .ansi = 14 }, .bold = true }, plain.style(.note));
    try testing.expectEqual(Style.Style{ .fg = .{ .ansi = 10 }, .bold = true }, plain.style(.help));
    // All bold, and glyph-set-independent (plain and unicode agree).
    inline for (.{ Severity.err, .warning, .note, .help }) |sev| {
        try testing.expect(plain.style(sev).bold);
        try testing.expectEqual(plain.style(sev), unicode.style(sev));
    }
}

test "severity words" {
    try testing.expectEqualStrings("error", plain.word(.err));
    try testing.expectEqualStrings("warning", plain.word(.warning));
    try testing.expectEqualStrings("note", plain.word(.note));
    try testing.expectEqualStrings("help", plain.word(.help));
}

test "plain glyph table exact bytes and ASCII-only guard" {
    try testing.expectEqualStrings("^", plain_glyphs.caret);
    try testing.expectEqualStrings("^", plain_glyphs.underline_primary);
    try testing.expectEqualStrings("-", plain_glyphs.underline_secondary);
    try testing.expectEqualStrings("|", plain_glyphs.rail_vertical);
    try testing.expectEqualStrings("/", plain_glyphs.rail_top);
    try testing.expectEqualStrings("\\", plain_glyphs.rail_bottom);
    try testing.expectEqualStrings("-", plain_glyphs.rail_horizontal);
    try testing.expectEqualStrings("=", plain_glyphs.note_bullet);
    // No byte >= 0x80: the plain set can never regress into unicode.
    inline for (@typeInfo(Glyphs).@"struct".fields) |f| {
        for (@field(plain_glyphs, f.name)) |b| try testing.expect(b < 0x80);
    }
}

test "unicode glyph table exact bytes" {
    try testing.expectEqualStrings("^", unicode_glyphs.caret);
    try testing.expectEqualStrings("\u{2500}", unicode_glyphs.underline_primary);
    try testing.expectEqualStrings("\u{2504}", unicode_glyphs.underline_secondary);
    try testing.expectEqualStrings("\u{2502}", unicode_glyphs.rail_vertical);
    try testing.expectEqualStrings("\u{256D}", unicode_glyphs.rail_top);
    try testing.expectEqualStrings("\u{2570}", unicode_glyphs.rail_bottom);
    try testing.expectEqualStrings("\u{2500}", unicode_glyphs.rail_horizontal);
    try testing.expectEqualStrings("\u{2550}", unicode_glyphs.note_bullet);
}

test "preset selection" {
    try testing.expect(!plain.is_unicode);
    try testing.expectEqual(plain_glyphs, plain.glyphs);
    try testing.expect(unicode.is_unicode);
    try testing.expectEqual(unicode_glyphs, unicode.glyphs);
    try testing.expectEqual(plain, forUnicode(false));
    try testing.expectEqual(unicode, forUnicode(true));
    try testing.expectEqualStrings("^", plain.caret());
    try testing.expectEqualStrings("\u{2500}", unicode.underline(.primary));
    try testing.expectEqualStrings("\u{2504}", unicode.underline(.secondary));
}

test "secondaryStyle is dim and uncoloured" {
    const s = plain.secondaryStyle();
    try testing.expectEqual(Style.Style{ .dim = true }, s);
    try testing.expectEqual(@as(?Style.Color, null), s.fg);
    try testing.expect(s.dim);
}

test "palette is downgrade-clean at every color level" {
    // Each severity's fg is a bare .ansi index, so downgrade is the identity at
    // both .ansi256 and .ansi16 — no rounding, matching Style.downgrade.
    inline for (.{ Severity.err, .warning, .note, .help }) |sev| {
        const c = plain.style(sev).fg.?;
        try testing.expectEqual(c, c.downgrade(.ansi256));
        try testing.expectEqual(c, c.downgrade(.ansi16));
    }
}

test "theme values are gate-safe when emitted through Style at .none" {
    // Theme itself never emits; this drives a Theme style THROUGH Style.styled at
    // .none to document that the values produce zero escape bytes on a dumb term.
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try plain.style(.err).styled(&w, .none, "error");
    try testing.expectEqualStrings("error", w.buffered());
    for (w.buffered()) |b| try testing.expect(b != 0x1b);
}
