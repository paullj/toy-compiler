//! The single deep home for decoding a numeric/string literal token: the range
//! verdict the checker needs (`fitsWidth`), the 64-bit `iconst` bit pattern codegen
//! needs (`value`), and the escape-decoded bytes string lowering needs
//! (`decodeString`). These once lived in three places (`BodyChecker.litMagnitude`
//! /`maxMagnitude`, `lower.parseInt`, `lower.decodeStringLiteral`) tied together only
//! by a shared separator-stripping helper and a comment; the base-0 grammar, the
//! per-width max table, and the escape table are now owned here so the range check
//! and codegen can never disagree on which literals are well-formed.

const std = @import("std");
const LayoutEngine = @import("../layout/Engine.zig");
const Type = @import("../layout/Type.zig").Type;

/// Pack a numeric-literal token into `buf`, dropping `_` digit separators, and return
/// the packed slice (or null if it overflows `buf`). The one base-0 grammar both the
/// range gate and codegen decode, so they can't disagree on well-formedness.
fn stripSeparators(raw: []const u8, buf: []u8) ?[]const u8 {
    var n: usize = 0;
    for (raw) |c| {
        if (c == '_') continue;
        if (n >= buf.len) return null;
        buf[n] = c;
        n += 1;
    }
    return buf[0..n];
}

/// The magnitude of a numeric-literal token (base-0: `0x`/`0o`/`0b` auto-detected,
/// `_` stripped) as a u128, or null if it does not parse or is absurdly long. u128
/// spans the full `uint64` range, so an over-range check is exact for every width.
fn magnitude(raw: []const u8) ?u128 {
    var buf: [128]u8 = undefined;
    const s = stripSeparators(raw, &buf) orelse return null;
    return std.fmt.parseInt(u128, s, 0) catch null;
}

/// The maximum in-range magnitude for integer type `t`, derived from its bit width:
/// signed spans `2^(bits-1)-1`, unsigned `2^bits-1`.
fn maxMagnitude(t: Type) u128 {
    const bits = t.intBits();
    return if (t.int_desc.signed)
        (@as(u128, 1) << @intCast(bits - 1)) - 1
    else
        (@as(u128, 1) << @intCast(bits)) - 1;
}

/// The 64-bit `iconst` bit pattern for a numeric-literal token, stripping `_`
/// separators. A `literal_number` token is always a non-negative magnitude (unary `-`
/// is a separate node), so it is decoded as u64 and bit-cast: this round-trips the FULL
/// `uint64` range, since the checker admits `uint`/`uint64` literals up to 2^64-1 while
/// signed widths cap at 2^63-1. Decoding as i64 instead would reject a legal
/// `uint64 = 0xFFFF...` at codegen even though the range check (u128) passed it. Base 0
/// auto-detects `0x`/`0o`/`0b`; a bare leading-zero decimal stays decimal. Null on u64
/// overflow / malformed.
pub fn value(raw: []const u8) ?i64 {
    var buf: [128]u8 = undefined;
    const s = stripSeparators(raw, &buf) orelse return null;
    return @bitCast(std.fmt.parseInt(u64, s, 0) catch return null);
}

/// Per-width range verdict for a numeric-literal token against integer type `t`. A
/// literal wider than u64 never fits any width. `negated and t.isSigned()` grants the
/// +1 signed-min allowance (`-128: int8`, `-9223372036854775808: int`), whose magnitude
/// is one past the positive max.
pub fn fitsWidth(raw: []const u8, t: Type, negated: bool) bool {
    const v = magnitude(raw) orelse return false;
    const max = maxMagnitude(t) + @intFromBool(negated and t.isSigned());
    return v <= max;
}

/// The hex value of one ASCII hex digit, or null. The single digit table `\xNN` /
/// `\u{…}` share.
fn hexDigit(c: u8) ?u4 {
    return switch (c) {
        '0'...'9' => @intCast(c - '0'),
        'a'...'f' => @intCast(c - 'a' + 10),
        'A'...'F' => @intCast(c - 'A' + 10),
        else => null,
    };
}

/// The result of decoding ONE escape sequence — the single escape table both the string
/// and char literal decoders route through, so their escape grammar can never drift.
/// `cp` is the decoded scalar; `unicode` is set ONLY for `\u{…}` (whose value the STRING
/// path UTF-8-encodes, while every other escape is a single raw byte ≤ 0xFF that it emits
/// verbatim — `\x80` is byte 0x80, not the two-byte UTF-8 of U+0080). A char literal takes
/// `cp` directly regardless of `unicode`.
const Escape = union(enum) {
    ok: struct { cp: u32, unicode: bool },
    unknown_escape,
    /// A `\x`/`\u{…}` with missing/non-hex digits, missing braces, or empty `\u{}`.
    bad_hex,
    /// A `\u{…}` naming a UTF-16 surrogate (`0xD800..=0xDFFF`) or a value `> 0x10FFFF`.
    bad_codepoint,
};

/// Decode the escape whose backslash was just consumed: `i` points AT the selector char
/// (guaranteed `< body.len` by the caller's dangling-backslash check); on `.ok` it is
/// advanced past the whole escape. Recognizes `\n \t \r \0 \\ \" \' \xNN \u{HEX…}`.
fn decodeEscape(body: []const u8, i: *usize) Escape {
    const sel = body[i.*];
    i.* += 1;
    switch (sel) {
        'n' => return .{ .ok = .{ .cp = 0x0A, .unicode = false } },
        't' => return .{ .ok = .{ .cp = 0x09, .unicode = false } },
        'r' => return .{ .ok = .{ .cp = 0x0D, .unicode = false } },
        '0' => return .{ .ok = .{ .cp = 0x00, .unicode = false } },
        '\\' => return .{ .ok = .{ .cp = '\\', .unicode = false } },
        '"' => return .{ .ok = .{ .cp = '"', .unicode = false } },
        '\'' => return .{ .ok = .{ .cp = '\'', .unicode = false } },
        'x' => {
            if (i.* + 2 > body.len) return .bad_hex;
            const hi = hexDigit(body[i.*]) orelse return .bad_hex;
            const lo = hexDigit(body[i.* + 1]) orelse return .bad_hex;
            i.* += 2;
            return .{ .ok = .{ .cp = @as(u32, hi) * 16 + lo, .unicode = false } };
        },
        'u' => {
            if (i.* >= body.len or body[i.*] != '{') return .bad_hex;
            i.* += 1;
            var cp: u32 = 0;
            var digits: usize = 0;
            while (i.* < body.len and body[i.*] != '}') : (i.* += 1) {
                const d = hexDigit(body[i.*]) orelse return .bad_hex;
                cp = cp * 16 + d;
                digits += 1;
                if (cp > 0x10FFFF) return .bad_codepoint; // caps the running value: no overflow
            }
            if (i.* >= body.len or body[i.*] != '}') return .bad_hex;
            i.* += 1; // consume `}`
            if (digits == 0) return .bad_hex;
            if (cp >= 0xD800 and cp <= 0xDFFF) return .bad_codepoint; // lone surrogate
            return .{ .ok = .{ .cp = cp, .unicode = true } };
        },
        else => return .unknown_escape,
    }
}

/// The outcome of decoding a string-literal token: the escape-decoded bytes (caller
/// owns), or one of the malformed-token kinds the caller renders as a note.
pub const StringDecode = union(enum) {
    ok: []u8,
    malformed,
    dangling_backslash,
    unknown_escape,
    bad_hex_escape,
    bad_codepoint,
};

/// Decode a string-literal token (`raw` includes the surrounding quotes) into its runtime
/// bytes: quotes stripped, escapes decoded through the shared `decodeEscape` table. A
/// `\u{…}` is UTF-8-encoded (multi-byte); every other escape emits one raw byte. Char
/// literals reuse `decodeEscape` too (see `decodeChar`) — the single escape source.
pub fn decodeString(gpa: std.mem.Allocator, raw: []const u8) error{OutOfMemory}!StringDecode {
    if (raw.len < 2 or raw[0] != '"' or raw[raw.len - 1] != '"') return .malformed;
    const body = raw[1 .. raw.len - 1];
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    while (i < body.len) {
        const c = body[i];
        if (c != '\\') {
            try out.append(gpa, c);
            i += 1;
            continue;
        }
        i += 1; // past the backslash
        if (i >= body.len) {
            out.deinit(gpa);
            return .dangling_backslash;
        }
        switch (decodeEscape(body, &i)) {
            .ok => |e| {
                if (e.unicode) {
                    var buf: [4]u8 = undefined;
                    const n = std.unicode.utf8Encode(@intCast(e.cp), &buf) catch {
                        out.deinit(gpa);
                        return .bad_codepoint;
                    };
                    try out.appendSlice(gpa, buf[0..n]);
                } else {
                    try out.append(gpa, @intCast(e.cp)); // non-`\u` escapes are ≤ 0xFF
                }
            },
            .unknown_escape => {
                out.deinit(gpa);
                return .unknown_escape;
            },
            .bad_hex => {
                out.deinit(gpa);
                return .bad_hex_escape;
            },
            .bad_codepoint => {
                out.deinit(gpa);
                return .bad_codepoint;
            },
        }
    }
    return .{ .ok = try out.toOwnedSlice(gpa) };
}

/// The outcome of decoding a char-literal token into a single Unicode scalar. Every
/// non-`ok` kind is a T0036 the checker renders; `ok` carries the codepoint the char's
/// inner `uint32` field holds.
pub const CharDecode = union(enum) {
    ok: u32,
    malformed, // not a `'…'` quoted form (unreachable for a lexed `char_lit`)
    empty, // `''`
    too_many, // more than one codepoint, or trailing bytes after an escape
    dangling_backslash,
    unknown_escape,
    bad_hex_escape,
    bad_codepoint,
    bad_utf8, // raw content is not a single well-formed UTF-8 scalar
};

/// Decode a char-literal token (`raw` includes the surrounding quotes) into exactly ONE
/// Unicode scalar: an escape (`decodeEscape`, must consume the whole body) OR a single raw
/// UTF-8 codepoint (multibyte SOURCE content — a raw `'€'` decodes here to `0x20AC`, not in
/// a later milestone). Reuses the shared escape table so `'\n'`/`'\u{20AC}'`/`'\x41'`
/// decode identically to their string-literal counterparts.
pub fn decodeChar(raw: []const u8) CharDecode {
    if (raw.len < 2 or raw[0] != '\'' or raw[raw.len - 1] != '\'') return .malformed;
    const body = raw[1 .. raw.len - 1];
    if (body.len == 0) return .empty;
    if (body[0] == '\\') {
        var i: usize = 1;
        if (i >= body.len) return .dangling_backslash; // lone `\` then closing quote
        const cp = switch (decodeEscape(body, &i)) {
            .ok => |e| e.cp,
            .unknown_escape => return .unknown_escape,
            .bad_hex => return .bad_hex_escape,
            .bad_codepoint => return .bad_codepoint,
        };
        if (i != body.len) return .too_many; // content after the escape
        return .{ .ok = cp };
    }
    // A raw scalar: exactly one UTF-8 codepoint, nothing trailing.
    const seq_len = std.unicode.utf8ByteSequenceLength(body[0]) catch return .bad_utf8;
    if (seq_len != body.len) return .too_many;
    const cp = std.unicode.utf8Decode(body[0..seq_len]) catch return .bad_utf8;
    return .{ .ok = cp };
}

const testing = std.testing;

test "value: base-aware decode round-trips the full uint64 range into the iconst bit pattern" {
    try testing.expectEqual(@as(?i64, 42), value("42"));
    try testing.expectEqual(@as(?i64, 0x2A), value("0x2A"));
    try testing.expectEqual(@as(?i64, 42), value("0o52"));
    try testing.expectEqual(@as(?i64, 42), value("0b0010_1010"));
    try testing.expectEqual(@as(?i64, 1000), value("1_000"));
    try testing.expectEqual(@as(?i64, std.math.maxInt(i64)), value("9223372036854775807"));
    // uint64 literals in [2^63, 2^64) decode to the correct 64-bit pattern.
    try testing.expectEqual(@as(?i64, @bitCast(@as(u64, 1) << 63)), value("0x8000000000000000"));
    try testing.expectEqual(@as(?i64, -1), value("0xFFFFFFFFFFFFFFFF")); // 2^64-1 as i64 bits
    try testing.expectEqual(@as(?i64, -1), value("18446744073709551615"));
    // Null past 2^64-1 (no width can admit it) and on a garbage token.
    try testing.expectEqual(@as(?i64, null), value("18446744073709551616"));
    try testing.expectEqual(@as(?i64, null), value("0xZZ"));
}

test "fitsWidth: per-width range verdict, base-aware, with the negated signed-min allowance" {
    try testing.expect(fitsWidth("127", Type.int8, false));
    try testing.expect(!fitsWidth("128", Type.int8, false));
    try testing.expect(fitsWidth("255", Type.uint8, false));
    try testing.expect(!fitsWidth("256", Type.uint8, false));
    // Base-aware: 0xC8 == 200, over int8.
    try testing.expect(!fitsWidth("0xC8", Type.int8, false));
    // Underscores are ignored when computing the value.
    try testing.expect(fitsWidth("1_000", Type.int16, false));
    // Platform int admits up to 2^63-1.
    try testing.expect(fitsWidth("9223372036854775807", Type.int, false));
    try testing.expect(!fitsWidth("9223372036854775808", Type.int, false));
    // Negated into a signed width: -128 fits int8, -129 does not.
    try testing.expect(fitsWidth("128", Type.int8, true));
    try testing.expect(!fitsWidth("129", Type.int8, true));
    // The negated allowance only applies to signed types.
    try testing.expect(!fitsWidth("256", Type.uint8, true));
    // Upper boundary of the unsigned widths — the [2^63, 2^64) band where the
    // signed/unsigned max split and the u128-vs-u64 decode actually differ.
    try testing.expect(fitsWidth("9223372036854775808", Type.uint64, false));
    try testing.expect(fitsWidth("18446744073709551615", Type.uint64, false));
    // Any width rejects a literal that overflows u64.
    try testing.expect(!fitsWidth("18446744073709551616", Type.uint64, false));
}

test "fitsWidth/value agree: anything the range gate admits, codegen can decode" {
    // The reason the two decoders live in one module: a token the range check accepts
    // must have a representable iconst pattern. Pin it across representative tokens x
    // widths x negation, so a future max-table skew (e.g. unsigned mis-derived as the
    // signed max) that over-rejects — or, worse, admits an undecodable token — is caught.
    const tokens = [_][]const u8{
        "0",                    "127",
        "128",                  "255",
        "9223372036854775807",  "9223372036854775808",
        "18446744073709551615", "18446744073709551616",
        "0xFFFFFFFFFFFFFFFF",   "1_000",
    };
    const widths = [_]Type{ Type.int8, Type.uint8, Type.int, Type.uint64 };
    for (tokens) |raw| for (widths) |t| for ([_]bool{ false, true }) |neg| {
        if (fitsWidth(raw, t, neg)) try testing.expect(value(raw) != null);
    };
}

test "decodeString: escapes, empty, and the three malformed kinds" {
    const gpa = testing.allocator;
    {
        const r = try decodeString(gpa, "\"hi\"");
        try testing.expectEqualStrings("hi", r.ok);
        gpa.free(r.ok);
    }
    {
        // Source body `a\nb\t\\\"` decodes to a, newline, b, tab, backslash, quote.
        const r = try decodeString(gpa, "\"a\\nb\\t\\\\\\\"\"");
        try testing.expectEqualSlices(u8, &[_]u8{ 'a', 0x0A, 'b', 0x09, '\\', '"' }, r.ok);
        gpa.free(r.ok);
    }
    {
        const r = try decodeString(gpa, "\"\"");
        try testing.expectEqualStrings("", r.ok);
        gpa.free(r.ok);
    }
    try testing.expectEqual(StringDecode.dangling_backslash, try decodeString(gpa, "\"x\\\""));
    try testing.expectEqual(StringDecode.unknown_escape, try decodeString(gpa, "\"\\q\""));
    try testing.expectEqual(StringDecode.malformed, try decodeString(gpa, "no-quotes"));
}

test "decodeString: the new escapes enrich strings (\\r \\0 \\' \\xNN \\u{…})" {
    const gpa = testing.allocator;
    {
        // `\r`, `\0`, `\'`, `\x41` each emit their single byte.
        const r = try decodeString(gpa, "\"\\r\\0\\'\\x41\"");
        try testing.expectEqualSlices(u8, &[_]u8{ 0x0D, 0x00, '\'', 0x41 }, r.ok);
        gpa.free(r.ok);
    }
    {
        // `\u{20AC}` UTF-8-encodes to the 3-byte euro sign; `\x80` stays the RAW byte 0x80
        // (NOT re-encoded), which is the string-vs-char difference.
        const r = try decodeString(gpa, "\"\\u{20AC}\\x80\"");
        try testing.expectEqualSlices(u8, &[_]u8{ 0xE2, 0x82, 0xAC, 0x80 }, r.ok);
        gpa.free(r.ok);
    }
    try testing.expectEqual(StringDecode.bad_hex_escape, try decodeString(gpa, "\"\\xZZ\""));
    try testing.expectEqual(StringDecode.bad_hex_escape, try decodeString(gpa, "\"\\u{}\""));
    try testing.expectEqual(StringDecode.bad_codepoint, try decodeString(gpa, "\"\\u{D800}\""));
    try testing.expectEqual(StringDecode.bad_codepoint, try decodeString(gpa, "\"\\u{110000}\""));
}

test "decodeChar: raw scalars, escapes, and every malformed kind" {
    // Simple ASCII + the escapes the spec pins.
    try testing.expectEqual(@as(u32, 65), decodeChar("'A'").ok);
    try testing.expectEqual(@as(u32, 10), decodeChar("'\\n'").ok);
    try testing.expectEqual(@as(u32, 9), decodeChar("'\\t'").ok);
    try testing.expectEqual(@as(u32, 13), decodeChar("'\\r'").ok);
    try testing.expectEqual(@as(u32, 0), decodeChar("'\\0'").ok);
    try testing.expectEqual(@as(u32, 92), decodeChar("'\\\\'").ok);
    try testing.expectEqual(@as(u32, 39), decodeChar("'\\''").ok);
    try testing.expectEqual(@as(u32, 0x20AC), decodeChar("'\\u{20AC}'").ok);
    try testing.expectEqual(@as(u32, 65), decodeChar("'\\x41'").ok);
    // A raw multibyte source codepoint decodes to one scalar (`'€'`).
    try testing.expectEqual(@as(u32, 0x20AC), decodeChar("'\u{20AC}'").ok);

    // Malformed kinds.
    try testing.expectEqual(CharDecode.empty, decodeChar("''"));
    try testing.expectEqual(CharDecode.too_many, decodeChar("'ab'"));
    try testing.expectEqual(CharDecode.too_many, decodeChar("'\\n\\n'"));
    try testing.expectEqual(CharDecode.unknown_escape, decodeChar("'\\q'"));
    try testing.expectEqual(CharDecode.bad_hex_escape, decodeChar("'\\xZZ'"));
    try testing.expectEqual(CharDecode.bad_codepoint, decodeChar("'\\u{D800}'")); // surrogate
    try testing.expectEqual(CharDecode.bad_codepoint, decodeChar("'\\u{110000}'")); // > 0x10FFFF
}
