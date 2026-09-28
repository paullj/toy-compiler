//! LSP positions <-> byte offsets <-> tokens: the coordinate plumbing every feature shares.

const std = @import("std");
const toyc = @import("toy_compiler");

const Token = toyc.Token;
const SourceMap = toyc.term.render.SourceMap;
const protocol = @import("protocol.zig");

/// How an LSP `character` counts: UTF-8 bytes, or UTF-16 code units (the protocol's
/// default, and what browser editors count in). Negotiated at `initialize`.
pub const Encoding = enum {
    utf8,
    utf16,

    pub fn name(e: Encoding) []const u8 {
        return switch (e) {
            .utf8 => "utf-8",
            .utf16 => "utf-16",
        };
    }
};

/// The byte offset of an LSP position. An out-of-range line yields null; a character past
/// the line's end clamps to the end (which `tokenAt` then reads as a gap -> null), as does
/// a UTF-16 position inside a surrogate pair's first half.
pub fn positionToOffset(sm: *const SourceMap, line: u32, character: u32, enc: Encoding) ?u32 {
    const lc = sm.lineCount();
    if (line >= lc) return null;
    const ls: u32 = sm.lineStart(@as(usize, line) + 1);
    const le: u32 = if (@as(usize, line) + 1 < lc) sm.lineStart(@as(usize, line) + 2) else @intCast(sm.bytes.len);
    var off = ls;
    var units: u32 = 0;
    while (off < le and units < character) {
        const n = std.unicode.utf8ByteSequenceLength(sm.bytes[off]) catch 1;
        units += switch (enc) {
            .utf8 => n,
            .utf16 => if (n == 4) 2 else 1,
        };
        off = @min(le, off + n);
    }
    return off;
}

/// The LSP position of byte `off`: its 0-based line, and the `character` counted in `enc`
/// from the line's start.
pub fn offsetToPosition(sm: *const SourceMap, off: u32, enc: Encoding) protocol.Position {
    const lc = sm.lineCol(off);
    const line: u32 = @intCast(lc.line - 1);
    const ls = sm.lineStart(lc.line);
    const end = @min(off, @as(u32, @intCast(sm.bytes.len)));
    const character: u32 = switch (enc) {
        .utf8 => end - ls,
        .utf16 => blk: {
            var units: u32 = 0;
            var i = ls;
            while (i < end) {
                const n = std.unicode.utf8ByteSequenceLength(sm.bytes[i]) catch 1;
                units += if (n == 4) 2 else 1;
                i += n;
            }
            break :blk units;
        },
    };
    return .{ .line = line, .character = character };
}

/// The LSP range of the byte span `[start, end)`.
pub fn rangeOf(sm: *const SourceMap, start: u32, end: u32, enc: Encoding) protocol.Range {
    return .{ .start = offsetToPosition(sm, start, enc), .end = offsetToPosition(sm, end, enc) };
}

/// LSP (line,character) -> byte offset within `text`, via a throwaway SourceMap. The SAME
/// basis the feature handlers use, so a spliced edit and a later hover/definition agree.
pub fn offsetIn(gpa: std.mem.Allocator, text: []const u8, line: u32, character: u32, enc: Encoding) !?u32 {
    var sm = try SourceMap.init(gpa, "d", text);
    defer sm.deinit(gpa);
    return positionToOffset(&sm, line, character, enc);
}

/// The token whose half-open `[start, end)` contains `off`, or null for a gap / EOF. Binary
/// search over the start-sorted, non-overlapping token stream.
pub fn tokenAt(tokens: []const Token, off: u32) ?u32 {
    var lo: usize = 0;
    var hi: usize = tokens.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (tokens[mid].start <= off) lo = mid + 1 else hi = mid;
    }
    if (lo == 0) return null;
    const i = lo - 1;
    return if (off < tokens[i].end) @intCast(i) else null;
}

const testing = std.testing;

test "tokenAt: hit, gap, past-last-end, empty" {
    const toks = [_]Token{
        .{ .tag = .identifier, .start = 0, .end = 3 },
        .{ .tag = .identifier, .start = 5, .end = 8 },
    };
    try testing.expectEqual(@as(?u32, 0), tokenAt(&toks, 0));
    try testing.expectEqual(@as(?u32, 0), tokenAt(&toks, 2));
    try testing.expectEqual(@as(?u32, null), tokenAt(&toks, 3)); // gap: end is exclusive
    try testing.expectEqual(@as(?u32, null), tokenAt(&toks, 4)); // whitespace gap
    try testing.expectEqual(@as(?u32, 1), tokenAt(&toks, 7));
    try testing.expectEqual(@as(?u32, null), tokenAt(&toks, 8)); // past last end
    try testing.expectEqual(@as(?u32, null), tokenAt(&.{}, 0)); // empty stream
}

test "positionToOffset: basis, OOB line, char past EOL clamps" {
    const gpa = testing.allocator;
    const src = "ab\ncde\n";
    var sm = try SourceMap.init(gpa, "t", src);
    defer sm.deinit(gpa);

    try testing.expectEqual(@as(?u32, 0), positionToOffset(&sm, 0, 0, .utf8));
    try testing.expectEqual(@as(?u32, 1), positionToOffset(&sm, 0, 1, .utf8));
    try testing.expectEqual(@as(?u32, 3), positionToOffset(&sm, 1, 0, .utf8)); // start of "cde"
    try testing.expectEqual(@as(?u32, 5), positionToOffset(&sm, 1, 2, .utf8));
    // A char past the line end clamps to the line's end (the trailing '\n' index).
    try testing.expectEqual(@as(?u32, 3), positionToOffset(&sm, 0, 50, .utf8));
    // OOB line -> null.
    try testing.expectEqual(@as(?u32, null), positionToOffset(&sm, 99, 0, .utf8));
}

test "positionToOffset is the exact inverse of lineCol" {
    const gpa = testing.allocator;
    const src = "fn f() -> int {\n  x := 1\n  return x\n}\n";
    var sm = try SourceMap.init(gpa, "t", src);
    defer sm.deinit(gpa);

    var off: u32 = 0;
    while (off <= src.len) : (off += 1) {
        const lc = sm.lineCol(off);
        const round = positionToOffset(&sm, @intCast(lc.line - 1), @intCast(lc.col - 1), .utf8);
        try testing.expectEqual(@as(?u32, off), round);
    }
}

test "utf-16 positions count code units, and round-trip with byte offsets" {
    const gpa = testing.allocator;
    // `é` is 2 bytes / 1 unit; `𝄞` is 4 bytes / 2 units (a surrogate pair).
    const src = "a\u{e9}b \u{1d11e}c\nx\n";
    var sm = try SourceMap.init(gpa, "t", src);
    defer sm.deinit(gpa);
    const c_off: u32 = @intCast(std.mem.indexOfScalar(u8, src, 'c').?);
    try testing.expectEqual(@as(?u32, c_off), positionToOffset(&sm, 0, 6, .utf16));
    try testing.expectEqual(@as(?u32, c_off), positionToOffset(&sm, 0, 9, .utf8));
    try testing.expectEqual(protocol.Position{ .line = 0, .character = 6 }, offsetToPosition(&sm, c_off, .utf16));
    var off: u32 = 0;
    while (off <= src.len) : (off += 1) {
        if (off < src.len and (src[off] & 0xC0) == 0x80) continue; // not a char boundary
        for ([_]Encoding{ .utf8, .utf16 }) |enc| {
            const p = offsetToPosition(&sm, off, enc);
            try testing.expectEqual(@as(?u32, off), positionToOffset(&sm, p.line, p.character, enc));
        }
    }
}
