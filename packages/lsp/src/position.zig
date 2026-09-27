//! LSP positions <-> byte offsets <-> tokens: the coordinate plumbing every feature shares.

const std = @import("std");
const toyc = @import("toy_compiler");

const Token = toyc.Token;
const SourceMap = toyc.term.render.SourceMap;

/// The byte offset of an LSP position (`off = line_starts[line] + character`). An
/// out-of-range line yields null; a character past the line's end clamps to the end (which
/// `tokenAt` then reads as a gap -> null). u64 arithmetic so the `line + 1` / `line + 2`
/// index used to bracket the line cannot overflow.
pub fn positionToOffset(sm: *const SourceMap, line: u32, character: u32) ?u32 {
    const lc = sm.lineCount();
    if (line >= lc) return null;
    const ls: u64 = sm.lineStart(@as(usize, line) + 1);
    const le: u64 = if (@as(usize, line) + 1 < lc) sm.lineStart(@as(usize, line) + 2) else sm.bytes.len;
    const want: u64 = ls + character;
    return @intCast(if (want >= le) le else want);
}

/// LSP (line,character) -> byte offset within `text`, via a throwaway SourceMap. Null for
/// an out-of-range line; a character past the line end clamps to the line end. This is the
/// SAME basis the feature handlers use, so a spliced edit and a later hover/definition agree.
pub fn offsetIn(gpa: std.mem.Allocator, text: []const u8, line: u32, character: u32) !?u32 {
    var sm = try SourceMap.init(gpa, "d", text);
    defer sm.deinit(gpa);
    return positionToOffset(&sm, line, character);
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

    try testing.expectEqual(@as(?u32, 0), positionToOffset(&sm, 0, 0));
    try testing.expectEqual(@as(?u32, 1), positionToOffset(&sm, 0, 1));
    try testing.expectEqual(@as(?u32, 3), positionToOffset(&sm, 1, 0)); // start of "cde"
    try testing.expectEqual(@as(?u32, 5), positionToOffset(&sm, 1, 2));
    // A char past the line end clamps to the line's end (the trailing '\n' index).
    try testing.expectEqual(@as(?u32, 3), positionToOffset(&sm, 0, 50));
    // OOB line -> null.
    try testing.expectEqual(@as(?u32, null), positionToOffset(&sm, 99, 0));
}

test "positionToOffset is the exact inverse of lineCol" {
    const gpa = testing.allocator;
    const src = "fn f() -> int {\n  x := 1\n  return x\n}\n";
    var sm = try SourceMap.init(gpa, "t", src);
    defer sm.deinit(gpa);

    var off: u32 = 0;
    while (off <= src.len) : (off += 1) {
        const lc = sm.lineCol(off);
        const round = positionToOffset(&sm, @intCast(lc.line - 1), @intCast(lc.col - 1));
        try testing.expectEqual(@as(?u32, off), round);
    }
}
