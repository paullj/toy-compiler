//! `toy explain <CODE>` support: per-code documentation EMBEDDED at compile time via
//! `@embedFile` (no runtime filesystem dependency — a released binary and tests run
//! from the repo root behave identically). The doc for a code lives at
//! `src/diagnostics/errors/<CODE>.md`; the comptime `docs` table below pairs each registry code
//! with its embed. A meta-test asserts every non-`.none` code has a non-empty doc.

const std = @import("std");
const codes = @import("codes.zig");

/// One (code, embedded-doc) pair. `text` is the whole `.md` file, embedded at build
/// time, so it is always in the binary and never read from disk at runtime.
const Doc = struct { code: codes.Code, text: []const u8 };

/// Every registry code's embedded documentation, generated from `codes.table` so the
/// code list has ONE source of truth. Each code's doc lives at `errors/<str>.md`;
/// adding a code without its file fails the build at `@embedFile` (it can't find the
/// file), so no separate coverage assert is needed.
pub const docs = blk: {
    @setEvalBranchQuota(20_000);
    var out: [codes.table.len]Doc = undefined;
    for (codes.table, 0..) |e, i|
        out[i] = .{ .code = e.code, .text = @embedFile("errors/" ++ e.str ++ ".md") };
    break :blk out;
};

/// The embedded doc for a code, or null if the code has no doc (only `.none`).
pub fn docFor(c: codes.Code) ?[]const u8 {
    for (docs) |d| if (d.code == c) return d.text;
    return null;
}

/// Look up a doc by its human code string ("R0001"), or null if the string is not a
/// known code. Used by the `explain` subcommand.
pub fn docForStr(s: []const u8) ?[]const u8 {
    const c = codes.fromStr(s) orelse return null;
    return docFor(c);
}

const testing = std.testing;

test "every non-none registry code has a non-empty embedded doc" {
    for (codes.table) |e| {
        const doc = docFor(e.code) orelse return error.TestUnexpectedResult;
        try testing.expect(doc.len > 0);
    }
    try testing.expectEqual(@as(?[]const u8, null), docFor(.none));
}

test "docForStr resolves a known code and rejects garbage" {
    const r1 = docForStr("R0001") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, r1, "undeclared identifier") != null);
    try testing.expectEqual(@as(?[]const u8, null), docForStr("Z9999"));

    const p1 = docForStr("P0001") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, p1, "P0001") != null);
    try testing.expect(std.mem.indexOf(u8, p1, "expected") != null);

    const w1 = docForStr("W0001") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, w1, "unused") != null);

    const w2 = docForStr("W0002") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, w2, "unused parameter") != null);

    const w3 = docForStr("W0003") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, w3, "unused function") != null);

    const w4 = docForStr("W0004") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, w4, "unused import") != null);

    const w5 = docForStr("W0005") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, w5, "unreachable") != null);

    const w6 = docForStr("W0006") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, w6, "unreachable match arm") != null);

    const w7 = docForStr("W0007") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, w7, "constant condition") != null);

    const w8 = docForStr("W0008") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, w8, "unused") != null);

    const w9 = docForStr("W0009") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, w9, "shadow") != null);

    const w10 = docForStr("W0010") orelse return error.TestUnexpectedResult;
    try testing.expect(std.mem.indexOf(u8, w10, "dead store") != null);
}
