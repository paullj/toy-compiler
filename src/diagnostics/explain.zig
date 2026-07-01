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

/// Every registry code's embedded documentation. Kept in lock-step with
/// `codes.table` by the comptime coverage assert below — adding a code without its
/// `src/diagnostics/errors/<CODE>.md` fails the build (`@embedFile` can't find the file).
pub const docs = [_]Doc{
    .{ .code = .R0001, .text = @embedFile("errors/R0001.md") },
    .{ .code = .R0002, .text = @embedFile("errors/R0002.md") },
    .{ .code = .R0003, .text = @embedFile("errors/R0003.md") },
    .{ .code = .R0004, .text = @embedFile("errors/R0004.md") },
    .{ .code = .R0005, .text = @embedFile("errors/R0005.md") },
    .{ .code = .R0006, .text = @embedFile("errors/R0006.md") },
    .{ .code = .R0007, .text = @embedFile("errors/R0007.md") },
    .{ .code = .R0008, .text = @embedFile("errors/R0008.md") },
    .{ .code = .R0009, .text = @embedFile("errors/R0009.md") },
    .{ .code = .T0001, .text = @embedFile("errors/T0001.md") },
    .{ .code = .T0002, .text = @embedFile("errors/T0002.md") },
    .{ .code = .T0003, .text = @embedFile("errors/T0003.md") },
    .{ .code = .T0004, .text = @embedFile("errors/T0004.md") },
    .{ .code = .T0005, .text = @embedFile("errors/T0005.md") },
    .{ .code = .T0006, .text = @embedFile("errors/T0006.md") },
    .{ .code = .T0007, .text = @embedFile("errors/T0007.md") },
    .{ .code = .T0008, .text = @embedFile("errors/T0008.md") },
    .{ .code = .T0009, .text = @embedFile("errors/T0009.md") },
    .{ .code = .T0010, .text = @embedFile("errors/T0010.md") },
    .{ .code = .T0011, .text = @embedFile("errors/T0011.md") },
    .{ .code = .T0012, .text = @embedFile("errors/T0012.md") },
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

// Coverage: every non-`.none` registry code MUST have a doc row here.
comptime {
    @setEvalBranchQuota(20_000);
    for (codes.table) |e| {
        var seen = false;
        for (docs) |d| if (d.code == e.code) {
            seen = true;
        };
        if (!seen) @compileError("code " ++ e.str ++ " has no docs/errors/<CODE>.md doc row in explain.zig");
    }
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
}
