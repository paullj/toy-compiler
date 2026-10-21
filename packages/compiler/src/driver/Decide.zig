//! Pure CLI decision policy: the exit-code, argument-validation, and path-derivation
//! helpers the `toy` entrypoint (`driver/main.zig`) consults. Split out of the exe
//! module so the correctness-critical parts — chiefly the check exit-code contract —
//! are library-visible and unit-testable: `test-bin` never analyzes exe entrypoints,
//! so any logic that must be covered lives here (mirroring `Check.zig`).
//!
//! Every function here is pure over its arguments (no `Io`, no allocation, no globals),
//! which is what makes the tests below independent of a running process.

const std = @import("std");
const Check = @import("Check.zig");
const codes = @import("../diagnostics/codes.zig");
const Diagnostic = @import("../diagnostics/Sink.zig").Diagnostic;

/// Map a `check` tally to the process exit code: 0 clean (or `--exit-zero`), 1 when at
/// least one diagnostic resolved to an error. `--deny-warnings` (-Werror) is applied
/// UPSTREAM in `SevCfg.resolve`, so a promoted warning already counts as an error in the
/// tally here — there is no per-warning branch. `--exit-zero` DOMINATES (editors that
/// read the stream, not the status). Usage and IO failures (exit 2) are handled by the
/// caller before reaching here.
pub fn checkExit(counts: Check.Counts, exit_zero: bool) u8 {
    if (exit_zero) return 0;
    if (counts.hasErrors()) return 1;
    return 0;
}

/// True when any diagnostic in `diags` carries an error-severity REGISTRY DEFAULT — the
/// gate that decides whether the graph is resolved enough to typecheck. Deliberately
/// reads the POD default (NOT the render-time config): the ability to typecheck
/// depends on whether resolution actually succeeded, which a severity rule (a
/// presentation choice) must never change.
pub fn resolveHasError(diags: []const Diagnostic) bool {
    for (diags) |d| if (d.severity == .err) return true;
    return false;
}

/// The registry spelling of a --deny/--warn/--allow spec: a known code ("r0001" gives
/// "R0001") or a band letter of `codes.prefix_bands`, or null when it names neither.
/// The result is static, so a severity rule never borrows the user's spelling.
pub fn canonicalSpec(m: []const u8) ?[]const u8 {
    if (codes.fromStr(m)) |c| return codes.str(c);
    return codes.bandOf(m);
}

/// The bands a severity flag takes, as "A|B": every band, or with `warnings_only`
/// (--warn/--allow) only the bands that hold a warning.
pub fn sevBands(comptime warnings_only: bool) []const u8 {
    comptime {
        @setEvalBranchQuota(20_000);
        var list: []const u8 = "";
        for (codes.prefix_bands) |b| {
            if (warnings_only and codes.errorOnly(&.{b})) continue;
            list = list ++ (if (list.len == 0) "" else "|") ++ .{b};
        }
        return list;
    }
}

/// What a severity flag's value must be, for its bad-value error.
pub fn sevExpected(comptime warnings_only: bool) []const u8 {
    comptime {
        const bands = sevBands(warnings_only);
        const code = if (warnings_only) "a warning code like W0001" else "a code like R0001";
        const one_of = if (std.mem.indexOfScalar(u8, bands, '|') != null) "one of " else "";
        return code ++ ", or a band: " ++ one_of ++ bands;
    }
}

/// The basename of a path (after the last '/'), used as the code-signing
/// identifier. Falls back to the whole string when there is no separator.
pub fn basename(path: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| return path[i + 1 ..];
    return path;
}

/// Strip a trailing `.toy` source extension for the default output binary name.
pub fn stemOf(name: []const u8) []const u8 {
    if (std.mem.endsWith(u8, name, ".toy")) return name[0 .. name.len - ".toy".len];
    return name;
}

test "checkExit: clean tally exits 0" {
    try std.testing.expectEqual(@as(u8, 0), checkExit(.{}, false));
    try std.testing.expectEqual(@as(u8, 0), checkExit(.{ .warnings = 3 }, false));
}

test "checkExit: any error exits 1" {
    try std.testing.expectEqual(@as(u8, 1), checkExit(.{ .errors = 1 }, false));
    try std.testing.expectEqual(@as(u8, 1), checkExit(.{ .errors = 2, .warnings = 5 }, false));
}

test "checkExit: --exit-zero dominates errors and warnings" {
    try std.testing.expectEqual(@as(u8, 0), checkExit(.{ .errors = 9 }, true));
    try std.testing.expectEqual(@as(u8, 0), checkExit(.{ .warnings = 9 }, true));
}

test "resolveHasError: only error severity blocks typecheck" {
    const err: Diagnostic = .{ .severity = .err, .message = "", .byte_offset = 0, .end = 0 };
    const warn: Diagnostic = .{ .severity = .warning, .message = "", .byte_offset = 0, .end = 0 };
    try std.testing.expect(!resolveHasError(&.{}));
    try std.testing.expect(!resolveHasError(&.{warn}));
    try std.testing.expect(resolveHasError(&.{ warn, err }));
}

test "canonicalSpec: bands come from codes, case is ignored, the rest is rejected" {
    for (codes.prefix_bands) |b| {
        try std.testing.expectEqualStrings(&.{b}, canonicalSpec(&.{b}).?);
        try std.testing.expectEqualStrings(&.{b}, canonicalSpec(&.{std.ascii.toLower(b)}).?);
    }
    try std.testing.expectEqualStrings("W0001", canonicalSpec("w0001").?);
    try std.testing.expectEqualStrings("R0001", canonicalSpec("R0001").?);
    try std.testing.expectEqual(@as(?[]const u8, null), canonicalSpec("Q"));
    try std.testing.expectEqual(@as(?[]const u8, null), canonicalSpec(""));
    try std.testing.expectEqual(@as(?[]const u8, null), canonicalSpec("ZZ999"));
}

test "sevBands: every band, or only the bands holding a warning" {
    try std.testing.expectEqualStrings("L|P|R|T|W", comptime sevBands(false));
    try std.testing.expectEqualStrings("W", comptime sevBands(true));
    try std.testing.expectEqualStrings("a code like R0001, or a band: one of L|P|R|T|W", comptime sevExpected(false));
    try std.testing.expectEqualStrings("a warning code like W0001, or a band: W", comptime sevExpected(true));
}

test "basename: with and without a separator" {
    try std.testing.expectEqualStrings("prog", basename("/tmp/build/prog"));
    try std.testing.expectEqualStrings("prog", basename("prog"));
    try std.testing.expectEqualStrings("", basename("trailing/"));
}

test "stemOf: strips only a trailing .toy" {
    try std.testing.expectEqualStrings("main", stemOf("main.toy"));
    try std.testing.expectEqualStrings("main", stemOf("main"));
    try std.testing.expectEqualStrings("a.toy.bak", stemOf("a.toy.bak"));
}
