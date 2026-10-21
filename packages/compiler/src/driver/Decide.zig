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
/// read the stream, not the status). IO/CLI failures (exit 2) are handled by the caller
/// before reaching here.
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

/// True when `m` names a diagnostic override target: a known code string ("R0001")
/// or a single band letter (L/P/R/T/W). Used to validate --deny/--warn/--allow specs.
pub fn validSpec(m: []const u8) bool {
    return codes.fromStr(m) != null or (m.len == 1 and (m[0] == 'L' or m[0] == 'P' or m[0] == 'R' or m[0] == 'T' or m[0] == 'W'));
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

test "validSpec: codes, band letters, and rejects" {
    try std.testing.expect(validSpec("L"));
    try std.testing.expect(validSpec("P"));
    try std.testing.expect(validSpec("R"));
    try std.testing.expect(validSpec("T"));
    try std.testing.expect(validSpec("W"));
    try std.testing.expect(!validSpec("X"));
    try std.testing.expect(!validSpec(""));
    try std.testing.expect(!validSpec("ZZ999"));
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
