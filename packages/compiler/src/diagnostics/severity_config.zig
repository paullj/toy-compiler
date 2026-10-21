//! Render-time severity OVERRIDE policy. Pure, zero-alloc, borrowed.
//! Applied LATE at render (reads the POD default, never rewrites it), so the cached
//! []Diagnostic blob stays rule-set-independent. Empty config == identity, so
//! no-flag runs render byte-identical. NON-GOAL: exit codes (render-only).
//! One-way imports: codes + model.

const std = @import("std");
const codes = @import("codes.zig");
const model = @import("model.zig");

/// What a rule does to a matched code: promote to error, show as a warning, or drop.
/// `.warning` and `.ignore` never apply to a code whose default is an error; the CLI
/// also rejects a spec that names only errors (`codes.errorOnly`).
pub const Action = enum { err, warning, ignore };

/// One override: `match` is a full code string ("R0001") or a single band letter
/// ("L","P","R","T"). `match` bytes are borrowed (from argv), so a rule slice must
/// not outlive them.
pub const Rule = struct { match: []const u8, action: Action };

/// A borrowed, ordered list of override rules. Empty (the default) is the identity
/// policy: `resolve` returns the POD default verbatim for every code.
pub const SeverityConfig = struct { rules: []const Rule = &.{}, deny_warnings: bool = false };

/// True when `rule` names `code`: a 1-char match is a band letter (matches every code
/// in that band); otherwise an exact code string. `.none` has no string so never
/// matches (guards the uncoded sentinel via the `orelse`).
fn matches(rule: Rule, code: codes.Code) bool {
    const s = codes.str(code) orelse return false; // .none never matches
    if (rule.match.len == 1) return s[0] == rule.match[0]; // band prefix
    return std.mem.eql(u8, rule.match, s); // exact
}

/// Effective severity for `code`, or null when the last matching rule is `.ignore`
/// (the diagnostic is dropped -> zero rendered bytes). With no matching rule the POD
/// `default` is returned verbatim, so an empty config is byte-identical. Last matching
/// rule wins, making repeatable flags deterministic.
pub fn resolve(code: codes.Code, default: model.Severity, cfg: SeverityConfig) ?model.Severity {
    var eff: ?model.Severity = default;
    for (cfg.rules) |r| {
        if (!matches(r, code)) continue;
        // An error stops the build whatever it renders as, so weakening one would only
        // hide why. The CLI rejects error-only specs; this also covers a band that
        // holds both errors and warnings.
        if (default == .err and r.action != .err) continue;
        eff = switch (r.action) {
            .err => .err,
            .warning => .warning,
            .ignore => null,
        };
    }
    // -Werror: a surviving warning becomes an error. An `.ignore`d code is already
    // null here and stays null — deny never resurrects a dropped diagnostic.
    if (eff) |e| {
        if (cfg.deny_warnings and e == .warning) eff = .err;
    }
    return eff;
}

const testing = std.testing;

test "empty config is identity (byte-identity gate)" {
    const cfg = SeverityConfig{};
    try testing.expectEqual(model.Severity.err, resolve(.R0001, .err, cfg).?);
    try testing.expectEqual(model.Severity.warning, resolve(.T0005, .warning, cfg).?);
    try testing.expectEqual(model.Severity.err, resolve(.none, .err, cfg).?);
}

test "exact downgrades; band downgrades a whole band; other bands untouched" {
    const deny = [_]Rule{ .{ .match = "W", .action = .err }, .{ .match = "W0001", .action = .warning } };
    try testing.expectEqual(model.Severity.warning, resolve(.W0001, .warning, .{ .rules = &deny }).?);
    try testing.expectEqual(model.Severity.err, resolve(.W0002, .warning, .{ .rules = &deny }).?);
    const band = [_]Rule{ .{ .match = "W", .action = .err }, .{ .match = "W", .action = .warning } };
    try testing.expectEqual(model.Severity.warning, resolve(.W0001, .warning, .{ .rules = &band }).?);
    try testing.expectEqual(model.Severity.warning, resolve(.W0002, .warning, .{ .rules = &band }).?);
    try testing.expectEqual(model.Severity.warning, resolve(.T0005, .warning, .{ .rules = &.{.{ .match = "W", .action = .err }} }).?);
}

test "ignore returns null (drop); last-match-wins" {
    const ig = [_]Rule{.{ .match = "W0001", .action = .ignore }};
    try testing.expectEqual(@as(?model.Severity, null), resolve(.W0001, .warning, .{ .rules = &ig }));
    const chain = [_]Rule{ .{ .match = "W", .action = .ignore }, .{ .match = "W0001", .action = .err } };
    try testing.expectEqual(model.Severity.err, resolve(.W0001, .warning, .{ .rules = &chain }).?);
    try testing.expectEqual(@as(?model.Severity, null), resolve(.W0002, .warning, .{ .rules = &chain }));
}

test "an error-default code is never weakened, even by a band rule that also names warnings" {
    const rules = [_]Rule{ .{ .match = "R", .action = .warning }, .{ .match = "R0001", .action = .ignore } };
    try testing.expectEqual(model.Severity.err, resolve(.R0001, .err, .{ .rules = &rules }).?);
    try testing.expectEqual(model.Severity.err, resolve(.R0002, .err, .{ .rules = &rules }).?);
    try testing.expectEqual(model.Severity.warning, resolve(.T0005, .warning, .{ .rules = &.{.{ .match = "T", .action = .warning }} }).?);
    try testing.expectEqual(@as(?model.Severity, null), resolve(.T0005, .warning, .{ .rules = &.{.{ .match = "T", .action = .ignore }} }));
}

test "deny_warnings promotes a surviving warning but never resurrects an ignored one" {
    try testing.expectEqual(model.Severity.err, resolve(.W0001, .warning, .{ .deny_warnings = true }).?);
    try testing.expectEqual(model.Severity.err, resolve(.R0001, .err, .{ .deny_warnings = true }).?);
    const ig = [_]Rule{.{ .match = "W0001", .action = .ignore }};
    try testing.expectEqual(@as(?model.Severity, null), resolve(.W0001, .warning, .{ .rules = &ig, .deny_warnings = true }));
    const w = [_]Rule{.{ .match = "W0001", .action = .warning }};
    try testing.expectEqual(model.Severity.err, resolve(.W0001, .warning, .{ .rules = &w, .deny_warnings = true }).?);
}

test ".none never matches a band/exact rule" {
    const band = [_]Rule{.{ .match = "R", .action = .ignore }};
    try testing.expectEqual(model.Severity.err, resolve(.none, .err, .{ .rules = &band }).?);
    const exact = [_]Rule{.{ .match = "R0001", .action = .ignore }};
    try testing.expectEqual(model.Severity.err, resolve(.none, .err, .{ .rules = &exact }).?);
}
