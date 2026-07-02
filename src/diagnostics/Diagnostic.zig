//! The one diagnostic type shared by every stage. `byte_offset` points at the
//! offending token; the driver/CLI renders byte_offset -> line:col uniformly.
//! `scope` carries the owning module id in graph mode (`NO_SCOPE` single-file).

const std = @import("std");
const codes = @import("codes.zig");
const model = @import("model.zig");

/// The "untagged" scope: a single-file diagnostic carries no module id.
pub const NO_SCOPE: u32 = std.math.maxInt(u32);

/// Sentinel for `Diagnostic.related`: the diagnostic has no related prior location.
pub const NO_RELATED: u32 = std.math.maxInt(u32);

/// The one diagnostic type shared by every stage. `byte_offset` points at the
/// offending token; the driver/CLI renders byte_offset -> line:col uniformly.
/// `scope` is the owning module id in graph mode, `NO_SCOPE` single-file.
///
/// `code`/`severity` are BOTH trivially-copyable enums, so this stays memcpy-trivial
/// POD (the Engine caches `[]const Diagnostic` as a raw blob). `code` is the stable
/// identity from `codes.zig`; `severity` is the registry DEFAULT (render-time config
/// overrides it late, never rewriting the stored value, so the cached blob stays
/// rule-set-independent). Every default keeps existing
/// `.{ .byte_offset = x, .message = m }` literals compiling unchanged.
pub const Diagnostic = struct {
    byte_offset: u32,
    message: []const u8,
    scope: u32 = NO_SCOPE,
    /// Stable identity (enum ordinal, cache-safe). `.none` => no `[code]` bracket.
    code: codes.Code = .none,
    /// Registry default severity; overridden LATE at render, never here.
    severity: model.Severity = .err,
    /// Optional byte offset of a RELATED prior location in the SAME scope (e.g. the
    /// first definition for a duplicate), rendered as a secondary "previously defined
    /// here" label. `NO_RELATED` => none. A memcpy-trivial `u32`, so the POD stays
    /// cache-safe; it is NOT part of the sort/dedup key.
    related: u32 = NO_RELATED,
};

const testing = std.testing;

test "the sink POD stays memcpy-trivial: no field beyond `message` is a slice/pointer (cache-stability gate)" {
    // This is the load-bearing cache-stability gate: the Engine caches a
    // `[]const Diagnostic` blob by memcpy, so no field may be a slice/pointer beyond
    // the already-borrowed `message`. C1 grew the POD to FIVE fields (code+severity,
    // both enums); C3 added a SIXTH, `related`, a plain `u32` — still trivially copyable.
    try testing.expectEqual(@as(usize, 6), @typeInfo(Diagnostic).@"struct".fields.len);
    try testing.expect(@FieldType(Diagnostic, "byte_offset") == u32);
    try testing.expect(@FieldType(Diagnostic, "message") == []const u8);
    try testing.expect(@FieldType(Diagnostic, "scope") == u32);
    try testing.expect(@FieldType(Diagnostic, "code") == codes.Code);
    try testing.expect(@FieldType(Diagnostic, "severity") == model.Severity);
    try testing.expect(@FieldType(Diagnostic, "related") == u32);
    try testing.expect(@typeInfo(@FieldType(Diagnostic, "code")) == .@"enum");
    try testing.expect(@typeInfo(@FieldType(Diagnostic, "severity")) == .@"enum");
    // Defaults keep every existing `.{ .byte_offset = x, .message = m }` literal valid.
    const d: Diagnostic = .{ .byte_offset = 1, .message = "m" };
    try testing.expectEqual(codes.Code.none, d.code);
    try testing.expectEqual(model.Severity.err, d.severity);
    try testing.expectEqual(NO_SCOPE, d.scope);
    try testing.expectEqual(NO_RELATED, d.related);
}
