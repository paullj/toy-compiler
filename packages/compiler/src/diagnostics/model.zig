//! The emit-side rich diagnostic MODEL: `Severity`, `Span`, `Label`, `Note`, and
//! the rich `Diagnostic` the Renderer draws. Lives on the emit side (not under
//! `term/render`) so the builder in `Sink.zig` can construct `Label`/`Note` slices
//! without a render->diagnostics import inversion. `term/render/Diagnostic.zig`
//! re-exports these types, so every existing `Rr.Diagnostic.*` call site is unchanged.
//!
//! Pure data: imports only `std` and the sink POD (for the `NO_SCOPE` sentinel it
//! pins `NO_SOURCE` to). Import ordering is ONE-WAY: `codes.zig` imports this file
//! (for `Severity`); this file imports NOTHING from `codes.zig`, avoiding a cycle.

const std = @import("std");
// The sink POD, imported from its home file (not via `Sink.zig`) so this leaf never
// pulls in the sink's ArrayList machinery. Only used to pin `NO_SOURCE == NO_SCOPE`.
const sink = @import("Diagnostic.zig");

/// Diagnostic severity, most-to-least severe. Append new severities at the end so
/// exhaustive switches break loudly. Spelled `err` (not the `error` keyword): the
/// user-facing word "error" is produced only by `Theme.word(.err)`.
pub const Severity = enum { err, warning, note, help };

/// A half-open byte range `[start, end)` into a single source. Intentionally
/// identical in shape to `SourceMap.Span` so the data model stays decoupled from it.
pub const Span = struct {
    start: u32,
    end: u32,

    /// True when the span covers no bytes (`start == end`) — a single caret point
    /// rather than an underline run.
    pub fn isZeroWidth(self: Span) bool {
        return self.start == self.end;
    }
};

/// Whether a label is the diagnostic's focus (`primary`, drawn with the caret /
/// primary underline) or supporting context (`secondary`, which recedes).
pub const LabelKind = enum { primary, secondary };

/// A span with an optional message pointing into one source. `message` and the
/// backing bytes are borrowed. An empty `message` means a bare underline (no text).
pub const Label = struct {
    kind: LabelKind,
    span: Span,
    /// May be "" for a bare underline with no text.
    message: []const u8 = "",
    /// Opaque source/scope id selecting which `SourceMap` the span indexes.
    /// `NO_SOURCE` == the single-source case (mirrors the sink's `NO_SCOPE`).
    source: u32 = NO_SOURCE,
};

/// A footer note kind: an informational `note` or an actionable `help`.
pub const NoteKind = enum { note, help };

/// A free-standing footer line (rustc's `= note:` / `= help:`). `message` borrowed.
pub const Note = struct {
    kind: NoteKind,
    message: []const u8,
};

/// The "untagged" source id: a single-source diagnostic carries no source id.
/// Pinned equal to `sink.NO_SCOPE` by the comptime assert below, so a single-file
/// `NO_SCOPE` lands on `NO_SOURCE` with no translation and no branch.
pub const NO_SOURCE: u32 = std.math.maxInt(u32);

/// The rich render-side diagnostic. `primary` is mandatory (always a focus);
/// `secondary`/`notes` default empty so the single-point case allocates nothing.
/// Slice fields borrowed. `code` is the resolved human string (e.g. "R0001"), null
/// when the diagnostic carries no code (`code == .none` in the sink POD).
pub const Diagnostic = struct {
    severity: Severity,
    /// Header message (the top line beside the severity word).
    message: []const u8,
    /// Focus label; `.kind == .primary` by construction.
    primary: Label,
    secondary: []const Label = &.{},
    notes: []const Note = &.{},
    /// Short code driving an `error[R0001]:` header; null => no bracket.
    code: ?[]const u8 = null,
    /// Opaque routing id carried from the sink; equals `primary.source`.
    scope: u32 = NO_SOURCE,
};

/// Build the rich render `Diagnostic` for a sink POD `d` at already-resolved effective
/// severity `eff` and resolved `code` string (null => no `[code]` bracket). The primary
/// covers `d.span()`; a set `d.related` becomes a secondary label reading
/// `related_label` over `d.relatedSpan()` in the same scope. `sec_buf` backs that
/// borrowed label, so it must outlive the render. Takes `code` and `related_label` as
/// params rather than calling `codes.zig` to keep this leaf's one-way import order.
pub fn richFromPod(d: sink.Diagnostic, eff: Severity, code: ?[]const u8, related_label: []const u8, sec_buf: *[1]Label) Diagnostic {
    const secondary: []const Label = if (d.relatedSpan()) |rs| blk: {
        sec_buf[0] = .{
            .kind = .secondary,
            .span = rs,
            .message = related_label,
            .source = d.scope,
        };
        break :blk sec_buf[0..1];
    } else &.{};
    return .{
        .severity = eff,
        .message = d.message,
        .primary = .{
            .kind = .primary,
            .span = d.span(),
            .message = d.message,
            .source = d.scope,
        },
        .secondary = secondary,
        .code = code,
        .scope = d.scope,
    };
}

// Pin the two "untagged" sentinels together. A single-file `NO_SCOPE` means
// `NO_SOURCE` with no translation table. Drift fails the build.
comptime {
    std.debug.assert(NO_SOURCE == sink.NO_SCOPE);
}

const testing = std.testing;

test "Span.isZeroWidth" {
    try testing.expect((Span{ .start = 3, .end = 3 }).isZeroWidth());
    try testing.expect(!(Span{ .start = 3, .end = 5 }).isZeroWidth());
}

test "a rich literal borrows secondary labels and notes with zero allocation" {
    const secondary = [_]Label{.{
        .kind = .secondary,
        .span = .{ .start = 10, .end = 14 },
        .message = "first defined here",
        .source = 2,
    }};
    const notes = [_]Note{
        .{ .kind = .note, .message = "expected because of this" },
        .{ .kind = .help, .message = "try adding a cast" },
    };
    const d = Diagnostic{
        .severity = .err,
        .message = "mismatched types",
        .primary = .{ .kind = .primary, .span = .{ .start = 4, .end = 8 }, .message = "here" },
        .secondary = &secondary,
        .notes = &notes,
        .code = "T0308",
    };
    try testing.expectEqual(@as(usize, 1), d.secondary.len);
    try testing.expectEqual(LabelKind.secondary, d.secondary[0].kind);
    try testing.expectEqual(@as(u32, 2), d.secondary[0].source);
    try testing.expectEqual(NoteKind.note, d.notes[0].kind);
    try testing.expectEqual(NoteKind.help, d.notes[1].kind);
    try testing.expectEqualStrings("T0308", d.code.?);
}

test "Diagnostic defaults: empty secondary/notes, null code, NO_SOURCE scope" {
    const d = Diagnostic{
        .severity = .warning,
        .message = "m",
        .primary = .{ .kind = .primary, .span = .{ .start = 0, .end = 3 } },
    };
    try testing.expectEqual(@as(usize, 0), d.secondary.len);
    try testing.expectEqual(@as(usize, 0), d.notes.len);
    try testing.expectEqual(@as(?[]const u8, null), d.code);
    try testing.expectEqual(NO_SOURCE, d.scope);
    try testing.expectEqualStrings("", d.primary.message);
    try testing.expectEqual(NO_SOURCE, d.primary.source);
}

test "a secondary Label constructs with its mandatory kind" {
    const l = Label{ .kind = .secondary, .span = .{ .start = 1, .end = 2 }, .message = "ctx", .source = 5 };
    try testing.expectEqual(LabelKind.secondary, l.kind);
    try testing.expectEqual(@as(u32, 5), l.source);
}

test "NO_SOURCE is pinned to the sink NO_SCOPE" {
    try testing.expectEqual(sink.NO_SCOPE, NO_SOURCE);
    try testing.expectEqual(std.math.maxInt(u32), NO_SOURCE);
}
