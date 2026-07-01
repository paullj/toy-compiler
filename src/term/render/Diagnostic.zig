//! Pretty-diagnostic data model (`Severity`, `Span`, `Label`, `Note`, `Diagnostic`)
//! plus `fromSink`, a read-only by-value adapter from the compiler's single-point
//! diagnostic. The render-side counterpart to the emit-side sink diagnostic.
//! - All slice fields are borrowed for the sink's lifetime; nothing here allocates.
//! - `fromSink` takes the sink diagnostic by value (can't mutate it) and returns
//!   a plain `Diagnostic` (no error union / no allocator = infallible + alloc-free).
//! - Pure data: imports only `std` and the sink type (the one sanctioned,
//!   read-only render->diagnostics cross-import); no Style/Theme/SourceMap.

const std = @import("std");
// The sink diagnostic type, imported from its home file (not via `Sink.zig`) so
// this leaf never pulls in the sink's ArrayList machinery. Read-only, by value.
const sink = @import("../../diagnostics/Diagnostic.zig");

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
    /// rather than an underline run. Every `fromSink` span is zero-width.
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
/// Pinned equal to `sink.NO_SCOPE` by the comptime assert below, so a
/// single-file `NO_SCOPE` lands on `NO_SOURCE` with no translation and no branch.
pub const NO_SOURCE: u32 = std.math.maxInt(u32);

/// The rich render-side diagnostic. `primary` is mandatory (always a focus, and
/// requiring it lets `fromSink` build one with no allocation); `secondary`/`notes`
/// default empty so the single-point case allocates nothing. Slice fields borrowed.
pub const Diagnostic = struct {
    severity: Severity,
    /// Header message (the top line beside the severity word).
    message: []const u8,
    /// Focus label; `.kind == .primary` by construction.
    primary: Label,
    secondary: []const Label = &.{},
    notes: []const Note = &.{},
    /// Short code driving an `error[E0001]:` header; `fromSink` -> null.
    code: ?[]const u8 = null,
    /// Opaque routing id carried from the sink; equals `primary.source`.
    scope: u32 = NO_SOURCE,
};

// Pin the two "untagged" sentinels together. `fromSink` forwards `scope` verbatim
// into `source`/`scope`, so this equality is what lets a single-file `NO_SCOPE`
// mean `NO_SOURCE` with no translation table. Drift fails the build.
comptime {
    std.debug.assert(NO_SOURCE == sink.NO_SCOPE);
}

/// Adapt the compiler's single-point diagnostic into the rich render model:
/// `.err` severity, borrowed message, a zero-width primary at `byte_offset`, and
/// `scope` forwarded verbatim (`NO_SCOPE` -> `NO_SOURCE`). Empty secondary/notes,
/// null code. By value + no allocator, so infallible and alloc-free.
pub fn fromSink(d: sink.Diagnostic) Diagnostic {
    return .{
        .severity = .err,
        .message = d.message, // borrowed: same .ptr, not duped
        .primary = .{
            .kind = .primary,
            .span = .{ .start = d.byte_offset, .end = d.byte_offset }, // zero-width point
            .message = d.message, // borrowed: aliases d.message
            .source = d.scope, // forwarded verbatim (NO_SCOPE -> NO_SOURCE)
        },
        // .secondary, .notes, .code take their struct defaults (empty / null).
        .scope = d.scope,
    };
}

const testing = std.testing;

test "fromSink synthesizes a zero-width primary error" {
    const d = fromSink(.{ .byte_offset = 42, .message = "boom", .scope = 7 });
    try testing.expectEqual(Severity.err, d.severity);
    try testing.expectEqual(LabelKind.primary, d.primary.kind);
    try testing.expectEqual(@as(u32, 42), d.primary.span.start);
    try testing.expectEqual(@as(u32, 42), d.primary.span.end);
    try testing.expect(d.primary.span.isZeroWidth());
    try testing.expectEqual(@as(u32, 7), d.primary.source);
    try testing.expectEqual(@as(usize, 0), d.secondary.len);
    try testing.expectEqual(@as(usize, 0), d.notes.len);
    try testing.expectEqual(@as(?[]const u8, null), d.code);
}

test "fromSink borrows the message (pointer identity, not a copy)" {
    const msg: []const u8 = "borrowed message";
    const d = fromSink(.{ .byte_offset = 3, .message = msg, .scope = 0 });
    // Same backing pointer on BOTH the header and the primary label.
    try testing.expectEqual(msg.ptr, d.message.ptr);
    try testing.expectEqual(msg.ptr, d.primary.message.ptr);
    try testing.expect(std.mem.eql(u8, d.message, msg));
}

test "fromSink forwards NO_SCOPE as NO_SOURCE" {
    // A default-scope sink diagnostic (single-file) -> NO_SOURCE everywhere.
    const d = fromSink(.{ .byte_offset = 1, .message = "m" });
    try testing.expectEqual(NO_SOURCE, d.primary.source);
    try testing.expectEqual(NO_SOURCE, d.scope);
    try testing.expectEqual(sink.NO_SCOPE, NO_SOURCE);
    try testing.expectEqual(std.math.maxInt(u32), NO_SOURCE);
}

test "fromSink takes the sink diagnostic BY VALUE" {
    // The first param's type is the sink struct itself (a value), never a pointer:
    // a pointer param would let it mutate the sink and would fail this gate.
    const P = @typeInfo(@TypeOf(fromSink)).@"fn".params[0].type.?;
    try testing.expect(P == sink.Diagnostic);
    try testing.expect(@typeInfo(P) == .@"struct");
}

test "sink diagnostic type is structurally unchanged (the 'never edited' gate)" {
    // If the sink diagnostic ever grows/renames a field, this fails the build,
    // proving fromSink's read-only contract still holds against the real type.
    try testing.expect(@hasField(sink.Diagnostic, "byte_offset"));
    try testing.expect(@hasField(sink.Diagnostic, "message"));
    try testing.expect(@hasField(sink.Diagnostic, "scope"));
    try testing.expectEqual(@as(usize, 3), @typeInfo(sink.Diagnostic).@"struct".fields.len);
    try testing.expect(@FieldType(sink.Diagnostic, "byte_offset") == u32);
    try testing.expect(@FieldType(sink.Diagnostic, "message") == []const u8);
    try testing.expect(@FieldType(sink.Diagnostic, "scope") == u32);
    try testing.expectEqual(std.math.maxInt(u32), sink.NO_SCOPE);
}

test "fromSink is infallible and alloc-free (no error union, no allocator)" {
    // Return type is exactly Diagnostic (not !Diagnostic), and the sole param is
    // the sink struct — no std.mem.Allocator anywhere. Compile-time proof.
    const info = @typeInfo(@TypeOf(fromSink)).@"fn";
    try testing.expect(info.return_type.? == Diagnostic);
    try testing.expectEqual(@as(usize, 1), info.params.len);
}

test "Span.isZeroWidth" {
    try testing.expect((Span{ .start = 3, .end = 3 }).isZeroWidth());
    try testing.expect(!(Span{ .start = 3, .end = 5 }).isZeroWidth());
}

test "a rich literal borrows secondary labels and notes with zero allocation" {
    const secondary = [_]Label{.{
        .kind = .secondary,
        .span = .{ .start = 10, .end = 14 },
        .message = "first defined here",
        .source = 2, // a different source than the primary
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
        .code = "E0308",
    };
    try testing.expectEqual(@as(usize, 1), d.secondary.len);
    try testing.expectEqual(LabelKind.secondary, d.secondary[0].kind);
    try testing.expectEqual(@as(u32, 2), d.secondary[0].source);
    try testing.expectEqual(NoteKind.note, d.notes[0].kind);
    try testing.expectEqual(NoteKind.help, d.notes[1].kind);
    try testing.expectEqualStrings("E0308", d.code.?);
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
    // Label message defaults to "" (bare underline) and source to NO_SOURCE.
    try testing.expectEqualStrings("", d.primary.message);
    try testing.expectEqual(NO_SOURCE, d.primary.source);
}

test "a secondary Label constructs with its mandatory kind" {
    // Label.kind is mandatory; a secondary label is spelled explicitly.
    const l = Label{ .kind = .secondary, .span = .{ .start = 1, .end = 2 }, .message = "ctx", .source = 5 };
    try testing.expectEqual(LabelKind.secondary, l.kind);
    try testing.expectEqual(@as(u32, 5), l.source);
}
