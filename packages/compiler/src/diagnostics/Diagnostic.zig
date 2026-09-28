//! The one diagnostic type shared by every stage. `byte_offset` points at the
//! offending token; the driver/CLI renders byte_offset -> line:col uniformly.
//! `scope` carries the owning module id in graph mode (`NO_SCOPE` single-file).

const std = @import("std");
const codes = @import("codes.zig");
const model = @import("model.zig");
const Token = @import("../ast/Token.zig").Token;

/// The "untagged" scope: a single-file diagnostic carries no module id.
pub const NO_SCOPE: u32 = std.math.maxInt(u32);

/// Sentinel for `Diagnostic.related`: the diagnostic has no related prior location.
pub const NO_RELATED: u32 = std.math.maxInt(u32);

/// Sentinel for `Diagnostic.end` / `related_end`: no explicit extent was emitted, so
/// `fillSpans` gives the span the extent of the token it starts at.
pub const NO_END: u32 = std.math.maxInt(u32);

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
    /// Exclusive end byte of the primary span. An emit site that knows the offending
    /// node's extent sets it; otherwise `fillSpans` does, from the token at `byte_offset`.
    end: u32 = NO_END,
    /// Exclusive end byte of the related span, filled like `end`.
    related_end: u32 = NO_END,

    /// The primary span `[byte_offset, end)`; empty before `fillSpans` has run.
    pub fn span(d: Diagnostic) model.Span {
        return .{ .start = d.byte_offset, .end = if (d.end == NO_END) d.byte_offset else d.end };
    }

    /// The related span, or null when there is no related location.
    pub fn relatedSpan(d: Diagnostic) ?model.Span {
        if (d.related == NO_RELATED) return null;
        return .{ .start = d.related, .end = if (d.related_end == NO_END) d.related else d.related_end };
    }
};

/// Give every diagnostic in `diags` without an explicit extent the extent of the token its
/// offset starts at. `tokensOf(scope)` returns the owning module's tokens. Run where a stage
/// hands its diagnostics out, so every renderer (CLI carets, NDJSON, the LSP) sees the same
/// spans and none has to re-derive them.
pub fn fillSpans(diags: []Diagnostic, ctx: anytype, comptime tokensOf: fn (@TypeOf(ctx), u32) []const Token) void {
    for (diags) |*d| {
        const tokens = tokensOf(ctx, d.scope);
        if (d.end == NO_END) d.end = tokenEnd(tokens, d.byte_offset);
        if (d.related != NO_RELATED and d.related_end == NO_END) d.related_end = tokenEnd(tokens, d.related);
    }
}

/// `fillSpans` for diagnostics that all belong to one module.
pub fn fillModuleSpans(diags: []Diagnostic, tokens: []const Token) void {
    const One = struct {
        fn tokensOf(t: []const Token, _: u32) []const Token {
            return t;
        }
    };
    fillSpans(diags, tokens, One.tokensOf);
}

/// The end of the token containing `off`, or `off` itself off any token (a gap, EOF).
pub fn tokenEnd(tokens: []const Token, off: u32) u32 {
    var lo: usize = 0;
    var hi: usize = tokens.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (tokens[mid].start <= off) lo = mid + 1 else hi = mid;
    }
    if (lo == 0) return off;
    const t = tokens[lo - 1];
    return if (off < t.end) t.end else off;
}

const testing = std.testing;

test "the sink POD stays memcpy-trivial: no field beyond `message` is a slice/pointer (cache-stability gate)" {
    // This is the load-bearing cache-stability gate: the Engine caches a
    // `[]const Diagnostic` blob by memcpy, so no field may be a slice/pointer beyond
    // the already-borrowed `message`.
    try testing.expectEqual(@as(usize, 8), @typeInfo(Diagnostic).@"struct".fields.len);
    try testing.expect(@FieldType(Diagnostic, "byte_offset") == u32);
    try testing.expect(@FieldType(Diagnostic, "message") == []const u8);
    try testing.expect(@FieldType(Diagnostic, "scope") == u32);
    try testing.expect(@FieldType(Diagnostic, "code") == codes.Code);
    try testing.expect(@FieldType(Diagnostic, "severity") == model.Severity);
    try testing.expect(@FieldType(Diagnostic, "related") == u32);
    try testing.expect(@FieldType(Diagnostic, "end") == u32);
    try testing.expect(@FieldType(Diagnostic, "related_end") == u32);
    try testing.expect(@typeInfo(@FieldType(Diagnostic, "code")) == .@"enum");
    try testing.expect(@typeInfo(@FieldType(Diagnostic, "severity")) == .@"enum");
    // Defaults keep every existing `.{ .byte_offset = x, .message = m }` literal valid.
    const d: Diagnostic = .{ .byte_offset = 1, .message = "m" };
    try testing.expectEqual(codes.Code.none, d.code);
    try testing.expectEqual(model.Severity.err, d.severity);
    try testing.expectEqual(NO_SCOPE, d.scope);
    try testing.expectEqual(NO_RELATED, d.related);
}

test "fillSpans gives an unspanned diagnostic its token's extent and keeps an explicit one" {
    const toks = [_]Token{
        .{ .tag = .identifier, .start = 0, .end = 3 },
        .{ .tag = .identifier, .start = 4, .end = 9 },
    };
    var diags = [_]Diagnostic{
        .{ .byte_offset = 4, .message = "token" },
        .{ .byte_offset = 0, .message = "explicit", .end = 9 },
        .{ .byte_offset = 3, .message = "gap" },
        .{ .byte_offset = 4, .message = "related", .related = 0 },
    };
    fillModuleSpans(&diags, &toks);
    try testing.expectEqual(model.Span{ .start = 4, .end = 9 }, diags[0].span());
    try testing.expectEqual(model.Span{ .start = 0, .end = 9 }, diags[1].span());
    try testing.expectEqual(model.Span{ .start = 3, .end = 3 }, diags[2].span());
    try testing.expectEqual(model.Span{ .start = 0, .end = 3 }, diags[3].relatedSpan().?);
}
