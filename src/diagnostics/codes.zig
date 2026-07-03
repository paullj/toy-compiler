//! The append-only diagnostic CODE registry. A `Code` is a trivially-copyable
//! `enum(u16)` stored INSIDE the sink POD (so the cached `[]Diagnostic` blob stays
//! memcpy-trivial and rule-set-independent); the human string ("R0001"), slug, and
//! default severity live in the comptime `table` looked up at render time — NOT in
//! the diagnostic. Severity is a registry DEFAULT here; render-time config overrides
//! it late, so identity (the code) is never conflated with presentation (severity).
//!
//! Prefix bands: `L####`=lex, `P####`=parse, `R####`=resolve, `T####`=type. The
//! comptime block below asserts code uniqueness, prefix membership, and per-band
//! contiguity (numeric tails run from 0001). Import ordering is ONE-WAY: this file
//! imports `model.zig` for `Severity`; `model.zig` imports nothing from here.

const std = @import("std");
const model = @import("model.zig");
pub const Severity = model.Severity;

/// Stability of a code (for a future `explain`/lint policy). Append-only.
pub const Status = enum { stable, preview, deprecated, removed };
/// Whether a diagnostic has an auto-fix (reserved for a future fix engine).
pub const Fixable = enum { no, safe, unsafe };

/// The append-only code enum. `none` (ordinal 0) is the "uncoded" sentinel every
/// existing emit site defaults to; it renders with NO `[code]` bracket, byte-identical
/// to pre-registry output. Non-exhaustive (`_`) so adding a code never breaks a switch.
pub const Code = enum(u16) {
    none = 0,

    // Parse band (P####).
    P0001, // expected-token       (missing punctuation/keyword: every expect() site)
    P0002, // expected-expression  (expression/argument position empty)
    P0003, // expected-declaration (top-level / after-`pub` decl dispatch)
    P0004, // expected-terminator  ("expected a newline or '}' after statement")
    P0005, // nesting-too-deep     (block/expression depth backstop — SIGBUS guard)
    P0006, // expected-name        (identifier: field/param/type/variant/label/loop-var/alias/path-seg)
    P0007, // invalid-label-target ("a label must prefix a loop, while, for, or block")
    P0008, // expected-pattern     (match sub-pattern / variant pattern)

    // Resolve band (R####).
    R0001, // undeclared-identifier
    R0002, // duplicate-function
    R0003, // unknown-imported-module
    R0004, // import-namespace-collision
    R0005, // not-exported
    R0006, // no-such-member
    R0007, // invalid-assignment-target
    R0008, // duplicate-label
    R0009, // undefined-label

    // Type band (T####).
    T0001, // unknown-type
    T0002, // unknown-module
    T0003, // module-has-no-type
    T0004, // recursive-type
    T0005, // empty-struct
    T0006, // empty-enum
    T0007, // unit-field
    T0008, // unit-payload
    T0009, // pub-exposes-non-pub
    T0010, // main-return-type
    T0011, // shadows-builtin
    T0012, // duplicate-struct
    T0013, // generics-unsupported
    T0014, // mono-depth (monomorphization instance/depth limit — M2 belt-and-suspenders)
    T0015, // type-arg-inference-conflict (a type-var bound to two different concrete types — M3)
    T0016, // type-args-not-inferable (a generic type-param left unbound after matching — M3)
    T0017, // instantiation-too-deep (unbounded generic-data instantiation depth — M4)
    T0018, // no-such-method (a method call names a receiver type that has no such method — M8)

    _,
};

/// One registry row. `str` is the human code; `slug` is the kebab-case identity used
/// for docs/fixture filenames; `default_severity` seeds the sink POD (render config
/// overrides it late). `redirect_to`/`fixable`/`help_slug` are reserved for later.
const Entry = struct {
    code: Code,
    str: []const u8,
    slug: []const u8,
    default_severity: Severity = .err,
    status: Status = .stable,
    redirect_to: Code = .none,
    fixable: Fixable = .no,
    help_slug: []const u8 = "",
};

/// The append-only registry. Every non-`.none` `Code` MUST have exactly one row (the
/// comptime block below enforces coverage, uniqueness, prefix, and contiguity).
pub const table = [_]Entry{
    .{ .code = .P0001, .str = "P0001", .slug = "expected-token" },
    .{ .code = .P0002, .str = "P0002", .slug = "expected-expression" },
    .{ .code = .P0003, .str = "P0003", .slug = "expected-declaration" },
    .{ .code = .P0004, .str = "P0004", .slug = "expected-terminator" },
    .{ .code = .P0005, .str = "P0005", .slug = "nesting-too-deep" },
    .{ .code = .P0006, .str = "P0006", .slug = "expected-name" },
    .{ .code = .P0007, .str = "P0007", .slug = "invalid-label-target" },
    .{ .code = .P0008, .str = "P0008", .slug = "expected-pattern" },
    .{ .code = .R0001, .str = "R0001", .slug = "undeclared-identifier" },
    .{ .code = .R0002, .str = "R0002", .slug = "duplicate-function" },
    .{ .code = .R0003, .str = "R0003", .slug = "unknown-imported-module" },
    .{ .code = .R0004, .str = "R0004", .slug = "import-namespace-collision" },
    .{ .code = .R0005, .str = "R0005", .slug = "not-exported" },
    .{ .code = .R0006, .str = "R0006", .slug = "no-such-member" },
    .{ .code = .R0007, .str = "R0007", .slug = "invalid-assignment-target" },
    .{ .code = .R0008, .str = "R0008", .slug = "duplicate-label" },
    .{ .code = .R0009, .str = "R0009", .slug = "undefined-label" },
    .{ .code = .T0001, .str = "T0001", .slug = "unknown-type" },
    .{ .code = .T0002, .str = "T0002", .slug = "unknown-module" },
    .{ .code = .T0003, .str = "T0003", .slug = "module-has-no-type" },
    .{ .code = .T0004, .str = "T0004", .slug = "recursive-type" },
    .{ .code = .T0005, .str = "T0005", .slug = "empty-struct" },
    .{ .code = .T0006, .str = "T0006", .slug = "empty-enum" },
    .{ .code = .T0007, .str = "T0007", .slug = "unit-field" },
    .{ .code = .T0008, .str = "T0008", .slug = "unit-payload" },
    .{ .code = .T0009, .str = "T0009", .slug = "pub-exposes-non-pub" },
    .{ .code = .T0010, .str = "T0010", .slug = "main-return-type" },
    .{ .code = .T0011, .str = "T0011", .slug = "shadows-builtin" },
    .{ .code = .T0012, .str = "T0012", .slug = "duplicate-struct" },
    .{ .code = .T0013, .str = "T0013", .slug = "generics-unsupported" },
    .{ .code = .T0014, .str = "T0014", .slug = "mono-depth" },
    .{ .code = .T0015, .str = "T0015", .slug = "type-arg-inference-conflict" },
    .{ .code = .T0016, .str = "T0016", .slug = "type-args-not-inferable" },
    .{ .code = .T0017, .str = "T0017", .slug = "instantiation-too-deep" },
    .{ .code = .T0018, .str = "T0018", .slug = "no-such-method" },
};

/// The human code string ("R0001") or null for `.none` (=> no `[code]` bracket, so
/// existing uncoded diagnostics render byte-identical). `unreachable` on an unlisted
/// code is impossible because the comptime coverage assert proves every code is in
/// the table.
pub fn str(c: Code) ?[]const u8 {
    if (c == .none) return null;
    for (table) |e| if (e.code == c) return e.str;
    unreachable;
}

/// The registry default severity for a code (`.err` for `.none`). Config overrides
/// this LATE at render time; nothing writes an overridden severity back into the POD.
pub fn defaultSeverity(c: Code) Severity {
    if (c == .none) return .err;
    for (table) |e| if (e.code == c) return e.default_severity;
    unreachable;
}

/// The kebab-case slug for a code, or null for `.none`.
pub fn slug(c: Code) ?[]const u8 {
    if (c == .none) return null;
    for (table) |e| if (e.code == c) return e.slug;
    unreachable;
}

/// Parse a human code string ("R0001") back to a `Code`, or null if unknown. Used by
/// `toy explain <CODE>` and the render-time severity config (`--error R0001`).
pub fn fromStr(s: []const u8) ?Code {
    for (table) |e| if (std.mem.eql(u8, e.str, s)) return e.code;
    return null;
}

/// True when `s` (a code string) has the given single-letter band prefix ('R','T',…).
pub fn hasPrefix(s: []const u8, band: u8) bool {
    return s.len > 0 and s[0] == band;
}

const prefix_bands = "LPRT";

// Build-time registry safety: coverage (every non-none code has exactly one row),
// uniqueness, prefix membership, and per-band contiguity (tails run 0001,0002,…).
comptime {
    @setEvalBranchQuota(20_000);
    // Coverage + uniqueness: every enum value except `none` and the `_` sentinel has
    // exactly one table row.
    for (@typeInfo(Code).@"enum".fields) |f| {
        if (std.mem.eql(u8, f.name, "none")) continue;
        var seen: usize = 0;
        for (table) |e| {
            if (std.mem.eql(u8, @tagName(e.code), f.name)) seen += 1;
        }
        if (seen == 0) @compileError("code " ++ f.name ++ " has no registry row");
        if (seen > 1) @compileError("code " ++ f.name ++ " has duplicate registry rows");
    }
    // str==tag name; prefix membership; slug non-empty; no duplicate str.
    for (table, 0..) |e, i| {
        if (!std.mem.eql(u8, e.str, @tagName(e.code)))
            @compileError("code str must equal its tag name: " ++ e.str);
        if (e.slug.len == 0) @compileError("code " ++ e.str ++ " has an empty slug");
        var band_ok = false;
        for (prefix_bands) |b| if (e.str[0] == b) {
            band_ok = true;
        };
        if (!band_ok) @compileError("code " ++ e.str ++ " has an out-of-band prefix");
        for (table[i + 1 ..]) |o| {
            if (e.code == o.code) @compileError("duplicate code enum in table: " ++ e.str);
            if (std.mem.eql(u8, e.str, o.str)) @compileError("duplicate code str in table: " ++ e.str);
        }
    }
    // Per-band contiguity: for each band, the numeric tails present must be exactly
    // 1..=count with no gaps. (Walk the table in declaration order per band.)
    for (prefix_bands) |band| {
        var next: u32 = 1;
        for (table) |e| {
            if (e.str[0] != band) continue;
            const tail = std.fmt.parseInt(u32, e.str[1..], 10) catch
                @compileError("code " ++ e.str ++ " has a non-numeric tail");
            if (tail != next) @compileError("code " ++ e.str ++ " breaks band contiguity");
            next += 1;
        }
    }
}

const testing = std.testing;

test "str/defaultSeverity/slug for none and a real code" {
    try testing.expectEqual(@as(?[]const u8, null), str(.none));
    try testing.expectEqual(Severity.err, defaultSeverity(.none));
    try testing.expectEqual(@as(?[]const u8, null), slug(.none));

    try testing.expectEqualStrings("R0001", str(.R0001).?);
    try testing.expectEqualStrings("undeclared-identifier", slug(.R0001).?);
    try testing.expectEqual(Severity.err, defaultSeverity(.R0001));

    // Parse band (P####).
    try testing.expectEqualStrings("P0001", str(.P0001).?);
    try testing.expectEqualStrings("expected-token", slug(.P0001).?);
    try testing.expectEqual(Severity.err, defaultSeverity(.P0001));
    try testing.expectEqual(Code.P0005, fromStr("P0005").?);
}

test "fromStr round-trips every table code and rejects garbage" {
    for (table) |e| {
        const c = fromStr(e.str) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(e.code, c);
    }
    try testing.expectEqual(@as(?Code, null), fromStr("Z9999"));
    try testing.expectEqual(@as(?Code, null), fromStr("R0001x"));
}

test "registry runtime invariants: unique, in-band prefix, contiguous tails" {
    // Uniqueness (code + str) and prefix membership.
    for (table, 0..) |e, i| {
        try testing.expect(e.str[0] == 'L' or e.str[0] == 'P' or e.str[0] == 'R' or e.str[0] == 'T');
        for (table[i + 1 ..]) |o| {
            try testing.expect(e.code != o.code);
            try testing.expect(!std.mem.eql(u8, e.str, o.str));
        }
    }
    // Per-band contiguity from 0001.
    inline for ("LPRT") |band| {
        var next: u32 = 1;
        for (table) |e| {
            if (e.str[0] != band) continue;
            const tail = try std.fmt.parseInt(u32, e.str[1..], 10);
            try testing.expectEqual(next, tail);
            next += 1;
        }
    }
}
