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
    T0019, // mut-self-not-place (a `mut self` method called on a temporary/non-place receiver — M9)
    T0020, // overlapping-impl (two impls of the same protocol for the same type-ctor — coherence, M11)
    T0021, // no-conformance (an `impl .. has P` omits a required method, or `has` names an undeclared protocol — M11)
    T0022, // mut-self-on-scalar (a `mut self` method called on a builtin scalar receiver — the by-address self ABI has no place to write back — M12)
    T0023, // unsatisfied-bound (a monomorphization type-arg does not conform to its generic param's `[T has P]` bound — use-site, M13)
    T0024, // conformance-signature-mismatch (a conforming impl method's signature does not match the protocol's declared signature — coherence, M13)
    T0025, // ambiguous-conformance (a use site of a type that conforms to one generic protocol multiple times omits the disambiguating type-args — M14)
    T0026, // missing-eq-impl (`==`/`!=` used on a value type with no `Eq` conformance — operator desugar, M15)
    T0027, // missing-ord-impl (`<`/`>`/`<=`/`>=` used on a value type with no `Ord` conformance — operator desugar, M16)
    T0028, // missing-arith-impl (`+`/`-`/`*`/`/` used on a value type with no matching `Add`/`Sub`/`Mul`/`Div` conformance — operator desugar, M17)
    T0029, // non-derivable-field (structural `Eq` derive blocked by a struct field whose type does not conform to `Eq` — use-site, M18)
    T0030, // non-hashable-field (structural `Hash` derive blocked by a field whose type does not conform to `Hash` — use-site, M20)
    T0031, // non-displayable-arg (`print(x)`/structural `Display` derive blocked by an arg/field whose type does not conform to `Display` — use-site, M22)
    T0032, // question-non-optionresult (`?` operand is not an Option/Result, or the enclosing return type cannot absorb the residual — M24)
    T0033, // question-constructor-mismatch (`?` operand family differs from the enclosing return, or a Result error-type mismatch — M24)
    T0034, // literal-out-of-range (an integer literal exceeds the range of its annotated width — M1)

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
    .{ .code = .T0019, .str = "T0019", .slug = "mut-self-not-place" },
    .{ .code = .T0020, .str = "T0020", .slug = "overlapping-impl" },
    .{ .code = .T0021, .str = "T0021", .slug = "no-conformance" },
    .{ .code = .T0022, .str = "T0022", .slug = "mut-self-on-scalar" },
    .{ .code = .T0023, .str = "T0023", .slug = "unsatisfied-bound" },
    .{ .code = .T0024, .str = "T0024", .slug = "conformance-signature-mismatch" },
    .{ .code = .T0025, .str = "T0025", .slug = "ambiguous-conformance" },
    .{ .code = .T0026, .str = "T0026", .slug = "missing-eq-impl" },
    .{ .code = .T0027, .str = "T0027", .slug = "missing-ord-impl" },
    .{ .code = .T0028, .str = "T0028", .slug = "missing-arith-impl" },
    .{ .code = .T0029, .str = "T0029", .slug = "non-derivable-field" },
    .{ .code = .T0030, .str = "T0030", .slug = "non-hashable-field" },
    .{ .code = .T0031, .str = "T0031", .slug = "non-displayable-arg" },
    .{ .code = .T0032, .str = "T0032", .slug = "question-non-optionresult" },
    .{ .code = .T0033, .str = "T0033", .slug = "question-constructor-mismatch" },
    .{ .code = .T0034, .str = "T0034", .slug = "literal-out-of-range" },
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

    // Type band (T####) — the M13 bound + conformance codes.
    try testing.expectEqualStrings("T0023", str(.T0023).?);
    try testing.expectEqualStrings("unsatisfied-bound", slug(.T0023).?);
    try testing.expectEqualStrings("T0024", str(.T0024).?);
    try testing.expectEqualStrings("conformance-signature-mismatch", slug(.T0024).?);
    try testing.expectEqual(Code.T0023, fromStr("T0023").?);
    try testing.expectEqual(Code.T0024, fromStr("T0024").?);

    // M14 generic-protocol ambiguity code.
    try testing.expectEqualStrings("T0025", str(.T0025).?);
    try testing.expectEqualStrings("ambiguous-conformance", slug(.T0025).?);
    try testing.expectEqual(Code.T0025, fromStr("T0025").?);

    // M15 operator `Eq` code.
    try testing.expectEqualStrings("T0026", str(.T0026).?);
    try testing.expectEqualStrings("missing-eq-impl", slug(.T0026).?);
    try testing.expectEqual(Code.T0026, fromStr("T0026").?);

    // M16 operator `Ord` code.
    try testing.expectEqualStrings("T0027", str(.T0027).?);
    try testing.expectEqualStrings("missing-ord-impl", slug(.T0027).?);
    try testing.expectEqual(Code.T0027, fromStr("T0027").?);

    // M17 operator `Add`/`Sub`/`Mul`/`Div` code.
    try testing.expectEqualStrings("T0028", str(.T0028).?);
    try testing.expectEqualStrings("missing-arith-impl", slug(.T0028).?);
    try testing.expectEqual(Code.T0028, fromStr("T0028").?);

    // M18 structural `Eq` derive blocker code.
    try testing.expectEqualStrings("T0029", str(.T0029).?);
    try testing.expectEqualStrings("non-derivable-field", slug(.T0029).?);
    try testing.expectEqual(Code.T0029, fromStr("T0029").?);

    // M20 structural `Hash` derive blocker code.
    try testing.expectEqualStrings("T0030", str(.T0030).?);
    try testing.expectEqualStrings("non-hashable-field", slug(.T0030).?);
    try testing.expectEqual(Code.T0030, fromStr("T0030").?);

    // M22 `print`/structural `Display` derive blocker code.
    try testing.expectEqualStrings("T0031", str(.T0031).?);
    try testing.expectEqualStrings("non-displayable-arg", slug(.T0031).?);
    try testing.expectEqual(Code.T0031, fromStr("T0031").?);

    // M24 `?` operator diagnostics.
    try testing.expectEqualStrings("T0032", str(.T0032).?);
    try testing.expectEqualStrings("question-non-optionresult", slug(.T0032).?);
    try testing.expectEqual(Code.T0032, fromStr("T0032").?);

    try testing.expectEqualStrings("T0033", str(.T0033).?);
    try testing.expectEqualStrings("question-constructor-mismatch", slug(.T0033).?);
    try testing.expectEqual(Code.T0033, fromStr("T0033").?);

    // M1 integer-literal range code.
    try testing.expectEqualStrings("T0034", str(.T0034).?);
    try testing.expectEqualStrings("literal-out-of-range", slug(.T0034).?);
    try testing.expectEqual(Code.T0034, fromStr("T0034").?);
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
