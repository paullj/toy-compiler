//! A lexical token: a tag plus its source byte range `[start, end)`.
//!
//! Laid out as an `extern struct` so a `[]Token` can be reinterpreted as raw
//! bytes (and back) for the content-addressed cache — see `Cache.zig`.

const std = @import("std");

pub const Token = extern struct {
    tag: Tag,
    /// Byte offset of the first character of the token in the source.
    start: u32,
    /// Byte offset one past the last character (i.e. exclusive end).
    end: u32,

    pub fn text(tok: Token, source: []const u8) []const u8 {
        return source[tok.start..tok.end];
    }

    comptime {
        // Pinned wire layout: `[]Token` is reinterpreted as raw cache bytes, so a
        // field change that shifts these offsets/size relocates padding and silently
        // invalidates old blobs — make it a deliberate (build-breaking) decision.
        if (@sizeOf(Token) != 12 or @offsetOf(Token, "start") != 4 or @offsetOf(Token, "end") != 8)
            @compileError("Token layout changed — bump the cache-blob format");
    }
};

/// Return `value` with its interior/trailing PADDING bytes zeroed. The cache-bound
/// `extern struct`s (`Token`, `Ast.Node`) have a `[]T` reinterpreted as raw bytes for
/// the content-addressed cache (and folded into the lex/parse fp); a field-wise struct
/// literal leaves padding UNDEFINED, so the raw blob — and the recorded fp — would
/// differ run-to-run on a cold build. Build every cached element through this so that
/// byte/fp stays deterministic. See `Ast.contentFp`.
pub fn zeroPad(comptime T: type, value: T) T {
    var clean: T = std.mem.zeroes(T);
    inline for (std.meta.fields(T)) |f| @field(clean, f.name) = @field(value, f.name);
    return clean;
}

pub const Tag = enum(u8) {
    invalid,
    eof,

    identifier,
    number,
    string,

    kw_fn,
    kw_return,
    kw_if,
    kw_else,
    kw_while,
    kw_true,
    kw_false,
    kw_struct,
    kw_loop,
    kw_for,
    kw_in,
    kw_break,
    kw_continue,
    kw_enum,
    kw_match,

    plus,
    minus,
    star,
    slash,
    eq,
    eq_eq,
    bang,
    bang_eq,
    lt,
    lt_eq,
    gt,
    gt_eq,
    amp_amp,
    pipe_pipe,
    /// `|` — the or-pattern separator in pattern position (not a general operator).
    pipe,
    l_paren,
    r_paren,
    l_brace,
    r_brace,
    comma,
    /// Statement terminator, inserted by the lexer at a newline that ends a
    /// statement (Go-style). There is no explicit terminator character.
    newline,
    colon,
    /// `:=` short variable declaration.
    colon_eq,
    dot,
    arrow,
    /// `..` range operator, valid only in a for-head.
    dotdot,
    /// `@` label sigil, prefixing a label name on a loop/while/for/block.
    at,

    // Module keywords. New `Tag` variants MUST extend the end: ordinals are frozen
    // because `Tag` is `enum(u8)` and `[]Token` is memcpy'd to/from the content cache.

    /// `import` — begins a module import declaration.
    kw_import,
    /// `pub` — exports the declaration that follows (`pub fn`/`pub struct`/`pub enum`).
    kw_pub,
    /// `as` — renames an imported namespace (`import a/b as r`).
    kw_as,

    // Error tokens. Like all `Tag` variants these are append-only (frozen
    // ordinals; `[]Token` is memcpy'd to/from the content cache). The lexer stays
    // total — it never fails — and instead emits these so error runs are
    // preserved as spans; the parser attaches diagnostics for them.

    /// A string literal with no closing quote (ran into a newline or EOF). The
    /// span starts at the opening quote and covers everything consumed, so a
    /// caret can point at the quote. Distinct from `.invalid` so a partial string
    /// is preserved as a string rather than an opaque byte run.
    string_unterminated,

    // Generics brackets. Appended at the END (frozen ordinals; `[]Token` is
    // memcpy'd to/from the content cache). `[` opens a generic-param list, a
    // type-application `Base[Arg,..]`, or explicit call type-args `f[int](..)`;
    // `]` closes it. Reserved for type position and postfix-call position only —
    // a future value-index `v[i]` must adopt a distinct form (see docs/typesystem.md).

    /// `[` — opens a generic-param list / type-application / call type-args.
    l_bracket,
    /// `]` — closes a generic-param list / type-application / call type-args.
    r_bracket,

    // Inherent methods (M8). Appended at the END (frozen ordinal; `[]Token` is
    // memcpy'd to/from the content cache). `impl` opens a keyword-led inherent-
    // method block `impl Type { fn .. }`. `self`/`Self` stay plain identifiers
    // recognized contextually by text (the `_` wildcard precedent) — NOT keywords.

    /// `impl` — opens an inherent-method block for a concrete type.
    kw_impl,

    // Mutating receiver (M9). Appended at the END (frozen ordinal; `[]Token` is
    // memcpy'd to/from the content cache). `mut` qualifies a method's `self`
    // receiver (`fn m(mut self, ..)`) so the receiver is passed by address and
    // mutated in place. It is a reserved keyword only in that position; the parser
    // rejects it elsewhere.

    /// `mut` — qualifies a `self` receiver as a by-address, in-place-mutating one.
    kw_mut,

    // Protocols (M11). Appended at the END (frozen ordinals; `[]Token` is memcpy'd
    // to/from the content cache). `protocol` opens a signature-only protocol decl;
    // `has` leads a conformance impl `impl T has P { .. }`. Both are reserved
    // keywords; the current corpus never uses either as an identifier.

    /// `protocol` — opens a signature-only protocol declaration.
    kw_protocol,
    /// `has` — the single conformance relation, leading `impl T has P { .. }`.
    kw_has,
};

/// Maps identifier text to its keyword tag, if any.
pub const keywords = std.StaticStringMap(Tag).initComptime(.{
    .{ "fn", .kw_fn },
    .{ "return", .kw_return },
    .{ "if", .kw_if },
    .{ "else", .kw_else },
    .{ "while", .kw_while },
    .{ "true", .kw_true },
    .{ "false", .kw_false },
    .{ "struct", .kw_struct },
    .{ "loop", .kw_loop },
    .{ "for", .kw_for },
    .{ "in", .kw_in },
    .{ "break", .kw_break },
    .{ "continue", .kw_continue },
    .{ "enum", .kw_enum },
    .{ "match", .kw_match },
    .{ "import", .kw_import },
    .{ "pub", .kw_pub },
    .{ "as", .kw_as },
    .{ "impl", .kw_impl },
    .{ "mut", .kw_mut },
    .{ "protocol", .kw_protocol },
    .{ "has", .kw_has },
});

comptime {
    // Every `kw_*` tag must appear in `keywords` exactly once, so a keyword cannot
    // be added to `Tag` without registering its spelling here (nor be registered
    // twice). The lexer's `keywords` lookup is then the sole keyword-spelling table.
    for (@typeInfo(Tag).@"enum".fields) |f| {
        if (!std.mem.startsWith(u8, f.name, "kw_")) continue;
        const want: Tag = @enumFromInt(f.value);
        var seen: usize = 0;
        for (keywords.values()) |v| {
            if (v == want) seen += 1;
        }
        if (seen != 1) @compileError("keyword tag '" ++ f.name ++ "' must be in `keywords` exactly once");
    }
}
