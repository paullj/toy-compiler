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
};

pub const Tag = enum(u8) {
    invalid,
    eof,

    identifier,
    number,
    string,

    // keywords
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

    // punctuation / operators
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

    // ---- M14: module keywords (appended LAST; ordinals of all tags above are
    // frozen because Tag is enum(u8) and `[]Token` is memcpy'd to/from the
    // content cache — new variants must extend the end). -----------------------

    /// `import` — begins a module import declaration.
    kw_import,
    /// `pub` — exports the declaration that follows (`pub fn`/`pub struct`/`pub enum`).
    kw_pub,
    /// `as` — renames an imported namespace (`import a/b as r`).
    kw_as,
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
