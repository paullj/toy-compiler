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

    pub fn lexeme(tag: Tag) ?[]const u8 {
        return switch (tag) {
            .kw_fn => "fn",
            .kw_return => "return",
            .kw_if => "if",
            .kw_else => "else",
            .kw_while => "while",
            .kw_true => "true",
            .kw_false => "false",
            .kw_struct => "struct",
            .kw_loop => "loop",
            .kw_for => "for",
            .kw_in => "in",
            .kw_break => "break",
            .kw_continue => "continue",
            .kw_enum => "enum",
            .kw_match => "match",
            .plus => "+",
            .minus => "-",
            .star => "*",
            .slash => "/",
            .eq => "=",
            .eq_eq => "==",
            .bang => "!",
            .bang_eq => "!=",
            .lt => "<",
            .lt_eq => "<=",
            .gt => ">",
            .gt_eq => ">=",
            .amp_amp => "&&",
            .pipe_pipe => "||",
            .pipe => "|",
            .l_paren => "(",
            .r_paren => ")",
            .l_brace => "{",
            .r_brace => "}",
            .comma => ",",
            .colon => ":",
            .colon_eq => ":=",
            .dot => ".",
            .arrow => "->",
            .dotdot => "..",
            .at => "@",
            else => null,
        };
    }
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
});
