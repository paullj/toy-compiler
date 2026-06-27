//! Hand-written lexer. Pure function over a source buffer: it never allocates
//! for itself and holds no IO, which is exactly what makes lexing cacheable and
//! safe to run on many cores at once (one `Lexer` per file per thread).

const std = @import("std");
const token = @import("ast/Token.zig");
const Token = token.Token;
const Tag = token.Tag;

const Lexer = @This();

source: []const u8,
index: u32,
/// Tag of the last token returned, for newline terminator insertion. Starts as
/// `.invalid` (which never ends a statement) so a leading newline inserts nothing.
prev: Tag,

pub fn init(source: []const u8) Lexer {
    return .{ .source = source, .index = 0, .prev = .invalid };
}

/// Lex the whole source into a slice of tokens, terminated by a single `.eof`.
/// Caller owns the returned memory.
pub fn tokenize(gpa: std.mem.Allocator, source: []const u8) ![]Token {
    var lexer = Lexer.init(source);
    var tokens: std.ArrayList(Token) = .empty;
    errdefer tokens.deinit(gpa);
    while (true) {
        const tok = lexer.next();
        // Rebuild from a zeroed value: `Token` is an `extern struct` with 3
        // padding bytes between `tag` and `start`, left undefined by `next`'s
        // struct literals. That garbage otherwise flows into the cached `[]Token`
        // blob (and the recorded lex result fp) via `sliceAsBytes`, breaking
        // cold-build byte/fp determinism.
        var clean: Token = std.mem.zeroes(Token);
        clean.tag = tok.tag;
        clean.start = tok.start;
        clean.end = tok.end;
        try tokens.append(gpa, clean);
        if (tok.tag == .eof) break;
    }
    return tokens.toOwnedSlice(gpa);
}

/// Produce the next token, advancing the cursor. Returns `.eof` repeatedly once
/// the end of source is reached.
pub fn next(l: *Lexer) Token {
    const crossed_newline = l.skipTrivia();

    // Go-style terminator insertion: a newline that follows a statement-ending
    // token becomes an implicit terminator. It is zero-width at the cursor.
    if (crossed_newline and canEndStatement(l.prev)) {
        l.prev = .newline;
        return .{ .tag = .newline, .start = l.index, .end = l.index };
    }

    const tok = l.lexToken();
    l.prev = tok.tag;
    return tok;
}

fn lexToken(l: *Lexer) Token {
    const start = l.index;
    if (l.index >= l.source.len) return l.make(.eof, start);

    const c = l.source[l.index];
    if (isIdentStart(c)) return l.lexIdentifier(start);
    if (isDigit(c)) return l.lexNumber(start);
    if (c == '"') return l.lexString(start);
    return l.lexSymbol(start);
}

/// Whether a token of this tag can end a statement, and so triggers terminator
/// insertion when a newline follows it.
fn canEndStatement(tag: Tag) bool {
    return switch (tag) {
        .identifier, .number, .string, .kw_true, .kw_false, .kw_return, .kw_break, .kw_continue, .r_paren, .r_brace => true,
        else => false,
    };
}

/// Skip whitespace and `#` line comments. Returns whether a newline was crossed
/// (consecutive blank lines collapse into a single crossing).
fn skipTrivia(l: *Lexer) bool {
    var crossed_newline = false;
    while (l.index < l.source.len) {
        switch (l.source[l.index]) {
            ' ', '\t', '\r' => l.index += 1,
            '\n' => {
                crossed_newline = true;
                l.index += 1;
            },
            '#' => while (l.index < l.source.len and l.source[l.index] != '\n') {
                l.index += 1;
            },
            else => return crossed_newline,
        }
    }
    return crossed_newline;
}

fn lexIdentifier(l: *Lexer, start: u32) Token {
    l.index += 1;
    while (l.index < l.source.len and isIdentCont(l.source[l.index])) l.index += 1;
    const text = l.source[start..l.index];
    const tag = token.keywords.get(text) orelse .identifier;
    return l.make(tag, start);
}

fn lexNumber(l: *Lexer, start: u32) Token {
    l.index += 1;
    while (l.index < l.source.len) {
        const c = l.source[l.index];
        if (!isDigit(c) and c != '_') break;
        l.index += 1;
    }
    return l.make(.number, start);
}

fn lexString(l: *Lexer, start: u32) Token {
    l.index += 1; // opening quote
    while (l.index < l.source.len) {
        switch (l.source[l.index]) {
            '\\' => l.index += 2, // skip escaped char
            '"' => {
                l.index += 1; // closing quote
                return l.make(.string, start);
            },
            '\n' => return l.make(.invalid, start), // unterminated on this line
            else => l.index += 1,
        }
    }
    return l.make(.invalid, start); // unterminated at EOF
}

fn lexSymbol(l: *Lexer, start: u32) Token {
    const c = l.source[l.index];
    l.index += 1;
    const tag: Tag = switch (c) {
        '+' => .plus,
        '-' => if (l.eat('>')) .arrow else .minus,
        '*' => .star,
        '/' => .slash,
        '=' => if (l.eat('=')) .eq_eq else .eq,
        '!' => if (l.eat('=')) .bang_eq else .bang,
        '<' => if (l.eat('=')) .lt_eq else .lt,
        '>' => if (l.eat('=')) .gt_eq else .gt,
        '&' => if (l.eat('&')) .amp_amp else .invalid,
        '|' => if (l.eat('|')) .pipe_pipe else .pipe,
        '(' => .l_paren,
        ')' => .r_paren,
        '{' => .l_brace,
        '}' => .r_brace,
        ',' => .comma,
        ':' => if (l.eat('=')) .colon_eq else .colon,
        '.' => if (l.eat('.')) .dotdot else .dot,
        '@' => .at,
        else => .invalid,
    };
    return l.make(tag, start);
}

/// Consume the next byte if it equals `c`; report whether it did.
fn eat(l: *Lexer, c: u8) bool {
    if (l.index < l.source.len and l.source[l.index] == c) {
        l.index += 1;
        return true;
    }
    return false;
}

fn make(l: *const Lexer, tag: Tag, start: u32) Token {
    return .{ .tag = tag, .start = start, .end = l.index };
}

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn isIdentStart(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_';
}

fn isIdentCont(c: u8) bool {
    return isIdentStart(c) or isDigit(c);
}

const testing = std.testing;

fn expectTags(source: []const u8, expected: []const Tag) !void {
    const tokens = try tokenize(testing.allocator, source);
    defer testing.allocator.free(tokens);
    try testing.expectEqual(expected.len, tokens.len);
    for (expected, tokens) |want, got| try testing.expectEqual(want, got.tag);
}

test "keywords and identifiers" {
    try expectTags("fn main return x", &.{ .kw_fn, .identifier, .kw_return, .identifier, .eof });
}

test "operators including := and other two-char" {
    try expectTags("== = != -> := < <= :", &.{ .eq_eq, .eq, .bang_eq, .arrow, .colon_eq, .lt, .lt_eq, .colon, .eof });
}

test "logical && and || are two-char operators" {
    try expectTags("a && b || c", &.{ .identifier, .amp_amp, .identifier, .pipe_pipe, .identifier, .eof });
}

test "lone & is invalid; lone | is a pattern separator" {
    try expectTags("&", &.{ .invalid, .eof });
    try expectTags("|", &.{ .pipe, .eof });
    try expectTags("||", &.{ .pipe_pipe, .eof });
}

test "short declaration with hash comment and string" {
    try expectTags(
        \\x := 1_000  # a comment
        \\msg := "hi"
    , &.{ .identifier, .colon_eq, .number, .newline, .identifier, .colon_eq, .string, .eof });
}

test "newline inserts terminator only after statement-ending tokens" {
    // `x` ends a statement → terminator; the `=` line continues (operator at
    // end of line suppresses insertion), so no terminator before `42`.
    try expectTags(
        \\x
        \\y =
        \\42
    , &.{ .identifier, .newline, .identifier, .eq, .number, .eof });
}

test "blank lines and trailing newline collapse to one terminator" {
    try expectTags("a\n\n\nb\n", &.{ .identifier, .newline, .identifier, .newline, .eof });
}

test "loop keywords and dotdot" {
    try expectTags("loop for in break continue ..", &.{ .kw_loop, .kw_for, .kw_in, .kw_break, .kw_continue, .dotdot, .eof });
}

test "bare break and continue terminate statements" {
    try expectTags("break\ncontinue\n", &.{ .kw_break, .newline, .kw_continue, .newline, .eof });
}

test "label sigil at lexes as a separate token before the name" {
    try expectTags("@outer loop", &.{ .at, .identifier, .kw_loop, .eof });
    try expectTags("break @done", &.{ .kw_break, .at, .identifier, .eof });
}

test "token spans map back to source text" {
    const src = "fn add";
    const tokens = try tokenize(testing.allocator, src);
    defer testing.allocator.free(tokens);
    try testing.expectEqualStrings("fn", tokens[0].text(src));
    try testing.expectEqualStrings("add", tokens[1].text(src));
}
