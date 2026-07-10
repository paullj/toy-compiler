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
/// Count of error tokens (`.invalid` / `.string_unterminated`) emitted so far.
/// Once it reaches `max_error_tokens` the lexer stops trying to structure the
/// rest of the file and swallows all remaining bytes into one trailing
/// `.invalid`, so a pathological all-garbage file can't produce a token per byte.
error_count: u32,

/// Cap on structured error tokens before the lexer gives up and coalesces the
/// whole tail into one `.invalid` (cf. Roslyn's `_badTokenCount`). Chosen so real
/// files with a handful of typos are fully tokenized, while adversarial garbage is
/// bounded to O(cap) tokens instead of O(bytes).
pub const max_error_tokens: u32 = 100;

pub fn init(source: []const u8) Lexer {
    return .{ .source = source, .index = 0, .prev = .invalid, .error_count = 0 };
}

/// Lex the whole source into a slice of tokens, terminated by a single `.eof`.
/// Caller owns the returned memory.
pub fn tokenize(gpa: std.mem.Allocator, source: []const u8) ![]Token {
    var lexer = Lexer.init(source);
    var tokens: std.ArrayList(Token) = .empty;
    errdefer tokens.deinit(gpa);
    // Debug-only invariant: token spans tile `[0, source.len)` monotonically —
    // each token's `start` is at or past the previous token's `end` (never
    // backtracks or overlaps; the gaps are skipped trivia), each span is
    // well-formed (`start <= end`), and the terminating `.eof` sits exactly at
    // `source.len`. This is what makes error tokens safe to preserve as spans: no
    // byte is double-counted and the cursor always makes forward progress, so a
    // coalesced `.invalid` run can never desync the offsets a caret points at.
    var prev_end: u32 = 0;
    while (true) {
        const tok = lexer.next();
        if (std.debug.runtime_safety) {
            std.debug.assert(tok.start >= prev_end); // monotonic, no overlap
            std.debug.assert(tok.end >= tok.start); // well-formed span
            prev_end = tok.end;
        }
        // Zero the extern-struct padding before the token enters the cached blob
        // (see `token.zeroPad`).
        try tokens.append(gpa, token.zeroPad(Token, tok));
        if (tok.tag == .eof) {
            if (std.debug.runtime_safety) std.debug.assert(tok.end == source.len);
            break;
        }
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
    if (isError(tok.tag)) l.error_count += 1;
    return tok;
}

/// Whether a tag is one of the lexer's error tokens. Kept in one place so the
/// spam-cap accounting and the parser's future diagnostic handling agree.
fn isError(tag: Tag) bool {
    return switch (tag) {
        .invalid, .string_unterminated, .char_unterminated => true,
        else => false,
    };
}

fn lexToken(l: *Lexer) Token {
    const start = l.index;
    if (l.index >= l.source.len) return l.make(.eof, start);

    // Spam cap: once a file has produced too many error tokens it is hopeless, so
    // stop structuring it — swallow everything left into one trailing `.invalid`
    // rather than emitting more per-construct error tokens (cf. `max_error_tokens`).
    if (l.error_count >= max_error_tokens) {
        l.index = @intCast(l.source.len);
        return l.make(.invalid, start);
    }

    const c = l.source[l.index];
    if (isIdentStart(c)) return l.lexIdentifier(start);
    if (isDigit(c)) return l.lexNumber(start);
    if (c == '"') return l.lexString(start);
    if (c == '\'') return l.lexChar(start);
    return l.lexSymbol(start);
}

/// Whether a token of this tag can end a statement, and so triggers terminator
/// insertion when a newline follows it.
fn canEndStatement(tag: Tag) bool {
    return switch (tag) {
        .identifier, .number, .string, .char_lit, .kw_true, .kw_false, .kw_return, .kw_break, .kw_continue, .r_paren, .r_brace, .question => true,
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
    // A `0x`/`0o`/`0b` prefix (only valid immediately after a leading `0`) switches to
    // the matching digit set. A stray invalid char just ends the token — there is no
    // malformed-number token; a bad literal (`0xZZ`) mis-spans and surfaces downstream.
    if (l.source[start] == '0' and l.index < l.source.len) {
        const base: ?u8 = switch (l.source[l.index]) {
            'x', 'X' => 16,
            'o', 'O' => 8,
            'b', 'B' => 2,
            else => null,
        };
        if (base) |b| {
            l.index += 1; // consume the base letter
            while (l.index < l.source.len) {
                const c = l.source[l.index];
                if (!isBaseDigit(c, b) and c != '_') break;
                l.index += 1;
            }
            return l.make(.number, start);
        }
    }
    while (l.index < l.source.len) {
        const c = l.source[l.index];
        if (!isDigit(c) and c != '_') break;
        l.index += 1;
    }
    return l.make(.number, start);
}

/// The single `\`-skip-2 quoted-literal scan shared by strings and chars (the escape
/// scan that must never drift between the two): consume to the closing `quote`, emitting
/// `closed_tag`; on a newline (span up to, NOT past, it — a total span from the opening
/// quote) or EOF before the close, emit `unterm_tag`. Content is NOT validated here — an
/// empty/multi-codepoint/bad-escape char is `decodeChar`'s decode-time diagnostic, and the
/// lexer stays total. `\`-skip-2 lets a `'\''`/`"\""` escaped quote and a raw multibyte
/// `'€'` span to their real closing quote.
fn lexQuoted(l: *Lexer, start: u32, quote: u8, closed_tag: Tag, unterm_tag: Tag) Token {
    l.index += 1; // opening quote
    while (l.index < l.source.len) {
        const c = l.source[l.index];
        if (c == '\\') {
            l.index = @min(l.index + 2, @as(u32, @intCast(l.source.len)));
        } else if (c == quote) {
            l.index += 1; // closing quote
            return l.make(closed_tag, start);
        } else if (c == '\n') {
            return l.make(unterm_tag, start);
        } else {
            l.index += 1;
        }
    }
    return l.make(unterm_tag, start); // unterminated at EOF
}

fn lexString(l: *Lexer, start: u32) Token {
    return lexQuoted(l, start, '"', .string, .string_unterminated);
}

fn lexChar(l: *Lexer, start: u32) Token {
    return lexQuoted(l, start, '\'', .char_lit, .char_unterminated);
}

fn lexSymbol(l: *Lexer, start: u32) Token {
    const c = l.source[l.index];

    // An unrecognizable lead byte (one that begins no token) coalesces with the
    // following unrecognizable bytes into ONE `.invalid`, so a garbage run is a
    // single span instead of a token per byte. A recognized lead byte that merely
    // fails to form an operator (e.g. a lone `&`) is NOT coalesced — it falls
    // through to the switch below and stays a one-byte `.invalid`.
    if (!beginsToken(c)) {
        l.index += 1;
        while (l.index < l.source.len and !beginsToken(l.source[l.index])) l.index += 1;
        return l.make(.invalid, start);
    }

    l.index += 1;
    const tag: Tag = switch (c) {
        '+' => .plus,
        '-' => if (l.eat('>')) .arrow else .minus,
        '*' => .star,
        '/' => .slash,
        '%' => .percent,
        '=' => if (l.eat('=')) .eq_eq else .eq,
        '!' => if (l.eat('=')) .bang_eq else .bang,
        '<' => if (l.eat('<')) .lt_lt else if (l.eat('=')) .lt_eq else .lt,
        '>' => if (l.eat('>')) .gt_gt else if (l.eat('=')) .gt_eq else .gt,
        '&' => if (l.eat('&')) .amp_amp else .amp,
        '^' => .caret,
        '~' => .tilde,
        '|' => if (l.eat('|')) .pipe_pipe else .pipe,
        '(' => .l_paren,
        ')' => .r_paren,
        '{' => .l_brace,
        '}' => .r_brace,
        '[' => .l_bracket,
        ']' => .r_bracket,
        ',' => .comma,
        ':' => if (l.eat('=')) .colon_eq else .colon,
        '.' => if (l.eat('.')) .dotdot else .dot,
        '@' => .at,
        '?' => .question,
        else => unreachable, // `beginsToken` already screened out non-lead bytes
    };
    return l.make(tag, start);
}

/// Whether byte `c` can begin a token or trivia — i.e. `next` would make progress
/// on it rather than treat it as an unrecognizable byte. Used to bound a coalesced
/// `.invalid` run: the run stops at the first byte that could start real input.
/// The symbol set here must mirror the recognized lead bytes in `lexSymbol`'s
/// switch and the trivia handled by `skipTrivia`.
fn beginsToken(c: u8) bool {
    if (isIdentStart(c) or isDigit(c)) return true;
    return switch (c) {
        '"' => true, // string
        '\'' => true, // char literal
        ' ', '\t', '\r', '\n', '#' => true, // trivia
        '+', '-', '*', '/', '%', '=', '!', '<', '>', '&', '^', '~', '|', '(', ')', '{', '}', '[', ']', ',', ':', '.', '@', '?' => true,
        else => false,
    };
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

fn isBaseDigit(c: u8, base: u8) bool {
    return switch (base) {
        2 => c == '0' or c == '1',
        8 => c >= '0' and c <= '7',
        16 => isDigit(c) or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F'),
        else => false,
    };
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

test "lone & is bitwise-and; lone | is a pattern separator" {
    try expectTags("&", &.{ .amp, .eof });
    try expectTags("&&", &.{ .amp_amp, .eof });
    try expectTags("|", &.{ .pipe, .eof });
    try expectTags("||", &.{ .pipe_pipe, .eof });
}

test "bitwise and shift operators lex to their tags" {
    try expectTags("& ^ ~ << >>", &.{ .amp, .caret, .tilde, .lt_lt, .gt_gt, .eof });
    try expectTags("a << b >> c", &.{ .identifier, .lt_lt, .identifier, .gt_gt, .identifier, .eof });
    // `<<`/`>>` do not swallow a following `=`; `<=`/`>=` still lex.
    try expectTags("< <= << > >= >>", &.{ .lt, .lt_eq, .lt_lt, .gt, .gt_eq, .gt_gt, .eof });
}

test "unterminated string at EOF is one string_unterminated token spanning to EOF" {
    const src = "\"abc"; // no closing quote, runs to EOF
    const tokens = try tokenize(testing.allocator, src);
    defer testing.allocator.free(tokens);
    try testing.expectEqual(@as(usize, 2), tokens.len);
    try testing.expectEqual(Tag.string_unterminated, tokens[0].tag);
    try testing.expectEqual(Tag.eof, tokens[1].tag);
    // Span starts at the opening quote and covers the whole partial string, so a
    // caret can point at the quote.
    try testing.expectEqualStrings("\"abc", tokens[0].text(src));
}

test "unterminated string ending in a backslash keeps its span within source" {
    const src = "\"ab\\"; // trailing backslash at EOF must not skip past the end
    const tokens = try tokenize(testing.allocator, src);
    defer testing.allocator.free(tokens);
    try testing.expectEqual(Tag.string_unterminated, tokens[0].tag);
    try testing.expectEqualStrings(src, tokens[0].text(src));
    try testing.expectEqual(Tag.eof, tokens[tokens.len - 1].tag);
    try testing.expectEqual(@as(u32, src.len), tokens[tokens.len - 1].end);
}

test "unterminated string at newline stops before the newline" {
    // The partial string ends at the newline (not consuming it); the newline then
    // becomes its own trivia/terminator handling — here `prev` is an error tag so
    // no terminator is inserted, and `x` follows on the next line.
    const src = "\"abc\nx";
    const tokens = try tokenize(testing.allocator, src);
    defer testing.allocator.free(tokens);
    try testing.expectEqual(Tag.string_unterminated, tokens[0].tag);
    try testing.expectEqualStrings("\"abc", tokens[0].text(src));
    try testing.expectEqual(Tag.identifier, tokens[1].tag);
    try testing.expectEqualStrings("x", tokens[1].text(src));
}

test "char literals lex to a single char_lit spanning both quotes" {
    try expectTags("'A'", &.{ .char_lit, .eof });
    // A `\`-skip-2 scan makes an escaped quote `'\''` and any escape lex correctly.
    try expectTags("'\\''", &.{ .char_lit, .eof });
    try expectTags("'\\n'", &.{ .char_lit, .eof });
    // `char_lit` ends a statement, so a newline after it inserts a terminator.
    try expectTags("c := 'A'\nd := 'B'\n", &.{
        .identifier, .colon_eq, .char_lit, .newline,
        .identifier, .colon_eq, .char_lit, .newline, .eof,
    });
}

test "a raw multibyte char literal spans its source bytes as one char_lit" {
    const src = "'\u{20AC}'"; // '€' — 3 source bytes between the quotes
    const tokens = try tokenize(testing.allocator, src);
    defer testing.allocator.free(tokens);
    try testing.expectEqual(@as(usize, 2), tokens.len);
    try testing.expectEqual(Tag.char_lit, tokens[0].tag);
    try testing.expectEqualStrings(src, tokens[0].text(src));
}

test "unterminated char literal is one char_unterminated token" {
    // At EOF: spans to EOF.
    {
        const src = "'ab";
        const tokens = try tokenize(testing.allocator, src);
        defer testing.allocator.free(tokens);
        try testing.expectEqual(Tag.char_unterminated, tokens[0].tag);
        try testing.expectEqualStrings("'ab", tokens[0].text(src));
        try testing.expectEqual(Tag.eof, tokens[tokens.len - 1].tag);
    }
    // At a newline: stops before it, leaving the newline as trivia.
    {
        const src = "'a\nx";
        const tokens = try tokenize(testing.allocator, src);
        defer testing.allocator.free(tokens);
        try testing.expectEqual(Tag.char_unterminated, tokens[0].tag);
        try testing.expectEqualStrings("'a", tokens[0].text(src));
        try testing.expectEqual(Tag.identifier, tokens[1].tag);
    }
}

test "a run of unknown bytes coalesces into one invalid token" {
    // `$` begins no token and is not trivia, so a run of them is a single
    // `.invalid` rather than one token per byte.
    const src = "$$$";
    const tokens = try tokenize(testing.allocator, src);
    defer testing.allocator.free(tokens);
    try testing.expectEqual(@as(usize, 2), tokens.len); // one .invalid + .eof
    try testing.expectEqual(Tag.invalid, tokens[0].tag);
    try testing.expectEqualStrings("$$$", tokens[0].text(src));
    // A run of unrecognized bytes bounded by real tokens is still one coalesced
    // span; the recognized `~` after it stops the run.
    try expectTags("a %$~ b", &.{ .identifier, .percent, .invalid, .tilde, .identifier, .eof });
}

/// Assert the debug span-tiling invariant holds directly: spans are monotone with
/// no overlap, cover to EOF, and (aside from skipped trivia) leave no byte behind.
fn expectSpansTile(source: []const u8) !void {
    const tokens = try tokenize(testing.allocator, source);
    defer testing.allocator.free(tokens);
    var prev_end: u32 = 0;
    for (tokens) |tok| {
        try testing.expect(tok.start >= prev_end);
        try testing.expect(tok.end >= tok.start);
        prev_end = tok.end;
    }
    try testing.expectEqual(@as(u32, @intCast(source.len)), tokens[tokens.len - 1].end);
    try testing.expectEqual(Tag.eof, tokens[tokens.len - 1].tag);
}

test "span totality holds on mixed valid and invalid input" {
    try expectSpansTile("fn f() { x := \"oops\n  y %% z + 1\n}\n");
    try expectSpansTile("$$$");
    try expectSpansTile("\"abc");
    try expectSpansTile(""); // just .eof at [0,0)
}

test "a valid program tokenizes exactly as before" {
    // Guards the invariant that valid programs are byte-identical after this change:
    // no error tokens appear and the tag stream is unchanged.
    try expectTags(
        \\fn add(a: int, b: int) -> int {
        \\    return a + b
        \\}
    , &.{
        // `{` does not end a statement, so the newline after it inserts no
        // terminator; the newline after `b` does.
        .kw_fn,      .identifier, .l_paren, .identifier, .colon,     .identifier,
        .comma,      .identifier, .colon,   .identifier, .r_paren,   .arrow,
        .identifier, .l_brace,    .kw_return, .identifier, .plus,    .identifier,
        .newline,    .r_brace,    .eof,
    });
}

test "invalid error tokens are capped: garbage past the cap becomes one trailing invalid" {
    // Build a source of `max_error_tokens + 50` isolated invalid bytes, each
    // separated by a space so they would each be their own `.invalid` if uncapped.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(testing.allocator);
    const n = max_error_tokens + 50;
    var i: u32 = 0;
    while (i < n) : (i += 1) try buf.appendSlice(testing.allocator, "$ ");
    const tokens = try tokenize(testing.allocator, buf.items);
    defer testing.allocator.free(tokens);
    // Cap error tokens, then a single trailing `.invalid` swallowing the rest, then
    // `.eof` — far fewer than `n` tokens.
    try testing.expect(tokens.len < n);
    try testing.expectEqual(Tag.eof, tokens[tokens.len - 1].tag);
    // The token just before EOF is the trailing swallow and reaches EOF.
    const last = tokens[tokens.len - 2];
    try testing.expectEqual(Tag.invalid, last.tag);
    try testing.expectEqual(@as(u32, @intCast(buf.items.len)), last.end);
}

test "short declaration with hash comment and string" {
    try expectTags(
        \\x := 1_000  # a comment
        \\msg := "hi"
    , &.{ .identifier, .colon_eq, .number, .newline, .identifier, .colon_eq, .string, .eof });
}

test "base-prefixed integer literals lex as a single number" {
    try expectTags("0x2A", &.{ .number, .eof });
    try expectTags("0o52", &.{ .number, .eof });
    try expectTags("0b0010_1010", &.{ .number, .eof });
    // A leading-zero decimal is NOT a base prefix; it stays one decimal number.
    try expectTags("042", &.{ .number, .eof });
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

test "brackets lex to l_bracket and r_bracket" {
    try expectTags("[ ]", &.{ .l_bracket, .r_bracket, .eof });
    // A generic-application stream tiles cleanly with the surrounding tokens.
    try expectTags("f[T]", &.{ .identifier, .l_bracket, .identifier, .r_bracket, .eof });
    try expectSpansTile("fn id[T](x: T) -> T { x }\n");
}

test "postfix ? lexes to question and can end a statement" {
    try expectTags("o?", &.{ .identifier, .question, .eof });
    // `?` is in `canEndStatement`, so a newline after `o?` inserts a terminator —
    // `v := o?` followed by another statement must not glue the two together.
    try expectTags(
        \\v := o?
        \\w
    , &.{ .identifier, .colon_eq, .identifier, .question, .newline, .identifier, .eof });
    // `f()?` composes: the `?` follows the call's `)`.
    try expectTags("f()?", &.{ .identifier, .l_paren, .r_paren, .question, .eof });
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
