#include "tree_sitter/parser.h"

// The toy language separates statements by newlines (Go-style automatic terminator
// insertion), but a raw newline is otherwise insignificant. This external scanner
// mirrors the compiler lexer's terminator insertion (packages/compiler/src/lex.zig):
// it emits a zero-width `_newline` token at a line break that the parser is willing to
// treat as a statement/arm terminator. Because the token is zero-width and only valid at
// terminator positions, an operator or an incomplete expression at end of line (where
// `_newline` is not in `valid_symbols`) is never terminated — the newline stays trivia
// and the expression continues onto the next line, exactly as the compiler does.
//
// `_newline` is trivia-excluded from the agreement comparison, so its presence changes
// only the grouping of the parse, never the leaf token stream.

enum TokenType {
    NEWLINE,
};

void *tree_sitter_toy_external_scanner_create(void) { return 0; }
void tree_sitter_toy_external_scanner_destroy(void *payload) { (void)payload; }
unsigned tree_sitter_toy_external_scanner_serialize(void *payload, char *buffer) {
    (void)payload;
    (void)buffer;
    return 0;
}
void tree_sitter_toy_external_scanner_deserialize(void *payload, const char *buffer, unsigned length) {
    (void)payload;
    (void)buffer;
    (void)length;
}

bool tree_sitter_toy_external_scanner_scan(void *payload, TSLexer *lexer, const bool *valid_symbols) {
    (void)payload;
    if (!valid_symbols[NEWLINE]) return false;

    // A zero-width token: mark the end at the current position before looking ahead, so
    // the whitespace/comment bytes scanned below are only peeked and get re-lexed (the
    // newline is still consumed by `extras`, the comment re-lexed as a `comment` node).
    lexer->result_symbol = NEWLINE;
    lexer->mark_end(lexer);

    for (;;) {
        int32_t c = lexer->lookahead;
        if (c == '\n') return true;                 // a line break ahead -> insert a terminator
        if (c == ' ' || c == '\t' || c == '\r') {   // horizontal whitespace: keep looking
            lexer->advance(lexer, true);
            continue;
        }
        if (c == '#') {                             // a `#` line comment runs to the newline
            while (lexer->lookahead != '\n' && lexer->lookahead != 0) {
                lexer->advance(lexer, true);
            }
            continue;
        }
        return false;                               // a real token on this line -> no terminator
    }
}
