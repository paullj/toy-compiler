//! A recursive-descent program parser with a Pratt expression core.
//!
//! A source file is a sequence of function declarations; each function has a
//! parameter list, an optional `-> Type`, and a brace-delimited block of
//! statements (`name := expr`, `name = expr`, `return expr?`, or a bare
//! expression statement). Statements are separated by the lexer-inserted
//! `.newline` terminator (Go-style ASI); a terminator before `}` or EOF is
//! optional. Expressions are parsed by precedence climbing, extended with a
//! call postfix `callee(args...)` that binds tighter than any infix operator.
//!
//! Parsing is sequential within one file, which is fine: our parallelism is
//! per-file (the driver runs one parser per file on its own thread). The parser
//! is allocation-light — it appends to a `[]Node` and a `[]u32` `extra` side
//! array (the variable-arity child runs) and holds no IO — so its `Tree` output
//! caches and threads exactly like the lexer's tokens.
//!
//! Invariant kept for the cache/root contract: child nodes are always appended
//! before their parents, and the `program` node is appended last, so
//! `Ast.root` is the last node. To preserve this with the `extra` side array we
//! never hold two ranges open across a recursive call — each producer collects
//! its children into a local list, parses the whole subtree, then writes its
//! `extra` run in one shot (see `addRange`).

const std = @import("std");
const token = @import("ast/Token.zig");
const Token = token.Token;
const Ast = @import("ast/Ast.zig");
const Node = Ast.Node;

const Parser = @This();

gpa: std.mem.Allocator,
tokens: []const Token,
/// Cursor into `tokens`.
index: u32,
nodes: std.ArrayList(Node),
/// Variable-arity child runs and range/proto headers (the "extra_data" side
/// array). See `Ast` for the encoding.
extra: std.ArrayList(u32),
/// Set when parsing fails; describes the first error.
diag: ?Diagnostic,
/// While true (only while parsing an `if`/`while` condition), a leading `{` or
/// `if` in `parsePrefix` is NOT taken as a block/if expression, so `if c { ... }`
/// reads `c` as the condition and `{ ... }` as the body. Parens are the escape
/// hatch. Reset (save/restore) inside `parseBlock` so a block body re-arms it.
no_block: bool = false,

pub const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;

const Error = error{ OutOfMemory, ParseFailed };

/// Parse a whole file into a `Tree`. On success returns the owned tree (root is
/// the last node, a `.program`). On a parse error returns null and fills `diag`.
pub fn parse(gpa: std.mem.Allocator, tokens: []const Token, diag: *?Diagnostic) error{OutOfMemory}!?Ast.Tree {
    var p: Parser = .{
        .gpa = gpa,
        .tokens = tokens,
        .index = 0,
        .nodes = .empty,
        .extra = .empty,
        .diag = null,
    };
    return p.parseProgram() catch |err| switch (err) {
        error.OutOfMemory => {
            p.deinit();
            return error.OutOfMemory;
        },
        error.ParseFailed => {
            diag.* = p.diag.?;
            p.deinit();
            return null;
        },
    };
}

fn deinit(p: *Parser) void {
    p.nodes.deinit(p.gpa);
    p.extra.deinit(p.gpa);
}

// ---- program / declarations ------------------------------------------------

fn parseProgram(p: *Parser) Error!Ast.Tree {
    var decls: std.ArrayList(Ast.Index) = .empty;
    defer decls.deinit(p.gpa);

    while (p.peek().tag == .newline) p.advance();
    while (p.peek().tag != .eof) {
        const decl = switch (p.peek().tag) {
            .kw_fn => try p.parseFnDecl(),
            .kw_struct => try p.parseStructDecl(),
            else => return p.fail(p.peek(), "expected a function or struct declaration"),
        };
        try decls.append(p.gpa, decl);
        while (p.peek().tag == .newline) p.advance();
    }
    try p.expect(.eof, "expected a function or struct declaration or end of input");

    const header = try p.addRange(decls.items);
    _ = try p.addNode(.{ .tag = .program, .main_token = 0, .lhs = header, .rhs = Ast.none });
    return Ast.Tree{
        .nodes = try p.nodes.toOwnedSlice(p.gpa),
        .extra = try p.extra.toOwnedSlice(p.gpa),
    };
}

fn parseFnDecl(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_fn, "expected 'fn'");
    const name_tok = p.index;
    try p.expect(.identifier, "expected a function name");
    try p.expect(.l_paren, "expected '(' after function name");

    var params: std.ArrayList(Ast.Index) = .empty;
    defer params.deinit(p.gpa);
    while (p.peek().tag != .r_paren) {
        const param_name = p.index;
        try p.expect(.identifier, "expected a parameter name");
        try p.expect(.colon, "expected ':' after parameter name");
        const type_node = try p.parseType();
        const param = try p.addNode(.{ .tag = .param, .main_token = param_name, .lhs = type_node, .rhs = Ast.none });
        try params.append(p.gpa, param);
        if (p.peek().tag == .comma) p.advance() else break;
    }
    try p.expect(.r_paren, "expected ')' to close parameter list");

    var ret_type: Ast.Index = Ast.none;
    if (p.peek().tag == .arrow) {
        p.advance();
        ret_type = try p.parseType();
    }

    const body = try p.parseBlock();

    // Write the params run, then the fixed 3-cell FnProto immediately after.
    const params_start: u32 = @intCast(p.extra.items.len);
    try p.extra.appendSlice(p.gpa, params.items);
    const proto_header = try p.addExtra(&.{ ret_type, params_start, @intCast(params.items.len) });

    return p.addNode(.{ .tag = .fn_decl, .main_token = name_tok, .lhs = proto_header, .rhs = body });
}

/// `struct Name { x: int, y: int }`. Fields are `name: Type`, comma-separated,
/// with newlines insignificant inside `{}`. Reuses the `.param` node for fields.
fn parseStructDecl(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_struct, "expected 'struct'");
    const name_tok = p.index;
    try p.expect(.identifier, "expected a struct name");
    try p.expect(.l_brace, "expected '{' after struct name");

    var fields: std.ArrayList(Ast.Index) = .empty;
    defer fields.deinit(p.gpa);
    while (true) {
        while (p.peek().tag == .newline) p.advance();
        if (p.peek().tag == .r_brace) break;
        const field_name = p.index;
        try p.expect(.identifier, "expected a field name");
        try p.expect(.colon, "expected ':' after field name");
        const type_node = try p.parseType();
        const field = try p.addNode(.{ .tag = .param, .main_token = field_name, .lhs = type_node, .rhs = Ast.none });
        try fields.append(p.gpa, field);
        // A field is separated by a comma OR a newline (both insignificant inside
        // `{}`); a `}` ends the list. An optional comma is consumed; the loop top
        // skips newlines and checks for `}`.
        if (p.peek().tag == .comma) p.advance();
    }
    try p.expect(.r_brace, "expected '}' to close struct body");

    const header = try p.addRange(fields.items);
    return p.addNode(.{ .tag = .struct_decl, .main_token = name_tok, .lhs = header, .rhs = Ast.none });
}

/// `Name { x: 1, y: 2 }` (or punning `Name { x, y }`). `name_ident` is the
/// already-parsed type-name `identifier` node. The `{` opens a fresh expression
/// context (reset `no_block`) so nested exprs and literals parse.
fn parseStructLiteral(p: *Parser, name_ident: Ast.Index) Error!Ast.Index {
    const lbrace = p.index;
    p.advance(); // {
    const saved_nb = p.no_block;
    p.no_block = false;
    defer p.no_block = saved_nb;

    var inits: std.ArrayList(Ast.Index) = .empty;
    defer inits.deinit(p.gpa);
    while (true) {
        while (p.peek().tag == .newline) p.advance();
        if (p.peek().tag == .r_brace) break;
        const field_tok = p.index;
        try p.expect(.identifier, "expected a field name");
        var value: Ast.Index = undefined;
        if (p.peek().tag == .colon) {
            p.advance(); // :
            value = try p.parseExpr(0);
        } else {
            // Punning shorthand: synthesize an identifier leaf on the field token.
            value = try p.addNode(.{ .tag = .identifier, .main_token = field_tok, .lhs = Ast.none, .rhs = Ast.none });
        }
        const fi = try p.addNode(.{ .tag = .field_init, .main_token = field_tok, .lhs = value, .rhs = Ast.none });
        try inits.append(p.gpa, fi);
        if (p.peek().tag == .comma) p.advance();
    }
    try p.expect(.r_brace, "expected '}' to close struct literal");

    const header = try p.addRange(inits.items);
    return p.addNode(.{ .tag = .struct_init, .main_token = lbrace, .lhs = name_ident, .rhs = header });
}

/// `recv.field`. Consumes the `.` then the field-name identifier.
fn parseFieldAccess(p: *Parser, recv: Ast.Index) Error!Ast.Index {
    p.advance(); // .
    const field_tok = p.index;
    try p.expect(.identifier, "expected a field name after '.'");
    return p.addNode(.{ .tag = .field_access, .main_token = field_tok, .lhs = recv, .rhs = Ast.none });
}

/// A type reference is written as an identifier (e.g. `int`, `bool`, `str`), or
/// the unit type `()`.
fn parseType(p: *Parser) Error!Ast.Index {
    if (p.peek().tag == .l_paren and p.peek2().tag == .r_paren) {
        const at = p.index;
        p.advance(); // (
        p.advance(); // )
        return p.addNode(.{ .tag = .literal_unit, .main_token = at, .lhs = Ast.none, .rhs = Ast.none });
    }
    const at = p.index;
    try p.expect(.identifier, "expected a type name");
    return p.addNode(.{ .tag = .identifier, .main_token = at, .lhs = Ast.none, .rhs = Ast.none });
}

fn parseBlock(p: *Parser) Error!Ast.Index {
    // A block body is a fresh expression context: re-allow `{`/`if` expressions
    // inside it even when reached from an `if`/`while` condition.
    const saved_nb = p.no_block;
    p.no_block = false;
    defer p.no_block = saved_nb;
    const lbrace = p.index;
    try p.expect(.l_brace, "expected '{' to open a block");

    var stmts: std.ArrayList(Ast.Index) = .empty;
    defer stmts.deinit(p.gpa);
    while (true) {
        while (p.peek().tag == .newline) p.advance();
        if (p.peek().tag == .r_brace) break;
        if (p.peek().tag == .eof) return p.fail(p.peek(), "expected '}' to close block");
        const stmt = try p.parseStmt();
        try stmts.append(p.gpa, stmt);
        try p.expectTerminator();
    }
    try p.expect(.r_brace, "expected '}' to close block");

    const header = try p.addRange(stmts.items);
    return p.addNode(.{ .tag = .block, .main_token = lbrace, .lhs = header, .rhs = Ast.none });
}

/// A statement ends at a `.newline`, or implicitly before `}`/EOF (Go ASI).
fn expectTerminator(p: *Parser) Error!void {
    switch (p.peek().tag) {
        .newline => {
            p.advance();
            while (p.peek().tag == .newline) p.advance();
        },
        .r_brace, .eof => {},
        else => return p.fail(p.peek(), "expected a newline or '}' after statement"),
    }
}

// ---- statements ------------------------------------------------------------

fn parseStmt(p: *Parser) Error!Ast.Index {
    const tok = p.peek();
    switch (tok.tag) {
        .kw_if => return p.parseIf(),
        .kw_while => return p.parseWhile(),
        .kw_for => return p.parseFor(),
        .kw_break => {
            const break_tok = p.index;
            p.advance();
            // A `@name` label may immediately follow `break`; parse it BEFORE the
            // value-terminator decision (the documented ordering hazard).
            const label_tok = try p.parseOptLabel();
            const expr: Ast.Index = switch (p.peek().tag) {
                .newline, .r_brace, .eof => Ast.none,
                else => try p.parseExpr(0),
            };
            return p.addNode(.{ .tag = .break_stmt, .main_token = break_tok, .lhs = expr, .rhs = label_tok });
        },
        .kw_continue => {
            const continue_tok = p.index;
            p.advance();
            const label_tok = try p.parseOptLabel();
            return p.addNode(.{ .tag = .continue_stmt, .main_token = continue_tok, .lhs = Ast.none, .rhs = label_tok });
        },
        // `@label <construct>` as a statement routes through parseExprStmt (like a
        // bare `loop`/`if`/block), so a trailing labeled loop/block is wrapped in an
        // expr_stmt and its value can satisfy a non-unit fn's trailing-expr rule.
        .kw_return => {
            const ret_tok = p.index;
            p.advance();
            const expr: Ast.Index = switch (p.peek().tag) {
                .newline, .r_brace, .eof => Ast.none,
                else => try p.parseExpr(0),
            };
            return p.addNode(.{ .tag = .return_stmt, .main_token = ret_tok, .lhs = expr, .rhs = Ast.none });
        },
        .identifier => switch (p.peek2().tag) {
            .colon_eq => {
                const name_tok = p.index;
                p.advance(); // name
                p.advance(); // :=
                const init_expr = try p.parseExpr(0);
                return p.addNode(.{ .tag = .var_decl, .main_token = name_tok, .lhs = init_expr, .rhs = Ast.none });
            },
            .eq => {
                const name_tok = p.index;
                const target = try p.leaf(.identifier, name_tok); // advances past name
                try p.expect(.eq, "expected '=' in assignment");
                const value = try p.parseExpr(0);
                return p.addNode(.{ .tag = .assign, .main_token = name_tok, .lhs = target, .rhs = value });
            },
            // A `.`-rooted place: `p.x = v`, `p.a.b = v`. Parse the place (a
            // postfix chain over a name); if `=` follows it is a field store,
            // otherwise it is an expression statement (continue the infix climb).
            .dot => {
                const first = p.index;
                const place = try p.parsePostfix(try p.parsePrefix());
                if (p.peek().tag == .eq) {
                    p.advance(); // =
                    const value = try p.parseExpr(0);
                    return p.addNode(.{ .tag = .assign, .main_token = first, .lhs = place, .rhs = value });
                }
                const expr = try p.continueInfix(place, 0);
                return p.addNode(.{ .tag = .expr_stmt, .main_token = first, .lhs = expr, .rhs = Ast.none });
            },
            else => return p.parseExprStmt(),
        },
        else => return p.parseExprStmt(),
    }
}

/// `while cond { body }`. The condition is a full expression (no parens
/// required, Go-style); the body is a brace block.
fn parseWhile(p: *Parser) Error!Ast.Index {
    const while_tok = p.index;
    p.advance(); // while
    p.no_block = true;
    const cond = try p.parseExpr(0);
    p.no_block = false;
    const body = try p.parseBlock();
    return p.addNode(.{ .tag = .while_stmt, .main_token = while_tok, .lhs = cond, .rhs = body });
}

/// `if cond { then } [else (block | if)]`. Recurses for `else if` chains: an
/// `else` may be followed by a block or another `if`. Go-style, the `else` must
/// sit on the same line as the closing `}` (the lexer inserts a `.newline` after
/// `}` only when a real newline byte follows, so `} else` on one line keeps
/// `.kw_else` as the immediate next token).
fn parseIf(p: *Parser) Error!Ast.Index {
    const if_tok = p.index;
    p.advance(); // if
    p.no_block = true;
    const cond = try p.parseExpr(0);
    p.no_block = false;
    const then_block = try p.parseBlock();
    var else_node: Ast.Index = Ast.none;
    if (p.peek().tag == .kw_else) {
        p.advance(); // else
        else_node = if (p.peek().tag == .kw_if) try p.parseIf() else try p.parseBlock();
    }
    const header = try p.addExtra(&.{ then_block, else_node });
    return p.addNode(.{ .tag = .if_stmt, .main_token = if_tok, .lhs = cond, .rhs = header });
}

/// If the cursor is at `@name`, consume both and return the identifier token
/// index; otherwise consume nothing and return `Ast.none`. Used for the optional
/// label on `break`/`continue`.
fn parseOptLabel(p: *Parser) Error!Ast.Index {
    if (p.peek().tag != .at) return Ast.none;
    p.advance(); // @
    const name_tok = p.index;
    try p.expect(.identifier, "expected a label name after '@'");
    return name_tok;
}

/// `@name <loop|while|for|block>`: a label prefixed onto a block-like construct.
/// The inner construct is built first (children precede parents), then wrapped in
/// a `labeled` node whose `main_token` is the identifier after `@`.
fn parseLabeled(p: *Parser) Error!Ast.Index {
    try p.expect(.at, "expected '@'");
    const name_tok = p.index;
    try p.expect(.identifier, "expected a label name after '@'");
    const inner: Ast.Index = switch (p.peek().tag) {
        .kw_loop => try p.parseLoop(),
        .kw_while => try p.parseWhile(),
        .kw_for => try p.parseFor(),
        .l_brace => try p.parseBlock(),
        else => return p.fail(p.peek(), "a label must prefix a loop, while, for, or block"),
    };
    return p.addNode(.{ .tag = .labeled, .main_token = name_tok, .lhs = inner, .rhs = Ast.none });
}

/// `loop { body }`. A value-yielding infinite loop expression.
fn parseLoop(p: *Parser) Error!Ast.Index {
    const loop_tok = p.index;
    p.advance(); // loop
    const body = try p.parseBlock();
    return p.addNode(.{ .tag = .loop_expr, .main_token = loop_tok, .lhs = body, .rhs = Ast.none });
}

/// `for ident in lo..hi { body }`. Iterates the half-open integer range
/// `[lo, hi)` with `ident: int` bound per-iteration. A `()` statement.
fn parseFor(p: *Parser) Error!Ast.Index {
    p.advance(); // for
    const ident_tok = p.index;
    try p.expect(.identifier, "expected a loop variable name");
    try p.expect(.kw_in, "expected 'in' after the loop variable");
    p.no_block = true;
    const lo = try p.parseExpr(0); // halts at `..` (no infix bp)
    try p.expect(.dotdot, "expected '..' in the for range");
    const hi = try p.parseExpr(0);
    p.no_block = false;
    const body = try p.parseBlock(); // re-arms no_block internally
    const header = try p.addExtra(&.{ lo, hi }); // children before parent
    return p.addNode(.{ .tag = .for_stmt, .main_token = ident_tok, .lhs = body, .rhs = header });
}

fn parseExprStmt(p: *Parser) Error!Ast.Index {
    const first = p.index;
    const expr = try p.parseExpr(0);
    return p.addNode(.{ .tag = .expr_stmt, .main_token = first, .lhs = expr, .rhs = Ast.none });
}

// ---- expressions (Pratt) ---------------------------------------------------

/// Precedence-climbing core. `min_bp` is the minimum binding power that an infix
/// operator must exceed to bind here; recursing with the operator's own bp makes
/// operators left-associative. The call postfix is applied to every operand
/// before the infix loop, so it binds tighter than any infix operator.
fn parseExpr(p: *Parser, min_bp: u8) Error!Ast.Index {
    const lhs = try p.parsePostfix(try p.parsePrefix());
    return p.continueInfix(lhs, min_bp);
}

/// The infix precedence-climbing loop, starting from an already-parsed `lhs`.
/// Factored out so the statement parser can resume an infix climb after probing
/// a `.`-rooted place that turned out not to be an assignment target.
fn continueInfix(p: *Parser, lhs0: Ast.Index, min_bp: u8) Error!Ast.Index {
    var lhs = lhs0;
    while (infixBp(p.peek().tag)) |bp| {
        if (bp <= min_bp) break;
        const op = p.index;
        p.advance();
        const rhs = try p.parseExpr(bp);
        lhs = try p.addNode(.{ .tag = .binary, .main_token = op, .lhs = lhs, .rhs = rhs });
    }
    return lhs;
}

fn parsePrefix(p: *Parser) Error!Ast.Index {
    const tok = p.peek();
    const at = p.index;
    switch (tok.tag) {
        .minus, .bang => {
            p.advance();
            const operand = try p.parseExpr(prefix_bp);
            return p.addNode(.{ .tag = .unary, .main_token = at, .lhs = operand, .rhs = Ast.none });
        },
        .number => return p.leaf(.literal_number, at),
        .string => return p.leaf(.literal_string, at),
        .kw_true, .kw_false => return p.leaf(.literal_bool, at),
        .identifier => return p.leaf(.identifier, at),
        .l_paren => {
            p.advance();
            if (p.peek().tag == .r_paren) { // the unit literal `()`
                p.advance(); // )
                return p.addNode(.{ .tag = .literal_unit, .main_token = at, .lhs = Ast.none, .rhs = Ast.none });
            }
            // A grouped sub-expression re-allows blocks (the escape hatch out of a
            // condition's `no_block`); restore the flag after.
            const saved_nb = p.no_block;
            p.no_block = false;
            const inner = try p.parseExpr(0);
            p.no_block = saved_nb;
            try p.expect(.r_paren, "expected ')' to close group");
            return inner;
        },
        .l_brace => {
            if (p.no_block) return p.fail(tok, "expected an expression");
            return p.parseBlock(); // a bare block as a value expression
        },
        .kw_if => {
            if (p.no_block) return p.fail(tok, "expected an expression");
            return p.parseIf(); // an if as a value expression
        },
        .kw_loop => {
            if (p.no_block) return p.fail(tok, "expected an expression");
            return p.parseLoop(); // a loop as a value expression
        },
        .at => {
            if (p.no_block) return p.fail(tok, "expected an expression");
            return p.parseLabeled(); // a labeled loop/block as a value expression
        },
        else => return p.fail(tok, "expected an expression"),
    }
}

/// Apply zero or more call postfixes to a primary expression, so `f(x)(y)` and
/// `f(g(x))` compose.
fn parsePostfix(p: *Parser, lhs0: Ast.Index) Error!Ast.Index {
    var lhs = lhs0;
    while (true) {
        switch (p.peek().tag) {
            .l_paren => lhs = try p.parseCall(lhs),
            // `.field` access. `..` is a separate token, so `0..5` is unaffected.
            .dot => lhs = try p.parseFieldAccess(lhs),
            // `Name { ... }` struct literal — only when blocks are allowed (so an
            // `if c { ... }` condition reads `c` not `c{...}`) and `lhs` is a bare
            // name. The call/group `( )` reset `no_block`, so `f(P{x:1})` works.
            .l_brace => {
                if (p.no_block or p.nodes.items[lhs].tag != .identifier) break;
                lhs = try p.parseStructLiteral(lhs);
            },
            else => break,
        }
    }
    return lhs;
}

fn parseCall(p: *Parser, callee: Ast.Index) Error!Ast.Index {
    const lparen = p.index;
    p.advance(); // (
    var args: std.ArrayList(Ast.Index) = .empty;
    defer args.deinit(p.gpa);
    // The call's `( )` open a fresh expression context, so re-allow blocks/if-exprs
    // in arguments even inside an if/while condition (`no_block`); restore after.
    const saved_nb = p.no_block;
    p.no_block = false;
    defer p.no_block = saved_nb;
    while (p.peek().tag != .r_paren) {
        try args.append(p.gpa, try p.parseExpr(0));
        if (p.peek().tag == .comma) p.advance() else break;
    }
    try p.expect(.r_paren, "expected ')' to close call");
    const header = try p.addRange(args.items);
    return p.addNode(.{ .tag = .call, .main_token = lparen, .lhs = callee, .rhs = header });
}

/// Infix binding power, or null if the tag is not an infix operator. Higher
/// binds tighter.
fn infixBp(tag: token.Tag) ?u8 {
    return switch (tag) {
        .pipe_pipe => 1,
        .amp_amp => 2,
        .eq_eq, .bang_eq => 3,
        .lt, .lt_eq, .gt, .gt_eq => 4,
        .plus, .minus => 5,
        .star, .slash => 6,
        else => null,
    };
}

/// Prefix operators bind tighter than any infix operator.
const prefix_bp: u8 = 7;

// ---- node / extra builders -------------------------------------------------

fn leaf(p: *Parser, tag: Node.Tag, tok_index: u32) Error!Ast.Index {
    p.advance();
    return p.addNode(.{ .tag = tag, .main_token = tok_index, .lhs = Ast.none, .rhs = Ast.none });
}

fn addNode(p: *Parser, node: Node) Error!Ast.Index {
    const idx: Ast.Index = @intCast(p.nodes.items.len);
    try p.nodes.append(p.gpa, node);
    return idx;
}

/// Append a run of node indices then a two-cell `{start, len}` header; return
/// the header cell index (what the parent `Node` stores).
fn addRange(p: *Parser, items: []const Ast.Index) Error!u32 {
    const start: u32 = @intCast(p.extra.items.len);
    try p.extra.appendSlice(p.gpa, items);
    const header: u32 = @intCast(p.extra.items.len);
    try p.extra.append(p.gpa, start);
    try p.extra.append(p.gpa, @intCast(items.len));
    return header;
}

/// Append raw cells; return the index of the first.
fn addExtra(p: *Parser, vals: []const u32) Error!u32 {
    const at: u32 = @intCast(p.extra.items.len);
    try p.extra.appendSlice(p.gpa, vals);
    return at;
}

// ---- cursor ----------------------------------------------------------------

fn peek(p: *const Parser) Token {
    return p.tokens[p.index];
}

/// One-token lookahead, clamped to the trailing `.eof`.
fn peek2(p: *const Parser) Token {
    const i = p.index + 1;
    return p.tokens[if (i < p.tokens.len) i else p.tokens.len - 1];
}

fn advance(p: *Parser) void {
    // The token stream always ends in `.eof`; never step past it.
    if (p.tokens[p.index].tag != .eof) p.index += 1;
}

fn expect(p: *Parser, tag: token.Tag, message: []const u8) Error!void {
    const tok = p.peek();
    if (tok.tag != tag) return p.fail(tok, message);
    p.advance();
}

fn fail(p: *Parser, tok: Token, message: []const u8) error{ParseFailed} {
    p.diag = .{ .byte_offset = tok.start, .message = message };
    return error.ParseFailed;
}

// ---- tests -----------------------------------------------------------------

const testing = std.testing;
const Lexer = @import("lex.zig");

/// Test-only: parse a single bare expression (the pre-M0 grammar) into a Tree,
/// so the expression-core tests below keep asserting on raw expressions without
/// the program/fn scaffolding.
fn parseExprOnly(gpa: std.mem.Allocator, tokens: []const Token, diag: *?Diagnostic) error{OutOfMemory}!?Ast.Tree {
    var p: Parser = .{
        .gpa = gpa,
        .tokens = tokens,
        .index = 0,
        .nodes = .empty,
        .extra = .empty,
        .diag = null,
    };
    const run = struct {
        fn go(pp: *Parser) Error!Ast.Tree {
            _ = try pp.parseExpr(0);
            while (pp.peek().tag == .newline) pp.advance();
            try pp.expect(.eof, "expected end of input");
            return Ast.Tree{
                .nodes = try pp.nodes.toOwnedSlice(pp.gpa),
                .extra = try pp.extra.toOwnedSlice(pp.gpa),
            };
        }
    }.go;
    return run(&p) catch |err| switch (err) {
        error.OutOfMemory => {
            p.deinit();
            return error.OutOfMemory;
        },
        error.ParseFailed => {
            diag.* = p.diag.?;
            p.deinit();
            return null;
        },
    };
}

fn freeTree(gpa: std.mem.Allocator, tree: Ast.Tree) void {
    gpa.free(tree.nodes);
    gpa.free(tree.extra);
}

fn expectSexpr(source: []const u8, want: []const u8) !void {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);

    var diag: ?Diagnostic = null;
    const tree = (try parseExprOnly(gpa, tokens, &diag)) orelse return error.UnexpectedParseFailure;
    defer freeTree(gpa, tree);

    var buf: [256]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try Ast.render(&w, tree, tokens, source);
    try testing.expectEqualStrings(want, w.buffered());
}

/// Parse a full program and assert its rendered S-expression.
fn expectProgram(source: []const u8, want: []const u8) !void {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);

    var diag: ?Diagnostic = null;
    const tree = (try parse(gpa, tokens, &diag)) orelse return error.UnexpectedParseFailure;
    defer freeTree(gpa, tree);

    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try Ast.render(&w, tree, tokens, source);
    try testing.expectEqualStrings(want, w.buffered());
}

// expression core (unchanged grammar, via parseExprOnly)

test "precedence: * binds tighter than +" {
    try expectSexpr("1 + 2 * 3", "(+ 1 (* 2 3))");
}

test "left associativity" {
    try expectSexpr("1 - 2 - 3", "(- (- 1 2) 3)");
}

test "unary binds tighter than binary" {
    try expectSexpr("-1 + 2", "(+ (- 1) 2)");
}

test "grouping overrides precedence" {
    try expectSexpr("(1 + 2) * 3", "(* (+ 1 2) 3)");
}

test "comparison and equality precedence" {
    try expectSexpr("1 + 2 == 3 < 4", "(== (+ 1 2) (< 3 4))");
}

test "trailing newline terminator is allowed" {
    try expectSexpr("a * b\n", "(* a b)");
}

test "parse error reports an offset and leaves a diagnostic" {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, "1 +");
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const result = try parseExprOnly(gpa, tokens, &diag);
    try testing.expect(result == null);
    try testing.expect(diag != null);
    try testing.expectEqualStrings("expected an expression", diag.?.message);
}

// program grammar

test "fn with params, return type, binary body" {
    try expectProgram(
        "fn add(a: int, b: int) -> int {\n return a + b\n}\n",
        "(program (fn add ((param a int) (param b int)) int (block (return (+ a b)))))",
    );
}

test "var_decl, call, assign, bare return" {
    try expectProgram(
        "fn main() {\n x := 1\n x = add(x, 2)\n return\n}\n",
        "(program (fn main () _ (block (:= x 1) (= x (call add x 2)) (return))))",
    );
}

test "empty block, no params, void return" {
    try expectProgram("fn f() {}\n", "(program (fn f () _ (block)))");
}

test "last statement without a trailing newline before }" {
    try expectProgram(
        "fn f() -> int { return 1 }\n",
        "(program (fn f () int (block (return 1))))",
    );
}

test "nested call precedence" {
    try expectProgram(
        "fn f() -> int { return g(1) + 2 }\n",
        "(program (fn f () int (block (return (+ (call g 1) 2)))))",
    );
}

test "root is program and children precede parents" {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, "fn add(a: int, b: int) -> int {\n return a + b\n}\n");
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const tree = (try parse(gpa, tokens, &diag)) orelse return error.UnexpectedParseFailure;
    defer freeTree(gpa, tree);

    try testing.expectEqual(Node.Tag.program, tree.nodes[Ast.root(tree.nodes)].tag);
    // Topological: every child node index is strictly less than its parent's.
    for (tree.nodes, 0..) |n, i| {
        const self: u32 = @intCast(i);
        switch (n.tag) {
            .unary => try testing.expect(n.lhs < self),
            .binary, .assign => {
                try testing.expect(n.lhs < self);
                try testing.expect(n.rhs < self);
            },
            .var_decl, .expr_stmt => try testing.expect(n.lhs < self),
            .return_stmt => if (n.lhs != Ast.none) try testing.expect(n.lhs < self),
            .param => try testing.expect(n.lhs < self),
            .call => {
                try testing.expect(n.lhs < self);
                for (Ast.rangeSlice(tree, n.rhs)) |c| try testing.expect(c < self);
            },
            .block => for (Ast.rangeSlice(tree, n.lhs)) |c| try testing.expect(c < self),
            .program => for (Ast.rangeSlice(tree, n.lhs)) |c| try testing.expect(c < self),
            .fn_decl => {
                const proto = Ast.protoAt(tree, n.lhs);
                if (proto.ret_type != Ast.none) try testing.expect(proto.ret_type < self);
                for (proto.params) |c| try testing.expect(c < self);
                try testing.expect(n.rhs < self);
            },
            .while_stmt => {
                try testing.expect(n.lhs < self);
                try testing.expect(n.rhs < self);
            },
            .if_stmt => {
                try testing.expect(n.lhs < self);
                const h = Ast.ifHeaderAt(tree, n.rhs);
                try testing.expect(h.then_block < self);
                if (h.else_node != Ast.none) try testing.expect(h.else_node < self);
            },
            .literal_unit => {},
            .loop_expr => try testing.expect(n.lhs < self),
            .for_stmt => {
                try testing.expect(n.lhs < self);
                const h = Ast.forHeaderAt(tree, n.rhs);
                try testing.expect(h.lo < self);
                try testing.expect(h.hi < self);
            },
            // break/continue overload `rhs` as a *token* index (the label), so
            // only `lhs` (the value expr) is a child node to check.
            .break_stmt => if (n.lhs != Ast.none) try testing.expect(n.lhs < self),
            .continue_stmt => {},
            .labeled => try testing.expect(n.lhs < self),
            .struct_decl => for (Ast.rangeSlice(tree, n.lhs)) |c| try testing.expect(c < self),
            .struct_init => {
                try testing.expect(n.lhs < self);
                for (Ast.rangeSlice(tree, n.rhs)) |c| try testing.expect(c < self);
            },
            .field_init => try testing.expect(n.lhs < self),
            .field_access => try testing.expect(n.lhs < self),
            else => {},
        }
    }
}

test "logical operator precedence: || is loosest, && next, then equality" {
    try expectSexpr("a || b && c == d", "(|| a (&& b (== c d)))");
}

test "logical operators are left-associative" {
    try expectSexpr("a || b || c", "(|| (|| a b) c)");
}

test "if statement parses to an if_stmt" {
    try expectProgram(
        "fn f() {\n if c { a }\n return\n}\n",
        "(program (fn f () _ (block (if c (block a)) (return))))",
    );
}

test "if/else parses with an else block" {
    try expectProgram(
        "fn f() {\n if c { a } else { b }\n return\n}\n",
        "(program (fn f () _ (block (if c (block a) (block b)) (return))))",
    );
}

test "else if chains as a nested if_stmt" {
    try expectProgram(
        "fn f() {\n if a {} else if b {} else {}\n return\n}\n",
        "(program (fn f () _ (block (if a (block) (if b (block) (block))) (return))))",
    );
}

test "while statement parses to a while_stmt" {
    try expectProgram(
        "fn f() {\n while c { a }\n return\n}\n",
        "(program (fn f () _ (block (while c (block a)) (return))))",
    );
}

test "unit type renders as ()" {
    try expectProgram("fn f() -> () {}\n", "(program (fn f () () (block)))");
}

test "if as an expression renders as an if node" {
    try expectProgram(
        "fn f() -> int {\n x := if a { 1 } else { 2 }\n return x\n}\n",
        "(program (fn f () int (block (:= x (if a (block 1) (block 2))) (return x))))",
    );
}

test "bare block expression renders as a block node" {
    try expectProgram(
        "fn f() -> int {\n x := { a := 1\n a + 1 }\n return x\n}\n",
        "(program (fn f () int (block (:= x (block (:= a 1) (+ a 1))) (return x))))",
    );
}

test "trailing-expression body parses with a trailing expr_stmt" {
    try expectProgram("fn f() -> int { 41 + 1 }\n", "(program (fn f () int (block (+ 41 1))))");
}

test "block/if expr in a call argument inside an if condition (call reopens block context)" {
    // `no_block` is set parsing the condition, but the call's `( )` open a fresh
    // expression context — the `{ 1 }` argument must parse, not error.
    try expectProgram(
        "fn g(x: int) -> int { x }\nfn main() -> int {\n if g({ 1 }) > 0 { 7 } else { 8 }\n}\n",
        "(program (fn g ((param x int)) int (block x)) (fn main () int (block (if (> (call g (block 1)) 0) (block 7) (block 8)))))",
    );
}

test "loop expression parses to a loop_expr" {
    try expectProgram(
        "fn f() -> int {\n loop { break 1 }\n}\n",
        "(program (fn f () int (block (loop (block (break 1))))))",
    );
}

test "for range parses to a for_stmt" {
    try expectProgram(
        "fn f() {\n for i in 0..5 { x = i }\n return\n}\n",
        "(program (fn f () _ (block (for i 0 5 (block (= x i))) (return))))",
    );
}

test "bare break and continue parse" {
    try expectProgram(
        "fn f() {\n loop { break }\n while c { continue }\n return\n}\n",
        "(program (fn f () _ (block (loop (block (break))) (while c (block (continue))) (return))))",
    );
}

test "value-break parses with its expression" {
    try expectProgram(
        "fn f() -> int {\n loop { break 1 + 2 }\n}\n",
        "(program (fn f () int (block (loop (block (break (+ 1 2)))))))",
    );
}

test "labeled loop parses to a labeled wrapper" {
    try expectProgram(
        "fn f() -> int {\n @outer loop { break @outer 1 }\n}\n",
        "(program (fn f () int (block (label outer (loop (block (break @outer 1)))))))",
    );
}

test "labeled while/for parse" {
    try expectProgram(
        "fn f() {\n @w while c { continue @w }\n @l for i in 0..3 { break @l }\n return\n}\n",
        "(program (fn f () _ (block (label w (while c (block (continue @w)))) (label l (for i 0 3 (block (break @l)))) (return))))",
    );
}

test "labeled bare block as an expression" {
    try expectProgram(
        "fn f() -> int {\n x := @calc { 3 }\n return x\n}\n",
        "(program (fn f () int (block (:= x (label calc (block 3))) (return x))))",
    );
}

test "labeled block as a call argument" {
    try expectProgram(
        "fn g(x: int) -> int { x }\nfn f() -> int { g(@b { 1 }) }\n",
        "(program (fn g ((param x int)) int (block x)) (fn f () int (block (call g (label b (block 1))))))",
    );
}

test "bare break and labeled break/continue render" {
    try expectProgram(
        "fn f() {\n @o loop { break @o\n continue @o }\n return\n}\n",
        "(program (fn f () _ (block (label o (loop (block (break @o) (continue @o)))) (return))))",
    );
}

test "label without a following construct is a parse error" {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, "fn f() {\n @x 1\n}\n");
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const result = try parse(gpa, tokens, &diag);
    try testing.expect(result == null);
    try testing.expect(diag != null);
}

// structs

test "struct declaration parses to a struct_decl" {
    try expectProgram(
        "struct Point { x: int, y: int }\n",
        "(program (struct Point (param x int) (param y int)))",
    );
}

test "struct declaration with newlines instead of commas" {
    try expectProgram(
        "struct Point {\n x: int\n y: int\n}\n",
        "(program (struct Point (param x int) (param y int)))",
    );
}

test "named struct construction" {
    try expectProgram(
        "fn f() -> int { p := Point { x: 1, y: 2 }\n return p.x }\n",
        "(program (fn f () int (block (:= p (new Point (field x 1) (field y 2))) (return (. p x)))))",
    );
}

test "field punning shorthand synthesizes an identifier" {
    try expectProgram(
        "fn f() -> int { p := Point { x, y }\n return 0 }\n",
        "(program (fn f () int (block (:= p (new Point (field x x) (field y y))) (return 0))))",
    );
}

test "nested field access" {
    try expectProgram(
        "fn f() -> int { return p.a.b }\n",
        "(program (fn f () int (block (return (. (. p a) b)))))",
    );
}

test "field place-store" {
    try expectProgram(
        "fn f() { p.x = 5\n return }\n",
        "(program (fn f () _ (block (= (. p x) 5) (return))))",
    );
}

test "nested field place-store" {
    try expectProgram(
        "fn f() { p.a.b = 5\n return }\n",
        "(program (fn f () _ (block (= (. (. p a) b) 5) (return))))",
    );
}

test "mixed fn and struct declarations" {
    try expectProgram(
        "struct P { x: int }\nfn f() -> int { 0 }\n",
        "(program (struct P (param x int)) (fn f () int (block 0)))",
    );
}

test "while condition does not consume a struct literal body" {
    try expectProgram(
        "fn f() { while c { x = 1 }\n return }\n",
        "(program (fn f () _ (block (while c (block (= x 1))) (return))))",
    );
}

test "struct literal as a call argument (call reopens block context)" {
    try expectProgram(
        "fn g(p: Point) -> int { 0 }\nfn f() -> int { g(Point { x: 1 }) }\n",
        "(program (fn g ((param p Point)) int (block 0)) (fn f () int (block (call g (new Point (field x 1))))))",
    );
}

test "struct program pack/unpack byte round-trip" {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, "struct Point { x: int, y: int }\nfn f() -> int { p := Point { x: 1, y: 2 }\n p.x }\n");
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const tree = (try parse(gpa, tokens, &diag)) orelse return error.UnexpectedParseFailure;
    defer freeTree(gpa, tree);

    const blob = try Ast.pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try Ast.unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer freeTree(gpa, got);

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
}

test "program pack/unpack byte round-trip" {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, "fn add(a: int, b: int) -> int {\n return a + b\n}\n");
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const tree = (try parse(gpa, tokens, &diag)) orelse return error.UnexpectedParseFailure;
    defer freeTree(gpa, tree);

    const blob = try Ast.pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try Ast.unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer freeTree(gpa, got);

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
}
