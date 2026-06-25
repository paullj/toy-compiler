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
/// Source bytes, for the rare token-TEXT check the parser needs (the `_`
/// wildcard pattern). Most of the parser works on token tags alone.
src: []const u8,
/// Cursor into `tokens`.
index: u32,
nodes: std.ArrayList(Node),
/// Node indices of top-level decls that carried a `pub` modifier (M14). Packed
/// into the `Tree.pub_bits` bitset after the `.program` node is appended.
pub_decls: std.ArrayList(Ast.Index),
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
pub fn parse(gpa: std.mem.Allocator, tokens: []const Token, source: []const u8, diag: *?Diagnostic) error{OutOfMemory}!?Ast.Tree {
    var p: Parser = .{
        .gpa = gpa,
        .tokens = tokens,
        .src = source,
        .index = 0,
        .nodes = .empty,
        .extra = .empty,
        .pub_decls = .empty,
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
    p.pub_decls.deinit(p.gpa);
}

// ---- program / declarations ------------------------------------------------

fn parseProgram(p: *Parser) Error!Ast.Tree {
    var decls: std.ArrayList(Ast.Index) = .empty;
    defer decls.deinit(p.gpa);

    while (p.peek().tag == .newline) p.advance();
    while (p.peek().tag != .eof) {
        // `import` decls have no `pub` modifier (imports are not re-exported).
        if (p.peek().tag == .kw_import) {
            try decls.append(p.gpa, try p.parseImport());
            while (p.peek().tag == .newline) p.advance();
            continue;
        }
        // An optional `pub` modifier precedes a fn/struct/enum decl and exports it.
        const is_pub = p.peek().tag == .kw_pub;
        if (is_pub) p.advance();
        const decl = switch (p.peek().tag) {
            .kw_fn => try p.parseFnDecl(),
            .kw_struct => try p.parseStructDecl(),
            .kw_enum => try p.parseEnumDecl(),
            else => return p.fail(p.peek(), if (is_pub)
                "expected a function, struct, or enum declaration after 'pub'"
            else
                "expected a function, struct, enum, or import declaration"),
        };
        if (is_pub) try p.pub_decls.append(p.gpa, decl);
        try decls.append(p.gpa, decl);
        while (p.peek().tag == .newline) p.advance();
    }
    try p.expect(.eof, "expected a declaration or end of input");

    const header = try p.addRange(decls.items);
    _ = try p.addNode(.{ .tag = .program, .main_token = 0, .lhs = header, .rhs = Ast.none });

    const nodes = try p.nodes.toOwnedSlice(p.gpa);
    errdefer p.gpa.free(nodes);
    const extra = try p.extra.toOwnedSlice(p.gpa);
    errdefer p.gpa.free(extra);
    const pub_bits = try p.buildPubBits(nodes.len);
    // The pub-decl scratch list is fully consumed into `pub_bits`; release it on
    // the success path (the error paths go through `p.deinit`).
    p.pub_decls.deinit(p.gpa);
    return Ast.Tree{ .nodes = nodes, .extra = extra, .pub_bits = pub_bits };
}

/// Materialize the `pub_bits` bitset from the collected `pub_decls` node indices.
/// Returns an empty slice when nothing is exported (the common single-file case),
/// so non-module programs pay nothing.
fn buildPubBits(p: *Parser, node_count: usize) Error![]u32 {
    if (p.pub_decls.items.len == 0) return &.{};
    const words = Ast.pubBitsLen(node_count);
    const bits = try p.gpa.alloc(u32, words);
    @memset(bits, 0);
    for (p.pub_decls.items) |idx| {
        bits[idx >> 5] |= @as(u32, 1) << @intCast(idx & 31);
    }
    return bits;
}

/// `import a/b/c [as alias]`. Path segments are `/`-separated identifiers. The
/// node stores the segment TOKEN indices (a `Range` in `extra`) and the alias
/// token (or `Ast.none`); `main_token` is the last segment (the default bind).
/// `/` appears ONLY here; `.` only in access — so there is no parse ambiguity.
fn parseImport(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_import, "expected 'import'");
    var segs: std.ArrayList(u32) = .empty;
    defer segs.deinit(p.gpa);
    const first = p.index;
    try p.expect(.identifier, "expected a module path after 'import'");
    try segs.append(p.gpa, first);
    while (p.peek().tag == .slash) {
        p.advance(); // /
        const seg = p.index;
        try p.expect(.identifier, "expected a path segment after '/'");
        try segs.append(p.gpa, seg);
    }
    var alias: Ast.Index = Ast.none;
    if (p.peek().tag == .kw_as) {
        p.advance(); // as
        alias = p.index;
        try p.expect(.identifier, "expected an alias name after 'as'");
    }
    const last_seg = segs.items[segs.items.len - 1];
    const header = try p.addRange(segs.items);
    return p.addNode(.{ .tag = .import_decl, .main_token = last_seg, .lhs = header, .rhs = alias });
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

/// `enum N { Empty, Circle(int), Rect { w: int, h: int } }`. Variants are
/// comma-separated, newlines insignificant inside `{}`. Three forms: unit,
/// tuple (positional payload types), struct (named `param` fields).
fn parseEnumDecl(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_enum, "expected 'enum'");
    const name_tok = p.index;
    try p.expect(.identifier, "expected an enum name");
    try p.expect(.l_brace, "expected '{' after enum name");

    var variants: std.ArrayList(Ast.Index) = .empty;
    defer variants.deinit(p.gpa);
    while (true) {
        while (p.peek().tag == .newline) p.advance();
        if (p.peek().tag == .r_brace) break;
        const vname = p.index;
        try p.expect(.identifier, "expected a variant name");
        var variant: Ast.Index = undefined;
        switch (p.peek().tag) {
            .l_paren => {
                p.advance(); // (
                var types: std.ArrayList(Ast.Index) = .empty;
                defer types.deinit(p.gpa);
                while (p.peek().tag != .r_paren) {
                    try types.append(p.gpa, try p.parseType());
                    if (p.peek().tag == .comma) p.advance() else break;
                }
                try p.expect(.r_paren, "expected ')' to close a tuple variant");
                const header = try p.addRange(types.items);
                variant = try p.addNode(.{ .tag = .enum_variant_tuple, .main_token = vname, .lhs = header, .rhs = Ast.none });
            },
            .l_brace => {
                const saved_nb = p.no_block;
                p.no_block = false;
                defer p.no_block = saved_nb;
                p.advance(); // {
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
                    if (p.peek().tag == .comma) p.advance();
                }
                try p.expect(.r_brace, "expected '}' to close a struct variant");
                const header = try p.addRange(fields.items);
                variant = try p.addNode(.{ .tag = .enum_variant_struct, .main_token = vname, .lhs = header, .rhs = Ast.none });
            },
            else => variant = try p.addNode(.{ .tag = .enum_variant_unit, .main_token = vname, .lhs = Ast.none, .rhs = Ast.none }),
        }
        try variants.append(p.gpa, variant);
        if (p.peek().tag == .comma) p.advance();
    }
    try p.expect(.r_brace, "expected '}' to close enum body");

    const header = try p.addRange(variants.items);
    return p.addNode(.{ .tag = .enum_decl, .main_token = name_tok, .lhs = header, .rhs = Ast.none });
}

/// `match scrut { pat -> body, ... }`. Scrutinee parsed in `no_block` (so a bare
/// `match x { ... }` reads `x`, the `{` opening the arm list). Arms are comma-
/// separated, newlines insignificant inside `{}`.
fn parseMatch(p: *Parser) Error!Ast.Index {
    const match_tok = p.index;
    p.advance(); // match
    const saved_nb = p.no_block;
    p.no_block = true;
    const scrut = try p.parseExpr(0);
    p.no_block = false;
    try p.expect(.l_brace, "expected '{' to open a match");

    var arms: std.ArrayList(Ast.Index) = .empty;
    defer arms.deinit(p.gpa);
    while (true) {
        while (p.peek().tag == .newline) p.advance();
        if (p.peek().tag == .r_brace) break;
        const pat = try p.parsePattern();
        var guard: Ast.Index = Ast.none;
        if (p.peek().tag == .kw_if) {
            p.advance();
            const g_nb = p.no_block;
            p.no_block = true; // stop the guard cond before `->`/`{`
            guard = try p.parseExpr(0);
            p.no_block = g_nb;
        }
        const arrow = p.index;
        try p.expect(.arrow, "expected '->' after a match pattern");
        const body = try p.parseExpr(0);
        const arm_hdr = try p.addExtra(&.{ guard, body });
        const arm = try p.addNode(.{ .tag = .match_arm, .main_token = arrow, .lhs = pat, .rhs = arm_hdr });
        try arms.append(p.gpa, arm);
        if (p.peek().tag == .comma) p.advance();
    }
    try p.expect(.r_brace, "expected '}' to close match");
    p.no_block = saved_nb;

    const header = try p.addRange(arms.items);
    return p.addNode(.{ .tag = .match_expr, .main_token = match_tok, .lhs = scrut, .rhs = header });
}

/// A match pattern, with or-alternatives: `subpat ('|' subpat)*`. Emits a
/// `pattern_or` only when there are >=2 alternatives; otherwise the bare subpat.
fn parsePattern(p: *Parser) Error!Ast.Index {
    const first = try p.parseSubPattern();
    if (p.peek().tag != .pipe) return first;
    var alts: std.ArrayList(Ast.Index) = .empty;
    defer alts.deinit(p.gpa);
    const first_tok = p.nodes.items[first].main_token;
    try alts.append(p.gpa, first);
    while (p.peek().tag == .pipe) {
        p.advance(); // |
        try alts.append(p.gpa, try p.parseSubPattern());
    }
    const hdr = try p.addRange(alts.items);
    return p.addNode(.{ .tag = .pattern_or, .main_token = first_tok, .lhs = hdr, .rhs = Ast.none });
}

/// A single (non-or) pattern: `_` (wildcard), an int/bool literal, a bare
/// identifier binding, or `.V`/`N.V` (variant) with optional payload sub-patterns.
/// Tuple payloads `.V(p, ...)` and struct payloads `.V { f, f: alias, f: subpat }`
/// each hold arbitrary sub-patterns (recursive via `parsePattern`).
fn parseSubPattern(p: *Parser) Error!Ast.Index {
    const tok = p.peek();
    // Literal patterns: int / true / false.
    if (tok.tag == .number or tok.tag == .kw_true or tok.tag == .kw_false) {
        const lt = p.index;
        p.advance();
        return p.addNode(.{ .tag = .pattern_literal, .main_token = lt, .lhs = Ast.none, .rhs = Ast.none });
    }
    if (tok.tag == .identifier and std.mem.eql(u8, tok.text(p.src), "_")) {
        const wt = p.index;
        p.advance();
        return p.addNode(.{ .tag = .pattern_wildcard, .main_token = wt, .lhs = Ast.none, .rhs = Ast.none });
    }
    var type_name: Ast.Index = Ast.none;
    if (tok.tag == .identifier) {
        // A bare identifier NOT followed by `.` is a whole-value binding.
        if (p.peek2().tag != .dot) {
            const bt = p.index;
            p.advance();
            return p.addNode(.{ .tag = .pattern_binding, .main_token = bt, .lhs = Ast.none, .rhs = Ast.none });
        }
        // Qualified `N.V`: build the type-name leaf, then expect `.V`.
        type_name = try p.leaf(.identifier, p.index);
        try p.expect(.dot, "expected '.' after an enum type name in a pattern");
    } else {
        try p.expect(.dot, "expected a variant pattern ('.V' or '_')");
    }
    const vname = p.index;
    try p.expect(.identifier, "expected a variant name in a pattern");
    var binders: Ast.Index = Ast.none;
    switch (p.peek().tag) {
        .l_paren => {
            p.advance(); // (
            var binds: std.ArrayList(Ast.Index) = .empty;
            defer binds.deinit(p.gpa);
            while (p.peek().tag != .r_paren) {
                // Each tuple element is an arbitrary sub-pattern (literal, binding,
                // wildcard, nested variant, or-pattern).
                try binds.append(p.gpa, try p.parsePattern());
                if (p.peek().tag == .comma) p.advance() else break;
            }
            try p.expect(.r_paren, "expected ')' to close a tuple pattern");
            binders = try p.addRange(binds.items);
        },
        .l_brace => {
            const saved_nb = p.no_block;
            p.no_block = false;
            defer p.no_block = saved_nb;
            p.advance(); // {
            var binds: std.ArrayList(Ast.Index) = .empty;
            defer binds.deinit(p.gpa);
            while (true) {
                while (p.peek().tag == .newline) p.advance();
                if (p.peek().tag == .r_brace) break;
                const field_tok = p.index;
                try p.expect(.identifier, "expected a field name in a struct pattern");
                var bind: Ast.Index = undefined;
                if (p.peek().tag == .colon) {
                    p.advance(); // :
                    // `field: alias` (rename to a bare ident) vs `field: subpat`
                    // (a literal/`.`/`_`/nested pattern matched against the field).
                    // A bare identifier NOT opening a payload is the M10 rename alias.
                    const after = p.peek();
                    const is_alias = after.tag == .identifier and
                        !std.mem.eql(u8, after.text(p.src), "_") and
                        p.peek2().tag != .dot;
                    const src_ident = try p.addNode(.{ .tag = .identifier, .main_token = field_tok, .lhs = Ast.none, .rhs = Ast.none });
                    if (is_alias) {
                        const alias_tok = p.index;
                        p.advance();
                        bind = try p.addNode(.{ .tag = .pattern_binding, .main_token = alias_tok, .lhs = src_ident, .rhs = Ast.none });
                    } else {
                        const subpat = try p.parsePattern();
                        bind = try p.addNode(.{ .tag = .pattern_binding, .main_token = field_tok, .lhs = src_ident, .rhs = subpat });
                    }
                } else {
                    bind = try p.addNode(.{ .tag = .pattern_binding, .main_token = field_tok, .lhs = Ast.none, .rhs = Ast.none });
                }
                try binds.append(p.gpa, bind);
                if (p.peek().tag == .comma) p.advance();
            }
            try p.expect(.r_brace, "expected '}' to close a struct pattern");
            binders = try p.addRange(binds.items);
        },
        else => {},
    }
    return p.addNode(.{ .tag = .pattern_variant, .main_token = vname, .lhs = type_name, .rhs = binders });
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

/// A type reference is written as an identifier (e.g. `int`, `bool`, `str`), the
/// unit type `()`, or a module-qualified type `mod.Type` (M14). A qualified type
/// reuses the `field_access` node: receiver = the module-name `identifier` leaf,
/// `main_token` = the type-name ident after `.`. The resolver disambiguates this
/// from value field access by its type position. `/` never appears in a type —
/// only `.` — so this stays unambiguous with the `import` path grammar.
fn parseType(p: *Parser) Error!Ast.Index {
    if (p.peek().tag == .l_paren and p.peek2().tag == .r_paren) {
        const at = p.index;
        p.advance(); // (
        p.advance(); // )
        return p.addNode(.{ .tag = .literal_unit, .main_token = at, .lhs = Ast.none, .rhs = Ast.none });
    }
    const at = p.index;
    try p.expect(.identifier, "expected a type name");
    var ty = try p.addNode(.{ .tag = .identifier, .main_token = at, .lhs = Ast.none, .rhs = Ast.none });
    // A `.ident` chain qualifies the type by its owning module (`mod.Type`). The
    // chain nests left like value field access, so a deeper `a.b.C` is supported
    // structurally (the resolver decides what is legal).
    while (p.peek().tag == .dot) {
        p.advance(); // .
        const field_tok = p.index;
        try p.expect(.identifier, "expected a type name after '.'");
        ty = try p.addNode(.{ .tag = .field_access, .main_token = field_tok, .lhs = ty, .rhs = Ast.none });
    }
    return ty;
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
        // Inferred variant construction `.V` in a type-known position. At an
        // expression START (no receiver) a leading `.` is a variant; `parsePostfix`
        // then upgrades it to a tuple/struct form if `(`/`{` follows. (A postfix
        // `.field` is handled in parsePostfix, after an operand.)
        .dot => {
            p.advance(); // .
            const name = p.index;
            try p.expect(.identifier, "expected a variant name after '.'");
            return p.addNode(.{ .tag = .enum_init_unit, .main_token = name, .lhs = Ast.none, .rhs = Ast.none });
        },
        .kw_match => {
            if (p.no_block) return p.fail(tok, "expected an expression");
            return p.parseMatch(); // a match as a value expression
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
            .l_paren => {
                // An inferred `.V` followed by `(args)` is a tuple-variant
                // construction; rebuild it in place (keep main_token / lhs=none).
                if (p.nodes.items[lhs].tag == .enum_init_unit and p.nodes.items[lhs].lhs == Ast.none) {
                    const rebuilt = try p.upgradeTupleInit(lhs, Ast.none);
                    lhs = rebuilt;
                } else {
                    lhs = try p.parseCall(lhs);
                }
            },
            // `.field` access. `..` is a separate token, so `0..5` is unaffected.
            .dot => lhs = try p.parseFieldAccess(lhs),
            // `Name { ... }` literal / variant construction — only when blocks are
            // allowed and `lhs` is a bare name (struct), an inferred `.V`
            // (struct-variant), or a `field_access` (qualified `N.V`). The call/
            // group `( )` reset `no_block`, so `f(P{x:1})` works.
            .l_brace => {
                if (p.no_block) break;
                const ltag = p.nodes.items[lhs].tag;
                switch (ltag) {
                    .identifier => lhs = try p.parseStructLiteral(lhs),
                    .enum_init_unit => if (p.nodes.items[lhs].lhs == Ast.none) {
                        lhs = try p.upgradeStructInit(lhs, Ast.none);
                    } else break,
                    // Qualified `N.V { ... }`: the type-name is the field_access's
                    // receiver and the variant is its field token.
                    .field_access => lhs = try p.upgradeStructInit(lhs, lhs),
                    else => break,
                }
            },
            else => break,
        }
    }
    return lhs;
}

/// Parse `(args)` onto a variant construction, producing an `enum_init_tuple`.
/// `node` is the `enum_init_unit` (inferred) to rebuild in place, or — for a
/// qualified `N.V(...)` — a `field_access` whose receiver is the type name and
/// whose field token is the variant. `type_name` is `none` for inferred.
fn upgradeTupleInit(p: *Parser, node: Ast.Index, type_name: Ast.Index) Error!Ast.Index {
    p.advance(); // (
    var args: std.ArrayList(Ast.Index) = .empty;
    defer args.deinit(p.gpa);
    const saved_nb = p.no_block;
    p.no_block = false;
    defer p.no_block = saved_nb;
    while (p.peek().tag != .r_paren) {
        try args.append(p.gpa, try p.parseExpr(0));
        if (p.peek().tag == .comma) p.advance() else break;
    }
    try p.expect(.r_paren, "expected ')' to close a variant construction");
    const header = try p.addRange(args.items);
    const vtok = p.nodes.items[node].main_token;
    p.nodes.items[node] = .{ .tag = .enum_init_tuple, .main_token = vtok, .lhs = type_name, .rhs = header };
    return node;
}

/// Parse `{ field: value, ... }` onto a variant construction, producing an
/// `enum_init_struct`. `node` is the `enum_init_unit` (inferred) or the
/// `field_access` (qualified `N.V`) to rebuild. `type_name` is `none` (inferred)
/// or the type-name node (for qualified, the field_access's receiver).
fn upgradeStructInit(p: *Parser, node: Ast.Index, qualified: Ast.Index) Error!Ast.Index {
    // For a qualified `N.V`, the node is the field_access: variant = its field
    // token, type-name = its receiver.
    const vtok = p.nodes.items[node].main_token;
    const type_name: Ast.Index = if (qualified == Ast.none) Ast.none else p.nodes.items[node].lhs;
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
            value = try p.addNode(.{ .tag = .identifier, .main_token = field_tok, .lhs = Ast.none, .rhs = Ast.none });
        }
        const fi = try p.addNode(.{ .tag = .field_init, .main_token = field_tok, .lhs = value, .rhs = Ast.none });
        try inits.append(p.gpa, fi);
        if (p.peek().tag == .comma) p.advance();
    }
    try p.expect(.r_brace, "expected '}' to close a variant construction");
    const header = try p.addRange(inits.items);
    p.nodes.items[node] = .{ .tag = .enum_init_struct, .main_token = vtok, .lhs = type_name, .rhs = header };
    return node;
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
fn parseExprOnly(gpa: std.mem.Allocator, tokens: []const Token, source: []const u8, diag: *?Diagnostic) error{OutOfMemory}!?Ast.Tree {
    var p: Parser = .{
        .gpa = gpa,
        .tokens = tokens,
        .src = source,
        .index = 0,
        .nodes = .empty,
        .extra = .empty,
        .pub_decls = .empty,
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
    if (tree.pub_bits.len != 0) gpa.free(@constCast(tree.pub_bits));
}

fn expectSexpr(source: []const u8, want: []const u8) !void {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);

    var diag: ?Diagnostic = null;
    const tree = (try parseExprOnly(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
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
    const tree = (try parse(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
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
    const source = "1 +";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const result = try parseExprOnly(gpa, tokens, source, &diag);
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
    const source = "fn add(a: int, b: int) -> int {\n return a + b\n}\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const tree = (try parse(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
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
    const source = "fn f() {\n @x 1\n}\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const result = try parse(gpa, tokens, source, &diag);
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

// enums + match (M10)

test "enum declaration with all three variant forms parses" {
    try expectProgram(
        "enum Shape { Empty, Circle(int), Rect { w: int, h: int } }\n",
        "(program (enum Shape (variant.unit Empty) (variant.tuple Circle int) (variant.struct Rect (param w int) (param h int))))",
    );
}

test "qualified and inferred variant construction parse" {
    // A qualified `N.V(...)` stays a `.call` over a field_access (typecheck/codegen
    // reinterpret it); inferred `.V` gets a dedicated enum_init_unit node.
    try expectProgram(
        "fn f() -> int { c := Shape.Circle(5)\n e := .Empty\n 0 }\n",
        "(program (fn f () int (block (:= c (call (. Shape Circle) 5)) (:= e (enew.unit Empty)) 0)))",
    );
}

test "qualified struct-variant construction parses (upgraded from field_access {)" {
    try expectProgram(
        "fn f() -> int { r := Shape.Rect { w: 3, h: 4 }\n 0 }\n",
        "(program (fn f () int (block (:= r (enew.struct Shape Rect (field w 3) (field h 4))) 0)))",
    );
}

test "match with tuple/struct/unit/wildcard arms parses" {
    try expectProgram(
        "fn f(s: Shape) -> int { match s { .Circle(r) -> r, .Rect { w, h } -> w, .Empty -> 0, _ -> 1 } }\n",
        "(program (fn f ((param s Shape)) int (block (match s (arm (pvar Circle (bind r)) r) (arm (pvar Rect (bind w) (bind h)) w) (arm (pvar Empty) 0) (arm (_) 1)))))",
    );
}

test "struct-rename pattern binding parses" {
    try expectProgram(
        "fn f(s: Shape) -> int { match s { .Rect { w: a, h: b } -> a } }\n",
        "(program (fn f ((param s Shape)) int (block (match s (arm (pvar Rect (bind a from w) (bind b from h)) a)))))",
    );
}

test "enum program pack/unpack byte round-trip" {
    const gpa = testing.allocator;
    const source = "enum Shape { Empty, Circle(int), Rect { w: int, h: int } }\nfn area(s: Shape) -> int { match s { .Circle(r) -> r, .Rect { w, h } -> w, .Empty -> 0 } }\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const tree = (try parse(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
    defer freeTree(gpa, tree);

    const blob = try Ast.pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try Ast.unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer freeTree(gpa, got);

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
}

test "struct program pack/unpack byte round-trip" {
    const gpa = testing.allocator;
    const source = "struct Point { x: int, y: int }\nfn f() -> int { p := Point { x: 1, y: 2 }\n p.x }\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const tree = (try parse(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
    defer freeTree(gpa, tree);

    const blob = try Ast.pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try Ast.unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer freeTree(gpa, got);

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
}

// M14: imports, pub, qualified access/types

test "import binds the last path segment" {
    try expectProgram(
        "import geometry/rect\nfn main() {}\n",
        "(program (import geometry/rect) (fn main () _ (block)))",
    );
}

test "single-segment import parses" {
    try expectProgram(
        "import util\nfn main() {}\n",
        "(program (import util) (fn main () _ (block)))",
    );
}

test "import with alias parses" {
    try expectProgram(
        "import geometry/rect as r\nfn main() {}\n",
        "(program (import geometry/rect as r) (fn main () _ (block)))",
    );
}

test "deep import path parses" {
    try expectProgram(
        "import a/b/c/d\nfn main() {}\n",
        "(program (import a/b/c/d) (fn main () _ (block)))",
    );
}

test "pub fn renders with a pub wrapper" {
    try expectProgram(
        "pub fn area() -> int { 0 }\n",
        "(program (pub (fn area () int (block 0))))",
    );
}

test "pub struct and pub enum parse" {
    try expectProgram(
        "pub struct Rect { w: int, h: int }\npub enum Shape { Empty }\n",
        "(program (pub (struct Rect (param w int) (param h int))) (pub (enum Shape (variant.unit Empty))))",
    );
}

test "non-pub decl stays bare among pub decls" {
    try expectProgram(
        "pub fn a() -> int { 0 }\nfn b() -> int { 1 }\n",
        "(program (pub (fn a () int (block 0))) (fn b () int (block 1)))",
    );
}

test "qualified member call parses via field_access" {
    // `rect.area()` is a call whose callee is a field_access (module member).
    try expectProgram(
        "fn main() -> int { rect.area() }\n",
        "(program (fn main () int (block (call (. rect area)))))",
    );
}

test "qualified type in a param parses via field_access" {
    try expectProgram(
        "fn f(r: rect.Rect) -> int { 0 }\n",
        "(program (fn f ((param r (. rect Rect))) int (block 0)))",
    );
}

test "qualified type as a return type parses" {
    try expectProgram(
        "fn make() -> rect.Rect { Point { x: 1 } }\n",
        "(program (fn make () (. rect Rect) (block (new Point (field x 1)))))",
    );
}

test "qualified type in a struct field parses" {
    try expectProgram(
        "struct Scene { r: rect.Rect }\n",
        "(program (struct Scene (param r (. rect Rect))))",
    );
}

test "module-qualified variant construction parses (3-level field_access)" {
    // `m.Color.Red` — receiver field_access (m.Color) carries the variant tail.
    try expectProgram(
        "fn f() -> int { c := m.Color.Red\n 0 }\n",
        "(program (fn f () int (block (:= c (. (. m Color) Red)) 0)))",
    );
}

test "pub modifier requires a declaration" {
    const gpa = testing.allocator;
    const source = "pub import a/b\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const result = try parse(gpa, tokens, source, &diag);
    try testing.expect(result == null);
    try testing.expect(diag != null);
}

test "import path missing a segment after slash is an error" {
    const gpa = testing.allocator;
    const source = "import a/\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const result = try parse(gpa, tokens, source, &diag);
    try testing.expect(result == null);
    try testing.expect(diag != null);
}

test "import pack/unpack byte round-trip carries pub_bits" {
    const gpa = testing.allocator;
    const source = "import geometry/rect as r\npub fn area() -> int { 0 }\nfn helper() -> int { 1 }\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const tree = (try parse(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
    defer freeTree(gpa, tree);

    // Exactly the `area` fn_decl node is pub; `helper` is not.
    var pub_count: usize = 0;
    for (tree.nodes, 0..) |node, i| {
        if (node.tag == .fn_decl and tree.isPub(@intCast(i))) pub_count += 1;
    }
    try testing.expectEqual(@as(usize, 1), pub_count);

    const blob = try Ast.pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try Ast.unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer freeTree(gpa, got);

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
    try testing.expectEqualSlices(u32, tree.pub_bits, got.pub_bits);
}

test "program pack/unpack byte round-trip" {
    const gpa = testing.allocator;
    const source = "fn add(a: int, b: int) -> int {\n return a + b\n}\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const tree = (try parse(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
    defer freeTree(gpa, tree);

    const blob = try Ast.pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try Ast.unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer freeTree(gpa, got);

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
}
