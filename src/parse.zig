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
const Code = @import("diagnostics/codes.zig").Code;

const Parser = @This();

gpa: std.mem.Allocator,
tokens: []const Token,
/// Source bytes, for the rare token-TEXT check the parser needs (the `_`
/// wildcard pattern). Most of the parser works on token tags alone.
src: []const u8,
/// Cursor into `tokens`.
index: u32,
nodes: std.ArrayList(Node),
/// Node indices of top-level decls that carried a `pub` modifier. Packed
/// into the `Tree.pub_bits` bitset after the `.program` node is appended.
pub_decls: std.ArrayList(Ast.Index),
/// Variable-arity child runs and range/proto headers (the "extra_data" side
/// array). See `Ast` for the encoding.
extra: std.ArrayList(u32),
/// Accumulated parse diagnostics. A parse that appends any diagnostic is a
/// TAINTED parse: `parse()` still returns the (partial) tree, but the caller must
/// not cache it, lower it, or codegen it. Unmanaged-style (`append(gpa, d)`).
diags: std.ArrayList(Diagnostic),
/// While true (only while parsing an `if`/`while` condition), a leading `{` or
/// `if` in `parsePrefix` is NOT taken as a block/if expression, so `if c { ... }`
/// reads `c` as the condition and `{ ... }` as the body. Parens are the escape
/// hatch. Reset (save/restore) inside `parseBlock` so a block body re-arms it.
no_block: bool = false,
/// Expression-recursion depth, bumped at the single `parseExpr` chokepoint and
/// restored on every unwind (via `defer`). Past `MAX_EXPR_DEPTH` the parser emits
/// one "nested too deeply" diagnostic and returns a bounded `error_node` instead
/// of recursing further — the anti-crash backstop for the project's frame-sizing
/// SIGBUS hazard on adversarially deep input.
depth: u16 = 0,
/// Cascade-suppression latch (matklad/ANTLR "in error recovery" flag). Set the
/// moment a diagnostic is appended (`warn`, which `fail` funnels through); while
/// set, `warn` still unwinds/repairs but does NOT append a FURTHER diagnostic, so
/// a burst of derived errors over the same unresynced region collapses to the
/// first, actionable one. CLEARED at a real resync boundary — a cleanly crossed
/// statement terminator (`expectTerminator`'s success arms) or the start of a new
/// top-level decl (`parseDecls` loop top) — which is what preserves genuinely
/// INDEPENDENT errors: after recovery reaches the next statement/decl the latch
/// clears, so the next independent error IS reported. It is deliberately NOT
/// cleared by every `expect`/`eat`/`advance`, because a statement re-parse after
/// `findNextStmt` lands on the TAIL of the same broken statement (e.g. the `x` in
/// a broken `let x =`) matches real tokens without having crossed a boundary.
in_error: bool = false,

pub const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;

const Error = error{ OutOfMemory, ParseError };

/// Cap on `parseExpr` recursion depth. Conservative versus the frame-sizing
/// SIGBUS hazard; no valid source nests expressions anywhere near this.
const MAX_EXPR_DEPTH: u16 = 256;

// Recovery lives as comptime `EnumSet(token.Tag)` FIRST sets and FOLLOW-union-
// ancestor-anchor recovery sets, matklad-style, rather than scattered `!= .x`
// conditionals. Every nested recovery set unions `decl_anchors` so a runaway
// inner construct breaks all the way out at the next top-level decl keyword or
// EOF, bounding cascade to the current declaration.

const TagSet = std.EnumSet(token.Tag);

fn setOf(comptime tags: []const token.Tag) TagSet {
    var s = TagSet.initEmpty();
    for (tags) |t| s.insert(t);
    return s;
}

/// FIRST(top-level decl): the exact arms of `parseDecls`' dispatch.
const decl_first = setOf(&.{ .kw_import, .kw_pub, .kw_fn, .kw_struct, .kw_enum, .kw_impl, .kw_protocol, .kw_type, .kw_extern });
/// The universal inherited ancestor anchor: `decl_first` ∪ {eof}.
const decl_anchors = decl_first.unionWith(setOf(&.{.eof}));
/// FIRST(expr): exactly `parsePrefix`'s accepted switch arms.
const expr_first = setOf(&.{ .identifier, .number, .float, .string, .char_lit, .kw_true, .kw_false, .l_paren, .l_brace, .l_bracket, .kw_if, .kw_loop, .kw_match, .kw_unsafe, .at, .dot, .minus, .bang, .amp, .star });
/// FIRST(type): an identifier (dot-chained) or the unit type `()`.
const type_first = setOf(&.{ .identifier, .l_paren });
/// FIRST(sub-pattern): a literal, a binding/wildcard identifier, or a `.V`/`N.V`.
const pattern_first = setOf(&.{ .identifier, .number, .kw_true, .kw_false, .dot });
/// FIRST(stmt): `expr_first` ∪ the statement-keyword starters.
const stmt_first = expr_first.unionWith(setOf(&.{ .kw_while, .kw_for, .kw_break, .kw_continue, .kw_return }));
/// FOLLOW(param) ∪ decl_anchors: own closer `)`, `->` (ret), `{` (body start).
const param_recovery = setOf(&.{ .r_paren, .arrow, .l_brace }).unionWith(decl_anchors);
/// FOLLOW(elem) ∪ decl_anchors for `( ... )` comma lists. No newline: newlines
/// are NOT skipped inside `( )` today — a newline there stays garbage (deleted).
const tuple_recovery = setOf(&.{ .r_paren, .comma }).unionWith(decl_anchors);
/// FOLLOW(field) ∪ decl_anchors for `{ ... }` item lists. Comma AND newline are
/// both legal separators inside braces (loop-top `skipNewlines` handles newline).
const field_recovery = setOf(&.{ .r_brace, .comma, .newline }).unionWith(decl_anchors);
/// Same shape as field_recovery (enum body is brace-delimited, comma/newline sep).
const variant_recovery = field_recovery;
/// Same shape again (match body is brace-delimited, arms comma/newline sep).
const arm_recovery = field_recovery;
/// FOLLOW(generic param/arg) ∪ decl_anchors for a `[ .. ]` generic list: own
/// closer `]`, separator `,`. No newline (a `[ .. ]` list stays on one line, like
/// the `( .. )` param list).
const generic_recovery = setOf(&.{ .r_bracket, .comma }).unionWith(decl_anchors);
/// `findNextStmt`'s STOP set (at the block's brace-depth). Must contain `.newline`
/// so a post-recovery `expectTerminator` does NOT spuriously cascade (resync lands
/// on a newline that `skipNewlines` then swallows).
const stmt_recovery = setOf(&.{ .newline, .r_brace, .eof }).unionWith(stmt_first).unionWith(decl_anchors);

comptime {
    // The headline audit: the 3-way list arms are unambiguous only if FIRST(expr)
    // is disjoint from FIRST(decl) — otherwise a stray `fn` inside `f(` could be
    // read as an argument instead of bailing to the decl loop.
    if (expr_first.intersectWith(decl_first).count() != 0) @compileError("expr_first overlaps decl_first: 3-way arms ambiguous");
    if (!tuple_recovery.contains(.r_paren) or !tuple_recovery.contains(.comma)) @compileError("tuple_recovery missing own closer/separator");
    if (!field_recovery.contains(.r_brace) or !field_recovery.contains(.newline)) @compileError("field_recovery missing own closer/separator");
    if (!stmt_recovery.contains(.newline)) @compileError("stmt_recovery must contain .newline (ASI anti-cascade)");
    for ([_]TagSet{ param_recovery, tuple_recovery, field_recovery, stmt_recovery, decl_anchors }) |s| {
        if (!s.contains(.eof)) @compileError("recovery set missing .eof anchor");
    }
}

// Brace/paren-DEPTH-aware. Because blocks/if/match/struct-literals are
// EXPRESSIONS, `r_brace` is structurally overloaded: a depth-naive scan would
// resync to a nested block-expression's `}` instead of the enclosing block's.
// Each scanner tracks depth and only STOPS at a target token when depth == 0.
// Termination is trivial — every iteration calls `advance()` (clamped at eof) —
// with a debug fuel guard as defense-in-depth against a depth-accounting bug.
// The scanners emit NO diagnostics (the triggering diag already fired — this is
// the no-cascade guarantee).

/// Scan to the next top-level decl keyword (or eof). If the cursor already sits
/// on a decl keyword (a decl that failed AT its own keyword), advance once first
/// so we always make progress.
///
/// A decl keyword is an UNCONDITIONAL anchor here, ignoring brace depth: this
/// grammar has no nested declarations, so `fn`/`struct`/`enum`/`import`/`pub`
/// never legitimately appears inside a balanced brace/paren region. Depth-gating
/// this stop (as the nested-block-aware `findNextStmt` must) would let an
/// UNbalanced stray `{` — e.g. the trailing `{` of `fn f( ) ) ) {` — inflate the
/// counter and swallow the next real `fn g`. Making decls a hard anchor keeps
/// recovery landing on the next declaration regardless.
fn findNextDecl(p: *Parser) void {
    if (decl_first.contains(p.peek().tag)) p.advance();
    var fuel: usize = p.tokens.len + 1;
    while (!p.at(.eof)) {
        std.debug.assert(fuel != 0);
        fuel -= 1;
        if (decl_first.contains(p.peek().tag)) return;
        p.advance();
    }
}

/// Scan to the next statement boundary at the block's brace-depth: a newline
/// (`skipNewlines` swallows it — anti-cascade), `r_brace` (block loop breaks), a
/// stmt-FIRST token (reparse the next statement), or a decl keyword (bail the
/// wrecked block to the decl loop so it cannot eat the next fn).
fn findNextStmt(p: *Parser) void {
    var depth: i32 = 0;
    var fuel: usize = p.tokens.len + 1;
    while (!p.at(.eof)) {
        std.debug.assert(fuel != 0);
        fuel -= 1;
        const t = p.peek().tag;
        if (depth == 0 and (t == .r_brace or t == .newline or stmt_first.contains(t) or decl_first.contains(t))) {
            // Landing on a `.newline`/`.r_brace` is a cleanly crossed statement
            // boundary — the same resync signal `expectTerminator`'s success arms
            // give — so clear the cascade latch: the NEXT statement's first error
            // must report. (When `expectTerminator` itself fails on a still-
            // unconsumed closer like `return )`, the block loop recovers via THIS
            // scan instead of the terminator's clear arm, so the clear has to live
            // here too or an independent error on the following line is swallowed.)
            // A stmt-FIRST/decl stop is deliberately NOT a clear: it can be the tail
            // of the same broken statement on the same physical line, still in the
            // unresynced region.
            if (t == .newline or t == .r_brace) p.in_error = false;
            return;
        }
        switch (t) {
            .l_brace, .l_paren, .l_bracket => depth += 1,
            .r_brace, .r_paren, .r_bracket => if (depth > 0) {
                depth -= 1;
            },
            else => {},
        }
        p.advance();
    }
}

/// Generic depth-aware scan stopping at `set` members at depth 0; backs the
/// match-arm resync shim.
fn resyncTo(p: *Parser, comptime set: TagSet) void {
    var depth: i32 = 0;
    var fuel: usize = p.tokens.len + 1;
    while (!p.at(.eof)) {
        std.debug.assert(fuel != 0);
        fuel -= 1;
        const t = p.peek().tag;
        if (depth == 0 and set.contains(t)) return;
        switch (t) {
            .l_brace, .l_paren, .l_bracket => depth += 1,
            .r_brace, .r_paren, .r_bracket => if (depth > 0) {
                depth -= 1;
            },
            else => {},
        }
        p.advance();
    }
}

/// What `parse()` returns: an always-present tree (on non-OOM runs) plus the
/// accumulated diagnostics. A run with a non-empty `diags` is a TAINTED parse —
/// the tree is partial/poisoned and must not be cached, lowered, or codegen'd.
/// The `diags` slice is owned by the caller (freed via `gpa.free`).
pub const Result = struct {
    tree: Ast.Tree,
    diags: []const Diagnostic,
};

/// Scoped override of `p.no_block`: set it to `value`, restore the PRIOR value on
/// `end()`. Every fresh-expression-context site sets it through this so set and
/// restore are symmetric by construction — restoring the saved value (not a hard
/// reset to false) is what keeps a nested site that was entered with `no_block`
/// already set from silently changing parse classification.
const NoBlockScope = struct {
    p: *Parser,
    saved: bool,
    fn enter(p: *Parser, value: bool) NoBlockScope {
        const s = NoBlockScope{ .p = p, .saved = p.no_block };
        p.no_block = value;
        return s;
    }
    fn end(s: NoBlockScope) void {
        s.p.no_block = s.saved;
    }
};

/// Parse a whole file into a `Tree`. ALWAYS returns a `Result` on a non-OOM run:
/// the tree is the (possibly partial) program built so far, and `diags` holds
/// every accumulated diagnostic. A non-empty `diags` means the parse is TAINTED —
/// the tree is usable for reporting/downstream inspection but must not be cached,
/// lowered, or codegen'd (see `Result`). Only `OutOfMemory` is a hard error.
pub fn parse(gpa: std.mem.Allocator, tokens: []const Token, source: []const u8) error{OutOfMemory}!Result {
    var p: Parser = .{
        .gpa = gpa,
        .tokens = tokens,
        .src = source,
        .index = 0,
        .nodes = .empty,
        .extra = .empty,
        .pub_decls = .empty,
        .diags = .empty,
    };
    const tree = p.parseProgram() catch |err| switch (err) {
        error.OutOfMemory => {
            p.deinitAll();
            return error.OutOfMemory;
        },
    };
    // Post-parse invariant sweep on the produced (possibly recovered) tree, gated
    // by `std.debug.runtime_safety` so it hardens the whole Debug/ReleaseSafe test
    // corpus at zero ReleaseFast cost. It asserts (panics) on a violation — a
    // sanity check on the parser itself, not a user-facing diagnostic.
    if (std.debug.runtime_safety) checkInvariants(tree, p.tokens, p.src, p.diags.items);
    // The diagnostics list is handed to the caller; everything else is either
    // moved into `tree` or already released inside `parseProgram`.
    const diags = p.diags.toOwnedSlice(p.gpa) catch |err| switch (err) {
        error.OutOfMemory => {
            freeTree(p.gpa, tree);
            p.diags.deinit(p.gpa);
            return error.OutOfMemory;
        },
    };
    return .{ .tree = tree, .diags = diags };
}

/// Test/consumer helper: parse a program that is expected to be VALID, returning
/// only the owned `Tree`. Frees the (empty, on a clean parse) diagnostics list and
/// returns `error.UnexpectedParseFailure` if the parse was tainted — so the many
/// call sites that only care about a valid-program tree stay one line and never
/// leak the diags slice. On taint, frees the partial tree too.
pub fn expectTree(gpa: std.mem.Allocator, tokens: []const Token, source: []const u8) !Ast.Tree {
    const res = try parse(gpa, tokens, source);
    if (res.diags.len != 0) {
        gpa.free(@constCast(res.diags));
        freeTree(gpa, res.tree);
        return error.UnexpectedParseFailure;
    }
    gpa.free(@constCast(res.diags));
    return res.tree;
}

/// Release all working buffers, including the diagnostics list. Used only on the
/// OOM path (the success path moves `nodes`/`extra` into the tree and hands
/// `diags` to the caller).
fn deinitAll(p: *Parser) void {
    p.nodes.deinit(p.gpa);
    p.extra.deinit(p.gpa);
    p.pub_decls.deinit(p.gpa);
    p.diags.deinit(p.gpa);
}

/// Release the working buffers EXCEPT the diagnostics list (the parseExprOnly
/// test helper owns the diags separately).
fn deinit(p: *Parser) void {
    p.nodes.deinit(p.gpa);
    p.extra.deinit(p.gpa);
    p.pub_decls.deinit(p.gpa);
}

/// Parse the whole program. The decl loop is resilient: each broken top-level
/// declaration records its diagnostics, resyncs to the next decl keyword, and the
/// loop continues — so `parse()` yields a tree covering ALL parsed decls and
/// reports many errors per file. `parseDecls` absorbs every `error.ParseError`
/// per-decl, so nothing but `OutOfMemory` propagates here.
fn parseProgram(p: *Parser) error{OutOfMemory}!Ast.Tree {
    var decls: std.ArrayList(Ast.Index) = .empty;
    defer decls.deinit(p.gpa);

    try p.parseDecls(&decls);

    const header = try p.addRange(decls.items);
    _ = try p.addNode(.{ .tag = .program, .main_token = 0, .lhs = header, .rhs = Ast.none });

    const nodes = try p.nodes.toOwnedSlice(p.gpa);
    errdefer p.gpa.free(nodes);
    const extra = try p.extra.toOwnedSlice(p.gpa);
    errdefer p.gpa.free(extra);
    const pub_bits = try p.buildPubBits(nodes.len);
    // The pub-decl scratch list is fully consumed into `pub_bits`; release it on
    // the success path (an OOM in this tail goes through `parse`'s `deinitAll`,
    // which re-`deinit`s the now-empty lists harmlessly).
    p.pub_decls.deinit(p.gpa);
    return Ast.Tree{ .nodes = nodes, .extra = extra, .pub_bits = pub_bits };
}

/// The resilient top-level declaration loop (the Zig-std per-item resync model):
/// parse one decl; on `error.ParseError` the diagnostic is already recorded, so
/// resync to the next decl keyword and continue. Every following decl still
/// parses, so N independent broken decls yield N (or more) diagnostics. The loop
/// exits at eof (no trailing `expect(.eof)` — the `while` condition owns that).
/// Narrowed to `error{OutOfMemory}`: the shim absorbs every `ParseError`.
fn parseDecls(p: *Parser, decls: *std.ArrayList(Ast.Index)) error{OutOfMemory}!void {
    p.skipNewlines();
    while (!p.at(.eof)) {
        // Each top-level decl is a fresh recovery unit: clear the cascade latch so
        // an independent error in THIS decl reports even after a prior decl broke.
        p.in_error = false;
        const entry = p.index;
        p.parseDeclRecoverable(decls) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.ParseError => p.findNextDecl(),
        };
        p.skipNewlines();
        std.debug.assert(p.index > entry or p.at(.eof));
    }
}

/// Parse ONE top-level declaration and append it (recording a `pub` export as a
/// side effect). On any nested `ParseError` the diagnostic is already in
/// `p.diags`; the error unwinds to the decl loop, which resyncs. This is the
/// per-decl tolerance unit.
fn parseDeclRecoverable(p: *Parser, decls: *std.ArrayList(Ast.Index)) Error!void {
    // `import` decls have no `pub` modifier (imports are not re-exported).
    if (p.at(.kw_import)) {
        try decls.append(p.gpa, try p.parseImport());
        return;
    }
    // `impl` blocks have no `pub` modifier either: a method's export follows its
    // receiver type, not an `impl`-level keyword (keeps methods module-private).
    if (p.at(.kw_impl)) {
        try decls.append(p.gpa, try p.parseImplDecl());
        return;
    }
    // An optional `pub` modifier precedes a fn/struct/enum decl and exports it.
    const is_pub = p.eat(.kw_pub);
    const decl = switch (p.peek().tag) {
        .kw_fn => try p.parseFnDecl(.top_level),
        .kw_extern => try p.parseExternFn(),
        .kw_struct => try p.parseStructDecl(),
        .kw_enum => try p.parseEnumDecl(),
        .kw_protocol => try p.parseProtocolDecl(),
        .kw_type => try p.parseTypeAlias(),
        else => return p.fail(p.peek(), .P0003, if (is_pub)
            "expected a function, struct, or enum declaration after 'pub'"
        else
            "expected a function, struct, enum, or import declaration"),
    };
    if (is_pub) try p.pub_decls.append(p.gpa, decl);
    try decls.append(p.gpa, decl);
}

/// Materialize the `pub_bits` bitset from the collected `pub_decls` node indices.
/// Returns an empty slice when nothing is exported (the common single-file case),
/// so non-module programs pay nothing.
fn buildPubBits(p: *Parser, node_count: usize) error{OutOfMemory}![]u32 {
    if (p.pub_decls.items.len == 0) return &.{};
    const words = Ast.pubBitsLen(node_count);
    const bits = try p.gpa.alloc(u32, words);
    @memset(bits, 0);
    for (p.pub_decls.items) |idx| {
        const i = idx.int();
        bits[i >> 5] |= @as(u32, 1) << @intCast(i & 31);
    }
    return bits;
}

/// `import a/b/c [as alias]`. Path segments are `/`-separated identifiers. The
/// node stores the segment TOKEN indices (a `Range` in `extra`) and the alias
/// token (or `Ast.none`); `main_token` is the last segment (the default bind).
/// `/` appears ONLY here; `.` only in access — so there is no parse ambiguity.
fn parseImport(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_import, "expected 'import'");
    var segs: std.ArrayList(Ast.TokIndex) = .empty;
    defer segs.deinit(p.gpa);
    const first = p.index;
    try p.expect(.identifier, "expected a module path after 'import'");
    try segs.append(p.gpa, Ast.TokIndex.from(first));
    while (p.eat(.slash)) {
        const seg = p.index;
        try p.expect(.identifier, "expected a path segment after '/'");
        try segs.append(p.gpa, Ast.TokIndex.from(seg));
    }
    var alias: Ast.TokIndex = Ast.TokIndex.none;
    if (p.eat(.kw_as)) {
        alias = Ast.TokIndex.from(p.index);
        try p.expect(.identifier, "expected an alias name after 'as'");
    }
    const last_seg = segs.items[segs.items.len - 1];
    const header = try p.addTokRange(segs.items);
    // The lhs holds a range header (a node-index slot by field type), and rhs
    // holds the alias TOKEN index; both are stored in the `Index` fields but the
    // decoders (`Ast.importPathToks`/`Ast.importAliasTok`) read them as tokens.
    return p.addNode(.{
        .tag = .import_decl,
        .main_token = last_seg.int(),
        .lhs = header,
        .rhs = Ast.Index.from(alias.int()),
    });
}

/// The four legal shapes a `fn` decl can take, one per call site. Modelled as a tagged
/// union so the receiver payload attaches ONLY to the two method forms and the derived
/// body-less / self-optional / self-synthesis behaviour comes from the tag — an illegal
/// combination is unrepresentable rather than merely discouraged by a comment.
///
///   - `top_level`: an ordinary `fn`; no receiver, so a bare `self` just parses as a
///     param named `self`. Parses a real body.
///   - `protocol_sig`: a protocol method SIGNATURE (`fn m(self, ..) -> R` with no
///     `{ .. }`). `proto_tok` is the protocol name, used as the synthetic `self`
///     receiver token; the body is a synthesized empty `block` node (bodyless) so
///     downstream shape checks and pack/unpack stay uniform.
///   - `inherent_method` / `conformance_method`: an impl method carrying a `Recv`.
///     `recv_tok` is the impl's receiver type-name token, so a leading bare `self` (a
///     plain identifier recognized by TEXT — the `_`-wildcard precedent) is consumed and
///     synthesized into `params[0] = self: <Receiver>`. Its type-ref is `self_type_ref`
///     when the impl supplies one (a `type_app` `Box[T]` for a generic receiver),
///     else a fresh `identifier` on the receiver token (a bare impl — byte-identical).
///     `impl_gparams` are the impl's generic-param nodes (`[T]`), PREPENDED into this
///     method's FnProto generic run so the method becomes a bona-fide generic template
///     (its `protoAt().generic_params.len > 0`), monomorphized per (type-instance,
///     method) exactly like a generic fn. Only `conformance_method` (`impl T has P`)
///     may OMIT the leading `self` (e.g. `fn from(s: Src) -> Self`): the "a method must
///     take 'self'" P0006 is suppressed and the T0024 coherence check governs correctness
///     against the protocol's sig. `inherent_method` still requires `self`.
const FnKind = union(enum) {
    top_level,
    protocol_sig: struct { proto_tok: u32 },
    inherent_method: Recv,
    conformance_method: Recv,
    /// A bodyless `extern fn` C-ABI declaration: no receiver, no generics, no body
    /// (like `protocol_sig` it synthesizes an empty block). Tagged `extern_fn_decl`
    /// at `addNode` so resolve/typecheck register it as a dyld import.
    extern_top_level,

    const Recv = struct { recv_tok: u32, self_type_ref: Ast.Index, impl_gparams: []const Ast.Index };
};

fn parseFnDecl(p: *Parser, kind: FnKind) Error!Ast.Index {
    const self_recv_tok: ?u32 = switch (kind) {
        .top_level, .extern_top_level => null,
        .protocol_sig => |ps| ps.proto_tok,
        .inherent_method, .conformance_method => |recv| recv.recv_tok,
    };
    const self_type_ref: Ast.Index = switch (kind) {
        .inherent_method, .conformance_method => |recv| recv.self_type_ref,
        else => Ast.none,
    };
    const impl_gparams: []const Ast.Index = switch (kind) {
        .inherent_method, .conformance_method => |recv| recv.impl_gparams,
        else => &.{},
    };
    const bodyless = switch (kind) {
        .protocol_sig, .extern_top_level => true,
        else => false,
    };
    // Both impl-method forms may declare a SELF-LESS method: a `conformance_method`
    // (`fn from(s: Src) -> Self`, governed by the T0024 coherence check) and an
    // `inherent_method` (an ASSOCIATED function `fn new() -> Vec[T]`, dispatched via
    // `Type[args].new()`). The receiver token is still recorded, so a leading `self`
    // is consumed when present; only the "must take 'self'" P0006 is lifted.
    const self_optional = switch (kind) {
        .conformance_method, .inherent_method => true,
        else => false,
    };

    try p.expect(.kw_fn, "expected 'fn'");
    const name_tok = p.index;
    try p.expect(.identifier, "expected a function name");

    // Optional generic-param list `[T, U]` between the name and the `(`. Its
    // `generic_param` nodes are created here (before the params/body), so they
    // still precede the `fn_decl` node.
    var generics: std.ArrayList(Ast.Index) = .empty;
    defer generics.deinit(p.gpa);
    if (p.at(.l_bracket)) try p.parseGenericParams(&generics);
    // An `extern fn` is a bare C-ABI symbol: it cannot be generic. Report and drop
    // any generic params so the rest of the signature still parses.
    if (kind == .extern_top_level and generics.items.len > 0) {
        try p.warn(p.peek(), .P0003, "'extern' functions cannot be generic");
        generics.clearRetainingCapacity();
    }

    try p.expect(.l_paren, "expected '(' after function name");

    var params: std.ArrayList(Ast.Index) = .empty;
    defer params.deinit(p.gpa);
    // A method's leading `self` receiver: a bare `self` with no `:` annotation
    // (peek2 != colon) is the by-value receiver — synthesize `params[0] = self:
    // <Receiver>` (child nodes created before the fn_decl, so children precede
    // parents). Its type-ref is a synthetic `identifier` on the impl's receiver
    // token. A method with no leading `self` is a static/associated fn, which is out
    // of scope: report P0006 and keep parsing the rest for recovery.
    if (self_recv_tok) |recv_tok| {
        // A `mut self` receiver: consume the leading `mut` ONLY when it qualifies
        // `self`, leaving `main_token` on the `self` token so `tokens[self_tok-1]` is
        // `kw_mut` (how `Ast.isMutParam` later detects mut-ness — no node-shape change).
        // A `mut` on a non-self first param is out of scope: report P0006 once and drop
        // the `mut` so the ordinary param loop still parses the rest.
        const has_mut = p.at(.kw_mut);
        const mut_self = has_mut and p.peek2().tag == .identifier and std.mem.eql(u8, p.peek2().text(p.src), "self");
        if (has_mut and !mut_self) try p.warn(p.peek(), .P0006, "'mut' may only qualify a 'self' receiver");
        if (has_mut) p.bump(.kw_mut);
        if (p.at(.identifier) and std.mem.eql(u8, p.peek().text(p.src), "self") and p.peek2().tag != .colon) {
            const self_tok = p.index;
            p.bump(.identifier);
            // A generic impl hands down its `type_app` receiver (`Box[T]`) so the
            // self param decodes to an `App`; a bare impl synthesizes a fresh
            // `identifier` on the receiver token (byte-identical to the old shape).
            const recv_ref = if (self_type_ref != Ast.none) self_type_ref else try p.addNode(.{ .tag = .identifier, .main_token = recv_tok, .lhs = Ast.none, .rhs = Ast.none });
            const self_param = try p.addNode(.{ .tag = .param, .main_token = self_tok, .lhs = recv_ref, .rhs = Ast.none });
            try params.append(p.gpa, self_param);
            _ = p.eat(.comma); // separator before the next param, if any
        } else if (!has_mut and !self_optional) {
            // A conformance impl (`self_optional`) may declare a SELF-LESS method
            // (`impl BigErr has From[SmallErr] { fn from(s: SmallErr) -> BigErr {..} }`):
            // the T0024 coherence signature check governs correctness there (a self-less
            // `from` matches From's self-less protocol sig). An INHERENT impl still requires
            // `self` — static/associated methods stay out of scope — so the diagnostic holds.
            try p.warn(p.peek(), .P0006, "a method must take 'self' as its first parameter");
        }
    }
    while (!p.at(.r_paren) and !p.at(.eof)) {
        const entry = p.index;
        if (p.at(.identifier)) {
            const param_name = p.index;
            p.bump(.identifier);
            try p.expect(.colon, "expected ':' after parameter name");
            const type_node = try p.parseType();
            const param = try p.addNode(.{ .tag = .param, .main_token = param_name, .lhs = type_node, .rhs = Ast.none });
            try params.append(p.gpa, param);
            if (!p.eat(.comma)) {
                if (p.at(.r_paren)) break;
            }
        } else if (param_recovery.contains(p.peek().tag)) {
            break;
        } else {
            _ = try p.advanceWithError(.P0006, "expected a parameter name or ')'");
        }
        std.debug.assert(p.index > entry or p.at(.r_paren) or p.at(.eof));
    }
    try p.expect(.r_paren, "expected ')' to close parameter list");

    var ret_type: Ast.Index = Ast.none;
    if (p.eat(.arrow)) {
        ret_type = try p.parseType();
    }

    // A protocol method signature has no body: synthesize an empty `block` (its
    // `main_token` is the current token — a leaf with an empty statement Range) so
    // the node shape matches an ordinary fn and pack/unpack stays uniform.
    const body = if (bodyless)
        try p.addNode(.{ .tag = .block, .main_token = p.index, .lhs = try p.addRange(&.{}), .rhs = Ast.none })
    else
        try p.parseBlock();

    // Write the params run, then the generics run, then the fixed 5-cell FnProto.
    // The layout is ADDITIVE: cells 0-2 (ret/params) are unchanged, so every
    // 3-cell decode site reads byte-identically; cells 3-4 hold the generics run.
    // An empty generics run leaves `generic_start` at the current extra length and
    // `generic_len` 0, which `protoAt` decodes to an empty (safe) slice.
    //
    // A generic impl's params (`[T]`) come FIRST in the run, then the method's
    // own `[U]` (parsed above but NOT yet inference-bound — deferred; only the impl's
    // params are matched at a call site). Both `impl_gparams` and `generics` are empty
    // for a bare impl / a top-level fn, so the run stays byte-identical there.
    const params_start: u32 = @intCast(p.extra.items.len);
    const param_cells: []const u32 = @ptrCast(params.items);
    try p.extra.appendSlice(p.gpa, param_cells);
    const generic_start: u32 = @intCast(p.extra.items.len);
    const impl_gp_cells: []const u32 = @ptrCast(impl_gparams);
    try p.extra.appendSlice(p.gpa, impl_gp_cells);
    const generic_cells: []const u32 = @ptrCast(generics.items);
    try p.extra.appendSlice(p.gpa, generic_cells);
    const proto_header = try p.addExtra(&.{
        ret_type.int(),
        params_start,
        @intCast(params.items.len),
        generic_start,
        @intCast(impl_gparams.len + generics.items.len),
    });

    const decl_tag: Ast.Node.Tag = if (kind == .extern_top_level) .extern_fn_decl else .fn_decl;
    return p.addNode(.{ .tag = decl_tag, .main_token = name_tok, .lhs = proto_header, .rhs = body });
}

/// `extern fn name(params) -> R` — a bodyless C-ABI declaration. Consumes the
/// leading `extern`, then reuses `parseFnDecl`'s bodyless path.
fn parseExternFn(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_extern, "expected 'extern'");
    return p.parseFnDecl(.extern_top_level);
}

/// `impl Type { fn m(self, ..) -> R { .. } }` — an inherent-method block —
/// OR `impl T has P { fn m.. {} }` — a protocol-conformance block. The receiver
/// is a bare type name (`impl P`), a GENERIC type application (`impl Box[T]` —
/// inherent only), or (conformance only) a QUALIFIED name (`impl mod.T has mod.P`).
/// `has` selects the conformance form: the protocol reference is a bare/qualified name
/// node, a generic (`type_app`) receiver is rejected, and the block is
/// an `impl_has_decl`. WITHOUT `has`, a qualified receiver is still rejected (inherent
/// impls stay bare-or-generic). The member loop is shared (`parseImplBody`);
/// each method is parsed with the receiver so its leading `self` is synthesized.
///
/// NODE SHARING (deliberate, safe): for a generic impl the ONE `type_app` node is both
/// `impl_decl.lhs` and every method's `self` type-ref, and its `generic_param` leaves
/// are shared with each method's FnProto generic run. All leaves are created before
/// every referencing parent (child<self still holds), the graph is a cycle-free DAG
/// (fine for the content fp + pack/unpack, which memcpy the flat node/extra arrays),
/// and the runtime invariant sweep only checks span totality + bracket pairing.
fn parseImplDecl(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_impl, "expected 'impl'");
    // A qualified `impl mod.T` receiver (coherence) builds a left-nested
    // `field_access` chain and keys `recv_tok` off the LAST segment (the type name).
    // Only a conformance impl (`has`) may be qualified; an inherent impl rejects it.
    const name_tok = p.index;
    const qname = try p.parseQualifiedName("expected a type name after 'impl'", "expected a type name after '.'");
    const recv_ref = qname.node;
    const recv_tok = qname.last_tok;
    const qualified = recv_tok != name_tok;
    // A generic receiver `impl Box[T]`: `[T]` declares the impl's type-params
    // (each a `generic_param` leaf) and the receiver becomes a `type_app`. The leaves
    // are prepended into every method's generic run (see `parseFnDecl`).
    var impl_gparams: std.ArrayList(Ast.Index) = .empty;
    defer impl_gparams.deinit(p.gpa);
    var recv_node: Ast.Index = recv_ref;
    var is_generic = false;
    if (p.at(.l_bracket)) {
        const lbracket = p.index;
        try p.parseGenericParams(&impl_gparams);
        const args_range = try p.addRange(impl_gparams.items);
        recv_node = try p.addNode(.{ .tag = .type_app, .main_token = lbracket, .lhs = recv_ref, .rhs = args_range });
        is_generic = true;
    }

    var methods: std.ArrayList(Ast.Index) = .empty;
    defer methods.deinit(p.gpa);

    // A conformance impl `impl T has P { .. }`.
    if (p.eat(.kw_has)) {
        // A generic (`Box[T]`) receiver is a BOUND (`impl Box[T] has P`); reject it
        // cleanly rather than mint an unsupported shape.
        if (is_generic) return p.fail(p.peek(), .P0001, "a generic 'impl ... has' receiver is not yet supported");
        const proto_ref = try p.parseProtocolRef();
        // A conformance impl permits self-less methods; coherence (T0024) then
        // governs signature correctness against the protocol's declared sig.
        try p.parseImplBody(recv_tok, recv_node, impl_gparams.items, &methods, true);
        // Write the method run, then the fixed 3-cell header {protocol_ref, start, len}
        // — decoded by `Ast.implHasAt`. The run precedes the header (start < header), so
        // every method node is created before the `impl_has_decl` node references it.
        const methods_start: u32 = @intCast(p.extra.items.len);
        const method_cells: []const u32 = @ptrCast(methods.items);
        try p.extra.appendSlice(p.gpa, method_cells);
        const header = try p.addExtra(&.{ proto_ref.int(), methods_start, @intCast(methods.items.len) });
        return p.addNode(.{ .tag = .impl_has_decl, .main_token = recv_tok, .lhs = recv_node, .rhs = header });
    }

    // Inherent impl: a qualified receiver is out of scope.
    if (qualified) return p.fail(p.peek(), .P0001, "an inherent 'impl' receiver must be a bare or generic type name");
    try p.parseImplBody(recv_tok, recv_node, impl_gparams.items, &methods, false);
    const header = try p.addRange(methods.items);
    return p.addNode(.{ .tag = .impl_decl, .main_token = recv_tok, .lhs = recv_node, .rhs = header });
}

/// The shared impl-member loop `{ fn m.. {}  fn n.. {} }`: parse each `fn` method with
/// the receiver so its leading `self` is synthesized, appending into `methods`
/// (newline/comma-separated). Used by both the inherent and conformance impl forms, so
/// the two cannot drift on member grammar / recovery.
fn parseImplBody(p: *Parser, recv_tok: u32, recv_node: Ast.Index, impl_gparams: []const Ast.Index, methods: *std.ArrayList(Ast.Index), self_optional: bool) Error!void {
    try p.expect(.l_brace, "expected '{' after the impl receiver type");
    while (!p.at(.eof)) {
        p.skipNewlines();
        if (p.at(.r_brace)) break;
        // `kw_fn` is the one decl keyword that legitimately STARTS an impl member, so
        // it is NOT a bail anchor here (unlike the struct-body loop); any OTHER decl
        // keyword means the impl body is wrecked → bail to the decl loop.
        if (!p.at(.kw_fn) and decl_anchors.contains(p.peek().tag)) break;
        const entry = p.index;
        if (p.at(.kw_fn)) {
            const recv: FnKind.Recv = .{ .recv_tok = recv_tok, .self_type_ref = recv_node, .impl_gparams = impl_gparams };
            const method = try p.parseFnDecl(if (self_optional) .{ .conformance_method = recv } else .{ .inherent_method = recv });
            try methods.append(p.gpa, method);
            // A method is separated by a newline (loop-top `skipNewlines`) or an
            // optional comma; a `}` ends the block.
            _ = p.eat(.comma);
        } else {
            _ = try p.advanceWithError(.P0006, "expected a method 'fn' or '}'");
        }
        std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
    }
    try p.expect(.r_brace, "expected '}' to close the impl block");
}

/// Parse a bare or dot-qualified protocol reference (`P` / `mod.P` / `a.b.P`) into an
/// `identifier` (bare) or a left-nested `field_access` chain (qualified) — the same
/// shape a qualified type-name uses, so the checker resolves it via the module tables.
/// A trailing `[..]` (generic protocol) wraps the ref in a `type_app` (`P[int]` ->
/// `type_app(P, [int])`), the SAME shape a generic type-application uses. NOT otherwise a
/// full type: no `()` unit. The built node lands in the `impl_has_decl`'s 3-cell header
/// or in a `generic_param`'s `lhs` bound slot (`[T has P[int]]`).
fn parseProtocolRef(p: *Parser) Error!Ast.Index {
    var node = (try p.parseQualifiedName("expected a protocol name after 'has'", "expected a protocol name after '.'")).node;
    // A generic protocol reference `P[int, ..]`: reuse the same `type_app` shape a
    // generic type-application uses (`parseTypeApp` wraps the base ref into a `type_app`).
    if (p.at(.l_bracket)) node = try p.parseTypeApp(node);
    return node;
}

/// `protocol P { fn m(self, ..) -> R }` — a signature-only protocol declaration.
/// Each member is a signature-only method parsed via `parseFnDecl(.., .protocol_sig)`
/// (bodyless is derived from `kind == .protocol_sig`), with the protocol name as the
/// synthetic receiver token (so a leading `self` is
/// consumed and its type-ref renders as the protocol name — inert, never decoded: a
/// protocol's method sigs never enter the fn table). The member loop mirrors the
/// impl-body loop (newline/comma-separated `fn`s).
fn parseProtocolDecl(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_protocol, "expected 'protocol'");
    const name_tok = p.index;
    try p.expect(.identifier, "expected a protocol name");

    // Optional generic-param list `[X, ..]` between the name and the body `{`,
    // parsed by the SAME `parseGenericParams` a generic struct/enum/fn uses. Rides the
    // otherwise-`none` `rhs` slot as a Range header, keeping non-generic protocols
    // byte-identical.
    var generics: std.ArrayList(Ast.Index) = .empty;
    defer generics.deinit(p.gpa);
    if (p.at(.l_bracket)) try p.parseGenericParams(&generics);

    try p.expect(.l_brace, "expected '{' after the protocol name");

    var sigs: std.ArrayList(Ast.Index) = .empty;
    defer sigs.deinit(p.gpa);
    while (!p.at(.eof)) {
        p.skipNewlines();
        if (p.at(.r_brace)) break;
        if (!p.at(.kw_fn) and decl_anchors.contains(p.peek().tag)) break;
        const entry = p.index;
        if (p.at(.kw_fn)) {
            const sig = try p.parseFnDecl(.{ .protocol_sig = .{ .proto_tok = name_tok } });
            try sigs.append(p.gpa, sig);
            _ = p.eat(.comma);
        } else {
            _ = try p.advanceWithError(.P0006, "expected a method 'fn' or '}'");
        }
        std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
    }
    try p.expect(.r_brace, "expected '}' to close the protocol block");

    const header = try p.addRange(sigs.items);
    const generic_hdr = try p.optRange(generics.items);
    return p.addNode(.{ .tag = .protocol_decl, .main_token = name_tok, .lhs = header, .rhs = generic_hdr });
}

/// `struct Name { x: int, y: int }`. Fields are `name: Type`, comma-separated,
/// with newlines insignificant inside `{}`. Reuses the `.param` node for fields.
fn parseStructDecl(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_struct, "expected 'struct'");
    const name_tok = p.index;
    try p.expect(.identifier, "expected a struct name");

    // Optional generic-param list `[T]` between the name and the body `{`.
    var generics: std.ArrayList(Ast.Index) = .empty;
    defer generics.deinit(p.gpa);
    if (p.at(.l_bracket)) try p.parseGenericParams(&generics);

    if (p.at(.l_paren)) {
        p.bump(.l_paren);
        var types: std.ArrayList(Ast.Index) = .empty;
        defer types.deinit(p.gpa);
        while (!p.at(.r_paren) and !p.at(.eof)) {
            const t_entry = p.index;
            if (type_first.contains(p.peek().tag)) {
                try types.append(p.gpa, try p.parseType());
                if (!p.eat(.comma)) {
                    if (p.at(.r_paren)) break;
                }
            } else if (tuple_recovery.contains(p.peek().tag)) {
                break;
            } else {
                _ = try p.advanceWithError(.P0006, "expected a type");
            }
            std.debug.assert(p.index > t_entry or p.at(.r_paren) or p.at(.eof));
        }
        try p.expect(.r_paren, "expected ')' to close a tuple struct");
        const header = try p.addRange(types.items);
        const generic_hdr = try p.optRange(generics.items);
        return p.addNode(.{ .tag = .tuple_struct_decl, .main_token = name_tok, .lhs = header, .rhs = generic_hdr });
    }

    try p.expect(.l_brace, "expected '{' after struct name");

    var fields: std.ArrayList(Ast.Index) = .empty;
    defer fields.deinit(p.gpa);
    while (!p.at(.eof)) {
        p.skipNewlines();
        if (p.at(.r_brace)) break;
        if (decl_anchors.contains(p.peek().tag)) break;
        const entry = p.index;
        if (p.at(.identifier)) {
            const field_name = p.index;
            p.bump(.identifier);
            try p.expect(.colon, "expected ':' after field name");
            const type_node = try p.parseType();
            const field = try p.addNode(.{ .tag = .param, .main_token = field_name, .lhs = type_node, .rhs = Ast.none });
            try fields.append(p.gpa, field);
            // A field is separated by a comma OR a newline (both insignificant
            // inside `{}`); a `}` ends the list. The comma is optional; the loop
            // top skips newlines and checks for `}`.
            _ = p.eat(.comma);
        } else {
            _ = try p.advanceWithError(.P0006, "expected a field name or '}'");
        }
        std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
    }
    try p.expect(.r_brace, "expected '}' to close struct body");

    const header = try p.addRange(fields.items);
    // Generics ride the otherwise-unused `rhs` slot as a Range header (or `none`
    // for a non-generic struct, keeping every existing struct byte-identical).
    const generic_hdr = try p.optRange(generics.items);
    return p.addNode(.{ .tag = .struct_decl, .main_token = name_tok, .lhs = header, .rhs = generic_hdr });
}

/// `type Name = <type-ref>` — a module-local transparent alias. Reuses `.eq`
/// (distinct from the `as` import-alias token) and the shared type-ref parser
/// (`parseType`, which carries the recursion-depth guard), so the target may be a
/// bare/qualified name or `()`. `main_token` = the alias name; `lhs` = the target
/// type-ref; `rhs` = none. Registration resolves it to the target's own `Type`.
fn parseTypeAlias(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_type, "expected 'type'");
    const name_tok = p.index;
    try p.expect(.identifier, "expected a type alias name");
    try p.expect(.eq, "expected '=' after the type alias name");
    const target = try p.parseType();
    return p.addNode(.{ .tag = .type_alias_decl, .main_token = name_tok, .lhs = target, .rhs = Ast.none });
}

/// `enum N { Empty, Circle(int), Rect { w: int, h: int } }`. Variants are
/// comma-separated, newlines insignificant inside `{}`. Three forms: unit,
/// tuple (positional payload types), struct (named `param` fields).
fn parseEnumDecl(p: *Parser) Error!Ast.Index {
    try p.expect(.kw_enum, "expected 'enum'");
    const name_tok = p.index;
    try p.expect(.identifier, "expected an enum name");

    // Optional generic-param list `[T]` between the name and the body `{`.
    var generics: std.ArrayList(Ast.Index) = .empty;
    defer generics.deinit(p.gpa);
    if (p.at(.l_bracket)) try p.parseGenericParams(&generics);

    try p.expect(.l_brace, "expected '{' after enum name");

    var variants: std.ArrayList(Ast.Index) = .empty;
    defer variants.deinit(p.gpa);
    while (!p.at(.eof)) {
        p.skipNewlines();
        if (p.at(.r_brace)) break;
        if (decl_anchors.contains(p.peek().tag)) break;
        const entry = p.index;
        if (!p.at(.identifier)) {
            _ = try p.advanceWithError(.P0006, "expected a variant name or '}'");
            std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
            continue;
        }
        const vname = p.index;
        p.bump(.identifier);
        var variant: Ast.Index = undefined;
        switch (p.peek().tag) {
            .l_paren => {
                p.bump(.l_paren);
                var types: std.ArrayList(Ast.Index) = .empty;
                defer types.deinit(p.gpa);
                while (!p.at(.r_paren) and !p.at(.eof)) {
                    const t_entry = p.index;
                    if (type_first.contains(p.peek().tag)) {
                        try types.append(p.gpa, try p.parseType());
                        if (!p.eat(.comma)) {
                            if (p.at(.r_paren)) break;
                        }
                    } else if (tuple_recovery.contains(p.peek().tag)) {
                        break;
                    } else {
                        _ = try p.advanceWithError(.P0006, "expected a type");
                    }
                    std.debug.assert(p.index > t_entry or p.at(.r_paren) or p.at(.eof));
                }
                try p.expect(.r_paren, "expected ')' to close a tuple variant");
                const header = try p.addRange(types.items);
                variant = try p.addNode(.{ .tag = .enum_variant_tuple, .main_token = vname, .lhs = header, .rhs = Ast.none });
            },
            .l_brace => {
                var nb = NoBlockScope.enter(p, false);
                defer nb.end();
                p.bump(.l_brace);
                var fields: std.ArrayList(Ast.Index) = .empty;
                defer fields.deinit(p.gpa);
                while (!p.at(.eof)) {
                    p.skipNewlines();
                    if (p.at(.r_brace)) break;
                    if (decl_anchors.contains(p.peek().tag)) break;
                    const f_entry = p.index;
                    if (p.at(.identifier)) {
                        const field_name = p.index;
                        p.bump(.identifier);
                        try p.expect(.colon, "expected ':' after field name");
                        const type_node = try p.parseType();
                        const field = try p.addNode(.{ .tag = .param, .main_token = field_name, .lhs = type_node, .rhs = Ast.none });
                        try fields.append(p.gpa, field);
                        _ = p.eat(.comma);
                    } else {
                        _ = try p.advanceWithError(.P0006, "expected a field name or '}'");
                    }
                    std.debug.assert(p.index > f_entry or p.at(.r_brace) or p.at(.eof));
                }
                try p.expect(.r_brace, "expected '}' to close a struct variant");
                const header = try p.addRange(fields.items);
                variant = try p.addNode(.{ .tag = .enum_variant_struct, .main_token = vname, .lhs = header, .rhs = Ast.none });
            },
            else => variant = try p.addNode(.{ .tag = .enum_variant_unit, .main_token = vname, .lhs = Ast.none, .rhs = Ast.none }),
        }
        try variants.append(p.gpa, variant);
        _ = p.eat(.comma);
        std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
    }
    try p.expect(.r_brace, "expected '}' to close enum body");

    const header = try p.addRange(variants.items);
    // Generics ride the otherwise-unused `rhs` slot as a Range header (see struct).
    const generic_hdr = try p.optRange(generics.items);
    return p.addNode(.{ .tag = .enum_decl, .main_token = name_tok, .lhs = header, .rhs = generic_hdr });
}

/// `match scrut { pat -> body, ... }`. Scrutinee parsed in `no_block` (so a bare
/// `match x { ... }` reads `x`, the `{` opening the arm list). Arms are comma-
/// separated, newlines insignificant inside `{}`.
fn parseMatch(p: *Parser) Error!Ast.Index {
    const match_tok = p.index;
    p.bump(.kw_match);
    // Scrutinee in `no_block` (a bare `match x { ... }` reads `x`, the `{` opening
    // the arm list); then the arm list parses with blocks re-allowed, restoring the
    // prior flag once the whole `match` is done.
    const scrut = scrut: {
        var scrut_nb = NoBlockScope.enter(p, true);
        defer scrut_nb.end();
        break :scrut try p.parseExpr(0);
    };
    var arms_nb = NoBlockScope.enter(p, false);
    defer arms_nb.end();
    try p.expect(.l_brace, "expected '{' to open a match");

    var arms: std.ArrayList(Ast.Index) = .empty;
    defer arms.deinit(p.gpa);
    while (!p.at(.eof)) {
        p.skipNewlines();
        if (p.at(.r_brace)) break;
        if (decl_anchors.contains(p.peek().tag)) break;
        const entry = p.index;
        // An arm is a compound (pattern/guard/arrow/body), so recover it by
        // catch-and-resync rather than a bare-item 3-way arm.
        const arm = p.parseMatchArm() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.ParseError => {
                p.resyncTo(arm_recovery);
                // `resyncTo` STOPS on (without consuming) the first anchor at depth 0,
                // so when the arm failed with the cursor already on a leading
                // separator (`,`/newline) it returns having moved nothing. Consume one
                // token in that case to guarantee forward progress (the loop-top
                // `skipNewlines` and trailing `eat(.comma)` already treat separators as
                // optional, so dropping one is safe). Never step past the arm-list end.
                if (p.index == entry and !p.at(.eof) and !p.at(.r_brace) and !decl_anchors.contains(p.peek().tag)) p.advance();
                std.debug.assert(p.index > entry or p.at(.r_brace) or decl_anchors.contains(p.peek().tag));
                continue;
            },
        };
        try arms.append(p.gpa, arm);
        _ = p.eat(.comma);
        std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
    }
    try p.expect(.r_brace, "expected '}' to close match");

    const header = try p.addRange(arms.items);
    return p.addNode(.{ .tag = .match_expr, .main_token = match_tok, .lhs = scrut, .rhs = header });
}

/// One `pat [if guard] -> body` match arm.
fn parseMatchArm(p: *Parser) Error!Ast.Index {
    const pat = try p.parsePattern();
    var guard: Ast.Index = Ast.none;
    if (p.eat(.kw_if)) {
        var guard_nb = NoBlockScope.enter(p, true); // stop the guard cond before `->`/`{`
        defer guard_nb.end();
        guard = try p.parseExpr(0);
    }
    const arrow = p.index;
    try p.expect(.arrow, "expected '->' after a match pattern");
    const body = try p.parseExpr(0);
    const arm_hdr = try p.addExtra(&.{ guard.int(), body.int() });
    return p.addNode(.{ .tag = .match_arm, .main_token = arrow, .lhs = pat, .rhs = arm_hdr });
}

/// A match pattern, with or-alternatives: `subpat ('|' subpat)*`. Emits a
/// `pattern_or` only when there are >=2 alternatives; otherwise the bare subpat.
fn parsePattern(p: *Parser) Error!Ast.Index {
    const first = try p.parseSubPattern();
    if (!p.at(.pipe)) return first;
    var alts: std.ArrayList(Ast.Index) = .empty;
    defer alts.deinit(p.gpa);
    const first_tok = p.nodes.items[first.int()].main_token;
    try alts.append(p.gpa, first);
    while (p.eat(.pipe)) {
        const entry = p.index;
        try alts.append(p.gpa, try p.parseSubPattern());
        std.debug.assert(p.index > entry);
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
            p.bump(.l_paren);
            var binds: std.ArrayList(Ast.Index) = .empty;
            defer binds.deinit(p.gpa);
            while (!p.at(.r_paren) and !p.at(.eof)) {
                const entry = p.index;
                if (pattern_first.contains(p.peek().tag)) {
                    // Each tuple element is an arbitrary sub-pattern (literal,
                    // binding, wildcard, nested variant, or-pattern).
                    try binds.append(p.gpa, try p.parsePattern());
                    if (!p.eat(.comma)) {
                        if (p.at(.r_paren)) break;
                    }
                } else if (tuple_recovery.contains(p.peek().tag)) {
                    break;
                } else {
                    _ = try p.advanceWithError(.P0008, "expected a pattern");
                }
                std.debug.assert(p.index > entry or p.at(.r_paren) or p.at(.eof));
            }
            try p.expect(.r_paren, "expected ')' to close a tuple pattern");
            binders = try p.addRange(binds.items);
        },
        .l_brace => {
            var nb = NoBlockScope.enter(p, false);
            defer nb.end();
            p.bump(.l_brace);
            var binds: std.ArrayList(Ast.Index) = .empty;
            defer binds.deinit(p.gpa);
            while (!p.at(.eof)) {
                p.skipNewlines();
                if (p.at(.r_brace)) break;
                if (decl_anchors.contains(p.peek().tag)) break;
                const entry = p.index;
                if (!p.at(.identifier)) {
                    _ = try p.advanceWithError(.P0006, "expected a field name in a struct pattern");
                    std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
                    continue;
                }
                const field_tok = p.index;
                p.bump(.identifier);
                var bind: Ast.Index = undefined;
                if (p.eat(.colon)) {
                    // `field: alias` (rename to a bare ident) vs `field: subpat`
                    // (a literal/`.`/`_`/nested pattern matched against the field).
                    // A bare identifier NOT opening a payload is the rename alias.
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
                _ = p.eat(.comma);
                std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
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
    p.bump(.l_brace);
    var nb = NoBlockScope.enter(p, false);
    defer nb.end();

    var inits: std.ArrayList(Ast.Index) = .empty;
    defer inits.deinit(p.gpa);
    while (!p.at(.eof)) {
        p.skipNewlines();
        if (p.at(.r_brace)) break;
        if (decl_anchors.contains(p.peek().tag)) break;
        const entry = p.index;
        if (!p.at(.identifier)) {
            _ = try p.advanceWithError(.P0006, "expected a field name or '}'");
            std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
            continue;
        }
        const field_tok = p.index;
        p.bump(.identifier);
        var value: Ast.Index = undefined;
        if (p.eat(.colon)) {
            value = try p.parseExpr(0);
        } else {
            // Punning shorthand: synthesize an identifier leaf on the field token.
            value = try p.addNode(.{ .tag = .identifier, .main_token = field_tok, .lhs = Ast.none, .rhs = Ast.none });
        }
        const fi = try p.addNode(.{ .tag = .field_init, .main_token = field_tok, .lhs = value, .rhs = Ast.none });
        try inits.append(p.gpa, fi);
        _ = p.eat(.comma);
        std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
    }
    try p.expect(.r_brace, "expected '}' to close struct literal");

    const header = try p.addRange(inits.items);
    return p.addNode(.{ .tag = .struct_init, .main_token = lbrace, .lhs = name_ident, .rhs = header });
}

/// `recv.field`. Consumes the `.` then the field-name identifier.
fn parseFieldAccess(p: *Parser, recv: Ast.Index) Error!Ast.Index {
    p.bump(.dot);
    if (p.at(.number)) {
        const num_tok = p.index;
        p.bump(.number);
        return p.addNode(.{ .tag = .tuple_field, .main_token = num_tok, .lhs = recv, .rhs = Ast.none });
    }
    const field_tok = p.index;
    try p.expect(.identifier, "expected a field name after '.'");
    return p.addNode(.{ .tag = .field_access, .main_token = field_tok, .lhs = recv, .rhs = Ast.none });
}

/// `[ ident (, ident)* ]` — a generic-parameter list following a decl name. Each
/// param becomes a `generic_param` leaf node appended to `out` (in source order,
/// created BEFORE the owning decl node so children still precede parents). Called
/// only when the cursor is already on `[`. Bracket-aware recovery: a non-identifier
/// is a P0006, resyncing to the list's own `]`/`,` (unioned with decl anchors so a
/// runaway bails to the next decl). A missing `]` unwinds to the decl loop.
fn parseGenericParams(p: *Parser, out: *std.ArrayList(Ast.Index)) Error!void {
    p.bump(.l_bracket);
    while (!p.at(.r_bracket) and !p.at(.eof)) {
        const entry = p.index;
        if (p.at(.identifier)) {
            const name_tok = p.index;
            p.bump(.identifier);
            // A constrained param `T has P` / `T has P[int]`: the bound
            // protocol-ref (created BEFORE the owning `generic_param` node, so children
            // precede parents) is stored in `lhs`; an unbounded `T` leaves it `none`.
            // Reuses the same `parseProtocolRef` an `impl .. has P` uses (bare or
            // dot-qualified, with an optional generic-protocol `[int, ..]` arg list).
            const bound: Ast.Index = if (p.eat(.kw_has)) try p.parseProtocolRef() else Ast.none;
            const gp = try p.addNode(.{ .tag = .generic_param, .main_token = name_tok, .lhs = bound, .rhs = Ast.none });
            try out.append(p.gpa, gp);
            if (!p.eat(.comma)) {
                if (p.at(.r_bracket)) break;
            }
        } else if (generic_recovery.contains(p.peek().tag)) {
            break;
        } else {
            _ = try p.advanceWithError(.P0006, "expected a type parameter name or ']'");
        }
        std.debug.assert(p.index > entry or p.at(.r_bracket) or p.at(.eof));
    }
    try p.expect(.r_bracket, "expected ']' to close a generic parameter list");
}

/// `Base[Arg, ..]` — a type application. `base` is the already-parsed base/callee
/// node (an `identifier` or a `field_access` dot-chain). Parses the bracketed
/// type-argument list (recursive `parseType`, so `Box[Vec[int]]` nests) and wraps
/// it in a `type_app` node (`main_token` = `[`, `lhs` = base, `rhs` = a `Range` of
/// the type-arg nodes). Called only when the cursor is already on `[`. The base
/// and every arg node are created before the `type_app`, so children precede it.
fn parseTypeApp(p: *Parser, base: Ast.Index) Error!Ast.Index {
    const lbracket = p.index;
    p.bump(.l_bracket);
    var args: std.ArrayList(Ast.Index) = .empty;
    defer args.deinit(p.gpa);
    while (!p.at(.r_bracket) and !p.at(.eof)) {
        const entry = p.index;
        if (type_first.contains(p.peek().tag)) {
            try args.append(p.gpa, try p.parseType());
            if (!p.eat(.comma)) {
                if (p.at(.r_bracket)) break;
            }
        } else if (generic_recovery.contains(p.peek().tag)) {
            break;
        } else {
            _ = try p.advanceWithError(.P0006, "expected a type");
        }
        std.debug.assert(p.index > entry or p.at(.r_bracket) or p.at(.eof));
    }
    try p.expect(.r_bracket, "expected ']' to close a type application");
    const header = try p.addRange(args.items);
    return p.addNode(.{ .tag = .type_app, .main_token = lbracket, .lhs = base, .rhs = header });
}

/// Peek past a bracketed run `[ .. ]` (from the `[` at the cursor to its depth-matched
/// `]`) and report whether the token AFTER it opens a call `(`, struct literal `{`, or
/// member access `.` — the three positions a `type_app` is validly consumed in. Every
/// other follower marks the `[ .. ]` as a value INDEX. An unbalanced run reads as an
/// index (its `parseIndex` reports the missing `]`). In a `no_block` header (if/while/
/// match/for scrutinee) a following `{` is the body brace, not a struct literal, so it
/// must not pull the index into a `type_app`.
fn turbofishFollows(p: *const Parser) bool {
    var depth: usize = 0;
    var i: u32 = p.index;
    while (i < p.tokens.len) : (i += 1) {
        switch (p.tokens[i].tag) {
            .l_bracket => depth += 1,
            .r_bracket => {
                depth -= 1;
                if (depth == 0) {
                    const next: token.Tag = if (i + 1 < p.tokens.len) p.tokens[i + 1].tag else .eof;
                    return next == .l_paren or (next == .l_brace and !p.no_block) or next == .dot;
                }
            },
            .eof => return false,
            else => {},
        }
    }
    return false;
}

/// Parse a value index `recv[idx]`. The receiver and index nodes are created before
/// the `index` node, so children precede the parent.
fn parseIndex(p: *Parser, recv: Ast.Index) Error!Ast.Index {
    const lbracket = p.index;
    p.bump(.l_bracket);
    // The `[ ]` open a fresh expression context (re-allow blocks/literals in the index).
    var nb = NoBlockScope.enter(p, false);
    defer nb.end();
    const idx = try p.parseExpr(0);
    try p.expect(.r_bracket, "expected ']' to close an index");
    return p.addNode(.{ .tag = .index, .main_token = lbracket, .lhs = recv, .rhs = idx });
}

/// Parse a bare or dot-qualified name (`A` / `mod.A` / `a.b.C`) into an `identifier`
/// leaf (bare) or a left-nested `field_access` chain (qualified) — the shared shape a
/// qualified type / protocol / impl-receiver name uses. `first_err`/`chain_err` are the
/// diagnostics for a missing name at the head and after a `.`. Returns the built node
/// and the LAST segment's token (the impl receiver keys its `main_token` off it). Each
/// child leaf is created before its `field_access` parent, so children precede parents.
fn parseQualifiedName(p: *Parser, first_err: []const u8, chain_err: []const u8) Error!struct { node: Ast.Index, last_tok: u32 } {
    const first_tok = p.index;
    try p.expect(.identifier, first_err);
    var node = try p.addNode(.{ .tag = .identifier, .main_token = first_tok, .lhs = Ast.none, .rhs = Ast.none });
    var last_tok = first_tok;
    while (p.at(.dot)) {
        p.bump(.dot);
        const seg_tok = p.index;
        try p.expect(.identifier, chain_err);
        node = try p.addNode(.{ .tag = .field_access, .main_token = seg_tok, .lhs = node, .rhs = Ast.none });
        last_tok = seg_tok;
    }
    return .{ .node = node, .last_tok = last_tok };
}

/// A type reference is written as an identifier (e.g. `int`, `bool`, `str`), the
/// unit type `()`, or a module-qualified type `mod.Type`. A qualified type
/// reuses the `field_access` node: receiver = the module-name `identifier` leaf,
/// `main_token` = the type-name ident after `.`. The resolver disambiguates this
/// from value field access by its type position. `/` never appears in a type —
/// only `.` — so this stays unambiguous with the `import` path grammar.
fn parseType(p: *Parser) Error!Ast.Index {
    // Recursion-depth guard at the single type chokepoint: parseType<->parseTypeApp
    // mutually recurse on nested generic type-applications (`Box[Box[..[int]..]]`, also
    // reachable through parseProtocolRef via `[T has P[P[..]]]` bounds), a chain that
    // never funnels through the parseExpr/parseBlock guards. Past the cap emit one
    // backstop and return a bounded error_node, consuming to a type-list boundary so the
    // surrounding parseTypeApp loop terminates via its `]`/eof guard (and its
    // forward-progress assert holds) instead of overflowing the stack. Compiled in ALL modes.
    p.depth += 1;
    defer p.depth -= 1;
    if (p.depth > MAX_EXPR_DEPTH) {
        try p.backstop(p.peek(), .P0005, "type nested too deeply");
        const et = p.index;
        while (!p.at(.eof) and !p.at(.r_bracket) and !p.at(.r_paren) and !p.at(.r_brace) and !p.at(.comma) and !p.at(.newline)) p.advance();
        return p.addNode(.{ .tag = .error_node, .main_token = et, .lhs = Ast.none, .rhs = Ast.none });
    }
    if (p.at(.l_paren) and p.peek2().tag == .r_paren) {
        const at_tok = p.index;
        p.bump(.l_paren);
        p.bump(.r_paren);
        return p.addNode(.{ .tag = .literal_unit, .main_token = at_tok, .lhs = Ast.none, .rhs = Ast.none });
    }
    // A `.ident` chain qualifies the type by its owning module (`mod.Type`), nesting
    // left like value field access so a deeper `a.b.C` is supported structurally (the
    // resolver decides what is legal).
    var ty = (try p.parseQualifiedName("expected a type name", "expected a type name after '.'")).node;
    // A trailing `[..]` applies type arguments in TYPE position (`Box[int]`,
    // `mod.Box[int]`). Parses to a `type_app`; Typecheck rejects it (T0013).
    if (p.at(.l_bracket)) ty = try p.parseTypeApp(ty);
    return ty;
}

fn parseBlock(p: *Parser) Error!Ast.Index {
    // Nesting-depth guard on the block/statement recursion. Statement-keyword
    // constructs (`if`/`while`/`for`/`loop`) recurse fn->parseBlock->parseStmt->fn
    // WITHOUT funneling through parseExpr (their only parseExpr call is the
    // condition, which unwinds `p.depth` back down before the body is parsed), so
    // the parseExpr guard alone cannot see this recursion. Sharing `p.depth`/
    // `MAX_EXPR_DEPTH` here caps total nesting: past the cap emit ONE diagnostic and
    // return a bounded `error_node` (the caller's stmt/decl recovery loop resyncs
    // depth-aware over the unparsed nested region) instead of overflowing the stack.
    // Compiled in ALL modes — it defends a real SIGBUS.
    p.depth += 1;
    defer p.depth -= 1;
    if (p.depth > MAX_EXPR_DEPTH) {
        try p.backstop(p.peek(), .P0005, "block nested too deeply");
        return p.addNode(.{ .tag = .error_node, .main_token = p.index, .lhs = Ast.none, .rhs = Ast.none });
    }
    // A block body is a fresh expression context: re-allow `{`/`if` expressions
    // inside it even when reached from an `if`/`while` condition.
    var nb = NoBlockScope.enter(p, false);
    defer nb.end();
    const lbrace = p.index;
    try p.expect(.l_brace, "expected '{' to open a block");

    var stmts: std.ArrayList(Ast.Index) = .empty;
    defer stmts.deinit(p.gpa);
    // Resilient statement loop (Zig-std model): a broken statement resyncs to the
    // next statement boundary rather than unwinding the whole block. The `while`
    // exits at eof; the trailing `expect(.r_brace)` reports a missing `}` ONCE
    // (no mid-loop unwind of the enclosing fn). A decl keyword at loop top bails
    // the wrecked block to the decl loop so it cannot eat the next fn.
    while (!p.at(.eof)) {
        p.skipNewlines();
        if (p.at(.r_brace)) break;
        if (decl_anchors.contains(p.peek().tag)) break;
        const entry = p.index;
        const stmt = p.parseStmt() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.ParseError => {
                p.findNextStmt();
                std.debug.assert(p.index > entry or p.at(.r_brace) or decl_anchors.contains(p.peek().tag));
                continue;
            },
        };
        try stmts.append(p.gpa, stmt);
        // A missing terminator after a statement resyncs (the recovery set includes
        // `.newline`, so `skipNewlines` at the next loop top swallows the landing
        // token — no spurious ASI cascade).
        p.expectTerminator() catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.ParseError => p.findNextStmt(),
        };
    }
    try p.expect(.r_brace, "expected '}' to close block");

    const header = try p.addRange(stmts.items);
    return p.addNode(.{ .tag = .block, .main_token = lbrace, .lhs = header, .rhs = Ast.none });
}

/// A statement ends at a `.newline`, or implicitly before `}`/EOF (Go ASI).
fn expectTerminator(p: *Parser) Error!void {
    switch (p.peek().tag) {
        .newline => {
            // A cleanly crossed statement boundary is a resync point: clear the
            // cascade latch so the NEXT statement's first error reports.
            p.in_error = false;
            p.advance();
            p.skipNewlines();
        },
        .r_brace, .eof => p.in_error = false,
        else => return p.fail(p.peek(), .P0004, "expected a newline or '}' after statement"),
    }
}

fn parseStmt(p: *Parser) Error!Ast.Index {
    const tok = p.peek();
    switch (tok.tag) {
        .kw_if => return p.parseIf(),
        .kw_while => return p.parseWhile(),
        .kw_for => return p.parseFor(),
        .kw_break => {
            const break_tok = p.index;
            p.bump(.kw_break);
            // A `@name` label may immediately follow `break`; parse it BEFORE the
            // value-terminator decision (the documented ordering hazard). The label
            // is a TOKEN index that overloads the (node-typed) `rhs` slot.
            const label_tok = try p.parseOptLabel();
            const expr: Ast.Index = switch (p.peek().tag) {
                .newline, .r_brace, .eof => Ast.none,
                else => try p.parseExpr(0),
            };
            return p.addNode(.{ .tag = .break_stmt, .main_token = break_tok, .lhs = expr, .rhs = Ast.Index.from(label_tok.int()) });
        },
        .kw_continue => {
            const continue_tok = p.index;
            p.bump(.kw_continue);
            const label_tok = try p.parseOptLabel();
            return p.addNode(.{ .tag = .continue_stmt, .main_token = continue_tok, .lhs = Ast.none, .rhs = Ast.Index.from(label_tok.int()) });
        },
        // `@label <construct>` as a statement routes through parseExprStmt (like a
        // bare `loop`/`if`/block), so a trailing labeled loop/block is wrapped in an
        // expr_stmt and its value can satisfy a non-unit fn's trailing-expr rule.
        .kw_return => {
            const ret_tok = p.index;
            p.bump(.kw_return);
            const expr: Ast.Index = switch (p.peek().tag) {
                .newline, .r_brace, .eof => Ast.none,
                else => try p.parseExpr(0),
            };
            return p.addNode(.{ .tag = .return_stmt, .main_token = ret_tok, .lhs = expr, .rhs = Ast.none });
        },
        .identifier => switch (p.peek2().tag) {
            .colon_eq => {
                const name_tok = p.index;
                p.bump(.identifier);
                p.bump(.colon_eq);
                const init_expr = try p.parseExpr(0);
                return p.addNode(.{ .tag = .var_decl, .main_token = name_tok, .lhs = init_expr, .rhs = Ast.none });
            },
            // `x: T = e` — explicitly-typed local. The type ref lands in the
            // otherwise-unused var_decl.rhs slot; the checker prefers it over
            // inferring from the initializer (the resolver ignores rhs).
            .colon => {
                const name_tok = p.index;
                p.bump(.identifier);
                p.bump(.colon);
                const type_ref = try p.parseType();
                try p.expect(.eq, "expected '=' after type in typed declaration");
                const init_expr = try p.parseExpr(0);
                return p.addNode(.{ .tag = .var_decl, .main_token = name_tok, .lhs = init_expr, .rhs = type_ref });
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
                if (p.eat(.eq)) {
                    const value = try p.parseExpr(0);
                    return p.addNode(.{ .tag = .assign, .main_token = first, .lhs = place, .rhs = value });
                }
                const expr = try p.continueInfix(place, 0);
                return p.addNode(.{ .tag = .expr_stmt, .main_token = first, .lhs = expr, .rhs = Ast.none });
            },
            else => return p.parseExprStmt(),
        },
        // A `*`-rooted place: `*r = v`. Parse the deref place; if `=` follows it is a
        // store-through-box, otherwise it is an expression statement (continue the infix
        // climb). Mirrors the `.`-rooted place path above.
        .star => {
            const first = p.index;
            const place = try p.parsePostfix(try p.parsePrefix());
            if (p.eat(.eq)) {
                const value = try p.parseExpr(0);
                return p.addNode(.{ .tag = .assign, .main_token = first, .lhs = place, .rhs = value });
            }
            const expr = try p.continueInfix(place, 0);
            return p.addNode(.{ .tag = .expr_stmt, .main_token = first, .lhs = expr, .rhs = Ast.none });
        },
        else => return p.parseExprStmt(),
    }
}

/// `while cond { body }`. The condition is a full expression (no parens
/// required, Go-style); the body is a brace block.
fn parseWhile(p: *Parser) Error!Ast.Index {
    const while_tok = p.index;
    p.bump(.kw_while);
    // `defer`-pair the scope so an error unwind while parsing the condition still
    // restores `no_block`; a nested scope ends it before the body is parsed.
    const cond = cond: {
        var nb = NoBlockScope.enter(p, true);
        defer nb.end();
        break :cond try p.parseExpr(0);
    };
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
    p.bump(.kw_if);
    const cond = cond: {
        var nb = NoBlockScope.enter(p, true);
        defer nb.end();
        break :cond try p.parseExpr(0);
    };
    const then_block = try p.parseBlock();
    var else_node: Ast.Index = Ast.none;
    if (p.eat(.kw_else)) {
        else_node = if (p.at(.kw_if)) try p.parseIf() else try p.parseBlock();
    }
    const header = try p.addExtra(&.{ then_block.int(), else_node.int() });
    return p.addNode(.{ .tag = .if_stmt, .main_token = if_tok, .lhs = cond, .rhs = header });
}

/// If the cursor is at `@name`, consume both and return the identifier TOKEN
/// index; otherwise consume nothing and return `TokIndex.none`. Used for the
/// optional label on `break`/`continue` (a token-overloaded slot).
fn parseOptLabel(p: *Parser) Error!Ast.TokIndex {
    if (!p.at(.at)) return Ast.TokIndex.none;
    p.bump(.at);
    const name_tok = p.index;
    try p.expect(.identifier, "expected a label name after '@'");
    return Ast.TokIndex.from(name_tok);
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
        else => return p.fail(p.peek(), .P0007, "a label must prefix a loop, while, for, or block"),
    };
    return p.addNode(.{ .tag = .labeled, .main_token = name_tok, .lhs = inner, .rhs = Ast.none });
}

/// `loop { body }`. A value-yielding infinite loop expression.
fn parseLoop(p: *Parser) Error!Ast.Index {
    const loop_tok = p.index;
    p.bump(.kw_loop);
    const body = try p.parseBlock();
    return p.addNode(.{ .tag = .loop_expr, .main_token = loop_tok, .lhs = body, .rhs = Ast.none });
}

/// `for ident in lo..hi { body }`. Iterates the half-open integer range
/// `[lo, hi)` with `ident: int` bound per-iteration. A `()` statement.
fn parseFor(p: *Parser) Error!Ast.Index {
    p.bump(.kw_for);
    const ident_tok = p.index;
    try p.expect(.identifier, "expected a loop variable name");
    try p.expect(.kw_in, "expected 'in' after the loop variable");
    const range = range: {
        var nb = NoBlockScope.enter(p, true);
        defer nb.end();
        const lo = try p.parseExpr(0); // halts at `..` (no infix bp)
        try p.expect(.dotdot, "expected '..' in the for range");
        const hi = try p.parseExpr(0);
        break :range .{ lo, hi };
    };
    const lo = range[0];
    const hi = range[1];
    const body = try p.parseBlock(); // re-arms no_block internally
    const header = try p.addExtra(&.{ lo.int(), hi.int() }); // children before parent
    return p.addNode(.{ .tag = .for_stmt, .main_token = ident_tok, .lhs = body, .rhs = header });
}

fn parseExprStmt(p: *Parser) Error!Ast.Index {
    const first = p.index;
    const expr = try p.parseExpr(0);
    return p.addNode(.{ .tag = .expr_stmt, .main_token = first, .lhs = expr, .rhs = Ast.none });
}

/// Precedence-climbing core. `min_bp` is the minimum binding power that an infix
/// operator must exceed to bind here; recursing with the operator's own bp makes
/// operators left-associative. The call postfix is applied to every operand
/// before the infix loop, so it binds tighter than any infix operator.
fn parseExpr(p: *Parser, min_bp: u8) Error!Ast.Index {
    // Recursion-depth guard at the single mutual-recursion chokepoint (nested
    // parens, unary chains, infix RHS via `continueInfix`, and call args all
    // funnel through here). Past the cap, emit one diagnostic and return a bounded
    // `error_node` (consuming to a boundary so the surrounding list/infix loop
    // terminates via its anchor/eof guard) instead of overflowing the stack. This
    // guard is compiled in ALL modes — it defends a real crash.
    p.depth += 1;
    defer p.depth -= 1;
    if (p.depth > MAX_EXPR_DEPTH) {
        try p.backstop(p.peek(), .P0005, "expression nested too deeply");
        const et = p.index;
        while (!p.at(.eof) and !p.at(.r_paren) and !p.at(.r_brace) and !p.at(.comma) and !p.at(.newline)) p.advance();
        return p.addNode(.{ .tag = .error_node, .main_token = et, .lhs = Ast.none, .rhs = Ast.none });
    }
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
    const at_tok = p.index;
    switch (tok.tag) {
        // `&x` boxes; `*r` derefs. Both bind as prefix unaries (tighter than any infix),
        // so `*r + 1` is `(*r) + 1`. Infix `a & b`/`a * b` are unaffected — they are
        // reached only via `continueInfix`, never here.
        .minus, .bang, .tilde, .amp, .star => {
            p.advance();
            const operand = try p.parseExpr(prefix_bp);
            return p.addNode(.{ .tag = .unary, .main_token = at_tok, .lhs = operand, .rhs = Ast.none });
        },
        .number => return p.leaf(.literal_number, at_tok),
        .float => return p.leaf(.literal_float, at_tok),
        .string => return p.leaf(.literal_string, at_tok),
        .char_lit => return p.leaf(.literal_char, at_tok),
        .kw_true, .kw_false => return p.leaf(.literal_bool, at_tok),
        .identifier => return p.leaf(.identifier, at_tok),
        .l_paren => {
            p.bump(.l_paren);
            if (p.at(.r_paren)) { // the unit literal `()`
                p.bump(.r_paren);
                return p.addNode(.{ .tag = .literal_unit, .main_token = at_tok, .lhs = Ast.none, .rhs = Ast.none });
            }
            // A grouped sub-expression re-allows blocks (the escape hatch out of a
            // condition's `no_block`); `defer`-restore the flag so an error unwind
            // while parsing the inner expr still restores the prior value.
            const inner = inner: {
                var nb = NoBlockScope.enter(p, false);
                defer nb.end();
                break :inner try p.parseExpr(0);
            };
            try p.expect(.r_paren, "expected ')' to close group");
            return inner;
        },
        .l_brace => {
            if (p.no_block) return p.fail(tok, .P0002, "expected an expression");
            return p.parseBlock(); // a bare block as a value expression
        },
        .kw_if => {
            if (p.no_block) return p.fail(tok, .P0002, "expected an expression");
            return p.parseIf(); // an if as a value expression
        },
        .kw_loop => {
            if (p.no_block) return p.fail(tok, .P0002, "expected an expression");
            return p.parseLoop(); // a loop as a value expression
        },
        .at => {
            if (p.no_block) return p.fail(tok, .P0002, "expected an expression");
            return p.parseLabeled(); // a labeled loop/block as a value expression
        },
        // Inferred variant construction `.V` in a type-known position. At an
        // expression START (no receiver) a leading `.` is a variant; `parsePostfix`
        // then upgrades it to a tuple/struct form if `(`/`{` follows. (A postfix
        // `.field` is handled in parsePostfix, after an operand.)
        .dot => {
            p.bump(.dot);
            const name = p.index;
            try p.expect(.identifier, "expected a variant name after '.'");
            return p.addNode(.{ .tag = .enum_init_unit, .main_token = name, .lhs = Ast.none, .rhs = Ast.none });
        },
        .kw_match => {
            if (p.no_block) return p.fail(tok, .P0002, "expected an expression");
            return p.parseMatch(); // a match as a value expression
        },
        .kw_unsafe => {
            if (p.no_block) return p.fail(tok, .P0002, "expected an expression");
            p.bump(.kw_unsafe);
            const blk = try p.parseBlock();
            return p.addNode(.{ .tag = .unsafe_block, .main_token = at_tok, .lhs = blk, .rhs = Ast.none });
        },
        // A list literal in expression-START position. The empty `[]` form is typed
        // bidirectionally by a `Vec[T]` annotation; a non-empty `[e0, ..]` infers its
        // element type from `e0`. (A postfix `id[..]` type-app / value index is reached
        // through parsePostfix after an operand, never here.)
        .l_bracket => {
            p.bump(.l_bracket);
            if (p.at(.r_bracket)) {
                p.bump(.r_bracket);
                return p.addNode(.{ .tag = .empty_list, .main_token = at_tok, .lhs = Ast.none, .rhs = Ast.none });
            }
            var elems: std.ArrayList(Ast.Index) = .empty;
            defer elems.deinit(p.gpa);
            // The `[ ]` open a fresh expression context, so re-allow blocks/literals in
            // elements even inside an if/while condition (mirrors `parseCall`).
            var nb = NoBlockScope.enter(p, false);
            defer nb.end();
            while (!p.at(.r_bracket) and !p.at(.eof)) {
                const entry = p.index;
                if (expr_first.contains(p.peek().tag)) {
                    try elems.append(p.gpa, try p.parseExpr(0));
                    if (!p.eat(.comma)) {
                        if (p.at(.r_bracket)) break;
                    }
                } else if (tuple_recovery.contains(p.peek().tag)) {
                    break;
                } else {
                    _ = try p.advanceWithError(.P0002, "expected a list element");
                }
                std.debug.assert(p.index > entry or p.at(.r_bracket) or p.at(.eof));
            }
            try p.expect(.r_bracket, "expected ']' to close a list literal");
            const header = try p.addRange(elems.items);
            return p.addNode(.{ .tag = .list_literal, .main_token = at_tok, .lhs = Ast.none, .rhs = header });
        },
        // No valid expression start. When the offending token is a structural
        // CLOSER an open enclosing construct still needs (`)` of a call/group, `}`
        // of a block/struct-literal), DELETING it (advanceWithError) would break
        // that construct's closing `expect` and cascade — one missing operand
        // (`g(1 + )`, or a trailing `:=`/`+` before `}`) would spray a diagnostic
        // per unfinished ancestor. So report the missing expression and return an
        // `error_node` WITHOUT consuming: the enclosing arg/group/block loop then
        // sees its closer (its anchor branch breaks, its `expect` consumes it),
        // collapsing the cascade to one diagnostic. Forward progress is still
        // guaranteed — `continueInfix` already consumed the operator before
        // recursing, and at statement start `findNextStmt` advances past a
        // still-unconsumed closer. For any OTHER invalid start, keep the
        // single-token DELETION repair (consume one token to make progress).
        else => {
            if (p.at(.r_paren) or p.at(.r_brace)) {
                try p.warn(tok, .P0002, "expected an expression");
                return p.addNode(.{ .tag = .error_node, .main_token = at_tok, .lhs = Ast.none, .rhs = Ast.none });
            }
            return p.advanceWithError(.P0002, "expected an expression");
        },
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
                if (p.nodes.items[lhs.int()].tag == .enum_init_unit and p.nodes.items[lhs.int()].lhs == Ast.none) {
                    const rebuilt = try p.upgradeTupleInit(lhs, Ast.none);
                    lhs = rebuilt;
                } else {
                    lhs = try p.parseCall(lhs);
                }
            },
            // `.field` access. `..` is a separate token, so `0..5` is unaffected.
            .dot => lhs = try p.parseFieldAccess(lhs),
            // Postfix `?`: wrap `lhs` in a `try_expr` and continue the loop so
            // `f()?`, `o?.x`, `o??` compose. Desugared below the parser (lower/types).
            .question => {
                const q = p.index;
                p.bump(.question);
                lhs = try p.addNode(.{ .tag = .try_expr, .main_token = q, .lhs = lhs, .rhs = Ast.none });
            },
            // Explicit call type-args `id[int](..)`: wrap ONLY a name / qualified
            // `mod.fn` callee into a `type_app`; the loop then sees the following
            // `(` and builds a normal `call` whose callee is the `type_app`. This
            // is the RESERVED postfix-call position; any other `lhs` (e.g. a call
            // result) breaks WITHOUT consuming `[`, keeping a future value-index
            // `v[i]` free to adopt a distinct form.
            .l_bracket => {
                const ltag = p.nodes.items[lhs.int()].tag;
                if (ltag != .identifier and ltag != .field_access) break;
                // Disambiguate a turbofish head (`id[T](..)` / `Box[T]{..}` /
                // `Vec[T].new()`) from a value index `v[i]` by the token AFTER the
                // matching `]`: a call / struct-literal / member access keeps the
                // type-app; anything else is a subscript. (A value index whose result is
                // itself indexed / member-accessed — `xs[i].f`, `xs[i][j]` — stays a
                // type-app; bind first, then access.)
                if (p.turbofishFollows()) {
                    lhs = try p.parseTypeApp(lhs);
                } else {
                    lhs = try p.parseIndex(lhs);
                }
            },
            // `Name { ... }` literal / variant construction — only when blocks are
            // allowed and `lhs` is a bare name (struct), an inferred `.V`
            // (struct-variant), or a `field_access` (qualified `N.V`). The call/
            // group `( )` reset `no_block`, so `f(P{x:1})` works.
            .l_brace => {
                if (p.no_block) break;
                const ltag = p.nodes.items[lhs.int()].tag;
                switch (ltag) {
                    .identifier => lhs = try p.parseStructLiteral(lhs),
                    // Generic-struct construction `Box[int] { ... }`: the `[..]`
                    // was already wrapped into a `type_app` by the `.l_bracket` arm, so
                    // build a `struct_init` whose lhs is that `type_app` (no new
                    // Node.Tag ⇒ ParseHeader.version unchanged). The `type_app` + its
                    // args were created first, so children still precede the parent.
                    .type_app => lhs = try p.parseStructLiteral(lhs),
                    .enum_init_unit => if (p.nodes.items[lhs.int()].lhs == Ast.none) {
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
    p.bump(.l_paren);
    var args: std.ArrayList(Ast.Index) = .empty;
    defer args.deinit(p.gpa);
    var nb = NoBlockScope.enter(p, false);
    defer nb.end();
    while (!p.at(.r_paren) and !p.at(.eof)) {
        const entry = p.index;
        if (expr_first.contains(p.peek().tag)) {
            try args.append(p.gpa, try p.parseExpr(0));
            if (!p.eat(.comma)) {
                if (p.at(.r_paren)) break;
            }
        } else if (tuple_recovery.contains(p.peek().tag)) {
            break;
        } else {
            _ = try p.advanceWithError(.P0002, "expected an argument");
        }
        std.debug.assert(p.index > entry or p.at(.r_paren) or p.at(.eof));
    }
    try p.expect(.r_paren, "expected ')' to close a variant construction");
    const header = try p.addRange(args.items);
    const vtok = p.nodes.items[node.int()].main_token;
    p.nodes.items[node.int()] = .{ .tag = .enum_init_tuple, .main_token = vtok, .lhs = type_name, .rhs = header };
    return node;
}

/// Parse `{ field: value, ... }` onto a variant construction, producing an
/// `enum_init_struct`. `node` is the `enum_init_unit` (inferred) or the
/// `field_access` (qualified `N.V`) to rebuild. `type_name` is `none` (inferred)
/// or the type-name node (for qualified, the field_access's receiver).
fn upgradeStructInit(p: *Parser, node: Ast.Index, qualified: Ast.Index) Error!Ast.Index {
    // For a qualified `N.V`, the node is the field_access: variant = its field
    // token, type-name = its receiver.
    const vtok = p.nodes.items[node.int()].main_token;
    const type_name: Ast.Index = if (qualified == Ast.none) Ast.none else p.nodes.items[node.int()].lhs;
    p.bump(.l_brace);
    var nb = NoBlockScope.enter(p, false);
    defer nb.end();
    var inits: std.ArrayList(Ast.Index) = .empty;
    defer inits.deinit(p.gpa);
    while (!p.at(.eof)) {
        p.skipNewlines();
        if (p.at(.r_brace)) break;
        if (decl_anchors.contains(p.peek().tag)) break;
        const entry = p.index;
        if (!p.at(.identifier)) {
            _ = try p.advanceWithError(.P0006, "expected a field name or '}'");
            std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
            continue;
        }
        const field_tok = p.index;
        p.bump(.identifier);
        var value: Ast.Index = undefined;
        if (p.eat(.colon)) {
            value = try p.parseExpr(0);
        } else {
            value = try p.addNode(.{ .tag = .identifier, .main_token = field_tok, .lhs = Ast.none, .rhs = Ast.none });
        }
        const fi = try p.addNode(.{ .tag = .field_init, .main_token = field_tok, .lhs = value, .rhs = Ast.none });
        try inits.append(p.gpa, fi);
        _ = p.eat(.comma);
        std.debug.assert(p.index > entry or p.at(.r_brace) or p.at(.eof));
    }
    try p.expect(.r_brace, "expected '}' to close a variant construction");
    const header = try p.addRange(inits.items);
    p.nodes.items[node.int()] = .{ .tag = .enum_init_struct, .main_token = vtok, .lhs = type_name, .rhs = header };
    return node;
}

fn parseCall(p: *Parser, callee: Ast.Index) Error!Ast.Index {
    const lparen = p.index;
    p.bump(.l_paren);
    var args: std.ArrayList(Ast.Index) = .empty;
    defer args.deinit(p.gpa);
    // The call's `( )` open a fresh expression context, so re-allow blocks/if-exprs
    // in arguments even inside an if/while condition (`no_block`); restore after.
    var nb = NoBlockScope.enter(p, false);
    defer nb.end();
    while (!p.at(.r_paren) and !p.at(.eof)) {
        const entry = p.index;
        if (expr_first.contains(p.peek().tag)) {
            try args.append(p.gpa, try p.parseExpr(0));
            if (!p.eat(.comma)) {
                if (p.at(.r_paren)) break;
            }
        } else if (tuple_recovery.contains(p.peek().tag)) {
            break;
        } else {
            _ = try p.advanceWithError(.P0002, "expected an argument");
        }
        std.debug.assert(p.index > entry or p.at(.r_paren) or p.at(.eof));
    }
    try p.expect(.r_paren, "expected ')' to close call");
    const header = try p.addRange(args.items);
    return p.addNode(.{ .tag = .call, .main_token = lparen, .lhs = callee, .rhs = header });
}

/// Infix binding power indexed by `token.Tag` ordinal; `-1` marks a non-infix
/// tag. Precedence is data the compiler validates rather than a hand-written
/// switch: `directEnumArrayDefault` with `max_unused_slots = 0` proves `Tag`
/// stays densely numbered, and a mistyped tag field name is a `@compileError`.
/// Higher binds tighter.
const infix_bp_table = std.enums.directEnumArrayDefault(token.Tag, i16, -1, 0, .{
    .pipe_pipe = 1,
    .amp_amp = 2,
    .pipe = 3, // bitwise OR
    .caret = 4, // bitwise XOR
    .amp = 5, // bitwise AND
    .eq_eq = 6,
    .bang_eq = 6,
    .lt = 7,
    .lt_eq = 7,
    .gt = 7,
    .gt_eq = 7,
    .lt_lt = 8, // shifts
    .gt_gt = 8,
    .plus = 9,
    .minus = 9,
    .star = 10,
    .slash = 10,
    .percent = 10,
    // Dotted float operators share the binding power of their integer twins.
    .lt_dot = 7,
    .gt_dot = 7,
    .le_dot = 7,
    .ge_dot = 7,
    .plus_dot = 9,
    .minus_dot = 9,
    .star_dot = 10,
    .slash_dot = 10,
});

/// Infix binding power, or null if the tag is not an infix operator. Higher
/// binds tighter.
fn infixBp(tag: token.Tag) ?u8 {
    const v = infix_bp_table[@intFromEnum(tag)];
    return if (v < 0) null else @intCast(v);
}

/// Comptime guard: every table operator must have a positive binding power (so
/// the `-1` sentinel unambiguously means "not infix"), and the operator set must
/// match exactly the tags the precedence-climbing loop recognizes. Returns an
/// error string on violation, else null.
fn checkInfixTable() ?[]const u8 {
    const infix_ops = [_]token.Tag{
        .pipe_pipe, .amp_amp,
        .pipe,      .caret,     .amp,
        .eq_eq,     .bang_eq,
        .lt,        .lt_eq,     .gt,   .gt_eq,
        .lt_lt,     .gt_gt,
        .plus,      .minus,
        .star,      .slash,     .percent,
        .lt_dot,    .gt_dot,    .le_dot, .ge_dot,
        .plus_dot,  .minus_dot,
        .star_dot,  .slash_dot,
    };
    // Every listed operator has a positive bp.
    for (infix_ops) |op| {
        if (infix_bp_table[@intFromEnum(op)] <= 0) {
            return "infix operator missing a positive binding power in infix_bp_table";
        }
    }
    // No tag outside the list carries a non-sentinel bp (the two sets match).
    for (@typeInfo(token.Tag).@"enum".fields) |f| {
        const tag: token.Tag = @enumFromInt(f.value);
        if (infix_bp_table[f.value] < 0) continue;
        var listed = false;
        for (infix_ops) |op| {
            if (op == tag) listed = true;
        }
        if (!listed) return "infix_bp_table entry not recognized by the infix loop";
    }
    return null;
}

comptime {
    if (checkInfixTable()) |m| @compileError(m);
}

/// Prefix operators bind tighter than any infix operator.
const prefix_bp: u8 = 11;

fn leaf(p: *Parser, tag: Node.Tag, tok_index: u32) Error!Ast.Index {
    p.advance();
    return p.addNode(.{ .tag = tag, .main_token = tok_index, .lhs = Ast.none, .rhs = Ast.none });
}

// The node/extra builders only ever fail with `OutOfMemory` (they never call
// `fail`), so they narrow their error set — which lets `parseProgram`'s tail
// (which runs AFTER the `ParseError`-catching decl loop) stay `error{OutOfMemory}`.
fn addNode(p: *Parser, node: Node) error{OutOfMemory}!Ast.Index {
    const idx = Ast.Index.from(@intCast(p.nodes.items.len));
    // Zero the extern-struct padding before the node enters the cached blob
    // (see `token.zeroPad`).
    try p.nodes.append(p.gpa, token.zeroPad(Node, node));
    return idx;
}

/// Append a run of node indices then a two-cell `{start, len}` header; return
/// the header cell as an `Ast.Index` (what the parent `Node` stores in an lhs/rhs
/// slot). The `Index` run is written into the `[]u32` `extra` verbatim — `Index`
/// is `enum(u32)`, so the bytes are identical.
fn addRange(p: *Parser, items: []const Ast.Index) error{OutOfMemory}!Ast.Index {
    const start: u32 = @intCast(p.extra.items.len);
    const cells: []const u32 = @ptrCast(items);
    try p.extra.appendSlice(p.gpa, cells);
    const header: u32 = @intCast(p.extra.items.len);
    try p.extra.append(p.gpa, start);
    try p.extra.append(p.gpa, @intCast(items.len));
    return Ast.Index.from(header);
}

/// A `Range` header for a generic-param list, or `Ast.none` when empty — so a
/// non-generic decl leaves its `rhs` slot `none` and stays byte-identical. Shared
/// by the struct/enum/protocol decls.
fn optRange(p: *Parser, items: []const Ast.Index) error{OutOfMemory}!Ast.Index {
    return if (items.len == 0) Ast.none else p.addRange(items);
}

/// Like `addRange` but for a run of TOKEN indices (an `import_decl`'s path
/// segments, which name tokens rather than child nodes). The header cell still
/// lands in an `Index`-typed slot, so it is returned as an `Ast.Index`.
fn addTokRange(p: *Parser, items: []const Ast.TokIndex) error{OutOfMemory}!Ast.Index {
    const start: u32 = @intCast(p.extra.items.len);
    const cells: []const u32 = @ptrCast(items);
    try p.extra.appendSlice(p.gpa, cells);
    const header: u32 = @intCast(p.extra.items.len);
    try p.extra.append(p.gpa, start);
    try p.extra.append(p.gpa, @intCast(items.len));
    return Ast.Index.from(header);
}

/// Append raw `u32` cells; return the index of the first as an `Ast.Index` (the
/// header always lands in an lhs/rhs slot). Cells are raw because a `FnProto`
/// header mixes a node index (`ret_type`) with `extra` offsets
/// (`params_start`/`params_len`); node-index cells are converted with `.int()`
/// at the call site.
fn addExtra(p: *Parser, vals: []const u32) error{OutOfMemory}!Ast.Index {
    const start: u32 = @intCast(p.extra.items.len);
    try p.extra.appendSlice(p.gpa, vals);
    return Ast.Index.from(start);
}

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

/// Is the cursor on `tag`? A predicate that never consumes — the readable form
/// of `p.peek().tag == tag`.
fn at(p: *const Parser, tag: token.Tag) bool {
    return p.peek().tag == tag;
}

/// Consume the current token iff it matches `tag`; return whether it did. For
/// OPTIONAL tokens (a trailing comma, an `as` alias, a leading `pub`).
fn eat(p: *Parser, tag: token.Tag) bool {
    if (!p.at(tag)) return false;
    p.advance();
    return true;
}

/// Consume a token the caller has ALREADY proven present (encodes the old
/// breadcrumb comment as an assert).
fn bump(p: *Parser, comptime tag: token.Tag) void {
    std.debug.assert(p.at(tag));
    p.advance();
}

/// Skip a run of statement terminators (Go-style newlines that are insignificant
/// inside `{}` / at the top level).
fn skipNewlines(p: *Parser) void {
    while (p.at(.newline)) p.advance();
}

/// Expect `tag`. On a match, consume it. On a MISMATCH (single-token
/// INSERTION repair): REPORT the diagnostic but do NOT consume — the caller
/// substitutes `Ast.none` for the missing child and keeps going. The propagated
/// `error.ParseError` unwinds the current producer to `parseProgram`, which
/// assembles the partial tree.
fn expect(p: *Parser, tag: token.Tag, message: []const u8) Error!void {
    const tok = p.peek();
    // A missing expected token is structurally one category (expected-token), so
    // expect() hardcodes P0001 — leaving all 49 call sites unchanged.
    if (tok.tag != tag) return p.fail(tok, .P0001, message);
    p.advance();
}

/// Append a diagnostic WITHOUT unwinding. Marks the parse as tainted. Used by the
/// repair primitives that recover in place; `fail` is `warn` + unwind. The `code`
/// is the diagnostic's registry category (a P#### parse code) and rides the same
/// cascade-latched append as the message.
fn warn(p: *Parser, tok: Token, code: Code, message: []const u8) error{OutOfMemory}!void {
    // Cascade suppression: once a diagnostic has fired for the current unresynced
    // region, swallow the derived ones (arm the latch on the FIRST). The caller
    // still gets its `error_node`/unwind — only the duplicate append is dropped.
    if (p.in_error) return;
    p.in_error = true;
    try p.diags.append(p.gpa, .{ .byte_offset = tok.start, .message = message, .code = code });
}

/// Emit a depth-limit BACKSTOP diagnostic. Unlike `warn` this bypasses the
/// cascade latch — the recursion-depth caps are the anti-SIGBUS crash backstop,
/// not a syntax cascade, so they must report even when a prior diagnostic (e.g. an
/// over-deep condition's own cap) already armed suppression. It does NOT arm the
/// latch either: the ordinary unwind noise (unclosed `}`) is still governed by the
/// normal `warn` latch, so this collapses to one backstop message plus one closer.
fn backstop(p: *Parser, tok: Token, code: Code, message: []const u8) error{OutOfMemory}!void {
    try p.diags.append(p.gpa, .{ .byte_offset = tok.start, .message = message, .code = code });
}

/// Report a diagnostic and unwind the current producer via `error.ParseError`.
/// The top-level loop catches this and finishes the partial tree. `warn` then
/// unwind — kept as one call so every `return p.fail(...)` site stays terse.
fn fail(p: *Parser, tok: Token, code: Code, message: []const u8) Error {
    // Route through `warn` so the cascade latch governs the append (a suppressed
    // append is still a `fail` — the producer unwinds either way). The `code`
    // rides the latch with the message.
    p.warn(tok, code, message) catch return error.OutOfMemory;
    return error.ParseError;
}

/// Single-token DELETION repair: report `message`, consume EXACTLY ONE token
/// (guaranteed forward progress, clamped at `.eof`), and return an `error_node`
/// leaf wrapping the offending token. Fills an operand/child slot in place so the
/// producer can keep building instead of unwinding.
fn advanceWithError(p: *Parser, code: Code, message: []const u8) Error!Ast.Index {
    const tok = p.peek();
    try p.warn(tok, code, message);
    const at_tok = p.index;
    p.advance(); // guaranteed progress (clamped at eof by `advance`)
    return p.addNode(.{ .tag = .error_node, .main_token = at_tok, .lhs = Ast.none, .rhs = Ast.none });
}

// A single sweep, run at the end of `parse()` under `std.debug.runtime_safety`,
// that pins three structural properties of every parse — valid OR recovered — so
// the whole test corpus doubles as an invariant check at zero release cost:
//
//   (a) SPAN TOTALITY — token spans tile `[0, source.len)` monotonically with no
//       gaps/overlaps, each span well-formed, the terminating `.eof` exactly at
//       `source.len`. This is the same contract the lexer asserts inline as it
//       emits tokens (see `lex.zig`); re-checking it on the FULL token list the
//       parser consumed lifts that guarantee to the parse boundary — a caret the
//       parser attaches to any token span can never point at a double-counted or
//       out-of-range byte.
//
//   (b) FORWARD PROGRESS — the recovery loops cannot spin. This is already pinned
//       by the fuel/advance guards in every resync scanner (`findNextDecl`,
//       `findNextStmt`, `resyncTo`) and every bounded list loop's
//       `p.index > entry or at-closer` assert; a passing parse (no hang) IS that
//       invariant holding. Nothing to re-check here — noted for completeness.
//
//   (c) STRUCTURAL PAIRING — the bracket delimiters (`(`/`)`, `{`/`}`, and the
//       generics `[`/`]`) are BALANCED for a clean parse. On a RECOVERED
//       parse (>=1 diagnostic, i.e. an `error_node` in the tree) imbalance is
//       allowed — recovery captured the syntax error. So the invariant is
//       "balanced OR the parse produced error nodes", which must NOT false-trip on
//       the adversarial-recovery corpus (e.g. `fn f( ) ) ) {`).

/// The post-parse invariant sweep (see the section header). Asserts (panics) on a
/// violation; only compiled where `std.debug.runtime_safety` is true.
fn checkInvariants(tree: Ast.Tree, tokens: []const Token, source: []const u8, diags: []const Diagnostic) void {
    checkSpanTotality(tokens, source);
    // (b) forward progress is enforced by the existing fuel guards; see the header.
    checkBracketPairing(tree, tokens, diags);
}

/// (a) SPAN TOTALITY. Re-verifies the lexer's tiling contract over the full token
/// list the parser consumed: spans are monotonic with no overlap, each well-formed,
/// and the trailing `.eof` sits exactly at `source.len`.
fn checkSpanTotality(tokens: []const Token, source: []const u8) void {
    var prev_end: u32 = 0;
    for (tokens) |tok| {
        std.debug.assert(tok.start >= prev_end); // monotonic, no overlap (gaps are trivia)
        std.debug.assert(tok.end >= tok.start); // well-formed span
        prev_end = tok.end;
    }
    // A tokenized source always ends in exactly one `.eof` sitting at `source.len`.
    std.debug.assert(tokens.len > 0);
    const last = tokens[tokens.len - 1];
    std.debug.assert(last.tag == .eof);
    std.debug.assert(last.end == source.len);
}

/// (c) STRUCTURAL PAIRING. Walks the token stream counting `(`/`)`, `{`/`}`, and
/// the generics `[`/`]` opens vs closes. For a CLEAN parse (no diagnostics, so no
/// `error_node`) all three pairs must be balanced and never go negative. On a
/// RECOVERED parse imbalance is tolerated. `error_node`-presence and `diags`
/// non-emptiness are equivalent recovery signals — either one licenses imbalance.
fn checkBracketPairing(tree: Ast.Tree, tokens: []const Token, diags: []const Diagnostic) void {
    const recovered = diags.len != 0 or hasErrorNode(tree);
    if (recovered) return; // recovery captured any imbalance; nothing to assert.
    var parens: i32 = 0;
    var braces: i32 = 0;
    var brackets: i32 = 0;
    for (tokens) |tok| switch (tok.tag) {
        .l_paren => parens += 1,
        .r_paren => parens -= 1,
        .l_brace => braces += 1,
        .r_brace => braces -= 1,
        .l_bracket => brackets += 1,
        .r_bracket => brackets -= 1,
        else => {},
    };
    // A clean parse is balanced AND never dipped below zero (a well-nested stream
    // can't end at 0 with a negative excursion, but the final == 0 check pins it).
    std.debug.assert(parens == 0);
    std.debug.assert(braces == 0);
    std.debug.assert(brackets == 0);
}

/// True if the tree contains at least one `.error_node` — the recovery marker that
/// licenses bracket imbalance in `checkBracketPairing`.
fn hasErrorNode(tree: Ast.Tree) bool {
    for (tree.nodes) |n| if (n.tag == .error_node) return true;
    return false;
}

const testing = std.testing;
const Lexer = @import("lex.zig");

/// Test-only: parse a single bare expression into a Tree,
/// so the expression-core tests below keep asserting on raw expressions without
/// the program/fn scaffolding. Unlike `parse()`, this helper still returns `null`
/// on a parse error (the expression-core tests want the terse "did it parse?"
/// shape); the first diagnostic (if any) is copied out through `diag`.
fn parseExprOnly(gpa: std.mem.Allocator, tokens: []const Token, source: []const u8, diag: *?Diagnostic) error{OutOfMemory}!?Ast.Tree {
    var p: Parser = .{
        .gpa = gpa,
        .tokens = tokens,
        .src = source,
        .index = 0,
        .nodes = .empty,
        .extra = .empty,
        .pub_decls = .empty,
        .diags = .empty,
    };
    defer p.diags.deinit(p.gpa);
    const run = struct {
        fn go(pp: *Parser) Error!Ast.Tree {
            _ = try pp.parseExpr(0);
            pp.skipNewlines();
            try pp.expect(.eof, "expected end of input");
            return Ast.Tree{
                .nodes = try pp.nodes.toOwnedSlice(pp.gpa),
                .extra = try pp.extra.toOwnedSlice(pp.gpa),
            };
        }
    }.go;
    const tree = run(&p) catch |err| switch (err) {
        error.OutOfMemory => {
            p.deinit();
            return error.OutOfMemory;
        },
        error.ParseError => {
            if (p.diags.items.len > 0) diag.* = p.diags.items[0];
            p.deinit();
            return null;
        },
    };
    // A tainted expression parse (a repair primitive fired without unwinding, e.g.
    // an `error_node` operand) still surfaces its first diagnostic.
    if (p.diags.items.len > 0) diag.* = p.diags.items[0];
    return tree;
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

/// Parse a full program and assert its rendered S-expression. A valid program
/// must parse cleanly (no diagnostics); `expectTree` asserts that.
fn expectProgram(source: []const u8, want: []const u8) !void {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);

    const tree = try expectTree(gpa, tokens, source);
    defer freeTree(gpa, tree);

    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try Ast.render(&w, tree, tokens, source);
    try testing.expectEqualStrings(want, w.buffered());
}

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

test "&x/*r parse as prefix unaries binding tighter than infix" {
    try expectSexpr("&x", "(& x)");
    try expectSexpr("*r", "(* r)");
    // Prefix binds tighter than infix: `*r + 1` is `(*r) + 1`, not `*(r + 1)`.
    try expectSexpr("*r + 1", "(+ (* r) 1)");
    // Infix `a * b` is UNAFFECTED (reached via the infix climb, never parsePrefix).
    try expectSexpr("a * b", "(* a b)");
}

test "*r = v parses as an assign whose target is a deref-unary place" {
    try expectProgram(
        "fn f() {\n *r = 1\n}\n",
        "(program (fn f () _ (block (= (* r) 1))))",
    );
}

test "comparison and equality precedence" {
    try expectSexpr("1 + 2 == 3 < 4", "(== (+ 1 2) (< 3 4))");
}

test "trailing newline terminator is allowed" {
    try expectSexpr("a * b\n", "(* a b)");
}

test "p.0 parses to tuple_field, p.x to field_access, p.0.1 chains" {
    const gpa = testing.allocator;
    // `p.0` is tuple_field; `p.x` is field_access. Assert the node TAGS (the renderer
    // prints both with the same `(. recv tok)` shape), so the KIND distinction, not the
    // text, is the load-bearing check.
    const single = [_]struct { src: []const u8, tag: Ast.Node.Tag }{
        .{ .src = "p.0", .tag = .tuple_field },
        .{ .src = "p.x", .tag = .field_access },
    };
    for (single) |c| {
        const tokens = try Lexer.tokenize(gpa, c.src);
        defer gpa.free(tokens);
        var diag: ?Diagnostic = null;
        const tree = (try parseExprOnly(gpa, tokens, c.src, &diag)) orelse return error.UnexpectedParseFailure;
        defer freeTree(gpa, tree);
        try testing.expectEqual(c.tag, tree.nodes[Ast.root(tree.nodes).int()].tag);
    }
    // `p.0.1` nests as a tuple_field whose lhs is a tuple_field (proves `0.1` did NOT
    // lex as one float — the middle `.` splits it into two dotted accesses).
    const tokens = try Lexer.tokenize(gpa, "p.0.1");
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const tree = (try parseExprOnly(gpa, tokens, "p.0.1", &diag)) orelse return error.UnexpectedParseFailure;
    defer freeTree(gpa, tree);
    const outer = tree.nodes[Ast.root(tree.nodes).int()];
    try testing.expectEqual(Ast.Node.Tag.tuple_field, outer.tag);
    try testing.expectEqual(Ast.Node.Tag.tuple_field, tree.nodes[outer.lhs.int()].tag);
}

test "parse error reports an offset and leaves a diagnostic" {
    // The missing operand after `+` no longer bails — `parsePrefix`'s else-arm
    // repairs it with a single-token DELETION (an `error_node` over the offending
    // token, here EOF), so `parseExprOnly` returns a (tainted) tree and surfaces
    // the diagnostic through `diag`.
    const gpa = testing.allocator;
    const source = "1 +";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var diag: ?Diagnostic = null;
    const result = try parseExprOnly(gpa, tokens, source, &diag);
    if (result) |t| freeTree(gpa, t);
    try testing.expect(result != null);
    try testing.expect(diag != null);
    try testing.expectEqualStrings("expected an expression", diag.?.message);
    // The diagnostic points at the offending token (the EOF after `1 +`).
    try testing.expectEqual(tokens[tokens.len - 1].start, diag.?.byte_offset);
}

/// Run the full `parse()` and return the `Result` (tree + owned diags). The caller
/// frees both. A test-only convenience for the fault-tolerance tests.
fn parseResult(gpa: std.mem.Allocator, source: []const u8) !Result {
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    return parse(gpa, tokens, source);
}

test "a syntax error still returns a tree covering all tokens to EOF" {
    // A bad expression start (`*` after `return`) triggers the single-token
    // DELETION repair, but parsing continues to the end of the file: the program
    // node still covers the whole fn, and the tree is non-empty + rooted at program.
    const gpa = testing.allocator;
    const source = "fn f() -> int {\n return *\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    // ALWAYS a tree: it is non-empty and its root is a `.program` node.
    try testing.expect(res.tree.nodes.len > 0);
    try testing.expectEqual(Node.Tag.program, res.tree.nodes[Ast.root(res.tree.nodes).int()].tag);
    // The program's decl range covers the (single) fn decl — the parse reached EOF
    // and closed the top-level node rather than truncating at the error.
    const prog = res.tree.nodes[Ast.root(res.tree.nodes).int()];
    try testing.expectEqual(@as(usize, 1), Ast.rangeSlice(res.tree, prog.lhs.int()).len);
}

test "the offending region is an error_node" {
    // The invalid `*` operand is repaired into an `error_node` leaf: the poison
    // node exists in the tree and renders as `(error)`.
    const gpa = testing.allocator;
    const source = "fn f() -> int {\n return *\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    var found_error_node = false;
    for (res.tree.nodes) |n| {
        if (n.tag == .error_node) found_error_node = true;
    }
    try testing.expect(found_error_node);

    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var buf: [256]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try Ast.render(&w, res.tree, tokens, source);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "(error)") != null);
}

test "a syntax error is reported (>=1 diagnostic) and the parse is tainted" {
    const gpa = testing.allocator;
    // `+` is binary-only (not a prefix operator), so it is invalid at expression start.
    const source = "fn f() -> int {\n return +\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expect(res.diags.len >= 1);
    try testing.expectEqualStrings("expected an expression", res.diags[0].message);
    // The diagnostic points at the offending `+` token.
    const bad_off = std.mem.indexOfScalar(u8, source, '+').?;
    try testing.expectEqual(@as(u32, @intCast(bad_off)), res.diags[0].byte_offset);
}

test "single-token INSERTION — a missing expected token reports and recovers" {
    // A missing `)` in the param list: the 3-way param loop sees `->` (a
    // param_recovery anchor) and breaks; the trailing `expect(.r_paren)` reports
    // the missing `)` and unwinds the decl, which the resilient decl loop recovers
    // from — so the parse still reaches EOF and yields a program tree.
    const gpa = testing.allocator;
    const source = "fn f( -> int { return 0 }\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expect(res.tree.nodes.len > 0);
    // Recovery reached EOF and closed the top-level node (a `.program` root).
    try testing.expectEqual(Node.Tag.program, res.tree.nodes[Ast.root(res.tree.nodes).int()].tag);
    try testing.expect(res.diags.len >= 1);
}

test "a clean program is untainted (no diagnostics, byte-identical tree)" {
    // The always-a-tree change must not perturb a VALID parse: zero diagnostics and
    // the same rendered tree as before.
    const gpa = testing.allocator;
    const source = "fn add(a: int, b: int) -> int {\n return a + b\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expectEqual(@as(usize, 0), res.diags.len);
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var buf: [256]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try Ast.render(&w, res.tree, tokens, source);
    try testing.expectEqualStrings(
        "(program (fn add ((param a int) (param b int)) int (block (return (+ a b)))))",
        w.buffered(),
    );
}

test "fn with params, return type, binary body" {
    try expectProgram(
        "fn add(a: int, b: int) -> int {\n return a + b\n}\n",
        "(program (fn add ((param a int) (param b int)) int (block (return (+ a b)))))",
    );
}

test "extern fn parses bodyless to extern_fn_decl" {
    try expectProgram(
        "extern fn labs(x: int) -> int\n",
        "(program (extern-fn labs ((param x int)) int))",
    );
}

test "pub extern fn parses and records the export" {
    try expectProgram(
        "pub extern fn labs(x: int) -> int\n",
        "(program (pub (extern-fn labs ((param x int)) int)))",
    );
}

test "unsafe block parses as a value expression wrapping its block" {
    try expectProgram(
        "fn main() -> int { return unsafe { 1 } }\n",
        "(program (fn main () int (block (return (unsafe (block 1))))))",
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

test "non-empty list literal parses to a list_literal of its elements" {
    try expectProgram(
        "fn main() -> int { xs := [1, 2, 3]\n return 0 }\n",
        "(program (fn main () int (block (:= xs (list 1 2 3)) (return 0))))",
    );
}

test "empty list literal still parses to empty_list" {
    try expectProgram(
        "fn main() -> int { xs: Vec[int] = []\n return 0 }\n",
        "(program (fn main () int (block (:= xs []) (return 0))))",
    );
}

test "value index parses to an index node" {
    try expectProgram(
        "fn main() -> int { return xs[0] }\n",
        "(program (fn main () int (block (return (index xs 0)))))",
    );
}

test "index on a field access parses to an index over the field" {
    try expectProgram(
        "fn main() -> int { return p.items[1] }\n",
        "(program (fn main () int (block (return (index (. p items) 1)))))",
    );
}

test "turbofish call `id[int](x)` still parses as a type_app callee (not an index)" {
    try expectProgram(
        "fn main() -> int { return f[int](3) }\n",
        "(program (fn main () int (block (return (call (tyapp f int) 3)))))",
    );
}

test "associated-fn `Vec[int].new()` still parses as a type_app receiver (not an index)" {
    try expectProgram(
        "fn main() -> int { return Vec[int].new() }\n",
        "(program (fn main () int (block (return (call (. (tyapp Vec int) new))))))",
    );
}

test "generic struct literal `Box[int]{..}` still parses as a type_app head (not an index)" {
    try expectProgram(
        "fn main() -> int { b := Box[int] { v: 1 }\n return 0 }\n",
        "(program (fn main () int (block (:= b (new (tyapp Box int) (field v 1))) (return 0))))",
    );
}

test "root is program and children precede parents" {
    const gpa = testing.allocator;
    const source = "fn add(a: int, b: int) -> int {\n return a + b\n}\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, source);
    defer freeTree(gpa, tree);

    try testing.expectEqual(Node.Tag.program, tree.nodes[Ast.root(tree.nodes).int()].tag);
    // Topological: every child node index is strictly less than its parent's.
    for (tree.nodes, 0..) |n, i| {
        const self: u32 = @intCast(i);
        switch (n.tag) {
            .unary => try testing.expect(n.lhs.int() < self),
            .binary, .assign => {
                try testing.expect(n.lhs.int() < self);
                try testing.expect(n.rhs.int() < self);
            },
            .var_decl, .expr_stmt => try testing.expect(n.lhs.int() < self),
            // Postfix `?`: the operand is `lhs`, created before the try_expr node.
            .try_expr => try testing.expect(n.lhs.int() < self),
            .return_stmt => if (n.lhs != Ast.none) try testing.expect(n.lhs.int() < self),
            .param => try testing.expect(n.lhs.int() < self),
            .call => {
                try testing.expect(n.lhs.int() < self);
                for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            .block => for (Ast.rangeSlice(tree, n.lhs.int())) |c| try testing.expect(c.int() < self),
            .program => for (Ast.rangeSlice(tree, n.lhs.int())) |c| try testing.expect(c.int() < self),
            .fn_decl => {
                const proto = Ast.protoAt(tree, n.lhs.int());
                if (proto.ret_type != Ast.none) try testing.expect(proto.ret_type.int() < self);
                for (proto.params) |c| try testing.expect(c.int() < self);
                for (proto.generic_params) |c| try testing.expect(c.int() < self);
                try testing.expect(n.rhs.int() < self);
            },
            .while_stmt => {
                try testing.expect(n.lhs.int() < self);
                try testing.expect(n.rhs.int() < self);
            },
            .if_stmt => {
                try testing.expect(n.lhs.int() < self);
                const h = Ast.ifHeaderAt(tree, n.rhs.int());
                try testing.expect(h.then_block.int() < self);
                if (h.else_node != Ast.none) try testing.expect(h.else_node.int() < self);
            },
            .literal_unit => {},
            .loop_expr => try testing.expect(n.lhs.int() < self),
            .for_stmt => {
                try testing.expect(n.lhs.int() < self);
                const h = Ast.forHeaderAt(tree, n.rhs.int());
                try testing.expect(h.lo.int() < self);
                try testing.expect(h.hi.int() < self);
            },
            // break/continue overload `rhs` as a *token* index (the label), so
            // only `lhs` (the value expr) is a child node to check.
            .break_stmt => if (n.lhs != Ast.none) try testing.expect(n.lhs.int() < self),
            .continue_stmt => {},
            .labeled => try testing.expect(n.lhs.int() < self),
            .struct_decl => {
                for (Ast.rangeSlice(tree, n.lhs.int())) |c| try testing.expect(c.int() < self);
                // Generics (when present) ride the `rhs` slot as a Range.
                if (n.rhs != Ast.none) for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            .struct_init => {
                try testing.expect(n.lhs.int() < self);
                for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            .field_init => try testing.expect(n.lhs.int() < self),
            .field_access => try testing.expect(n.lhs.int() < self),
            // Leaves: `main_token` only; no child node indices to order.
            .literal_number, .literal_float, .literal_string, .literal_bool, .literal_char, .identifier, .empty_list => {},
            .list_literal => for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self),
            .index => {
                try testing.expect(n.lhs.int() < self);
                try testing.expect(n.rhs.int() < self);
            },
            // A poison leaf holds only its offending token; no child nodes.
            .error_node => {},
            .enum_decl => {
                for (Ast.rangeSlice(tree, n.lhs.int())) |c| try testing.expect(c.int() < self);
                if (n.rhs != Ast.none) for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            .enum_variant_unit => {},
            .enum_variant_tuple, .enum_variant_struct => for (Ast.rangeSlice(tree, n.lhs.int())) |c| try testing.expect(c.int() < self),
            .enum_init_unit => if (n.lhs != Ast.none) try testing.expect(n.lhs.int() < self),
            .enum_init_tuple, .enum_init_struct => {
                if (n.lhs != Ast.none) try testing.expect(n.lhs.int() < self);
                for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            .match_expr => {
                try testing.expect(n.lhs.int() < self);
                for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            .match_arm => {
                try testing.expect(n.lhs.int() < self);
                const h = Ast.armHeaderAt(tree, n.rhs.int());
                if (h.guard != Ast.none) try testing.expect(h.guard.int() < self);
                try testing.expect(h.body.int() < self);
            },
            .pattern_variant => {
                if (n.lhs != Ast.none) try testing.expect(n.lhs.int() < self);
                if (n.rhs != Ast.none) for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            .pattern_wildcard, .pattern_literal => {},
            .pattern_binding => {
                if (n.lhs != Ast.none) try testing.expect(n.lhs.int() < self);
                if (n.rhs != Ast.none) try testing.expect(n.rhs.int() < self);
            },
            .pattern_or => for (Ast.rangeSlice(tree, n.lhs.int())) |c| try testing.expect(c.int() < self),
            // `import_decl` overloads `lhs`/`rhs` as TOKEN indices (path segments,
            // alias) like break/continue — no node children to order.
            .import_decl => {},
            // A declared type parameter: its name token, plus an optional bound
            // protocol-ref in `lhs` (`[T has P]`), created before this node.
            .generic_param => if (n.lhs != Ast.none) try testing.expect(n.lhs.int() < self),
            // `Base[Arg, ..]`: the base is `lhs`, the type-args are a Range in `rhs`.
            .type_app => {
                try testing.expect(n.lhs.int() < self);
                for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            // `impl T { fn .. }`: the receiver type-ref is `lhs`, the methods are a
            // Range of `fn_decl`s in `rhs` (all created before the impl node).
            .impl_decl => {
                try testing.expect(n.lhs.int() < self);
                for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            // `protocol P[X] { fn .. }`: the method sigs are a Range in `lhs`; the
            // optional generic-param Range rides `rhs` (or `none`).
            .protocol_decl => {
                for (Ast.rangeSlice(tree, n.lhs.int())) |c| try testing.expect(c.int() < self);
                if (n.rhs != Ast.none) for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            // `impl T has P { fn .. }`: receiver type-ref is `lhs`; the protocol ref +
            // methods live in the 3-cell header decoded by `implHasAt`.
            .impl_has_decl => {
                try testing.expect(n.lhs.int() < self);
                const h = Ast.implHasAt(tree, n.rhs.int());
                try testing.expect(h.protocol.int() < self);
                for (h.methods) |c| try testing.expect(c.int() < self);
            },
            // `type N = T`: the target type-ref is `lhs`, created before this node.
            .type_alias_decl => try testing.expect(n.lhs.int() < self),
            // `struct N(T0, ..)`: the positional type-refs are a Range in `lhs`;
            // the optional generic-param Range rides `rhs` (or `none`).
            .tuple_struct_decl => {
                for (Ast.rangeSlice(tree, n.lhs.int())) |c| try testing.expect(c.int() < self);
                if (n.rhs != Ast.none) for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            // `recv.N`: the receiver is `lhs`, created before this node.
            .tuple_field => try testing.expect(n.lhs.int() < self),
            // `extern fn ..`: same shape as fn_decl (proto in `lhs`, synthesized empty
            // block in `rhs`), so its proto/body indices precede this node too.
            .extern_fn_decl => {
                const proto = Ast.protoAt(tree, n.lhs.int());
                if (proto.ret_type != Ast.none) try testing.expect(proto.ret_type.int() < self);
                for (proto.params) |c| try testing.expect(c.int() < self);
                try testing.expect(n.rhs.int() < self);
            },
            // `unsafe { .. }`: the inner block is `lhs`, created before this node.
            .unsafe_block => try testing.expect(n.lhs.int() < self),
        }
    }
}

// --- generics front-end: parse-only (no semantics; Typecheck rejects them) ---

test "generic fn parses with a [T] segment" {
    try expectProgram("fn id[T](x: T) -> T { x }\n", "(program (fn id [T] ((param x T)) T (block x)))");
}

test "generic fn with multiple type params" {
    try expectProgram("fn f[T, U]() {}\n", "(program (fn f [T U] () _ (block)))");
}

test "generic struct stores its params in the rhs slot" {
    try expectProgram("struct Box[T] { v: T }\n", "(program (struct Box [T] (param v T)))");
}

test "generic enum stores its params in the rhs slot" {
    try expectProgram("enum E[T] { Some(T), None }\n", "(program (enum E [T] (variant.tuple Some T) (variant.unit None)))");
}

test "type application in type position parses to a tyapp" {
    try expectProgram("fn f(x: Box[int]) {}\n", "(program (fn f ((param x (tyapp Box int))) _ (block)))");
}

test "explicit call type-args wrap the callee in a tyapp" {
    try expectSexpr("id[int](7)", "(call (tyapp id int) 7)");
    // A qualified callee `mod.f[int](..)` wraps the field_access base.
    try expectSexpr("m.f[int](7)", "(call (tyapp (. m f) int) 7)");
}

test "nested type application nests tyapp nodes" {
    // Empty arg list renders with no trailing args after the callee.
    try expectSexpr("f[Box[int]]()", "(call (tyapp f (tyapp Box int)))");
}

test "postfix ? wraps its operand in a try_expr" {
    try expectSexpr("o?", "(try o)");
    // Composes onto a call result and onto a subsequent `.field`.
    try expectSexpr("parse(a)?", "(try (call parse a))");
    try expectSexpr("o?.x", "(. (try o) x)");
    // Double `?` nests left-to-right.
    try expectSexpr("o??", "(try (try o))");
}

test "constrained generic param stores the bound protocol-ref in its lhs" {
    const gpa = testing.allocator;
    const src = "fn twice[T has Doubler](v: T) -> int { 0 }\n";
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, src);
    defer freeTree(gpa, tree);

    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    const fn_decl_idx = Ast.rangeSlice(tree, prog.lhs.int())[0];
    const proto = Ast.protoAt(tree, tree.nodes[fn_decl_idx.int()].lhs.int());
    try testing.expectEqual(@as(usize, 1), proto.generic_params.len);
    const bound = Ast.genericParamBound(tree, proto.generic_params[0]) orelse return error.MissingBound;
    const bn = tree.nodes[bound.int()];
    try testing.expectEqual(Node.Tag.identifier, bn.tag);
    try testing.expectEqualStrings("Doubler", tokens[bn.main_token].text(src));
}

test "unbounded generic param leaves its lhs none" {
    const gpa = testing.allocator;
    const src = "fn id[T](x: T) -> T { x }\n";
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, src);
    defer freeTree(gpa, tree);

    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    const fn_decl_idx = Ast.rangeSlice(tree, prog.lhs.int())[0];
    const proto = Ast.protoAt(tree, tree.nodes[fn_decl_idx.int()].lhs.int());
    try testing.expectEqual(@as(?Ast.Index, null), Ast.genericParamBound(tree, proto.generic_params[0]));
}

test "a generic protocol stores its type-params in the rhs slot" {
    try expectProgram(
        "protocol Into[U] {\n fn into(self) -> U\n }\n",
        "(program (protocol Into [U] (fn into ((param self Into)) U (block))))",
    );
}

test "a non-generic protocol leaves its rhs none" {
    try expectProgram(
        "protocol Named {\n fn name(self) -> int\n }\n",
        "(program (protocol Named (fn name ((param self Named)) int (block))))",
    );
    const gpa = testing.allocator;
    const src = "protocol Named {\n fn name(self) -> int\n }\n";
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, src);
    defer freeTree(gpa, tree);
    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    const decl = Ast.rangeSlice(tree, prog.lhs.int())[0];
    try testing.expectEqual(Ast.none, tree.nodes[decl.int()].rhs);
    try testing.expectEqual(@as(usize, 0), Ast.protocolGenericParams(tree, decl).len);
}

test "impl P has Into[int] builds a type_app protocol-ref" {
    const gpa = testing.allocator;
    const src = "impl P has Into[int] {\n fn into(self) -> int { self.x }\n }\n";
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, src);
    defer freeTree(gpa, tree);
    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    const decl = tree.nodes[Ast.rangeSlice(tree, prog.lhs.int())[0].int()];
    const proto = Ast.implProtocol(tree, decl) orelse return error.MissingProtocol;
    try testing.expectEqual(Node.Tag.type_app, tree.nodes[proto.int()].tag);
    const base = Ast.protocolRefBase(tree, proto);
    try testing.expectEqual(Node.Tag.identifier, tree.nodes[base.int()].tag);
    try testing.expectEqualStrings("Into", tokens[tree.nodes[base.int()].main_token].text(src));
    const args = Ast.protocolRefArgs(tree, proto);
    try testing.expectEqual(@as(usize, 1), args.len);
    try testing.expectEqualStrings("int", tokens[tree.nodes[args[0].int()].main_token].text(src));
}

test "[T has Into[int]] bound builds a type_app protocol-ref" {
    const gpa = testing.allocator;
    const src = "fn use[T has Into[int]](v: T) -> int { 0 }\n";
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, src);
    defer freeTree(gpa, tree);
    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    const fn_decl_idx = Ast.rangeSlice(tree, prog.lhs.int())[0];
    const proto = Ast.protoAt(tree, tree.nodes[fn_decl_idx.int()].lhs.int());
    const bound = Ast.genericParamBound(tree, proto.generic_params[0]) orelse return error.MissingBound;
    try testing.expectEqual(Node.Tag.type_app, tree.nodes[bound.int()].tag);
    const base = Ast.protocolRefBase(tree, bound);
    try testing.expectEqualStrings("Into", tokens[tree.nodes[base.int()].main_token].text(src));
    try testing.expectEqual(@as(usize, 1), Ast.protocolRefArgs(tree, bound).len);
}

// --- inherent methods: impl block parsing ---

test "impl block parses a method with a synthesized self param" {
    try expectProgram(
        "struct P { x: int, y: int }\nimpl P { fn sum(self) -> int { self.x + self.y } }\n",
        "(program (struct P (param x int) (param y int)) (impl P (fn sum ((param self P)) int (block (+ (. self x) (. self y))))))",
    );
}

test "impl method with an extra param keeps self as params[0]" {
    try expectProgram(
        "struct P { x: int }\nimpl P { fn add(self, a: int) -> int { self.x + a } }\n",
        "(program (struct P (param x int)) (impl P (fn add ((param self P) (param a int)) int (block (+ (. self x) a)))))",
    );
}

test "impl block with multiple methods" {
    try expectProgram(
        "struct P { x: int }\nimpl P {\n fn get(self) -> int { self.x }\n fn zero(self) -> int { 0 }\n}\n",
        "(program (struct P (param x int)) (impl P (fn get ((param self P)) int (block (. self x))) (fn zero ((param self P)) int (block 0))))",
    );
}

test "mut self parses to the same synthesized self-param shape (no node change)" {
    // `mut self` carries no dedicated node/cell — it renders identically to plain
    // `self`; the `mut` is detected later by token adjacency (`Ast.isMutParam`).
    try expectProgram(
        "struct P { x: int }\nimpl P { fn bump(mut self, d: int) { self.x = self.x + d } }\n",
        "(program (struct P (param x int)) (impl P (fn bump ((param self P) (param d int)) _ (block (= (. self x) (+ (. self x) d))))))",
    );
}

test "a generic impl parses; the receiver is a type_app and the method's self type-ref is it" {
    // The method carries the impl's `[T]` in its FnProto generic run and its `self`
    // type-ref is the receiver `type_app` `Box[T]` (so it decodes to an App).
    try expectProgram(
        "struct Box[T] { v: T }\nimpl Box[T] { fn get(self) -> T { self.v } }\n",
        "(program (struct Box [T] (param v T)) (impl Box (fn get [T] ((param self (tyapp Box T))) T (block (. self v)))))",
    );
}

test "the generic impl's receiver type_app is SHARED as impl.lhs and each method's self type-ref" {
    const gpa = testing.allocator;
    const source = "struct Box[T] { v: T }\nimpl Box[T] { fn get(self) -> T { self.v } }\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, source);
    defer freeTree(gpa, tree);

    var impl_idx: ?Ast.Index = null;
    for (tree.nodes, 0..) |n, i| {
        if (n.tag == .impl_decl) impl_idx = Ast.Index.from(@intCast(i));
    }
    try testing.expect(impl_idx != null);
    const impl = tree.nodes[impl_idx.?.int()];
    // impl_decl.lhs is a `type_app` (the generic receiver), not a bare identifier.
    try testing.expectEqual(Node.Tag.type_app, tree.nodes[impl.lhs.int()].tag);

    const methods = Ast.rangeSlice(tree, impl.rhs.int());
    try testing.expectEqual(@as(usize, 1), methods.len);
    const proto = Ast.protoAt(tree, tree.nodes[methods[0].int()].lhs.int());
    // The method has exactly one generic param, named "T" (the impl's).
    try testing.expectEqual(@as(usize, 1), proto.generic_params.len);
    try testing.expectEqualStrings("T", tokens[tree.nodes[proto.generic_params[0].int()].main_token].text(source));
    // The self param's type-ref IS the shared receiver type_app node.
    try testing.expectEqual(@as(usize, 1), proto.params.len);
    try testing.expectEqual(impl.lhs, tree.nodes[proto.params[0].int()].lhs);
}

test "a qualified `impl mod.T` receiver is still rejected (P0001)" {
    const gpa = testing.allocator;
    const source = "impl a.b { fn m(self) -> int { 0 } }\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const res = try parse(gpa, tokens, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);
    try testing.expect(res.diags.len >= 1);
}

test "a protocol decl parses signature-only methods (empty body)" {
    // The self param's type-ref renders as the protocol name (inert — never decoded),
    // and the bodyless signature carries a synthesized empty `(block)`.
    try expectProgram(
        "protocol Named { fn name(self) -> int }\n",
        "(program (protocol Named (fn name ((param self Named)) int (block))))",
    );
}

test "a protocol decl with multiple signatures" {
    try expectProgram(
        "protocol Shape {\n fn area(self) -> int\n fn sides(self) -> int\n}\n",
        "(program (protocol Shape (fn area ((param self Shape)) int (block)) (fn sides ((param self Shape)) int (block))))",
    );
}

test "an impl-has decl parses (recv then protocol then methods)" {
    try expectProgram(
        "struct P { x: int }\nimpl P has Named { fn name(self) -> int { self.x } }\n",
        "(program (struct P (param x int)) (impl-has P Named (fn name ((param self P)) int (block (. self x)))))",
    );
}

test "a qualified impl-has parses (qualified receiver AND protocol)" {
    // Both the receiver `lib.W` and the protocol `lib.Show` render as field_access
    // chains; the self param's type-ref is the shared qualified receiver node.
    try expectProgram(
        "impl lib.W has lib.Show { fn show(self) -> int { self.n } }\n",
        "(program (impl-has (. lib W) (. lib Show) (fn show ((param self (. lib W))) int (block (. self n)))))",
    );
}

test "a generic `impl Box[T] has P` receiver is rejected (P0001)" {
    const gpa = testing.allocator;
    const source = "impl Box[T] has P { fn m(self) -> int { 0 } }\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const res = try parse(gpa, tokens, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);
    try testing.expect(res.diags.len >= 1);
}

test "`pub protocol` records the pub export" {
    try expectProgram(
        "pub protocol Named { fn name(self) -> int }\n",
        "(program (pub (protocol Named (fn name ((param self Named)) int (block)))))",
    );
}

test "isMutParam is true for `mut self`, false for plain self / a non-self first param" {
    const gpa = testing.allocator;
    const source =
        "struct P { x: int }\n" ++
        "impl P {\n" ++
        "  fn bump(mut self, d: int) { self.x = self.x + d }\n" ++
        "  fn get(self) -> int { self.x }\n" ++
        "}\n" ++
        "fn free(a: int) -> int { a }\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, source);
    defer freeTree(gpa, tree);

    // Collect the three fn_decls in source order: bump (mut self), get (self), free.
    var fns: [3]Ast.Index = undefined;
    var nfns: usize = 0;
    for (tree.nodes, 0..) |n, i| {
        if (n.tag == .fn_decl) {
            fns[nfns] = Ast.Index.from(@intCast(i));
            nfns += 1;
        }
    }
    try testing.expectEqual(@as(usize, 3), nfns);

    const bump_proto = Ast.protoAt(tree, tree.nodes[fns[0].int()].lhs.int());
    const get_proto = Ast.protoAt(tree, tree.nodes[fns[1].int()].lhs.int());
    const free_proto = Ast.protoAt(tree, tree.nodes[fns[2].int()].lhs.int());

    try testing.expect(Ast.isMutParam(tree, tokens, bump_proto.params[0])); // `mut self`
    try testing.expect(!Ast.isMutParam(tree, tokens, bump_proto.params[1])); // `d: int`
    try testing.expect(!Ast.isMutParam(tree, tokens, get_proto.params[0])); // plain `self`
    try testing.expect(!Ast.isMutParam(tree, tokens, free_proto.params[0])); // `a: int`
}

test "a mut on a non-self first param is rejected (P0006), mut self is not" {
    const gpa = testing.allocator;
    // `mut x` on a top-level fn is out of scope: the reserved `mut` keyword makes
    // the param loop's identifier expectation fail → a diagnostic. A `mut self` method
    // parses cleanly.
    const bad = "fn f(mut x: int) -> int { x }\n";
    {
        const tokens = try Lexer.tokenize(gpa, bad);
        defer gpa.free(tokens);
        const res = try parse(gpa, tokens, bad);
        defer gpa.free(@constCast(res.diags));
        defer freeTree(gpa, res.tree);
        try testing.expect(res.diags.len >= 1);
    }
    const good = "struct P { x: int }\nimpl P { fn bump(mut self, d: int) { self.x = self.x + d } }\nfn main() -> int { 0 }\n";
    {
        const tokens = try Lexer.tokenize(gpa, good);
        defer gpa.free(tokens);
        const res = try parse(gpa, tokens, good);
        defer gpa.free(@constCast(res.diags));
        defer freeTree(gpa, res.tree);
        try testing.expectEqual(@as(usize, 0), res.diags.len);
    }
}

test "a self-less inherent method is an associated function: a clean parse, no P0006" {
    const gpa = testing.allocator;
    const source = "struct P { x: int }\nimpl P { fn make() -> int { 0 } }\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const res = try parse(gpa, tokens, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);
    for (res.diags) |d| try testing.expect(d.code != .P0006);
}

test "generic nodes precede their parents (children-before-parents on generics)" {
    const gpa = testing.allocator;
    const source =
        \\struct Box[T] { v: T }
        \\enum E[U] { Some(U), None }
        \\fn f[A, B](x: Box[int]) -> A { x }
        \\
    ;
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, source);
    defer freeTree(gpa, tree);

    var saw_generic_param = false;
    var saw_type_app = false;
    for (tree.nodes, 0..) |n, i| {
        const self: u32 = @intCast(i);
        switch (n.tag) {
            .generic_param => saw_generic_param = true,
            .type_app => {
                saw_type_app = true;
                try testing.expect(n.lhs.int() < self);
                for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            .fn_decl => {
                const proto = Ast.protoAt(tree, n.lhs.int());
                for (proto.generic_params) |c| try testing.expect(c.int() < self);
            },
            .struct_decl, .enum_decl => {
                if (n.rhs != Ast.none) for (Ast.rangeSlice(tree, n.rhs.int())) |c| try testing.expect(c.int() < self);
            },
            else => {},
        }
    }
    try testing.expect(saw_generic_param);
    try testing.expect(saw_type_app);
}

test "generic-struct construction Box[int]{ v: 1 } builds struct_init over a type_app" {
    const gpa = testing.allocator;
    const source = "fn main() -> int {\n b := Box[int]{ v: 1 }\n b.v\n}\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, source);
    defer freeTree(gpa, tree);

    var saw = false;
    for (tree.nodes, 0..) |n, i| {
        if (n.tag != .struct_init) continue;
        // The construction's lhs is the `type_app` (Box[int]), created before it, so
        // children still precede the parent (the on-every-parse invariant sweep holds).
        try testing.expect(tree.nodes[n.lhs.int()].tag == .type_app);
        try testing.expect(n.lhs.int() < @as(u32, @intCast(i)));
        saw = true;
    }
    try testing.expect(saw);
}

test "malformed generic list recovers without crashing (tainted parse)" {
    const gpa = testing.allocator;
    const source = "fn f[ , ]() {}\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const res = try parse(gpa, tokens, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);
    // A garbage generic list is rejected at parse time; the parse is tainted but the
    // on-every-parse invariant sweep (span totality + bracket pairing) still holds.
    try testing.expect(res.diags.len > 0);
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
    // Contract: parse ALWAYS returns a tree; the error travels in `diags`.
    const gpa = testing.allocator;
    const source = "fn f() {\n @x 1\n}\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const result = try parse(gpa, tokens, source);
    defer gpa.free(@constCast(result.diags));
    defer freeTree(gpa, result.tree);
    try testing.expect(result.diags.len >= 1);
    try testing.expectEqualStrings("a label must prefix a loop, while, for, or block", result.diags[0].message);
}

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
    const tree = try expectTree(gpa, tokens, source);
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
    const tree = try expectTree(gpa, tokens, source);
    defer freeTree(gpa, tree);

    const blob = try Ast.pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try Ast.unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer freeTree(gpa, got);

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
}

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
    // Contract: parse ALWAYS returns a tree; the error travels in `diags`.
    const gpa = testing.allocator;
    const source = "pub import a/b\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const result = try parse(gpa, tokens, source);
    defer gpa.free(@constCast(result.diags));
    defer freeTree(gpa, result.tree);
    try testing.expect(result.diags.len >= 1);
    try testing.expectEqualStrings("expected a function, struct, or enum declaration after 'pub'", result.diags[0].message);
}

test "import path missing a segment after slash is an error" {
    // Contract: parse ALWAYS returns a tree; the error travels in `diags`.
    const gpa = testing.allocator;
    const source = "import a/\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const result = try parse(gpa, tokens, source);
    defer gpa.free(@constCast(result.diags));
    defer freeTree(gpa, result.tree);
    try testing.expect(result.diags.len >= 1);
    try testing.expectEqualStrings("expected a path segment after '/'", result.diags[0].message);
}

test "import pack/unpack byte round-trip carries pub_bits" {
    const gpa = testing.allocator;
    const source = "import geometry/rect as r\npub fn area() -> int { 0 }\nfn helper() -> int { 1 }\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const tree = try expectTree(gpa, tokens, source);
    defer freeTree(gpa, tree);

    // Exactly the `area` fn_decl node is pub; `helper` is not.
    var pub_count: usize = 0;
    for (tree.nodes, 0..) |node, i| {
        if (node.tag == .fn_decl and tree.isPub(Ast.Index.from(@intCast(i)))) pub_count += 1;
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
    const tree = try expectTree(gpa, tokens, source);
    defer freeTree(gpa, tree);

    const blob = try Ast.pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try Ast.unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer freeTree(gpa, got);

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
}

/// Render a (possibly tainted) parse result's tree into `buf`, returning the
/// rendered slice — a test-only convenience for the recovery tests that inspect
/// the shape of a recovered tree.
fn renderResult(res: Result, source: []const u8, buf: []u8) ![]const u8 {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    var w = std.Io.Writer.fixed(buf);
    try Ast.render(&w, res.tree, tokens, source);
    return w.buffered();
}

test "two independent errors in one file both report and both decls survive" {
    // `return )` in a, `return )` in b — two INDEPENDENT broken statements, one per
    // decl. Each stray `)` in return-value position is the cascade signature (a
    // structural closer in a statement value slot): before the cascade fix each site
    // sprayed TWO diagnostics ("expected an expression" + "expected a newline or
    // '}'"), so the file yielded FOUR. The latch must collapse each site to one AND
    // clear at the decl boundary so the second site still reports — EXACTLY TWO
    // total, not four (over-cascade) and not one (over-suppression).
    const gpa = testing.allocator;
    const source = "fn a() -> int {\n  return )\n}\nfn b() -> int {\n  return )\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expectEqual(@as(usize, 2), res.diags.len);
    for (res.diags) |d| try testing.expectEqualStrings("expected an expression", d.message);
    const prog = res.tree.nodes[Ast.root(res.tree.nodes).int()];
    try testing.expectEqual(Node.Tag.program, prog.tag);
    // BOTH fn decls present: the top-level decl loop recovered across the break.
    try testing.expectEqual(@as(usize, 2), Ast.rangeSlice(res.tree, prog.lhs.int()).len);
    // The two error offsets straddle b's declaration keyword (distinct sites).
    const b_off: u32 = @intCast(std.mem.indexOf(u8, source, "fn b").?);
    var before: usize = 0;
    var after: usize = 0;
    for (res.diags) |d| {
        if (d.byte_offset < b_off) before += 1 else after += 1;
    }
    try testing.expectEqual(@as(usize, 1), before);
    try testing.expectEqual(@as(usize, 1), after);
}

test "a stray ')' in return-value position yields exactly one diagnostic (no cascade)" {
    // The confirmed cascade defect: `return )` — a structural closer where an
    // expression is expected. parsePrefix reports "expected an expression" and
    // returns an error_node WITHOUT consuming the `)`; before the cascade latch the
    // unconsumed `)` then tripped expectTerminator into a SECOND spurious "expected a
    // newline or '}'" at the same offset. The latch collapses this to ONE.
    const gpa = testing.allocator;
    const source = "fn a() -> int {\n  return )\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("expected an expression", res.diags[0].message);
    // Point at the stray `)` in return-value position (the one after `return `),
    // not the `()` in the fn signature.
    const paren_off: u32 = @intCast(std.mem.indexOf(u8, source, "return )").? + "return ".len);
    try testing.expectEqual(paren_off, res.diags[0].byte_offset);
}

test "a stray ')' after '=' assignment value yields exactly one diagnostic (no cascade)" {
    // The same cascade signature across a different statement form: `x = )`. Proves
    // the fix is systematic (not special-cased to `return`), collapsing the
    // parsePrefix + expectTerminator pair over the unconsumed `)` to ONE diagnostic.
    const gpa = testing.allocator;
    const source = "fn a() -> int {\n  x = )\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("expected an expression", res.diags[0].message);
}

test "two adjacent broken statements (no good stmt between) each report — no over-suppression" {
    // Over-suppression regression guard. `return )` on line 2 and `x = )` on line 3
    // are two INDEPENDENT sites on distinct lines with NO valid statement between
    // them. The first site's stray `)` is left unconsumed, so recovery goes through
    // `findNextStmt` (not `expectTerminator`'s clean newline arm). If the cascade
    // latch is not cleared when that scan crosses the line's newline boundary, the
    // second site's "expected an expression" is silently swallowed and the file
    // reports only ONE diagnostic. It must report EXACTLY TWO, one per site.
    const gpa = testing.allocator;
    const source = "fn a() -> int {\n  return )\n  x = )\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expectEqual(@as(usize, 2), res.diags.len);
    for (res.diags) |d| try testing.expectEqualStrings("expected an expression", d.message);
    // The two diagnostics sit at two distinct offsets (the two stray `)`s), proving
    // they are two independent sites, not a duplicated single-site cascade.
    try testing.expect(res.diags[0].byte_offset != res.diags[1].byte_offset);
    const first_paren: u32 = @intCast(std.mem.indexOf(u8, source, "return )").? + "return ".len);
    const second_paren: u32 = @intCast(std.mem.indexOf(u8, source, "x = )").? + "x = ".len);
    try testing.expectEqual(first_paren, res.diags[0].byte_offset);
    try testing.expectEqual(second_paren, res.diags[1].byte_offset);
}

test "a valid statement between two broken sites still yields exactly two diagnostics" {
    // The complementary guard: a well-formed statement (`y := 1`) between the two
    // broken sites must NOT itself add a diagnostic, and both broken sites must
    // still report — exactly two total. Proves the latch clears cleanly across a
    // successful statement without either over-reporting or over-suppressing.
    const gpa = testing.allocator;
    const source = "fn a() -> int {\n  return )\n  y := 1\n  x = )\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expectEqual(@as(usize, 2), res.diags.len);
    for (res.diags) |d| try testing.expectEqualStrings("expected an expression", d.message);
}

test "adversarial `fn f( ) ) ) {` terminates (no hang) and yields a tree" {
    // The anti-hang backstop: this must RETURN (a hanging test is the failure),
    // yield a non-empty program tree, and report at least one diagnostic.
    const gpa = testing.allocator;
    const source = "fn f( ) ) ) {\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expect(res.tree.nodes.len > 0);
    try testing.expectEqual(Node.Tag.program, res.tree.nodes[Ast.root(res.tree.nodes).int()].tag);
    try testing.expect(res.diags.len >= 1);
}

test "adversarial garbage recovers to a following well-formed decl" {
    // After the adversarial decl the parser must resync to a real following decl.
    const gpa = testing.allocator;
    const source = "fn f( ) ) ) {\nfn g() -> int { 0 }\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    var buf: [512]u8 = undefined;
    const rendered = try renderResult(res, source, &buf);
    try testing.expect(std.mem.indexOf(u8, rendered, "(fn g") != null);
    try testing.expect(res.diags.len >= 1);
}

test "a broken statement recovers to the next statement" {
    // `return )` is a broken statement (stray `)`); `y := 2` and the final
    // `return` must still parse — a broken statement does not poison its siblings.
    const gpa = testing.allocator;
    const source = "fn f() {\n  return )\n  y := 2\n  return\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expect(res.diags.len >= 1);
    var buf: [512]u8 = undefined;
    const rendered = try renderResult(res, source, &buf);
    // The later statements survived the broken one (resync landed on `y`).
    try testing.expect(std.mem.indexOf(u8, rendered, "(:= y 2)") != null);
    // The final bare `return` also parsed.
    try testing.expect(std.mem.indexOf(u8, rendered, "(return))") != null);
}

test "a broken decl recovers to the next decl" {
    // `fn a( { }` is a malformed decl; `fn b` must still parse.
    const gpa = testing.allocator;
    const source = "fn a( { }\nfn b() -> int { 0 }\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expect(res.diags.len >= 1);
    var buf: [512]u8 = undefined;
    const rendered = try renderResult(res, source, &buf);
    try testing.expect(std.mem.indexOf(u8, rendered, "(fn b") != null);
}

test "one root error yields exactly one diagnostic (no cascade)" {
    // A single bad operand (`+` after `return`; `+` is binary-only, invalid at
    // expression start) must produce exactly ONE diagnostic — the ASI/newline
    // anti-cascade guards against duplicates.
    const gpa = testing.allocator;
    const source = "fn f() -> int {\n  return +\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("expected an expression", res.diags[0].message);
    const bad_off: u32 = @intCast(std.mem.indexOfScalar(u8, source, '+').?);
    try testing.expectEqual(bad_off, res.diags[0].byte_offset);
}

test "a missing call operand before ')' yields exactly one diagnostic (no closer-delete cascade)" {
    // `g(1 + )` — the RHS of `+` is missing and the next token is the call's own
    // `)`. parsePrefix must NOT delete that `)` (doing so would break the call's
    // closing `expect`, then the block's, spraying a diagnostic per open ancestor).
    // It returns an error_node without consuming, so the arg loop sees `)`, breaks,
    // and `expect(.r_paren)` consumes it — collapsing the whole cascade to ONE
    // diagnostic (the missing expression).
    const gpa = testing.allocator;
    const source = "fn g(a: int) -> int { 0 }\nfn f() -> int {\n  g(1 + )\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("expected an expression", res.diags[0].message);
}

test "a trailing ':=' before '}' yields exactly one diagnostic (no closer-delete cascade)" {
    // `x := \n}` — the trailing `:=` suppresses the newline, so the initializer's
    // parsePrefix lands on the block's `}`. Deleting it would swallow the block
    // closer and add a spurious "expected '}'"; instead the error_node is returned
    // without consuming, `expectTerminator` accepts the implicit `}`, and the block
    // loop closes normally — ONE diagnostic.
    const gpa = testing.allocator;
    const source = "fn a() -> int {\n  x := \n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("expected an expression", res.diags[0].message);
}

test "a broken call argument list reports and resyncs without hanging" {
    // `g(1, , 3)` — the doubled comma is a tuple_recovery anchor, so the arg loop
    // breaks and the trailing `expect(.r_paren)` fails at the stray comma; the
    // block-statement loop then resyncs to the next statement (`3`). The call is
    // abandoned rather than repaired in place, but recovery is bounded: parsing
    // reaches EOF, a diagnostic is reported, and there is no hang. (The design's
    // recovery-set code — comma ∈ tuple_recovery → break — is authoritative over
    // its own test-plan prose, which imagined the comma being eaten in place.)
    const gpa = testing.allocator;
    const source = "fn g(a: int, b: int, c: int) -> int { 0 }\nfn f() -> int { g(1, , 3) }\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expect(res.diags.len >= 1);
    // Both fn decls survive: the error stayed inside f's body.
    const prog = res.tree.nodes[Ast.root(res.tree.nodes).int()];
    try testing.expectEqual(@as(usize, 2), Ast.rangeSlice(res.tree, prog.lhs.int()).len);
    // Recovery resynced to the trailing `3` statement inside f's block.
    var buf: [512]u8 = undefined;
    const rendered = try renderResult(res, source, &buf);
    try testing.expect(std.mem.indexOf(u8, rendered, "(fn f () int (block 3))") != null);
}

test "deep expression nesting is capped instead of overflowing the stack" {
    // ~300 nested parens (> MAX_EXPR_DEPTH). parse() must RETURN (no SIGBUS) with a
    // "nested too deeply" diagnostic and a present tree.
    const gpa = testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, "fn f() -> int {\n  return ");
    const n = 300;
    try buf.appendNTimes(gpa, '(', n);
    try buf.append(gpa, '1');
    try buf.appendNTimes(gpa, ')', n);
    try buf.appendSlice(gpa, "\n}\n");

    const res = try parseResult(gpa, buf.items);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expect(res.tree.nodes.len > 0);
    var found = false;
    for (res.diags) |d| {
        if (std.mem.eql(u8, d.message, "expression nested too deeply")) found = true;
    }
    try testing.expect(found);
}

test "deep statement nesting is capped instead of overflowing the stack" {
    // Statement-keyword recursion (`while`/`if`) nests through parseBlock, NOT
    // parseExpr, so the parseExpr guard cannot see it. ~2000 unclosed `while true{`
    // is well above MAX_EXPR_DEPTH: parse() must RETURN (no SIGBUS) with a "nested
    // too deeply" diagnostic and a present tree. Unclosed braces taint the parse,
    // isolating the crash to the parser's recursion.
    const gpa = testing.allocator;
    inline for (.{ "while true{\n", "if a{\n" }) |nest| {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, "fn f() -> int {\n");
        var i: usize = 0;
        while (i < 2000) : (i += 1) try buf.appendSlice(gpa, nest);

        const res = try parseResult(gpa, buf.items);
        defer gpa.free(@constCast(res.diags));
        defer freeTree(gpa, res.tree);

        try testing.expect(res.tree.nodes.len > 0);
        var found = false;
        for (res.diags) |d| {
            if (std.mem.eql(u8, d.message, "block nested too deeply")) found = true;
        }
        try testing.expect(found);
    }
}

test "deep generic type nesting is capped instead of overflowing the stack" {
    // parseType<->parseTypeApp recurse on nested type-applications without funneling
    // through the parseExpr/parseBlock guards. A `Box[Box[..[int]..]]` far past
    // MAX_EXPR_DEPTH must RETURN (no SIGBUS) with a "type nested too deeply" diagnostic
    // and a present tree.
    const gpa = testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, "fn f(x: ");
    const n = 1000;
    var i: usize = 0;
    while (i < n) : (i += 1) try buf.appendSlice(gpa, "Box[");
    try buf.appendSlice(gpa, "int");
    try buf.appendNTimes(gpa, ']', n);
    try buf.appendSlice(gpa, ") {}\n");

    const res = try parseResult(gpa, buf.items);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expect(res.tree.nodes.len > 0);
    var found = false;
    for (res.diags) |d| {
        if (d.code == .P0005 and std.mem.eql(u8, d.message, "type nested too deeply")) found = true;
    }
    try testing.expect(found);
}

test "match arm starting on a separator makes forward progress (no hang)" {
    // A leading `,`/newline arm was the one recovery loop that could stall:
    // resyncTo stops on the separator without consuming, so the loop must skip it.
    // Each of these must RETURN with >=1 diagnostic and a present tree.
    const gpa = testing.allocator;
    inline for (.{
        "fn f(s: S) -> int {\n match s {\n ,\n }\n}\n",
        "fn f(s: S) -> int {\n match s { , , , }\n}\n",
    }) |source| {
        const res = try parseResult(gpa, source);
        defer gpa.free(@constCast(res.diags));
        defer freeTree(gpa, res.tree);

        try testing.expect(res.tree.nodes.len > 0);
        try testing.expect(res.diags.len >= 1);
    }
}

test "deep infix chain is bounded (RHS recursion depth)" {
    // A long `1 + 1 + ... + 1` exercises infix-RHS recursion through parseExpr.
    // Left-associative infix does not deepen parseExpr recursion, so a 200-term
    // chain stays well under the cap and parses cleanly.
    const gpa = testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, "fn f() -> int {\n  return 1");
    var i: usize = 0;
    while (i < 200) : (i += 1) try buf.appendSlice(gpa, " + 1");
    try buf.appendSlice(gpa, "\n}\n");

    const res = try parseResult(gpa, buf.items);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    try testing.expectEqual(@as(usize, 0), res.diags.len);
}

test "a recovered error_node-bearing tree round-trips byte-identically" {
    // Cache soundness under recovery: an error_node's contentFp uses only
    // tag+main_token+lhs+rhs, so a recovered tree packs/unpacks byte-identically.
    const gpa = testing.allocator;
    const source = "fn f() -> int {\n  return *\n}\n";
    const res = try parseResult(gpa, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    // Sanity: the tree actually carries an error_node.
    var has_error = false;
    for (res.tree.nodes) |nd| {
        if (nd.tag == .error_node) has_error = true;
    }
    try testing.expect(has_error);

    const blob = try Ast.pack(gpa, res.tree);
    defer gpa.free(blob);
    const got = (try Ast.unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer freeTree(gpa, got);

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(res.tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, res.tree.extra, got.extra);
}

const codes = @import("diagnostics/codes.zig");

test "parse diagnostics carry P-codes by syntactic category" {
    const gpa = testing.allocator;

    // Expression position: `return )` => expected-expression => P0002.
    {
        const res = try parseResult(gpa, "fn a() -> int {\n  return )\n}\n");
        defer gpa.free(@constCast(res.diags));
        defer freeTree(gpa, res.tree);
        try testing.expectEqual(@as(usize, 1), res.diags.len);
        try testing.expectEqual(codes.Code.P0002, res.diags[0].code);
    }
    // Decl dispatch: a bare `pub 3` at the top level => expected-declaration => P0003.
    {
        const res = try parseResult(gpa, "pub 3\n");
        defer gpa.free(@constCast(res.diags));
        defer freeTree(gpa, res.tree);
        try testing.expect(res.diags.len >= 1);
        try testing.expectEqual(codes.Code.P0003, res.diags[0].code);
    }
    // Invalid label target: `@lbl 3` => P0007.
    {
        const res = try parseResult(gpa, "fn a() -> int {\n  @lbl 3\n  return 0\n}\n");
        defer gpa.free(@constCast(res.diags));
        defer freeTree(gpa, res.tree);
        try testing.expect(res.diags.len >= 1);
        try testing.expectEqual(codes.Code.P0007, res.diags[0].code);
    }
    // Missing expected token (expect() hardcodes P0001): a missing `)` in the
    // param list surfaces via the trailing expect(.r_paren).
    {
        const res = try parseResult(gpa, "fn f( -> int { return 0 }\n");
        defer gpa.free(@constCast(res.diags));
        defer freeTree(gpa, res.tree);
        try testing.expect(res.diags.len >= 1);
        try testing.expectEqual(codes.Code.P0001, res.diags[0].code);
    }
}

test "a self-less method parses with no P0006 in BOTH a conformance and an inherent impl" {
    const gpa = testing.allocator;

    // A CONFORMANCE impl (`impl T has P`) permits a self-less method (`From`): no
    // P0006, and the single declared param is read as an ordinary param (no synthetic self).
    {
        const res = try parseResult(gpa,
            \\enum SmallErr { bad }
            \\enum BigErr { small }
            \\impl BigErr has From[SmallErr] {
            \\ fn from(s: SmallErr) -> BigErr { BigErr.small }
            \\}
            \\
        );
        defer gpa.free(@constCast(res.diags));
        defer freeTree(gpa, res.tree);
        for (res.diags) |d| try testing.expect(d.code != .P0006);
        // Locate the `impl_has_decl` and its single `from` method; assert exactly one param
        // (the self-less `s: SmallErr`) — no synthesized self prepended.
        const prog = res.tree.nodes[Ast.root(res.tree.nodes).int()];
        var saw_from = false;
        for (Ast.rangeSlice(res.tree, prog.lhs.int())) |decl_idx| {
            const decl = res.tree.nodes[decl_idx.int()];
            if (decl.tag != .impl_has_decl) continue;
            for (Ast.implMethods(res.tree, decl)) |mnode| {
                const m = res.tree.nodes[mnode.int()];
                const proto = Ast.protoAt(res.tree, m.lhs.int());
                try testing.expectEqual(@as(usize, 1), proto.params.len);
                saw_from = true;
            }
        }
        try testing.expect(saw_from);
    }

    // An INHERENT impl (`impl T`) now permits a self-less ASSOCIATED function
    // (`fn new() -> Vec[T]`): no P0006, and its declared params carry no synthesized self.
    {
        const res = try parseResult(gpa,
            \\struct Vec[T] { n: int }
            \\impl Vec[T] {
            \\ fn new() -> Vec[T] { Vec[T] { n: 0 } }
            \\}
            \\
        );
        defer gpa.free(@constCast(res.diags));
        defer freeTree(gpa, res.tree);
        for (res.diags) |d| try testing.expect(d.code != .P0006);
        const prog = res.tree.nodes[Ast.root(res.tree.nodes).int()];
        var saw_new = false;
        for (Ast.rangeSlice(res.tree, prog.lhs.int())) |decl_idx| {
            const decl = res.tree.nodes[decl_idx.int()];
            if (decl.tag != .impl_decl) continue;
            for (Ast.implMethods(res.tree, decl)) |mnode| {
                const m = res.tree.nodes[mnode.int()];
                const proto = Ast.protoAt(res.tree, m.lhs.int());
                try testing.expectEqual(@as(usize, 0), proto.params.len); // no synthesized self
                saw_new = true;
            }
        }
        try testing.expect(saw_new);
    }
}

test "a nesting-depth backstop carries the P0005 code" {
    const gpa = testing.allocator;
    // Mirror the block-depth fixture (~2000 unclosed `while true{`), which is
    // known to terminate with a "block nested too deeply" backstop; assert the code.
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, "fn f() -> int {\n");
    var i: usize = 0;
    while (i < 2000) : (i += 1) try buf.appendSlice(gpa, "while true{\n");

    const res = try parseResult(gpa, buf.items);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    var saw_p0005 = false;
    for (res.diags) |d| if (d.code == .P0005) {
        saw_p0005 = true;
    };
    try testing.expect(saw_p0005);
}

test "a valid program stamps no code (byte-identical contract)" {
    const gpa = testing.allocator;
    const res = try parseResult(gpa, "fn main() -> int {\n  return 0\n}\n");
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);
    // No diagnostic => no code is ever stamped on a clean parse.
    try testing.expectEqual(@as(usize, 0), res.diags.len);
}

test "report-once keeps exactly one diagnostic with exactly one code" {
    const gpa = testing.allocator;
    // A cascade fixture: one broken statement must collapse to a single coded diag.
    const res = try parseResult(gpa, "fn a() -> int {\n  return )\n}\n");
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expect(res.diags[0].code != .none);
}

test "span totality holds over the full token list of a clean parse" {
    const gpa = testing.allocator;
    const source = "fn main() -> int {\n  return 0\n}\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    // Direct check of the (a) sweep: monotonic tiling, well-formed spans, eof==len.
    checkSpanTotality(tokens, source);
    // And the whole parse (which runs the sweep under runtime_safety) is clean.
    const res = try parse(gpa, tokens, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);
    try testing.expectEqual(@as(usize, 0), res.diags.len);
}

test "a clean parse has balanced brackets and no error node" {
    const gpa = testing.allocator;
    const source = "fn f(a: int) -> int {\n  return (a + 1)\n}\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const res = try parse(gpa, tokens, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);
    try testing.expectEqual(@as(usize, 0), res.diags.len);
    try testing.expect(!hasErrorNode(res.tree));
    // The pairing check must pass on this clean tree (no assert trip).
    checkBracketPairing(res.tree, tokens, res.diags);
}

test "the adversarial `fn f( ) ) ) {` recovers with imbalance TOLERATED" {
    // The load-bearing case: the token stream is bracket-IMBALANCED (three `)` vs
    // one `(`, one unclosed `{`), yet the parse must not false-trip the pairing
    // invariant because it recovered (>=1 diagnostic / error_node). `parse()` runs
    // the sweep under runtime_safety, so reaching this point without a panic proves
    // the "balanced OR recovered" arm.
    const gpa = testing.allocator;
    const source = "fn f( ) ) ) {\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const res = try parse(gpa, tokens, source);
    defer gpa.free(@constCast(res.diags));
    defer freeTree(gpa, res.tree);

    // It genuinely recovered (this is what licenses the imbalance).
    try testing.expect(res.diags.len >= 1);
    // The raw token stream really IS imbalanced — otherwise the test proves nothing.
    var parens: i32 = 0;
    var braces: i32 = 0;
    for (tokens) |t| switch (t.tag) {
        .l_paren => parens += 1,
        .r_paren => parens -= 1,
        .l_brace => braces += 1,
        .r_brace => braces -= 1,
        else => {},
    };
    try testing.expect(parens != 0 or braces != 0);
    // And the pairing check tolerates it because the parse recovered.
    checkBracketPairing(res.tree, tokens, res.diags);
}
