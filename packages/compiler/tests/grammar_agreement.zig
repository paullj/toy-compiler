//! Token/highlight-level agreement between the compiler lexer and the tree-sitter-toy
//! grammar, over the `tests/corpora/language-features` corpus.
//!
//! For every fixture the compiler considers PARSE-CLEAN (lex-clean AND `Parser.parse`
//! reports no diagnostics), this asserts that:
//!   1. the tree-sitter grammar parses it with ZERO ERROR/MISSING nodes, and
//!   2. the classified non-trivia leaf tokens of the tree-sitter parse match, span for
//!      span and class for class, the classified tokens of the compiler lexer.
//!
//! Agreement is at token granularity, not AST shape: newlines/whitespace/comments and the
//! external `_newline` terminator are trivia on both sides and excluded. Where the two
//! engines cannot align 1:1 the reconciliation is documented at `classifyTag` / `leafClass`
//! — nothing is silently dropped.
//!
//! The suite is non-vacuous: it asserts floors on the number of files and tokens compared,
//! that exactly the 7 known syntactic-error fixtures are parse-tainted, that the grammar
//! rejects the certain ones, and (a live comparator check) that a deliberately-wrong
//! ONE-SIDED classifier makes the comparison FAIL while the true one passes.

const std = @import("std");
const toyc = @import("toy_compiler");
const ts = @import("tree_sitter");
const testing = std.testing;
const Io = std.Io;

extern fn tree_sitter_toy() callconv(.c) *const ts.Language;

const corpus_dir = "tests/corpora/language-features";

// ---- Taxonomy ---------------------------------------------------------------

/// The coarse highlight classes both engines are reduced onto. Editor-only distinctions
/// (type vs function vs variable; boolean vs keyword) are intentionally folded away here —
/// they live only in `queries/highlights.scm`.
const Class = enum { keyword, number, float, string, char, operator, punctuation, identifier };

/// Map a compiler `Tag` onto a `Class`, or null for a trivia/error tag that is excluded
/// from the comparison. Exhaustive: a newly-added `Tag` fails to compile until classified.
fn classifyTag(tag: toyc.Tag) ?Class {
    return switch (tag) {
        // Excluded: statement-terminator, end marker, and the lexer's error tokens (an
        // error token only appears in a parse-tainted file, which is never compared).
        .newline, .eof, .invalid, .string_unterminated, .char_unterminated => null,

        .identifier => .identifier,
        .number => .number,
        .float => .float,
        .string => .string,
        .char_lit => .char,

        // Every keyword, including the boolean literals `true`/`false` (which the compiler
        // lexes as keywords). Highlighting may show `true`/`false` as @boolean, but for
        // agreement they fold to keyword.
        .kw_fn, .kw_return, .kw_if, .kw_else, .kw_while, .kw_true, .kw_false, .kw_struct, .kw_loop, .kw_for, .kw_in, .kw_break, .kw_continue, .kw_enum, .kw_match, .kw_import, .kw_pub, .kw_as, .kw_impl, .kw_mut, .kw_protocol, .kw_has, .kw_type, .kw_extern, .kw_unsafe => .keyword,

        .plus, .minus, .star, .slash, .percent, .eq, .eq_eq, .bang, .bang_eq, .lt, .lt_eq, .gt, .gt_eq, .amp_amp, .pipe_pipe, .pipe_gt, .pipe, .amp, .caret, .tilde, .lt_lt, .gt_gt, .arrow, .question, .dotdot, .colon_eq, .plus_dot, .minus_dot, .star_dot, .slash_dot, .lt_dot, .gt_dot, .le_dot, .ge_dot => .operator,

        .l_paren, .r_paren, .l_brace, .r_brace, .l_bracket, .r_bracket, .comma, .colon, .dot, .at => .punctuation,
    };
}

/// The one-sided negative control: identical to `classifyTag` except numbers are
/// mislabeled as strings. Applied to the COMPILER side only (the tree-sitter side keeps
/// the true classifier), so it produces a genuine asymmetry the comparator must catch.
fn classifyTagWrong(tag: toyc.Tag) ?Class {
    if (tag == .number) return .string;
    return classifyTag(tag);
}

// ---- Token spans ------------------------------------------------------------

const Span = struct { start: u32, end: u32, class: Class };

const ClassifyFn = *const fn (toyc.Tag) ?Class;

/// Re-lex a single anonymous tree-sitter leaf's spelling through the compiler lexer and
/// return its class. Asserts the spelling lexes to exactly one non-trivia token — a drift
/// guard: if the grammar ever emitted a multi-token or unlexable anonymous literal this
/// fails loudly rather than silently agreeing.
fn reLexAnonClass(gpa: std.mem.Allocator, spelling: []const u8) !Class {
    const toks = try toyc.Lexer.tokenize(gpa, spelling);
    var found: ?toyc.Tag = null;
    for (toks) |t| {
        if (t.tag == .eof or t.tag == .newline) continue;
        if (found != null) return error.AnonLeafReLexedToMultipleTokens;
        found = t.tag;
    }
    const tag = found orelse return error.AnonLeafReLexedToNoToken;
    return classifyTag(tag) orelse error.AnonLeafReLexedToTrivia;
}

/// Class of a tree-sitter leaf, or null if it is trivia (a comment, a zero-width terminator,
/// or a missing node). Named literal leaves map by kind — importantly `number` maps by kind
/// so a post-`.` integer index (`x.0`) is a number, matching the compiler's context-sensitive
/// lexer without re-lexing. Anonymous leaves (keywords, operators, punctuation) re-lex.
fn leafClass(gpa: std.mem.Allocator, node: ts.Node) !?Class {
    if (node.isExtra() or node.isMissing()) return null;
    if (node.startByte() == node.endByte()) return null; // zero-width, e.g. `_newline`
    const kind = node.kind();
    if (std.mem.eql(u8, kind, "identifier")) return .identifier;
    if (std.mem.eql(u8, kind, "number")) return .number;
    if (std.mem.eql(u8, kind, "float")) return .float;
    if (std.mem.eql(u8, kind, "string")) return .string;
    if (std.mem.eql(u8, kind, "char")) return .char;
    if (std.mem.eql(u8, kind, "comment")) return null;
    // An anonymous leaf: its kind is its literal spelling.
    return try reLexAnonClass(gpa, kind);
}

/// Collect every classified leaf of the tree, in source order, into `out`.
fn collectLeaves(gpa: std.mem.Allocator, cursor: *ts.TreeCursor, out: *std.ArrayList(Span)) !void {
    const node = cursor.node();
    if (node.childCount() == 0) {
        if (try leafClass(gpa, node)) |c| {
            try out.append(gpa, .{ .start = node.startByte(), .end = node.endByte(), .class = c });
        }
        return;
    }
    if (cursor.gotoFirstChild()) {
        while (true) {
            try collectLeaves(gpa, cursor, out);
            if (!cursor.gotoNextSibling()) break;
        }
        _ = cursor.gotoParent();
    }
}

// ---- Corpus classification --------------------------------------------------

const Scope = enum { in, out_lex, out_parse };

/// Classify a fixture exactly as the compiler does: lex-error tags → out_lex; else a
/// non-empty `Parser.parse` diagnostic set → out_parse; else parse-clean → in. `gpa` is an
/// arena, so the tree and diags it allocates are freed with the arena.
fn scope(gpa: std.mem.Allocator, src: []const u8) !Scope {
    const toks = try toyc.Lexer.tokenize(gpa, src);
    for (toks) |t| switch (t.tag) {
        .invalid, .string_unterminated, .char_unterminated => return .out_lex,
        else => {},
    };
    const res = try toyc.Parser.parse(gpa, toks, src);
    if (res.diags.len != 0) return .out_parse;
    return .in;
}

// ---- Core comparator --------------------------------------------------------

const CompareResult = struct { tokens: usize };

/// The IN-path comparison, parameterized only by which classifier decorates the COMPILER
/// tokens; the tree-sitter side always uses the true `classifyTag`/kind map. Asserts a
/// zero-ERROR tree-sitter parse and span+class equality of the two classified leaf streams.
/// `report` gates the divergence diagnostics: the negative control expects a disagreement,
/// so it passes `report=false` — a stdout write on a PASSING test would break the
/// `zig build test` results pipe.
fn compareFile(gpa: std.mem.Allocator, src: []const u8, compiler_classify: ClassifyFn, report: bool) !CompareResult {
    // Compiler side.
    const toks = try toyc.Lexer.tokenize(gpa, src);
    var a: std.ArrayList(Span) = .empty;
    for (toks) |t| {
        if (compiler_classify(t.tag)) |c| try a.append(gpa, .{ .start = t.start, .end = t.end, .class = c });
    }

    // Tree-sitter side.
    const parser = ts.Parser.create();
    defer parser.destroy();
    try parser.setLanguage(tree_sitter_toy());
    const tree = parser.parseString(src, null) orelse return error.TreeSitterParseReturnedNull;
    defer tree.destroy();
    const root = tree.rootNode();
    if (root.hasError()) return error.GrammarHasErrorNode;

    var b: std.ArrayList(Span) = .empty;
    var cursor = root.walk();
    defer cursor.destroy();
    try collectLeaves(gpa, &cursor, &b);

    if (a.items.len != b.items.len) {
        if (report) {
            std.debug.print("token COUNT mismatch: compiler={d} tree-sitter={d}\n", .{ a.items.len, b.items.len });
            dumpDivergence(src, a.items, b.items);
        }
        return error.GrammarDisagreement;
    }
    for (a.items, b.items, 0..) |ta, tb, i| {
        if (ta.start != tb.start or ta.end != tb.end or ta.class != tb.class) {
            if (report) std.debug.print(
                "token[{d}] mismatch: compiler [{d},{d}) {s} vs tree-sitter [{d},{d}) {s}\n  text: '{s}'\n",
                .{ i, ta.start, ta.end, @tagName(ta.class), tb.start, tb.end, @tagName(tb.class), src[ta.start..@min(ta.end, src.len)] },
            );
            return error.GrammarDisagreement;
        }
    }
    return .{ .tokens = a.items.len };
}

fn dumpDivergence(src: []const u8, a: []const Span, b: []const Span) void {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (a[i].start != b[i].start or a[i].end != b[i].end or a[i].class != b[i].class) {
            std.debug.print("  first divergence at index {d}: '{s}'\n", .{ i, src[a[i].start..@min(a[i].end, src.len)] });
            return;
        }
    }
}

// ---- The corpus agreement test ---------------------------------------------

test "grammar agreement: tree-sitter-toy matches the compiler lexer over the corpus" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var root = try Io.Dir.cwd().openDir(io, corpus_dir, .{ .iterate = true });
    defer root.close(io);
    var walker = try root.walk(gpa);
    defer walker.deinit();

    var files_compared: usize = 0;
    var tokens_compared: usize = 0;
    var out_seen: usize = 0;
    var out_errored: usize = 0;
    // The four syntactic-error fixtures whose rejection is certain: the grammar MUST leave
    // an ERROR/MISSING node on each. (In practice all 7 parse-tainted fixtures error; only
    // these four are hard-asserted so a future, more-permissive edit can't quietly regress
    // them, while the soft floor `out_errored >= 4` tolerates GLR accepting the others.)
    const certain_out = [_][]const u8{
        "errors/stray_semicolon.toy",
        "errors/c_style_for.toy",
        "errors/top_level_expr.toy",
        "errors/unclosed_block.toy",
    };
    var certain_seen: usize = 0;

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".toy")) continue; // skips READMEs, scripts

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();

        const rel = try std.fmt.allocPrint(a, "{s}/{s}", .{ corpus_dir, entry.path });
        const src = try Io.Dir.cwd().readFileAlloc(io, rel, a, .unlimited);

        switch (try scope(a, src)) {
            .in => {
                const r = compareFile(a, src, classifyTag, true) catch |err| {
                    std.debug.print("agreement failure in {s}: {s}\n", .{ entry.path, @errorName(err) });
                    return err;
                };
                files_compared += 1;
                tokens_compared += r.tokens;
            },
            .out_lex, .out_parse => {
                out_seen += 1;
                const parser = ts.Parser.create();
                defer parser.destroy();
                try parser.setLanguage(tree_sitter_toy());
                const tree = parser.parseString(src, null) orelse return error.TreeSitterParseReturnedNull;
                defer tree.destroy();
                const errored = tree.rootNode().hasError();
                if (errored) out_errored += 1;
                for (certain_out) |name| {
                    if (std.mem.endsWith(u8, entry.path, name)) {
                        certain_seen += 1;
                        if (!errored) {
                            std.debug.print("certain-OUT fixture parsed WITHOUT an ERROR node: {s}\n", .{entry.path});
                            return error.CertainOutFixtureAccepted;
                        }
                    }
                }
            },
        }
    }

    // Non-vacuity floors (grounded on measured counts: IN = 304, OUT = 7). Printed only on
    // failure to keep the passing run silent under `zig build test`'s results pipe.
    if (files_compared < 300 or tokens_compared < 15000 or out_seen != 7 or out_errored < 4 or certain_seen != certain_out.len) {
        std.debug.print(
            "agreement summary: files_compared={d} tokens_compared={d} out_seen={d} out_errored={d} certain_seen={d}\n",
            .{ files_compared, tokens_compared, out_seen, out_errored, certain_seen },
        );
        return error.AgreementNonVacuityFloor;
    }
}

// ---- Deterministic control table (hermetic) --------------------------------

test "grammar agreement: deterministic control snippets agree span-for-span" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Every snippet must parse zero-ERROR in the grammar (compareFile asserts it) and its
    // lexer tokens must agree span-for-span. These pin the tricky, load-bearing cases:
    // the post-`.` number-not-float rule (`o.0.0`), base/underscore/exponent literals,
    // dotted-float operators, char escapes, value indexing, labels, and `rawptr`/unit.
    const snippets = [_][]const u8{
        "fn f() -> int {\n  for i in 0..5 { }\n  return 0\n}\n",
        "fn f(o: int) -> int { return o.0.0 }\n", // post-`.` integers, never a float
        "fn f(o: int) -> int { return o.0 }\n",
        "fn f() -> int { return 1_000 }\n",
        "fn f() -> int { return 0x2A }\n",
        "fn f() -> int { return 0o52 }\n",
        "fn f() -> int { return 0b1010 }\n",
        "fn f() -> float { return 3.0 }\n",
        "fn f() -> float { return 1.5e-2 }\n",
        "fn f() -> float { return 1e3 }\n",
        "fn f() -> int { return 255.into() }\n", // `255` number, `.`, then a method call
        "fn f() -> int { return a << b >> c }\n",
        "fn f() -> bool { return x <=. y }\n",
        "fn f() -> char { return 'a' }\n",
        "fn f() -> char { return '\\u{20AC}' }\n",
        "fn f() { unsafe { } }\n",
        "fn f(p: rawptr) -> rawptr { return p }\n",
        "fn f() -> () { return () }\n",
        "import a/b as r\nfn main() -> int { return 0 }\n",
        "fn f() -> int {\n  @outer loop { break @outer 1 }\n}\n",
        "fn f(counts: Map[str, int]) -> int { return counts[\"the\"] }\n",
    };
    for (snippets) |src| {
        _ = compareFile(a, src, classifyTag, true) catch |err| {
            std.debug.print("control snippet failed ({s}): {s}\n", .{ @errorName(err), src });
            return err;
        };
    }
}

// ---- One-sided negative control (liveness of the comparator) ---------------

test "grammar agreement: a one-sided wrong classifier DISAGREES, the true one AGREES" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const src = "fn add(x: int) -> int { return x + 1 }\n";
    // WRONG on the compiler side only (`1` -> string) while the tree-sitter side keeps the
    // true classifier (`1` -> number): a genuine asymmetry the comparator must catch.
    try testing.expectError(error.GrammarDisagreement, compareFile(a, src, classifyTagWrong, false));
    // TRUE classifier on both sides: must agree.
    _ = try compareFile(a, src, classifyTag, true);
}
