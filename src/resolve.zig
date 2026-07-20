//! Name-resolution shared types (the `Resolution` value re-export).
//!
//! The resolution PASS itself lives in `resolve_graph.zig`: a lone source file is
//! resolved as the trivial one-module graph (`Graph.single` → `resolveGraph`), so
//! there is exactly ONE resolver, whose `ResolveGraph.GraphResult` is consumed
//! whole everywhere (the driver `FileResult`, `lower`). This module just owns the
//! peer type those readers spell as `Resolve.Resolution`.
//!
//! Resolution is lexical and index-based: a result's `resolutions` is a
//! `[]Resolution` parallel to the node array (only identifier-expression nodes
//! carry a value; others stay `.unresolved`). The pass never stops at the first
//! error.

const std = @import("std");
const Token = @import("ast/Token.zig").Token;
const Ast = @import("ast/Ast.zig");

/// What an identifier expression refers to, once resolved. Definition lives in
/// the `symbols/` peer data module; re-exported so `Resolve.Resolution` keeps
/// working for downstream consumers.
pub const Resolution = @import("symbols/Resolution.zig").Resolution;

const testing = std.testing;
const Lexer = @import("lex.zig");
const Parser = @import("parse.zig");
const Graph = @import("driver/Graph.zig");
const ResolveGraph = @import("resolve_graph.zig");

const Parsed = struct {
    tokens: []Token,
    tree: Ast.Tree,
    source: []const u8,

    fn deinit(self: *Parsed, gpa: std.mem.Allocator) void {
        gpa.free(self.tokens);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
    }
};

fn parseSource(gpa: std.mem.Allocator, source: []const u8) !Parsed {
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, source);
    return .{ .tokens = tokens, .tree = tree, .source = source };
}

/// Resolve an already-parsed source as the trivial one-module graph (the ONE
/// resolver). Returns the whole-graph result; for a single source `diags` is the
/// flat diagnostic stream and `resolutions[0]` is the node-parallel array. The
/// graph BORROWS `parsed` (which the caller keeps alive), and the result owns its
/// own arrays, so the graph is torn down immediately. Free the result with
/// `res.deinit(gpa)` (GraphResult.deinit).
fn resolveParsed(gpa: std.mem.Allocator, parsed: Parsed) !ResolveGraph.GraphResult {
    var g = try Graph.single(gpa, "main", "", parsed.source, parsed.tokens, parsed.tree.nodes, parsed.tree.extra, parsed.tree.pub_bits);
    defer g.deinit(gpa);
    return ResolveGraph.resolveGraph(gpa, &g);
}

/// Resolve a source and return the diagnostic count (and free everything).
fn resolveDiagCount(source: []const u8) !usize {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, source);
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    return res.diags.len;
}

test "clean program resolves with zero diagnostics" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        "fn add(a: int, b: int) -> int {\n return a + b\n}\n",
    ));
}

test "forward function reference resolves" {
    // main calls add/neg declared below it.
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        \\fn main() {
        \\ x := add(1, 2)
        \\ _y := neg(x)
        \\ return
        \\}
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn neg(x: int) -> int { return -x }
        \\
    ));
}

test "undeclared identifier yields exactly one diagnostic" {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, "fn f() {\n _y := x\n return\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'x'", res.diags[0].message);
}

test "a single root undeclared name yields exactly one diagnostic (report-once)" {
    // Mirrors the report-once smoke: one undeclared name → one diagnostic, no
    // cascade through the `return` that consumes its poison.
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, "fn main() -> int {\n return nope\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expect(std.mem.startsWith(u8, res.diags[0].message, "undeclared identifier 'nope'"));
}

test "a close typo of an in-scope local yields a did-you-mean hint" {
    const gpa = testing.allocator;
    // `count` is bound (and used); `cont` is undeclared and one deletion away → hint.
    var parsed = try parseSource(gpa, "fn f() -> int {\n count := 1\n return count + cont\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'cont'; did you mean 'count'?", res.diags[0].message);
}

test "a close typo of a fn name yields a did-you-mean hint" {
    const gpa = testing.allocator;
    // `panic` is always in the fn table; `pnic` is a transposition/deletion away.
    var parsed = try parseSource(gpa, "fn f() {\n pnic(\"hi\")\n return\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'pnic'; did you mean 'panic'?", res.diags[0].message);
}

test "a close typo of a SHADOWED local still yields a did-you-mean hint" {
    const gpa = testing.allocator;
    // `count` is declared in an outer scope and re-declared (shadowed) in an inner
    // block, so candidateIter yields the string `count` twice. That duplicate must
    // NOT be treated as an ambiguous tie — the hint must still fire.
    var parsed = try parseSource(gpa, "fn f() -> int {\n count := 1\n count = count + 1\n {\n count := 2\n return count + cont\n }\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'cont'; did you mean 'count'?", res.diags[0].message);
}

test "a close typo of a fn name shadowed by a local still yields a hint" {
    const gpa = testing.allocator;
    // A local named `panic` shadows the built-in fn `panic`: the string `panic` is
    // yielded by both the local scope and the fn table. The duplicate must not
    // suppress the hint for a typo of it.
    var parsed = try parseSource(gpa, "fn f() -> int {\n panic := 1\n return pani + panic\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'pani'; did you mean 'panic'?", res.diags[0].message);
}

test "a distant undeclared name yields NO hint" {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, "fn f() -> int {\n _count := 1\n return zzzzzz\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'zzzzzz'", res.diags[0].message);
}

test "an ambiguous near-miss (tie) yields NO hint" {
    const gpa = testing.allocator;
    // `cat` and `bar` are each distance 1 from `bat` → strict-unique-winner fails → no hint.
    var parsed = try parseSource(gpa, "fn f() -> int {\n cat := 1\n bar := 2\n return bat + cat + bar\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'bat'", res.diags[0].message);
}

test "an unimported std type in expression position hints its import" {
    const gpa = testing.allocator;
    // `Map` binds nothing without the import; the type-args are not descended, so
    // there is exactly one diagnostic — the import hint on `Map`.
    var parsed = try parseSource(gpa, "fn main() -> int {\n _m := Map[int, int].new()\n return 0\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'Map'; add 'import std/map'", res.diags[0].message);
}

test "an unimported print uses the corrected spelling and names its import" {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, "fn main() -> int {\n print(\"hi\")\n return 0\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'print'; use 'io.print' and add 'import std/io'", res.diags[0].message);
}

test "a typo of a std name gets NO import hint (near-miss owns typos)" {
    const gpa = testing.allocator;
    // `Mapp` is one edit from stdlib `Map` but is NOT an exact key, and nothing named
    // `Map` is in scope, so neither an import hint nor a near-miss fires → bare message.
    var parsed = try parseSource(gpa, "fn main() -> int {\n _x := Mapp\n return 0\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'Mapp'", res.diags[0].message);
}

test "duplicate parameter is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "fn f(a: int, a: int) {\n return\n}\n",
    ));
}

test "same-scope := redeclaration is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "fn f() {\n x := 1\n x := 2\n return x\n}\n",
    ));
}

test "x := x with no outer is undeclared" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "fn f() {\n x := x\n return x\n}\n",
    ));
}

test "x := x with an outer in scope resolves to the outer" {
    // Inner var_decl initializer must see the parameter `x`, not itself.
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, "fn f(x: int) {\n x := x\n return x\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    // The initializer `x` (an identifier expr) must resolve to the parameter's
    // slot 0; the new `:=` local takes slot 1. No diagnostics either way.
    try testing.expectEqual(@as(usize, 0), res.diags.len);
    // Find the var_decl node and assert its initializer resolved to slot 0.
    var found = false;
    for (parsed.tree.nodes, 0..) |n, i| {
        if (n.tag == .var_decl) {
            const init_res = res.resolutions[0][n.lhs.int()];
            try testing.expect(init_res == .local);
            try testing.expectEqual(@as(u32, 0), init_res.local);
            // The decl itself binds a fresh slot (1).
            const decl_res = res.resolutions[0][i];
            try testing.expect(decl_res == .local);
            try testing.expectEqual(@as(u32, 1), decl_res.local);
            found = true;
        }
    }
    try testing.expect(found);
}

test "if/while conditions and arm bodies resolve cleanly" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        \\fn f(n: int) -> int {
        \\ if n < 0 {
        \\  x := 1
        \\  return x
        \\ } else {
        \\  y := 2
        \\  return y
        \\ }
        \\ while n > 0 { n = n - 1 }
        \\ return n
        \\}
        \\
    ));
}

test "undeclared name in an else-if condition is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f(n: int) {
        \\ if n > 0 {} else if bad < 1 {}
        \\ return
        \\}
        \\
    ));
}

test "assignment to an undeclared name is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "fn f() {\n x = 1\n return\n}\n",
    ));
}

test "for loop binds its variable and resolves its body cleanly" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        \\fn f() -> int {
        \\ s := 0
        \\ for i in 0..5 { s = s + i }
        \\ return s
        \\}
        \\
    ));
}

test "for loop variable is in scope only inside the body" {
    // `i` referenced after the loop is undeclared.
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() -> int {
        \\ for i in 0..5 { _x := i }
        \\ return i
        \\}
        \\
    ));
}

test "undeclared range bound is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() {
        \\ for i in 0..n { _x := i }
        \\ return
        \\}
        \\
    ));
}

test "loop body and break value resolve cleanly" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        \\fn f() -> int {
        \\ i := 0
        \\ loop {
        \\  if i > 3 { break i }
        \\  i = i + 1
        \\ }
        \\}
        \\
    ));
}

test "labeled loop and break @label resolve cleanly and target the named construct" {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa,
        \\fn f() -> int {
        \\ @outer loop {
        \\  @inner loop { break @outer 1 }
        \\ }
        \\}
        \\
    );
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), res.diags.len);
    // The `@outer` labeled wrapper's construct is the FIRST loop_expr (outermost);
    // emitted last among the two loop_exprs (children precede parents), so it has
    // the larger node index. The break must resolve to THAT node, not the inner.
    var outer_loop: Ast.Index = Ast.none;
    for (parsed.tree.nodes, 0..) |n, i| {
        if (n.tag == .loop_expr) outer_loop = Ast.Index.from(@intCast(i)); // last wins → outermost
    }
    var found = false;
    for (parsed.tree.nodes, 0..) |n, i| {
        if (n.tag == .break_stmt) {
            try testing.expect(res.resolutions[0][i] == .label);
            try testing.expectEqual(outer_loop, res.resolutions[0][i].label);
            found = true;
        }
    }
    try testing.expect(found);
}

test "break to an undefined label is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() -> int {
        \\ @outer loop { break @nope 1 }
        \\}
        \\
    ));
}

test "continue to an undefined label is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() {
        \\ @outer loop { continue @nope }
        \\}
        \\
    ));
}

test "label out of scope (after the construct) is reported" {
    // `@a` is in scope only inside its loop body; a break after it sees nothing.
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() {
        \\ @a loop { break }
        \\ @b while true { break @a }
        \\ return
        \\}
        \\
    ));
}

test "duplicate label in scope is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() {
        \\ @x loop { @x loop { break } }
        \\ return
        \\}
        \\
    ));
}

test "labeled for binds its loop variable on the inner for_stmt" {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa,
        \\fn f() -> int {
        \\ s := 0
        \\ @l for i in 0..5 { s = s + i }
        \\ return s
        \\}
        \\
    );
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), res.diags.len);
    // The inner for_stmt node carries the loop-var slot resolution.
    var found = false;
    for (parsed.tree.nodes, 0..) |n, i| {
        if (n.tag == .for_stmt) {
            try testing.expect(res.resolutions[0][i] == .local);
            found = true;
        }
    }
    try testing.expect(found);
}

test "assignment to a function name is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "fn g() { return }\nfn f() {\n g = 1\n return\n}\n",
    ));
}

test "struct construction resolves clean (type name is not flagged)" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P { x: 1 }\n return p.x }\n",
    ));
}

test "field punning resolves the in-scope locals" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        "struct P { x: int, y: int }\nfn f() -> int { x := 1\n y := 2\n p := P { x, y }\n return p.x }\n",
    ));
}

test "punning an out-of-scope local is reported as undeclared" {
    // `y` is never declared; the punning-synthesized identifier must flag it.
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "struct P { x: int, y: int }\nfn f() -> int { x := 1\n p := P { x, y }\n return p.x }\n",
    ));
}

test "field place-store resolves its receiver local" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        "struct P { x: int }\nfn f() { p := P { x: 1 }\n p.x = 5\n return }\n",
    ));
}

test "a struct-named callee resolves quietly (no double diagnostic with Typecheck)" {
    // `P(1)` is positional construction (Typecheck rejects it). Resolve must NOT
    // also flag `P` as undeclared — so the resolve diagnostic count is 0.
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P(1)\n return p.x }\n",
    ));
}

test "forward call to a function resolves the callee" {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, "fn f() {\n g()\n return\n}\nfn g() { return }\n");
    defer parsed.deinit(gpa);
    var res = try resolveParsed(gpa, parsed);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), res.diags.len);
    // The call's callee identifier resolves to a function.
    var found = false;
    for (parsed.tree.nodes) |n| {
        if (n.tag == .call) {
            const callee_res = res.resolutions[0][n.lhs.int()];
            try testing.expect(callee_res == .func);
            found = true;
        }
    }
    try testing.expect(found);
}
