//! Transitive content fingerprint of a function — the cache key + parallel unit
//! for M5 incremental codegen.
//!
//! `fingerprint(fn)` is a PURE function of frozen inputs: it returns a u64 that
//! is determined ENTIRELY by what the lowered code would depend on, and NOTHING
//! else. That makes it both the incremental cache key (a hit means "this exact
//! function already lowered") and safe to compute in parallel.
//!
//! The key is TRANSITIVE: it folds (a) a structural walk of the fn BODY, (b) the
//! SIGNATURES (not bodies) of every function this one calls, (c) the layouts of
//! every type it touches, and (d) the compiler+target stamp (supplied externally
//! via the cache `Key`/dir, not mixed here). The (a)+(b) split is the whole point
//! of M5: a callee BODY change must NOT recompile its callers (their fingerprint
//! is unchanged), but a callee SIGNATURE change MUST (their fingerprint flips).
//!
//! The (a) body walk is defined ONCE in `AstWalk`; here `AstWalk.HashVisitor`
//! folds its event stream. The walk's two correctness invariants (which this fold
//! relies on) live with the walk:
//!
//!   * NO ABSOLUTE INDICES, NO SOURCE OFFSETS — node TAGS and leaf token TEXT
//!     only, never an index/offset that shifts when a sibling is edited. [C3]
//!   * ORDER-SENSITIVE, NEVER XOR — children fold in wiring order with arity
//!     sentinels, so `a - b` and `b - a` cannot collide. [C7]

const std = @import("std");
const Token = @import("../ast/Token.zig").Token;
const Ast = @import("../ast/Ast.zig");
const Typecheck = @import("../types.zig");
const AstWalk = @import("AstWalk.zig");

/// The callee identity-and-signature datum (`symbols/Sig.zig`); the `(b)` fold
/// site in `fingerprint` below is the home for why its `kind` is load-bearing.
pub const Sig = @import("../symbols/Sig.zig").Sig;

/// A type this function touches, with an index-free layout descriptor. For
/// a struct, `layout` is the Driver-precomputed byte string (name + per-field
/// name+kind+offset, recursing nested structs, + size + align). Editing a touched
/// struct's fields changes `layout`, flipping every using fn's hash. [C3]
pub const TouchedType = struct {
    kind: Typecheck.Kind,
    layout: []const u8 = &.{},
};

/// Bumped when the in-memory layout encoding of any type changes, so a stale blob
/// from a prior layout is invalidated.
const type_layout_version: u8 = 3;

const seed: u64 = 0x46_50_52_4e; // "FPRN"

/// Compute the transitive content fingerprint of the function at `fn_decl`.
///
///   * `callee_sigs` — the signature of every function the body calls, in the
///     order the body walk encounters the calls. The walk re-derives that order,
///     so the caller supplies the sigs in the SAME order (one per call site).
///   * `touched` — the types this function touches (its node types + param/ret).
pub fn fingerprint(
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    fn_decl: Ast.Index,
    callee_sigs: []const Sig,
    touched: []const TouchedType,
) u64 {
    var h = std.hash.Wyhash.init(seed);

    // (a) structural body walk (tags + leaf text, order-sensitive, index-free).
    // The walk order is defined ONCE in `AstWalk`; `HashVisitor` folds the same
    // bytes this function folded when it owned the walk.
    var hv = AstWalk.HashVisitor{ .h = &h };
    // HashVisitor folds into the Wyhash (no allocation), so its error set is empty
    // and this walk is infallible.
    AstWalk.walk(.{ .tree = tree, .tokens = tokens, .source = source }, fn_decl, &hv) catch unreachable;

    // (b) callee identities + signatures, in walk order. The callee's resolved
    // SymName{kind,name} is folded BEFORE its sig: it is what the `.func` reloc
    // target carries, so a shadow/unshadow that changes the bound symbol (e.g.
    // builtin `print` vs a user `fn print` of the same sig) flips the caller's
    // hash even though the sig is identical. [C6]
    AstWalk.updateU32(&h, @intCast(callee_sigs.len));
    for (callee_sigs) |s| {
        h.update(&[_]u8{@intFromEnum(s.kind)});
        AstWalk.updateLeaf(&h, s.name);
        AstWalk.updateU32(&h, @intCast(s.params.len));
        for (s.params) |p| h.update(&[_]u8{@intFromEnum(p.kind)});
        h.update(&[_]u8{@intFromEnum(s.ret.kind)});
    }

    // (c) touched type layouts. A struct folds its full layout descriptor so an
    // edit to its fields (names/types/offsets/size) flips every using fn's hash.
    AstWalk.updateU32(&h, @intCast(touched.len));
    for (touched) |ty| {
        h.update(&[_]u8{ @intFromEnum(ty.kind), type_layout_version });
        if (ty.kind == .@"struct" or ty.kind == .@"enum") AstWalk.updateLeaf(&h, ty.layout);
    }

    return h.final();
}

// Tests — the cheapest proofs of the miscompile-class ledger (C3, C5, C7).

const testing = std.testing;
const Lexer = @import("../lex.zig");
const Parser = @import("../parse.zig");

const Built = struct {
    tokens: []Token,
    tree: Ast.Tree,
    source: []const u8,

    fn deinit(self: *Built, gpa: std.mem.Allocator) void {
        gpa.free(self.tokens);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
    }

    /// The fn_decl node index of the `idx`-th top-level function (source order).
    fn fnDecl(self: *const Built, idx: usize) Ast.Index {
        const prog = self.tree.nodes[Ast.root(self.tree.nodes).int()];
        return Ast.rangeSlice(self.tree, prog.lhs.int())[idx];
    }
};

fn build(gpa: std.mem.Allocator, source: []const u8) !Built {
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    var diag: ?Parser.Diagnostic = null;
    const tree = (try Parser.parse(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
    return .{ .tokens = tokens, .tree = tree, .source = source };
}

fn fp(b: *const Built, fn_idx: usize) u64 {
    return fingerprint(b.tree, b.tokens, b.source, b.fnDecl(fn_idx), &.{}, &.{});
}

test "position-independent: a fn's hash is the same regardless of sibling order [C3]" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn helper() -> int {\n return 7\n}\nfn main() -> int {\n return 1\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn main() -> int {\n return 1\n}\nfn helper() -> int {\n return 7\n}\n");
    defer b.deinit(gpa);

    // helper is fn 0 in `a`, fn 1 in `b`; identical body → identical hash.
    try testing.expectEqual(fp(&a, 0), fp(&b, 1));
    // main is fn 1 in `a`, fn 0 in `b`.
    try testing.expectEqual(fp(&a, 1), fp(&b, 0));
}

test "swapped operands hash differently: a-b ≠ b-a [C7]" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f(a: int, b: int) -> int {\n return a - b\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f(a: int, b: int) -> int {\n return b - a\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
}

test "arity sentinel: f() ≠ f(0) [C5]" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn g() -> int {\n return 0\n}\nfn f() -> int {\n return g()\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn g(x: int) -> int {\n return x\n}\nfn f() -> int {\n return g(0)\n}\n");
    defer b.deinit(gpa);
    // f calls g() vs g(0): the call arity sentinel must differ.
    try testing.expect(fp(&a, 1) != fp(&b, 1));
}

test "optionality sentinel: bare return ≠ return 0" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f() {\n return\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f() -> int {\n return 0\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
}

test "callee signature folds in: a sig change flips the caller's hash [C2]" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn main() -> int {\n return 1\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const sig_a = [_]Sig{.{ .kind = .user_fn, .name = "g", .params = &.{.int}, .ret = .int }};
    const sig_b = [_]Sig{.{ .kind = .user_fn, .name = "g", .params = &.{ .int, .int }, .ret = .int }};
    const ha = fingerprint(b.tree, b.tokens, b.source, decl, &sig_a, &.{});
    const hb = fingerprint(b.tree, b.tokens, b.source, decl, &sig_b, &.{});
    try testing.expect(ha != hb);
}

test "callee kind folds in: builtin vs user_fn of an identical sig flips the hash [C6]" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn main() {\n print(\"hi\")\n return\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    // Same name, same sig — only the resolved KIND differs (builtin print vs a
    // user `fn print`). The caller's fingerprint MUST differ, else a
    // shadow/unshadow edit is a stale-cache miscompile.
    const builtin_callee = [_]Sig{.{ .kind = .builtin, .name = "print", .params = &.{.str}, .ret = .unit }};
    const user_callee = [_]Sig{.{ .kind = .user_fn, .name = "print", .params = &.{.str}, .ret = .unit }};
    const hb = fingerprint(b.tree, b.tokens, b.source, decl, &builtin_callee, &.{});
    const hu = fingerprint(b.tree, b.tokens, b.source, decl, &user_callee, &.{});
    try testing.expect(hb != hu);
}

test "loop break-value folds in: different break values hash differently" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f() -> int {\n loop { break 1 }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f() -> int {\n loop { break 2 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
}

test "break optionality sentinel: bare break ≠ break 0; break ≠ continue" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f() {\n loop { break }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f() -> int {\n loop { break 0 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
    var c = try build(gpa, "fn f() {\n loop { continue }\n}\n");
    defer c.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&c, 0));
}

test "for-loop var name and bounds fold in" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f() {\n for i in 0..5 { x = i }\n return\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f() {\n for j in 0..5 { x = j }\n return\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0)); // var name differs
    var c = try build(gpa, "fn f() {\n for i in 0..6 { x = i }\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&c, 0)); // hi bound differs
}

test "break label target folds in: break @a ≠ break @b; break ≠ break @a" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f() -> int {\n @a loop { @b loop { break @a 1 } }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f() -> int {\n @a loop { @b loop { break @b 1 } }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
    var c = try build(gpa, "fn f() -> int {\n loop { break 1 }\n}\n");
    defer c.deinit(gpa);
    var d = try build(gpa, "fn f() -> int {\n @a loop { break @a 1 }\n}\n");
    defer d.deinit(gpa);
    try testing.expect(fp(&c, 0) != fp(&d, 0));
}

test "continue label target folds in: continue ≠ continue @a" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f() {\n @o loop { while c { continue } }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f() {\n @o loop { while c { continue @o } }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
}

test "labeled-block value folds in" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f() -> int {\n @c { break @c 1 }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f() -> int {\n @c { break @c 2 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
}

test "editing a sibling fn does not change a labeled fn's hash [C3]" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn g() -> int {\n @l loop { break @l 5 }\n}\nfn h() -> int {\n return 1\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn g() -> int {\n @l loop { break @l 5 }\n}\nfn h() -> int {\n return 99\n}\n");
    defer b.deinit(gpa);
    try testing.expectEqual(fp(&a, 0), fp(&b, 0)); // g unchanged despite h's edit
}

test "identical body + identical sigs hash identically (cache hit)" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn add(a: int, b: int) -> int {\n return a + b\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn add(a: int, b: int) -> int {\n return a + b\n}\n");
    defer b.deinit(gpa);
    try testing.expectEqual(fp(&a, 0), fp(&b, 0));
}

// ---- struct layout fold ----

/// Two struct layouts whose declared fields differ must hash differently when
/// folded as a touched type; an unchanged layout must hash identically (cache
/// hit). The descriptors below mirror what the Driver precomputes.
fn touchedStruct(name: []const u8, layout: []const u8) [1]TouchedType {
    _ = name;
    return .{.{ .kind = .@"struct", .layout = layout }};
}

test "touched struct layout folds in: a field-layout edit flips the hash" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    // Same struct name, different layout bytes (a field added) → different hash.
    const v1 = touchedStruct("Point", "Point\x00x\x00\x02\x00\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00");
    const v2 = touchedStruct("Point", "Point\x00x\x00\x02\x00\x00\x00\x00y\x00\x02\x08\x00\x00\x00\x10\x00\x00\x00\x08\x00\x00\x00");
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v1);
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v2);
    try testing.expect(h1 != h2);
}

test "touched struct layout: identical layout hashes identically (cache hit)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const v = touchedStruct("Point", "Point\x00x\x00\x02\x00\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00");
    const v2 = touchedStruct("Point", "Point\x00x\x00\x02\x00\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00");
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v);
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v2);
    try testing.expectEqual(h1, h2);
}

test "struct decl/literal/field-access spelling folds into the body walk" {
    const gpa = testing.allocator;
    var a = try build(gpa, "struct P { x: int }\nfn f() -> int {\n p := P { x: 1 }\n return p.x\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "struct P { x: int }\nfn f() -> int {\n p := P { x: 2 }\n return p.x\n}\n");
    defer b.deinit(gpa);
    // f is fn 1 (struct P is decl 0); a different literal field value flips it.
    try testing.expect(fp(&a, 1) != fp(&b, 1));
}

test "a fn NOT touching a struct is unaffected by an unrelated touched-struct fold" {
    // Isolation: folding a struct into one fn's touched set must not affect a fn
    // whose own touched set is empty (its hash is computed independently).
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f() -> int {\n return 7\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const v = touchedStruct("Q", "Q\x00n\x00\x02\x00\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00");
    const with = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v);
    const without = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{});
    // They legitimately differ (the touched set is part of the key); the point is
    // each is a pure function of its OWN inputs — recomputing `without` matches.
    try testing.expectEqual(without, fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}));
    try testing.expect(with != without);
}

// ---- enum layout + variant/match spelling fold (M10) ----

test "variant spelling in a fn body folds into the body walk" {
    const gpa = testing.allocator;
    var a = try build(gpa, "enum S { A, B }\nfn f() -> S { return S.A }\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "enum S { A, B }\nfn f() -> S { return S.B }\n");
    defer b.deinit(gpa);
    // f is fn 1; constructing a different variant flips the body walk.
    try testing.expect(fp(&a, 1) != fp(&b, 1));
}

test "qualified N.V differs from inferred .V in the body walk" {
    const gpa = testing.allocator;
    var a = try build(gpa, "enum S { A, B }\nfn f(s: S) -> S { match s { .A -> S.B, .B -> S.A } }\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "enum S { A, B }\nfn f(s: S) -> S { match s { .A -> .B, .B -> .A } }\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 1) != fp(&b, 1));
}

test "match arm spelling folds (different variant binding name)" {
    const gpa = testing.allocator;
    var a = try build(gpa, "enum S { C(int) }\nfn f(s: S) -> int { match s { .C(r) -> r } }\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "enum S { C(int) }\nfn f(s: S) -> int { match s { .C(q) -> q } }\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 1) != fp(&b, 1));
}

test "touched enum layout folds in: a variant-layout edit flips the hash" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const v1 = [1]TouchedType{.{ .kind = .@"enum", .layout = "E\x00tag" }};
    const v2 = [1]TouchedType{.{ .kind = .@"enum", .layout = "E\x00TAG-changed" }};
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v1);
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v2);
    try testing.expect(h1 != h2);
}

test "touched enum layout: identical layout hashes identically (cache hit)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const v = [1]TouchedType{.{ .kind = .@"enum", .layout = "E\x00same" }};
    const v2 = [1]TouchedType{.{ .kind = .@"enum", .layout = "E\x00same" }};
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v);
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v2);
    try testing.expectEqual(h1, h2);
}

test "M11: a match literal-value edit flips the hash" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f(n: int) -> int {\n match n { 0 -> 1, _ -> 2 }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f(n: int) -> int {\n match n { 1 -> 1, _ -> 2 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
}

test "M11: adding a guard flips the hash (guard sentinel)" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f(n: int) -> int {\n match n { 0 -> 1, _ -> 2 }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f(n: int) -> int {\n match n { 0 if n > 0 -> 1, _ -> 2 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
}

test "M11: reordering or-pattern alternatives flips the hash (order-sensitive)" {
    const gpa = testing.allocator;
    var a = try build(gpa, "enum E { A, B, C }\nfn f(e: E) -> int {\n match e { .A | .B -> 1, .C -> 2 }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "enum E { A, B, C }\nfn f(e: E) -> int {\n match e { .B | .A -> 1, .C -> 2 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 1) != fp(&b, 1));
}

test "M11: adding a nested sub-pattern flips the hash" {
    const gpa = testing.allocator;
    var a = try build(gpa, "enum E { C(int), N }\nfn f(e: E) -> int {\n match e { .C(r) -> r, .N -> 0 }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "enum E { C(int), N }\nfn f(e: E) -> int {\n match e { .C(0) -> 1, .C(r) -> r, .N -> 0 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 1) != fp(&b, 1));
}
