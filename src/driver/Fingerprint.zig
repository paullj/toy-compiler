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
//! Two correctness invariants the walk must never violate:
//!
//!   * NO ABSOLUTE INDICES, NO SOURCE OFFSETS. The walk hashes node TAGS and leaf
//!     token TEXT, never a node index, token index, or source byte offset. Those
//!     all shift when an unrelated sibling function is edited; an unchanged fn
//!     must keep an identical hash so it stays a cache hit. [C3]
//!   * ORDER-SENSITIVE, NEVER XOR. Children are folded with a streaming Wyhash in
//!     their exact wiring order, with arity/optionality sentinels. XOR (or +) is
//!     commutative, so `a - b` and `b - a` — or swapped comparison operands —
//!     would collide and miscompile. [C7]

const std = @import("std");
const Token = @import("../ast/Token.zig").Token;
const Ast = @import("../ast/Ast.zig");
const Typecheck = @import("../types.zig");

/// A callee's identity-and-signature, folded into a caller's fingerprint so a
/// signature OR symbol-identity change (but not a body change) invalidates the
/// caller. [C1][C2]
///
/// The callee's resolved KIND (user_fn vs builtin vs import) is load-bearing,
/// not just its sig: a user `fn print(s: str)` has the SAME sig {[str],void} as
/// the builtin `print`, but lowers to a DIFFERENT `.func` reloc target
/// (`{user_fn,"print"}` vs `{builtin,"print"}`). Folding only the sig would let
/// a shadow-and-unshadow edit keep an identical fingerprint while the call binds
/// to a different symbol — a stale-cache miscompile. So we fold the full
/// `SymName{kind,name}` (which IS what enters the lowered bytes) too. [Cx]
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
const type_layout_version: u8 = 2;

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
    walk(&h, tree, tokens, source, fn_decl);

    // (b) callee identities + signatures, in walk order. The callee's resolved
    // SymName{kind,name} is folded BEFORE its sig: it is what the `.func` reloc
    // target carries, so a shadow/unshadow that changes the bound symbol (e.g.
    // builtin `print` vs a user `fn print` of the same sig) flips the caller's
    // hash even though the sig is identical. [Cx]
    updateU32(&h, @intCast(callee_sigs.len));
    for (callee_sigs) |s| {
        h.update(&[_]u8{@intFromEnum(s.kind)});
        updateLeaf(&h, s.name);
        updateU32(&h, @intCast(s.params.len));
        for (s.params) |p| h.update(&[_]u8{@intFromEnum(p.kind)});
        h.update(&[_]u8{@intFromEnum(s.ret.kind)});
    }

    // (c) touched type layouts. A struct folds its full layout descriptor so an
    // edit to its fields (names/types/offsets/size) flips every using fn's hash.
    updateU32(&h, @intCast(touched.len));
    for (touched) |ty| {
        h.update(&[_]u8{ @intFromEnum(ty.kind), type_layout_version });
        if (ty.kind == .@"struct") updateLeaf(&h, ty.layout);
    }

    return h.final();
}

fn updateU32(h: *std.hash.Wyhash, v: u32) void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    h.update(&buf);
}

/// Length-prefixed leaf text, so `"ab"+"c"` cannot collide with `"a"+"bc"`.
fn updateLeaf(h: *std.hash.Wyhash, text: []const u8) void {
    updateU32(h, @intCast(text.len));
    h.update(text);
}

/// Walk the subtree at `idx`, folding tag + leaf text + ordered children. Mirrors
/// the `Ast` node wiring exactly (and the statement/expr sets the Codegen lowers).
fn walk(h: *std.hash.Wyhash, tree: Ast.Tree, tokens: []const Token, source: []const u8, idx: Ast.Index) void {
    if (idx == Ast.none) return;
    const n = tree.nodes[idx];
    h.update(&[_]u8{@intFromEnum(n.tag)});
    const leaf = tokens[n.main_token].text(source);
    switch (n.tag) {
        // Leaf-bearing literals/identifiers: the spelling IS the content.
        .literal_number, .literal_string, .literal_bool, .identifier => updateLeaf(h, leaf),

        .unary => {
            updateLeaf(h, leaf); // operator text
            walk(h, tree, tokens, source, n.lhs);
        },
        .binary => {
            updateLeaf(h, leaf); // operator text
            walk(h, tree, tokens, source, n.lhs); // lhs THEN rhs: a-b ≠ b-a [C7]
            walk(h, tree, tokens, source, n.rhs);
        },
        .call => {
            // The callee leaf is folded by recursing the lhs identifier; the arity
            // sentinel distinguishes f() from f(0). [C5]
            walk(h, tree, tokens, source, n.lhs);
            const args = Ast.rangeSlice(tree, n.rhs);
            updateU32(h, @intCast(args.len));
            for (args) |a| walk(h, tree, tokens, source, a);
        },
        .var_decl => {
            updateLeaf(h, leaf); // bound name
            walk(h, tree, tokens, source, n.lhs); // initializer
        },
        .assign => {
            walk(h, tree, tokens, source, n.lhs); // target
            walk(h, tree, tokens, source, n.rhs); // value
        },
        .return_stmt => {
            h.update(&[_]u8{@intFromBool(n.lhs != Ast.none)}); // bare return ≠ return v
            if (n.lhs != Ast.none) walk(h, tree, tokens, source, n.lhs);
        },
        .expr_stmt => walk(h, tree, tokens, source, n.lhs),
        .block => {
            const stmts = Ast.rangeSlice(tree, n.lhs);
            updateU32(h, @intCast(stmts.len));
            for (stmts) |s| walk(h, tree, tokens, source, s);
        },
        .param => {
            updateLeaf(h, leaf); // param name
            walk(h, tree, tokens, source, n.lhs); // type-ref identifier
        },
        .fn_decl => {
            updateLeaf(h, leaf); // fn name
            const proto = Ast.protoAt(tree, n.lhs);
            updateU32(h, @intCast(proto.params.len));
            for (proto.params) |p| walk(h, tree, tokens, source, p);
            h.update(&[_]u8{@intFromBool(proto.ret_type != Ast.none)});
            if (proto.ret_type != Ast.none) walk(h, tree, tokens, source, proto.ret_type);
            walk(h, tree, tokens, source, n.rhs); // body block
        },
        .while_stmt => {
            walk(h, tree, tokens, source, n.lhs); // cond
            walk(h, tree, tokens, source, n.rhs); // body
        },
        .if_stmt => {
            walk(h, tree, tokens, source, n.lhs); // cond
            const head = Ast.ifHeaderAt(tree, n.rhs);
            walk(h, tree, tokens, source, head.then_block);
            h.update(&[_]u8{@intFromBool(head.else_node != Ast.none)});
            if (head.else_node != Ast.none) walk(h, tree, tokens, source, head.else_node);
        },
        .program => {}, // never a fingerprint root
        // Zero-sized leaf: the tag byte (folded before the switch) IS its content,
        // so it is structurally distinct from any other node. Reached as a value
        // literal and as a `()` type-ref (under param/fn_decl ret).
        .literal_unit => {},
        .loop_expr => walk(h, tree, tokens, source, n.lhs), // body
        .for_stmt => {
            updateLeaf(h, leaf); // loop-var name
            const head = Ast.forHeaderAt(tree, n.rhs);
            walk(h, tree, tokens, source, head.lo);
            walk(h, tree, tokens, source, head.hi);
            walk(h, tree, tokens, source, n.lhs); // body
        },
        .break_stmt => {
            // The label target identity is load-bearing (it picks the join), so
            // fold its TEXT (not a token index — index-free [C3]).
            h.update(&[_]u8{@intFromBool(n.rhs != Ast.none)}); // bare ≠ @label
            if (n.rhs != Ast.none) updateLeaf(h, tokens[n.rhs].text(source));
            h.update(&[_]u8{@intFromBool(n.lhs != Ast.none)}); // bare break ≠ break v
            if (n.lhs != Ast.none) walk(h, tree, tokens, source, n.lhs);
        },
        .continue_stmt => {
            h.update(&[_]u8{@intFromBool(n.rhs != Ast.none)}); // bare ≠ @label
            if (n.rhs != Ast.none) updateLeaf(h, tokens[n.rhs].text(source));
        },
        // The label NAME at the def-site is folded so a break-target rename stays
        // consistent with the wrapper; the inner construct is then walked.
        .labeled => {
            updateLeaf(h, leaf); // label name
            walk(h, tree, tokens, source, n.lhs);
        },
        // Structs: the decl folds the type's SPELLING (name + fields); the
        // resolved layout enters via the touched-types (c) component. A struct
        // literal folds its type-name + each field-init; a field access folds the
        // field-name + receiver. (Construction/access SPELLING is here; LAYOUT is (c).)
        .struct_decl => {
            updateLeaf(h, leaf); // struct name
            const fields = Ast.rangeSlice(tree, n.lhs);
            updateU32(h, @intCast(fields.len));
            for (fields) |f| walk(h, tree, tokens, source, f);
        },
        .struct_init => {
            walk(h, tree, tokens, source, n.lhs); // type-name identifier
            const inits = Ast.rangeSlice(tree, n.rhs);
            updateU32(h, @intCast(inits.len));
            for (inits) |fi| walk(h, tree, tokens, source, fi);
        },
        .field_init => {
            updateLeaf(h, leaf); // field name
            walk(h, tree, tokens, source, n.lhs); // value
        },
        .field_access => {
            updateLeaf(h, leaf); // field name
            walk(h, tree, tokens, source, n.lhs); // receiver
        },
    }
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
        const prog = self.tree.nodes[Ast.root(self.tree.nodes)];
        return Ast.rangeSlice(self.tree, prog.lhs)[idx];
    }
};

fn build(gpa: std.mem.Allocator, source: []const u8) !Built {
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    var diag: ?Parser.Diagnostic = null;
    const tree = (try Parser.parse(gpa, tokens, &diag)) orelse return error.UnexpectedParseFailure;
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

test "callee kind folds in: builtin vs user_fn of an identical sig flips the hash [Cx]" {
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
