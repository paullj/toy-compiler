//! Transitive content fingerprint of a function — the cache key + parallel unit
//! for incremental codegen.
//!
//! `fingerprint(fn)` is a PURE function of frozen inputs: it returns a u64 that
//! is determined ENTIRELY by what the lowered code would depend on, and NOTHING
//! else. That makes it both the incremental cache key (a hit means "this exact
//! function already lowered") and safe to compute in parallel.
//!
//! The key is TRANSITIVE: it folds (a) a structural walk of the fn BODY, (b) the
//! SIGNATURES (not bodies) of every function this one calls, (c) the layouts of
//! every type it touches, and (d) the compiler+target stamp (supplied externally
//! via the cache `Key`/dir, not mixed here). The (a)+(b) split is the whole point:
//! a callee BODY change must NOT recompile its callers (their fingerprint
//! is unchanged), but a callee SIGNATURE change MUST (their fingerprint flips).
//!
//! The (a) body walk is defined ONCE in `AstWalk`; here `AstWalk.HashVisitor`
//! folds its event stream. The walk's two correctness invariants (which this fold
//! relies on) live with the walk:
//!
//!   * NO ABSOLUTE INDICES, NO SOURCE OFFSETS — node TAGS and leaf token TEXT
//!     only, never an index/offset that shifts when a sibling is edited.
//!   * ORDER-SENSITIVE, NEVER XOR — children fold in wiring order with arity
//!     sentinels, so `a - b` and `b - a` cannot collide.

const std = @import("std");
const Token = @import("../ast/Token.zig").Token;
const Ast = @import("../ast/Ast.zig");
const Typecheck = @import("../types.zig");
const AstWalk = @import("AstWalk.zig");
const Derive = @import("../symbols/Derive.zig");

/// The callee identity-and-signature datum (`symbols/Sig.zig`); the `(b)` fold
/// site in `fingerprint` below is the home for why its `kind` is load-bearing.
pub const Sig = @import("../symbols/Sig.zig").Sig;

/// A type this function touches, with an index-free layout descriptor. For
/// a struct, `layout` is the Driver-precomputed byte string (name + per-field
/// name+kind+offset, recursing nested structs, + size + align). Editing a touched
/// struct's fields changes `layout`, flipping every using fn's hash.
pub const TouchedType = struct {
    kind: Typecheck.Kind,
    layout: []const u8 = &.{},
};

/// A resolved `[T has P]` bound witnessing an instance's conformance (M13), folded
/// as the ordered `(e)` component below. `conform` is the conforming type's index-free
/// layout descriptor (built by the SAME `appendTouched` the type-arg fold uses);
/// `witness_syms` are the witnessing method SymNames in protocol-declared order — NOT
/// the impl body-fp (that would over-invalidate). Toggling a sibling conformance to a
/// different witness / conforming layout flips this datum, and thus the dependent
/// instance's fingerprint, deterministically.
pub const ResolvedConformance = struct {
    protocol_name: []const u8,
    conform: TouchedType,
    witness_syms: []const []const u8,
    /// The protocol type-args of this conformance (M14, `Into[int]` -> `[int]`), each an
    /// index-free layout descriptor built by the SAME `appendTouched` the type-arg fold
    /// uses. Folded ordered (never XOR) into the (e) component so an `Into[int]` ->
    /// `Into[bool]` edit flips the dependent instance's fingerprint (stale-witness guard).
    /// Empty for a non-generic protocol -> that conformance folds byte-identically to M13.
    protocol_args: []const TouchedType = &.{},
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
    /// (d) The concrete type-args of a monomorphized instance (M2), in
    /// generic-param order — each an index-free layout descriptor built by
    /// `appendTouched`. Empty for every non-generic (non-instance) fn: the fold is
    /// then SKIPPED ENTIRELY, so an existing fn's fingerprint is byte-identical and
    /// the warm cache is preserved. Two instances of one template get distinct
    /// fingerprints even when the template body/sig folds are identical.
    type_args: []const TouchedType,
    /// (e) The resolved `[T has P]` bound conformances of a monomorphized instance
    /// (M13), in generic-param order — each folding the witnessing impl's STRUCTURAL
    /// identity (protocol name + conforming-type layout + witness SymNames), NOT its
    /// body-fp. Empty for every non-bounded (fn/instance): the fold is then SKIPPED
    /// ENTIRELY, so every existing fp is byte-identical and the warm cache is
    /// preserved. This is the incremental-correctness core the M15+ operator branch
    /// depends on: toggling a sibling-module conformance flips exactly the dependent
    /// monomorphizations' keys.
    conformances: []const ResolvedConformance,
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
    // hash even though the sig is identical.
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

    // (d) monomorphization type-args. Folded ONLY when present (an instance), so a
    // non-generic fn's fold is byte-identical — the warm cache survives M2. Ordered
    // (a `[T,U]` reorder flips it) with a full per-arg layout descriptor so `id[int]`
    // and `id[Point]` diverge and a struct-layout edit to a type-arg invalidates
    // exactly the dependent instance. Never XOR, never the interned index.
    if (type_args.len > 0) {
        AstWalk.updateU32(&h, @intCast(type_args.len));
        for (type_args) |ty| {
            h.update(&[_]u8{ @intFromEnum(ty.kind), type_layout_version });
            if (ty.kind == .@"struct" or ty.kind == .@"enum") AstWalk.updateLeaf(&h, ty.layout);
        }
    }

    // (e) resolved bound conformances (M13). Folded ONLY when present (a bounded
    // instance), so every non-bounded fp is byte-identical (warm cache preserved).
    // ORDERED (count sentinel + per-conformance leaves in generic-param order, then
    // witness SymNames in protocol-declared order); never XOR, never a build-local
    // index — only the source-borrowed protocol NAME, the conforming type's layout
    // descriptor (mirroring (c)/(d)), and the witnessing method mangled SymNames. This
    // is the stale-cache-miscompile guard: a conformance edit (witness or conforming
    // layout) flips exactly the dependent monomorphization's key.
    if (conformances.len > 0) {
        AstWalk.updateU32(&h, @intCast(conformances.len));
        for (conformances) |rc| {
            AstWalk.updateLeaf(&h, rc.protocol_name);
            h.update(&[_]u8{ @intFromEnum(rc.conform.kind), type_layout_version });
            if (rc.conform.kind == .@"struct" or rc.conform.kind == .@"enum") AstWalk.updateLeaf(&h, rc.conform.layout);
            AstWalk.updateU32(&h, @intCast(rc.witness_syms.len));
            for (rc.witness_syms) |w| AstWalk.updateLeaf(&h, w);
            // (M14) The conformance's protocol type-args, ordered + structural (mirroring
            // (d)); gated `len > 0` so a non-generic-protocol conformance folds
            // byte-identically to M13 (warm cache preserved). NEVER XOR / interned index.
            if (rc.protocol_args.len > 0) {
                AstWalk.updateU32(&h, @intCast(rc.protocol_args.len));
                for (rc.protocol_args) |pa| {
                    h.update(&[_]u8{ @intFromEnum(pa.kind), type_layout_version });
                    if (pa.kind == .@"struct" or pa.kind == .@"enum") AstWalk.updateLeaf(&h, pa.layout);
                }
            }
        }
    }

    return h.final();
}

/// The distinct seed for a SOURCE-LESS derive unit (M18). Different from `seed` so a
/// derive fingerprint can never alias a real fn's body-walk fingerprint even if the
/// folded bytes happened to coincide.
const derive_seed: u64 = 0x44_52_56_46; // "DRVF"

/// The NON-AST fingerprint of a source-less auto-derive codegen unit (M18). There is
/// no `fn_decl` to walk, so the key is built ENTIRELY from the recipe: a distinct
/// derive seed + the derive kind + the protocol name + the conforming type's
/// index-free layout descriptor (the SAME encoding the (c)/(d) touched fold uses) +
/// each resolved field-witness (its `FieldWitness` tag + the witness SymName), ordered
/// (never XOR). So a field-layout edit flips it (via `conform.layout`), and a nested
/// field gaining/losing an explicit impl flips it (via the witness tag/name change) —
/// the stale-cache-miscompile guards a real fn gets from (a)/(c)/(e).
pub fn deriveFingerprint(
    protocol_name: []const u8,
    kind: Derive.Kind,
    conform: TouchedType,
    field_witnesses: []const Derive.FieldWitness,
) u64 {
    var h = std.hash.Wyhash.init(derive_seed);
    h.update(&[_]u8{ @intFromEnum(kind), type_layout_version });
    AstWalk.updateLeaf(&h, protocol_name);
    // The conforming type's layout descriptor (mirrors (c)/(d)); a struct/enum folds
    // its full index-free layout so a field-layout edit flips the unit's key.
    h.update(&[_]u8{@intFromEnum(conform.kind)});
    if (conform.kind == .@"struct" or conform.kind == .@"enum") AstWalk.updateLeaf(&h, conform.layout);
    // The resolved per-field witnesses, ORDERED (count sentinel + per-field tag + the
    // witness SymName). A nested field gaining an explicit impl changes its `FieldWitness`
    // (tag and/or name), flipping the key — so the override never serves a stale blob.
    AstWalk.updateU32(&h, @intCast(field_witnesses.len));
    for (field_witnesses) |fw| {
        h.update(&[_]u8{@intFromEnum(fw)});
        switch (fw) {
            .inline_kind => {},
            .eq_call => |n| AstWalk.updateLeaf(&h, n),
            .cmp_eq => |n| AstWalk.updateLeaf(&h, n),
            .cmp_call => |n| AstWalk.updateLeaf(&h, n),
            .hash_call => |n| AstWalk.updateLeaf(&h, n),
            .display_call => |n| AstWalk.updateLeaf(&h, n),
        }
    }
    return h.final();
}

// Tests — the cheapest proofs of the miscompile-class ledger.

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
    const tree = try Parser.expectTree(gpa, tokens, source);
    return .{ .tokens = tokens, .tree = tree, .source = source };
}

fn fp(b: *const Built, fn_idx: usize) u64 {
    return fingerprint(b.tree, b.tokens, b.source, b.fnDecl(fn_idx), &.{}, &.{}, &.{}, &.{});
}

test "position-independent: a fn's hash is the same regardless of sibling order" {
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

test "swapped operands hash differently: a-b ≠ b-a" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f(a: int, b: int) -> int {\n return a - b\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f(a: int, b: int) -> int {\n return b - a\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
}

test "arity sentinel: f() ≠ f(0)" {
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

test "callee signature folds in: a sig change flips the caller's hash" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn main() -> int {\n return 1\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const sig_a = [_]Sig{.{ .kind = .user_fn, .name = "g", .params = &.{.int}, .ret = .int }};
    const sig_b = [_]Sig{.{ .kind = .user_fn, .name = "g", .params = &.{ .int, .int }, .ret = .int }};
    const ha = fingerprint(b.tree, b.tokens, b.source, decl, &sig_a, &.{}, &.{}, &.{});
    const hb = fingerprint(b.tree, b.tokens, b.source, decl, &sig_b, &.{}, &.{}, &.{});
    try testing.expect(ha != hb);
}

test "callee kind folds in: builtin vs user_fn of an identical sig flips the hash" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn main() {\n print(\"hi\")\n return\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    // Same name, same sig — only the resolved KIND differs (builtin print vs a
    // user `fn print`). The caller's fingerprint MUST differ, else a
    // shadow/unshadow edit is a stale-cache miscompile.
    const builtin_callee = [_]Sig{.{ .kind = .builtin, .name = "print", .params = &.{.str}, .ret = .unit }};
    const user_callee = [_]Sig{.{ .kind = .user_fn, .name = "print", .params = &.{.str}, .ret = .unit }};
    const hb = fingerprint(b.tree, b.tokens, b.source, decl, &builtin_callee, &.{}, &.{}, &.{});
    const hu = fingerprint(b.tree, b.tokens, b.source, decl, &user_callee, &.{}, &.{}, &.{});
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

test "editing a sibling fn does not change a labeled fn's hash" {
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
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v1, &.{}, &.{});
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v2, &.{}, &.{});
    try testing.expect(h1 != h2);
}

test "touched struct layout: identical layout hashes identically (cache hit)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const v = touchedStruct("Point", "Point\x00x\x00\x02\x00\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00");
    const v2 = touchedStruct("Point", "Point\x00x\x00\x02\x00\x00\x00\x00\x08\x00\x00\x00\x08\x00\x00\x00");
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v, &.{}, &.{});
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v2, &.{}, &.{});
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
    const with = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v, &.{}, &.{});
    const without = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &.{});
    // They legitimately differ (the touched set is part of the key); the point is
    // each is a pure function of its OWN inputs — recomputing `without` matches.
    try testing.expectEqual(without, fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &.{}));
    try testing.expect(with != without);
}

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
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v1, &.{}, &.{});
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v2, &.{}, &.{});
    try testing.expect(h1 != h2);
}

test "touched enum layout: identical layout hashes identically (cache hit)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const v = [1]TouchedType{.{ .kind = .@"enum", .layout = "E\x00same" }};
    const v2 = [1]TouchedType{.{ .kind = .@"enum", .layout = "E\x00same" }};
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v, &.{}, &.{});
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &v2, &.{}, &.{});
    try testing.expectEqual(h1, h2);
}

test "a match literal-value edit flips the hash" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f(n: int) -> int {\n match n { 0 -> 1, _ -> 2 }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f(n: int) -> int {\n match n { 1 -> 1, _ -> 2 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
}

test "adding a guard flips the hash (guard sentinel)" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f(n: int) -> int {\n match n { 0 -> 1, _ -> 2 }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f(n: int) -> int {\n match n { 0 if n > 0 -> 1, _ -> 2 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&b, 0));
}

test "reordering or-pattern alternatives flips the hash (order-sensitive)" {
    const gpa = testing.allocator;
    var a = try build(gpa, "enum E { A, B, C }\nfn f(e: E) -> int {\n match e { .A | .B -> 1, .C -> 2 }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "enum E { A, B, C }\nfn f(e: E) -> int {\n match e { .B | .A -> 1, .C -> 2 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 1) != fp(&b, 1));
}

test "adding a nested sub-pattern flips the hash" {
    const gpa = testing.allocator;
    var a = try build(gpa, "enum E { C(int), N }\nfn f(e: E) -> int {\n match e { .C(r) -> r, .N -> 0 }\n}\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "enum E { C(int), N }\nfn f(e: E) -> int {\n match e { .C(0) -> 1, .C(r) -> r, .N -> 0 }\n}\n");
    defer b.deinit(gpa);
    try testing.expect(fp(&a, 1) != fp(&b, 1));
}

test "type-args fold: id[int] and id[Point] get distinct fingerprints (M2)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn id(x: int) -> int {\n return x\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    // Same body/callees/touched; the SOLE difference is the concrete type-arg — an
    // int scalar vs a struct with a layout. The (d) fold must separate them.
    const as_int = [1]TouchedType{.{ .kind = .int }};
    const as_pt = [1]TouchedType{.{ .kind = .@"struct", .layout = "Point\x00x\x00\x02y" }};
    const h_int = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &as_int, &.{});
    const h_pt = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &as_pt, &.{});
    try testing.expect(h_int != h_pt);
}

test "type-args fold: empty type_args is byte-identical to no fold (warm cache preserved)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    // The (d) fold is CONDITIONAL on a non-empty slice, so an empty type_args folds
    // NOTHING and a non-generic fn's fingerprint is stable across calls — the warm
    // cache survives M2. (A pre-M2 blob keyed on the same fp still hits.)
    const a = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &.{});
    const c = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &.{});
    try testing.expectEqual(a, c);
    // A non-empty type_args MUST diverge from the empty fold (proving the gate fires).
    const v = [1]TouchedType{.{ .kind = .int }};
    const with = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &v, &.{});
    try testing.expect(with != a);
}

test "generic-param carryover: reorder/add [T,U] flips the fp; a non-generic fn is stable" {
    const gpa = testing.allocator;
    var a = try build(gpa, "fn f[T, U](x: T) -> T { x }\n");
    defer a.deinit(gpa);
    var b = try build(gpa, "fn f[U, T](x: T) -> T { x }\n");
    defer b.deinit(gpa);
    // Identical body + param/ret spelling; ONLY the generic-param ORDER differs, so
    // the (ordered) carryover fold must separate them.
    try testing.expect(fp(&a, 0) != fp(&b, 0));
    // Removing a generic param also flips it.
    var c = try build(gpa, "fn f[T](x: T) -> T { x }\n");
    defer c.deinit(gpa);
    try testing.expect(fp(&a, 0) != fp(&c, 0));
    // A non-generic fn folds NOTHING new (the carryover is conditional on non-empty),
    // so two identical non-generic fns are byte-identical — the warm cache survives.
    var d = try build(gpa, "fn g(x: int) -> int { x }\n");
    defer d.deinit(gpa);
    var e = try build(gpa, "fn g(x: int) -> int { x }\n");
    defer e.deinit(gpa);
    try testing.expectEqual(fp(&d, 0), fp(&e, 0));
}

test "type-args fold: identical type-args hash identically (instance cache hit)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn id(x: int) -> int {\n return x\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const v1 = [1]TouchedType{.{ .kind = .@"struct", .layout = "P\x00same" }};
    const v2 = [1]TouchedType{.{ .kind = .@"struct", .layout = "P\x00same" }};
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &v1, &.{});
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &v2, &.{});
    try testing.expectEqual(h1, h2);
}

test "conformance fold (e): empty conformances is byte-identical to no fold (warm cache)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    // The (e) fold is CONDITIONAL on a non-empty slice, so an empty conformances folds
    // NOTHING — a non-bounded fn's fingerprint is byte-identical to the pre-M13 fold,
    // preserving the warm cache. Both a plain call and the (d)+(e) call must agree.
    const none1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &.{});
    const none2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &.{});
    try testing.expectEqual(none1, none2);
    const witness = [_][]const u8{"lib.P.dbl"};
    const rc = [1]ResolvedConformance{.{ .protocol_name = "Doubler", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &witness }};
    const with = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc);
    try testing.expect(with != none1); // a non-empty fold MUST diverge (the gate fires)
}

test "conformance fold (e): a witness-SymName change flips the fingerprint" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const w1 = [_][]const u8{"lib.P.dbl"};
    const w2 = [_][]const u8{"lib.Q.dbl"}; // a DIFFERENT witnessing symbol (override swap)
    const rc1 = [1]ResolvedConformance{.{ .protocol_name = "Doubler", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w1 }};
    const rc2 = [1]ResolvedConformance{.{ .protocol_name = "Doubler", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w2 }};
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc1);
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc2);
    try testing.expect(h1 != h2);
}

test "conformance fold (e): a conform-layout change flips the fingerprint" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const w = [_][]const u8{"lib.P.dbl"};
    const rc1 = [1]ResolvedConformance{.{ .protocol_name = "Doubler", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w }};
    const rc2 = [1]ResolvedConformance{.{ .protocol_name = "Doubler", .conform = .{ .kind = .@"struct", .layout = "P\x00x\x00y" }, .witness_syms = &w }};
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc1);
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc2);
    try testing.expect(h1 != h2);
}

test "conformance fold (e): identical conformances hash identically (cache hit)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const w = [_][]const u8{"lib.P.dbl"};
    const rc1 = [1]ResolvedConformance{.{ .protocol_name = "Doubler", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w }};
    const rc2 = [1]ResolvedConformance{.{ .protocol_name = "Doubler", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w }};
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc1);
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc2);
    try testing.expectEqual(h1, h2);
}

test "conformance fold (e) M14: a protocol-args change (Into[int] -> Into[bool]) flips the fp" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const w = [_][]const u8{"lib.P.into$Into$int"};
    // Same protocol, same conforming layout, same witness NAME — only the protocol
    // type-args differ. This is the load-bearing stale-witness guard: an `Into[int]`
    // instance must NOT serve a cached `Into[bool]` witness.
    const int_arg = [_]TouchedType{.{ .kind = .int }};
    const bool_arg = [_]TouchedType{.{ .kind = .bool }};
    const rc_int = [1]ResolvedConformance{.{ .protocol_name = "Into", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w, .protocol_args = &int_arg }};
    const rc_bool = [1]ResolvedConformance{.{ .protocol_name = "Into", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w, .protocol_args = &bool_arg }};
    const h_int = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc_int);
    const h_bool = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc_bool);
    try testing.expect(h_int != h_bool);
}

test "conformance fold (e) M14: identical protocol-args hash identically" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const w = [_][]const u8{"lib.P.into$Into$int"};
    const a1 = [_]TouchedType{.{ .kind = .int }};
    const a2 = [_]TouchedType{.{ .kind = .int }};
    const rc1 = [1]ResolvedConformance{.{ .protocol_name = "Into", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w, .protocol_args = &a1 }};
    const rc2 = [1]ResolvedConformance{.{ .protocol_name = "Into", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w, .protocol_args = &a2 }};
    const h1 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc1);
    const h2 = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc2);
    try testing.expectEqual(h1, h2);
}

test "deriveFingerprint: a conform-layout edit flips the derive unit's key (M18)" {
    // Same protocol/kind/witnesses; only the conforming type's layout bytes differ (a
    // field added). The (c)-style layout fold must separate them, or an edited struct
    // serves a stale derived-eq blob.
    const v1 = TouchedType{ .kind = .@"struct", .layout = "P\x00x" };
    const v2 = TouchedType{ .kind = .@"struct", .layout = "P\x00x\x00y" };
    const h1 = deriveFingerprint("Eq", .eq, v1, &.{ .inline_kind });
    const h2 = deriveFingerprint("Eq", .eq, v2, &.{ .inline_kind });
    try testing.expect(h1 != h2);
}

test "deriveFingerprint: a field-witness swap flips the key (nested override guard, M18)" {
    // Same layout; a nested aggregate field's resolved witness changes (e.g. the field
    // type gained an explicit impl, so its witness SymName differs). Must flip the key.
    const cf = TouchedType{ .kind = .@"struct", .layout = "Outer\x00inner" };
    const w1 = [_]Derive.FieldWitness{.{ .eq_call = "Eq$eq$s0" }};
    const w2 = [_]Derive.FieldWitness{.{ .eq_call = "main.Inner.eq" }};
    const h1 = deriveFingerprint("Eq", .eq, cf, &w1);
    const h2 = deriveFingerprint("Eq", .eq, cf, &w2);
    try testing.expect(h1 != h2);
    // The FieldWitness TAG also folds: an `eq_call` vs a `cmp_eq` witness (Ord-only field)
    // to the same symbol must diverge.
    const w3 = [_]Derive.FieldWitness{.{ .cmp_eq = "Eq$eq$s0" }};
    try testing.expect(deriveFingerprint("Eq", .eq, cf, &w1) != deriveFingerprint("Eq", .eq, cf, &w3));
}

test "deriveFingerprint: identical recipe hashes identically (cache hit) + struct/enum differ" {
    const cf = TouchedType{ .kind = .@"struct", .layout = "P\x00x" };
    const w = [_]Derive.FieldWitness{.{ .eq_call = "Eq$eq$s1" }};
    try testing.expectEqual(deriveFingerprint("Eq", .eq, cf, &w), deriveFingerprint("Eq", .eq, cf, &w));
    // A struct-derive vs an enum-derive of the same name/layout-bytes must differ (the
    // conform.kind marker separates the two id spaces).
    const en = TouchedType{ .kind = .@"enum", .layout = "P\x00x" };
    try testing.expect(deriveFingerprint("Eq", .eq, cf, &w) != deriveFingerprint("Eq", .eq, en, &w));
}

test "deriveFingerprint M19: an Ord recipe's key differs by kind and flips on a cmp_call swap" {
    const cf = TouchedType{ .kind = .@"struct", .layout = "P\x00x" };
    // Same protocol name + layout + a scalar witness, but `.eq` vs `.ord` kind: the kind
    // marker separates the two units (an `Eq$eq$s0` and an `Ord$cmp$s0` never alias).
    const eq_key = deriveFingerprint("Eq", .eq, cf, &.{.inline_kind});
    const ord_key = deriveFingerprint("Ord", .ord, cf, &.{.inline_kind});
    try testing.expect(eq_key != ord_key);
    // An Ord recipe's aggregate field carries a `cmp_call` witness; a witness-SymName swap
    // (the field gained an explicit `impl has Ord`) flips the key so no stale blob serves.
    const w1 = [_]Derive.FieldWitness{.{ .cmp_call = "Ord$cmp$s1" }};
    const w2 = [_]Derive.FieldWitness{.{ .cmp_call = "main.Inner.cmp" }};
    try testing.expect(deriveFingerprint("Ord", .ord, cf, &w1) != deriveFingerprint("Ord", .ord, cf, &w2));
    // Identical Ord recipe hashes identically (cache hit).
    try testing.expectEqual(deriveFingerprint("Ord", .ord, cf, &w1), deriveFingerprint("Ord", .ord, cf, &w1));
}

test "deriveFingerprint M20: a Hash recipe's key differs by kind and flips on a hash_call swap" {
    const cf = TouchedType{ .kind = .@"struct", .layout = "P\x00x" };
    // Same protocol name + layout + a scalar witness, but the `.hash` kind separates the
    // unit from the Eq/Ord units (a `Hash$hash$s0` never aliases an `Eq$eq$s0`/`Ord$cmp$s0`).
    const eq_key = deriveFingerprint("Eq", .eq, cf, &.{.inline_kind});
    const ord_key = deriveFingerprint("Ord", .ord, cf, &.{.inline_kind});
    const hash_key = deriveFingerprint("Hash", .hash, cf, &.{.inline_kind});
    try testing.expect(hash_key != eq_key);
    try testing.expect(hash_key != ord_key);
    // A Hash recipe's aggregate field carries a `hash_call` witness; a witness-SymName swap
    // (the field gained an explicit `impl has Hash`) flips the key so no stale blob serves.
    const w1 = [_]Derive.FieldWitness{.{ .hash_call = "Hash$hash$s1" }};
    const w2 = [_]Derive.FieldWitness{.{ .hash_call = "main.Inner.hash" }};
    try testing.expect(deriveFingerprint("Hash", .hash, cf, &w1) != deriveFingerprint("Hash", .hash, cf, &w2));
    // Identical Hash recipe hashes identically (cache hit).
    try testing.expectEqual(deriveFingerprint("Hash", .hash, cf, &w1), deriveFingerprint("Hash", .hash, cf, &w1));
}

test "deriveFingerprint M22: a Display recipe's key differs by kind and flips on a display_call swap" {
    const cf = TouchedType{ .kind = .@"struct", .layout = "P\x00x" };
    // Same protocol name + layout + a scalar witness, but the `.display` kind separates the
    // unit from the Eq/Ord/Hash units (a `Display$display$s0` never aliases the others).
    const eq_key = deriveFingerprint("Eq", .eq, cf, &.{.inline_kind});
    const hash_key = deriveFingerprint("Hash", .hash, cf, &.{.inline_kind});
    const disp_key = deriveFingerprint("Display", .display, cf, &.{.inline_kind});
    try testing.expect(disp_key != eq_key);
    try testing.expect(disp_key != hash_key);
    // A Display recipe's aggregate field carries a `display_call` witness; a witness-SymName
    // swap (the field gained an explicit `impl has Display`) flips the key so no stale blob
    // serves — the override/stale-cache guard.
    const w1 = [_]Derive.FieldWitness{.{ .display_call = "Display$display$s1" }};
    const w2 = [_]Derive.FieldWitness{.{ .display_call = "main.Inner.display" }};
    try testing.expect(deriveFingerprint("Display", .display, cf, &w1) != deriveFingerprint("Display", .display, cf, &w2));
    // Identical Display recipe hashes identically (cache hit).
    try testing.expectEqual(deriveFingerprint("Display", .display, cf, &w1), deriveFingerprint("Display", .display, cf, &w1));
}

test "conformance fold (e) M14: empty protocol-args is byte-identical to a pre-M14 fold" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);
    const w = [_][]const u8{"lib.P.dbl"};
    // A non-generic-protocol conformance leaves `protocol_args` empty; the `len > 0` gate
    // must make its (e) fold byte-identical to an M13 conformance that never had the field.
    const rc_default = [1]ResolvedConformance{.{ .protocol_name = "Doubler", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w }};
    const rc_empty = [1]ResolvedConformance{.{ .protocol_name = "Doubler", .conform = .{ .kind = .@"struct", .layout = "P\x00x" }, .witness_syms = &w, .protocol_args = &.{} }};
    const h_default = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc_default);
    const h_empty = fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{}, &rc_empty);
    try testing.expectEqual(h_default, h_empty);
}
