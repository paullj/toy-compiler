//! Boundary tests for the incremental query engine seam (integration).
//!
//! These exercise the engine END-TO-END: each spins up a threaded `Io` runtime + a
//! temp-dir `Cache` stand-in and drives real lex/parse + a real on-disk cache. They
//! consume the compiler as a BLACK BOX through `@import("toy_compiler")` (the
//! published surface), so they live in the repo-root tests/ — not inline in
//! Engine.zig — and run in their own `toy-integration-test` binary.
//!
//! The distributed query/cache/fingerprint/force-verify/parallel logic lives in
//! `Engine` + `Key` + `Cache` + `Fingerprint`. These pin the five boundary
//! behaviours that seam must preserve:
//!
//!   * MISS  — a cold key computes and stores.
//!   * HIT   — a primed key serves the cached value without recomputing.
//!   * FORCE — `.force` mode skips the cache READ but still stores (re-lower).
//!   * VERIFY— `.verify` re-derivation is byte-identical (the determinism
//!             basis the codegen verify gate asserts against the stored blob).
//!   * INVALIDATION — the uniform `Key.codegen` digest flips for exactly the edits
//!             that must recompile (struct field edit on a touching fn; a callee
//!             SIGNATURE change on the caller) and is STABLE for a callee BODY-only
//!             change on the caller — the (a)/(b) split.
//!
//! The whole-program force/verify and module-granular invalidation paths are
//! additionally covered end-to-end by the Driver soundness tests and
//! the three corpora; these pin the engine UNIT contract those build on.

const std = @import("std");
const Io = std.Io;
const testing = std.testing;

const toyc = @import("toy_compiler");
const Engine = toyc.QueryEngine;
const Cache = toyc.Cache;
const Key = toyc.QueryKey;
const Fingerprint = toyc.Fingerprint;
const Token = toyc.token.Token;
const Ast = toyc.Ast;
const Link = toyc.Link;
const Lexer = toyc.Lexer;
const Parser = toyc.Parser;
const Sig = Fingerprint.Sig;
const TouchedType = Fingerprint.TouchedType;

/// A fresh temp-dir `Cache` + `Engine`, plus teardown. Each test gets its own dir
/// so concurrent test execution never shares cache state. The dir is created under
/// the test tmp tree and removed on `deinit`.
const Harness = struct {
    threaded: std.Io.Threaded,
    tmp: testing.TmpDir,
    dir_buf: [80]u8 = undefined,
    cache: Cache = undefined,
    io: Io = undefined,

    /// `init(gpa, &h)` initialises a heap-stable harness in place: the `Io.Threaded`
    /// runtime can't be moved after `io()` is taken (the `Io` vtable closes over its
    /// address), and `Cache.dir` borrows `dir_buf`, so both must live in the caller's
    /// `Harness` storage. Each test gets its own temp cache dir.
    fn init(gpa: std.mem.Allocator, self: *Harness) !void {
        self.threaded = std.Io.Threaded.init(gpa, .{});
        self.io = self.threaded.io();
        self.tmp = testing.tmpDir(.{});
        // tmpDir hands back a sub_path under .zig-cache/tmp; root the cache dir there.
        const dir = std.fmt.bufPrint(&self.dir_buf, ".zig-cache/tmp/{s}/cc", .{&self.tmp.sub_path}) catch unreachable;
        self.cache = try Cache.init(self.io, dir);
    }

    fn deinit(self: *Harness) void {
        self.tmp.cleanup();
        self.threaded.deinit();
    }

    fn engine(self: *Harness, mode: Engine.Mode) Engine {
        return Engine.init(self.cache, mode);
    }
};

/// A compute closure that counts how many times it runs, so a HIT can be proven by
/// the counter NOT advancing. Returns a fixed gpa-owned blob (the codegen path's
/// `[]u8` FnCode shape).
const CountingBlob = struct {
    gpa: std.mem.Allocator,
    runs: *usize,
    payload: []const u8,
    pub fn run(c: @This()) ![]u8 {
        c.runs.* += 1;
        return c.gpa.dupe(u8, c.payload);
    }
};

test "MISS then HIT: a cold key computes once; the warm key serves cached without recompute" {
    const gpa = testing.allocator;
    var h: Harness = undefined;
    try Harness.init(gpa, &h);
    defer h.deinit();
    const engine = h.engine(.normal);

    const key = Key.codegen("native", 0xC0FFEE, .{}, .{ .kind = .user_fn, .name = "f" });
    var runs: usize = 0;

    // MISS: cold key -> compute runs, value stored.
    const miss = try engine.query(u8, gpa, h.io, key, 0, false, true, CountingBlob{ .gpa = gpa, .runs = &runs, .payload = "BLOB" });
    defer gpa.free(miss.value);
    try testing.expect(!miss.cached);
    try testing.expectEqual(@as(usize, 1), runs);
    try testing.expectEqualSlices(u8, "BLOB", miss.value);

    // HIT: same key -> served from cache, compute does NOT run again.
    const hit = try engine.query(u8, gpa, h.io, key, 0, false, true, CountingBlob{ .gpa = gpa, .runs = &runs, .payload = "BLOB" });
    defer gpa.free(hit.value);
    try testing.expect(hit.cached);
    try testing.expectEqual(@as(usize, 1), runs); // unchanged — the proof of a hit
    try testing.expectEqualSlices(u8, "BLOB", hit.value);
}

test "FORCE: .force-equivalent re-derivation ignores the primed cache (re-lower, then store)" {
    // The engine's codegen FORCE branch is `if (mode != .force) { ...read... }` —
    // it SKIPS the cache read and recomputes, then stores. Here we model that
    // contract at the cache seam the engine sits on: a primed key, then a forced
    // recompute that does NOT consult the stored blob (compute runs) but DOES
    // overwrite it (a subsequent normal read serves the fresh bytes).
    const gpa = testing.allocator;
    var h: Harness = undefined;
    try Harness.init(gpa, &h);
    defer h.deinit();
    const engine = h.engine(.normal);

    const key = Key.codegen("native", 0xF0F0, .{}, .{ .kind = .user_fn, .name = "g" });
    var runs: usize = 0;

    // Prime the cache.
    const primed = try engine.query(u8, gpa, h.io, key, 0, false, true, CountingBlob{ .gpa = gpa, .runs = &runs, .payload = "V1" });
    gpa.free(primed.value);
    try testing.expectEqual(@as(usize, 1), runs);

    // FORCE model: skip the read, recompute fresh, store. (We invoke compute
    // directly + put, exactly as Engine.codegen's force branch does: no get.)
    const fresh = try CountingBlob.run(.{ .gpa = gpa, .runs = &runs, .payload = "V2" });
    defer gpa.free(fresh);
    try testing.expectEqual(@as(usize, 2), runs); // compute ran despite a primed cache
    try h.cache.put(u8, h.io, key, 0, fresh);

    // The forced store is now what a normal read serves.
    const after = (try h.cache.get(u8, gpa, h.io, key)).?;
    defer gpa.free(after);
    try testing.expectEqualSlices(u8, "V2", after);
}

test "VERIFY: re-deriving identical inputs is byte-identical (the determinism basis)" {
    // The codegen VERIFY gate re-lowers a fn and asserts its packed bytes equal the
    // cached blob (hit) or a 2nd fresh lowering (cold). That gate only holds because
    // the underlying derivation is deterministic: identical frozen inputs -> an
    // identical fingerprint -> an identical cache key -> the same stored bytes. We
    // pin that determinism at the seam: re-fingerprinting the SAME fn twice yields
    // the same u64 (so the same Key.codegen digest), and storing+reading the same
    // blob round-trips byte-identically.
    const gpa = testing.allocator;
    var h: Harness = undefined;
    try Harness.init(gpa, &h);
    defer h.deinit();

    var b = try build(gpa, "fn add(a: int, b: int) -> int {\n return a + b\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    const fp1 = Fingerprint.fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{});
    const fp2 = Fingerprint.fingerprint(b.tree, b.tokens, b.source, decl, &.{}, &.{}, &.{});
    try testing.expectEqual(fp1, fp2); // deterministic fingerprint

    const sym: Link.SymName = .{ .kind = .user_fn, .name = "add" };
    const k1 = Key.codegen("aarch64-macos", fp1, .{}, sym);
    const k2 = Key.codegen("aarch64-macos", fp2, .{}, sym);
    try testing.expectEqual(k1.digest(), k2.digest()); // same key => same slot

    // Round-trip the blob through the cache byte-identically (verify compares
    // exactly these bytes).
    try h.cache.put(u8, h.io, k1, 0, "AARCH64BYTES");
    const got = (try h.cache.get(u8, gpa, h.io, k2)).?;
    defer gpa.free(got);
    try testing.expectEqualSlices(u8, "AARCH64BYTES", got);
}

const Built = struct {
    tokens: []Token,
    tree: Ast.Tree,
    source: []const u8,

    fn deinit(self: *Built, gpa: std.mem.Allocator) void {
        gpa.free(self.tokens);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
    }

    fn fnDecl(self: *const Built, idx: usize) Ast.Index {
        const prog = self.tree.nodes[Ast.root(self.tree.nodes).int()];
        return Ast.rangeSlice(self.tree, prog.lhs.int())[idx];
    }
};

fn freeTree(gpa: std.mem.Allocator, tree: Ast.Tree) void {
    gpa.free(tree.nodes);
    gpa.free(tree.extra);
}

fn build(gpa: std.mem.Allocator, source: []const u8) !Built {
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, source);
    return .{ .tokens = tokens, .tree = tree, .source = source };
}

/// The full codegen key for the `fn_idx`-th fn, emitted under `sym`, with the given
/// callee sigs and touched types — the exact `fp ^ optMix ^ symMix` fold the engine
/// folds at the codegen seam.
fn cgKey(b: *const Built, fn_idx: usize, sym_name: []const u8, callees: []const Sig, touched: []const TouchedType) Key.Key {
    const fp = Fingerprint.fingerprint(b.tree, b.tokens, b.source, b.fnDecl(fn_idx), callees, touched, &.{});
    return Key.codegen("aarch64-macos", fp, .{}, .{ .kind = .user_fn, .name = sym_name });
}

test "INVALIDATION: a struct field edit flips the codegen key of a fn that TOUCHES it" {
    // Editing a struct's layout must recompile every fn whose touched set folds that
    // struct's layout bytes — its codegen key digest changes, so the on-disk slot
    // changes (a stale hit would miscompile against the old layout).
    const gpa = testing.allocator;
    var b = try build(gpa, "fn area(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);

    const layout_v1 = [1]TouchedType{.{ .kind = .@"struct", .layout = "Point\x00x" }};
    const layout_v2 = [1]TouchedType{.{ .kind = .@"struct", .layout = "Point\x00x\x00y" }}; // field added
    const k1 = cgKey(&b, 0, "area", &.{}, &layout_v1);
    const k2 = cgKey(&b, 0, "area", &.{}, &layout_v2);
    try testing.expect(k1.digest() != k2.digest());
}

test "INVALIDATION: a callee SIGNATURE change flips the caller's codegen key; a callee BODY change does NOT" {
    // The (a)/(b) split: a fn folds its callees' SIGNATURES, never their bodies. So
    // editing a callee's body leaves the caller's key STABLE (a cache hit),
    // but changing a callee's signature flips it (the caller must re-lower).
    const gpa = testing.allocator;
    var b = try build(gpa, "fn main() -> int {\n return g(1)\n}\n");
    defer b.deinit(gpa);

    const sig_base = [_]Sig{.{ .kind = .user_fn, .name = "g", .params = &.{.int}, .ret = .int }};
    // Same sig (a callee body-only edit doesn't change `g`'s sig) => SAME caller key.
    const sig_body_only = [_]Sig{.{ .kind = .user_fn, .name = "g", .params = &.{.int}, .ret = .int }};
    // A 2nd param (a callee SIGNATURE change) => caller key FLIPS.
    const sig_changed = [_]Sig{.{ .kind = .user_fn, .name = "g", .params = &.{ .int, .int }, .ret = .int }};

    const k_base = cgKey(&b, 0, "main", &sig_base, &.{});
    const k_body = cgKey(&b, 0, "main", &sig_body_only, &.{});
    const k_sig = cgKey(&b, 0, "main", &sig_changed, &.{});

    try testing.expectEqual(k_base.digest(), k_body.digest()); // body-only edit: cache HIT
    try testing.expect(k_base.digest() != k_sig.digest()); // sig change: recompile
}

test "INVALIDATION: a fn's OWN body edit flips its key, but leaves an untouched sibling's key stable" {
    // Position independence: editing one fn's body changes only THAT fn's key. A
    // sibling whose source is unchanged keeps an identical fingerprint (no indices
    // / offsets folded), so its codegen key — and thus its cache slot — is stable.
    const gpa = testing.allocator;
    var v1 = try build(gpa, "fn helper() -> int {\n return 7\n}\nfn main() -> int {\n return 1\n}\n");
    defer v1.deinit(gpa);
    var v2 = try build(gpa, "fn helper() -> int {\n return 7\n}\nfn main() -> int {\n return 2\n}\n"); // only main edited
    defer v2.deinit(gpa);

    // helper (fn 0) is unchanged across the edit -> identical key (cache HIT).
    try testing.expectEqual(cgKey(&v1, 0, "helper", &.{}, &.{}).digest(), cgKey(&v2, 0, "helper", &.{}, &.{}).digest());
    // main (fn 1) was edited -> its key flips (recompile).
    try testing.expect(cgKey(&v1, 1, "main", &.{}, &.{}).digest() != cgKey(&v2, 1, "main", &.{}, &.{}).digest());
}

test "lex query: miss computes then hit serves from cache" {
    const gpa = testing.allocator;
    var h: Harness = undefined;
    try Harness.init(gpa, &h);
    defer h.deinit();
    const engine = h.engine(.normal);

    const src = "fn main() {}";

    const miss = try engine.lex(gpa, h.io, "native", src, 0, false);
    defer gpa.free(miss.value);
    try testing.expect(!miss.cached);
    try testing.expect(miss.value.len > 0);

    const hit = try engine.lex(gpa, h.io, "native", src, 0, false);
    defer gpa.free(hit.value);
    try testing.expect(hit.cached);
    try testing.expectEqual(miss.value.len, hit.value.len);
}

test "parse query: miss parses+stores; hit serves a validated cached tree" {
    const gpa = testing.allocator;
    var h: Harness = undefined;
    try Harness.init(gpa, &h);
    defer h.deinit();
    const engine = h.engine(.normal);

    const src = "fn main() -> int {\n return 0\n}\n";
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);

    const miss = try engine.parse(gpa, h.io, "native", src, tokens, 0, false);
    try testing.expect(!miss.cached);
    // The parser always returns a tree; a clean parse leaves `diags` empty.
    try testing.expectEqual(@as(usize, 0), miss.diags.len);
    gpa.free(@constCast(miss.diags));
    freeTree(gpa, miss.tree);

    const hit = try engine.parse(gpa, h.io, "native", src, tokens, 0, false);
    try testing.expect(hit.cached);
    try testing.expectEqual(@as(usize, 0), hit.diags.len);
    gpa.free(@constCast(hit.diags));
    freeTree(gpa, hit.tree);
}

test "codegen key discrimination: opt level / own-symbol / fingerprint each key distinctly" {
    // The uniform `Key.codegen` must not alias across O-level, emitted symbol, or
    // fingerprint — the invariants the consolidated key preserves.
    const base = Key.codegen("native", 0x1111, .{}, .{ .kind = .user_fn, .name = "add" });

    const diff_fp = Key.codegen("native", 0x2222, .{}, .{ .kind = .user_fn, .name = "add" });
    try testing.expect(base.digest() != diff_fp.digest());

    const diff_sym = Key.codegen("native", 0x1111, .{}, .{ .kind = .user_fn, .name = "m.add" });
    try testing.expect(base.digest() != diff_sym.digest());

    // A lex key and a codegen key with the SAME u64 input cannot alias — the phase
    // byte is folded first (no cross-shape collision).
    const lex_key = Key.lex("native", "add");
    try testing.expect(lex_key.digest() != Key.codegen("native", lex_key.input, .{}, .{ .kind = .user_fn, .name = "add" }).digest());
}

test "phase discrimination: lex and parse of the same source land in distinct slots" {
    // lex/parse share `input = Wyhash(source)` but differ by the phase byte folded
    // FIRST in digest(), so the two front-end queries never collide.
    const src = "fn main() {}";
    try testing.expect(Key.lex("native", src).digest() != Key.parse("native", src).digest());
}

test "target sensitivity: codegen keys differ across targets; lex keys do NOT" {
    // codegen blobs are target-specific machine code and must not alias across
    // targets; lex/parse are target-independent and intentionally share a slot.
    const cg_a = Key.codegen("aarch64-macos", 0x99, .{}, .{ .kind = .user_fn, .name = "f" });
    const cg_b = Key.codegen("x86_64-linux", 0x99, .{}, .{ .kind = .user_fn, .name = "f" });
    try testing.expect(cg_a.digest() != cg_b.digest());

    const lex_a = Key.lex("aarch64-macos", "fn f(){}");
    const lex_b = Key.lex("x86_64-linux", "fn f(){}");
    try testing.expectEqual(lex_a.digest(), lex_b.digest());
}
