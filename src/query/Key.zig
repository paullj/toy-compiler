//! The uniform query key. A thin typed constructor over `Cache.Key` that
//! discriminates every CACHEABLE query shape (lex / parse / codegen, single-file
//! and graph) WITHOUT a new digest formula and WITHOUT cross-shape collision.
//!
//! Why no collision (and why we do NOT change `Cache.Key.digest()`):
//!   * `digest()` folds `@intFromEnum(phase)` BEFORE `input`, so a lex key and a
//!     codegen key with the same `u64` input cannot alias — distinct phase byte
//!     => distinct Wyhash stream. lex/parse share `input = Wyhash(source)` but
//!     differ by phase byte.
//!   * codegen's `input` is `fp ^ optMix(opt) ^ symMix(sym)` — a DIFFERENT u64
//!     namespace, a different phase byte, AND target-sensitive ([C10]).
//!   * single-file vs graph codegen needs NO extra bit: `symMix` folds the
//!     emitted `SymName` (bare `add` vs qualified `m.add` => different input),
//!     the existing [Cx] M14 invariant.
//!
//! This module ONLY consolidates the `fp ^ optMix ^ symMix` arithmetic (formerly
//! duplicated in Driver) into one place. The seeds and `digest()` are sacred:
//! changing them changes on-disk identity (a cold rebuild is fine) but must NOT
//! change emitted bytes.

const std = @import("std");
const Cache = @import("Cache.zig");
const Opt = @import("../opt/Opt.zig");
const Link = @import("../link/Link.zig");

pub const Phase = Cache.Phase;
pub const Key = Cache.Key;

/// lex query: keyed by source-hash, target-independent.
pub fn lex(target: []const u8, source: []const u8) Key {
    return Cache.Key.fromSource(.lex, target, source);
}

/// parse query: keyed by source-hash, target-independent.
pub fn parse(target: []const u8, source: []const u8) Key {
    return Cache.Key.fromSource(.parse, target, source);
}

/// codegen query: keyed by the content fingerprint xor'd with the opt-level and
/// the fn's own emitted symbol name, target-sensitive. This is the ONE place the
/// `fp ^ optMix(opt) ^ symMix(sym)` fold lives (was duplicated in Driver).
pub fn codegen(target: []const u8, fp: u64, opt: Opt.Config, sym: Link.SymName) Key {
    return Cache.Key.fromFingerprint(.codegen, target, fp ^ optMix(opt) ^ symMix(sym));
}

/// Hash the opt config's fixed wire byte into a u64 to xor into the codegen
/// fingerprint. Wyhash for good distribution; deterministic (the byte is a fixed
/// packed struct with a 0 pad). The ONLY place opt level enters the cache key.
pub fn optMix(cfg: Opt.Config) u64 {
    return std.hash.Wyhash.hash(0x4f_50_54_4d, &[_]u8{cfg.bits()}); // "OPTM"
}

/// The STABLE codegen DAG-node identity — distinct from `codegen(...).digest()`,
/// which folds the transitive content fingerprint and so makes a node's identity its
/// content hash. This id folds ONLY a phase tag + the target (target-sensitive) + the
/// opt level + the program-wide global fn id (`gid`, the `res.fns` index) + whether the
/// fn is the program ENTRY. `gid` is globally unique across modules (unlike the
/// bare-name signature id), so this id is unique across (kind=codegen, module, target,
/// opt, is_entry) — a stable per-fn name independent of the content fp.
///
/// It names the in-memory DAG node recorded for OBSERVABILITY (consumed by the
/// `--query-stats` red-green REPORTER, `--verify`'s auditor, and `--dump-dag`).
/// Reuse/cutoff is NOT decided here — the content-fp cache (`Key.codegen`) is the sole
/// driver; this id only makes the observed node STABLE across rebuilds so the reporter
/// can track a fn's codegen node by identity.
///
/// `is_entry` is folded because it is a codegen-root input NOT carried by the fn's
/// body subtree (`body(gid)`): the entry fn gets a distinct prologue. Folding it keeps
/// the observed node honest — an entry flip lands under a DIFFERENT stable id rather
/// than aliasing the prior (non-entry) node's fp. Costs nothing on the common path
/// (is_entry is stable per fn).
///
/// NOTE: this is NOT the on-disk cache key. `Key.codegen` (content-fp-derived) stays
/// the cache-entry name so cold-cache bytes are byte-identical; this id only names the
/// in-memory DAG node the red-green reporter reads.
pub fn codegenIdentity(target: []const u8, opt: Opt.Config, gid: u32, is_entry: bool) u64 {
    var h = std.hash.Wyhash.init(0x43_47_49_44); // "CGID"
    h.update(&[_]u8{@intFromEnum(Phase.codegen)});
    h.update(target);
    var ob: [8]u8 = undefined;
    std.mem.writeInt(u64, &ob, optMix(opt), .little);
    h.update(&ob);
    var gb: [4]u8 = undefined;
    std.mem.writeInt(u32, &gb, gid, .little);
    h.update(&gb);
    h.update(&[_]u8{@intFromBool(is_entry)});
    return h.final();
}

/// Mix a function's OWN emitted `SymName{kind,name}` into the codegen cache key.
///
/// WHY (M14, [Cx] cache soundness): the fingerprint folds a fn's body and its
/// CALLEES' identities, but NOT the fn's own internal name — and a leaf fn (no
/// callees) with identical source produces an identical fingerprint regardless of
/// the name it is emitted under. In M14 the SAME `fn add` body is emitted bare
/// (`add`) single-file but module-qualified (`m.add`) in a graph build; that name
/// IS baked into the cached `FnCode.sym` (and into every caller's reloc target).
/// Without folding it, a graph build would serve a single-file build's cached
/// blob carrying the WRONG (bare) sym → `error.UnresolvedSymbol` at link, or worse
/// a cross-module stale-name miscompile. Folding the own name keys each emitted
/// identity to its own cache slot. Deterministic; the only other key input beside
/// the pure fingerprint and `optMix`.
pub fn symMix(sym: Link.SymName) u64 {
    var h = std.hash.Wyhash.init(0x53_59_4d_4e); // "SYMN"
    h.update(&[_]u8{@intFromEnum(sym.kind)});
    h.update(sym.name);
    return h.final();
}

const testing = std.testing;

test "codegenIdentity is unique per (gid, target, opt, is_entry) and ignores content fp" {
    const o0: Opt.Config = .{};
    const t_arm = "aarch64-macos";
    const t_x = "x86_64-macos";

    // Distinct global fn ids => distinct stable node ids (the program-wide-unique
    // precondition for naming each fn's observability node by stable identity).
    try testing.expect(codegenIdentity(t_arm, o0, 0, false) != codegenIdentity(t_arm, o0, 1, false));
    try testing.expect(codegenIdentity(t_arm, o0, 5, false) != codegenIdentity(t_arm, o0, 6, false));

    // Same gid, different target => distinct (target-sensitive: a cross-target node
    // must not alias the wrong target's observed fp).
    try testing.expect(codegenIdentity(t_arm, o0, 3, false) != codegenIdentity(t_x, o0, 3, false));

    // Same gid+target, different opt level => distinct.
    const o1: Opt.Config = .{ .fold = true };
    try testing.expect(codegenIdentity(t_arm, o0, 3, false) != codegenIdentity(t_arm, o1, 3, false));

    // Same gid+target+opt, different ENTRY flag => distinct (an entry flip lands under
    // a different id rather than aliasing the prior non-entry node's fp).
    try testing.expect(codegenIdentity(t_arm, o0, 3, false) != codegenIdentity(t_arm, o0, 3, true));

    // Stable for identical inputs (no content-fp dependence: identity is decoupled
    // from the body bytes — that is the whole point).
    try testing.expectEqual(codegenIdentity(t_arm, o0, 9, true), codegenIdentity(t_arm, o0, 9, true));
}
