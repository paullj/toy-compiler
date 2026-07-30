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
//!     namespace, a different phase byte, AND target-sensitive.
//!   * single-file vs graph codegen needs NO extra bit: `symMix` folds the
//!     emitted `SymName` (bare `add` vs qualified `m.add` => different input),
//!     the existing module-qualified-name invariant.
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

/// Mix a function's OWN emitted `SymName{kind,name}` into the codegen cache key.
///
/// WHY (cache soundness): the fingerprint folds a fn's body and its
/// CALLEES' identities, but NOT the fn's own internal name — and a leaf fn (no
/// callees) with identical source produces an identical fingerprint regardless of
/// the name it is emitted under. The SAME `fn add` body is emitted bare
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
