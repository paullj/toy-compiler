//! The compilation-phase taxonomy — the SINGLE source of truth shared by the
//! on-disk `Cache` (re-exported there as `Cache.Phase`) and the static
//! `StageGraph`. The CACHEABLE subset (`Kind.cacheable()` below: lex/parse/
//! codegen/source) names the on-disk cache entries; the rest are the fine-grained
//! typecheck phases and the scheduling BARRIER joins (`collect`/`global_tables`).
//!
//! EXPLICIT discriminants pin the CACHEABLE subset's bytes (lex=0, parse=1,
//! codegen=3, source=11) so `Cache.Key.digest()` folds a stable byte per phase — folding a
//! different byte would cold-rebuild every codegen entry. The remaining kinds'
//! bytes are kept distinct, but their values are not load-bearing.

const std = @import("std");

/// The KIND of a compilation phase. The cacheable kinds map to themselves 1:1
/// (no bridge switch): a cache key records its phase directly off its kind.
///
/// `collect` and `global_tables` are the `Engine.barrier` fan-IN joins the
/// `StageGraph` schedules: `collect`'s fold is over every module's parse digest
/// (the global name-resolution tables built from them); `global_tables`'s fold is
/// over every global fn's resolve digest (the program-wide layout/sig tables the
/// per-fn body region depends on). Neither names a cache entry (never cacheable).
pub const Kind = enum(u8) {
    lex = 0,
    parse = 1,
    signature = 2,
    /// KEEPS byte 3 (the old `Cache.Phase.codegen` value) — see the type doc.
    codegen = 3,
    body = 4,
    type_of = 5,
    layout = 6,
    resolve_name = 7,
    /// The DISCOVER barrier join (folds the entry-path digest -> the module graph).
    discover = 8,
    /// The resolve COLLECT barrier join (folds the per-module parse digests).
    collect = 9,
    /// The typecheck Pass-A GLOBAL_TABLES barrier join (folds the per-fn resolve
    /// digests).
    global_tables = 10,
    /// The raw module SOURCE bytes, keyed by their own content fingerprint. Stored
    /// alongside lex/parse so the warm-discover fast path can serve a file's source +
    /// tokens + AST entirely from the bulk-loaded pack (no per-file read syscall) when
    /// the manifest's (mtime, size) unchanged-predicate says the file is unchanged. Its
    /// byte (11) is pinned alongside every other cacheable phase by the exhaustive
    /// byte-pin test in `Cache.zig`.
    source = 11,

    /// Whether a node of this kind names an on-disk `Cache` entry (the CACHEABLE
    /// subset) vs an in-memory-only phase. Exhaustive no-else: a new kind FAILS TO
    /// COMPILE until it declares its tier here.
    pub fn cacheable(kind: Kind) bool {
        return switch (kind) {
            .lex, .parse, .codegen, .source => true,
            .signature, .body, .type_of, .layout, .resolve_name, .discover, .collect, .global_tables => false,
        };
    }

    /// Whether results for this kind depend on the compilation target. Target is
    /// folded into the on-disk cache key only when true, so target-independent
    /// phases share one entry across targets. Exhaustive no-else: a new kind FAILS
    /// TO COMPILE until it declares its target-sensitivity here.
    pub fn targetSensitive(kind: Kind) bool {
        return switch (kind) {
            .codegen => true, // aarch64 blobs must not alias across targets
            .lex, .parse, .signature, .body, .type_of, .layout, .resolve_name, .discover, .collect, .global_tables, .source => false,
        };
    }
};
