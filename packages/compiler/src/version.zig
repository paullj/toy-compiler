//! Compiler identity, used as the top-level cache namespace.
//!
//! The cache must never serve results produced by a *different* compiler. We
//! derive a stamp from `source_digest` — a hash of the compiler's own source —
//! so any edit (committed or not) lands cache entries in a fresh
//! `.toy/<stamp>/cache/` subdir and stale results are never read.
//!
//! `source_digest` is computed at *build time* by hashing every file under
//! `src/` (see `sourceDigest` in build.zig) and injected via the
//! `build_options` module. Doing it at build time — rather than via comptime
//! `@embedFile` — means new source files are picked up automatically, only the
//! 64-bit digest (not the source bytes) is baked into the binary, and we never
//! pay to embed or comptime-hash the growing source tree.

const std = @import("std");
const build_options = @import("build_options");

/// Human-facing version. Bump on releases; combined with `source_digest` it
/// forms the cache stamp.
pub const semver = build_options.semver;

/// Content hash of the whole compiler. Changes whenever any `src/` file does.
pub const source_digest: u64 = build_options.source_digest;

/// DEV pipeline-inspection switch (see build.zig): true in a Debug build (and every
/// test binary), false in a release build. The CLI schema (`Cli.zig`) reads it at
/// comptime to gate the `--emit lex|parse|ir` flag's presence. Re-exported here so the
/// single library-module `build_options` is the ONE owner of that generated file — the
/// exe module reaches it via `toyc.version.dev_inspect` rather than importing
/// `build_options` a second time (which Zig rejects: one file, one module).
pub const dev_inspect: bool = build_options.dev_inspect;

/// The cache stamp: identifies "this exact compiler build". Written into a
/// buffer (no allocation). Used as the per-compiler cache subdirectory name.
pub fn stamp(buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}+{x:0>16}", .{ semver, source_digest }) catch unreachable;
}

/// Upper bound on the length of `stamp`, for sizing stack buffers.
pub const stamp_max = semver.len + 1 + 16;
