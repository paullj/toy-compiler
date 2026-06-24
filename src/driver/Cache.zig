//! Content-addressed, on-disk cache of compilation results, organized as a
//! demand-driven (query) memo table: each phase asks the cache for a `Key`, and
//! a hit lets the driver skip the work entirely.
//!
//! Cache key = (compiler identity, phase, target, inputs):
//!
//!   * Compiler identity is the *directory* — every entry lives under
//!     `.toy-cache/<stamp>/` (see `version.zig`). A different compiler build
//!     simply writes to a different subdir, so we never read stale results and
//!     never need wipe-on-startup logic. Old subdirs can be GC'd lazily.
//!   * Phase, target, and a hash of the inputs are folded into the per-entry
//!     `Key.digest`, which names the file inside that subdir.
//!
//! Target is only mixed in for phases that actually depend on it
//! (`Phase.targetSensitive`): lexing is target-independent, so the same tokens
//! are reused across targets instead of being needlessly re-lexed per target.
//!
//! Entries are written via a unique temp file + atomic rename, so many worker
//! threads share the cache without locking.

const std = @import("std");
const Io = std.Io;

const Cache = @This();

/// Compilation phases that participate in the cache. Each is a distinct query.
pub const Phase = enum(u8) {
    lex,
    parse,
    /// Resolve + typecheck. A pipeline *level* only in M0 (run in-memory, not
    /// stored), but ordered after `parse` so `@intFromEnum` gates the pipeline.
    check,
    /// Per-function lowering (M5+; M12 Ast→Ir→aarch64). Cached by a transitive
    /// content fingerprint, not a source hash, and target-sensitive (the blob is
    /// aarch64 machine code). The IR is built INSIDE this query and never cached
    /// separately — the cache stays ONE-TIER ([C8]).
    codegen,

    /// Whether results for this phase depend on the compilation target. Target
    /// is only folded into the cache key when true, so target-independent
    /// phases (lex, parse, check) share a single entry across targets.
    pub fn targetSensitive(phase: Phase) bool {
        return switch (phase) {
            .lex, .parse, .check => false,
            .codegen => true, // aarch64 blobs must not alias across targets [C10]
        };
    }
};

/// Identifies a single cacheable unit of work.
pub const Key = struct {
    phase: Phase,
    /// Target triple (e.g. "native", "aarch64-macos"). Ignored for
    /// target-independent phases.
    target: []const u8,
    /// Hash of the phase's inputs (for lexing, the source bytes).
    input: u64,

    pub fn fromSource(phase: Phase, target: []const u8, source: []const u8) Key {
        return .{ .phase = phase, .target = target, .input = std.hash.Wyhash.hash(0, source) };
    }

    /// Key a phase directly by a precomputed content fingerprint (M5 codegen),
    /// rather than by rehashing a source byte range. `digest()` folds `input`
    /// verbatim, so the fingerprint IS the cache identity.
    pub fn fromFingerprint(phase: Phase, target: []const u8, fp: u64) Key {
        return .{ .phase = phase, .target = target, .input = fp };
    }

    /// Fold the key components into the on-disk entry name.
    pub fn digest(k: Key) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(&[_]u8{@intFromEnum(k.phase)});
        if (k.phase.targetSensitive()) h.update(k.target);
        h.update(std.mem.asBytes(&k.input));
        return h.final();
    }
};

/// Per-compiler cache directory, e.g. ".toy-cache/0.0.0-1a2b3c...". Borrowed,
/// must outlive the `Cache` (the driver keeps it on its stack for the run).
dir: []const u8,

/// Create the per-stamp cache directory (and the `.toy-cache` root) if needed.
/// `dir` must already include the compiler stamp.
pub fn init(io: Io, dir: []const u8) !Cache {
    try Io.Dir.cwd().createDirPath(io, dir);
    return .{ .dir = dir };
}

/// Every entry is `[u64 checksum][payload]`. The checksum is a hash of the
/// payload bytes; on load we recompute and treat any mismatch as a miss. This
/// turns payload corruption — bitrot, a torn write, a cache copied between hosts
/// of different endianness (we serialize native-endian) — into a clean re-run
/// instead of feeding a garbage `Token`/`Node` array downstream (which would
/// out-of-bounds panic on the bad indices). The temp-file + atomic rename in
/// `put` already prevents readers from seeing half-written entries.
const checksum_seed: u64 = 0xCAC4E5EED;
const checksum_len = @sizeOf(u64);

/// Load a cached entry for `key` as a slice of `T`, or null on a miss (absent or
/// corrupt). `T` must be a fixed-size value type (e.g. `Token`, `Ast.Node`).
/// Caller owns the result.
pub fn get(c: Cache, comptime T: type, gpa: std.mem.Allocator, io: Io, key: Key) !?[]T {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = c.entryPath(&buf, key);

    const bytes = Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer gpa.free(bytes);

    if (bytes.len < checksum_len) return null; // missing/truncated header
    const stored = std.mem.readInt(u64, bytes[0..checksum_len], .little);
    const payload = bytes[checksum_len..];
    if (std.hash.Wyhash.hash(checksum_seed, payload) != stored) return null; // corrupt
    if (payload.len % @sizeOf(T) != 0) return null; // foreign element size

    const items = try gpa.alloc(T, payload.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(items), payload);
    return items;
}

/// Store `items` under `key`. `tmp_tag` must be unique among concurrent writers
/// (the file index works) so temp files never collide.
pub fn put(c: Cache, comptime T: type, io: Io, key: Key, tmp_tag: usize, items: []const T) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = c.entryPath(&buf, key);
    const tmp = std.fmt.bufPrint(&tmp_buf, "{s}/{x:0>16}.{d}.tmp", .{ c.dir, key.digest(), tmp_tag }) catch return;

    const payload = std.mem.sliceAsBytes(items);
    var header: [checksum_len]u8 = undefined;
    std.mem.writeInt(u64, &header, std.hash.Wyhash.hash(checksum_seed, payload), .little);

    {
        var file = try Io.Dir.cwd().createFile(io, tmp, .{});
        defer file.close(io);
        try file.writeStreamingAll(io, &header);
        try file.writeStreamingAll(io, payload);
    }
    try Io.Dir.cwd().rename(tmp, Io.Dir.cwd(), path, io);
}

fn entryPath(c: Cache, buf: []u8, key: Key) []const u8 {
    return std.fmt.bufPrint(buf, "{s}/{x:0>16}", .{ c.dir, key.digest() }) catch unreachable;
}
