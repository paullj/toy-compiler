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

// ---- M17 per-program DAG artifact -----------------------------------------
// The red-green dependency graph is ONE artifact per build identity, not a
// content-keyed per-fn entry, so it cannot use `Key`/`entryPath`. It rides in
// the same per-stamp dir under a fixed name `dag.<program_key>`. The compiler
// stamp already namespaces the dir by compiler identity, so a compiler change =>
// fresh dir => no stale DAG (the same guarantee the per-entry cache relies on).
// It reuses the SAME checksum + temp-file + atomic-rename writer as `put`, so a
// torn/foreign blob round-trips to a clean miss (the caller falls back to a full
// re-validation — never a partial/false-green).

/// Fold a build identity (entry canonical path + target) into the per-program
/// DAG artifact name. Target is folded unconditionally because codegen fps are
/// target-sensitive, so a per-(entry,target) DAG is the cleanest keying.
pub fn programKey(entry_canon: []const u8, target: []const u8) u64 {
    var h = std.hash.Wyhash.init(0x44_41_47_50); // "PGAD"
    h.update(entry_canon);
    h.update(&[_]u8{0});
    h.update(target);
    return h.final();
}

fn dagPath(c: Cache, buf: []u8, program_key: u64) []const u8 {
    return std.fmt.bufPrint(buf, "{s}/dag.{x:0>16}", .{ c.dir, program_key }) catch unreachable;
}

/// Store the serialized DAG blob under `program_key`. Same `[checksum][payload]`
/// framing + temp-file + atomic-rename as `put`, so concurrent invocations never
/// observe a torn artifact.
pub fn putDag(c: Cache, io: Io, program_key: u64, blob: []const u8) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = c.dagPath(&buf, program_key);
    const tmp = std.fmt.bufPrint(&tmp_buf, "{s}/dag.{x:0>16}.tmp", .{ c.dir, program_key }) catch return;

    var header: [checksum_len]u8 = undefined;
    std.mem.writeInt(u64, &header, std.hash.Wyhash.hash(checksum_seed, blob), .little);

    {
        var file = try Io.Dir.cwd().createFile(io, tmp, .{});
        defer file.close(io);
        try file.writeStreamingAll(io, &header);
        try file.writeStreamingAll(io, blob);
    }
    try Io.Dir.cwd().rename(tmp, Io.Dir.cwd(), path, io);
}

/// Load the serialized DAG blob for `program_key`, or null on a miss (absent or
/// checksum-corrupt). Caller owns the returned payload bytes and must pass them
/// to `Dag.deserialize`, which independently re-validates magic/version/size — a
/// torn or foreign artifact thus fails twice over to a clean full re-validation.
pub fn getDag(c: Cache, gpa: std.mem.Allocator, io: Io, program_key: u64) !?[]u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = c.dagPath(&buf, program_key);

    const bytes = Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    errdefer gpa.free(bytes);

    if (bytes.len < checksum_len) {
        gpa.free(bytes);
        return null;
    }
    const stored = std.mem.readInt(u64, bytes[0..checksum_len], .little);
    const payload = bytes[checksum_len..];
    if (std.hash.Wyhash.hash(checksum_seed, payload) != stored) {
        gpa.free(bytes);
        return null;
    }
    // Hand back JUST the payload (drop the checksum header). Move it to the front
    // so the caller frees one allocation.
    std.mem.copyForwards(u8, bytes[0..payload.len], payload);
    return try gpa.realloc(bytes, payload.len);
}

// ---- tests: cache-key digest semantics ------------------------------------
// The e2e cold/warm run exercises the on-disk round-trip; these pin down the
// digest logic in isolation.

const testing = std.testing;

test "same phase + content => same digest" {
    const a = Cache.Key.fromSource(.lex, "native", "fn main() {}");
    const b = Cache.Key.fromSource(.lex, "native", "fn main() {}");
    try testing.expectEqual(a.digest(), b.digest());
}

test "different content => different digest" {
    const a = Cache.Key.fromSource(.lex, "native", "fn a() {}");
    const b = Cache.Key.fromSource(.lex, "native", "fn b() {}");
    try testing.expect(a.digest() != b.digest());
}

test "lexing is target-independent: target does not change the key" {
    // Lexing doesn't depend on the target, so the same source must reuse one
    // cache entry across targets rather than re-lex per target.
    const native = Cache.Key.fromSource(.lex, "native", "fn main() {}");
    const arm = Cache.Key.fromSource(.lex, "aarch64-macos", "fn main() {}");
    try testing.expectEqual(native.digest(), arm.digest());
}

test "programKey folds entry + target distinctly" {
    const a = Cache.programKey("/x/main.toy", "aarch64-macos");
    const same = Cache.programKey("/x/main.toy", "aarch64-macos");
    try testing.expectEqual(a, same);
    try testing.expect(a != Cache.programKey("/x/other.toy", "aarch64-macos"));
    try testing.expect(a != Cache.programKey("/x/main.toy", "native"));
}

test "putDag/getDag round-trips a blob; absent => null; corruption => null" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toyc-ut-dag";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const cache = try Cache.init(io, dir);

    const key = Cache.programKey("/p/main.toy", "aarch64-macos");

    // Absent => miss.
    try testing.expect((try cache.getDag(gpa, io, key)) == null);

    const blob = "TDAG-ish payload bytes \x00\x01\x02";
    try cache.putDag(io, key, blob);

    const got = (try cache.getDag(gpa, io, key)).?;
    defer gpa.free(got);
    try testing.expectEqualSlices(u8, blob, got);

    // A different key is still a miss (no cross-program aliasing).
    try testing.expect((try cache.getDag(gpa, io, Cache.programKey("/p/other.toy", "aarch64-macos"))) == null);
}
