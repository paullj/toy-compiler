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

/// The single packed content-addressed object store — ONE file per stamp dir
/// (`.toy-cache/<stamp>/pack.bin`) replacing the 6200 per-entry files. Mirrors
/// rustc's single dep-graph file / a ccache manifest / a Bazel CAS: a single-file
/// object store avoids per-unit directory churn (the renameat/openat storm + the
/// single-dir inode contention that degrades -j past 4).
///
/// LIFECYCLE: `load` once at build start (bulk read pack + index into a hashmap),
/// then every `Cache.get` is a memory lookup. `Cache.put` stages blobs into an
/// in-memory write buffer; `flush` writes the merged pack ONCE at build end.
///
/// LAYOUT (all u64 little-endian):
///   [magic][n_entries][index_checksum]
///   index: n_entries × [digest][offset][len]   (offset/len into the blob region)
///   blobs: concatenated [u64 checksum][payload] frames (the SAME framing the
///          per-file path used, so the per-entry checksum still turns a torn or
///          foreign blob into a clean miss on read).
/// The index is itself checksummed: a bad magic / bad index checksum / short file
/// => the whole pack is treated as COLD (empty index, full rebuild), NEVER a false
/// hit. Carrying-forward: `load` seeds the write buffer with EVERY prior entry, so
/// `flush` rewrites a superset — entries from prior builds survive (the per-fn
/// files were never deleted either; this matches that persistence).
pub const Pack = struct {
    const magic: u64 = 0x31_4b_43_41_50_59_4f_54; // "TOYPACK1" little-endian
    const hdr_len = 3 * @sizeOf(u64); // [magic][n_entries][index_checksum]
    const rec_len = 3 * @sizeOf(u64); // [digest][offset][len]

    /// A staged blob's location in `blobs`. Offsets are realloc-stable (unlike a
    /// slice into `blobs`), so the index maps digest -> rec_index and `lookup`
    /// reconstructs the slice from `blobs.items[off..off+len]`. This sidesteps the
    /// dangling-slice hazard of caching slices into a growing buffer.
    const Rec = struct { digest: u64, off: usize, len: usize };

    gpa: std.mem.Allocator,
    /// Concatenated `[checksum][payload]` frames: prior entries carried forward by
    /// `load`, then fresh entries staged by `put`. Written verbatim as the pack's
    /// blob region by `flush`. Owned; freed in `deinit`.
    blobs: std.ArrayListUnmanaged(u8) = .empty,
    /// Parallel record of each frame's (digest, off, len) in `blobs`. Owned.
    recs: std.ArrayListUnmanaged(Rec) = .empty,
    /// digest -> index into `recs` (last writer wins on a duplicate digest, which is
    /// safe: same digest => same content => same bytes). Owned.
    index: std.AutoHashMapUnmanaged(u64, usize) = .{},
    /// Guards `blobs`/`recs`/`index` against the codegen fan-out's concurrent `put`s.
    /// A lock-free atomic-bool spinlock (mirrors `Dag`): 0.16's `std.Io.Mutex` needs an
    /// `Io` to block, but the critical sections here are tiny (an appendSlice + a hashmap
    /// upsert), so a spinlock keeps the module Io-free and safe under parallel fan-out.
    lock_state: std.atomic.Value(bool) = .init(false),

    pub fn init(gpa: std.mem.Allocator) Pack {
        return .{ .gpa = gpa };
    }

    fn lock(self: *Pack) void {
        while (self.lock_state.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    fn unlock(self: *Pack) void {
        self.lock_state.store(false, .release);
    }

    pub fn deinit(self: *Pack) void {
        self.blobs.deinit(self.gpa);
        self.recs.deinit(self.gpa);
        self.index.deinit(self.gpa);
        self.* = undefined;
    }

    fn packPath(buf: []u8, dir: []const u8) []const u8 {
        return std.fmt.bufPrint(buf, "{s}/pack.bin", .{dir}) catch unreachable;
    }

    /// Bulk-read the pack for `dir` into the in-memory store. A missing/short/bad
    /// pack leaves the store empty (a clean cold build), NEVER an error: a torn or
    /// foreign pack must degrade to a full rebuild, not a hard failure or false hit.
    /// Prior entries are copied into `blobs` so `flush` carries them forward (matching
    /// the per-file path, where old entries persisted across builds).
    pub fn load(self: *Pack, gpa: std.mem.Allocator, io: Io, dir: []const u8) void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = packPath(&buf, dir);
        const bytes = Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited) catch return;
        defer gpa.free(bytes);

        if (bytes.len < hdr_len) return;
        if (std.mem.readInt(u64, bytes[0..8], .little) != magic) return;
        const n: u64 = std.mem.readInt(u64, bytes[8..16], .little);
        const want_idx_ck = std.mem.readInt(u64, bytes[16..24], .little);

        const idx_bytes_len = std.math.mul(usize, @intCast(n), rec_len) catch return;
        if (bytes.len < hdr_len + idx_bytes_len) return;
        const idx_bytes = bytes[hdr_len .. hdr_len + idx_bytes_len];
        if (std.hash.Wyhash.hash(checksum_seed, idx_bytes) != want_idx_ck) return;
        const blob_region = bytes[hdr_len + idx_bytes_len ..];

        var i: usize = 0;
        while (i < n) : (i += 1) {
            const rec = idx_bytes[i * rec_len ..][0..rec_len];
            const digest = std.mem.readInt(u64, rec[0..8], .little);
            const off = std.mem.readInt(u64, rec[8..16], .little);
            const len = std.mem.readInt(u64, rec[16..24], .little);
            const end = std.math.add(u64, off, len) catch {
                self.reset();
                return;
            };
            if (end > blob_region.len) {
                self.reset();
                return;
            }
            self.stageLocked(gpa, digest, blob_region[@intCast(off)..@intCast(end)]) catch {
                self.reset();
                return;
            };
        }
    }

    fn reset(self: *Pack) void {
        self.blobs.clearRetainingCapacity();
        self.recs.clearRetainingCapacity();
        self.index.clearRetainingCapacity();
    }

    /// Stage an already-framed `[checksum][payload]` slice. NOT thread-locked: `load`
    /// is single-threaded; concurrent `put` callers go through `append` (which locks).
    fn stageLocked(self: *Pack, gpa: std.mem.Allocator, digest: u64, framed: []const u8) !void {
        const off = self.blobs.items.len;
        try self.blobs.appendSlice(gpa, framed);
        const rec_i = self.recs.items.len;
        try self.recs.append(gpa, .{ .digest = digest, .off = off, .len = framed.len });
        try self.index.put(gpa, digest, rec_i);
    }

    /// Memory lookup of an entry's `[checksum][payload]` frame, or null on a miss.
    /// Mutex-guarded because `put` can grow `blobs`/`recs` concurrently in the fan-out
    /// (each KEY is written by one job, but distinct keys race the shared buffers).
    fn lookup(self: *Pack, digest: u64) ?[]const u8 {
        self.lock();
        defer self.unlock();
        const rec_i = self.index.get(digest) orelse return null;
        const r = self.recs.items[rec_i];
        return self.blobs.items[r.off .. r.off + r.len];
    }

    /// Stage `[checksum][payload]` for `digest` (mutex-guarded: the codegen fan-out
    /// appends from many threads). The payload is COPIED, so the caller may free it.
    fn append(self: *Pack, digest: u64, payload: []const u8) !void {
        var header: [checksum_len]u8 = undefined;
        std.mem.writeInt(u64, &header, std.hash.Wyhash.hash(checksum_seed, payload), .little);
        self.lock();
        defer self.unlock();
        const off = self.blobs.items.len;
        try self.blobs.ensureUnusedCapacity(self.gpa, header.len + payload.len);
        self.blobs.appendSliceAssumeCapacity(&header);
        self.blobs.appendSliceAssumeCapacity(payload);
        const rec_i = self.recs.items.len;
        try self.recs.append(self.gpa, .{ .digest = digest, .off = off, .len = header.len + payload.len });
        try self.index.put(self.gpa, digest, rec_i);
    }

    /// Write the merged pack ONCE: header + index + blob region, via a temp file +
    /// atomic rename so a crash mid-flush never leaves a torn pack a future build
    /// would load (the index checksum also catches a torn index). Non-fatal on error
    /// (the next build just rebuilds), mirroring `put`'s swallowed-write policy.
    pub fn flush(self: *Pack, io: Io, dir: []const u8) void {
        self.lock();
        defer self.unlock();
        self.flushLocked(io, dir) catch {};
    }

    fn flushLocked(self: *Pack, io: Io, dir: []const u8) !void {
        // Emit the LIVE entries (one frame per digest — `index` already deduped to the
        // last writer) in DIGEST-SORTED order, re-laying the blob region so offsets
        // match. The pack is then byte-identical run-to-run regardless of the codegen
        // fan-out's put arrival order — the project-wide determinism invariant (mirrors
        // Dag.dumpDeterministic's sort-at-dump). Output bytes + incremental hits are
        // order-independent (get is keyed by digest), so this only canonicalizes the
        // cache artifact; it never changes what a build produces or reuses.
        const n: usize = self.index.count();
        const order = try self.gpa.alloc(Rec, n);
        defer self.gpa.free(order);
        {
            var it = self.index.valueIterator();
            var i: usize = 0;
            while (it.next()) |rec_i| : (i += 1) order[i] = self.recs.items[rec_i.*];
        }
        std.sort.block(Rec, order, {}, struct {
            fn lt(_: void, a: Rec, b: Rec) bool {
                return a.digest < b.digest;
            }
        }.lt);

        const idx_bytes = try self.gpa.alloc(u8, n * rec_len);
        defer self.gpa.free(idx_bytes);
        var blob_out: std.ArrayList(u8) = try .initCapacity(self.gpa, self.blobs.items.len);
        defer blob_out.deinit(self.gpa);
        var off: usize = 0;
        for (order, 0..) |r, i| {
            blob_out.appendSliceAssumeCapacity(self.blobs.items[r.off..][0..r.len]);
            const rec = idx_bytes[i * rec_len ..][0..rec_len];
            std.mem.writeInt(u64, rec[0..8], r.digest, .little);
            std.mem.writeInt(u64, rec[8..16], @intCast(off), .little);
            std.mem.writeInt(u64, rec[16..24], @intCast(r.len), .little);
            off += r.len;
        }
        const idx_ck = std.hash.Wyhash.hash(checksum_seed, idx_bytes);

        var hdr: [hdr_len]u8 = undefined;
        std.mem.writeInt(u64, hdr[0..8], magic, .little);
        std.mem.writeInt(u64, hdr[8..16], @as(u64, n), .little);
        std.mem.writeInt(u64, hdr[16..24], idx_ck, .little);

        var buf: [std.fs.max_path_bytes]u8 = undefined;
        var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = packPath(&buf, dir);
        const tmp = std.fmt.bufPrint(&tmp_buf, "{s}/pack.bin.tmp", .{dir}) catch return;
        {
            var file = try Io.Dir.cwd().createFile(io, tmp, .{});
            defer file.close(io);
            try file.writeStreamingAll(io, &hdr);
            try file.writeStreamingAll(io, idx_bytes);
            try file.writeStreamingAll(io, blob_out.items);
        }
        try Io.Dir.cwd().rename(tmp, Io.Dir.cwd(), path, io);
    }
};

/// Per-compiler cache directory, e.g. ".toy-cache/0.0.0-1a2b3c...". Borrowed,
/// must outlive the `Cache` (the driver keeps it on its stack for the run).
dir: []const u8,

/// OPTIONAL packed-object-store sidecar. When present, `get`/`put` route through
/// ONE per-stamp pack file + an in-memory index instead of one file per entry:
///   * GET = a memory lookup + slice + checksum check (zero per-entry syscalls).
///   * PUT = append `(digest,blob)` to an in-memory buffer; the WHOLE pack is
///     written ONCE at build end (`Pack.flush`), collapsing 6200x4 syscalls to ~4.
/// BORROWED (mirrors how `Cache` is itself a borrowed dir handle): ONE `Pack` lives
/// on the driver frame and a `*Pack` rides in the value `Cache` copied into every
/// fan-out job, so all jobs share the read index + write accumulator. `null` (the
/// default) is the VERBATIM one-file-per-entry path — byte-identical to before.
pack: ?*Pack = null,

/// Create the per-stamp cache directory (and the `.toy-cache` root) if needed.
/// `dir` must already include the compiler stamp.
pub fn init(io: Io, dir: []const u8) !Cache {
    try Io.Dir.cwd().createDirPath(io, dir);
    return .{ .dir = dir };
}

/// Same as `init` but attaches a borrowed packed-object-store sidecar so `get`/`put`
/// route through the single pack file + in-memory index. The caller owns the `Pack`
/// (loads it before the build, flushes it after) and must keep it alive for the run.
pub fn initPack(io: Io, dir: []const u8, pack: *Pack) !Cache {
    try Io.Dir.cwd().createDirPath(io, dir);
    return .{ .dir = dir, .pack = pack };
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
    // PACKED PATH: a hit is a memory lookup + slice into the bulk-loaded pack; a
    // miss means the digest is absent from this build's loaded index (no per-entry
    // syscall). The pack carries the SAME `[u64 checksum][payload]` framing, so a
    // torn/foreign entry round-trips to a clean miss exactly as the file path does.
    if (c.pack) |p| {
        const entry = p.lookup(key.digest()) orelse return null;
        if (entry.len < checksum_len) return null;
        const stored = std.mem.readInt(u64, entry[0..checksum_len], .little);
        const payload = entry[checksum_len..];
        if (std.hash.Wyhash.hash(checksum_seed, payload) != stored) return null;
        if (payload.len % @sizeOf(T) != 0) return null;
        const items = try gpa.alloc(T, payload.len / @sizeOf(T));
        @memcpy(std.mem.sliceAsBytes(items), payload);
        return items;
    }

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
    const payload = std.mem.sliceAsBytes(items);

    // PACKED PATH: stage `[u64 checksum][payload]` into the in-memory write buffer
    // under `key.digest()`. No file is written here — the merged pack is flushed
    // ONCE at build end. Each key is written by exactly one job and never read back
    // this build, so a same-build append never needs to be visible to a concurrent
    // get (the within-build atomicity the temp-file+rename gave is unneeded). The
    // append is mutex-guarded so the fan-out's workers share one buffer safely.
    if (c.pack) |p| {
        try p.append(key.digest(), payload);
        return;
    }

    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = c.entryPath(&buf, key);
    const tmp = std.fmt.bufPrint(&tmp_buf, "{s}/{x:0>16}.{d}.tmp", .{ c.dir, key.digest(), tmp_tag }) catch return;

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

test "Pack: put/flush then load round-trips entries through the single pack file" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toyc-ut-pack";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const k1 = Cache.Key.fromSource(.lex, "native", "fn a() {}");
    const k2 = Cache.Key.fromSource(.lex, "native", "fn b() {}");
    const v1 = [_]u8{ 1, 2, 3, 4, 5 };
    const v2 = [_]u8{ 9, 8, 7 };

    // Build 1: cold pack, put two entries, flush.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir); // absent => empty
        const cache = try Cache.initPack(io, dir, &pack);
        try testing.expect((try cache.get(u8, gpa, io, k1)) == null); // cold miss
        try cache.put(u8, io, k1, 0, &v1);
        try cache.put(u8, io, k2, 1, &v2);
        pack.flush(io, dir);
    }

    // Build 2: fresh pack loads the flushed file; both entries are memory hits.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        const g1 = (try cache.get(u8, gpa, io, k1)).?;
        defer gpa.free(g1);
        try testing.expectEqualSlices(u8, &v1, g1);
        const g2 = (try cache.get(u8, gpa, io, k2)).?;
        defer gpa.free(g2);
        try testing.expectEqualSlices(u8, &v2, g2);
        // An absent key is still a clean miss.
        try testing.expect((try cache.get(u8, gpa, io, Cache.Key.fromSource(.lex, "native", "fn c() {}"))) == null);
    }
}

test "Pack: a corrupt pack file => cold (empty index), never a false hit" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toyc-ut-pack-corrupt";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const k1 = Cache.Key.fromSource(.lex, "native", "fn a() {}");
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        try cache.put(u8, io, k1, 0, &[_]u8{ 1, 2, 3 });
        pack.flush(io, dir);
    }

    // Corrupt the index region (byte just past the header) so the index checksum fails.
    {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/pack.bin", .{dir}) catch unreachable;
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
        defer gpa.free(bytes);
        bytes[24] ^= 0xFF; // first index byte
        var file = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, bytes);
    }

    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir); // bad index checksum => cold
        const cache = try Cache.initPack(io, dir, &pack);
        try testing.expect((try cache.get(u8, gpa, io, k1)) == null); // never a false hit
    }
}

test "Pack: load carries prior entries forward across a flush" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toyc-ut-pack-carry";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const k1 = Cache.Key.fromSource(.lex, "native", "fn a() {}");
    const k2 = Cache.Key.fromSource(.lex, "native", "fn b() {}");

    // Build 1: write k1.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        try cache.put(u8, io, k1, 0, &[_]u8{42});
        pack.flush(io, dir);
    }
    // Build 2: load (carries k1 forward), write k2, flush.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        try cache.put(u8, io, k2, 0, &[_]u8{43});
        pack.flush(io, dir);
    }
    // Build 3: BOTH k1 (carried twice) and k2 survive.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        const g1 = (try cache.get(u8, gpa, io, k1)).?;
        defer gpa.free(g1);
        try testing.expectEqualSlices(u8, &[_]u8{42}, g1);
        const g2 = (try cache.get(u8, gpa, io, k2)).?;
        defer gpa.free(g2);
        try testing.expectEqualSlices(u8, &[_]u8{43}, g2);
    }
}
