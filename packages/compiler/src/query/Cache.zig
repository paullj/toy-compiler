//! Content-addressed, on-disk cache of compilation results, organized as a
//! demand-driven (query) memo table: each phase asks the cache for a `Key`, and
//! a hit lets the driver skip the work entirely.
//!
//! Cache key = (compiler identity, phase, target, inputs):
//!
//!   * Compiler identity is the *directory* — every entry lives under
//!     `.toy/<stamp>/cache/` (see `version.zig`). A different compiler build
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
const Phase_ = @import("Phase.zig");

const Cache = @This();

/// Compilation phases that participate in the cache. The on-disk cache and the
/// static `StageGraph` share ONE phase taxonomy (`Phase.Kind`): the phases the
/// cache keys are exactly its CACHEABLE subset (`Phase.Kind.cacheable`: lex/parse/
/// codegen). Re-exported under this name so the cache reads as "phases".
///
/// codegen (Ast→Ir→aarch64) is cached by a transitive content fingerprint,
/// not a source hash, and is target-sensitive (the blob is aarch64 machine code).
/// The IR is built INSIDE that query and never cached separately — the cache stays
/// ONE-TIER.
pub const Phase = Phase_.Kind;

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

    /// Key a phase directly by a precomputed content fingerprint (codegen),
    /// rather than by rehashing a source byte range. `digest()` folds `input`
    /// verbatim, so the fingerprint IS the cache identity.
    pub fn fromFingerprint(phase: Phase, target: []const u8, fp: u64) Key {
        return .{ .phase = phase, .target = target, .input = fp };
    }

    /// Fold the key components into the on-disk entry name. Only the CACHEABLE
    /// subset of `Phase` may name an on-disk entry; an in-memory-only kind here is a
    /// caller bug (the assert compiles out under ReleaseFast — it never folds, so
    /// the digest bytes are unchanged).
    pub fn digest(k: Key) u64 {
        std.debug.assert(k.phase.cacheable());
        var h = std.hash.Wyhash.init(0);
        h.update(&[_]u8{@intFromEnum(k.phase)});
        if (k.phase.targetSensitive()) h.update(k.target);
        h.update(std.mem.asBytes(&k.input));
        return h.final();
    }
};

/// The single packed content-addressed object store — ONE file per stamp dir
/// (`.toy/<stamp>/cache/pack.bin`) replacing the 6200 per-entry files. Mirrors
/// rustc's single dep-graph file / a ccache manifest / a Bazel CAS: a single-file
/// object store avoids per-unit directory churn (the renameat/openat storm + the
/// single-dir inode contention that degrades -j past 4).
///
/// LIFECYCLE: `load` once at build start (bulk read pack + index into a hashmap),
/// then every `Cache.get` is a memory lookup. `Cache.put` stages blobs into an
/// in-memory write buffer; `flush` writes the merged pack ONCE at build end.
///
/// LAYOUT (all u64 little-endian):
///   [magic][n_entries][index_checksum][n_manifest][manifest_checksum]
///   manifest: n_manifest × [path_len:u32][path bytes][mtime][size][ctime][content_fp]
///             (the warm-discover unchanged-predicate; see `ManifestEntry`)
///   index:    n_entries × [digest][offset][len]   (offsets into the blob region)
///   blobs:    codegen [u64 checksum][payload] frames, concatenated (the SAME framing the
///             per-file path used, so the per-entry checksum still turns a torn or foreign
///             blob into a clean miss).
/// The index is itself checksummed: a bad magic / bad index checksum / short file
/// => the whole pack is treated as COLD (empty index, full rebuild), NEVER a false
/// hit. The manifest carries its OWN checksum and sits BEFORE the codegen index (so the
/// index offset is a fixed function of the manifest extent). Two corruption cases, both
/// SOUND (degrade to a re-read/rebuild, never a false hit):
///   * a manifest CONTENT/checksum corruption drops ONLY the manifest (every
///     warm-discover lookup then misses and discover re-reads), codegen still loads;
///   * a manifest FRAMING corruption (a bad n_manifest or path_len) shifts the computed
///     idx_base, so the codegen index checksum then fails and the WHOLE pack goes cold
///     (codegen dropped too) — still a clean rebuild, just a coarser degrade.
/// Carrying-forward: `load`
/// seeds the write buffer with EVERY prior entry, so `flush` rewrites a superset —
/// entries from prior builds survive (the per-fn files were never deleted either; this
/// matches that persistence). The manifest is rebuilt fresh each build from the files
/// discover actually touched (it is NOT carried forward blindly): a file dropped from
/// the graph drops out of the manifest, and a re-read file overwrites its entry.
pub const Pack = struct {
    // "TOYPACK5": bumped when `ctime` was added to each manifest record (TOYPACK4 added
    // the manifest section itself). An older-magic file fails the magic check in `load` =>
    // treated as COLD (full rebuild), never a false hit — the same degrade-to-miss the
    // checksums give. The bump is LOAD-BEARING for the ctime hardening: a TOYPACK4 manifest
    // record has no ctime field, so reusing the magic would let `load` read an old record
    // as if it carried a ctime (a framing mismatch) — the bump cold-misses it cleanly
    // instead, so warm-serve never trusts a ctime-less entry.
    const magic: u64 = 0x35_4b_43_41_50_59_4f_54;
    // [magic][n_entries][index_checksum][n_manifest][manifest_checksum]
    const hdr_len = 5 * @sizeOf(u64);
    const rec_len = 3 * @sizeOf(u64); // [digest][offset][len]

    /// A staged blob's location in `blobs`. Offsets are realloc-stable (unlike a
    /// slice into `blobs`), so the index maps digest -> rec_index and `lookup`
    /// reconstructs the slice from `blobs.items[off..off+len]`. This sidesteps the
    /// dangling-slice hazard of caching slices into a growing buffer.
    const Rec = struct { digest: u64, off: usize, len: usize };

    /// One warm-discover manifest record: the last build's snapshot of a module file.
    /// `path` is the file's CANONICAL path (the same physical-identity key discover
    /// interns by, so symlink/case duplicates collapse to one entry).
    ///
    /// SOUNDNESS CONTRACT — read carefully, a wrong oracle here is a silent MISCOMPILE.
    /// `(mtime, size, ctime)` is the UNCHANGED-PREDICATE: when a fresh `stat` matches ALL
    /// THREE, discover treats the file as unchanged since the build that recorded this entry
    /// and serves the cached source + tokens + AST keyed by `content_fp` WITHOUT reading the
    /// file. `content_fp` is the `Wyhash(0, source)` of the bytes that build read; it is
    /// the CACHE KEY for the {source, lex, parse} blobs, not a re-checked authority (we do
    /// not read the file, so there is nothing to rehash).
    ///
    /// This rests on the BUILD CONTRACT: the source tree is not mutated mid-build and a
    /// write updates the file's mtime (POSIX writes do; so do editors, `cp`, and the
    /// compilers/codegen tools that emit source). That is the SAME assumption discover's
    /// `realpath_cache` already relies on — interning by physical identity assumes a path
    /// resolves the same way for the whole build because the tree is immutable mid-build.
    /// The manifest extends that one build-window further (across builds), keyed off mtime.
    /// Every UNCERTAIN case falls back to a full read + lex + parse: no entry, a failed
    /// stat, a different size, a moved mtime, a moved ctime, or any of the three cache blobs
    /// missing.
    ///
    /// `ctime` (the inode STATUS-CHANGE time) closes the same-size-mtime-restore window that
    /// `(mtime, size)` alone could not: ANY metadata write moves ctime, so the out-of-band
    /// shapes that fake a matching mtime+size are caught by a moved ctime WITHOUT a read —
    /// ctime rides the SAME `stat` that already reads mtime+size, so the warm read-skip win
    /// is unchanged. Three such shapes:
    ///   * `touch -r ref f` (and `cp -p`/`rsync --times` that copy mtime+size): the `touch`
    ///     itself is a metadata write that bumps `f`'s ctime, so ctime no longer matches;
    ///   * `cp -p src f` onto a fresh inode: a new inode carries a fresh ctime, so it can
    ///     never match the recorded one;
    ///   * a same-mtime edit on a coarse-granularity FS tick (the mtime resolution didn't
    ///     advance): the write still moves ctime, so it is caught.
    /// The ONLY residual hole is a tool that fakes ALL of mtime + size + ctime + content in
    /// one go (e.g. restoring a verbatim inode image) — not a shape a normal build produces.
    /// Re-reading + rehashing every file every warm build (the only way to close even that)
    /// defeats the entire purpose of the manifest: the read IS the warm-discover bottleneck,
    /// since lex/parse are already cache hits.
    pub const ManifestEntry = struct {
        mtime: i64,
        size: u64,
        ctime: i64,
        content_fp: u64,

        /// The unchanged-predicate as ONE call: true iff a fresh `stat` matches all three of
        /// (mtime, size, ctime). An unrepresentable timestamp (a `std.math.cast` fail) returns
        /// false so the caller falls back to a full read — never a trap or a coerced compare.
        pub fn matches(entry: ManifestEntry, st: Io.Dir.Stat) bool {
            const mtime = std.math.cast(i64, st.mtime.nanoseconds) orelse return false;
            const ctime = std.math.cast(i64, st.ctime.nanoseconds) orelse return false;
            return mtime == entry.mtime and st.size == entry.size and ctime == entry.ctime;
        }
    };

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
    /// The PREVIOUS build's warm-discover manifest, keyed by canonical path: what
    /// discover consults to decide if a file is unchanged (`manifestGet`). Seeded by
    /// `load`; read-only thereafter. Owns its keys (the canonical path strings).
    manifest_prev: std.StringHashMapUnmanaged(ManifestEntry) = .{},
    /// THIS build's manifest, accumulated by discover as it touches each file
    /// (`manifestPut`) — whether it served the file from cache (carries the prior
    /// entry forward) or re-read it (records the fresh mtime/size/content_fp). This is
    /// what `flush` serializes, so a file no longer in the graph drops out next build.
    /// Owns its keys; guarded by `lock` (discover is serial today, but the lock keeps
    /// it safe if a future change parallelizes load).
    manifest_cur: std.StringHashMapUnmanaged(ManifestEntry) = .{},
    /// Guards `blobs`/`recs`/`index` against the codegen fan-out's concurrent `put`s.
    /// A lock-free atomic-bool spinlock: 0.16's `std.Io.Mutex` needs an `Io` to block,
    /// but the critical sections here are tiny (an appendSlice + a hashmap upsert), so a
    /// spinlock keeps the module Io-free and safe under parallel fan-out.
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
        freeManifest(self.gpa, &self.manifest_prev);
        freeManifest(self.gpa, &self.manifest_cur);
        self.* = undefined;
    }

    fn freeManifest(gpa: std.mem.Allocator, m: *std.StringHashMapUnmanaged(ManifestEntry)) void {
        var it = m.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        m.deinit(gpa);
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
        const n_manifest: u64 = std.mem.readInt(u64, bytes[24..32], .little);
        const want_manifest_ck = std.mem.readInt(u64, bytes[32..40], .little);

        // The manifest records sit between the header and the codegen index. Parse
        // their EXTENT first (so the codegen index offset is known) and validate the
        // checksum; a bad manifest drops ONLY the manifest (cleared below) — the
        // codegen index still loads, so warm-discover just misses and re-reads.
        const manifest_bytes_len = manifestExtent(bytes[hdr_len..], n_manifest) catch return;
        const manifest_bytes = bytes[hdr_len .. hdr_len + manifest_bytes_len];
        const manifest_ok = std.hash.Wyhash.hash(checksum_seed, manifest_bytes) == want_manifest_ck;
        if (manifest_ok) self.loadManifest(gpa, manifest_bytes, n_manifest) catch self.clearManifestPrev();

        const idx_base = hdr_len + manifest_bytes_len;
        const idx_bytes_len = std.math.mul(usize, @intCast(n), rec_len) catch return;
        if (bytes.len < idx_base + idx_bytes_len) return;
        const idx_bytes = bytes[idx_base .. idx_base + idx_bytes_len];
        if (std.hash.Wyhash.hash(checksum_seed, idx_bytes) != want_idx_ck) return;

        const blob_region = bytes[idx_base + idx_bytes_len ..];

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
        // The codegen tier only — `manifest_prev` is an INDEPENDENT section. A corrupt
        // blob region wipes codegen entries; the manifest (already parsed + checksum-
        // validated above) can stay, since every warm-serve still re-checks the
        // lex/parse cache via `lookup` (which now misses) and falls back to a re-read.
        self.blobs.clearRetainingCapacity();
        self.recs.clearRetainingCapacity();
        self.index.clearRetainingCapacity();
    }

    // A manifest record's TRAILING fixed fields after `[path_len:u32][path]`:
    // [mtime:u64][size:u64][ctime:u64][content_fp:u64].
    const manifest_fixed = 4 * @sizeOf(u64);

    fn clearManifestPrev(self: *Pack) void {
        freeManifest(self.gpa, &self.manifest_prev);
        self.manifest_prev = .{};
    }

    /// Walk `n_manifest` variable-length records from the start of `region` and return
    /// their total byte length, bounds-checking every field. Errors (a truncated
    /// record, a path_len that runs off the end) so `load` degrades to cold rather than
    /// reading past the buffer. Does NOT validate content — just the framing extent.
    fn manifestExtent(region: []const u8, n_manifest: u64) !usize {
        var off: usize = 0;
        var i: u64 = 0;
        while (i < n_manifest) : (i += 1) {
            if (off + @sizeOf(u32) > region.len) return error.Truncated;
            const path_len = std.mem.readInt(u32, region[off..][0..4], .little);
            const rec_len_total = std.math.add(usize, @sizeOf(u32) + manifest_fixed, path_len) catch return error.Truncated;
            const next = std.math.add(usize, off, rec_len_total) catch return error.Truncated;
            if (next > region.len) return error.Truncated;
            off = next;
        }
        return off;
    }

    /// Parse the manifest records into `manifest_prev` (keyed by the OWNED canonical
    /// path). `region` is exactly the bytes `manifestExtent` measured, so every field
    /// is already in-bounds; this only decodes them. Last-writer-wins on a duplicate
    /// path (flush emits each path once, so duplicates never occur in a sound pack).
    fn loadManifest(self: *Pack, gpa: std.mem.Allocator, region: []const u8, n_manifest: u64) !void {
        var off: usize = 0;
        var i: u64 = 0;
        while (i < n_manifest) : (i += 1) {
            const path_len = std.mem.readInt(u32, region[off..][0..4], .little);
            off += @sizeOf(u32);
            const path = region[off .. off + path_len];
            off += path_len;
            const mtime = std.mem.readInt(i64, region[off..][0..8], .little);
            const size = std.mem.readInt(u64, region[off + 8 ..][0..8], .little);
            const ctime = std.mem.readInt(i64, region[off + 16 ..][0..8], .little);
            const content_fp = std.mem.readInt(u64, region[off + 24 ..][0..8], .little);
            off += manifest_fixed;

            const key = try gpa.dupe(u8, path);
            errdefer gpa.free(key);
            const gop = try self.manifest_prev.getOrPut(gpa, key);
            if (gop.found_existing) gpa.free(key); // duplicate path: keep the first key, overwrite value
            gop.value_ptr.* = .{ .mtime = mtime, .size = size, .ctime = ctime, .content_fp = content_fp };
        }
    }

    /// Warm-discover lookup: the PREVIOUS build's snapshot for `canon_path`, or null if
    /// this file was not in the last build's manifest (=> a full read + lex + parse). A
    /// `(mtime, size, ctime)` all-match against the returned entry is the UNCHANGED-PREDICATE
    /// the caller serves the cached {source, tokens, AST} off of without reading the file (see
    /// `ManifestEntry`'s soundness contract for why this is sound under the build contract).
    pub fn manifestGet(self: *Pack, canon_path: []const u8) ?ManifestEntry {
        return self.manifest_prev.get(canon_path);
    }

    /// Record THIS build's manifest entry for `canon_path` (copied). Called by discover
    /// for every file it touches — both a warm serve (carries the prior entry forward)
    /// and a re-read (the fresh mtime/size/content_fp). Last-writer-wins. Lock-guarded
    /// so a future parallel discover is safe; today discover is serial. A failed put is
    /// swallowed by the caller (worst case: next build re-reads that one file).
    pub fn manifestPut(self: *Pack, canon_path: []const u8, e: ManifestEntry) !void {
        self.lock();
        defer self.unlock();
        const gop = try self.manifest_cur.getOrPut(self.gpa, canon_path);
        if (!gop.found_existing) {
            gop.key_ptr.* = self.gpa.dupe(u8, canon_path) catch |err| {
                _ = self.manifest_cur.remove(canon_path);
                return err;
            };
        }
        gop.value_ptr.* = e;
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

    /// Canonicalize the codegen section into a written index region + a blob region:
    /// emit the LIVE entries (one frame per key — the index already deduped to the last
    /// writer) in KEY-SORTED order, re-laying the blob region so offsets match. The
    /// result is byte-identical run-to-run regardless of arrival order (the project-wide
    /// determinism invariant). Caller owns `idx_bytes` (allocated here); `blob_out` is
    /// appended to (not reset).
    fn layoutSection(
        gpa: std.mem.Allocator,
        index: *const std.AutoHashMapUnmanaged(u64, usize),
        recs: []const Rec,
        blobs: []const u8,
        blob_out: *std.ArrayList(u8),
    ) !struct { idx_bytes: []u8, idx_ck: u64 } {
        const n: usize = index.count();
        const order = try gpa.alloc(Rec, n);
        defer gpa.free(order);
        {
            var it = index.valueIterator();
            var i: usize = 0;
            while (it.next()) |rec_i| : (i += 1) order[i] = recs[rec_i.*];
        }
        std.sort.block(Rec, order, {}, struct {
            fn lt(_: void, a: Rec, b: Rec) bool {
                return a.digest < b.digest;
            }
        }.lt);

        const idx_bytes = try gpa.alloc(u8, n * rec_len);
        errdefer gpa.free(idx_bytes);
        // Offsets are RELATIVE to this section's blob region (where blob_out currently
        // starts), so each section's index is self-contained and load can split the two.
        const base = blob_out.items.len;
        try blob_out.ensureUnusedCapacity(gpa, blobs.len);
        var off: usize = 0;
        for (order, 0..) |r, i| {
            blob_out.appendSliceAssumeCapacity(blobs[r.off..][0..r.len]);
            const rec = idx_bytes[i * rec_len ..][0..rec_len];
            std.mem.writeInt(u64, rec[0..8], r.digest, .little);
            std.mem.writeInt(u64, rec[8..16], @intCast(off), .little);
            std.mem.writeInt(u64, rec[16..24], @intCast(r.len), .little);
            off += r.len;
        }
        std.debug.assert(blob_out.items.len - base == off);
        return .{ .idx_bytes = idx_bytes, .idx_ck = std.hash.Wyhash.hash(checksum_seed, idx_bytes) };
    }

    /// Serialize `manifest_cur` into a single byte buffer of variable-length records,
    /// PATH-SORTED for byte-identical output run-to-run regardless of insertion order
    /// (the same determinism the codegen index gets). Caller owns the returned bytes.
    /// Record: `[path_len:u32][path][mtime:u64][size:u64][ctime:u64][content_fp:u64]`.
    fn layoutManifest(self: *Pack) !struct { bytes: []u8, ck: u64 } {
        const n: usize = self.manifest_cur.count();
        const Pair = struct { path: []const u8, e: ManifestEntry };
        const order = try self.gpa.alloc(Pair, n);
        defer self.gpa.free(order);
        {
            var it = self.manifest_cur.iterator();
            var i: usize = 0;
            while (it.next()) |kv| : (i += 1) order[i] = .{ .path = kv.key_ptr.*, .e = kv.value_ptr.* };
        }
        std.sort.block(Pair, order, {}, struct {
            fn lt(_: void, a: Pair, b: Pair) bool {
                return std.mem.lessThan(u8, a.path, b.path);
            }
        }.lt);

        var total: usize = 0;
        for (order) |p| total += @sizeOf(u32) + p.path.len + manifest_fixed;
        const bytes = try self.gpa.alloc(u8, total);
        errdefer self.gpa.free(bytes);
        var off: usize = 0;
        for (order) |p| {
            std.mem.writeInt(u32, bytes[off..][0..4], @intCast(p.path.len), .little);
            off += @sizeOf(u32);
            @memcpy(bytes[off .. off + p.path.len], p.path);
            off += p.path.len;
            std.mem.writeInt(i64, bytes[off..][0..8], p.e.mtime, .little);
            std.mem.writeInt(u64, bytes[off + 8 ..][0..8], p.e.size, .little);
            std.mem.writeInt(i64, bytes[off + 16 ..][0..8], p.e.ctime, .little);
            std.mem.writeInt(u64, bytes[off + 24 ..][0..8], p.e.content_fp, .little);
            off += manifest_fixed;
        }
        std.debug.assert(off == total);
        return .{ .bytes = bytes, .ck = std.hash.Wyhash.hash(checksum_seed, bytes) };
    }

    fn flushLocked(self: *Pack, io: Io, dir: []const u8) !void {
        var blob_out: std.ArrayList(u8) = try .initCapacity(self.gpa, self.blobs.items.len);
        defer blob_out.deinit(self.gpa);

        const cg = try layoutSection(self.gpa, &self.index, self.recs.items, self.blobs.items, &blob_out);
        defer self.gpa.free(cg.idx_bytes);

        const man = try self.layoutManifest();
        defer self.gpa.free(man.bytes);

        const n: u64 = self.index.count();
        const n_manifest: u64 = self.manifest_cur.count();

        var hdr: [hdr_len]u8 = undefined;
        std.mem.writeInt(u64, hdr[0..8], magic, .little);
        std.mem.writeInt(u64, hdr[8..16], n, .little);
        std.mem.writeInt(u64, hdr[16..24], cg.idx_ck, .little);
        std.mem.writeInt(u64, hdr[24..32], n_manifest, .little);
        std.mem.writeInt(u64, hdr[32..40], man.ck, .little);

        var buf: [std.fs.max_path_bytes]u8 = undefined;
        var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = packPath(&buf, dir);
        const tmp = std.fmt.bufPrint(&tmp_buf, "{s}/pack.bin.tmp", .{dir}) catch return;
        {
            var file = try Io.Dir.cwd().createFile(io, tmp, .{});
            defer file.close(io);
            try file.writeStreamingAll(io, &hdr);
            // Manifest BEFORE the codegen index so the index offset is a fixed function
            // of the manifest extent (load splits the two sections by that extent).
            try file.writeStreamingAll(io, man.bytes);
            try file.writeStreamingAll(io, cg.idx_bytes);
            try file.writeStreamingAll(io, blob_out.items);
        }
        try Io.Dir.cwd().rename(tmp, Io.Dir.cwd(), path, io);
    }
};

/// Per-compiler cache directory, e.g. ".toy/0.0.0+1a2b3c.../cache". Borrowed,
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

/// False => every `get` misses and every `put` is dropped, with no I/O. A long-lived
/// editor session checks a fresh buffer per keystroke: a disk cache would grow without
/// bound, and a browser host has no disk at all.
enabled: bool = true,

pub const disabled: Cache = .{ .dir = "", .enabled = false };

/// Create the per-stamp cache directory (and the `.toy` root) if needed.
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
    if (!c.enabled) return null;
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
    if (!c.enabled) return;
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

/// Warm-discover unchanged-predicate lookup (see `Pack.ManifestEntry`): the PREVIOUS
/// build's snapshot for `canon_path`, or null when there is no pack sidecar (the per-file
/// path never persists a manifest) or no prior entry. A null is always a SAFE miss — the
/// caller re-reads + re-lexes + re-parses. A non-null entry whose `(mtime, size, ctime)` all
/// match a fresh stat authorizes serving the cached {source, tokens, AST} without reading the file.
pub fn manifestGet(c: Cache, canon_path: []const u8) ?Pack.ManifestEntry {
    const p = c.pack orelse return null;
    return p.manifestGet(canon_path);
}

/// Record THIS build's manifest entry for `canon_path`. A no-op without a pack
/// sidecar. Non-fatal on OOM (the caller swallows it: worst case the file is re-read
/// next build), mirroring `put`'s swallowed-write policy.
pub fn manifestPut(c: Cache, canon_path: []const u8, e: Pack.ManifestEntry) void {
    const p = c.pack orelse return;
    p.manifestPut(canon_path, e) catch {};
}

// ---- tests: cache-key digest semantics ------------------------------------
// The e2e cold/warm run exercises the on-disk round-trip; these pin down the
// digest logic in isolation.

const testing = std.testing;

test "a disabled cache misses every get and drops every put without touching io" {
    const c = Cache.disabled;
    const k = Key.fromSource(.lex, "native", "fn main() {}");
    try c.put(u8, Io.failing, k, 0, &[_]u8{ 1, 2, 3 });
    try testing.expect((try c.get(u8, testing.allocator, Io.failing, k)) == null);
}

test "cacheable subset's enum bytes are PINNED (digest byte-identity across the enum collapse)" {
    // `digest()` folds `@intFromEnum(phase)`, so the on-disk entry name for every
    // cacheable phase depends on these exact byte values. A different byte is a
    // silent cold rebuild of every entry of that phase. Exhaustive (no else): every
    // cacheable phase must pin its byte HERE, and any new kind must be classified,
    // or this fails to compile.
    inline for (@typeInfo(Phase).@"enum".fields) |f| {
        const k: Phase = @enumFromInt(f.value);
        const pinned: ?u8 = switch (k) {
            .lex => 0,
            .parse => 1,
            .codegen => 3,
            .source => 11,
            .signature, .body, .type_of, .layout, .resolve_name, .discover, .collect, .global_tables => null,
        };
        if (pinned) |want| {
            try testing.expect(k.cacheable());
            try testing.expectEqual(want, @intFromEnum(k));
        } else {
            try testing.expect(!k.cacheable());
        }
    }
}

test "cacheable/targetSensitive classify the unified enum's tiers" {
    // The cacheable subset names on-disk entries; the rest are in-memory-only.
    try testing.expect(Phase.lex.cacheable());
    try testing.expect(Phase.parse.cacheable());
    try testing.expect(Phase.codegen.cacheable());
    try testing.expect(!Phase.signature.cacheable());
    try testing.expect(!Phase.body.cacheable());
    try testing.expect(!Phase.layout.cacheable());
    // Only codegen depends on the target; the rest share one entry across
    // targets (lex/parse) or never reach the on-disk key at all.
    try testing.expect(Phase.codegen.targetSensitive());
    try testing.expect(!Phase.lex.targetSensitive());
    try testing.expect(!Phase.parse.targetSensitive());
}

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

    // Corrupt the codegen index checksum in the header so the index validation fails.
    {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/pack.bin", .{dir}) catch unreachable;
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
        defer gpa.free(bytes);
        bytes[16] ^= 0xFF; // codegen index checksum field ([magic][n][idx_ck]...)
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

test "Pack: flush is byte-identical regardless of insertion order" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_a = ".toyc-ut-pack-det-a";
    const dir_b = ".toyc-ut-pack-det-b";
    Io.Dir.cwd().deleteTree(io, dir_a) catch {};
    Io.Dir.cwd().deleteTree(io, dir_b) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir_a) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir_b) catch {};

    const k1 = Cache.Key.fromSource(.lex, "native", "fn a() {}");
    const k2 = Cache.Key.fromSource(.lex, "native", "fn b() {}");

    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir_a);
        const cache = try Cache.initPack(io, dir_a, &pack);
        try cache.put(u8, io, k1, 0, &[_]u8{ 1, 2 });
        try cache.put(u8, io, k2, 1, &[_]u8{ 3, 4, 5 });
        pack.flush(io, dir_a);
    }
    // Same entries, REVERSED insertion order => the sorted flush must produce identical bytes.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir_b);
        const cache = try Cache.initPack(io, dir_b, &pack);
        try cache.put(u8, io, k2, 0, &[_]u8{ 3, 4, 5 });
        try cache.put(u8, io, k1, 1, &[_]u8{ 1, 2 });
        pack.flush(io, dir_b);
    }

    var buf_a: [std.fs.max_path_bytes]u8 = undefined;
    var buf_b: [std.fs.max_path_bytes]u8 = undefined;
    const pa = std.fmt.bufPrint(&buf_a, "{s}/pack.bin", .{dir_a}) catch unreachable;
    const pb = std.fmt.bufPrint(&buf_b, "{s}/pack.bin", .{dir_b}) catch unreachable;
    const ba = try Io.Dir.cwd().readFileAlloc(io, pa, gpa, .unlimited);
    defer gpa.free(ba);
    const bb = try Io.Dir.cwd().readFileAlloc(io, pb, gpa, .unlimited);
    defer gpa.free(bb);
    try testing.expectEqualSlices(u8, ba, bb);
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

// ---- tests: warm-discover manifest section --------------------------------

test "Pack: manifest entries round-trip through flush then load (by canonical path)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toyc-ut-manifest-rt";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const ea: Cache.Pack.ManifestEntry = .{ .mtime = 111, .size = 222, .ctime = 333, .content_fp = 0xABCDEF };
    const eb: Cache.Pack.ManifestEntry = .{ .mtime = -5, .size = 0, .ctime = -7, .content_fp = 0 };

    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir); // cold => empty manifest
        const cache = try Cache.initPack(io, dir, &pack);
        try testing.expect(cache.manifestGet("/a/main.toy") == null); // no prior build
        cache.manifestPut("/a/main.toy", ea);
        cache.manifestPut("/a/util.toy", eb);
        pack.flush(io, dir);
    }
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        const ga = cache.manifestGet("/a/main.toy").?;
        try testing.expectEqual(ea, ga);
        const gb = cache.manifestGet("/a/util.toy").?;
        try testing.expectEqual(eb, gb);
        try testing.expect(cache.manifestGet("/a/absent.toy") == null);
    }
}

test "Pack: manifest flush is byte-identical regardless of insertion order" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_a = ".toyc-ut-manifest-det-a";
    const dir_b = ".toyc-ut-manifest-det-b";
    for ([_][]const u8{ dir_a, dir_b }) |d| Io.Dir.cwd().deleteTree(io, d) catch {};
    defer for ([_][]const u8{ dir_a, dir_b }) |d| Io.Dir.cwd().deleteTree(io, d) catch {};

    const e1: Cache.Pack.ManifestEntry = .{ .mtime = 1, .size = 2, .ctime = 3, .content_fp = 4 };
    const e2: Cache.Pack.ManifestEntry = .{ .mtime = 5, .size = 6, .ctime = 7, .content_fp = 8 };

    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir_a);
        const cache = try Cache.initPack(io, dir_a, &pack);
        cache.manifestPut("aaa.toy", e1);
        cache.manifestPut("bbb.toy", e2);
        pack.flush(io, dir_a);
    }
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir_b);
        const cache = try Cache.initPack(io, dir_b, &pack);
        cache.manifestPut("bbb.toy", e2); // REVERSED insertion order
        cache.manifestPut("aaa.toy", e1);
        pack.flush(io, dir_b);
    }

    var buf_a: [std.fs.max_path_bytes]u8 = undefined;
    var buf_b: [std.fs.max_path_bytes]u8 = undefined;
    const pa = std.fmt.bufPrint(&buf_a, "{s}/pack.bin", .{dir_a}) catch unreachable;
    const pb = std.fmt.bufPrint(&buf_b, "{s}/pack.bin", .{dir_b}) catch unreachable;
    const ba = try Io.Dir.cwd().readFileAlloc(io, pa, gpa, .unlimited);
    defer gpa.free(ba);
    const bb = try Io.Dir.cwd().readFileAlloc(io, pb, gpa, .unlimited);
    defer gpa.free(bb);
    try testing.expectEqualSlices(u8, ba, bb);
}

test "Pack: manifest is rebuilt per build (a file not re-put drops out)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toyc-ut-manifest-rebuild";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const e: Cache.Pack.ManifestEntry = .{ .mtime = 1, .size = 1, .ctime = 1, .content_fp = 1 };
    // Build 1: two files in the graph.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        cache.manifestPut("keep.toy", e);
        cache.manifestPut("drop.toy", e);
        pack.flush(io, dir);
    }
    // Build 2: only one file is still in the graph (only it is re-put). The dropped
    // file must NOT carry forward (the manifest is the current graph, not a union).
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        try testing.expect(cache.manifestGet("keep.toy") != null); // last build saw it
        try testing.expect(cache.manifestGet("drop.toy") != null); // last build saw it too
        cache.manifestPut("keep.toy", e); // only keep is touched this build
        pack.flush(io, dir);
    }
    // Build 3: `drop.toy` is gone from the persisted manifest.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        try testing.expect(cache.manifestGet("keep.toy") != null);
        try testing.expect(cache.manifestGet("drop.toy") == null);
    }
}

test "Pack: codegen + manifest coexist; a corrupt manifest drops only the manifest" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toyc-ut-manifest-coexist";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const k1 = Cache.Key.fromSource(.lex, "native", "fn a() {}");
    const e: Cache.Pack.ManifestEntry = .{ .mtime = 9, .size = 9, .ctime = 9, .content_fp = 9 };
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        try cache.put(u8, io, k1, 0, &[_]u8{ 7, 7, 7 });
        cache.manifestPut("m.toy", e);
        pack.flush(io, dir);
    }
    // Corrupt the manifest checksum field (header bytes 32..40). The codegen index
    // checksum is untouched, so codegen entries must STILL load while the manifest
    // is dropped (every warm-serve then misses and discover re-reads — sound).
    {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/pack.bin", .{dir}) catch unreachable;
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
        defer gpa.free(bytes);
        bytes[32] ^= 0xFF; // manifest checksum field
        var file = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, bytes);
    }
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        try testing.expect(cache.manifestGet("m.toy") == null); // manifest dropped
        const g1 = (try cache.get(u8, gpa, io, k1)).?; // codegen survived
        defer gpa.free(g1);
        try testing.expectEqualSlices(u8, &[_]u8{ 7, 7, 7 }, g1);
    }
}

test "Pack: a corrupt manifest FRAMING (bad n_manifest) degrades the whole pack to cold (still sound)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toyc-ut-manifest-framing";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const k1 = Cache.Key.fromSource(.lex, "native", "fn a() {}");
    const e: Cache.Pack.ManifestEntry = .{ .mtime = 9, .size = 9, .ctime = 9, .content_fp = 9 };
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        try cache.put(u8, io, k1, 0, &[_]u8{ 7, 7, 7 });
        cache.manifestPut("m.toy", e);
        pack.flush(io, dir);
    }
    // Corrupt the n_manifest FRAMING count (header bytes 24..32), NOT the checksum.
    // Unlike a checksum corruption, a wrong extent shifts idx_base, so the codegen index
    // checksum then fails and the WHOLE pack goes cold — codegen is dropped too. This is
    // still SOUND (a clean rebuild, never a false hit); it just pins that the codegen
    // tier is NOT preserved under a framing (vs content) corruption.
    {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/pack.bin", .{dir}) catch unreachable;
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
        defer gpa.free(bytes);
        bytes[24] ^= 0x01; // n_manifest field: 1 -> 0, so the manifest record is read as codegen-index bytes
        var file = try Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, bytes);
    }
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, dir);
        const cache = try Cache.initPack(io, dir, &pack);
        try testing.expect(cache.manifestGet("m.toy") == null); // manifest gone
        try testing.expect((try cache.get(u8, gpa, io, k1)) == null); // codegen ALSO cold — never a false hit
    }
}
