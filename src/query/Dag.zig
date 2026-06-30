//! M16 SPIKE — the AUTOMATIC dependency-recording infrastructure.
//!
//! A per-build directed acyclic graph of query dependencies. Whenever a `query()`
//! runs INSIDE another query's `compute`, the engine records a caller->callee edge
//! plus the callee's RESULT FINGERPRINT into a shared `*Dag`. The structure is
//! purely OBSERVATIONAL this milestone: invalidation is UNCHANGED (still
//! content-fingerprint), and emitted bytes are byte-identical. The recorded DAG is
//! exactly the structure M17's red-green early-cutoff will consume.
//!
//! KEY DESIGN POINTS (locked M16):
//!
//!   * AUTOMATIC active-query-stack. The active query is a per-thread `threadlocal`
//!     pointer (`Active.current`); a nested `query()` call reads it as its PARENT,
//!     enters itself, runs `compute` (which may nest further), then restores the
//!     saved parent. The C call stack IS the dependency stack — no edges are passed
//!     explicitly. Each `fanOut` worker thread has its own `current`, while the
//!     shared `*Dag` (mutex-guarded) collects edges from every thread.
//!
//!   * BORROWED `*Dag`, never an Engine value field. The Engine is re-`init`'d
//!     per job from a copied cache+mode; a Dag VALUE field would give each job a
//!     private empty graph and the edges would vanish. So a single `Dag` lives in
//!     the driver scope and a `*Dag` is threaded into every job, mirroring how
//!     `Cache` is itself a borrowed dir handle.
//!
//!   * DETERMINISM FROM SORT-AT-DUMP ([C11]). Edges record in
//!     schedule-dependent order (parallel fan-out, hit-vs-miss). `dumpDeterministic`
//!     SORTS nodes and each callee list by (kind,id) before emitting, so the dump
//!     is byte-identical run-to-run and single-threaded == fanned-out.
//!
//!   * recordEdge runs UNCONDITIONALLY after a result is obtained — on a cache/memo
//!     HIT and on a fresh compute alike — else the DAG would differ hit-vs-miss.

const std = @import("std");

const Dag = @This();

/// The KIND of a query node — the SINGLE source of truth shared by the on-disk
/// `Cache` (re-exported there as `Cache.Phase`) and the in-memory dependency
/// graph. The CACHEABLE subset (`cacheable()` below: lex/parse/codegen) names the
/// on-disk entries; the rest are IN-MEMORY-ONLY fine-grained typecheck nodes and
/// the two scheduling BARRIER joins (`collect`/`global_tables`).
///
/// `signature` and `body` are DISTINCT kinds for the same fn id: a caller depends
/// on `signature(callee)`, NEVER on `body(callee)` — the firewall the spike proves.
/// The cacheable kinds map to themselves 1:1 (no bridge switch): a query records
/// its node directly off its kind.
///
/// `collect` and `global_tables` are the `Engine.barrier` fan-IN joins the
/// `StageGraph` schedules: `collect`'s id is the order-independent fold of every
/// module's parse digest (the global name-resolution tables built from them);
/// `global_tables`'s id is the fold of every global fn's resolve digest (the
/// program-wide layout/sig tables the per-fn body region depends on). They are
/// OBSERVABILITY-only roots in the recorded DAG (never cacheable), so they have no
/// 1:1 cache entry — only the fold value names them.
///
/// EXPLICIT discriminants pin the CACHEABLE subset's bytes (lex=0, parse=1,
/// codegen=3) so `Cache.Key.digest()` is byte-identical to before this enum was
/// unified — folding a different byte would cold-rebuild every codegen entry. The
/// in-memory-only kinds' bytes are free to renumber (their only persisted use is
/// the DAG blob, whose `format_version` gates a mismatch to a clean re-validation).
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

    /// Whether a node of this kind names an on-disk `Cache` entry (the CACHEABLE
    /// subset) vs an in-memory-only dependency node. Exhaustive no-else: a new kind
    /// FAILS TO COMPILE until it declares its tier here.
    pub fn cacheable(kind: Kind) bool {
        return switch (kind) {
            .lex, .parse, .codegen => true,
            .signature, .body, .type_of, .layout, .resolve_name, .discover, .collect, .global_tables => false,
        };
    }

    /// Whether results for this kind depend on the compilation target. Target is
    /// folded into the on-disk cache key only when true, so target-independent
    /// phases share one entry across targets. Exhaustive no-else: a new kind FAILS
    /// TO COMPILE until it declares its target-sensitivity here.
    pub fn targetSensitive(kind: Kind) bool {
        return switch (kind) {
            .codegen => true, // aarch64 blobs must not alias across targets [C10]
            .lex, .parse, .signature, .body, .type_of, .layout, .resolve_name, .discover, .collect, .global_tables => false,
        };
    }
};

/// An IN-MEMORY node key. `id` is the deterministic global id ([C11]) for the
/// fine-grained typecheck kinds; for lex/parse it folds the `Cache.Key.digest()`
/// so DAG nodes align with cache identity, and codegen folds a STABLE per-fn
/// identity (`Key.codegenIdentity`) decoupled from the content fp.
pub const NodeKey = struct {
    kind: Kind,
    id: u64,

    /// Total order for deterministic dump: (kind, id). [C11]
    fn lessThan(_: void, a: NodeKey, b: NodeKey) bool {
        const ak = @intFromEnum(a.kind);
        const bk = @intFromEnum(b.kind);
        if (ak != bk) return ak < bk;
        return a.id < b.id;
    }
};

/// A self-contained spinlock guarding all mutation/read of the maps below. Zig
/// 0.16's `std.Io.Mutex` needs an `Io` to block; the DAG is touched from both the
/// engine (which has an `Io`) AND pure unit tests (which do not), and the critical
/// sections are tiny (a hashmap upsert), so a lock-free atomic spinlock keeps the
/// module Io-free and testable while staying safe under parallel fan-out. [C11]
lock_state: std.atomic.Value(bool) = .init(false),
/// caller -> ordered (deduped) list of callees. A `null` caller is a ROOT demand
/// and records NO edge (only the callee's result fingerprint).
edges: std.AutoHashMapUnmanaged(NodeKey, std.ArrayListUnmanaged(NodeKey)) = .{},
/// node -> its last-observed result fingerprint. Upserted on EVERY observation
/// (hit or fresh compute) so the dump shows a stable fp.
result_fp: std.AutoHashMapUnmanaged(NodeKey, u64) = .{},
/// M17 RED-GREEN SCAFFOLDING (persisted, not yet driving invalidation this
/// stage). A global revision counter, bumped when an input changes; per-node
/// `changed_at`/`verified_at` stamps (invariant `changed_at <= verified_at`).
/// These round-trip through serialize/deserialize so a fresh process loads the
/// prior build's stamps; the maybe_changed_after walk (a later stage) consumes
/// them. Today they are recorded as 0 and persisted verbatim.
revision: u64 = 0,
/// node -> the revision at which its result fp last CHANGED.
changed_at: std.AutoHashMapUnmanaged(NodeKey, u64) = .{},
/// node -> the revision at which it was last VERIFIED unchanged (green/red).
verified_at: std.AutoHashMapUnmanaged(NodeKey, u64) = .{},

pub fn init(_: std.mem.Allocator) Dag {
    return .{};
}

fn lock(self: *Dag) void {
    while (self.lock_state.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
        std.atomic.spinLoopHint();
    }
}

fn unlock(self: *Dag) void {
    self.lock_state.store(false, .release);
}

pub fn deinit(self: *Dag, gpa: std.mem.Allocator) void {
    var it = self.edges.valueIterator();
    while (it.next()) |list| list.deinit(gpa);
    self.edges.deinit(gpa);
    self.result_fp.deinit(gpa);
    self.changed_at.deinit(gpa);
    self.verified_at.deinit(gpa);
}

/// Record `parent -> child` (deduped) and ALWAYS upsert `result_fp[child]`.
/// `parent == null` is a root demand: no edge, but the child's fp is still
/// recorded. Mutex-guarded so parallel fan-out workers are safe. recordEdge is a
/// non-recursive upsert, so it is safe to call on a re-entrant / poisoned node
/// (the layout cycle case). Must be called UNCONDITIONALLY after a result is
/// obtained (HIT == fresh compute) for determinism.
pub fn recordEdge(self: *Dag, gpa: std.mem.Allocator, parent: ?NodeKey, child: NodeKey, child_fp: u64) void {
    self.lock();
    defer self.unlock();

    self.result_fp.put(gpa, child, child_fp) catch {};

    const p = parent orelse return;
    const gop = self.edges.getOrPut(gpa, p) catch return;
    if (!gop.found_existing) gop.value_ptr.* = .empty;
    for (gop.value_ptr.items) |existing| {
        if (existing.kind == child.kind and existing.id == child.id) return; // dedup
    }
    gop.value_ptr.append(gpa, child) catch {};
}

/// Look up a node's recorded result fingerprint, if any. Test-facing: the firewall
/// test reads `signature(fn)`'s fp here and asserts stable-vs-flipped.
pub fn fingerprintOf(self: *Dag, node: NodeKey) ?u64 {
    self.lock();
    defer self.unlock();
    return self.result_fp.get(node);
}

/// A lock-guarded snapshot of every node that has a recorded result fp (caller owns
/// the slice). The red-green walk demands these as roots; taking the snapshot under
/// the lock then walking lock-free avoids the non-reentrant spinlock self-deadlock
/// (the recursive walk must never hold the lock).
pub fn nodeKeys(self: *Dag, gpa: std.mem.Allocator) ![]NodeKey {
    self.lock();
    defer self.unlock();
    var out: std.ArrayListUnmanaged(NodeKey) = .empty;
    errdefer out.deinit(gpa);
    var it = self.result_fp.keyIterator();
    while (it.next()) |k| try out.append(gpa, k.*);
    return out.toOwnedSlice(gpa);
}

/// Emit the DAG in DETERMINISTIC order: nodes sorted by (kind,id), each callee
/// list sorted by (kind,id), one `caller -> callee fp=<hex>` line per edge, plus
/// a `root fp=<hex>` line for every node that has a recorded fp. Determinism
/// comes from SORT-AT-DUMP, so parallel schedule and hit/miss order are erased.
/// [C11]
pub fn dumpDeterministic(self: *Dag, gpa: std.mem.Allocator, writer: anytype) !void {
    self.lock();
    defer self.unlock();

    // (1) every node that has a recorded fp -> a stable, sorted fp line.
    var fp_nodes: std.ArrayListUnmanaged(NodeKey) = .empty;
    defer fp_nodes.deinit(gpa);
    {
        var it = self.result_fp.keyIterator();
        while (it.next()) |k| try fp_nodes.append(gpa, k.*);
    }
    std.mem.sort(NodeKey, fp_nodes.items, {}, NodeKey.lessThan);
    for (fp_nodes.items) |n| {
        const fp = self.result_fp.get(n).?;
        try writer.print("node {s}#{d} fp={x}\n", .{ @tagName(n.kind), n.id, fp });
    }

    // (2) edges: callers sorted, each callee list sorted.
    var callers: std.ArrayListUnmanaged(NodeKey) = .empty;
    defer callers.deinit(gpa);
    {
        var it = self.edges.keyIterator();
        while (it.next()) |k| try callers.append(gpa, k.*);
    }
    std.mem.sort(NodeKey, callers.items, {}, NodeKey.lessThan);
    for (callers.items) |caller| {
        const list = self.edges.get(caller).?;
        const callees = try gpa.dupe(NodeKey, list.items);
        defer gpa.free(callees);
        std.mem.sort(NodeKey, callees, {}, NodeKey.lessThan);
        for (callees) |callee| {
            const fp = self.result_fp.get(callee) orelse 0;
            try writer.print("{s}#{d} -> {s}#{d} fp={x}\n", .{
                @tagName(caller.kind), caller.id,
                @tagName(callee.kind), callee.id,
                fp,
            });
        }
    }
}

// ===========================================================================
// M17 PERSISTENCE — flat, deterministic on-disk blob (one per program+target).
// ===========================================================================

/// On-disk format magic + version. A foreign/old blob (mismatch) deserializes to
/// `null` => "no prior DAG" => the build re-validates from scratch (the
/// corruption FALLBACK; never a partial/false-green). The version MUST bump on
/// ANY layout change to this format.
const magic: u32 = 0x47_41_44_54; // "TDAG" little-endian
// v3: the `Kind` enum was unified with `Cache.Phase` and its in-memory-only kinds
// renumbered (codegen pinned to byte 3, body/type_of/layout/resolve_name shifted);
// a v2 blob's kind bytes would now mis-decode, so the bump forces those to
// deserialize to null (a clean full re-validation, never a misread node). v2 itself
// had dropped the never-recorded `Kind.resolve` for the same reason.
const format_version: u32 = 3;

/// Header: magic, version, revision, node_count, edge_count. All u64-padded so
/// the body alignment is trivially 8 and no struct padding is ever hashed/written
/// (we emit field-by-field LE, never `sliceAsBytes` of a record — the exact
/// Ast.Node padding bug, avoided at the persistence layer by construction).
const header_bytes = 4 + 4 + 8 + 8 + 8;
/// NODE record: kind(u8) + id(u64) + result_fp(u64) + changed_at(u64) +
/// verified_at(u64), written field-by-field LE (no padding).
const node_rec_bytes = 1 + 8 + 8 + 8 + 8;
/// EDGE record: parent{kind(u8),id(u64)} + child{kind(u8),id(u64)}, LE.
const edge_rec_bytes = (1 + 8) + (1 + 8);

/// A flat snapshot of one persisted node (mirrors the NODES section). Returned by
/// deserialize so a later red-green walk can read stamps without touching maps.
pub const NodeRecord = struct {
    key: NodeKey,
    result_fp: u64,
    changed_at: u64,
    verified_at: u64,
};

/// A flat snapshot of one persisted edge.
pub const EdgeRecord = struct {
    parent: NodeKey,
    child: NodeKey,
};

/// The result of loading a prior DAG: the revision baseline plus flat node/edge
/// records, owned by the caller. Kept deliberately map-free and lock-free: a
/// fresh process loads this read-only snapshot, never the live mutable maps.
pub const Loaded = struct {
    revision: u64,
    nodes: []NodeRecord,
    edges: []EdgeRecord,

    pub fn deinit(self: *Loaded, gpa: std.mem.Allocator) void {
        gpa.free(self.nodes);
        gpa.free(self.edges);
        self.* = undefined;
    }

    /// Look up a node's persisted result fp. `nodes` is (kind,id)-sorted by `serialize`,
    /// so this binary-searches: the red-green walk does one lookup per node + per edge, so
    /// a linear scan made the whole walk O(N^2) — at 60k nodes a fully-cached build spent
    /// >10s here. Binary search drops it to O(N log N) with zero extra allocation.
    pub fn nodeFp(self: *const Loaded, key: NodeKey) ?u64 {
        const kk = @intFromEnum(key.kind);
        var lo: usize = 0;
        var hi: usize = self.nodes.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const m = self.nodes[mid].key;
            const mk = @intFromEnum(m.kind);
            if (mk == kk and m.id == key.id) return self.nodes[mid].result_fp;
            if (mk < kk or (mk == kk and m.id < key.id)) lo = mid + 1 else hi = mid;
        }
        return null;
    }
};

/// Serialize the DAG into one flat, deterministic blob (caller owns the bytes).
/// Emission order mirrors `dumpDeterministic`: nodes sorted by (kind,id), edges
/// by (parent,child), so the on-disk bytes are byte-identical run-to-run ([C11];
/// doubles as a reproducibility self-check). Records are written field-by-field
/// little-endian — never `sliceAsBytes` of a struct — so no uninitialized padding
/// can leak (the Ast.Node fp-nondeterminism bug, structurally precluded here).
pub fn serialize(self: *Dag, gpa: std.mem.Allocator) ![]u8 {
    self.lock();
    defer self.unlock();

    var nodes: std.ArrayListUnmanaged(NodeKey) = .empty;
    defer nodes.deinit(gpa);
    {
        var it = self.result_fp.keyIterator();
        while (it.next()) |k| try nodes.append(gpa, k.*);
    }
    std.mem.sort(NodeKey, nodes.items, {}, NodeKey.lessThan);

    var edges: std.ArrayListUnmanaged(EdgeRecord) = .empty;
    defer edges.deinit(gpa);
    {
        var callers: std.ArrayListUnmanaged(NodeKey) = .empty;
        defer callers.deinit(gpa);
        var it = self.edges.keyIterator();
        while (it.next()) |k| try callers.append(gpa, k.*);
        std.mem.sort(NodeKey, callers.items, {}, NodeKey.lessThan);
        for (callers.items) |caller| {
            const list = self.edges.get(caller).?;
            const callees = try gpa.dupe(NodeKey, list.items);
            defer gpa.free(callees);
            std.mem.sort(NodeKey, callees, {}, NodeKey.lessThan);
            for (callees) |callee| try edges.append(gpa, .{ .parent = caller, .child = callee });
        }
    }

    const total = header_bytes + nodes.items.len * node_rec_bytes + edges.items.len * edge_rec_bytes;
    const buf = try gpa.alloc(u8, total);
    errdefer gpa.free(buf);

    var w: usize = 0;
    const putU32 = struct {
        fn f(b: []u8, off: *usize, v: u32) void {
            std.mem.writeInt(u32, b[off.*..][0..4], v, .little);
            off.* += 4;
        }
    }.f;
    const putU64 = struct {
        fn f(b: []u8, off: *usize, v: u64) void {
            std.mem.writeInt(u64, b[off.*..][0..8], v, .little);
            off.* += 8;
        }
    }.f;
    const putU8 = struct {
        fn f(b: []u8, off: *usize, v: u8) void {
            b[off.*] = v;
            off.* += 1;
        }
    }.f;

    putU32(buf, &w, magic);
    putU32(buf, &w, format_version);
    putU64(buf, &w, self.revision);
    putU64(buf, &w, nodes.items.len);
    putU64(buf, &w, edges.items.len);

    for (nodes.items) |n| {
        putU8(buf, &w, @intFromEnum(n.kind));
        putU64(buf, &w, n.id);
        putU64(buf, &w, self.result_fp.get(n).?);
        putU64(buf, &w, self.changed_at.get(n) orelse 0);
        putU64(buf, &w, self.verified_at.get(n) orelse 0);
    }
    for (edges.items) |e| {
        putU8(buf, &w, @intFromEnum(e.parent.kind));
        putU64(buf, &w, e.parent.id);
        putU8(buf, &w, @intFromEnum(e.child.kind));
        putU64(buf, &w, e.child.id);
    }
    std.debug.assert(w == total);
    return buf;
}

/// Parse a blob produced by `serialize` into a `Loaded` snapshot, or `null` on
/// ANY structural mismatch (bad magic/version, truncation, a count that overruns
/// the buffer, an out-of-range kind byte). A `null` => "no/garbage prior DAG" =>
/// full re-validation; deserialize NEVER returns a partial record set, so a torn
/// or foreign blob can never produce a false-green. Caller owns the result.
pub fn deserialize(gpa: std.mem.Allocator, bytes: []const u8) !?Loaded {
    if (bytes.len < header_bytes) return null;
    var r: usize = 0;
    const getU32 = struct {
        fn f(b: []const u8, off: *usize) u32 {
            const v = std.mem.readInt(u32, b[off.*..][0..4], .little);
            off.* += 4;
            return v;
        }
    }.f;
    const getU64 = struct {
        fn f(b: []const u8, off: *usize) u64 {
            const v = std.mem.readInt(u64, b[off.*..][0..8], .little);
            off.* += 8;
            return v;
        }
    }.f;

    if (getU32(bytes, &r) != magic) return null;
    if (getU32(bytes, &r) != format_version) return null;
    const revision = getU64(bytes, &r);
    const node_count = getU64(bytes, &r);
    const edge_count = getU64(bytes, &r);

    // Reject a count that would overrun the buffer BEFORE allocating, so a
    // foreign/torn header can never drive a huge alloc or an OOB read.
    const nc: usize = std.math.cast(usize, node_count) orelse return null;
    const ec: usize = std.math.cast(usize, edge_count) orelse return null;
    const node_bytes = std.math.mul(usize, nc, node_rec_bytes) catch return null;
    const edge_bytes = std.math.mul(usize, ec, edge_rec_bytes) catch return null;
    const need_total = std.math.add(usize, header_bytes, std.math.add(usize, node_bytes, edge_bytes) catch return null) catch return null;
    if (bytes.len != need_total) return null;

    const max_kind = @typeInfo(Kind).@"enum".fields.len;

    const nodes = try gpa.alloc(NodeRecord, nc);
    errdefer gpa.free(nodes);
    for (nodes) |*n| {
        const kb = bytes[r];
        r += 1;
        if (kb >= max_kind) return null;
        n.* = .{
            .key = .{ .kind = @enumFromInt(kb), .id = getU64(bytes, &r) },
            .result_fp = getU64(bytes, &r),
            .changed_at = getU64(bytes, &r),
            .verified_at = getU64(bytes, &r),
        };
    }

    const edges = try gpa.alloc(EdgeRecord, ec);
    errdefer gpa.free(edges);
    for (edges) |*e| {
        const pk = bytes[r];
        r += 1;
        if (pk >= max_kind) return null;
        const pid = getU64(bytes, &r);
        const ck = bytes[r];
        r += 1;
        if (ck >= max_kind) return null;
        const cid = getU64(bytes, &r);
        e.* = .{
            .parent = .{ .kind = @enumFromInt(pk), .id = pid },
            .child = .{ .kind = @enumFromInt(ck), .id = cid },
        };
    }

    return Loaded{ .revision = revision, .nodes = nodes, .edges = edges };
}

// ===========================================================================
// M17 RED-GREEN CORE — the maybe_changed_after walk + early-cutoff/backdating.
// ===========================================================================

/// A node's verdict in the red-green walk: GREEN (verified unchanged — reuse the
/// cached result, no re-execution) or RED (a dep changed, so the node must
/// re-execute). Memoized per node so the walk visits each node once.
pub const Verdict = enum { green, red };

/// The error a re-entered in-progress query raises: the dependency graph is a DAG,
/// so re-visiting a node already on the walk stack is a cycle. Recursive *functions*
/// do NOT cycle the query graph — the signature firewall means a body depends on a
/// callee's SIGNATURE, never its body, so a self/mutual-recursive call recurses the
/// signature node (a leaf w.r.t. the body), not back into the body.
pub const WalkError = error{QueryCycle} || std.mem.Allocator.Error;

/// The RED-GREEN re-validation walker. Drives `maybeChangedAfter` over a PRIOR
/// build's `Loaded` snapshot (its recorded child edges + per-node result fps),
/// using a FRESH `Dag` (this build's just-recorded fps) as the comparator.
///
/// THE INVARIANT (soundness + cutoff in one rule): a node is "changed" — visible to
/// its dependents — iff its FRESH result fp differs from its PRIOR result fp (a new
/// node, absent from the prior DAG, counts as changed). Dependency recursion decides
/// only whether a node must RE-EXECUTE (the recompute set / RED); BACKDATING is then
/// automatic — a node that re-executes because a dep changed but whose own fresh fp
/// equals its prior fp is NOT marked changed, so its dependents cut off. The
/// comparator is M16's `result_fp` verbatim; the walk introduces no new fingerprint.
///
/// USAGE is lock-free + single-threaded: build it AFTER the fan-out has fully
/// recorded the fresh Dag (every fp present), never inside a held spinlock — the
/// recursive walk would self-deadlock the non-reentrant lock. It snapshots the
/// fresh fps it needs via `fingerprintOf` (which locks per call) before recursing.
pub const RedGreen = struct {
    gpa: std.mem.Allocator,
    prior: *const Loaded,
    fresh: *Dag,
    /// node -> resolved verdict (green/red), so each node is walked once.
    verdict: std.AutoHashMapUnmanaged(NodeKey, Verdict) = .{},
    /// nodes currently on the walk stack — re-entry is a cycle.
    on_stack: std.AutoHashMapUnmanaged(NodeKey, void) = .{},
    /// Prior-edge index: parent -> its recorded children (built once from the flat
    /// `Loaded.edges`, so recursion is O(children) not O(all edges) per node).
    children: std.AutoHashMapUnmanaged(NodeKey, std.ArrayListUnmanaged(NodeKey)) = .{},
    /// How many nodes the walk re-executes (RED). The cutoff metric: a strict
    /// subset of the full node set on a localized edit.
    red_count: usize = 0,
    /// How many nodes verify unchanged (GREEN). green > 0 is the liveness signal.
    green_count: usize = 0,

    pub fn init(gpa: std.mem.Allocator, prior: *const Loaded, fresh: *Dag) !RedGreen {
        var rg: RedGreen = .{ .gpa = gpa, .prior = prior, .fresh = fresh };
        errdefer rg.deinit();
        for (prior.edges) |e| {
            const gop = try rg.children.getOrPut(gpa, e.parent);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(gpa, e.child);
        }
        return rg;
    }

    pub fn deinit(self: *RedGreen) void {
        var it = self.children.valueIterator();
        while (it.next()) |list| list.deinit(self.gpa);
        self.children.deinit(self.gpa);
        self.verdict.deinit(self.gpa);
        self.on_stack.deinit(self.gpa);
        self.* = undefined;
    }

    fn priorChildren(self: *const RedGreen, node: NodeKey) []const NodeKey {
        if (self.children.get(node)) |list| return list.items;
        return &.{};
    }

    /// Whether `node`'s own result fp CHANGED vs the prior build (a new node — absent
    /// from the prior DAG — counts as changed). This is the sole soundness comparator.
    fn fpChanged(self: *RedGreen, node: NodeKey) bool {
        const prior_fp = self.prior.nodeFp(node) orelse return true; // new node
        const fresh_fp = self.fresh.fingerprintOf(node) orelse return true; // gone => treat as changed
        return prior_fp != fresh_fp;
    }

    /// The red-green re-validation of one node. Returns its `Verdict`:
    ///   * recurse the PRIOR-recorded deps first (depth-first);
    ///   * if EVERY dep is green AND the node's own fp is unchanged -> GREEN
    ///     (reuse the cached result, no re-execution);
    ///   * otherwise RED (a dep changed, or a leaf input changed -> re-execute).
    /// BACKDATING: a node is reported CHANGED to its dependents only via `fpChanged`,
    /// so a RED node that re-executes to an identical fp does NOT propagate as changed
    /// — `changedAfter` (below) reads the fp, not the verdict, to gate dependents.
    /// A re-entered on-stack node raises `error.QueryCycle`.
    pub fn verdictOf(self: *RedGreen, node: NodeKey) WalkError!Verdict {
        if (self.verdict.get(node)) |v| return v;
        if (self.on_stack.contains(node)) return error.QueryCycle;
        try self.on_stack.put(self.gpa, node, {});
        defer _ = self.on_stack.remove(node);

        var any_dep_changed = false;
        for (self.priorChildren(node)) |child| {
            _ = try self.verdictOf(child); // walk + memoize the subtree
            if (self.changedAfter(child)) any_dep_changed = true;
        }

        // RED iff a dep changed OR (it is a leaf input that itself changed). For a
        // node WITH deps, the dep-change drives re-execution; for a leaf (no deps),
        // its own fp flip is the only change source. Either way an unchanged-fp node
        // with all-green deps is GREEN.
        const v: Verdict = if (any_dep_changed or self.fpChanged(node)) .red else .green;
        try self.verdict.put(self.gpa, node, v);
        if (v == .red) self.red_count += 1 else self.green_count += 1;
        return v;
    }

    /// Whether `node` is CHANGED as seen by its dependents — the backdating gate.
    /// Reads the fp comparison, NOT the verdict: a RED node that re-executed to an
    /// identical fp is "not changed", so its dependents cut off. Safe to call only
    /// after `verdictOf(node)` has run (the walk is depth-first, so it has).
    pub fn changedAfter(self: *RedGreen, node: NodeKey) bool {
        return self.fpChanged(node);
    }

    /// Drive the walk from a set of ROOT demands (e.g. every codegen node) and
    /// resolve their whole transitive dep set. Returns the recompute set size (RED
    /// count) so the caller can report the cutoff. `error.QueryCycle` on a cycle.
    pub fn run(self: *RedGreen, roots: []const NodeKey) WalkError!void {
        for (roots) |r| _ = try self.verdictOf(r);
    }
};

/// Whether `parent -> child` is a recorded edge. Test-facing: the firewall /
/// under-recording guard asserts edge presence/absence.
pub fn hasEdge(self: *Dag, parent: NodeKey, child: NodeKey) bool {
    self.lock();
    defer self.unlock();
    const list = self.edges.get(parent) orelse return false;
    for (list.items) |c| {
        if (c.kind == child.kind and c.id == child.id) return true;
    }
    return false;
}

/// The AUTOMATIC active-query-stack. `current` is the node whose `compute` is
/// presently running ON THIS THREAD; a nested `query()` reads it as its parent.
/// Per-thread (`threadlocal`) so each fan-out worker has its own stack while the
/// shared `*Dag` collects from all of them. `enter` returns the previous value;
/// the caller `defer`s `leave(prev)` to restore (save/restore = the C call stack
/// is the dependency stack).
pub const Active = struct {
    threadlocal var current: ?NodeKey = null;

    pub fn get() ?NodeKey {
        return current;
    }

    pub fn enter(node: NodeKey) ?NodeKey {
        const prev = current;
        current = node;
        return prev;
    }

    pub fn leave(prev: ?NodeKey) void {
        current = prev;
    }
};

/// Fold a fn's SIGNATURE — kind, length-prefixed name, param kinds (incl. global
/// struct/enum ids), return kind — into a u64, mirroring the codegen callee-sig
/// fold (Fingerprint.zig L84-91) EXACTLY and folding NO body content. This is the
/// `signature(fn)` fingerprint the spike's firewall proves: a body-only edit leaves
/// it STABLE; a signature edit (param/ret type or arity) FLIPS it. Under-fold =>
/// body edit flips sig (firewall fails); over-fold/miss => caller flips wrongly.
pub fn sigFingerprint(sig: anytype) u64 {
    var h = std.hash.Wyhash.init(0x53_49_47_4e); // "SIGN"
    h.update(&[_]u8{@intFromEnum(sig.kind)});
    var name_len: [4]u8 = undefined;
    std.mem.writeInt(u32, &name_len, @intCast(sig.name.len), .little);
    h.update(&name_len);
    h.update(sig.name);
    var nparams: [4]u8 = undefined;
    std.mem.writeInt(u32, &nparams, @intCast(sig.params.len), .little);
    h.update(&nparams);
    for (sig.params) |p| {
        h.update(&[_]u8{@intFromEnum(p.kind)});
        // Global type identity (struct/enum id), NOT byte layout — keeps the sig a
        // pure ABI-shape hash and avoids a sig<->layout cycle.
        var id_buf: [8]u8 = undefined;
        std.mem.writeInt(u32, id_buf[0..4], p.struct_id, .little);
        std.mem.writeInt(u32, id_buf[4..8], p.enum_id, .little);
        h.update(&id_buf);
    }
    h.update(&[_]u8{@intFromEnum(sig.ret.kind)});
    {
        var id_buf: [8]u8 = undefined;
        std.mem.writeInt(u32, id_buf[0..4], sig.ret.struct_id, .little);
        std.mem.writeInt(u32, id_buf[4..8], sig.ret.enum_id, .little);
        h.update(&id_buf);
    }
    return h.final();
}

// ===========================================================================
// Inline unit tests (pure DAG mechanics — no engine/cache needed).
// ===========================================================================

const testing = std.testing;

test "recordEdge dedups callees and upserts result fp" {
    var dag: Dag = .init(testing.allocator);
    defer dag.deinit(testing.allocator);

    const caller: NodeKey = .{ .kind = .body, .id = 1 };
    const callee: NodeKey = .{ .kind = .signature, .id = 2 };

    dag.recordEdge(testing.allocator, caller, callee, 0xAAAA);
    dag.recordEdge(testing.allocator, caller, callee, 0xBBBB); // dup edge, new fp

    try testing.expect(dag.hasEdge(caller, callee));
    // dedup: only one callee recorded.
    try testing.expectEqual(@as(usize, 1), dag.edges.get(caller).?.items.len);
    // fp upserted to the latest observation.
    try testing.expectEqual(@as(?u64, 0xBBBB), dag.fingerprintOf(callee));
}

test "root demand records fp but no edge" {
    var dag: Dag = .init(testing.allocator);
    defer dag.deinit(testing.allocator);

    const root: NodeKey = .{ .kind = .codegen, .id = 7 };
    dag.recordEdge(testing.allocator, null, root, 0x1234);

    try testing.expectEqual(@as(?u64, 0x1234), dag.fingerprintOf(root));
    try testing.expectEqual(@as(usize, 0), dag.edges.count());
}

test "dumpDeterministic is sort-stable regardless of insertion order" {
    const gpa = testing.allocator;

    const caller: NodeKey = .{ .kind = .body, .id = 1 };
    const c_sig: NodeKey = .{ .kind = .signature, .id = 9 };
    const c_layout: NodeKey = .{ .kind = .layout, .id = 3 };

    var a: Dag = .init(gpa);
    defer a.deinit(gpa);
    a.recordEdge(gpa, caller, c_sig, 0x11);
    a.recordEdge(gpa, caller, c_layout, 0x22);

    var b: Dag = .init(gpa);
    defer b.deinit(gpa);
    // reverse insertion order + a redundant re-record (hit vs miss).
    b.recordEdge(gpa, caller, c_layout, 0x22);
    b.recordEdge(gpa, caller, c_sig, 0x11);
    b.recordEdge(gpa, caller, c_sig, 0x11);

    var aw_a: std.Io.Writer.Allocating = .init(gpa);
    defer aw_a.deinit();
    var aw_b: std.Io.Writer.Allocating = .init(gpa);
    defer aw_b.deinit();
    try a.dumpDeterministic(gpa, &aw_a.writer);
    try b.dumpDeterministic(gpa, &aw_b.writer);

    try testing.expectEqualStrings(aw_a.written(), aw_b.written());
}

test "sigFingerprint is stable across body changes, flips on signature changes" {
    const types = @import("../types.zig");
    const Sym = @import("../symbols/Sym.zig");
    const Sig = @import("../symbols/Sig.zig").Sig;

    const params_int = [_]types.Type{types.Type.int};
    const base: Sig = .{ .kind = Sym.SymKind.user_fn, .name = "add", .params = &params_int, .ret = types.Type.int };

    // Identical signature => identical fp (a body-only edit does not touch these
    // fields, so the sig fp is STABLE — the firewall).
    const same: Sig = .{ .kind = Sym.SymKind.user_fn, .name = "add", .params = &params_int, .ret = types.Type.int };
    try testing.expectEqual(sigFingerprint(base), sigFingerprint(same));

    // A return-type change FLIPS the fp.
    const ret_changed: Sig = .{ .kind = Sym.SymKind.user_fn, .name = "add", .params = &params_int, .ret = types.Type.bool };
    try testing.expect(sigFingerprint(base) != sigFingerprint(ret_changed));

    // An arity change FLIPS the fp.
    const params_two = [_]types.Type{ types.Type.int, types.Type.int };
    const arity_changed: Sig = .{ .kind = Sym.SymKind.user_fn, .name = "add", .params = &params_two, .ret = types.Type.int };
    try testing.expect(sigFingerprint(base) != sigFingerprint(arity_changed));

    // A param-type change FLIPS the fp.
    const params_bool = [_]types.Type{types.Type.bool};
    const param_changed: Sig = .{ .kind = Sym.SymKind.user_fn, .name = "add", .params = &params_bool, .ret = types.Type.int };
    try testing.expect(sigFingerprint(base) != sigFingerprint(param_changed));
}

test "Active stack save/restore nests" {
    try testing.expectEqual(@as(?NodeKey, null), Active.get());
    const a: NodeKey = .{ .kind = .body, .id = 1 };
    const prev_a = Active.enter(a);
    try testing.expectEqual(@as(?NodeKey, a), Active.get());
    {
        const b: NodeKey = .{ .kind = .signature, .id = 2 };
        const prev_b = Active.enter(b);
        try testing.expectEqual(@as(?NodeKey, b), Active.get());
        Active.leave(prev_b);
    }
    try testing.expectEqual(@as(?NodeKey, a), Active.get());
    Active.leave(prev_a);
    try testing.expectEqual(@as(?NodeKey, null), Active.get());
}

test "serialize/deserialize round-trips nodes, edges, fps, revision, stamps" {
    const gpa = testing.allocator;
    var dag: Dag = .init(gpa);
    defer dag.deinit(gpa);

    const caller: NodeKey = .{ .kind = .codegen, .id = 100 };
    const sig: NodeKey = .{ .kind = .signature, .id = 7 };
    const body: NodeKey = .{ .kind = .body, .id = 7 };

    dag.recordEdge(gpa, null, caller, 0xC0DE); // caller demanded as a root => its own fp
    dag.recordEdge(gpa, caller, sig, 0xDEAD);
    dag.recordEdge(gpa, null, body, 0xBEEF); // root demand: fp, no edge
    dag.recordEdge(gpa, caller, body, 0xBEEF);
    dag.revision = 42;
    try dag.changed_at.put(gpa, sig, 7);
    try dag.verified_at.put(gpa, sig, 42);

    const blob = try dag.serialize(gpa);
    defer gpa.free(blob);

    var loaded = (try deserialize(gpa, blob)).?;
    defer loaded.deinit(gpa);

    try testing.expectEqual(@as(u64, 42), loaded.revision);
    try testing.expectEqual(@as(usize, 3), loaded.nodes.len); // caller, sig, body
    try testing.expectEqual(@as(usize, 2), loaded.edges.len); // caller->sig, caller->body

    try testing.expectEqual(@as(?u64, 0xDEAD), loaded.nodeFp(sig));
    try testing.expectEqual(@as(?u64, 0xBEEF), loaded.nodeFp(body));

    // The sig node's stamps survived the round-trip.
    for (loaded.nodes) |n| if (n.key.kind == .signature and n.key.id == 7) {
        try testing.expectEqual(@as(u64, 7), n.changed_at);
        try testing.expectEqual(@as(u64, 42), n.verified_at);
    };
}

test "serialize is byte-deterministic regardless of insertion order [C11]" {
    const gpa = testing.allocator;
    const caller: NodeKey = .{ .kind = .body, .id = 1 };
    const c_sig: NodeKey = .{ .kind = .signature, .id = 9 };
    const c_layout: NodeKey = .{ .kind = .layout, .id = 3 };

    var a: Dag = .init(gpa);
    defer a.deinit(gpa);
    a.recordEdge(gpa, caller, c_sig, 0x11);
    a.recordEdge(gpa, caller, c_layout, 0x22);

    var b: Dag = .init(gpa);
    defer b.deinit(gpa);
    b.recordEdge(gpa, caller, c_layout, 0x22);
    b.recordEdge(gpa, caller, c_sig, 0x11);
    b.recordEdge(gpa, caller, c_sig, 0x11); // redundant re-record (hit vs miss)

    const ba = try a.serialize(gpa);
    defer gpa.free(ba);
    const bb = try b.serialize(gpa);
    defer gpa.free(bb);
    try testing.expectEqualSlices(u8, ba, bb);
}

/// Build a `Loaded` snapshot directly from a `Dag` (round-trip via serialize) so
/// the red-green tests can model a "prior build". Caller deinits the `Loaded`.
fn loadedOf(gpa: std.mem.Allocator, dag: *Dag) !Loaded {
    const blob = try dag.serialize(gpa);
    defer gpa.free(blob);
    return (try deserialize(gpa, blob)).?;
}

test "red-green: an unchanged build is ALL GREEN (no recompute)" {
    const gpa = testing.allocator;
    // prior: codegen(caller) -> signature(callee); plus body(callee).
    const caller: NodeKey = .{ .kind = .codegen, .id = 1 };
    const sig: NodeKey = .{ .kind = .signature, .id = 2 };
    const body: NodeKey = .{ .kind = .body, .id = 2 };

    var prior_dag: Dag = .init(gpa);
    defer prior_dag.deinit(gpa);
    prior_dag.recordEdge(gpa, null, caller, 0xAAAA);
    prior_dag.recordEdge(gpa, caller, sig, 0xBBBB);
    prior_dag.recordEdge(gpa, null, body, 0xCCCC);

    var prior = try loadedOf(gpa, &prior_dag);
    defer prior.deinit(gpa);

    // fresh: IDENTICAL fps (nothing edited).
    var fresh: Dag = .init(gpa);
    defer fresh.deinit(gpa);
    fresh.recordEdge(gpa, null, caller, 0xAAAA);
    fresh.recordEdge(gpa, caller, sig, 0xBBBB);
    fresh.recordEdge(gpa, null, body, 0xCCCC);

    var rg = try RedGreen.init(gpa, &prior, &fresh);
    defer rg.deinit();
    try rg.run(&[_]NodeKey{ caller, body });

    try testing.expectEqual(@as(usize, 0), rg.red_count);
    try testing.expect(rg.green_count > 0);
    try testing.expectEqual(Verdict.green, (try rg.verdictOf(caller)));
}

test "red-green: a CHANGED signature turns the caller RED (cutoff does NOT fire)" {
    const gpa = testing.allocator;
    const caller: NodeKey = .{ .kind = .codegen, .id = 1 };
    const sig: NodeKey = .{ .kind = .signature, .id = 2 };

    var prior_dag: Dag = .init(gpa);
    defer prior_dag.deinit(gpa);
    prior_dag.recordEdge(gpa, null, caller, 0xAAAA);
    prior_dag.recordEdge(gpa, caller, sig, 0xBBBB);
    var prior = try loadedOf(gpa, &prior_dag);
    defer prior.deinit(gpa);

    // fresh: the callee SIGNATURE fp flipped (a signature edit). The caller's own fp
    // also flips (its fingerprint folds the callee sig), so it is RED on both counts.
    var fresh: Dag = .init(gpa);
    defer fresh.deinit(gpa);
    fresh.recordEdge(gpa, null, caller, 0x9999);
    fresh.recordEdge(gpa, caller, sig, 0xDEAD);

    var rg = try RedGreen.init(gpa, &prior, &fresh);
    defer rg.deinit();
    try rg.run(&[_]NodeKey{caller});

    try testing.expectEqual(Verdict.red, (try rg.verdictOf(sig)));
    try testing.expectEqual(Verdict.red, (try rg.verdictOf(caller)));
}

test "red-green BACKDATING: a body edit cuts off the caller (its signature fp is stable)" {
    // THE HEADLINE: edit a fn body -> body(callee) fp flips RED, but signature(callee)
    // fp stays stable (the firewall). The caller depends on signature(callee), NOT
    // body(callee), so the caller verifies GREEN even though the callee body changed.
    const gpa = testing.allocator;
    const caller: NodeKey = .{ .kind = .codegen, .id = 1 };
    const sig: NodeKey = .{ .kind = .signature, .id = 2 };
    const body: NodeKey = .{ .kind = .body, .id = 2 };
    const callee_cg: NodeKey = .{ .kind = .codegen, .id = 2 };

    var prior_dag: Dag = .init(gpa);
    defer prior_dag.deinit(gpa);
    prior_dag.recordEdge(gpa, null, caller, 0xAAAA);
    prior_dag.recordEdge(gpa, caller, sig, 0xBBBB); // caller -> callee SIGNATURE
    prior_dag.recordEdge(gpa, null, callee_cg, 0xC0DE);
    prior_dag.recordEdge(gpa, callee_cg, sig, 0xBBBB); // callee codegen -> its own sig
    prior_dag.recordEdge(gpa, callee_cg, body, 0xBEEF); // callee codegen -> its body
    var prior = try loadedOf(gpa, &prior_dag);
    defer prior.deinit(gpa);

    // fresh: ONLY body(callee)'s fp flips; signature(callee) is stable. The callee's
    // OWN codegen fp flips (it lowers the edited body); the caller's fp is unchanged.
    var fresh: Dag = .init(gpa);
    defer fresh.deinit(gpa);
    fresh.recordEdge(gpa, null, caller, 0xAAAA); // caller fp STABLE
    fresh.recordEdge(gpa, caller, sig, 0xBBBB); // sig STABLE
    fresh.recordEdge(gpa, null, callee_cg, 0xCAFE); // callee codegen fp FLIPS
    fresh.recordEdge(gpa, callee_cg, sig, 0xBBBB);
    fresh.recordEdge(gpa, callee_cg, body, 0xF00D); // body fp FLIPS

    var rg = try RedGreen.init(gpa, &prior, &fresh);
    defer rg.deinit();
    try rg.run(&[_]NodeKey{ caller, callee_cg });

    // The callee's body + its own codegen are RED (re-lowered); the caller cuts off.
    try testing.expectEqual(Verdict.red, (try rg.verdictOf(body)));
    try testing.expectEqual(Verdict.red, (try rg.verdictOf(callee_cg)));
    try testing.expectEqual(Verdict.green, (try rg.verdictOf(sig)));
    try testing.expectEqual(Verdict.green, (try rg.verdictOf(caller)));
    // CUTOFF: a strict subset re-executed (the caller was spared).
    try testing.expect(rg.red_count < rg.verdict.count());
    try testing.expect(rg.green_count > 0);
}

test "red-green BACKDATING: a re-executed node with an UNCHANGED fp does not propagate" {
    // A node whose dep changed (so it re-executes / RED) but whose OWN fp is identical
    // must NOT be reported changed to ITS dependents — the backdating cutoff.
    const gpa = testing.allocator;
    const top: NodeKey = .{ .kind = .codegen, .id = 1 };
    const mid: NodeKey = .{ .kind = .body, .id = 2 };
    const leaf: NodeKey = .{ .kind = .parse, .id = 3 };

    var prior_dag: Dag = .init(gpa);
    defer prior_dag.deinit(gpa);
    prior_dag.recordEdge(gpa, null, top, 0x1111);
    prior_dag.recordEdge(gpa, top, mid, 0x2222);
    prior_dag.recordEdge(gpa, mid, leaf, 0x3333);
    var prior = try loadedOf(gpa, &prior_dag);
    defer prior.deinit(gpa);

    // fresh: leaf changed; mid re-executes but lands on the SAME fp (backdated); top
    // therefore stays GREEN.
    var fresh: Dag = .init(gpa);
    defer fresh.deinit(gpa);
    fresh.recordEdge(gpa, null, top, 0x1111); // top fp stable
    fresh.recordEdge(gpa, top, mid, 0x2222); // mid fp stable (backdated)
    fresh.recordEdge(gpa, mid, leaf, 0x4444); // leaf fp FLIPPED

    var rg = try RedGreen.init(gpa, &prior, &fresh);
    defer rg.deinit();
    try rg.run(&[_]NodeKey{top});

    try testing.expectEqual(Verdict.red, (try rg.verdictOf(leaf))); // input changed
    try testing.expectEqual(Verdict.red, (try rg.verdictOf(mid))); // re-executes (dep changed)
    try testing.expect(!rg.changedAfter(mid)); // ...but backdated: fp unchanged
    try testing.expectEqual(Verdict.green, (try rg.verdictOf(top))); // so top cuts off
}

test "red-green: a new node (absent from prior DAG) is RED" {
    const gpa = testing.allocator;
    const old: NodeKey = .{ .kind = .codegen, .id = 1 };
    const new: NodeKey = .{ .kind = .codegen, .id = 2 };

    var prior_dag: Dag = .init(gpa);
    defer prior_dag.deinit(gpa);
    prior_dag.recordEdge(gpa, null, old, 0xAAAA);
    var prior = try loadedOf(gpa, &prior_dag);
    defer prior.deinit(gpa);

    var fresh: Dag = .init(gpa);
    defer fresh.deinit(gpa);
    fresh.recordEdge(gpa, null, old, 0xAAAA);
    fresh.recordEdge(gpa, null, new, 0xBBBB); // a brand-new fn

    var rg = try RedGreen.init(gpa, &prior, &fresh);
    defer rg.deinit();
    try rg.run(&[_]NodeKey{ old, new });

    try testing.expectEqual(Verdict.green, (try rg.verdictOf(old)));
    try testing.expectEqual(Verdict.red, (try rg.verdictOf(new)));
}

test "red-green: a cycle in the prior edges raises QueryCycle" {
    const gpa = testing.allocator;
    const a: NodeKey = .{ .kind = .body, .id = 1 };
    const b: NodeKey = .{ .kind = .body, .id = 2 };

    var prior_dag: Dag = .init(gpa);
    defer prior_dag.deinit(gpa);
    prior_dag.recordEdge(gpa, null, a, 0xAA);
    prior_dag.recordEdge(gpa, a, b, 0xBB);
    prior_dag.recordEdge(gpa, b, a, 0xAA); // a -> b -> a, a cycle
    var prior = try loadedOf(gpa, &prior_dag);
    defer prior.deinit(gpa);

    var fresh: Dag = .init(gpa);
    defer fresh.deinit(gpa);
    fresh.recordEdge(gpa, null, a, 0xAA);
    fresh.recordEdge(gpa, a, b, 0xBB);
    fresh.recordEdge(gpa, b, a, 0xAA);

    var rg = try RedGreen.init(gpa, &prior, &fresh);
    defer rg.deinit();
    try testing.expectError(error.QueryCycle, rg.run(&[_]NodeKey{a}));
}

test "deserialize rejects bad magic, version, and truncation -> null (full rebuild fallback)" {
    const gpa = testing.allocator;
    var dag: Dag = .init(gpa);
    defer dag.deinit(gpa);
    dag.recordEdge(gpa, null, .{ .kind = .codegen, .id = 1 }, 0xAB);

    const blob = try dag.serialize(gpa);
    defer gpa.free(blob);

    // A valid blob loads.
    {
        var ok = (try deserialize(gpa, blob)).?;
        ok.deinit(gpa);
    }

    // Empty / too-short -> null.
    try testing.expectEqual(@as(?Loaded, null), try deserialize(gpa, &[_]u8{}));
    try testing.expectEqual(@as(?Loaded, null), try deserialize(gpa, blob[0..3]));

    // Corrupt magic -> null.
    {
        const bad = try gpa.dupe(u8, blob);
        defer gpa.free(bad);
        bad[0] ^= 0xFF;
        try testing.expectEqual(@as(?Loaded, null), try deserialize(gpa, bad));
    }
    // Corrupt version -> null.
    {
        const bad = try gpa.dupe(u8, blob);
        defer gpa.free(bad);
        bad[4] ^= 0xFF;
        try testing.expectEqual(@as(?Loaded, null), try deserialize(gpa, bad));
    }
    // A trailing-byte size mismatch -> null (size guard).
    {
        const bad = try gpa.alloc(u8, blob.len + 1);
        defer gpa.free(bad);
        @memcpy(bad[0..blob.len], blob);
        bad[blob.len] = 0;
        try testing.expectEqual(@as(?Loaded, null), try deserialize(gpa, bad));
    }
}
