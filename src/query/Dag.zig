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

/// The KIND of a query node. `signature` and `body` are DISTINCT kinds for the
/// same fn id: a caller depends on `signature(callee)`, NEVER on `body(callee)` —
/// the firewall the spike proves. `lex`/`parse`/`codegen` align with the on-disk
/// Cache identity (their id folds the Cache.Key digest); the fine-grained
/// typecheck kinds (`signature`/`body`/`type_of`/`layout`/`resolve_name`) carry a
/// deterministic global id and are memoized in-build (NOT a Cache phase).
pub const Kind = enum(u8) {
    lex,
    parse,
    resolve,
    signature,
    body,
    type_of,
    layout,
    resolve_name,
    codegen,
};

/// An IN-MEMORY node key. NOT a `Cache.Phase` — adding Cache phases would change
/// on-disk identity (cold-rebuild), which M16 avoids. `id` is the deterministic
/// global id ([C11]) for the fine-grained typecheck kinds; for lex/parse/codegen
/// it folds the existing `Cache.Key.digest()` so DAG nodes align with cache
/// identity.
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
