//! The diagnostic accumulator every semantic pass emits through. It owns the
//! diagnostic list AND the message-string lifetimes: a heap message allocated by
//! `emitFmt` is freed exactly once (on `deinit`, or by the `Owned` handed to a
//! `Result`); a static message passed to `emit` is borrowed and never freed.
//! Ownership is structural, not by convention — callers no longer hand-manage a
//! parallel `owned_msgs` list.
//!
//! Module tagging rides INSIDE each `Diagnostic` as `scope` (single-file leaves
//! it `NO_SCOPE`; graph mode stamps the owning module id via `setScope`). Keeping
//! it in the struct makes the old "diags.len == diag_mods.len" length-sync
//! hazard a type-level guarantee.
//!
//! Ordering is a single stable composite-key sort `(scope, byte_offset, code, message)`.
//! `code` is a stable enum ordinal (deterministic across `-j`); for uncoded (`.none`)
//! diagnostics it is inert, so the message is the final tiebreak so duplicates land ADJACENT for
//! `dedupAdjacent` (even interleaved same-offset repeats). The sort key depends only
//! on the diagnostic fields, so the merged stream is byte-identical at every `-j`:
//! the parallel Pass-C typecheck merges per-worker sinks in fn-id (slot) order and
//! then sorts ONCE.

const std = @import("std");

// The diagnostic datum + its `NO_SCOPE` sentinel are defined in `Diagnostic.zig`
// (the type's home); re-exported so `Sink.Diagnostic`/`Sink.NO_SCOPE` and the
// internal uses below keep working.
const diag = @import("Diagnostic.zig");
pub const NO_SCOPE = diag.NO_SCOPE;
pub const NO_RELATED = diag.NO_RELATED;
pub const Diagnostic = diag.Diagnostic;

// The code registry + model types the coded builder threads into each diagnostic.
const codes = @import("codes.zig");
const model = @import("model.zig");
pub const Code = codes.Code;

const DiagnosticSink = @This();

gpa: std.mem.Allocator,
diags: std.ArrayList(Diagnostic) = .empty,
/// Heap messages only (the ones `emitFmt` allocated); freed once on deinit.
/// Static `emit` messages are never in here, so they are never freed.
owned: std.ArrayList([]u8) = .empty,
/// Stamped onto every subsequent emit. Set per fn/module in graph mode; a
/// single-file caller leaves it `NO_SCOPE`.
cur_scope: u32 = NO_SCOPE,

pub fn init(gpa: std.mem.Allocator) DiagnosticSink {
    return .{ .gpa = gpa };
}

pub fn deinit(self: *DiagnosticSink) void {
    for (self.owned.items) |m| self.gpa.free(m);
    self.owned.deinit(self.gpa);
    self.diags.deinit(self.gpa);
    self.* = undefined;
}

/// Set the module id stamped onto subsequent emits. Single-file callers never
/// call this (the sink stays `NO_SCOPE`).
pub fn setScope(self: *DiagnosticSink, scope: u32) void {
    self.cur_scope = scope;
}

/// Record a static (borrowed) message. NOT freed by the sink.
pub fn emit(self: *DiagnosticSink, byte_offset: u32, message: []const u8) !void {
    try self.diags.append(self.gpa, .{
        .byte_offset = byte_offset,
        .message = message,
        .scope = self.cur_scope,
    });
}

/// Format a message, take ownership of the buffer, and record the diagnostic.
/// Ordering is load-bearing for OOM safety: reserve the diag slot FIRST (so the
/// final append is infallible), THEN allocPrint, THEN track in `owned`. The
/// `errdefer free(msg)` covers only the window before `owned.append` succeeds —
/// once the message is tracked, the infallible append can't fail, so the message
/// is never both tracked-and-freed (double free) nor freed-and-untracked (leak).
pub fn emitFmt(self: *DiagnosticSink, byte_offset: u32, comptime fmt: []const u8, args: anytype) !void {
    try self.diags.ensureUnusedCapacity(self.gpa, 1);
    const msg = try std.fmt.allocPrint(self.gpa, fmt, args);
    errdefer self.gpa.free(msg);
    try self.owned.append(self.gpa, msg);
    self.diags.appendAssumeCapacity(.{
        .byte_offset = byte_offset,
        .message = msg,
        .scope = self.cur_scope,
    });
}

/// Record a static (borrowed) coded message: like `emit`, but stamps the stable
/// `code` and its registry default severity onto the diagnostic. Existing sites keep
/// calling `emit` (code stays `.none`); coded sites call this or the builder.
pub fn emitCode(self: *DiagnosticSink, code: codes.Code, byte_offset: u32, message: []const u8) !void {
    try self.diags.append(self.gpa, .{
        .byte_offset = byte_offset,
        .message = message,
        .scope = self.cur_scope,
        .code = code,
        .severity = codes.defaultSeverity(code),
    });
}

/// Format + own a coded message. Same load-bearing reserve->allocPrint->track OOM
/// ordering as `emitFmt`; additionally stamps `code` + its default severity.
pub fn emitFmtCode(self: *DiagnosticSink, code: codes.Code, byte_offset: u32, comptime fmt: []const u8, args: anytype) !void {
    return self.emitFmtCodeRelated(code, byte_offset, diag.NO_RELATED, fmt, args);
}

/// Like `emitFmtCode`, but also records a RELATED prior location: `related` is a byte
/// offset in the SAME scope (e.g. a duplicate's first definition), rendered as a
/// secondary "previously defined here" label. Same load-bearing reserve->allocPrint->
/// track OOM ordering; `related` is a memcpy-trivial `u32` on the POD, not part of the
/// sort/dedup key.
pub fn emitFmtCodeRelated(self: *DiagnosticSink, code: codes.Code, byte_offset: u32, related: u32, comptime fmt: []const u8, args: anytype) !void {
    try self.diags.ensureUnusedCapacity(self.gpa, 1);
    const msg = try std.fmt.allocPrint(self.gpa, fmt, args);
    errdefer self.gpa.free(msg);
    try self.owned.append(self.gpa, msg);
    self.diags.appendAssumeCapacity(.{
        .byte_offset = byte_offset,
        .message = msg,
        .scope = self.cur_scope,
        .code = code,
        .severity = codes.defaultSeverity(code),
        .related = related,
    });
}

/// Start a fluent coded diagnostic: `sink.err(.R0001).span(a,b).emit()` or
/// `.emitFmt(fmt, args)`. Severity is sourced from the registry default only — no
/// builder method accepts a raw severity, so no rule-dependent state can leak into
/// the cached POD. The builder is purely additive; `emit`/`emitFmt` are untouched.
pub fn err(self: *DiagnosticSink, code: codes.Code) Builder {
    return .{ .sink = self, .code = code, .severity = codes.defaultSeverity(code) };
}

/// Fluent builder for a coded diagnostic. `span` sets the primary byte offset (its
/// `start`); `emit`/`emitFmt` terminate, reusing `emitCode`/`emitFmtCode`'s owning +
/// OOM discipline. Value-typed (each setter returns a copy), so it never aliases.
pub const Builder = struct {
    sink: *DiagnosticSink,
    code: codes.Code,
    severity: model.Severity,
    primary: ?model.Span = null,

    /// Set the primary span; the emitted diagnostic's `byte_offset` is `s`.
    pub fn span(b: Builder, s: u32, e: u32) Builder {
        var n = b;
        n.primary = .{ .start = s, .end = e };
        return n;
    }

    fn offset(b: Builder) u32 {
        return if (b.primary) |p| p.start else 0;
    }

    /// Terminate with a static (borrowed) message.
    pub fn emit(b: Builder, message: []const u8) !void {
        try b.sink.emitCode(b.code, b.offset(), message);
    }

    /// Terminate with an owned formatted message.
    pub fn emitFmt(b: Builder, comptime fmt: []const u8, args: anytype) !void {
        try b.sink.emitFmtCode(b.code, b.offset(), fmt, args);
    }
};

/// Serial merge of a worker's sink into this one, transferring ownership of both
/// its diags and its owned messages, then emptying it (so its `deinit` is a
/// no-op and there is no double-free). Capacity is reserved FIRST so the appends
/// are infallible — no `try` after the reserve means no half-merged window.
/// Callers MUST merge in slot (fn-id) order to preserve the equal-key tiebreak.
pub fn merge(self: *DiagnosticSink, other: *DiagnosticSink) !void {
    try self.diags.ensureUnusedCapacity(self.gpa, other.diags.items.len);
    try self.owned.ensureUnusedCapacity(self.gpa, other.owned.items.len);
    self.diags.appendSliceAssumeCapacity(other.diags.items);
    self.owned.appendSliceAssumeCapacity(other.owned.items);
    other.diags.clearAndFree(other.gpa);
    other.owned.clearAndFree(other.gpa);
}

/// Deterministic stable total order: key `(scope, byte_offset, code, message)`, with any
/// residual ties (fully identical tuples) broken by pre-sort (insertion) index.
/// ONE stable sort drives BOTH modes — single-file degenerates because every
/// scope == NO_SCOPE. Idempotent; call once after all emits/merges.
pub fn sort(self: *DiagnosticSink) void {
    std.sort.insertionContext(0, self.diags.items.len, SortCtx{ .diags = self.diags.items });
    self.dedupAdjacent();
}

/// Collapse EXACT-duplicate diagnostics keyed on `(scope, byte_offset, code, message)`.
/// PRECONDITION: called right after the stable sort, whose full key is
/// `(scope, byte_offset, code, message)` — so every exact-tuple repeat is contiguous,
/// even interleaved same-offset duplicates like emission `A, B, A` (the message
/// tiebreak pulls the two `A`s together). One linear compaction keeps the first of
/// each adjacent run and drops later exact repeats. Idempotent (dedup of an
/// already-deduped array is a no-op). COLLECTION stays complete: this is a post-sort
/// view collapse of identical repeats, never distinct errors.
///
/// Frees NOTHING: a dropped duplicate's `emitFmt` buffer stays tracked in the
/// parallel `owned` list and is freed exactly once on `deinit`/`Owned.deinit`. Only
/// `diags` shrinks — freeing here would risk double-freeing (a surviving identical
/// message may be a DIFFERENT `emitFmt` buffer).
fn dedupAdjacent(self: *DiagnosticSink) void {
    const d = self.diags.items;
    if (d.len < 2) return;
    var w: usize = 1;
    for (d[1..]) |cur| {
        const prev = d[w - 1];
        const same = cur.scope == prev.scope and
            cur.byte_offset == prev.byte_offset and
            cur.code == prev.code and
            std.mem.eql(u8, cur.message, prev.message);
        if (!same) {
            d[w] = cur;
            w += 1;
        }
    }
    self.diags.shrinkRetainingCapacity(w);
}

const SortCtx = struct {
    diags: []Diagnostic,
    pub fn lessThan(c: SortCtx, a: usize, b: usize) bool {
        if (c.diags[a].scope != c.diags[b].scope) return c.diags[a].scope < c.diags[b].scope;
        if (c.diags[a].byte_offset != c.diags[b].byte_offset) return c.diags[a].byte_offset < c.diags[b].byte_offset;
        // Code before message: a stable enum ordinal (deterministic across `-j`,
        // unlike a pointer). For all existing `.none` diagnostics this component is
        // constant/inert, so the message tiebreak still drives ordering there.
        if (c.diags[a].code != c.diags[b].code)
            return @intFromEnum(c.diags[a].code) < @intFromEnum(c.diags[b].code);
        // Final tiebreak: message bytes. This groups exact-triple duplicates
        // ADJACENTLY so `dedupAdjacent` catches interleaved same-offset repeats
        // (emission order `A, B, A` would otherwise leave the two `A`s split by
        // `B`). Depends only on message bytes, so it stays stable across `-j`
        // modes. Distinct messages at one key now come out lexicographically
        // (not emission order); that is fine — dedup only needs equal triples
        // adjacent, and no test pins intra-offset emission order for DISTINCT
        // messages under a shared key.
        return std.mem.order(u8, c.diags[a].message, c.diags[b].message) == .lt;
    }
    pub fn swap(c: SortCtx, a: usize, b: usize) void {
        std.mem.swap(Diagnostic, &c.diags[a], &c.diags[b]);
    }
};

pub fn items(self: *const DiagnosticSink) []const Diagnostic {
    return self.diags.items;
}

pub fn count(self: *const DiagnosticSink) usize {
    return self.diags.items.len;
}

/// The owned slices handed to a stage `Result`. The Result frees them via
/// `deinit`; static messages are never in `owned`, so they are never freed.
pub const Owned = struct {
    diags: []Diagnostic,
    owned: [][]u8,

    pub fn deinit(o: *Owned, gpa: std.mem.Allocator) void {
        gpa.free(o.diags);
        for (o.owned) |m| gpa.free(m);
        gpa.free(o.owned);
        o.* = undefined;
    }
};

/// Hand the owned slices to a Result and reset this sink to empty. After this
/// the Result owns + frees the memory; the sink may be reused or deinit'd. If the
/// second detach fails, the first is freed so nothing strands.
pub fn toOwned(self: *DiagnosticSink) !Owned {
    const diags = try self.diags.toOwnedSlice(self.gpa);
    errdefer self.gpa.free(diags);
    const owned = try self.owned.toOwnedSlice(self.gpa);
    return .{ .diags = diags, .owned = owned };
}

// ---- tests -----------------------------------------------------------------

const testing = std.testing;

test "sort orders by (scope, byte_offset, message) deterministically for equal (scope, offset)" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();

    // Emit OUT of order, including diagnostics that share a byte_offset across
    // different scopes, AND a long run of same-(scope, byte_offset) diagnostics with
    // DISTINCT messages emitted in a scrambled order. The message is the final sort
    // tiebreak, so that run must come out in LEXICOGRAPHIC message order regardless of
    // emission order — the deterministic total order dedup relies on. The run is long
    // + scrambled so any failure to apply the message tiebreak (e.g. an unstable sort
    // that leaves same-(scope, offset) items in arrival order) mis-orders a pair.
    sink.setScope(1);
    try sink.emit(50, "s1@50");
    sink.setScope(0);
    try sink.emit(50, "s0@50"); // same byte_offset as s1@50, lower scope
    // 32 same-key diagnostics at (scope 0, byte_offset 10). Their messages are
    // zero-padded (`eq00`..`eq31`) so lexicographic order equals numeric index order;
    // they are EMITTED in bit-reversed (scrambled) order, so recovering ascending
    // order proves the message tiebreak drove the sort.
    const n_equal = 32;
    var emit_order: [n_equal]u32 = undefined;
    for (0..n_equal) |i| {
        var r: u32 = 0;
        var v: u32 = @intCast(i);
        for (0..5) |_| {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        emit_order[i] = r;
    }
    var payloads: [n_equal][16]u8 = undefined;
    for (0..n_equal) |i| {
        const s = std.fmt.bufPrint(&payloads[i], "eq{d:0>2}", .{emit_order[i]}) catch unreachable;
        try sink.emit(10, s);
    }
    sink.setScope(1);
    try sink.emit(5, "s1@5");

    sink.sort();
    const got = sink.items();
    try testing.expectEqual(@as(usize, n_equal + 3), got.len);
    // scope 0 group first: the same-key run (byte_offset 10) in LEXICOGRAPHIC message
    // order, i.e. eq00, eq01, ..., eq31 (ascending index), NOT emission order.
    for (0..n_equal) |i| {
        try testing.expectEqual(@as(u32, 0), got[i].scope);
        try testing.expectEqual(@as(u32, 10), got[i].byte_offset);
        var buf: [16]u8 = undefined;
        const want = std.fmt.bufPrint(&buf, "eq{d:0>2}", .{i}) catch unreachable;
        try testing.expectEqualStrings(want, got[i].message);
    }
    try testing.expectEqualStrings("s0@50", got[n_equal].message);
    // scope 1 group, byte_offset ascending.
    try testing.expectEqualStrings("s1@5", got[n_equal + 1].message);
    try testing.expectEqualStrings("s1@50", got[n_equal + 2].message);
}

test "sort orders by (scope, byte_offset) — small smoke check" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    sink.setScope(1);
    try sink.emit(50, "s1@50");
    sink.setScope(0);
    try sink.emit(50, "s0@50");
    try sink.emit(10, "s0@10-first");
    try sink.emit(10, "s0@10-second");
    sink.setScope(1);
    try sink.emit(5, "s1@5");

    sink.sort();
    const got = sink.items();
    try testing.expectEqual(@as(usize, 5), got.len);
    try testing.expectEqualStrings("s0@10-first", got[0].message);
    try testing.expectEqualStrings("s0@10-second", got[1].message); // message tiebreak: "first" < "second"
    try testing.expectEqualStrings("s0@50", got[2].message);
    try testing.expectEqualStrings("s1@5", got[3].message);
    try testing.expectEqualStrings("s1@50", got[4].message);

    // The two same-(scope, byte_offset) diagnostics must come out message-ordered
    // (deterministic), not in an arbitrary order an unstable sort could pick.
    try testing.expect(got[0].byte_offset == got[1].byte_offset);
    try testing.expect(got[0].scope == got[1].scope);
}

test "single-file (NO_SCOPE) diagnostics survive sort and round-trip untagged" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.emit(30, "a");
    try sink.emitFmt(10, "val {d}", .{7});
    sink.sort();
    var o = try sink.toOwned();
    defer o.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), o.diags.len);
    try testing.expectEqual(NO_SCOPE, o.diags[0].scope);
    try testing.expectEqual(@as(u32, 10), o.diags[0].byte_offset);
    try testing.expectEqualStrings("val 7", o.diags[0].message);
    try testing.expectEqual(NO_SCOPE, o.diags[1].scope);
}

test "merge preserves slot-order tiebreak and equals a single-sink emit" {
    const gpa = testing.allocator;

    // Three worker sinks; sinks 2 and 0 share a byte_offset (cross-scope tie).
    var w0 = DiagnosticSink.init(gpa);
    w0.setScope(0);
    try w0.emit(100, "w0");
    var w1 = DiagnosticSink.init(gpa);
    w1.setScope(1);
    try w1.emit(20, "w1");
    var w2 = DiagnosticSink.init(gpa);
    w2.setScope(2);
    try w2.emit(100, "w2");

    var out = DiagnosticSink.init(gpa);
    defer out.deinit();
    try out.merge(&w0);
    try out.merge(&w1);
    try out.merge(&w2);
    // Merge transfers ownership + empties each worker; their deinit is a no-op.
    w0.deinit();
    w1.deinit();
    w2.deinit();
    out.sort();

    // A single sink that emitted the same diagnostics in slot order then sorted.
    var ref = DiagnosticSink.init(gpa);
    defer ref.deinit();
    ref.setScope(0);
    try ref.emit(100, "w0");
    ref.setScope(1);
    try ref.emit(20, "w1");
    ref.setScope(2);
    try ref.emit(100, "w2");
    ref.sort();

    try testing.expectEqual(ref.count(), out.count());
    for (ref.items(), out.items()) |a, b| {
        try testing.expectEqual(a.scope, b.scope);
        try testing.expectEqual(a.byte_offset, b.byte_offset);
        try testing.expectEqualStrings(a.message, b.message);
    }
}

test "deinit frees a mix of static and heap messages with no leak" {
    var sink = DiagnosticSink.init(testing.allocator);
    try sink.emit(1, "static one");
    try sink.emitFmt(2, "heap {s}", .{"two"});
    try sink.emit(3, "static three");
    try sink.emitFmt(4, "heap {d}", .{4});
    sink.deinit(); // testing.allocator fails the test on any leak.
}

test "toOwned then Owned.deinit frees heap messages with no leak" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.emit(1, "static");
    try sink.emitFmt(2, "heap {d}", .{2});
    var o = try sink.toOwned();
    o.deinit(testing.allocator);
    // The sink is now empty; deinit (deferred) is a safe no-op.
    try testing.expectEqual(@as(usize, 0), sink.count());
}

test "emitFmt OOM after allocPrint leaks nothing and never double-frees" {
    // Fail the `owned.append` that follows a successful `allocPrint`: the errdefer
    // must free the message (it is not yet tracked), so deinit under the leak-
    // detecting backing allocator sees no leak and no double-free. This guards the
    // load-bearing reserve->allocPrint->track ordering: if the message were tracked
    // before this failure, the errdefer would double-free it on deinit.
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 2 });
    var sink = DiagnosticSink.init(failing.allocator());
    // alloc #0 = diags.ensureUnusedCapacity, #1 = allocPrint, #2 = owned.append.
    try testing.expectError(error.OutOfMemory, sink.emitFmt(7, "msg {d}", .{1}));
    try testing.expectEqual(@as(usize, 0), sink.count());
    sink.deinit();
}

test "dedup collapses adjacent exact duplicates (keeps the first-emitted)" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    // Same (scope, byte_offset, message) triple emitted twice, plus a DISTINCT
    // message at the same offset. After sort()+dedup the exact repeat collapses; the
    // distinct one survives. The survivor of the collapsed pair is the first-emitted.
    try sink.emit(10, "dup");
    try sink.emit(10, "dup");
    try sink.emit(10, "other");
    sink.sort();
    try testing.expectEqual(@as(usize, 2), sink.count());
    try testing.expectEqualStrings("dup", sink.items()[0].message);
    try testing.expectEqualStrings("other", sink.items()[1].message);
}

test "dedup collapses INTERLEAVED same-offset duplicates (emission A,B,A)" {
    // The two `A`s are emitted non-adjacently, split by `B`, all at one offset. The
    // message sort tiebreak pulls the identical triples together so dedupAdjacent
    // catches them: result is A, B (both survivors, no `A` repeat). This is the case
    // a `(scope, byte_offset)`-only sort would miss (it would leave A, B, A).
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.emit(10, "A");
    try sink.emit(10, "B");
    try sink.emit(10, "A");
    sink.sort();
    try testing.expectEqual(@as(usize, 2), sink.count());
    try testing.expectEqualStrings("A", sink.items()[0].message);
    try testing.expectEqualStrings("B", sink.items()[1].message);
}

test "dedup keeps distinct messages at the same (scope, byte_offset)" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.emit(10, "a");
    try sink.emit(10, "b");
    sink.sort();
    try testing.expectEqual(@as(usize, 2), sink.count());
}

test "dedup keeps the same message at different byte_offsets" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.emit(10, "same");
    try sink.emit(20, "same");
    sink.sort();
    try testing.expectEqual(@as(usize, 2), sink.count());
}

test "dedup keeps the same (offset,message) under different scopes" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    sink.setScope(0);
    try sink.emit(10, "x");
    sink.setScope(1);
    try sink.emit(10, "x");
    sink.sort();
    try testing.expectEqual(@as(usize, 2), sink.count());
}

test "dedup with emitFmt (heap) messages frees exactly once (no double-free)" {
    // Under testing.allocator: a dropped duplicate's heap buffer must STILL be freed
    // exactly once on deinit (it stays tracked in `owned`), proving dedup shrinks only
    // `diags` and never touches `owned`.
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.emitFmt(10, "val {d}", .{7});
    try sink.emitFmt(10, "val {d}", .{7}); // identical text, DIFFERENT buffer
    try sink.emitFmt(10, "val {d}", .{9});
    sink.sort();
    try testing.expectEqual(@as(usize, 2), sink.count());
    try testing.expectEqualStrings("val 7", sink.items()[0].message);
    try testing.expectEqualStrings("val 9", sink.items()[1].message);
}

test "dedup is idempotent (a second sort changes nothing)" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.emit(10, "dup");
    try sink.emit(10, "dup");
    try sink.emit(5, "keep");
    sink.sort();
    try testing.expectEqual(@as(usize, 2), sink.count());
    sink.sort();
    try testing.expectEqual(@as(usize, 2), sink.count());
    try testing.expectEqualStrings("keep", sink.items()[0].message);
    try testing.expectEqualStrings("dup", sink.items()[1].message);
}

test "sort orders by code BEFORE message at an equal (scope, byte_offset)" {
    // Three diagnostics at one (scope, byte_offset), same message, DIFFERENT codes.
    // The code component (an enum ordinal) sorts before message, so they come out
    // none(0) < R0001 < R0002 regardless of emission order.
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.emitCode(.R0002, 10, "x");
    try sink.emit(10, "x"); // .none
    try sink.emitCode(.R0001, 10, "x");
    sink.sort();
    const got = sink.items();
    try testing.expectEqual(@as(usize, 3), got.len);
    try testing.expectEqual(codes.Code.none, got[0].code);
    try testing.expectEqual(codes.Code.R0001, got[1].code);
    try testing.expectEqual(codes.Code.R0002, got[2].code);
}

test "dedup respects code: identical (scope, offset, message) but different codes both survive" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.emitCode(.R0001, 10, "same");
    try sink.emitCode(.R0002, 10, "same");
    sink.sort();
    try testing.expectEqual(@as(usize, 2), sink.count());
    // And two truly-identical coded diagnostics DO collapse.
    var s2 = DiagnosticSink.init(testing.allocator);
    defer s2.deinit();
    try s2.emitCode(.R0001, 10, "same");
    try s2.emitCode(.R0001, 10, "same");
    s2.sort();
    try testing.expectEqual(@as(usize, 1), s2.count());
}

test "the coded builder stamps code + registry default severity" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.err(.R0001).span(7, 9).emitFmt("undeclared '{s}'", .{"x"});
    try sink.err(.T0004).emit("recursive"); // no span => byte_offset 0
    sink.sort();
    const got = sink.items();
    try testing.expectEqual(@as(usize, 2), got.len);
    // T0004 sorts first (byte_offset 0 < 7).
    try testing.expectEqual(@as(u32, 0), got[0].byte_offset);
    try testing.expectEqual(codes.Code.T0004, got[0].code);
    try testing.expectEqual(model.Severity.err, got[0].severity);
    try testing.expectEqual(@as(u32, 7), got[1].byte_offset);
    try testing.expectEqual(codes.Code.R0001, got[1].code);
    try testing.expectEqualStrings("undeclared 'x'", got[1].message);
}

test "builder emitFmt OOM after allocPrint leaks nothing (mirrors the emitFmt OOM gate)" {
    // The builder routes through `emitFmtCode`, which reuses the reserve->allocPrint->
    // track ordering. Fail the `owned.append` after a successful `allocPrint`: the
    // errdefer must free the message, so deinit sees no leak and no double-free.
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 2 });
    var sink = DiagnosticSink.init(failing.allocator());
    try testing.expectError(error.OutOfMemory, sink.err(.R0001).span(1, 2).emitFmt("msg {d}", .{1}));
    try testing.expectEqual(@as(usize, 0), sink.count());
    sink.deinit();
}

test "sort is idempotent and count tracks items" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();
    try sink.emit(3, "c");
    try sink.emit(1, "a");
    try sink.emit(2, "b");
    sink.sort();
    const first = [_][]const u8{ sink.items()[0].message, sink.items()[1].message, sink.items()[2].message };
    sink.sort();
    try testing.expectEqualStrings(first[0], sink.items()[0].message);
    try testing.expectEqualStrings(first[1], sink.items()[1].message);
    try testing.expectEqualStrings(first[2], sink.items()[2].message);
    try testing.expectEqual(@as(usize, 3), sink.count());
}
