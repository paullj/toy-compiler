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
//! Ordering is a single stable composite-key sort `(scope, byte_offset)` with
//! ties broken by emission (insertion) order. The stable sort is load-bearing:
//! the parallel Pass-C typecheck merges per-worker sinks in fn-id (slot) order
//! and then sorts ONCE, so the merged stream is byte-identical at every `-j`.

const std = @import("std");

/// The "untagged" scope: a single-file diagnostic carries no module id.
pub const NO_SCOPE: u32 = std.math.maxInt(u32);

/// The one diagnostic type shared by every stage. `byte_offset` points at the
/// offending token; the driver/CLI renders byte_offset -> line:col uniformly.
/// `scope` is the owning module id in graph mode, `NO_SCOPE` single-file. The
/// default keeps every existing `.{ .byte_offset = x, .message = m }` literal
/// compiling unchanged.
pub const Diagnostic = struct {
    byte_offset: u32,
    message: []const u8,
    scope: u32 = NO_SCOPE,
};

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

/// Deterministic stable total order: key `(scope, byte_offset)`, ties broken by
/// pre-sort (insertion) index. ONE stable sort drives BOTH modes — single-file
/// degenerates because every scope == NO_SCOPE. Idempotent; call once after all
/// emits/merges.
pub fn sort(self: *DiagnosticSink) void {
    std.sort.insertionContext(0, self.diags.items.len, SortCtx{ .diags = self.diags.items });
}

const SortCtx = struct {
    diags: []Diagnostic,
    pub fn lessThan(c: SortCtx, a: usize, b: usize) bool {
        if (c.diags[a].scope != c.diags[b].scope) return c.diags[a].scope < c.diags[b].scope;
        return c.diags[a].byte_offset < c.diags[b].byte_offset;
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

test "sort orders by (scope, byte_offset) and is stable for equal keys" {
    var sink = DiagnosticSink.init(testing.allocator);
    defer sink.deinit();

    // Emit OUT of (scope, byte_offset) order, including diagnostics that share a
    // byte_offset across different scopes, AND a long run of equal-key (scope,
    // byte_offset) diagnostics whose emission order must be preserved. The run is
    // long + scrambled so ANY non-stable reordering scrambles at least one pair —
    // this is what makes the test a real guard against reintroducing an unstable
    // sort (verified: it FAILS under std.sort.pdq/block), not a happy-path check.
    sink.setScope(1);
    try sink.emit(50, "s1@50");
    sink.setScope(0);
    try sink.emit(50, "s0@50"); // same byte_offset as s1@50, lower scope
    // 32 equal-key diagnostics at (scope 0, byte_offset 10), carrying a bit-
    // reversed payload as their distinguishing "emission tag". Stability means the
    // run comes out in EMISSION order (tags in bit-reversed sequence, NOT
    // ascending), which an unstable sort on 32 scrambled items will not reproduce.
    const n_equal = 32;
    var tags: [n_equal]u32 = undefined;
    for (0..n_equal) |i| {
        var r: u32 = 0;
        var v: u32 = @intCast(i);
        for (0..5) |_| {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        tags[i] = r;
    }
    var payloads: [n_equal][16]u8 = undefined;
    for (0..n_equal) |i| {
        const s = std.fmt.bufPrint(&payloads[i], "eq{d}", .{tags[i]}) catch unreachable;
        try sink.emit(10, s);
    }
    sink.setScope(1);
    try sink.emit(5, "s1@5");

    sink.sort();
    const got = sink.items();
    try testing.expectEqual(@as(usize, n_equal + 3), got.len);
    // scope 0 group first: the equal-key run (byte_offset 10) in EMISSION order.
    for (0..n_equal) |i| {
        try testing.expectEqual(@as(u32, 0), got[i].scope);
        try testing.expectEqual(@as(u32, 10), got[i].byte_offset);
        var buf: [16]u8 = undefined;
        const want = std.fmt.bufPrint(&buf, "eq{d}", .{tags[i]}) catch unreachable;
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
    try testing.expectEqualStrings("s0@10-second", got[1].message); // stability
    try testing.expectEqualStrings("s0@50", got[2].message);
    try testing.expectEqualStrings("s1@5", got[3].message);
    try testing.expectEqualStrings("s1@50", got[4].message);

    // An unstable sort would be ALLOWED to swap the equal-key pair; assert it did
    // not. This is the guard that pins the block->insertion (stable) change.
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
