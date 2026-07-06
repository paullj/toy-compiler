//! Composite (`App`) type intern table — CHECK-TIME only (M4). A peer data module,
//! sibling of `Mono.zig`.
//!
//! An `App` is a generic-struct application `Ctor[args..]` (e.g. `Box[int]`). It is a
//! `Type` whose `Kind == .app` and whose `struct_id` field is an index INTO THIS
//! TABLE. `App`s are FORMED during (parallel) Pass-C body checking + the (serial)
//! monomorphization tail; each reachable ground `App` is later REIFIED to an ordinary
//! `struct_id` before the layout snapshot, so this table lives only for the duration
//! of one whole-graph typecheck.
//!
//! CONTENT-ADDRESSED interning is the correctness keystone: `intern(ctor, args)` maps
//! a structurally-equal `(ctor, args)` to the SAME index within a run, so `Type.eql`
//! comparing two `.app`s by their index is exactly structural equality. A nested-`App`
//! arg contributes its own index (via the reused `struct_id`), so the flat dedup key
//! is well-defined once inner `App`s are interned first.
//!
//! DETERMINISM: `intern` is a pure function of `(ctor, args)`, but the INDEX it
//! returns is run-order-dependent (whichever thread interns a key first wins the next
//! slot). The index therefore MUST NOT feed any fingerprint / mangled symbol / codegen
//! decision — those consume the REIFIED `struct_id`, assigned serially in
//! `writeStructuralKey` order (an index-INDEPENDENT recursive key), not this index.
//!
//! THREAD-SAFETY: `App`s are formed during parallel Pass C. A single mutex guards both
//! `intern` (write) and `at` (read); `at` returns the `Entry` BY VALUE (its `args`
//! slice is an independent, never-freed-until-teardown heap allocation, so the returned
//! slice header stays valid after the lock is released even if the entry array reallocs).

const std = @import("std");
const Type = @import("../layout/Engine.zig").Type;

const Composite = @This();

/// One interned composite. `args` is OWNED (freed at teardown); its elements are
/// concrete/`type_var`/nested-`App` types in generic-param order. `ctor_is_enum`
/// (M6) disambiguates the `ctor` id space: struct ids and enum ids are independent,
/// so a struct-App `S[..]` and an enum-App `E[..]` with the SAME `ctor` number must
/// never share a dedup/structural key (else one would reify as the other). It folds
/// into both `writeFlatKey` (interning) and `writeStructuralKey` (reify order).
pub const Entry = struct {
    ctor: u32,
    args: []const Type,
    ctor_is_enum: bool = false,
};

entries: std.ArrayList(Entry) = .empty,
/// Flat structural key (`writeFlatKey`) -> entry index. Keys are OWNED.
dedup: std.StringHashMapUnmanaged(u32) = .empty,
/// A lock-free atomic-bool spinlock (mirrors `query/Cache.zig`): 0.16's `std.Io.Mutex`
/// needs an `Io` to block, but the critical sections here are tiny (a key build + a
/// hashmap upsert / a slice copy), so a spinlock keeps this module Io-free and safe if
/// Pass-C `App` interning runs in parallel.
lock_state: std.atomic.Value(bool) = .init(false),

fn lock(c: *Composite) void {
    while (c.lock_state.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
        std.atomic.spinLoopHint();
    }
}

fn unlock(c: *Composite) void {
    c.lock_state.store(false, .release);
}

pub fn deinit(c: *Composite, gpa: std.mem.Allocator) void {
    for (c.entries.items) |e| gpa.free(@constCast(e.args));
    c.entries.deinit(gpa);
    var it = c.dedup.keyIterator();
    while (it.next()) |k| gpa.free(k.*);
    c.dedup.deinit(gpa);
}

/// The flat dedup key for `(ctor, args)`: `ctor` then each arg's
/// `(kind, struct_id, enum_id)` at fixed width. A nested `.app` arg contributes its
/// own interned index (via `struct_id`), and a `.type_var` arg its ordinal — so the
/// key is injective within a run (inner `App`s interned first).
fn writeFlatKey(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), ctor: u32, args: []const Type, ctor_is_enum: bool) !void {
    // Struct and enum ctor ids share one index space, so the leading discriminator
    // byte keeps struct#N and enum#N from aliasing to the same key.
    try buf.append(gpa, @intFromBool(ctor_is_enum));
    var w: [4]u8 = undefined;
    std.mem.writeInt(u32, &w, ctor, .little);
    try buf.appendSlice(gpa, &w);
    for (args) |a| try a.appendKeyBytes(gpa, buf);
}

/// Intern `(ctor, args)` to a stable composite index; structurally-equal calls return
/// the same index. Mutex-guarded (Apps form during parallel Pass C).
pub fn intern(c: *Composite, gpa: std.mem.Allocator, ctor: u32, args: []const Type, ctor_is_enum: bool) !u32 {
    c.lock();
    defer c.unlock();
    var keybuf: std.ArrayList(u8) = .empty;
    defer keybuf.deinit(gpa);
    try writeFlatKey(gpa, &keybuf, ctor, args, ctor_is_enum);
    const gop = try c.dedup.getOrPut(gpa, keybuf.items);
    if (gop.found_existing) return gop.value_ptr.*;
    gop.key_ptr.* = try gpa.dupe(u8, keybuf.items);
    const idx: u32 = @intCast(c.entries.items.len);
    gop.value_ptr.* = idx;
    const owned = try gpa.dupe(Type, args);
    try c.entries.append(gpa, .{ .ctor = ctor, .args = owned, .ctor_is_enum = ctor_is_enum });
    return idx;
}

/// The interned `(ctor, args)` at `idx`, BY VALUE. The `args` slice remains valid after
/// the lock is released (independent heap allocation, freed only at teardown).
pub fn at(c: *Composite, idx: u32) Entry {
    c.lock();
    defer c.unlock();
    return c.entries.items[idx];
}

/// Append the INDEX-INDEPENDENT recursive structural key of `ty` to `buf`: an `.app`
/// expands to a marker + `ctor` + the (recursively-expanded) args + an end marker; any
/// other type folds `(kind, struct_id, enum_id)`. Because it expands nested `App`s to
/// their contents rather than their interned index, the key is a pure function of
/// SOURCE — the driver of deterministic reified-`struct_id` assignment under `-jN`.
pub fn writeStructuralKey(c: *Composite, gpa: std.mem.Allocator, ty: Type, buf: *std.ArrayList(u8)) !void {
    if (ty.isApp()) {
        const e = c.at(ty.appIdx());
        try buf.append(gpa, 0xAA); // app-open marker
        try buf.append(gpa, @intFromBool(e.ctor_is_enum)); // struct-App vs enum-App: distinct reify order
        var w: [4]u8 = undefined;
        std.mem.writeInt(u32, &w, e.ctor, .little);
        try buf.appendSlice(gpa, &w);
        for (e.args) |a| try c.writeStructuralKey(gpa, a, buf);
        try buf.append(gpa, 0xBB); // app-close marker (arity/nesting delimiter)
    } else {
        try ty.appendKeyBytes(gpa, buf);
    }
}

/// The `App`-nesting depth of `ty`: a non-`App` is 0; `App(c, args)` is
/// `1 + max(depth(arg))`. Finite (each interned `App` is a finite structure). Drives
/// the M4 termination guard (T0017) — an unbounded `f[T] -> f[Box[T]]` chain forms
/// ever-deeper `App`s and is rejected before it can hang/OOM.
pub fn appDepth(c: *Composite, ty: Type) u32 {
    if (!ty.isApp()) return 0;
    const e = c.at(ty.appIdx());
    var m: u32 = 0;
    for (e.args) |a| {
        const d = c.appDepth(a);
        if (d > m) m = d;
    }
    return 1 + m;
}

const testing = std.testing;

test "intern is content-addressed: structurally-equal Apps share an index" {
    const gpa = testing.allocator;
    var c: Composite = .{};
    defer c.deinit(gpa);
    const a = try c.intern(gpa, 3, &.{Type.int}, false);
    const b = try c.intern(gpa, 3, &.{Type.int}, false);
    try testing.expectEqual(a, b); // same (ctor, args) => same index
    const d = try c.intern(gpa, 3, &.{Type.bool}, false);
    try testing.expect(a != d); // different arg => different index
    const e = try c.intern(gpa, 4, &.{Type.int}, false);
    try testing.expect(a != e); // different ctor => different index
    // Nested App: Box[Box[int]] vs a fresh Box[Box[int]] share an index (the inner
    // App interns to the same index first, so the outer flat key matches).
    const inner = try c.intern(gpa, 3, &.{Type.int}, false);
    const outer1 = try c.intern(gpa, 3, &.{Type.app(inner)}, false);
    const outer2 = try c.intern(gpa, 3, &.{Type.app(try c.intern(gpa, 3, &.{Type.int}, false))}, false);
    try testing.expectEqual(outer1, outer2);
}

test "M6: a struct-App and an enum-App with the SAME ctor intern to DIFFERENT indices" {
    const gpa = testing.allocator;
    var c: Composite = .{};
    defer c.deinit(gpa);
    // ctor id 3 as a STRUCT vs as an ENUM: the id spaces are independent, so these
    // are distinct types and must not alias (the discriminator is in the flat key).
    const s = try c.intern(gpa, 3, &.{Type.int}, false);
    const e = try c.intern(gpa, 3, &.{Type.int}, true);
    try testing.expect(s != e);
    // ...and their INDEX-INDEPENDENT structural keys differ too (so struct-App vs
    // enum-App get a stable, distinct place in the (depth, key) reify order).
    var ks: std.ArrayList(u8) = .empty;
    defer ks.deinit(gpa);
    var ke: std.ArrayList(u8) = .empty;
    defer ke.deinit(gpa);
    try c.writeStructuralKey(gpa, Type.app(s), &ks);
    try c.writeStructuralKey(gpa, Type.app(e), &ke);
    try testing.expect(!std.mem.eql(u8, ks.items, ke.items));
    // The bit round-trips on the entry.
    try testing.expect(!c.at(s).ctor_is_enum);
    try testing.expect(c.at(e).ctor_is_enum);
}

test "writeStructuralKey is index-independent (order-stable under -jN)" {
    const gpa = testing.allocator;
    // Two tables that intern the SAME App in DIFFERENT orders assign it different
    // indices, but its structural key must be byte-identical.
    var c1: Composite = .{};
    defer c1.deinit(gpa);
    var c2: Composite = .{};
    defer c2.deinit(gpa);
    // c1: intern an unrelated App first, so Box[int] lands at index 1.
    _ = try c1.intern(gpa, 9, &.{Type.bool}, false);
    const idx1 = try c1.intern(gpa, 3, &.{Type.int}, false);
    // c2: intern Box[int] first, at index 0.
    const idx2 = try c2.intern(gpa, 3, &.{Type.int}, false);
    try testing.expect(idx1 != idx2); // indices differ across tables

    var k1: std.ArrayList(u8) = .empty;
    defer k1.deinit(gpa);
    var k2: std.ArrayList(u8) = .empty;
    defer k2.deinit(gpa);
    try c1.writeStructuralKey(gpa, Type.app(idx1), &k1);
    try c2.writeStructuralKey(gpa, Type.app(idx2), &k2);
    try testing.expectEqualSlices(u8, k1.items, k2.items);
}

test "appDepth counts nesting" {
    const gpa = testing.allocator;
    var c: Composite = .{};
    defer c.deinit(gpa);
    try testing.expectEqual(@as(u32, 0), c.appDepth(Type.int));
    const b1 = try c.intern(gpa, 3, &.{Type.int}, false); // Box[int]
    try testing.expectEqual(@as(u32, 1), c.appDepth(Type.app(b1)));
    const b2 = try c.intern(gpa, 3, &.{Type.app(b1)}, false); // Box[Box[int]]
    try testing.expectEqual(@as(u32, 2), c.appDepth(Type.app(b2)));
}

test "at returns the entry by value with a stable args slice" {
    const gpa = testing.allocator;
    var c: Composite = .{};
    defer c.deinit(gpa);
    const idx = try c.intern(gpa, 7, &.{ Type.int, Type.bool }, false);
    const e = c.at(idx);
    try testing.expectEqual(@as(u32, 7), e.ctor);
    try testing.expectEqual(@as(usize, 2), e.args.len);
    try testing.expect(Type.eql(e.args[0], Type.int));
    try testing.expect(Type.eql(e.args[1], Type.bool));
}
