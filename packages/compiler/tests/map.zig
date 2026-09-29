//! End-to-end coverage for the erased Map: runtime descriptor dispatch on raw key cells,
//! insertion-order iteration (including across a grow), single-handle sharing, survival of
//! a forced collection, and -jN determinism. Each check surfaces as the child's EXIT CODE
//! (0 on success), the only thing this harness observes.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const harness = @import("harness.zig");

const skipUnlessBackend = harness.skipUnlessBackend;

const compile = harness.compile;

const runExit = harness.runExit;

const collision_src =
    \\import core/map_selftest
    \\fn main() -> int { return map_selftest.collision_survives() }
    \\
;

test "map: collisions + a grow survive a forced collection (conservative scan)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-map-collision";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const prog = try compile(gpa, io, dir, collision_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

const order_src =
    \\import std/map
    \\import std/vec
    \\fn main() -> int {
    \\    m := Map[int, int].new()
    \\    order := [5, 13, 21, 3, 11, 19, 7, 2]
    \\    for k in order { m.set(k, k * 2) }
    \\    j := 0
    \\    for k, v in m {
    \\        if k != order.get(j).unwrap() { return 1 }
    \\        if v != k * 2 { return 2 }
    \\        j = j + 1
    \\    }
    \\    if j != 8 { return 3 }
    \\    return 0
    \\}
    \\
;

test "map: for k, v iterates in insertion order across a grow" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-map-order";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // 8 keys force a grow from cap 8 -> 16; iteration must still be insertion order.
    const prog = try compile(gpa, io, dir, order_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

const share_src =
    \\import std/map
    \\fn main() -> int {
    \\    m := Map[int, int].new()
    \\    n := m
    \\    m.set(1, 100)
    \\    if n.get(1).unwrap_or(0) != 100 { return 1 }
    \\    n.set(2, 200)
    \\    if m.get(2).unwrap_or(0) != 200 { return 2 }
    \\    if m.len() != 2 { return 3 }
    \\    return 0
    \\}
    \\
;

test "map: `n := m` shares the one table (single handle)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-map-share";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const prog = try compile(gpa, io, dir, share_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

const custom_eq_src =
    \\import std/map
    \\struct DK { id: int, tag: int }
    \\impl DK has Hashable {
    \\    fn hash(self) -> int { return self.id }
    \\    fn eq(self, o: DK) -> bool { return self.id == o.id }
    \\}
    \\fn main() -> int {
    \\    m := Map[DK, int].new()
    \\    m.set(DK{ id: 1, tag: 10 }, 100)
    \\    m.set(DK{ id: 1, tag: 99 }, 200)
    \\    if m.len() != 1 { return 1 }
    \\    if m.get(DK{ id: 1, tag: 0 }).unwrap_or(0) != 200 { return 2 }
    \\    return 0
    \\}
    \\
;

test "map: an explicit `impl has Hashable` custom eq is honored (dedups on one field)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-map-custom-eq";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // Two inserts with the SAME custom key (same id, different tag) must update in place
    // (len 1, value 200) — the erased dispatch resolves the key's explicit Hashable eq/hash,
    // not the structural field walk (which would keep both entries).
    const prog = try compile(gpa, io, dir, custom_eq_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

test "map: an explicit-impl-Hashable key program is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-map-custom-eq-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const p1 = try compile(gpa, io, dir ++ "/j1", custom_eq_src, &.{ "--no-cache", "-j1" });
    defer gpa.free(p1);
    const p8 = try compile(gpa, io, dir ++ "/j8", custom_eq_src, &.{ "--no-cache", "-j8" });
    defer gpa.free(p8);

    const b1 = try Io.Dir.cwd().readFileAlloc(io, p1, gpa, .unlimited);
    defer gpa.free(b1);
    const b8 = try Io.Dir.cwd().readFileAlloc(io, p8, gpa, .unlimited);
    defer gpa.free(b8);

    try std.testing.expectEqual(b1.len, b8.len);
    var differing: usize = 0;
    for (b1, b8) |x, y| {
        if (x != y) differing += 1;
    }
    try std.testing.expect(differing <= 1);
}

const struct_key_src =
    \\import std/map
    \\struct P { x: int, y: int }
    \\fn main() -> int {
    \\    m := Map[P, int].new()
    \\    m.set(P{ x: 1, y: 1 }, 11)
    \\    m.set(P{ x: 2, y: 2 }, 22)
    \\    m.set(P{ x: 1, y: 1 }, 33)
    \\    if m.len() != 2 { return 1 }
    \\    i := 0
    \\    for k, v in m {
    \\        if i == 0 { if k.x != 1 || v != 33 { return 2 } }
    \\        if i == 1 { if k.x != 2 || v != 22 { return 3 } }
    \\        i = i + 1
    \\    }
    \\    if i != 2 { return 4 }
    \\    return 0
    \\}
    \\
;

test "map: a plain struct key round-trips and iterates in insertion order" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-map-struct-key";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // A plain struct is structurally Hashable, so it works as a key with no explicit impl;
    // an update keeps the original insertion position.
    const prog = try compile(gpa, io, dir, struct_key_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

test "map: the Map program is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-map-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // Same `-o` basename (`prog`) in both dirs, so the ad-hoc code-sign IDENTIFIER is
    // identical — only the one-byte code-sign nonce may differ. The descriptor table +
    // erased units + the Map's mono instances are pure functions of source, so codegen
    // is `-jN` byte-identical.
    const p1 = try compile(gpa, io, dir ++ "/j1", share_src, &.{ "--no-cache", "-j1" });
    defer gpa.free(p1);
    const p8 = try compile(gpa, io, dir ++ "/j8", share_src, &.{ "--no-cache", "-j8" });
    defer gpa.free(p8);

    const b1 = try Io.Dir.cwd().readFileAlloc(io, p1, gpa, .unlimited);
    defer gpa.free(b1);
    const b8 = try Io.Dir.cwd().readFileAlloc(io, p8, gpa, .unlimited);
    defer gpa.free(b8);

    try std.testing.expectEqual(b1.len, b8.len);
    var differing: usize = 0;
    for (b1, b8) |x, y| {
        if (x != y) differing += 1;
    }
    try std.testing.expect(differing <= 1);
}

// A zero-sized value column (`V = ()`, the Set backing): the cell stride is `size_of[K] + 0`
// and every value store/load is a no-op. Exercises `mp_set`/`mp_val_at`/`mp_grow` with a unit
// value across a forced grow (this whole path was a lower crash before `()` became a value).
const unit_val_int_src =
    \\import std/map
    \\import std/vec
    \\fn main() -> int {
    \\    m := Map[int, ()].new()
    \\    order := [5, 13, 21, 3, 11, 19, 7, 2]
    \\    for k in order { m.set(k, ()) }
    \\    m.set(5, ())
    \\    if m.len() != 8 { return 1 }
    \\    if !m.get(21).is_some() { return 2 }
    \\    if m.get(99).is_some() { return 3 }
    \\    j := 0
    \\    for k, v in m {
    \\        if k != order.get(j).unwrap() { return 4 }
    \\        j = j + 1
    \\    }
    \\    if j != 8 { return 5 }
    \\    return 0
    \\}
    \\
;

test "map: Map[int, ()] set/get/for-k,v with a zero-sized value column across a grow" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-map-unit-int";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const prog = try compile(gpa, io, dir, unit_val_int_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

const unit_val_str_src =
    \\import std/map
    \\fn main() -> int {
    \\    m := Map[str, ()].new()
    \\    m.set("aaa", ())
    \\    m.set("bbb", ())
    \\    m.set("aaa", ())
    \\    if m.len() != 2 { return 1 }
    \\    if !m.get("bbb").is_some() { return 2 }
    \\    if m.get("zzz").is_some() { return 3 }
    \\    return 0
    \\}
    \\
;

test "map: Map[str, ()] dedups an owned key with a zero-sized value" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-map-unit-str";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const prog = try compile(gpa, io, dir, unit_val_str_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}
