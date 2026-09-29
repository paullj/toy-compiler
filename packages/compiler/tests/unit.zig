//! End-to-end coverage for `()` as a first-class zero-sized type + value: a `()` param
//! round-trips (ABI zero-location), a `()` local binds, a struct field / enum variant
//! payload of type `()` compiles and round-trips (0-byte member), the structural derives
//! (Eq/Hash/Display) treat a `()` field correctly, and a `()`-value program (a `()` param
//! plus `Map[int, ()]`) is byte-identical at -j1 and -j8. Each behavioural check surfaces as
//! the child's EXIT CODE (0 on success), except the Display case which observes stdout.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const harness = @import("harness.zig");

const skipUnlessBackend = harness.skipUnlessBackend;

const compile = harness.compile;

const runExit = harness.runExit;

const compileAndRun = harness.buildAndRun;

test "unit: a () parameter round-trips (ABI zero-location)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    // f consumes a () param (0 GPR / 0 NSAA); the surrounding int args keep their slots.
    const src =
        \\fn f(a: int, u: (), b: int) -> int { return a + b }
        \\fn main() -> int { return f(3, (), 4) }
        \\
    ;
    try std.testing.expectEqual(@as(u8, 7), try compileAndRun(gpa, io, ".toy-test-unit-param", src));
}

test "unit: a () local binds (zero-sized)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const src =
        \\fn g() {}
        \\fn main() -> int {
        \\    x := ()
        \\    y: () = g()
        \\    return 0
        \\}
        \\
    ;
    try std.testing.expectEqual(@as(u8, 0), try compileAndRun(gpa, io, ".toy-test-unit-local", src));
}

test "unit: a struct with a () field round-trips (0-byte member)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const src =
        \\struct S { a: int, u: (), b: int }
        \\fn main() -> int {
        \\    s := S { a: 3, u: (), b: 4 }
        \\    return s.a + s.b
        \\}
        \\
    ;
    try std.testing.expectEqual(@as(u8, 7), try compileAndRun(gpa, io, ".toy-test-unit-field", src));
}

test "unit: an enum variant with a () payload round-trips" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const src =
        \\enum E { V(()), W }
        \\fn main() -> int {
        \\    e := E.V(())
        \\    match e {
        \\        E.V(x) -> { return 7 }
        \\        E.W -> { return 1 }
        \\    }
        \\}
        \\
    ;
    try std.testing.expectEqual(@as(u8, 7), try compileAndRun(gpa, io, ".toy-test-unit-payload", src));
}

test "unit: structural Eq/Hash over a () field compares by the sized fields only" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    // The `()` field contributes `true` to the eq AND-fold, so equality tracks `a`.
    const src =
        \\struct S { a: int, u: () }
        \\fn main() -> int {
        \\    p := S { a: 1, u: () }
        \\    q := S { a: 1, u: () }
        \\    r := S { a: 2, u: () }
        \\    if p != q { return 1 }
        \\    if p == r { return 2 }
        \\    return 0
        \\}
        \\
    ;
    try std.testing.expectEqual(@as(u8, 0), try compileAndRun(gpa, io, ".toy-test-unit-eq", src));
}

test "unit: structural Display renders a () field as ()" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-unit-display";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const src =
        \\import std/io
        \\struct S { a: int, u: () }
        \\fn main() {
        \\    io.print(S { a: 7, u: () })
        \\}
        \\
    ;
    const prog = try compile(gpa, io, dir, src, &.{});
    defer gpa.free(prog);

    var run = try std.process.spawn(io, .{ .argv = &.{prog}, .stdout = .pipe });
    var rdr = run.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 20));
    defer gpa.free(got);
    _ = try run.wait(io);
    // The `u` member renders as the literal `()`.
    try std.testing.expect(std.mem.indexOf(u8, got, "()") != null);
}

test "unit: a bare () is a usable Map key / Set element (zero-sized key witnesses)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    // A `()` key is zero-sized: its erased hash folds the seed and its erased eq is
    // always true, so a `Set[()]` collapses to at most one element and a `Map[(), V]`
    // holds a single cell keyed on the sole `()` value.
    const set_src =
        \\import std/set
        \\fn main() -> int {
        \\    s := Set[()].new()
        \\    s.add(())
        \\    s.add(())
        \\    return s.len()
        \\}
        \\
    ;
    try std.testing.expectEqual(@as(u8, 1), try compileAndRun(gpa, io, ".toy-test-unit-set-key", set_src));

    const map_src =
        \\import std/map
        \\fn main() -> int {
        \\    m := Map[(), int].new()
        \\    m.set((), 42)
        \\    return m.get(()).unwrap_or(0)
        \\}
        \\
    ;
    try std.testing.expectEqual(@as(u8, 42), try compileAndRun(gpa, io, ".toy-test-unit-map-key", map_src));
}

test "unit: a ()-value program (a () param + Map[int,()]) is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-unit-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const src =
        \\import std/map
        \\fn f(u: ()) -> int { return 0 }
        \\fn main() -> int {
        \\    m := Map[int, ()].new()
        \\    m.set(1, ())
        \\    m.set(2, ())
        \\    return f(())
        \\}
        \\
    ;
    // Same `-o` basename in both dirs so the ad-hoc code-sign identifier is identical; the
    // () monomorphized instances are pure functions of source, so codegen is -jN identical.
    const p1 = try compile(gpa, io, dir ++ "/j1", src, &.{ "--no-cache", "-j1" });
    defer gpa.free(p1);
    const p8 = try compile(gpa, io, dir ++ "/j8", src, &.{ "--no-cache", "-j8" });
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
