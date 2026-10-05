//! End-to-end coverage for the `std/option` closure methods on the prelude `Option` and
//! `Result`. Each case compiles a program with the installed `toy`, runs it, and asserts
//! the exit code: the payload's lifetime across collections, which instance a call seals
//! to, and parallel determinism.

const std = @import("std");
const Io = std.Io;
const harness = @import("harness.zig");

test "option: Option.map over a str payload survives forced collections" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    // Every callback forces a collection while the method holds the only reference to a
    // fresh str (its payload or its result), so a missed root shows up as a wrong str.
    // 10 * 3 + 10 * 4 = 70.
    const codes = try harness.exitAtBothLevels(gpa, io, ".toy-test-option-gc",
        \\import std/option
        \\import core/mem_selftest
        \\fn fresh(i: int) -> str { "k".concat(i.to_string()) }
        \\fn main() -> int {
        \\    total := 0
        \\    i := 0
        \\    while i < 20 {
        \\        o := Option[str].some(fresh(i))
        \\        m := o.map(fn(s) {
        \\            _c := mem_selftest.forced_collect_count()
        \\            s.concat("!")
        \\        })
        \\        _c := mem_selftest.forced_collect_count()
        \\        want := fresh(i).concat("!")
        \\        if m.unwrap() != want { return 1 }
        \\        t := m.then(fn(s) {
        \\            _c := mem_selftest.forced_collect_count()
        \\            Option.some(s.concat("?"))
        \\        })
        \\        if t.unwrap() != want.concat("?") { return 2 }
        \\        n: Option[str] = Option.none
        \\        k := i + 100
        \\        d := n.or_else(fn() {
        \\            _c := mem_selftest.forced_collect_count()
        \\            fresh(k)
        \\        })
        \\        if d != fresh(k) { return 3 }
        \\        r: Result[int, str] = Result.err(fresh(i))
        \\        e := r.map_err(fn(s) {
        \\            _c := mem_selftest.forced_collect_count()
        \\            s.concat("#")
        \\        })
        \\        _d := mem_selftest.forced_collect_count()
        \\        if e.or_else(fn(s) { s.len() }) != fresh(i).len() + 1 { return 4 }
        \\        total = total + m.unwrap().len()
        \\        i = i + 1
        \\    }
        \\    return total
        \\}
        \\
    );
    try std.testing.expectEqual([2]u8{ 70, 70 }, codes);
}

test "option: two Option.map instances on one payload type run their own code" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    // One receiver type, three `U`s: each call must seal to its own instance row. Without
    // specialization the generic instances themselves run, so a call sealed to the wrong
    // row would hand back the wrong payload type.
    const src =
        \\import std/option
        \\fn dec(v: int) -> int { v - 1 }
        \\fn main() -> int {
        \\    o: Option[int] = Option.some(6)
        \\    a := o.map(fn(v) { v * 7 }).unwrap()
        \\    b := o.map(fn(v) { v.to_string().concat("x") }).unwrap()
        \\    c := o.map(fn(v) { v + 1 }).unwrap()
        \\    d := o.map(dec).unwrap()
        \\    big := o.map(fn(v) { v > 5 }).unwrap()
        \\    if b != "6x" || !big { return 1 }
        \\    r: Result[int, str] = Result.ok(3)
        \\    e := r.map(fn(v) { v * 2 }).or_else(fn(_m) { 0 })
        \\    f := r.map(fn(v) { v.to_string() }).or_else(fn(m) { m })
        \\    if f != "3" { return 2 }
        \\    return a + c + d + e
        \\}
        \\
    ;
    const dir = ".toy-test-option-rows";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const configs = [_][]const []const u8{
        &.{ "--no-cache", "-O0" },
        &.{ "--no-cache", "-O1" },
        &.{ "--no-cache", "-O0", "--no-specialize" },
    };
    for (configs, 0..) |flags, i| {
        const sub = try std.fmt.allocPrint(gpa, "{s}/{d}", .{ dir, i });
        defer gpa.free(sub);
        const prog = try harness.compile(gpa, io, sub, src, flags);
        defer gpa.free(prog);
        try std.testing.expectEqual(@as(u8, 60), try harness.runExit(gpa, io, prog));
    }
}

test "option: Option and Result methods are byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    // std/vec brings the unit count past the 64 at which Pass C and codegen fan out.
    const src =
        \\import std/option
        \\import std/vec
        \\fn half(n: int) -> Option[int] {
        \\    if n % 2 == 0 { return Option.some(n / 2) }
        \\    return Option.none
        \\}
        \\fn parse(s: str) -> Result[int, str] {
        \\    if s.len() == 0 { return Result.err("empty") }
        \\    return Result.ok(s.len())
        \\}
        \\fn main() -> int {
        \\    xs: Vec[int] = [5, 3, 8, 1]
        \\    a := xs.find(fn(x) { x > 6 }) |> .map(fn(x) { x * 2 }) |> .or_else(fn() { 0 })
        \\    b := xs.map(half).fold(0, fn(acc, o) { acc + o.then(half).or_else(fn() { 1 }) })
        \\    words := xs.map(fn(x) { Option[int].some(x).map(fn(v) { v.to_string() }).or_else(fn() { "" }) })
        \\    c := parse("abc").map(fn(v) { v + 1 }).then(fn(v) { Result[int, str].ok(v * 2) }).or_else(fn(_e) { 0 })
        \\    d := parse("").map_err(fn(m) { m.len() }).or_else(fn(k) { k * 3 })
        \\    return a + b + c + d + words.len()
        \\}
        \\
    ;
    const dir = ".toy-test-option-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const p1 = try harness.compile(gpa, io, dir ++ "/j1", src, &.{ "--no-cache", "-j1" });
    defer gpa.free(p1);
    const p8 = try harness.compile(gpa, io, dir ++ "/j8", src, &.{ "--no-cache", "-j8" });
    defer gpa.free(p8);

    const b1 = try Io.Dir.cwd().readFileAlloc(io, p1, gpa, .unlimited);
    defer gpa.free(b1);
    const b8 = try Io.Dir.cwd().readFileAlloc(io, p8, gpa, .unlimited);
    defer gpa.free(b8);

    try std.testing.expectEqual(b1.len, b8.len);
    // Only the one-byte code-sign nonce may differ.
    var differing: usize = 0;
    for (b1, b8) |x, y| {
        if (x != y) differing += 1;
    }
    try std.testing.expect(differing <= 1);
    try std.testing.expectEqual(@as(u8, 48), try harness.runExit(gpa, io, p1));
}
