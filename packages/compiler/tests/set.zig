//! End-to-end coverage for `Set[T]` over the erased `Map[T, ()]`: `.add` dedup, `.contains`
//! hit/miss, insertion-order key iteration across a grow (also through `each` and `filter`),
//! single-handle sharing, survival of a forced collection (conservative scan of the Set ->
//! Map -> entries -> str-key buffers), and -jN determinism. Each check surfaces as the
//! child's EXIT CODE (0 on success), the only thing this harness observes. Set is pure
//! library wiring over the Map: no new core surface.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const harness = @import("harness.zig");

const skipUnlessBackend = harness.skipUnlessBackend;

const compile = harness.compile;

const runExit = harness.runExit;

const dedup_src =
    \\import std/set
    \\fn main() -> int {
    \\    s := Set[int].new()
    \\    s.add(1)
    \\    s.add(1)
    \\    if s.len() != 1 { return 1 }
    \\    s.add(2)
    \\    if s.len() != 2 { return 2 }
    \\    return 0
    \\}
    \\
;

test "set: add de-duplicates in place (two add(1) -> len 1)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-set-dedup";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const prog = try compile(gpa, io, dir, dedup_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

const contains_src =
    \\import std/set
    \\fn main() -> int {
    \\    s := Set[int].new()
    \\    s.add(1)
    \\    s.add(2)
    \\    if !s.contains(1) { return 1 }
    \\    if s.contains(9) { return 2 }
    \\    return 0
    \\}
    \\
;

test "set: contains hit/miss" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-set-contains";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const prog = try compile(gpa, io, dir, contains_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

const order_src =
    \\import std/set
    \\import std/vec
    \\fn main() -> int {
    \\    s := Set[int].new()
    \\    order := [5, 13, 21, 3, 11, 19, 7, 2]
    \\    for k in order { s.add(k) }
    \\    j := 0
    \\    for x in s {
    \\        if x != order.get(j).unwrap() { return 1 }
    \\        j = j + 1
    \\    }
    \\    if j != 8 { return 2 }
    \\    return 0
    \\}
    \\
;

test "set: for-x iterates keys in insertion order across a grow" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-set-order";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // 8 keys force a grow from cap 8 -> 16 (grow at the 6th insert, load factor 0.75);
    // single-binding key iteration must still be insertion order.
    const prog = try compile(gpa, io, dir, order_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

const share_src =
    \\import std/set
    \\fn main() -> int {
    \\    s := Set[int].new()
    \\    n := s
    \\    s.add(1)
    \\    if !n.contains(1) { return 1 }
    \\    n.add(2)
    \\    if !s.contains(2) { return 2 }
    \\    if s.len() != 2 { return 3 }
    \\    return 0
    \\}
    \\
;

test "set: `n := s` shares the one table (single handle)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-set-share";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const prog = try compile(gpa, io, dir, share_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

const str_collect_src =
    \\import std/set
    \\import std/vec
    \\fn main() -> int {
    \\    s := Set[str].new()
    \\    s.add("a".concat("aa"))
    \\    s.add("b".concat("bb"))
    \\    s.add("c".concat("cc"))
    \\    i := 0
    \\    loop {
    \\        if i >= 20000 { break }
    \\        junk: Vec[int] = []
    \\        junk.push(i)
    \\        i = i + 1
    \\    }
    \\    if !s.contains("aaa") { return 1 }
    \\    if !s.contains("bbb") { return 2 }
    \\    if !s.contains("ccc") { return 3 }
    \\    if s.len() != 3 { return 4 }
    \\    return 0
    \\}
    \\
;

test "set: Set[str] with owned keys survives a forced collection (conservative scan)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-set-str-collect";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // Owned (concat'd) str keys are heap buffers; churning ~20000 dead Vec[int] forces at
    // least one collection. The conservative object scan must keep the Set's Map header,
    // entries array, and each 8-aligned str-key ptr's buffer alive, so every lookup by a
    // content-equal literal still hits. A mis-swept key buffer changes the exit.
    const prog = try compile(gpa, io, dir, str_collect_src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

test "set: a Set program is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    // Set's mono instances + the shared descriptor table are pure functions of source, so
    // codegen is -jN byte-identical.
    try harness.expectDeterministicBuild(gpa, io, ".toy-test-set-determinism", share_src, &.{}, null);
}

test "set: each, filter and map follow insertion order across a grow" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // Ten keys cross a grow and two are added again; every walk must follow the first adds,
    // and `map` keeps the first place of the two keys whose tens digit collides.
    const codes = try harness.exitAtBothLevels(gpa, io, ".toy-test-set-walk-order",
        \\import std/set
        \\import std/vec
        \\fn main() -> int {
        \\    s := Set[int].new()
        \\    order := [5, 13, 21, 3, 11, 19, 7, 2, 17, 23]
        \\    for k in order { s.add(k) }
        \\    s.add(5)
        \\    s.add(19)
        \\    j := 0
        \\    bad := 0
        \\    s.each(fn(x) {
        \\        if x != order[j] { bad = bad + 1 }
        \\        j = j + 1
        \\    })
        \\    if j != 10 || bad != 0 { return 1 }
        \\    big := s.filter(fn(x) { x > 10 })
        \\    want := order.filter(fn(x) { x > 10 })
        \\    if big.len() != want.len() { return 2 }
        \\    i := 0
        \\    big.each(fn(x) {
        \\        if x != want[i] { bad = bad + 1 }
        \\        i = i + 1
        \\    })
        \\    if bad != 0 || i != 6 { return 3 }
        \\    tens := s.map(fn(x) { x / 10 })
        \\    seen: Vec[int] = []
        \\    tens.each(fn(t) { seen.push(t) })
        \\    if seen.len() != 3 || seen[0] != 0 || seen[1] != 1 || seen[2] != 2 { return 4 }
        \\    return 0
        \\}
        \\
    );
    try std.testing.expectEqual([2]u8{ 0, 0 }, codes);
}

test "set: Set closure methods are byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The lambdas and the Vec instances bring the unit count past the 64 at which Pass C and
    // codegen fan out.
    const src =
        \\import std/set
        \\import std/vec
        \\fn main() -> int {
        \\    words: Vec[str] = ["fig", "ash", "elm", "ash", "oak", "fig"]
        \\    s := Set[str].new()
        \\    words.each(fn(w) { s.add(w) })
        \\    n := 0
        \\    s.each(fn(w) { n = n + w.len() })
        \\    short := s.filter(fn(w) { w != "elm" })
        \\    if !short.contains("oak") || !words.any(fn(w) { w == "ash" }) { return 1 }
        \\    lens := Set[int].new()
        \\    short.each(fn(w) { lens.add(w.len()) })
        \\    odd := Set[int].new()
        \\    words.map(fn(w) { w.len() * 2 + 1 }).each(fn(x) { odd.add(x) })
        \\    upper := s.map(fn(w) { w.concat("!") })
        \\    return n + short.len() * 10 + lens.len() + odd.filter(fn(x) { x > 5 }).len() + upper.len()
        \\}
        \\
    ;

    try harness.expectDeterministicBuild(gpa, io, ".toy-test-set-methods-determinism", src, &.{}, 48);
}
