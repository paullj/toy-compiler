//! Warm-cache soundness for parameter modes across modules (#25). A `mut` receiver or
//! param changes how a caller passes the argument and how the callee reads it, so flipping
//! one must rebuild both sides even when no body changes. Each test builds in a cwd of its
//! own (the `.toy/` cache is cwd-relative), and builds the edited program cold in a second
//! cwd as the reference, because `--no-cache` still writes the cache.

const std = @import("std");
const Io = std.Io;
const harness = @import("harness.zig");

const File = struct { name: []const u8, src: []const u8 };

fn writeAll(gpa: std.mem.Allocator, io: Io, dir: []const u8, files: []const File) !void {
    for (files) |f| {
        const path = try std.fs.path.join(gpa, &.{ dir, f.name });
        defer gpa.free(path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = f.src });
    }
}

/// `root/sub`, created, as an absolute path (owned).
fn makeDir(gpa: std.mem.Allocator, io: Io, root: []const u8, sub: []const u8) ![:0]u8 {
    const rel = try std.fs.path.join(gpa, &.{ root, sub });
    defer gpa.free(rel);
    try Io.Dir.cwd().createDirPath(io, rel);
    const out_a = try std.fs.path.join(gpa, &.{ rel, "a" });
    defer gpa.free(out_a);
    try Io.Dir.cwd().createDirPath(io, out_a);
    const out_b = try std.fs.path.join(gpa, &.{ rel, "b" });
    defer gpa.free(out_b);
    try Io.Dir.cwd().createDirPath(io, out_b);
    return Io.Dir.cwd().realPathFileAlloc(io, rel, gpa);
}

/// Build `v1` warm-cache style in `root/w` and run it, apply `v2` (the files it replaces)
/// over it, then: a warm build runs to `v2_exit`, a warm `--verify` build passes, and a
/// cold build of the same sources in `root/c` runs to `v2_exit` too. Each build writes a
/// fresh `-o` dir, since macOS kills an image re-signed in place.
fn flip(root: []const u8, v1: []const File, v1_exit: u8, v2: []const File, v2_exit: u8) !void {
    return flipAt(root, &.{}, v1, v1_exit, v2, v2_exit);
}

/// `flip` with `flags` on every build.
fn flipAt(root: []const u8, flags: []const []const u8, v1: []const File, v1_exit: u8, v2: []const File, v2_exit: u8) !void {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    Io.Dir.cwd().deleteTree(io, root) catch {};
    defer Io.Dir.cwd().deleteTree(io, root) catch {};
    const w = try makeDir(gpa, io, root, "w");
    defer gpa.free(w);
    const c = try makeDir(gpa, io, root, "c");
    defer gpa.free(c);

    try writeAll(gpa, io, w, v1);
    try buildWith(gpa, io, w, &.{ "build", "main.toy", "-o", "a/prog" }, flags);
    try std.testing.expectEqual(v1_exit, try harness.runExitIn(gpa, io, w, "a/prog"));

    try writeAll(gpa, io, w, v2);
    try buildWith(gpa, io, w, &.{ "build", "main.toy", "-o", "b/prog" }, flags);
    try std.testing.expectEqual(v2_exit, try harness.runExitIn(gpa, io, w, "b/prog"));
    try buildWith(gpa, io, w, &.{ "build", "main.toy", "-o", "b/prog", "--verify" }, flags);

    try writeAll(gpa, io, c, v1);
    try writeAll(gpa, io, c, v2);
    try buildWith(gpa, io, c, &.{ "build", "main.toy", "-o", "b/prog", "--no-cache" }, flags);
    try std.testing.expectEqual(v2_exit, try harness.runExitIn(gpa, io, c, "b/prog"));
}

fn buildWith(gpa: std.mem.Allocator, io: Io, cwd_abs: []const u8, args: []const []const u8, flags: []const []const u8) !void {
    const all = try std.mem.concat(gpa, []const u8, &.{ args, flags });
    defer gpa.free(all);
    try harness.buildIn(gpa, io, cwd_abs, all);
}

test "warm cache: an imported method's self->mut self flip rebuilds its caller (#25)" {
    const main_src =
        \\import p
        \\fn main() -> int {
        \\    q := p.mk()
        \\    _s := q.show()
        \\    return q.x
        \\}
        \\
    ;
    const v1 =
        \\pub struct P { pub x: int }
        \\impl P {
        \\    fn show(self) -> str { "pee" }
        \\}
        \\pub fn mk() -> P { P{ x: 1 } }
        \\
    ;
    const v2 =
        \\pub struct P { pub x: int }
        \\impl P {
        \\    fn show(mut self) -> str {
        \\        self.x = 5
        \\        "pee"
        \\    }
        \\}
        \\pub fn mk() -> P { P{ x: 1 } }
        \\
    ;
    try flip(".toy-test-places-recv", &.{ .{ .name = "main.toy", .src = main_src }, .{ .name = "p.toy", .src = v1 } }, 1, &.{.{ .name = "p.toy", .src = v2 }}, 5);
}

test "warm cache: an imported fn's x->mut x flip rebuilds the caller" {
    const main_src =
        \\import lib
        \\fn main() -> int {
        \\    c := 1
        \\    lib.bump(c)
        \\    return c
        \\}
        \\
    ;
    try flip(
        ".toy-test-places-param",
        &.{ .{ .name = "main.toy", .src = main_src }, .{ .name = "lib.toy", .src = "pub fn bump(n: int) { n = n + 1 }\n" } },
        1,
        &.{.{ .name = "lib.toy", .src = "pub fn bump(mut n: int) { n = n + 1 }\n" }},
        2,
    );
}

test "warm cache: an instance whose type param no value param names flips its caller's key" {
    // No value param of `bump` names `T` (as with `mem.ga_new[T]()` in every Vec), yet the
    // caller passes `c` as the instance declares it, so the instance's modes are in its key.
    const main_src =
        \\import lib
        \\fn main() -> int {
        \\    c := 1
        \\    lib.bump[bool](c)
        \\    return c
        \\}
        \\
    ;
    try flip(
        ".toy-test-places-turbofish",
        &.{ .{ .name = "main.toy", .src = main_src }, .{ .name = "lib.toy", .src = "pub fn bump[T](n: int) { n = n + 1 }\n" } },
        1,
        &.{.{ .name = "lib.toy", .src = "pub fn bump[T](mut n: int) { n = n + 1 }\n" }},
        2,
    );
}

test "warm cache: a mode flip with an unchanged body plus a caller edit serves no stale callee (#25 callee half)" {
    const lib_v1 =
        \\pub struct P { pub x: int }
        \\impl P {
        \\    fn show(self) -> int { self.x }
        \\}
        \\pub fn mk() -> P { P{ x: 1 } }
        \\
    ;
    const lib_v2 =
        \\pub struct P { pub x: int }
        \\impl P {
        \\    fn show(mut self) -> int { self.x }
        \\}
        \\pub fn mk() -> P { P{ x: 1 } }
        \\
    ;
    const main_v1 =
        \\import p
        \\fn main() -> int {
        \\    q := p.mk()
        \\    return q.show() + q.x + 0
        \\}
        \\
    ;
    const main_v2 =
        \\import p
        \\fn main() -> int {
        \\    q := p.mk()
        \\    return q.show() + q.x + 1
        \\}
        \\
    ;
    try flip(
        ".toy-test-places-callee",
        &.{ .{ .name = "main.toy", .src = main_v1 }, .{ .name = "p.toy", .src = lib_v1 } },
        2,
        &.{ .{ .name = "main.toy", .src = main_v2 }, .{ .name = "p.toy", .src = lib_v2 } },
        3,
    );
}

const escape_main =
    \\import std/vec
    \\import lib
    \\fn make(keep: Vec[fn() -> int], n: int) -> int {
    \\    k := n * 2
    \\    lib.run(fn() -> int { k + n }, keep)
    \\}
    \\fn main() -> int {
    \\    keep: Vec[fn() -> int] = []
    \\    r := make(keep, 5)
    \\    if keep.len() == 0 { return r }
    \\    return keep[0]() + 20
    \\}
    \\
;
const lib_calls =
    \\import std/vec
    \\pub fn run(f: fn() -> int, keep: Vec[fn() -> int]) -> int { f() }
    \\
;
const lib_keeps =
    \\import std/vec
    \\pub fn run(f: fn() -> int, keep: Vec[fn() -> int]) -> int {
    \\    keep.push(f)
    \\    f()
    \\}
    \\
;

test "warm -O1: an imported fn that starts keeping its fn param rebuilds the caller's frame env" {
    try flipAt(".toy-test-places-esc-on", &.{"-O1"}, &.{ .{ .name = "main.toy", .src = escape_main }, .{ .name = "lib.toy", .src = lib_calls } }, 15, &.{.{ .name = "lib.toy", .src = lib_keeps }}, 35);
}

test "warm -O1: an imported fn that stops keeping its fn param rebuilds the caller's heap env" {
    try flipAt(".toy-test-places-esc-off", &.{"-O1"}, &.{ .{ .name = "main.toy", .src = escape_main }, .{ .name = "lib.toy", .src = lib_keeps } }, 35, &.{.{ .name = "lib.toy", .src = lib_calls }}, 15);
}

// The env (a 320-byte copy plus the accumulator) is over the frame budget, so it is a GC
// cell whichever way the callee goes and only the accumulator's box flips: these pin the
// kept-box half of the caller's key, apart from its frame-env bits.
const big_env_main =
    \\import std/vec
    \\import lib
    \\struct Q { a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int }
    \\struct Big { p: Q, q: Q, r: Q, s: Q, t: Q }
    \\fn setup(keep: Vec[fn(int)]) -> int {
    \\    q := Q { a: 1, b: 0, c: 0, d: 0, e: 0, f: 0, g: 0, h: 0 }
    \\    x := Big { p: q, q: q, r: q, s: q, t: q }
    \\    total := 0
    \\    lib.each(fn(v: int) { total = total + v + x.p.a }, keep)
    \\    total
    \\}
    \\fn smash(n: int) -> int {
    \\    a := n
    \\    b := n + 1
    \\    a + b
    \\}
    \\fn main() -> int {
    \\    keep: Vec[fn(int)] = []
    \\    r := setup(keep)
    \\    _z := smash(1000)
    \\    if keep.len() > 0 {
    \\        keep[0](5)
    \\        keep[0](5)
    \\        return r + 50
    \\    }
    \\    return r
    \\}
    \\
;
const lib_each_calls =
    \\import std/vec
    \\pub fn each(f: fn(int), keep: Vec[fn(int)]) {
    \\    f(1)
    \\    f(2)
    \\}
    \\
;
const lib_each_keeps =
    \\import std/vec
    \\pub fn each(f: fn(int), keep: Vec[fn(int)]) {
    \\    keep.push(f)
    \\    f(1)
    \\    f(2)
    \\}
    \\
;

test "warm -O1: a callee that starts keeping its fn param re-boxes the caller's accumulator in a heap env" {
    try flipAt(".toy-test-places-box-on", &.{"-O1"}, &.{ .{ .name = "main.toy", .src = big_env_main }, .{ .name = "lib.toy", .src = lib_each_calls } }, 5, &.{.{ .name = "lib.toy", .src = lib_each_keeps }}, 55);
}

test "warm -O1: a callee that stops keeping its fn param unboxes the caller's accumulator in a heap env" {
    try flipAt(".toy-test-places-box-off", &.{"-O1"}, &.{ .{ .name = "main.toy", .src = big_env_main }, .{ .name = "lib.toy", .src = lib_each_keeps } }, 55, &.{.{ .name = "lib.toy", .src = lib_each_calls }}, 5);
}

const acc_main =
    \\import std/vec
    \\import lib
    \\fn main() -> int {
    \\    xs: Vec[int] = [1, 2, 3]
    \\    keep: Vec[fn(int)] = []
    \\    total := 0
    \\    lib.each(xs, fn(v: int) { total = total + v }, keep)
    \\    return total
    \\}
    \\
;
const lib_acc_calls = "import std/vec\npub fn each(xs: Vec[int], f: fn(int), keep: Vec[fn(int)]) {\n    for x in xs { f(x) }\n}\n";
const lib_acc_keeps = "import std/vec\npub fn each(xs: Vec[int], f: fn(int), keep: Vec[fn(int)]) {\n    keep.push(f)\n    for x in xs { f(x) }\n}\n";

test "warm -O1: a callee that starts keeping its fn param re-boxes the caller's accumulator" {
    try flipAt(".toy-test-places-acc-on", &.{"-O1"}, &.{ .{ .name = "main.toy", .src = acc_main }, .{ .name = "lib.toy", .src = lib_acc_calls } }, 6, &.{.{ .name = "lib.toy", .src = lib_acc_keeps }}, 6);
}

test "warm -O1: a callee that stops keeping its fn param unboxes the caller's accumulator" {
    try flipAt(".toy-test-places-acc-off", &.{"-O1"}, &.{ .{ .name = "main.toy", .src = acc_main }, .{ .name = "lib.toy", .src = lib_acc_keeps } }, 6, &.{.{ .name = "lib.toy", .src = lib_acc_calls }}, 6);
}

// A user generic method, because the flip needs a callee whose body can start or stop
// keeping f; the caller's key must follow its instance's summary.
const bag_main =
    \\import std/vec
    \\import lib
    \\fn main() -> int {
    \\    keep: Vec[fn(int)] = []
    \\    items: Vec[int] = [4, 5, 6]
    \\    bag := lib.mk(items, keep)
    \\    total := 0
    \\    bag.each(fn(v: int) { total = total + v })
    \\    return total
    \\}
    \\
;
const lib_bag_head =
    \\import std/vec
    \\pub struct Bag[T] { items: Vec[T], keep: Vec[fn(T)] }
    \\pub fn mk[T](items: Vec[T], keep: Vec[fn(T)]) -> Bag[T] { Bag { items: items, keep: keep } }
    \\
;
const lib_bag_calls = lib_bag_head ++ "impl Bag[T] {\n    fn each(self, f: fn(T)) {\n        for x in self.items { f(x) }\n    }\n}\n";
const lib_bag_keeps = lib_bag_head ++ "impl Bag[T] {\n    fn each(self, f: fn(T)) {\n        self.keep.push(f)\n        for x in self.items { f(x) }\n    }\n}\n";

test "warm -O1: a generic method instance that starts keeping its fn param rebuilds the caller" {
    try flipAt(".toy-test-places-bag-on", &.{"-O1"}, &.{ .{ .name = "main.toy", .src = bag_main }, .{ .name = "lib.toy", .src = lib_bag_calls } }, 15, &.{.{ .name = "lib.toy", .src = lib_bag_keeps }}, 15);
}

test "warm -O1: a generic method instance that stops keeping its fn param rebuilds the caller" {
    try flipAt(".toy-test-places-bag-off", &.{"-O1"}, &.{ .{ .name = "main.toy", .src = bag_main }, .{ .name = "lib.toy", .src = lib_bag_keeps } }, 15, &.{.{ .name = "lib.toy", .src = lib_bag_calls }}, 15);
}

/// The diagnostics a `toy` run in `cwd_abs` that must fail prints (owned).
fn buildErrIn(gpa: std.mem.Allocator, io: Io, cwd_abs: []const u8, args: []const []const u8) ![]u8 {
    const bin_abs = harness.toy_exe;
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.append(gpa, bin_abs);
    try argv.appendSlice(gpa, args);
    var child = try std.process.spawn(io, .{ .argv = argv.items, .cwd = .{ .path = cwd_abs }, .stdout = .pipe, .stderr = .ignore });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const err = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    errdefer gpa.free(err);
    const term = try child.wait(io);
    if (term == .exited and term.exited == 0) return error.TestExpectedCompileError;
    return err;
}

test "warm: an imported fn that starts keeping its fn param makes a mut-param capture T0052, at either level" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    const root = ".toy-test-places-esc-legal";
    Io.Dir.cwd().deleteTree(io, root) catch {};
    defer Io.Dir.cwd().deleteTree(io, root) catch {};
    const main_src =
        \\import std/vec
        \\import lib
        \\fn add_all(mut n: int, xs: Vec[int], keep: Vec[fn(int)]) {
        \\    lib.each(xs, fn(v: int) { n = n + v }, keep)
        \\}
        \\fn main() -> int {
        \\    xs: Vec[int] = [1, 2, 3]
        \\    keep: Vec[fn(int)] = []
        \\    t := 0
        \\    add_all(t, xs, keep)
        \\    return t
        \\}
        \\
    ;
    const calls = "import std/vec\npub fn each(xs: Vec[int], f: fn(int), keep: Vec[fn(int)]) {\n    for x in xs { f(x) }\n}\n";
    const keeps = "import std/vec\npub fn each(xs: Vec[int], f: fn(int), keep: Vec[fn(int)]) {\n    keep.push(f)\n    for x in xs { f(x) }\n}\n";
    for ([_][]const u8{ "-O0", "-O1" }) |lvl| {
        const w = try makeDir(gpa, io, root, lvl);
        defer gpa.free(w);
        try writeAll(gpa, io, w, &.{ .{ .name = "main.toy", .src = main_src }, .{ .name = "lib.toy", .src = calls } });
        try harness.buildIn(gpa, io, w, &.{ "build", "main.toy", "-o", "a/prog", lvl });
        try std.testing.expectEqual(@as(u8, 6), try harness.runExitIn(gpa, io, w, "a/prog"));
        try writeAll(gpa, io, w, &.{.{ .name = "lib.toy", .src = keeps }});
        const err = try buildErrIn(gpa, io, w, &.{ "build", "main.toy", "-o", "b/prog", lvl });
        defer gpa.free(err);
        try std.testing.expect(std.mem.indexOf(u8, err, "T0052") != null);
        try writeAll(gpa, io, w, &.{.{ .name = "lib.toy", .src = calls }});
        try harness.buildIn(gpa, io, w, &.{ "build", "main.toy", "-o", "b/prog", lvl });
        try std.testing.expectEqual(@as(u8, 6), try harness.runExitIn(gpa, io, w, "b/prog"));
    }
}

test "warm: a callee flip makes a bound method on a mut param T0052 and back" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    const root = ".toy-test-places-esc-bound";
    Io.Dir.cwd().deleteTree(io, root) catch {};
    defer Io.Dir.cwd().deleteTree(io, root) catch {};
    const main_src =
        \\import std/vec
        \\import lib
        \\fn f(mut p: lib.P, keep: Vec[fn() -> int]) -> int {
        \\    p.x = p.x + 1
        \\    lib.run(p.get, keep)
        \\}
        \\fn main() -> int {
        \\    keep: Vec[fn() -> int] = []
        \\    p := lib.mk(6)
        \\    return f(p, keep) + p.x
        \\}
        \\
    ;
    const head = "import std/vec\npub struct P { pub x: int }\nimpl P {\n    fn get(self) -> int { self.x }\n}\npub fn mk(x: int) -> P { P { x: x } }\n";
    const calls = head ++ "pub fn run(f: fn() -> int, keep: Vec[fn() -> int]) -> int { f() }\n";
    const keeps = head ++ "pub fn run(f: fn() -> int, keep: Vec[fn() -> int]) -> int {\n    keep.push(f)\n    f()\n}\n";
    for ([_][]const u8{ "-O0", "-O1" }) |lvl| {
        const w = try makeDir(gpa, io, root, lvl);
        defer gpa.free(w);
        try writeAll(gpa, io, w, &.{ .{ .name = "main.toy", .src = main_src }, .{ .name = "lib.toy", .src = calls } });
        try harness.buildIn(gpa, io, w, &.{ "build", "main.toy", "-o", "a/prog", lvl });
        try std.testing.expectEqual(@as(u8, 14), try harness.runExitIn(gpa, io, w, "a/prog"));
        try writeAll(gpa, io, w, &.{.{ .name = "lib.toy", .src = keeps }});
        const err = try buildErrIn(gpa, io, w, &.{ "build", "main.toy", "-o", "b/prog", lvl });
        defer gpa.free(err);
        try std.testing.expect(std.mem.indexOf(u8, err, "T0052") != null);
        try std.testing.expect(std.mem.indexOf(u8, err, "the 'mut' parameter 'p'") != null);
        try writeAll(gpa, io, w, &.{.{ .name = "lib.toy", .src = calls }});
        try harness.buildIn(gpa, io, w, &.{ "build", "main.toy", "-o", "b/prog", lvl, "--verify" });
        try std.testing.expectEqual(@as(u8, 14), try harness.runExitIn(gpa, io, w, "b/prog"));
    }
}

test "warm: an own-generic method call follows its argument's new type" {
    const main_src =
        \\import lib
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\    fn map[U](self, f: fn(T) -> U) -> Box[U] { Box { v: f(self.v) } }
        \\    fn get(self) -> T { self.v }
        \\}
        \\fn main() -> int {
        \\    b := Box { v: 41 }
        \\    return b.map(lib.f).get().to_string().len()
        \\}
        \\
    ;
    try flip(
        ".toy-test-places-own-generic",
        &.{ .{ .name = "main.toy", .src = main_src }, .{ .name = "lib.toy", .src = "pub fn f(x: int) -> int { x + 1 }\n" } },
        2,
        &.{.{ .name = "lib.toy", .src = "pub fn f(x: int) -> bool { x > 0 }\n" }},
        4,
    );
}
