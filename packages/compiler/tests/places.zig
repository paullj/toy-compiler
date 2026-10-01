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
    try harness.buildIn(gpa, io, w, &.{ "build", "main.toy", "-o", "a/prog" });
    try std.testing.expectEqual(v1_exit, try harness.runExitIn(gpa, io, w, "a/prog"));

    try writeAll(gpa, io, w, v2);
    try harness.buildIn(gpa, io, w, &.{ "build", "main.toy", "-o", "b/prog" });
    try std.testing.expectEqual(v2_exit, try harness.runExitIn(gpa, io, w, "b/prog"));
    try harness.buildIn(gpa, io, w, &.{ "build", "main.toy", "-o", "b/prog", "--verify" });

    try writeAll(gpa, io, c, v1);
    try writeAll(gpa, io, c, v2);
    try harness.buildIn(gpa, io, c, &.{ "build", "main.toy", "-o", "b/prog", "--no-cache" });
    try std.testing.expectEqual(v2_exit, try harness.runExitIn(gpa, io, c, "b/prog"));
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
        \\pub struct P { x: int }
        \\impl P {
        \\    fn show(self) -> str { "pee" }
        \\}
        \\pub fn mk() -> P { P{ x: 1 } }
        \\
    ;
    const v2 =
        \\pub struct P { x: int }
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

test "warm cache: a mode flip with an unchanged body plus a caller edit serves no stale callee (#25 callee half)" {
    const lib_v1 =
        \\pub struct P { x: int }
        \\impl P {
        \\    fn show(self) -> int { self.x }
        \\}
        \\pub fn mk() -> P { P{ x: 1 } }
        \\
    ;
    const lib_v2 =
        \\pub struct P { x: int }
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
