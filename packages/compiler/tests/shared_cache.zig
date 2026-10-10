//! Warm-cache soundness across programs that share one `.toy/` cache directory (#35).
//! Two programs whose `main` bodies are the same text get the same codegen key unless the
//! key folds every symbol the body relocates to. The symbols of a module's own methods
//! carry the module's name (`e.It.iter$int` against `f.It.iter$int`), so a key that missed
//! a desugared call served the first program's code to the second, which then named a
//! symbol it does not define ("internal: unresolved symbol after codegen").

const std = @import("std");
const Io = std.Io;
const harness = @import("harness.zig");

/// A user iterator walked by `for x in it` and by `for k, v in ps`: both desugar to calls
/// of `iter` and `next`, which no call node names.
const iter_prog =
    \\import std/iter
    \\import std/map
    \\
    \\struct It[T] { i: int, n: int }
    \\
    \\impl It[T] has iter.Iterator[int] {
    \\    fn next(mut self) -> Option[int] {
    \\        self.i = self.i + 1
    \\        if self.i > self.n { return Option[int].none }
    \\        return Option.some(self.i)
    \\    }
    \\}
    \\
    \\impl It[T] {
    \\    fn iter(self) -> It[T] { self }
    \\}
    \\
    \\struct Pairs[T] { i: int }
    \\
    \\impl Pairs[T] has iter.Iterator[map.Entry[int, int]] {
    \\    fn next(mut self) -> Option[map.Entry[int, int]] {
    \\        self.i = self.i + 1
    \\        if self.i > 2 { return Option[map.Entry[int, int]].none }
    \\        return Option.some(map.Entry[int, int] { key: self.i, val: 10 * self.i })
    \\    }
    \\}
    \\
    \\impl Pairs[T] {
    \\    fn iter(self) -> Pairs[T] { self }
    \\}
    \\
    \\fn main() -> int {
    \\    t := 0
    \\    it := It[int] { i: 0, n: 4 }
    \\    for x in it { t = t + x }
    \\    ps := Pairs[int] { i: 0 }
    \\    for k, v in ps { t = t + k + v }
    \\    return t
    \\}
    \\
;

test "two programs with one for-in main share a cache dir and each calls its own iterator (#35)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    const root = ".toy-test-shared-cache";
    Io.Dir.cwd().deleteTree(io, root) catch {};
    defer Io.Dir.cwd().deleteTree(io, root) catch {};
    try Io.Dir.cwd().createDirPath(io, root ++ "/out");
    const dir = try Io.Dir.cwd().realPathFileAlloc(io, root, gpa);
    defer gpa.free(dir);

    const commented = "# the same program, one line longer\n" ++ iter_prog;
    const progs = [_]struct { name: []const u8, src: []const u8 }{
        .{ .name = "e", .src = iter_prog },
        .{ .name = "f", .src = iter_prog },
        .{ .name = "m", .src = commented },
    };
    for (progs) |p| {
        const path = try std.fmt.allocPrint(gpa, "{s}/{s}.toy", .{ root, p.name });
        defer gpa.free(path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = p.src });
    }
    // 1+2+3+4 from `It`, then (1+10) + (2+20) from `Pairs`.
    const want: u8 = 43;
    for (progs) |p| {
        const src = try std.fmt.allocPrint(gpa, "{s}.toy", .{p.name});
        defer gpa.free(src);
        const out = try std.fmt.allocPrint(gpa, "out/{s}", .{p.name});
        defer gpa.free(out);
        try harness.buildIn(gpa, io, dir, &.{ "build", src, "-o", out });
        try std.testing.expectEqual(want, try harness.runExitIn(gpa, io, dir, out));
    }
    // A warm `--verify` build re-lowers every unit and compares it with the cache entry
    // its key finds, so a key two programs share for different code fails it.
    for (progs) |p| {
        const src = try std.fmt.allocPrint(gpa, "{s}.toy", .{p.name});
        defer gpa.free(src);
        const out = try std.fmt.allocPrint(gpa, "out/{s}-verify", .{p.name});
        defer gpa.free(out);
        try harness.buildIn(gpa, io, dir, &.{ "build", src, "-o", out, "--verify" });
    }
}
