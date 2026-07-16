//! End-to-end coverage for `Set[T]` over the erased `Map[T, bool]`: `.add` dedup, `.contains`
//! hit/miss, insertion-order key iteration across a grow, single-handle sharing, survival of
//! a forced collection (conservative scan of the Set -> Map -> entries -> str-key buffers),
//! and -jN determinism. Each check surfaces as the child's EXIT CODE (0 on success), the only
//! thing this harness observes. Set is pure library wiring over the Map: no new core surface.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

fn skipUnlessBackend(io: Io) !void {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;
}

fn compile(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, entry_src: []const u8, flags: []const []const u8) ![:0]u8 {
    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, "zig-out/bin/toy", gpa);
    defer gpa.free(bin_abs);

    try Io.Dir.cwd().createDirPath(io, dir_name);
    const main_path = try std.fmt.allocPrint(gpa, "{s}/main.toy", .{dir_name});
    defer gpa.free(main_path);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = main_path, .data = entry_src });

    const out_bin = try std.fmt.allocPrint(gpa, "{s}/prog", .{dir_name});
    defer gpa.free(out_bin);

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.append(gpa, bin_abs);
    try argv.append(gpa, "build");
    try argv.append(gpa, main_path);
    try argv.append(gpa, "-o");
    try argv.append(gpa, out_bin);
    for (flags) |f| try argv.append(gpa, f);

    var child = try std.process.spawn(io, .{ .argv = argv.items, .stdout = .pipe, .stderr = .pipe });
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) return error.CompileFailed;

    return Io.Dir.cwd().realPathFileAlloc(io, out_bin, gpa);
}

fn runExit(gpa: std.mem.Allocator, io: Io, prog: []const u8) !u8 {
    var run = try std.process.spawn(io, .{ .argv = &.{prog} });
    const term = try run.wait(io);
    _ = gpa;
    return switch (term) {
        .exited => |c| c,
        else => error.ChildCrashed,
    };
}

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

    const dir = ".toy-test-set-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // Same `-o` basename in both dirs, so the ad-hoc code-sign IDENTIFIER is identical —
    // only the one-byte code-sign nonce may differ. Set's mono instances + the shared
    // descriptor table are pure functions of source, so codegen is -jN byte-identical.
    const p1 = try compile(gpa, io, dir ++ "/j1", share_src, &.{ "--force", "-j1" });
    defer gpa.free(p1);
    const p8 = try compile(gpa, io, dir ++ "/j8", share_src, &.{ "--force", "-j8" });
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
