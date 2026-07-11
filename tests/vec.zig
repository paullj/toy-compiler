//! End-to-end coverage for the growable `Vec[T]` over `core/mem`'s `GcArray`.
//!
//! A `Vec` has no in-process observable surface (the child's heap is opaque), so each case
//! writes a tiny entry program, compiles it with the installed `toy`, runs it, and asserts
//! the EXIT CODE. Covered: growth past the initial capacity (forcing at least one grow, the
//! old backing leaking), reference sharing (a push through one binding seen through
//! another), a post-grow read-back, and the two constructor surfaces (`Vec[int].new()` and
//! the annotated empty-list literal) producing identical behavior.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

fn skipUnlessBackend(io: Io) !void {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;
}

/// Compile `entry_src` with the installed `toy` (extra flags via `flags`) into
/// `dir_name/prog`, returning the produced binary's absolute path (owned by `gpa`).
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

fn buildAndRun(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, entry_src: []const u8) !u8 {
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const prog = try compile(gpa, io, dir_name, entry_src, &.{});
    defer gpa.free(prog);
    var run = try std.process.spawn(io, .{ .argv = &.{prog} });
    const term = try run.wait(io);
    return switch (term) {
        .exited => |c| c,
        else => error.ChildCrashed,
    };
}

test "vec: growth past the initial capacity reads back a post-grow element" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // Push 10 elements (i) past the initial cap of 4, forcing >= 1 grow (the old backing
    // leaks). Read back element 7 (a post-grow slot) — it must survive the copy. 7 == 7.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-grow",
        \\import std/vec
        \\fn main() -> int {
        \\    xs := Vec[int].new()
        \\    i := 0
        \\    while i < 10 {
        \\        xs.push(i)
        \\        i = i + 1
        \\    }
        \\    return xs.len() + xs.get(7).unwrap()   # 10 + 7 = 17
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 17), code);
}

test "vec: `:=` shares the same growable (a push through one alias is seen through the other)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // ys := xs copies the 8-byte handle, so both alias one GcArray. A value-copy would
    // leave xs.len() == 1 -> exit 1; sharing makes it 3 -> exit 3.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-share",
        \\import std/vec
        \\fn main() -> int {
        \\    xs := Vec[int].new()
        \\    xs.push(9)
        \\    ys := xs
        \\    ys.push(9)
        \\    ys.push(9)
        \\    return xs.len()   # 3 iff shared
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 3), code);
}

test "vec: `Vec[int].new()` and the empty-list literal build an identical empty Vec" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // Both constructors produce an empty Vec that grows past its initial cap identically;
    // the two lengths + a post-grow read must agree. (7*2 + 6) == 20 for each -> 40.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-ctors",
        \\import std/vec
        \\fn fill(v: Vec[int]) -> int {
        \\    i := 0
        \\    while i < 7 {
        \\        v.push(i)
        \\        i = i + 1
        \\    }
        \\    return v.len() * 2 + v.get(6).unwrap()   # 7*2 + 6 = 20
        \\}
        \\fn main() -> int {
        \\    a := Vec[int].new()
        \\    b: Vec[int] = []
        \\    return fill(a) + fill(b)   # 20 + 20 = 40
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 40), code);
}

test "vec: get returns none for a negative or out-of-range index" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // A negative index must fall to `.none`, not read `elems + i*size` BEFORE the backing
    // (a bare upper-bound guard would let -1 through and SIGSEGV on a large negative). Both
    // the negative and the past-len index yield `.none`; the in-range read is `.some(5)`.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-bounds",
        \\import std/vec
        \\fn main() -> int {
        \\    xs := Vec[int].new()
        \\    xs.push(5)
        \\    n1 := match xs.get(0 - 1) { .some(_) -> 0, .none -> 7 }
        \\    n2 := match xs.get(100) { .some(_) -> 0, .none -> 7 }
        \\    h := match xs.get(0) { .some(v) -> v, .none -> 0 }   # 5
        \\    return n1 + n2 + h   # 7 + 7 + 5 = 19
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 19), code);
}

test "vec: a Vec[bool] round-trips scalar elements through get" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The second scalar element type: push/get of `bool` round-trips through the
    // `load`-returns-`expected` payload typing, and an out-of-range get is `.none`.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-bool",
        \\import std/vec
        \\fn main() -> int {
        \\    xs := Vec[bool].new()
        \\    xs.push(true)
        \\    xs.push(false)
        \\    xs.push(true)
        \\    a := match xs.get(0) { .some(v) -> v, .none -> false }
        \\    b := match xs.get(1) { .some(v) -> v, .none -> true }
        \\    c := match xs.get(2) { .some(v) -> v, .none -> false }
        \\    miss := xs.get(9).is_none()
        \\    return if a && !b && c && miss { 42 } else { 0 }
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "vec: a Vec program is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const src =
        \\import std/vec
        \\fn main() -> int {
        \\    xs := Vec[int].new()
        \\    xs.push(3)
        \\    ys := xs
        \\    ys.push(0)
        \\    return xs.len() * 10 + xs.get(0).unwrap()
        \\}
        \\
    ;
    const dir = ".toy-test-vec-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const p1 = try compile(gpa, io, dir ++ "/j1", src, &.{ "--force", "-j1" });
    defer gpa.free(p1);
    const p8 = try compile(gpa, io, dir ++ "/j8", src, &.{ "--force", "-j8" });
    defer gpa.free(p8);

    const b1 = try Io.Dir.cwd().readFileAlloc(io, p1, gpa, .unlimited);
    defer gpa.free(b1);
    const b8 = try Io.Dir.cwd().readFileAlloc(io, p8, gpa, .unlimited);
    defer gpa.free(b8);

    try std.testing.expectEqual(b1.len, b8.len);
    // The reified Vec[int]/GcArray struct ids ride the (depth, structural-key) order, so
    // codegen is a pure function of source — the two images differ ONLY in the one-byte
    // code-sign nonce (re-randomized every build, never a codegen input).
    var differing: usize = 0;
    for (b1, b8) |x, y| {
        if (x != y) differing += 1;
    }
    try std.testing.expect(differing <= 1);
}
