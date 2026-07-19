//! End-to-end coverage for the managed heap box `Ref[T]`.
//!
//! `Ref` has no in-process observable surface (the child's heap is opaque), so each case
//! writes a tiny entry program, compiles it with the installed `toy`, runs it, and asserts
//! the EXIT CODE. The determinism case instead compiles one program at `-j1` and `-j8` and
//! asserts the two images are byte-identical bar the one-byte code-sign nonce — the
//! run-order-independence the reify-out-of-App-index design guarantees.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

fn skipUnlessBackend(io: Io) !void {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;
}

/// Compile `entry_src` with the installed `toy` (any extra flags via `flags`) into
/// `dir_name/prog` and return the produced binary's absolute path (owned by `gpa`). The
/// caller owns the temp dir cleanup token it passes in.
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

test "ref: two independent &41 boxes compare UNEQUAL (cell identity, not structural)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // Returns 1 iff the two distinct cells are unequal AND a handle equals itself.
    const code = try buildAndRun(gpa, io, ".toy-test-ref-identity",
        \\fn main() -> int {
        \\    r := &41
        \\    if r == r && &41 != &41 { return 1 }
        \\    return 0
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 1), code);
}

test "ref: Option[Ref[int]] unwrap-some yields the boxed value; absence takes the default" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    const code = try buildAndRun(gpa, io, ".toy-test-ref-option",
        \\fn pick(present: bool) -> Option[Ref[int]] {
        \\    if present { return Option.some(&40) }
        \\    return Option[Ref[int]].none
        \\}
        \\fn main() -> int {
        \\    a := pick(true)
        \\    base := if a.is_some() { *a.unwrap() } else { 0 }
        \\    b := pick(false)
        \\    extra := if b.is_some() { 0 } else { 2 }
        \\    return base + extra
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "ref: a program building Ref[int] AND Ref[bool] is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const src =
        \\fn main() -> int {
        \\    a := &41
        \\    b := &true
        \\    *a = *a + 1
        \\    if *b { return *a }
        \\    return 0
        \\}
        \\
    ;
    const dir = ".toy-test-ref-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const p1 = try compile(gpa, io, dir ++ "/j1", src, &.{ "--no-cache", "-j1" });
    defer gpa.free(p1);
    const p8 = try compile(gpa, io, dir ++ "/j8", src, &.{ "--no-cache", "-j8" });
    defer gpa.free(p8);

    const b1 = try Io.Dir.cwd().readFileAlloc(io, p1, gpa, .unlimited);
    defer gpa.free(b1);
    const b8 = try Io.Dir.cwd().readFileAlloc(io, p8, gpa, .unlimited);
    defer gpa.free(b8);

    try std.testing.expectEqual(b1.len, b8.len);
    // The reified Ref[int]/Ref[bool] struct ids ride the (depth, structural-key) order, so
    // codegen is a pure function of source — the two images differ ONLY in the one-byte
    // code-sign nonce (which is re-randomized every build, never a codegen input).
    var differing: usize = 0;
    for (b1, b8) |x, y| {
        if (x != y) differing += 1;
    }
    try std.testing.expect(differing <= 1);
}

test "ref: a growable holding Ref[int] elements pushes then reads-back the boxed values" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // A box element is an 8-byte handle: `ga_push` stores it and `ga_get` reads it back as a
    // `Ref[int]` (not a truncated int). `&x` boxes the value; the push round-trips through the
    // same element slot. 3 + 4 == 7.
    const code = try buildAndRun(gpa, io, ".toy-test-ref-box-element",
        \\import core/mem
        \\fn main() -> int {
        \\    a := mem.ga_new[Ref[int]]()
        \\    mem.ga_push[Ref[int]](a, &3)
        \\    mem.ga_push[Ref[int]](a, &4)
        \\    return *mem.ga_get[Ref[int]](a, 0) + *mem.ga_get[Ref[int]](a, 1)
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "ref: a growable of Ref[int] elements is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const src =
        \\import core/mem
        \\fn main() -> int {
        \\    a := mem.ga_new[Ref[int]]()
        \\    mem.ga_push[Ref[int]](a, &3)
        \\    mem.ga_push[Ref[int]](a, &4)
        \\    return *mem.ga_get[Ref[int]](a, 0) + *mem.ga_get[Ref[int]](a, 1)
        \\}
        \\
    ;
    const dir = ".toy-test-ref-box-element-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

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

test "ref: a derivable + Ref-containing type emits a deterministic trace unit (-j1 == -j8)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    // `Node` derives `Eq` (used via `==`) AND holds a `Ref` field, so the one descriptor
    // producer co-emits a `trace` unit that `bl`s `gc_mark`. Both the trace unit's synthetic
    // name (minted off the reified struct id, which rides the (depth, structural-key) order)
    // and `gc_mark`'s appended body are a pure function of source — so the two images are
    // byte-identical bar the one-byte code-sign nonce, the spike's determinism guarantee.
    const src =
        \\struct Node { v: int, next: Ref[int] }
        \\fn main() -> int {
        \\    r := &1
        \\    a := Node{ v: 42, next: r }
        \\    b := Node{ v: 42, next: r }
        \\    return if a == b { a.v } else { 0 }
        \\}
        \\
    ;
    const dir = ".toy-test-ref-trace-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

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
