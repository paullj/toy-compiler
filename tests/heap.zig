//! End-to-end coverage for the leaking span bump allocator.
//!
//! The allocator has no user-facing surface, so it is exercised through a bundled,
//! allowlisted `core/mem_selftest` module whose `pub` self-checks prove each invariant
//! in toy and surface the result as the program's EXIT CODE — the only thing this
//! harness can observe of the compiled child (it cannot inspect the child's heap
//! in-process). Each case writes a tiny entry program that `import core/mem_selftest`,
//! compiles it with the installed `toy` binary, runs the result, and asserts the exit.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

/// Compile `entry_src` (written into a fresh temp dir) with the installed `toy`, run the
/// produced binary, and return its exit code. Skips when `toy` is absent (the corpus
/// gate builds it first) or off aarch64-macos (the only backend target).
fn buildAndRun(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, entry_src: []const u8) !u8 {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;
    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, "zig-out/bin/toy", gpa);
    defer gpa.free(bin_abs);

    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const main_path = try std.fmt.allocPrint(gpa, "{s}/main.toy", .{dir_name});
    defer gpa.free(main_path);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = main_path, .data = entry_src });

    const out_bin = try std.fmt.allocPrint(gpa, "{s}/prog", .{dir_name});
    defer gpa.free(out_bin);
    {
        var child = try std.process.spawn(io, .{ .argv = &.{ bin_abs, "build", main_path, "-o", out_bin }, .stdout = .pipe, .stderr = .pipe });
        const term = try child.wait(io);
        if (term != .exited or term.exited != 0) return error.CompileFailed;
    }

    const prog_abs = try Io.Dir.cwd().realPathFileAlloc(io, out_bin, gpa);
    defer gpa.free(prog_abs);
    var run = try std.process.spawn(io, .{ .argv = &.{prog_abs} });
    const term = try run.wait(io);
    return switch (term) {
        .exited => |c| c,
        else => error.ChildCrashed,
    };
}

test "heap: store + read-back through a rawptr cell returns 42" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = buildAndRun(gpa, io, ".toy-test-heap-selftest",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.selftest() }
        \\
    ) catch |e| if (e == error.SkipZigTest) return e else return e;
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "heap: two cells are distinct, non-overlapping, and zeroed" {
    // The assertion that actually proves the allocator: a no-op readback would also
    // return 42 in the first test, but only real distinct+zeroed storage passes this.
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-heap-distinct",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.check_distinct_zeroed() }
        \\
    );
    try std.testing.expectEqual(@as(u8, 1), code);
}

test "heap: repeated span exhaustion keeps mmapping fresh spans (arena count >= 3)" {
    // Allocating past two full spans forces two refills. This catches a refill that
    // fails to persist the new span's limit: such a bug bumps unbounded off the second
    // span into unmapped memory (SIGSEGV on the store) and never mmaps a third.
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-heap-growth",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.check_span_growth() }
        \\
    );
    try std.testing.expect(code >= 3);
}
