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

test "gc: an explicit collection bumps the collection count to 1" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-forced",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.forced_collect_count() }
        \\
    );
    try std.testing.expectEqual(@as(u8, 1), code);
}

test "gc: churning unreachable arrays forces at least one collection (falsifies never-collect)" {
    // The decisive proof that a collection ACTUALLY runs — a bump-only allocator would
    // leave gc_stats().collections at 0 here even though collect.toy still exits 15.
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-churn",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.churn_collects() }
        \\
    );
    try std.testing.expect(code >= 1);
}

test "gc: a struct-embedded growable survives a forced collection (conservative whole-frame scan)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-aggregate",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.aggregate_survivor() }
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "gc: a self-referential cycle is traced without looping and swept without crashing" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-cycle",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.ref_cycle_reclaimed() }
        \\
    );
    try std.testing.expectEqual(@as(u8, 1), code);
}

test "gc: a rooted Vec survives churning ~20000 unreachable Vecs (collect.toy spike gate)" {
    // The spike gate: exit 15 (= keep.get(0)==7 + 8) proves the collector does NOT
    // over-collect the survivor rooted in main's frame while reclaiming the churn.
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-collect",
        \\import std/vec
        \\fn main() -> int {
        \\    keep: Vec[int] = []
        \\    keep.push(7)
        \\    i := 0
        \\    loop {
        \\        if i >= 20000 { break }
        \\        junk: Vec[int] = []
        \\        junk.push(i)
        \\        i = i + 1
        \\    }
        \\    return keep.get(0).unwrap_or(0) + 8
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 15), code);
}

test "gc: a >4096-wide object forces the mark stack to realloc-grow; all children survive" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-markstack-grow",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.mark_stack_grow() }
        \\
    );
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "gc: a large object (> 8 KiB direct mmap) survives a forced collection" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-large-survive",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.large_object_survivor() }
        \\
    );
    try std.testing.expectEqual(@as(u8, 9), code);
}

test "gc: churning dropped large objects is bounded — dead spans munmapped, rooted one kept" {
    // Guards the large-object sweep: each dropped > 8 KiB object is a direct mmap the
    // collector must munmap. Without watermark-triggered collection on the large path plus
    // the unlink+munmap sweep, this churn leaks unbounded; the sweep must also never free
    // the rooted large object, so the survivor still reads back (exit 9).
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-large-churn",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.large_churn_survivor() }
        \\
    );
    try std.testing.expectEqual(@as(u8, 9), code);
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
