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

/// Compile `entry_src` with extra `args` (e.g. `--no-cache -j8`) into `dir_name/prog` and return
/// the produced image bytes (caller frees). Same `-o prog` basename across calls, so the
/// ad-hoc code-sign IDENTIFIER matches and only the one-byte code-sign nonce may differ —
/// the seam a `-jN` determinism check compares. Skips off the aarch64-macos backend.
fn compileBytes(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, entry_src: []const u8, args: []const []const u8) ![]u8 {
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

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.appendSlice(gpa, &.{ bin_abs, "build", main_path, "-o", out_bin });
    try argv.appendSlice(gpa, args);

    var child = try std.process.spawn(io, .{ .argv = argv.items, .stdout = .pipe, .stderr = .pipe });
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) return error.CompileFailed;

    return Io.Dir.cwd().readFileAlloc(io, out_bin, gpa, .unlimited);
}

/// The load-bearing precise-trace entry: 64 OWNED (heap `concat`-built) string keys inserted
/// into a `Map[str,int]` (which grows `cap` to 128, scattering the 64 dense entries into
/// physical buckets ≥ len), then a forced collection while ~20000 dead heap strings churn the
/// same size classes so a swept key buffer is reused and overwritten. It exits 42 iff every
/// key still maps to its value. The precise `mp_trace` keeps them by walking the DENSE
/// `entries[0..len)` and marking each key buffer — the backings are leaf-marked, so a
/// physical-bucket or truncated walk misses the scattered keys and this flips to a low count.
const map_high_bucket_src =
    \\import std/vec
    \\import std/map
    \\fn main() -> int {
    \\    m := Map[str, int].new()
    \\    n := 0
    \\    loop {
    \\        if n >= 64 { break }
    \\        m.set("k".concat(n.to_string()), n)
    \\        n = n + 1
    \\    }
    \\    i := 0
    \\    loop {
    \\        if i >= 20000 { break }
    \\        dead := "x".concat(i.to_string())
    \\        junk: Vec[int] = []
    \\        junk.push(i)
    \\        i = i + 1
    \\    }
    \\    ok := 0
    \\    j := 0
    \\    loop {
    \\        if j >= 64 { break }
    \\        v := m.get("k".concat(j.to_string())).unwrap_or(-1)
    \\        if v == j { ok = ok + 1 }
    \\        j = j + 1
    \\    }
    \\    if ok == 64 {
    \\        if m.len() == 64 { return 42 }
    \\    }
    \\    return ok
    \\}
    \\
;

test "gc: high-bucket-collision Map[str,int] keys survive a forced collection (precise mp_trace)" {
    // THE load-bearing proof of descriptor-driven container tracing: with the entries backing
    // leaf-marked, the scattered collision keys are reached ONLY through the dense
    // entries[0..len) walk that dispatches each key through the str descriptor's trace unit.
    // A "walk physical buckets[0..len)" mutation sweeps the scattered keys → a low exit code;
    // exit 42 confirms the precise walk is live (output-identity alone cannot show this).
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-map-highbucket", map_high_bucket_src);
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "gc: a descriptor-traced Map program is byte-identical at -j1 and -j8" {
    // The widened stack-map (off,tag) payload + the descriptor/trace-unit bytes are pure
    // functions of source, so the container-tracing collector image is `-jN` byte-identical
    // (only the one-byte code-sign nonce may differ).
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const b1 = compileBytes(gpa, io, ".toy-test-gc-jN-1", map_high_bucket_src, &.{ "--no-cache", "-j1" }) catch |e| return if (e == error.SkipZigTest) e else e;
    defer gpa.free(b1);
    const b8 = try compileBytes(gpa, io, ".toy-test-gc-jN-8", map_high_bucket_src, &.{ "--no-cache", "-j8" });
    defer gpa.free(b8);
    try std.testing.expectEqual(b1.len, b8.len);
    var differing: usize = 0;
    for (b1, b8) |x, y| {
        if (x != y) differing += 1;
    }
    try std.testing.expect(differing <= 1);
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

test "gc: a by-value aggregate SLOT survivor is kept by recursing the precise root map" {
    // The precise map must project a by-value struct's CONTAINED reference, not just bare
    // Ref locals: after make() returns, the backing gc_array's only root is `keep`'s Bag
    // slot, reached only via Bag -> Vec -> gc_array recursion. Exit 8 proves the recursion;
    // an isReference()-only map would skip the slot and sweep the survivor (exit 0).
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-roots",
        \\import std/vec
        \\struct Bag { items: Vec[int], n: int }
        \\fn make() -> Bag {
        \\    v: Vec[int] = []
        \\    v.push(8)
        \\    return Bag{ items: v, n: 1 }
        \\}
        \\fn main() -> int {
        \\    keep: Bag = make()
        \\    i := 0
        \\    loop {
        \\        if i >= 20000 { break }
        \\        junk: Vec[int] = []
        \\        junk.push(i)
        \\        i = i + 1
        \\    }
        \\    it := keep.items
        \\    return it.get(0).unwrap_or(0)
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 8), code);
}

test "gc: a box VALUE held across an alloc-triggered collection is a root (retyped-box gate)" {
    // The box handle from `&8` lives ONLY as an SSA value in main's frame while the second
    // argument (`churn_ref()`) forces a collection. It survives only because a box VALUE is
    // recorded as a managed cell (`.rawptr`) the precise map projects — the retype whose
    // absence UAFs the boxed 8. Exit 17 (= 8 + 9); a swept 8 would read reused memory.
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-boxval",
        \\import std/vec
        \\fn use2(x: Ref[int], y: Ref[int]) -> int { return *x + *y }
        \\fn churn_ref() -> Ref[int] {
        \\    i := 0
        \\    loop {
        \\        if i >= 20000 { break }
        \\        j: Vec[int] = []
        \\        j.push(i)
        \\        i = i + 1
        \\    }
        \\    return &9
        \\}
        \\fn main() -> int { return use2(&8, churn_ref()) }
        \\
    );
    try std.testing.expectEqual(@as(u8, 17), code);
}

test "gc: a struct-embedded heap str survives churn (str leaf projected at its ptr half)" {
    // A `str` field holds a pointer into a heap buffer; the precise map must project that
    // pointer (base+0). Box{s: to_string(700)} survives 20000 churned vectors and still
    // compares equal to "700" (exit 7); without the str leaf the buffer is swept (exit 0).
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-strbox",
        \\import std/vec
        \\struct Box { s: str, n: int }
        \\fn main() -> int {
        \\    b := Box{ s: 700.to_string(), n: 0 }
        \\    i := 0
        \\    loop {
        \\        if i >= 20000 { break }
        \\        junk: Vec[int] = []
        \\        junk.push(i)
        \\        i = i + 1
        \\    }
        \\    if b.s == "700" { return 7 }
        \\    return 0
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "gc: forced collection count is deterministic across identical runs (watermark-driven)" {
    // Collection count is driven by the allocation watermark, not by liveness/reclamation,
    // so two identical programs must report the SAME count regardless of what the precise
    // map retains. Assert count-equality only (never live-bytes / span count).
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const a = try buildAndRun(gpa, io, ".toy-test-gc-detcount-a",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.forced_collect_count() }
        \\
    );
    const b = try buildAndRun(gpa, io, ".toy-test-gc-detcount-b",
        \\import core/mem_selftest
        \\fn main() -> int { return mem_selftest.forced_collect_count() }
        \\
    );
    try std.testing.expectEqual(a, b);
}

test "gc: a str-only aggregate as a Vec element survives churn (trace unit reaches the buffer)" {
    // A struct whose only managed content is a `str` is still trace-requiring: as a container
    // element its buffer is reached ONLY through the descriptor's trace unit dispatched by the
    // dense element walk. If the aggregate earned no trace unit (trace_off == 0) the leaf-marked
    // backing keeps the struct but the str buffers are swept → the read-back fails (exit != 7).
    // Built in a separate frame so a stale stack pointer cannot conservatively over-retain them.
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-vec-strwrap",
        \\import std/vec
        \\struct Wrap { s: str }
        \\fn build() -> Vec[Wrap] {
        \\    v: Vec[Wrap] = []
        \\    v.push(Wrap{ s: 700.to_string() })
        \\    v.push(Wrap{ s: 800.to_string() })
        \\    return v
        \\}
        \\fn main() -> int {
        \\    keep := build()
        \\    i := 0
        \\    loop {
        \\        if i >= 20000 { break }
        \\        dead := "x".concat(i.to_string())
        \\        junk: Vec[int] = []
        \\        junk.push(i)
        \\        i = i + 1
        \\    }
        \\    ok := 0
        \\    for w in keep {
        \\        if w.s == "700" { ok = ok + 3 }
        \\        if w.s == "800" { ok = ok + 4 }
        \\    }
        \\    return ok
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "gc: a multi-entry Map[str,Ref] keeps EVERY value's pointee across a collection (not just entry 0)" {
    // The dense entries walk dispatches the key trace then the value trace at cell+key_size. A
    // value trace routing through gc_mark must not leave a stale key_size for the NEXT entry's
    // value dispatch, else every entry after the first marks the key slot instead of the value
    // → the later values' boxed cells are swept. Sum of all three pointees (10+20+40) proves
    // every entry's value survived, not just insertion-order entry 0.
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-map-refval",
        \\import std/vec
        \\import std/map
        \\fn build() -> Map[str, Ref[int]] {
        \\    m := Map[str, Ref[int]].new()
        \\    m.set("a", &10)
        \\    m.set("b", &20)
        \\    m.set("c", &40)
        \\    return m
        \\}
        \\fn main() -> int {
        \\    m := build()
        \\    i := 0
        \\    loop {
        \\        if i >= 20000 { break }
        \\        dead := "x".concat(i.to_string())
        \\        junk: Vec[int] = []
        \\        junk.push(i)
        \\        i = i + 1
        \\    }
        \\    sum := 0
        \\    for k, v in m {
        \\        sum = sum + *v
        \\    }
        \\    if sum == 70 { return 7 }
        \\    return 0
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}
