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
const harness = @import("harness.zig");

/// Compile `entry_src` (written into a fresh temp dir) with the installed `toy`, run the
/// produced binary, and return its exit code. Skips when `toy` is absent (the corpus
/// gate builds it first) or off aarch64-macos (the only backend target).
fn buildAndRun(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, entry_src: []const u8) !u8 {
    return buildAndRunWith(gpa, io, dir_name, entry_src, &.{});
}

/// `buildAndRun` with extra build `flags`.
fn buildAndRunWith(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, entry_src: []const u8, flags: []const []const u8) !u8 {
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
        const argv = try std.mem.concat(gpa, []const u8, &.{ &.{ bin_abs, "build", main_path, "-o", out_bin }, flags });
        defer gpa.free(argv);
        var child = try std.process.spawn(io, .{ .argv = argv, .stdout = .pipe, .stderr = .pipe });
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
    // functions of source, so the container-tracing collector image is `-jN` byte-identical.
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.expectDeterministicBuild(gpa, io, ".toy-test-gc-jN", map_high_bucket_src, &.{}, null);
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
    // The box handle from `Ref.new(8)` lives ONLY as an SSA value in main's frame while the
    // second argument (`churn_ref()`) forces a collection. It survives only because a box
    // VALUE is recorded as a managed cell (`.rawptr`) the precise map projects — the retype
    // whose absence UAFs the boxed 8. Exit 17 (= 8 + 9); a swept 8 would read reused memory.
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-boxval",
        \\import std/ref
        \\import std/vec
        \\fn use2(x: Ref[int], y: Ref[int]) -> int { return x.get() + y.get() }
        \\fn churn_ref() -> Ref[int] {
        \\    i := 0
        \\    loop {
        \\        if i >= 20000 { break }
        \\        j: Vec[int] = []
        \\        j.push(i)
        \\        i = i + 1
        \\    }
        \\    return Ref.new(9)
        \\}
        \\fn main() -> int { return use2(Ref.new(8), churn_ref()) }
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
        \\import std/ref
        \\import std/vec
        \\import std/map
        \\fn build() -> Map[str, Ref[int]] {
        \\    m := Map[str, Ref[int]].new()
        \\    m.set("a", Ref.new(10))
        \\    m.set("b", Ref.new(20))
        \\    m.set("c", Ref.new(40))
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
        \\        sum = sum + v.get()
        \\    }
        \\    if sum == 70 { return 7 }
        \\    return 0
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

/// Three named fns pushed into a `Vec[fn(int) -> int]`, then a forced collection and a
/// churn of same-size-class allocations before the elements are read back and called.
/// The pairs survive only if the Vec's backing is traced through the fn pair's trace unit.
/// triple(2) + double(3) + 1 forced + 5 churn collections = 18.
const fn_vec_src =
    \\import std/vec
    \\import core/mem_selftest
    \\fn double(x: int) -> int { x * 2 }
    \\fn triple(x: int) -> int { x * 3 }
    \\fn square(x: int) -> int { x * x }
    \\fn main() -> int {
    \\    fs: Vec[fn(int) -> int] = []
    \\    fs.push(double)
    \\    fs.push(triple)
    \\    fs.push(square)
    \\    n := mem_selftest.forced_collect_count()
    \\    c := mem_selftest.churn_collects()
    \\    return fs[1](2) + fs[0](3) + n + c
    \\}
    \\
;

test "gc: a Vec of fn values survives a forced collection" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-fn-vec", fn_vec_src);
    try std.testing.expectEqual(@as(u8, 18), code);
}

test "fn-value programs are byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.expectDeterministicBuild(gpa, io, ".toy-test-fn-jN", fn_vec_src, &.{}, null);
}

/// Closure GC-pressure fixtures. Every victim env is an 8- or 16-byte cell; `churn` makes
/// 20000 closures whose 8-byte envs hold junk, so a victim env swept for want of a root
/// is reused and overwritten before the victim is called. Each churn closure escapes
/// through `keep_one`, or at -O1 its env would sit in `churn`'s frame and overwrite no
/// cell. `collect` forces one collection and then churns.
const closure_gc_prelude =
    \\import std/vec
    \\import std/map
    \\import core/mem_selftest
    \\type Op = fn(int) -> int
    \\fn make_adder(n: int) -> Op {
    \\    fn(x: int) -> int { x + n }
    \\}
    \\fn keep_one(f: fn() -> int) -> fn() -> int { f }
    \\fn churn() -> int {
    \\    i := 0
    \\    loop {
    \\        if i >= 20000 { break }
    \\        j := i + 1000
    \\        _g := keep_one(fn() -> int { j })
    \\        i = i + 1
    \\    }
    \\    return mem_selftest.churn_collects()
    \\}
    \\fn collect() -> bool {
    \\    n := mem_selftest.forced_collect_count()
    \\    c := churn()
    \\    n >= 1 && c >= 1
    \\}
    \\
;

/// Each fixture runs at -O0 and at -O1, where its closures that do not escape keep their
/// envs in frame slots that the collections read through the frames' root maps.
fn expectClosureGc(dir_name: []const u8, main_src: []const u8) !void {
    try expectClosureGcAt(dir_name, main_src, &.{});
    const o1_dir = try std.fmt.allocPrint(std.testing.allocator, "{s}-O1", .{dir_name});
    defer std.testing.allocator.free(o1_dir);
    try expectClosureGcAt(o1_dir, main_src, &.{"-O1"});
}

fn expectClosureGcAt(dir_name: []const u8, main_src: []const u8, flags: []const []const u8) !void {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const src = try std.mem.concat(gpa, u8, &.{ closure_gc_prelude, main_src });
    defer gpa.free(src);
    const code = try buildAndRunWith(gpa, io, dir_name, src, flags);
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "gc: a closure held only in a local keeps its env across a collection (frame root)" {
    try expectClosureGc(".toy-test-gc-closure-local",
        \\fn main() -> int {
        \\    f := make_adder(40)
        \\    if !collect() { return 1 }
        \\    return f(2)
        \\}
        \\
    );
}

test "gc: closures held only in a Vec keep their envs (element trace)" {
    try expectClosureGc(".toy-test-gc-closure-vec",
        \\fn fill() -> Vec[Op] {
        \\    fs: Vec[Op] = []
        \\    for i in 0..5 { fs.push(make_adder(i * 10)) }
        \\    fs
        \\}
        \\fn main() -> int {
        \\    fs := fill()
        \\    if !collect() { return 1 }
        \\    s := 0
        \\    for i in 0..5 { s = s + fs[i](1) }
        \\    return s - 63
        \\}
        \\
    );
}

test "gc: closures held only as Map values keep their envs" {
    try expectClosureGc(".toy-test-gc-closure-map",
        \\fn fill() -> Map[int, Op] {
        \\    m := Map[int, Op].new()
        \\    for i in 0..5 { m.set(i, make_adder(i * 10)) }
        \\    m
        \\}
        \\fn main() -> int {
        \\    m := fill()
        \\    if !collect() { return 1 }
        \\    s := 0
        \\    for k, v in m { s = s + v(k) }
        \\    return s - 68
        \\}
        \\
    );
}

test "gc: a closure in a struct in a Vec keeps its env" {
    try expectClosureGc(".toy-test-gc-closure-struct-vec",
        \\struct H { k: int, f: Op }
        \\fn fill() -> Vec[H] {
        \\    hs: Vec[H] = []
        \\    for i in 0..5 { hs.push(H { k: i, f: make_adder(i * 10) }) }
        \\    hs
        \\}
        \\fn main() -> int {
        \\    hs := fill()
        \\    if !collect() { return 1 }
        \\    s := 0
        \\    for i in 0..5 {
        \\        h := hs[i]
        \\        s = s + h.f(h.k)
        \\    }
        \\    return s - 68
        \\}
        \\
    );
}

test "gc: a closure captured by another closure is reached through the outer env (word scan)" {
    try expectClosureGc(".toy-test-gc-closure-nested",
        \\fn twice(f: Op) -> Op {
        \\    fn(x: int) -> int { f(f(x)) }
        \\}
        \\fn main() -> int {
        \\    g := twice(make_adder(20))
        \\    if !collect() { return 1 }
        \\    return g(2)
        \\}
        \\
    );
}

test "gc: an Option[fn] payload keeps its env" {
    try expectClosureGc(".toy-test-gc-closure-option",
        \\fn pick(n: int) -> Option[Op] {
        \\    if n > 0 { return .some(make_adder(n)) }
        \\    return .none
        \\}
        \\fn main() -> int {
        \\    o := pick(40)
        \\    if !collect() { return 1 }
        \\    return match o {
        \\        .some(f) -> f(2),
        \\        .none -> 3,
        \\    }
        \\}
        \\
    );
}

test "gc: a closure argument survives a collection while a later argument is evaluated" {
    try expectClosureGc(".toy-test-gc-closure-arg",
        \\fn apply(f: Op, x: int) -> int { f(x) }
        \\fn collect_then(x: int) -> int {
        \\    if !collect() { return 0 }
        \\    x
        \\}
        \\fn main() -> int {
        \\    return apply(make_adder(40), collect_then(2))
        \\}
        \\
    );
}

test "gc: a boxed counter survives collections between calls" {
    try expectClosureGc(".toy-test-gc-box-counter",
        \\struct Pt { x: int, y: int }
        \\fn counter() -> fn() -> int {
        \\    n := 0
        \\    fn() {
        \\        n = n + 1
        \\        n
        \\    }
        \\}
        \\fn main() -> int {
        \\    next := counter()
        \\    next()
        \\    if !collect() { return 1 }
        \\    next()
        \\    total := 0
        \\    p := Pt { x: 0, y: 0 }
        \\    add := fn(v: int) {
        \\        total = total + v
        \\        p.y = p.y + 1
        \\    }
        \\    add(10)
        \\    if !collect() { return 2 }
        \\    add(next())
        \\    if !collect() { return 3 }
        \\    return total + next() + p.y + 23
        \\}
        \\
    );
}

test "gc: a local fn's env survives collections, inside its own recursion too" {
    try expectClosureGc(".toy-test-gc-local-fn",
        \\fn make(k: int) -> Op {
        \\    base := k
        \\    fn go(x: int) -> int {
        \\        if x == 5 && !collect() { return 0 - 1000 }
        \\        if x <= 0 { base } else { go(x - 1) + 1 }
        \\    }
        \\    go
        \\}
        \\fn main() -> int {
        \\    f := make(30)
        \\    if !collect() { return 1 }
        \\    hits := 0
        \\    fn tick() { hits = hits + 1 }
        \\    tick()
        \\    if !collect() { return 2 }
        \\    tick()
        \\    # f(10) = 30 + 10
        \\    return f(10) + hits
        \\}
        \\
    );
}

test "gc: a bound method's receiver env survives a collection" {
    try expectClosureGc(".toy-test-gc-bound",
        \\struct Acc { n: int }
        \\impl Acc {
        \\    fn plus(self, x: int) -> int { self.n + x }
        \\    fn bump(mut self) { self.n = self.n + 1 }
        \\}
        \\fn make_bound(n: int) -> Op {
        \\    a := Acc { n: n }
        \\    a.plus
        \\}
        \\fn make_counter() -> fn() -> int {
        \\    c := Acc { n: 0 }
        \\    inc := c.bump
        \\    fn() -> int {
        \\        inc()
        \\        c.n
        \\    }
        \\}
        \\fn main() -> int {
        \\    f := make_bound(32)
        \\    fs: Vec[Op] = []
        \\    for i in 0..3 { fs.push(make_bound(i)) }
        \\    g := make_counter()
        \\    g()
        \\    if !collect() { return 1 }
        \\    # 34 + (1 + 2 + 3) + 2
        \\    return f(2) + fs[0](1) + fs[1](1) + fs[2](1) + g()
        \\}
        \\
    );
}

test "gc: a specialized call's lambda env survives a collection inside the callee" {
    try expectClosureGc(".toy-test-gc-spec",
        \\fn twice_gc[T](f: fn(T) -> T, x: T) -> T {
        \\    y := f(x)
        \\    if !collect() { return x }
        \\    f(y)
        \\}
        \\fn main() -> int {
        \\    n := 20
        \\    s := "ab".concat("c")
        \\    a := twice_gc(fn(x: int) -> int { x + n + s.len() - 3 }, 2)
        \\    if a != 42 { return 1 }
        \\    b := twice_gc(make_adder(21), 0)
        \\    if b != 42 { return 2 }
        \\    return a
        \\}
        \\
    );
}

test "gc: per-iteration boxes survive" {
    try expectClosureGc(".toy-test-gc-box-loop",
        \\fn fill() -> Vec[fn() -> int] {
        \\    fs: Vec[fn() -> int] = []
        \\    for i in 0..3 {
        \\        fs.push(fn() {
        \\            i = i + 1
        \\            i
        \\        })
        \\    }
        \\    xs: Vec[int] = [10, 20]
        \\    for x in xs {
        \\        fs.push(fn() {
        \\            x = x + 1
        \\            x
        \\        })
        \\    }
        \\    m := Map[int, int].new()
        \\    m.set(1, 100)
        \\    for k, v in m {
        \\        fs.push(fn() {
        \\            v = v + k
        \\            v
        \\        })
        \\    }
        \\    fs
        \\}
        \\fn main() -> int {
        \\    fs := fill()
        \\    if !collect() { return 1 }
        \\    s := 0
        \\    for j in 0..fs.len() { s = s + fs[j]() }
        \\    if !collect() { return 2 }
        \\    for j in 0..fs.len() { s = s + fs[j]() }
        \\    # First round 1+2+3 + 11+21 + 101 = 139, second 2+3+4 + 12+22 + 102 = 145.
        \\    return s - 242
        \\}
        \\
    );
}

test "gc: a box whose closures are gone is kept by its owner" {
    try expectClosureGc(".toy-test-gc-box-owner",
        \\fn main() -> int {
        \\    n := 40
        \\    g := fn() { n = n + 1 }
        \\    g()
        \\    g = fn() { }
        \\    g()
        \\    if !collect() { return 1 }
        \\    n = n + 1
        \\    return n
        \\}
        \\
    );
}

/// A `Vec` slot is not written before the churn loop's collections, so it still holds a
/// word an earlier call left in the same stack bytes (`gs`'s element backing). Each
/// collection walks that slot as a container header unless the prologue clears it.
const stale_slot_src =
    \\import std/ref
    \\import std/vec
    \\struct C { n: int, tag: str, xs: Vec[int] }
    \\struct Pt { x: int, y: int }
    \\fn churn_mk(k: int, _j: int) -> C {
    \\    i := 0
    \\    loop {
    \\        if i >= 30000 { break }
    \\        p := Pt { x: i, y: i }
    \\        f := Pt { x: 1, y: 2 }
    \\        _q := f
    \\        _r := Ref.new(p.x)
    \\        i = i + 1
    \\    }
    \\    xs: Vec[int] = [1, 2, 3]
    \\    C { n: k, tag: "ab", xs: xs }
    \\}
    \\fn churn_closures(k: int) -> C {
    \\    i := 0
    \\    loop {
    \\        if i >= 30000 { break }
    \\        p := Pt { x: i, y: i }
    \\        f := fn() -> int { p.x }
    \\        _q := f
    \\        i = i + 1
    \\    }
    \\    xs: Vec[int] = [1, 2, 3]
    \\    C { n: k, tag: "ab", xs: xs }
    \\}
    \\fn main() -> int {
    \\    gs: Vec[int] = []
    \\    c := churn_mk(1, 0)
    \\    gs.push(c.n)
    \\    c2 := churn_mk(2, 0)
    \\    gs.push(c2.n)
    \\    c3 := churn_closures(3)
    \\    gs.push(c3.n)
    \\    c4 := churn_closures(4)
    \\    gs.push(c4.n)
    \\    return gs.len() + c.xs.len() + c4.xs.len()
    \\}
    \\
;

test "gc: a container slot is cleared before a collection can scan it" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const code = try buildAndRun(gpa, io, ".toy-test-gc-stale-slot", stale_slot_src);
    try std.testing.expectEqual(@as(u8, 10), code);
}

test "gc: an enum word that is a container in one variant only is word-scanned" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    // The Vec of variant `b` shares its payload offset with a closure env, a `Ref` and a
    // `str` buffer of variant `a`, the live one.
    const cases = [_]struct { src: []const u8, want: u8 }{
        .{ .src =
        \\import std/vec
        \\import core/mem_selftest
        \\enum E { b(int, int, Vec[int]), a(int, fn() -> int) }
        \\struct Pt { x: int, y: int }
        \\fn mk(k: int) -> fn() -> int {
        \\    p := Pt { x: k, y: k }
        \\    fn() -> int { p.x + p.y }
        \\}
        \\fn main() -> int {
        \\    e := E.a(1, mk(5))
        \\    n := mem_selftest.churn_collects()
        \\    m := mem_selftest.forced_collect_count()
        \\    r := match e { .a(k, f) -> k + f(), .b(_, _, v) -> v.len() }
        \\    if n < 1 || m < 1 { return 1 }
        \\    return r
        \\}
        \\
        , .want = 11 },
        .{ .src =
        \\import std/ref
        \\import std/vec
        \\import core/mem_selftest
        \\enum E { b(int, int, Vec[int]), a(int, int, Ref[int]) }
        \\fn main() -> int {
        \\    e := E.a(1, 2, Ref.new(5))
        \\    n := mem_selftest.churn_collects()
        \\    m := mem_selftest.forced_collect_count()
        \\    r := match e { .a(k, _, rr) -> k + rr.get(), .b(_, _, v) -> v.len() }
        \\    if n < 1 || m < 1 { return 1 }
        \\    return r
        \\}
        \\
        , .want = 6 },
        .{ .src =
        \\import std/vec
        \\import core/mem_selftest
        \\enum E { b(int, Vec[int]), a(int, str) }
        \\fn main() -> int {
        \\    x := 12345
        \\    e := E.a(1, x.to_string())
        \\    n := mem_selftest.churn_collects()
        \\    m := mem_selftest.forced_collect_count()
        \\    r := match e { .a(k, s) -> k + s.len(), .b(_, v) -> v.len() }
        \\    if n < 1 || m < 1 { return 1 }
        \\    return r
        \\}
        \\
        , .want = 6 },
    };
    for (cases, 0..) |cs, i| {
        const dir = try std.fmt.allocPrint(gpa, ".toy-test-gc-enum-overlap-{d}", .{i});
        defer gpa.free(dir);
        const code = try buildAndRun(gpa, io, dir, cs.src);
        try std.testing.expectEqual(cs.want, code);
    }
}

test "gc: at -O1 a lambda whose Vec, str, Ref and closure captures sit in a frame env runs across collections in its callee" {
    const main_src =
        \\import std/ref
        \\fn each_gc(xs: Vec[int], f: fn(int)) {
        \\    for x in xs {
        \\        if !collect() { return }
        \\        f(x)
        \\    }
        \\}
        \\fn main() -> int {
        \\    v: Vec[int] = [10, 20]
        \\    s := "ab".concat("c")
        \\    r := Ref.new(5)
        \\    g := make_adder(1)
        \\    total := 0
        \\    each_gc([1, 2], fn(x: int) { total = total + x + v[1] + s.len() + r.get() + g(0) })
        \\    # (1 + 20 + 3 + 5 + 1) + (2 + 20 + 3 + 5 + 1) = 61
        \\    return total - 19
        \\}
        \\
    ;
    try expectClosureGcAt(".toy-test-gc-frame-env", main_src, &.{"-O1"});
    // `main`'s own slots root the same values as the env's copies, so this run cannot see
    // a missing env root (the `Closure.envStruct` and FrameLayout unit tests pin the env's
    // root typing); it pins that the env it ran was a frame env.
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    const dir = ".toy-test-gc-frame-env-ir";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const src = try std.mem.concat(gpa, u8, &.{ closure_gc_prelude, main_src });
    defer gpa.free(src);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir ++ "/main.toy", .data = src });
    const o1 = try harness.emitIr(gpa, io, dir ++ "/main.toy", &.{"-O1"});
    defer gpa.free(o1);
    const o0 = try harness.emitIr(gpa, io, dir ++ "/main.toy", &.{"-O0"});
    defer gpa.free(o0);
    const body1 = harness.irBody(o1, "main") orelse return error.TestExpectedMain;
    const body0 = harness.irBody(o0, "main") orelse return error.TestExpectedMain;
    try std.testing.expect(std.mem.indexOf(u8, body1, ":env$main$l0") != null);
    // The env cell and `total`'s box are gone.
    try std.testing.expectEqual(std.mem.count(u8, body0, "call @gc_alloc"), std.mem.count(u8, body1, "call @gc_alloc") + 2);
}

test "gc: a capturing lambda passed to a generic fn survives a collection in it" {
    try expectClosureGc(".toy-test-gc-generic-arg",
        \\fn twice_gc[T](f: fn(T) -> T, x: T) -> T {
        \\    y := f(x)
        \\    if !collect() { return x }
        \\    f(y)
        \\}
        \\fn main() -> int {
        \\    n := 20
        \\    s := "ab".concat("c")
        \\    a := twice_gc(fn(x: int) -> int { x + n + s.len() - 3 }, 2)
        \\    if a != 42 { return 1 }
        \\    b := twice_gc(make_adder(21), 0)
        \\    if b != 42 { return 2 }
        \\    return a
        \\}
        \\
    );
}
