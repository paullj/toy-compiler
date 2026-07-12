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

test "vec: a scalar `[1,2,3]` literal is built and read via xs[i]" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The non-empty literal infers V=int from the first element and builds a populated
    // Vec[int]; bracket indexing reads each element in bounds. 2 + 5 + 7 = 14.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-lit-scalar",
        \\import std/vec
        \\fn main() -> int {
        \\    xs := [2, 5, 7]
        \\    return xs[0] + xs[1] + xs[2]
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 14), code);
}

test "vec: an aggregate `[P{..},..]` literal is read via xs[i] (struct element)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // V=P inferred from the first element; ga_push copies size_of[P] bytes per element and
    // xs[i] copies the struct out. (10+20) + (30+40) = 100.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-lit-agg",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\fn main() -> int {
        \\    ps := [P{ x: 10, y: 20 }, P{ x: 30, y: 40 }]
        \\    a := ps[0]
        \\    b := ps[1]
        \\    return a.x + a.y + b.x + b.y
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 100), code);
}

test "vec: an aggregate element survives a regrow, read via xs[i] below the initial cap" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // Push 6 structs (past the initial cap of 4) so ga_grow copies the live prefix by
    // size_of[P] bytes; then read index 1 (< the initial cap) — it must survive the copy.
    // ps[1] = {x:1,y:2} -> 1+2 = 3, plus len 6 -> 9.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-agg-regrow",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\fn main() -> int {
        \\    xs := Vec[P].new()
        \\    i := 0
        \\    while i < 6 {
        \\        xs.push(P{ x: i, y: i + 1 })
        \\        i = i + 1
        \\    }
        \\    e := xs[1]
        \\    return xs.len() + e.x + e.y
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 9), code);
}

test "vec: Option[P] unwrap over an aggregate payload (get) yields the struct" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The SPIKE: get(1) -> Option[P], unwrap of an aggregate payload routes through a
    // result slot + copy. p.x + p.y = 3 + 4 = 7.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-opt-struct",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\fn main() -> int {
        \\    ps := [P{ x: 1, y: 2 }, P{ x: 3, y: 4 }]
        \\    p := ps.get(1).unwrap()
        \\    return p.x + p.y
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "vec: Option[Vec[int]] unwrap round-trips a Vec payload" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The unwrap payload is a Vec[int] (a struct aggregate — the 8-byte handle rides the
    // result-slot copy). Build a Vec, wrap it in Option[Vec[int]], unwrap, and read back.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-opt-vec",
        \\import std/vec
        \\fn main() -> int {
        \\    xs := [10, 20, 30]
        \\    o := Option[Vec[int]].some(xs)
        \\    v := o.unwrap()
        \\    return v.len() * 10 + v[2]   # 3*10 + 30 = 60
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 60), code);
}

test "vec: Option[Ref[int]] unwrap rides the scalar/ref payload path" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // A managed-box payload is an 8-byte cell pointer, so unwrap keeps the scalar
    // block-arg join (byte-identical to HEAD). Deref the unwrapped Ref: *r = 42.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-opt-ref",
        \\fn main() -> int {
        \\    r := &42
        \\    o := Option[Ref[int]].some(r)
        \\    return *o.unwrap()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "vec: unwrap_or with an aggregate default returns the default on none" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // unwrap_or's none-branch writes the aggregate default into the result slot via
    // lowerExprInto. An empty Vec's get(0) is none -> the default P{x:5,y:6} -> 11.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-unwrap-or",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\fn main() -> int {
        \\    xs := Vec[P].new()
        \\    p := xs.get(0).unwrap_or(P{ x: 5, y: 6 })
        \\    return p.x + p.y
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 11), code);
}

test "vec: .first() returns the head element" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    const code = try buildAndRun(gpa, io, ".toy-test-vec-first",
        \\import std/vec
        \\fn main() -> int {
        \\    xs := [8, 1, 2]
        \\    a := xs.first().unwrap()
        \\    empty := Vec[int].new()
        \\    b := empty.first().unwrap_or(3)
        \\    return a + b   # 8 + 3 = 11
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 11), code);
}

test "vec: `for x in xs` over a Vec[int] sums each element" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // `for x in xs` desugars to `it := xs.iter()` then a loop over `it.next()` until None,
    // binding `x` to each Some payload. VecIter[int] is a GENERIC conforming receiver of
    // Iterator[int]. 4 + 5 + 6 + 7 = 22.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-forin-int",
        \\import std/vec
        \\fn main() -> int {
        \\    xs := [4, 5, 6, 7]
        \\    total := 0
        \\    for x in xs {
        \\        total = total + x
        \\    }
        \\    return total
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 22), code);
}

test "vec: `for p in ps` over a Vec[P] binds the aggregate element (Some payload is a struct)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The Item is an aggregate P, so `next` yields Option[P] and the loop copies the whole
    // struct payload into the loop var (the aggregate-safe payload copy, not a scalar load).
    // (10+20) + (30+40) + (1+2) = 103.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-forin-agg",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\fn main() -> int {
        \\    ps := [P{ x: 10, y: 20 }, P{ x: 30, y: 40 }, P{ x: 1, y: 2 }]
        \\    total := 0
        \\    for p in ps {
        \\        total = total + p.x + p.y
        \\    }
        \\    return total
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 103), code);
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

test "vec: xs[i].field reads a struct element's field in place (variable index)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // `ps[i].x + ps[i].y` with a VARIABLE index exercises the value-headed bracketed-
    // postfix path (the parser cannot tell `ps[i].x` from a `Vec[int].new()` turbofish;
    // the checker reinterprets a value-resolved head as an index). 3 + 4 = 7.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-index-field",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\fn main() -> int {
        \\    ps := [P{ x: 1, y: 2 }, P{ x: 3, y: 4 }]
        \\    i := 1
        \\    return ps[i].x + ps[i].y
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "vec: a := ps[0]; a.x workaround still reads the element field" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The pre-fix workaround (bind the element to a local, then access) must keep working.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-index-bind",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\fn main() -> int {
        \\    ps := [P{ x: 5, y: 2 }, P{ x: 3, y: 4 }]
        \\    a := ps[0]
        \\    return a.x + a.y
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "vec: xs[i].method() dispatches an inherent method on the element" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // `ps[i].sum()` with a variable index: the element read composes with instance-method
    // dispatch. 20 + 22 = 42.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-index-method",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\impl P { fn sum(self) -> int { self.x + self.y } }
        \\fn main() -> int {
        \\    ps := [P{ x: 20, y: 22 }, P{ x: 1, y: 1 }]
        \\    i := 0
        \\    return ps[i].sum()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "vec: indexing a non-indexable value is a clean compile error" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // `n[0]` on an int must be rejected (not a spurious "unknown type 'n'"); the compile
    // fails cleanly.
    const dir = ".toy-test-vec-index-badtype";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    try std.testing.expectError(error.CompileFailed, compile(gpa, io, dir,
        \\fn main() -> int {
        \\    n := 3
        \\    return n[0]
        \\}
        \\
    , &.{}));
}

test "vec: a float element round-trips through push/get (float as a monomorphization type-arg)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // `float` must be admitted as a generic type-argument (so Vec[float]/GcArray[float]
    // monomorphize), and the element store must carry the value's type.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-float",
        \\import std/vec
        \\fn main() -> int {
        \\    xs: Vec[float] = []
        \\    xs.push(1.5)
        \\    xs.push(2.5)
        \\    return if xs.get(0).unwrap() +. xs.get(1).unwrap() == 4.0 { 42 } else { 0 }
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 42), code);
}
