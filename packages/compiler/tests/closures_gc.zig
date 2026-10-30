//! End-to-end GC coverage for function values: fn values and closures whose pairs and
//! envs must survive collections, held in locals, containers, other closures' envs, bound
//! method values and specialized calls, at -O0 and with -O1's frame envs. Each case
//! builds a program with the installed `toy` and asserts its exit code.

const std = @import("std");
const Io = std.Io;
const harness = @import("harness.zig");
const heap = @import("heap.zig");
const buildAndRun = heap.buildAndRun;
const buildAndRunWith = heap.buildAndRunWith;

/// Three named fns pushed into a `Vec[fn(int) -> int]`, then a forced collection and a
/// churn of same-size-class allocations before the elements are read back and called.
/// The pairs survive only if the Vec's backing is traced through the fn pair's trace unit.
/// triple(2) + double(3) + 6 when the forced collection was the first and the churn
/// collected again = 18.
const fn_vec_src =
    \\import std/gc
    \\import std/vec
    \\fn double(x: int) -> int { x * 2 }
    \\fn triple(x: int) -> int { x * 3 }
    \\fn square(x: int) -> int { x * x }
    \\fn churn() -> int {
    \\    for i in 0..20000 {
    \\        junk: Vec[int] = []
    \\        junk.push(i)
    \\    }
    \\    return gc.collections()
    \\}
    \\fn main() -> int {
    \\    fs: Vec[fn(int) -> int] = []
    \\    fs.push(double)
    \\    fs.push(triple)
    \\    fs.push(square)
    \\    gc.collect()
    \\    n := gc.collections()
    \\    c := churn()
    \\    if n != 1 || c <= n { return 1 }
    \\    return fs[1](2) + fs[0](3) + 6
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
    \\import std/gc
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
    \\        junk: Vec[int] = []
    \\        junk.push(i)
    \\        i = i + 1
    \\    }
    \\    return gc.collections()
    \\}
    \\fn collect() -> bool {
    \\    gc.collect()
    \\    n := gc.collections()
    \\    c := churn()
    \\    n >= 1 && c > n
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

test "gc: a group of local fns keeps its one shared env, and the cell of its mut capture, across collections" {
    try expectClosureGc(".toy-test-gc-local-fn-group",
        \\fn make(start: int, tag: str) -> Vec[fn(int) -> int] {
        \\    n := start
        \\    fn even(k: int) -> int { if k == 0 { n + tag.len() } else { odd(k - 1) } }
        \\    fn odd(k: int) -> int {
        \\        n = n + 1
        \\        if k == 0 { 0 } else { even(k - 1) }
        \\    }
        \\    fs: Vec[fn(int) -> int] = []
        \\    fs.push(odd)
        \\    fs.push(even)
        \\    fs
        \\}
        \\fn main() -> int {
        \\    fs := make(10, "abcd")
        \\    if !collect() { return 1 }
        \\    a := fs[0](3)
        \\    if !collect() { return 2 }
        \\    # odd(3): n = 12, then even(0) = 16; even(2): n = 13, then even(0) = 17
        \\    return a + fs[1](2) + 9
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
