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
const harness = @import("harness.zig");

const skipUnlessBackend = harness.skipUnlessBackend;

const compile = harness.compile;

const buildAndRun = harness.buildAndRun;

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
    // block-arg join (byte-identical to HEAD). Read the unwrapped Ref: 42.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-opt-ref",
        \\import std/ref
        \\fn main() -> int {
        \\    r := Ref.new(42)
        \\    o := Option[Ref[int]].some(r)
        \\    return o.unwrap().get()
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

    // The reified Vec[int]/GcArray struct ids ride the (depth, structural-key) order, so
    // codegen is a pure function of source.
    try harness.expectDeterministicBuild(gpa, io, ".toy-test-vec-determinism", src, &.{}, null);
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

test "vec: b.items[i].field indexes a Vec-typed struct FIELD and reads the element field" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The field-headed value index: `b.items` is a `Vec[P]` struct FIELD, not a local. The
    // bracket head is a field_access, so the value-index predicate must walk to the LEFTMOST
    // identifier (`b` -> .local). Formerly misparsed as a qualified type ("unknown module
    // 'b'"). 3 + 4 = 7.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-field-index",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\struct Bag { items: Vec[P] }
        \\fn main() -> int {
        \\    b := Bag { items: [P{ x: 1, y: 2 }, P{ x: 3, y: 4 }] }
        \\    i := 1
        \\    return b.items[i].x + b.items[i].y
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "vec: b.items[i].method() dispatches an inherent method off a Vec-typed field element" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    const code = try buildAndRun(gpa, io, ".toy-test-vec-field-method",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\impl P { fn sum(self) -> int { self.x + self.y } }
        \\struct Bag { items: Vec[P] }
        \\fn main() -> int {
        \\    b := Bag { items: [P{ x: 20, y: 22 }, P{ x: 1, y: 1 }] }
        \\    i := 0
        \\    return b.items[i].sum()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "vec: v := b.items[i]; v.x workaround still reads a Vec-typed field element" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The pre-fix workaround for the field-headed case must keep working.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-field-bind",
        \\import std/vec
        \\struct P { x: int, y: int }
        \\struct Bag { items: Vec[P] }
        \\fn main() -> int {
        \\    b := Bag { items: [P{ x: 5, y: 2 }, P{ x: 3, y: 4 }] }
        \\    v := b.items[1]
        \\    return v.x + v.y
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "vec: indexing a non-indexable struct FIELD is a clean type error (not 'unknown module')" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // `b.n[i]` where `n: int`: the leftmost `b` is .local so this is treated as an index,
    // and indexing an int is rejected as a type error — NOT misrouted to a qualified-type
    // "unknown module 'b'" path, and NO crash. The compile fails cleanly.
    const dir = ".toy-test-vec-field-badtype";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    try std.testing.expectError(error.CompileFailed, compile(gpa, io, dir,
        \\struct Bag { n: int }
        \\fn main() -> int {
        \\    b := Bag { n: 3 }
        \\    i := 0
        \\    return b.n[i]
        \\}
        \\
    , &.{}));
}

test "vec: assigning to a field of an index element writes the element back (never a silent write-drop)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // `b.items[i].x = 99` reads its element into a temp copy; the store must reach the
    // Vec through the write-back, not stop at the copy.
    const dir = ".toy-test-vec-index-assign";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const prog = try compile(gpa, io, dir,
        \\import std/vec
        \\struct P { x: int, y: int }
        \\struct Bag { items: Vec[P] }
        \\fn main() -> int {
        \\    b := Bag { items: [P{ x: 1, y: 2 }, P{ x: 3, y: 4 }] }
        \\    i := 1
        \\    b.items[i].x = 99
        \\    return b.items[i].x + b.items[i].y + b.items[0].x
        \\}
        \\
    , &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 104), try harness.runExit(gpa, io, prog));
}

test "vec: a managed-box element passed to a mut param or a mut self receiver is written back" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // A `Vec[int]` element is a one-word handle: its copy lives in a slot the callee writes
    // through, and the write-back stores the new handle into the outer Vec.
    const dir = ".toy-test-vec-elem-mut-box";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const prog = try compile(gpa, io, dir,
        \\import std/vec
        \\struct W { v: Vec[int] }
        \\impl W {
        \\    fn swap_in(mut self) { self.v = [40, 2] }
        \\}
        \\fn reset(mut v: Vec[int]) { v = [7, 8, 9] }
        \\fn main() -> int {
        \\    vs: Vec[Vec[int]] = [[1], [2]]
        \\    reset(vs[1])
        \\    ws: Vec[W] = [W { v: [] }]
        \\    ws[0].swap_in()
        \\    return vs[1][2] + vs[1].len() + vs[0][0] + ws[0].v[0] + ws[0].v[1]
        \\}
        \\
    , &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 55), try harness.runExit(gpa, io, prog));
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

test "vec: Vec[Vec[int]].new() nested container round-trips" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The outer element type is itself a generic instance (`Vec[int]`), so the outer
    // `new()` receiver `Vec[Vec[int]]` must resolve through the reified-arg -> App-form
    // inverse. A local for the unwrap avoids method-chaining on the inner handle.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-nested-vec",
        \\import std/vec
        \\fn main() -> int {
        \\    outer := Vec[Vec[int]].new()
        \\    inner := Vec[int].new()
        \\    inner.push(5)
        \\    outer.push(inner)
        \\    got: Vec[int] = outer.get(0).unwrap()
        \\    return got.get(0).unwrap()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 5), code);
}

test "vec: Vec[Ref[int]].new() round-trips managed-box elements" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The element is a managed box (`Ref[int]`, an `isRefPayload` type). `Ref.new(x)` boxes
    // the value; `get(i).unwrap().get()` reads the boxed cell back.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-ref",
        \\import std/ref
        \\import std/vec
        \\fn main() -> int {
        \\    xs := Vec[Ref[int]].new()
        \\    xs.push(Ref.new(3))
        \\    xs.push(Ref.new(4))
        \\    return xs.get(0).unwrap().get() + xs.get(1).unwrap().get()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "vec: Vec[Option[int]].new() constructs + round-trips an enum element" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The element is a generic enum instance (`Option[int]`), reaching the same
    // reified-arg -> App-form inverse via the `enumT` id-space.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-opt-elem",
        \\import std/vec
        \\fn main() -> int {
        \\    xs := Vec[Option[int]].new()
        \\    xs.push(Option.some(9))
        \\    return xs.get(0).unwrap().unwrap()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 9), code);
}

test "vec: a nested-generic-container program is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const src =
        \\import std/ref
        \\import std/vec
        \\fn main() -> int {
        \\    xs := Vec[Ref[int]].new()
        \\    xs.push(Ref.new(3))
        \\    xs.push(Ref.new(4))
        \\    return xs.get(0).unwrap().get() + xs.get(1).unwrap().get()
        \\}
        \\
    ;

    try harness.expectDeterministicBuild(gpa, io, ".toy-test-vec-nested-determinism", src, &.{}, null);
}

test "vec: a .field off a call/method result materializes the aggregate rvalue" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // A `.field` off an expression with no named place — `get(i).unwrap().field`, a
    // nested `.unwrap().i.x`, a direct `mk().x` — must materialize the aggregate rvalue
    // into a temp once and read the field from the copy. Regression: the lowerer had no
    // arm for a call base, so it read a stale register (garbage) or crashed codegen when
    // the field fed arithmetic. 3 + 5 + 3 + 5 == 16.
    const code = try buildAndRun(gpa, io, ".toy-test-vec-field-thru-call",
        \\import std/vec
        \\struct I { x: int }
        \\struct P { x: int, i: I }
        \\fn mk() -> P { return P{ x: 3, i: I{ x: 5 } } }
        \\fn main() -> int {
        \\    ps := Vec[P].new()
        \\    ps.push(mk())
        \\    a := ps.get(0).unwrap().x
        \\    b := ps.get(0).unwrap().i.x
        \\    c := mk().x
        \\    return a + b + c + mk().i.x
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 16), code);
}

test "vec: a program's own Vec method wins over std's of the same name, whatever the import order" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The library's `each` walks backwards, so the first element it hands over is 3.
    const lib =
        \\import std/vec
        \\impl Vec[T] {
        \\    fn each(self, f: fn(T)) {
        \\        i := self.len() - 1
        \\        while i >= 0 {
        \\            f(self.get(i).unwrap())
        \\            i = i - 1
        \\        }
        \\    }
        \\}
        \\pub fn zero() -> int { 0 }
        \\
    ;
    const body =
        \\fn main() -> int {
        \\    xs: Vec[int] = [1, 2, 3]
        \\    first := 0
        \\    xs.each(fn(v: int) { if first == 0 { first = v } })
        \\    return first + lib.zero()
        \\}
        \\
    ;
    for ([_][]const u8{ "import std/vec\nimport lib\n", "import lib\nimport std/vec\n" }, 0..) |imports, k| {
        const dir = try std.fmt.allocPrint(gpa, ".toy-test-vec-own-method-{d}", .{k});
        defer gpa.free(dir);
        Io.Dir.cwd().deleteTree(io, dir) catch {};
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        try Io.Dir.cwd().createDirPath(io, dir);
        const lib_path = try std.fmt.allocPrint(gpa, "{s}/lib.toy", .{dir});
        defer gpa.free(lib_path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = lib_path, .data = lib });
        const src = try std.mem.concat(gpa, u8, &.{ imports, body });
        defer gpa.free(src);
        const prog = try harness.compile(gpa, io, dir, src, &.{"--no-cache"});
        defer gpa.free(prog);
        try std.testing.expectEqual(@as(u8, 3), try harness.runExit(gpa, io, prog));
    }
}

test "vec: a program's own Vec and Option methods never change what std's methods do" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // std's `split`, `map` and `join` call `push`, `len`, `get` and `unwrap_or`, `sort`
    // calls `sort_by`, and a `for` walks with `get`: each keeps std's own method or the
    // builtin. The program's own calls reach its methods: `len` is 99, `get` finds nothing
    // and `push` drops its element (the `fold` count, which reads std's length, stays 3).
    try harness.expectStdoutAtBothLevels(gpa, io, ".toy-test-vec-shadow-std",
        \\import std/vec
        \\import std/str
        \\import std/io
        \\impl Vec[T] {
        \\    fn len(self) -> int { 99 }
        \\    fn push(self, _x: T) {}
        \\    fn get(self, _i: int) -> Option[T] { Option[T].none }
        \\    fn sort_by(self, _less: fn(T, T) -> bool) {}
        \\}
        \\impl Option[T] {
        \\    fn unwrap_or(self, d: T) -> T { d }
        \\}
        \\fn main() -> int {
        \\    parts := "c a b".split(" ")
        \\    shout := parts.map(fn(s) { s.concat("!") })
        \\    parts.sort()
        \\    io.println(", ".join(parts))
        \\    io.println("".join(shout))
        \\    seen := ""
        \\    for p in parts { seen = seen.concat(p) }
        \\    io.println(seen)
        \\    parts.push("z")
        \\    io.println(parts.fold(0, fn(n, _s) { n + 1 }))
        \\    io.println(parts.len())
        \\    io.println(parts.get(0).is_none())
        \\    io.println(Option.some(5).unwrap_or(6))
        \\    return 0
        \\}
        \\
    , "a, b, c\nc!a!b!\nabc\n3\n99\ntrue\n6\n");
}

test "vec: Vec closure methods keep their results across forced collections" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // Every callback forces a collection while the method holds fresh strs only in its own
    // locals and buffers (the map's output, the sort's scratch array), so a missed root
    // shows up as a wrong str. The sort by length is stable: the 3-byte strs keep their
    // insertion order. 10 * 3 + 30 * 4 = 150.
    const codes = try harness.exitAtBothLevels(gpa, io, ".toy-test-vec-closure-gc",
        \\import std/vec
        \\import core/mem_selftest
        \\fn main() -> int {
        \\    xs: Vec[str] = []
        \\    i := 0
        \\    while i < 40 {
        \\        xs.push("k".concat(((i * 17) % 40).to_string()))
        \\        i = i + 1
        \\    }
        \\    ys := xs.map(fn(s) {
        \\        _c := mem_selftest.forced_collect_count()
        \\        s.concat("!")
        \\    })
        \\    i = 0
        \\    while i < 40 {
        \\        if ys.get(i).unwrap() != xs.get(i).unwrap().concat("!") { return 1 }
        \\        i = i + 1
        \\    }
        \\    ys.sort_by(fn(a, b) {
        \\        _c := mem_selftest.forced_collect_count()
        \\        a.len() < b.len()
        \\    })
        \\    short := ys.filter(fn(s) {
        \\        _c := mem_selftest.forced_collect_count()
        \\        s.len() == 3
        \\    })
        \\    joined := short.fold("", fn(acc, s) {
        \\        _c := mem_selftest.forced_collect_count()
        \\        acc.concat(s)
        \\    })
        \\    if joined != "k0!k5!k4!k9!k3!k8!k2!k7!k1!k6!" { return 2 }
        \\    if ys.get(10).unwrap() != "k17!" { return 3 }
        \\    return ys.fold(0, fn(acc, s) { acc + s.len() })
        \\}
        \\
    );
    try std.testing.expectEqual([2]u8{ 150, 150 }, codes);
}

test "vec: sort_by is stable and sort agrees on 1000 elements" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // Keys 0..9 from an LCG repeat about 100 times each, so stability is tested on long
    // runs of equal keys. `R`'s derived Ord compares `k` then `seq`, so `sort()` must give
    // exactly the order a stable sort by `k` gives; the ints are checked against an
    // insertion sort that shares no code with the merge sort.
    const codes = try harness.exitAtBothLevels(gpa, io, ".toy-test-vec-sort-stable",
        \\import std/vec
        \\struct R { k: int, seq: int }
        \\fn insertion_sorted(xs: Vec[int]) -> Vec[int] {
        \\    out: Vec[int] = []
        \\    for x in xs {
        \\        next: Vec[int] = []
        \\        placed := false
        \\        for y in out {
        \\            if !placed && x < y {
        \\                next.push(x)
        \\                placed = true
        \\            }
        \\            next.push(y)
        \\        }
        \\        if !placed { next.push(x) }
        \\        out = next
        \\    }
        \\    out
        \\}
        \\fn main() -> int {
        \\    rs: Vec[R] = []
        \\    ks: Vec[int] = []
        \\    x := 12345
        \\    i := 0
        \\    while i < 1000 {
        \\        x = (x * 1103515245 + 12345) % 2147483648
        \\        k := (x / 65536) % 10
        \\        rs.push(R { k: k, seq: i })
        \\        ks.push(k)
        \\        i = i + 1
        \\    }
        \\    by_ord := rs.filter(fn(_r) { true })
        \\    rs.sort_by(fn(a, b) { a.k < b.k })
        \\    i = 1
        \\    while i < 1000 {
        \\        a := rs.get(i - 1).unwrap()
        \\        b := rs.get(i).unwrap()
        \\        if a.k > b.k { return 1 }
        \\        if a.k == b.k && a.seq >= b.seq { return 2 }
        \\        i = i + 1
        \\    }
        \\    by_ord.sort()
        \\    want := insertion_sorted(ks)
        \\    ks.sort()
        \\    i = 0
        \\    while i < 1000 {
        \\        if ks.get(i).unwrap() != want.get(i).unwrap() { return 3 }
        \\        if by_ord.get(i).unwrap().seq != rs.get(i).unwrap().seq { return 4 }
        \\        i = i + 1
        \\    }
        \\    if rs.get(0).unwrap().k != 0 || rs.get(999).unwrap().k != 9 { return 5 }
        \\    return 42
        \\}
        \\
    );
    try std.testing.expectEqual([2]u8{ 42, 42 }, codes);
}

test "vec: Vec closure methods are byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const src =
        \\import std/vec
        \\struct P { k: int, s: str }
        \\fn main() -> int {
        \\    xs: Vec[int] = [5, 3, 8, 1]
        \\    total := 0
        \\    xs.each(fn(v) { total = total + v })
        \\    words := xs.map(fn(x) { x.to_string() })
        \\    big := xs.filter(fn(x) { x > 2 })
        \\    sum := big.fold(0, fn(a, x) { a + x })
        \\    found := xs.find(fn(x) { x > 6 }).unwrap_or(0)
        \\    flags := if xs.any(fn(x) { x == 1 }) && xs.all(fn(x) { x > 0 }) { 1 } else { 0 }
        \\    ps: Vec[P] = [P { k: 2, s: "a" }, P { k: 1, s: "b" }]
        \\    ps.sort_by(fn(a, b) { a.k < b.k })
        \\    xs.sort()
        \\    digits := words.fold("", fn(acc, w) { acc.concat(w) })
        \\    return total + sum + found + flags + xs.get(0).unwrap() + ps.len() + digits.len()
        \\}
        \\
    ;

    try harness.expectDeterministicBuild(gpa, io, ".toy-test-vec-closure-determinism", src, &.{}, 49);
}
