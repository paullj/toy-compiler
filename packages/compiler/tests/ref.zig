//! End-to-end coverage for the managed heap box `Ref[T]`.
//!
//! `Ref` has no in-process observable surface (the child's heap is opaque), so each case
//! writes a tiny entry program, compiles it with the installed `toy`, runs it, and asserts
//! the EXIT CODE. The determinism case instead compiles one program at `-j1` and `-j8` and
//! asserts the two images are byte-identical — the run-order-independence the
//! reify-out-of-App-index design guarantees.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const harness = @import("harness.zig");

const skipUnlessBackend = harness.skipUnlessBackend;

const compile = harness.compile;

const buildAndRun = harness.buildAndRun;

test "ref: two independent Ref.new(41) boxes compare UNEQUAL (cell identity, not structural)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // Returns 1 iff the two distinct cells are unequal AND a handle equals itself.
    const code = try buildAndRun(gpa, io, ".toy-test-ref-identity",
        \\import std/ref
        \\fn main() -> int {
        \\    r := Ref.new(41)
        \\    if r == r && Ref.new(41) != Ref.new(41) { return 1 }
        \\    return 0
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 1), code);
}

test "ref: Option[Ref[int]] expect-some yields the boxed value; absence takes the default" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    const code = try buildAndRun(gpa, io, ".toy-test-ref-option",
        \\import std/ref
        \\fn pick(present: bool) -> Option[Ref[int]] {
        \\    if present { return Option.some(Ref.new(40)) }
        \\    return Option[Ref[int]].none
        \\}
        \\fn main() -> int {
        \\    a := pick(true)
        \\    base := if a.is_some() { a.expect("no value").get() } else { 0 }
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
        \\import std/ref
        \\fn main() -> int {
        \\    a := Ref.new(41)
        \\    b := Ref.new(true)
        \\    a.set(a.get() + 1)
        \\    if b.get() { return a.get() }
        \\    return 0
        \\}
        \\
    ;

    // The reified Ref[int]/Ref[bool] struct ids ride the (depth, structural-key) order, so
    // codegen is a pure function of source.
    try harness.expectDeterministicBuild(gpa, io, ".toy-test-ref-determinism", src, &.{}, null);
}

test "ref: a Vec[Ref[int]] pushes then reads back the boxed values" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // A box element is an 8-byte handle: `push` stores it and `xs[i]` reads it back as a
    // `Ref[int]` (not a truncated int). 3 + 4 == 7.
    const code = try buildAndRun(gpa, io, ".toy-test-ref-box-element",
        \\import std/ref
        \\import std/vec
        \\fn main() -> int {
        \\    xs: Vec[Ref[int]] = []
        \\    xs.push(Ref.new(3))
        \\    xs.push(Ref.new(4))
        \\    return xs[0].get() + xs[1].get()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}

test "ref: a Vec[Ref[int]] program is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const src =
        \\import std/ref
        \\import std/vec
        \\fn main() -> int {
        \\    xs: Vec[Ref[int]] = []
        \\    xs.push(Ref.new(3))
        \\    xs.push(Ref.new(4))
        \\    return xs[0].get() + xs[1].get()
        \\}
        \\
    ;

    try harness.expectDeterministicBuild(gpa, io, ".toy-test-ref-box-element-determinism", src, &.{}, null);
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
    // byte-identical, the spike's determinism guarantee.
    const src =
        \\import std/ref
        \\struct Node { v: int, next: Ref[int] }
        \\fn main() -> int {
        \\    r := Ref.new(1)
        \\    a := Node{ v: 42, next: r }
        \\    b := Node{ v: 42, next: r }
        \\    return if a == b { a.v } else { 0 }
        \\}
        \\
    ;

    try harness.expectDeterministicBuild(gpa, io, ".toy-test-ref-trace-determinism", src, &.{}, null);
}

test "ref: a set through a copy of a handle is seen through the original" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // `bump` takes the handle BY VALUE, so its `set` goes through a copy; the caller's handle
    // names the same cell and sees both writes.
    const code = try buildAndRun(gpa, io, ".toy-test-ref-copy-alias",
        \\import std/ref
        \\fn bump(r: Ref[int]) { r.set(r.get() + 10) }
        \\fn main() -> int {
        \\    a := Ref.new(1)
        \\    b := a
        \\    b.set(5)
        \\    bump(b)
        \\    return a.get()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 15), code);
}

test "ref: Ref[P] and Ref[str] payloads round-trip through set and get" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    const code = try buildAndRun(gpa, io, ".toy-test-ref-aggregate",
        \\import std/ref
        \\struct P { x: int, y: int }
        \\fn main() -> int {
        \\    p := Ref.new(P { x: 1, y: 2 })
        \\    q := p
        \\    q.set(P { x: 30, y: 4 })
        \\    s := Ref.new("ab")
        \\    s.set("abcdefgh")
        \\    return p.get().x + p.get().y + s.get().len()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "ref: Ref.new inside a generic mk[T] and a user Box.new(5) infer their type params" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    const code = try buildAndRun(gpa, io, ".toy-test-ref-infer",
        \\import std/ref
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\    fn new(v: T) -> Box[T] { Box { v: v } }
        \\}
        \\fn mk[T](v: T) -> Ref[T] { Ref.new(v) }
        \\fn main() -> int {
        \\    b := Box.new(5)
        \\    f := mk(true)
        \\    n := mk(30)
        \\    pinned := Ref[int].new(7)
        \\    if !f.get() { return 1 }
        \\    return b.v + n.get() + pinned.get()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 42), code);
}
