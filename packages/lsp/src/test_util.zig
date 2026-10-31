//! Cursor positions for LSP tests, computed from the fixture text so no test hand-counts
//! a column. ASCII fixtures, so a byte offset equals a character offset.

const std = @import("std");
const protocol = @import("protocol.zig");

pub fn posAt(src: []const u8, idx: usize) protocol.Position {
    var line: u32 = 0;
    var col: u32 = 0;
    for (src[0..idx]) |ch| {
        if (ch == '\n') {
            line += 1;
            col = 0;
        } else col += 1;
    }
    return .{ .line = line, .character = col };
}

/// The position of the first occurrence of `needle`.
pub fn posOf(src: []const u8, needle: []const u8) protocol.Position {
    return posAt(src, std.mem.indexOf(u8, src, needle).?);
}

/// The position just past the last occurrence of `needle`.
pub fn posAfterLast(src: []const u8, needle: []const u8) protocol.Position {
    return posAt(src, std.mem.lastIndexOf(u8, src, needle).? + needle.len);
}

/// A program with every method-callee shape (concrete, generic-impl instance and static,
/// bounded `T has P`, derive, scalar/str builtins, Option) plus a same-named enum variant.
pub const method_fixture =
    \\struct Acc { n: int }
    \\impl Acc {
    \\    fn plus(self, a: int, b: int) -> int { return self.n + a + b }
    \\}
    \\struct Box[T] { v: T }
    \\impl Box[T] {
    \\    fn make(v: T) -> Box[T] { Box[T] { v: v } }
    \\    fn get(self) -> T { self.v }
    \\}
    \\protocol Doubler {
    \\    fn dbl(self) -> int
    \\}
    \\impl Acc has Doubler {
    \\    fn dbl(self) -> int { self.n * 2 }
    \\}
    \\struct Key { a: int }
    \\enum Kind { plus, other }
    \\fn twice[T has Doubler](v: T) -> int { v.dbl() + v.dbl() }
    \\fn main() -> int {
    \\    acc := Acc { n: 1 }
    \\    x := acc.plus(1, 2)
    \\    y := 1 |> acc.plus(2)
    \\    b := Box[int].make(3)
    \\    g := b.get()
    \\    s := "hi".concat("!")
    \\    e := if (3).eq(3) { 1 } else { 0 }
    \\    o: Option[int] = Option.some(4)
    \\    u := o.expect("no value") + o.or(0)
    \\    k := Key { a: 1 }.hash()
    \\    _k := Kind.plus
    \\    return x + y + g + s.len() + e + u + k + twice(acc)
    \\}
    \\
;

const Analysis = @import("Analysis.zig");
const Documents = @import("Documents.zig");
const Workspace = @import("Workspace.zig");

/// The analysis of `src` as the unsaved document `file:///p/main.toy`, with no other
/// document open and no disk; free with `deinit`.
pub fn analyze(gpa: std.mem.Allocator, src: []const u8) !*Analysis {
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    return analyzeIn(gpa, .{ .io = std.Io.failing, .docs = &docs, .disk = false }, "file:///p/main.toy", src);
}

/// The analysis of `src` as the unsaved document `uri` in `ws`; free with `deinit`.
pub fn analyzeIn(gpa: std.mem.Allocator, ws: Workspace, uri: []const u8, src: []const u8) !*Analysis {
    return Analysis.init(gpa, ws, uri, src, 1);
}

/// Method calls inside generic bodies (issue #33): a generic impl's method calling another,
/// a generic fn calling a generic type's method, and a concrete type's method called and
/// bound inside a generic fn with two instances.
pub const generic_body_fixture =
    \\struct Acc { n: int }
    \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
    \\struct Box[T] { v: T }
    \\impl Box[T] {
    \\    fn get(self) -> T { self.v }
    \\    fn twice(self) -> T { self.get() }
    \\}
    \\fn use_box[T](b: Box[T]) -> T { b.get() }
    \\fn with_acc[T](x: T) -> int {
    \\    acc := Acc { n: 1 }
    \\    g := acc.plus
    \\    return acc.plus(2) + g(3)
    \\}
    \\fn main() -> int {
    \\    b := Box[int]{ v: 3 }
    \\    x := b.get()
    \\    return x + b.twice() + use_box(b) + with_acc(1) + with_acc("s")
    \\}
    \\
;

/// Field privacy across modules: the entry `privacy_main` (at `privacy_main_uri`) reads a
/// pub and a private field of `lib.P` and its own struct's fields; `privacy_lib` is open
/// beside it at `privacy_lib_uri`. The private field comes first so an order by visibility
/// differs from declaration order. `p.y` is the one T0078.
pub const privacy_lib_uri = "file:///priv/lib.toy";
pub const privacy_main_uri = "file:///priv/main.toy";
pub const privacy_lib =
    \\pub struct P { y: int, pub x: int }
    \\pub fn mk() -> P { P { y: 2, x: 1 } }
    \\
;
pub const privacy_main =
    \\import lib
    \\struct Q { b: int, pub a: int }
    \\fn main() -> int {
    \\    p := lib.mk()
    \\    q := Q { b: 2, a: 1 }
    \\    return p.x + p.y + q.b
    \\}
    \\
;

/// An entry that sets a private field of `lib.P` in a struct literal: T0078 at `y`.
pub const privacy_literal =
    \\import lib
    \\fn main() -> int {
    \\    r := lib.P { y: 3, x: 4 }
    \\    return r.x
    \\}
    \\
;

/// A workspace with only `privacy_lib` open, disk off; free `docs` after every analysis
/// made in it.
pub fn privacyWorkspace(gpa: std.mem.Allocator, docs: *Documents) !Workspace {
    try docs.put(gpa, privacy_lib_uri, privacy_lib, 1);
    return .{ .io = std.Io.failing, .docs = docs, .disk = false };
}

/// A folder module `util` (`util/mod.toy`) that re-exports its child `util/math` whole and
/// selectively, one item renamed, beside an entry `reexport_main` that imports through it,
/// renames again, and imports the child directly. Open beside each other, disk off.
pub const reexport_math_uri = "file:///rx/util/math.toy";
pub const reexport_util_uri = "file:///rx/util/mod.toy";
pub const reexport_main_uri = "file:///rx/main.toy";
pub const reexport_math =
    \\pub fn sq(x: int) -> int { x * x }
    \\pub struct Pt { pub x: int }
    \\fn hidden() -> int { 0 }
    \\
;
pub const reexport_util =
    \\pub import util/math.{sq as square, Pt}
    \\pub import util/math
    \\fn helper() -> int { 1 }
    \\pub fn side() -> int { helper() }
    \\
;
pub const reexport_main =
    \\import util.{square, Pt as P, side}
    \\import util/math.{sq}
    \\import util
    \\
    \\fn main() -> int {
    \\    p := P { x: 2 }
    \\    return square(p.x) + sq(1) + side() + util.square(1)
    \\}
    \\
;

/// A workspace with `reexport_math` and `reexport_util` open, disk off; free `docs` after
/// every analysis made in it.
pub fn reexportWorkspace(gpa: std.mem.Allocator, docs: *Documents) !Workspace {
    try docs.put(gpa, reexport_math_uri, reexport_math, 1);
    try docs.put(gpa, reexport_util_uri, reexport_util, 1);
    return .{ .io = std.Io.failing, .docs = docs, .disk = false };
}
