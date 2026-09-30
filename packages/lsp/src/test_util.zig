//! Cursor positions for LSP tests, computed from the fixture text so no test hand-counts
//! a column. ASCII fixtures, so a byte offset equals a character offset.

const std = @import("std");
const protocol = @import("protocol.zig");

fn posAt(src: []const u8, idx: usize) protocol.Position {
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
    \\    u := o.unwrap() + o.unwrap_or(0)
    \\    k := Key { a: 1 }.hash()
    \\    _k := Kind.plus
    \\    return x + y + g + s.len() + e + u + k + twice(acc)
    \\}
    \\
;
