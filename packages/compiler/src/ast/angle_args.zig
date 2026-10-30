//! C++/Rust-style `<...>` type arguments in EXPRESSION position: `Vec<int>.new()`,
//! `f<int>(x)`, `Map<str, int>.new()`. There `<` and `>` are comparisons, and
//! `a < b > (c)` is a legal parse, so the shape is never acted on while parsing.
//! Instead a stage that has already failed hands each error to `claim`: an error
//! inside the shape that the shape itself explains becomes one P0010 over the
//! `<...>`, so a valid program can never change (issue #23 ruling).

const std = @import("std");
const Token = @import("Token.zig").Token;
const Tag = @import("Token.zig").Tag;
const Diagnostic = @import("../diagnostics/Diagnostic.zig").Diagnostic;
const codes = @import("../diagnostics/codes.zig");

pub const message = "type arguments use '[]', not '<>': write 'Vec[int].new()' or 'f[int](x)'";

/// One `Name<...>` followed by `(` or `.member`, as token indices.
const Shape = struct {
    lt: usize,
    gt: usize,
    /// The last token an error can sit on and still be an artifact of the shape:
    /// the `>` before a call, or the member after `.` (`new` reads as undeclared).
    last: usize,
};

fn typeToken(tag: Tag) bool {
    return switch (tag) {
        .identifier, .dot, .comma, .lt, .gt, .gt_gt, .gt_dot, .l_bracket, .r_bracket => true,
        else => false,
    };
}

fn shapeAt(tokens: []const Token, lt: usize) ?Shape {
    if (lt == 0 or tokens[lt - 1].tag != .identifier) return null;
    if (lt + 1 >= tokens.len or tokens[lt + 1].tag != .identifier) return null;
    var depth: i32 = 0;
    var k = lt;
    while (k < tokens.len) : (k += 1) {
        switch (tokens[k].tag) {
            .lt => depth += 1,
            .gt, .gt_dot => depth -= 1,
            .gt_gt => depth -= 2,
            .identifier, .dot, .comma, .l_bracket, .r_bracket => {},
            else => return null,
        }
        if (depth == 0) break;
        if (depth < 0) return null;
    } else return null;
    if (k + 1 >= tokens.len) return null;
    // `>.` lexes as the float comparison, so `Vec<int>.new` ends on it.
    if (tokens[k].tag == .gt_dot)
        return if (tokens[k + 1].tag == .identifier) .{ .lt = lt, .gt = k, .last = k + 1 } else null;
    return switch (tokens[k + 1].tag) {
        .l_paren => .{ .lt = lt, .gt = k, .last = k },
        .dot => if (k + 2 < tokens.len and tokens[k + 2].tag == .identifier)
            .{ .lt = lt, .gt = k, .last = k + 2 }
        else
            null,
        else => null,
    };
}

const Claimable = struct { shape: Shape, at: usize };

/// The outermost shape holding the token at byte `offset`.
fn around(tokens: []const Token, offset: u32) ?Claimable {
    const at = std.sort.partitionPoint(Token, tokens, offset, struct {
        fn before(off: u32, t: Token) bool {
            return t.end <= off;
        }
    }.before);
    if (at >= tokens.len) return null;
    var best: ?Claimable = null;
    var j = at;
    while (typeToken(tokens[j].tag)) : (j -= 1) {
        if (tokens[j].tag == .lt) if (shapeAt(tokens, j)) |s| {
            if (s.last >= at) best = .{ .shape = s, .at = at };
        };
        if (j == 0) break;
    }
    return best;
}

/// Only the errors a type-argument reading explains are claimed: a parse error (the
/// comma), a type used as a value, or the member after `>.` read as undeclared. Any
/// other error, such as a typo in `f(a < b, cz > (d))`, is real in a comparison too,
/// so it is left alone.
fn explained(code: codes.Code, at: usize, s: Shape) bool {
    if (code == .T0056) return true;
    if (code == .R0001) return s.last > s.gt and at == s.last;
    const str = codes.str(code) orelse return false;
    return codes.hasPrefix(str, 'P');
}

/// Turn the error `d` into P0010 when it sits inside an expression-position
/// `<...>`. Duplicates this creates share one key, so the sink's dedup folds them.
pub fn claim(d: *Diagnostic, tokens: []const Token) void {
    if (d.severity != .err) return;
    const c = around(tokens, d.byte_offset) orelse return;
    if (!explained(d.code, c.at, c.shape)) return;
    const s = c.shape;
    d.* = .{
        .byte_offset = tokens[s.lt].start,
        .end = tokens[s.gt].end,
        .message = message,
        .scope = d.scope,
        .code = .P0010,
        .severity = codes.defaultSeverity(.P0010),
    };
}

const testing = std.testing;
const Lexer = @import("../lex.zig");

fn claimedAt(src: []const u8, needle: []const u8) !?[]const u8 {
    return claimedAs(src, needle, .T0056);
}

fn claimedAs(src: []const u8, needle: []const u8, code: codes.Code) !?[]const u8 {
    const tokens = try Lexer.tokenize(testing.allocator, src);
    defer testing.allocator.free(tokens);
    const off: u32 = @intCast(std.mem.indexOf(u8, src, needle).?);
    var d: Diagnostic = .{ .byte_offset = off, .end = off + 1, .message = "x", .code = code };
    claim(&d, tokens);
    if (d.code != .P0010) return null;
    return src[d.byte_offset..d.end];
}

test "claim covers the type arguments of a call or a member access" {
    try testing.expectEqualStrings("<int>", (try claimedAt("f<int>(x)\n", "int")).?);
    try testing.expectEqualStrings("<int>.", (try claimedAt("v := Vec<int>.new()\n", "new")).?);
    try testing.expectEqualStrings("<str, int>.", (try claimedAt("m := Map<str, int>.new()\n", ", int")).?);
    try testing.expectEqualStrings("<Vec<int>>", (try claimedAt("v := Vec<Vec<int>>.new()\n", "int>")).?);
    try testing.expectEqualStrings("<Option[int]>.", (try claimedAt("v := Vec<Option[int]>.new()\n", "int]")).?);
}

test "claim leaves errors outside the shape, and comparisons, alone" {
    try testing.expect(try claimedAt("v := Vec<int>.new(q)\n", "q") == null);
    try testing.expect(try claimedAt("v := Vec<int>.new()\n", "Vec") == null);
    try testing.expect(try claimedAt("x := a < b > c\n", "b") == null);
    try testing.expect(try claimedAt("x := a < b > 1 + (c)\n", "b") == null);
    try testing.expect(try claimedAt("fn id(x: Vec<int>) -> int\n", "int") == null);
}

test "claim keeps an undeclared name inside comparisons that only look like the shape" {
    try testing.expect(try claimedAs("x := foo(a < b, cz > (a))\n", "cz", .R0001) == null);
    try testing.expect(try claimedAs("if a < bb > (c) {}\n", "bb", .R0001) == null);
    try testing.expect(try claimedAs("x := foo(a < b, c > (a))\n", "c >", .T0017) == null);
    try testing.expectEqualStrings("<int>.", (try claimedAs("v := Vec<int>.new()\n", "new", .R0001)).?);
    try testing.expectEqualStrings("<int>", (try claimedAs("v := Vec<int> .new()\n", "new", .R0001)).?);
    try testing.expectEqualStrings("<str, int>.", (try claimedAs("m := Map<str, int>.new()\n", ", int", .P0004)).?);
}
