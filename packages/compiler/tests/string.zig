//! End-to-end coverage for the unified `str`: concat + read-back, method-based and
//! `==` content equality (including independent backing), int/bool `to_string`, an
//! owned (concat'd) str key surviving a forced collection in a Map, the std/str methods
//! (shared-byte substrings across forced collections, UTF-8 `chars`), and -jN
//! determinism. Each check surfaces as the child's EXIT CODE (0 == success).

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const harness = @import("harness.zig");

const skipUnlessBackend = harness.skipUnlessBackend;

const compile = harness.compile;

const runExit = harness.runExit;

const runStdout = harness.runStdout;

/// A kind's rendering under `io.print(x)` and `io.print(x.to_string())` must be the identical
/// bytes — the invariant `io.print(x) ≡ io.print(x.to_string())`, proven per kind against the
/// running program's stdout (not a compiler internal).
const PrintCase = struct { name: []const u8, defs: []const u8, expr: []const u8 };

const print_ts_cases = [_]PrintCase{
    .{ .name = "str", .defs = "", .expr = "\"hi there\"" },
    .{ .name = "int", .defs = "", .expr = "42" },
    .{ .name = "int-neg", .defs = "", .expr = "(0 - 7)" },
    .{ .name = "int-min", .defs = "", .expr = "(-9223372036854775807 - 1)" },
    .{ .name = "bool", .defs = "", .expr = "true" },
    .{ .name = "unit", .defs = "", .expr = "()" },
    .{ .name = "record", .defs = "struct Point { x: int, y: int }\n", .expr = "Point{ x: 4, y: 2 }" },
    .{ .name = "tuple", .defs = "struct Tup(int, bool)\n", .expr = "Tup(9, true)" },
    .{ .name = "enum-payload", .defs = "enum Color { R(int), G, B }\n", .expr = "Color.R(7)" },
    .{ .name = "enum-bare", .defs = "enum Color { R(int), G, B }\n", .expr = "Color.G" },
    .{ .name = "char-multibyte", .defs = "", .expr = "'\\u{20AC}'" },
    .{ .name = "nested", .defs = "struct Inner { u: int, v: int }\nstruct Outer { lo: Inner, hi: Inner }\n", .expr = "Outer{ lo: Inner{ u: 1, v: 2 }, hi: Inner{ u: 3, v: 4 } }" },
};

test "string: io.print(x) and io.print(x.to_string()) are byte-identical stdout for every kind" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    for (print_ts_cases) |c| {
        const src_print = try std.fmt.allocPrint(gpa, "import std/io\n{s}fn main() -> int {{ x := {s}\n io.print(x)\n return 0 }}\n", .{ c.defs, c.expr });
        defer gpa.free(src_print);
        const src_ts = try std.fmt.allocPrint(gpa, "import std/io\n{s}fn main() -> int {{ x := {s}\n io.print(x.to_string())\n return 0 }}\n", .{ c.defs, c.expr });
        defer gpa.free(src_ts);

        const dir_p = try std.fmt.allocPrint(gpa, ".toy-test-pts-{s}-p", .{c.name});
        defer gpa.free(dir_p);
        const dir_t = try std.fmt.allocPrint(gpa, ".toy-test-pts-{s}-t", .{c.name});
        defer gpa.free(dir_t);
        Io.Dir.cwd().deleteTree(io, dir_p) catch {};
        Io.Dir.cwd().deleteTree(io, dir_t) catch {};
        defer Io.Dir.cwd().deleteTree(io, dir_p) catch {};
        defer Io.Dir.cwd().deleteTree(io, dir_t) catch {};

        const prog_p = try compile(gpa, io, dir_p, src_print, &.{});
        defer gpa.free(prog_p);
        const prog_t = try compile(gpa, io, dir_t, src_ts, &.{});
        defer gpa.free(prog_t);

        const out_p = try runStdout(gpa, io, prog_p);
        defer gpa.free(out_p);
        const out_t = try runStdout(gpa, io, prog_t);
        defer gpa.free(out_t);

        std.testing.expectEqualStrings(out_p, out_t) catch |e| {
            std.debug.print("print==to_string mismatch for kind '{s}'\n", .{c.name});
            return e;
        };
    }
}

const custom_display_src =
    \\struct Money { cents: int }
    \\impl Money has Display {
    \\    fn display(self) -> str { return "$".concat(self.cents.to_string()) }
    \\}
    \\fn main() -> int {
    \\    m := Money{ cents: 42 }
    \\    if !(m.to_string() == "$42") { return 1 }
    \\    if !(m.to_string().concat("!") == "$42!") { return 2 }
    \\    return 0
    \\}
    \\
;

test "string: to_string on a custom-Display type returns the impl's str (universal, round-trips)" {
    try expectPass(".toy-test-string-custom-display", custom_display_src);
}

const wide_ts_src =
    \\struct Wide { a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int, j: int, k: int, l: int, m: int }
    \\fn main() -> int {
    \\    w := Wide{ a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7, h: 8, i: 9, j: 10, k: 11, l: 12, m: 13 }
    \\    if !(w.to_string() == "Wide{a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7, h: 8, i: 9, j: 10, k: 11, l: 12, m: 13}") { return 1 }
    \\    return 0
    \\}
    \\
;

test "string: a 13-field struct renders under to_string (no field-count cliff)" {
    try expectPass(".toy-test-string-wide-ts", wide_ts_src);
}

fn expectPass(dir: []const u8, src: []const u8) !void {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const prog = try compile(gpa, io, dir, src, &.{});
    defer gpa.free(prog);
    try std.testing.expectEqual(@as(u8, 0), try runExit(gpa, io, prog));
}

const concat_src =
    \\fn main() -> int {
    \\    s := "ab".concat("cd")
    \\    return if s.len() == 4 && s.byte_at(0) == 97 && s.byte_at(3) == 100 { 0 } else { 1 }
    \\}
    \\
;

test "string: concat allocates a heap buffer and reads back (numeric ASCII)" {
    try expectPass(".toy-test-string-concat", concat_src);
}

const eq_src =
    \\fn main() -> int {
    \\    if !("th".concat("e").eq("the")) { return 1 }
    \\    if "th".concat("e").eq("no") { return 2 }
    \\    if !(("th".concat("e")) == "the") { return 3 }
    \\    return 0
    \\}
    \\
;

test "string: concat result content-equals a literal via .eq() and ==, differs from another" {
    try expectPass(".toy-test-string-eq", eq_src);
}

const eq_backing_src =
    \\fn main() -> int {
    \\    a := "xy".concat("z")
    \\    c := "x".concat("yz")
    \\    return if a.eq(c) { 0 } else { 1 }
    \\}
    \\
;

test "string: two independently-backed concat results compare equal by content" {
    try expectPass(".toy-test-string-eq-backing", eq_backing_src);
}

const long_src =
    \\fn main() -> int {
    \\    s := ""
    \\    i := 0
    \\    while i < 50 {
    \\        s = s.concat("ab")
    \\        i = i + 1
    \\    }
    \\    return if s.len() == 100 && s.byte_at(0) == 97 && s.byte_at(99) == 98 { 0 } else { 1 }
    \\}
    \\
;

test "string: a long repeated concat (heap buffer) reads back correct" {
    try expectPass(".toy-test-string-long", long_src);
}

const to_string_src =
    \\fn main() -> int {
    \\    if !(42.to_string().eq("42")) { return 1 }
    \\    if !((0 - 7).to_string().eq("-7")) { return 2 }
    \\    if !(0.to_string().eq("0")) { return 3 }
    \\    if !(true.to_string().eq("true")) { return 4 }
    \\    return 0
    \\}
    \\
;

test "string: int/bool to_string render into a str comparable by content" {
    try expectPass(".toy-test-string-to-string", to_string_src);
}

const map_owned_key_src =
    \\import std/gc
    \\import std/map
    \\fn main() -> int {
    \\    m := Map[str, int].new()
    \\    m.set("a".concat("aa"), 1)
    \\    m.set("b".concat("bb"), 2)
    \\    m.set("c".concat("cc"), 3)
    \\    gc.collect()
    \\    m.set("a".concat("aa"), 10)
    \\    gc.collect()
    \\    if m.len() != 3 { return 1 }
    \\    a := match m.get("aaa") { .some(v) -> v, .none -> -1 }
    \\    if a != 10 { return 2 }
    \\    b := match m.get("bbb") { .some(v) -> v, .none -> -1 }
    \\    if b != 2 { return 3 }
    \\    return 0
    \\}
    \\
;

test "string: Map with owned (concat'd) str keys survives a forced collection and dedups by content" {
    try expectPass(".toy-test-string-map-owned", map_owned_key_src);
}

const det_src =
    \\import std/io
    \\fn main() -> int {
    \\    io.print("hello".concat(" ").concat("world"))
    \\    return 0
    \\}
    \\
;

test "string: a concat program is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    // The packed String codegen is a pure function of source.
    try harness.expectDeterministicBuild(gpa, io, ".toy-test-string-determinism", det_src, &.{}, null);
}

const ts_det_src =
    \\import std/io
    \\struct Point { x: int, y: int }
    \\fn main() -> int {
    \\    p := Point{ x: 4, y: 2 }
    \\    io.print(p)
    \\    io.print(p.to_string())
    \\    io.print(42.to_string())
    \\    return 0
    \\}
    \\
;

test "string: a print/to_string program is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    try harness.expectDeterministicBuild(gpa, io, ".toy-test-string-ts-determinism", ts_det_src, &.{}, null);
}

test "string: split, trim and join results survive forced collections" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The parts share the concat'd parent's buffer through interior pointers, and the
    // parent itself is dead after `split`, so only those pointers keep it alive. 'x' = 120.
    const codes = try harness.exitAtBothLevels(gpa, io, ".toy-test-string-split-gc",
        \\import std/str
        \\import std/vec
        \\import std/gc
        \\fn main() -> int {
        \\    src := "x".concat("alpha, beta ,gamma,delta  ")
        \\    parts := src.split(",") |> .map(fn(p) { p.trim() })
        \\    i := 0
        \\    while i < 200 {
        \\        junk: Vec[int] = [i, i, i]
        \\        _j := junk.len()
        \\        i = i + 1
        \\    }
        \\    gc.collect()
        \\    keep := parts.filter(fn(p) { p.ends_with("a") })
        \\    gc.collect()
        \\    joined := "|".join(keep)
        \\    if joined != "xalpha|beta|gamma|delta" { return 1 }
        \\    if "a-b".replace("-", "x".concat("y")) != "axyb" { return 2 }
        \\    it := joined.chars()
        \\    first := it.next().expect("no value")
        \\    v: int = first.into()
        \\    return v
        \\}
        \\
    );
    try std.testing.expectEqual([2]u8{ 120, 120 }, codes);
}

test "string: a join's buffer survives collections around it, over thousands of parts" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // Each join holds its buffer only in a `rawptr` until the str is made, and the junk
    // between joins keeps the collector busy, so a buffer the collector lost shows up as
    // wrong bytes. 13890 part bytes + 2999 commas = 16889, and 16889 % 251 = 72.
    const codes = try harness.exitAtBothLevels(gpa, io, ".toy-test-string-join-gc",
        \\import std/str
        \\import std/vec
        \\import std/gc
        \\fn main() -> int {
        \\    parts: Vec[str] = []
        \\    i := 0
        \\    while i < 3000 {
        \\        parts.push("p".concat(i.to_string()))
        \\        i = i + 1
        \\    }
        \\    n := 0
        \\    k := 0
        \\    while k < 40 {
        \\        s := ",".join(parts)
        \\        gc.collect()
        \\        junk: Vec[str] = ["a".concat("b"), "c".concat("d")]
        \\        _j := junk.len()
        \\        if !s.starts_with("p0,p1,p2,") || !s.ends_with(",p2998,p2999") { return 1 }
        \\        n = s.len()
        \\        k = k + 1
        \\    }
        \\    return n % 251
        \\}
        \\
    );
    try std.testing.expectEqual([2]u8{ 72, 72 }, codes);
}

const chars_src =
    \\import std/str
    \\import std/io
    \\fn main() -> int {
    \\    s := "aé€😀\xC3"
    \\    for c in s.chars() {
    \\        v: int = c.into()
    \\        io.print(v.to_string().concat(" "))
    \\    }
    \\    it := s.chars()
    \\    n := 0
    \\    loop {
    \\        o := it.next()
    \\        if o.is_none() { break }
    \\        io.print(o.expect("no value"))
    \\        n = n + 1
    \\    }
    \\    return n - 5
    \\}
    \\
;

test "string: chars decodes UTF-8 under for and a manual next, at -O0 and -O1" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-string-chars";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    for ([_][]const u8{ "-O0", "-O1" }) |level| {
        const sub = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, level[1..] });
        defer gpa.free(sub);
        const prog = try compile(gpa, io, sub, chars_src, &.{ "--no-cache", level });
        defer gpa.free(prog);
        const out = try runStdout(gpa, io, prog);
        defer gpa.free(out);
        try std.testing.expectEqualStrings("97 233 8364 128512 65533 aé€😀\u{FFFD}", out);
    }
}

test "string: a chars iterator keeps its string alive" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The iterator's copy of the str is the only reference to the concat'd buffer: under
    // `for` while each step forces a collection, and in a struct returned from a fn.
    const live = try harness.exitAtBothLevels(gpa, io, ".toy-test-string-chars-live",
        \\import std/str
        \\import std/vec
        \\import std/gc
        \\fn churn() {
        \\    i := 0
        \\    while i < 500 {
        \\        g := Vec[int].new()
        \\        g.push(i)
        \\        i = i + 1
        \\    }
        \\}
        \\fn main() -> int {
        \\    total := 0
        \\    for c in "h".concat("é€😀").chars() {
        \\        churn()
        \\        gc.collect()
        \\        k: int = c.into()
        \\        total = total + k
        \\    }
        \\    return total - 104 - 233 - 8364 - 128512 + 7
        \\}
        \\
    );
    try std.testing.expectEqual([2]u8{ 7, 7 }, live);
    const held = try harness.exitAtBothLevels(gpa, io, ".toy-test-string-chars-held",
        \\import std/str
        \\import std/vec
        \\import std/gc
        \\fn mk() -> Chars[char] {
        \\    return "h".concat("é€").chars()
        \\}
        \\fn main() -> int {
        \\    it := mk()
        \\    gc.collect()
        \\    g := Vec[int].new()
        \\    i := 0
        \\    while i < 2000 {
        \\        g.push(i)
        \\        i = i + 1
        \\    }
        \\    gc.collect()
        \\    total := 0
        \\    for c in it {
        \\        k: int = c.into()
        \\        total = total + k
        \\    }
        \\    return total - 104 - 233 - 8364 + 9
        \\}
        \\
    );
    try std.testing.expectEqual([2]u8{ 9, 9 }, held);
}

test "string: str methods are byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The lambdas and Vec instances bring the unit count past the 64 at which Pass C and
    // codegen fan out.
    const src =
        \\import std/str
        \\import std/vec
        \\fn words(s: str) -> Vec[str] {
        \\    return s.split(",") |> .map(fn(w) { w.trim() })
        \\}
        \\fn wide(s: str) -> int {
        \\    n := 0
        \\    for c in s.chars() {
        \\        k: int = c.into()
        \\        if k > 127 { n = n + 1 }
        \\    }
        \\    return n
        \\}
        \\fn main() -> int {
        \\    ws := words(" the quick,brown , fox ")
        \\    n := ws.fold(0, fn(a, w) { a + w.len() })
        \\    if !"-".join(ws).contains("quick-brown") { return 1 }
        \\    if !"a.b.c".replace(".", "::").starts_with("a::") { return 2 }
        \\    if !"name.toy".ends_with(".toy") { return 3 }
        \\    long := ws.filter(fn(w) { w.len() > 3 })
        \\    if !long.any(fn(w) { w.contains("q") }) || !ws.all(fn(w) { w.len() > 0 }) { return 4 }
        \\    ws.sort_by(fn(a, b) { a.len() < b.len() })
        \\    if ws.find(fn(w) { w.starts_with("b") }).or("") != "brown" { return 5 }
        \\    ws.each(fn(w) { n = n + wide(w) })
        \\    lens := "é€,😀".split(",").map(fn(p) { wide(p) })
        \\    return n + lens.fold(0, fn(a, x) { a + x }) + ws.get(0).expect("no value").len()
        \\}
        \\
    ;

    try harness.expectDeterministicBuild(gpa, io, ".toy-test-string-methods-determinism", src, &.{}, 23);
}

test "string: a program's len and byte_at on str steer its own calls, never trim, split or join" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // Were std's calls to reach them, `trim` would take 100000 bytes from a 6-byte str and
    // print the heap past it.
    try harness.expectStdoutAtBothLevels(gpa, io, ".toy-test-string-shadow-len",
        \\import std/str
        \\import std/io
        \\impl str {
        \\    fn len(self) -> int { 100000 }
        \\    fn byte_at(self, _i: int) -> int { 120 }
        \\}
        \\fn main() -> int {
        \\    io.println("  abc ".trim().concat("|"))
        \\    io.println("-".join("a b".split(" ")))
        \\    io.println("x".len())
        \\    io.println("x".byte_at(0))
        \\    return 0
        \\}
        \\
    , "abc|\na-b\n100000\n120\n");
}

test "string: core/text panics on a range outside its str or buffer, at -O0 and -O1" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // std/str computes in-range bounds, so only a program importing core/text itself can
    // pass a wrong one; each must stop with its message, not read or write past the bytes.
    const Case = struct { body: []const u8, msg: []const u8 };
    const cases = [_]Case{
        .{ .body = "_s := text.slice(\"hello\", 2, 9)", .msg = "str slice out of bounds" },
        .{ .body = "_s := text.slice(\"hello\", -1, 2)", .msg = "str slice out of bounds" },
        .{ .body = "_s := text.slice(\"hello\", 3, 2)", .msg = "str slice out of bounds" },
        .{ .body = "b := text.alloc(4)\n    text.put(b, 2, \"abc\")", .msg = "str buffer write out of bounds" },
        .{ .body = "b := text.alloc(4)\n    text.put(b, -1, \"a\")", .msg = "str buffer write out of bounds" },
        .{ .body = "b := text.alloc(4)\n    _s := text.take(b, 5)", .msg = "str buffer read out of bounds" },
        .{ .body = "_b := text.alloc(-1)", .msg = "str buffer size is negative" },
    };
    const dir = ".toy-test-string-text-bounds";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    for (cases, 0..) |c, k| for ([_][]const u8{ "-O0", "-O1" }) |level| {
        const src = try std.fmt.allocPrint(gpa, "import core/text\nfn main() -> int {{\n    {s}\n    return 0\n}}\n", .{c.body});
        defer gpa.free(src);
        const sub = try std.fmt.allocPrint(gpa, "{s}/{d}{s}", .{ dir, k, level[1..] });
        defer gpa.free(sub);
        const prog = try harness.compile(gpa, io, sub, src, &.{ "--no-cache", level, harness.dev_core_imports });
        defer gpa.free(prog);
        const run = try harness.runStderr(gpa, io, prog);
        defer gpa.free(run.err);
        std.testing.expect(run.term == .exited and run.term.exited == 1 and std.mem.startsWith(u8, run.err, c.msg)) catch |e| {
            std.debug.print("case {d} at {s}: {any}, stderr: {s}\n", .{ k, level, run.term, run.err });
            return e;
        };
    };
    // The same parts in range build the str.
    try harness.expectStdoutAtBothLevelsWith(gpa, io, dir,
        \\import core/text
        \\import std/io
        \\fn main() -> int {
        \\    b := text.alloc(5)
        \\    text.put(b, 0, "he")
        \\    text.put(b, 2, text.slice("ollo", 1, 4))
        \\    io.println(text.take(b, 5))
        \\    io.println(text.take(b, 0).len())
        \\    io.println(text.slice("hello", 5, 5).len())
        \\    return 0
        \\}
        \\
    , &.{harness.dev_core_imports}, "hello\n0\n0\n");
}
