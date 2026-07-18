//! End-to-end coverage for the unified `str`: concat + read-back, method-based and
//! `==` content equality (including independent backing), int/bool `to_string`, an
//! owned (concat'd) str key surviving a forced collection in a Map, and -jN
//! determinism. Each check surfaces as the child's EXIT CODE (0 == success).

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

fn skipUnlessBackend(io: Io) !void {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;
}

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

fn runExit(gpa: std.mem.Allocator, io: Io, prog: []const u8) !u8 {
    var run = try std.process.spawn(io, .{ .argv = &.{prog} });
    const term = try run.wait(io);
    _ = gpa;
    return switch (term) {
        .exited => |c| c,
        else => error.ChildCrashed,
    };
}

fn runStdout(gpa: std.mem.Allocator, io: Io, prog: []const u8) ![]u8 {
    var child = try std.process.spawn(io, .{ .argv = &.{prog}, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const out = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    errdefer gpa.free(out);
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) return error.ChildFailed;
    return out;
}

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
    \\import core/map_selftest
    \\fn main() -> int { return map_selftest.concat_str_key_survives() }
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

    const dir = ".toy-test-string-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // Same `-o` basename (`prog`) in both dirs → identical basename-derived code-sign
    // IDENTIFIER, and the packed String codegen is a pure function of source, so the two
    // binaries are byte-for-byte identical (assert EXACT equality so a genuine 1-byte
    // non-determinism cannot hide behind a slack tolerance).
    const p1 = try compile(gpa, io, dir ++ "/j1", det_src, &.{ "--force", "-j1" });
    defer gpa.free(p1);
    const p8 = try compile(gpa, io, dir ++ "/j8", det_src, &.{ "--force", "-j8" });
    defer gpa.free(p8);

    const b1 = try Io.Dir.cwd().readFileAlloc(io, p1, gpa, .unlimited);
    defer gpa.free(b1);
    const b8 = try Io.Dir.cwd().readFileAlloc(io, p8, gpa, .unlimited);
    defer gpa.free(b8);

    try std.testing.expectEqualSlices(u8, b1, b8);
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

    const dir = ".toy-test-string-ts-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const p1 = try compile(gpa, io, dir ++ "/j1", ts_det_src, &.{ "--force", "-j1" });
    defer gpa.free(p1);
    const p8 = try compile(gpa, io, dir ++ "/j8", ts_det_src, &.{ "--force", "-j8" });
    defer gpa.free(p8);

    const b1 = try Io.Dir.cwd().readFileAlloc(io, p1, gpa, .unlimited);
    defer gpa.free(b1);
    const b8 = try Io.Dir.cwd().readFileAlloc(io, p8, gpa, .unlimited);
    defer gpa.free(b8);

    try std.testing.expectEqualSlices(u8, b1, b8);
}
