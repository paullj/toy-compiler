//! End-to-end coverage for the packed String: str/String concatenation + read-back,
//! content `==` across mixed str/String operands (including independent backing),
//! a push-forced grow (store-before-alloc + byte copy), single-handle reference
//! sharing, and -jN determinism. Each check surfaces as the child's EXIT CODE (0 on
//! success), the only thing this harness observes.

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
    \\import std/string
    \\fn main() -> int {
    \\    s := "ab" + "cd"
    \\    return if s.len() == 4 && s.byte_at(0) == 97 && s.byte_at(3) == 100 { 0 } else { 1 }
    \\}
    \\
;

test "string: str + str concatenates and reads back (numeric ASCII)" {
    try expectPass(".toy-test-string-concat", concat_src);
}

const eq_src =
    \\import std/string
    \\fn main() -> int {
    \\    if !(("th" + "e") == "the") { return 1 }
    \\    if ("th" + "e") == "no" { return 2 }
    \\    return 0
    \\}
    \\
;

test "string: (\"th\"+\"e\") content-equals \"the\" and differs from \"no\"" {
    try expectPass(".toy-test-string-eq", eq_src);
}

const eq_str_str_src =
    \\import std/string
    \\fn main() -> int {
    \\    a := "xy" + "z"
    \\    c := "x" + "yz"
    \\    return if a == c { 0 } else { 1 }
    \\}
    \\
;

test "string: String == String compares content across independent backing" {
    try expectPass(".toy-test-string-eq-ss", eq_str_str_src);
}

const grow_src =
    \\import std/string
    \\fn main() -> int {
    \\    s := String.new()
    \\    i := 0
    \\    while i < 100 {
    \\        s.push_byte(65 + (i % 26))
    \\        i = i + 1
    \\    }
    \\    return if s.len() == 100 && s.byte_at(50) == 65 + (50 % 26) { 0 } else { 1 }
    \\}
    \\
;

test "string: a push-forced grow copies bytes correctly and survives" {
    try expectPass(".toy-test-string-grow", grow_src);
}

const share_src =
    \\import std/string
    \\fn main() -> int {
    \\    s := String.new()
    \\    s.push_byte(65)
    \\    s2 := s
    \\    s2.push_byte(66)
    \\    return if s.len() == 2 { 0 } else { 1 }
    \\}
    \\
;

test "string: `s2 := s` shares the one buffer (single handle)" {
    try expectPass(".toy-test-string-share", share_src);
}

const det_src =
    \\import std/string
    \\fn main() -> int {
    \\    print("hello" + " " + "world")
    \\    return 0
    \\}
    \\
;

test "string: a String program is byte-identical at -j1 and -j8" {
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
