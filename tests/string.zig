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
    \\fn main() -> int {
    \\    print("hello".concat(" ").concat("world"))
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
