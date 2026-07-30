//! End-to-end coverage for `std/math`: the pure-integer helpers (min/max/clamp/sign/pow,
//! plus `abs` over `core/ffi`'s `labs`) and the libm float ops (sqrt/floor/ceil) reached
//! through `core/ffi`'s safe wrappers — the float extern round-trips an f64 across the
//! AAPCS64 V-register boundary. Each check surfaces as the child's EXIT CODE (0/computed).

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

fn buildAndRun(gpa: std.mem.Allocator, io: Io, dir: []const u8, src: []const u8) !u8 {
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const prog = try compile(gpa, io, dir, src, &.{});
    defer gpa.free(prog);
    return runExit(gpa, io, prog);
}

test "math: integer helpers min/max/clamp/sign/pow/abs" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // 3 + 7 + 3 + 32 + 4 == 49, then + sign(-2)+sign(0)+sign(9) == 49 + 0 == 49.
    const code = try buildAndRun(gpa, io, ".toy-test-math-int",
        \\import std/math
        \\fn main() -> int {
        \\    base := math.min(3, 7) + math.max(3, 7) + math.clamp(5, 0, 3) + math.pow(2, 5) + math.abs(-4)
        \\    return base + math.sign(-2) + math.sign(0) + math.sign(9)
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 49), code);
}

test "math: libm sqrt over the float extern (16.0 -> 4)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The f64 arg rides a V-register into libSystem `sqrt`; the f64 result rides back and
    // truncates to 4. Proves float marshals correctly across the extern boundary.
    const code = try buildAndRun(gpa, io, ".toy-test-math-sqrt",
        \\import std/math
        \\fn main() -> int {
        \\    return math.sqrt(16.0).try_into().unwrap()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 4), code);
}

test "math: libm floor/ceil round toward -inf/+inf" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // floor(3.7)=3, ceil(3.2)=4 -> 3 + 4 == 7.
    const code = try buildAndRun(gpa, io, ".toy-test-math-floorceil",
        \\import std/math
        \\fn main() -> int {
        \\    f: int = math.floor(3.7).try_into().unwrap()
        \\    c: int = math.ceil(3.2).try_into().unwrap()
        \\    return f + c
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}
