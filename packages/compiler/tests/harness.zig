//! Shared helpers for the integration tests that build a toy program with the installed
//! `zig-out/bin/toy` and run the result.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

/// The emitted binary is macOS/aarch64 Mach-O, and these tests drive the installed CLI.
pub fn skipUnlessBackend(io: Io) !void {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;
}

/// Write `entry_src` to `dir_name/main.toy`, build it to `dir_name/prog`, and return the
/// binary's absolute path (owned by `gpa`). Keeping the `main.toy`/`prog` basenames fixed
/// matters: the binary embeds them, so two builds only compare equal with the same names.
pub fn compile(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, entry_src: []const u8, flags: []const []const u8) ![:0]u8 {
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

pub fn runExit(gpa: std.mem.Allocator, io: Io, prog: []const u8) !u8 {
    var run = try std.process.spawn(io, .{ .argv = &.{prog} });
    const term = try run.wait(io);
    _ = gpa;
    return switch (term) {
        .exited => |c| c,
        else => error.ChildCrashed,
    };
}

/// `compile` + `runExit` in a fresh `dir`, removed afterwards.
pub fn buildAndRun(gpa: std.mem.Allocator, io: Io, dir: []const u8, src: []const u8) !u8 {
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const prog = try compile(gpa, io, dir, src, &.{});
    defer gpa.free(prog);
    return runExit(gpa, io, prog);
}
