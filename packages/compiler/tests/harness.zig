//! Shared helpers for the integration tests that build a toy program with the `toy` built
//! alongside this test binary and run the result.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

/// The absolute install path of the `toy` this test binary was built with. build.zig
/// installs it before any integration test runs, under `test` and `test-bin` alike, so a
/// test never drives a stale or missing compiler.
pub const toy_exe: []const u8 = @import("test_options").toy_exe;

/// A missing compiler is a failure, never a skip: a skip here once hid every CLI test on a
/// fresh checkout, where nothing had installed `toy` yet.
pub fn requireToy(io: Io) !void {
    Io.Dir.accessAbsolute(io, toy_exe, .{}) catch |e| {
        std.debug.print("no toy at {s} ({t}); run the integration tests through 'zig build test' or 'zig build test-bin'\n", .{ toy_exe, e });
        return error.ToyMissing;
    };
}

/// The emitted binary is macOS/aarch64 Mach-O, and these tests drive the CLI.
pub fn skipUnlessBackend(io: Io) !void {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    try requireToy(io);
}

/// Write `entry_src` to `dir_name/main.toy`, build it to `dir_name/prog`, and return the
/// binary's absolute path (owned by `gpa`). Keeping the `main.toy`/`prog` basenames fixed
/// matters: the binary embeds them, so two builds only compare equal with the same names.
pub fn compile(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, entry_src: []const u8, flags: []const []const u8) ![:0]u8 {
    const bin_abs = toy_exe;

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

/// Build `src` with `flags` at -j1 and at -j8 (each `--no-cache`, in fresh subdirs of
/// `dir_name`) and expect the two images to be identical; with `exit` set, also expect the
/// -j1 binary to exit with it. Every byte must match: the code signature has no nonce, only
/// the `-o` basename (which `compile` keeps fixed), so any difference is a -jN leak.
pub fn expectDeterministicBuild(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, src: []const u8, flags: []const []const u8, exit: ?u8) !void {
    try skipUnlessBackend(io);
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const one = try buildImage(gpa, io, dir_name, "-j1", src, flags);
    defer one.deinit(gpa);
    const many = try buildImage(gpa, io, dir_name, "-j8", src, flags);
    defer many.deinit(gpa);
    if (std.mem.indexOfDiff(u8, one.bytes, many.bytes)) |at| {
        std.debug.print("-j1 and -j8 images differ at byte {d} of {d} and {d}\n", .{ at, one.bytes.len, many.bytes.len });
        return error.TestExpectedEqual;
    }
    if (exit) |code| try std.testing.expectEqual(code, try runExit(gpa, io, one.prog));
}

const Image = struct {
    prog: [:0]u8,
    bytes: []u8,

    fn deinit(img: Image, gpa: std.mem.Allocator) void {
        gpa.free(img.prog);
        gpa.free(img.bytes);
    }
};

/// `compile` at `jobs` (`-jN`) into `dir_name/jN`, with the image read back.
fn buildImage(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, jobs: []const u8, src: []const u8, flags: []const []const u8) !Image {
    const sub = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir_name, jobs[1..] });
    defer gpa.free(sub);
    const args = try std.mem.concat(gpa, []const u8, &.{ &.{ "--no-cache", jobs }, flags });
    defer gpa.free(args);
    const prog = try compile(gpa, io, sub, src, args);
    errdefer gpa.free(prog);
    return .{ .prog = prog, .bytes = try Io.Dir.cwd().readFileAlloc(io, prog, gpa, .unlimited) };
}

/// Run the built `toy` with `args` in the directory `cwd_abs` (absolute), which owns the
/// `.toy/` cache the build reads and writes. Fails unless toy exits 0.
pub fn buildIn(gpa: std.mem.Allocator, io: Io, cwd_abs: []const u8, args: []const []const u8) !void {
    const bin_abs = toy_exe;
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.append(gpa, bin_abs);
    try argv.appendSlice(gpa, args);
    var child = try std.process.spawn(io, .{ .argv = argv.items, .cwd = .{ .path = cwd_abs }, .stdout = .pipe, .stderr = .pipe });
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) return error.CompileFailed;
}

/// `runExit` of the program at `rel`, a path relative to `cwd_abs`.
pub fn runExitIn(gpa: std.mem.Allocator, io: Io, cwd_abs: []const u8, rel: []const u8) !u8 {
    const prog = try std.fs.path.join(gpa, &.{ cwd_abs, rel });
    defer gpa.free(prog);
    return runExit(gpa, io, prog);
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

/// The stdout of `prog`, which must exit 0.
pub fn runStdout(gpa: std.mem.Allocator, io: Io, prog: []const u8) ![]u8 {
    var child = try std.process.spawn(io, .{ .argv = &.{prog}, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const out = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    errdefer gpa.free(out);
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) return error.ChildFailed;
    return out;
}

/// What `prog` wrote to stderr, and how it ended.
pub fn runStderr(gpa: std.mem.Allocator, io: Io, prog: []const u8) !struct { err: []u8, term: std.process.Child.Term } {
    var child = try std.process.spawn(io, .{ .argv = &.{prog}, .stdout = .ignore, .stderr = .pipe });
    var rdr = child.stderr.?.readerStreaming(io, &.{});
    const err = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    errdefer gpa.free(err);
    return .{ .err = err, .term = try child.wait(io) };
}

/// Build `src` at -O0 and at -O1 into separate dirs under `dir`, run each, and expect
/// `want` on stdout from both.
pub fn expectStdoutAtBothLevels(gpa: std.mem.Allocator, io: Io, dir: []const u8, src: []const u8, want: []const u8) !void {
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    for ([_][]const u8{ "-O0", "-O1" }) |level| {
        const sub = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, level[1..] });
        defer gpa.free(sub);
        const prog = try compile(gpa, io, sub, src, &.{ "--no-cache", level });
        defer gpa.free(prog);
        const out = try runStdout(gpa, io, prog);
        defer gpa.free(out);
        std.testing.expectEqualStrings(want, out) catch |e| {
            std.debug.print("at {s}\n", .{level});
            return e;
        };
    }
}

/// `compile` + `runExit` in a fresh `dir`, removed afterwards.
pub fn buildAndRun(gpa: std.mem.Allocator, io: Io, dir: []const u8, src: []const u8) !u8 {
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    const prog = try compile(gpa, io, dir, src, &.{});
    defer gpa.free(prog);
    return runExit(gpa, io, prog);
}

/// Build `src` at -O0 and at -O1 into separate dirs under `dir` and return both exit codes.
pub fn exitAtBothLevels(gpa: std.mem.Allocator, io: Io, dir: []const u8, src: []const u8) ![2]u8 {
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var codes: [2]u8 = undefined;
    for ([_][]const u8{ "-O0", "-O1" }, 0..) |level, i| {
        const sub = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, level[1..] });
        defer gpa.free(sub);
        const prog = try compile(gpa, io, sub, src, &.{ "--no-cache", level });
        defer gpa.free(prog);
        codes[i] = try runExit(gpa, io, prog);
    }
    return codes;
}

/// The `--emit ir` text of `path` built with `flags` (owned).
pub fn emitIr(gpa: std.mem.Allocator, io: Io, path: []const u8, flags: []const []const u8) ![]u8 {
    const bin = toy_exe;
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.appendSlice(gpa, &.{ bin, "--emit", "ir" });
    try argv.appendSlice(gpa, flags);
    try argv.append(gpa, path);
    var child = try std.process.spawn(io, .{ .argv = argv.items, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const ir = try rdr.interface.allocRemaining(gpa, .limited(1 << 22));
    errdefer gpa.free(ir);
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) return error.EmitFailed;
    return ir;
}

/// The body of `fn <name>(` in `ir`, up to its closing brace.
pub fn irBody(ir: []const u8, name: []const u8) ?[]const u8 {
    var buf: [128]u8 = undefined;
    const head = std.fmt.bufPrint(&buf, "fn {s}(", .{name}) catch return null;
    const at = if (std.mem.startsWith(u8, ir, head)) 0 else (std.mem.indexOf(u8, ir, head) orelse return null);
    const end = std.mem.indexOfPos(u8, ir, at, "\n}\n") orelse ir.len;
    return ir[at..end];
}
