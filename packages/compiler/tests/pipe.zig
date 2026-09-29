//! Proof that `|>` is pure syntax: every run case in
//! `tests/corpora/language-features/pipe/` compiles to the same binary as its
//! hand-desugared twin in `pipe_twins/`. The corpus harness already checks that each
//! case runs correctly; this checks that it is the SAME program as the call form.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const harness = @import("harness.zig");

const cases_dir = "tests/corpora/language-features/pipe";
const twins_dir = "packages/compiler/tests/pipe_twins";

const skipUnlessBackend = harness.skipUnlessBackend;

/// Build `src_path` as `<dir>/main.toy` -> `<dir>/prog` and return the binary's bytes.
/// The binary embeds the input and output basenames, so both sides of a pair must use
/// the same ones or every pair would differ.
fn build(gpa: std.mem.Allocator, io: Io, dir: []const u8, src_path: []const u8) ![]u8 {
    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, "zig-out/bin/toy", gpa);
    defer gpa.free(bin_abs);
    try Io.Dir.cwd().createDirPath(io, dir);

    const src = try Io.Dir.cwd().readFileAlloc(io, src_path, gpa, .unlimited);
    defer gpa.free(src);
    const main_path = try std.fmt.allocPrint(gpa, "{s}/main.toy", .{dir});
    defer gpa.free(main_path);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = main_path, .data = src });
    const out_bin = try std.fmt.allocPrint(gpa, "{s}/prog", .{dir});
    defer gpa.free(out_bin);

    // `--no-cache`: a warm cache could hand the twin the pipe build's cached code, and the
    // comparison would pass without testing anything. stderr is inherited so a twin that
    // stops compiling shows its diagnostics.
    var child = try std.process.spawn(io, .{
        .argv = &.{ bin_abs, "build", main_path, "-o", out_bin, "--no-cache" },
        .stdout = .ignore,
        .stderr = .inherit,
    });
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) {
        std.debug.print("failed to build {s}\n", .{src_path});
        return error.CompileFailed;
    }
    return Io.Dir.cwd().readFileAlloc(io, out_bin, gpa, .unlimited);
}

test "pipe: every pipe run case is byte-identical to its hand-desugared twin" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const tmp = ".toy-test-pipe-twins";
    Io.Dir.cwd().deleteTree(io, tmp) catch {};
    defer Io.Dir.cwd().deleteTree(io, tmp) catch {};

    var root = try Io.Dir.cwd().openDir(io, cases_dir, .{ .iterate = true });
    defer root.close(io);
    var walker = try root.walk(gpa);
    defer walker.deinit();

    var compared: usize = 0;
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".toy")) continue;
        const case_path = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ cases_dir, entry.path });
        defer gpa.free(case_path);
        const twin_path = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ twins_dir, entry.path });
        defer gpa.free(twin_path);
        Io.Dir.cwd().access(io, twin_path, .{}) catch {
            std.debug.print("pipe case {s} has no twin at {s}\n", .{ case_path, twin_path });
            return error.MissingTwin;
        };

        const pipe_dir = try std.fmt.allocPrint(gpa, "{s}/{s}/pipe", .{ tmp, entry.path });
        defer gpa.free(pipe_dir);
        const twin_dir = try std.fmt.allocPrint(gpa, "{s}/{s}/twin", .{ tmp, entry.path });
        defer gpa.free(twin_dir);
        const a = try build(gpa, io, pipe_dir, case_path);
        defer gpa.free(a);
        const b = try build(gpa, io, twin_dir, twin_path);
        defer gpa.free(b);

        if (!std.mem.eql(u8, a, b)) {
            std.debug.print("pipe case {s} differs from its twin ({d} vs {d} bytes)\n", .{ case_path, a.len, b.len });
            return error.TestExpectedEqual;
        }
        compared += 1;
    }
    // Every twin must have a case, so a renamed or deleted case cannot leave a stale twin.
    try std.testing.expect(compared > 0);
    try std.testing.expectEqual(try countToy(gpa, io, twins_dir), compared);
}

fn countToy(gpa: std.mem.Allocator, io: Io, path: []const u8) !usize {
    var dir = try Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(gpa);
    defer walker.deinit();
    var n: usize = 0;
    while (try walker.next(io)) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.path, ".toy")) n += 1;
    }
    return n;
}
