//! End-to-end coverage for the indirect-call subsystem + descriptor table.
//!
//! The subsystem has no direct user surface, so it is exercised through the bundled,
//! allowlisted `core/desc_selftest` module: it obtains a type's descriptor
//! (`descriptor_of[T]()`), reads the hash/eq function offsets from it, resolves those
//! to runtime addresses via `text_base()`, and `call_indirect`s the erased hash/eq on
//! raw key bytes — then compares against the inline `.hash()` / `==`. Each check
//! surfaces as the child's EXIT CODE (0 on success), the only thing this harness
//! observes. A `-j1`/`-j8` byte-identity case pins the new determinism surface.

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

const selftest_src =
    \\import core/desc_selftest
    \\fn main() -> int { return desc_selftest.selftest() }
    \\
;

test "desc: descriptor_of + call_indirect on struct/str/int keys matches the inline witnesses" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-desc-selftest";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const prog = try compile(gpa, io, dir, selftest_src, &.{});
    defer gpa.free(prog);
    var run = try std.process.spawn(io, .{ .argv = &.{prog} });
    const term = try run.wait(io);
    const code = switch (term) {
        .exited => |c| c,
        else => return error.ChildCrashed,
    };
    try std.testing.expectEqual(@as(u8, 0), code);
}

test "desc: the descriptor-table program is byte-identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);

    const dir = ".toy-test-desc-determinism";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // Same `-o` basename (`prog`) in both dirs, so the ad-hoc code-sign IDENTIFIER (derived
    // from the basename) is identical — only the one-byte code-sign nonce may differ. The
    // descriptor table + erased units are pure functions of source (canonical order), so
    // codegen is `-jN` byte-identical.
    const p1 = try compile(gpa, io, dir ++ "/j1", selftest_src, &.{ "--force", "-j1" });
    defer gpa.free(p1);
    const p8 = try compile(gpa, io, dir ++ "/j8", selftest_src, &.{ "--force", "-j8" });
    defer gpa.free(p8);

    const b1 = try Io.Dir.cwd().readFileAlloc(io, p1, gpa, .unlimited);
    defer gpa.free(b1);
    const b8 = try Io.Dir.cwd().readFileAlloc(io, p8, gpa, .unlimited);
    defer gpa.free(b8);

    try std.testing.expectEqual(b1.len, b8.len);
    var differing: usize = 0;
    for (b1, b8) |x, y| {
        if (x != y) differing += 1;
    }
    try std.testing.expect(differing <= 1);
}
