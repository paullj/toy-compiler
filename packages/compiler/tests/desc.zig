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
const harness = @import("harness.zig");

const skipUnlessBackend = harness.skipUnlessBackend;

const compile = harness.compile;

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
    const p1 = try compile(gpa, io, dir ++ "/j1", selftest_src, &.{ "--no-cache", "-j1" });
    defer gpa.free(p1);
    const p8 = try compile(gpa, io, dir ++ "/j8", selftest_src, &.{ "--no-cache", "-j8" });
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
