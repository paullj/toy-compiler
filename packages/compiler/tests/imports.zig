//! Warm-cache soundness for selective imports and re-exports. A selection binds a local
//! name to another module's item, so changing what a name selects, or what a `pub import`
//! re-exports under it, must rebuild every fn that names it, even though the fn's own text
//! is unchanged. The search-root tests check the same for `--root` and `-I`: which file
//! an import names depends on the roots, so changing them must never serve a stale module.

const std = @import("std");
const Io = std.Io;
const harness = @import("harness.zig");
const flip = @import("places.zig").flip;

test "warm cache: re-pointing a selected fn's alias rebuilds its caller" {
    const lib =
        \\pub fn one() -> int { 1 }
        \\pub fn two() -> int { 2 }
        \\
    ;
    try flip(
        ".toy-test-imports-fn",
        &.{ .{ .name = "main.toy", .src = "import lib.{one as f}\nfn main() -> int { f() }\n" }, .{ .name = "lib.toy", .src = lib } },
        1,
        &.{.{ .name = "main.toy", .src = "import lib.{two as f}\nfn main() -> int { f() }\n" }},
        2,
    );
}

test "warm cache: re-pointing a selected struct's alias rebuilds the fns that touch it" {
    const lib =
        \\pub struct A { pub x: int }
        \\pub struct B { pub pad: int, pub x: int }
        \\
    ;
    try flip(
        ".toy-test-imports-struct",
        &.{ .{ .name = "main.toy", .src = "import lib.{A as S}\nfn get(s: S) -> int { s.x }\nfn main() -> int { get(S { x: 3 }) }\n" }, .{ .name = "lib.toy", .src = lib } },
        3,
        &.{.{ .name = "main.toy", .src = "import lib.{B as S}\nfn get(s: S) -> int { s.x }\nfn main() -> int { get(S { pad: 9, x: 4 }) }\n" }},
        4,
    );
}

test "warm cache: a selected fn whose module changes its body serves the new body" {
    try flip(
        ".toy-test-imports-body",
        &.{ .{ .name = "main.toy", .src = "import lib.{val}\nfn main() -> int { val() }\n" }, .{ .name = "lib.toy", .src = "pub fn val() -> int { 5 }\n" } },
        5,
        &.{.{ .name = "lib.toy", .src = "pub fn val() -> int { 6 }\n" }},
        6,
    );
}

test "warm cache: re-pointing a re-export rebuilds the importer's caller" {
    const lib =
        \\pub fn one() -> int { 1 }
        \\pub fn two() -> int { 2 }
        \\
    ;
    try flip(
        ".toy-test-imports-reexport",
        &.{ .{ .name = "main.toy", .src = "import mid.{f}\nimport mid\nfn main() -> int { f() + mid.f() }\n" }, .{ .name = "mid.toy", .src = "pub import lib.{one as f}\n" }, .{ .name = "lib.toy", .src = lib } },
        2,
        &.{.{ .name = "mid.toy", .src = "pub import lib.{two as f}\n" }},
        4,
    );
}

test "warm cache: a re-exported struct whose layout changes rebuilds the fns that touch it" {
    try flip(
        ".toy-test-imports-reexport-struct",
        &.{ .{ .name = "main.toy", .src = "import mid.{S, mk}\nfn get(s: S) -> int { s.x }\nfn main() -> int { get(mk(3)) }\n" }, .{ .name = "mid.toy", .src = "pub import lib.{S, mk}\n" }, .{ .name = "lib.toy", .src = "pub struct S { pub x: int }\npub fn mk(x: int) -> S { S { x: x } }\n" } },
        3,
        &.{.{ .name = "lib.toy", .src = "pub struct S { pub pad: int, pub x: int }\npub fn mk(x: int) -> S { S { pad: 9, x: x + 1 } }\n" }},
        4,
    );
}

/// Run the built `toy` with `args` in `cwd_abs`, capturing its output (caller frees).
fn toyIn(gpa: std.mem.Allocator, io: Io, cwd_abs: []const u8, args: []const []const u8) !struct { out: []u8, err: []u8, term: std.process.Child.Term } {
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.append(gpa, harness.toy_exe);
    try argv.appendSlice(gpa, args);
    var child = try std.process.spawn(io, .{ .argv = argv.items, .cwd = .{ .path = cwd_abs }, .stdout = .pipe, .stderr = .pipe });
    var ro = child.stdout.?.readerStreaming(io, &.{});
    const out = try ro.interface.allocRemaining(gpa, .limited(1 << 20));
    errdefer gpa.free(out);
    var re = child.stderr.?.readerStreaming(io, &.{});
    const err = try re.interface.allocRemaining(gpa, .limited(1 << 20));
    errdefer gpa.free(err);
    return .{ .out = out, .err = err, .term = try child.wait(io) };
}

/// A fresh `root` with `files` written under it; returns its absolute path (owned).
fn project(gpa: std.mem.Allocator, io: Io, root: []const u8, files: []const [2][]const u8) ![:0]u8 {
    Io.Dir.cwd().deleteTree(io, root) catch {};
    for (files) |f| {
        const path = try std.fs.path.join(gpa, &.{ root, f[0] });
        defer gpa.free(path);
        try Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(path).?);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = f[1] });
    }
    try Io.Dir.cwd().createDirPath(io, root);
    return Io.Dir.cwd().realPathFileAlloc(io, root, gpa);
}

test "search roots: -I resolves a sibling folder; a warm rebuild follows a changed -I set and an added clash" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    const root = ".toy-test-imports-roots";
    defer Io.Dir.cwd().deleteTree(io, root) catch {};
    const w = try project(gpa, io, root, &.{
        .{ "src/main.toy", "import extra\nfn main() -> int { extra.v() }\n" },
        .{ "lib/extra.toy", "pub fn v() -> int { 1 }\n" },
        .{ "alt/extra.toy", "pub fn v() -> int { 2 }\n" },
    });
    defer gpa.free(w);

    {
        const r = try toyIn(gpa, io, w, &.{ "build", "src/main.toy", "-o", "p0" });
        defer gpa.free(r.out);
        defer gpa.free(r.err);
        try std.testing.expectEqual(std.process.Child.Term{ .exited = 1 }, r.term);
        try std.testing.expect(std.mem.indexOf(u8, r.out, "imported module 'extra' not found (looked for 'src/extra.toy')") != null);
    }
    try harness.buildIn(gpa, io, w, &.{ "build", "src/main.toy", "-I", "lib", "-o", "p1" });
    try std.testing.expectEqual(@as(u8, 1), try harness.runExitIn(gpa, io, w, "p1"));
    try harness.buildIn(gpa, io, w, &.{ "build", "src/main.toy", "--include", "alt", "-o", "p2" });
    try std.testing.expectEqual(@as(u8, 2), try harness.runExitIn(gpa, io, w, "p2"));
    try harness.buildIn(gpa, io, w, &.{ "build", "src/main.toy", "-Ilib", "-o", "p3", "--verify" });
    try std.testing.expectEqual(@as(u8, 1), try harness.runExitIn(gpa, io, w, "p3"));

    try Io.Dir.cwd().writeFile(io, .{ .sub_path = root ++ "/src/extra.toy", .data = "pub fn v() -> int { 3 }\n" });
    inline for (.{ "check", "build" }) |verb| {
        const r = try toyIn(gpa, io, w, &.{ verb, "src/main.toy", "-I", "lib", "-I", "alt" });
        defer gpa.free(r.out);
        defer gpa.free(r.err);
        try std.testing.expectEqual(std.process.Child.Term{ .exited = 1 }, r.term);
        try std.testing.expect(std.mem.indexOf(u8, r.out, "error[R0014]: imported module 'extra' is found in more than one root ('src/extra.toy', 'lib/extra.toy', 'alt/extra.toy')") != null);
        try std.testing.expect(std.mem.indexOf(u8, r.out, "--> src/main.toy:1:8") != null);
    }
}

test "search roots: --root starts import paths at another folder, and a missing root dir is a usage error" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    const root = ".toy-test-imports-rootopt";
    defer Io.Dir.cwd().deleteTree(io, root) catch {};
    const w = try project(gpa, io, root, &.{
        .{ "app/src/main.toy", "import lib/extra\nfn main() -> int { extra.v() }\n" },
        .{ "app/lib/extra.toy", "pub fn v() -> int { 7 }\n" },
    });
    defer gpa.free(w);

    try harness.buildIn(gpa, io, w, &.{ "build", "app/src/main.toy", "--root", "app", "-o", "p1" });
    try std.testing.expectEqual(@as(u8, 7), try harness.runExitIn(gpa, io, w, "p1"));

    inline for (.{ .{ "--root", "nope" }, .{ "-I", "nope" } }) |flag| {
        inline for (.{ "check", "build", "run" }) |verb| {
            const r = try toyIn(gpa, io, w, &.{ verb, "app/src/main.toy", flag[0], flag[1] });
            defer gpa.free(r.out);
            defer gpa.free(r.err);
            try std.testing.expectEqual(std.process.Child.Term{ .exited = 2 }, r.term);
            try std.testing.expect(std.mem.indexOf(u8, r.err, "invalid value 'nope' for " ++ flag[0] ++ " (a directory)") != null);
        }
    }
}
