//! Hybrid specialization is an optimization only: every `closures/spec_*.toy` program
//! behaves the same with `--no-specialize`, the number of spec units a template gets is
//! capped, and a specialized build is as deterministic as any other.

const std = @import("std");
const Io = std.Io;
const harness = @import("harness.zig");

const cases_dir = "tests/corpora/language-features/closures";

const Ran = struct { exit: u8, stdout: []u8 };

fn run(gpa: std.mem.Allocator, io: Io, prog: []const u8) !Ran {
    var child = try std.process.spawn(io, .{ .argv = &.{prog}, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const out = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    errdefer gpa.free(out);
    const term = try child.wait(io);
    return switch (term) {
        .exited => |c| .{ .exit = c, .stdout = out },
        else => error.ChildCrashed,
    };
}

/// Every `spec_*.toy` in the closures corpus, as `cases_dir`-relative paths (owned).
fn specCases(gpa: std.mem.Allocator, io: Io) !std.ArrayList([]u8) {
    var list: std.ArrayList([]u8) = .empty;
    errdefer {
        for (list.items) |p| gpa.free(p);
        list.deinit(gpa);
    }
    var dir = try Io.Dir.cwd().openDir(io, cases_dir, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.startsWith(u8, entry.name, "spec_") or !std.mem.endsWith(u8, entry.name, ".toy")) continue;
        try list.append(gpa, try std.fmt.allocPrint(gpa, "{s}/{s}", .{ cases_dir, entry.name }));
    }
    std.mem.sort([]u8, list.items, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.lt);
    return list;
}

fn freeCases(gpa: std.mem.Allocator, list: *std.ArrayList([]u8)) void {
    for (list.items) |p| gpa.free(p);
    list.deinit(gpa);
}

/// Build `path` into `dir/prog` (same basenames for every build, so images compare) and
/// return the image bytes.
fn buildBytes(gpa: std.mem.Allocator, io: Io, dir: []const u8, path: []const u8, flags: []const []const u8) ![]u8 {
    const src = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(src);
    const prog = try harness.compile(gpa, io, dir, src, flags);
    defer gpa.free(prog);
    return Io.Dir.cwd().readFileAlloc(io, prog, gpa, .unlimited);
}

test "every closures/spec_*.toy has the same exit and stdout with --no-specialize" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    const tmp = ".toy-test-closures-spec";
    Io.Dir.cwd().deleteTree(io, tmp) catch {};
    defer Io.Dir.cwd().deleteTree(io, tmp) catch {};

    var cases = try specCases(gpa, io);
    defer freeCases(gpa, &cases);
    try std.testing.expect(cases.items.len >= 5);
    for (cases.items, 0..) |path, i| {
        const src = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
        defer gpa.free(src);
        // A fresh path per build: macOS kills a process whose signed image was rewritten in
        // place after an earlier run of it.
        const on_dir = try std.fmt.allocPrint(gpa, tmp ++ "/{d}/on", .{i});
        defer gpa.free(on_dir);
        const off_dir = try std.fmt.allocPrint(gpa, tmp ++ "/{d}/off", .{i});
        defer gpa.free(off_dir);
        const on = try harness.compile(gpa, io, on_dir, src, &.{"--no-cache"});
        defer gpa.free(on);
        const off = try harness.compile(gpa, io, off_dir, src, &.{ "--no-cache", "--no-specialize" });
        defer gpa.free(off);
        const a = try run(gpa, io, on);
        defer gpa.free(a.stdout);
        const b = try run(gpa, io, off);
        defer gpa.free(b.stdout);
        if (a.exit != b.exit or !std.mem.eql(u8, a.stdout, b.stdout)) {
            std.debug.print("{s}: exit {d} vs {d} without specialization\n", .{ path, a.exit, b.exit });
            return error.TestExpectedEqual;
        }
        // The case must actually specialize, or this compares a program with itself: it
        // has a spec unit, and every spec unit has a caller.
        const ir = try emitIr(gpa, io, path);
        defer gpa.free(ir);
        var units: usize = 0;
        var lines = std.mem.splitScalar(u8, ir, '\n');
        while (lines.next()) |line| {
            if (!std.mem.startsWith(u8, line, "fn ") or std.mem.indexOf(u8, line, "$sp") == null) continue;
            const name = line[3..std.mem.indexOfScalar(u8, line, '(').?];
            const call = try std.fmt.allocPrint(gpa, "call @{s}(", .{name});
            defer gpa.free(call);
            if (std.mem.indexOf(u8, ir, call) == null) {
                std.debug.print("{s}: spec unit {s} has no caller\n", .{ path, name });
                return error.TestExpectedSpecCall;
            }
            units += 1;
        }
        if (units == 0) {
            std.debug.print("{s} does not specialize\n", .{path});
            return error.TestExpectedSpecialization;
        }
    }
}

/// The `--emit ir` text of `path` (owned).
fn emitIr(gpa: std.mem.Allocator, io: Io, path: []const u8) ![]u8 {
    const bin = try Io.Dir.cwd().realPathFileAlloc(io, "zig-out/bin/toy", gpa);
    defer gpa.free(bin);
    var child = try std.process.spawn(io, .{ .argv = &.{ bin, "--emit", "ir", path }, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const ir = try rdr.interface.allocRemaining(gpa, .limited(1 << 22));
    errdefer gpa.free(ir);
    const term = try child.wait(io);
    if (term != .exited or term.exited != 0) return error.EmitFailed;
    return ir;
}

test "spec_cap.toy has exactly 1 erased + 8 spec units" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    const ir = try emitIr(gpa, io, cases_dir ++ "/spec_cap.toy");
    defer gpa.free(ir);

    var erased: usize = 0;
    var spec: usize = 0;
    var lines = std.mem.splitScalar(u8, ir, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "fn ")) continue;
        if (std.mem.indexOf(u8, line, ".apply$int(") != null) erased += 1;
        if (std.mem.indexOf(u8, line, ".apply$int$sp0$") != null) spec += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), erased);
    try std.testing.expectEqual(@as(usize, 8), spec);
    // The cap keeps the first 8 by name: the last two lambdas take the erased instance.
    try std.testing.expect(std.mem.indexOf(u8, ir, "$sp0$main$l7(") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "$sp0$main$l8") == null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "$sp0$main$l9") == null);
}

test "specialized binaries are identical at -j1 and -j8" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    const tmp = ".toy-test-closures-jn";
    Io.Dir.cwd().deleteTree(io, tmp) catch {};
    defer Io.Dir.cwd().deleteTree(io, tmp) catch {};

    var cases = try specCases(gpa, io);
    defer freeCases(gpa, &cases);
    for (cases.items) |path| {
        const a = try buildBytes(gpa, io, tmp, path, &.{ "--no-cache", "-j1" });
        defer gpa.free(a);
        const b = try buildBytes(gpa, io, tmp, path, &.{ "--no-cache", "-j8" });
        defer gpa.free(b);
        if (!std.mem.eql(u8, a, b)) {
            std.debug.print("{s} differs between -j1 and -j8\n", .{path});
            return error.TestExpectedEqual;
        }
    }
}

test "a _ capture builds the same binary as the lambda it means" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    const tmp = ".toy-test-closures-hole";
    Io.Dir.cwd().deleteTree(io, tmp) catch {};
    defer Io.Dir.cwd().deleteTree(io, tmp) catch {};

    const pre =
        \\struct Acc { n: int }
        \\impl Acc {
        \\    fn plus(self, k: int) -> int { self.n + k }
        \\}
        \\fn add(a: int, b: int) -> int { a + b }
        \\fn apply(f: fn(int) -> int, x: int) -> int { f(x) }
        \\fn main() -> int {
        \\    a := 3
        \\    acc := Acc { n: 4 }
        \\
    ;
    const pairs = [_]struct { capture: []const u8, lambda: []const u8 }{
        .{ .capture = "g := add(1, _)\n    return g(a) + acc.n", .lambda = "g := fn(v: int) { add(1, v) }\n    return g(a) + acc.n" },
        .{ .capture = "return apply(add(a, _), acc.n)", .lambda = "return apply(fn(v) { add(a, v) }, acc.n)" },
        .{ .capture = "p := acc.plus(_)\n    return p(a)", .lambda = "p := fn(v: int) { acc.plus(v) }\n    return p(a)" },
    };
    for (pairs, 0..) |pair, i| for ([_][]const u8{ "-O0", "-O1" }) |opt| {
        const cap_src = try std.fmt.allocPrint(gpa, "{s}    {s}\n}}\n", .{ pre, pair.capture });
        defer gpa.free(cap_src);
        const lam_src = try std.fmt.allocPrint(gpa, "{s}    {s}\n}}\n", .{ pre, pair.lambda });
        defer gpa.free(lam_src);
        const cap_dir = try std.fmt.allocPrint(gpa, tmp ++ "/{d}{s}/cap", .{ i, opt });
        defer gpa.free(cap_dir);
        const lam_dir = try std.fmt.allocPrint(gpa, tmp ++ "/{d}{s}/lam", .{ i, opt });
        defer gpa.free(lam_dir);
        const cap = try harness.compile(gpa, io, cap_dir, cap_src, &.{ "--no-cache", opt });
        defer gpa.free(cap);
        const lam = try harness.compile(gpa, io, lam_dir, lam_src, &.{ "--no-cache", opt });
        defer gpa.free(lam);
        const a = try Io.Dir.cwd().readFileAlloc(io, cap, gpa, .unlimited);
        defer gpa.free(a);
        const b = try Io.Dir.cwd().readFileAlloc(io, lam, gpa, .unlimited);
        defer gpa.free(b);
        if (!std.mem.eql(u8, a, b)) {
            std.debug.print("'{s}' differs from '{s}' at {s}\n", .{ pair.capture, pair.lambda, opt });
            return error.TestExpectedEqual;
        }
    };
}

test "200 nested captures build and run" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    const tmp = ".toy-test-closures-nest";
    Io.Dir.cwd().deleteTree(io, tmp) catch {};
    defer Io.Dir.cwd().deleteTree(io, tmp) catch {};

    // Each `ap(_, f)` is `fn(v) { ap(v, f) }`, a fn(int) -> int, so it is the next level's f.
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try src.appendSlice(gpa, "fn id(x: int) -> int { x }\nfn ap(x: int, f: fn(int) -> int) -> int { f(x) + 1 }\nfn main() -> int {\n    f := ");
    for (0..200) |_| try src.appendSlice(gpa, "ap(_, ");
    try src.appendSlice(gpa, "id");
    for (0..200) |_| try src.append(gpa, ')');
    try src.appendSlice(gpa, "\n    return f(0)\n}\n");
    const prog = try harness.compile(gpa, io, tmp, src.items, &.{ "--no-cache", "-O0", "-j1" });
    defer gpa.free(prog);
    const r = try run(gpa, io, prog);
    defer gpa.free(r.stdout);
    try std.testing.expectEqual(@as(u8, 200), r.exit);
}
