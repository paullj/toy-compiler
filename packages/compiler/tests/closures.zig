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
        const ir = try harness.emitIr(gpa, io, path, &.{});
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

test "spec_cap.toy has exactly 1 erased + 8 spec units" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    const ir = try harness.emitIr(gpa, io, cases_dir ++ "/spec_cap.toy", &.{});
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

    var cases = try specCases(gpa, io);
    defer freeCases(gpa, &cases);
    for (cases.items) |path| {
        const src = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
        defer gpa.free(src);
        for ([_][]const u8{ "-O0", "-O1" }) |opt| {
            errdefer std.debug.print("{s} at {s}\n", .{ path, opt });
            try harness.expectDeterministicBuild(gpa, io, ".toy-test-closures-jn", src, &.{opt}, null);
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

// `Vec.each` and the program's own `each_of` only call `f`, so the lambda's env cannot
// escape the owners: a std method's summary and a user generic fn instance's both count.
test "a capturing lambda passed to Vec.each or to a generic fn that only calls it allocates no env and no box in its caller at -O1" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    const o1 = try harness.emitIr(gpa, io, cases_dir ++ "/escape_each.toy", &.{"-O1"});
    defer gpa.free(o1);
    const o0 = try harness.emitIr(gpa, io, cases_dir ++ "/escape_each.toy", &.{"-O0"});
    defer gpa.free(o0);
    const Owner = struct { name: []const u8, callee: []const u8 };
    for ([_]Owner{
        .{ .name = "main", .callee = "call @std/vec.Vec.each$int$sp" },
        .{ .name = "escape_each.direct_call", .callee = "call @std/vec.Vec.each$int$sp" },
        .{ .name = "escape_each.via_free_fn", .callee = "call @escape_each.each_of$int$sp" },
    }) |o| {
        const owner = o.name;
        const body1 = harness.irBody(o1, owner) orelse return error.TestExpectedOwner;
        try std.testing.expect(std.mem.indexOf(u8, body1, "call @gc_alloc") == null);
        var buf: [64]u8 = undefined;
        try std.testing.expect(std.mem.indexOf(u8, body1, try std.fmt.bufPrint(&buf, ":env${s}$l0", .{owner})) != null);
        try std.testing.expect(std.mem.indexOf(u8, body1, o.callee) != null);
        // -O0 keeps the box and the env cell, so the check can fail.
        const body0 = harness.irBody(o0, owner) orelse return error.TestExpectedOwner;
        try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, body0, "call @gc_alloc"));
    }
}

test "every std method that takes a closure keeps its caller's env in the frame at -O1" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    const o1 = try harness.emitIr(gpa, io, cases_dir ++ "/escape_std_methods.toy", &.{"-O1"});
    defer gpa.free(o1);
    const o0 = try harness.emitIr(gpa, io, cases_dir ++ "/escape_std_methods.toy", &.{"-O0"});
    defer gpa.free(o0);
    const owners = [_][]const u8{
        "escape_std_methods.vec_methods",
        "escape_std_methods.option_methods",
        "escape_std_methods.map_methods",
        "escape_std_methods.set_methods",
    };
    for (owners) |owner| {
        const body1 = harness.irBody(o1, owner) orelse return error.TestExpectedOwner;
        if (std.mem.indexOf(u8, body1, "call @gc_alloc") != null) {
            std.debug.print("{s} allocates at -O1: a std method it calls keeps its closure\n", .{owner});
            return error.TestUnexpectedAlloc;
        }
        var buf: [96]u8 = undefined;
        try std.testing.expect(std.mem.indexOf(u8, body1, try std.fmt.bufPrint(&buf, ":env${s}$l0", .{owner})) != null);
        const body0 = harness.irBody(o0, owner) orelse return error.TestExpectedOwner;
        try std.testing.expect(std.mem.count(u8, body0, "call @gc_alloc") >= 2);
    }
}

/// A `main` whose frame holds an `H.Small`, `H`'s other variant carrying `g_bytes` of
/// nested structs, beside a 240-byte struct a lambda captures: the lambda's env is a frame
/// env at -O1 when it fits. Nothing copies the big payload, so the frame costs no code.
fn nearCapFrame(gpa: std.mem.Allocator, g_bytes: u32) ![]u8 {
    var src: std.Io.Writer.Allocating = .init(gpa);
    errdefer src.deinit();
    const w = &src.writer;
    try w.writeAll("struct A { a0: int, a1: int, a2: int, a3: int, a4: int, a5: int, a6: int, a7: int }\n");
    const nest = [_][2][]const u8{ .{ "B", "A" }, .{ "C", "B" }, .{ "D", "C" }, .{ "E", "D" }, .{ "F", "E" } };
    for (nest) |pair| {
        try w.print("struct {s} {{ ", .{pair[0]});
        for (0..8) |i| try w.print("{s}x{d}: {s}, ", .{ pair[0], i, pair[1] });
        try w.writeAll("}\n");
    }
    // `g_bytes` as at most seven of each nested struct, largest first, then ints.
    const sizes = [_]u32{ 2097152, 262144, 32768, 4096, 512, 64 };
    const names = [_][]const u8{ "F", "E", "D", "C", "B", "A" };
    var left = g_bytes;
    try w.writeAll("struct G { ");
    for (sizes, names) |size, name| {
        const n = @min(7, left / size);
        for (0..n) |i| try w.print("g{s}{d}: {s}, ", .{ name, i, name });
        left -= n * size;
    }
    for (0..left / 8) |i| try w.print("gi{d}: int, ", .{i});
    try w.writeAll("}\nenum H { Small, Huge(G) }\nstruct S { ");
    for (0..30) |i| try w.print("f{d}: int, ", .{i});
    try w.writeAll("}\nfn run(f: fn() -> int) -> int { f() }\nfn main() -> int {\n h := H.Small\n s := S { ");
    for (0..30) |i| try w.print("f{d}: {d}, ", .{ i, i });
    try w.writeAll("}\n r := run(fn() -> int { s.f3 + s.f29 })\n return match h { H.Small -> r, H.Huge(_) -> 0 }\n}\n");
    return src.toOwnedSlice();
}

/// The exit code of `prog` run with a 32 MiB stack, room for a frame near codegen's 16 MiB
/// cap (the default 8 MiB stack faults on the frame's first page past it).
fn runExitBigStack(io: Io, prog: []const u8) !u8 {
    var child = try std.process.spawn(io, .{ .argv = &.{ "/bin/sh", "-c", "ulimit -s 32768 && exec \"$0\"", prog } });
    const term = try child.wait(io);
    return switch (term) {
        .exited => |c| c,
        else => error.ChildCrashed,
    };
}

test "a body whose -O1 frame would overflow with frame envs falls back to heap envs" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    // Since #32 the cap is 16 MiB, so only a frame within an env's size of it reaches the
    // fallback. At this size `main`'s frame fits at -O0 and with a heap env at -O1, but not
    // with the 240-byte frame env. If frames change size, these tripwires fire: retune
    // `g_bytes` so the build still falls back and the lean one still keeps its frame env.
    const g_bytes: u32 = 16776096;
    const tmp = ".toy-test-closures-frame";
    Io.Dir.cwd().deleteTree(io, tmp) catch {};
    defer Io.Dir.cwd().deleteTree(io, tmp) catch {};
    try Io.Dir.cwd().createDirPath(io, tmp);
    const src = try nearCapFrame(gpa, g_bytes);
    defer gpa.free(src);
    const p0 = try harness.compile(gpa, io, tmp ++ "/o0", src, &.{ "--no-cache", "-O0" });
    defer gpa.free(p0);
    const p1 = try harness.compile(gpa, io, tmp ++ "/o1", src, &.{ "--no-cache", "-O1" });
    defer gpa.free(p1);
    try std.testing.expectEqual(@as(u8, 32), try runExitBigStack(io, p0));
    try std.testing.expectEqual(@as(u8, 32), try runExitBigStack(io, p1));
    const path = tmp ++ "/near.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });
    const ir = try harness.emitIr(gpa, io, path, &.{"-O1"});
    defer gpa.free(ir);
    const body = harness.irBody(ir, "main") orelse return error.TestExpectedMain;
    try std.testing.expect(std.mem.indexOf(u8, body, "call @gc_alloc") != null);

    // 120 bytes less, and the frame env fits: the cell above comes from the fallback.
    const lean = try nearCapFrame(gpa, g_bytes - 120);
    defer gpa.free(lean);
    const lean_path = tmp ++ "/lean.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = lean_path, .data = lean });
    const lean_ir = try harness.emitIr(gpa, io, lean_path, &.{"-O1"});
    defer gpa.free(lean_ir);
    const lean_body = harness.irBody(lean_ir, "main") orelse return error.TestExpectedMain;
    try std.testing.expect(std.mem.indexOf(u8, lean_body, "call @gc_alloc") == null);
    try std.testing.expect(std.mem.indexOf(u8, lean_body, ":env$main$l0") != null);
}

test "a capturing lambda passed to Vec.each in a loop triggers no collection at -O1" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    const tmp = ".toy-test-closures-alloc";
    Io.Dir.cwd().deleteTree(io, tmp) catch {};
    defer Io.Dir.cwd().deleteTree(io, tmp) catch {};
    // `forced_collect_count` runs one collection and returns the count, so the difference
    // is 1 exactly when nothing in the loop allocated enough to trigger one.
    const src =
        \\import std/vec
        \\import core/mem_selftest
        \\fn main() -> int {
        \\    xs: Vec[int] = [1, 2, 3]
        \\    a := mem_selftest.forced_collect_count()
        \\    total := 0
        \\    for _ in 0..100000 { xs.each(fn(v: int) { total = total + v }) }
        \\    b := mem_selftest.forced_collect_count()
        \\    if total != 600000 { return 255 }
        \\    return b - a
        \\}
        \\
    ;
    const o1 = try harness.compile(gpa, io, tmp ++ "/o1", src, &.{ "--no-cache", "-O1" });
    defer gpa.free(o1);
    try std.testing.expectEqual(@as(u8, 1), try harness.runExit(gpa, io, o1));
    const o0 = try harness.compile(gpa, io, tmp ++ "/o0", src, &.{ "--no-cache", "-O0" });
    defer gpa.free(o0);
    try std.testing.expect(try harness.runExit(gpa, io, o0) > 1);
}

test "every closures/ and places/ program and modules/escape_summary has the same exit and stdout at -O0, -O1 and -O1 --no-specialize" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);

    const tmp = ".toy-test-closures-diffopt";
    Io.Dir.cwd().deleteTree(io, tmp) catch {};
    defer Io.Dir.cwd().deleteTree(io, tmp) catch {};
    const Entry = struct { path: []u8, lib: ?[]u8 = null };
    var entries: std.ArrayList(Entry) = .empty;
    defer {
        for (entries.items) |e| {
            gpa.free(e.path);
            if (e.lib) |l| gpa.free(l);
        }
        entries.deinit(gpa);
    }
    for ([_][]const u8{ cases_dir, "tests/corpora/language-features/places" }) |dir_path| {
        var dir = try Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".toy")) continue;
            try entries.append(gpa, .{ .path = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir_path, entry.name }) });
        }
    }
    const mod_dir = "tests/corpora/language-features/modules/escape_summary";
    try entries.append(gpa, .{ .path = try gpa.dupe(u8, mod_dir ++ "/main.toy"), .lib = try gpa.dupe(u8, mod_dir ++ "/lib.toy") });
    const levels = [_][]const []const u8{ &.{ "--no-cache", "-O0" }, &.{ "--no-cache", "-O1" }, &.{ "--no-cache", "-O1", "--no-specialize" } };
    for (entries.items, 0..) |e, n| {
        const src = try Io.Dir.cwd().readFileAlloc(io, e.path, gpa, .unlimited);
        defer gpa.free(src);
        var ref: ?Ran = null;
        defer if (ref) |r| gpa.free(r.stdout);
        for (levels, 0..) |flags, li| {
            const d = try std.fmt.allocPrint(gpa, tmp ++ "/{d}/{d}", .{ n, li });
            defer gpa.free(d);
            if (e.lib) |lib_path| {
                const lib = try Io.Dir.cwd().readFileAlloc(io, lib_path, gpa, .unlimited);
                defer gpa.free(lib);
                try Io.Dir.cwd().createDirPath(io, d);
                const dst = try std.fmt.allocPrint(gpa, "{s}/lib.toy", .{d});
                defer gpa.free(dst);
                try Io.Dir.cwd().writeFile(io, .{ .sub_path = dst, .data = lib });
            }
            const p = try harness.compile(gpa, io, d, src, flags);
            defer gpa.free(p);
            const r = try run(gpa, io, p);
            if (ref) |a| {
                defer gpa.free(r.stdout);
                if (a.exit != r.exit or !std.mem.eql(u8, a.stdout, r.stdout)) {
                    std.debug.print("{s}: exit {d} at -O0, {d} at level {d}\n", .{ e.path, a.exit, r.exit, li });
                    return error.TestExpectedEqual;
                }
            } else ref = r;
        }
    }
    try std.testing.expect(entries.items.len >= 60);
}
