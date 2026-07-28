//! Driver end-to-end corpus (integration).
//!
//! Relocated out of `driver/Driver.zig`: these tests drive the whole compilation
//! pipeline (discover → resolve → typecheck → codegen → link), compile-and-RUN real
//! Mach-O binaries, and pin cache soundness / byte-identity across rebuilds. They
//! exercise the driver as a black box through the published `toy_compiler` surface,
//! so they belong in tests/ (not
//! inline in the library) and run in the `toy-integration-test` binary. The bare
//! symbol names below alias the driver's published surface so the moved test bodies
//! stay byte-verbatim.
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const toyc = @import("toy_compiler");

const Driver = toyc.Driver;
const Codegen = toyc.DriverCodegen;
const Ast = toyc.Ast;
const Cache = toyc.Cache;
const Engine = toyc.QueryEngine;
const Resolve = toyc.Resolve;
const Typecheck = toyc.Typecheck;
const Graph = toyc.Graph;
const Opt = toyc.Opt;
const lower = toyc.lower;
const version = toyc.version;
const Mono = toyc.Mono;
const AstWalk = toyc.AstWalk;
const Fingerprint = toyc.Fingerprint;

const cache_root = Driver.cache_root;
const cache_dir_buf_len = Driver.cache_dir_buf_len;
const run = Driver.run;
const pipeline = Driver.pipeline;
const job = Driver.job;
const lowerGraphProgram = Codegen.lowerGraphProgram;
const buildImage = Codegen.buildImage;
const FileResult = Driver.FileResult;
const LinkedProgram = Codegen.LinkedProgram;
const LowerProgramResult = Codegen.LowerProgramResult;

const testing = std.testing;

/// Test-only: lower an already-`.check`ed single-file `FileResult` through the SAME
/// whole-graph codegen orchestration the production `-o` build drives. A lone file IS
/// the trivial one-module graph, so we wrap it in `Graph.single` and feed its existing
/// whole-graph `resolve`/`typecheck` results (graph-of-one == program-wide tables) to
/// `lowerGraphProgram` — there is ONE codegen path, exercised here too. `io == null` to
/// `lowerGraphProgram` would block on the pool; tests pass a real threaded `io`. Caller
/// owns the result on `.ok`.
fn lowerSingleFile(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    target: []const u8,
    r: *const FileResult,
    mode: Engine.Mode,
    opt: Opt.Config,
) !LowerProgramResult {
    var graph = try Graph.single(gpa, "main", r.path, r.source, r.tokens, r.nodes, r.extra, r.pub_bits);
    defer graph.deinit(gpa);
    return lowerGraphProgram(gpa, io, cache, target, &graph, &r.resolve.?, &r.typecheck.?, mode, opt, null, null, 0);
}

test "cold then warm parse: 2nd run hits cache and renders identically" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // Write a source file into a unique temp dir so we don't disturb the repo.
    const dir_name = ".toy-test-driver";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // A source unlikely to collide with any real run's cache entry, so the
    // first pipeline pass is genuinely a cold miss.
    const src = "fn drv_cold_warm_zzq(a: int, b: int) -> int {\n return a + b\n}\n";
    const path = dir_name ++ "/p.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    // Ensure a genuinely cold start: drop any prior parse entry for this source.
    {
        const key = Cache.Key.fromSource(.parse, "native", src);
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        const epath = std.fmt.bufPrint(&pbuf, "{s}/{x:0>16}", .{ dir, key.digest() }) catch unreachable;
        Io.Dir.cwd().deleteFile(io, epath) catch {};
    }

    var first: FileResult = .{ .path = path };
    try pipeline(gpa, io, cache, .parse, "native", &first, 0);
    defer first.deinit(gpa);
    try testing.expect(!first.nodes_cached);

    var second: FileResult = .{ .path = path };
    try pipeline(gpa, io, cache, .parse, "native", &second, 1);
    defer second.deinit(gpa);
    try testing.expect(second.nodes_cached);

    // Both renders must match.
    var b1: [256]u8 = undefined;
    var w1 = std.Io.Writer.fixed(&b1);
    try Ast.render(&w1, .{ .nodes = first.nodes, .extra = first.extra }, first.tokens, first.source);
    var b2: [256]u8 = undefined;
    var w2 = std.Io.Writer.fixed(&b2);
    try Ast.render(&w2, .{ .nodes = second.nodes, .extra = second.extra }, second.tokens, second.source);
    try testing.expectEqualStrings(w1.buffered(), w2.buffered());
}

test "emit=check on a clean program resolves with no diagnostics" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-driver-check-ok";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "pub fn add(a: int, b: int) -> int {\n return a + b\n}\n";
    const path = dir_name ++ "/p.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    var r: FileResult = .{ .path = path };
    try pipeline(gpa, io, cache, .check, "native", &r, 0);
    defer r.deinit(gpa);

    try testing.expect(r.checked);
    try testing.expect(r.err == null);
    try testing.expect(r.resolve != null);
    try testing.expectEqual(@as(usize, 0), r.resolve.?.diags.len);
}

test "emit=check on a bad program reports a resolve error" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-driver-check-bad";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn f() {\n y := undefined_name\n return\n}\n";
    const path = dir_name ++ "/p.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    var r: FileResult = .{ .path = path };
    try pipeline(gpa, io, cache, .check, "native", &r, 0);
    defer r.deinit(gpa);

    try testing.expect(r.checked);
    try testing.expectEqual(@as(?anyerror, error.ResolveError), r.err);
    try testing.expect(r.resolve != null);
    try testing.expect(r.resolve.?.diags.len > 0);
}

test "a many-error file collects the FULL uncapped diagnostic set (render cap is output-only)" {
    // The render-time cap (DiagRender.DIAG_CAP = 100) never truncates the COLLECTED /
    // returned diagnostics — the Sink/Result stays complete so incremental/cache
    // fingerprints stay stable. Build a file with 120 distinct undeclared names and
    // assert all 120 diagnostics are retained on the resolve result.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-driver-cap";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // 120 statements each referencing a distinct undeclared name `uNNN`.
    const n_errs = 120;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try src.appendSlice(gpa, "fn _f() {\n");
    for (0..n_errs) |i| {
        var line: [32]u8 = undefined;
        try src.appendSlice(gpa, std.fmt.bufPrint(&line, " _x{d} := undecl{d}\n", .{ i, i }) catch unreachable);
    }
    try src.appendSlice(gpa, " return\n}\n");

    const path = dir_name ++ "/many.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src.items });

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    var r: FileResult = .{ .path = path };
    try pipeline(gpa, io, cache, .check, "native", &r, 0);
    defer r.deinit(gpa);

    try testing.expectEqual(@as(?anyerror, error.ResolveError), r.err);
    try testing.expect(r.resolve != null);
    // The FULL set is collected — NOT capped to DIAG_CAP (100).
    try testing.expectEqual(@as(usize, n_errs), r.resolve.?.diags.len);
}

test "`toy check` on a many-error file caps output at DIAG_CAP primaries + a summary line" {
    // End-to-end render check via the built `toy` binary (the only path that exercises
    // DiagRender). Migrated from the removed `--emit check` inspection table to the
    // `toy check` subcommand. Skips gracefully if the binary isn't present. DIAG_CAP = 100
    // primary `-->` carets are drawn, then one `... and N more` line; the summary count
    // is total - 100.
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const toy_bin = "zig-out/bin/toy";
    Io.Dir.cwd().access(io, toy_bin, .{}) catch return error.SkipZigTest;

    const dir_name = ".toy-test-driver-cap-render";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const n_errs = 120;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try src.appendSlice(gpa, "fn _f() {\n");
    for (0..n_errs) |i| {
        var line: [32]u8 = undefined;
        try src.appendSlice(gpa, std.fmt.bufPrint(&line, " _x{d} := undecl{d}\n", .{ i, i }) catch unreachable);
    }
    try src.appendSlice(gpa, " return\n}\n");
    const path = dir_name ++ "/many.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src.items });

    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, toy_bin, gpa);
    defer gpa.free(bin_abs);

    var child = try std.process.spawn(io, .{ .argv = &.{ bin_abs, "check", path }, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 20));
    defer gpa.free(got);
    _ = try child.wait(io);

    // Count `-->` primaries: exactly DIAG_CAP (100) rendered.
    var carets: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, got, i, "-->")) |at| {
        carets += 1;
        i = at + 3;
    }
    try testing.expectEqual(@as(usize, 100), carets);
    // The trailing "... and 20 more" summary line is present.
    try testing.expect(std.mem.indexOf(u8, got, "... and 20 more") != null);
}

test "explain: a known code prints its doc (exit 0); an unknown code arg-errors (exit 2)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const toy_bin = "zig-out/bin/toy";
    Io.Dir.cwd().access(io, toy_bin, .{}) catch return error.SkipZigTest;
    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, toy_bin, gpa);
    defer gpa.free(bin_abs);

    // Known code: prints the doc, exit 0.
    {
        var child = try std.process.spawn(io, .{ .argv = &.{ bin_abs, "explain", "R0001" }, .stdout = .pipe });
        var rdr = child.stdout.?.readerStreaming(io, &.{});
        const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
        defer gpa.free(got);
        const term = try child.wait(io);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, term);
        try testing.expect(std.mem.indexOf(u8, got, "R0001") != null);
        try testing.expect(std.mem.indexOf(u8, got, "undeclared identifier") != null);
    }
    // Unknown code: an argument error, exit 2.
    {
        var child = try std.process.spawn(io, .{ .argv = &.{ bin_abs, "explain", "BOGUS" }, .stdout = .pipe });
        var rdr = child.stdout.?.readerStreaming(io, &.{});
        const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
        defer gpa.free(got);
        const term = try child.wait(io);
        try testing.expectEqual(std.process.Child.Term{ .exited = 2 }, term);
        try testing.expect(std.mem.indexOf(u8, got, "unknown diagnostic code") != null);
    }
}

test "explain --list enumerates EVERY registered code (and `explain list` works too)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const toy_bin = "zig-out/bin/toy";
    Io.Dir.cwd().access(io, toy_bin, .{}) catch return error.SkipZigTest;
    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, toy_bin, gpa);
    defer gpa.free(bin_abs);

    // `--list`: exit 0; band headers; and — the drift guard — EVERY registry code +
    // its kebab title appears, so a newly-added code shows up automatically.
    {
        var child = try std.process.spawn(io, .{ .argv = &.{ bin_abs, "explain", "--list" }, .stdout = .pipe });
        var rdr = child.stdout.?.readerStreaming(io, &.{});
        const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
        defer gpa.free(got);
        const term = try child.wait(io);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, term);
        try testing.expect(std.mem.indexOf(u8, got, "Parser") != null);
        try testing.expect(std.mem.indexOf(u8, got, "Name resolution") != null);
        try testing.expect(std.mem.indexOf(u8, got, "Type checking") != null);
        for (toyc.diagnostics.codes.table) |e| {
            try testing.expect(std.mem.indexOf(u8, got, e.str) != null);
            try testing.expect(std.mem.indexOf(u8, got, e.slug) != null);
        }
    }
    // The bare word `list` is an alias for `--list`.
    {
        var child = try std.process.spawn(io, .{ .argv = &.{ bin_abs, "explain", "list" }, .stdout = .pipe });
        var rdr = child.stdout.?.readerStreaming(io, &.{});
        const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
        defer gpa.free(got);
        const term = try child.wait(io);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, term);
        try testing.expect(std.mem.indexOf(u8, got, "R0002") != null);
    }
}

test "coded render: `return nope` renders `error[R0001]:` and stays report-once (one -->)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const toy_bin = "zig-out/bin/toy";
    Io.Dir.cwd().access(io, toy_bin, .{}) catch return error.SkipZigTest;

    const dir_name = ".toy-test-driver-coded";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const path = dir_name ++ "/one.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "fn main() -> int {\n  return nope\n}\n" });

    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, toy_bin, gpa);
    defer gpa.free(bin_abs);

    // Migrated from `--emit check` (removed inspection table) to the `toy check` subcommand.
    var child = try std.process.spawn(io, .{ .argv = &.{ bin_abs, "check", path }, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    defer gpa.free(got);
    _ = try child.wait(io);

    // The authorized output change: the coded header bracket.
    try testing.expect(std.mem.indexOf(u8, got, "error[R0001]:") != null);
    // Single-file check's `-->` header must name the ON-DISK path, byte-for-byte as the
    // pre-graph-fix single-file path did — NOT the module stem. Regression lock for the
    // routing fix: `check` now discovers a graph-of-one, and the multi-module renderer
    // would spell the entry as its bare stem (`one`, from `Module.path`) instead of the
    // input path (`.../one.toy`, `Module.file`). Single-file check must stay
    // byte-identical, so the header carries the full path here.
    try testing.expect(std.mem.indexOf(u8, got, "--> " ++ path ++ ":2:10") != null);
    // report-once preserved: exactly one primary caret line.
    var carets: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, got, i, "-->")) |at| {
        carets += 1;
        i = at + 3;
    }
    try testing.expectEqual(@as(usize, 1), carets);
}

test "explain: a parse code (P0001) prints its doc (exit 0)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const toy_bin = "zig-out/bin/toy";
    Io.Dir.cwd().access(io, toy_bin, .{}) catch return error.SkipZigTest;
    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, toy_bin, gpa);
    defer gpa.free(bin_abs);

    var child = try std.process.spawn(io, .{ .argv = &.{ bin_abs, "explain", "P0001" }, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    defer gpa.free(got);
    const term = try child.wait(io);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, term);
    try testing.expect(std.mem.indexOf(u8, got, "P0001") != null);
}

test "coded render: a parse error renders `error[P0002]:` and stays report-once (one -->)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const toy_bin = "zig-out/bin/toy";
    Io.Dir.cwd().access(io, toy_bin, .{}) catch return error.SkipZigTest;

    const dir_name = ".toy-test-driver-parse-coded";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const path = dir_name ++ "/one.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "fn a() -> int {\n  return )\n}\n" });

    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, toy_bin, gpa);
    defer gpa.free(bin_abs);

    // Migrated from `--emit check` (removed inspection table) to the `toy check` subcommand.
    var child = try std.process.spawn(io, .{ .argv = &.{ bin_abs, "check", path }, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    defer gpa.free(got);
    _ = try child.wait(io);

    // The authorized C output change: coded parse-diagnostic header bracket.
    try testing.expect(std.mem.indexOf(u8, got, "error[P0002]:") != null);
    // report-once preserved: exactly one primary caret line.
    try testing.expectEqual(@as(usize, 1), countCarets(got));
}

/// Spawn the built `toy` binary with `argv_tail` (already including the entry path),
/// capturing stdout and the exit term. Skips when the binary is absent. Used by the
/// multi-module `check` regression below, which writes several files under `dir_name`
/// then drives `check` over the entry.
fn spawnToy(gpa: std.mem.Allocator, io: Io, argv_tail: []const []const u8) !struct { out: []u8, term: std.process.Child.Term } {
    const toy_bin = "zig-out/bin/toy";
    Io.Dir.cwd().access(io, toy_bin, .{}) catch return error.SkipZigTest;
    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, toy_bin, gpa);
    defer gpa.free(bin_abs);
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.append(gpa, bin_abs);
    for (argv_tail) |a| try argv.append(gpa, a);
    var child = try std.process.spawn(io, .{ .argv = argv.items, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    const term = try child.wait(io);
    return .{ .out = got, .term = term };
}

test "check follows imports: a VALID multi-module program reports zero diagnostics (agrees with build), a REAL import error is reported at the right module" {
    // Regression for the fixed soundness bug: `toy check <entry>` used to fan out over
    // the entry as an INDEPENDENT single file (no import discovery), so a valid import
    // spuriously reported `R0003 unknown imported module` + `R0001 undeclared identifier`
    // that `build` did not. The fix routes `check` through the SAME graph front-end
    // `build` uses (discover -> resolve -> typecheck), so imports ARE followed. This test
    // asserts (a) check + build AGREE on a valid two-module program (both exit 0, zero
    // diagnostics) and (b) a real error in the IMPORTED module is reported against THAT
    // module and exits 1 (matching build).
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;

    const dir_name = ".toy-test-driver-check-multimod";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const main_path = dir_name ++ "/main.toy";
    const helper_path = dir_name ++ "/helper.toy";
    // A valid program: main imports helper and calls its pub fn.
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = main_path, .data = "import helper\nfn main() -> int {\n  return helper.answer()\n}\n" });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = helper_path, .data = "pub fn answer() -> int {\n  return 42\n}\n" });

    // (a) VALID: check reports zero diagnostics and exits 0 — no spurious R0003/R0001.
    {
        const res = try spawnToy(gpa, io, &.{ "check", main_path });
        defer gpa.free(res.out);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
        try testing.expectEqual(@as(usize, 0), countCarets(res.out)); // no diagnostics rendered
        try testing.expect(std.mem.indexOf(u8, res.out, "R0003") == null);
        try testing.expect(std.mem.indexOf(u8, res.out, "R0001") == null);
    }
    // (a') DIFFERENTIAL: build agrees — the same valid program compiles + exits 0.
    {
        const out_bin = dir_name ++ "/prog";
        const res = try spawnToy(gpa, io, &.{ "build", main_path, "-o", out_bin });
        defer gpa.free(res.out);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
    }

    // (b) A REAL error in the IMPORTED module: helper references an undeclared name. Check
    // must report it against helper (its owning module) and exit 1 — build fails too.
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = helper_path, .data = "pub fn answer() -> int {\n  return nope_zzq\n}\n" });
    {
        const res = try spawnToy(gpa, io, &.{ "check", main_path });
        defer gpa.free(res.out);
        try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
        // The diagnostic is coded R0001 and its location names the IMPORTED module.
        try testing.expect(std.mem.indexOf(u8, res.out, "error[R0001]:") != null);
        try testing.expect(std.mem.indexOf(u8, res.out, "nope_zzq") != null);
        try testing.expect(std.mem.indexOf(u8, res.out, "helper:2") != null);
        try testing.expectEqual(@as(usize, 1), countCarets(res.out));
    }
    // (b') DIFFERENTIAL: build fails on the same broken import too (exit != 0).
    {
        const out_bin = dir_name ++ "/prog2";
        const res = try spawnToy(gpa, io, &.{ "build", main_path, "-o", out_bin });
        defer gpa.free(res.out);
        try testing.expect(res.term != .exited or res.term.exited != 0);
    }
}

test "io.print(char) emits byte-exact UTF-8 for 1/2/3/4-byte codepoints (no extra bytes)" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;

    const dir_name = ".toy-test-m10-charutf8";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const out_bin = dir_name ++ "/prog";
    {
        const res = try spawnToy(gpa, io, &.{ "build", "examples/io/char_utf8.toy", "-o", out_bin });
        defer gpa.free(res.out);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
    }

    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, out_bin, gpa);
    defer gpa.free(bin_abs);
    var child = try std.process.spawn(io, .{ .argv = &.{bin_abs}, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    defer gpa.free(got);
    const term = try child.wait(io);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, term);
    // A(41) ñ(C3 B1) €(E2 82 AC) 😀(F0 9F 98 80) = EXACTLY 10 bytes, no trailing newline.
    try testing.expectEqualSlices(u8, "\x41\xC3\xB1\xE2\x82\xAC\xF0\x9F\x98\x80", got);
}

test "io.print(int) renders decimal for 0 / negative / i64 max / i64 MIN (never-negate loop); io.print(str) separates" {
    // BEHAVIORAL coverage for the two hand-asm builtins whose unit tests only assert
    // instruction SHAPE: __int_to_str's i64::MIN-safe digit loop (it must NOT negate the
    // running value) and the str write. i64::MIN is the critical case — negating it
    // overflows. Asserts exact stdout bytes.
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;

    const dir_name = ".toy-test-display-int-edges";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const main_path = dir_name ++ "/main.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = main_path, .data =
        \\import std/io
        \\fn main() {
        \\    io.print(0)
        \\    io.print("\n")
        \\    io.print(-1)
        \\    io.print("\n")
        \\    io.print(9223372036854775807)
        \\    io.print("\n")
        \\    io.print(-9223372036854775808)
        \\    io.print("\n")
        \\}
        \\
    });

    const out_bin = dir_name ++ "/prog";
    {
        const res = try spawnToy(gpa, io, &.{ "build", main_path, "-o", out_bin });
        defer gpa.free(res.out);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
    }

    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, out_bin, gpa);
    defer gpa.free(bin_abs);
    var child = try std.process.spawn(io, .{ .argv = &.{bin_abs}, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    defer gpa.free(got);
    const term = try child.wait(io);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, term);
    try testing.expectEqualSlices(u8, "0\n-1\n9223372036854775807\n-9223372036854775808\n", got);
}

test "char Display is a shared witness — >6 boundary codepoints compile + byte-exact UTF-8" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;

    const dir_name = ".toy-test-c4-charbands";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // 7 char displays (> the pre-fix frame-overflow threshold of 6) spanning every UTF-8
    // length band and BOTH edges of each: U+7F|U+80, U+7FF|U+800, U+FFFF|U+10000, U+10FFFF.
    const main_path = dir_name ++ "/main.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = main_path, .data =
        \\import std/io
        \\fn main() {
        \\    io.print('\u{7F}')
        \\    io.print('\u{80}')
        \\    io.print('\u{7FF}')
        \\    io.print('\u{800}')
        \\    io.print('\u{FFFF}')
        \\    io.print('\u{10000}')
        \\    io.print('\u{10FFFF}')
        \\}
        \\
    });

    const out_bin = dir_name ++ "/prog";
    {
        const res = try spawnToy(gpa, io, &.{ "build", main_path, "-o", out_bin });
        defer gpa.free(res.out);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term); // pre-fix: CodegenDiagnostic
    }

    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, out_bin, gpa);
    defer gpa.free(bin_abs);
    var child = try std.process.spawn(io, .{ .argv = &.{bin_abs}, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    defer gpa.free(got);
    const term = try child.wait(io);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, term);
    // 7F | C2 80 | DF BF | E0 A0 80 | EF BF BF | F0 90 80 80 | F4 8F BF BF  = 19 bytes exactly.
    try testing.expectEqualSlices(u8, "\x7F\xC2\x80\xDF\xBF\xE0\xA0\x80\xEF\xBF\xBF\xF0\x90\x80\x80\xF4\x8F\xBF\xBF", got);
}

test "int->char try_into is a shared witness — 12 in one fn compile (pre-fix frame overflow)" {
    // 12 int->char `try_into` in ONE fn — >10, the pre-fix inline (~24 SSA cells/site)
    // overflowed the imm12 frame cap (error.CodegenDiagnostic). Each is now one CALL to
    // the shared `TryInto$int_to_char` witness (O(1)/site), so the fn compiles. Each result
    // is consumed via a char-typed match (pins int->char) then char->int `.into()`; the sum
    // of codepoints 65..76 is 846, so a correct run exits 42.
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-c4-icframe";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn f() -> int {
        \\  c0: char = match (65).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c1: char = match (66).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c2: char = match (67).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c3: char = match (68).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c4: char = match (69).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c5: char = match (70).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c6: char = match (71).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c7: char = match (72).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c8: char = match (73).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c9: char = match (74).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c10: char = match (75).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  c11: char = match (76).try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  n0: int = c0.into()
        \\  n1: int = c1.into()
        \\  n2: int = c2.into()
        \\  n3: int = c3.into()
        \\  n4: int = c4.into()
        \\  n5: int = c5.into()
        \\  n6: int = c6.into()
        \\  n7: int = c7.into()
        \\  n8: int = c8.into()
        \\  n9: int = c9.into()
        \\  n10: int = c10.into()
        \\  n11: int = c11.into()
        \\  return n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11
        \\}
        \\fn main() -> int {
        \\  if f() == 846 { return 42 }
        \\  return 1
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term); // pre-fix: CodegenDiagnostic
}

test "char->byte try_into is a shared witness — 22 in one fn compile (pre-fix frame overflow)" {
    // 22 char->byte `try_into` in ONE fn — >20, the pre-fix inline overflowed the frame.
    // Each is now one CALL to the shared `TryInto$char_to_byte` witness. Codepoints 65..86
    // all fit `byte`, so every `unwrap_or(0)` yields the codepoint; the two edge checks
    // (b0, b21) exit non-42 on any miscompile, else 42.
    // 22 is the smallest count that clears the pre-fix ~21-site overflow; the spill-all frame
    // now overflows at 25, so this test sits ~2 sites under the cap. A per-site frame-cost bump
    // (wider ret_slot, an extra spill) could regress it — and raising N cannot buy more margin,
    // since frame overflow at high site counts is inherent to spill-all (plain local-heavy code
    // overflows at ~90 locals), not to this witness.
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-c4-cbframe";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn g() -> int {
        \\  b0: byte = 'A'.try_into().unwrap_or(0)
        \\  b1: byte = 'B'.try_into().unwrap_or(0)
        \\  b2: byte = 'C'.try_into().unwrap_or(0)
        \\  b3: byte = 'D'.try_into().unwrap_or(0)
        \\  b4: byte = 'E'.try_into().unwrap_or(0)
        \\  b5: byte = 'F'.try_into().unwrap_or(0)
        \\  b6: byte = 'G'.try_into().unwrap_or(0)
        \\  b7: byte = 'H'.try_into().unwrap_or(0)
        \\  b8: byte = 'I'.try_into().unwrap_or(0)
        \\  b9: byte = 'J'.try_into().unwrap_or(0)
        \\  b10: byte = 'K'.try_into().unwrap_or(0)
        \\  b11: byte = 'L'.try_into().unwrap_or(0)
        \\  b12: byte = 'M'.try_into().unwrap_or(0)
        \\  b13: byte = 'N'.try_into().unwrap_or(0)
        \\  b14: byte = 'O'.try_into().unwrap_or(0)
        \\  b15: byte = 'P'.try_into().unwrap_or(0)
        \\  b16: byte = 'Q'.try_into().unwrap_or(0)
        \\  b17: byte = 'R'.try_into().unwrap_or(0)
        \\  b18: byte = 'S'.try_into().unwrap_or(0)
        \\  b19: byte = 'T'.try_into().unwrap_or(0)
        \\  b20: byte = 'U'.try_into().unwrap_or(0)
        \\  b21: byte = 'V'.try_into().unwrap_or(0)
        \\  if b0 != 65 { return 1 }
        \\  if b21 != 86 { return 2 }
        \\  return 42
        \\}
        \\fn main() -> int { return g() }
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term);
}

test "int->char witness — surrogate/range boundaries are byte-exact through the CALL" {
    // The witness's `validScalarValue` predicate must reproduce the inline verdicts exactly:
    // valid scalars Ok (their codepoint), surrogates + >0x10FFFF Err (the '?'=63 sentinel).
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-c4-icbounds";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn ci(c: char) -> int { return c.into() }
        \\fn ic(n: int) -> int {
        \\  c: char = match n.try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  return ci(c)
        \\}
        \\fn main() -> int {
        \\  if ic(0) != 0 { return 1 }
        \\  if ic(0xD7FF) != 55295 { return 2 }
        \\  if ic(0xD800) != 63 { return 3 }
        \\  if ic(0xDFFF) != 63 { return 4 }
        \\  if ic(0xE000) != 57344 { return 5 }
        \\  if ic(0x10FFFF) != 1114111 { return 6 }
        \\  if ic(0x110000) != 63 { return 7 }
        \\  return 42
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term);
}

test "char->byte witness — 0xFF fits, 0x100 does not (byte-exact through the CALL)" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-c4-cbbounds";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn main() -> int {
        \\  ok255: byte = '\u{FF}'.try_into().unwrap_or(0)
        \\  if ok255 != 255 { return 1 }
        \\  err256: byte = '\u{100}'.try_into().unwrap_or(0)
        \\  if err256 != 0 { return 2 }
        \\  return 42
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term);
}

test "a uint-source int->char passes the raw value to the witness (no-widen arg path)" {
    // Locks that the call site hands the witness the receiver value RAW (no width coercion):
    // a `uint`-typed source converts to the correct char just like an `int` source.
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-c4-icnonint";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn ci(c: char) -> int { return c.into() }
        \\fn main() -> int {
        \\  u: uint = 0x41
        \\  c: char = match u.try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  if ci(c) != 65 { return 1 }
        \\  u2: uint = 0x20AC
        \\  c2: char = match u2.try_into() { .ok(c) -> c, .err(_) -> '?' }
        \\  if ci(c2) != 8364 { return 2 }
        \\  return 42
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term);
}

test "float->int try_into is a shared witness — 12 in one fn compile (pre-fix frame overflow)" {
    // 12 float->int `try_into` in ONE fn — >10; the pre-fix inline would overflow the imm12
    // frame cap. Each is now one CALL to the shared `TryInto$float_to_int` witness (O(1)/site),
    // so the fn compiles. The 12 floats 0.0..11.0 all convert; the sum is 66, so a correct run exits 42.
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-c5-fiframe";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn f() -> int {
        \\  a0: int = (0.0).try_into().unwrap_or(0)
        \\  a1: int = (1.0).try_into().unwrap_or(0)
        \\  a2: int = (2.0).try_into().unwrap_or(0)
        \\  a3: int = (3.0).try_into().unwrap_or(0)
        \\  a4: int = (4.0).try_into().unwrap_or(0)
        \\  a5: int = (5.0).try_into().unwrap_or(0)
        \\  a6: int = (6.0).try_into().unwrap_or(0)
        \\  a7: int = (7.0).try_into().unwrap_or(0)
        \\  a8: int = (8.0).try_into().unwrap_or(0)
        \\  a9: int = (9.0).try_into().unwrap_or(0)
        \\  a10: int = (10.0).try_into().unwrap_or(0)
        \\  a11: int = (11.0).try_into().unwrap_or(0)
        \\  return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11
        \\}
        \\fn main() -> int {
        \\  if f() == 66 { return 42 }
        \\  return 1
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term); // pre-fix: CodegenDiagnostic
}

test "float->int witness — NaN/+-inf/out-of-range Err, 2^63 boundary is byte-exact through the CALL" {
    // The fcmp range guard must reproduce the fallible-narrow verdicts: NaN and +-inf and
    // |f|>=2^63 -> Err (unwrap_or(-1) sentinel); the largest f64 < 2^63 -> Ok. The exactly-
    // representable 2^63 is the silent-saturation gate: fcvtzs alone would saturate it to
    // i64_max, but the strict `< 2^63` guard makes it Err.
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-c5-fibounds";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn main() -> int {
        \\  nan: int = (0.0 /. 0.0).try_into().unwrap_or(-1)
        \\  pinf: int = (1.0 /. 0.0).try_into().unwrap_or(-1)
        \\  huge: int = (1e30).try_into().unwrap_or(-1)
        \\  nhuge: int = (0.0 -. 1e30).try_into().unwrap_or(-1)
        \\  at63: int = (9223372036854775808.0).try_into().unwrap_or(-1)
        \\  below: int = (9223372036854774784.0).try_into().unwrap()
        \\  if nan != -1 { return 1 }
        \\  if pinf != -1 { return 2 }
        \\  if huge != -1 { return 3 }
        \\  if nhuge != -1 { return 4 }
        \\  if at63 != -1 { return 5 }
        \\  if below != 9223372036854774784 { return 6 }
        \\  return 42
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term);
}

test "an Option[int] find/match program compiles + runs; exit is the unwrapped payload" {
    // The prelude `Option[T]` is nameable with no import; `Option[int]` reifies to a
    // plain concrete enum through the reification path, so this compiles + runs on the real backend.
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-m23-option";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn find(n: int) -> Option[int] {
        \\  if n > 0 { return Option.some(n) }
        \\  return Option[int].none
        \\}
        \\fn main() -> int {
        \\  return match find(42) { .some(v) -> v, .none -> 0 }
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term);
}

test "a Result[int,str] construct+match program compiles + runs; exit is the ok payload" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-m23-result";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn checked_div(a: int, b: int) -> Result[int, str] {
        \\  if b == 0 { return Result[int, str].err("divide by zero") }
        \\  return Result.ok(a / b)
        \\}
        \\fn main() -> int {
        \\  return match checked_div(84, 2) { .ok(v) -> v, .err(_) -> 1 }
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term);
}

test "a bare `Option.none` with no inferable target reports T0016" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-m23-neg";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src = "fn main() -> int {\n  x := Option.none\n  return 0\n}\n";
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"check"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
    try testing.expect(std.mem.indexOf(u8, res.out, "T0016") != null);
}

test "coherence: distinct int widths (int8 vs uint8) conform to the same protocol without a T0020 collision" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-coherence-int-widths";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\protocol Foo { fn foo(self) -> int }
        \\impl int8 has Foo { fn foo(self) -> int { 1 } }
        \\impl uint8 has Foo { fn foo(self) -> int { 2 } }
        \\fn main() -> int { 0 }
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"check"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
    try testing.expect(std.mem.indexOf(u8, res.out, "T0020") == null);
}

test "coherence: two identical int8 impls DO still collide (rejected)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-coherence-dup-int8";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    // Two impls of the SAME (int8, Foo) are still illegal — here the duplicate-method
    // check (R0002) fires before the coherence barrier, but either way it is rejected.
    const src =
        \\protocol Foo { fn foo(self) -> int }
        \\impl int8 has Foo { fn foo(self) -> int { 1 } }
        \\impl int8 has Foo { fn foo(self) -> int { 2 } }
        \\fn main() -> int { 0 }
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"check"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
    try testing.expect(std.mem.indexOf(u8, res.out, "int8") != null);
}

test "literals: the most-negative signed literal (int8 -128, int INT_MIN) is accepted" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-int-min-literal";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn main() -> int {
        \\  a: int8 = -128
        \\  b: int = -9223372036854775808
        \\  return 0
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"check"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
    try testing.expect(std.mem.indexOf(u8, res.out, "T0034") == null);
}

test "literals: an unannotated literal past 2^64-1 reports T0034 (not a codegen note)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-int-unannotated-overflow";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src = "fn main() -> int {\n  x := 18446744073709551616\n  return 0\n}\n";
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"check"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
    try testing.expect(std.mem.indexOf(u8, res.out, "T0034") != null);
    // The verdict now lives at the checker; the old codegen-time note is gone.
    try testing.expect(std.mem.indexOf(u8, res.out, "out of range for codegen") == null);
}

test "literals: one past the signed min (int8 -129) still reports T0034" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-int-min-overflow";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src = "fn main() -> int {\n  a: int8 = -129\n  return 0\n}\n";
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"check"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
    try testing.expect(std.mem.indexOf(u8, res.out, "T0034") != null);
}

test "match: an out-of-range literal pattern on a narrow-int scrutinee reports T0034 once" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-match-narrow-oor";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src = "fn main() -> int {\n  x: int8 = 0\n  return match x { 300 -> 1, _ -> 2 }\n}\n";
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"check"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
    try testing.expect(std.mem.indexOf(u8, res.out, "T0034") != null);
    // Single diagnostic — the width-adopt keeps the assignability mismatch silent.
    try testing.expect(std.mem.indexOf(u8, res.out, "does not match scrutinee") == null);
}

test "an Option is_some/unwrap_or program compiles + runs; exit is the guarded payload" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-m23-option-methods";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn main() -> int {
        \\  a := Option[int].some(40)
        \\  base := if a.is_some() { a.unwrap_or(0) } else { 0 }
        \\  b := Option[int].none
        \\  return base + b.unwrap_or(2)
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term);
}

test "a Result is_ok/unwrap_or program compiles + runs; exit is the ok payload" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-m23-result-methods";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    const src =
        \\fn main() -> int {
        \\  r := Result[int, str].ok(40)
        \\  base := if r.is_ok() { r.unwrap_or(0) } else { 99 }
        \\  e := Result[int, str].err("boom")
        \\  return base + e.unwrap_or(2)
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, res.term);
}

test "unwrap on an aggregate payload compiles and runs (result-slot copy, not a crash)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-agg-unwrap";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    // Aggregate-payload `unwrap_or` once PANICKED codegen (unassigned join block-arg
    // offset) then was deferred with a T0018 reject; it now routes the struct payload
    // through a result slot + copy, so it compiles and runs — `.none.unwrap_or(P{x:2})`
    // takes the default, returning 2.
    const src =
        \\struct P { x: int }
        \\fn main() -> int {
        \\  d := P { x: 2 }
        \\  p := Option[P].none.unwrap_or(d)
        \\  return p.x
        \\}
        \\
    ;
    const res = try runToyOnFixture(gpa, io, dir_name, src, &.{"run"});
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 2 }, res.term);
}

/// Spawn `toy` with `args` over a one-file fixture (the `return nope` R0001 program),
/// capturing stdout+stderr merged and the exit code. Returns the captured bytes (the
/// caller frees) and the term. Skips when the built binary is absent.
fn runToyOnFixture(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, src: []const u8, args: []const []const u8) !struct { out: []u8, term: std.process.Child.Term } {
    const toy_bin = "zig-out/bin/toy";
    Io.Dir.cwd().access(io, toy_bin, .{}) catch return error.SkipZigTest;
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    var path_buf: [256]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/one.toy", .{dir_name});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });
    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, toy_bin, gpa);
    defer gpa.free(bin_abs);

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.append(gpa, bin_abs);
    for (args) |a| try argv.append(gpa, a);
    try argv.append(gpa, path);

    var child = try std.process.spawn(io, .{ .argv = argv.items, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    const term = try child.wait(io);
    return .{ .out = got, .term = term };
}

const c3_fixture = "fn main() -> int {\n  return nope\n}\n";

fn countCarets(got: []const u8) usize {
    var carets: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, got, i, "-->")) |at| {
        carets += 1;
        i = at + 3;
    }
    return carets;
}

test "--warn downgrades an error to a warning (render-only, one -->)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-c3-warn";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // Migrated from `--emit check` (removed inspection table) to the `toy check` subcommand.
    const res = runToyOnFixture(gpa, io, dir_name, c3_fixture, &.{ "check", "--warn", "R0001" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);

    try testing.expect(std.mem.indexOf(u8, res.out, "warning[R0001]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "error[R0001]:") == null);
    // report-once preserved: exactly one primary caret line.
    try testing.expectEqual(@as(usize, 1), countCarets(res.out));
    // `toy check`'s summary counts the downgraded diagnostic as a warning; no errors
    // survive so the exit is 0 (the removed inspection table had no such gate).
    try testing.expect(std.mem.indexOf(u8, res.out, "1 warning(s)") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check` --ignore suppresses a code entirely (zero diagnostic bytes; no errors survive => exit 0)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-c3-ignore";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // Migrated from `--emit check` (removed inspection table) to the `toy check` subcommand.
    const res = runToyOnFixture(gpa, io, dir_name, c3_fixture, &.{ "check", "--allow", "R0001" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);

    // The ignored diagnostic renders zero bytes: no code, no caret.
    try testing.expect(std.mem.indexOf(u8, res.out, "R0001") == null);
    try testing.expectEqual(@as(usize, 0), countCarets(res.out));
    // `toy check` gates its exit on the effective severity: with the ONLY diagnostic
    // ignored, zero errors survive, so there is NO error/warning summary line and the
    // process exits 0. (The removed inspection table always exited 1 on a failed file;
    // `check` is the new, faithful behaviour.)
    try testing.expect(std.mem.indexOf(u8, res.out, "error(s)") == null);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning(s)") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "band flag affects the whole band (--warn R downgrades R0001)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-c3-band";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // Migrated from `--emit check` (removed inspection table) to the `toy check` subcommand.
    const res = runToyOnFixture(gpa, io, dir_name, c3_fixture, &.{ "check", "--warn", "R" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);

    try testing.expect(std.mem.indexOf(u8, res.out, "warning[R0001]:") != null);
}

test "`toy check` unknown --warn spec is a USAGE error (exit 2)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-c3-bad";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // Migrated from `--emit check` to `toy check`, which fixes the exit code: a bad
    // severity-flag value on the check path is a USAGE error (exit 2), consistent with
    // `check`'s missing-input-file exit 2 — NOT the generic CLI-parse exit 1.
    const res = runToyOnFixture(gpa, io, dir_name, c3_fixture, &.{ "check", "--warn", "BOGUS" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);

    try testing.expect(std.mem.indexOf(u8, res.out, "unknown code or band") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 2 }, res.term);
}

test "`toy check` with no severity flags renders the coded error + a 1-error summary (exit 1)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-c3-baseline";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // Migrated from `--emit check` to `toy check`.
    const res = runToyOnFixture(gpa, io, dir_name, c3_fixture, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);

    // Empty config == identity: the coded error header, never a warning token.
    try testing.expect(std.mem.indexOf(u8, res.out, "error[R0001]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning") == null);
    // `toy check`'s program-wide summary (one error) + the compile-failure exit code.
    try testing.expect(std.mem.indexOf(u8, res.out, "1 error(s)") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
}

test "duplicate function renders a 'previously defined here' secondary label + a summary" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-c3-secondary";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const dup =
        \\fn f() -> int {
        \\  return 1
        \\}
        \\fn f() -> int {
        \\  return 2
        \\}
        \\fn main() -> int {
        \\  return f()
        \\}
    ;
    // Migrated from `--emit check` to `toy check`.
    const res = runToyOnFixture(gpa, io, dir_name, dup, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);

    // The R0002 primary at the duplicate AND a secondary label at the FIRST definition —
    // both locations visible (the flagship enrichment).
    try testing.expect(std.mem.indexOf(u8, res.out, "error[R0002]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "duplicate function 'f'") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "previously defined here") != null);
    // The diagnostic-count summary line (one root error, no cascade).
    try testing.expect(std.mem.indexOf(u8, res.out, "1 error(s)") != null);
}

test "the summary line counts respect severity config (--warn, --ignore)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-c3-summary";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // Three independent undeclared names => three R0001 diagnostics (nothing near, so no
    // "did you mean" hints).
    const multi =
        \\fn main() -> int {
        \\  aaa
        \\  bbb
        \\  ccc
        \\  return 0
        \\}
    ;
    // Migrated from `--emit check` to `toy check` (all three invocations).
    // Default: three errors.
    {
        const res = runToyOnFixture(gpa, io, dir_name, multi, &.{"check"}) catch |e| {
            if (e == error.SkipZigTest) return error.SkipZigTest;
            return e;
        };
        defer gpa.free(res.out);
        try testing.expect(std.mem.indexOf(u8, res.out, "3 error(s)") != null);
    }
    // --warn R0001: counted as warnings, not errors.
    {
        const res = runToyOnFixture(gpa, io, dir_name, multi, &.{ "check", "--warn", "R0001" }) catch |e| {
            if (e == error.SkipZigTest) return error.SkipZigTest;
            return e;
        };
        defer gpa.free(res.out);
        try testing.expect(std.mem.indexOf(u8, res.out, "3 warning(s)") != null);
        try testing.expect(std.mem.indexOf(u8, res.out, "error(s)") == null);
    }
    // --ignore R0001: excluded from the summary entirely (no diagnostic-count line).
    {
        const res = runToyOnFixture(gpa, io, dir_name, multi, &.{ "check", "--allow", "R0001" }) catch |e| {
            if (e == error.SkipZigTest) return error.SkipZigTest;
            return e;
        };
        defer gpa.free(res.out);
        try testing.expect(std.mem.indexOf(u8, res.out, "error(s)") == null);
        try testing.expect(std.mem.indexOf(u8, res.out, "warning(s)") == null);
    }
}

test "`toy --emit check` is rejected as an invalid --emit value (nonzero exit)" {
    // `--emit check` no longer exists: the CLI value list is {lex,parse,ir} in a dev
    // build. `toy --emit check <file>` must fail as a bad value. `--emit parse` still
    // works in the default Debug build (dev_inspect on).
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-d2-emitcheck";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const clean = "fn main() -> int {\n  return 0\n}\n";
    // `--emit check`: rejected (nonzero).
    {
        const res = runToyOnFixture(gpa, io, dir_name, clean, &.{ "--emit", "check" }) catch |e| {
            if (e == error.SkipZigTest) return error.SkipZigTest;
            return e;
        };
        defer gpa.free(res.out);
        try testing.expect(res.term != .exited or res.term.exited != 0);
    }
    // `--emit parse`: still valid in the Debug build (dev_inspect on) -> exit 0.
    {
        const res = runToyOnFixture(gpa, io, dir_name, clean, &.{ "--emit", "parse" }) catch |e| {
            if (e == error.SkipZigTest) return error.SkipZigTest;
            return e;
        };
        defer gpa.free(res.out);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
    }
}

test "`toy check --format ndjson` enriches each line with file, rendered, labels (valid JSON)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-d2-ndjson";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // A duplicate-fn program: R0002 carries a `previously defined here` secondary label.
    const dup =
        \\fn f() -> int {
        \\  return 1
        \\}
        \\fn f() -> int {
        \\  return 2
        \\}
        \\fn main() -> int {
        \\  return f()
        \\}
    ;
    const res = runToyOnFixture(gpa, io, dir_name, dup, &.{ "check", "--format", "ndjson" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);

    // Find the R0002 line and assert the new fields are present on it.
    var it = std.mem.tokenizeScalar(u8, res.out, '\n');
    var saw_r0002 = false;
    while (it.next()) |line| {
        if (std.mem.indexOf(u8, line, "\"code\":\"R0002\"") == null) continue;
        saw_r0002 = true;
        // The enrichment fields.
        try testing.expect(std.mem.indexOf(u8, line, "\"file\":\"") != null);
        try testing.expect(std.mem.indexOf(u8, line, dir_name) != null); // file is the input path
        try testing.expect(std.mem.indexOf(u8, line, "\"rendered\":\"") != null);
        // The rendered snippet is JSON-escaped: embedded newlines are `\n`, not raw.
        try testing.expect(std.mem.indexOf(u8, line, "error[R0002]") != null);
        // The secondary label is present in the labels array with is_primary:false.
        try testing.expect(std.mem.indexOf(u8, line, "\"labels\":[") != null);
        try testing.expect(std.mem.indexOf(u8, line, "previously defined here") != null);
        try testing.expect(std.mem.indexOf(u8, line, "\"is_primary\":false") != null);
        // Every line stays one physical line (no raw newline inside the record).
    }
    try testing.expect(saw_r0002);
}

test "`toy check --deny-warnings` promotes a surviving warning to a nonzero exit" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-d2-eow";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    // --warn R0001 downgrades the only error to a warning. Without --deny-warnings the
    // exit is 0 (no errors survive); WITH it, the surviving warning forces exit 1.
    {
        const res = runToyOnFixture(gpa, io, dir_name, c3_fixture, &.{ "check", "--warn", "R0001" }) catch |e| {
            if (e == error.SkipZigTest) return error.SkipZigTest;
            return e;
        };
        defer gpa.free(res.out);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
    }
    {
        const res = runToyOnFixture(gpa, io, dir_name, c3_fixture, &.{ "check", "--warn", "R0001", "--deny-warnings" }) catch |e| {
            if (e == error.SkipZigTest) return error.SkipZigTest;
            return e;
        };
        defer gpa.free(res.out);
        try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
    }
}

// The unused-variable warning fixtures: a `:=` local and a `for` loop var that are
// never referenced. Both must warn W0001 and NOT fail the build/check (exit 0), be
// promoted to an error by `--deny-warnings`, and be silenced by `--allow`.
const unused_local_fixture = "fn main() -> int {\n  i := 41\n  return 1\n}\n";
const unused_loopvar_fixture = "fn main() -> int {\n  for i in 0..3 {}\n  return 1\n}\n";

test "`toy check` renders W0001 for an unused local and exits 0" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0001-check";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_local_fixture, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0001]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "unused variable 'i'") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "1 warning(s)") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check` warns W0001 for an unused `for` loop variable (exit 0)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0001-loop";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_loopvar_fixture, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0001]") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "a warning-only build renders W0001, still produces a binary, and exits 0" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0001-build";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    var out_buf: [256]u8 = undefined;
    const out_path = try std.fmt.bufPrint(&out_buf, "{s}/out", .{dir_name});
    const res = runToyOnFixture(gpa, io, dir_name, unused_local_fixture, &.{ "-o", out_path }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0001") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
    // The binary was actually produced despite the warning.
    try Io.Dir.cwd().access(io, out_path, .{});
}

test "`toy check --deny-warnings` promotes W0001 to an error (exit 1)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0001-deny-check";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_local_fixture, &.{ "check", "--deny-warnings" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "error[W0001]") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "1 error(s)") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
}

test "a build with --deny-warnings fails on a surviving W0001 (nonzero exit)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0001-deny-build";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    var out_buf: [256]u8 = undefined;
    const out_path = try std.fmt.bufPrint(&out_buf, "{s}/out", .{dir_name});
    const res = runToyOnFixture(gpa, io, dir_name, unused_local_fixture, &.{ "-o", out_path, "--deny-warnings" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
}

test "`--allow W0001` and `--allow W` silence the warning (no bytes, exit 0)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    inline for (.{ "W0001", "W" }) |spec| {
        const dir_name = ".toy-test-driver-w0001-allow-" ++ spec;
        defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
        const res = runToyOnFixture(gpa, io, dir_name, unused_local_fixture, &.{ "check", "--allow", spec }) catch |e| {
            if (e == error.SkipZigTest) return error.SkipZigTest;
            return e;
        };
        defer gpa.free(res.out);
        try testing.expect(std.mem.indexOf(u8, res.out, "W0001") == null);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
    }
}

// The unused-parameter warning: a fn parameter never referenced in the body. It
// must warn W0002 and NOT fail the check (exit 0), be promoted by `--deny-warnings`,
// and be silenced by `--allow W0002`.
const unused_param_fixture = "fn f(x: int) -> int {\n  return 0\n}\nfn main() -> int {\n  return f(1)\n}\n";

test "`toy check` renders W0002 for an unused parameter and exits 0" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0002-check";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_param_fixture, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0002]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "unused parameter 'x'") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check --deny-warnings` promotes W0002 to an error (exit 1)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0002-deny";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_param_fixture, &.{ "check", "--deny-warnings" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "error[W0002]") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
}

test "`--allow W0002` silences the unused-parameter warning (exit 0)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0002-allow";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_param_fixture, &.{ "check", "--allow", "W0002" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0002") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

// The unused-function warning: a non-`pub` top-level fn that nothing in the module
// graph calls. It must warn W0003 and NOT fail the check (exit 0), be promoted by
// `--deny-warnings`, and be silenced by `--allow W0003`.
const unused_fn_fixture = "fn helper() -> int {\n  return 7\n}\nfn main() -> int {\n  return 0\n}\n";

test "`toy check` renders W0003 for an unused function and exits 0" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0003";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_fn_fixture, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0003]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "unused function 'helper'") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check --deny-warnings` promotes W0003 to an error (exit 1)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0003-deny";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_fn_fixture, &.{ "check", "--deny-warnings" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "error[W0003]") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
}

// The unused-import warning: an `import a/b` referenced neither qualified nor via a
// bare imported type. The bundled `std/io` is an unused import here (nothing names `io`,
// and io declares no `impl` and no bare-nameable type, so no branch credits it). It must
// warn W0004 and NOT fail the check (exit 0), be promoted by `--deny-warnings`, and be
// silenced by `--allow W0004`.
const unused_import_fixture = "import std/io\nfn main() -> int {\n  return 0\n}\n";

test "`toy check` renders W0004 for an unused import and exits 0" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0004";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_import_fixture, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0004]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "unused import 'std/io'") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check --deny-warnings` promotes W0004 to an error (exit 1)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0004-deny";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_import_fixture, &.{ "check", "--deny-warnings" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "error[W0004]") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
}

test "`--allow W0004` silences the unused-import warning (exit 0)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0004-allow";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_import_fixture, &.{ "check", "--allow", "W0004" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0004") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`--allow W0003` silences the unused-function warning (exit 0)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0003-allow";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unused_fn_fixture, &.{ "check", "--allow", "W0003" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0003") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

// The unreachable-code warning: a statement that can never run because a preceding
// statement in the same block always diverges. `return 0\n return 1` is the cleanest
// case — a single W0005 with no co-emitted unused-variable or fall-off noise. It must
// warn W0005 and NOT fail the check (exit 0), be promoted by `--deny-warnings`, and be
// silenced by `--allow W0005`.
const unreachable_after_return_fixture = "fn main() -> int {\n  return 0\n  return 1\n}\n";

test "`toy check` renders W0005 for code after a return and exits 0" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0005";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unreachable_after_return_fixture, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0005]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "unreachable code") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check --deny-warnings` promotes W0005 to an error (exit 1)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0005-deny";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unreachable_after_return_fixture, &.{ "check", "--deny-warnings" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "error[W0005]") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
}

test "`--allow W0005` silences the unreachable-code warning (exit 0)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0005-allow";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unreachable_after_return_fixture, &.{ "check", "--allow", "W0005" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0005") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check` flags code after a panic (the unit-typed diverging builtin)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0005-panic";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn main() -> int {\n  panic(\"x\")\n  return 0\n}\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0005]:") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check` flags code after an if/else whose arms both diverge" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0005-ifelse";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn f(c: bool) -> int {\n  if c {\n    return 1\n  } else {\n    return 2\n  }\n  return 3\n}\nfn main() -> int { return f(true) }\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0005]:") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check` flags code after a break inside a loop" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0005-break";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn main() -> int {\n  loop {\n    break\n    _x := 1\n  }\n  return 0\n}\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0005]:") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check` flags dead code nested inside a reachable block" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0005-nested";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn f() {\n  {\n    return\n    _x := 1\n  }\n}\nfn main() -> int {\n  f()\n  return 0\n}\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0005]:") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

// Only the FIRST unreachable statement warns: two dead statements after a `return`
// produce exactly one W0005 (no cascade). The tail is a unit function so the fall-off
// error does not co-emit and muddy the count.
test "`toy check` reports only the first unreachable statement (no cascade)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0005-once";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn f() {\n  return\n  _x := 1\n  _y := 2\n}\nfn main() -> int {\n  f()\n  return 0\n}\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    var count: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, res.out, i, "W0005")) |at| {
        count += 1;
        i = at + 5;
    }
    try testing.expectEqual(@as(usize, 1), count);
}

// A construct control can fall through — an else-less `if`, a `while`, a bare `for` —
// does NOT make what follows unreachable. None of these fixtures may emit W0005.
test "`toy check` does not flag code after an else-less if or a while" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const cases = [_][]const u8{
        // else-less if: the then-arm diverges but the if may be skipped.
        "fn f(c: bool) -> int {\n  if c {\n    return 1\n  }\n  return 3\n}\nfn main() -> int { return f(true) }\n",
        // while: the body may run zero times, so control falls through.
        "fn main() -> int {\n  i := 0\n  while i < 3 {\n    i = i + 1\n  }\n  return 0\n}\n",
    };
    for (cases, 0..) |src, n| {
        var buf: [64]u8 = undefined;
        const dir_name = try std.fmt.bufPrint(&buf, ".toy-test-driver-w0005-neg-{d}", .{n});
        defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
        const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
            if (e == error.SkipZigTest) return error.SkipZigTest;
            return e;
        };
        defer gpa.free(res.out);
        try testing.expect(std.mem.indexOf(u8, res.out, "W0005") == null);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
    }
}

// The unreachable-match-arm warning: a match arm that can never match because the
// preceding UNGUARDED arms already cover every value the scrutinee can take. An arm
// after an unguarded `_` is the cleanest case; it must warn W0006 and NOT fail the
// check (exit 0), be promoted by `--deny-warnings`, and be silenced by `--allow W0006`.
const unreachable_arm_after_wildcard_fixture = "fn f(n: int) -> int {\n  match n {\n    _ -> 1,\n    0 -> 2\n  }\n}\nfn main() -> int { return f(0) }\n";

test "`toy check` renders W0006 for an arm after an unguarded wildcard and exits 0" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0006";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unreachable_arm_after_wildcard_fixture, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0006]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "unreachable match arm") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check` flags an arm after full enum coverage" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0006-enum";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "enum E { A, B }\nfn f(e: E) -> int {\n  match e {\n    .A -> 1,\n    .B -> 2,\n    _ -> 3\n  }\n}\nfn main() -> int { return f(E.A) }\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0006]:") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check` flags an arm after both bools" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0006-bool";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn f(b: bool) -> int {\n  match b {\n    true -> 1,\n    false -> 0,\n    _ -> 2\n  }\n}\nfn main() -> int { return f(true) }\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0006]:") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

// A required, reachable int `_` (the exhaustiveness provider for an infinite domain)
// COMPLETES coverage and must not warn — it is tested before its own contribution.
test "`toy check` does not flag a required int wildcard catch-all" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0006-int-req";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn f(n: int) -> int {\n  match n {\n    0 -> 1,\n    _ -> 2\n  }\n}\nfn main() -> int { return f(0) }\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0006") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

// A guarded arm may fall through, so it never saturates the match: the normal arm that
// follows a guarded-only prefix is reachable and must not warn.
test "`toy check` does not flag a normal arm after a guarded arm" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0006-guard";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn f(n: int) -> int {\n  match n {\n    _ if n > 0 -> 1,\n    _ -> 2\n  }\n}\nfn main() -> int { return f(1) }\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0006") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

// Only the FIRST arm reached after saturation warns: two arms after an unguarded `_`
// produce exactly one W0006 (no cascade).
test "`toy check` reports only the first unreachable arm (no cascade)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0006-once";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn f(n: int) -> int {\n  match n {\n    _ -> 1,\n    0 -> 2,\n    1 -> 3\n  }\n}\nfn main() -> int { return f(0) }\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    var count: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, res.out, i, "W0006")) |at| {
        count += 1;
        i = at + 5;
    }
    try testing.expectEqual(@as(usize, 1), count);
}

test "`toy check --deny-warnings` promotes W0006 to an error (exit 1)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0006-deny";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unreachable_arm_after_wildcard_fixture, &.{ "check", "--deny-warnings" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "error[W0006]") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
}

test "`--allow W0006` silences the unreachable-match-arm warning (exit 0)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0006-allow";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, unreachable_arm_after_wildcard_fixture, &.{ "check", "--allow", "W0006" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0006") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

// The constant-condition warning: an `if` whose condition is a bare `true`/`false`
// literal (one branch is dead). It must warn W0007 and NOT fail the check (exit 0), be
// promoted by `--deny-warnings`, and be silenced by `--allow W0007`. Only a DIRECT
// literal-bool condition warns — a real condition and a `while true` do not.
const const_if_true_fixture = "fn main() -> int {\n  if true { return 1 } else { return 2 }\n}\n";
const const_if_false_fixture = "fn main() -> int {\n  if false { return 1 } else { return 2 }\n}\n";

test "`toy check` renders W0007 for `if true` and exits 0" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0007-true";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, const_if_true_fixture, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0007]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "if condition is always true") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check` renders W0007 for `if false` naming the value" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0007-false";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, const_if_false_fixture, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0007]:") != null);
    try testing.expect(std.mem.indexOf(u8, res.out, "if condition is always false") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

// A value-position `if` (an `x := if .. {} else {}`) with a literal-bool condition warns
// through the same helper at the value-if site.
test "`toy check` renders W0007 for a value-if with a literal condition" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0007-value";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn main() -> int {\n  x := if true { 1 } else { 2 }\n  return x\n}\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "warning[W0007]:") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

// A real (non-literal) condition must never warn.
test "`toy check` does not flag a real if condition" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0007-real";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn f(n: int) -> int {\n  if n > 0 { return 1 } else { return 2 }\n}\nfn main() -> int { return f(1) }\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0007") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

// `while true` is a separate construct (a deliberate infinite loop): out of scope.
test "`toy check` does not flag a `while true` loop" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0007-while";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const src = "fn main() -> int {\n  while true { break }\n  return 0\n}\n";
    const res = runToyOnFixture(gpa, io, dir_name, src, &.{"check"}) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0007") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "`toy check --deny-warnings` promotes W0007 to an error (exit 1)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0007-deny";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, const_if_true_fixture, &.{ "check", "--deny-warnings" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "error[W0007]") != null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 1 }, res.term);
}

test "`--allow W0007` silences the constant-condition warning (exit 0)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir_name = ".toy-test-driver-w0007-allow";
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const res = runToyOnFixture(gpa, io, dir_name, const_if_true_fixture, &.{ "check", "--allow", "W0007" }) catch |e| {
        if (e == error.SkipZigTest) return error.SkipZigTest;
        return e;
    };
    defer gpa.free(res.out);
    try testing.expect(std.mem.indexOf(u8, res.out, "W0007") == null);
    try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
}

test "codegen reports missing main and lowers a simple main" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-driver-lower";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    // No `main` -> a clear error, no codegen.
    {
        const src = "fn helper() -> int {\n return 1\n}\n";
        const path = dir_name ++ "/nomain.toy";
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });
        var r: FileResult = .{ .path = path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, 0);
        defer r.deinit(gpa);
        try testing.expect(r.err == null);
        var lowered = try lowerSingleFile(gpa, io, cache, "aarch64-macos", &r, .normal, .O0);
        switch (lowered) {
            .err => |e| try testing.expect(e.byte_offset == null),
            .ok => |*lp| {
                lp.deinit(gpa);
                return error.TestUnexpectedResult;
            },
        }
    }

    // A simple `main` lowers and links to a non-empty, word-aligned __text blob
    // with `main` at offset 0 (it is the only function).
    {
        const src = "fn main() -> int {\n x := 40\n y := 2\n return x + y\n}\n";
        const path = dir_name ++ "/main.toy";
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });
        var r: FileResult = .{ .path = path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, 1);
        defer r.deinit(gpa);
        try testing.expect(r.err == null);
        var lowered = try lowerSingleFile(gpa, io, cache, "aarch64-macos", &r, .normal, .O0);
        switch (lowered) {
            .err => return error.TestUnexpectedResult,
            .ok => |*lp| {
                defer lp.deinit(gpa);
                try testing.expectEqual(@as(usize, 0), lp.diags.len);
                try testing.expect(lp.text.len > 0);
                try testing.expectEqual(@as(usize, 0), lp.text.len % 4);
                try testing.expectEqual(@as(u32, 0), lp.entry_off);
            },
        }
    }
}

test "a syntax-error file is tainted, reported, and never reaches check/codegen" {
    // The parser now ALWAYS returns a (partial) tree, so the driver must gate
    // check/codegen on `!tainted`: a diagnostic-bearing parse stops with a
    // ParseError and never populates resolve/typecheck — which is what keeps a
    // poisoned tree (with `error_node`s) out of lower/codegen.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-driver-tainted";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    const src = "fn main() -> int {\n return *\n}\n";
    const path = dir_name ++ "/bad.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });
    var r: FileResult = .{ .path = path };
    try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, 0);
    defer r.deinit(gpa);

    // Reported as a parse error, tainted, with at least one diagnostic surfaced.
    try testing.expectEqual(@as(?anyerror, error.ParseError), r.err);
    try testing.expect(r.tainted);
    try testing.expect(r.diags.len >= 1);
    // A tree was still produced (parse() is always-a-tree)...
    try testing.expect(r.parsed);
    try testing.expect(r.nodes.len > 0);
    // ...but the file NEVER reached check: no resolve/typecheck ran, so there is no
    // path to codegen. (`checked` is only set once resolve begins.)
    try testing.expect(!r.checked);
    try testing.expect(r.resolve == null);
    try testing.expect(r.typecheck == null);
}

// End-to-end on the real OS: compile a `main`, write a signed 0o755 executable,
// run it, and assert the masked exit code. Gated to this host because only here
// can we exec what we produced. Proves the whole back-end (Codegen + MachO +
// CodeSign) yields a binary the kernel accepts and runs.
test "integration: emitted binary runs with the right exit code" {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // Unique per-process scratch dir. `zig build` runs the two test executables
    // (module + exe) in parallel and BOTH include this test, so a fixed dir name
    // races (one process's deleteTree vs the other's writeFile -> FileNotFound).
    // tmpDir gives each run a random-named, auto-cleaned dir under .zig-cache/tmp.
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_name_buf: [64]u8 = undefined;
    const dir_name = std.fmt.bufPrint(&dir_name_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    const Case = struct { src: []const u8, name: []const u8, expect: u8 };
    const cases = [_]Case{
        .{ .src = "fn main() -> int {\n x := 40\n y := 2\n return x + y\n}\n", .name = "add", .expect = 42 },
        // 300 & 0xFF == 44: dyld's start glue masks main's return to a byte.
        .{ .src = "fn main() -> int {\n return 300\n}\n", .name = "big", .expect = 44 },
        // Multi-function cases — each exercises a back-end capability that a
        // unit assert can't catch: a wrong ABI/alignment/reloc/entry-offset bug
        // still faults or returns the wrong code when actually executed.
        // A forward call into a helper.
        .{ .src = "fn addfn(a: int, b: int) -> int {\n return a + b\n}\nfn main() -> int {\n return addfn(40, 2)\n}\n", .name = "call", .expect = 42 },
        // `main` is NOT first: nonzero entry_off + a backward bl to `helper`.
        .{ .src = "fn main() -> int {\n return helper()\n}\nfn helper() -> int {\n return 7\n}\n", .name = "mainfirst", .expect = 7 },
        // Nested call: id(40)'s result must be spilled before id(2) runs, else the
        // first arg is clobbered. Proves eval-to-temps-then-marshal.
        .{ .src = "fn id(x: int) -> int {\n return x\n}\nfn add2(a: int, b: int) -> int {\n return a + b\n}\nfn main() -> int {\n return add2(id(40), id(2))\n}\n", .name = "nested", .expect = 42 },
        // > 8 args: a 10-int-param sum called with 1..10 → 55. Proves the outgoing
        // region + the incoming-stack-arg ldrFp copy + sp 16-alignment at the call
        // (a misaligned sp would fault on the kernel's alignment check).
        .{ .src = "fn sum10(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int, j: int) -> int {\n return a + b + c + d + e + f + g + h + i + j\n}\nfn main() -> int {\n return sum10(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)\n}\n", .name = "tenargs", .expect = 55 },
        // A void call as an expression statement: result discarded, then return 5.
        .{ .src = "fn noop() {\n return\n}\nfn main() -> int {\n noop()\n return 5\n}\n", .name = "voidcall", .expect = 5 },
        // Control flow — each RUN proves a back-end capability a byte assert
        // can't: a wrong b.cond sense, a sign-flipped branch offset, an un-backpatched
        // placeholder, or a missing frame-walk arm only shows when actually executed.
        // if true-arm taken.
        .{ .src = "fn main() -> int {\n if 3 > 2 {\n return 7\n }\n return 0\n}\n", .name = "if_true", .expect = 7 },
        // if false → else taken (flip the operands of the above).
        .{ .src = "fn main() -> int {\n if 2 > 3 {\n return 7\n } else {\n return 9\n }\n}\n", .name = "if_else", .expect = 9 },
        // SIGNED-condition canary: -3 < 0 must use a SIGNED b.lt/b.ge (an unsigned
        // b.lo would treat -3 as a huge unsigned and take the wrong branch).
        .{ .src = "fn main() -> int {\n if -3 < 0 {\n return 1\n }\n return 0\n}\n", .name = "signed", .expect = 1 },
        // else-if 3-way ladder, middle arm taken (nested if_stmt as else-node).
        .{ .src = "fn main() -> int {\n x := 2\n if x == 1 {\n return 10\n } else if x == 2 {\n return 20\n } else {\n return 30\n }\n}\n", .name = "elseif", .expect = 20 },
        // while loop summing 0..4 = 10. The BACKWARD branch must have a correct,
        // sign-extended negative offset or the program hangs / faults instead of
        // returning 10. (Runs to completion below: a hang would fail the test.)
        .{ .src = "fn main() -> int {\n i := 0\n s := 0\n while i < 5 {\n s = s + i\n i = i + 1\n }\n return s\n}\n", .name = "while_sum", .expect = 10 },
        // Nested if inside a while: count how many of 0..9 are >= 5 → 5.
        .{ .src = "fn main() -> int {\n i := 0\n c := 0\n while i < 10 {\n if i >= 5 {\n c = c + 1\n }\n i = i + 1\n }\n return c\n}\n", .name = "nested", .expect = 5 },
        // && short-circuit as a VALUE materialized to 0/1: (1<2)&&(3<2) is false → 0.
        .{ .src = "fn main() -> int {\n b := (1 < 2) && (3 < 2)\n if b {\n return 1\n }\n return 0\n}\n", .name = "and_val", .expect = 0 },
        // || short-circuit value: (1<2)||(3<2) is true → 1.
        .{ .src = "fn main() -> int {\n b := (1 < 2) || (3 < 2)\n if b {\n return 1\n }\n return 0\n}\n", .name = "or_val", .expect = 1 },
        // && short-circuit OBSERVABLE: the rhs is a call that would set a flag; with
        // a false lhs it must NOT run. Here lhs false ⇒ helper() not called ⇒ 0.
        .{ .src = "fn helper() -> int {\n return 1\n}\nfn main() -> int {\n if (1 > 2) && (helper() == 1) {\n return 99\n }\n return 0\n}\n", .name = "and_short", .expect = 0 },
        // An early return inside a branch, with code AFTER the if (return-continues:
        // the then-arm's inline epilogue must not stop codegen of the trailing return).
        .{ .src = "fn main() -> int {\n x := 5\n if x > 0 {\n return 42\n }\n return 7\n}\n", .name = "early_ret", .expect = 42 },
        // Recursive factorial, now TERMINATING via an if base case: 5! = 120.
        .{ .src = "fn fact(n: int) -> int {\n if n <= 1 {\n return 1\n }\n return n * fact(n - 1)\n}\nfn main() -> int {\n return fact(5)\n}\n", .name = "fact", .expect = 120 },
        // fib(10) = 55: return-continues + intra-fn backpatch + recursive calls all
        // at once. 55 & 0xFF == 55.
        .{ .src = "fn fib(n: int) -> int {\n if n < 2 {\n return n\n }\n return fib(n - 1) + fib(n - 2)\n}\nfn main() -> int {\n return fib(10)\n}\n", .name = "fib", .expect = 55 },
        // Expression orientation — each RUN proves the value-merge/trailing-expr
        // lowering with an actual exit code.
        // value-if, then-arm taken.
        .{ .src = "fn main() -> int {\n c := 1\n x := if c > 0 { 7 } else { 9 }\n return x\n}\n", .name = "ifval_then", .expect = 7 },
        // value-if, else-arm taken.
        .{ .src = "fn main() -> int {\n c := 0\n x := if c > 0 { 7 } else { 9 }\n return x\n}\n", .name = "ifval_else", .expect = 9 },
        // a fn body that is JUST a trailing expression (no return) → 42.
        .{ .src = "fn answer() -> int { 41 + 1 }\nfn main() -> int {\n return answer()\n}\n", .name = "trailing", .expect = 42 },
        // a bare block expression → 2.
        .{ .src = "fn main() -> int {\n x := { a := 1\n a + 1 }\n return x\n}\n", .name = "blockval", .expect = 2 },
        // a unit-returning fn (explicit `-> ()`), called for effect.
        .{ .src = "fn noop() -> () {\n }\nfn main() -> int {\n noop()\n return 5\n}\n", .name = "unitret", .expect = 5 },
        // an if-expr as a function argument → 3.
        .{ .src = "fn id(x: int) -> int { x }\nfn main() -> int {\n return id(if 1 < 2 { 3 } else { 4 })\n}\n", .name = "ifval_arg", .expect = 3 },
        // a nested if-expr inside another if-arm → 8.
        .{ .src = "fn main() -> int {\n return if 1 < 2 { if 3 > 4 { 0 } else { 8 } } else { 5 }\n}\n", .name = "ifval_nested", .expect = 8 },
        // a value-if where the then-arm diverges (returns on all paths): the live
        // else-arm value reaches the join → 8.
        .{ .src = "fn f(n: int) -> int {\n y := if n < 0 { return 0 } else { 8 }\n return y\n}\nfn main() -> int {\n return f(5)\n}\n", .name = "ifval_div", .expect = 8 },
        // FRAME-SIZING REGRESSION: a value-if as the 3rd call arg whose arm has a
        // depth-3 nested binary. The arm lowers at the INHERITED temp depth, so its
        // internal spills must be sized from there — sizing from 0 under-counts the
        // frame and the top spill lands on the saved x30/LR slot → SIGBUS in this
        // non-leaf fn (compute). add3(10,20, 1+(2+(3+(4+(5+6))))) = 51.
        .{ .src = "fn add3(a: int, b: int, c: int) -> int {\n return a + b + c\n}\nfn compute(k: int) -> int {\n return add3(10, 20, if k > 0 { 1 + (2 + (3 + (4 + (5 + 6)))) } else { 0 })\n}\nfn main() -> int {\n return compute(1)\n}\n", .name = "ifval_deeparg", .expect = 51 },
        // Same regression via a BARE-BLOCK expr as the deep 3rd arg.
        .{ .src = "fn add3(a: int, b: int, c: int) -> int {\n return a + b + c\n}\nfn compute(k: int) -> int {\n return add3(10, 20, { 1 + (2 + (3 + (4 + (5 + 6)))) })\n}\nfn main() -> int {\n return compute(1)\n}\n", .name = "blockval_deeparg", .expect = 51 },
        // a str-returning fn whose body is a value-if, consumed through a CALL → its
        // (ptr,len) must reach the caller in (x0,x1) and bind cleanly. Exits 0. (No
        // print: this single-file pipeline cannot import `std/io`; binding the str
        // still forces the reg-pair return to materialize in the caller.)
        .{ .src = "fn pick(b: int) -> str {\n if b > 0 { \"yes\" } else { \"no\" }\n}\nfn main() -> int {\n s := pick(1)\n return 0\n}\n", .name = "strret_call", .expect = 0 },
        // ENTRY-EPILOGUE REGRESSION: a non-unit `main` whose body is a TRAILING value
        // expression (no explicit return) lands its value in x0 at the fall-through
        // epilogue — the entry's `movz x0,#0` must NOT clobber it (only a UNIT entry
        // forces 0). Trailing if-value → 7.
        .{ .src = "fn main() -> int {\n if 1 > 0 { 7 } else { 8 }\n}\n", .name = "trailing_main_if", .expect = 7 },
        // Trailing bare-block value as the non-unit main body → 15.
        .{ .src = "fn main() -> int {\n {\n a := 10\n a + 5\n }\n}\n", .name = "trailing_main_block", .expect = 15 },

        // Loops — each RUN is a HANG-CANARY: a wrong/sign-flipped back-edge would
        // hang or fault instead of returning, so a completed run with the right exit
        // code proves the loop terminated.
        // value-loop yielding via break-value: i goes 0..5, breaks 5*10 = 50.
        .{ .src = "fn main() -> int {\n i := 0\n v := loop {\n if i >= 5 { break i * 10 }\n i = i + 1\n }\n return v\n}\n", .name = "loopval", .expect = 50 },
        // while with an early break: sum 0,1,2 then break at i==3 → 3.
        .{ .src = "fn main() -> int {\n i := 0\n s := 0\n while i < 100 {\n if i == 3 { break }\n s = s + i\n i = i + 1\n }\n return s\n}\n", .name = "while_break", .expect = 3 },
        // while with a continue: advance i FIRST, skip adding when i==2.
        // 1 + 3 + 4 + 5 = 13.
        .{ .src = "fn main() -> int {\n i := 0\n s := 0\n while i < 5 {\n i = i + 1\n if i == 2 { continue }\n s = s + i\n }\n return s\n}\n", .name = "while_cont", .expect = 13 },
        // for over a half-open range [1,5): 1+2+3+4 = 10.
        .{ .src = "fn main() -> int {\n s := 0\n for i in 1..5 {\n s = s + i\n }\n return s\n}\n", .name = "for_sum", .expect = 10 },
        // for with continue (hits the increment so it still terminates): skip i==2.
        // 0+1+3+4 = 8.
        .{ .src = "fn main() -> int {\n s := 0\n for i in 0..5 {\n if i == 2 { continue }\n s = s + i\n }\n return s\n}\n", .name = "for_cont", .expect = 8 },
        // nested loops: a bare break hits the INNERMOST loop only. Inner adds 1 then
        // breaks at j==1, per each of 3 outer iters → 3.
        .{ .src = "fn main() -> int {\n s := 0\n for i in 0..3 {\n for j in 0..3 {\n if j == 1 { break }\n s = s + 1\n }\n }\n return s\n}\n", .name = "nested_break", .expect = 3 },
        // nested loops: a continue hits the INNERMOST loop. Inner skips j==1 → adds at
        // j=0 and j=2 → 2 per outer, 3 outers → 6.
        .{ .src = "fn main() -> int {\n s := 0\n for i in 0..3 {\n for j in 0..3 {\n if j == 1 { continue }\n s = s + 1\n }\n }\n return s\n}\n", .name = "nested_cont", .expect = 6 },
        // a break-less `loop` typed `never`, used where a value is expected: the loop
        // only `return`s. f returns 7 once ready(i) holds.
        .{ .src = "fn ready(n: int) -> bool {\n return n >= 3\n}\nfn f() -> int {\n i := 0\n loop {\n if ready(i) { return 7 }\n i = i + 1\n }\n}\nfn main() -> int {\n return f()\n}\n", .name = "never_loop", .expect = 7 },
        // FRAME-SIZING / SIGBUS canary: a value-loop as a deep call argument whose
        // break-value is a depth-5 nested binary. The loop result slot + body must be
        // sized from the inherited temp depth or the spill overruns onto saved x29/x30.
        // add3(10,20, loop{ break 1+(2+(3+(4+(5+6)))) }) = 10+20+21 = 51.
        .{ .src = "fn add3(a: int, b: int, c: int) -> int {\n return a + b + c\n}\nfn compute() -> int {\n return add3(10, 20, loop { break 1 + (2 + (3 + (4 + (5 + 6)))) })\n}\nfn main() -> int {\n return compute()\n}\n", .name = "loopval_deeparg", .expect = 51 },
        // back-edge sign canary: a tight count to 42 then break 42.
        .{ .src = "fn main() -> int {\n i := 0\n v := loop {\n if i >= 42 { break i }\n i = i + 1\n }\n return v\n}\n", .name = "loop_canary", .expect = 42 },

        // Labels & multi-level exits — each RUN is a HANG-CANARY (a wrong outer
        // back-edge/target would hang or fault). LABEL-RESOLUTION canaries too: a
        // break/continue hitting the WRONG (innermost vs named) context miscompiles
        // to a visibly wrong exit code.
        // (1) break a VALUE out of an OUTER loop from inside an INNER for-loop. j==4
        // → break @outer 4*10 = 40 (exits BOTH loops). A wrong target would loop.
        .{ .src = "fn main() -> int {\n @outer loop {\n for j in 0..10 {\n if j == 4 { break @outer j * 10 }\n }\n }\n}\n", .name = "break_outer", .expect = 40 },
        // (2) labeled BARE BLOCK expression with early exits. a=false,b=true → 2.
        .{ .src = "fn main() -> int {\n a := false\n b := true\n x := @calc {\n if a { break @calc 1 }\n if b { break @calc 2 }\n 3\n }\n x\n}\n", .name = "labeledblock", .expect = 2 },
        // (2b) labeled bare block where NO break fires → the trailing expr (3).
        .{ .src = "fn main() -> int {\n a := false\n b := false\n @calc {\n if a { break @calc 1 }\n if b { break @calc 2 }\n 3\n }\n}\n", .name = "labeledblock_fall", .expect = 3 },
        // (3) `continue @outer` from an inner loop skips to the outer's next iter.
        // On j==3 continue @outer; each outer iter adds 3 (j=0,1,2) → 3 outers → 9.
        .{ .src = "fn main() -> int {\n s := 0\n @outer for i in 0..3 {\n for j in 0..10 {\n if j == 3 { continue @outer }\n s = s + 1\n }\n }\n s\n}\n", .name = "continue_outer", .expect = 9 },
        // (4) bare break/continue STILL hit the INNERMOST loop under a labeled outer
        // (unchanged): inner `for` adds 1 then bare-breaks at j==1 → 1 per outer,
        // 3 outers → 3 (the bare break must NOT escape to @outer).
        .{ .src = "fn main() -> int {\n s := 0\n @outer loop {\n for i in 0..3 {\n for j in 0..3 {\n if j == 1 { break }\n s = s + 1\n }\n }\n break @outer s\n }\n}\n", .name = "bare_inner_under_labeled", .expect = 3 },
        // (5) FRAME-SIZING / SIGBUS canary: a labeled bare block as a deep call arg
        // whose break-value is a depth-5 nested binary. The block result slot + body
        // must be sized from the inherited temp depth (lowerLabeledBlock body@depth+1)
        // or the spill overruns onto saved x29/x30. add3(10,20,@blk{break @blk ...})=51.
        .{ .src = "fn add3(a: int, b: int, c: int) -> int {\n return a + b + c\n}\nfn compute() -> int {\n return add3(10, 20, @blk { break @blk 1 + (2 + (3 + (4 + (5 + 6)))) })\n}\nfn main() -> int {\n return compute()\n}\n", .name = "labeledblock_deeparg", .expect = 51 },
        // (5b) deep break @outer value as the SIGBUS canary on a labeled loop too.
        .{ .src = "fn add3(a: int, b: int, c: int) -> int {\n return a + b + c\n}\nfn compute() -> int {\n return add3(10, 20, @lp loop { break @lp 1 + (2 + (3 + (4 + (5 + 6)))) })\n}\nfn main() -> int {\n return compute()\n}\n", .name = "break_outer_deeparg", .expect = 51 },
        // (6) str-typed result canary: a labeled bare block yielding a str (16-byte
        // result slot + x0/x1) bound to a local. Exits 0; proves the fat-value
        // store/load through the labeled-block result slot. (No print: single-file
        // pipeline can't import `std/io`; binding the str still exercises the slot.)
        .{ .src = "fn greet(s: str) -> str {\n s\n}\nfn main() -> int {\n s := @blk { break @blk greet(\"hi\") }\n return 0\n}\n", .name = "labeledblock_str", .expect = 0 },

        // Structs — each RUN proves a back-end capability (layout/ABI/copy) that
        // a byte assert can't: a wrong field offset, ABI class, or missed copy
        // silently corrupts data or faults only when actually executed.
        // (1) construct + read two fields → 42.
        .{ .src = "struct P { x: int, y: int }\nfn main() -> int {\n p := P { x: 40, y: 2 }\n return p.x + p.y\n}\n", .name = "struct_read", .expect = 42 },
        // (2) field punning → 42.
        .{ .src = "struct P { x: int, y: int }\nfn main() -> int {\n x := 40\n y := 2\n p := P { x, y }\n return p.x + p.y\n}\n", .name = "struct_pun", .expect = 42 },
        // (3) field place-store then read-back → 99.
        .{ .src = "struct P { x: int, y: int }\nfn main() -> int {\n p := P { x: 7, y: 1 }\n p.x = 99\n return p.x\n}\n", .name = "struct_mut", .expect = 99 },
        // (4) COPY semantics (small struct): pass by value, callee mutates its param,
        // the caller's copy is UNCHANGED → 7. A missed copy would read 1000.
        .{ .src = "struct P { x: int, y: int }\nfn bump(q: P) -> int {\n q.x = 1000\n return q.x\n}\nfn main() -> int {\n p := P { x: 7, y: 0 }\n z := bump(p)\n return p.x\n}\n", .name = "struct_copy", .expect = 7 },
        // (5) return a small (<=16B) struct by value → 42 (reg-pair return path).
        .{ .src = "struct P { x: int, y: int }\nfn make() -> P {\n return P { x: 40, y: 2 }\n}\nfn main() -> int {\n p := make()\n return p.x + p.y\n}\n", .name = "struct_ret_small", .expect = 42 },
        // (6) a >16B struct (3 ints) passed AND returned via x8 sret. add1 bumps a:
        // {1,2,3}→{2,2,3} sum 7. Proves the indirect arg + x8 sret paths.
        .{ .src = "struct V3 { a: int, b: int, c: int }\nfn add1(v: V3) -> V3 {\n v.a = v.a + 1\n return v\n}\nfn main() -> int {\n p := V3 { a: 1, b: 2, c: 3 }\n q := add1(p)\n return q.a + q.b + q.c\n}\n", .name = "struct_big", .expect = 7 },
        // (7) COPY semantics (large struct): callee mutates its >16B param, caller
        // unchanged → 6. A missed indirect-arg copy would corrupt the caller.
        .{ .src = "struct V3 { a: int, b: int, c: int }\nfn bump(v: V3) -> int {\n v.a = 1000\n return v.a\n}\nfn main() -> int {\n p := V3 { a: 1, b: 2, c: 3 }\n z := bump(p)\n return p.a + p.b + p.c\n}\n", .name = "struct_big_copy", .expect = 6 },
        // (8) a free function computing over a struct (area) → 42.
        .{ .src = "struct Rect { w: int, h: int }\nfn area(r: Rect) -> int { r.w * r.h }\nfn main() -> int {\n r := Rect { w: 6, h: 7 }\n return area(r)\n}\n", .name = "struct_area", .expect = 42 },
        // (9) nested struct field access o.i.v → 42.
        .{ .src = "struct Inner { v: int }\nstruct Outer { i: Inner, w: int }\nfn main() -> int {\n o := Outer { i: Inner { v: 40 }, w: 2 }\n return o.i.v + o.w\n}\n", .name = "struct_nested", .expect = 42 },
        // (10) FRAME-SIZING / SIGBUS canary: a small struct literal as the deep 3rd
        // call arg. use3(1,2,P{3,4})+32 = 42. The struct temp span must be sized.
        .{ .src = "struct P { x: int, y: int }\nfn use3(a: int, b: int, p: P) -> int { a + b + p.x + p.y }\nfn main() -> int {\n return use3(1, 2, P { x: 3, y: 4 }) + 32\n}\n", .name = "struct_deeparg", .expect = 42 },
        // (11) SIGBUS canary: a >16B struct temp as a deep arg. sum(36,V3{1,2,3})=42.
        .{ .src = "struct V3 { a: int, b: int, c: int }\nfn sum(x: int, v: V3) -> int { x + v.a + v.b + v.c }\nfn main() -> int {\n return sum(36, V3 { a: 1, b: 2, c: 3 })\n}\n", .name = "struct_big_deeparg", .expect = 42 },
        // (12) SIGBUS canary: nested field access (o.i.v) as a deep call arg → 42.
        .{ .src = "struct Inner { v: int }\nstruct Outer { i: Inner, w: int }\nfn id(a: int, b: int, c: int) -> int { a + b + c }\nfn main() -> int {\n o := Outer { i: Inner { v: 30 }, w: 2 }\n return id(o.i.v, o.w, 10)\n}\n", .name = "struct_fieldarg", .expect = 42 },
        // SIGBUS regression (frame undersizing): RETURN a SMALL struct whose
        // source is a bare param identifier (`return p`) — genStructExprPair spills
        // a temp at cg.depth that measureExpr's .identifier arm must reserve, else
        // the return-copy store lands on saved x29/x30 → SIGBUS. id(P{40,2})=42.
        .{ .src = "struct P { x: int, y: int }\nfn id(p: P) -> P {\n return p\n}\nfn main() -> int {\n p := P { x: 40, y: 2 }\n q := id(p)\n return q.x + q.y\n}\n", .name = "struct_ret_ident", .expect = 42 },
        // Same, trailing-expr form (`fn id(p:P) -> P { p }`).
        .{ .src = "struct P { x: int, y: int }\nfn id(p: P) -> P { p }\nfn main() -> int {\n p := P { x: 40, y: 2 }\n q := id(p)\n return q.x + q.y\n}\n", .name = "struct_ret_ident_trailing", .expect = 42 },
        // Return a LOCAL-rooted SMALL struct FIELD (`return o.i`, i a substruct):
        // the local-rooted .field_access arm of measureExpr must reserve the
        // reg-pair temp when the field itself is a struct. geti(Outer).a+.b = 42.
        .{ .src = "struct Inner { a: int, b: int }\nstruct Outer { i: Inner, z: int }\nfn geti(o: Outer) -> Inner {\n return o.i\n}\nfn main() -> int {\n o := Outer { i: Inner { a: 40, b: 2 }, z: 99 }\n r := geti(o)\n return r.a + r.b\n}\n", .name = "struct_ret_subfield", .expect = 42 },
        // sret regression: a >16B struct returned via a VALUE-IF body (not an
        // expr_stmt) must be written through x8 — was lowered as a statement-if and
        // x8 left unwritten (garbage). make(1)={10,20,30}, sum = 60.
        .{ .src = "struct V3 { a: int, b: int, c: int }\nfn make(c: int) -> V3 {\n if c == 1 { V3 { a: 10, b: 20, c: 30 } } else { V3 { a: 1, b: 1, c: 1 } }\n}\nfn main() -> int {\n p := make(1)\n return p.a + p.b + p.c\n}\n", .name = "struct_sret_valueif", .expect = 60 },
        // Miscompile regression: a struct literal as a call arg whose field init
        // is itself a CALL — the field value must spill PAST the struct's own temp
        // bytes (not onto x). area(Point{40, id(2)}) = 42.
        .{ .src = "struct Point { x: int, y: int }\nfn id(n: int) -> int { return n }\nfn area(p: Point) -> int { p.x + p.y }\nfn main() -> int {\n return area(Point { x: 40, y: id(2) })\n}\n", .name = "struct_arg_callinit", .expect = 42 },
        // SIGSEGV regression: a >16B struct returned, a field init calls a 9-arg
        // fn (whose arg marshal dirties x9) — the sret dest pointer must NOT be held
        // in caller-saved x9 across the bl. val(1..9)=45, mk()={45,1,1}, sum = 47.
        .{ .src = "struct V3 { a: int, b: int, c: int }\nfn val(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int) -> int {\n return a + b + c + d + e + f + g + h + i\n}\nfn mk() -> V3 {\n return V3 { a: val(1,2,3,4,5,6,7,8,9), b: 1, c: 1 }\n}\nfn main() -> int {\n q := mk()\n return q.a + q.b + q.c\n}\n", .name = "struct_sret_x9", .expect = 47 },
        // Coverage regression: a SMALL struct produced by a value-if, BOUND to a
        // local (a non-return sink) — was hard-rejected "struct expression
        // unsupported in codegen". P{10,20} → 30.
        .{ .src = "struct P { x: int, y: int }\nfn main() -> int {\n c := 1\n p := if c == 1 { P { x: 10, y: 20 } } else { P { x: 1, y: 1 } }\n return p.x + p.y\n}\n", .name = "struct_valueif_bind", .expect = 30 },
        // Same for a LARGE struct bound from a value-if. V3{10,20,30} → 60.
        .{ .src = "struct V3 { a: int, b: int, c: int }\nfn main() -> int {\n c := 1\n p := if c == 1 { V3 { a: 10, b: 20, c: 30 } } else { V3 { a: 1, b: 1, c: 1 } }\n return p.a + p.b + p.c\n}\n", .name = "struct_valueif_bind_big", .expect = 60 },
        // A struct from a labeled-block break, bound to a local. break @blk P{40,2}.
        .{ .src = "struct P { x: int, y: int }\nfn main() -> int {\n p := @blk { break @blk P { x: 40, y: 2 } }\n return p.x + p.y\n}\n", .name = "struct_labeledblock_break", .expect = 42 },
        // A LARGE struct from a loop break, bound to a local (struct break-value
        // must copy full bytes, not just x0/x1). break V3{40,1,1} → 42.
        .{ .src = "struct V3 { a: int, b: int, c: int }\nfn main() -> int {\n i := 0\n v := loop {\n i = i + 1\n if i == 3 { break V3 { a: 40, b: 1, c: 1 } }\n }\n return v.a + v.b + v.c\n}\n", .name = "struct_loop_break_big", .expect = 42 },
        // Enums — each RUN proves an enum capability a byte assert can't.
        // (e1) all three variant forms, qualified + inferred, match binding tuple +
        // struct payloads + unit + a value-bound match. Circle(5)=25, Rect{3,4}=12,
        // Empty=0 → 37.
        .{ .src = "enum Shape { Empty, Circle(int), Rect { w: int, h: int } }\nfn area(s: Shape) -> int { match s { .Circle(r) -> r * r, .Rect { w, h } -> w * h, .Empty -> 0 } }\nfn main() -> int {\n c := Shape.Circle(5)\n r := Shape.Rect { w: 3, h: 4 }\n return area(c) + area(r) + area(.Empty)\n}\n", .name = "enum_shape", .expect = 37 },
        // (e2) a wildcard arm. A 4-variant enum, match .B + `_`. B → 7.
        .{ .src = "enum E { A, B, C, D }\nfn f(e: E) -> int { match e { .B -> 7, _ -> 0 } }\nfn main() -> int {\n return f(E.B)\n}\n", .name = "enum_wildcard", .expect = 7 },
        // (e3) a small enum passed BY VALUE and RETURNED by value (reg pair). The
        // callee returns the same variant; round-trips C(42) → 42.
        .{ .src = "enum E { C(int), N }\nfn echo(e: E) -> E { e }\nfn main() -> int {\n x := echo(E.C(42))\n return match x { .C(r) -> r, .N -> 0 }\n}\n", .name = "enum_byvalue_small", .expect = 42 },
        // (e4) a >16B enum (struct variant 3 ints + 8B tag = 32B) passed + returned
        // via the indirect/x8 path; sum of fields. A{10,20,30} → 60.
        .{ .src = "enum Big { A { p: int, q: int, r: int }, B(int) }\nfn echo(b: Big) -> Big { b }\nfn main() -> int {\n x := echo(Big.A { p: 10, q: 20, r: 30 })\n return match x { .A { p, q, r } -> p + q + r, .B(n) -> n }\n}\n", .name = "enum_byvalue_big", .expect = 60 },
        // (e5) COPY semantics: a callee binds (by value) its enum param's payload;
        // the caller's value is unchanged. callee reads C(5)→5, caller still C(5)→5,
        // 5 + 5 = 10. (enums have no field-store; the by-value bind is the copy.)
        .{ .src = "enum E { C(int) }\nfn peek(e: E) -> int { match e { .C(r) -> r } }\nfn main() -> int {\n e := E.C(5)\n a := peek(e)\n b := match e { .C(r) -> r }\n return a + b\n}\n", .name = "enum_copy", .expect = 10 },
        // (e6) match as a value bound to a LOCAL and as a fn's TRAILING expression.
        // f's body IS the match; x binds a match. C(20) → 20, then +22 = 42.
        .{ .src = "enum E { C(int), N }\nfn f(e: E) -> int { match e { .C(r) -> r, .N -> 0 } }\nfn main() -> int {\n x := match E.C(20) { .C(r) -> r, .N -> 0 }\n return f(E.C(22)) + x\n}\n", .name = "enum_match_expr", .expect = 42 },
        // (e7) FRAME-SIZING / SIGBUS canary: a variant constructed as a deep 3rd call
        // arg whose payload is a deeply-nested binary. add3(10,20,C(1+(2+(3+(4+(5+6))))))
        // → area uses the payload = 21; 10+20+21 = 51... but here we sum the payload
        // directly: add3(10,20, <Circle payload via match>) = 10+20+ (1+2+3+4+5+6)=51.
        .{ .src = "enum E { C(int), N }\nfn add3(a: int, b: int, c: int) -> int { return a + b + c }\nfn pay(e: E) -> int { match e { .C(r) -> r, .N -> 0 } }\nfn main() -> int {\n return add3(10, 20, pay(.C(1 + (2 + (3 + (4 + (5 + 6)))))))\n}\n", .name = "enum_deeparg", .expect = 51 },
        // (e8) FRAME-SIZING: a >16B enum temp constructed as a deep call arg. The
        // large-variant construction is the 2nd arg (indirect). sum(36, A{1,2,3})=42.
        .{ .src = "enum Big { A { p: int, q: int, r: int }, B(int) }\nfn sum(x: int, b: Big) -> int { x + match b { .A { p, q, r } -> p + q + r, .B(n) -> n } }\nfn main() -> int {\n return sum(36, Big.A { p: 1, q: 2, r: 3 })\n}\n", .name = "enum_big_deeparg", .expect = 42 },
        // (e9) FRAME-SIZING: a match as a deep call sub-operand with nested arm
        // temps. add3(10,20, match B(6) { .B(r) -> 1+(2+(3+r)), ... }) = 10+20+12 = 42.
        .{ .src = "enum E { A(int), B(int) }\nfn add3(a: int, b: int, c: int) -> int { return a + b + c }\nfn main() -> int {\n return add3(10, 20, match E.B(6) { .A(r) -> r, .B(r) -> 1 + (2 + (3 + r)) })\n}\n", .name = "enum_match_deeparg", .expect = 42 },
        // (e10) a match RETURNING a large enum via the sret path (trailing expr):
        // remap a small enum to a large one. C → A{10,20,12} sum 42.
        .{ .src = "enum Sel { C, D }\nenum Big { A { p: int, q: int, r: int }, B(int) }\nfn pick(s: Sel) -> Big { match s { .C -> Big.A { p: 10, q: 20, r: 12 }, .D -> Big.B(0) } }\nfn main() -> int {\n x := pick(Sel.C)\n return match x { .A { p, q, r } -> p + q + r, .B(v) -> v }\n}\n", .name = "enum_match_sret", .expect = 42 },
        // (e11) FRAME-SIZING REGRESSION: an aggregate-RETURNING call passed DIRECTLY
        // as an aggregate ARG to another call — f(g(...)). genCallInner pre-advances
        // the depth past a struct/enum arg's own span BEFORE materializing it, so the
        // inner call's sret temp lowers one span deeper; measureExpr's .call arm must
        // measure that aggregate arg's interior at d+span (NOT d) or the inner spill
        // overruns the frame onto saved x29/x30 → SIGBUS at ret. >16B enum: echo is
        // identity, consume sums fields. consume(echo(A{10,20,30})) = 60.
        .{ .src = "enum Big { A { p: int, q: int, r: int }, B(int) }\nfn echo(b: Big) -> Big { b }\nfn consume(b: Big) -> int { match b { .A { p, q, r } -> p + q + r, .B(n) -> n } }\nfn main() -> int {\n return consume(echo(Big.A { p: 10, q: 20, r: 30 }))\n}\n", .name = "enum_big_call_in_arg", .expect = 60 },
        // (e12) same shape, <=16B enum (reg-pair ABI), QUALIFIED: pick(id(E.C(42)))=42.
        .{ .src = "enum E { C(int), N }\nfn id(e: E) -> E { e }\nfn pick(e: E) -> int { match e { .C(r) -> r, .N -> 0 } }\nfn main() -> int {\n return pick(id(E.C(42)))\n}\n", .name = "enum_small_call_in_arg", .expect = 42 },
        // (e13) same, <=16B enum, INFERRED .C(42): the inner CALL result is the agg
        // arg, so inferred-vs-qualified does not change framing. pick(id(.C(42)))=42.
        .{ .src = "enum E { C(int), N }\nfn id(e: E) -> E { e }\nfn pick(e: E) -> int { match e { .C(r) -> r, .N -> 0 } }\nfn main() -> int {\n return pick(id(.C(42)))\n}\n", .name = "enum_small_call_in_arg_inferred", .expect = 42 },
        // (e14) PRE-EXISTING STRUCT widening of the same defect: a >16B struct
        // returned by a call passed as the struct arg of another call. echo identity,
        // consume sums. consume(echo(Big{10,20,30,40})) = 100.
        .{ .src = "struct Big { p: int, q: int, r: int, s: int }\nfn echo(b: Big) -> Big { b }\nfn consume(b: Big) -> int { b.p + b.q + b.r + b.s }\nfn main() -> int {\n return consume(echo(Big { p: 10, q: 20, r: 30, s: 40 }))\n}\n", .name = "struct_big_call_in_arg", .expect = 100 },
        // Match enrichment.
        // (m1) int literal match + wildcard. f(1)=20.
        .{ .src = "fn f(n: int) -> int { match n { 0 -> 10, 1 -> 20, _ -> 99 } }\nfn main() -> int { return f(1) }\n", .name = "match_int", .expect = 20 },
        // (m1b) int literal match falls to wildcard. f(7)=99.
        .{ .src = "fn f(n: int) -> int { match n { 0 -> 10, 1 -> 20, _ -> 99 } }\nfn main() -> int { return f(7) }\n", .name = "match_int_wild", .expect = 99 },
        // (m2) bool match, no `_`. f via x==1 → true → 1.
        .{ .src = "fn f(b: bool) -> int { match b { true -> 1, false -> 0 } }\nfn main() -> int { return f(1 == 1) }\n", .name = "match_bool", .expect = 1 },
        // (m3) nested literal taken over a broader binding BY ORDER. C(0)→100.
        .{ .src = "enum E { C(int), N }\nfn f(e: E) -> int { match e { .C(0) -> 100, .C(r) -> r, .N -> 0 } }\nfn main() -> int { return f(E.C(0)) }\n", .name = "match_nested_zero", .expect = 100 },
        // (m3b) same match, the broader arm wins for a non-zero payload. C(5)→5.
        .{ .src = "enum E { C(int), N }\nfn f(e: E) -> int { match e { .C(0) -> 100, .C(r) -> r, .N -> 0 } }\nfn main() -> int { return f(E.C(5)) }\n", .name = "match_nested_bind", .expect = 5 },
        // (m4) nested ENUM pattern binds x two levels deep. Outer(Inner(7))→7.
        .{ .src = "enum In { V(int) }\nenum Out { O(In), Z }\nfn f(o: Out) -> int { match o { .O(.V(x)) -> x, .Z -> 0 } }\nfn main() -> int { return f(Out.O(In.V(7))) }\n", .name = "match_nested_enum", .expect = 7 },
        // (m5) or-pattern with no bindings. A|B → 1, C → 2. A→1.
        .{ .src = "enum E { A, B, C }\nfn f(e: E) -> int { match e { .A | .B -> 1, .C -> 2 } }\nfn main() -> int { return f(E.A) }\n", .name = "match_or_empty", .expect = 1 },
        // (m6) or-pattern with a SHARED binding. A(7)|B(7)→7; here B(9)→9.
        .{ .src = "enum E { A(int), B(int), N }\nfn f(e: E) -> int { match e { .A(x) | .B(x) -> x, .N -> 0 } }\nfn main() -> int { return f(E.B(9)) }\n", .name = "match_or_payload", .expect = 9 },
        // (m7) guard FALLS THROUGH to a later catch-all when false. n=5 → 2.
        .{ .src = "fn f(n: int) -> int { match n { _ if n > 10 -> 1, _ -> 2 } }\nfn main() -> int { return f(5) }\n", .name = "match_guard_false", .expect = 2 },
        // (m7b) guard TRUE takes the guarded arm. n=20 → 1.
        .{ .src = "fn f(n: int) -> int { match n { _ if n > 10 -> 1, _ -> 2 } }\nfn main() -> int { return f(20) }\n", .name = "match_guard_true", .expect = 1 },
        // (m8) first-match-wins with an overlapping guarded arm BEFORE the same
        // pattern unguarded. n=3 (not >10) falls to the second `_`. → 7.
        .{ .src = "fn f(n: int) -> int { match n { _ if n > 10 -> 1, _ -> 7 } }\nfn main() -> int { return f(3) }\n", .name = "match_first_wins", .expect = 7 },
        // (m9) a dense int match (0,1,2,3,_) → compare-chain dispatch. f(2)=22.
        .{ .src = "fn f(n: int) -> int { match n { 0 -> 20, 1 -> 21, 2 -> 22, 3 -> 23, _ -> 0 } }\nfn main() -> int { return f(2) }\n", .name = "match_dense", .expect = 22 },
        // (m10) FRAME canary: a guard with a deeply-nested sub-expression. The
        // guard 1+(2+(3+(4+(5+6)))) = 21 > r(=5) is TRUE → 1.
        .{ .src = "enum E { C(int), N }\nfn f(e: E) -> int { match e { .C(r) if (1 + (2 + (3 + (4 + (5 + 6))))) > r -> 1, _ -> 0 } }\nfn main() -> int { return f(E.C(5)) }\n", .name = "match_guard_nested", .expect = 1 },
        // (m11) FRAME canary: a guarded/nested match as a deep call arg. add3(10,20,
        // match B(6) { .B(r) if r > 0 -> 1+(2+(3+r)), ... }) = 10+20+12 = 42.
        .{ .src = "enum E { A(int), B(int) }\nfn add3(a: int, b: int, c: int) -> int { return a + b + c }\nfn main() -> int {\n return add3(10, 20, match E.B(6) { .B(r) if r > 0 -> 1 + (2 + (3 + r)), _ -> 0 })\n}\n", .name = "match_guard_deeparg", .expect = 42 },
        // (m12) FRAME canary: f(g(...)) where the inner match returns a >16B enum
        // via sret with a guarded arm, the outer sums it. pick guarded by s→true.
        .{ .src = "enum Sel { C, D }\nenum Big { A { p: int, q: int, r: int }, B(int) }\nfn pick(s: Sel) -> Big { match s { .C if 3 > 1 -> Big.A { p: 10, q: 20, r: 12 }, _ -> Big.B(0) } }\nfn consume(b: Big) -> int { match b { .A { p, q, r } -> p + q + r, .B(v) -> v } }\nfn main() -> int {\n return consume(pick(Sel.C))\n}\n", .name = "match_guard_sret_arg", .expect = 42 },
        // (m13) A match STATEMENT whose every arm body `return`s: the match type is
        // `never`, so lowering must emit each arm's return and seal the (unreachable)
        // join rather than fall through to the fn exit with no value.
        .{ .src = "enum Opt { some(int), none }\nfn pick(o: Opt) -> int {\n match o { .some(v) -> { return v }, .none -> { return 0 } }\n}\nfn main() -> int { return pick(Opt.some(42)) }\n", .name = "match_arm_return", .expect = 42 },
        // (m14) Same over a GENERIC enum instance — the App scrutinee reifies but the
        // arm-return lowering path is identical, so it must run to the payload too.
        .{ .src = "enum Opt[T] { some(T), none }\nfn pick(o: Opt[int]) -> int {\n match o { .some(v) -> { return v }, .none -> { return 0 } }\n}\nfn main() -> int { return pick(Opt[int].some(42)) }\n", .name = "match_arm_return_generic", .expect = 42 },
        // (m15) MIXED arms: one yields a value, one `return`s — the match type stays
        // `int`, exercising the scalar produce-into path's diverging-arm guard.
        .{ .src = "enum Opt { some(int), none }\nfn pick(o: Opt) -> int {\n x := match o { .some(v) -> v, .none -> { return 7 } }\n return x + 1\n}\nfn main() -> int { return pick(Opt.some(41)) }\n", .name = "match_arm_return_mixed", .expect = 42 },

        // Methods — each RUN proves STATIC method dispatch + self-by-value: a
        // wrong callee / a dropped or misplaced self arg would fault or mis-total.
        // (me1) `p.sum()` dispatches to the impl method; self passed by value → 42.
        .{ .src = "struct P { x: int, y: int }\nimpl P { fn sum(self) -> int { self.x + self.y } }\nfn main() -> int {\n p := P{ x: 20, y: 22 }\n return p.sum()\n}\n", .name = "method_sum", .expect = 42 },
        // (me2) a method with an extra struct param + a `Self`-returning method,
        // chained: a.scaled(2)={4,6}, then {4,6}.dot({4,5}) = 16+30 = 46.
        .{ .src = "struct Vec { x: int, y: int }\nimpl Vec {\n fn dot(self, o: Vec) -> int { self.x * o.x + self.y * o.y }\n fn scaled(self, k: int) -> Self { Vec{ x: self.x * k, y: self.y * k } }\n}\nfn main() -> int {\n a := Vec{ x: 2, y: 3 }\n b := Vec{ x: 4, y: 5 }\n s := a.scaled(2)\n return s.dot(b)\n}\n", .name = "method_args", .expect = 46 },
        // (me3) a >16B receiver (3 ints) passed by value as `self` via the indirect-arg
        // (sret-class) path; the method sums its fields → 6.
        .{ .src = "struct V3 { a: int, b: int, c: int }\nimpl V3 { fn total(self) -> int { self.a + self.b + self.c } }\nfn main() -> int {\n v := V3{ a: 1, b: 2, c: 3 }\n return v.total()\n}\n", .name = "method_bigself", .expect = 6 },

        // Integer literal bases: each decodes to 42 through an independent oracle
        // (hex 0x2A / octal 0o52 / binary 0b0010_1010). Base-0 parse + width lowering.
        .{ .src = "fn main() -> int {\n return 0x2A\n}\n", .name = "hexlit", .expect = 42 },
        .{ .src = "fn main() -> int {\n return 0o52\n}\n", .name = "octlit", .expect = 42 },
        .{ .src = "fn main() -> int {\n return 0b0010_1010\n}\n", .name = "binlit", .expect = 42 },
        // uint8 width + Eq across two bases + `_` grouping: 0b0010_1010 == 0o52 == 42.
        .{ .src = "fn main() -> int {\n x: uint8 = 0b0010_1010\n y: uint8 = 0o52\n return if x == y { 42 } else { 0 }\n}\n", .name = "uint8_bases", .expect = 42 },
        // Homogeneous-width result type: `c: int8 = a + b` must type as int8 (was
        // platform int under the old arm) so the annotated bind holds; 100+20=120 > 20.
        .{ .src = "fn main() -> int {\n a: int8 = 100\n b: int8 = 20\n c: int8 = a + b\n return if c > b { 42 } else { 0 }\n}\n", .name = "resultwidth", .expect = 42 },
        // Bare-literal sibling adoption (E3): `a + 1` adopts a's int8 width; 41+1=42.
        .{ .src = "fn main() -> int {\n a: int8 = 41\n c: int8 = a + 1\n d: int8 = 42\n return if c == d { 42 } else { 0 }\n}\n", .name = "bare_sibling", .expect = 42 },
        // Symmetric left-literal sibling adoption: `1 + a` must adopt a's int8 width
        // exactly as `a + 1` does — pins the l_lit branch of the binary arm.
        .{ .src = "fn main() -> int {\n a: int8 = 41\n c: int8 = 1 + a\n d: int8 = 42\n return if c == d { 42 } else { 0 }\n}\n", .name = "bare_sibling_left", .expect = 42 },
        // Both operands bare literals under a width annotation: `c: int8 = 1 + 2`
        // re-types each literal under the outer int8 expected so the annotated bind holds.
        .{ .src = "fn main() -> int {\n c: int8 = 40 + 2\n d: int8 = 42\n return if c == d { 42 } else { 0 }\n}\n", .name = "both_lit_width", .expect = 42 },
        // uint64 literal above 2^63: the checker admits it (u128 range gate) and
        // lowering now round-trips the full u64 into the iconst via bitcast — a plain i64
        // decode would have rejected it at codegen. 2^64-1 compares equal to itself → 42.
        .{ .src = "fn main() -> int {\n x: uint64 = 0xFFFFFFFFFFFFFFFF\n return if x == 0xFFFFFFFFFFFFFFFF { 42 } else { 0 }\n}\n", .name = "uint64_full_range", .expect = 42 },
        // WIDTH-CORRECT WRAPPING: 100+100 on two int8 overflows and wraps to -56
        // (sxtb). Observed via a SIGNED compare (never return the raw int8 — macOS masks
        // main's return to 8 bits). -56 < 100 is true only after the wrap → 42.
        .{ .src = "fn main() -> int {\n a: int8 = 100\n b: int8 = 100\n c: int8 = a + b\n return if c < b { 42 } else { 0 }\n}\n", .name = "wrap_int8", .expect = 42 },
        // Register-domain normalization: the wrapped sum feeds the icmp DIRECTLY (no
        // memory round-trip), so genArith's post-op sxtb is the only thing that makes
        // 200 wrap to -56. Without it, 200 < 100 is false → 0.
        .{ .src = "fn main() -> int {\n a: int8 = 100\n b: int8 = 100\n return if a + b < b { 42 } else { 0 }\n}\n", .name = "wrap_arith_value", .expect = 42 },
        // UNSIGNED compare: 2^63 > 1 is true UNSIGNED (ugt→hi) but false signed
        // (bit63 reads as negative). x must be uint64 so the operand type drives the
        // unsigned dispatch. → 42.
        .{ .src = "fn main() -> int {\n x: uint64 = 0x8000000000000000\n return if x > 1 { 42 } else { 0 }\n}\n", .name = "uint64_gt_unsigned", .expect = 42 },
        // UNSIGNED divide: 2^63 / 2 == 2^62 via udiv; sdiv would give 0xC000... The
        // sign-agnostic `==` isolates the divide from the compare. → 42.
        .{ .src = "fn main() -> int {\n x: uint64 = 0x8000000000000000\n y: uint64 = 2\n z: uint64 = x / y\n return if z == 0x4000000000000000 { 42 } else { 0 }\n}\n", .name = "udiv_unsigned", .expect = 42 },
        // NARROW STRUCT FIELD round-trip (semantic leg of the GOAL): the wrapped -56
        // stores into an int8 field and loads back canonical (8-byte-strided cell + 64-bit
        // str/ldr, sound because the stored value is already canonicalized). -56 < 0 → 42.
        .{ .src = "struct S { v: int8 }\nfn main() -> int {\n a: int8 = 100\n b: int8 = 100\n c: int8 = a + b\n s := S{ v: c }\n return if s.v < 0 { 42 } else { 0 }\n}\n", .name = "struct_narrow_field", .expect = 42 },
        // NARROW NEG re-wrap (signed leg): a+a=128 wraps to -128, then -(-128)=128
        // must re-wrap to -128 (sxtb on the neg result) for `d < 0` to hold. Without the
        // neg-leg normalize, d stays +128 → 0. Feeds the icmp directly (no memory hop).
        .{ .src = "fn main() -> int {\n a: int8 = 64\n c: int8 = a + a\n d: int8 = -c\n return if d < 0 { 42 } else { 0 }\n}\n", .name = "neg_int8_rewrap", .expect = 42 },
        // NARROW NEG re-wrap (unsigned/and-mask leg): -(5) = -5 masks to 251 in uint8.
        // The `and #0xff` on the neg result is the only thing that makes b == 251. → 42.
        .{ .src = "fn main() -> int {\n a: uint8 = 5\n b: uint8 = -a\n return if b == 251 { 42 } else { 0 }\n}\n", .name = "neg_uint8_mask", .expect = 42 },
        // UNSIGNED `<` (ult→lo): 1 < 2^63 is true unsigned, false signed (bit63<0). → 42.
        .{ .src = "fn main() -> int {\n x: uint64 = 0x8000000000000000\n return if 1 < x { 42 } else { 0 }\n}\n", .name = "uint64_lt_unsigned", .expect = 42 },
        // UNSIGNED `<=` (ule→ls): 1 <= 2^63 unsigned true, signed false. → 42.
        .{ .src = "fn main() -> int {\n x: uint64 = 0x8000000000000000\n return if 1 <= x { 42 } else { 0 }\n}\n", .name = "uint64_le_unsigned", .expect = 42 },
        // UNSIGNED `>=` (uge→hs): 2^63 >= 1 unsigned true, signed false. → 42.
        .{ .src = "fn main() -> int {\n x: uint64 = 0x8000000000000000\n return if x >= 1 { 42 } else { 0 }\n}\n", .name = "uint64_ge_unsigned", .expect = 42 },

        // Into[T] WIDENING: `small.into()` widens uint8→uint losslessly, driven by
        // the `wide:` annotation. 200 round-trips → 42.
        .{ .src = "fn main() -> int {\n small: uint8 = 200\n wide: uint = small.into()\n return if wide == 200 { 42 } else { 0 }\n}\n", .name = "into_widen", .expect = 42 },
        // TryInto[T] OK: 200 fits uint8 so `try_into().unwrap()` yields Ok(200) → 42.
        .{ .src = "fn main() -> int {\n wide: uint = 200\n back: uint8 = wide.try_into().unwrap()\n return if back == 200 { 42 } else { 0 }\n}\n", .name = "tryinto_ok", .expect = 42 },
        // TryInto[T] ERR (same-sign narrowing): 300 does NOT fit uint8 ⇒ Err ⇒
        // `unwrap_or(0)` gives 0. The negative range gate. → 0.
        .{ .src = "fn main() -> int {\n wide2: uint = 300\n narrow: uint8 = wide2.try_into().unwrap_or(0)\n return if narrow == 0 { 0 } else { 1 }\n}\n", .name = "tryinto_err", .expect = 0 },
        // TryInto[T] SIGN-FLIP ERR (cross-sign block): -1→uint has `fits` true (same
        // 64 bits) but v<0 signed ⇒ Err ⇒ `unwrap_or(9)` = 9. The exact case the
        // sign-check block exists for. → 42.
        .{ .src = "fn main() -> int {\n n: int = -1\n u: uint = n.try_into().unwrap_or(9)\n return if u == 9 { 42 } else { 0 }\n}\n", .name = "tryinto_signflip_err", .expect = 42 },
        // TryInto[T] SIGN-FLIP OK (cross-sign, same width, non-negative): 5→uint is
        // Ok(5). → 42.
        .{ .src = "fn main() -> int {\n n: int = 5\n u: uint = n.try_into().unwrap()\n return if u == 5 { 42 } else { 0 }\n}\n", .name = "tryinto_signflip_ok", .expect = 42 },
        // Into[T] SIGNED WIDEN of a NEGATIVE value: int16 -5 -> int drives the sxth
        // sign-extend arm of `normalizeWidth` (the unsigned cases only exercise uxt). The
        // widened register must read as -5 in the full 64-bit width. → 42.
        .{ .src = "fn main() -> int {\n a: int16 = -5\n w: int = a.into()\n return if w == -5 { 42 } else { 0 }\n}\n", .name = "into_signed_widen_neg", .expect = 42 },
        // Into[T] target from a RETURN type: `a.into()` widens uint8->uint with the
        // conversion target derived from `widen`'s return type, not a let-annotation. → 42.
        .{ .src = "fn widen(a: uint8) -> uint {\n return a.into()\n}\nfn main() -> int {\n return if widen(200) == 200 { 42 } else { 0 }\n}\n", .name = "into_ret_pos", .expect = 42 },
        // TryInto[T] target from a RETURN type: the `try_into().unwrap()` payload
        // target flows from `narrow`'s uint8 return type. 200 fits → Ok(200). → 42.
        .{ .src = "fn narrow(w: uint) -> uint8 {\n return w.try_into().unwrap()\n}\nfn main() -> int {\n return if narrow(200) == 200 { 42 } else { 0 }\n}\n", .name = "tryinto_ret_pos", .expect = 42 },
        // Into[T] target from a CALL-ARGUMENT slot: `small.into()` widens uint8->uint
        // with the target derived from `takes`'s param type. → 42.
        .{ .src = "fn takes(u: uint) -> int {\n return if u == 200 { 42 } else { 0 }\n}\nfn main() -> int {\n small: uint8 = 200\n return takes(small.into())\n}\n", .name = "into_arg_slot", .expect = 42 },

        // Derived `Ord` on an UNSIGNED field compares UNSIGNED. hi has the high bit set
        // (2^63), so a signed field compare would rank it below lo and make `a < b` true;
        // unsigned, 2^63 > 1, so `a < b` is FALSE → 0.
        .{ .src = "struct W { v: uint64 }\nfn main() -> int {\n hi: uint64 = 0x8000000000000000\n lo: uint64 = 1\n a := W{ v: hi }\n b := W{ v: lo }\n return if a < b { 1 } else { 0 }\n}\n", .name = "derive_ord_unsigned_lt", .expect = 0 },
        // Positive direction of the same: `a > b` IS true unsigned (2^63 > 1) → 42.
        .{ .src = "struct W { v: uint64 }\nfn main() -> int {\n hi: uint64 = 0x8000000000000000\n lo: uint64 = 1\n a := W{ v: hi }\n b := W{ v: lo }\n return if a > b { 42 } else { 0 }\n}\n", .name = "derive_ord_unsigned_gt", .expect = 42 },
        // A HEX literal in a `match` pattern decodes base-aware: 0x2A == 42 matches → 1.
        // A decimal-only pattern decoder would read 0x2A as 0 and fall to the wildcard.
        .{ .src = "fn main() -> int {\n x := 42\n return match x { 0x2A -> 1, _ -> 0 }\n}\n", .name = "match_hex_pattern", .expect = 1 },
        // Same hex pattern that must NOT match (scrutinee 7 ≠ 0x2A) → wildcard → 0.
        .{ .src = "fn main() -> int {\n x := 7\n return match x { 0x2A -> 1, _ -> 0 }\n}\n", .name = "match_hex_pattern_miss", .expect = 0 },
        // A narrow-int scrutinee accepts int literal patterns: the pattern `0` adopts the
        // int8 width instead of platform int, matches → 1.
        .{ .src = "fn main() -> int {\n x: int8 = 0\n return match x { 0 -> 1, _ -> 2 }\n}\n", .name = "match_narrow_int", .expect = 1 },
        // Same narrow-int match falling to the wildcard (5 ≠ 0) → 2.
        .{ .src = "fn main() -> int {\n x: int8 = 5\n return match x { 0 -> 1, _ -> 2 }\n}\n", .name = "match_narrow_int_wild", .expect = 2 },
        // The most-negative signed literals are writable (magnitude one past the positive
        // max). -128:int8 stored to a field loads back < 0 → 42.
        .{ .src = "struct S { v: int8 }\nfn main() -> int {\n a: int8 = -128\n s := S{ v: a }\n return if s.v < 0 { 42 } else { 0 }\n}\n", .name = "int_min_int8", .expect = 42 },
    };

    for (cases, 0..) |c, i| {
        const src_path = std.fmt.allocPrint(gpa, "{s}/{s}.toy", .{ dir_name, c.name }) catch unreachable;
        defer gpa.free(src_path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = src_path, .data = c.src });

        var r: FileResult = .{ .path = src_path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, i);
        defer r.deinit(gpa);
        try testing.expect(r.err == null);

        var lowered = try lowerSingleFile(gpa, io, cache, "aarch64-macos", &r, .normal, .O0);
        const lp = switch (lowered) {
            .ok => |*ok| ok,
            .err => return error.TestUnexpectedResult,
        };
        defer lp.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), lp.diags.len);

        const image = try buildImage(io, gpa, c.name, lp.text, lp.entry_off, lp.cstrings, lp.data_relocs);
        defer gpa.free(image);

        const out_path = std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir_name, c.name }) catch unreachable;
        defer gpa.free(out_path);
        {
            const perms: Io.File.Permissions = .fromMode(0o755);
            var f = try Io.Dir.cwd().createFile(io, out_path, .{ .permissions = perms });
            defer f.close(io);
            try f.writeStreamingAll(io, image);
            try f.setPermissions(io, perms);
        }

        // The signature we embedded must satisfy the system verifier, otherwise
        // the kernel would refuse to exec it below. Check it explicitly so a
        // signing regression fails here with a clear cause rather than as a
        // mysterious spawn error.
        const abs = try Io.Dir.cwd().realPathFileAlloc(io, out_path, gpa);
        defer gpa.free(abs);
        {
            var cs = try std.process.spawn(io, .{ .argv = &.{ "codesign", "-v", abs } });
            const cs_term = try cs.wait(io);
            try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, cs_term);
        }

        // Spawn the executable by absolute path and check its exit code.
        var child = try std.process.spawn(io, .{ .argv = &.{abs} });
        const term = try child.wait(io);
        try testing.expectEqual(std.process.Child.Term{ .exited = c.expect }, term);
    }
}

test "integration: monomorphized generic instances get distinct symbols, run to exit 42, and cost zero units when uncalled" {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_name_buf: [64]u8 = undefined;
    const dir_name = std.fmt.bufPrint(&dir_name_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    // Case A: the exact demo — id[int](7) + id[P]({20,15}) → 7 + 20 + 15 = 42.
    // The struct-typed instance exercises the reg-pair/indirect struct ABI unchanged.
    {
        const src =
            "fn id[T](x: T) -> T { x }\n" ++
            "struct P { x: int, y: int }\n" ++
            "fn main() -> int {\n a := id[int](7)\n p := id[P](P{ x: 20, y: 15 })\n return a + p.x + p.y\n}\n";
        const src_path = std.fmt.allocPrint(gpa, "{s}/idexp.toy", .{dir_name}) catch unreachable;
        defer gpa.free(src_path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = src_path, .data = src });

        var r: FileResult = .{ .path = src_path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, 0);
        defer r.deinit(gpa);
        try testing.expect(r.err == null);

        // Two reachable instances, in canonical order (scalar kind < struct kind),
        // with DISTINCT mangled symbols.
        const insts = r.typecheck.?.instances;
        try testing.expectEqual(@as(usize, 2), insts.len);
        try testing.expectEqualStrings("main.id$int", insts[0].name.?);
        try testing.expect(!std.mem.eql(u8, insts[0].name.?, insts[1].name.?));

        // Force a fresh lowering: main + 2 instances = exactly 3 codegen units.
        var lowered = try lowerSingleFile(gpa, io, cache, "aarch64-macos", &r, .force, .O0);
        const lp = switch (lowered) {
            .ok => |*ok| ok,
            .err => return error.TestUnexpectedResult,
        };
        defer lp.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), lp.diags.len);
        try testing.expectEqual(@as(usize, 3), lp.codegen_compiled);

        const image = try buildImage(io, gpa, "idexp", lp.text, lp.entry_off, lp.cstrings, lp.data_relocs);
        defer gpa.free(image);
        const out_path = std.fmt.allocPrint(gpa, "{s}/idexp", .{dir_name}) catch unreachable;
        defer gpa.free(out_path);
        {
            const perms: Io.File.Permissions = .fromMode(0o755);
            var f = try Io.Dir.cwd().createFile(io, out_path, .{ .permissions = perms });
            defer f.close(io);
            try f.writeStreamingAll(io, image);
            try f.setPermissions(io, perms);
        }
        const abs = try Io.Dir.cwd().realPathFileAlloc(io, out_path, gpa);
        defer gpa.free(abs);
        var child = try std.process.spawn(io, .{ .argv = &.{abs} });
        const term = try child.wait(io);
        try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, term);
    }

    // Case B: an UNCALLED generic fn mints zero instances and emits zero extra units
    // (free in the binary). Only `main` is lowered.
    {
        const src = "fn unused[T](x: T) -> T { x }\nfn main() -> int {\n return 9\n}\n";
        const src_path = std.fmt.allocPrint(gpa, "{s}/uncalled.toy", .{dir_name}) catch unreachable;
        defer gpa.free(src_path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = src_path, .data = src });

        var r: FileResult = .{ .path = src_path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, 1);
        defer r.deinit(gpa);
        try testing.expect(r.err == null);
        try testing.expectEqual(@as(usize, 0), r.typecheck.?.instances.len);

        var lowered = try lowerSingleFile(gpa, io, cache, "aarch64-macos", &r, .force, .O0);
        const lp = switch (lowered) {
            .ok => |*ok| ok,
            .err => return error.TestUnexpectedResult,
        };
        defer lp.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), lp.diags.len);
        try testing.expectEqual(@as(usize, 1), lp.codegen_compiled); // just `main`
    }
}

test "integration: a derived-Eq struct runs to exit 42, mints one source-less unit, and costs zero when uncalled" {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_name_buf: [64]u8 = undefined;
    const dir_name = std.fmt.bufPrint(&dir_name_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    // Case A: an all-Eq-fields struct compared with `==` derives structurally (no impl)
    // and runs to 42. main + one SOURCE-LESS derive unit = 2 codegen units.
    {
        const src =
            "struct P { x: int, y: int }\n" ++
            "fn main() -> int {\n a := P{ x: 1, y: 2 }\n b := P{ x: 1, y: 2 }\n return if a == b { 42 } else { 0 }\n}\n";
        const src_path = std.fmt.allocPrint(gpa, "{s}/deq.toy", .{dir_name}) catch unreachable;
        defer gpa.free(src_path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = src_path, .data = src });

        var r: FileResult = .{ .path = src_path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, 0);
        defer r.deinit(gpa);
        try testing.expect(r.err == null);

        // Exactly one synthetic recipe (Eq for struct id 0), canonically named.
        const derives = r.typecheck.?.derives;
        try testing.expectEqual(@as(usize, 1), derives.len);
        try testing.expectEqualStrings("Eq$eq$s0", derives[0].name.?);

        var lowered = try lowerSingleFile(gpa, io, cache, "aarch64-macos", &r, .force, .O0);
        const lp = switch (lowered) {
            .ok => |*ok| ok,
            .err => return error.TestUnexpectedResult,
        };
        defer lp.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), lp.diags.len);
        try testing.expectEqual(@as(usize, 2), lp.codegen_compiled); // main + Eq$eq$s0

        const image = try buildImage(io, gpa, "deq", lp.text, lp.entry_off, lp.cstrings, lp.data_relocs);
        defer gpa.free(image);
        const out_path = std.fmt.allocPrint(gpa, "{s}/deq", .{dir_name}) catch unreachable;
        defer gpa.free(out_path);
        {
            const perms: Io.File.Permissions = .fromMode(0o755);
            var f = try Io.Dir.cwd().createFile(io, out_path, .{ .permissions = perms });
            defer f.close(io);
            try f.writeStreamingAll(io, image);
            try f.setPermissions(io, perms);
        }
        const abs = try Io.Dir.cwd().realPathFileAlloc(io, out_path, gpa);
        defer gpa.free(abs);
        var child = try std.process.spawn(io, .{ .argv = &.{abs} });
        const term = try child.wait(io);
        try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, term);
    }

    // Case B: a derivable struct never compared with `==` mints ZERO derive units
    // (lazy — free in the binary). Only `main` is lowered.
    {
        const src = "struct Q { v: int }\nfn main() -> int {\n q := Q{ v: 9 }\n return q.v\n}\n";
        const src_path = std.fmt.allocPrint(gpa, "{s}/dunused.toy", .{dir_name}) catch unreachable;
        defer gpa.free(src_path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = src_path, .data = src });

        var r: FileResult = .{ .path = src_path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, 1);
        defer r.deinit(gpa);
        try testing.expect(r.err == null);
        try testing.expectEqual(@as(usize, 0), r.typecheck.?.derives.len);

        var lowered = try lowerSingleFile(gpa, io, cache, "aarch64-macos", &r, .force, .O0);
        const lp = switch (lowered) {
            .ok => |*ok| ok,
            .err => return error.TestUnexpectedResult,
        };
        defer lp.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), lp.codegen_compiled); // just `main`
    }
}

test "integration: print writes the expected bytes to stdout" {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_name_buf: [64]u8 = undefined;
    const dir_name = std.fmt.bufPrint(&dir_name_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    const Case = struct { src: []const u8, name: []const u8, want: []const u8 };
    const cases = [_]Case{
        .{ .src = "import std/io\nfn main() {\n io.print(\"hello world\\n\")\n}\n", .name = "hw", .want = "hello world\n" },
        // A str local + a second literal: two distinct cstrings + a 16-byte slot.
        .{ .src = "import std/io\nfn main() {\n io.print(\"AB\")\n s := \"CD\\n\"\n io.print(s)\n}\n", .name = "two", .want = "ABCD\n" },
    };

    _ = cache;
    for (cases, 0..) |c, i| {
        _ = i;
        const src_path = std.fmt.allocPrint(gpa, "{s}/{s}.toy", .{ dir_name, c.name }) catch unreachable;
        defer gpa.free(src_path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = src_path, .data = c.src });

        // `io.print` is ordinary library code in `std/io`, so these programs import a
        // module — the WHOLE-BINARY build (with graph discovery) is the only path that
        // follows imports; the single-file `pipeline` cannot. Build via the real `toy`.
        const out_path = std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir_name, c.name }) catch unreachable;
        defer gpa.free(out_path);
        {
            const res = try spawnToy(gpa, io, &.{ "build", src_path, "-o", out_path });
            defer gpa.free(res.out);
            try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, res.term);
        }

        const abs = try Io.Dir.cwd().realPathFileAlloc(io, out_path, gpa);
        defer gpa.free(abs);

        // Spawn capturing stdout; read to EOF (child closes it on exit), then reap.
        var child = try std.process.spawn(io, .{ .argv = &.{abs}, .stdout = .pipe });
        var rdr = child.stdout.?.readerStreaming(io, &.{});
        const got = try rdr.interface.allocRemaining(gpa, .limited(4096));
        defer gpa.free(got);
        const term = try child.wait(io);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, term);
        try testing.expectEqualStrings(c.want, got);
    }
}

/// Lower `src` (written to `path`) through the codegen cache in `cache`, then
/// return the linked program. Caller owns the result (deinit) and `r` (deinit).
fn checkAndLower(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    path: []const u8,
    src: []const u8,
    mode: Engine.Mode,
    r_out: *FileResult,
) !LinkedProgram {
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });
    r_out.* = .{ .path = path };
    try pipeline(gpa, io, cache, .check, "aarch64-macos", r_out, 0);
    try testing.expect(r_out.err == null);
    const lowered = try lowerSingleFile(gpa, io, cache, "aarch64-macos", r_out, mode, .O0);
    return switch (lowered) {
        .ok => |ok| ok,
        .err => error.TestUnexpectedResult,
    };
}

test "edit-one-fn: only the edited fn recompiles; callers unaffected by a body change" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // 2 fns: add + main calling it. Cold build: both compiled, none cached.
    const v1 = "fn add(a: int, b: int) -> int {\n return a + b\n}\nfn main() -> int {\n return add(40, 2)\n}\n";
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, v1, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp1.codegen_compiled);
    try testing.expectEqual(@as(usize, 0), lp1.codegen_cached);

    // Edit ONLY add's body (a+b -> a+b+0); main's source + signature unchanged.
    // main's fingerprint folds add's SIGNATURE only, so main is NOT recompiled.
    const v2 = "fn add(a: int, b: int) -> int {\n return a + b + 0\n}\nfn main() -> int {\n return add(40, 2)\n}\n";
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, v2, .normal, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    // Exactly one fn recompiles (add); main is a cache hit.
    try testing.expectEqual(@as(usize, 1), lp2.codegen_compiled);
    try testing.expectEqual(@as(usize, 1), lp2.codegen_cached);

    // Now change add's SIGNATURE (add a 3rd param). main calls add, so its
    // fingerprint folds add's sig → main MUST also recompile.
    const v3 = "fn add(a: int, b: int, c: int) -> int {\n return a + b\n}\nfn main() -> int {\n return add(40, 2, 0)\n}\n";
    var r3: FileResult = undefined;
    var lp3 = try checkAndLower(gpa, io, cache, path, v3, .normal, &r3);
    defer r3.deinit(gpa);
    defer lp3.deinit(gpa);
    // main's call-arg count also changed here, so main recompiles regardless; the
    // point proven by v1->v2 is the body-change isolation. Both recompile now.
    try testing.expectEqual(@as(usize, 2), lp3.codegen_compiled);
    try testing.expectEqual(@as(usize, 0), lp3.codegen_cached);
}

test "verify-mode: re-lowering every fn matches the cached blob" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    const src = "fn add(a: int, b: int) -> int {\n return a + b\n}\nfn main() -> int {\n return add(40, 2)\n}\n";
    // Cold build to populate the cache.
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, src, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);

    // VERIFY: every cache hit is re-lowered and asserted byte-identical to its
    // cached blob. A mismatch would panic (std.debug.assert) inside the job.
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, src, .verify, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp2.codegen_cached);
}

test "cache soundness: a method body edit recompiles only the method; a method return-type edit recompiles its caller" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // `main` calls the method `m` for effect (result discarded), so a change to m's
    // RETURN TYPE keeps main's source valid while altering what main must codegen —
    // the stale-cache hazard the method-sig fold guards. Cold: 2 units (m + main).
    const v1 = "struct P { x: int }\nimpl P { fn m(self) -> int { self.x } }\nfn main() -> int {\n p := P{ x: 41 }\n p.m()\n return 0\n}\n";
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, v1, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp1.codegen_compiled);
    try testing.expectEqual(@as(usize, 0), lp1.codegen_cached);

    // Edit ONLY m's body (self.x -> self.x + 0). main's source, the call, and the
    // call node's type (int) are unchanged, so main is a cache HIT; only m recompiles.
    const v2 = "struct P { x: int }\nimpl P { fn m(self) -> int { self.x + 0 } }\nfn main() -> int {\n p := P{ x: 41 }\n p.m()\n return 0\n}\n";
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, v2, .normal, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), lp2.codegen_compiled);
    try testing.expectEqual(@as(usize, 1), lp2.codegen_cached);

    // Change m's RETURN TYPE (int -> ()). main still compiles (the discarded call is
    // valid either way), but its codegen depends on the callee's result ABI, so main
    // MUST recompile. A missing method-sig/return fold would stale-hit main here
    // (compiled==1) — a wrong-return miscompile. Both units recompile => compiled==2.
    const v3 = "struct P { x: int }\nimpl P { fn m(self) -> () { } }\nfn main() -> int {\n p := P{ x: 41 }\n p.m()\n return 0\n}\n";
    var r3: FileResult = undefined;
    var lp3 = try checkAndLower(gpa, io, cache, path, v3, .normal, &r3);
    defer r3.deinit(gpa);
    defer lp3.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp3.codegen_compiled);
    try testing.expectEqual(@as(usize, 0), lp3.codegen_cached);
}

test "cache soundness: editing a value-if fn recompiles only it; verify passes" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // g uses a value-if + bare-block expr; main calls g. Cold build: both compiled.
    const v1 = "fn g(n: int) -> int {\n x := { a := if n < 0 { 1 } else { 2 }\n a + 1 }\n return x\n}\nfn main() -> int {\n return g(5)\n}\n";
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, v1, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp1.codegen_compiled);
    try testing.expectEqual(@as(usize, 0), lp1.codegen_cached);

    // Edit ONLY g's then-arm value (1 -> 3). main's sig/source unchanged, so main
    // is a cache hit; only g recompiles. An unwalked new tag would be a stale hit,
    // failing compiled==1 (it would falsely cache g).
    const v2 = "fn g(n: int) -> int {\n x := { a := if n < 0 { 3 } else { 2 }\n a + 1 }\n return x\n}\nfn main() -> int {\n return g(5)\n}\n";
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, v2, .normal, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), lp2.codegen_compiled);
    try testing.expectEqual(@as(usize, 1), lp2.codegen_cached);

    // VERIFY: re-lower every cache hit and assert byte-identical to the cached blob.
    var r3: FileResult = undefined;
    var lp3 = try checkAndLower(gpa, io, cache, path, v2, .verify, &r3);
    defer r3.deinit(gpa);
    defer lp3.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp3.codegen_cached);
}

test "cache soundness: editing a loop/for/break fn recompiles only it; verify passes" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // g uses a for-loop AND a value-loop with a break-constant; main calls g.
    const v1 = "fn g(n: int) -> int {\n s := 0\n for i in 0..n {\n s = s + i\n }\n v := loop {\n break 7\n }\n return s + v\n}\nfn main() -> int {\n return g(3)\n}\n";
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, v1, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp1.codegen_compiled);
    try testing.expectEqual(@as(usize, 0), lp1.codegen_cached);

    // Edit ONLY g's break constant (7 -> 9). main is a cache hit; only g recompiles.
    // A new tag missing from the fingerprint/walkers would be a stale hit (compiled==0).
    const v2 = "fn g(n: int) -> int {\n s := 0\n for i in 0..n {\n s = s + i\n }\n v := loop {\n break 9\n }\n return s + v\n}\nfn main() -> int {\n return g(3)\n}\n";
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, v2, .normal, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), lp2.codegen_compiled);
    try testing.expectEqual(@as(usize, 1), lp2.codegen_cached);

    // VERIFY: re-lower every cache hit and assert byte-identical to the cached blob.
    var r3: FileResult = undefined;
    var lp3 = try checkAndLower(gpa, io, cache, path, v2, .verify, &r3);
    defer r3.deinit(gpa);
    defer lp3.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp3.codegen_cached);
}

test "cache soundness: editing a labeled/break fn recompiles only it; verify passes" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // g uses a labeled value-loop with a `break @L` constant; main calls g.
    const v1 = "fn g() -> int {\n @L loop {\n break @L 7\n }\n}\nfn main() -> int {\n return g()\n}\n";
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, v1, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp1.codegen_compiled);
    try testing.expectEqual(@as(usize, 0), lp1.codegen_cached);

    // Edit ONLY g's break value (7 -> 9). main is a cache hit; only g recompiles.
    // The new label/break tags must be fingerprint-reachable AND walkTouchedSig must
    // descend the labeled wrapper, else g would be a STALE HIT (compiled==0).
    const v2 = "fn g() -> int {\n @L loop {\n break @L 9\n }\n}\nfn main() -> int {\n return g()\n}\n";
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, v2, .normal, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), lp2.codegen_compiled);
    try testing.expectEqual(@as(usize, 1), lp2.codegen_cached);

    // Edit ONLY the label NAME used by the break-target (@L -> @M). The break still
    // targets the (renamed) enclosing loop, so the program is equivalent; but the
    // folded label text differs, so g recompiles (the conservative, sound choice).
    const v3 = "fn g() -> int {\n @M loop {\n break @M 9\n }\n}\nfn main() -> int {\n return g()\n}\n";
    var r3: FileResult = undefined;
    var lp3 = try checkAndLower(gpa, io, cache, path, v3, .normal, &r3);
    defer r3.deinit(gpa);
    defer lp3.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), lp3.codegen_compiled);
    try testing.expectEqual(@as(usize, 1), lp3.codegen_cached);

    // VERIFY: re-lower every cache hit and assert byte-identical to the cached blob.
    var r4: FileResult = undefined;
    var lp4 = try checkAndLower(gpa, io, cache, path, v3, .verify, &r4);
    defer r4.deinit(gpa);
    defer lp4.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp4.codegen_cached);
}

test "cache soundness: editing a struct's fields recompiles every fn that touches it; verify passes" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // `area` touches Point (param); `main` constructs Point; `other` does NOT.
    const v1 = "struct Point { x: int }\nfn area(p: Point) -> int { p.x }\nfn other() -> int { 5 }\nfn main() -> int {\n p := Point { x: 42 }\n return area(p) + other() - 5\n}\n";
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, v1, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);
    try testing.expectEqual(@as(usize, 3), lp1.codegen_compiled);
    try testing.expectEqual(@as(usize, 0), lp1.codegen_cached);

    // Add a field to Point (a LAYOUT change). The bodies of area/other/main are
    // byte-identical, but area+main TOUCH Point, so they MUST recompile (a stale
    // hit would miscompile against the old 8-byte layout). `other` stays cached.
    // This is the "touched type layouts" hook made REAL.
    const v2 = "struct Point { x: int, y: int }\nfn area(p: Point) -> int { p.x }\nfn other() -> int { 5 }\nfn main() -> int {\n p := Point { x: 42, y: 0 }\n return area(p) + other() - 5\n}\n";
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, v2, .normal, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp2.codegen_compiled); // area + main
    try testing.expectEqual(@as(usize, 1), lp2.codegen_cached); // other

    // VERIFY: re-lower every cache hit and assert byte-identical to the cached blob.
    var r3: FileResult = undefined;
    var lp3 = try checkAndLower(gpa, io, cache, path, v2, .verify, &r3);
    defer r3.deinit(gpa);
    defer lp3.deinit(gpa);
    try testing.expectEqual(@as(usize, 3), lp3.codegen_cached);
}

test "cache soundness: a struct touched ONLY via a param/return type folds its layout (no stale hit across the ABI boundary)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // `consume` touches Box ONLY via its param TYPE (its body never reads a field,
    // so no body node carries the struct type). Box starts 16B (reg-pair ABI).
    const v1 = "struct Box { a: int, b: int }\nfn consume(p: Box) -> int { return 7 }\nfn main() -> int {\n b := Box { a: 1, b: 2 }\n return consume(b)\n}\n";
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, v1, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp1.codegen_compiled);

    // Add a 3rd field (16B reg-pair -> 24B INDIRECT, an ABI-class change). `consume`
    // touches Box only via its param type, so its fingerprint MUST fold the layout
    // (the param/ret-only fold in walkTouchedSig). A stale hit here would bake in the
    // wrong ABI. Both consume + main recompile.
    const v2 = "struct Box { a: int, b: int, c: int }\nfn consume(p: Box) -> int { return 7 }\nfn main() -> int {\n b := Box { a: 1, b: 2, c: 3 }\n return consume(b)\n}\n";
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, v2, .normal, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp2.codegen_compiled); // consume + main, NOT a stale hit
    try testing.expectEqual(@as(usize, 0), lp2.codegen_cached);

    // VERIFY: re-lower every cache hit and assert byte-identical (no stale blob).
    var r3: FileResult = undefined;
    var lp3 = try checkAndLower(gpa, io, cache, path, v2, .verify, &r3);
    defer r3.deinit(gpa);
    defer lp3.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp3.codegen_cached);
}

test "cache soundness: editing an enum's variants recompiles every fn that touches it; verify passes" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // `area` matches Shape; `main` constructs Shape; `other` does NOT touch it.
    const v1 = "enum Shape { C(int), R { w: int, h: int } }\nfn area(s: Shape) -> int { match s { .C(r) -> r, .R { w, h } -> w * h } }\nfn other() -> int { 5 }\nfn main() -> int {\n s := Shape.R { w: 6, h: 7 }\n return area(s) + other() - 5\n}\n";
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, v1, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);
    try testing.expectEqual(@as(usize, 3), lp1.codegen_compiled);
    try testing.expectEqual(@as(usize, 0), lp1.codegen_cached);

    // Add a field to the R variant (a LAYOUT change crossing the 16B↔24B ABI
    // boundary for Shape). area + main TOUCH Shape → MUST recompile; other stays.
    const v2 = "enum Shape { C(int), R { w: int, h: int, d: int } }\nfn area(s: Shape) -> int { match s { .C(r) -> r, .R { w, h, d } -> w * h * d } }\nfn other() -> int { 5 }\nfn main() -> int {\n s := Shape.R { w: 6, h: 7, d: 1 }\n return area(s) + other() - 5\n}\n";
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, v2, .normal, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp2.codegen_compiled); // area + main
    try testing.expectEqual(@as(usize, 1), lp2.codegen_cached); // other

    var r3: FileResult = undefined;
    var lp3 = try checkAndLower(gpa, io, cache, path, v2, .verify, &r3);
    defer r3.deinit(gpa);
    defer lp3.deinit(gpa);
    try testing.expectEqual(@as(usize, 3), lp3.codegen_cached);
}

test "cache soundness: an enum touched ONLY via a param type folds its layout across the 16B<->24B ABI boundary" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // `consume` touches E ONLY via its param TYPE (its body never matches/reads it).
    // E starts 16B (tag + one int → reg-pair ABI).
    const v1 = "enum E { A(int), B }\nfn consume(e: E) -> int { return 7 }\nfn main() -> int {\n e := E.B\n return consume(e)\n}\n";
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, v1, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp1.codegen_compiled);

    // Grow the A payload to a 3-int struct variant → E becomes 32B (indirect ABI).
    // `consume` reads no field, but its PARAM ABI changed — it MUST recompile.
    const v2 = "enum E { A { p: int, q: int, r: int }, B }\nfn consume(e: E) -> int { return 7 }\nfn main() -> int {\n e := E.B\n return consume(e)\n}\n";
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, v2, .normal, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp2.codegen_compiled); // both touch E
    try testing.expectEqual(@as(usize, 0), lp2.codegen_cached);

    var r3: FileResult = undefined;
    var lp3 = try checkAndLower(gpa, io, cache, path, v2, .verify, &r3);
    defer r3.deinit(gpa);
    defer lp3.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lp3.codegen_cached);
}

test "errors: non-exhaustive/unknown variant/arity/type; recursive enum; uninferable .V" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-driver-m10-err";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    const cases = [_][]const u8{
        "enum S { A, B }\nfn f(s: S) -> int { match s { .A -> 1 } }\nfn main() -> int { return f(S.A) }\n", // non-exhaustive
        "enum S { A, B }\nfn main() -> int { x := S.Nope\n return 0 }\n", // unknown variant
        "enum S { C(int) }\nfn main() -> int { x := S.C(1, 2)\n return 0 }\n", // wrong arity
        "enum S { C(int) }\nfn main() -> int { x := S.C(true)\n return 0 }\n", // wrong payload type
        "enum R { A(R), B }\nfn main() -> int { return 0 }\n", // directly-recursive enum
        "enum A { X(B), Y }\nenum B { Z(A), W }\nfn main() -> int { return 0 }\n", // indirect-recursive
        "enum S { A }\nfn main() -> int { x := .A\n return 0 }\n", // inferred .V, no expected type
    };
    for (cases, 0..) |src, i| {
        const path = std.fmt.allocPrint(gpa, "{s}/e{d}.toy", .{ dir_name, i }) catch unreachable;
        defer gpa.free(path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });
        var r: FileResult = .{ .path = path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, i);
        defer r.deinit(gpa);
        try testing.expectEqual(@as(?anyerror, error.TypeError), r.err);
        try testing.expect(r.typecheck != null);
        try testing.expect(r.typecheck.?.diags.len > 0);
    }
}

test "errors: non-exhaustive int/bool; guarded-only/partial-nested variant; inconsistent or-bindings; non-bool guard" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-driver-m11-err";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    const cases = [_][]const u8{
        "fn f(n: int) -> int { match n { 0 -> 1, 1 -> 2 } }\nfn main() -> int { return f(0) }\n", // int, no `_`
        "fn f(b: bool) -> int { match b { true -> 1 } }\nfn main() -> int { return f(1 == 1) }\n", // bool missing false
        "enum E { C(int), N }\nfn f(e: E) -> int { match e { .C(r) if r > 0 -> 1, .N -> 0 } }\nfn main() -> int { return f(E.N) }\n", // C covered only by a guarded arm
        "enum E { C(int), N }\nfn f(e: E) -> int { match e { .C(0) -> 1, .N -> 0 } }\nfn main() -> int { return f(E.N) }\n", // C covered only by a partial nested pattern
        "enum E { A(int), B(int) }\nfn f(e: E) -> int { match e { .A(x) | .B(y) -> 1 } }\nfn main() -> int { return f(E.A(1)) }\n", // or-pattern: different names
        "enum E { A(int), B }\nfn f(e: E) -> int { match e { .A(x) | .B -> 1 } }\nfn main() -> int { return f(E.B) }\n", // or-pattern: one binds, one doesn't
        "fn f(n: int) -> int { match n { _ if n -> 1, _ -> 0 } }\nfn main() -> int { return f(0) }\n", // guard not bool
    };
    for (cases, 0..) |src, i| {
        const path = std.fmt.allocPrint(gpa, "{s}/e{d}.toy", .{ dir_name, i }) catch unreachable;
        defer gpa.free(path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });
        var r: FileResult = .{ .path = path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, i);
        defer r.deinit(gpa);
        try testing.expectEqual(@as(?anyerror, error.TypeError), r.err);
        try testing.expect(r.typecheck != null);
        try testing.expect(r.typecheck.?.diags.len > 0);
    }
}

test "cache soundness: editing a literal/guard recompiles only that fn; verify byte-identical" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // `f` matches; `other` is a sibling that must stay cached across edits.
    const v1 = "fn f(n: int) -> int { match n { 0 -> 1, _ -> 2 } }\nfn other() -> int { 5 }\nfn main() -> int {\n return f(0) + other() - 5\n}\n";
    var r1: FileResult = undefined;
    var lp1 = try checkAndLower(gpa, io, cache, path, v1, .normal, &r1);
    defer r1.deinit(gpa);
    defer lp1.deinit(gpa);
    try testing.expectEqual(@as(usize, 3), lp1.codegen_compiled);
    try testing.expectEqual(@as(usize, 0), lp1.codegen_cached);

    // Edit the literal VALUE 0→1: f's pattern AST changes → f recompiles; other cached.
    const v2 = "fn f(n: int) -> int { match n { 1 -> 1, _ -> 2 } }\nfn other() -> int { 5 }\nfn main() -> int {\n return f(0) + other() - 5\n}\n";
    var r2: FileResult = undefined;
    var lp2 = try checkAndLower(gpa, io, cache, path, v2, .normal, &r2);
    defer r2.deinit(gpa);
    defer lp2.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), lp2.codegen_compiled); // f
    try testing.expectEqual(@as(usize, 2), lp2.codegen_cached); // other + main

    // Add a GUARD to f's first arm → f recompiles again; other stays cached.
    const v3 = "fn f(n: int) -> int { match n { 1 if n > 0 -> 1, _ -> 2 } }\nfn other() -> int { 5 }\nfn main() -> int {\n return f(0) + other() - 5\n}\n";
    var r3: FileResult = undefined;
    var lp3 = try checkAndLower(gpa, io, cache, path, v3, .normal, &r3);
    defer r3.deinit(gpa);
    defer lp3.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), lp3.codegen_compiled); // f
    try testing.expectEqual(@as(usize, 2), lp3.codegen_cached);

    // Verify: re-lower every fn against the cache; all bytes identical.
    var r4: FileResult = undefined;
    var lp4 = try checkAndLower(gpa, io, cache, path, v3, .verify, &r4);
    defer r4.deinit(gpa);
    defer lp4.deinit(gpa);
    try testing.expectEqual(@as(usize, 3), lp4.codegen_cached);
}

test "errors: missing/unknown/mismatched fields; positional construction; recursive struct" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-driver-m9-err";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    // All these are caught in Typecheck (after Resolve), so each is a TypeError.
    const cases = [_][]const u8{
        "struct P { x: int, y: int }\nfn main() -> int { p := P { x: 1 }\n return p.x }\n", // missing field
        "struct P { x: int }\nfn main() -> int { p := P { x: 1, z: 2 }\n return p.x }\n", // unknown field (init)
        "struct P { x: int }\nfn main() -> int { p := P { x: 1 }\n return p.q }\n", // unknown field (access)
        "struct P { x: int }\nfn main() -> int { p := P { x: true }\n return p.x }\n", // field type mismatch
        "struct P { x: int }\nfn main() -> int { p := P(1)\n return p.x }\n", // positional construction
        "struct R { r: R }\nfn main() -> int { return 0 }\n", // directly-recursive struct
        // A bare struct NAME used as a value (not P{...}/P(...)) — Resolve quietly
        // skips struct-named idents, so Typecheck must report it (else it escapes to
        // a codegen `unreachable`). Both the var-init and field-access-base subcases.
        "struct P { x: int }\nfn main() -> int { q := P\n return 7 }\n", // struct name as value
        "struct P { x: int }\nfn main() -> int { return P.x }\n", // struct-name field-access base
        // A struct named after a builtin scalar/str type (permanently unreachable).
        "struct int { x: int }\nfn main() -> int { return 0 }\n", // shadows builtin
        // An empty struct (size-0 aggregate — an ABI/codegen hazard).
        "struct E {}\nfn main() -> int { return 0 }\n", // empty struct
    };
    for (cases, 0..) |src, i| {
        const path = std.fmt.allocPrint(gpa, "{s}/e{d}.toy", .{ dir_name, i }) catch unreachable;
        defer gpa.free(path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });
        var r: FileResult = .{ .path = path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, i);
        defer r.deinit(gpa);
        try testing.expectEqual(@as(?anyerror, error.TypeError), r.err);
        try testing.expect(r.typecheck != null);
        try testing.expect(r.typecheck.?.diags.len > 0);
    }
}

test "errors: undefined/duplicate labels (Resolve); continue-block & value-break-while (Type); bad label prefix (Parse)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-driver-m8-err";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    const ErrCase = struct { src: []const u8, want: anyerror };
    // The pipeline short-circuits Resolve (ResolveError) BEFORE Typecheck
    // (TypeError) BEFORE codegen; the surface (label-prefix) failure is a ParseError.
    const cases = [_]ErrCase{
        // ResolveError: break/continue to an undefined/out-of-scope label.
        .{ .src = "fn f() -> int {\n @o loop { break @undef 1 }\n}\n", .want = error.ResolveError },
        .{ .src = "fn f() {\n @o loop { continue @undef }\n return\n}\n", .want = error.ResolveError },
        // ResolveError: a break to a label that is out of scope (closed already).
        .{ .src = "fn f() {\n @a loop { break }\n @b while true { break @a }\n return\n}\n", .want = error.ResolveError },
        // ResolveError: a duplicate label in scope.
        .{ .src = "fn f() {\n @x loop { @x loop { break } }\n return\n}\n", .want = error.ResolveError },
        // TypeError: `continue @blk` where @blk labels a bare block (not a loop).
        .{ .src = "fn f() {\n @blk { continue @blk }\n return\n}\n", .want = error.TypeError },
        // TypeError: `break @w 5` where @w labels a while (a `()` loop).
        .{ .src = "fn f() {\n @w while true { break @w 5 }\n return\n}\n", .want = error.TypeError },
        // TypeError: `break @ff 5` where @ff labels a for (a `()` loop).
        .{ .src = "fn f() {\n @ff for i in 0..3 { break @ff 5 }\n return\n}\n", .want = error.TypeError },
        // ParseError: a label prefixing a non-block-like construct.
        .{ .src = "fn f() {\n @x 1\n return\n}\n", .want = error.ParseError },
    };
    for (cases, 0..) |c, i| {
        const path = std.fmt.allocPrint(gpa, "{s}/e{d}.toy", .{ dir_name, i }) catch unreachable;
        defer gpa.free(path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = c.src });
        var r: FileResult = .{ .path = path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, i);
        defer r.deinit(gpa);
        try testing.expectEqual(@as(?anyerror, c.want), r.err);
        if (c.want == error.ResolveError) {
            try testing.expect(r.resolve != null);
            try testing.expect(r.resolve.?.diags.len > 0);
        } else if (c.want == error.TypeError) {
            try testing.expect(r.typecheck != null);
            try testing.expect(r.typecheck.?.diags.len > 0);
        }
    }
}

test "error: break/continue outside a loop and break-value in while/for are rejected" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-driver-m7-err";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    // The outside-loop and break-value checks live in Typecheck, so each is a TypeError.
    const cases = [_][]const u8{
        "fn f() {\n break\n return\n}\n", // break outside a loop
        "fn f() {\n continue\n return\n}\n", // continue outside a loop
        "fn f() {\n while true { break 5 }\n return\n}\n", // value-break in a while
        "fn f() {\n for i in 0..5 { break i }\n return\n}\n", // value-break in a for
    };
    for (cases, 0..) |src, i| {
        const path = std.fmt.allocPrint(gpa, "{s}/e{d}.toy", .{ dir_name, i }) catch unreachable;
        defer gpa.free(path);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });
        var r: FileResult = .{ .path = path };
        try pipeline(gpa, io, cache, .check, "aarch64-macos", &r, i);
        defer r.deinit(gpa);
        try testing.expectEqual(@as(?anyerror, error.TypeError), r.err);
        try testing.expect(r.typecheck != null);
        try testing.expect(r.typecheck.?.diags.len > 0);
    }
}

test "byte-identical: a warm build equals a from-scratch (.force) build" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    const src = "fn add(a: int, b: int) -> int {\n return a + b\n}\nfn main() -> int {\n return add(40, 2)\n}\n";

    // Warm (populate, then read back from cache).
    {
        var r0: FileResult = undefined;
        var lp0 = try checkAndLower(gpa, io, cache, path, src, .normal, &r0);
        r0.deinit(gpa);
        lp0.deinit(gpa);
    }
    var rw: FileResult = undefined;
    var warm = try checkAndLower(gpa, io, cache, path, src, .normal, &rw);
    defer rw.deinit(gpa);
    defer warm.deinit(gpa);
    const warm_img = try buildImage(io, gpa, "p", warm.text, warm.entry_off, warm.cstrings, warm.data_relocs);
    defer gpa.free(warm_img);

    // From scratch (ignore the cache).
    var rf: FileResult = undefined;
    var fresh = try checkAndLower(gpa, io, cache, path, src, .force, &rf);
    defer rf.deinit(gpa);
    defer fresh.deinit(gpa);
    const fresh_img = try buildImage(io, gpa, "p", fresh.text, fresh.entry_off, fresh.cstrings, fresh.data_relocs);
    defer gpa.free(fresh_img);

    try testing.expectEqualSlices(u8, fresh_img, warm_img);
}

test "reorder: swapping fn order is all cache hits and keeps correct linkage" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var src_buf: [64]u8 = undefined;
    const src_dir = std.fmt.bufPrint(&src_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;
    var cache_buf: [80]u8 = undefined;
    const cache_dir = std.fmt.bufPrint(&cache_buf, ".zig-cache/tmp/{s}/cc", .{&tmp.sub_path}) catch unreachable;
    const cache = try Cache.init(io, cache_dir);

    const path = std.fmt.allocPrint(gpa, "{s}/p.toy", .{src_dir}) catch unreachable;
    defer gpa.free(path);

    // [main, helper] then [helper, main] — bodies identical, order swapped.
    const a = "fn main() -> int {\n return helper()\n}\nfn helper() -> int {\n return 7\n}\n";
    const b = "fn helper() -> int {\n return 7\n}\nfn main() -> int {\n return helper()\n}\n";

    var ra: FileResult = undefined;
    var lpa = try checkAndLower(gpa, io, cache, path, a, .normal, &ra);
    defer ra.deinit(gpa);
    defer lpa.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lpa.codegen_compiled);

    // Reordered build: same fingerprints (position-independent) → all cache hits.
    var rb: FileResult = undefined;
    var lpb = try checkAndLower(gpa, io, cache, path, b, .normal, &rb);
    defer rb.deinit(gpa);
    defer lpb.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), lpb.codegen_cached);
    try testing.expectEqual(@as(usize, 0), lpb.codegen_compiled);

    // The reordered binary still links correctly (call resolves by name): build
    // it and run it — must exit 7. (macOS/aarch64 only.)
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return;
    const image = try buildImage(io, gpa, "p", lpb.text, lpb.entry_off, lpb.cstrings, lpb.data_relocs);
    defer gpa.free(image);
    const out_path = std.fmt.allocPrint(gpa, "{s}/p", .{src_dir}) catch unreachable;
    defer gpa.free(out_path);
    {
        const perms: Io.File.Permissions = .fromMode(0o755);
        var f = try Io.Dir.cwd().createFile(io, out_path, .{ .permissions = perms });
        defer f.close(io);
        try f.writeStreamingAll(io, image);
        try f.setPermissions(io, perms);
    }
    const abs = try Io.Dir.cwd().realPathFileAlloc(io, out_path, gpa);
    defer gpa.free(abs);
    var child = try std.process.spawn(io, .{ .argv = &.{abs} });
    const term = try child.wait(io);
    try testing.expectEqual(std.process.Child.Term{ .exited = 7 }, term);
}

test "cross-serializer parity: every int-aware identity serializer folds int_desc in lockstep" {
    // The cross-serializer property this pins: the four independent encoders of a
    // Type's identity (the flat dedup key, the coherence key, the Mono mangle, and the
    // fingerprint's TouchedType) must AGREE on which int variants are distinct. An encoder
    // that dropped int_desc (the coherence bug) would fold two widths into one here and
    // fail the biconditional. Since every int variant is pairwise distinct, this asserts
    // each encoder separates exactly the pairs the others do.
    const gpa = testing.allocator;
    const Type = Typecheck.Type;
    const ints = [_]Type{
        Type.int,   Type.uint,
        Type.int8,  Type.int16,  Type.int32,  Type.int64,
        Type.uint8, Type.uint16, Type.uint32, Type.uint64,
    };

    // appendTouched only reads `frozen` on the struct/enum branches; an int never reaches
    // them, so empty layout tables are a sound throwaway.
    const throwaway: struct {
        layouts: []const Typecheck.Layout = &.{},
        enum_layouts: []const Typecheck.EnumLayout = &.{},
    } = .{};

    for (ints, 0..) |a, i| {
        for (ints, 0..) |b, j| {
            const same = (i == j);

            var ka: std.ArrayList(u8) = .empty;
            defer ka.deinit(gpa);
            var kb: std.ArrayList(u8) = .empty;
            defer kb.deinit(gpa);
            try a.appendKeyBytes(gpa, &ka);
            try b.appendKeyBytes(gpa, &kb);
            try testing.expectEqual(same, std.mem.eql(u8, ka.items, kb.items));

            var ca: std.ArrayList(u8) = .empty;
            defer ca.deinit(gpa);
            var cb: std.ArrayList(u8) = .empty;
            defer cb.deinit(gpa);
            try Typecheck.coherence.appendKeyType(gpa, &ca, a);
            try Typecheck.coherence.appendKeyType(gpa, &cb, b);
            try testing.expectEqual(same, std.mem.eql(u8, ca.items, cb.items));

            const ma = try Mono.mangle(gpa, "id", &.{a});
            defer gpa.free(ma);
            const mb = try Mono.mangle(gpa, "id", &.{b});
            defer gpa.free(mb);
            try testing.expectEqual(same, std.mem.eql(u8, ma, mb));

            var ta: std.ArrayList(Fingerprint.TouchedType) = .empty;
            defer ta.deinit(gpa);
            var tb: std.ArrayList(Fingerprint.TouchedType) = .empty;
            defer tb.deinit(gpa);
            try AstWalk.appendTouched(gpa, &throwaway, a, &ta);
            try AstWalk.appendTouched(gpa, &throwaway, b, &tb);
            const touched_same = ta.items[0].kind == tb.items[0].kind and
                ta.items[0].int_desc == tb.items[0].int_desc;
            try testing.expectEqual(same, touched_same);
        }
    }

    // Derive's mangle/key legitimately OMIT int_desc: its domain is struct/enum recipes,
    // neither of which carries the descriptor. Documents why that omission is sound (and
    // is the invariant Derive.writeKey/mangle now assert).
    try testing.expect(!Type.carriesIntDesc(.@"struct"));
    try testing.expect(!Type.carriesIntDesc(.@"enum"));
}
