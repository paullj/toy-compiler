//! Differential check-vs-build test (integration).
//!
//! REGRESSION GUARD for the class of bug just fixed: `toy check` used to fan out
//! over the entry as an INDEPENDENT single file (no import discovery), so a valid
//! import spuriously reported `R0003 unknown imported module` + `R0001 undeclared
//! identifier` that `build` did not. The fix routes `check` through the SAME graph
//! front-end `build` uses (`Graph.discover` -> `resolveGraph` -> `checkGraph`), so
//! imports ARE followed. This test asserts `check` and `build` AGREE.
//!
//! The CORE guard runs IN-PROCESS through the published surface, so it always
//! executes under `test-bin` (no built-binary or macOS dependency): it drives the
//! very discover -> resolve -> typecheck sequence `runCheck` drives, and asserts
//!   (i)  a single-file VALID program -> no errors;
//!   (ii) a multi-module VALID program with a real import -> no errors (EXACTLY the
//!        regressed case: if `check` stopped discovering imports the import symbol
//!        would go undeclared and this would FAIL);
//!   (iii) a multi-module program with a REAL error in the IMPORTED module -> an
//!        error reported AGAINST the imported module (not the entry).
//! A separate, binary-gated block additionally spawns the real `toy check` and
//! `toy build` and asserts their EXIT CODES agree on the same three fixtures — the
//! literal check==build differential over the CLI — skipping cleanly when the
//! binary is absent (build additionally needs macOS codegen).

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const toyc = @import("toy_compiler");

const Driver = toyc.Driver;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Diagnostic = toyc.DiagnosticSink.Diagnostic;
const NO_SCOPE = toyc.DiagnosticSink.NO_SCOPE;
const testing = std.testing;

/// The outcome of the in-process front-end over one entry: whether discovery hit a
/// structural error (a real missing import, cycle, escape, or a tainted parse),
/// plus the count of error-severity resolve/typecheck diagnostics and, for the
/// first such diagnostic, its owning module scope. This mirrors what `runCheck`
/// consults to pick its exit code, so "no structural error and zero error diags"
/// is exactly `check`'s exit-0 condition — and `build`'s.
const CheckOutcome = struct {
    structural: bool,
    errors: usize,
    first_error_scope: u32 = NO_SCOPE,
};

/// Run the SAME front-end `runCheck`/`build` run over `entry`: discover the graph
/// FOLLOWING imports, resolve the whole graph, and (only if resolve is clean)
/// typecheck it. Returns the error tally + first error's owning module. This is the
/// in-process differential subject: if discovery ever stopped following imports,
/// (ii) below would surface an undeclared-symbol error and the test would FAIL.
fn checkEntry(gpa: std.mem.Allocator, io: Io, entry: []const u8) !CheckOutcome {
    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);

    var graph = try Graph.discover(gpa, io, cache, "native", entry, null);
    defer graph.deinit(gpa);
    if (graph.err != null) return .{ .structural = true, .errors = 0 };

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    if (countErrors(res.diags) > 0) {
        return .{ .structural = false, .errors = countErrors(res.diags), .first_error_scope = firstErrorScope(res.diags) };
    }

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, io, 0);
    defer tc.deinit(gpa);
    return .{ .structural = false, .errors = countErrors(tc.diags), .first_error_scope = firstErrorScope(tc.diags) };
}

fn countErrors(diags: []const Diagnostic) usize {
    var n: usize = 0;
    for (diags) |d| if (d.severity == .err) {
        n += 1;
    };
    return n;
}

fn firstErrorScope(diags: []const Diagnostic) u32 {
    for (diags) |d| if (d.severity == .err) return d.scope;
    return NO_SCOPE;
}

/// Write a fresh, unique temp fixture dir (torn down by the caller's defer) and
/// return its name. `files` is a list of {relative-path, contents} pairs.
fn writeFixture(io: Io, comptime dir_name: []const u8, files: []const [2][]const u8) !void {
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    for (files) |f| {
        // Ensure any subdir in the relative path exists before writing.
        if (std.fs.path.dirnamePosix(f[0])) |sub| {
            var buf: [512]u8 = undefined;
            const full = std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir_name, sub }) catch unreachable;
            try Io.Dir.cwd().createDirPath(io, full);
        }
        var pbuf: [512]u8 = undefined;
        const path = std.fmt.bufPrint(&pbuf, "{s}/{s}", .{ dir_name, f[0] }) catch unreachable;
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = f[1] });
    }
}

test "differential: check agrees with build in-process (single-file + multi-module valid + imported-module error)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // (i) SINGLE-FILE VALID -> no structural error, zero error diagnostics (check
    // exit 0; build exit 0).
    {
        const dir = ".toy-test-diff-single";
        try writeFixture(io, dir, &.{
            .{ "main.toy", "fn main() -> int {\n  return 7\n}\n" },
        });
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const outcome = try checkEntry(gpa, io, dir ++ "/main.toy");
        try testing.expect(!outcome.structural);
        try testing.expectEqual(@as(usize, 0), outcome.errors);
    }

    // (ii) MULTI-MODULE VALID with a REAL import -> no structural error, zero error
    // diagnostics. THIS IS THE REGRESSED CASE: `main` imports `helper` and calls its
    // pub fn. If `check` stopped discovering imports, `helper` would be an unknown
    // module and `helper.answer` an undeclared identifier, so `outcome.errors` would
    // be > 0 and this assertion would FAIL — the exact guard the fix demands.
    {
        const dir = ".toy-test-diff-multi-ok";
        try writeFixture(io, dir, &.{
            .{ "main.toy", "import helper\nfn main() -> int {\n  return helper.answer()\n}\n" },
            .{ "helper.toy", "pub fn answer() -> int {\n  return 42\n}\n" },
        });
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const outcome = try checkEntry(gpa, io, dir ++ "/main.toy");
        try testing.expect(!outcome.structural);
        try testing.expectEqual(@as(usize, 0), outcome.errors);
    }

    // (iii) MULTI-MODULE with a REAL error in the IMPORTED module -> at least one
    // error diagnostic, reported AGAINST the imported module (scope != entry). This
    // is only reachable if discovery FOLLOWED the import into `helper` and typed it —
    // a single-file check of `main` alone could never see `helper`'s body error.
    {
        const dir = ".toy-test-diff-multi-err";
        try writeFixture(io, dir, &.{
            .{ "main.toy", "import helper\nfn main() -> int {\n  return helper.answer()\n}\n" },
            .{ "helper.toy", "pub fn answer() -> int {\n  return nope_zzq\n}\n" },
        });
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const outcome = try checkEntry(gpa, io, dir ++ "/main.toy");
        try testing.expect(!outcome.structural);
        try testing.expect(outcome.errors >= 1);
        // The error is in the IMPORTED module, not the entry (entry is discovery id 0).
        try testing.expect(outcome.first_error_scope != NO_SCOPE);
        try testing.expect(outcome.first_error_scope != 0);
    }

    // (iv) M8 check-vs-build parity: a single-file program calling a MISSING method
    // reports EXACTLY one error (T0018) on the shared `checkGraph` front-end — the
    // same path `build` runs, so `check` and `build` agree it is broken.
    {
        const dir = ".toy-test-diff-nomethod";
        try writeFixture(io, dir, &.{
            .{ "main.toy", "struct P { x: int }\nfn main() -> int {\n  p := P{ x: 1 }\n  return p.nope()\n}\n" },
        });
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const outcome = try checkEntry(gpa, io, dir ++ "/main.toy");
        try testing.expect(!outcome.structural);
        try testing.expectEqual(@as(usize, 1), outcome.errors);
    }
}

/// Spawn the built `toy` with `argv_tail`, returning the exit code (or an error).
/// Skips (via error.SkipZigTest at the call site) when the binary is absent.
fn spawnExit(gpa: std.mem.Allocator, io: Io, argv_tail: []const []const u8) !std.process.Child.Term {
    const bin_abs = try Io.Dir.cwd().realPathFileAlloc(io, "zig-out/bin/toy", gpa);
    defer gpa.free(bin_abs);
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(gpa);
    try argv.append(gpa, bin_abs);
    for (argv_tail) |a| try argv.append(gpa, a);
    var child = try std.process.spawn(io, .{ .argv = argv.items, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    gpa.free(got);
    return child.wait(io);
}

fn exitCode(term: std.process.Child.Term) ?u8 {
    return switch (term) {
        .exited => |c| c,
        else => null,
    };
}

test "differential: the real `toy check` and `toy build` exit codes AGREE on the three fixtures" {
    // The literal CLI differential: spawn both subcommands and compare exit codes.
    // `build` runs codegen, so it needs macOS + the built binary; skip cleanly
    // otherwise. The in-process test above is the always-on guard; this pins the
    // end-to-end CLI agreement.
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;

    // (i) single-file valid: check exit 0 AND build exit 0.
    {
        const dir = ".toy-test-diff-cli-single";
        try writeFixture(io, dir, &.{
            .{ "main.toy", "fn main() -> int {\n  return 7\n}\n" },
        });
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const chk = try spawnExit(gpa, io, &.{ "check", dir ++ "/main.toy" });
        const bld = try spawnExit(gpa, io, &.{ "build", dir ++ "/main.toy", "-o", dir ++ "/prog" });
        try testing.expectEqual(@as(?u8, 0), exitCode(chk));
        try testing.expectEqual(@as(?u8, 0), exitCode(bld));
    }

    // (ii) multi-module valid with a real import: check exit 0 AND build exit 0. If
    // check stopped following imports it would exit 1 here while build exits 0 — a
    // divergence this equality assertion catches.
    {
        const dir = ".toy-test-diff-cli-multi-ok";
        try writeFixture(io, dir, &.{
            .{ "main.toy", "import helper\nfn main() -> int {\n  return helper.answer()\n}\n" },
            .{ "helper.toy", "pub fn answer() -> int {\n  return 42\n}\n" },
        });
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const chk = try spawnExit(gpa, io, &.{ "check", dir ++ "/main.toy" });
        const bld = try spawnExit(gpa, io, &.{ "build", dir ++ "/main.toy", "-o", dir ++ "/prog" });
        try testing.expectEqual(@as(?u8, 0), exitCode(chk));
        try testing.expectEqual(@as(?u8, 0), exitCode(bld));
    }

    // (iii) multi-module with a real error in the imported module: check exit 1 AND
    // build exit != 0 (they agree that the program is broken, reported at helper).
    {
        const dir = ".toy-test-diff-cli-multi-err";
        try writeFixture(io, dir, &.{
            .{ "main.toy", "import helper\nfn main() -> int {\n  return helper.answer()\n}\n" },
            .{ "helper.toy", "pub fn answer() -> int {\n  return nope_zzq\n}\n" },
        });
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const chk = try spawnExit(gpa, io, &.{ "check", dir ++ "/main.toy" });
        const bld = try spawnExit(gpa, io, &.{ "build", dir ++ "/main.toy", "-o", dir ++ "/prog" });
        try testing.expectEqual(@as(?u8, 1), exitCode(chk));
        try testing.expect(exitCode(bld) == null or exitCode(bld).? != 0);
    }
}
