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
const codes = toyc.codes;
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

    var graph = try Graph.discover(gpa, io, cache, "native", entry, .{});
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

test "type-alias privacy: `type S = mod.Private` does not launder a non-pub cross-module type" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // Aliasing a NON-pub struct must be rejected exactly as a direct `geom.Secret`
    // reference is (R0005) — the alias target is resolved through the same
    // export-visibility check, not laundered past it.
    {
        const dir = ".toy-test-alias-privacy";
        try writeFixture(io, dir, &.{
            .{ "geom.toy", "struct Secret { x: int }\npub fn ping() -> int { return 1 }\n" },
            .{ "main.toy", "import geom\ntype S = geom.Secret\nfn describe(s: S) -> int { return s.x }\nfn main() -> int { return geom.ping() }\n" },
        });
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const outcome = try checkEntry(gpa, io, dir ++ "/main.toy");
        try testing.expect(!outcome.structural);
        try testing.expect(outcome.errors >= 1);
    }

    // Control: aliasing a `pub` struct is clean — the fix must not over-reject.
    {
        const dir = ".toy-test-alias-privacy-ok";
        try writeFixture(io, dir, &.{
            .{ "geom.toy", "pub struct Rect { w: int, h: int }\npub fn ping() -> int { return 1 }\n" },
            .{ "main.toy", "import geom\ntype R = geom.Rect\nfn takes(r: R) -> int { return 0 }\nfn main() -> int { return geom.ping() }\n" },
        });
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const outcome = try checkEntry(gpa, io, dir ++ "/main.toy");
        try testing.expect(!outcome.structural);
        try testing.expectEqual(@as(usize, 0), outcome.errors);
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

    // (iv) check-vs-build parity: a single-file program calling a MISSING method
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

// regression: the T0024 coherence signature check must anchor its diagnostic in
// the CURRENTLY-scanned module's tree, not at the `decl_node` of `findMethod`'s
// whole-program first match. When the SAME (protocol, receiver) is impl'd in two
// different modules for a shared imported type (an overlapping impl -> T0020),
// `findMethod` returns the first-registered method, whose `decl_node` indexes a
// DIFFERENT (here far larger) module's tree; dereferencing it against the active,
// shorter tree used to panic with `index out of bounds`, crashing the compiler on
// mere source input (breaking the report-all contract). Here module `a` is imported
// first and padded so its node array dwarfs tiny module `b`'s, and `a`'s impl
// signature is the mismatched one. The check must NOT crash and must attribute
// EXACTLY one T0024 (to `a`'s genuinely-wrong impl) plus one T0020 (the overlap) —
// never a spurious T0024 against `b`'s correct sibling impl.
test "coherence: overlapping cross-module impls with a mismatched sig report T0024+T0020 without crashing" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const a_src = comptime blk: {
        var s: []const u8 = "import lib\n";
        var i: usize = 0;
        while (i < 30) : (i += 1) {
            s = s ++ std.fmt.comptimePrint("fn pad_{d}(a: int) -> int {{ return a + {d} }}\n", .{ i, i });
        }
        s = s ++ "impl lib.P has lib.Doubler {\n    fn dbl(self) -> bool { self.x > 0 }\n}\n";
        break :blk s;
    };

    const dir = ".toy-test-diff-coherence-xmod";
    try writeFixture(io, dir, &.{
        .{ "lib.toy", "pub protocol Doubler {\n    fn dbl(self) -> int\n}\npub struct P {\n    x: int\n}\n" },
        .{ "a.toy", a_src },
        .{ "b.toy", "import lib\nimpl lib.P has lib.Doubler {\n    fn dbl(self) -> int { self.x + self.x }\n}\n" },
        .{ "main.toy", "import lib\nimport a\nimport b\nfn main() -> int {\n    return 0\n}\n" },
    });
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);
    var graph = try Graph.discover(gpa, io, cache, "native", dir ++ "/main.toy", .{});
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), countErrors(res.diags));

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, io, 0);
    defer tc.deinit(gpa);

    var n_t0024: usize = 0;
    var n_t0020: usize = 0;
    for (tc.diags) |d| {
        if (d.severity != .err) continue;
        switch (d.code) {
            .T0024 => n_t0024 += 1,
            .T0020 => n_t0020 += 1,
            else => {},
        }
    }
    try testing.expectEqual(@as(usize, 1), n_t0024);
    try testing.expectEqual(@as(usize, 1), n_t0020);
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

/// Spawn an already-built binary by absolute path and return its Term. `spawnExit`
/// only ever spawns `toy`; this runs a compiled toy PROGRAM directly so a `brk #0`
/// abort surfaces as `Term.signal` (exitCode==null) — `toy run` would cook it into a
/// clean 128+signo instead.
fn spawnBinaryTerm(gpa: std.mem.Allocator, io: Io, abs_path: []const u8) !std.process.Child.Term {
    var child = try std.process.spawn(io, .{ .argv = &.{abs_path}, .stdout = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    gpa.free(got);
    return child.wait(io);
}

const BinTermErr = struct { term: std.process.Child.Term, stderr: []u8 };
/// Like `spawnBinaryTerm` but ALSO captures the child's STDERR (fd 2). Drains stdout
/// (discarded) so a program that also prints cannot deadlock on a full pipe; the panic
/// fixtures print nothing, so their tiny stderr never blocks. Caller frees `.stderr`. A
/// clean `exit(1)` panic surfaces as `Term.exited{1}` with the message on `.stderr`.
fn spawnBinaryTermStderr(gpa: std.mem.Allocator, io: Io, abs_path: []const u8) !BinTermErr {
    var child = try std.process.spawn(io, .{ .argv = &.{abs_path}, .stdout = .pipe, .stderr = .pipe });
    var out = child.stdout.?.readerStreaming(io, &.{});
    const out_bytes = try out.interface.allocRemaining(gpa, .limited(1 << 16));
    gpa.free(out_bytes);
    var errr = child.stderr.?.readerStreaming(io, &.{});
    const err_bytes = try errr.interface.allocRemaining(gpa, .limited(1 << 16));
    return .{ .term = try child.wait(io), .stderr = err_bytes };
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

/// A protocol reusing an operator/derive witness name (`eq`/`cmp`) on a type that ALSO gets
/// the structural derive: the operator must bind the DERIVE witness by protocol id, not go
/// `.ambiguous`. The checker types `==`/`<` off conformance EXISTENCE (the derive satisfies
/// it), so if lower selected the witness by NAME it would see two same-named conformance
/// methods, resolve `.ambiguous`, and ABORT codegen on a program `check` accepted. Both
/// programs must `check` clean AND run to 111 (the derive result), ignoring the sibling.
const sibling_eq_src =
    \\protocol Weird { fn eq(self, other: Self) -> bool }
    \\struct Point { x: int, y: int }
    \\impl Point has Weird { fn eq(self, other: Point) -> bool { return false } }
    \\fn main() -> int {
    \\  a := Point{x: 1, y: 2}
    \\  b := Point{x: 1, y: 2}
    \\  return if a == b { 111 } else { 222 }
    \\}
    \\
;

const sibling_cmp_src =
    \\protocol Weird { fn cmp(self, other: Self) -> int }
    \\struct Point { x: int, y: int }
    \\impl Point has Weird { fn cmp(self, other: Point) -> int { return 999 } }
    \\fn main() -> int {
    \\  a := Point{x: 1, y: 2}
    \\  b := Point{x: 3, y: 4}
    \\  return if a < b { 111 } else { 222 }
    \\}
    \\
;

test "sibling protocol reusing eq/cmp: check is clean (in-process, always-on)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    inline for (.{ sibling_eq_src, sibling_cmp_src }, .{ ".toy-test-sibling-eq", ".toy-test-sibling-cmp" }) |src, dir| {
        try writeFixture(io, dir, &.{.{ "main.toy", src }});
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const outcome = try checkEntry(gpa, io, dir ++ "/main.toy");
        try testing.expect(!outcome.structural);
        try testing.expectEqual(@as(usize, 0), outcome.errors);
    }
}

test "sibling protocol reusing eq/cmp: build+run yields the derive result (exit 111), not a codegen abort" {
    // The end-to-end guard for the check/lower divergence: `check` accepted these, so a
    // name-selected witness would `.ambiguous` in lower and abort `build`. Needs macOS
    // codegen + the built `toy`; skips cleanly otherwise (the in-process check above is
    // the always-on guard).
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;

    inline for (.{ sibling_eq_src, sibling_cmp_src }, .{ ".toy-test-sibling-eq-run", ".toy-test-sibling-cmp-run" }) |src, dir| {
        try writeFixture(io, dir, &.{.{ "main.toy", src }});
        defer Io.Dir.cwd().deleteTree(io, dir) catch {};
        const chk = try spawnExit(gpa, io, &.{ "check", dir ++ "/main.toy" });
        const runt = try spawnExit(gpa, io, &.{ "run", dir ++ "/main.toy" });
        try testing.expectEqual(@as(?u8, 0), exitCode(chk));
        try testing.expectEqual(@as(?u8, 111), exitCode(runt));
    }
}

test "panic: div/mod/unwrap traps + user panic() write msg + a symbolized-backtrace dump to STDERR and exit nonzero at BOTH -O levels; nonzero controls exit normally" {
    // `__panic` writes the message to fd 2, then a symbolized backtrace — one `0x<hex>`
    // line per frame walked off the x29 chain (each is the call site's slide-independent
    // __text offset) — then SYS_exit(1): a clean nonzero exit, no SIGILL/brk. A panicking
    // program yields >= 2 frames (the panic-site fn + at least its caller up to the C
    // runtime). The -O1 legs are the SOLE proof fold's const-0 skip held (that -O1 did
    // not fold `/0`/`%0` to a value and erase the panic). The control legs (nonzero
    // divisor) run to their normal exit with NO backtrace — their panic block is now
    // `bl panic` but never reached (or DCE'd at -O1 for the const divisor).
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    Io.Dir.cwd().access(io, "zig-out/bin/toy", .{}) catch return error.SkipZigTest;

    const fixtures = .{
        .{ .dir = ".toy-test-panic-div0", .src = "fn main() -> int { return 10 / 0 }\n", .want = @as(?u8, 1), .err = "division by zero" },
        .{ .dir = ".toy-test-panic-mod0", .src = "fn main() -> int { return 10 % 0 }\n", .want = @as(?u8, 1), .err = "remainder by zero" },
        .{ .dir = ".toy-test-panic-unwrap", .src = "fn main() -> int {\n  x := Option[int].none\n  return x.unwrap()\n}\n", .want = @as(?u8, 1), .err = "unwrap of empty" },
        .{ .dir = ".toy-test-panic-user", .src = "fn main() -> int {\n  panic(\"boom\")\n  return 0\n}\n", .want = @as(?u8, 1), .err = "boom" },
        .{ .dir = ".toy-test-panic-oob", .src = "import std/vec\nfn main() -> int {\n  xs := [10]\n  return xs[5]\n}\n", .want = @as(?u8, 1), .err = "index out of bounds" },
        .{ .dir = ".toy-test-panic-ctldiv", .src = "fn main() -> int { return 10 / 3 }\n", .want = @as(?u8, 3), .err = "" },
        .{ .dir = ".toy-test-panic-ctlmod", .src = "fn main() -> int { return 17 % 5 }\n", .want = @as(?u8, 2), .err = "" },
    };
    inline for (fixtures) |fx| {
        defer Io.Dir.cwd().deleteTree(io, fx.dir) catch {};
        inline for (.{ "-O0", "-O1" }) |lvl| {
            try writeFixture(io, fx.dir, &.{.{ "main.toy", fx.src }});
            const bld = try spawnExit(gpa, io, &.{ "build", lvl, fx.dir ++ "/main.toy", "-o", fx.dir ++ "/prog" ++ lvl });
            try testing.expectEqual(@as(?u8, 0), exitCode(bld)); // compiled cleanly
            const abs = try Io.Dir.cwd().realPathFileAlloc(io, fx.dir ++ "/prog" ++ lvl, gpa);
            defer gpa.free(abs);
            const r = try spawnBinaryTermStderr(gpa, io, abs);
            defer gpa.free(r.stderr);
            try testing.expectEqual(fx.want, exitCode(r.term)); // clean nonzero exit (1), not a signal
            const frames = countBacktraceFrames(r.stderr);
            if (fx.err.len > 0) {
                try testing.expect(std.mem.indexOf(u8, r.stderr, fx.err) != null);
                try testing.expect(frames >= 2); // panic dumps >= 2 walked frames
                // Every fixture panics in `main`, so its symbolized name appears on a
                // frame line (` main` follows the offset — the symbol-table hit).
                try testing.expect(std.mem.indexOf(u8, r.stderr, " main\n") != null);
            } else {
                try testing.expectEqual(@as(usize, 0), frames); // a clean run prints no backtrace
            }
        }
    }
}

/// Count backtrace frame lines (`0x<hex>`) in a panic's stderr. Each frame the
/// `__panic` FP-chain walk emits is its own `0x`-prefixed line.
fn countBacktraceFrames(stderr: []const u8) usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, stderr, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "0x")) n += 1;
    }
    return n;
}
