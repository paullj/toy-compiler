//! `toyc` CLI entry point.
//!
//! Usage:
//!   toyc [--emit lex|parse|check] [--dump] [--target <triple>] <file...>
//!
//! Runs the pipeline (lex → parse → check) over each file in parallel, with an
//! on-disk cache, and prints a summary. `--dump` prints the artifact of the emit
//! phase: tokens for `lex`, the AST S-expression for `parse`, and the AST plus a
//! per-function signature summary for `check` (name resolution + typecheck).
//! `-o` carries on through lower → codegen → link → sign, all hung off this same
//! driver; `--emit ir` dumps the target-independent IR text.

const std = @import("std");
const Io = std.Io;
// The CLI entry is the exe module's root; it lives in driver/ (the engine), but
// because a Zig module cannot import files above its root source file, it reaches
// the rest of the compiler through the `toy_compiler` library module (rooted at
// src/root.zig) rather than via relative `../` paths.
const toyc = @import("toy_compiler");
const Driver = toyc.Driver;
const Ast = toyc.Ast;
const Graph = toyc.Graph;
const Codegen = toyc.DriverCodegen;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const CodegenIr = toyc.CodegenIr;
const Opt = toyc.Opt;
const Engine = toyc.QueryEngine;
const version = toyc.version;

pub fn main(init: std.process.Init) !void {
    // The per-fn codegen fan-out + parallel link tail hammer the allocator from
    // every worker thread. `std.process.Init.gpa` is a mutex-guarded DebugAllocator
    // in Debug/ReleaseSafe, so concurrent allocs serialize on one lock and the
    // parallelism collapses to ~1x regardless of -j. The lock-free per-CPU
    // `smp_allocator` lets the workers allocate without contending, so -jN actually
    // scales in EVERY build mode. The allocator choice never affects output bytes
    // (determinism is structural, not alloc-order dependent), so this is safe.
    const gpa = std.heap.smp_allocator;
    const io = init.io;

    var stdout_buf: [4096]u8 = undefined;
    // Streaming (not positional) — stdout may be a pipe, tty, or append target,
    // none of which are seekable. Positional writes corrupt ordering there.
    var stdout_writer = Io.File.stdout().writerStreaming(io, &stdout_buf);
    const out = &stdout_writer.interface;

    var dump = false;
    var emit: Driver.Emit = .parse;
    // The DEFAULT action is to BUILD an executable; `--emit` opts into an inspection
    // mode instead. This tracks whether the user asked for one, so a bare `toyc <file>`
    // builds (output → the default build dir) rather than printing a report.
    var emit_explicit = false;
    // The optional leading subcommand: `build` (compile to an executable, the default)
    // or `run` (build, then execute the produced binary and report how it ended).
    const Command = enum { build, run };
    var command: Command = .build;
    var verb_seen = false;
    var target: []const u8 = "native";
    var out_path: ?[]const u8 = null;
    var codegen_stats = false;
    var mode: Engine.Mode = .normal;
    // opt level / pass selection. Default -O0 (no opt). Last flag wins,
    // left-to-right; --opt= / --no-opt= toggle individual passes from current.
    var opt: Opt.Config = .O0;
    var opt_stats = false;
    // Per-stage wall-clock profile (discover/resolve/typecheck/lower/image+sign).
    // Run at `-j1` for a clean serial breakdown of where time is spent.
    var timings = false;
    // the `-j N` jobs knob. `null` => default (a cpu-based pool, i.e. the
    // std runtime's `.unlimited` concurrent_limit); `1` => `.limited(0)` (forces
    // every `fanOut` onto its inline serial fallback — the true serial baseline);
    // `N>=2` => `.limited(N)` (cap the worker pool at N). The CLI's `init.io` is
    // fixed at `.unlimited` and not reconfigurable, so the `-o` path builds its OWN
    // pool from this limit (see emitExecutable).
    var jlimit: Io.Limit = .unlimited;
    // Worker count for the "built with N threads" line: the `-j N` value, or the host
    // cpu count for the default (`.unlimited`) pool. 0 means "not set → use cpu count".
    var job_count: usize = 0;
    var paths: std.ArrayList([]const u8) = .empty;
    defer paths.deinit(gpa);

    var args = init.minimal.args.iterate();
    _ = args.skip(); // program name
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--dump")) {
            dump = true;
        } else if (std.mem.eql(u8, arg, "--codegen-stats")) {
            codegen_stats = true;
        } else if (std.mem.eql(u8, arg, "--verify")) {
            mode = .verify;
        } else if (std.mem.eql(u8, arg, "--force")) {
            mode = .force;
        } else if (std.mem.eql(u8, arg, "-O0")) {
            opt = .O0;
        } else if (std.mem.eql(u8, arg, "-O1")) {
            opt = .O1;
        } else if (std.mem.eql(u8, arg, "--opt-stats")) {
            opt_stats = true;
        } else if (std.mem.eql(u8, arg, "--timings")) {
            timings = true;
        } else if (std.mem.startsWith(u8, arg, "--opt=")) {
            // Start from level-off, then turn ON each named pass.
            opt = .O0;
            var it = std.mem.splitScalar(u8, arg["--opt=".len..], ',');
            while (it.next()) |name| {
                if (name.len == 0) return argError(out, "--opt expects a comma-separated pass list (fold,branch,dce,forward)");
                const p = passByName(name) orelse return argError(out, "--opt: unknown pass (expected fold,branch,dce,forward)");
                opt.set(p, true);
            }
        } else if (std.mem.startsWith(u8, arg, "--no-opt=")) {
            // Turn the named pass OFF from the current config (e.g. -O1 --no-opt=forward).
            const name = arg["--no-opt=".len..];
            const p = passByName(name) orelse return argError(out, "--no-opt: unknown pass (expected fold,branch,dce,forward)");
            opt.set(p, false);
        } else if (std.mem.eql(u8, arg, "--emit")) {
            emit_explicit = true;
            const v = args.next() orelse return argError(out, "--emit requires a value (lex|parse|check|ir)");
            if (std.mem.eql(u8, v, "lex")) {
                emit = .lex;
            } else if (std.mem.eql(u8, v, "parse")) {
                emit = .parse;
            } else if (std.mem.eql(u8, v, "check")) {
                emit = .check;
            } else if (std.mem.eql(u8, v, "ir")) {
                emit = .ir;
            } else {
                return argError(out, "--emit must be 'lex', 'parse', 'check', or 'ir'");
            }
        } else if (std.mem.eql(u8, arg, "-j")) {
            const v = args.next() orelse return argError(out, "-j requires a thread count (N>=1)");
            const n = std.fmt.parseInt(usize, v, 10) catch return argError(out, "-j expects a positive integer");
            if (n < 1) return argError(out, "-j must be >= 1");
            // -j1 => .limited(0): `Io.concurrent` then always returns
            // ConcurrencyUnavailable (Threaded busy_count >= 0 is always true), so
            // every fanOut takes its verbatim inline serial path — zero workers.
            // -jN => .limited(N): the pool grows to at most N concurrent workers.
            jlimit = if (n == 1) .limited(0) else .limited(n);
            job_count = n;
        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
            out_path = args.next() orelse return argError(out, "-o/--output requires an output path");
        } else if (std.mem.eql(u8, arg, "--target")) {
            target = args.next() orelse return argError(out, "--target requires a value");
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try usage(out);
            return;
        } else if (!verb_seen and paths.items.len == 0 and (std.mem.eql(u8, arg, "build") or std.mem.eql(u8, arg, "run"))) {
            // A leading `build`/`run` verb (first positional only) selects the action.
            command = if (std.mem.eql(u8, arg, "run")) .run else .build;
            verb_seen = true;
        } else {
            try paths.append(gpa, arg);
        }
    }

    if (paths.items.len == 0) {
        // Asking to emit (`-o`) with no input is an error, not usage.
        if (out_path != null) {
            try argError(out, "no input file");
            std.process.exit(1);
        }
        try usage(out);
        return;
    }

    // The DEFAULT action is to BUILD a signed executable: a bare `toyc <file>` (and any
    // `-o`/`--output`) compiles + links a binary, output defaulting to the build dir.
    // `--emit lex|parse|check|ir` opts into an inspection mode (no binary). Executable
    // emission is locked to aarch64-macos. A `build`/`run` verb or `-o`/`--output`
    // forces executable emission; otherwise a bare invocation builds by default unless an
    // inspection `--emit` was requested.
    const build_exe = verb_seen or out_path != null or !emit_explicit;
    const run_after = command == .run;
    // Worker threads the build will use (for the "built with N threads" line).
    const threads: usize = if (job_count != 0) job_count else (std.Thread.getCpuCount() catch 1);
    if (build_exe and !isAarch64Macos(target)) {
        try argError(out, "code emission only supports aarch64-macos in M1");
        std.process.exit(1);
    }

    // Build the executable: output → `-o`/`--output`, else the default build dir.
    if (build_exe) {
        std.process.exit(try emitExecutable(gpa, out, target, paths.items, out_path, run_after, mode, codegen_stats, opt, opt_stats, timings, jlimit, threads));
    }

    // `--emit ir`: print the target-independent IR for the whole program.
    if (emit == .ir) {
        std.process.exit(try emitIr(gpa, out, target, paths.items, opt, jlimit));
    }

    const results = try Driver.run(gpa, io, emit, target, paths.items);
    defer {
        for (results) |*r| r.deinit(gpa);
        gpa.free(results);
    }

    const failures = try report(out, results, emit, target, dump);
    try out.flush();
    // Signal compilation failure to scripts/CI. (Flush first; exit skips defers.)
    if (failures > 0) std.process.exit(1);
}

/// Map a `--opt`/`--no-opt` pass name to its `Opt.Pass`, or null if unknown.
fn passByName(name: []const u8) ?Opt.Pass {
    if (std.mem.eql(u8, name, "fold")) return .fold;
    if (std.mem.eql(u8, name, "branch")) return .branch;
    if (std.mem.eql(u8, name, "dce")) return .dce;
    if (std.mem.eql(u8, name, "forward")) return .forward;
    return null;
}

/// True if `target` names the aarch64-macos triple we can emit for (or `native`,
/// which on this host is aarch64-macos). Accepts the common spellings.
fn isAarch64Macos(target: []const u8) bool {
    const ok = [_][]const u8{
        "native",
        "aarch64-macos",
        "arm64-macos",
        "aarch64-apple-macos",
        "aarch64-apple-darwin",
    };
    for (ok) |t| if (std.mem.eql(u8, target, t)) return true;
    return false;
}

/// The basename of a path (after the last '/'), used as the code-signing
/// identifier. Falls back to the whole string when there is no separator.
fn basename(path: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| return path[i + 1 ..];
    return path;
}

/// Strip a trailing `.toy` source extension for the default output binary name.
fn stemOf(name: []const u8) []const u8 {
    if (std.mem.endsWith(u8, name, ".toy")) return name[0 .. name.len - ".toy".len];
    return name;
}

/// The default executable output when no `-o`/`--output` is given:
/// `.toy/<stamp>/build/<entry-stem>`. Creates the build dir on demand; the path is
/// written into `buf` (caller-owned, must outlive the write).
fn defaultOutputPath(io: Io, buf: []u8, entry: []const u8) ![]const u8 {
    var stamp_buf: [version.stamp_max]u8 = undefined;
    var dir_buf: [Driver.cache_root.len + 1 + version.stamp_max + "/build".len]u8 = undefined;
    const build_dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/build", .{ Driver.cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    try Io.Dir.cwd().createDirPath(io, build_dir);
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ build_dir, stemOf(basename(entry)) }) catch unreachable;
}

/// Print the always-on `built with <threads> thread(s) in <n> <unit>` line. Single
/// integers for ns/ms/s; minutes carry a seconds remainder. Units: ns, ms, s, min.
fn printBuildTime(out: *Io.Writer, threads: usize, ns: u64) !void {
    const unit: []const u8 = if (threads == 1) "thread" else "threads";
    if (ns < std.time.ns_per_ms) {
        try out.print("built with {d} {s} in {d} ns\n", .{ threads, unit, ns });
    } else if (ns < std.time.ns_per_s) {
        try out.print("built with {d} {s} in {d} ms\n", .{ threads, unit, ns / std.time.ns_per_ms });
    } else if (ns < std.time.ns_per_min) {
        try out.print("built with {d} {s} in {d} s\n", .{ threads, unit, ns / std.time.ns_per_s });
    } else {
        try out.print("built with {d} {s} in {d} min {d} s\n", .{ threads, unit, ns / std.time.ns_per_min, (ns % std.time.ns_per_min) / std.time.ns_per_s });
    }
}

/// `run` subcommand tail: execute the freshly built binary, wait for it, report how it
/// ended, and adopt its exit status. A clean exit returns the child's code; a signal
/// returns 128+signo (the shell convention); anything else → 1.
fn runBinary(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, path: []const u8) !u8 {
    // Spawn by ABSOLUTE path: the codesigned binary must be exec'd by a real path, and a
    // bare name without a `/` would be looked up on PATH rather than in the build dir.
    const abs = Io.Dir.cwd().realPathFileAlloc(io, path, gpa) catch |e| {
        try out.print("run: cannot locate {s}: {t}\n", .{ path, e });
        try out.flush();
        return 1;
    };
    defer gpa.free(abs);

    var child = std.process.spawn(io, .{ .argv = &.{abs} }) catch |e| {
        try out.print("run: failed to launch {s}: {t}\n", .{ basename(path), e });
        try out.flush();
        return 1;
    };
    const term = child.wait(io) catch |e| {
        try out.print("run: error awaiting {s}: {t}\n", .{ basename(path), e });
        try out.flush();
        return 1;
    };
    const name = basename(path);
    switch (term) {
        .exited => |code| {
            try out.print("{s} exited with code {d}\n", .{ name, code });
            try out.flush();
            return code;
        },
        .signal => |sig| {
            const s: u32 = @intCast(@intFromEnum(sig));
            try out.print("{s} killed by signal {d}\n", .{ name, s });
            try out.flush();
            return @intCast(128 + (s & 0x7f));
        },
        .stopped => |sig| {
            const s: u32 = @intCast(@intFromEnum(sig));
            try out.print("{s} stopped by signal {d}\n", .{ name, s });
            try out.flush();
            return 1;
        },
        .unknown => |st| {
            try out.print("{s} terminated abnormally (status {d})\n", .{ name, st });
            try out.flush();
            return 1;
        },
    }
}

/// Monotonic nanoseconds from the `Io` clock (CLOCK_UPTIME_RAW on macOS). Zig 0.16
/// has no `std.time.Timer`; timing is an `Io` capability.
fn nowNs(io: Io) i128 {
    return Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
}

/// Delta since `last` (advancing it), in ns, for the `--timings` profile. Reports 0
/// when timings are off, so a plain build never reads the clock.
fn lapNs(io: Io, timings: bool, last: *i128) u64 {
    if (!timings) return 0;
    const n = nowNs(io);
    const dt = n - last.*;
    last.* = n;
    return if (dt > 0) @intCast(dt) else 0;
}

/// Print the `--timings` per-stage breakdown (ms + % of the wall-clock total).
/// `total` is the BUILD WALL-CLOCK (`elapsed`), and the printed stage rows + the
/// `other/overhead` reconciliation row SUM TO IT — there is no unattributed time left
/// hiding. Each cache-backed stage (discover, lower) carries a compute / cache-get /
/// cache-put sub-split (the discover split is file-read+lex+parse compute vs the
/// lex/parse content cache; the lower split is codegen compute vs the codegen cache);
/// the barrier-join stages (resolve, typecheck) have no cache so report compute only.
/// `setup` is the pre-stage window (cache open + the one bulk `pack.load`);
/// `other/overhead` = total - sum(setup..image) is the small residual the named rows
/// don't cover (the stats-print window + measurement slop).
fn printTimings(out: *Io.Writer, total: u64, setup: u64, discover: u64, resolve: u64, typecheck: u64, lower: u64, post: u64, image: u64, disc_sub: ?StageSub, low_sub: ?LowerSub) !void {
    const tot_f: f64 = @floatFromInt(if (total == 0) 1 else total);
    const row = struct {
        fn p(w: *Io.Writer, name: []const u8, ns: u64, tf: f64) !void {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const pct = @as(f64, @floatFromInt(ns)) * 100.0 / tf;
            try w.print("  {s:<11}{d:>9.3} ms  ({d:>4.1}%)\n", .{ name, ms, pct });
        }
        // A stage's SUB-rows are attributed as a fraction of the PARENT stage (not the
        // whole build): they answer "where does this stage's time go". At -jN the
        // per-unit compute/get/put are summed across workers, so they can exceed the
        // wall-clock parent — that ratio IS the parallel-overlap signal.
        fn sub_p(w: *Io.Writer, name: []const u8, ns: u64, parent_ns: u64, parent: []const u8) !void {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const pf: f64 = @floatFromInt(if (parent_ns == 0) 1 else parent_ns);
            const pct = @as(f64, @floatFromInt(ns)) * 100.0 / pf;
            try w.print("    {s:<13}{d:>9.3} ms  ({d:>5.1}% of {s})\n", .{ name, ms, pct, parent });
        }
    };
    try out.print("timings:\n", .{});
    try row.p(out, "setup", setup, tot_f);
    try row.p(out, "discover", discover, tot_f);
    if (disc_sub) |s| {
        try row.sub_p(out, "compute", s.compute_ns, discover, "discover");
        try row.sub_p(out, "cache-get", s.get_ns, discover, "discover");
        try row.sub_p(out, "cache-put", s.put_ns, discover, "discover");
    }
    try row.p(out, "resolve", resolve, tot_f);
    try row.p(out, "typecheck", typecheck, tot_f);
    try row.p(out, "lower", lower, tot_f);
    if (low_sub) |s| {
        try row.sub_p(out, "compute", s.compute_ns, lower, "lower");
        try row.sub_p(out, "cache-get", s.get_ns, lower, "lower");
        try row.sub_p(out, "cache-put", s.put_ns, lower, "lower");
        try row.sub_p(out, "link-tail", s.link_ns, lower, "lower");
    }
    // The stats-print window (reads ~0 unless `--codegen-stats`/`--opt-stats` print).
    try row.p(out, "post", post, tot_f);
    try row.p(out, "image+sign", image, tot_f);
    // RECONCILE TO TOTAL: everything the named rows didn't cover (measurement slop +
    // the tiny tail after the image lap). Printed, never hidden — a saturating
    // subtraction so clock jitter can't underflow it.
    const named = setup +| discover +| resolve +| typecheck +| lower +| post +| image;
    const other = total -| named;
    try row.p(out, "other/ovh", other, tot_f);
    try out.print("  {s:<11}{d:>9.3} ms  (wall-clock)\n", .{ "total", @as(f64, @floatFromInt(total)) / 1_000_000.0 });
    try out.flush();
}

/// A cache-backed stage's compute/get/put sub-split (the generic `StageProbe` snapshot).
/// Used for `discover` (file-read+lex+parse compute vs the lex/parse content cache).
const StageSub = struct { compute_ns: u64, get_ns: u64, put_ns: u64 };

/// The `--timings` sub-breakdown of `lower`: codegen compute (summed across workers),
/// cache get/put I/O (summed across workers), and the serial relink/link tail.
const LowerSub = struct { compute_ns: u64, get_ns: u64, put_ns: u64, link_ns: u64 };

/// Run lex→parse→check→codegen→link→sign for a single input and write the signed
/// executable to `out_path` (mode 0o755). Returns the process exit code: 0 on
/// success, 1 on any failure (front-end errors, no `main`, unsupported node).
fn emitExecutable(
    gpa: std.mem.Allocator,
    out: *Io.Writer,
    target: []const u8,
    paths: []const []const u8,
    out_path: ?[]const u8,
    run_after: bool,
    mode: Engine.Mode,
    codegen_stats: bool,
    opt: Opt.Config,
    opt_stats: bool,
    timings: bool,
    jlimit: Io.Limit,
    threads: usize,
) !u8 {
    // `-o` takes the 1 ROOT (entry) file; the driver discovers the transitive
    // import graph from it and compiles the whole program.
    if (paths.len != 1) {
        try argError(out, "-o takes exactly one input file (the entry module)");
        return 1;
    }

    // the `-o` build runs on OUR OWN pool sized by `-j`, not the CLI's fixed
    // `.unlimited` io. `.limited(0)` (from `-j1`) drives every fanOut serial so the
    // single-thread build is the byte-identity baseline; `.limited(N)` caps workers.
    var tail_io: std.Io.Threaded = .init(gpa, .{ .concurrent_limit = jlimit });
    defer tail_io.deinit();
    const io = tail_io.io();

    // Total build wall-clock for the always-on "built binary in X" line (one clock read).
    const build_start = nowNs(io);

    // Resolve the output: `-o`/`--output` wins; otherwise default to
    // `.toy/<stamp>/build/<entry-stem>` (the build dir is created on demand).
    var out_buf: [std.fs.max_path_bytes]u8 = undefined;
    const resolved_out = out_path orelse try defaultOutputPath(io, &out_buf, paths[0]);

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    var pack = toyc.Cache.Pack.init(gpa);
    defer pack.deinit();
    const cache = try Driver.openCachePack(io, &dir_buf, &pack);
    // BULK-READ the packed object store ONCE: every subsequent `cache.get` (front-end
    // lex/parse + the per-fn codegen fan-out) is then a memory lookup instead of a
    // per-entry `readFileAlloc` syscall. A missing/torn pack stays cold (empty index).
    pack.load(gpa, io, cache.dir);
    // WRITE the merged pack ONCE at build end (deferred so it runs on every exit path,
    // matching the per-file path's best-effort persistence regardless of late errors).
    // This collapses the 6200×4 per-entry syscalls to ~4 (one create+writev+rename).
    defer pack.flush(io, cache.dir);

    // Per-stage wall-clock (only when `--timings`). The monotonic timer is lapped at
    // each stage boundary; accumulators live at fn scope so the success path can print
    // the breakdown. The discover/resolve barrier joins + typecheck's Pass-A prologue
    // are serial; the per-fn body checks (inside the typecheck stage) and `lower` fan
    // out with more workers — so a `-j1 --timings` run is the clean "where does the
    // time go" profile that scopes the parallelization work.
    var last_ns: i128 = if (timings) nowNs(io) else 0;
    // reconcile-to-total: the SETUP window (build_start -> here) covers the
    // output-dir create, cache open, and the ONE bulk `pack.load` — all BEFORE the first
    // stage lap, so previously it was unattributed "other". Lap it explicitly as its own
    // row so the printed columns sum to the wall-clock total with only a small labeled
    // residual.
    const ns_setup: u64 = if (timings) blk: {
        const dt = last_ns - build_start;
        break :blk if (dt > 0) @intCast(dt) else 0;
    } else 0;
    var ns_discover: u64 = 0;
    var ns_resolve: u64 = 0;
    var ns_typecheck: u64 = 0;
    var ns_lower: u64 = 0;
    // The post-lower window: the `--codegen-stats`/`--opt-stats` prints. Previously
    // DISCARDED (lap-reset) so it wouldn't pollute the image bucket, which left it as
    // unattributed "other". Captured as its own labeled row so the residual
    // stays small (it reads ~0 now that nothing heavy happens here).
    var ns_post: u64 = 0;
    var ns_image: u64 = 0;

    // `--timings` SUB-stage attribution of `lower` and `discover`: a borrowed probe per
    // stage splits its cache-backed queries into compute (the miss path) vs cache
    // get/put I/O. The lower probe splits the codegen fan-out (`lowerOne` vs cache I/O)
    // and `link_ns` captures the serial relink/link tail; the discover probe splits the
    // per-module file-read+lex+parse compute vs the lex/parse content-cache reads. Null
    // when timings are off => zero clock reads in the hot paths.
    var lower_probe: Engine.StageProbe = .{};
    var discover_probe: Engine.StageProbe = .{};
    var link_ns: u64 = 0;
    const probe_ptr: ?*Engine.StageProbe = if (timings) &lower_probe else null;
    const discover_probe_ptr: ?*Engine.StageProbe = if (timings) &discover_probe else null;
    const link_ns_ptr: ?*u64 = if (timings) &link_ns else null;

    // The whole build is ONE `StageGraph.pipeline` the interpreter drives; the
    // `Orchestrator` (below) supplies each stage's compute and writes its result into
    // one of these frame-local optionals, torn down by the `defer`s regardless of how
    // far the build got.
    var graph: ?Graph.Graph = null;
    defer if (graph) |*g| g.deinit(gpa);
    var res: ?ResolveGraph.GraphResult = null;
    defer if (res) |*r| r.deinit(gpa);
    var tc: ?TypecheckGraph.GraphResult = null;
    defer if (tc) |*t| t.deinit(gpa);
    var lowered: ?Codegen.LowerProgramResult = null;
    defer if (lowered) |*lw| switch (lw.*) {
        .ok => |*lp| lp.deinit(gpa),
        .err => {},
    };
    var ir_unused: ?Codegen.IrResult = null; // the `.lower` tail never writes this
    var failed_stage: ?Orchestrator.Stage = null;

    // The DISCOVER barrier's single contributor: the entry path's digest (which
    // program is being built). The COLLECT / GLOBAL_TABLES barriers fold a real
    // multi-element multiset; their scratch lists live here so the `contributors` hook
    // can fill + return a slice that outlives the `barrier` call.
    const entry_contributors = [_]u64{std.hash.Wyhash.hash(0x44_53_43_56, paths[0])}; // "DSCV"
    var collect_contribs: std.ArrayList(u64) = .empty;
    defer collect_contribs.deinit(gpa);
    var gt_contribs: std.ArrayList(u64) = .empty;
    defer gt_contribs.deinit(gpa);

    const orch = Orchestrator{
        .gpa = gpa,
        .io = io,
        .cache = cache,
        .target = target,
        .entry = paths[0],
        .tail = .lower,
        .entry_contributors = &entry_contributors,
        .collect_contribs = &collect_contribs,
        .gt_contribs = &gt_contribs,
        .mode = mode,
        .opt = opt,
        .discover_probe = discover_probe_ptr,
        .probe = probe_ptr,
        .link_ns = link_ns_ptr,
        // `-j` chunk-count basis for the body-check + codegen fan-outs. `threads` is the
        // resolved jobs count (the `-j N` value, else the host cpu count).
        .ncpu = threads,
        .timings = timings,
        .last_ns = &last_ns,
        .ns_discover = &ns_discover,
        .ns_resolve = &ns_resolve,
        .ns_typecheck = &ns_typecheck,
        .ns_lower = &ns_lower,
        .graph = &graph,
        .res = &res,
        .tc = &tc,
        .lowered = &lowered,
        .ir = &ir_unused,
        .failed_stage = &failed_stage,
    };

    const engine = Engine.initProbe(cache, mode, probe_ptr);
    if (!try runPipeline(out, gpa, engine, &orch)) return 1;

    // The interpreter ran every stage clean: `lowered` is `.ok` with no diagnostics.
    const lp = &lowered.?.ok;

    if (codegen_stats) {
        try out.print("codegen: compiled={d} cached={d}\n", .{ lp.codegen_compiled, lp.codegen_cached });
        try out.flush();
    }

    // dual-metric counters, fixed field order, deterministic. Cached fns
    // contribute 0 to the opt counters; use --force for honest numbers.
    if (opt_stats) {
        const s = lp.opt_stats;
        try out.print("opt: rounds={d} ir_instrs={d}->{d} emitted_instrs={d}\n", .{
            s.rounds, s.ir_instrs_before, lp.ir_instrs, lp.emitted_instrs,
        });
        try out.print("  folded={d} branches={d} blocks={d} dced={d} forwarded={d} values_pruned={d}\n", .{
            s.consts_folded, s.branches_folded, s.blocks_removed, s.instrs_dced, s.loads_forwarded, s.values_pruned,
        });
        try out.flush();
    }

    // CAPTURE (not discard) the stats-print window into its own `post` row, so it is
    // attributed rather than leaking into image+sign or the residual. Laps the timer so
    // image+sign measures only the image build below.
    ns_post = lapNs(io, timings, &last_ns);
    const image = try Codegen.buildImage(
        io,
        gpa,
        basename(resolved_out),
        lp.text,
        lp.entry_off,
        lp.cstrings,
        lp.data_relocs,
        lp.uses_write,
    );
    defer gpa.free(image);

    try writeExecutable(io, resolved_out, image);
    ns_image = lapNs(io, timings, &last_ns);
    const elapsed = nowNs(io) - build_start;
    try printBuildTime(out, threads, if (elapsed > 0) @intCast(elapsed) else 0);
    try out.flush();
    if (timings) {
        const low_sub: LowerSub = .{
            .compute_ns = lower_probe.compute_ns.load(.monotonic),
            .get_ns = lower_probe.get_ns.load(.monotonic),
            .put_ns = lower_probe.put_ns.load(.monotonic),
            .link_ns = link_ns,
        };
        const disc_sub: StageSub = .{
            .compute_ns = discover_probe.compute_ns.load(.monotonic),
            .get_ns = discover_probe.get_ns.load(.monotonic),
            .put_ns = discover_probe.put_ns.load(.monotonic),
        };
        const total_wall: u64 = if (elapsed > 0) @intCast(elapsed) else 0;
        try printTimings(out, total_wall, ns_setup, ns_discover, ns_resolve, ns_typecheck, ns_lower, ns_post, ns_image, disc_sub, low_sub);
    }
    // `run`: execute the freshly built binary and adopt its exit status.
    if (run_after) return runBinary(gpa, io, out, resolved_out);
    return 0;
}

/// The SINGLE stage adapter for `StageGraph.interpret`, shared by every
/// program-producing build (`-o`, `--emit ir`). It supplies each stage's compute
/// (discover / resolve / typecheck / codegen) and writes the result into a frame-local
/// optional on the caller's stack, indexed by the comptime stage position in
/// `StageGraph.pipeline` (which defines the cadence and what each barrier folds; the
/// `comptime` block below pins this `Stage` enum to it). A barrier's compute is the JOIN
/// work that produces the tables the next stage demands.
///
/// The codegen stage's behavior is selected by `tail`: `.lower` (the `-o` path) runs
/// `lowerGraphProgram` into `lowered`; `.render_ir` (`--emit ir`) runs `renderGraphIr`
/// into `ir`. The discover/resolve/typecheck stages are IDENTICAL across paths — there
/// is ONE orchestration, not one per output kind.
///
/// A stage whose result carries diagnostics (or a structural discover error) raises
/// `error.StageDiagnostics` after setting `failed_stage`, so the interpreter stops and
/// `runPipeline` renders that stage's diagnostics. Hard errors (OOM/IO) propagate
/// as-is. Every result slot is owned by the caller (torn down by its `defer`s), so a
/// mid-pipeline failure never leaks. `--timings` laps (`ns_*`) are best-effort and may
/// be null pointers on the inspection paths that don't profile.
const Orchestrator = struct {
    pub const Stage = enum { discover, resolve, typecheck, codegen };
    /// What the codegen region (stage 3) does: lower to a `LinkedProgram` (image/sign
    /// tail) or render the IR text. The front-end stages are identical.
    pub const Tail = enum { lower, render_ir };

    comptime {
        // The hooks dispatch on `Stage` derived from the pipeline position
        // (`@enumFromInt(stage_i)`); this pins each `Stage` to its `StageGraph.pipeline`
        // entry, so a pipeline reorder/insert is a compile error here (a rename to fix),
        // not a silent hook↔stage misdispatch.
        const pipeline = toyc.StageGraph.pipeline;
        const fields = @typeInfo(Stage).@"enum".fields;
        if (pipeline.len != fields.len)
            @compileError("Orchestrator.Stage count must match StageGraph.pipeline length");
        for (fields) |f| {
            const aligned = switch (@as(Stage, @enumFromInt(f.value))) {
                .discover => pipeline[f.value].kind == .discover,
                .resolve => pipeline[f.value].kind == .collect,
                .typecheck => pipeline[f.value].kind == .global_tables,
                .codegen => pipeline[f.value].kind == .codegen,
            };
            if (!aligned)
                @compileError("Orchestrator stage '" ++ f.name ++ "' is not aligned with its StageGraph.pipeline entry");
        }
    }

    gpa: std.mem.Allocator,
    io: Io,
    cache: toyc.Cache,
    target: []const u8,
    entry: []const u8,
    tail: Tail,
    /// The single DISCOVER-barrier contributor (the entry-path digest), borrowed from
    /// the caller's frame so the `contributors` hook can return a stable slice.
    entry_contributors: []const u64,
    /// Scratch lists (on the caller's frame) the `contributors` hook fills for the
    /// COLLECT / GLOBAL_TABLES barriers: the per-module parse digests and the per-fn
    /// resolve digests. Filled lazily when their barrier is reached (the upstream
    /// stage's result is ready by then) and returned as a stable slice for the fold.
    collect_contribs: *std.ArrayList(u64),
    gt_contribs: *std.ArrayList(u64),
    mode: Engine.Mode,
    opt: Opt.Config,
    /// the discover stage's compute/cache probe (file-read+lex+parse compute
    /// vs the lex/parse content cache). Distinct from `probe` (the lower stage's), so
    /// the two stages' sub-splits never co-mingle. Null when not profiling.
    discover_probe: ?*Engine.StageProbe,
    probe: ?*Engine.LowerProbe,
    link_ns: ?*u64,
    /// The `-j` jobs knob (0 => host cpus): the chunk-count basis for the two hot
    /// per-fn fan-outs the orchestrator drives — the GLOBAL_TABLES body checks
    /// (`checkGraph`) and the codegen region (`lowerGraphProgram`).
    ncpu: usize,

    // `--timings` per-stage laps (each stage closure charges its own bucket). Null
    // pointers on paths that don't profile (`--emit ir`) => no lap.
    timings: bool,
    last_ns: ?*i128,
    ns_discover: ?*u64,
    ns_resolve: ?*u64,
    ns_typecheck: ?*u64,
    ns_lower: ?*u64,

    // Result slots on the caller's frame (it owns teardown). `lowered` is written by
    // the `.lower` tail, `ir` by the `.render_ir` tail.
    graph: *?Graph.Graph,
    res: *?ResolveGraph.GraphResult,
    tc: *?TypecheckGraph.GraphResult,
    lowered: *?Codegen.LowerProgramResult,
    ir: *?Codegen.IrResult,
    failed_stage: *?Stage,

    /// Lap a `--timings` bucket if both the accumulator and the running timer are
    /// present; a no-op on the inspection paths (null pointers).
    fn lap(self: Orchestrator, bucket: ?*u64) void {
        if (bucket) |b| if (self.last_ns) |ln| {
            b.* = lapNs(self.io, self.timings, ln);
        };
    }

    /// The contributor multiset for barrier stage `i`, folded (order-independently) by
    /// `Engine.barrier` into the join's id:
    ///   0 DISCOVER      — the entry-path digest (borrowed, identifies the program);
    ///   1 COLLECT       — every discovered module's parse digest (`Ast.contentFp`),
    ///                     the inputs the global resolve tables are built from;
    ///   2 GLOBAL_TABLES — every global fn's resolve digest (module+name+decl), the
    ///                     inputs the program-wide layout/sig tables are built from.
    /// Stages 1/2 fill a caller-frame scratch list (the upstream result is ready) and
    /// return its slice; it outlives the `barrier` call that reads it.
    pub fn contributors(self: Orchestrator, comptime stage_i: usize) []const u64 {
        switch (@as(Stage, @enumFromInt(stage_i))) {
            .discover => return self.entry_contributors,
            .resolve => {
                const list = self.collect_contribs;
                list.clearRetainingCapacity();
                for (self.graph.*.?.modules) |*m| list.append(self.gpa, Ast.contentFp(m.tree())) catch {};
                return list.items;
            },
            .typecheck => {
                const list = self.gt_contribs;
                list.clearRetainingCapacity();
                for (self.res.*.?.fns) |gf| list.append(self.gpa, fnResolveDigest(gf)) catch {};
                return list.items;
            },
            .codegen => comptime unreachable,
        }
    }

    pub fn barrierCompute(self: Orchestrator, comptime stage_i: usize) switch (@as(Stage, @enumFromInt(stage_i))) {
        .discover => DiscoverCompute,
        .resolve => ResolveCompute,
        .typecheck => TypecheckCompute,
        .codegen => unreachable,
    } {
        return switch (@as(Stage, @enumFromInt(stage_i))) {
            .discover, .resolve, .typecheck => .{ .o = self },
            .codegen => comptime unreachable,
        };
    }

    /// A barrier produced its join: lap the stage's timing bucket. The `fold` carried
    /// by `BarrierResult` is observability (recorded as the barrier node id); the join
    /// tables themselves live in the frame slots the compute wrote.
    pub fn recordBarrier(self: Orchestrator, comptime stage_i: usize, _: Engine.BarrierResult(void)) void {
        switch (@as(Stage, @enumFromInt(stage_i))) {
            .discover => self.lap(self.ns_discover),
            .resolve => self.lap(self.ns_resolve),
            .typecheck => self.lap(self.ns_typecheck),
            .codegen => comptime unreachable,
        }
    }

    /// The codegen REGION (stage 3): run the stage, lap its timing bucket, and raise
    /// `error.StageDiagnostics` (after recording which stage) if it produced
    /// diagnostics. The region owns its internal per-fn `Engine.fanOut` + relink join.
    pub fn region(self: Orchestrator, comptime stage_i: usize) !void {
        comptime std.debug.assert(@as(Stage, @enumFromInt(stage_i)) == .codegen);
        switch (self.tail) {
            .lower => {
                self.lowered.* = try Codegen.lowerGraphProgram(self.gpa, self.io, self.cache, self.target, &self.graph.*.?, &self.res.*.?, &self.tc.*.?, self.mode, self.opt, self.probe, self.link_ns, self.ncpu);
                self.lap(self.ns_lower);
                const bad = switch (self.lowered.*.?) {
                    .err => true,
                    .ok => |lp| lp.diags.len > 0,
                };
                if (bad) {
                    self.failed_stage.* = .codegen;
                    return error.StageDiagnostics;
                }
            },
            .render_ir => {
                self.ir.* = try Codegen.renderGraphIr(self.gpa, &self.graph.*.?, &self.res.*.?, &self.tc.*.?, self.opt);
                self.lap(self.ns_lower);
                if (self.ir.*.? == .err) {
                    self.failed_stage.* = .codegen;
                    return error.StageDiagnostics;
                }
            },
        }
    }
};

/// The per-fn GLOBAL_TABLES contributor: fold one global fn's resolve identity
/// (owning module + qualified name + decl node) into a u64. This is what the resolve
/// stage settled for the fn; the typecheck Pass-A tables are built from the whole set,
/// so folding all of them is the genuine fan-in the GLOBAL_TABLES barrier joins.
fn fnResolveDigest(gf: ResolveGraph.GlobalFn) u64 {
    var h = std.hash.Wyhash.init(0x52_53_4c_56); // "RSLV"
    var b: [8]u8 = undefined;
    std.mem.writeInt(u32, b[0..4], gf.module, .little);
    std.mem.writeInt(u32, b[4..8], gf.decl_node, .little);
    h.update(&b);
    h.update(gf.name);
    return h.final();
}

/// The DISCOVER barrier's compute: build the module graph from the entry path. A
/// structural discover error lands in `graph.err`; signal it as a stage failure so
/// the interpreter stops and `runPipeline` renders it. Returns `void` — the graph is
/// written into the frame slot as a side effect (uniform with the other barriers).
const DiscoverCompute = struct {
    o: Orchestrator,
    pub fn run(c: DiscoverCompute) !void {
        c.o.graph.* = try Graph.discover(c.o.gpa, c.o.io, c.o.cache, c.o.target, c.o.entry, c.o.discover_probe);
        if (c.o.graph.*.?.err != null) {
            c.o.failed_stage.* = .discover;
            return error.StageDiagnostics;
        }
    }
};

/// The COLLECT barrier's compute: build the whole-graph name-resolution tables
/// (`resolveGraph` — global fn/type tables + per-fn binding). The join over the module
/// parse digests; its result is written into the `res` frame slot. Resolve diagnostics
/// raise `error.StageDiagnostics`.
const ResolveCompute = struct {
    o: Orchestrator,
    pub fn run(c: ResolveCompute) !void {
        c.o.res.* = try ResolveGraph.resolveGraph(c.o.gpa, &c.o.graph.*.?);
        if (c.o.res.*.?.diags.len > 0) {
            c.o.failed_stage.* = .resolve;
            return error.StageDiagnostics;
        }
    }
};

/// The GLOBAL_TABLES barrier's compute: build the program-wide layout/sig tables
/// (typecheck Pass-A) and check every fn body — the per-fn body checks fan out INSIDE
/// `checkGraph` (via `Engine.fanOut` over the worker pool) as the PER_UNIT region past
/// this barrier. The join over the per-fn resolve digests; result -> the `tc` frame
/// slot. Type diagnostics raise `error.StageDiagnostics`.
const TypecheckCompute = struct {
    o: Orchestrator,
    pub fn run(c: TypecheckCompute) !void {
        c.o.tc.* = try TypecheckGraph.checkGraph(c.o.gpa, &c.o.graph.*.?, &c.o.res.*.?, c.o.io, c.o.ncpu);
        if (c.o.tc.*.?.diags.len > 0) {
            c.o.failed_stage.* = .typecheck;
            return error.StageDiagnostics;
        }
    }
};

/// Drive the shared front-end (discover -> resolve -> typecheck -> codegen) through
/// `StageGraph.interpret` and render any stage's diagnostics. This is the ONE
/// sequencer every program-producing build runs — `-o` and `--emit ir` both call it,
/// differing only in the codegen `tail` and the post-pipeline output.
/// Returns `true` on clean completion (the caller reads the result slots), `false`
/// when a stage produced diagnostics (already printed; the caller returns exit 1).
fn runPipeline(out: *Io.Writer, gpa: std.mem.Allocator, engine: Engine, orch: *const Orchestrator) !bool {
    toyc.StageGraph.interpret(&toyc.StageGraph.pipeline, engine, gpa, orch.*) catch |e| switch (e) {
        // A stage produced diagnostics: print them against the owning module and tell
        // the caller to exit non-zero. `failed_stage` says which stage's diagnostics.
        error.StageDiagnostics => {
            const g = &orch.graph.*.?;
            switch (orch.failed_stage.*.?) {
                .discover => try printGraphError(out, g, g.err.?),
                .resolve => for (orch.res.*.?.diags) |d| try printModuleDiag(out, g, d.scope, d.byte_offset, d.message),
                .typecheck => for (orch.tc.*.?.diags) |d| try printModuleDiag(out, g, d.scope, d.byte_offset, d.message),
                .codegen => switch (orch.tail) {
                    .lower => switch (orch.lowered.*.?) {
                        .err => |ee| try printGraphEmitError(out, g, ee),
                        .ok => |lp| for (lp.diags) |d| try printGraphEmitError(out, g, .{ .message = d.message, .byte_offset = d.byte_offset }),
                    },
                    .render_ir => switch (orch.ir.*.?) {
                        .err => |ee| try printGraphEmitError(out, g, ee),
                        .ok => {}, // an .ok IR result never raises StageDiagnostics
                    },
                },
            }
            try out.flush();
            return false;
        },
        else => return e,
    };
    return true;
}

/// `--emit ir`: run the SAME front-end StageGraph the `-o` build drives, with the
/// codegen `tail` set to `.render_ir` so the final stage renders the deterministic IR
/// text instead of lowering. Prints the IR to stdout. Returns the process exit code.
fn emitIr(
    gpa: std.mem.Allocator,
    out: *Io.Writer,
    target: []const u8,
    paths: []const []const u8,
    opt: Opt.Config,
    jlimit: Io.Limit,
) !u8 {
    // `--emit ir` takes the 1 ROOT (entry) file; discover the whole graph.
    if (paths.len != 1) {
        try argError(out, "--emit ir takes exactly one input file (the entry module)");
        return 1;
    }

    // the whole-graph typecheck Pass-C fans out per-fn body checks across our own
    // `-j`-sized pool. `.limited(0)` (-j1) drives every job onto its inline serial path
    // — the byte-identity baseline against which -jN must produce IDENTICAL diagnostics.
    var tail_io: std.Io.Threaded = .init(gpa, .{ .concurrent_limit = jlimit });
    defer tail_io.deinit();
    const io = tail_io.io();

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);

    // ONE orchestration: the SAME discover->resolve->typecheck->codegen StageGraph the
    // `-o` build drives, with the codegen `tail` set to `.render_ir` so stage 3 renders
    // the IR text instead of lowering. No second hand-sequenced chain.
    var graph: ?Graph.Graph = null;
    defer if (graph) |*g| g.deinit(gpa);
    var res: ?ResolveGraph.GraphResult = null;
    defer if (res) |*r| r.deinit(gpa);
    var tc: ?TypecheckGraph.GraphResult = null;
    defer if (tc) |*t| t.deinit(gpa);
    var lowered_unused: ?Codegen.LowerProgramResult = null; // the `.render_ir` tail never writes this
    var ir: ?Codegen.IrResult = null;
    defer if (ir) |*r| switch (r.*) {
        .ok => |text| gpa.free(text),
        .err => {},
    };
    var failed_stage: ?Orchestrator.Stage = null;

    const entry_contributors = [_]u64{std.hash.Wyhash.hash(0x44_53_43_56, paths[0])}; // "DSCV"
    var collect_contribs: std.ArrayList(u64) = .empty;
    defer collect_contribs.deinit(gpa);
    var gt_contribs: std.ArrayList(u64) = .empty;
    defer gt_contribs.deinit(gpa);
    const orch = Orchestrator{
        .gpa = gpa,
        .io = io,
        .cache = cache,
        .target = target,
        .entry = paths[0],
        .tail = .render_ir,
        .entry_contributors = &entry_contributors,
        .collect_contribs = &collect_contribs,
        .gt_contribs = &gt_contribs,
        .mode = .normal,
        .opt = opt,
        .discover_probe = null,
        .probe = null,
        .link_ns = null,
        // `-j` chunk basis from the pool limit: `.limited(N)`->N, `.unlimited`->host
        // cpus, `-j1` (`.limited(0)`)->1 (one chunk = the serial inline path).
        .ncpu = @max(@as(usize, 1), jlimit.toInt() orelse Engine.hostCpus()),
        .timings = false,
        .last_ns = null,
        .ns_discover = null,
        .ns_resolve = null,
        .ns_typecheck = null,
        .ns_lower = null,
        .graph = &graph,
        .res = &res,
        .tc = &tc,
        .lowered = &lowered_unused,
        .ir = &ir,
        .failed_stage = &failed_stage,
    };

    const engine = Engine.init(cache, .normal);
    if (!try runPipeline(out, gpa, engine, &orch)) return 1;

    try out.writeAll(ir.?.ok);
    try out.flush();
    return 0;
}

/// Render a graph-discovery structural error against the owning module's source
/// (or the entry path when no module loaded). Appends the cycle/file detail.
fn printGraphError(out: *Io.Writer, graph: *const Graph.Graph, e: Graph.Error) !void {
    const path = if (e.module) |m| graph.modules[m].path else if (graph.modules.len > 0) graph.entry().path else "<entry>";
    const src = if (e.module) |m| graph.modules[m].source else &[_]u8{};
    if (e.byte_offset) |off| {
        const loc = lineCol(src, off);
        try out.print("{s}:{d}:{d}: error: {s}", .{ path, loc.line, loc.col, e.message });
    } else {
        try out.print("{s}: error: {s}", .{ path, e.message });
    }
    if (e.detail.len > 0) try out.print(" ({s})", .{e.detail});
    try out.writeByte('\n');
}

/// Render a cross-module resolve/typecheck diagnostic against its owning module.
fn printModuleDiag(out: *Io.Writer, graph: *const Graph.Graph, scope: u32, byte_offset: u32, message: []const u8) !void {
    // An untagged (single-file) diagnostic renders against the entry module; a
    // graph diagnostic against its owning module.
    const m = if (scope == toyc.DiagnosticSink.NO_SCOPE) graph.entry() else &graph.modules[scope];
    const loc = lineCol(m.source, byte_offset);
    try out.print("{s}:{d}:{d}: error: {s}\n", .{ m.path, loc.line, loc.col, message });
}

/// Render a code-emission `EmitError` from a graph build against its owning module
/// (or the entry module when `module` is null).
fn printGraphEmitError(out: *Io.Writer, graph: *const Graph.Graph, e: Codegen.EmitError) !void {
    const m = if (e.module) |mi| &graph.modules[mi] else graph.entry();
    if (e.byte_offset) |off| {
        const loc = lineCol(m.source, off);
        try out.print("{s}:{d}:{d}: error: {s}\n", .{ m.path, loc.line, loc.col, e.message });
    } else {
        try out.print("{s}: error: {s}\n", .{ m.path, e.message });
    }
}

/// Write `image` to `path` as an executable (mode 0o755). `createFile`'s
/// `permissions` is subject to the process umask, so we also `setPermissions`
/// explicitly afterwards to guarantee the executable bits land.
fn writeExecutable(io: Io, path: []const u8, image: []const u8) !void {
    const exec_perms: Io.File.Permissions = .fromMode(0o755);
    var file = try Io.Dir.cwd().createFile(io, path, .{ .permissions = exec_perms });
    defer file.close(io);
    try file.writeStreamingAll(io, image);
    try file.setPermissions(io, exec_perms);
}

/// Render a code-emission error as `file:line:col: message` (or `file: message`
/// when there is no source location), matching the front-end diagnostic style.
fn printEmitError(out: *Io.Writer, r: *const Driver.FileResult, e: Codegen.EmitError) !void {
    if (e.byte_offset) |off| {
        const loc = lineCol(r.source, off);
        try out.print("{s}:{d}:{d}: error: {s}\n", .{ r.path, loc.line, loc.col, e.message });
    } else {
        try out.print("{s}: error: {s}\n", .{ r.path, e.message });
    }
}

fn argError(out: *Io.Writer, message: []const u8) !void {
    try out.print("error: {s}\n", .{message});
    try out.flush();
}

fn usage(out: *Io.Writer) !void {
    var stamp_buf: [version.stamp_max]u8 = undefined;
    try out.print(
        \\toy {s} — toy compiler (lexer + parser + name resolution + typecheck)
        \\
        \\usage: toy [build|run] [options] <file...>
        \\  build <file>      compile to a signed executable (the default action)
        \\  run <file>        build, then execute the binary and report its exit status
        \\  -o, --output <p>  output path for the built binary (default: .toy/<stamp>/build/<name>)
        \\  --emit lex|parse|check|ir  inspect the pipeline instead of building (no binary)
        \\  -j <N>            build worker threads (N>=1; -j1 = serial; default cpu-based)
        \\  --dump            print the emit phase's artifact (tokens, or the AST)
        \\  --target <triple> compilation target (default: native)
        \\  --codegen-stats   print compiled-vs-cached function counts (with -o)
        \\  --verify          re-lower cached functions and assert they match (with -o)
        \\  --force           ignore the codegen cache; lower every function (with -o)
        \\  -O0 | -O1         IR optimization level (default -O0; -O1 = all passes)
        \\  --opt=<list>      enable only these passes (fold,branch,dce,forward)
        \\  --no-opt=<pass>   disable one pass from the current level (e.g. -O1 --no-opt=forward)
        \\  --opt-stats       print per-pass opt counters + dual metric (with -o; use --force)
        \\
    , .{version.stamp(&stamp_buf)});
    try out.flush();
}

/// Print the per-file summary; returns the number of files that failed (so the
/// caller can set a non-zero exit status).
fn report(out: *Io.Writer, results: []const Driver.FileResult, emit: Driver.Emit, target: []const u8, dump: bool) !usize {
    var stamp_buf: [version.stamp_max]u8 = undefined;
    try out.print("compiler {s}  target {s}  emit {t}\n", .{ version.stamp(&stamp_buf), target, emit });

    var total_tokens: usize = 0;
    var total_bytes: usize = 0;
    var lex_hits: usize = 0;
    var parse_hits: usize = 0;
    var failures: usize = 0;

    try out.print("{s: <28} {s: >8} {s: >7} {s: >6}  {s}\n", .{ "file", "bytes", "tokens", "nodes", "status" });
    try out.writeAll("-" ** 72 ++ "\n");

    for (results) |r| {
        total_bytes += r.source.len;
        if (r.tokens_cached) lex_hits += 1;
        if (r.nodes_cached) parse_hits += 1;

        if (r.err) |err| {
            failures += 1;
            try printFailure(out, r, err);
            continue;
        }

        // Every token stream ends in a synthetic .eof we don't count as "real".
        const real_tokens = r.tokens.len -| 1;
        total_tokens += real_tokens;

        try out.print("{s: <28} {d: >8} {d: >7} ", .{ r.path, r.source.len, real_tokens });
        if (r.parsed) try out.print("{d: >6}", .{r.nodes.len}) else try out.print("{s: >6}", .{"-"});
        try out.print("  {s}\n", .{cacheNote(r, emit)});

        if (dump) try dumpArtifact(out, r, emit);
    }

    try out.writeAll("-" ** 72 ++ "\n");
    try out.print(
        "{d} file(s): {d} bytes, {d} tokens, cache hits lex={d} parse={d}, {d} failure(s)\n",
        .{ results.len, total_bytes, total_tokens, lex_hits, parse_hits, failures },
    );
    return failures;
}

/// Short per-file status: which phases were fresh vs. served from cache.
fn cacheNote(r: Driver.FileResult, emit: Driver.Emit) []const u8 {
    const lex = if (r.tokens_cached) "cached" else "lexed";
    if (emit == .lex) return lex;
    // Resolution runs in-memory (uncached); report it explicitly when reached.
    if (emit == .check and r.checked) return "checked";
    const parse = if (r.nodes_cached) "cached" else "parsed";
    if (r.tokens_cached and r.nodes_cached) return "cached";
    if (!r.tokens_cached and !r.nodes_cached) return "lexed+parsed";
    // Mixed: name the phase that actually ran.
    return if (r.tokens_cached) parse else lex;
}

fn printFailure(out: *Io.Writer, r: Driver.FileResult, err: anyerror) !void {
    if (r.diag) |d| {
        // Parse error: a single diagnostic.
        const loc = lineCol(r.source, d.byte_offset);
        try out.print("{s: <28} {d: >8} {d: >7} {s: >6}  error: {s} at {d}:{d}\n", .{
            r.path, r.source.len, r.tokens.len -| 1, "-", d.message, loc.line, loc.col,
        });
    } else if (err == error.ResolveError) {
        // Resolution error(s): list each one as file:line:col under the row.
        try out.print("{s: <28} {d: >8} {d: >7} {s: >6}  error: {d} resolve diagnostic(s)\n", .{
            r.path, r.source.len, r.tokens.len -| 1, "-", if (r.resolve) |res| res.diags.len else 0,
        });
        if (r.resolve) |res| for (res.diags) |d| {
            const loc = lineCol(r.source, d.byte_offset);
            try out.print("    {s}:{d}:{d}: {s}\n", .{ r.path, loc.line, loc.col, d.message });
        };
    } else if (err == error.TypeError) {
        // Type error(s): list each one as file:line:col under the row.
        try out.print("{s: <28} {d: >8} {d: >7} {d: >6}  error: {d} type diagnostic(s)\n", .{
            r.path, r.source.len, r.tokens.len -| 1, r.nodes.len, if (r.typecheck) |tc| tc.diags.len else 0,
        });
        if (r.typecheck) |tc| for (tc.diags) |d| {
            const loc = lineCol(r.source, d.byte_offset);
            try out.print("    {s}:{d}:{d}: {s}\n", .{ r.path, loc.line, loc.col, d.message });
        };
    } else {
        try out.print("{s: <28} {s: >8} {s: >7} {s: >6}  error: {t}\n", .{ r.path, "-", "-", "-", err });
    }
}

fn dumpArtifact(out: *Io.Writer, r: Driver.FileResult, emit: Driver.Emit) !void {
    switch (emit) {
        .lex => for (r.tokens, 0..) |tok, i| {
            try out.print("    [{d: >4}] {s: <12} {d: >5}..{d: <5} {s}\n", .{
                i, @tagName(tok.tag), tok.start, tok.end,
                if (tok.tag == .eof) "" else tok.text(r.source),
            });
        },
        .parse => {
            try out.writeAll("    ");
            try Ast.render(out, .{ .nodes = r.nodes, .extra = r.extra, .pub_bits = r.pub_bits }, r.tokens, r.source);
            try out.writeByte('\n');
        },
        // Check dump: the AST plus a per-function signature summary with the
        // inferred return type, then a tally of diagnostics (0 when clean).
        // `ir` shares the front-end with `check`; its artifact (the IR text) is
        // printed by `emitIr`, not the table dumper.
        .check, .ir => {
            try out.writeAll("    ");
            try Ast.render(out, .{ .nodes = r.nodes, .extra = r.extra, .pub_bits = r.pub_bits }, r.tokens, r.source);
            try out.writeByte('\n');
            try dumpCheck(out, r);
        },
    }
}

/// Per-function summary under `--emit check --dump`: each function's resolved
/// signature with its inferred return type, then a diagnostic count.
fn dumpCheck(out: *Io.Writer, r: Driver.FileResult) !void {
    if (r.nodes.len == 0) return;
    const tree: Ast.Tree = .{ .nodes = r.nodes, .extra = r.extra };
    const prog = r.nodes[Ast.root(r.nodes)];
    if (prog.tag != .program) return;

    const tc = r.typecheck;
    for (Ast.rangeSlice(tree, prog.lhs)) |fn_idx| {
        const decl = r.nodes[fn_idx];
        if (decl.tag != .fn_decl) continue;
        const name = r.tokens[decl.main_token].text(r.source);
        const proto = Ast.protoAt(tree, decl.lhs);

        try out.print("    fn {s}(", .{name});
        for (proto.params, 0..) |p_idx, i| {
            if (i != 0) try out.writeAll(", ");
            const p = r.nodes[p_idx];
            const p_name = r.tokens[p.main_token].text(r.source);
            const p_type = r.tokens[r.nodes[p.lhs].main_token].text(r.source);
            try out.print("{s}: {s}", .{ p_name, p_type });
        }
        const ret = if (proto.ret_type == Ast.none)
            "()"
        else if (r.nodes[proto.ret_type].tag == .literal_unit)
            "()"
        else
            r.tokens[r.nodes[proto.ret_type].main_token].text(r.source);
        // `rhs` is the body block; its inferred type isn't tracked, so report the
        // declared return spelling — the typechecker has already verified it.
        try out.print(") -> {s}\n", .{ret});
    }

    const n_diags = if (tc) |t| t.diags.len else 0;
    try out.print("    checked: {d} diagnostic(s)\n", .{n_diags});
}

const LineCol = struct { line: usize, col: usize };

fn lineCol(source: []const u8, offset: u32) LineCol {
    var line: usize = 1;
    var col: usize = 1;
    const end = @min(offset, source.len);
    for (source[0..end]) |c| {
        if (c == '\n') {
            line += 1;
            col = 1;
        } else col += 1;
    }
    return .{ .line = line, .col = col };
}
