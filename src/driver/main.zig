//! `toyc` CLI entry point.
//!
//! Usage:
//!   toy [build|run] [OPTIONS] <file...>
//!
//! With NO subcommand a bare `toy <file>` builds a signed executable (output → the
//! default build dir); `--emit lex|parse|check|ir` opts into an INSPECTION mode
//! instead (no binary). The `build`/`run` subcommands (and any `-o`/`--output`)
//! force a build; `run` additionally execs the produced binary and reports its exit
//! status. `--dump` prints the emit phase's artifact: tokens for `lex`, the AST
//! S-expression for `parse`, and the AST plus a per-function signature summary for
//! `check`; `--emit ir` prints the target-independent IR text. `-o` carries on
//! through lower → codegen → link → sign.
//!
//! CLI + DIAGNOSTICS. Argument parsing is the typed `toyc.cli` framework: the app
//! schema is `Cli.zig` (a sibling `@import`), parsed by `cli.Parser` with a
//! collect-all `cli.Sink`; `-h`/`--help` and `-V`/`--version` route to
//! `cli.Help`. Per-diagnostic lines render through the pretty
//! `toyc.term.render.Renderer` (snippet + caret), and the STATUS output (the report
//! table, --timings, the build-time line, the run-status line) is colourised via
//! `toyc.term.Style`. Colour is resolved ONCE per phase from `--color` + a
//! `detectTty(stdout)` probe + the environment (NO_COLOR / CLICOLOR / TERM), so a
//! piped/redirected/CI run resolves to `.none` and emits ZERO escape bytes — the
//! load-bearing gate that keeps diff.sh and any captured output byte-stable.

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
const Orchestrator = @import("Orchestrator.zig").Orchestrator;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const CodegenIr = toyc.CodegenIr;
const Opt = toyc.Opt;
const Engine = toyc.QueryEngine;
const version = toyc.version;
// The CLI framework + terminal library, reached through the library module (never
// via relative `../cli`/`../term` paths). `AppCli` is the sibling app schema.
const cli = toyc.cli;
const term = toyc.term;
const Style = term.Style;
const Terminal = term.Terminal;
const Rr = term.render;
const AppCli = @import("Cli.zig");

// ---- status palette --------------------------------------------------------
// Colours from Theme's bright-ansi indices so `Color.downgrade` is the identity at
// ansi16/ansi256; dim/bold are pure attributes. Every use goes through
// `Style.styled`, which writes ZERO bytes at `.none` — so on a pipe/CI (`.none`)
// the coloured path is byte-identical to plain. `err` matches the Renderer's own
// header word hue (bright red 9, bold) instead of a hardcoded normal red.
const sty_err: Style.Style = .{ .fg = .{ .ansi = 9 }, .bold = true }; // Theme.plain.style(.err)
const sty_ok: Style.Style = .{ .fg = .{ .ansi = 10 } }; // bright green
const sty_head: Style.Style = .{ .bold = true };
const sty_faint: Style.Style = .{ .dim = true };

/// The renderer options for a colour level: ASCII carets (unicode off) keep the
/// diagnostic output byte-stable and gate-safe under any future golden.
fn renderOpts(level: Style.ColorLevel) Rr.Renderer.RenderOpts {
    return .{ .color = level, .unicode = false };
}

/// Resolve the colour level ONCE from an explicit `choice`, the tty-ness of STDOUT
/// (diagnostics + status all go to `out` == stdout, so probing stdout is what makes
/// a pipe/redirect/CI resolve `.none`), and the environment (NO_COLOR / CLICOLOR /
/// CLICOLOR_FORCE / TERM). Called twice by design: with `.auto` for the pre-parse
/// error path (a bad `--color` must not break error colouring), then with the
/// parsed `--color` for all status + diagnostics.
fn resolveLevel(init: std.process.Init, io: Io, choice: Terminal.ColorChoice) Style.ColorLevel {
    const is_tty = Terminal.detectTty(std.Io.File.stdout(), io);
    const env = Terminal.EnvView.fromEnv(init.environ_map.*);
    return Terminal.resolve(choice, is_tty, env);
}

pub fn main(init: std.process.Init) !void {
    // The per-fn codegen fan-out + parallel link tail hammer the allocator from
    // every worker thread. `std.process.Init.gpa` is a mutex-guarded DebugAllocator
    // in Debug/ReleaseSafe, so concurrent allocs serialize on one lock and the
    // parallelism collapses to ~1x regardless of -j. The lock-free per-CPU
    // `smp_allocator` lets the workers allocate without contending, so -jN actually
    // scales in EVERY build mode. The allocator choice never affects output bytes
    // (determinism is structural, not alloc-order dependent), so this is safe.
    // The comptime `Parsed(cmd)` reify + the `inline for` subcommand re-entry walk
    // this spec's ~14 options across root/build/run, exceeding the default 1000
    // backwards-branch quota. Raise it here (a caller-side comptime budget, not a
    // framework edit) so the reification compiles.
    @setEvalBranchQuota(20_000);

    const gpa = std.heap.smp_allocator;
    const io = init.io;

    var stdout_buf: [4096]u8 = undefined;
    // Streaming (not positional) — stdout may be a pipe, tty, or append target,
    // none of which are seekable. Positional writes corrupt ordering there.
    var stdout_writer = Io.File.stdout().writerStreaming(io, &stdout_buf);
    const out = &stdout_writer.interface;

    // The DRIVER STATE the post-parse tail reads. Declared up front (identical set
    // to the old hand-rolled loop's locals) so the post-parse tail below stays a
    // faithful reproduction. `applyParsed` fills these from a `Parsed` result.
    var st: State = .{ .paths = .empty };
    defer st.paths.deinit(gpa);

    // COLLECT argv WITHOUT the program name into a gpa-owned list kept alive for the
    // WHOLE fn: Sink error strings AND Parsed string/append/variadic slices BORROW
    // from argv (and the parse arena), so argv + the parse Result must outlive every
    // render and the whole build. Never free this early or copy `p.file` out.
    var argv_list: std.ArrayList([]const u8) = .empty;
    defer argv_list.deinit(gpa);
    {
        var it = init.minimal.args.iterate();
        _ = it.skip(); // program name
        while (it.next()) |a| try argv_list.append(gpa, a);
    }
    const argv = argv_list.items;

    // The collect-all parse-error accumulator. The parser never prints/exits — it
    // fills this and the driver formats + prints the accumulated errors itself.
    var sink: cli.Sink.Sink = .{};
    defer sink.deinit(gpa);

    // Resolve a PRE-PARSE colour level for the error path with `.auto` (env + tty
    // only): a parse error may itself be a bad `--color`, so the error colouring
    // must not depend on the value we are about to reject.
    const err_level = resolveLevel(init, io, .auto);

    var root_res = try cli.Parser.parse(gpa, comptime AppCli.spec.root, argv, &sink);
    defer root_res.deinit();
    switch (root_res.value) {
        // -h/--help and -V/--version route to the framework's plain-text renderers.
        .help => |h| {
            try cli.Help.renderHelp(comptime AppCli.spec.root, out, if (h.mode == .long) .long else .short);
            try out.flush();
            return;
        },
        .version => {
            try cli.Help.renderVersion(comptime AppCli.spec, out);
            try out.flush();
            return;
        },
        .errors => {
            try printCliErrors(out, err_level, sink.items());
            try out.flush();
            std.process.exit(1);
        },
        .ok => |p| {
            if (try applyParsed(gpa, AppCli.spec.root, p, out, err_level, &st)) |code| {
                try out.flush(); // exit skips defers; flush the styled arg-error line
                std.process.exit(code);
            }
        },
        .subcommand => |sc| {
            // FLAT subcommand model: re-enter `parse` on the matched subcommand.
            // `build`/`run` select the action; the shared option set means
            // `applyParsed` maps their `Parsed` identically to the root's.
            st.verb_seen = true;
            st.command = if (std.mem.eql(u8, sc.name, "run")) .run else .build;
            // Re-parse the FULL argv with just the verb token removed — NOT only
            // `sc.rest` (the post-verb tail). Options BEFORE the verb (e.g. `toy
            // -O1 build x`) were consumed by the root parse and would be lost if we
            // re-parsed the tail alone; the old single-pass loop honored them, so we
            // must too. `sc.rest` is the argv suffix after the verb, so the verb sits
            // at `argv.len - rest.len - 1`; splice it out.
            const verb_idx = argv.len - sc.rest.len - 1;
            var merged: std.ArrayList([]const u8) = .empty;
            defer merged.deinit(gpa);
            try merged.appendSlice(gpa, argv[0..verb_idx]);
            try merged.appendSlice(gpa, argv[verb_idx + 1 ..]);
            inline for (AppCli.spec.root.subcommands, 0..) |sub, j| {
                if (j == sc.index) {
                    var sub_res = try cli.Parser.parse(gpa, sub, merged.items, &sink);
                    defer sub_res.deinit();
                    switch (sub_res.value) {
                        .help => |h| {
                            try cli.Help.renderHelp(sub, out, if (h.mode == .long) .long else .short);
                            try out.flush();
                            return;
                        },
                        .version => {
                            try cli.Help.renderVersion(comptime AppCli.spec, out);
                            try out.flush();
                            return;
                        },
                        .errors => {
                            try printCliErrors(out, err_level, sink.items());
                            try out.flush();
                            std.process.exit(1);
                        },
                        .ok => |p| {
                            if (try applyParsed(gpa, sub, p, out, err_level, &st)) |code| {
                                try out.flush(); // exit skips defers; flush the styled arg-error line
                                std.process.exit(code);
                            }
                        },
                        // A subcommand has no sub-subcommands in this spec.
                        .subcommand => unreachable,
                    }
                }
            }
        },
    }

    // The FINAL colour level: now that `--color` is known, resolve once for all
    // status + diagnostics. Still probes stdout, so a piped run stays `.none`.
    const level = resolveLevel(init, io, st.color_choice);

    // ---- POST-PARSE TAIL (reproduces the old lines 157-201 verbatim) -----------

    if (st.paths.items.len == 0) {
        // Asking to emit (`-o`) with no input is an error, not usage.
        if (st.out_path != null) argErr(out, level, "no input file"); // flushes + exit(1)
        // No input, no `-o`: print short help and exit 0 (replaces usage()).
        try cli.Help.renderHelp(comptime AppCli.spec.root, out, .short);
        try out.flush();
        return;
    }

    // The DEFAULT action is to BUILD a signed executable: a bare `toy <file>` (and any
    // `-o`/`--output`) compiles + links a binary, output defaulting to the build dir.
    // `--emit lex|parse|check|ir` opts into an inspection mode (no binary). Executable
    // emission is locked to aarch64-macos. A `build`/`run` subcommand or `-o`/`--output`
    // forces executable emission; otherwise a bare invocation builds by default unless an
    // inspection `--emit` was requested.
    const build_exe = st.verb_seen or st.out_path != null or !st.emit_explicit;
    const run_after = st.command == .run;
    // Worker threads the build will use (for the "built with N threads" line).
    const threads: usize = if (st.job_count != 0) st.job_count else (std.Thread.getCpuCount() catch 1);
    if (build_exe and !isAarch64Macos(st.target)) argErr(out, level, "code emission only supports aarch64-macos in M1");

    // Build the executable: output → `-o`/`--output`, else the default build dir.
    if (build_exe) {
        std.process.exit(try emitExecutable(gpa, out, level, st.target, st.paths.items, st.out_path, run_after, st.mode, st.codegen_stats, st.opt, st.opt_stats, st.timings, st.jlimit, threads));
    }

    // `--emit ir`: print the target-independent IR for the whole program.
    if (st.emit == .ir) {
        std.process.exit(try emitIr(gpa, out, level, st.target, st.paths.items, st.opt, st.jlimit));
    }

    const results = try Driver.run(gpa, io, st.emit, st.target, st.paths.items);
    defer {
        for (results) |*r| r.deinit(gpa);
        gpa.free(results);
    }

    const failures = try report(out, gpa, level, results, st.emit, st.target, st.dump);
    try out.flush();
    // Signal compilation failure to scripts/CI. (Flush first; exit skips defers.)
    if (failures > 0) std.process.exit(1);
}

/// The optional leading subcommand: `build` (compile to an executable, the default)
/// or `run` (build, then execute the produced binary and report how it ended).
const Command = enum { build, run };

/// The driver state the CLI fills and the post-parse tail reads — one field per old
/// `main()` local so the tail stays a faithful reproduction of the hand-rolled loop.
const State = struct {
    dump: bool = false,
    emit: Driver.Emit = .parse,
    // The DEFAULT action is to BUILD; `--emit` opts into inspection. This tracks
    // whether the user asked for one, so a bare `toy <file>` builds rather than reports.
    emit_explicit: bool = false,
    command: Command = .build,
    verb_seen: bool = false,
    target: []const u8 = "native",
    out_path: ?[]const u8 = null,
    codegen_stats: bool = false,
    mode: Engine.Mode = .normal,
    // opt level / pass selection. Default -O0. Base -O then --opt (resets+sets) then
    // each --no-opt; the flat model loses cross-flag argv order (documented deviation).
    opt: Opt.Config = .O0,
    opt_stats: bool = false,
    timings: bool = false,
    // the `-j N` jobs knob. `.unlimited` => cpu pool; `.limited(0)` (-j1) => serial;
    // `.limited(N)` => cap at N.
    jlimit: Io.Limit = .unlimited,
    // 0 => "not set → use cpu count" for the "built with N threads" line.
    job_count: usize = 0,
    // The resolved `--color` choice, fed once into `resolveLevel` after the parse.
    color_choice: Terminal.ColorChoice = .auto,
    paths: std.ArrayList([]const u8),
};

/// Map a `Parsed(cmd)` onto the driver `State`. GENERIC over the comptime command +
/// `p: anytype` because `Parsed(root)`/`Parsed(build)`/`Parsed(run)` are DISTINCT
/// reified types with the identical field set (the shared option/positional consts);
/// fields are read by name. Returns a non-null `?u8` exit code when the driver
/// detects a bad value the grammar can't express (an unknown/empty opt pass name).
fn applyParsed(gpa: std.mem.Allocator, comptime cmd: cli.Spec.Command, p: anytype, out: *Io.Writer, level: Style.ColorLevel, st: *State) !?u8 {
    _ = cmd;
    st.dump = p.dump;
    st.codegen_stats = p.codegen_stats;
    st.opt_stats = p.opt_stats;
    st.timings = p.timings;
    // --force DOMINATES --verify when both are given: the flat model loses argv
    // order, so apply .verify first then let .force win (no script passes both).
    if (p.verify) st.mode = .verify;
    if (p.force) st.mode = .force;
    st.target = p.target orelse "native";
    st.out_path = p.output;
    st.color_choice = switch (p.color orelse .auto) {
        .auto => .auto,
        .always => .always,
        .never => .never,
    };
    if (p.emit) |e| {
        st.emit_explicit = true;
        st.emit = switch (e) {
            .lex => .lex,
            .parse => .parse,
            .check => .check,
            .ir => .ir,
        };
    }
    if (p.j) |n| {
        // n >= 1 guaranteed by the Range. -j1 => .limited(0) (serial baseline);
        // -jN>=2 => .limited(N) (cap the pool). job_count feeds the threads line.
        const u: usize = @intCast(n);
        st.jlimit = if (u == 1) .limited(0) else .limited(u);
        st.job_count = u;
    }
    // Base opt level from -O (short-only => field `O`); .max=1 rejects -O2 upstream.
    st.opt = if (p.O) |lvl| (if (lvl == 1) Opt.Config.O1 else Opt.Config.O0) else Opt.Config.O0;
    // --opt=<list>: reset to level-off, then turn ON each named pass. An empty
    // element or unknown pass name is a driver arg error (the grammar can't express
    // pass-name validity), reported + exit 1.
    if (p.opt) |list| {
        st.opt = .O0;
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |name| {
            if (name.len == 0) return argErrCode(out, level, "--opt expects a comma-separated pass list (fold,branch,dce,forward)");
            const pass = passByName(name) orelse return argErrCode(out, level, "--opt: unknown pass (expected fold,branch,dce,forward)");
            st.opt.set(pass, true);
        }
    }
    // --no-opt=<pass>: turn each named pass OFF from the current config.
    for (p.no_opt) |name| {
        const pass = passByName(name) orelse return argErrCode(out, level, "--no-opt: unknown pass (expected fold,branch,dce,forward)");
        st.opt.set(pass, false);
    }
    // p.file is a variadic slice borrowing argv/arena; copy the ELEMENTS (slices) —
    // not the bytes — into paths, which shares argv's lifetime, so nothing dangles.
    for (p.file) |f| try st.paths.append(gpa, f);
    return null;
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
/// The whole line is dim (`sty_faint`); at `.none` `styled` writes only the text.
fn printBuildTime(out: *Io.Writer, level: Style.ColorLevel, threads: usize, ns: u64) !void {
    const unit: []const u8 = if (threads == 1) "thread" else "threads";
    var buf: [128]u8 = undefined;
    const line = if (ns < std.time.ns_per_ms)
        std.fmt.bufPrint(&buf, "built with {d} {s} in {d} ns", .{ threads, unit, ns }) catch unreachable
    else if (ns < std.time.ns_per_s)
        std.fmt.bufPrint(&buf, "built with {d} {s} in {d} ms", .{ threads, unit, ns / std.time.ns_per_ms }) catch unreachable
    else if (ns < std.time.ns_per_min)
        std.fmt.bufPrint(&buf, "built with {d} {s} in {d} s", .{ threads, unit, ns / std.time.ns_per_s }) catch unreachable
    else
        std.fmt.bufPrint(&buf, "built with {d} {s} in {d} min {d} s", .{ threads, unit, ns / std.time.ns_per_min, (ns % std.time.ns_per_min) / std.time.ns_per_s }) catch unreachable;
    try sty_faint.styled(out, level, line);
    try out.writeByte('\n');
}

/// `run` subcommand tail: execute the freshly built binary, wait for it, report how it
/// ended, and adopt its exit status. A clean exit returns the child's code; a signal
/// returns 128+signo (the shell convention); anything else → 1. The status line is
/// styled: a clean exit-0 green, any non-zero / signal / spawn failure err-red bold;
/// at `.none` (piped) `styledLine` writes only the text, so the line stays byte-stable.
fn runBinary(gpa: std.mem.Allocator, io: Io, out: *Io.Writer, level: Style.ColorLevel, path: []const u8) !u8 {
    // Spawn by ABSOLUTE path: the codesigned binary must be exec'd by a real path, and a
    // bare name without a `/` would be looked up on PATH rather than in the build dir.
    const abs = Io.Dir.cwd().realPathFileAlloc(io, path, gpa) catch |e| {
        try styledLine(out, level, sty_err, "run: cannot locate {s}: {t}", .{ path, e });
        try out.flush();
        return 1;
    };
    defer gpa.free(abs);

    var child = std.process.spawn(io, .{ .argv = &.{abs} }) catch |e| {
        try styledLine(out, level, sty_err, "run: failed to launch {s}: {t}", .{ basename(path), e });
        try out.flush();
        return 1;
    };
    const term_status = child.wait(io) catch |e| {
        try styledLine(out, level, sty_err, "run: error awaiting {s}: {t}", .{ basename(path), e });
        try out.flush();
        return 1;
    };
    const name = basename(path);
    switch (term_status) {
        .exited => |code| {
            try styledLine(out, level, if (code == 0) sty_ok else sty_err, "{s} exited with code {d}", .{ name, code });
            try out.flush();
            return code;
        },
        .signal => |sig| {
            const s: u32 = @intCast(@intFromEnum(sig));
            try styledLine(out, level, sty_err, "{s} killed by signal {d}", .{ name, s });
            try out.flush();
            return @intCast(128 + (s & 0x7f));
        },
        .stopped => |sig| {
            const s: u32 = @intCast(@intFromEnum(sig));
            try styledLine(out, level, sty_err, "{s} stopped by signal {d}", .{ name, s });
            try out.flush();
            return 1;
        },
        .unknown => |status| {
            try styledLine(out, level, sty_err, "{s} terminated abnormally (status {d})", .{ name, status });
            try out.flush();
            return 1;
        },
    }
}

/// Format `fmt`+`args` into a stack buffer and emit it as one styled line (no
/// trailing newline of its own — the caller supplies none; this writes `\n`). At
/// `.none` `styled` writes only the text, so the line is byte-identical to plain.
fn styledLine(out: *Io.Writer, level: Style.ColorLevel, style: Style.Style, comptime fmt: []const u8, args: anytype) !void {
    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch buf[0..0];
    try style.styled(out, level, s);
    try out.writeByte('\n');
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
/// CRITICAL colour discipline: only the LABEL is ever styled — the `{d} ms` numeric
/// token is emitted RAW so `--color=always | awk '/total/{print $2}'` in profile.sh
/// still parses. The `timings:` header and the `total` row's label are bold; every
/// stage/sub label is plain. At `.none` all `styled` calls write only text, so the
/// coloured output is byte-identical to plain (and profile.sh always pipes => `.none`).
fn printTimings(out: *Io.Writer, level: Style.ColorLevel, total: u64, setup: u64, discover: u64, resolve: u64, typecheck: u64, lower: u64, post: u64, image: u64, disc_sub: ?StageSub, low_sub: ?LowerSub) !void {
    const tot_f: f64 = @floatFromInt(if (total == 0) 1 else total);
    const row = struct {
        // The label field is width-11; pad with RAW spaces to that width (SGR bytes
        // must not count toward the field), styling ONLY the visible label glyphs.
        fn p(w: *Io.Writer, lvl: Style.ColorLevel, name: []const u8, ns: u64, tf: f64) !void {
            const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
            const pct = @as(f64, @floatFromInt(ns)) * 100.0 / tf;
            try w.writeAll("  ");
            try Style.Style.styled(.{}, w, lvl, name); // plain label, gate-safe
            try w.splatByteAll(' ', 11 -| name.len);
            try w.print("{d:>9.3} ms  ({d:>4.1}%)\n", .{ ms, pct });
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
    try sty_head.styled(out, level, "timings:");
    try out.writeByte('\n');
    try row.p(out, level, "setup", setup, tot_f);
    try row.p(out, level, "discover", discover, tot_f);
    if (disc_sub) |s| {
        try row.sub_p(out, "compute", s.compute_ns, discover, "discover");
        try row.sub_p(out, "cache-get", s.get_ns, discover, "discover");
        try row.sub_p(out, "cache-put", s.put_ns, discover, "discover");
    }
    try row.p(out, level, "resolve", resolve, tot_f);
    try row.p(out, level, "typecheck", typecheck, tot_f);
    try row.p(out, level, "lower", lower, tot_f);
    if (low_sub) |s| {
        try row.sub_p(out, "compute", s.compute_ns, lower, "lower");
        try row.sub_p(out, "cache-get", s.get_ns, lower, "lower");
        try row.sub_p(out, "cache-put", s.put_ns, lower, "lower");
        try row.sub_p(out, "link-tail", s.link_ns, lower, "lower");
    }
    // The stats-print window (reads ~0 unless `--codegen-stats`/`--opt-stats` print).
    try row.p(out, level, "post", post, tot_f);
    try row.p(out, level, "image+sign", image, tot_f);
    // RECONCILE TO TOTAL: everything the named rows didn't cover (measurement slop +
    // the tiny tail after the image lap). Printed, never hidden — a saturating
    // subtraction so clock jitter can't underflow it.
    const named = setup +| discover +| resolve +| typecheck +| lower +| post +| image;
    const other = total -| named;
    try row.p(out, level, "other/ovh", other, tot_f);
    // total row: label bold (padded raw first), numeric raw.
    try out.writeAll("  ");
    try sty_head.styled(out, level, "total");
    try out.splatByteAll(' ', 11 - "total".len);
    try out.print("{d:>9.3} ms  (wall-clock)\n", .{@as(f64, @floatFromInt(total)) / 1_000_000.0});
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
    level: Style.ColorLevel,
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
    // import graph from it and compiles the whole program. This is an in-body check
    // that RETURNS 1 (not exit); `argLine` prints the styled error line AND flushes,
    // because the `1` bubbles up through `std.process.exit(try emitExecutable(...))`
    // in main, which skips defers and never flushes `out` on its own.
    if (paths.len != 1) {
        try argLine(out, level, "-o takes exactly one input file (the entry module)");
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
    // `Orchestrator` supplies each stage's compute and writes its result into
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
    if (!try runPipeline(out, gpa, level, engine, &orch)) return 1;

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
    try printBuildTime(out, level, threads, if (elapsed > 0) @intCast(elapsed) else 0);
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
        try printTimings(out, level, total_wall, ns_setup, ns_discover, ns_resolve, ns_typecheck, ns_lower, ns_post, ns_image, disc_sub, low_sub);
    }
    // `run`: execute the freshly built binary and adopt its exit status.
    if (run_after) return runBinary(gpa, io, out, level, resolved_out);
    return 0;
}

/// Drive the shared front-end (discover -> resolve -> typecheck -> codegen) through
/// `StageGraph.interpret` and render any stage's diagnostics. This is the ONE
/// sequencer every program-producing build runs — `-o` and `--emit ir` both call it,
/// differing only in the codegen `tail` and the post-pipeline output.
/// Returns `true` on clean completion (the caller reads the result slots), `false`
/// when a stage produced diagnostics (already rendered; the caller returns exit 1).
/// Each stage's diagnostics route through the pretty `Renderer` (snippet + caret)
/// against the owning module's source, styled at `level` (gate-safe at `.none`).
fn runPipeline(out: *Io.Writer, gpa: std.mem.Allocator, level: Style.ColorLevel, engine: Engine, orch: *const Orchestrator) !bool {
    toyc.StageGraph.interpret(&toyc.StageGraph.pipeline, engine, gpa, orch.*) catch |e| switch (e) {
        // A stage produced diagnostics: render them against the owning module and tell
        // the caller to exit non-zero. `failed_stage` says which stage's diagnostics.
        error.StageDiagnostics => {
            const g = &orch.graph.*.?;
            switch (orch.failed_stage.*.?) {
                .discover => try renderGraphError(gpa, out, level, g, g.err.?),
                .resolve => try renderScopedDiags(gpa, out, level, g, orch.res.*.?.diags),
                .typecheck => try renderScopedDiags(gpa, out, level, g, orch.tc.*.?.diags),
                .codegen => switch (orch.tail) {
                    .lower => switch (orch.lowered.*.?) {
                        .err => |ee| try renderGraphEmit(gpa, out, level, g, ee),
                        .ok => |lp| for (lp.diags) |d| try renderGraphEmit(gpa, out, level, g, .{ .message = d.message, .byte_offset = d.byte_offset }),
                    },
                    .render_ir => switch (orch.ir.*.?) {
                        .err => |ee| try renderGraphEmit(gpa, out, level, g, ee),
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
    level: Style.ColorLevel,
    target: []const u8,
    paths: []const []const u8,
    opt: Opt.Config,
    jlimit: Io.Limit,
) !u8 {
    // `--emit ir` takes the 1 ROOT (entry) file; discover the whole graph. In-body
    // check that RETURNS 1 (not exit); `argLine` prints the styled error line AND
    // flushes, because the `1` bubbles up through `std.process.exit(try emitIr(...))`
    // in main, which skips defers and never flushes `out` on its own.
    if (paths.len != 1) {
        try argLine(out, level, "--emit ir takes exactly one input file (the entry module)");
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
    if (!try runPipeline(out, gpa, level, engine, &orch)) return 1;

    try out.writeAll(ir.?.ok);
    try out.flush();
    return 0;
}

// ---- diagnostics rendering (pretty snippets via the render Renderer) -------
//
// FRAMEWORK GAP (worked around here, not patched): the Renderer needs a Span into a
// source to draw a snippet, so a Graph.Error / EmitError with `byte_offset == null`
// (or a discover error with no module loaded, i.e. no source) cannot snippet. We
// fall back to `renderPlainError` — `path: error: msg (detail)` — preserving the old
// no-location shape exactly, minus escapes at `.none`.

/// Render a resolve/type sink diagnostic against a prepared SourceMap: a zero-width
/// primary `.err` at `d.byte_offset` (via `fromSink`), snippet + caret.
fn renderSinkDiag(out: *Io.Writer, level: Style.ColorLevel, sm: *const Rr.SourceMap, d: toyc.DiagnosticSink.Diagnostic) !void {
    try Rr.Renderer.render(Rr.Diagnostic.fromSink(d), sm, out, renderOpts(level));
}

/// Render a LOCATED error: with an offset, build a one-off SourceMap over `name`/`src`
/// and a primary label at `[off, off)` carrying `message`; a non-empty `detail`
/// becomes a `= note:` footer. With no offset, fall back to `renderPlainError`
/// (detail inline as `(detail)`), preserving the old no-location shape.
fn renderLocated(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, name: []const u8, src: []const u8, byte_offset: ?u32, message: []const u8, detail: []const u8) !void {
    const off = byte_offset orelse return renderPlainError(out, level, name, message, detail);
    var sm = try Rr.SourceMap.init(gpa, name, src);
    defer sm.deinit(gpa);
    var notes_buf: [1]Rr.Diagnostic.Note = undefined;
    const notes: []const Rr.Diagnostic.Note = if (detail.len != 0) blk: {
        notes_buf[0] = .{ .kind = .note, .message = detail };
        break :blk notes_buf[0..1];
    } else &.{};
    const d = Rr.Diagnostic.Diagnostic{
        .severity = .err,
        .message = message,
        .primary = .{ .kind = .primary, .span = .{ .start = off, .end = off }, .message = message },
        .notes = notes,
    };
    try Rr.Renderer.render(d, &sm, out, renderOpts(level));
}

/// The LOCATION-LESS fallback (no offset, or the arg-error path). Emits the old
/// `path: error: msg (detail)` shape with only the `error` word styled at `sty_err`
/// (bright red 9 bold — matching the Renderer's own header word hue). Gate-safe: at
/// `.none` `styled` writes only "error".
fn renderPlainError(out: *Io.Writer, level: Style.ColorLevel, path: []const u8, message: []const u8, detail: []const u8) !void {
    try out.print("{s}: ", .{path});
    try sty_err.styled(out, level, "error");
    try out.print(": {s}", .{message});
    if (detail.len != 0) try out.print(" ({s})", .{detail});
    try out.writeByte('\n');
}

/// Render a batch of sink diagnostics (resolve/type) against their owning modules,
/// REUSING one SourceMap per scope: the diagnostics are sorted by (scope, offset) so
/// same-scope diagnostics are contiguous — we build a map per scope and keep it while
/// the scope holds. Scope `NO_SCOPE` (single-file) picks the entry module.
fn renderScopedDiags(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, g: *const Graph.Graph, diags: []const toyc.DiagnosticSink.Diagnostic) !void {
    var cached_scope: ?u32 = null;
    var sm: Rr.SourceMap = undefined;
    defer if (cached_scope != null) sm.deinit(gpa);
    for (diags) |d| {
        if (cached_scope == null or cached_scope.? != d.scope) {
            if (cached_scope != null) sm.deinit(gpa);
            const m = if (d.scope == toyc.DiagnosticSink.NO_SCOPE) g.entry() else &g.modules[d.scope];
            // Clear `cached_scope` BEFORE the `try init`: `sm` was just deinit'd (set to
            // `undefined`) above, so if `SourceMap.init` hits OOM and propagates, the
            // `defer if (cached_scope != null) sm.deinit(gpa)` must NOT fire on the freed/
            // undefined `sm` (double-free of an undefined pointer). Reassign after init.
            cached_scope = null;
            sm = try Rr.SourceMap.init(gpa, m.path, m.source); // Module.source []u8 coerces to []const u8
            cached_scope = d.scope;
        }
        try renderSinkDiag(out, level, &sm, d);
    }
}

/// Render a graph-discovery structural error against the owning module's source (or
/// the entry path when no module loaded). With a location, snippet + `= note: detail`;
/// otherwise the plain `path: error: msg (detail)` fallback.
fn renderGraphError(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, g: *const Graph.Graph, e: Graph.Error) !void {
    const name = if (e.module) |m| g.modules[m].path else if (g.modules.len > 0) g.entry().path else "<entry>";
    const has_src = e.module != null or g.modules.len > 0;
    if (e.byte_offset != null and has_src) {
        const src: []const u8 = if (e.module) |m| g.modules[m].source else g.entry().source;
        try renderLocated(gpa, out, level, name, src, e.byte_offset, e.message, e.detail);
    } else {
        try renderPlainError(out, level, name, e.message, e.detail);
    }
}

/// Render a code-emission `EmitError` from a graph build against its owning module
/// (or the entry module when `module` is null). EmitError carries no detail.
fn renderGraphEmit(gpa: std.mem.Allocator, out: *Io.Writer, level: Style.ColorLevel, g: *const Graph.Graph, e: Codegen.EmitError) !void {
    const m = if (e.module) |mi| &g.modules[mi] else g.entry();
    if (e.byte_offset) |off| {
        try renderLocated(gpa, out, level, m.path, m.source, off, e.message, "");
    } else {
        try renderPlainError(out, level, m.path, e.message, "");
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

// ---- CLI / arg-error output ------------------------------------------------

/// Emit one `error: <message>` line styled at `sty_err` (the `error` word coloured,
/// gate-safe at `.none`), then FLUSH. The in-body file-count checks that RETURN (not
/// exit) a `1` bubble up through `std.process.exit(try emit*(...))` in `main`, which
/// skips defers and never flushes `out` — so the flush MUST happen here or the error
/// line is silently dropped. `argErr`/`argErrCode` re-flush (idempotent, harmless).
fn argLine(out: *Io.Writer, level: Style.ColorLevel, message: []const u8) !void {
    try sty_err.styled(out, level, "error");
    try out.print(": {s}\n", .{message});
    try out.flush();
}

/// A fatal arg error for the post-parse checks (no-input, aarch64, ...): print a
/// styled `error:` line, flush, and EXIT 1. Never returns.
fn argErr(out: *Io.Writer, level: Style.ColorLevel, message: []const u8) noreturn {
    argLine(out, level, message) catch {};
    out.flush() catch {};
    std.process.exit(1);
}

/// Like `argErr` but for `applyParsed`, which returns `!?u8` rather than exiting
/// directly: print the styled `error:` line and return `1` as the exit code.
fn argErrCode(out: *Io.Writer, level: Style.ColorLevel, message: []const u8) !?u8 {
    try argLine(out, level, message);
    return 1;
}

/// Format + print every accumulated `cli.Sink` error (the parser never prints —
/// this REPLACES the old `argError` for grammar/parse failures). One styled
/// `error: <detail>` line per error; a trailing `Try 'toy --help' ...` hint. Every
/// string in a Sink.Error is BORROWED from argv, so this must run before argv frees.
fn printCliErrors(out: *Io.Writer, level: Style.ColorLevel, errs: []const cli.Sink.Error) !void {
    for (errs) |e| {
        try sty_err.styled(out, level, "error");
        try out.writeAll(": ");
        switch (e.kind) {
            .unknown_flag => try out.print("unknown flag: {s}", .{e.arg}),
            .missing_value => try out.print("{s} requires a value ({s})", .{ e.arg, e.expected }),
            .bad_value => try out.print("invalid value for {s}: '{s}' (expected {s})", .{ e.arg, e.got, e.expected }),
            .missing_required => try out.print("missing required argument: {s}", .{e.arg}),
            .unexpected_arg => try out.print("unexpected argument: {s}", .{e.arg}),
            .conflict => try out.print("{s} conflicts with {s}", .{ e.arg, e.where }),
            .unmet_requirement => try out.print("{s} requires {s}", .{ e.arg, e.where }),
        }
        try out.writeByte('\n');
    }
    try sty_faint.styled(out, level, "Try 'toy --help' for more information.");
    try out.writeByte('\n');
}

/// Print the per-file summary; returns the number of files that failed (so the
/// caller can set a non-zero exit status). The summary TABLE is kept (now styled);
/// per-diagnostic detail lines route through the Renderer via `printFailure`.
fn report(out: *Io.Writer, gpa: std.mem.Allocator, level: Style.ColorLevel, results: []const Driver.FileResult, emit: Driver.Emit, target: []const u8, dump: bool) !usize {
    var stamp_buf: [version.stamp_max]u8 = undefined;
    // Header: the version stamp bold, labels plain.
    try out.writeAll("compiler ");
    try sty_head.styled(out, level, version.stamp(&stamp_buf));
    try out.print("  target {s}  emit {t}\n", .{ target, emit });

    var total_tokens: usize = 0;
    var total_bytes: usize = 0;
    var lex_hits: usize = 0;
    var parse_hits: usize = 0;
    var failures: usize = 0;

    // Table header row: bold (the whole line is a header, no data cells to keep plain).
    {
        var hbuf: [80]u8 = undefined;
        const hdr = std.fmt.bufPrint(&hbuf, "{s: <28} {s: >8} {s: >7} {s: >6}  {s}", .{ "file", "bytes", "tokens", "nodes", "status" }) catch unreachable;
        try sty_head.styled(out, level, hdr);
        try out.writeByte('\n');
    }
    // Separator: dim.
    try sty_faint.styled(out, level, "-" ** 72);
    try out.writeByte('\n');

    for (results) |r| {
        total_bytes += r.source.len;
        if (r.tokens_cached) lex_hits += 1;
        if (r.nodes_cached) parse_hits += 1;

        if (r.err) |err| {
            failures += 1;
            try printFailure(out, gpa, level, r, err);
            continue;
        }

        // Every token stream ends in a synthetic .eof we don't count as "real".
        const real_tokens = r.tokens.len -| 1;
        total_tokens += real_tokens;

        // Data cells (file/bytes/tokens/nodes) stay PLAIN; only the STATUS cell is
        // coloured. The status cell is LAST (no trailing padding), so styling its
        // whole content is width-safe.
        try out.print("{s: <28} {d: >8} {d: >7} ", .{ r.path, r.source.len, real_tokens });
        if (r.parsed) try out.print("{d: >6}", .{r.nodes.len}) else try out.print("{s: >6}", .{"-"});
        try out.writeAll("  ");
        const note = cacheNote(r, emit);
        // `cached` recedes (dim); a fresh phase reads green.
        const nstyle: Style.Style = if (std.mem.eql(u8, note, "cached")) sty_faint else sty_ok;
        try nstyle.styled(out, level, note);
        try out.writeByte('\n');

        if (dump) try dumpArtifact(out, r, emit);
    }

    try sty_faint.styled(out, level, "-" ** 72);
    try out.writeByte('\n');
    // Totals: the `F failure(s)` fragment err-red bold when failures>0, else the
    // whole line dim. Pad-plain (the numbers) then style only the coloured fragment.
    if (failures > 0) {
        try out.print(
            "{d} file(s): {d} bytes, {d} tokens, cache hits lex={d} parse={d}, ",
            .{ results.len, total_bytes, total_tokens, lex_hits, parse_hits },
        );
        var fbuf: [32]u8 = undefined;
        const frag = std.fmt.bufPrint(&fbuf, "{d} failure(s)", .{failures}) catch unreachable;
        try sty_err.styled(out, level, frag);
        try out.writeByte('\n');
    } else {
        var lbuf: [160]u8 = undefined;
        const line = std.fmt.bufPrint(&lbuf, "{d} file(s): {d} bytes, {d} tokens, cache hits lex={d} parse={d}, {d} failure(s)", .{ results.len, total_bytes, total_tokens, lex_hits, parse_hits, failures }) catch unreachable;
        try sty_faint.styled(out, level, line);
        try out.writeByte('\n');
    }
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

/// Emit a failing file's summary ROW (kept, now styled — the `error: ...` cell
/// err-red bold, the path plain), then route each per-file diagnostic through the
/// Renderer over ONE SourceMap of the file (single-file => scope == NO_SCOPE).
fn printFailure(out: *Io.Writer, gpa: std.mem.Allocator, level: Style.ColorLevel, r: Driver.FileResult, err: anyerror) !void {
    if (r.diag) |d| {
        // Parse error: a single diagnostic. The row's `error:` cell is styled; the
        // pretty snippet follows via the Renderer.
        try rowThenErr(out, level, r.path, r.source.len, r.tokens.len -| 1, "-", "parse error", 0);
        var sm = try Rr.SourceMap.init(gpa, r.path, r.source);
        defer sm.deinit(gpa);
        try renderSinkDiag(out, level, &sm, d);
    } else if (err == error.ResolveError) {
        const n = if (r.resolve) |res| res.diags.len else 0;
        var msgbuf: [48]u8 = undefined;
        try rowThenErr(out, level, r.path, r.source.len, r.tokens.len -| 1, "-", std.fmt.bufPrint(&msgbuf, "{d} resolve diagnostic(s)", .{n}) catch "resolve diagnostics", 0);
        if (r.resolve) |res| {
            var sm = try Rr.SourceMap.init(gpa, r.path, r.source);
            defer sm.deinit(gpa);
            for (res.diags) |d| try renderSinkDiag(out, level, &sm, d);
        }
    } else if (err == error.TypeError) {
        const n = if (r.typecheck) |tc| tc.diags.len else 0;
        var msgbuf: [48]u8 = undefined;
        try rowThenErr(out, level, r.path, r.source.len, r.tokens.len -| 1, null, std.fmt.bufPrint(&msgbuf, "{d} type diagnostic(s)", .{n}) catch "type diagnostics", r.nodes.len);
        if (r.typecheck) |tc| {
            var sm = try Rr.SourceMap.init(gpa, r.path, r.source);
            defer sm.deinit(gpa);
            for (tc.diags) |d| try renderSinkDiag(out, level, &sm, d);
        }
    } else {
        var msgbuf: [48]u8 = undefined;
        try rowDashThenErr(out, level, r.path, std.fmt.bufPrint(&msgbuf, "{t}", .{err}) catch "error");
    }
}

/// A failing-file summary row `path bytes tokens nodes  error: <msg>` where the data
/// cells (path/bytes/tokens/nodes) stay PLAIN and only the trailing `error: <msg>`
/// is styled (err-red bold, last on the line so width-safe). `nodes` is either the
/// literal "-" (pass `"-"` as `nodes_str`, `nodes_num` unused) or a count (pass
/// `null` as `nodes_str` and the count as `nodes_num`).
fn rowThenErr(out: *Io.Writer, level: Style.ColorLevel, path: []const u8, bytes: usize, tokens: usize, nodes_str: ?[]const u8, msg: []const u8, nodes_num: usize) !void {
    try out.print("{s: <28} {d: >8} {d: >7} ", .{ path, bytes, tokens });
    if (nodes_str) |s| try out.print("{s: >6}", .{s}) else try out.print("{d: >6}", .{nodes_num});
    try out.writeAll("  ");
    try errWord(out, level, msg);
}

/// Variant of `rowThenErr` for the catch-all error row: every data cell is "-".
fn rowDashThenErr(out: *Io.Writer, level: Style.ColorLevel, path: []const u8, msg: []const u8) !void {
    try out.print("{s: <28} {s: >8} {s: >7} {s: >6}  ", .{ path, "-", "-", "-" });
    try errWord(out, level, msg);
}

/// Emit a styled `error: <msg>` fragment (the `error:` prefix + message together in
/// err-red bold) ending the line. Gate-safe at `.none`.
fn errWord(out: *Io.Writer, level: Style.ColorLevel, msg: []const u8) !void {
    var buf: [80]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "error: {s}", .{msg}) catch "error";
    try sty_err.styled(out, level, s);
    try out.writeByte('\n');
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
