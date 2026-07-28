//! `toyc` CLI entry point: `toy [build|run] [OPTIONS] <file...>`.
//!
//! A bare `toy <file>` builds a signed executable; `--emit lex|parse|check|ir`
//! inspects the pipeline instead (no binary); `build`/`run` (and any `-o`) force a
//! build, `run` also execs the binary. See `Cli.zig` for the schema.
//! - Argument parsing is the typed `toyc.cli` framework (schema `Cli.zig`, parsed
//!   by `cli.Parser` with a collect-all `cli.Sink`). Diagnostics render through
//!   `toyc.term.render.Renderer`; status output colourises via `toyc.term.Style`.
//! - Colour is resolved once per phase from `--color` + a `detectTty(stdout)` probe
//!   + the environment, so a piped/redirected/CI run resolves to `.none` and emits
//!   zero escape bytes — the gate that keeps diff.sh and captured output byte-stable.

const std = @import("std");
const Io = std.Io;
// The CLI entry is the exe module's root; it lives in driver/ (the engine), but
// because a Zig module cannot import files above its root source file, it reaches
// the rest of the compiler through the `toy_compiler` library module (rooted at
// src/root.zig) rather than via relative `../` paths.
const toyc = @import("toy_compiler");
const Driver = toyc.Driver;
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
const AppCli = @import("Cli.zig");
// The extracted status-table + diagnostics-rendering modules (sibling files). The
// shared status palette (`sty_*`) lives in DiagRender — the lower layer both this and
// Report style with — so the dependency runs main -> Report -> DiagRender (no cycle).
const Report = @import("Report.zig");
const DiagRender = @import("DiagRender.zig");
const Check = toyc.Check;
const Decide = toyc.Decide;
const SevCfg = toyc.diagnostics.severity_config;
const sty_err = DiagRender.sty_err;
const sty_ok = DiagRender.sty_ok;
const sty_head = DiagRender.sty_head;
const sty_faint = DiagRender.sty_faint;

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

    // The severity-override rule list. Fn-scope (like `st.paths`) so it outlives
    // every render; `applyParsed` fills it in FIXED (error, warn, ignore) order from
    // the repeatable flags, borrowing argv bytes, then points `st.sev` at its items.
    var sev_rules: std.ArrayList(SevCfg.Rule) = .empty;
    defer sev_rules.deinit(gpa);

    // The collect-all parse-error accumulator. The parser never prints/exits — it
    // fills this and the driver formats + prints the accumulated errors itself.
    var sink: cli.Sink = .{};
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
            try DiagRender.printCliErrors(out, err_level, sink.items());
            try out.flush();
            std.process.exit(1);
        },
        .ok => |p| {
            if (try applyParsed(gpa, AppCli.spec.root, p, out, err_level, &st, &sev_rules)) |code| {
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
                            try DiagRender.printCliErrors(out, err_level, sink.items());
                            try out.flush();
                            std.process.exit(1);
                        },
                        .ok => |p| {
                            // `explain` is a distinct action (print a code's docs, no
                            // build); it has a `code` positional, not the shared option
                            // set, so it bypasses `applyParsed` entirely.
                            if (comptime std.mem.eql(u8, sub.name, "explain")) {
                                try runExplain(out, err_level, p.code, p.list);
                                try out.flush();
                                return;
                            } else if (comptime std.mem.eql(u8, sub.name, "check")) {
                                // `check` is its OWN action with its OWN restricted option
                                // set (no codegen/emit/opt flags), so it has a distinct
                                // `applyCheckParsed` — `applyParsed` reads build fields
                                // (`emit`,`dump`,`O`,...) that `Parsed(check)` no longer has.
                                // It fills `st` (paths, target, color, sev rules) + the
                                // check-only knobs, and exit-2's on a bad severity spec.
                                if (try applyCheckParsed(gpa, p, out, err_level, &st, &sev_rules)) |code| {
                                    try out.flush(); // exit skips defers; flush the styled arg-error line
                                    std.process.exit(code);
                                }
                                st.check_seen = true;
                            } else if (try applyParsed(gpa, sub, p, out, err_level, &st, &sev_rules)) |code| {
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

    // Post-parse tail (reproduces the old hand-rolled tail verbatim).

    if (st.paths.items.len == 0) {
        // `check` with no input is an argument error (exit 2, like a CLI/IO failure —
        // distinct from a compile failure's 1). Asking to emit (`-o`) with no input is
        // likewise an error, not usage.
        if (st.check_seen) {
            try argLine(out, level, "check: no input file");
            std.process.exit(2);
        }
        if (st.out_path != null) argErr(out, level, "no input file"); // flushes + exit(1)
        // No input, no `-o`: print short help and exit 0 (replaces usage()).
        try cli.Help.renderHelp(comptime AppCli.spec.root, out, .short);
        try out.flush();
        return;
    }

    // `toy check`: the diagnostics-focused front-end. Runs lex->parse->resolve->typecheck
    // over every input (NEVER codegen), reports ALL diagnostics (no early-bail), applies
    // the severity config, and returns an exit code: 0 clean, 1 when >=1 error survives
    // the config, 2 on a hard CLI/IO error. `--exit-zero` forces 0 regardless.
    if (st.check_seen) {
        std.process.exit(try runCheck(gpa, out, level, st.target, st.paths.items, st.check_ndjson, st.check_exit_zero, st.sev));
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
    if (build_exe and !Decide.isAarch64Macos(st.target)) argErr(out, level, "code emission only supports aarch64-macos");

    // Build the executable: output → `-o`/`--output`, else the default build dir.
    if (build_exe) {
        std.process.exit(try emitExecutable(gpa, out, level, st.target, st.paths.items, st.out_path, run_after, st.mode, st.codegen_stats, st.opt, st.opt_stats, st.timings, st.jlimit, threads, st.sev));
    }

    // `--emit ir`: print the target-independent IR for the whole program.
    if (st.emit == .ir) {
        std.process.exit(try emitIr(gpa, out, level, st.target, st.paths.items, st.opt, st.jlimit, st.sev));
    }

    const results = try Driver.run(gpa, io, st.emit, st.target, st.paths.items);
    defer {
        for (results) |*r| r.deinit(gpa);
        gpa.free(results);
    }

    const failures = try Report.report(out, gpa, level, results, st.emit, st.target, st.dump, st.sev);
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
    // The `check` subcommand: run the front-end (lex->parse->resolve->typecheck, never
    // codegen), report EVERY diagnostic, and exit on the diagnostic gate. Distinct from
    // `--emit check` (the inspection TABLE), which stays for back-compat.
    check_seen: bool = false,
    // `check --format ndjson`: emit line-delimited JSON instead of pretty snippets.
    check_ndjson: bool = false,
    // `check --exit-zero`: always exit 0 even with errors (editor/LSP streaming).
    check_exit_zero: bool = false,
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
    // Render-time severity overrides (--deny/--warn/--allow) + `--deny-warnings`. Empty
    // by default == identity, so no-flag runs render byte-identical. `rules` BORROWS argv
    // bytes via a fn-scope backing list `applyParsed` fills; resolve reads it LATE at render.
    sev: SevCfg.SeverityConfig = .{},
    paths: std.ArrayList([]const u8),
};

/// Map a `Parsed(cmd)` onto the driver `State`. GENERIC over the comptime command +
/// `p: anytype` because `Parsed(root)`/`Parsed(build)`/`Parsed(run)` are DISTINCT
/// reified types with the identical field set (the shared option/positional consts);
/// fields are read by name. Returns a non-null `?u8` exit code when the driver
/// detects a bad value the grammar can't express (an unknown/empty opt pass name).
fn applyParsed(gpa: std.mem.Allocator, comptime cmd: cli.Spec.Command, p: anytype, out: *Io.Writer, level: Style.ColorLevel, st: *State, sev_rules: *std.ArrayList(SevCfg.Rule)) !?u8 {
    _ = cmd;
    st.dump = p.dump;
    st.codegen_stats = p.codegen_stats;
    st.opt_stats = p.opt_stats;
    st.timings = p.timings;
    // --no-cache DOMINATES --verify when both are given: the flat model loses argv
    // order, so apply .verify first then let .force win (no script passes both).
    if (p.verify) st.mode = .verify;
    if (p.no_cache) st.mode = .force;
    st.target = p.target orelse "native";
    st.out_path = p.output;
    st.color_choice = switch (p.color orelse .auto) {
        .auto => .auto,
        .always => .always,
        .never => .never,
    };
    // `--emit lex|parse|ir` is registered ONLY in a dev build (dev_inspect); in a release
    // build the option — and thus the `Parsed` field — does not exist, so guard the read
    // with `@hasField`. The CLI value list dropped `check` (it is its own subcommand now),
    // so the mapping is lex/parse/ir only; the internal `Driver.Emit.check` is untouched.
    if (@hasField(@TypeOf(p), "emit")) {
        if (p.emit) |e| {
            st.emit_explicit = true;
            st.emit = switch (e) {
                .lex => .lex,
                .parse => .parse,
                .ir => .ir,
            };
        }
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
            const pass = Decide.passByName(name) orelse return argErrCode(out, level, "--opt: unknown pass (expected fold,branch,dce,forward)");
            st.opt.set(pass, true);
        }
    }
    // --no-opt=<pass>: turn each named pass OFF from the current config.
    for (p.no_opt) |name| {
        const pass = Decide.passByName(name) orelse return argErrCode(out, level, "--no-opt: unknown pass (expected fold,branch,dce,forward)");
        st.opt.set(pass, false);
    }
    // Severity overrides; a bad spec is a build arg-error (exit 1). See `buildSevRules`.
    if (try buildSevRules(gpa, p, sev_rules)) |msg| return argErrCode(out, level, msg);
    st.sev = .{ .rules = sev_rules.items, .deny_warnings = p.deny_warnings };
    // p.file is a variadic slice borrowing argv/arena; copy the ELEMENTS (slices) —
    // not the bytes — into paths, which shares argv's lifetime, so nothing dangles.
    for (p.file) |f| try st.paths.append(gpa, f);
    return null;
}

/// Map the `check` subcommand's RESTRICTED `Parsed` onto the driver `State`. Distinct
/// from `applyParsed` because `check_opts` (Cli.zig) carries no codegen/emit/opt flags,
/// so `Parsed(check)` has no `emit`/`dump`/`O`/... fields to read — only the diagnostic-
/// shaping + input knobs. Fills paths, target, color, the severity rules, and the
/// check-only knobs. Returns a non-null `?u8` exit code on a bad severity spec: unlike
/// the build path's arg-errors (exit 1), a check-path usage error exits 2 (consistent
/// with `check`'s missing-input-file exit 2 — a bad `--warn BOGUS` is the same class).
fn applyCheckParsed(gpa: std.mem.Allocator, p: anytype, out: *Io.Writer, level: Style.ColorLevel, st: *State, sev_rules: *std.ArrayList(SevCfg.Rule)) !?u8 {
    st.target = p.target orelse "native";
    st.color_choice = switch (p.color orelse .auto) {
        .auto => .auto,
        .always => .always,
        .never => .never,
    };
    st.check_ndjson = if (p.format) |f| (f == .ndjson) else false;
    st.check_exit_zero = p.exit_zero;
    // Severity overrides; a bad spec is a check USAGE error -> exit 2 (not the build
    // path's 1). See `buildSevRules`.
    if (try buildSevRules(gpa, p, sev_rules)) |msg| return usageCode(out, level, msg);
    st.sev = .{ .rules = sev_rules.items, .deny_warnings = p.deny_warnings };
    for (p.file) |f| try st.paths.append(gpa, f);
    return null;
}

/// Append the `--deny`/`--warn`/`--allow` severity rules onto `sev_rules` in FIXED
/// severity order — all deny, then all warn, then all allow — so that under `resolve`'s
/// last-match-wins semantics `allow` dominates `warn` dominates `deny` for a code named
/// by multiple flags, deterministically and independent of cross-flag argv position (the
/// flat model loses cross-flag order — the same documented deviation as --verify/--no-cache).
/// Each spec must be a known code (R0001) or a band letter (L/P/R/T/W). Returns the
/// offending "<flag>: ..." message on the first invalid spec, or null on success; the
/// caller maps a non-null message to its own exit code (build exit 1, check exit 2).
/// `--deny-warnings` (read by the caller) promotes surviving warnings uniformly via `resolve`.
fn buildSevRules(gpa: std.mem.Allocator, p: anytype, sev_rules: *std.ArrayList(SevCfg.Rule)) !?[]const u8 {
    for (p.deny) |m| {
        if (!Decide.validSpec(m)) return "--deny: unknown code or band (e.g. R0001 or R)";
        try sev_rules.append(gpa, .{ .match = m, .action = .err });
    }
    for (p.warn) |m| {
        if (!Decide.validSpec(m)) return "--warn: unknown code or band (e.g. R0001 or R)";
        try sev_rules.append(gpa, .{ .match = m, .action = .warning });
    }
    for (p.allow) |m| {
        if (!Decide.validSpec(m)) return "--allow: unknown code or band (e.g. R0001 or R)";
        try sev_rules.append(gpa, .{ .match = m, .action = .ignore });
    }
    return null;
}

/// Like `argErrCode` but returns `2` (a USAGE error, not a compile failure): the `check`
/// path treats a bad severity-flag value as a usage error, consistent with a missing
/// input file (exit 2). Prints the styled `error:` line and returns `2`.
fn usageCode(out: *Io.Writer, level: Style.ColorLevel, message: []const u8) !?u8 {
    try argLine(out, level, message);
    return 2;
}

/// The default executable output when no `-o`/`--output` is given:
/// `.toy/<stamp>/build/<entry-stem>`. Creates the build dir on demand; the path is
/// written into `buf` (caller-owned, must outlive the write).
fn defaultOutputPath(io: Io, buf: []u8, entry: []const u8) ![]const u8 {
    var stamp_buf: [version.stamp_max]u8 = undefined;
    var dir_buf: [Driver.cache_root.len + 1 + version.stamp_max + "/build".len]u8 = undefined;
    const build_dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/build", .{ Driver.cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    try Io.Dir.cwd().createDirPath(io, build_dir);
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ build_dir, Decide.stemOf(Decide.basename(entry)) }) catch unreachable;
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
        try styledLine(out, level, sty_err, "run: failed to launch {s}: {t}", .{ Decide.basename(path), e });
        try out.flush();
        return 1;
    };
    const term_status = child.wait(io) catch |e| {
        try styledLine(out, level, sty_err, "run: error awaiting {s}: {t}", .{ Decide.basename(path), e });
        try out.flush();
        return 1;
    };
    const name = Decide.basename(path);
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
    sev: SevCfg.SeverityConfig,
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
        .sev = sev,
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
    if (!try runPipeline(out, gpa, level, engine, &orch, sev)) return 1;

    // The interpreter ran every stage clean: `lowered` is `.ok` with no EFFECTIVE error.
    const lp = &lowered.?.ok;

    // Render any surviving warnings + a summary, then build the binary. No warning
    // survives to here under `deny_warnings` (it would have promoted to an error and
    // bailed a stage), so the surviving warnings render as warnings and the build still
    // exits 0. A truly clean build renders nothing (`renderDiagSummary` no-ops at 0/0),
    // so its stdout stays byte-identical.
    {
        const g = &graph.?;
        var counts: Check.Counts = .{};
        const stages = [_][]const toyc.DiagnosticSink.Diagnostic{
            res.?.diags,
            if (tc) |t| t.diags else &.{},
            lp.diags,
        };
        for (stages) |s| {
            try DiagRender.renderScopedDiags(gpa, out, level, g, s, sev);
            const c = Check.tallyGraph(s, sev);
            counts.errors += c.errors;
            counts.warnings += c.warnings;
        }
        try DiagRender.renderDiagSummary(out, level, counts.errors, counts.warnings);
    }

    if (codegen_stats) {
        try out.print("codegen: compiled={d} cached={d}\n", .{ lp.codegen_compiled, lp.codegen_cached });
        try out.flush();
    }

    // dual-metric counters, fixed field order, deterministic. Cached fns
    // contribute 0 to the opt counters; use --no-cache for honest numbers.
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
        Decide.basename(resolved_out),
        lp.text,
        lp.entry_off,
        lp.cstrings,
        lp.data_relocs,
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
fn runPipeline(out: *Io.Writer, gpa: std.mem.Allocator, level: Style.ColorLevel, engine: Engine, orch: *const Orchestrator, cfg: SevCfg.SeverityConfig) !bool {
    toyc.StageGraph.interpret(&toyc.StageGraph.pipeline, engine, gpa, orch.*) catch |e| switch (e) {
        // A stage produced diagnostics: render them against the owning module and tell
        // the caller to exit non-zero. `failed_stage` says which stage's diagnostics.
        error.StageDiagnostics => {
            const g = &orch.graph.*.?;
            switch (orch.failed_stage.*.?) {
                .discover => try DiagRender.renderGraphError(gpa, out, level, g, g.err.?, cfg),
                .resolve => try DiagRender.renderScopedDiags(gpa, out, level, g, orch.res.*.?.diags, cfg),
                .typecheck => try DiagRender.renderScopedDiags(gpa, out, level, g, orch.tc.*.?.diags, cfg),
                .codegen => switch (orch.tail) {
                    .lower => switch (orch.lowered.*.?) {
                        .err => |ee| try DiagRender.renderGraphEmit(gpa, out, level, g, ee),
                        .ok => |lp| for (lp.diags) |d| try DiagRender.renderGraphEmit(gpa, out, level, g, .{
                            .message = d.message,
                            .byte_offset = d.byte_offset,
                            // Carry the diagnostic's owning module so it renders against the
                            // right source once multi-module lowering emits per-module diags;
                            // NO_SCOPE (single-file) stays null -> entry module.
                            .module = if (d.scope == toyc.DiagnosticSink.NO_SCOPE) null else d.scope,
                        }),
                    },
                    .render_ir => switch (orch.ir.*.?) {
                        .err => |ee| try DiagRender.renderGraphEmit(gpa, out, level, g, ee),
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
    sev: SevCfg.SeverityConfig,
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
        .sev = sev,
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
    if (!try runPipeline(out, gpa, level, engine, &orch, sev)) return 1;

    try out.writeAll(ir.?.ok);
    try out.flush();
    return 0;
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

/// `toy check <entry>`: the diagnostics-focused front-end. Runs the SAME graph
/// front-end `build` drives — `Graph.discover` (FOLLOWING imports) -> `resolveGraph`
/// -> `checkGraph` — then STOPS before codegen/lower/link. Reports EVERY diagnostic
/// from the discovered graph (whole-graph resolve diagnostics; if resolve is clean,
/// whole-graph typecheck diagnostics) against each diagnostic's OWN module source,
/// applies the severity config, and emits either pretty human snippets or a stable
/// NDJSON stream. Returns the process exit code: 0 when nothing survives the config as
/// an error, 1 when >=1 error remains (incl. a structural graph error like a real
/// missing import — matching `build`), 2 on a hard CLI/IO failure (a missing/unreadable
/// ENTRY file). `exit_zero` forces 0. `ndjson` picks the wire form.
///
/// This is the bug fix: the old path fanned out over `paths` as INDEPENDENT single
/// files (`Driver.run(.check)`), discovering no imports, so a valid multi-module
/// program over-reported spurious R0003/R0001 diagnostics that `build` did not. Routing
/// through the graph makes `check` agree with `build`.
fn runCheck(
    gpa: std.mem.Allocator,
    out: *Io.Writer,
    level: Style.ColorLevel,
    target: []const u8,
    paths: []const []const u8,
    ndjson: bool,
    exit_zero: bool,
    sev: SevCfg.SeverityConfig,
) !u8 {
    // `check` takes the 1 ROOT (entry) file, exactly like a `-o`/`--emit ir` build:
    // discover its transitive import graph. More than one bare input is a usage error
    // (exit 2), consistent with `emitExecutable`/`emitIr` rejecting multi-entry.
    if (paths.len != 1) {
        try argLine(out, level, "check takes exactly one input file (the entry module)");
        return 2;
    }
    const entry = paths[0];

    // Own pool for the whole-graph typecheck Pass-C body fan-out (`checkGraph` needs an
    // `io`); a plain `check` uses the host cpu pool. Diagnostics are deterministic
    // regardless of `-jN` (sorted by (scope, offset, code, message)).
    var tail_io: std.Io.Threaded = .init(gpa, .{});
    defer tail_io.deinit();
    const cio = tail_io.io();

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = Driver.openCache(cio, &dir_buf) catch |e| {
        try argLine(out, level, @errorName(e));
        return 2;
    };

    // DISCOVER the whole module graph (following imports), reusing `build`'s discoverer.
    // A true I/O/OOM failure propagates as a Zig error (exit 2); a STRUCTURAL problem
    // (missing import, escape, cycle, tainted parse) is recorded in `graph.err`.
    var graph = Graph.discover(gpa, cio, cache, target, entry, null) catch |e| {
        try argLine(out, level, @errorName(e));
        return 2;
    };
    defer graph.deinit(gpa);

    if (graph.err) |ge| {
        // A tainted parse in the ENTRY renders the entry's FULL coded parse-diagnostic
        // list (`error[P0002]:` + a caret) — `discover` retained it on `ge.parse_diags`,
        // so `check` reports every diagnostic, matching `build`'s per-file output. A parse
        // error in an IMPORTED module instead falls through to `renderGraphError` (matching
        // `build` — discovery never enters an import past a broken one).
        const parse_in_entry = ge.kind == .parse and (ge.module == null or ge.module == graph.entry_index);
        if (parse_in_entry) return reportEntryFile(gpa, out, level, &graph, ndjson, ge.parse_diags, &.{}, &.{}, sev, exit_zero);
        // A missing/unreadable ENTRY file (kind=.missing, module=null) is a hard CLI/IO
        // failure -> exit 2, matching `check`'s single-file missing-input contract. Every
        // OTHER structural error (a real missing/mis-cased IMPORT, a path escape, an import
        // cycle, a tainted parse in a discovered import) is a compile-level error `build`
        // reports too -> render it against its owning module and exit 1.
        try DiagRender.renderGraphError(gpa, out, level, &graph, ge, sev);
        try out.flush();
        if (exit_zero) return 0;
        return if (ge.kind == .missing and ge.module == null) 2 else 1;
    }

    // RESOLVE the whole graph. Report ALL resolve diagnostics; only when resolve is clean
    // do we proceed to typecheck (you cannot type unresolved code, but each stage reports
    // ALL of its own diagnostics — `resolveGraph`/`checkGraph` do so internally).
    var res = ResolveGraph.resolveGraph(gpa, &graph) catch |e| {
        try argLine(out, level, @errorName(e));
        return 2;
    };
    defer res.deinit(gpa);

    // Typecheck only when resolve is error-free (you cannot type unresolved code). BOTH
    // stages' diagnostics are reported — resolve carries the unused-variable warnings
    // (W0001), which the old code dropped by collapsing to `tc.?.diags` on a clean
    // resolve. `tc` is kept alive (fn-scope optional) so its borrowed messages outlive
    // every render below.
    var tc: ?TypecheckGraph.GraphResult = null;
    defer if (tc) |*t| t.deinit(gpa);
    if (!Decide.resolveHasError(res.diags)) {
        tc = TypecheckGraph.checkGraph(gpa, &graph, &res, cio, 0) catch |e| {
            try argLine(out, level, @errorName(e));
            return 2;
        };
    }
    const type_diags: []const toyc.DiagnosticSink.Diagnostic = if (tc) |t| t.diags else &.{};

    // SINGLE-MODULE (no imports discovered): render against the entry's ON-DISK `file`,
    // not its module `path` (a bare stem like `fix_bad` vs `/tmp/fix_bad.toy`), so the
    // human `-->` header matches the per-file build output. All entry parse errors already
    // routed above, so a single-module graph carries resolve (incl. warnings) + type
    // diagnostics. Both wire forms share `reportEntryFile` with the tainted-entry path.
    if (graph.modules.len == 1)
        return reportEntryFile(gpa, out, level, &graph, ndjson, &.{}, res.diags, type_diags, sev, exit_zero);

    // Multi-module: combine resolve + type diagnostics into ONE slice (single cap + single
    // tally). Each stage sorts its own half, but the resolve-then-type concatenation is
    // not globally ordered — a later-line resolve warning would precede an earlier-line
    // type error. One stable re-sort on the canonical key restores per-scope contiguity
    // (renderScopedDiags/emitNdjsonGraph require it) and source-monotonic order.
    const combined = try gpa.alloc(toyc.DiagnosticSink.Diagnostic, res.diags.len + type_diags.len);
    defer gpa.free(combined);
    @memcpy(combined[0..res.diags.len], res.diags);
    @memcpy(combined[res.diags.len..], type_diags);
    toyc.DiagnosticSink.sortSlice(combined);
    if (ndjson) {
        // NDJSON over the graph's flat slice: each diagnostic's `file` is its OWNING
        // module path (not always the entry), so an imported module's error is attributed
        // correctly. Schema byte-identical to the single-file form.
        const counts = try Check.emitNdjsonGraph(out, gpa, &graph, combined, sev);
        try out.flush();
        return Decide.checkExit(counts, exit_zero);
    }
    // Human form: pretty per-module snippets (reusing the multi-module renderer, which
    // builds ONE SourceMap per scope) + a program-wide `N error(s), M warning(s)` summary.
    try DiagRender.renderScopedDiags(gpa, out, level, &graph, combined, sev);
    const counts = Check.tallyGraph(combined, sev);
    try DiagRender.renderDiagSummary(out, level, counts.errors, counts.warnings);
    try out.flush();
    return Decide.checkExit(counts, exit_zero);
}

/// Report the diagnostics belonging to the ENTRY file — a tainted entry parse, or a
/// single-module graph with no imports — against its on-disk `file`/`source`, then a
/// program-wide summary and the mapped exit code. The two entry-only cases share this one
/// render + summary + exit path so their output and exit gate cannot drift. The three
/// stage batches are merged into one slice and re-sorted so diagnostics print in source
/// order across stages (a resolve warning and a type error interleave by line).
fn reportEntryFile(
    gpa: std.mem.Allocator,
    out: *Io.Writer,
    level: Style.ColorLevel,
    g: *const Graph.Graph,
    ndjson: bool,
    parse_diags: []const toyc.DiagnosticSink.Diagnostic,
    resolve_diags: []const toyc.DiagnosticSink.Diagnostic,
    type_diags: []const toyc.DiagnosticSink.Diagnostic,
    sev: SevCfg.SeverityConfig,
    exit_zero: bool,
) !u8 {
    // Concatenate the (at most three) stage batches into ONE slice and sort on the
    // canonical key so diagnostics print in source order regardless of which stage
    // produced them: a resolve warning and a typecheck error on adjacent lines must
    // interleave by line, not by stage. Heap-allocated, freed on return. Sorting only
    // permutes the POD structs — message buffers stay borrowed from parse/resolve/type
    // diags, which outlive this call, so no ownership changes.
    const combined = try gpa.alloc(toyc.DiagnosticSink.Diagnostic, parse_diags.len + resolve_diags.len + type_diags.len);
    defer gpa.free(combined);
    @memcpy(combined[0..parse_diags.len], parse_diags);
    @memcpy(combined[parse_diags.len..][0..resolve_diags.len], resolve_diags);
    @memcpy(combined[parse_diags.len + resolve_diags.len ..], type_diags);
    toyc.DiagnosticSink.sortSlice(combined);

    if (ndjson) {
        // The sorted union attributed to the entry (scope -> entry file); schema matches
        // the multi-module form (`emitNdjsonGraph` is shared).
        const counts = try Check.emitNdjsonGraph(out, gpa, g, combined, sev);
        try out.flush();
        return Decide.checkExit(counts, exit_zero);
    }
    const e = g.entry();
    try DiagRender.renderFileDiags(gpa, out, level, e.file, e.source, combined, sev);
    const counts = Check.tallyGraph(combined, sev);
    try DiagRender.renderDiagSummary(out, level, counts.errors, counts.warnings);
    try out.flush();
    return Decide.checkExit(counts, exit_zero);
}

/// `toy explain <CODE>`: print the code's embedded documentation. A known code prints
/// its doc and returns (caller flushes + exits 0); an unknown code prints a styled
/// arg error and EXITS 2 (an argument error, distinct from a compile failure's 1).
/// `--list` (or the bare word `list`) enumerates every code instead; a bare `explain`
/// with neither a code nor `--list` is a usage error (exit 2).
fn runExplain(out: *Io.Writer, level: Style.ColorLevel, code: ?[]const u8, list: bool) !void {
    // `--list`, or `toy explain list`, prints the whole registry.
    if (list or (code != null and std.mem.eql(u8, code.?, "list"))) {
        try listCodes(out, level);
        return;
    }
    const c = code orelse {
        argLine(out, level, "specify a diagnostic code (e.g. R0001) or --list") catch {};
        out.flush() catch {};
        std.process.exit(2);
    };
    if (toyc.diagnostics.explain.docForStr(c)) |doc| {
        try out.writeAll(doc);
        if (doc.len == 0 or doc[doc.len - 1] != '\n') try out.writeByte('\n');
        return;
    }
    var buf: [128]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "unknown diagnostic code: {s}", .{c}) catch "unknown diagnostic code";
    argLine(out, level, msg) catch {};
    out.flush() catch {};
    std.process.exit(2);
}

/// The band a code prefix belongs to (for `explain --list` grouping headers).
fn bandTitle(prefix: u8) []const u8 {
    return switch (prefix) {
        'L' => "Lexer (L)",
        'P' => "Parser (P)",
        'R' => "Name resolution (R)",
        'T' => "Type checking (T)",
        'W' => "Warnings (W)",
        else => "Other",
    };
}

/// `toy explain --list`: print every registered diagnostic code with its default
/// severity and kebab title, grouped by band. Driven straight from `codes.table`, so it
/// can never drift from the registry (a new code shows up here automatically). Sorted by
/// the table's append order, which is band-grouped + numerically contiguous.
fn listCodes(out: *Io.Writer, level: Style.ColorLevel) !void {
    const reg = toyc.diagnostics.codes;
    var cur_band: u8 = 0;
    for (reg.table) |e| {
        const prefix = e.str[0];
        if (prefix != cur_band) {
            cur_band = prefix;
            if (e.code != reg.table[0].code) try out.writeByte('\n');
            try DiagRender.sty_head.styled(out, level, bandTitle(prefix));
            try out.writeByte('\n');
        }
        const sev = switch (e.default_severity) {
            .err => "error",
            .warning => "warning",
            .note => "note",
            .help => "help",
        };
        try out.print("  {s: <6} {s: <8} {s}\n", .{ e.str, sev, e.slug });
    }
}
