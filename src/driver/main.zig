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
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const CodegenIr = toyc.CodegenIr;
const Opt = toyc.Opt;
const Dag = toyc.QueryDag;
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
    var target: []const u8 = "native";
    var out_path: ?[]const u8 = null;
    var codegen_stats = false;
    var mode: CodegenIr.Mode = .normal;
    // M13 opt level / pass selection. Default -O0 (no opt). Last flag wins,
    // left-to-right; --opt= / --no-opt= toggle individual passes from current.
    var opt: Opt.Config = .O0;
    var opt_stats = false;
    // M16: `--dump-dag` compiles the whole program with the per-build dependency
    // DAG threaded into every query, then prints the deterministic dump to stdout.
    var dump_dag = false;
    // M17: `--query-stats` (with `-o`) threads the dependency DAG through the real
    // executable build, LOADS the prior on-disk DAG, runs the red-green
    // re-validation walk, and reports the recompute set (the cutoff) to stderr,
    // then persists the fresh DAG. The emitted bytes are byte-identical to a plain
    // `-o` build (the content cache, not the walk, drives correctness); the walk is
    // the live in-production proof of the early-cutoff architecture.
    var query_stats = false;
    // M18: the `-j N` jobs knob. `null` => default (a cpu-based pool, i.e. the
    // std runtime's `.unlimited` concurrent_limit); `1` => `.limited(0)` (forces
    // every `fanOut` onto its inline serial fallback — the true serial baseline);
    // `N>=2` => `.limited(N)` (cap the worker pool at N). The CLI's `init.io` is
    // fixed at `.unlimited` and not reconfigurable, so the `-o`/`--dump-dag` paths
    // build their OWN pool from this limit (see emitExecutable / emitDumpDag).
    var jlimit: Io.Limit = .unlimited;
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
        } else if (std.mem.eql(u8, arg, "--dump-dag")) {
            dump_dag = true;
        } else if (std.mem.eql(u8, arg, "--query-stats")) {
            query_stats = true;
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
        } else if (std.mem.eql(u8, arg, "-o")) {
            out_path = args.next() orelse return argError(out, "-o requires an output path");
        } else if (std.mem.eql(u8, arg, "--target")) {
            target = args.next() orelse return argError(out, "--target requires a value");
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try usage(out);
            return;
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

    // Code emission (`-o`) is locked to aarch64-macos in M1.
    if (out_path != null and !isAarch64Macos(target)) {
        try argError(out, "code emission only supports aarch64-macos in M1");
        std.process.exit(1);
    }

    // `--dump-dag`: compile the whole program with the per-build dependency DAG
    // threaded through every query, then print the deterministic dump to stdout.
    // This is the LIVE proof the recording infra runs in production (anti-dead-code).
    if (dump_dag) {
        if (!isAarch64Macos(target)) {
            try argError(out, "--dump-dag only supports aarch64-macos (it runs codegen)");
            std.process.exit(1);
        }
        std.process.exit(try emitDumpDag(gpa, out, target, paths.items, opt, jlimit));
    }

    // `-o`: lower `main` and write a signed, runnable executable.
    if (out_path) |path| {
        std.process.exit(try emitExecutable(gpa, out, target, paths.items, path, mode, codegen_stats, opt, opt_stats, query_stats, jlimit));
    }

    // `--emit ir`: print the target-independent IR for the whole program.
    if (emit == .ir) {
        std.process.exit(try emitIr(gpa, io, out, target, paths.items, opt));
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

/// Run lex→parse→check→codegen→link→sign for a single input and write the signed
/// executable to `out_path` (mode 0o755). Returns the process exit code: 0 on
/// success, 1 on any failure (front-end errors, no `main`, unsupported node).
fn emitExecutable(
    gpa: std.mem.Allocator,
    out: *Io.Writer,
    target: []const u8,
    paths: []const []const u8,
    out_path: []const u8,
    mode: CodegenIr.Mode,
    codegen_stats: bool,
    opt: Opt.Config,
    opt_stats: bool,
    query_stats: bool,
    jlimit: Io.Limit,
) !u8 {
    // M14: `-o` takes the 1 ROOT (entry) file; the driver discovers the transitive
    // import graph from it and compiles the whole program.
    if (paths.len != 1) {
        try argError(out, "-o takes exactly one input file (the entry module)");
        return 1;
    }

    // M18: the `-o` build runs on OUR OWN pool sized by `-j`, not the CLI's fixed
    // `.unlimited` io. `.limited(0)` (from `-j1`) drives every fanOut serial so the
    // single-thread build is the byte-identity baseline; `.limited(N)` caps workers.
    var tail_io: std.Io.Threaded = .init(gpa, .{ .concurrent_limit = jlimit });
    defer tail_io.deinit();
    const io = tail_io.io();

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);

    // Red-green incremental is the DEFAULT drive: a single per-build dependency `Dag`
    // lives on this frame and is threaded through discovery, typecheck, and the per-fn
    // codegen fan-out. The prior on-disk DAG (loaded below) DRIVES codegen reuse — a
    // fn whose body subtree verifies green is served from the prior blob without
    // re-deriving its fingerprint; the fresh DAG is persisted after a successful build
    // for the next one. `--force` bypasses reuse (full rebuild); `--verify` re-lowers
    // everything and audits each green verdict; `--query-stats` additionally PRINTS the
    // recompute set. The recorded-edge path is byte-identical to the old verbatim path.
    var dag_storage: Dag = .init(gpa);
    defer dag_storage.deinit(gpa);
    const dag: ?*Dag = &dag_storage;

    // The program key (entry canonical path + target) names this program's on-disk DAG
    // artifact for BOTH the prior-DAG load (drive) and the post-build persist.
    const entry_canon: ?[]u8 = blk: {
        if (Io.Dir.cwd().realPathFileAlloc(io, paths[0], gpa) catch null) |rp| {
            defer gpa.free(rp);
            break :blk gpa.dupe(u8, rp) catch null;
        }
        break :blk gpa.dupe(u8, paths[0]) catch null;
    };
    defer if (entry_canon) |ec| gpa.free(ec);
    const prog_key: ?u64 = if (entry_canon) |ec| toyc.Cache.programKey(ec, target) else null;

    // --- discover the module graph from the entry file ---
    var graph = try Graph.discoverDag(gpa, io, cache, target, paths[0], dag.?);
    defer graph.deinit(gpa);
    if (graph.err) |ge| {
        try printGraphError(out, &graph, ge);
        try out.flush();
        return 1;
    }

    // --- whole-graph name resolution ---
    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    if (res.diags.len > 0) {
        for (res.diags) |d| try printModuleDiag(out, &graph, d.module, d.byte_offset, d.message);
        try out.flush();
        return 1;
    }

    // --- whole-graph typecheck ---
    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, dag);
    defer tc.deinit(gpa);
    if (tc.diags.len > 0) {
        for (tc.diags) |d| try printModuleDiag(out, &graph, d.module, d.byte_offset, d.message);
        try out.flush();
        return 1;
    }

    // Load the prior on-disk DAG BEFORE codegen so the red-green walk can DRIVE codegen
    // reuse (the verdict decides reuse, not post-hoc observation). A load/parse failure
    // => `null` => full content-fp rebuild (the FALLBACK-SAFETY invariant — never a
    // partial/false-green result).
    var prior_loaded: ?Dag.Loaded = null;
    defer if (prior_loaded) |*pl| pl.deinit(gpa);
    if (prog_key) |pk| {
        if (cache.getDag(gpa, io, pk) catch null) |blob| {
            defer gpa.free(blob);
            prior_loaded = Dag.deserialize(gpa, blob) catch null;
        }
    }
    const prior_ptr: ?*const Dag.Loaded = if (prior_loaded) |*pl| pl else null;

    var lowered = try Driver.lowerGraphProgram(gpa, io, cache, target, &graph, &res, &tc, mode, opt, dag, prior_ptr);
    switch (lowered) {
        .err => |e| {
            try printGraphEmitError(out, &graph, e);
            try out.flush();
            return 1;
        },
        .ok => |*lp| {
            defer lp.deinit(gpa);
            if (lp.diags.len > 0) {
                for (lp.diags) |d| try printGraphEmitError(out, &graph, .{ .message = d.message, .byte_offset = d.byte_offset });
                try out.flush();
                return 1;
            }

            // How many fns were freshly lowered vs. served from the codegen cache.
            if (codegen_stats) {
                try out.print("codegen: compiled={d} cached={d}\n", .{ lp.codegen_compiled, lp.codegen_cached });
                try out.flush();
            }

            // Persist the fresh DAG for the next build's red-green drive. Best-effort:
            // a serialize/write failure only costs the next build a full re-validation,
            // never correctness. Done on EVERY successful build (this is what makes the
            // NEXT build incremental), not just under `--query-stats`.
            if (prog_key) |pk| if (dag) |d| {
                if (d.serialize(gpa)) |blob| {
                    defer gpa.free(blob);
                    cache.putDag(io, pk, blob) catch {};
                } else |_| {}
            };

            // `--query-stats`: print the red-green recompute set (the cutoff) over the
            // just-recorded graph vs the prior snapshot. Purely diagnostic — the emitted
            // bytes are identical with or without it.
            if (query_stats) if (dag) |d| {
                try reportQueryStats(out, gpa, prior_ptr, d, lp.codegen_compiled, lp.codegen_cached);
                try out.flush();
            };

            // M13 dual-metric counters, fixed field order, deterministic. Cached
            // fns contribute 0 to the opt counters; use --force for honest numbers.
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

            const image = try Driver.buildImage(
                io,
                gpa,
                basename(out_path),
                lp.text,
                lp.entry_off,
                lp.cstrings,
                lp.data_relocs,
                lp.uses_write,
            );
            defer gpa.free(image);

            try writeExecutable(io, out_path, image);
            return 0;
        },
    }
}


/// `--emit ir`: run the front-end + the `lower` stage over every function and
/// print the deterministic IR text to stdout. Returns the process exit code.
fn emitIr(
    gpa: std.mem.Allocator,
    io: Io,
    out: *Io.Writer,
    target: []const u8,
    paths: []const []const u8,
    opt: Opt.Config,
) !u8 {
    // M14: `--emit ir` takes the 1 ROOT (entry) file; discover the whole graph.
    if (paths.len != 1) {
        try argError(out, "--emit ir takes exactly one input file (the entry module)");
        return 1;
    }

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);

    var graph = try Graph.discover(gpa, io, cache, target, paths[0]);
    defer graph.deinit(gpa);
    if (graph.err) |ge| {
        try printGraphError(out, &graph, ge);
        try out.flush();
        return 1;
    }

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    if (res.diags.len > 0) {
        for (res.diags) |d| try printModuleDiag(out, &graph, d.module, d.byte_offset, d.message);
        try out.flush();
        return 1;
    }

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, null);
    defer tc.deinit(gpa);
    if (tc.diags.len > 0) {
        for (tc.diags) |d| try printModuleDiag(out, &graph, d.module, d.byte_offset, d.message);
        try out.flush();
        return 1;
    }

    switch (try Driver.renderGraphIr(gpa, &graph, &res, &tc, opt)) {
        .err => |e| {
            try printGraphEmitError(out, &graph, e);
            try out.flush();
            return 1;
        },
        .ok => |text| {
            defer gpa.free(text);
            try out.writeAll(text);
            try out.flush();
            return 0;
        },
    }
}

/// `--dump-dag`: compile the whole program with a per-build dependency `Dag`
/// threaded through EVERY query (discovery lex/parse + per-fn codegen), then print
/// the deterministic dump to stdout. This is the LIVE anti-dead-code proof: the
/// recording infra runs in production, not just in unit tests.
///
/// OWNERSHIP: ONE `Dag` value lives on this stack frame and a `*Dag` is threaded
/// down through `discoverDag` and `lowerGraphProgram`'s graph fan-out. A `*Dag`
/// (never a value) is what reaches each per-fn job, so every worker's recorded
/// edges land in the single shared graph (a value field would vanish per-job).
/// The DAG is purely OBSERVATIONAL — emitted bytes are byte-identical to a normal
/// build (default builds keep dag=null; only this path opts in).
fn emitDumpDag(
    gpa: std.mem.Allocator,
    out: *Io.Writer,
    target: []const u8,
    paths: []const []const u8,
    opt: Opt.Config,
    jlimit: Io.Limit,
) !u8 {
    if (paths.len != 1) {
        try argError(out, "--dump-dag takes exactly one input file (the entry module)");
        return 1;
    }

    // M18: same OWN-pool construction as emitExecutable so `-j` governs the
    // dag-threaded codegen fan-out too (and the dump stays byte-identical at any -j).
    var tail_io: std.Io.Threaded = .init(gpa, .{ .concurrent_limit = jlimit });
    defer tail_io.deinit();
    const io = tail_io.io();

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);

    // The single per-build dependency sink; outlives every fan-out await below.
    var dag: Dag = .init(gpa);
    defer dag.deinit(gpa);

    // --- discover (lex/parse routed through the dag-threaded engine) ---
    var graph = try Graph.discoverDag(gpa, io, cache, target, paths[0], &dag);
    defer graph.deinit(gpa);
    if (graph.err) |ge| {
        try printGraphError(out, &graph, ge);
        try out.flush();
        return 1;
    }

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    if (res.diags.len > 0) {
        for (res.diags) |d| try printModuleDiag(out, &graph, d.module, d.byte_offset, d.message);
        try out.flush();
        return 1;
    }

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, &dag);
    defer tc.deinit(gpa);
    if (tc.diags.len > 0) {
        for (tc.diags) |d| try printModuleDiag(out, &graph, d.module, d.byte_offset, d.message);
        try out.flush();
        return 1;
    }

    // --- whole-program codegen with the dag threaded into every per-fn job ---
    // `--dump-dag` is purely observational: pass `prior=null` so codegen never takes
    // the green reuse path (the dump shows the freshly-recorded structure).
    var lowered = try Driver.lowerGraphProgram(gpa, io, cache, target, &graph, &res, &tc, .normal, opt, &dag, null);
    switch (lowered) {
        .err => |e| {
            try printGraphEmitError(out, &graph, e);
            try out.flush();
            return 1;
        },
        .ok => |*lp| {
            defer lp.deinit(gpa);
            if (lp.diags.len > 0) {
                for (lp.diags) |d| try printGraphEmitError(out, &graph, .{ .message = d.message, .byte_offset = d.byte_offset });
                try out.flush();
                return 1;
            }
        },
    }

    // --- M17 persistence: LOAD the prior DAG (round-trip demo), then WRITE the
    // fresh one. Purely additive this stage: the artifact is recorded but does
    // NOT yet drive invalidation, so the dump on stdout is byte-identical to a
    // build without persistence (the fp-determinism gate diffs stdout). The
    // load/write status is logged to STDERR so it never perturbs stdout.
    const entry_canon = blk: {
        if (Io.Dir.cwd().realPathFileAlloc(io, paths[0], gpa) catch null) |rp| {
            defer gpa.free(rp);
            break :blk try gpa.dupe(u8, rp);
        }
        break :blk try gpa.dupe(u8, paths[0]);
    };
    defer gpa.free(entry_canon);
    const prog_key = toyc.Cache.programKey(entry_canon, target);
    persistDag(gpa, io, cache, prog_key, &dag) catch |e| {
        // Persistence is best-effort scaffolding; a write/load failure must never
        // fail the build (the fallback is always a full re-validation).
        std.debug.print("--dump-dag: persistence skipped ({s})\n", .{@errorName(e)});
    };

    // --- emit the deterministic dump ---
    try dag.dumpDeterministic(gpa, out);
    try out.flush();
    return 0;
}

/// M17 `--query-stats` (with `-o`): the production proof of the early-cutoff
/// architecture. Load the prior on-disk DAG, run the red-green re-validation walk
/// over the just-recorded fresh `dag`, report the recompute set (the CUTOFF) and
/// the codegen cache outcome (the byte-level cutoff that actually drives
/// correctness), then persist the fresh DAG for the next build. Output is to
/// STDOUT (an explicit stats request, like `--codegen-stats`); the emitted
/// executable is byte-identical to a build without this flag.
///
/// Two cutoff numbers are reported because they live at two granularities:
///   * red-green over the recorded query DAG (fine-grained: per signature/body/
///     type_of/layout/codegen node) — the architecture's recompute set;
///   * codegen compiled/cached — the byte-level cutoff the content cache delivers
///     (a fn whose transitive fingerprint is unchanged is served from cache, never
///     re-lowered). These agree in spirit: a localized edit recompiles a strict
///     subset, the rest cut off.
fn reportQueryStats(
    out: *Io.Writer,
    gpa: std.mem.Allocator,
    prior: ?*const Dag.Loaded,
    dag: *Dag,
    codegen_compiled: usize,
    codegen_cached: usize,
) !void {
    const total_nodes = dag.result_fp.count();

    if (prior) |loaded| {
        const roots = try dag.nodeKeys(gpa);
        defer gpa.free(roots);
        var rg = try Dag.RedGreen.init(gpa, loaded, dag);
        defer rg.deinit();
        if (rg.run(roots)) |_| {
            try out.print("query: red-green recompute={d} cutoff={d} of {d} nodes\n", .{ rg.red_count, rg.green_count, rg.red_count + rg.green_count });
        } else |e| switch (e) {
            // A cycle in the prior edges is reported, not fatal — the build already
            // succeeded and this walk is purely the diagnostic stats reporter.
            error.QueryCycle => try out.writeAll("query: red-green walk found a cycle in the prior DAG (skipped)\n"),
            else => return e,
        }
    } else {
        try out.print("query: no prior DAG (cold build) -> {d} nodes recorded\n", .{total_nodes});
    }
    try out.print("query: codegen compiled={d} cached={d}\n", .{ codegen_compiled, codegen_cached });
}

/// Round-trip the per-build DAG through the on-disk artifact: LOAD any prior blob,
/// run the RED-GREEN re-validation walk against it (reporting the recompute set —
/// the cutoff), then WRITE the fresh serialized DAG. Logs to stderr so stdout (the
/// deterministic dump) stays byte-identical. Errors propagate to the best-effort
/// caller.
fn persistDag(gpa: std.mem.Allocator, io: Io, cache: toyc.Cache, prog_key: u64, dag: *Dag) !void {
    const fresh = try dag.serialize(gpa);
    defer gpa.free(fresh);

    if (try cache.getDag(gpa, io, prog_key)) |blob| {
        defer gpa.free(blob);
        if (try Dag.deserialize(gpa, blob)) |loaded_const| {
            var loaded = loaded_const;
            defer loaded.deinit(gpa);
            std.debug.print(
                "--dump-dag: loaded prior DAG (rev={d}, {d} nodes, {d} edges); fresh build has {d} nodes\n",
                .{ loaded.revision, loaded.nodes.len, loaded.edges.len, dag.result_fp.count() },
            );
            try reportRedGreen(gpa, &loaded, dag);
        } else {
            std.debug.print("--dump-dag: prior DAG present but unparseable -> full re-validation\n", .{});
        }
    } else {
        std.debug.print("--dump-dag: no prior DAG (cold) -> full re-validation\n", .{});
    }

    try cache.putDag(io, prog_key, fresh);
}

/// Run the RED-GREEN re-validation walk (`Dag.RedGreen`) over the just-recorded
/// fresh `dag` against the `prior` build snapshot, demanding every recorded node as
/// a root so the whole graph is re-validated. Prints the recompute set — the CUTOFF
/// metric — to stderr (so stdout's deterministic dump is untouched):
///   `--dump-dag: red-green walk: N red (recompute), M green (cutoff) of T nodes`
/// A cycle in the prior edges is reported, not fatal (the build already succeeded;
/// the walk is observational this stage).
fn reportRedGreen(gpa: std.mem.Allocator, prior: *const Dag.Loaded, dag: *Dag) !void {
    const roots = try dag.nodeKeys(gpa);
    defer gpa.free(roots);

    var rg = try Dag.RedGreen.init(gpa, prior, dag);
    defer rg.deinit();
    rg.run(roots) catch |e| switch (e) {
        error.QueryCycle => {
            std.debug.print("--dump-dag: red-green walk: cycle detected in prior edges (skipped)\n", .{});
            return;
        },
        else => return e,
    };
    std.debug.print(
        "--dump-dag: red-green walk: {d} red (recompute), {d} green (cutoff) of {d} nodes\n",
        .{ rg.red_count, rg.green_count, rg.red_count + rg.green_count },
    );
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
fn printModuleDiag(out: *Io.Writer, graph: *const Graph.Graph, module: u32, byte_offset: u32, message: []const u8) !void {
    const m = &graph.modules[module];
    const loc = lineCol(m.source, byte_offset);
    try out.print("{s}:{d}:{d}: error: {s}\n", .{ m.path, loc.line, loc.col, message });
}

/// Render a code-emission `EmitError` from a graph build against its owning module
/// (or the entry module when `module` is null).
fn printGraphEmitError(out: *Io.Writer, graph: *const Graph.Graph, e: Driver.EmitError) !void {
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
fn printEmitError(out: *Io.Writer, r: *const Driver.FileResult, e: Driver.EmitError) !void {
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
        \\toyc {s} — toy compiler (lexer + parser + name resolution + typecheck)
        \\
        \\usage: toyc [options] <file...>
        \\  --emit lex|parse|check|ir  how far to run the pipeline (default: parse)
        \\  -o <path>         emit a signed, runnable executable (aarch64-macos only)
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
        \\  --dump-dag        compile the program and print the per-build query DAG (aarch64-macos)
        \\  --query-stats     with -o: red-green recompute set vs the prior build (the cutoff)
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
