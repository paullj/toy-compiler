//! Compilation driver: runs the pipeline (lex → parse → check) over many files
//! in parallel.
//!
//! One job per file is dispatched onto the `Io` runtime's worker threads, so a
//! build with N files uses up to N cores. Each job runs the pipeline up to
//! `emit` for its file — read, then a cache-or-run step per phase — and writes
//! only into its own result slot, so no locking is needed. The on-disk phases
//! (lex, parse) memoize through the content cache, keyed by `Cache.Phase`. The
//! `check` level (name resolution + typecheck) runs in-memory and is not cached
//! yet, so it always re-runs on top of the cached parse artifact.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Token = @import("../ast/Token.zig").Token;
const Parser = @import("../parse.zig");
const Ast = @import("../ast/Ast.zig");
const Cache = @import("../query/Cache.zig");
const Engine = @import("../query/Engine.zig");
const Dag = @import("../query/Dag.zig");
const Resolve = @import("../resolve.zig");
const Typecheck = @import("../types.zig");
const Graph = @import("Graph.zig");
const ResolveGraph = @import("../resolve_graph.zig");
const TypecheckGraph = @import("../types_graph.zig");
const CodegenIr = @import("../codegen/CodegenIr.zig");
const Ir = @import("../ir/Ir.zig");
const Opt = @import("../opt/Opt.zig");
const lower = @import("../lower.zig");
const Fingerprint = @import("../query/Fingerprint.zig");
const Link = @import("../link/Link.zig");
const link = @import("../link/emit.zig");
const version = @import("../version.zig");

pub const cache_root = ".toy";

/// The buffer size the cache `dir_buf` needs: `.toy/<stamp>/cache`.
pub const cache_dir_buf_len = cache_root.len + 1 + version.stamp_max + "/cache".len;

/// Open the per-compiler cache directory (`.toy/<stamp>/cache/`). `dir_buf` must
/// outlive the returned `Cache` (it borrows the formatted path). Used by the CLI
/// to share one cache across `Driver.run` (front-end) and `lowerGraphProgram` (codegen).
pub fn openCache(io: Io, dir_buf: []u8) !Cache {
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    return Cache.init(io, dir);
}

/// Like `openCache` but attaches a borrowed packed-object-store sidecar so `get`/`put`
/// route through ONE pack file + an in-memory index (bulk read at start, bulk write at
/// end) instead of one file per entry. The caller owns `pack` and must `load` it before
/// the build and `flush` it after.
pub fn openCachePack(io: Io, dir_buf: []u8, pack: *Cache.Pack) !Cache {
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    return Cache.initPack(io, dir, pack);
}

/// How far to run the pipeline. Distinct from `Cache.Phase`: `lex`/`parse` are
/// cached on disk, but `check` (name resolution, and later typecheck) is an
/// in-memory level only — it has no cache phase of its own yet.
pub const Emit = enum {
    lex,
    parse,
    check,

    /// Dump the target-independent IR for the whole program (`--emit ir`). Runs
    /// the full front-end (resolve + typecheck) like `check`, then lowers every
    /// function via the `lower` stage and renders the IR text.
    ir,

    /// The deepest cached phase this emit level needs. `check`/`ir` build on the
    /// parse artifact, so they cache through `parse`.
    fn cachePhase(emit: Emit) Cache.Phase {
        return switch (emit) {
            .lex => .lex,
            .parse, .check, .ir => .parse,
        };
    }

    /// `ir` needs the same in-memory front-end as `check`.
    fn runsCheck(emit: Emit) bool {
        return emit == .check or emit == .ir;
    }
};

/// Result of running the pipeline on one file. Owns its `source`, `tokens`, and
/// `nodes`.
pub const FileResult = struct {
    path: []const u8,
    source: []u8 = &.{},
    tokens: []Token = &.{},
    nodes: []Ast.Node = &.{},
    /// Variable-arity child runs paired with `nodes` (see `Ast`). Owned.
    extra: []u32 = &.{},
    /// `pub` export bitset paired with `nodes` (M14, see `Ast.Tree.pub_bits`).
    /// Owned; empty for a program with no exports.
    pub_bits: []const u32 = &.{},
    /// Whether each phase's result was loaded from cache rather than recomputed.
    tokens_cached: bool = false,
    nodes_cached: bool = false,
    /// True once the parse phase ran (or hit cache) for this file.
    parsed: bool = false,
    /// True once name resolution ran for this file (emit == .check).
    checked: bool = false,
    err: ?anyerror = null,
    /// On a parse error, where and what.
    diag: ?Parser.Diagnostic = null,
    /// Whole-graph name-resolution result (emit == .check). A lone file IS the
    /// trivial one-module graph, so the graph result is stored whole. SOLE owner of
    /// the fn-name strings (`fns[i].name`); the typecheck `sigs[i].name` borrow them,
    /// lifetime-tied to this sibling field. Owned; freed in deinit.
    resolve: ?ResolveGraph.GraphResult = null,
    /// Whole-graph type-check result (emit == .check, run only if resolve was
    /// clean). Stored whole; its `sigs[i].name` BORROW `resolve.?.fns[i].name`
    /// (same lifetime — sibling fields, torn down together). Owned; freed in deinit.
    typecheck: ?Typecheck.GraphResult = null,

    /// The entry module's node-parallel resolution array. A `FileResult` is a
    /// graph-of-one, so module 0 IS the file (asserted, to trip loudly if a
    /// multi-module result is ever stored here).
    pub fn resolutions(r: *const FileResult) []const Resolve.Resolution {
        std.debug.assert(r.resolve.?.resolutions.len == 1);
        return r.resolve.?.resolutions[0];
    }

    /// The entry module's node-parallel inferred-type array (graph-of-one).
    pub fn nodeTypes(r: *const FileResult) []const Typecheck.Type {
        std.debug.assert(r.typecheck.?.node_types.len == 1);
        return r.typecheck.?.node_types[0];
    }

    pub fn deinit(r: *FileResult, gpa: std.mem.Allocator) void {
        gpa.free(r.source);
        gpa.free(r.tokens);
        gpa.free(r.nodes);
        gpa.free(r.extra);
        if (r.pub_bits.len != 0) gpa.free(@constCast(r.pub_bits));
        if (r.resolve) |*res| res.deinit(gpa);
        if (r.typecheck) |*tc| tc.deinit(gpa);
        r.* = undefined;
    }
};

/// Run the pipeline (up to `emit`) over every path for `target`. Returns one
/// `FileResult` per input (same order); per-file failures are reported in
/// `FileResult.err`, not as a hard error. Caller owns the slice and must
/// `deinit` each result.
pub fn run(gpa: std.mem.Allocator, io: Io, emit: Emit, target: []const u8, paths: []const []const u8) ![]FileResult {
    // Cache is namespaced by compiler identity: .toy/<stamp>/cache/. The buffer
    // lives on this stack frame, which outlives every job (we await below).
    var dir_buf: [cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    const results = try gpa.alloc(FileResult, paths.len);
    for (results, paths) |*r, path| r.* = .{ .path = path };

    const Ctx = struct {
        gpa: std.mem.Allocator,
        io: Io,
        cache: Cache,
        emit: Emit,
        target: []const u8,
        results: []FileResult,
        pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(job)) {
            return .{ c.gpa, c.io, c.cache, c.emit, c.target, &c.results[i], i };
        }
    };
    Engine.fanOut(io, results.len, job, Ctx{
        .gpa = gpa,
        .io = io,
        .cache = cache,
        .emit = emit,
        .target = target,
        .results = results,
    });

    return results;
}

fn job(gpa: std.mem.Allocator, io: Io, cache: Cache, emit: Emit, target: []const u8, result: *FileResult, index: usize) void {
    pipeline(gpa, io, cache, emit, target, result, index) catch |err| {
        result.err = err;
    };
}

fn pipeline(gpa: std.mem.Allocator, io: Io, cache: Cache, emit: Emit, target: []const u8, result: *FileResult, index: usize) !void {
    const cache_phase = emit.cachePhase();
    result.source = try Io.Dir.cwd().readFileAlloc(io, result.path, gpa, .unlimited);

    // --- lex --- (routed through the query engine, the M15 seam)
    const engine = Engine.init(cache, .normal);
    const lexed = try engine.lex(gpa, io, target, result.source, index, false);
    result.tokens = lexed.value;
    result.tokens_cached = lexed.cached;

    if (@intFromEnum(cache_phase) < @intFromEnum(Cache.Phase.parse)) return;

    // --- parse ---
    // The parse output is a Tree (nodes + extra). We pack both into one flat
    // []u8 blob and store/load it through the existing generic byte cache;
    // Ast.unpack validates a hit and treats a corrupt/foreign blob as a miss.
    result.parsed = true;
    const parsed = try engine.parse(gpa, io, target, result.source, result.tokens, index, false);
    const tree = parsed.tree orelse {
        result.diag = parsed.diag;
        result.err = error.ParseError;
        return;
    };
    result.nodes_cached = parsed.cached;
    result.nodes = tree.nodes;
    result.extra = tree.extra;
    result.pub_bits = tree.pub_bits;

    if (!emit.runsCheck()) return;

    // --- check: name resolution then typecheck (in-memory only; uncached). ---
    // A lone source file is compiled as the trivial one-module graph: the graph
    // front-end (`resolveGraph`/`checkGraph`) is the ONLY resolve/typecheck path,
    // and its whole-graph results (one module → program-wide tables == local
    // tables) are stored WHOLE on the `FileResult`. The two results are sibling
    // fields with one shared lifetime, so the resolve `fns` table (owner of every
    // fn name) provably outlives the typecheck `sigs[].name` that borrow it.
    result.checked = true;
    var graph = try Graph.single(gpa, "main", result.path, result.source, result.tokens, result.nodes, result.extra, result.pub_bits);
    defer graph.deinitSingle(gpa);

    var gr = try ResolveGraph.resolveGraph(gpa, &graph);
    // Move `gr` into the FileResult before any later fallible step so its teardown
    // is owned structurally (FileResult.deinit), never by a leak-prone errdefer.
    // checkGraph below borrows `result.resolve.?` (resolutions/fns), which is fine:
    // the move only transfers ownership, not the data's address.
    result.resolve = gr;
    gr = undefined;

    if (result.resolve.?.diags.len > 0) {
        // A name error would poison every dependent type; don't typecheck. The
        // diag-bearing resolve result is already on the FileResult; stop here.
        result.err = error.ResolveError;
        return;
    }

    // dag=null + io=null => serial Pass-C: the per-file `run` fan-out is the only
    // active parallelism for `--emit check`, so there is no nested-pool blowup. A
    // 1-fn graph never spawns regardless; serial == parallel byte-for-byte.
    result.typecheck = try TypecheckGraph.checkGraph(gpa, &graph, &result.resolve.?, null, null, 0);
    if (result.typecheck.?.diags.len > 0) result.err = error.TypeError;
}

// ---- code emission (the `-o` path) -----------------------------------------

/// A user-facing failure while emitting code: a message plus the source byte
/// offset to render as `line:col` (or `null` for whole-file errors). The driver
/// returns these to `main`, which renders them and exits non-zero.
pub const EmitError = struct {
    message: []const u8,
    byte_offset: ?u32,
    /// Graph builds: the owning module id (index into `Graph.modules`) whose
    /// source `byte_offset` points into. `null` (single-file path, or a
    /// whole-program error with no single owner) renders against the entry file.
    module: ?u32 = null,
};

/// The linked whole-program lowering: the joined `__text` blob, `main`'s resolved
/// entry offset within it, and any codegen diagnostics (the caller checks
/// `diags.len` and prints them, matching the prior contract). `owned_msgs` are the
/// heap-allocated diagnostic messages. Caller frees via `deinit`.
pub const LinkedProgram = struct {
    text: []u8,
    entry_off: u32,
    diags: []CodegenIr.Diagnostic,
    owned_msgs: [][]u8,
    /// Interned `__cstring` bytes (M2). Owned; empty for string-free programs.
    cstrings: []u8 = &.{},
    /// Cross-segment relocs (adrp/add/ldr to __cstring/__got) rebased to absolute
    /// __text offsets, patched by `Link.applyDataRelocs` after MachO assigns
    /// vmaddrs. Owned; empty for M1/M3 programs.
    data_relocs: []Link.Reloc = &.{},
    /// Whether the program calls `print` (→ one `_write` import).
    uses_write: bool = false,
    /// M5 incremental counters: how many functions were freshly lowered vs served
    /// from the codegen cache this build. Surfaced via `--codegen-stats`.
    codegen_compiled: usize = 0,
    codegen_cached: usize = 0,
    /// M13 dual-metric: summed opt counters over freshly-lowered fns, the total
    /// IR instruction count after opt, and the emitted aarch64 instruction count
    /// (text.len/4). No owned slices → deinit unchanged. Cached fns contribute 0
    /// to opt_stats/ir_instrs (their opt ran on a prior build), so honest
    /// `--opt-stats` numbers require `--force`.
    opt_stats: Opt.Stats = .{},
    ir_instrs: usize = 0,
    emitted_instrs: usize = 0,

    pub fn deinit(self: *LinkedProgram, gpa: std.mem.Allocator) void {
        gpa.free(self.text);
        gpa.free(self.diags);
        for (self.owned_msgs) |m| gpa.free(m);
        gpa.free(self.owned_msgs);
        gpa.free(self.cstrings);
        // `.import` data-reloc targets carry an owned name copy (the source
        // FnCodes were freed by the relink tail); `.cstr` targets are offsets.
        for (self.data_relocs) |rl| switch (rl.target) {
            .import => |s| gpa.free(s.name),
            .func, .cstr => {},
        };
        gpa.free(self.data_relocs);
        self.* = undefined;
    }
};

/// What `lowerGraphProgram` produced for an already type-checked program: either a
/// linked program (caller frees) or one `EmitError`.
pub const LowerProgramResult = union(enum) {
    ok: LinkedProgram,
    err: EmitError,
};

/// One parallel codegen job's slot. Each job owns exactly one slot (no locks).
const FnSlot = struct {
    fc: ?Link.FnCode = null,
    cached: bool = false,
    err: ?anyerror = null,
    /// M13 dual-metric, set only when the fn is freshly lowered (cached fns
    /// leave these at 0). Accumulated into the LinkedProgram in the collect loop.
    opt_stats: Opt.Stats = .{},
    ir_instrs: usize = 0,
};

/// The single-fn, read-only view one codegen job runs against (thread-safe: nothing
/// here is mutated during the fan-out). Built per job by `GraphFrozen.frozenFor` from
/// the job's owning-module per-module arrays plus the program-wide tables.
const Frozen = struct {
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const Resolve.Resolution,
    node_types: []const Typecheck.Type,
    layouts: []const Typecheck.Layout,
    enum_layouts: []const Typecheck.EnumLayout,
    names: []const Link.SymName,
    /// A one-element slice naming this job's fn (`frozenFor` points it at a stack buf).
    fn_nodes: []const Ast.Index,
    sigs: []const Fingerprint.Sig,
    /// M13 opt level / pass selection. Mixed into the codegen cache key so
    /// toggling `-O` lands on a different entry, and threaded into `lowerOne`.
    opt: Opt.Config,
};

/// What `renderGraphIr` produced: either the rendered IR text (caller frees)
/// or one `EmitError`.
pub const IrResult = union(enum) {
    ok: []u8,
    err: EmitError,
};

/// The SERIAL relink tail (every build, uncached): derive `uses_write`, append
/// the print body, intern strings program-wide (deterministic: fn source order
/// then in-fn literal order), rewrite `.cstr` hashes to offsets, then link and
/// rebase cross-segment relocs. `slots` is consumed (each FnCode freed). [C8]
fn relink(
    io: Io,
    gpa: std.mem.Allocator,
    slots: []FnSlot,
    names: []const Link.SymName,
    entry_fn: u32,
    compiled: usize,
    cached_n: usize,
    opt_stats: Opt.Stats,
    ir_instrs: usize,
) !LowerProgramResult {
    // Gather the lowered fns into a contiguous slice in source order. `linkAndTail`
    // CONSUMES the elements (frees each FnCode on every path), so we only free the
    // backing array here, never its elements.
    var fns: std.ArrayList(Link.FnCode) = .empty;
    defer fns.deinit(gpa);
    for (slots) |*s| {
        try fns.append(gpa, s.fc.?);
        s.fc = null; // ownership moved into `fns`, then into linkAndTail
    }

    var lp = linkAndTail(io, gpa, fns.items, names, entry_fn) catch |e| switch (e) {
        error.CallTargetTooFar => return .{ .err = .{ .message = "call target out of range for M1 codegen", .byte_offset = null } },
        error.UnresolvedSymbol, error.NoEntry => return .{ .err = .{ .message = "internal: unresolved symbol after codegen", .byte_offset = null } },
        else => |err| return err,
    };
    lp.diags = try gpa.alloc(CodegenIr.Diagnostic, 0);
    lp.owned_msgs = try gpa.alloc([]u8, 0);
    lp.codegen_compiled = compiled;
    lp.codegen_cached = cached_n;
    lp.opt_stats = opt_stats;
    lp.ir_instrs = ir_instrs;
    // Dual-metric machine-code count: aarch64 is fixed-width 4-byte instrs, so
    // the emitted instruction count is the linked __text length / 4.
    lp.emitted_instrs = lp.text.len / 4;
    return .{ .ok = lp };
}

/// Shared back half of both paths: append the print body if referenced, intern
/// strings deterministically, rewrite `.cstr` targets, then `Link.link`. CONSUMES
/// `fns` (frees each FnCode and any appended print body). Returns a LinkedProgram
/// with `diags`/`owned_msgs` left empty for the caller to fill.
fn linkAndTail(io: Io, gpa: std.mem.Allocator, fns: []Link.FnCode, names: []const Link.SymName, entry_fn: u32) !LinkedProgram {
    const lk = try link.linkProgram(io, gpa, fns, names[entry_fn]);
    return LinkedProgram{
        .text = lk.text,
        .entry_off = lk.entry_off,
        .diags = &.{},
        .owned_msgs = &.{},
        .cstrings = lk.cstrings,
        .data_relocs = lk.data_relocs,
        .uses_write = lk.uses_write,
    };
}

// ---- M14 whole-graph code emission -----------------------------------------

/// Frozen, read-only inputs shared by every codegen job in a WHOLE-GRAPH build.
/// Per-module arrays (trees/tokens/sources/resolutions/node_types) are indexed by
/// module id; `layouts`/`enum_layouts`/`names`/`sigs` are PROGRAM-WIDE (one global
/// id space). `fn_decls`/`fn_modules` are parallel to the global fn id space (one
/// entry per global fn; `print` excluded — only real fn bodies are lowered). A
/// codegen job selects its fn's owning module to build a single-fn `Frozen` view,
/// so `walkCalls`/`walkTouchedSig`/`typeRefToType`/`fingerprint`/`lowerOne` run
/// UNCHANGED — the only difference from single-file is which module's tree/resolutions
/// they read. [design 8/10]
const GraphFrozen = struct {
    trees: []const Ast.Tree,
    tokens: []const []const Token,
    sources: []const []const u8,
    resolutions: []const []const Resolve.Resolution,
    node_types: []const []const Typecheck.Type,
    /// Program-wide tables (global ids).
    layouts: []const Typecheck.Layout,
    enum_layouts: []const Typecheck.EnumLayout,
    /// Program-wide: one SymName per global fn id (qualified user fns, bare `main`,
    /// `{builtin,"print"}` at the print id). Parallel to `sigs`.
    names: []const Link.SymName,
    sigs: []const Fingerprint.Sig,
    /// The `fn_decl` node for each LOWERABLE global fn (parallel to `lower_ids`).
    fn_decls: []const Ast.Index,
    /// The owning module id for each lowerable fn (parallel to `fn_decls`).
    fn_modules: []const u32,
    /// Global fn id of each lowerable fn (parallel to `fn_decls`); indexes
    /// `names`/`sigs`. Excludes `print` (bodyless).
    lower_ids: []const u32,
    /// Global fn id of the entry `main` (indexes `names`).
    entry_id: u32,
    opt: Opt.Config,
    /// M16: the per-build dependency-recording sink, threaded only by the
    /// `--dump-dag` path. `null` on every default (`-o`/`run`/`--emit`) build so
    /// each per-fn codegen job runs through `Engine.init` (dag=null) verbatim —
    /// the byte-identity guarantee. When non-null each job runs through
    /// `Engine.initDag` so `query()` records caller->callee codegen edges.
    dag: ?*Dag = null,

    /// `--timings` sub-stage probe (codegen-compute vs cache get/put I/O), threaded
    /// only when `--timings` is on. BORROWED; null on a plain build => zero overhead.
    probe: ?*Engine.LowerProbe = null,

    /// Build the single-fn `Frozen` view a codegen job runs against: this fn's
    /// owning module's per-module arrays + the program-wide tables. `fn_nodes` is
    /// a one-element slice naming this fn (the job reads `fn_nodes[0]`).
    fn frozenFor(gf: *const GraphFrozen, lower_i: usize, fn_node_buf: *[1]Ast.Index) Frozen {
        const mod = gf.fn_modules[lower_i];
        fn_node_buf[0] = gf.fn_decls[lower_i];
        return .{
            .tree = gf.trees[mod],
            .tokens = gf.tokens[mod],
            .source = gf.sources[mod],
            .resolutions = gf.resolutions[mod],
            .node_types = gf.node_types[mod],
            .layouts = gf.layouts,
            .enum_layouts = gf.enum_layouts,
            .names = gf.names,
            .fn_nodes = fn_node_buf[0..1],
            .sigs = gf.sigs,
            .opt = gf.opt,
        };
    }
};

/// Lower EVERY function across the whole module `graph` (in parallel, memoized
/// through the codegen cache), then run the serial relink tail. The entry `main`
/// MUST live in the entry module (error otherwise). `res`/`tc` are the whole-graph
/// resolve/typecheck results (program-wide global ids). Caller owns the returned
/// `LinkedProgram` on success. [design 5/8]
pub fn lowerGraphProgram(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    target: []const u8,
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const Typecheck.GraphResult,
    mode: CodegenIr.Mode,
    opt: Opt.Config,
    dag: ?*Dag,
    probe: ?*Engine.LowerProbe,
    link_ns: ?*u64,
    /// The `-j` jobs knob: the chunk-count basis for the per-fn codegen fan-out (0 =>
    /// host cpu count). PERF P1 — codegen dispatches ~`ncpu` chunks, not one task per
    /// fn, so `-jN` scales. Determinism is unchanged (ranges are disjoint, slots read
    /// back in index order), so this is a perf lever only.
    ncpu: usize,
) !LowerProgramResult {
    const n_mods = graph.modules.len;

    // --- per-module Frozen inputs (one Tree view + arrays per module) ---
    const trees = try gpa.alloc(Ast.Tree, n_mods);
    defer gpa.free(trees);
    const toks = try gpa.alloc([]const Token, n_mods);
    defer gpa.free(toks);
    const srcs = try gpa.alloc([]const u8, n_mods);
    defer gpa.free(srcs);
    const resols = try gpa.alloc([]const Resolve.Resolution, n_mods);
    defer gpa.free(resols);
    const ntypes = try gpa.alloc([]const Typecheck.Type, n_mods);
    defer gpa.free(ntypes);
    for (graph.modules, 0..) |*m, i| {
        trees[i] = m.tree();
        toks[i] = m.tokens;
        srcs[i] = m.source;
        resols[i] = res.resolutions[i];
        ntypes[i] = tc.node_types[i];
    }

    // --- program-wide SymName table (one per global fn id) ---
    // Mirrors the resolver's global fn order exactly: a user fn -> {user_fn, its
    // qualified/bare name from the typecheck sig}; the synthetic bodyless `print`
    // -> {builtin,"print"}. The qualified name MUST equal the typecheck sig name
    // so the fingerprint callee fold lines up with the reloc target. [design 8]
    const names = try buildGraphNames(gpa, res.fns, tc.sigs);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }

    // --- enumerate lowerable fns (skip the bodyless print) + find entry main ---
    var fn_decls: std.ArrayList(Ast.Index) = .empty;
    defer fn_decls.deinit(gpa);
    var fn_modules: std.ArrayList(u32) = .empty;
    defer fn_modules.deinit(gpa);
    var lower_ids: std.ArrayList(u32) = .empty;
    defer lower_ids.deinit(gpa);
    var entry_id: ?u32 = null;
    for (res.fns, 0..) |gf, gid| {
        if (gf.decl_node == Ast.none) continue; // synthetic print: no body to lower
        // The entry `main` is the bare {user_fn,"main"} fn in the entry module.
        if (gf.module == graph.entry_index and std.mem.eql(u8, gf.name, "main"))
            entry_id = @intCast(gid);
        try fn_decls.append(gpa, gf.decl_node);
        try fn_modules.append(gpa, gf.module);
        try lower_ids.append(gpa, @intCast(gid));
    }

    const eid = entry_id orelse return .{ .err = .{
        .message = "-o requires a function named 'main' in the entry module",
        .byte_offset = null,
    } };

    // Parameters on main are unsupported (codegen also guards; cleaner up front).
    {
        const em = &graph.modules[graph.entry_index];
        const main_decl = em.nodes[res.fns[eid].decl_node];
        const main_proto = Ast.protoAt(em.tree(), main_decl.lhs);
        if (main_proto.params.len > 0) return .{ .err = .{
            .message = "parameters on main unsupported in M1 codegen",
            .byte_offset = em.tokens[main_decl.main_token].start,
            .module = graph.entry_index,
        } };

        // Rule 7: `main` may only yield `int` (the process exit code) or `()`
        // (nothing) — any other return type has no entry-point semantics. Enforced
        // primarily in the type checker (`Typecheck.checkMainReturn`, sorted into the
        // type-diagnostic stream); this is the codegen-entry backstop.
        const main_ret = tc.sigs[eid].ret;
        if (main_ret.kind != .int and main_ret.kind != .unit) return .{ .err = .{
            .message = "main must return int or ()",
            .byte_offset = em.tokens[main_decl.main_token].start,
            .module = graph.entry_index,
        } };
    }

    const gf = GraphFrozen{
        .trees = trees,
        .tokens = toks,
        .sources = srcs,
        .resolutions = resols,
        .node_types = ntypes,
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = names,
        .sigs = tc.sigs,
        .fn_decls = fn_decls.items,
        .fn_modules = fn_modules.items,
        .lower_ids = lower_ids.items,
        .entry_id = eid,
        .opt = opt,
        .dag = dag,
        .probe = probe,
    };

    // --- parallel per-fn fan-out (one slot per lowerable fn) ---
    const slots = try gpa.alloc(FnSlot, fn_decls.items.len);
    defer gpa.free(slots);
    for (slots) |*s| s.* = .{};

    const Ctx = struct {
        gpa: std.mem.Allocator,
        io: Io,
        cache: Cache,
        target: []const u8,
        mode: CodegenIr.Mode,
        gf: *const GraphFrozen,
        slots: []FnSlot,
        pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(graphFnJob)) {
            return .{ c.gpa, c.io, c.cache, c.target, c.mode, c.gf, i, &c.slots[i] };
        }
    };
    Engine.chunkedFanOut(io, fn_decls.items.len, ncpu, Engine.Chunk.codegen.threshold, Engine.Chunk.codegen.chunks_per_cpu, graphFnJob, Ctx{
        .gpa = gpa,
        .io = io,
        .cache = cache,
        .target = target,
        .mode = mode,
        .gf = &gf,
        .slots = slots,
    });

    var first_err: ?anyerror = null;
    for (slots) |s| if (s.err) |e| {
        if (first_err == null) first_err = e;
    };
    if (first_err) |e| {
        for (slots) |*s| if (s.fc) |*fc| fc.deinit(gpa);
        return e;
    }

    var compiled: usize = 0;
    var cached_n: usize = 0;
    var opt_stats: Opt.Stats = .{};
    var ir_instrs: usize = 0;
    for (slots) |s| {
        if (s.cached) cached_n += 1 else compiled += 1;
        opt_stats.add(s.opt_stats);
        ir_instrs += s.ir_instrs;
    }

    // The relink entry index is the entry fn's POSITION in the lowered `slots`
    // (source-order = the lowerable-fn enumeration order), not its global id. Find
    // it by matching the entry global id in `lower_ids`.
    var entry_pos: u32 = 0;
    for (lower_ids.items, 0..) |gid, i| if (gid == eid) {
        entry_pos = @intCast(i);
        break;
    };

    // Build a names slice parallel to the LOWERED fns (relink/link index by
    // lowered-fn position, not global id). Each lowered fn's name is its global
    // SymName; borrowed (not owned) — the backing `names` outlives relink here.
    const lowered_names = try gpa.alloc(Link.SymName, fn_decls.items.len);
    defer gpa.free(lowered_names);
    for (lower_ids.items, 0..) |gid, i| lowered_names[i] = names[gid];

    const link_t0: i128 = if (link_ns != null) nowNs(io) else 0;
    const out = relink(io, gpa, slots, lowered_names, entry_pos, compiled, cached_n, opt_stats, ir_instrs);
    if (link_ns) |lp| {
        const dt = nowNs(io) - link_t0;
        lp.* = if (dt > 0) @intCast(dt) else 0;
    }
    return out;
}

fn nowNs(io: Io) i128 {
    return Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
}

/// One per-fn codegen job: build the single-fn `Frozen` view for this fn's owning
/// module, then run the fingerprint→cache→lower path via `Engine.codegen`. The
/// cross-module callee identity + touched layouts ride in through the program-wide
/// `names`/`sigs`/`layouts`, so the fingerprint folds a qualified callee's SymName+sig
/// distinctly with NO engine change. [design 8/10]
fn graphFnJob(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    target: []const u8,
    mode: CodegenIr.Mode,
    gf: *const GraphFrozen,
    lower_i: usize,
    slot: *FnSlot,
) void {
    graphFnJobInner(gpa, io, cache, target, mode, gf, lower_i, slot) catch |e| {
        slot.err = e;
    };
}

fn graphFnJobInner(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    target: []const u8,
    mode: CodegenIr.Mode,
    gf: *const GraphFrozen,
    lower_i: usize,
    slot: *FnSlot,
) !void {
    var fn_node_buf: [1]Ast.Index = undefined;
    const frozen = gf.frozenFor(lower_i, &fn_node_buf);
    const fn_decl = frozen.fn_nodes[0];
    const gid = gf.lower_ids[lower_i];
    const sym = gf.names[gid];
    const is_entry = gid == gf.entry_id;
    // Thread this fn's typecheck sig (program-wide, indexed by global id) so the
    // fn_decl param/return fold uses the ABI-correct GLOBAL type ids — including a
    // CROSS-MODULE qualified `b: rect.Rect`. A bare-name re-resolution would mis-pick
    // the first same-named type in the merged layout table, missing a pub-type
    // layout edit at the importer (cross-module M9 / TOP-RISK-#1 hole). [design 10]
    const my_sig: ?Fingerprint.Sig = if (gid < gf.sigs.len) gf.sigs[gid] else null;

    // The cross-module callee identity + touched layouts ride in through the
    // program-wide `names`/`sigs`/`layouts` of this fn's `frozen` view, so the
    // fingerprint folds a qualified callee distinctly with NO engine change.
    // tmp_tag = `lower_i`.
    const engine = Engine.initProbe(cache, mode, gf.dag, gf.probe);
    try engine.codegen(gpa, io, target, &frozen, fn_decl, sym, is_entry, my_sig, lower_i, gid, slot);
}

/// Build the program-wide index→SymName table for a graph build: one entry per
/// global fn id, parallel to `tc.sigs`. A user fn -> {user_fn, sig name} (the
/// qualified spelling, or bare `main`); the synthetic bodyless `print` ->
/// {builtin,"print"}. The name MUST equal the sig name so the fingerprint callee
/// fold matches the reloc target. Caller owns the names.
fn buildGraphNames(gpa: std.mem.Allocator, fns: []const ResolveGraph.GlobalFn, sigs: []const Fingerprint.Sig) ![]Link.SymName {
    const names = try gpa.alloc(Link.SymName, fns.len);
    var built: usize = 0;
    errdefer {
        for (names[0..built]) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, sigs[i].name) };
        built += 1;
    }
    return names;
}

/// `--emit ir` for a whole-graph build: run the `lower` stage over every fn across
/// the graph and render the deterministic IR text (each fn rendered against its
/// owning module's tree). Caller owns the returned text on success.
pub fn renderGraphIr(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const Typecheck.GraphResult,
    opt: Opt.Config,
) !IrResult {
    const names = try buildGraphNames(gpa, res.fns, tc.sigs);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }

    var aw: std.Io.Writer.Allocating = .init(gpa);
    errdefer aw.deinit();
    var diags: std.ArrayList(CodegenIr.Diagnostic) = .empty;
    defer diags.deinit(gpa);

    var first = true;
    for (res.fns, 0..) |gf, gid| {
        if (gf.decl_node == Ast.none) continue; // skip bodyless print
        const m = &graph.modules[gf.module];
        const is_entry = gf.module == graph.entry_index and std.mem.eql(u8, gf.name, "main");
        const in = lower.Inputs{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[gf.module],
            .node_types = tc.node_types[gf.module],
            .layouts = tc.layouts,
            .enum_layouts = tc.enum_layouts,
            .names = names,
            .sig = if (gid < tc.sigs.len) tc.sigs[gid] else null,
        };
        var func = try lower.lowerFn(gpa, in, gf.decl_node, names[gid], is_entry, &diags);
        defer func.deinit(gpa);
        var opt_st: Opt.Stats = .{};
        try Opt.run(gpa, &func, opt, &opt_st);
        if (!first) try aw.writer.writeAll("\n");
        try Ir.render(&aw.writer, &func, in.layouts, in.enum_layouts);
        first = false;
    }

    var list = aw.toArrayList();
    return .{ .ok = try list.toOwnedSlice(gpa) };
}

/// Build the fully signed, runnable Mach-O image for `code`. Thin pass-through to
/// the single backend boundary (`link.assembleAndSign`), which owns the
/// order-sensitive assemble → applyDataRelocs → sign-last pipeline. Kept so the
/// asm path and the byte-identity tests call it on `LinkedProgram` fields.
/// Caller owns the returned bytes and writes them mode 0o755.
pub fn buildImage(
    io: Io,
    gpa: std.mem.Allocator,
    identifier: []const u8,
    code: []const u8,
    entry_off: u32,
    cstrings: []const u8,
    data_relocs: []const Link.Reloc,
    uses_write: bool,
) ![]u8 {
    return link.assembleAndSign(io, gpa, identifier, code, entry_off, cstrings, data_relocs, uses_write);
}

// ---- tests -----------------------------------------------------------------

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
    mode: CodegenIr.Mode,
    opt: Opt.Config,
) !LowerProgramResult {
    var graph = try Graph.single(gpa, "main", r.path, r.source, r.tokens, r.nodes, r.extra, r.pub_bits);
    defer graph.deinitSingle(gpa);
    return lowerGraphProgram(gpa, io, cache, target, &graph, &r.resolve.?, &r.typecheck.?, mode, opt, null, null, null, 0);
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

    const src = "fn add(a: int, b: int) -> int {\n return a + b\n}\n";
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
        // M3 multi-function cases — each exercises a back-end capability that a
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
        // M4 control flow — each RUN proves a back-end capability a byte assert
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
        // M6 expression orientation — each RUN proves the value-merge/trailing-expr
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
        // (ptr,len) must reach the caller in (x0,x1) and feed `print`. Exits 0.
        .{ .src = "fn pick(b: int) -> str {\n if b > 0 { \"yes\" } else { \"no\" }\n}\nfn main() -> int {\n print(pick(1))\n return 0\n}\n", .name = "strret_call", .expect = 0 },
        // ENTRY-EPILOGUE REGRESSION: a non-unit `main` whose body is a TRAILING value
        // expression (no explicit return) lands its value in x0 at the fall-through
        // epilogue — the entry's `movz x0,#0` must NOT clobber it (only a UNIT entry
        // forces 0). Trailing if-value → 7.
        .{ .src = "fn main() -> int {\n if 1 > 0 { 7 } else { 8 }\n}\n", .name = "trailing_main_if", .expect = 7 },
        // Trailing bare-block value as the non-unit main body → 15.
        .{ .src = "fn main() -> int {\n {\n a := 10\n a + 5\n }\n}\n", .name = "trailing_main_block", .expect = 15 },

        // M7 loops — each RUN is a HANG-CANARY: a wrong/sign-flipped back-edge would
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

        // M8 labels & multi-level exits — each RUN is a HANG-CANARY (a wrong outer
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
        // (M7 unchanged): inner `for` adds 1 then bare-breaks at j==1 → 1 per outer,
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
        // result slot + x0/x1) consumed by `print`. Exits 0; proves the fat-value
        // store/load through the labeled-block result slot.
        .{ .src = "fn greet(s: str) -> str {\n s\n}\nfn main() -> int {\n print(@blk { break @blk greet(\"hi\") })\n return 0\n}\n", .name = "labeledblock_str", .expect = 0 },

        // M9 structs — each RUN proves a back-end capability (layout/ABI/copy) that
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
        // M9 SIGBUS regression (frame undersizing): RETURN a SMALL struct whose
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
        // M9 sret regression: a >16B struct returned via a VALUE-IF body (not an
        // expr_stmt) must be written through x8 — was lowered as a statement-if and
        // x8 left unwritten (garbage). make(1)={10,20,30}, sum = 60.
        .{ .src = "struct V3 { a: int, b: int, c: int }\nfn make(c: int) -> V3 {\n if c == 1 { V3 { a: 10, b: 20, c: 30 } } else { V3 { a: 1, b: 1, c: 1 } }\n}\nfn main() -> int {\n p := make(1)\n return p.a + p.b + p.c\n}\n", .name = "struct_sret_valueif", .expect = 60 },
        // M9 miscompile regression: a struct literal as a call arg whose field init
        // is itself a CALL — the field value must spill PAST the struct's own temp
        // bytes (not onto x). area(Point{40, id(2)}) = 42.
        .{ .src = "struct Point { x: int, y: int }\nfn id(n: int) -> int { return n }\nfn area(p: Point) -> int { p.x + p.y }\nfn main() -> int {\n return area(Point { x: 40, y: id(2) })\n}\n", .name = "struct_arg_callinit", .expect = 42 },
        // M9 SIGSEGV regression: a >16B struct returned, a field init calls a 9-arg
        // fn (whose arg marshal dirties x9) — the sret dest pointer must NOT be held
        // in caller-saved x9 across the bl. val(1..9)=45, mk()={45,1,1}, sum = 47.
        .{ .src = "struct V3 { a: int, b: int, c: int }\nfn val(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int) -> int {\n return a + b + c + d + e + f + g + h + i\n}\nfn mk() -> V3 {\n return V3 { a: val(1,2,3,4,5,6,7,8,9), b: 1, c: 1 }\n}\nfn main() -> int {\n q := mk()\n return q.a + q.b + q.c\n}\n", .name = "struct_sret_x9", .expect = 47 },
        // M9 coverage regression: a SMALL struct produced by a value-if, BOUND to a
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
        // M10 enums — each RUN proves an enum capability a byte assert can't.
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
        // 5 + 5 = 10. (M10 enums have no field-store; the by-value bind is the copy.)
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
        // (e14) PRE-EXISTING M9 STRUCT widening of the same defect: a >16B struct
        // returned by a call passed as the struct arg of another call. echo identity,
        // consume sums. consume(echo(Big{10,20,30,40})) = 100.
        .{ .src = "struct Big { p: int, q: int, r: int, s: int }\nfn echo(b: Big) -> Big { b }\nfn consume(b: Big) -> int { b.p + b.q + b.r + b.s }\nfn main() -> int {\n return consume(echo(Big { p: 10, q: 20, r: 30, s: 40 }))\n}\n", .name = "struct_big_call_in_arg", .expect = 100 },
        // M11 match enrichment.
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

        const image = try buildImage(io, gpa, c.name, lp.text, lp.entry_off, lp.cstrings, lp.data_relocs, lp.uses_write);
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
        .{ .src = "fn main() {\n print(\"hello world\\n\")\n}\n", .name = "hw", .want = "hello world\n" },
        // A str local + a second literal: two distinct cstrings + a 16-byte slot.
        .{ .src = "fn main() {\n print(\"AB\")\n s := \"CD\\n\"\n print(s)\n}\n", .name = "two", .want = "ABCD\n" },
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

        const image = try buildImage(io, gpa, c.name, lp.text, lp.entry_off, lp.cstrings, lp.data_relocs, lp.uses_write);
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

// ---- M5 incremental + parallel codegen tests -------------------------------

/// Lower `src` (written to `path`) through the codegen cache in `cache`, then
/// return the linked program. Caller owns the result (deinit) and `r` (deinit).
fn checkAndLower(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    path: []const u8,
    src: []const u8,
    mode: CodegenIr.Mode,
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

test "M5 edit-one-fn: only the edited fn recompiles; callers unaffected by a body change [iii]" {
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
    // Exactly one fn recompiles (add); main is a cache hit. [C1]
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

test "M5 verify-mode: re-lowering every fn matches the cached blob [v]" {
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

test "M6 cache soundness: editing a value-if fn recompiles only it; verify passes [v]" {
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

test "M7 cache soundness: editing a loop/for/break fn recompiles only it; verify passes [v]" {
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

test "M8 cache soundness: editing a labeled/break fn recompiles only it; verify passes [v]" {
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

test "M9 cache soundness: editing a struct's fields recompiles every fn that touches it; verify passes [v]" {
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
    // This is the M5 "touched type layouts" hook made REAL.
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

test "M9 cache soundness: a struct touched ONLY via a param/return type folds its layout (no stale hit across the ABI boundary) [v]" {
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

test "M10 cache soundness: editing an enum's variants recompiles every fn that touches it; verify passes [v]" {
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

test "M10 cache soundness: an enum touched ONLY via a param type folds its layout across the 16B<->24B ABI boundary [v]" {
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

test "M10 errors: non-exhaustive/unknown variant/arity/type; recursive enum; uninferable .V" {
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

test "M11 errors: non-exhaustive int/bool; guarded-only/partial-nested variant; inconsistent or-bindings; non-bool guard [iv]" {
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

test "M11 cache soundness: editing a literal/guard recompiles only that fn; verify byte-identical [v]" {
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

test "M9 errors: missing/unknown/mismatched fields; positional construction; recursive struct" {
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

test "M8 errors: undefined/duplicate labels (Resolve); continue-block & value-break-while (Type); bad label prefix (Parse)" {
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

test "M7 error: break/continue outside a loop and break-value in while/for are rejected" {
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

test "M5 byte-identical: a warm build equals a from-scratch (.force) build [iv]" {
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
    const warm_img = try buildImage(io, gpa, "p", warm.text, warm.entry_off, warm.cstrings, warm.data_relocs, warm.uses_write);
    defer gpa.free(warm_img);

    // From scratch (ignore the cache).
    var rf: FileResult = undefined;
    var fresh = try checkAndLower(gpa, io, cache, path, src, .force, &rf);
    defer rf.deinit(gpa);
    defer fresh.deinit(gpa);
    const fresh_img = try buildImage(io, gpa, "p", fresh.text, fresh.entry_off, fresh.cstrings, fresh.data_relocs, fresh.uses_write);
    defer gpa.free(fresh_img);

    try testing.expectEqualSlices(u8, fresh_img, warm_img);
}

test "M5 reorder: swapping fn order is all cache hits and keeps correct linkage [vi]" {
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
    const image = try buildImage(io, gpa, "p", lpb.text, lpb.entry_off, lpb.cstrings, lpb.data_relocs, lpb.uses_write);
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
