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
const Io = std.Io;
const Token = @import("../ast/Token.zig").Token;
const Parser = @import("../parse.zig");
const Ast = @import("../ast/Ast.zig");
const Cache = @import("../query/Cache.zig");
const Engine = @import("../query/Engine.zig");
const Typecheck = @import("../types.zig");
const Graph = @import("Graph.zig");
const ResolveGraph = @import("../resolve_graph.zig");
const TypecheckGraph = @import("../types_graph.zig");
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
    /// `pub` export bitset paired with `nodes` (see `Ast.Tree.pub_bits`).
    /// Owned; empty for a program with no exports.
    pub_bits: []const u32 = &.{},
    /// Whether each phase's result was loaded from cache rather than recomputed.
    tokens_cached: bool = false,
    nodes_cached: bool = false,
    /// True once the parse phase ran (or hit cache) for this file.
    parsed: bool = false,
    /// A TAINTED parse produced at least one diagnostic. The (partial) tree is
    /// still populated for reporting, but the file does NOT proceed to check/codegen.
    tainted: bool = false,
    /// True once name resolution ran for this file (emit == .check).
    checked: bool = false,
    err: ?anyerror = null,
    /// On a parse error, every accumulated parser diagnostic. Owned; freed in
    /// deinit. Empty on a clean parse.
    diags: []const Parser.Diagnostic = &.{},
    /// Whole-graph name-resolution result (emit == .check). A lone file IS the
    /// trivial one-module graph, so the graph result is stored whole. SOLE owner of
    /// the fn-name strings (`fns[i].name`); the typecheck `sigs[i].name` borrow them,
    /// lifetime-tied to this sibling field. Owned; freed in deinit.
    resolve: ?ResolveGraph.GraphResult = null,
    /// Whole-graph type-check result (emit == .check, run only if resolve was
    /// clean). Stored whole; its `sigs[i].name` BORROW `resolve.?.fns[i].name`
    /// (same lifetime — sibling fields, torn down together). Owned; freed in deinit.
    typecheck: ?Typecheck.GraphResult = null,

    pub fn deinit(r: *FileResult, gpa: std.mem.Allocator) void {
        gpa.free(r.source);
        gpa.free(r.tokens);
        gpa.free(r.nodes);
        gpa.free(r.extra);
        if (r.pub_bits.len != 0) gpa.free(@constCast(r.pub_bits));
        if (r.diags.len != 0) gpa.free(@constCast(r.diags));
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

pub fn job(gpa: std.mem.Allocator, io: Io, cache: Cache, emit: Emit, target: []const u8, result: *FileResult, index: usize) void {
    pipeline(gpa, io, cache, emit, target, result, index) catch |err| {
        result.err = err;
    };
}

pub fn pipeline(gpa: std.mem.Allocator, io: Io, cache: Cache, emit: Emit, target: []const u8, result: *FileResult, index: usize) !void {
    const cache_phase = emit.cachePhase();
    result.source = try Io.Dir.cwd().readFileAlloc(io, result.path, gpa, .unlimited);

    // lex: routed through the query engine
    const engine = Engine.init(cache, .normal);
    const lexed = try engine.lex(gpa, io, target, result.source, index, false);
    result.tokens = lexed.value;
    result.tokens_cached = lexed.cached;

    if (@intFromEnum(cache_phase) < @intFromEnum(Cache.Phase.parse)) return;

    // The parse output is a Tree (nodes + extra). We pack both into one flat
    // []u8 blob and store/load it through the existing generic byte cache;
    // Ast.unpack validates a hit and treats a corrupt/foreign blob as a miss.
    result.parsed = true;
    const parsed = try engine.parse(gpa, io, target, result.source, result.tokens, index, false);
    const tree = parsed.tree;
    result.nodes_cached = parsed.cached;
    result.nodes = tree.nodes;
    result.extra = tree.extra;
    result.pub_bits = tree.pub_bits;

    // A TAINTED parse still yields a (partial) tree, but the file does NOT
    // proceed to check/codegen — a poisoned tree would only cascade spurious
    // resolve/type errors (and the `error_node` arms in lower/codegen assume a
    // tainted tree is gated out here). Report the diagnostics and stop.
    if (parsed.diags.len > 0) {
        result.tainted = true;
        result.diags = parsed.diags;
        result.err = error.ParseError;
        return;
    }

    if (!emit.runsCheck()) return;

    // check: name resolution then typecheck, in-memory only; uncached.
    // A lone source file is compiled as the trivial one-module graph: the graph
    // front-end (`resolveGraph`/`checkGraph`) is the ONLY resolve/typecheck path,
    // and its whole-graph results (one module → program-wide tables == local
    // tables) are stored WHOLE on the `FileResult`. The two results are sibling
    // fields with one shared lifetime, so the resolve `fns` table (owner of every
    // fn name) provably outlives the typecheck `sigs[].name` that borrow it.
    result.checked = true;
    var graph = try Graph.single(gpa, "main", result.path, result.source, result.tokens, result.nodes, result.extra, result.pub_bits);
    defer graph.deinit(gpa);

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

    // io=null => serial Pass-C: the per-file `run` fan-out is the only active
    // parallelism for `--emit check`, so there is no nested-pool blowup. A 1-fn
    // graph never spawns regardless; serial == parallel byte-for-byte.
    result.typecheck = try TypecheckGraph.checkGraph(gpa, &graph, &result.resolve.?, null, 0);
    if (result.typecheck.?.diags.len > 0) result.err = error.TypeError;
}
