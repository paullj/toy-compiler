//! The incremental query engine: one testable seam that owns the on-disk memo
//! (`Cache`) and the uniform `query(key, compute) -> result` entry. M15 lifts the
//! distributed query/cache/fingerprint/force-verify/parallel logic out of the
//! driver and into this module; the driver becomes a thin orchestrator that
//! sequences phases and demands each unit through here.
//!
//! This is a BEHAVIOR-PRESERVING refactor: each query is still pull-computed or
//! served from the on-disk content-addressed cache exactly as before. The cache
//! stays the SOLE memo (lock-free atomic-rename writers, [C11] determinism); no
//! shared mutable in-memory map is introduced — each unit is computed once per
//! build, so a map would only race for zero gain.
//!
//! Scaffold stage: the engine owns the borrowed `Cache` + the build `Mode`, and
//! exposes the generic `query` entry plus the `lex` front-end helper as the first
//! query routed through it. Later stages fold codegen / force-verify / parallel
//! fan-out in behind the same seam.

const std = @import("std");
const Io = std.Io;
const Token = @import("../ast/Token.zig").Token;
const Lexer = @import("../lex.zig");
const Cache = @import("Cache.zig");
const Key = @import("Key.zig");
const CodegenIr = @import("../codegen/CodegenIr.zig");
const Ast = @import("../ast/Ast.zig");
const Fingerprint = @import("Fingerprint.zig");
const Walks = @import("Walks.zig");
const Link = @import("../link/Link.zig");
const Ir = @import("../ir/Ir.zig");
const Opt = @import("../opt/Opt.zig");
const lower = @import("../lower.zig");

const Engine = @This();

/// How force/verify behave for cached queries (threaded from `main.zig`). Aliased
/// to the codegen mode so the driver and engine share one type.
pub const Mode = CodegenIr.Mode;

/// The on-disk content-addressed memo. Borrowed; must outlive the engine (the CLI
/// keeps the cache + its dir buffer on its stack for the whole run).
cache: Cache,

/// force/verify mode for cached compute queries. Front-end queries (lex/parse)
/// ignore it — they are pure source-hash lookups with no re-lower gate.
mode: Mode = .normal,

pub fn init(cache: Cache, mode: Mode) Engine {
    return .{ .cache = cache, .mode = mode };
}

/// The result of a single query: the value plus whether it was served from the
/// on-disk cache (vs freshly computed). `cached` preserves the per-phase
/// `*_cached` bookkeeping the driver reports.
pub fn Result(comptime T: type) type {
    return struct { value: []T, cached: bool };
}

/// The uniform demand entry: serve `key` from the cache, else run `compute` and
/// store the result. `compute` returns a freshly allocated `[]T` the caller owns
/// on a miss; on a hit the cached slice is returned (also caller-owned).
///
/// `tmp_tag` must be unique among concurrent writers (the file/fn index) so temp
/// files never collide on store. `swallow_put` mirrors the front-end's policy of
/// treating a cache-write failure as non-fatal (`cache.put(...) catch {}`): a
/// failed store just means the next build re-runs the query. `get` errors always
/// propagate.
pub fn query(
    self: Engine,
    comptime T: type,
    gpa: std.mem.Allocator,
    io: Io,
    key: Cache.Key,
    tmp_tag: usize,
    comptime swallow_get: bool,
    comptime swallow_put: bool,
    compute: anytype,
) !Result(T) {
    const hit: ?[]T = if (swallow_get) (self.cache.get(T, gpa, io, key) catch null) else (try self.cache.get(T, gpa, io, key));
    if (hit) |h| {
        return .{ .value = h, .cached = true };
    }
    const fresh = try compute.run();
    if (swallow_put) {
        self.cache.put(T, io, key, tmp_tag, fresh) catch {};
    } else {
        try self.cache.put(T, io, key, tmp_tag, fresh);
    }
    return .{ .value = fresh, .cached = false };
}

/// lex query (the M15 proof seam): tokenize `source` for `target`, served from
/// cache or freshly lexed. Preserves the pipeline's policy: `get` propagates,
/// `put` is swallowed, `tmp_tag` is the caller's file index.
pub fn lex(
    self: Engine,
    gpa: std.mem.Allocator,
    io: Io,
    target: []const u8,
    source: []const u8,
    tmp_tag: usize,
    comptime swallow_get: bool,
) !Result(Token) {
    const LexCompute = struct {
        gpa: std.mem.Allocator,
        source: []const u8,
        fn run(c: @This()) ![]Token {
            return Lexer.tokenize(c.gpa, c.source);
        }
    };
    return self.query(Token, gpa, io, Key.lex(target, source), tmp_tag, swallow_get, true, LexCompute{
        .gpa = gpa,
        .source = source,
    });
}

/// The parse query: serve the file's `Ast.Tree` from cache or freshly parse. The
/// parse artifact is a flat `[]u8` blob (`Ast.pack`/`unpack`); a hit must be
/// VALIDATED by `Ast.unpack`, which treats a corrupt/foreign blob as a miss (so
/// the query re-parses rather than trusting a bad blob). Unlike the generic
/// `query`, parse cannot use the raw memo path because of this unpack validation.
///
/// `swallow_get` selects the caller's read policy: the per-file pipeline propagates
/// `get`/`unpack` errors (`false`), while module-graph discovery swallows them and
/// treats a failed read as a plain miss (`true`). `put` is always swallowed (a
/// failed store just means the next build re-parses). On a parse FAILURE the tree
/// is `null` and `diag` holds the parser diagnostic — the caller owns reporting it.
pub const ParseResult = struct {
    tree: ?Ast.Tree,
    cached: bool = false,
    diag: ?@import("../parse.zig").Diagnostic = null,
};

pub fn parse(
    self: Engine,
    gpa: std.mem.Allocator,
    io: Io,
    target: []const u8,
    source: []const u8,
    tokens: []const Token,
    tmp_tag: usize,
    comptime swallow_get: bool,
) !ParseResult {
    const Parser = @import("../parse.zig");
    const cache = self.cache;
    const key = Key.parse(target, source);

    // --- cache read + validate (a corrupt blob is treated as a miss) ---
    const blob: ?[]u8 = if (swallow_get) (cache.get(u8, gpa, io, key) catch null) else (try cache.get(u8, gpa, io, key));
    if (blob) |bytes| {
        defer gpa.free(bytes);
        const unpacked: ?Ast.Tree = if (swallow_get) (Ast.unpack(gpa, bytes) catch null) else (try Ast.unpack(gpa, bytes));
        if (unpacked) |t| return .{ .tree = t, .cached = true };
    }

    // --- miss: parse fresh, store on success ---
    var diag: ?Parser.Diagnostic = null;
    if (try Parser.parse(gpa, tokens, source, &diag)) |t| {
        if (Ast.pack(gpa, t) catch null) |b| {
            defer gpa.free(b);
            cache.put(u8, io, key, tmp_tag, b) catch {};
        }
        return .{ .tree = t, .cached = false };
    }
    return .{ .tree = null, .diag = diag };
}

/// The parallel fan-out COORDINATOR: dispatch `n` jobs onto the `Io` runtime's
/// worker threads, then await them. Lifts the `concurrent(...) catch <inline>`
/// pattern used identically by the per-file front-end and the per-fn codegen
/// (single-file + graph). Each job writes ONLY its own caller-owned slot, so no
/// locks are taken; the inline fallback (when the runtime can't provide true
/// concurrency) is LOAD-BEARING for single-threaded determinism and is preserved
/// verbatim.
///
/// `jobFn` is the per-unit work function; `ctx.args(i)` returns the exact
/// `ArgsTuple` for job `i` (including its slot pointer and its `tmp_tag` index).
/// Determinism ([C11]) is unaffected: jobs are independent and order-free, and the
/// caller's collect loop reads slots in deterministic index order afterward.
pub fn fanOut(io: Io, n: usize, comptime jobFn: anytype, ctx: anytype) void {
    var group: Io.Group = .init;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const args = ctx.args(i);
        group.concurrent(io, jobFn, args) catch @call(.auto, jobFn, args);
    }
    group.await(io) catch {};
}

/// Dual-metric output of a fresh `lowerOne`: the summed opt counters for this fn
/// and its post-opt IR instruction count. Surfaced via `--opt-stats`.
pub const OptOut = struct { stats: Opt.Stats = .{}, ir_instrs: usize = 0 };

/// Lower one function (Ast→Ir→OPT→FnCode) with a throwaway diag sink (a job-local
/// diagnostic still fails the build at relink, surfaced via `error.CodegenDiagnostic`).
/// The IR is built INSIDE this query and never escapes — `irf` owns its arrays and
/// is freed here, keeping the codegen cache ONE-TIER ([C8]). The M13 opt stage runs
/// in-place on `irf` between lower and codegen; `opt_out` (when non-null) receives
/// the per-fn dual-metric numbers.
///
/// `frozen` is the single-fn read-only view (`Driver.Frozen` / a graph `frozenFor`
/// slice); taken `anytype` so the `Frozen` type can stay in the driver this stage.
fn lowerOne(gpa: std.mem.Allocator, frozen: anytype, fn_decl: Ast.Index, sym: Link.SymName, is_entry: bool, sig: ?Fingerprint.Sig, opt_out: ?*OptOut) !Link.FnCode {
    var diags: std.ArrayList(CodegenIr.Diagnostic) = .empty;
    defer diags.deinit(gpa);

    const in: lower.Inputs = .{
        .tree = frozen.tree,
        .tokens = frozen.tokens,
        .source = frozen.source,
        .resolutions = frozen.resolutions,
        .node_types = frozen.node_types,
        .layouts = frozen.layouts,
        .enum_layouts = frozen.enum_layouts,
        .names = frozen.names,
        .sig = sig,
    };
    var irf = try lower.lowerFn(gpa, in, fn_decl, sym, is_entry, &diags);
    defer irf.deinit(gpa);
    // A lower diagnostic (an unsupported/not-yet-lowered construct) fails the build
    // before we emit a partial FnCode.
    if (diags.items.len > 0) return error.CodegenDiagnostic;

    // M13: run the opt stage in-place on the IR between lower and codegen. Stays
    // ONE-TIER ([C8]) — the IR never escapes this query. With no passes enabled
    // this is a pure no-op (the scaffold O0==O1 gate).
    var opt_st: Opt.Stats = .{};
    try Opt.run(gpa, &irf, frozen.opt, &opt_st);
    if (opt_out) |o| o.* = .{ .stats = opt_st, .ir_instrs = Ir.instrCount(&irf) };

    const fc = try CodegenIr.lowerIr(gpa, &irf, frozen.layouts, frozen.enum_layouts, is_entry, &diags);
    // A codegen diagnostic means an unsupported construct slipped past the
    // front-end; surface it as an error so the build fails cleanly rather than
    // emitting a half-lowered function.
    if (diags.items.len > 0) {
        var tmp = fc;
        tmp.deinit(gpa);
        return error.CodegenDiagnostic;
    }
    return fc;
}

/// The per-function codegen query (single-file AND graph): fingerprint this fn,
/// fold the uniform codegen key, then serve it from the cache or freshly lower —
/// honoring `self.mode` (normal / force / verify). Writes ONLY the caller-owned
/// `slot` (no locks), exactly as the driver's per-fn jobs did.
///
/// The caller derives the per-fn view: `frozen` (a single-fn `Frozen` read-only
/// view), the fn's emitted `sym`, `is_entry`, its typecheck `my_sig`, and a
/// `tmp_tag` unique among concurrent writers (the fn index / lowerable-fn index).
/// `slot` is a `*FnSlot`-shaped out pointer the engine fills (taken `anytype` so
/// the `FnSlot`/`Frozen` types can stay in the driver this stage). [C8] one-tier:
/// the IR is built+freed inside `lowerOne`; only the packed `FnCode` blob is cached.
pub fn codegen(
    self: Engine,
    gpa: std.mem.Allocator,
    io: Io,
    target: []const u8,
    frozen: anytype,
    fn_decl: Ast.Index,
    sym: Link.SymName,
    is_entry: bool,
    my_sig: ?Fingerprint.Sig,
    tmp_tag: usize,
    slot: anytype,
) !void {
    const cache = self.cache;
    const mode = self.mode;

    // Gather this fn's callee sigs (in walk order) and touched types for the
    // fingerprint. The walk order is the body's call order; `walkCalls` mirrors
    // `Fingerprint`'s walk so the supplied order matches.
    var callee_sigs: std.ArrayList(Fingerprint.Sig) = .empty;
    defer callee_sigs.deinit(gpa);
    try Walks.walkCalls(gpa, frozen, fn_decl, &callee_sigs);
    var touched: std.ArrayList(Fingerprint.TouchedType) = .empty;
    defer {
        Walks.freeTouched(gpa, touched.items);
        touched.deinit(gpa);
    }
    try Walks.walkTouchedSig(gpa, frozen, fn_decl, my_sig, &touched);

    const fp = Fingerprint.fingerprint(frozen.tree, frozen.tokens, frozen.source, fn_decl, callee_sigs.items, touched.items);
    // The uniform codegen key: the content fingerprint xor'd with the opt level
    // and the fn's OWN emitted symbol, target-sensitive. `Key.codegen` is the ONE
    // place the `fp ^ optMix ^ symMix` fold lives (was duplicated in Driver).
    const key = Key.codegen(target, fp, frozen.opt, sym);

    if (mode == .verify) {
        // [C11] determinism + cache-soundness gate. ALWAYS re-lower the fn fresh
        // and assert its packed FnCode bytes are byte-identical to a reference:
        //   * cache HIT  -> compare against the stored blob (cache soundness).
        //   * cache MISS -> compare against a SECOND fresh lowering (determinism).
        // Either way the assertion runs unconditionally — it is NOT gated on a
        // primed cache, so it can never silently no-op (the dead-gate bug, where
        // `--force` skipped the cache-read branch and thus the assert entirely).
        var opt_out: OptOut = .{};
        var fresh = try lowerOne(gpa, frozen, fn_decl, sym, is_entry, my_sig, &opt_out);
        errdefer fresh.deinit(gpa);
        const fb = try Link.pack(gpa, fresh);
        defer gpa.free(fb);

        var was_cached = false;
        if (cache.get(u8, gpa, io, key) catch null) |blob| {
            defer gpa.free(blob);
            std.debug.assert(std.mem.eql(u8, fb, blob));
            was_cached = true;
        } else {
            // Cold: no reference blob to compare against, so lower a second time
            // and assert the two fresh lowerings agree (pure determinism).
            var fresh2 = try lowerOne(gpa, frozen, fn_decl, sym, is_entry, my_sig, null);
            defer fresh2.deinit(gpa);
            const fb2 = try Link.pack(gpa, fresh2);
            defer gpa.free(fb2);
            std.debug.assert(std.mem.eql(u8, fb, fb2));
            // Populate the cache so subsequent fns/runs see a primed entry.
            cache.put(u8, io, key, tmp_tag, fb) catch {};
        }
        // A verify build re-lowers fresh, so it has honest opt stats even on a
        // cache hit (was_cached==true). Record them.
        slot.* = .{ .fc = fresh, .cached = was_cached, .opt_stats = opt_out.stats, .ir_instrs = opt_out.ir_instrs };
        return;
    }

    if (mode != .force) {
        if (cache.get(u8, gpa, io, key) catch null) |blob| {
            defer gpa.free(blob);
            if (Link.unpack(gpa, blob) catch null) |fc| {
                slot.* = .{ .fc = fc, .cached = true };
                return;
            }
        }
    }

    var opt_out: OptOut = .{};
    var fc = try lowerOne(gpa, frozen, fn_decl, sym, is_entry, my_sig, &opt_out);
    errdefer fc.deinit(gpa);
    if (Link.pack(gpa, fc) catch null) |b| {
        defer gpa.free(b);
        cache.put(u8, io, key, tmp_tag, b) catch {};
    }
    slot.* = .{ .fc = fc, .cached = false, .opt_stats = opt_out.stats, .ir_instrs = opt_out.ir_instrs };
}

// Engine boundary tests (hit/miss/force/verify/invalidation + key discrimination)
// are an INTEGRATION suite (temp-dir cache + threaded runtime), so they live in
// `src/tests/query_engine.zig` rather than inline here; `root.zig` pulls them.
