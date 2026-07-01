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
const Phase = @import("Phase.zig");

const Engine = @This();

/// How force/verify behave for cached queries (threaded from `main.zig`), the
/// cache/verify policy this engine owns and the verify gate enforces:
///   normal — use a cached blob if present, else lower+cache.
///   force  — ignore cache, always re-lower (cold build).
///   verify — re-lower every fn TWICE and assert byte-identity ([C11]),
///            independent of cache state. Implies force (the gate must hold on
///            a cold build, not only on a primed cache hit).
pub const Mode = enum { normal, force, verify };

/// The on-disk content-addressed memo. Borrowed; must outlive the engine (the CLI
/// keeps the cache + its dir buffer on its stack for the whole run).
cache: Cache,

/// force/verify mode for cached compute queries. Front-end queries (lex/parse)
/// ignore it — they are pure source-hash lookups with no re-lower gate.
mode: Mode = .normal,

/// `--timings` per-stage compute/cache probe (see `StageProbe`). BORROWED; null on a
/// plain build so the query path reads no clock. EVERY cache-backed query — `lex` and
/// `parse` (the `discover` stage) and `codegen` (the `lower` stage) — charges the
/// engine's probe, so each stage attributes compute vs cache-get vs cache-put. A
/// different probe instance is threaded per stage (discover's vs lower's), so the two
/// don't co-mingle even though both run through this one field.
probe: ?*StageProbe = null,

/// `--timings` per-stage accumulator, split into the three costs a cache-backed stage
/// divides into: COMPUTE (the miss path — `lowerOne` for codegen, file-read+tokenize
/// for lex, the AST build for parse), cache GET I/O (the hit-or-miss read), and cache
/// PUT I/O (the atomic-rename temp-file write on a miss). Atomic because a stage's
/// fan-out (codegen, and discovery's per-module queries) runs jobs in parallel and
/// each adds its own deltas. BORROWED: one probe per stage lives on the
/// driver frame and a `*StageProbe` is threaded into every job. `null` (the default)
/// is zero-overhead — no clock is read.
pub const StageProbe = struct {
    compute_ns: std.atomic.Value(u64) = .init(0),
    get_ns: std.atomic.Value(u64) = .init(0),
    put_ns: std.atomic.Value(u64) = .init(0),

    fn add(field: *std.atomic.Value(u64), dt: u64) void {
        _ = field.fetchAdd(dt, .monotonic);
    }

    /// Charge `now - start` to the COMPUTE bucket from OUTSIDE the query path — for a
    /// stage's miss-side work that is not itself a cached query (e.g. discovery's
    /// `readFileAlloc`, which is always paid and is the file-read part of the
    /// "file-read+lex+parse" discover compute). Reads the clock once; callers gate the
    /// call on the probe being present so a plain build pays nothing.
    pub fn lapCompute(self: *StageProbe, io: Io, start: i128) void {
        lap(io, &self.compute_ns, start);
    }

    /// Charge `now - start` to the cache-GET bucket from OUTSIDE the query path — for a
    /// cached serve that doesn't go through `Engine.query`/`lex`/`parse` (the warm-discover
    /// fast path's direct `cache.get`s of the source/lex/parse blobs). Mirrors `lapCompute`;
    /// callers gate on the probe being present so a plain build reads no clock.
    pub fn lapGet(self: *StageProbe, io: Io, start: i128) void {
        lap(io, &self.get_ns, start);
    }

    /// The monotonic clock the probe charges against (`Io.Clock`, since Zig 0.16 has no
    /// `std.time.Timer`). Exposed so an out-of-query caller can snapshot a start stamp.
    pub fn now(io: Io) i128 {
        return nowNs(io);
    }
};

/// The original name of `StageProbe`, kept so call sites that wired the lower-only
/// probe keep compiling. The probe is stage-agnostic (three compute/get/put buckets),
/// so the generalized name is `StageProbe`; this alias is the lower stage's view of it.
pub const LowerProbe = StageProbe;

fn nowNs(io: Io) i128 {
    return Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
}

/// Charge `dt = now - start` to `field`, but only when a probe is present. Reading
/// the clock per get/put/compute is cheap relative to the syscalls they bracket, and
/// it is only paid under `--timings` (probe != null), so a plain build is unaffected.
fn lap(io: Io, field: *std.atomic.Value(u64), start: i128) void {
    const dt = nowNs(io) - start;
    if (dt > 0) StageProbe.add(field, @intCast(dt));
}

pub fn init(cache: Cache, mode: Mode) Engine {
    return .{ .cache = cache, .mode = mode };
}

/// Like `init` but also threads a borrowed `--timings` sub-stage probe.
pub fn initProbe(cache: Cache, mode: Mode, probe: ?*LowerProbe) Engine {
    return .{ .cache = cache, .mode = mode, .probe = probe };
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
    // get->compute->put, with the `--timings` probe laps gated on `self.probe != null`
    // so a plain build reads no clock (matching the codegen path's discipline).
    const get_t0: i128 = if (self.probe != null) nowNs(io) else 0;
    const hit: ?[]T = if (swallow_get) (self.cache.get(T, gpa, io, key) catch null) else (try self.cache.get(T, gpa, io, key));
    if (self.probe) |p| lap(io, &p.get_ns, get_t0);
    if (hit) |h| {
        return .{ .value = h, .cached = true };
    }
    const comp_t0: i128 = if (self.probe != null) nowNs(io) else 0;
    const fresh = try compute.run();
    if (self.probe) |p| lap(io, &p.compute_ns, comp_t0);
    const put_t0: i128 = if (self.probe != null) nowNs(io) else 0;
    if (swallow_put) {
        self.cache.put(T, io, key, tmp_tag, fresh) catch {};
    } else {
        try self.cache.put(T, io, key, tmp_tag, fresh);
    }
    if (self.probe) |p| lap(io, &p.put_ns, put_t0);
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
/// failed store just means the next build re-parses).
///
/// B2: the parser now ALWAYS produces a tree, so `tree` is non-optional. A TAINTED
/// parse (any accumulated diagnostic) sets `diags` non-empty; the caller must not
/// proceed to codegen and reports every diagnostic. The cache PUT is gated on
/// `diags.len == 0`, so a poisoned partial tree never becomes a cached "good"
/// parse. `diags` is owned by the caller (freed via `gpa.free`); a cache HIT (a
/// clean parse, never tainted) yields an empty `diags`.
pub const ParseResult = struct {
    tree: Ast.Tree,
    cached: bool = false,
    diags: []const @import("../parse.zig").Diagnostic = &.{},
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
    // `--timings`: the read+unpack-validate is the GET cost; gated on `self.probe`
    // so a plain build reads no clock (the codegen-probe discipline).
    const get_t0: i128 = if (self.probe != null) nowNs(io) else 0;
    const blob: ?[]u8 = if (swallow_get) (cache.get(u8, gpa, io, key) catch null) else (try cache.get(u8, gpa, io, key));
    if (blob) |bytes| {
        defer gpa.free(bytes);
        const unpacked: ?Ast.Tree = if (swallow_get) (Ast.unpack(gpa, bytes) catch null) else (try Ast.unpack(gpa, bytes));
        if (unpacked) |t| {
            if (self.probe) |p| lap(io, &p.get_ns, get_t0);
            return .{ .tree = t, .cached = true };
        }
    }
    if (self.probe) |p| lap(io, &p.get_ns, get_t0);

    // --- miss: parse fresh; store ONLY a clean (untainted) parse ---
    const comp_t0: i128 = if (self.probe != null) nowNs(io) else 0;
    const res = try Parser.parse(gpa, tokens, source);
    if (self.probe) |p| lap(io, &p.compute_ns, comp_t0);
    // Gate the cache PUT on a clean parse: a tainted (diagnostic-bearing) partial
    // tree must never become a cached "good" parse — the next build would serve it
    // as a hit and skip the diagnostics. A clean parse caches as before.
    if (res.diags.len == 0) {
        if (Ast.pack(gpa, res.tree) catch null) |b| {
            defer gpa.free(b);
            const put_t0: i128 = if (self.probe != null) nowNs(io) else 0;
            cache.put(u8, io, key, tmp_tag, b) catch {};
            if (self.probe) |p| lap(io, &p.put_ns, put_t0);
        }
    }
    return .{ .tree = res.tree, .cached = false, .diags = res.diags };
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

/// PER-STAGE chunking heuristics for `chunkedFanOut`, tuned by the work-per-unit of
/// each fan-out site (see PERF P1). `threshold` = stay serial at or below this unit
/// count (chunking would cost more than it saves); `chunks_per_cpu` = how many ranges
/// per cpu (more = finer ranges = better load-balance when unit cost varies, at a few
/// more enqueues). The ratio "per-unit work vs per-task overhead" sets both: the
/// tinier the unit, the LOWER the threshold has to be to ever activate on real
/// programs, and a slightly higher chunk count keeps any one fat range from stalling.
pub const Chunk = struct {
    /// Per-fn body type-check (`types.checkBodies`): the TINIEST unit (~0.05-0.5ms, an
    /// AST walk), so the worst overhead ratio — chunk aggressively. Low threshold so a
    /// few-hundred-fn program already benefits; 8 ranges/cpu to even out fat bodies.
    pub const body = .{ .threshold = 64, .chunks_per_cpu = 8 };
    /// Per-fn codegen lower (`Driver.lowerGraphProgram`): more work/unit than body
    /// (~1-5ms cold; a warm cache hit is ~0.02ms = back to tiny, so still chunk). 8
    /// ranges/cpu because a warm/cold MIX makes per-unit cost highly uneven.
    pub const codegen = .{ .threshold = 64, .chunks_per_cpu = 8 };
    /// Per-fn link copy+call26 patch (`Link.link`): ~0.01-0.1ms/fn, one memcpy + a few
    /// reloc patches. 4 ranges/cpu — units are uniform (no cache mix) so coarse is fine.
    pub const fn_link = .{ .threshold = 128, .chunks_per_cpu = 4 };
    /// Per-fn cstring-reloc rewrite (`emit.rewriteCstr`): ~0.001-0.01ms/fn, by-key map
    /// lookups only. Tiny + uniform; chunk only on big programs, coarse ranges.
    pub const cstr = .{ .threshold = 256, .chunks_per_cpu = 4 };
    /// Per-reloc data-reloc patch (`Link.applyDataRelocs`) and per-page code-sign hash
    /// (`CodeSign`): small unit COUNT (relocs/pages), not 10K, so contention is already
    /// bounded. High threshold leaves them serial except on the largest images.
    pub const small_count = .{ .threshold = 256, .chunks_per_cpu = 2 };
};

/// The host CPU count, the default chunk-count basis when a caller passes no `-j`
/// hint (`ncpu_hint == 0`). It is the runtime knob — NOT a hardcoded constant — and
/// over-providing chunks vs. the real pool size is harmless: surplus ranges queue on
/// the pool and run as workers free up, and at `-j1` (`.limited(0)`) every range
/// takes the inline serial path regardless of count, so byte-identity is unaffected.
pub fn hostCpus() usize {
    return std.Thread.getCpuCount() catch 1;
}

/// CHUNKED parallel fan-out: the PERF P1 primitive that makes `-jN` actually scale.
///
/// `Engine.fanOut` dispatches ONE task per unit; with 10K µs-scale units that is 10K
/// `group.concurrent` enqueues on the main thread plus 10K completion atomics, and the
/// per-task overhead (enqueue lock + worker dequeue + completion signal, all on shared
/// cache lines) swamps the work — so `-jN` ends up SLOWER than `-j1`, worse with more
/// cores. This splits `[0,n)` into ~`ncpu` CONTIGUOUS ranges and dispatches ONE task
/// per range; each range loops its units SERIALLY. Dispatch count drops from `n` to
/// ~`ncpu`, the shared-atomic traffic with it, so contention stops and it scales.
///
/// `threshold`: below it, run every unit inline on this thread (no dispatch) — the
/// verbatim serial path for small programs where chunking would only add overhead.
/// `chunks_per_cpu`: ranges = `min(n, ncpu * chunks_per_cpu)`; >1 gives the pool more,
/// smaller ranges so an uneven unit-cost distribution still load-balances (static
/// ranges, no work-stealing). `ncpu_hint`: the `-j` jobs knob; `0` => `hostCpus()`.
///
/// DETERMINISM ([C11]): ranges own DISJOINT unit spans (range r writes only units
/// `start..end`); there is NO shared mutable state across ranges, and each range calls
/// the SAME `jobFn(ctx.args(i))` the per-unit `fanOut` would, in ascending `i`. The
/// caller still reads its slots back in unit-index order, so the merged result is
/// byte-identical to `-j1` regardless of range count or thread-arrival order. The
/// inline fallback (when `group.concurrent` can't provide concurrency, e.g. `-j1`'s
/// `.limited(0)`) runs the ranges in order on this thread = a plain serial loop over
/// all `n` units, the exact serial baseline.
pub fn chunkedFanOut(
    io: Io,
    n: usize,
    ncpu_hint: usize,
    threshold: usize,
    chunks_per_cpu: usize,
    comptime jobFn: anytype,
    ctx: anytype,
) void {
    if (n == 0) return;
    // Small `n`: the serial path. Chunking here only pays dispatch overhead for work
    // that is already a rounding error, and it keeps tiny programs byte-for-byte on
    // the same code path `-j1` takes.
    if (n <= threshold) {
        var i: usize = 0;
        while (i < n) : (i += 1) @call(.auto, jobFn, ctx.args(i));
        return;
    }

    const ncpu = if (ncpu_hint != 0) ncpu_hint else hostCpus();
    const target_chunks = @max(@as(usize, 1), ncpu * @max(@as(usize, 1), chunks_per_cpu));
    const nchunks = @min(n, target_chunks);

    // One task per CONTIGUOUS range; each runs its units serially. The range job
    // closes over the user's `ctx`/`jobFn` and the `[start,end)` it owns. Args is a
    // 3-tuple `{ctx, start, end}` — ONE enqueue per range, not per unit.
    const RangeJob = struct {
        fn run(c: @TypeOf(ctx), start: usize, end: usize) void {
            var i = start;
            while (i < end) : (i += 1) @call(.auto, jobFn, c.args(i));
        }
    };

    var group: Io.Group = .init;
    // Balanced split: the first `rem` ranges get one extra unit so every unit is
    // covered with at most a 1-unit imbalance (no padding, no leftover tail).
    const base = n / nchunks;
    const rem = n % nchunks;
    var start: usize = 0;
    var c: usize = 0;
    while (c < nchunks) : (c += 1) {
        const len = base + @as(usize, if (c < rem) 1 else 0);
        const end = start + len;
        group.concurrent(io, RangeJob.run, .{ ctx, start, end }) catch RangeJob.run(ctx, start, end);
        start = end;
    }
    group.await(io) catch {};
}

/// A DETERMINISTIC, ORDER-INDEPENDENT fold of a contributor multiset into one u64 —
/// the aggregate identity of a fan-in BARRIER (e.g. "the global tables built from all
/// module parse-result digests"). Each contributor is Wyhash'd, the per-element hashes
/// are SUMMED (commutative => the fold is identical regardless of contributor order or
/// the thread-arrival order that produced them), then a final Wyhash mix breaks the
/// linearity of a raw sum. Summing (not XOR) so duplicate contributors do not cancel.
/// This is the set-fold the two-region scheduler's BARRIER nodes are keyed by; its
/// order-independence is what keeps -j1 and -jN byte-identical across a barrier.
pub fn aggKey(contributors: []const u64) u64 {
    var acc: u64 = 0;
    for (contributors) |c| {
        acc +%= std.hash.Wyhash.hash(0x42_41_52_52, std.mem.asBytes(&c)); // "BARR"
    }
    return std.hash.Wyhash.hash(0x46_4f_4c_44, std.mem.asBytes(&acc)); // "FOLD"
}

/// The result of an `Engine.barrier`: the compute's aggregate `value` plus the
/// order-independent `fold` of the contributors that fed it (the barrier node's id).
pub fn BarrierResult(comptime T: type) type {
    return struct { value: T, fold: u64 };
}

/// The success payload `T` of a barrier `compute`'s `pub fn run(self) error{...}!T`.
/// Used in the `barrier` return type, where `try` is illegal (it is outside function
/// scope there); resolved via the call's type (`@TypeOf` never executes it).
fn ComputePayload(comptime Compute: type) type {
    const ret = @TypeOf(@as(Compute, undefined).run());
    return @typeInfo(ret).error_union.payload;
}

/// The fan-IN seam: a JOIN between two fan-out regions. Folds `contributors` (the
/// upstream region's per-unit result digests) into one order-independent barrier id
/// and runs `compute` ONCE to build the aggregate the downstream region depends on.
/// The aggregate is a build-local value, never an on-disk artifact (barriers are
/// joins, not cacheable units), so nothing is cached.
///
/// This is the seam the StageGraph interpreter calls at each BARRIER transition;
/// `kind` is the stage's `Phase.Kind` (carried for `--timings` attribution).
/// `compute` is `anytype` with a `pub fn run(self) !T` method (it crosses the module
/// boundary into here, so `run` must be `pub` — unlike `query`'s same-file computes).
pub fn barrier(
    self: Engine,
    gpa: std.mem.Allocator,
    kind: Phase.Kind,
    contributors: []const u64,
    compute: anytype,
) !BarrierResult(ComputePayload(@TypeOf(compute))) {
    _ = self;
    _ = gpa;
    _ = kind;
    const fold = aggKey(contributors);
    const value = try compute.run();
    return .{ .value = value, .fold = fold };
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
/// REUSE = CONTENT-FP CACHE HIT, the SOLE driver. A `.normal` build serves the
/// prior blob iff `cache.get(Key.codegen(...))` hits the fn's transitive content
/// fingerprint; the StageGraph schedules, the content cache decides. `.force` skips
/// the cache; `.verify` re-lowers fresh and audits the bytes ([C11]).
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
        // [C11] determinism + cache-soundness gate. ALWAYS re-lower the fn fresh and
        // check its packed FnCode bytes against a reference:
        //   * cache HIT  -> compare against the stored blob (cache soundness).
        //   * cache MISS -> compare against a SECOND fresh lowering (determinism).
        // The checks are REAL runtime comparisons that return an error on mismatch —
        // NOT `std.debug.assert`, which compiles out under ReleaseFast/Small and would
        // make the whole safety net silently no-op in a release build.
        var opt_out: OptOut = .{};
        var fresh = try lowerOne(gpa, frozen, fn_decl, sym, is_entry, my_sig, &opt_out);
        errdefer fresh.deinit(gpa);
        const fb = try Link.pack(gpa, fresh);
        defer gpa.free(fb);

        var was_cached = false;
        if (cache.get(u8, gpa, io, key) catch null) |blob| {
            defer gpa.free(blob);
            if (!std.mem.eql(u8, fb, blob)) return error.VerifyCacheMismatch;
            was_cached = true;
        } else {
            // Cold: no reference blob to compare against, so lower a second time
            // and require the two fresh lowerings agree (pure determinism).
            var fresh2 = try lowerOne(gpa, frozen, fn_decl, sym, is_entry, my_sig, null);
            defer fresh2.deinit(gpa);
            const fb2 = try Link.pack(gpa, fresh2);
            defer gpa.free(fb2);
            if (!std.mem.eql(u8, fb, fb2)) return error.VerifyNondeterministic;
            // Populate the cache so subsequent fns/runs see a primed entry.
            cache.put(u8, io, key, tmp_tag, fb) catch {};
        }
        // A verify build re-lowers fresh, so it has honest opt stats even on a
        // cache hit (was_cached==true). Record them.
        slot.* = .{ .fc = fresh, .cached = was_cached, .opt_stats = opt_out.stats, .ir_instrs = opt_out.ir_instrs };
        return;
    }

    if (mode != .force) {
        const get_t0: i128 = if (self.probe != null) nowNs(io) else 0;
        const got = cache.get(u8, gpa, io, key) catch null;
        if (self.probe) |p| lap(io, &p.get_ns, get_t0);
        if (got) |blob| {
            defer gpa.free(blob);
            if (Link.unpack(gpa, blob) catch null) |fc| {
                slot.* = .{ .fc = fc, .cached = true };
                return;
            }
        }
    }

    var opt_out: OptOut = .{};
    const comp_t0: i128 = if (self.probe != null) nowNs(io) else 0;
    var fc = try lowerOne(gpa, frozen, fn_decl, sym, is_entry, my_sig, &opt_out);
    if (self.probe) |p| lap(io, &p.compute_ns, comp_t0);
    errdefer fc.deinit(gpa);
    if (Link.pack(gpa, fc) catch null) |b| {
        defer gpa.free(b);
        const put_t0: i128 = if (self.probe != null) nowNs(io) else 0;
        cache.put(u8, io, key, tmp_tag, b) catch {};
        if (self.probe) |p| lap(io, &p.put_ns, put_t0);
    }
    slot.* = .{ .fc = fc, .cached = false, .opt_stats = opt_out.stats, .ir_instrs = opt_out.ir_instrs };
}

const testing = std.testing;

test "aggKey is order-independent and does not cancel duplicates" {
    const a = [_]u64{ 0x11, 0x22, 0x33 };
    const a_rev = [_]u64{ 0x33, 0x22, 0x11 };
    const a_shuf = [_]u64{ 0x22, 0x11, 0x33 };
    // The set-fold is identical regardless of contributor order (the property that
    // keeps a barrier byte-identical at -j1 and -jN).
    try testing.expectEqual(aggKey(&a), aggKey(&a_rev));
    try testing.expectEqual(aggKey(&a), aggKey(&a_shuf));

    // A different multiset yields a different fold.
    const b = [_]u64{ 0x11, 0x22, 0x44 };
    try testing.expect(aggKey(&a) != aggKey(&b));

    // Summing (not XOR) so a duplicated contributor does NOT cancel: {x,x} != {} and
    // != {x}.
    const dup = [_]u64{ 0xABCD, 0xABCD };
    const one = [_]u64{0xABCD};
    try testing.expect(aggKey(&dup) != aggKey(&one));
    try testing.expect(aggKey(&dup) != aggKey(&.{}));
}

test "barrier folds contributors and runs compute exactly once" {
    const gpa = testing.allocator;
    const cache: Cache = .{ .dir = "unused-barrier-test" }; // barrier never touches it
    const engine = Engine.init(cache, .normal);

    var runs: usize = 0;
    const Compute = struct {
        runs: *usize,
        pub fn run(c: @This()) !u32 {
            c.runs.* += 1;
            return 7;
        }
    };

    const contributors = [_]u64{ 0xDEAD, 0xBEEF, 0xC0DE };
    const res = try engine.barrier(gpa, .signature, &contributors, Compute{ .runs = &runs });

    // compute ran EXACTLY once; the aggregate flows through unchanged.
    try testing.expectEqual(@as(usize, 1), runs);
    try testing.expectEqual(@as(u32, 7), res.value);
    // The fold matches the standalone set-fold of the same contributors.
    try testing.expectEqual(aggKey(&contributors), res.fold);
}

test "chunkedFanOut covers every unit exactly once, in disjoint ranges (any ncpu)" {
    const gpa = testing.allocator;
    // .limited(0) forces the inline path (every range runs on this thread): the
    // serial baseline. A real pool is exercised by the integration/perf suite; here
    // we assert the partition is total + disjoint regardless of the chunk count.
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(0) });
    defer threaded.deinit();
    const io = threaded.io();

    const Ctx = struct {
        hits: []u32,
        pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(job)) {
            return .{ c.hits, i };
        }
        fn job(hits: []u32, i: usize) void {
            hits[i] += 1;
        }
    };

    // n=1000, threshold=10 so chunking activates; sweep ncpu incl. cases where n is
    // NOT divisible by the chunk count (1000 % 7, % 13, % 64 != 0).
    for ([_]usize{ 1, 2, 3, 4, 7, 8, 13, 64, 1000, 2000 }) |ncpu| {
        const hits = try gpa.alloc(u32, 1000);
        defer gpa.free(hits);
        @memset(hits, 0);
        chunkedFanOut(io, 1000, ncpu, 10, 4, Ctx.job, Ctx{ .hits = hits });
        for (hits) |h| try testing.expectEqual(@as(u32, 1), h);
    }
}

test "chunkedFanOut below threshold runs the serial inline path" {
    const gpa = testing.allocator;
    // .unlimited would still let a tiny n go inline, but pin .limited(0) so the test
    // asserts the threshold branch (not pool behavior): n <= threshold => inline.
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(0) });
    defer threaded.deinit();
    const io = threaded.io();

    var order: std.ArrayListUnmanaged(usize) = .empty;
    defer order.deinit(gpa);
    const Ctx = struct {
        order: *std.ArrayListUnmanaged(usize),
        gpa: std.mem.Allocator,
        pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(job)) {
            return .{ c.order, c.gpa, i };
        }
        fn job(o: *std.ArrayListUnmanaged(usize), g: std.mem.Allocator, i: usize) void {
            o.append(g, i) catch {};
        }
    };
    // n=5, threshold=100 => serial: the inline path visits units in ascending order.
    chunkedFanOut(io, 5, 8, 100, 4, Ctx.job, Ctx{ .order = &order, .gpa = gpa });
    try testing.expectEqual(@as(usize, 5), order.items.len);
    for (order.items, 0..) |v, i| try testing.expectEqual(i, v);
}

test "chunkedFanOut result is identical to fanOut (per-unit slot writes)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(0) });
    defer threaded.deinit();
    const io = threaded.io();

    const Ctx = struct {
        out: []u64,
        pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(job)) {
            return .{ c.out, i };
        }
        fn job(out: []u64, i: usize) void {
            // A deterministic per-unit value written to the unit's OWN slot.
            out[i] = std.hash.Wyhash.hash(0xABCD, std.mem.asBytes(&i));
        }
    };

    const a = try gpa.alloc(u64, 777);
    defer gpa.free(a);
    const b = try gpa.alloc(u64, 777);
    defer gpa.free(b);
    @memset(a, 0);
    @memset(b, 0);

    fanOut(io, 777, Ctx.job, Ctx{ .out = a });
    chunkedFanOut(io, 777, 8, 16, 4, Ctx.job, Ctx{ .out = b });
    try testing.expectEqualSlices(u64, a, b);
}

test "barrier returns the fold for any kind" {
    const gpa = testing.allocator;
    const cache: Cache = .{ .dir = "unused-barrier-test" };
    const engine = Engine.init(cache, .normal);

    const Compute = struct {
        pub fn run(_: @This()) !u32 {
            return 42;
        }
    };
    const contributors = [_]u64{ 1, 2, 3 };
    const res = try engine.barrier(gpa, .layout, &contributors, Compute{});
    try testing.expectEqual(@as(u32, 42), res.value);
    try testing.expectEqual(aggKey(&contributors), res.fold);
}

// Engine boundary tests (hit/miss/force/verify/invalidation + key discrimination)
// are an INTEGRATION suite (temp-dir cache + threaded runtime), so they live in
// `src/tests/query_engine.zig` rather than inline here; `root.zig` pulls them.
