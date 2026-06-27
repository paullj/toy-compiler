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
const builtin = @import("builtin");
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
const Dag = @import("Dag.zig");

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

/// M16 SPIKE — the OBSERVATIONAL dependency-recording sink. When non-null,
/// `query()` records a caller->callee edge + the callee's result fingerprint into
/// this shared `*Dag` (the active-query-stack supplies the caller). BORROWED, never
/// a value field: the Engine is re-`init`'d per job from a copied cache+mode, so a
/// value DAG would give each job a private empty graph and the edges would vanish —
/// a single Dag lives in the driver scope and a `*Dag` is threaded into every job
/// (mirrors how `Cache` is itself a borrowed dir handle).
///
/// When `null` (the default `init`), `query()` is VERBATIM-today: zero overhead,
/// zero behavior change — the byte-identity guarantee for this milestone.
dag: ?*Dag = null,

pub fn init(cache: Cache, mode: Mode) Engine {
    return .{ .cache = cache, .mode = mode };
}

/// Same as `init`, but threads a borrowed `*Dag` so `query()` records dependency
/// edges. Existing call sites keep using `init` (dag = null) so they compile +
/// behave unchanged; only the driver scope that owns the per-build `Dag` opts in.
pub fn initDag(cache: Cache, mode: Mode, dag: *Dag) Engine {
    return .{ .cache = cache, .mode = mode, .dag = dag };
}

/// Map a `Cache.Key` to its in-memory DAG `NodeKey`. lex/parse/codegen fold the
/// Cache.Key digest so DAG nodes align 1:1 with cache identity. (The fine-grained
/// typecheck nodes — signature/body/type_of/layout — are recorded directly via
/// `dag.recordEdge` with their global ids in later stages; this spike wires the
/// engine seam + proves the signature firewall in tests.)
fn nodeKeyFor(key: Cache.Key) Dag.NodeKey {
    const kind: Dag.Kind = switch (key.phase) {
        .lex => .lex,
        .parse => .parse,
        .check => .body, // coarse: the M15 single check phase (no fine-grained split yet here)
        .codegen => .codegen,
    };
    return .{ .kind = kind, .id = key.digest() };
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
    // FAST PATH: no DAG sink => VERBATIM-today (zero overhead, byte-identical).
    if (self.dag == null) {
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

    // OBSERVATIONAL PATH: same get->compute->put, but the active-query-stack
    // records caller->this edge + this node's result fp. The C call stack IS the
    // dependency stack: read the current active node as our PARENT, enter ourselves
    // (so nested query() calls in `compute` see us as their parent), restore on
    // exit. recordEdge runs UNCONDITIONALLY after the result is obtained — on a HIT
    // and on a fresh compute alike — so the DAG is identical hit-vs-miss. [C11]
    const d = self.dag.?;
    const node = nodeKeyFor(key);
    const parent = Dag.Active.get();
    const prev = Dag.Active.enter(node);
    defer Dag.Active.leave(prev);

    const hit: ?[]T = if (swallow_get) (self.cache.get(T, gpa, io, key) catch null) else (try self.cache.get(T, gpa, io, key));
    if (hit) |h| {
        d.recordEdge(gpa, parent, node, fpOfBytes(T, h));
        return .{ .value = h, .cached = true };
    }
    const fresh = try compute.run();
    if (swallow_put) {
        self.cache.put(T, io, key, tmp_tag, fresh) catch {};
    } else {
        try self.cache.put(T, io, key, tmp_tag, fresh);
    }
    d.recordEdge(gpa, parent, node, fpOfBytes(T, fresh));
    return .{ .value = fresh, .cached = false };
}

/// A deterministic content fingerprint of a query RESULT slice for the recorded
/// DAG. Observational only — it lets the dump show a stable per-node fp that flips
/// iff the result bytes change. (The codegen path's authoritative content
/// fingerprint is the `Fingerprint`-derived cache key; this engine-level fold is a
/// uniform stand-in across result types `T` for the spike's edge recording.)
fn fpOfBytes(comptime T: type, items: []const T) u64 {
    return std.hash.Wyhash.hash(0x44_41_47_46, std.mem.sliceAsBytes(items)); // "DAGF"
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

    // OBSERVATIONAL DAG: parse cannot use the generic `query()` (its unpack
    // validation needs a custom hit path), so it records its node/edge directly.
    // `enter` makes this parse the Active parent for any nested query in `compute`
    // (lex during discovery runs separately, but this keeps the seam uniform); the
    // edge + fp are recorded UNCONDITIONALLY after the result is obtained. No-op +
    // zero-overhead when dag == null.
    const node: Dag.NodeKey = .{ .kind = .parse, .id = key.digest() };
    const dag_parent: ?Dag.NodeKey = if (self.dag != null) Dag.Active.get() else null;
    const prev: ?Dag.NodeKey = if (self.dag != null) Dag.Active.enter(node) else null;
    defer if (self.dag != null) Dag.Active.leave(prev);

    // --- cache read + validate (a corrupt blob is treated as a miss) ---
    const blob: ?[]u8 = if (swallow_get) (cache.get(u8, gpa, io, key) catch null) else (try cache.get(u8, gpa, io, key));
    if (blob) |bytes| {
        defer gpa.free(bytes);
        const unpacked: ?Ast.Tree = if (swallow_get) (Ast.unpack(gpa, bytes) catch null) else (try Ast.unpack(gpa, bytes));
        if (unpacked) |t| {
            if (self.dag) |d| d.recordEdge(gpa, dag_parent, node, Ast.contentFp(t));
            return .{ .tree = t, .cached = true };
        }
    }

    // --- miss: parse fresh, store on success ---
    var diag: ?Parser.Diagnostic = null;
    if (try Parser.parse(gpa, tokens, source, &diag)) |t| {
        if (Ast.pack(gpa, t) catch null) |b| {
            defer gpa.free(b);
            cache.put(u8, io, key, tmp_tag, b) catch {};
            if (self.dag) |d| d.recordEdge(gpa, dag_parent, node, Ast.contentFp(t));
        } else if (self.dag) |d| d.recordEdge(gpa, dag_parent, node, 0);
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

/// The per-fn red-green REUSE decision the driver computes single-threaded BEFORE
/// the fan-out (the walk is not thread-safe) and hands to `codegen`. `green` means
/// the codegen node + its whole transitive dep closure verified unchanged vs the
/// prior build, so the prior cached `FnCode` may be served WITHOUT re-deriving the
/// transitive content fp and WITHOUT the body walk — that skip is the perf win.
///
/// `green=false` (the default) is the VERBATIM path: recompute exactly as today.
/// A green decision is only trusted when an oracle is actually present; a green
/// node whose on-disk cache blob was evicted falls back to a fresh recompute (a
/// cache miss on a green node is treated as red — never an error). The recorded
/// `stamp` is the prior node fp so the green-reused node round-trips an IDENTICAL
/// fresh DAG (hit == miss recording).
pub const Reuse = struct {
    green: bool = false,
    stamp: u64 = 0,
};

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
    gid: u32,
    reuse: ?Reuse,
    slot: anytype,
) !void {
    const cache = self.cache;
    const mode = self.mode;

    // The STABLE codegen node identity (target+opt+global fn id), distinct from the
    // content-fp-derived cache key below. This names the in-memory DAG node the
    // red-green walk decides reuse for, so a node's identity is NO LONGER its content
    // hash. Used for BOTH the recorded node and (matching) `nodeKeyFor` alignment.
    const stable_id = Key.codegenIdentity(target, frozen.opt, gid, is_entry);

    // In a DEBUG build, AUDIT every green verdict instead of trusting the stamp blind
    // (rustc's -Zincremental-verify-ich model): skip the fast-path shortcut, recompute
    // the real content fp below, and require it equals the oracle's stamp — a mismatch
    // is a dependency UNDER-RECORDING bug (a false green that would miscompile). The
    // recompute is the fp WALK, not a re-lower, and the unchanged fn still cuts off via
    // the content-fp cache below, so Debug stays incremental while self-checking. A
    // RELEASE build keeps the full skip (the actual perf win).
    const debug_audit = builtin.mode == .Debug;

    // GREEN FAST-PATH: the driver's red-green oracle verified this fn's codegen node
    // + its whole transitive dep closure unchanged vs the prior build. Serve the
    // prior cached blob WITHOUT re-deriving the transitive fp or the body walk (the
    // perf win), and STILL record the node + body edge with the prior stamp so the
    // fresh DAG round-trips identically (hit == miss recording). A cache miss on a
    // green node falls back to the normal recompute below (never an error). Gated on
    // an actual oracle + `.normal` mode, so dag==null / force / verify stay verbatim;
    // skipped under `debug_audit` so the verdict is cross-checked, not trusted.
    if (mode == .normal and !debug_audit) if (reuse) |rd| if (rd.green) {
        if (try self.greenReuse(gpa, io, target, stable_id, gid, rd, slot)) return;
    };

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

    // DEBUG green-verdict audit: the oracle marked this fn green, meaning its content
    // is unchanged — so the freshly recomputed key input MUST equal the reused stamp.
    // A mismatch means the dependency closure UNDER-RECORDED and the green path would
    // have served a stale blob = miscompile. Catch it in Debug/test builds; the sound
    // content-fp cache lookup below still serves the (genuinely unchanged) blob.
    if (debug_audit and mode == .normal) if (reuse) |rd| if (rd.green and rd.stamp != key.input) return error.VerifyFalseGreen;

    if (self.dag) |d| recordCodegenNode(d, gpa, stable_id, gid, key.input);

    if (mode == .verify) {
        // [C11] determinism + cache-soundness + red-green-soundness gate. ALWAYS
        // re-lower the fn fresh and check its packed FnCode bytes against a reference:
        //   * cache HIT  -> compare against the stored blob (cache soundness).
        //   * cache MISS -> compare against a SECOND fresh lowering (determinism).
        // The checks are REAL runtime comparisons that return an error on mismatch —
        // NOT `std.debug.assert`, which compiles out under ReleaseFast/Small and would
        // make the whole safety net silently no-op in a release build.
        //
        // RED-GREEN cross-check: if the driver's oracle marked this fn GREEN, its
        // trusted stamp MUST equal the freshly recomputed key input (the content fp the
        // green path would have reused blind). A mismatch means the dependency closure
        // UNDER-RECORDED — a false green that would miscompile on the normal path. We
        // recompute the real fp here (the body walk the green path skips) precisely to
        // catch that. `--verify` never takes the green skip (the fast-path is gated on
        // `.normal`), so this is the one mode that re-derives + audits the verdict.
        if (reuse) |rd| if (rd.green and rd.stamp != key.input) return error.VerifyFalseGreen;

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

/// Record the codegen DAG node under a STABLE identity plus its complete dependency
/// CLOSURE. The ONLY edge the codegen root records is `codegen(gid) -> body(gid)`:
/// the body subtree the typecheck pass recorded (body -> signature(callee) / layout
/// / type_of / resolve_name) carries every input that affects the emitted bytes, so
/// the codegen root reaches all of them transitively through this one edge. The sig
/// firewall is preserved — a caller's codegen reaches a callee's SIGNATURE (not its
/// body) via codegen(caller) -> body(caller) -> signature(callee). WITHOUT this edge
/// a body/layout edit recorded under body(gid) never reaches the codegen root and the
/// node verifies green off its own stamp alone = MISCOMPILE.
///
/// `gid` is the SAME global fn id the typecheck pass keyed body(fid) under (checkFn
/// enters `body` with .id = fid). `stamp` is the WIDENED node fp (fp ^ optMix ^
/// symMix, target-sensitive via the key) so a green verdict flips iff the emitted
/// bytes would change. Recording the SAME single edge on every path (normal / hit /
/// green-reuse) keeps the fresh DAG byte-identical regardless of how the fn was
/// served (the HIT == MISS invariant the persisted snapshot relies on). [C11]
fn recordCodegenNode(d: *Dag, gpa: std.mem.Allocator, stable_id: u64, gid: u32, stamp: u64) void {
    const node: Dag.NodeKey = .{ .kind = .codegen, .id = stable_id };
    const parent = Dag.Active.get();
    d.recordEdge(gpa, parent, node, stamp);
    const body_node: Dag.NodeKey = .{ .kind = .body, .id = gid };
    d.recordEdge(gpa, node, body_node, d.fingerprintOf(body_node) orelse 0);
}

/// The GREEN fast-path body: serve the prior cached `FnCode` by reconstructing the
/// content-fp cache key FROM THE STAMP (no body walk, no `Fingerprint.fingerprint`).
/// The stamp the prior build recorded for a codegen node IS `fp ^ optMix ^ symMix`,
/// which is exactly `Key.codegen(...).input`, so `Key.fromFingerprint(.codegen,
/// target, stamp)` rebuilds the cache key without re-deriving the transitive fp —
/// THAT skip is the perf win. Returns true if the prior blob was found and reused;
/// false (cache miss / unpack failure) tells the caller to fall back to a full
/// recompute (a cache miss on a green node is RED, never an error). On reuse it
/// records the codegen node + body edge with the prior stamp so the fresh DAG is
/// byte-identical to a recomputed one.
fn greenReuse(
    self: Engine,
    gpa: std.mem.Allocator,
    io: Io,
    target: []const u8,
    stable_id: u64,
    gid: u32,
    rd: Reuse,
    slot: anytype,
) !bool {
    const key = Key.Key.fromFingerprint(.codegen, target, rd.stamp);
    const blob = (self.cache.get(u8, gpa, io, key) catch null) orelse return false;
    defer gpa.free(blob);
    const fc = (Link.unpack(gpa, blob) catch null) orelse return false;

    // Record the IDENTICAL codegen root + body edge a recomputed node would, so the
    // fresh persisted DAG round-trips byte-identically whether this fn was reused or
    // recomputed (HIT == MISS == green-reuse). The body subtree itself was already
    // recorded by the typecheck pass that ran before the fan-out.
    if (self.dag) |d| recordCodegenNode(d, gpa, stable_id, gid, rd.stamp);
    slot.* = .{ .fc = fc, .cached = true };
    return true;
}

const testing = std.testing;

test "green-reuse cache key is reconstructible from the recorded stamp alone" {
    // The load-bearing equation for the green fast-path: a codegen node's recorded
    // STAMP is `fp ^ optMix ^ symMix`, which is EXACTLY `Key.codegen(...).input`. So
    // `Key.fromFingerprint(.codegen, target, stamp)` rebuilds the on-disk cache key
    // WITHOUT re-deriving the transitive content fp (the body walk we skip on green).
    // If this drifts, the green path would look up the wrong cache slot.
    const target = "aarch64-macos";
    const sym: Link.SymName = .{ .kind = .user_fn, .name = "m.add" };
    const opt: Opt.Config = .{ .fold = true };
    const fp: u64 = 0xDEAD_BEEF_CAFE_F00D;

    const key = Key.codegen(target, fp, opt, sym);
    const stamp = fp ^ Key.optMix(opt) ^ Key.symMix(sym);
    try testing.expectEqual(key.input, stamp);

    const rebuilt = Key.Key.fromFingerprint(.codegen, target, stamp);
    try testing.expectEqual(key.digest(), rebuilt.digest());
}

test "Reuse defaults to a non-green (verbatim) decision" {
    // A zero-valued Reuse must be the safe recompute path, so an unset slot never
    // triggers a green reuse.
    const rd: Reuse = .{};
    try testing.expect(!rd.green);
}

// Engine boundary tests (hit/miss/force/verify/invalidation + key discrimination)
// are an INTEGRATION suite (temp-dir cache + threaded runtime), so they live in
// `src/tests/query_engine.zig` rather than inline here; `root.zig` pulls them.
