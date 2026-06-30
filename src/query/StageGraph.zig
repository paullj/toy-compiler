//! The compiler pipeline as DATA: an ordered, STATIC list of stages, each tagged
//! with its representative node kind and the SCHEDULING transition it imposes. A
//! thin `interpret` walks the list and drives the engine's two coordination
//! primitives — `Engine.barrier` at a fan-IN join, a driver-supplied REGION
//! otherwise — so the cadence lives in ONE declarative place. This is THE
//! orchestration for every program-producing build: `main.zig`'s `interpretPipeline`
//! drives `-o`, `--dump-dag`, and `--emit ir` through `interpret`, not a
//! hand-sequenced chain. (`--emit lex|parse|check` is the per-file inspection
//! report, a different shape, served by `Driver.run`.)
//!
//! The static graph is purely a SCHEDULING + OBSERVABILITY artifact: it says WHEN
//! to barrier vs run a region, never WHAT the result is. Correctness/cutoff stays the
//! content-fp cache (the codegen region serves a hit or re-lowers); the StageGraph
//! introduces no thread-arrival dependence, so -j1 and -jN stay byte-identical (the
//! barrier set-fold is order-free; a region's per-fn jobs are key-disjoint, read
//! back in index order, and its diagnostics merge in fn-id order + one stable sort).
//!
//! A REGION stage is `driver.region(i)`: the driver runs the stage and owns its own
//! internal scheduling. The codegen region FANS OUT per-fn internally (via
//! `Engine.fanOut`, the one parallelism mechanism) and owns its post-region join (the
//! relink tail). The interpreter owns only the cadence — it runs the stage, so the
//! boundary BETWEEN a region and the next stage is a sequential join (the region fully
//! completes before the next stage starts).
//!
//! A BARRIER stage is a fan-IN keyed by `Engine.barrier`'s order-independent set-fold
//! over its `contributors`. There are THREE, and each folds a real, multi-element
//! contributor multiset on every program build (so the set-fold is load-bearing, not a
//! 1-element stub):
//!   * DISCOVER folds the entry-path digest -> the discovered module graph;
//!   * COLLECT folds every module's parse digest -> the global name-resolution tables
//!     (resolve's whole-graph collect + per-fn bind is the join compute);
//!   * GLOBAL_TABLES folds every global fn's resolve digest -> the program-wide
//!     layout/sig tables (typecheck Pass-A is the join compute; the per-fn body checks
//!     then fan out INSIDE that compute as the PER_UNIT region past the barrier).
//! Each fold is order-independent (sum-then-mix over the contributor digests), so the
//! join id — and the recorded barrier node — is identical at -j1 and -jN. The body and
//! codegen per-fn jobs are key-disjoint and read back in index order, diagnostics merge
//! in fn-id order + one stable sort, and codegen keys are content-fp pure, so the whole
//! pipeline stays byte-identical across thread counts.

const std = @import("std");
const Engine = @import("Engine.zig");
const Dag = @import("Dag.zig");

/// How a stage joins the schedule. A `BARRIER` is a fan-IN: `Engine.barrier` folds
/// the stage's `contributors` into one aggregate id (recorded under `Stage.kind`) and
/// runs the join compute once. A `REGION` stage is `driver.region(i)`: the driver runs
/// the stage with its own internal scheduling (serial, or a per-fn `Engine.fanOut` +
/// post-region join). The boundary between consecutive REGIONs is sequential (region i
/// completes before region i+1 starts) — the hard ordering the two-region design needs.
pub const Transition = enum { BARRIER, REGION };

/// One stage of the static pipeline. For a `BARRIER`, `kind` is the Dag.Kind of the
/// recorded barrier node (load-bearing — `interpret` passes it to `Engine.barrier`).
/// For a `REGION`, `kind` is the representative node kind (documentary: the region
/// records its OWN fine-grained DAG nodes internally via its pass, e.g. typecheck's
/// body/signature/layout nodes; `interpret` does not read `kind` for a REGION).
pub const Stage = struct {
    kind: Dag.Kind,
    transition: Transition,
};

/// The compiler pipeline declared as data — the cadence the interpreter drives. The
/// DISCOVER barrier folds the entry-path digest into the module graph; the COLLECT
/// barrier folds every module's parse digest into the global name-resolution tables
/// (its compute is resolve's whole-graph collect + per-fn bind); the GLOBAL_TABLES
/// barrier folds every global fn's resolve digest into the program-wide layout/sig
/// tables (its compute is typecheck Pass-A, after which the per-fn body checks fan out
/// internally); the codegen region fans out per-fn behind the frozen tables then
/// relinks. The three barriers are the hard joins between fan-out regions; codegen's
/// boundary is the sequential one `interpret` imposes (the region completes before the
/// build's tail runs).
///
/// `kind` names each barrier's recorded node (`Dag.Kind.discover`/`collect`/
/// `global_tables`); for the codegen REGION it is the representative kind (the region
/// records its own fine-grained codegen/body nodes internally for observability). The
/// transitions are what `interpret` schedules on.
pub const pipeline = [_]Stage{
    .{ .kind = .discover, .transition = .BARRIER }, // DISCOVER: entry path -> module graph
    .{ .kind = .collect, .transition = .BARRIER }, // COLLECT: module parse digests -> resolve tables
    .{ .kind = .global_tables, .transition = .BARRIER }, // GLOBAL_TABLES: fn resolve digests -> layout/sig + per-fn body fan-out
    .{ .kind = .codegen, .transition = .REGION }, // codegen: per-fn lower fan-out + relink
};

/// Walk `stages` in order, driving each through the engine's coordination primitive
/// for its transition: `engine.barrier` at a `BARRIER` (a root join — never inside a
/// job), the driver's `region` at a `REGION`. The `driver` supplies the per-stage
/// pieces via COMPTIME-index-keyed hooks (so each stage's compute / region may close
/// over distinct state):
///
///   * BARRIER stage `i`: `driver.contributors(i) []const u64`,
///     `driver.barrierCompute(i)` (a value with `pub fn run(self) !T`), and
///     `driver.recordBarrier(i, Engine.BarrierResult(T))` to receive the aggregate.
///   * REGION stage `i`: `driver.region(i) !void` — runs the region with its own
///     internal scheduling (serial, or an `Engine.fanOut` over key-disjoint units
///     then the post-region join). The region owns parallelism + ordering; the
///     interpreter only sequences it, so a region fully completes before the next.
///
/// `stages` is comptime so the loop unrolls and only the hooks a stage actually uses
/// are referenced — a BARRIER stage never forces the driver to define `region` and
/// vice versa.
pub fn interpret(
    comptime stages: []const Stage,
    engine: Engine,
    gpa: std.mem.Allocator,
    driver: anytype,
) !void {
    inline for (stages, 0..) |stage, i| {
        switch (stage.transition) {
            .BARRIER => {
                const r = try engine.barrier(gpa, stage.kind, driver.contributors(i), driver.barrierCompute(i));
                driver.recordBarrier(i, r);
            },
            .REGION => try driver.region(i),
        }
    }
}

// ===========================================================================
// Unit tests — a MOCK pipeline asserts the interpreter's cadence: a barrier runs
// between two fan-out regions (never inside a job), and each region's jobs are
// dispatched + read back in index order. No engine cache / real compute is touched.
// ===========================================================================

const testing = std.testing;

test "interpret drives a mock pipeline: barrier between fan-out regions, jobs in index order" {
    const gpa = testing.allocator;

    // A 3-stage mock: REGION (3 jobs) -> BARRIER -> REGION (2 jobs). The mock records
    // a flat EVENT LOG of every interpreter action so the test can assert the exact
    // cadence + ordering.
    const mock_stages = [_]Stage{
        .{ .kind = .lex, .transition = .REGION },
        .{ .kind = .signature, .transition = .BARRIER },
        .{ .kind = .codegen, .transition = .REGION },
    };

    const Event = union(enum) {
        job: struct { stage: usize, idx: usize },
        barrier: struct { stage: usize, fold: u64 },
    };

    const Log = struct {
        events: std.ArrayListUnmanaged(Event) = .empty,
        active_in_job: bool = false, // true while a fan-out job is "running"
        saw_barrier_in_job: bool = false,
        fn deinit(l: *@This(), g: std.mem.Allocator) void {
            l.events.deinit(g);
        }
    };

    var log: Log = .{};
    defer log.deinit(gpa);

    // The fan-out job: records (stage, idx). The region reads jobs back in index
    // order; the inline (-j1) fallback runs them in index order on this thread.
    const Job = struct {
        fn run(l: *Log, g: std.mem.Allocator, stage: usize, idx: usize) void {
            l.active_in_job = true;
            l.events.append(g, .{ .job = .{ .stage = stage, .idx = idx } }) catch {};
            l.active_in_job = false;
        }
    };

    const JobCtx = struct {
        log: *Log,
        gpa: std.mem.Allocator,
        stage: usize,
        pub fn args(c: @This(), j: usize) std.meta.ArgsTuple(@TypeOf(Job.run)) {
            return .{ c.log, c.gpa, c.stage, j };
        }
    };

    // The barrier compute: asserts (via the log flag) it is NOT running inside a job,
    // then returns a sentinel so recordBarrier can log the fold.
    const BarrierCompute = struct {
        log: *Log,
        pub fn run(c: @This()) !u32 {
            if (c.log.active_in_job) c.log.saw_barrier_in_job = true;
            return 0;
        }
    };

    const Driver = struct {
        log: *Log,
        gpa: std.mem.Allocator,
        io: std.Io,
        fn contributors(_: @This(), comptime stage_i: usize) []const u64 {
            // stage 1 is the barrier; fold a fixed contributor set.
            comptime std.debug.assert(stage_i == 1);
            return &[_]u64{ 0xA, 0xB };
        }
        fn barrierCompute(self: @This(), comptime stage_i: usize) BarrierCompute {
            comptime std.debug.assert(stage_i == 1);
            return .{ .log = self.log };
        }
        fn recordBarrier(self: @This(), comptime stage_i: usize, r: Engine.BarrierResult(u32)) void {
            self.log.events.append(self.gpa, .{ .barrier = .{ .stage = stage_i, .fold = r.fold } }) catch {};
        }
        // A fan-out region: dispatch this stage's jobs (index-ordered) via the one
        // parallelism mechanism, exactly as the real fan-out regions do.
        fn region(self: @This(), comptime stage_i: usize) !void {
            const n: usize = switch (stage_i) {
                0 => 3,
                2 => 2,
                else => unreachable,
            };
            Engine.fanOut(self.io, n, Job.run, JobCtx{ .log = self.log, .gpa = self.gpa, .stage = stage_i });
        }
    };

    // -j1 (.limited(0)) forces the inline fan-out path: jobs run in index order on
    // this thread, the true serial baseline the determinism contract rests on.
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(0) });
    defer threaded.deinit();
    const io = threaded.io();

    const cache: @import("Cache.zig") = .{ .dir = "unused-stagegraph-test" };
    const engine = Engine.init(cache, .normal);

    try interpret(&mock_stages, engine, gpa, Driver{ .log = &log, .gpa = gpa, .io = io });

    // The barrier compute NEVER ran inside a fan-out job.
    try testing.expect(!log.saw_barrier_in_job);

    // Event log: 3 jobs (stage 0, idx 0..2) in order, then the barrier (stage 1),
    // then 2 jobs (stage 2, idx 0..1) in order.
    try testing.expectEqual(@as(usize, 6), log.events.items.len);
    try testing.expectEqual(@as(usize, 0), log.events.items[0].job.stage);
    try testing.expectEqual(@as(usize, 0), log.events.items[0].job.idx);
    try testing.expectEqual(@as(usize, 1), log.events.items[1].job.idx);
    try testing.expectEqual(@as(usize, 2), log.events.items[2].job.idx);
    // The barrier fires AFTER region-1's jobs and BEFORE region-2's.
    try testing.expectEqual(@as(usize, 1), log.events.items[3].barrier.stage);
    try testing.expectEqual(Engine.aggKey(&[_]u64{ 0xA, 0xB }), log.events.items[3].barrier.fold);
    try testing.expectEqual(@as(usize, 2), log.events.items[4].job.stage);
    try testing.expectEqual(@as(usize, 0), log.events.items[4].job.idx);
    try testing.expectEqual(@as(usize, 1), log.events.items[5].job.idx);
}

test "the static pipeline's shape matches the documented cadence" {
    // THREE fan-IN barriers lead — DISCOVER (entry path), COLLECT (module parse
    // digests -> resolve tables), GLOBAL_TABLES (fn resolve digests -> layout/sig +
    // the per-fn body fan-out INSIDE its compute) — then codegen as the one REGION.
    // Each barrier folds a real multi-element contributor multiset on a program build,
    // so `Engine.aggKey` is load-bearing, not a 1-element stub.
    try testing.expectEqual(@as(usize, 4), pipeline.len);

    // DISCOVER -> COLLECT -> GLOBAL_TABLES are barriers, in this order.
    try testing.expectEqual(Transition.BARRIER, pipeline[0].transition);
    try testing.expectEqual(Dag.Kind.discover, pipeline[0].kind);
    try testing.expectEqual(Transition.BARRIER, pipeline[1].transition);
    try testing.expectEqual(Dag.Kind.collect, pipeline[1].kind);
    try testing.expectEqual(Transition.BARRIER, pipeline[2].transition);
    try testing.expectEqual(Dag.Kind.global_tables, pipeline[2].kind);

    var n_barriers: usize = 0;
    for (pipeline) |s| if (s.transition == .BARRIER) {
        n_barriers += 1;
    };
    try testing.expectEqual(@as(usize, 3), n_barriers);

    // codegen is the sole REGION (its per-fn lower fan-out owns its relink join).
    try testing.expectEqual(Transition.REGION, pipeline[3].transition);
    try testing.expectEqual(Dag.Kind.codegen, pipeline[3].kind);
}
