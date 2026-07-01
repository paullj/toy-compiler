//! The build-pipeline orchestrator: the single stage adapter `StageGraph.interpret`
//! drives for every program-producing build (`-o`, `--emit ir`). It supplies each
//! stage's compute (discover / resolve / typecheck / codegen), folds the barrier
//! contributor multisets, writes each stage's result into a caller-frame slot, and
//! laps the `--timings` buckets. The three barrier-compute structs + the per-fn
//! resolve digest live here too, as the unit that owns the stage-dispatch coupling.
const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");
const Ast = toyc.Ast;
const Engine = toyc.QueryEngine;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Codegen = toyc.DriverCodegen;
const Opt = toyc.Opt;

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
pub const Orchestrator = struct {
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
        if (!self.timings) return;
        if (bucket) |b| if (self.last_ns) |ln| {
            const n = Io.Clock.Timestamp.now(self.io, .awake).raw.nanoseconds;
            const dt = n - ln.*;
            ln.* = n;
            b.* = if (dt > 0) @intCast(dt) else 0;
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
    std.mem.writeInt(u32, b[4..8], gf.decl_node.int(), .little);
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
