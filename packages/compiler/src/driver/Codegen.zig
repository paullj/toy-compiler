//! Whole-graph codegen + link tail: turns a resolved+typechecked module graph into
//! a linked Mach-O image (or rendered IR text). One `Plan` lists every unit to lower; the
//! per-unit fan-out lowers each against its single-unit `Engine.Frozen` view (thread-safe,
//! program-wide tables + its owner's arrays), and `--emit ir` lowers the same units through
//! the same path. Then the SERIAL relink tail interns strings,
//! rebases cross-segment relocs, links, and signs — the order-sensitive
//! assemble → applyDataRelocs → sign-last pipeline. Distinct from the per-file
//! front-end (`Driver.run`/`pipeline`, discover → lex/parse/check).
const std = @import("std");
const Span = @import("../diagnostics/model.zig").Span;
const Io = std.Io;
const Ast = @import("../ast/Ast.zig");
const Cache = @import("../query/Cache.zig");
const Engine = @import("../query/Engine.zig");
const Typecheck = @import("../types.zig");
const Mono = @import("../symbols/Mono.zig");
const Graph = @import("Graph.zig");
const ResolveGraph = @import("../resolve_graph.zig");
const CodegenIr = @import("../codegen/CodegenIr.zig");
const Ir = @import("../ir/Ir.zig");
const Opt = @import("../opt/Opt.zig");
const lower = @import("../lower.zig");
const Closure = @import("../symbols/Closure.zig");
const Spec = @import("../symbols/Spec.zig");
const AstWalk = @import("../query/AstWalk.zig");
const Walk = @import("../ast/Walk.zig");
const Fingerprint = @import("../query/Fingerprint.zig");
const Link = @import("../link/Link.zig");
const link = @import("../link/emit.zig");
const Abi = @import("../codegen/abi/Abi.zig");
const Derive = @import("../symbols/Derive.zig");
const Body = @import("../symbols/Body.zig");

/// The mangled name of the erased `(ty, kind)` witness in `tc.erased_units` — a linear
/// scan matching `(Type.eql, kind)`. Borrowed (the plan holds it by reference); `null`
/// only if the unit was not synthesized (a non-Hashable type never asks for one).
fn erasedNameFor(tc: *const Typecheck.GraphResult, ty: Typecheck.Type, kind: Derive.ErasedKind) ?[]const u8 {
    for (tc.erased_units) |u| {
        if (u.kind == kind and Typecheck.Type.eql(u.ty, ty)) return u.name.?;
    }
    return null;
}

/// The mangled trace unit a descriptor's `trace_off` points at, or `null` when `ty` needs
/// no tracing (a scalar / managed-free aggregate). The collector calls it as
/// `(text_base + trace_off)(cell_addr)` to mark the managed heap object a `ty`-typed cell
/// references. Only `str`/`Ref`/`gc_array` primitives carry one today (their erased trace
/// unit); a managed-free type resolves to `null` → `trace_off` 0 → the container walk
/// treats its cells as leaves.
fn traceNameFor(tc: *const Typecheck.GraphResult, ty: Typecheck.Type) ?[]const u8 {
    return erasedNameFor(tc, ty, .trace);
}

/// A user-facing failure while emitting code: a message plus the source span to
/// render (or `null` for whole-file errors). The driver returns these to `main`,
/// which renders them and exits non-zero.
pub const EmitError = struct {
    message: []const u8,
    span: ?Span,
    /// Graph builds: the owning module id (index into `Graph.modules`) whose
    /// source `span` points into. `null` (single-file path, or a
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
    /// Interned `__cstring` bytes. Owned; empty for string-free programs.
    cstrings: []u8 = &.{},
    /// Cross-segment relocs (adrp/add/ldr to __cstring/__got) rebased to absolute
    /// __text offsets, patched by `Link.applyDataRelocs` after MachO assigns
    /// vmaddrs. Owned; empty for reloc-free programs. The dyld import set is DERIVED
    /// from these (`.import` targets) in `assembleAndSign`.
    data_relocs: []Link.Reloc = &.{},
    /// Incremental counters: how many functions were freshly lowered vs served
    /// from the codegen cache this build. Surfaced via `--codegen-stats`.
    codegen_compiled: usize = 0,
    codegen_cached: usize = 0,
    /// dual-metric: summed opt counters over freshly-lowered fns, the total
    /// IR instruction count after opt, and the emitted aarch64 instruction count
    /// (text.len/4). No owned slices → deinit unchanged. Cached fns contribute 0
    /// to opt_stats/ir_instrs (their opt ran on a prior build), so honest
    /// `--opt-stats` numbers require `--no-cache`.
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
        for (self.data_relocs) |rl| if (rl.target.name()) |nm| gpa.free(nm);
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

/// A lowering diagnostic as the CLI renders it. One with no source position of its own (a
/// frame that outgrew codegen) points at the failing unit's declaration, so the user learns
/// which function to split; a unit with no declaration renders without a snippet.
fn emitErrorOf(graph: *const Graph.Graph, d: CodegenIr.Diagnostic, module: u32, decl: Ast.Index) EmitError {
    if (d.end != 0) return .{ .message = d.message, .span = d.span(), .module = module };
    const m = &graph.modules[module];
    if (decl == Ast.none or decl.int() >= m.nodes.len) return .{ .message = d.message, .span = null, .module = module };
    const tok = m.tokens[m.nodes[decl.int()].main_token];
    return .{ .message = d.message, .span = .{ .start = tok.start, .end = tok.end }, .module = module };
}

/// What `renderGraphIr` produced: either the rendered IR text (caller frees)
/// or one `EmitError`.
pub const IrResult = union(enum) {
    ok: []u8,
    err: EmitError,
};

/// The SERIAL relink tail (every build, uncached): append referenced hand-emitted
/// builtin bodies, intern strings program-wide (deterministic: fn source order
/// then in-fn literal order), rewrite `.cstr` hashes to offsets, then link and
/// rebase cross-segment relocs. `slots` is consumed (each FnCode freed).
fn relink(
    io: Io,
    gpa: std.mem.Allocator,
    slots: []Engine.Slot,
    names: []const Link.SymName,
    entry_fn: u32,
    compiled: usize,
    cached_n: usize,
    opt_stats: Opt.Stats,
    ir_instrs: usize,
    desc_plan: []const Link.DescEntry,
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

    var lp = linkAndTail(io, gpa, fns.items, names, entry_fn, desc_plan) catch |e| switch (e) {
        error.CallTargetTooFar => return .{ .err = .{ .message = "call target out of range", .span = null } },
        error.UnresolvedSymbol, error.NoEntry => return .{ .err = .{ .message = "internal: unresolved symbol after codegen", .span = null } },
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

/// Shared back half of both paths: append any referenced hand-emitted builtin bodies,
/// intern strings deterministically, rewrite `.cstr` targets, then `Link.link`. CONSUMES
/// `fns` (frees each FnCode and any appended builtin body). Returns a LinkedProgram
/// with `diags`/`owned_msgs` left empty for the caller to fill.
fn linkAndTail(io: Io, gpa: std.mem.Allocator, fns: []Link.FnCode, names: []const Link.SymName, entry_fn: u32, desc_plan: []const Link.DescEntry) !LinkedProgram {
    const lk = try link.linkProgram(io, gpa, fns, names[entry_fn], desc_plan);
    return LinkedProgram{
        .text = lk.text,
        .entry_off = lk.entry_off,
        .diags = &.{},
        .owned_msgs = &.{},
        .cstrings = lk.cstrings,
        .data_relocs = lk.data_relocs,
    };
}

/// One lowered unit, by class and its index in that class's table. Its position in the
/// plan is its codegen slot, its link order and its `--emit ir` order; a body's unit is
/// the body's own index, so the bodies lead the plan.
const Unit = union(enum) {
    /// A body written in source (`Body`): a non-generic fn, or a monomorphized instance.
    body: Body.Index,
    /// A source-less auto-derive recipe.
    derive: u32,
    /// An erased hash/eq/trace witness.
    erased: u32,
    /// A lambda or local fn of a body, in `(owner, node)` order.
    lambda: u32,
    /// A fn-value thunk, in `(kind, key)` order.
    adapter: u32,
    /// A specialized copy of an instance, in name order.
    spec: u32,
};

/// How `Plan.init` lowers: the opt passes, whether generic calls with a known fn argument
/// get spec units, and frame storage (see `frameStorage`).
pub const Options = struct { opt: Opt.Config, specialize: bool = true, frame: bool };

/// The whole-graph codegen plan: every unit in canonical order, and the program-wide
/// tables each unit's `Engine.Frozen` view is cut from. Read-only once built, so the
/// parallel jobs share it.
const Plan = struct {
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const Typecheck.GraphResult,
    /// Each module's tree, tokens and source, by module id. Owned.
    modules: []Walk.Source,
    /// One SymName per global fn id, parallel to `tc.sigs`. Owned.
    names: []Link.SymName,
    /// Owned.
    units: []Unit,
    /// Owned.
    specs: Spec.Table,
    /// Per body, its closure facts; empty (no entries) when the program has no lambda unit
    /// and no boxed local. Owned.
    closure_facts: []const []const u8,
    /// The unit of the entry `main`: the bare `main` with a body in the entry module.
    entry: ?u32,
    o: Options,

    fn init(gpa: std.mem.Allocator, graph: *const Graph.Graph, res: *const ResolveGraph.GraphResult, tc: *const Typecheck.GraphResult, o: Options) !Plan {
        const modules = try gpa.alloc(Walk.Source, graph.modules.len);
        errdefer gpa.free(modules);
        for (graph.modules, modules) |*m, *ms| ms.* = .{ .tree = m.tree(), .tokens = m.tokens, .source = m.source };
        const names = try buildGraphNames(gpa, res.fns, tc.sigs);
        errdefer freeNames(gpa, names);

        const bodies = tc.bodies.items;
        var units: std.ArrayList(Unit) = .empty;
        errdefer units.deinit(gpa);
        var entry: ?u32 = null;
        for (bodies, 0..) |b, i| {
            if (b.of == .base) {
                const f = res.fns[b.of.base];
                if (f.module == graph.entry_index and std.mem.eql(u8, f.name, "main")) entry = @intCast(i);
            }
            try units.append(gpa, .{ .body = @intCast(i) });
        }
        for (0..tc.derives.len) |k| try units.append(gpa, .{ .derive = @intCast(k) });
        for (0..tc.erased_units.len) |k| try units.append(gpa, .{ .erased = @intCast(k) });
        for (0..tc.lambda_units.len) |k| try units.append(gpa, .{ .lambda = @intCast(k) });
        for (0..tc.adapters.len) |k| try units.append(gpa, .{ .adapter = @intCast(k) });

        const specs = if (o.specialize) try Spec.build(gpa, .{
            .modules = modules,
            .resolutions = res.resolutions,
            .closures = res.closures,
            .bodies = tc.bodies,
            .prog = .{ .sigs = tc.sigs, .methods = tc.methods, .instances = tc.instances, .names = names, .layouts = tc.layouts },
            .lambda_units = tc.lambda_units,
        }) else Spec.Table{};
        errdefer specs.deinit(gpa);
        for (0..specs.units.len) |k| try units.append(gpa, .{ .spec = @intCast(k) });

        var p: Plan = .{
            .graph = graph,
            .res = res,
            .tc = tc,
            .modules = modules,
            .names = names,
            .units = try units.toOwnedSlice(gpa),
            .specs = specs,
            .closure_facts = &.{},
            .entry = entry,
            .o = o,
        };
        errdefer gpa.free(p.units);
        p.closure_facts = try p.ownerClosureFacts(gpa);
        return p;
    }

    fn deinit(p: *Plan, gpa: std.mem.Allocator) void {
        freeClosureFacts(gpa, p.closure_facts);
        p.specs.deinit(gpa);
        gpa.free(p.units);
        freeNames(gpa, p.names);
        gpa.free(p.modules);
        p.* = undefined;
    }

    /// The body whose tables unit `u` reads: its own, or for a lambda unit its owner and
    /// for a spec unit its instance. Null for a source-less unit, whose emitter is
    /// table-driven.
    fn ownerOf(p: *const Plan, u: Unit) ?Closure.Owner {
        return switch (u) {
            .body => |i| i,
            .lambda => |k| p.tc.lambda_units[k].owner,
            .spec => |k| p.tc.bodies.ofInstance(p.specs.units[k].inst),
            .derive, .erased, .adapter => null,
        };
    }

    /// The module whose tree unit `u` reads, or which a source-less unit is charged to.
    fn moduleOf(p: *const Plan, u: Unit) u32 {
        const tc = p.tc;
        return switch (u) {
            .body, .lambda, .spec => tc.bodies.items[p.ownerOf(u).?].mod,
            .derive => |k| tc.derives[k].mod,
            .erased => |k| tc.erased_units[k].mod,
            // An adapter reads no AST; module 0 only gives its view a tree to carry.
            .adapter => 0,
        };
    }

    /// The declaration a diagnostic with no position of its own blames: the fn a body is
    /// written in (a lambda unit's owner), none for a source-less unit.
    fn declOf(p: *const Plan, u: Unit) Ast.Index {
        const o = p.ownerOf(u) orelse return Ast.none;
        return p.tc.bodies.items[o].decl;
    }

    /// The symbol unit `u` is emitted and linked under. Borrowed from the plan's tables.
    fn symOf(p: *const Plan, u: Unit) Link.SymName {
        const tc = p.tc;
        return .{ .kind = .user_fn, .name = switch (u) {
            .body => |i| tc.bodies.items[i].sig.name,
            .derive => |k| tc.derives[k].name.?,
            .erased => |k| tc.erased_units[k].name.?,
            .lambda => |k| tc.lambda_units[k].name,
            .adapter => |k| tc.adapters[k].name,
            .spec => |k| p.specs.units[k].name,
        } };
    }

    /// The single-unit view job `i` lowers against: its owner body's tables (an instance's
    /// OWN `node_types`, so two instances of one template never alias) and the program-wide
    /// ones.
    fn frozenFor(p: *const Plan, i: usize) Engine.Frozen {
        const tc = p.tc;
        const u = p.units[i];
        const mod = p.moduleOf(u);
        const owner = p.ownerOf(u);
        const ob: ?*const Body = if (owner) |o| &tc.bodies.items[o] else null;
        const sym = p.symOf(u);
        const inst: ?*const Mono.Instance = if (u == .body and ob.?.of == .instance) &tc.instances[ob.?.of.instance] else null;
        const body: Engine.Body = switch (u) {
            .body => .{ .source = .{ .decl = ob.?.decl, .is_entry = p.entry == @as(?u32, @intCast(i)) } },
            .spec => .{ .source = .{ .decl = ob.?.decl, .cached = false } },
            .lambda => |k| .{ .lambda = k },
            .derive => |k| .{ .derive = &tc.derives[k] },
            .erased => |k| .{ .erased = &tc.erased_units[k] },
            .adapter => |k| .{ .adapter = &tc.adapters[k] },
        };
        const sig: ?Fingerprint.Sig = switch (u) {
            .body => ob.?.sig,
            .spec => blk: {
                var s = ob.?.sig;
                s.name = sym.name;
                break :blk s;
            },
            .derive, .erased, .lambda, .adapter => null,
        };
        const src = p.modules[mod];
        return .{
            .in = .{
                .tree = src.tree,
                .tokens = src.tokens,
                .source = src.source,
                .resolutions = p.res.resolutions[mod],
                .node_types = if (ob) |b| b.node_types else &.{},
                .callees = if (ob) |b| b.callees else &.{},
                .layouts = tc.layouts,
                .enum_layouts = tc.enum_layouts,
                .names = p.names,
                .sig = sig,
                .instances = tc.instances,
                .sigs = tc.sigs,
                .methods = tc.methods,
                .derives = tc.derives,
                .prelude_ids = tc.prelude_ids,
                .char_struct = tc.char_struct,
                .closure = if (owner) |o| p.closureFor(o, switch (u) {
                    .spec => |k| p.specs.units[k].targets,
                    else => &.{},
                }) else .none,
            },
            .body = body,
            .sym = sym,
            .opt = p.o.opt,
            .type_args = if (inst) |x| x.args else &.{},
            .conformances = if (inst) |x| x.conformances else &.{},
            .closure_facts = if (u == .body and i < p.closure_facts.len) p.closure_facts[i] else &.{},
        };
    }

    /// The closure tables a body of `owner` lowers against.
    fn closureFor(p: *const Plan, owner: Closure.Owner, spec_targets: []const Spec.Target) lower.ClosureInputs {
        return .{
            .adapters = p.tc.adapters,
            .lambda_units = p.tc.lambda_units,
            .owner = owner,
            .storage = Closure.ownerStorage(p.tc.bodies, p.tc.owner_closures, owner, p.o.frame),
            .frame_envs = p.o.frame,
            .specs = p.specs,
            .spec_targets = spec_targets,
        };
    }

    /// Each body's closure facts, indexed like the body list: `Fingerprint.ownerFacts` of the
    /// closure tables its lowering gets (`closureFor`), so its key and its code read one
    /// storage. Serial, once per body. Empty when there is no lambda unit and no boxed local.
    fn ownerClosureFacts(p: *const Plan, gpa: std.mem.Allocator) ![]const []const u8 {
        if (p.tc.lambda_units.len == 0 and p.tc.owner_closures.len == 0) return &.{};
        const facts = try gpa.alloc([]const u8, p.tc.bodies.items.len);
        @memset(facts, &.{});
        errdefer freeClosureFacts(gpa, facts);
        for (facts, 0..) |*f, i| {
            const c = p.closureFor(@intCast(i), &.{});
            f.* = try Fingerprint.ownerFacts(gpa, Closure.ownerUnits(c.lambda_units, c.owner.?), c.storage, c.frame_envs);
        }
        return facts;
    }

    fn emitError(p: *const Plan, d: CodegenIr.Diagnostic, i: usize) EmitError {
        const u = p.units[i];
        return emitErrorOf(p.graph, d, p.moduleOf(u), p.declOf(u));
    }
};

/// Whether a build keeps the envs of closures that never outlive their maker in frame
/// slots, and unboxes the captures only they share. Any one opt pass turns it on, not the
/// level as such: `-O1` with every pass off lowers closures as `-O0` does, and `-O0` with
/// one pass on (`--opt=dce`) gets frame envs, so a pass bisection that ends with every pass
/// off also turns this off.
fn frameStorage(opt: Opt.Config) bool {
    return opt.any();
}

fn freeClosureFacts(gpa: std.mem.Allocator, facts: []const []const u8) void {
    for (facts) |f| gpa.free(f);
    gpa.free(facts);
}

fn freeNames(gpa: std.mem.Allocator, names: []const Link.SymName) void {
    for (names) |nm| gpa.free(nm.name);
    gpa.free(names);
}

/// Lower EVERY unit across the whole module `graph` (in parallel, memoized through the
/// codegen cache), then run the serial relink tail. The entry `main` MUST live in the
/// entry module (error otherwise). `res`/`tc` are the whole-graph resolve/typecheck results
/// (program-wide global ids). Caller owns the returned `LinkedProgram` on success.
pub fn lowerGraphProgram(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    target: []const u8,
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const Typecheck.GraphResult,
    mode: Engine.Mode,
    opt: Opt.Config,
    /// Give generic calls with a named fn or lambda argument their own spec unit; off
    /// sends every call through the erased instance.
    specialize: bool,
    probe: ?*Engine.LowerProbe,
    link_ns: ?*u64,
    /// The `-j` jobs knob: the chunk-count basis for the per-unit codegen fan-out (0 =>
    /// host cpu count). Codegen dispatches ~`ncpu` chunks, not one task per unit, so
    /// `-jN` scales. Determinism is unchanged (ranges are disjoint, slots read back in
    /// index order), so this is a perf lever only.
    ncpu: usize,
) !LowerProgramResult {
    var plan = try Plan.init(gpa, graph, res, tc, .{ .opt = opt, .specialize = specialize, .frame = frameStorage(opt) });
    defer plan.deinit(gpa);

    const entry = plan.entry orelse return .{ .err = .{
        .message = "-o requires a function named 'main' in the entry module",
        .span = null,
    } };

    // Parameters on main are unsupported (codegen also guards; cleaner up front).
    {
        const em = &graph.modules[graph.entry_index];
        const main_gid = tc.bodies.items[plan.units[entry].body].of.base;
        const main_decl = em.nodes[res.fns[main_gid].decl_node.int()];
        const main_proto = Ast.protoAt(em.tree(), main_decl.lhs.int());
        if (main_proto.params.len > 0) return .{ .err = .{
            .message = "parameters on main are unsupported",
            .span = .{ .start = em.tokens[main_decl.main_token].start, .end = em.tokens[main_decl.main_token].end },
            .module = graph.entry_index,
        } };

        // Rule 7: `main` may only yield `int` (the process exit code) or `()`
        // (nothing) — any other return type has no entry-point semantics. Enforced
        // primarily in the type checker (`Typecheck.checkMainReturn`, sorted into the
        // type-diagnostic stream); this is the codegen-entry backstop.
        const main_ret = tc.sigs[main_gid].ret;
        if (main_ret.kind != .int and main_ret.kind != .unit) return .{ .err = .{
            .message = "main must return int or ()",
            .span = .{ .start = em.tokens[main_decl.main_token].start, .end = em.tokens[main_decl.main_token].end },
            .module = graph.entry_index,
        } };
    }

    const slots = try gpa.alloc(Engine.Slot, plan.units.len);
    defer gpa.free(slots);
    for (slots) |*s| s.* = .{};

    const Ctx = struct {
        gpa: std.mem.Allocator,
        io: Io,
        engine: Engine,
        target: []const u8,
        plan: *const Plan,
        slots: []Engine.Slot,
        pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(unitJob)) {
            return .{ c.gpa, c.io, c.engine, c.target, c.plan, i, &c.slots[i] };
        }
    };
    Engine.chunkedFanOut(io, plan.units.len, ncpu, Engine.Chunk.codegen.threshold, Engine.Chunk.codegen.chunks_per_cpu, unitJob, Ctx{
        .gpa = gpa,
        .io = io,
        .engine = Engine.initProbe(cache, mode, probe),
        .target = target,
        .plan = &plan,
        .slots = slots,
    });

    for (slots, 0..) |s, i| if (s.err) |e| {
        for (slots) |*sl| if (sl.fc) |*fc| fc.deinit(gpa);
        if (e == error.CodegenDiagnostic) if (s.diag) |d| return .{ .err = plan.emitError(d, i) };
        return e;
    };

    var compiled: usize = 0;
    var cached_n: usize = 0;
    var opt_stats: Opt.Stats = .{};
    var ir_instrs: usize = 0;
    for (slots) |s| {
        if (s.cached) cached_n += 1 else compiled += 1;
        opt_stats.add(s.opt_stats);
        ir_instrs += s.ir_instrs;
    }

    // relink/link index by unit position, not global id. Each name is borrowed from the
    // plan's tables, which outlive relink here.
    const unit_names = try gpa.alloc(Link.SymName, plan.units.len);
    defer gpa.free(unit_names);
    for (plan.units, unit_names) |u, *nm| nm.* = plan.symOf(u);

    // Build the descriptor plan (canonical order = `tc.descriptor_types`), each entry's
    // `{size, align, size_class}` from the layout snapshot + the erased hash/eq unit names
    // (borrowed from `tc.erased_units`); `trace_name` points at the type's trace unit (its
    // erased str/Ref/gc_array trace, else null), so the collector resolves `trace_off` and
    // dispatches container-element tracing. Threaded through relink → linkAndTail →
    // linkProgram → Link.link.
    var desc_plan: std.ArrayList(Link.DescEntry) = .empty;
    defer desc_plan.deinit(gpa);
    for (tc.descriptor_types) |dt| {
        const d = Abi.descriptorFor(dt.ty, tc.layouts, tc.enum_layouts);
        try desc_plan.append(gpa, .{
            .desc_hash = try Link.descHash(gpa, dt.ty),
            .size = d.size,
            .@"align" = d.@"align",
            .size_class = d.size_class,
            .hash_name = if (dt.hashable) erasedNameFor(tc, dt.ty, .hash) else null,
            .eq_name = if (dt.hashable) erasedNameFor(tc, dt.ty, .eq) else null,
            .trace_name = traceNameFor(tc, dt.ty),
        });
    }

    const link_t0: i128 = if (link_ns != null) nowNs(io) else 0;
    const out = relink(io, gpa, slots, unit_names, entry, compiled, cached_n, opt_stats, ir_instrs, desc_plan.items);
    if (link_ns) |lp| {
        const dt = nowNs(io) - link_t0;
        lp.* = if (dt > 0) @intCast(dt) else 0;
    }
    return out;
}

fn nowNs(io: Io) i128 {
    return Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
}

/// One codegen job: unit `i`'s view through `Engine.codegen`'s fingerprint → cache →
/// lower path. A callee's cross-module identity and the layouts it touches ride in
/// through the program-wide `names`/`sigs`/`layouts`, so the fingerprint folds a qualified
/// callee distinctly.
fn unitJob(gpa: std.mem.Allocator, io: Io, engine: Engine, target: []const u8, plan: *const Plan, i: usize, slot: *Engine.Slot) void {
    const fr = plan.frozenFor(i);
    engine.codegen(gpa, io, target, &fr, i, slot) catch |e| {
        slot.err = e;
    };
}

/// Build the program-wide index→SymName table for a graph build: one entry per
/// global fn id, parallel to `tc.sigs`. A user fn -> {user_fn, sig name} (the
/// qualified spelling, or bare `main`); a synthetic bodyless builtin (`panic`) ->
/// {builtin,"panic"}. The name MUST equal the sig name so the fingerprint callee
/// fold matches the reloc target. Caller owns the names.
fn buildGraphNames(gpa: std.mem.Allocator, fns: []const ResolveGraph.GlobalFn, sigs: []const Fingerprint.Sig) ![]Link.SymName {
    const names = try gpa.alloc(Link.SymName, fns.len);
    var built: usize = 0;
    errdefer {
        for (names[0..built]) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (fns, 0..) |gf, i| {
        // The resolver's kind carries straight through: `.builtin` (`panic`/intrinsics),
        // `.import` (an `extern` — a bare dyld symbol reached via `__got`), or
        // `.user_fn`. An extern's name is the bare C symbol (`labs`), matching its reloc.
        names[i] = .{ .kind = gf.kind, .name = try gpa.dupe(u8, sigs[i].name) };
        built += 1;
    }
    return names;
}

/// `--emit ir` for a whole-graph build: lower every unit of the plan through the build's
/// own per-unit path and render the deterministic IR text, in plan order. Caller owns the
/// returned text on success.
pub fn renderGraphIr(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const Typecheck.GraphResult,
    opt: Opt.Config,
    specialize: bool,
) !IrResult {
    return renderIr(gpa, graph, res, tc, .{ .opt = opt, .specialize = specialize, .frame = frameStorage(opt) });
}

/// `renderGraphIr` with frame storage chosen apart from the opt passes, which lets a test
/// read frame-storage IR that no pass has rewritten.
pub fn renderIr(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const Typecheck.GraphResult,
    o: Options,
) !IrResult {
    var plan = try Plan.init(gpa, graph, res, tc, o);
    defer plan.deinit(gpa);
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    var diags: std.ArrayList(CodegenIr.Diagnostic) = .empty;
    defer diags.deinit(gpa);
    for (0..plan.units.len) |i| {
        const fr = plan.frozenFor(i);
        var stats: Opt.Stats = .{};
        var func = try Engine.lowerUnit(gpa, &fr, &diags, &stats);
        defer func.deinit(gpa);
        if (diags.items.len > 0) return .{ .err = plan.emitError(diags.items[0], i) };
        if (i > 0) try aw.writer.writeAll("\n");
        try Ir.render(&aw.writer, &func, fr.in.layouts, fr.in.enum_layouts);
    }
    return .{ .ok = try aw.toOwnedSlice() };
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
) ![]u8 {
    return link.assembleAndSign(io, gpa, identifier, code, entry_off, cstrings, data_relocs);
}

const testing = std.testing;

/// One module checked as a graph, kept alive so a plan can borrow it. Heap-pinned because
/// `res` and `tc` are built against `&g`.
const Checked = struct {
    tokens: []@import("../ast/Token.zig").Token,
    tree: Ast.Tree,
    g: Graph.Graph,
    res: ResolveGraph.GraphResult,
    tc: Typecheck.GraphResult,

    fn create(gpa: std.mem.Allocator, src: []const u8) !*Checked {
        const self = try gpa.create(Checked);
        errdefer gpa.destroy(self);
        self.tokens = try @import("../lex.zig").tokenize(gpa, src);
        errdefer gpa.free(self.tokens);
        self.tree = try @import("../parse.zig").expectTree(gpa, self.tokens, src);
        errdefer {
            gpa.free(self.tree.nodes);
            gpa.free(self.tree.extra);
        }
        self.g = try Graph.single(gpa, "main", "", src, self.tokens, self.tree.nodes, self.tree.extra, self.tree.pub_bits);
        errdefer self.g.deinit(gpa);
        self.res = try ResolveGraph.resolveGraph(gpa, &self.g);
        errdefer self.res.deinit(gpa);
        self.tc = try @import("../types_graph.zig").checkGraph(gpa, &self.g, &self.res, null, 0);
        return self;
    }

    fn destroy(self: *Checked, gpa: std.mem.Allocator) void {
        self.tc.deinit(gpa);
        self.res.deinit(gpa);
        self.g.deinit(gpa);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
        gpa.free(self.tokens);
        gpa.destroy(self);
    }

    fn plan(self: *const Checked, gpa: std.mem.Allocator, opt: Opt.Config) !Plan {
        return Plan.init(gpa, &self.g, &self.res, &self.tc, .{ .opt = opt, .frame = frameStorage(opt) });
    }
};

/// A base fn calling a named fn and a lambda, two generic instances (one with a spec
/// unit), a derived `==` and a fn value: every unit class but the erased witnesses, which
/// only bundled std code asks for.
const every_class_src =
    \\struct P { x: int, y: int }
    \\fn add(a: int, b: int) -> int { a + b }
    \\fn apply[T](f: fn(T, T) -> T, x: T) -> T { f(x, x) }
    \\fn keep[T](x: T) -> T { x }
    \\fn main() -> int {
    \\    p := P { x: 1, y: 2 }
    \\    q := P { x: 1, y: 2 }
    \\    k := 3
    \\    g := add
    \\    h := fn(a: int, b: int) -> int { a * b + k }
    \\    e := if p == q { 1 } else { 0 }
    \\    apply(add, 2) + apply(h, 1) + keep(e) + g(1, 2)
    \\}
    \\
;

test "the plan lists every unit class in canonical order, each under its own symbol" {
    const gpa = testing.allocator;
    const ck = try Checked.create(gpa, every_class_src);
    defer ck.destroy(gpa);
    var plan = try ck.plan(gpa, .O0);
    defer plan.deinit(gpa);

    const want = [_]struct { std.meta.Tag(Unit), []const u8 }{
        .{ .body, "main.add" },
        .{ .body, "main" },
        .{ .body, "main.apply$int" },
        .{ .body, "main.keep$int" },
        .{ .derive, "Eq$eq$s0" },
        .{ .lambda, "main$l0" },
        .{ .adapter, "main.add$fnval" },
        .{ .spec, "main.apply$int$sp0$main.add" },
    };
    try testing.expectEqual(want.len, plan.units.len);
    for (want, plan.units) |w, u| {
        try testing.expectEqual(w[0], std.meta.activeTag(u));
        try testing.expectEqualStrings(w[1], plan.symOf(u).name);
    }
    try testing.expectEqual(@as(?u32, 1), plan.entry);
    // The bodies lead the plan: a body's unit is its index in the body list.
    try testing.expectEqual(@as(usize, 4), ck.tc.bodies.items.len);
    for (ck.tc.bodies.items, 0..) |b, i| {
        try testing.expectEqual(@as(Body.Index, @intCast(i)), plan.units[i].body);
        try testing.expectEqualStrings(b.sig.name, plan.symOf(plan.units[i]).name);
    }
}

test "--emit ir renders exactly the plan's units, in plan order" {
    const gpa = testing.allocator;
    const ck = try Checked.create(gpa, every_class_src);
    defer ck.destroy(gpa);
    for ([_]Opt.Config{ .O0, .O1 }) |opt| {
        var plan = try ck.plan(gpa, opt);
        defer plan.deinit(gpa);
        const out = try renderGraphIr(gpa, &ck.g, &ck.res, &ck.tc, opt, true);
        const ir = out.ok;
        defer gpa.free(ir);
        var at: usize = 0;
        var n: usize = 0;
        while (std.mem.indexOfPos(u8, ir, at, "fn ")) |start| : (n += 1) {
            const open = std.mem.indexOfScalarPos(u8, ir, start, '(').?;
            try testing.expectEqualStrings(plan.symOf(plan.units[n]).name, ir[start + 3 .. open]);
            at = std.mem.indexOfPos(u8, ir, start, "\n}\n").? + 3;
        }
        try testing.expectEqual(plan.units.len, n);
    }
}

test "each unit reads its owner's tables: an instance its own, a lambda its owner's" {
    const gpa = testing.allocator;
    const ck = try Checked.create(gpa, every_class_src);
    defer ck.destroy(gpa);
    var plan = try ck.plan(gpa, .O1);
    defer plan.deinit(gpa);
    const tc = &ck.tc;
    for (plan.units, 0..) |u, i| {
        const fr = plan.frozenFor(i);
        switch (u) {
            .body => |bi| {
                try testing.expectEqual(@as(?Closure.Owner, bi), fr.in.closure.owner);
                try testing.expect(fr.body.source.cached);
                switch (tc.bodies.items[bi].of) {
                    .base => |gid| {
                        try testing.expectEqual(tc.node_types[0].ptr, fr.in.node_types.ptr);
                        try testing.expectEqual(tc.sigs[gid].mut_mask, fr.in.sig.?.mut_mask);
                    },
                    .instance => |k| {
                        try testing.expectEqual(tc.instances[k].node_types.ptr, fr.in.node_types.ptr);
                        try testing.expectEqual(tc.instances[k].args.ptr, fr.type_args.ptr);
                    },
                }
            },
            .lambda => |k| {
                try testing.expectEqual(tc.node_types[0].ptr, fr.in.node_types.ptr);
                try testing.expectEqual(tc.lambda_units[k].owner, fr.in.closure.owner.?);
                try testing.expect(fr.in.closure.frame_envs);
            },
            .spec => |k| {
                const inst = &tc.instances[plan.specs.units[k].inst];
                try testing.expectEqual(inst.node_types.ptr, fr.in.node_types.ptr);
                try testing.expectEqual(@as(usize, 1), fr.in.closure.spec_targets.len);
                try testing.expect(!fr.body.source.cached);
                try testing.expectEqualStrings(fr.sym.name, fr.in.sig.?.name);
            },
            .derive, .erased, .adapter => {
                try testing.expectEqual(@as(usize, 0), fr.in.node_types.len);
                try testing.expectEqual(@as(?Closure.Owner, null), fr.in.closure.owner);
                try testing.expectEqual(@as(?Fingerprint.Sig, null), fr.in.sig);
            },
        }
    }
}

test "the build lowers and links every unit of the plan" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const ck = try Checked.create(gpa, every_class_src);
    defer ck.destroy(gpa);
    for ([_]Opt.Config{ .O0, .O1 }) |opt| {
        var out = try lowerGraphProgram(gpa, threaded.io(), Cache.disabled, "aarch64-macos", &ck.g, &ck.res, &ck.tc, .verify, opt, true, null, null, 1);
        defer if (out == .ok) out.ok.deinit(gpa);
        try testing.expect(out == .ok);
        try testing.expectEqual(@as(usize, 8), out.ok.codegen_compiled);
    }
}

/// The calls each body's code makes to a unit of `plan` that the body's key does not fold
/// ([C2]), into `unfolded` (names borrowed from `plan`); returns how many calls it checked.
fn unfoldedCalls(gpa: std.mem.Allocator, plan: *const Plan, unfolded: *std.ArrayList([]const u8)) !usize {
    var unit_syms: std.StringHashMapUnmanaged(void) = .empty;
    defer unit_syms.deinit(gpa);
    for (plan.units) |u| try unit_syms.put(gpa, plan.symOf(u).name, {});

    var calls: usize = 0;
    for (plan.units, 0..) |u, i| {
        if (u != .body) continue;
        const fr = plan.frozenFor(i);
        var sigs: std.ArrayList(Fingerprint.Sig) = .empty;
        defer sigs.deinit(gpa);
        try AstWalk.walkCalls(gpa, &fr.in, fr.body.source.decl, fr.in.sig, &sigs);
        var diags: std.ArrayList(CodegenIr.Diagnostic) = .empty;
        defer diags.deinit(gpa);
        var stats: Opt.Stats = .{};
        var func = try Engine.lowerUnit(gpa, &fr, &diags, &stats);
        defer func.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), diags.items.len);
        for (func.blocks) |blk| for (blk.instrs) |ins| switch (ins.op) {
            .call => |c| if (unit_syms.contains(c.callee.name)) {
                calls += 1;
                const folded = for (sigs.items) |sg| {
                    if (std.mem.eql(u8, sg.name, c.callee.name)) break true;
                } else false;
                if (!folded) try unfolded.append(gpa, c.callee.name);
            },
            else => {},
        };
    }
    return calls;
}

test "every call a body makes relocates to a symbol its key folds, but for unkeyed instances" {
    const gpa = testing.allocator;
    const ck = try Checked.create(gpa,
        \\struct P { x: int, y: int }
        \\struct B[T] { v: int }
        \\impl B[T] { fn get(self) -> int { self.v } }
        \\fn add(a: int, b: int) -> int { a + b }
        \\fn apply[T](f: fn(T, T) -> T, x: T) -> T { f(x, x) }
        \\fn keep[T](x: T) -> T { x }
        \\fn mk[T]() -> B[T] { B[T] { v: 4 } }
        \\fn main() -> int {
        \\    p := P { x: 1, y: 2 }
        \\    q := P { x: 1, y: 2 }
        \\    k := 3
        \\    h := fn(a: int, b: int) -> int { a * b + k }
        \\    e := if p == q { 1 } else { 0 }
        \\    apply(add, 2) + apply(h, 1) + keep(e) + keep[int](1) + mk[bool]().get()
        \\}
        \\
    );
    defer ck.destroy(gpa);
    var plan = try ck.plan(gpa, .O0);
    defer plan.deinit(gpa);
    var unfolded: std.ArrayList([]const u8) = .empty;
    defer unfolded.deinit(gpa);
    try testing.expect(try unfoldedCalls(gpa, &plan, &unfolded) >= 6);
    // `mk[bool]()` names no type param in its value params, so its key misses it ([C2]).
    try testing.expectEqual(@as(usize, 1), unfolded.items.len);
    try testing.expectEqualStrings("main.mk$bool", unfolded.items[0]);
}

test "the calls a for-in and a try_into make are folded into the body's key (#35)" {
    const gpa = testing.allocator;
    const ck = try Checked.create(gpa,
        \\protocol Iterator[T] { fn next(mut self) -> Option[T] }
        \\struct It[T] { i: int, n: int }
        \\impl It[T] has Iterator[int] {
        \\    fn next(mut self) -> Option[int] {
        \\        self.i = self.i + 1
        \\        if self.i > self.n { return Option[int].none }
        \\        return Option.some(self.i)
        \\    }
        \\}
        \\impl It[T] { fn iter(self) -> It[T] { self } }
        \\fn main() -> int {
        \\    t := 0
        \\    it := It[int] { i: 0, n: 4 }
        \\    for x in it { t = t + x }
        \\    _c: char = t.try_into().unwrap()
        \\    return t
        \\}
        \\
    );
    defer ck.destroy(gpa);
    var plan = try ck.plan(gpa, .O0);
    defer plan.deinit(gpa);
    var unfolded: std.ArrayList([]const u8) = .empty;
    defer unfolded.deinit(gpa);
    // `iter`, `next` and the `TryInto` witness; no call node names any of them.
    try testing.expectEqual(@as(usize, 3), try unfoldedCalls(gpa, &plan, &unfolded));
    try testing.expectEqual(@as(usize, 0), unfolded.items.len);
}
