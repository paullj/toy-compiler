//! Whole-graph codegen + link tail: turns a resolved+typechecked module graph into
//! a linked Mach-O image (or rendered IR text). The per-fn codegen fan-out lowers
//! each function against a single-fn `Frozen` view (thread-safe, program-wide tables
//! + the fn's owning-module arrays), then the SERIAL relink tail interns strings,
//! rebases cross-segment relocs, links, and signs — the order-sensitive
//! assemble → applyDataRelocs → sign-last pipeline. Distinct from the per-file
//! front-end (`Driver.run`/`pipeline`, discover → lex/parse/check).
const std = @import("std");
const Span = @import("../diagnostics/model.zig").Span;
const builtin = @import("builtin");
const Io = std.Io;
const Token = @import("../ast/Token.zig").Token;
const Ast = @import("../ast/Ast.zig");
const Cache = @import("../query/Cache.zig");
const Engine = @import("../query/Engine.zig");
const Resolve = @import("../resolve.zig");
const Typecheck = @import("../types.zig");
const Mono = @import("../symbols/Mono.zig");
const Graph = @import("Graph.zig");
const ResolveGraph = @import("../resolve_graph.zig");
const CodegenIr = @import("../codegen/CodegenIr.zig");
const Ir = @import("../ir/Ir.zig");
const Opt = @import("../opt/Opt.zig");
const lower = @import("../lower.zig");
const DeriveEmit = @import("../lower/derive_emit.zig");
const AdapterEmit = @import("../lower/adapter_emit.zig");
const Closure = @import("../symbols/Closure.zig");
const Spec = @import("../symbols/Spec.zig");
const AstWalk = @import("../query/AstWalk.zig");
const Fingerprint = @import("../query/Fingerprint.zig");
const Link = @import("../link/Link.zig");
const link = @import("../link/emit.zig");
const Abi = @import("../codegen/abi/Abi.zig");
const Derive = @import("../symbols/Derive.zig");

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

/// One parallel codegen job's slot. Each job owns exactly one slot (no locks).
const FnSlot = struct {
    fc: ?Link.FnCode = null,
    cached: bool = false,
    err: ?anyerror = null,
    /// With `err == error.CodegenDiagnostic`: the lowering's first diagnostic.
    diag: ?CodegenIr.Diagnostic = null,
    /// dual-metric, set only when the fn is freshly lowered (cached fns
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
    /// Parallel to `node_types`: the checker's method-call selections.
    callees: []const Typecheck.Callee,
    layouts: []const Typecheck.Layout,
    enum_layouts: []const Typecheck.EnumLayout,
    names: []const Link.SymName,
    /// A one-element slice naming this job's fn (`frozenFor` points it at a stack buf).
    fn_nodes: []const Ast.Index,
    sigs: []const Fingerprint.Sig,
    /// Opt level / pass selection. Mixed into the codegen cache key so
    /// toggling `-O` lands on a different entry, and threaded into `lowerOne`.
    opt: Opt.Config,
    /// The whole monomorphization instance table, so `walkCalls`/`lowerCall`
    /// resolve a generic call site to its reified instance. Empty for a program with
    /// no generics; shared read-only across all jobs.
    instances: []const Mono.Instance = &.{},
    /// This unit's OWN concrete type-args when it is a monomorphized instance
    /// (`&.{}` for a base fn) — folded into the (d) fingerprint component so two
    /// instances of one template get distinct cache keys.
    type_args: []const Typecheck.Type = &.{},
    /// This unit's OWN resolved `[T has P]` bound conformances when it is a bounded
    /// monomorphized instance (`&.{}` for a base or unbounded-template instance) —
    /// folded into the (e) fingerprint component so toggling a sibling conformance
    /// invalidates exactly the dependent monomorphizations.
    conformances: []const Mono.ResolvedConformance = &.{},
    /// The program-wide inherent-method table, for method-call dispatch in
    /// `lowerCall` and the method-sig fold in `CallVisitor`. Shared read-only.
    methods: []const Typecheck.Method = &.{},
    /// The authorized structural auto-derive recipes: so `lowerStructEq`/`lowerDeriveEq`
    /// resolve a derived witness to its synthetic unit, and `CallVisitor.foldWitness`
    /// folds the derived-Eq witness identity. Shared read-only.
    derives: []const Typecheck.DeriveRecipe = &.{},
    /// The erased hash/eq witnesses, so `Engine.codegenErased` reads its unit's type/kind/
    /// name off this job's view. Shared read-only.
    erased_units: []const Typecheck.ErasedUnit = &.{},
    /// The prelude protocol ids: so `lowerStructEq`/`CallVisitor` resolve each
    /// operator/derive/`?`-widen witness by its SPECIFIC protocol (a sibling protocol
    /// reusing the name is excluded). Copied verbatim from the checker; lower and the
    /// fingerprint fold read the SAME bundle. Shared read-only.
    prelude_ids: Typecheck.PreludeProtocolIds = .{},
    /// The compiler-provided `char` struct id, for the conversion recognizer +
    /// char-literal lowering. Copied verbatim from the checker. Shared read-only.
    char_struct: ?u32 = null,
    /// The function-value tables this job's lowering reads. Shared read-only.
    closure: lower.ClosureInputs,
    /// This owner body's closure facts (`Closure.ownerFacts`), folded into its key; empty
    /// for every other unit and for a body with no env slot and no boxed local.
    closure_facts: []const u8,

    /// The template facts `Mono.callInstanceRef` needs for this job's read-only view:
    /// derived from the callee's `Sig` (the fingerprint side of the shared discovery/
    /// fold arg-extraction). `null` for a non-generic callee.
    pub fn genericTemplate(self: *const Frozen, gid: u32) ?Mono.TemplateRef {
        return Mono.templateRef(self.sigs, gid);
    }
};

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
    slots: []FnSlot,
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

/// Frozen, read-only inputs shared by every codegen job in a WHOLE-GRAPH build.
/// Per-module arrays (trees/tokens/sources/resolutions/node_types) are indexed by
/// module id; `layouts`/`enum_layouts`/`names`/`sigs` are PROGRAM-WIDE (one global
/// id space). `fn_decls`/`fn_modules` are parallel to the global fn id space (one
/// entry per global fn; bodyless builtins excluded — only real fn bodies are lowered). A
/// codegen job selects its fn's owning module to build a single-fn `Frozen` view,
/// so `walkCalls`/`walkTouchedSig`/`typeRefToType`/`fingerprint`/`lowerOne` run
/// UNCHANGED — the only difference from single-file is which module's tree/resolutions
/// they read.
const GraphFrozen = struct {
    trees: []const Ast.Tree,
    tokens: []const []const Token,
    sources: []const []const u8,
    resolutions: []const []const Resolve.Resolution,
    node_types: []const []const Typecheck.Type,
    callees: []const []const Typecheck.Callee,
    /// Program-wide tables (global ids).
    layouts: []const Typecheck.Layout,
    enum_layouts: []const Typecheck.EnumLayout,
    /// Program-wide: one SymName per global fn id (qualified user fns, bare `main`,
    /// `{builtin,"panic"}` at the panic id). Parallel to `sigs`.
    names: []const Link.SymName,
    sigs: []const Fingerprint.Sig,
    /// The `fn_decl` node for each LOWERABLE global fn (parallel to `lower_ids`).
    fn_decls: []const Ast.Index,
    /// The owning module id for each lowerable fn (parallel to `fn_decls`).
    fn_modules: []const u32,
    /// Global fn id of each lowerable BASE fn (parallel to the first `base_count`
    /// entries of `fn_decls`/`fn_modules`); indexes `names`/`sigs`. Excludes bodyless
    /// builtins and generic templates (lowered only as instances).
    lower_ids: []const u32,
    /// Number of base fn units; `fn_decls`/`fn_modules` entries at `[base_count..]`
    /// are monomorphized instances (parallel to `instances`).
    base_count: usize,
    /// The monomorphization instance table, in canonical order. Appended after
    /// the base fns as extra lowerable units.
    instances: []const Mono.Instance,
    /// Global fn id of the entry `main` (indexes `names`).
    entry_id: u32,
    /// The program-wide inherent-method table, threaded into every job's `Frozen`.
    methods: []const Typecheck.Method = &.{},
    /// The authorized structural auto-derive recipes, in canonical order. Appended
    /// after the base fns + Mono instances as a THIRD class of lowerable units, and
    /// threaded into every job's `Frozen` (a base fn's `==` on a derived type resolves the
    /// synthetic witness through it too).
    derives: []const Typecheck.DeriveRecipe = &.{},
    /// The prelude protocol ids, copied into every job's `Frozen` by `frozenFor`, so
    /// each operator/derive/`?`-widen witness resolves by its SPECIFIC protocol.
    prelude_ids: Typecheck.PreludeProtocolIds = .{},
    /// The compiler-provided `char` struct id, copied into every job's `Frozen` by
    /// `frozenFor` for the conversion recognizer + char-literal lowering.
    char_struct: ?u32 = null,
    /// The count of BASE fns + Mono instances; `fn_decls`/`fn_modules` entries at
    /// `[derive_base, erased_base)` are the source-less derive units (parallel to `derives`).
    derive_base: usize = 0,
    /// The erased hash/eq witnesses, appended after the derive units as a FOURTH class of
    /// lowerable units. Routed to `Engine.codegenErased` (uncached).
    erased_units: []const Typecheck.ErasedUnit = &.{},
    /// The count of BASE fns + Mono instances + derive units; `fn_decls`/`fn_modules`
    /// entries at `[erased_base, adapter_base)` are the erased units (parallel to `erased_units`).
    erased_base: usize = 0,
    /// The lambda units, appended after the erased units as a FIFTH class of lowerable
    /// units (parallel to `lambda_units`). Each lowers under its owner's view. Routed to
    /// `Engine.codegenClosure` (uncached).
    lambda_units: []const Closure.LambdaUnit = &.{},
    /// Entries at `[lambda_base, adapter_base)` are the lambda units.
    lambda_base: usize,
    /// The fn-value thunks, appended after the lambda units as a SIXTH class of lowerable
    /// units (parallel to `adapters`). Routed to `Engine.codegenClosure` (uncached).
    adapters: []const Closure.Adapter = &.{},
    /// Entries at `[adapter_base, spec_base)` are the adapter units.
    adapter_base: usize,
    /// The spec units, appended after the adapters as a SEVENTH class of lowerable units
    /// (parallel to `specs.units`). Each lowers under its instance's view. Routed to
    /// `Engine.codegenClosure` (uncached).
    specs: Spec.Table = .{},
    /// Entries at `[spec_base..]` are the spec units.
    spec_base: usize,
    /// Each owner's boxed locals, sorted by owner; a lambda unit reads its owner's.
    owner_closures: []const Closure.OwnerClosure = &.{},
    /// Per base fn and instance unit (`[0, derive_base)`), its closure facts; empty (no
    /// entries) when the program has no lambda unit and no boxed local.
    closure_facts: []const []const u8,
    opt: Opt.Config,
    /// `frameStorage(opt)`, decided once so the lowering and the closure facts in its key
    /// see the same storage.
    frame: bool,

    /// `--timings` sub-stage probe (codegen-compute vs cache get/put I/O), threaded
    /// only when `--timings` is on. BORROWED; null on a plain build => zero overhead.
    probe: ?*Engine.LowerProbe = null,

    /// Build the single-fn `Frozen` view a codegen job runs against: this fn's
    /// owning module's per-module arrays + the program-wide tables. `fn_nodes` is
    /// a one-element slice naming this fn (the job reads `fn_nodes[0]`).
    fn frozenFor(gf: *const GraphFrozen, lower_i: usize, fn_node_buf: *[1]Ast.Index) Frozen {
        const mod = gf.fn_modules[lower_i];
        fn_node_buf[0] = gf.fn_decls[lower_i];
        // The lowerable-unit classes: base fns `[0, base_count)`, Mono instances
        // `[base_count, derive_base)`, the source-less derive and erased units
        // `[derive_base, lambda_base)`, lambda units `[lambda_base, adapter_base)`, the
        // source-less adapters `[adapter_base, spec_base)`, and spec units `[spec_base, ..)`.
        // A body reads its owner's tables: an instance its OWN per-instance `node_types` (so
        // two instances of one template never alias), a base fn its module's slice, a lambda
        // unit whichever its owner reads, a spec unit its instance's. A source-less unit
        // reads NO AST/node_types (its emitter is table-driven), so leave them empty.
        const is_lambda = lower_i >= gf.lambda_base and lower_i < gf.adapter_base;
        const spec: ?Spec.Unit = if (lower_i >= gf.spec_base) gf.specs.units[lower_i - gf.spec_base] else null;
        const owner: ?Closure.Owner = if (is_lambda)
            gf.lambda_units[lower_i - gf.lambda_base].owner
        else if (spec) |u|
            .{ .instance = u.inst }
        else if (lower_i < gf.base_count)
            .{ .base = gf.lower_ids[lower_i] }
        else if (lower_i < gf.derive_base)
            .{ .instance = @intCast(lower_i - gf.base_count) }
        else
            null;
        const inst: ?*const Mono.Instance = if (owner) |o| switch (o) {
            .instance => |k| &gf.instances[k],
            .base => null,
        } else null;
        const is_derive = owner == null;
        const nts: []const Typecheck.Type = if (inst) |i| i.node_types else if (is_derive) &.{} else gf.node_types[mod];
        const cs: []const Typecheck.Callee = if (inst) |i| i.callees else if (is_derive) &.{} else gf.callees[mod];
        const targs: []const Typecheck.Type = if (inst) |i| i.args else &.{};
        const confs: []const Mono.ResolvedConformance = if (inst) |i| i.conformances else &.{};
        return .{
            .tree = gf.trees[mod],
            .tokens = gf.tokens[mod],
            .source = gf.sources[mod],
            .resolutions = gf.resolutions[mod],
            .node_types = nts,
            .callees = cs,
            .layouts = gf.layouts,
            .enum_layouts = gf.enum_layouts,
            .names = gf.names,
            .fn_nodes = fn_node_buf[0..1],
            .sigs = gf.sigs,
            .opt = gf.opt,
            .instances = gf.instances,
            .type_args = targs,
            .conformances = confs,
            .methods = gf.methods,
            .derives = gf.derives,
            .erased_units = gf.erased_units,
            .prelude_ids = gf.prelude_ids,
            .char_struct = gf.char_struct,
            .closure = if (is_derive) .none else closureFor(gf.adapters, gf.lambda_units, gf.owner_closures, owner.?, gf.specs, if (spec) |u| u.targets else &.{}, gf.frame),
            .closure_facts = if (lower_i < gf.closure_facts.len) gf.closure_facts[lower_i] else &.{},
        };
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

/// The closure tables a body of `owner` lowers against, under frame storage when `frame`.
fn closureFor(
    adapters: []const Closure.Adapter,
    units: []const Closure.LambdaUnit,
    owner_closures: []const Closure.OwnerClosure,
    owner: Closure.Owner,
    specs: Spec.Table,
    spec_targets: []const Spec.Target,
    frame: bool,
) lower.ClosureInputs {
    const st = Closure.ownerStorage(owner_closures, owner, frame);
    return .{
        .adapters = adapters,
        .lambda_units = units,
        .owner = owner,
        .boxed = st.boxed,
        .unboxed = st.unboxed,
        .frame_envs = frame,
        .specs = specs,
        .spec_targets = spec_targets,
    };
}

/// Each base fn and instance unit's closure facts, indexed like `fn_decls[0..derive_base)`:
/// its storage under `frame` (as `closureFor` hands it to the lowering) and its own lambda
/// units (a contiguous run of the `(owner, node)`-sorted table) laid out by
/// `Closure.ownerFacts`. Serial, once per owner. Empty when there is no lambda unit and no
/// boxed local.
fn buildClosureFacts(
    gpa: std.mem.Allocator,
    units: []const Closure.LambdaUnit,
    owner_closures: []const Closure.OwnerClosure,
    lower_ids: []const u32,
    base_count: usize,
    derive_base: usize,
    frame: bool,
) ![]const []const u8 {
    if (units.len == 0 and owner_closures.len == 0) return &.{};
    const facts = try gpa.alloc([]const u8, derive_base);
    @memset(facts, &.{});
    errdefer freeClosureFacts(gpa, facts);
    // Every owner with a unit or a box, each once: both tables are sorted by owner.
    var i: usize = 0;
    var oi: usize = 0;
    while (i < units.len or oi < owner_closures.len) {
        const owner = if (oi == owner_closures.len or (i < units.len and !Closure.ownerLess(owner_closures[oi].owner, units[i].owner)))
            units[i].owner
        else
            owner_closures[oi].owner;
        var j = i;
        while (j < units.len and std.meta.eql(units[j].owner, owner)) j += 1;
        if (oi < owner_closures.len and std.meta.eql(owner_closures[oi].owner, owner)) oi += 1;
        const at: ?usize = switch (owner) {
            .base => |gid| std.sort.binarySearch(u32, lower_ids, gid, struct {
                fn order(key: u32, x: u32) std.math.Order {
                    return std.math.order(key, x);
                }
            }.order),
            .instance => |k| base_count + k,
        };
        if (at) |ui| facts[ui] = try Closure.ownerFacts(gpa, units[i..j], Closure.ownerStorage(owner_closures, owner, frame), frame);
        i = j;
    }
    return facts;
}

fn freeClosureFacts(gpa: std.mem.Allocator, facts: []const []const u8) void {
    for (facts) |f| gpa.free(f);
    gpa.free(facts);
}

/// What `buildSpecTable` reads: per-module arrays by module id, the rest program-wide.
const SpecSource = struct {
    trees: []const Ast.Tree,
    tokens: []const []const Token,
    sources: []const []const u8,
    resolutions: []const []const Resolve.Resolution,
    node_types: []const []const Typecheck.Type,
    callees: []const []const Typecheck.Callee,
    closures: []const []const Closure.FnClosures,
    names: []const Link.SymName,
    sigs: []const Fingerprint.Sig,
    instances: []const Mono.Instance,
    methods: []const Typecheck.Method,
    lambda_units: []const Closure.LambdaUnit,
    /// The base owners in lower order: global fn id, module and `fn_decl`, parallel.
    base_ids: []const u32,
    base_mods: []const u32,
    base_decls: []const Ast.Index,
};

/// One owner body's read-only tables, as `AstWalk.instanceOfCall` and `Spec.targetsOf`
/// read them.
const SpecView = struct {
    tree: Ast.Tree,
    resolutions: []const Resolve.Resolution,
    node_types: []const Typecheck.Type,
    callees: []const Typecheck.Callee,
    names: []const Link.SymName,
    sigs: []const Fingerprint.Sig,
    instances: []const Mono.Instance,
    methods: []const Typecheck.Method,
    closure: lower.ClosureInputs,

    pub fn genericTemplate(self: *const SpecView, gid: u32) ?Mono.TemplateRef {
        return Mono.templateRef(self.sigs, gid);
    }
};

/// The spec table: a mask per template with an eligible param, and a unit per distinct
/// generic call site (in any lowerable owner: base fns, then instances, lambdas included)
/// whose argument for an eligible param is a named fn or a lambda literal. Serial, and the
/// cap and the unit order go by name, so it is identical at any `-j`.
fn buildSpecTable(gpa: std.mem.Allocator, src: SpecSource) !Spec.Table {
    const masks = try specMasks(gpa, src);
    if (masks.len == 0) {
        gpa.free(masks);
        return .{};
    }
    var cands: std.ArrayList(Spec.Candidate) = .empty;
    collectSpecCandidates(gpa, src, masks, &cands) catch |e| {
        Spec.freeUnits(gpa, cands.items);
        cands.deinit(gpa);
        gpa.free(masks);
        return e;
    };
    return Spec.finish(gpa, &cands, masks);
}

/// The eligible params of each instantiated template that has one, sorted by gid.
fn specMasks(gpa: std.mem.Allocator, src: SpecSource) ![]Spec.TemplateMask {
    var masks: std.ArrayList(Spec.TemplateMask) = .empty;
    errdefer masks.deinit(gpa);
    var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer seen.deinit(gpa);
    for (src.instances) |inst| {
        const gid = inst.template_gid;
        if ((try seen.getOrPut(gpa, gid)).found_existing) continue;
        if (gid >= src.sigs.len) continue;
        const fcs: []const Closure.FnClosures = if (inst.mod < src.closures.len) src.closures[inst.mod] else &.{};
        const walk_src: AstWalk.Source = .{ .tree = src.trees[inst.mod], .tokens = src.tokens[inst.mod], .source = src.sources[inst.mod] };
        const mv = .{ .tree = src.trees[inst.mod], .resolutions = src.resolutions[inst.mod], .callees = inst.callees, .sigs = src.sigs, .methods = src.methods };
        const m = Spec.eligibleMask(walk_src, mv, inst.decl_node, src.sigs[gid].params, Closure.findFn(fcs, inst.decl_node) != null);
        if (m != 0) try masks.append(gpa, .{ .gid = gid, .eligible = m });
    }
    std.mem.sort(Spec.TemplateMask, masks.items, {}, struct {
        fn lt(_: void, a: Spec.TemplateMask, b: Spec.TemplateMask) bool {
            return a.gid < b.gid;
        }
    }.lt);
    return masks.toOwnedSlice(gpa);
}

fn collectSpecCandidates(gpa: std.mem.Allocator, src: SpecSource, masks: []const Spec.TemplateMask, cands: *std.ArrayList(Spec.Candidate)) !void {
    const V = struct {
        gpa: std.mem.Allocator,
        view: *const SpecView,
        masks: Spec.Table,
        cands: *std.ArrayList(Spec.Candidate),
        targets: std.ArrayList(Spec.Target) = .empty,

        pub fn on(self: *@This(), ev: AstWalk.Event) error{OutOfMemory}!void {
            const e = switch (ev) {
                .enter => |e| e,
                else => return,
            };
            if (e.tag != .call or self.view.tree.nodes[e.idx.int()].lhs == Ast.none) return;
            const site: Spec.Site = if (try AstWalk.instanceOfCall(self.gpa, self.view, e.idx)) |ii|
                .{ .ii = ii, .self_n = 0 }
            else
                Spec.methodSite(self.view, e.idx) orelse return;
            const inst = self.view.instances[site.ii];
            const m = Spec.mask(self.masks, inst.template_gid);
            if (m == 0) return;
            self.targets.clearRetainingCapacity();
            try Spec.targetsOf(self.gpa, self.view, e.idx, site, m, &self.targets);
            if (self.targets.items.len == 0) return;
            try Spec.addCandidate(self.gpa, self.cands, inst.template_gid, site.ii, inst.name.?, self.targets.items);
        }
    };
    const owners = src.base_ids.len + src.instances.len;
    for (0..owners) |k| {
        const is_base = k < src.base_ids.len;
        const inst: ?*const Mono.Instance = if (is_base) null else &src.instances[k - src.base_ids.len];
        const mod = if (inst) |i| i.mod else src.base_mods[k];
        const decl = if (inst) |i| i.decl_node else src.base_decls[k];
        const view: SpecView = .{
            .tree = src.trees[mod],
            .resolutions = src.resolutions[mod],
            .node_types = if (inst) |i| i.node_types else src.node_types[mod],
            .callees = if (inst) |i| i.callees else src.callees[mod],
            .names = src.names,
            .sigs = src.sigs,
            .instances = src.instances,
            .methods = src.methods,
            .closure = .{
                .adapters = &.{},
                .lambda_units = src.lambda_units,
                .owner = if (is_base) .{ .base = src.base_ids[k] } else .{ .instance = @intCast(k - src.base_ids.len) },
            },
        };
        var v: V = .{ .gpa = gpa, .view = &view, .masks = .{ .masks = masks }, .cands = cands };
        defer v.targets.deinit(gpa);
        try AstWalk.walk(.{ .tree = view.tree, .tokens = src.tokens[mod], .source = src.sources[mod] }, decl, &v);
    }
}

/// Lower EVERY function across the whole module `graph` (in parallel, memoized
/// through the codegen cache), then run the serial relink tail. The entry `main`
/// MUST live in the entry module (error otherwise). `res`/`tc` are the whole-graph
/// resolve/typecheck results (program-wide global ids). Caller owns the returned
/// `LinkedProgram` on success.
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
    /// The `-j` jobs knob: the chunk-count basis for the per-fn codegen fan-out (0 =>
    /// host cpu count). Codegen dispatches ~`ncpu` chunks, not one task per
    /// fn, so `-jN` scales. Determinism is unchanged (ranges are disjoint, slots read
    /// back in index order), so this is a perf lever only.
    ncpu: usize,
) !LowerProgramResult {
    const n_mods = graph.modules.len;

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
    const cs = try gpa.alloc([]const Typecheck.Callee, n_mods);
    defer gpa.free(cs);
    for (graph.modules, 0..) |*m, i| {
        trees[i] = m.tree();
        toks[i] = m.tokens;
        srcs[i] = m.source;
        resols[i] = res.resolutions[i];
        ntypes[i] = tc.node_types[i];
        cs[i] = tc.callees[i];
    }

    // Mirrors the resolver's global fn order exactly: a user fn -> {user_fn, its
    // qualified/bare name from the typecheck sig}; a synthetic bodyless builtin (`panic`)
    // -> {builtin,"panic"}. The qualified name MUST equal the typecheck sig name
    // so the fingerprint callee fold lines up with the reloc target.
    const names = try buildGraphNames(gpa, res.fns, tc.sigs);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }

    var fn_decls: std.ArrayList(Ast.Index) = .empty;
    defer fn_decls.deinit(gpa);
    var fn_modules: std.ArrayList(u32) = .empty;
    defer fn_modules.deinit(gpa);
    var lower_ids: std.ArrayList(u32) = .empty;
    defer lower_ids.deinit(gpa);
    var entry_id: ?u32 = null;
    for (res.fns, 0..) |gf, gid| {
        if (!lowerableBase(graph, gf)) continue;
        // The entry `main` is the bare {user_fn,"main"} fn in the entry module.
        if (gf.module == graph.entry_index and std.mem.eql(u8, gf.name, "main"))
            entry_id = @intCast(gid);
        try fn_decls.append(gpa, gf.decl_node);
        try fn_modules.append(gpa, gf.module);
        try lower_ids.append(gpa, @intCast(gid));
    }
    // The instance units follow the base fns (canonical order = the mono tail's
    // sort), each a fully-concrete per-fn codegen unit. Its `node_types`/`args`/sig
    // ride the `instances` table (consumed by `frozenFor`/`graphFnJobInner`).
    const base_count = fn_decls.items.len;
    for (tc.instances) |inst| {
        try fn_decls.append(gpa, inst.decl_node);
        try fn_modules.append(gpa, inst.mod);
    }
    // The SOURCE-LESS auto-derive units follow the instances, in canonical order.
    // Each has a sentinel `decl_node` (never read — the emitter is layout-driven) + the
    // recipe's `mod` (a real module id for the per-module tree/tokens the job's `Frozen`
    // carries but never walks). Routed to `Engine.codegenSynthetic` in `graphFnJobInner`.
    const derive_base = fn_decls.items.len;
    for (tc.derives) |d| {
        try fn_decls.append(gpa, Ast.none);
        try fn_modules.append(gpa, d.mod);
    }
    // The ERASED hash/eq units follow the derive units, in canonical order. Sentinel
    // `decl_node` (layout-driven emitter) + the unit's `mod`. Routed to
    // `Engine.codegenErased` (uncached) in `graphFnJobInner`.
    const erased_base = fn_decls.items.len;
    for (tc.erased_units) |u| {
        try fn_decls.append(gpa, Ast.none);
        try fn_modules.append(gpa, u.mod);
    }
    // The lambda units follow, in canonical `(owner, node)` order, each under its owner's
    // module (and, through `frozenFor`, its owner's tables).
    const lambda_base = fn_decls.items.len;
    for (tc.lambda_units) |u| {
        try fn_decls.append(gpa, u.fn_node);
        try fn_modules.append(gpa, u.mod);
    }
    // The fn-value thunks follow, in canonical `(kind, key)` order. An adapter reads no
    // AST; module 0 only gives its `Frozen` a tree to carry.
    const adapter_base = fn_decls.items.len;
    for (tc.adapters) |_| {
        try fn_decls.append(gpa, Ast.none);
        try fn_modules.append(gpa, 0);
    }
    // The spec units follow, sorted by name, each under its instance's module.
    const specs = if (specialize) try buildSpecTable(gpa, .{
        .trees = trees,
        .tokens = toks,
        .sources = srcs,
        .resolutions = resols,
        .node_types = ntypes,
        .callees = cs,
        .closures = res.closures,
        .names = names,
        .sigs = tc.sigs,
        .instances = tc.instances,
        .methods = tc.methods,
        .lambda_units = tc.lambda_units,
        .base_ids = lower_ids.items,
        .base_mods = fn_modules.items[0..base_count],
        .base_decls = fn_decls.items[0..base_count],
    }) else Spec.Table{};
    defer specs.deinit(gpa);
    const spec_base = fn_decls.items.len;
    for (specs.units) |u| {
        try fn_decls.append(gpa, tc.instances[u.inst].decl_node);
        try fn_modules.append(gpa, tc.instances[u.inst].mod);
    }

    const eid = entry_id orelse return .{ .err = .{
        .message = "-o requires a function named 'main' in the entry module",
        .span = null,
    } };

    // Parameters on main are unsupported (codegen also guards; cleaner up front).
    {
        const em = &graph.modules[graph.entry_index];
        const main_decl = em.nodes[res.fns[eid].decl_node.int()];
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
        const main_ret = tc.sigs[eid].ret;
        if (main_ret.kind != .int and main_ret.kind != .unit) return .{ .err = .{
            .message = "main must return int or ()",
            .span = .{ .start = em.tokens[main_decl.main_token].start, .end = em.tokens[main_decl.main_token].end },
            .module = graph.entry_index,
        } };
    }

    const frame = frameStorage(opt);
    const closure_facts = try buildClosureFacts(gpa, tc.lambda_units, tc.owner_closures, lower_ids.items, base_count, derive_base, frame);
    defer freeClosureFacts(gpa, closure_facts);

    const gf = GraphFrozen{
        .trees = trees,
        .tokens = toks,
        .sources = srcs,
        .resolutions = resols,
        .node_types = ntypes,
        .callees = cs,
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = names,
        .sigs = tc.sigs,
        .fn_decls = fn_decls.items,
        .fn_modules = fn_modules.items,
        .lower_ids = lower_ids.items,
        .base_count = base_count,
        .instances = tc.instances,
        .entry_id = eid,
        .methods = tc.methods,
        .derives = tc.derives,
        .prelude_ids = tc.prelude_ids,
        .char_struct = tc.char_struct,
        .derive_base = derive_base,
        .erased_units = tc.erased_units,
        .erased_base = erased_base,
        .lambda_units = tc.lambda_units,
        .lambda_base = lambda_base,
        .adapters = tc.adapters,
        .adapter_base = adapter_base,
        .specs = specs,
        .spec_base = spec_base,
        .owner_closures = tc.owner_closures,
        .closure_facts = closure_facts,
        .opt = opt,
        .frame = frame,
        .probe = probe,
    };

    const slots = try gpa.alloc(FnSlot, fn_decls.items.len);
    defer gpa.free(slots);
    for (slots) |*s| s.* = .{};

    const Ctx = struct {
        gpa: std.mem.Allocator,
        io: Io,
        cache: Cache,
        target: []const u8,
        mode: Engine.Mode,
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

    var first_err: ?usize = null;
    for (slots, 0..) |s, i| if (s.err != null) {
        if (first_err == null) first_err = i;
    };
    if (first_err) |i| {
        for (slots) |*s| if (s.fc) |*fc| fc.deinit(gpa);
        const e = slots[i].err.?;
        if (e == error.CodegenDiagnostic) if (slots[i].diag) |d| return .{ .err = emitErrorOf(graph, d, fn_modules.items[i], fn_decls.items[i]) };
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
    // Instance units carry their mangled SymName (borrowed from the instance table,
    // which outlives relink). Parallel to the appended instance codegen units.
    for (tc.instances, 0..) |inst, k| lowered_names[base_count + k] = .{ .kind = .user_fn, .name = inst.name.? };
    // Derive units carry their synthetic mangled SymName (borrowed from `tc.derives`).
    for (tc.derives, 0..) |d, k| lowered_names[derive_base + k] = .{ .kind = .user_fn, .name = d.name.? };
    // Erased units carry their synthetic mangled SymName (borrowed from `tc.erased_units`).
    for (tc.erased_units, 0..) |u, k| lowered_names[erased_base + k] = .{ .kind = .user_fn, .name = u.name.? };
    for (tc.lambda_units, 0..) |u, k| lowered_names[lambda_base + k] = .{ .kind = .user_fn, .name = u.name };
    for (tc.adapters, 0..) |a, k| lowered_names[adapter_base + k] = .{ .kind = .user_fn, .name = a.name };
    for (specs.units, 0..) |u, k| lowered_names[spec_base + k] = .{ .kind = .user_fn, .name = u.name };

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
    const out = relink(io, gpa, slots, lowered_names, entry_pos, compiled, cached_n, opt_stats, ir_instrs, desc_plan.items);
    if (link_ns) |lp| {
        const dt = nowNs(io) - link_t0;
        lp.* = if (dt > 0) @intCast(dt) else 0;
    }
    return out;
}

fn nowNs(io: Io) i128 {
    return Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
}

/// Whether global fn `gf` is lowered as a base unit: a fn with a body. A synthetic builtin
/// has no body, an `extern` is referenced only via `names[gid]`, and a generic TEMPLATE's
/// params/ret are `type_var`s with no ABI, so it is lowered only through its instances (an
/// uncalled generic fn emits ZERO codegen units).
fn lowerableBase(graph: *const Graph.Graph, gf: ResolveGraph.GlobalFn) bool {
    if (gf.decl_node == Ast.none) return false;
    if (gf.kind == .import) return false;
    const gm = &graph.modules[gf.module];
    const gdecl = gm.nodes[gf.decl_node.int()];
    return Ast.protoAt(gm.tree(), gdecl.lhs.int()).generic_params.len == 0;
}

/// One per-fn codegen job: build the single-fn `Frozen` view for this fn's owning
/// module, then run the fingerprint→cache→lower path via `Engine.codegen`. The
/// cross-module callee identity + touched layouts ride in through the program-wide
/// `names`/`sigs`/`layouts`, so the fingerprint folds a qualified callee's SymName+sig
/// distinctly with NO engine change.
fn graphFnJob(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    target: []const u8,
    mode: Engine.Mode,
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
    mode: Engine.Mode,
    gf: *const GraphFrozen,
    lower_i: usize,
    slot: *FnSlot,
) !void {
    var fn_node_buf: [1]Ast.Index = undefined;
    const frozen = gf.frozenFor(lower_i, &fn_node_buf);
    const fn_decl = frozen.fn_nodes[0];
    var sym: Link.SymName = undefined;
    var is_entry = false;
    // Thread this fn's typecheck sig (program-wide, indexed by global id) so the
    // fn_decl param/return fold uses the ABI-correct GLOBAL type ids — including a
    // CROSS-MODULE qualified `b: rect.Rect`. A bare-name re-resolution would mis-pick
    // the first same-named type in the merged layout table, missing a pub-type
    // layout edit at the importer (cross-module hole).
    var my_sig: ?Fingerprint.Sig = null;
    if (lower_i >= gf.spec_base) {
        // A spec unit: uncached like a lambda; each caller's key folds its name.
        const engine = Engine.initProbe(cache, mode, gf.probe);
        try engine.codegenClosure(gpa, io, target, &frozen, .{ .spec = @intCast(lower_i - gf.spec_base) }, lower_i, slot);
        return;
    } else if (lower_i >= gf.adapter_base) {
        // A fn-value thunk: source-less and uncached, like an erased unit.
        const engine = Engine.initProbe(cache, mode, gf.probe);
        try engine.codegenClosure(gpa, io, target, &frozen, .{ .adapter = gf.adapters[lower_i - gf.adapter_base] }, lower_i, slot);
        return;
    } else if (lower_i >= gf.lambda_base) {
        // A lambda body: uncached, so its owner's key alone (which folds the lambda's body)
        // decides reuse; the unit is rebuilt from the owner's tables every build.
        const engine = Engine.initProbe(cache, mode, gf.probe);
        try engine.codegenClosure(gpa, io, target, &frozen, .{ .lambda = @intCast(lower_i - gf.lambda_base) }, lower_i, slot);
        return;
    } else if (lower_i >= gf.erased_base) {
        // An ERASED hash/eq unit: layout-driven, uncached (pure over `(ty, kind, layout)` +
        // enumerated in canonical order, so `-jN`-deterministic without a cache key).
        const ei = lower_i - gf.erased_base;
        const engine = Engine.initProbe(cache, mode, gf.probe);
        try engine.codegenErased(gpa, io, target, &frozen, ei, lower_i, slot);
        return;
    } else if (lower_i >= gf.derive_base) {
        // A SOURCE-LESS auto-derive unit: no fn_decl / no AST fingerprint. Its
        // identity is the recipe's synthetic mangled name; route to the synthetic engine
        // entry which builds a NON-AST fingerprint from the recipe.
        const di = lower_i - gf.derive_base;
        const engine = Engine.initProbe(cache, mode, gf.probe);
        try engine.codegenSynthetic(gpa, io, target, &frozen, di, lower_i, slot);
        return;
    } else if (lower_i < gf.base_count) {
        const gid = gf.lower_ids[lower_i];
        sym = gf.names[gid];
        is_entry = gid == gf.entry_id;
        my_sig = if (gid < gf.sigs.len) gf.sigs[gid] else null;
    } else {
        // A monomorphized instance: its identity is the mangled SymName and its sig
        // is the SUBSTITUTED (fully concrete) params/ret. Never the entry.
        const inst = gf.instances[lower_i - gf.base_count];
        sym = .{ .kind = .user_fn, .name = inst.name.? };
        my_sig = .{ .kind = .user_fn, .name = inst.name.?, .params = inst.params, .ret = inst.ret };
    }

    // The cross-module callee identity + touched layouts ride in through the
    // program-wide `names`/`sigs`/`layouts` of this fn's `frozen` view, so the
    // fingerprint folds a qualified callee distinctly with NO engine change.
    const engine = Engine.initProbe(cache, mode, gf.probe);
    try engine.codegen(gpa, io, target, &frozen, fn_decl, sym, is_entry, my_sig, lower_i, slot);
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

/// `--emit ir` for a whole-graph build: run the `lower` stage over every fn across
/// the graph and render the deterministic IR text (each fn rendered against its
/// owning module's tree). Caller owns the returned text on success.
pub fn renderGraphIr(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const Typecheck.GraphResult,
    opt: Opt.Config,
    specialize: bool,
) !IrResult {
    const names = try buildGraphNames(gpa, res.fns, tc.sigs);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }

    const specs = if (specialize) try renderSpecTable(gpa, graph, res, tc, names) else Spec.Table{};
    defer specs.deinit(gpa);
    const frame = frameStorage(opt);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    errdefer aw.deinit();
    var diags: std.ArrayList(CodegenIr.Diagnostic) = .empty;
    defer diags.deinit(gpa);

    var first = true;
    // The unit that produced `diags[0]`: noted before each unit lowers, until one fails.
    var fail: struct { mod: u32 = 0, decl: Ast.Index = Ast.none } = .{};
    for (res.fns, 0..) |gf, gid| {
        if (!lowerableBase(graph, gf)) continue;
        const m = &graph.modules[gf.module];
        if (diags.items.len == 0) fail = .{ .mod = gf.module, .decl = gf.decl_node };
        const is_entry = gf.module == graph.entry_index and std.mem.eql(u8, gf.name, "main");
        const in = lower.Inputs{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[gf.module],
            .node_types = tc.node_types[gf.module],
            .callees = tc.callees[gf.module],
            .layouts = tc.layouts,
            .enum_layouts = tc.enum_layouts,
            .names = names,
            .sig = if (gid < tc.sigs.len) tc.sigs[gid] else null,
            .instances = tc.instances,
            .sigs = tc.sigs,
            .methods = tc.methods,
            .derives = tc.derives,
            .prelude_ids = tc.prelude_ids,
            .char_struct = tc.char_struct,
            .closure = closureFor(tc.adapters, tc.lambda_units, tc.owner_closures, .{ .base = @intCast(gid) }, specs, &.{}, frame),
        };
        var opt_st: Opt.Stats = .{};
        var func = try Engine.lowerFitted(gpa, in, opt, Engine.FnBody{ .decl = gf.decl_node, .sym = names[gid], .is_entry = is_entry }, &diags, &opt_st);
        defer func.deinit(gpa);
        if (!first) try aw.writer.writeAll("\n");
        try Ir.render(&aw.writer, &func, in.layouts, in.enum_layouts);
        first = false;
    }

    // Render each monomorphized instance as a concrete unit (its own node_types +
    // substituted sig), in canonical order — the same units codegen lowers.
    for (tc.instances, 0..) |inst, k| {
        const m = &graph.modules[inst.mod];
        if (diags.items.len == 0) fail = .{ .mod = inst.mod, .decl = inst.decl_node };
        const sym = Link.SymName{ .kind = .user_fn, .name = inst.name.? };
        const in = lower.Inputs{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[inst.mod],
            .node_types = inst.node_types,
            .callees = inst.callees,
            .layouts = tc.layouts,
            .enum_layouts = tc.enum_layouts,
            .names = names,
            .sig = .{ .kind = .user_fn, .name = inst.name.?, .params = inst.params, .ret = inst.ret },
            .instances = tc.instances,
            .sigs = tc.sigs,
            .methods = tc.methods,
            .derives = tc.derives,
            .prelude_ids = tc.prelude_ids,
            .char_struct = tc.char_struct,
            .closure = closureFor(tc.adapters, tc.lambda_units, tc.owner_closures, .{ .instance = @intCast(k) }, specs, &.{}, frame),
        };
        var opt_st: Opt.Stats = .{};
        var func = try Engine.lowerFitted(gpa, in, opt, Engine.FnBody{ .decl = inst.decl_node, .sym = sym, .is_entry = false }, &diags, &opt_st);
        defer func.deinit(gpa);
        if (!first) try aw.writer.writeAll("\n");
        try Ir.render(&aw.writer, &func, in.layouts, in.enum_layouts);
        first = false;
    }

    // Render each SOURCE-LESS auto-derive unit via the layout-walking emitter, in
    // canonical order — the same units codegen lowers, so `--emit ir` shows exactly one
    // `Eq$eq$…`/`Ord$cmp$…` unit per derived type (the zero-codegen / override proofs read
    // this). Dispatch on the recipe kind, mirroring `Engine.lowerSynthetic`.
    for (tc.derives) |d| {
        const m = &graph.modules[d.mod];
        if (diags.items.len == 0) fail = .{ .mod = d.mod };
        const sym = Link.SymName{ .kind = .user_fn, .name = d.name.? };
        const in = lower.Inputs{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[d.mod],
            .node_types = &.{},
            .callees = &.{},
            .layouts = tc.layouts,
            .enum_layouts = tc.enum_layouts,
            .names = names,
            .sig = .{ .kind = .user_fn, .name = d.name.?, .params = d.params, .ret = d.ret },
            .instances = tc.instances,
            .sigs = tc.sigs,
            .methods = tc.methods,
            .derives = tc.derives,
            .prelude_ids = tc.prelude_ids,
            .char_struct = tc.char_struct,
            .closure = .none,
        };
        var func = try DeriveEmit.lower(gpa, in, d, sym, &diags);
        defer func.deinit(gpa);
        var opt_st: Opt.Stats = .{};
        try Opt.run(gpa, &func, opt, &opt_st);
        if (!first) try aw.writer.writeAll("\n");
        try Ir.render(&aw.writer, &func, in.layouts, in.enum_layouts);
        first = false;
    }

    // Render each ERASED hash/eq unit via the key-by-pointer emitter, in canonical order —
    // the same units codegen lowers, so `--emit ir` shows one `descriptor$hash$…`/
    // `descriptor$eq$…` per hashable descriptor type.
    for (tc.erased_units) |u| {
        const m = &graph.modules[u.mod];
        if (diags.items.len == 0) fail = .{ .mod = u.mod };
        const sym = Link.SymName{ .kind = .user_fn, .name = u.name.? };
        const in = lower.Inputs{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[u.mod],
            .node_types = &.{},
            .callees = &.{},
            .layouts = tc.layouts,
            .enum_layouts = tc.enum_layouts,
            .names = names,
            .sig = null,
            .instances = tc.instances,
            .sigs = tc.sigs,
            .methods = tc.methods,
            .derives = tc.derives,
            .prelude_ids = tc.prelude_ids,
            .char_struct = tc.char_struct,
            .closure = .none,
        };
        var func = try DeriveEmit.lowerErased(gpa, in, u.ty, u.kind, sym, &diags);
        defer func.deinit(gpa);
        var opt_st: Opt.Stats = .{};
        try Opt.run(gpa, &func, opt, &opt_st);
        if (!first) try aw.writer.writeAll("\n");
        try Ir.render(&aw.writer, &func, in.layouts, in.enum_layouts);
        first = false;
    }

    // Render each lambda unit under its owner's tables, in canonical order — the same
    // units codegen lowers.
    for (tc.lambda_units, 0..) |u, ui| {
        const m = &graph.modules[u.mod];
        if (diags.items.len == 0) fail = .{ .mod = u.mod, .decl = u.fn_node };
        const sym = Link.SymName{ .kind = .user_fn, .name = u.name };
        const inst: ?*const Mono.Instance = switch (u.owner) {
            .instance => |k| &tc.instances[k],
            .base => null,
        };
        const in = lower.Inputs{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[u.mod],
            .node_types = if (inst) |i| i.node_types else tc.node_types[u.mod],
            .callees = if (inst) |i| i.callees else tc.callees[u.mod],
            .layouts = tc.layouts,
            .enum_layouts = tc.enum_layouts,
            .names = names,
            .sig = null,
            .instances = tc.instances,
            .sigs = tc.sigs,
            .methods = tc.methods,
            .derives = tc.derives,
            .prelude_ids = tc.prelude_ids,
            .char_struct = tc.char_struct,
            .closure = closureFor(tc.adapters, tc.lambda_units, tc.owner_closures, u.owner, specs, &.{}, frame),
        };
        var opt_st: Opt.Stats = .{};
        var func = try Engine.lowerFitted(gpa, in, opt, Engine.LambdaBody{ .ui = @intCast(ui), .sym = sym }, &diags, &opt_st);
        defer func.deinit(gpa);
        if (!first) try aw.writer.writeAll("\n");
        try Ir.render(&aw.writer, &func, in.layouts, in.enum_layouts);
        first = false;
    }

    // Render each fn-value thunk, in canonical order — the same units codegen lowers.
    for (tc.adapters) |a| {
        const m = &graph.modules[0];
        if (diags.items.len == 0) fail = .{};
        const sym = Link.SymName{ .kind = .user_fn, .name = a.name };
        const in = lower.Inputs{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[0],
            .node_types = &.{},
            .callees = &.{},
            .layouts = tc.layouts,
            .enum_layouts = tc.enum_layouts,
            .names = names,
            .sig = null,
            .instances = tc.instances,
            .sigs = tc.sigs,
            .methods = tc.methods,
            .derives = tc.derives,
            .prelude_ids = tc.prelude_ids,
            .char_struct = tc.char_struct,
            .closure = .none,
        };
        var func = try AdapterEmit.lowerAdapter(gpa, in, a, sym, &diags);
        defer func.deinit(gpa);
        var opt_st: Opt.Stats = .{};
        try Opt.run(gpa, &func, opt, &opt_st);
        if (!first) try aw.writer.writeAll("\n");
        try Ir.render(&aw.writer, &func, in.layouts, in.enum_layouts);
        first = false;
    }

    // Render each spec unit under its instance's tables, in name order — the same units
    // codegen lowers.
    for (specs.units) |u| {
        const inst = &tc.instances[u.inst];
        const m = &graph.modules[inst.mod];
        if (diags.items.len == 0) fail = .{ .mod = inst.mod, .decl = inst.decl_node };
        const sym = Link.SymName{ .kind = .user_fn, .name = u.name };
        const in = lower.Inputs{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[inst.mod],
            .node_types = inst.node_types,
            .callees = inst.callees,
            .layouts = tc.layouts,
            .enum_layouts = tc.enum_layouts,
            .names = names,
            .sig = .{ .kind = .user_fn, .name = u.name, .params = inst.params, .ret = inst.ret },
            .instances = tc.instances,
            .sigs = tc.sigs,
            .methods = tc.methods,
            .derives = tc.derives,
            .prelude_ids = tc.prelude_ids,
            .char_struct = tc.char_struct,
            .closure = closureFor(tc.adapters, tc.lambda_units, tc.owner_closures, .{ .instance = u.inst }, specs, u.targets, frame),
        };
        var func = try lower.lowerFn(gpa, in, inst.decl_node, sym, false, &diags);
        defer func.deinit(gpa);
        var opt_st: Opt.Stats = .{};
        try Opt.run(gpa, &func, opt, &opt_st);
        if (!first) try aw.writer.writeAll("\n");
        try Ir.render(&aw.writer, &func, in.layouts, in.enum_layouts);
        first = false;
    }

    if (diags.items.len > 0) {
        aw.deinit();
        return .{ .err = emitErrorOf(graph, diags.items[0], fail.mod, fail.decl) };
    }
    var list = aw.toArrayList();
    return .{ .ok = try list.toOwnedSlice(gpa) };
}

/// `buildSpecTable` over a graph, for `renderGraphIr`, which has no `GraphFrozen`.
fn renderSpecTable(gpa: std.mem.Allocator, graph: *const Graph.Graph, res: *const ResolveGraph.GraphResult, tc: *const Typecheck.GraphResult, names: []const Link.SymName) !Spec.Table {
    const n_mods = graph.modules.len;
    const trees = try gpa.alloc(Ast.Tree, n_mods);
    defer gpa.free(trees);
    const toks = try gpa.alloc([]const Token, n_mods);
    defer gpa.free(toks);
    const srcs = try gpa.alloc([]const u8, n_mods);
    defer gpa.free(srcs);
    for (graph.modules, 0..) |*m, i| {
        trees[i] = m.tree();
        toks[i] = m.tokens;
        srcs[i] = m.source;
    }
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var mods: std.ArrayList(u32) = .empty;
    defer mods.deinit(gpa);
    var decls: std.ArrayList(Ast.Index) = .empty;
    defer decls.deinit(gpa);
    for (res.fns, 0..) |gf, gid| {
        if (!lowerableBase(graph, gf)) continue;
        try ids.append(gpa, @intCast(gid));
        try mods.append(gpa, gf.module);
        try decls.append(gpa, gf.decl_node);
    }
    return buildSpecTable(gpa, .{
        .trees = trees,
        .tokens = toks,
        .sources = srcs,
        .resolutions = res.resolutions,
        .node_types = tc.node_types,
        .callees = tc.callees,
        .closures = res.closures,
        .names = names,
        .sigs = tc.sigs,
        .instances = tc.instances,
        .methods = tc.methods,
        .lambda_units = tc.lambda_units,
        .base_ids = ids.items,
        .base_mods = mods.items,
        .base_decls = decls.items,
    });
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
