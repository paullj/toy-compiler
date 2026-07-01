//! Whole-graph codegen + link tail: turns a resolved+typechecked module graph into
//! a linked Mach-O image (or rendered IR text). The per-fn codegen fan-out lowers
//! each function against a single-fn `Frozen` view (thread-safe, program-wide tables
//! + the fn's owning-module arrays), then the SERIAL relink tail interns strings,
//! rebases cross-segment relocs, links, and signs — the order-sensitive
//! assemble → applyDataRelocs → sign-last pipeline. Distinct from the per-file
//! front-end (`Driver.run`/`pipeline`, discover → lex/parse/check).
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Token = @import("../ast/Token.zig").Token;
const Ast = @import("../ast/Ast.zig");
const Cache = @import("../query/Cache.zig");
const Engine = @import("../query/Engine.zig");
const Resolve = @import("../resolve.zig");
const Typecheck = @import("../types.zig");
const Graph = @import("Graph.zig");
const ResolveGraph = @import("../resolve_graph.zig");
const CodegenIr = @import("../codegen/CodegenIr.zig");
const Ir = @import("../ir/Ir.zig");
const Opt = @import("../opt/Opt.zig");
const lower = @import("../lower.zig");
const Fingerprint = @import("../query/Fingerprint.zig");
const Link = @import("../link/Link.zig");
const link = @import("../link/emit.zig");

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
    /// Incremental counters: how many functions were freshly lowered vs served
    /// from the codegen cache this build. Surfaced via `--codegen-stats`.
    codegen_compiled: usize = 0,
    codegen_cached: usize = 0,
    /// dual-metric: summed opt counters over freshly-lowered fns, the total
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
    layouts: []const Typecheck.Layout,
    enum_layouts: []const Typecheck.EnumLayout,
    names: []const Link.SymName,
    /// A one-element slice naming this job's fn (`frozenFor` points it at a stack buf).
    fn_nodes: []const Ast.Index,
    sigs: []const Fingerprint.Sig,
    /// Opt level / pass selection. Mixed into the codegen cache key so
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

/// Frozen, read-only inputs shared by every codegen job in a WHOLE-GRAPH build.
/// Per-module arrays (trees/tokens/sources/resolutions/node_types) are indexed by
/// module id; `layouts`/`enum_layouts`/`names`/`sigs` are PROGRAM-WIDE (one global
/// id space). `fn_decls`/`fn_modules` are parallel to the global fn id space (one
/// entry per global fn; `print` excluded — only real fn bodies are lowered). A
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
    for (graph.modules, 0..) |*m, i| {
        trees[i] = m.tree();
        toks[i] = m.tokens;
        srcs[i] = m.source;
        resols[i] = res.resolutions[i];
        ntypes[i] = tc.node_types[i];
    }

    // Mirrors the resolver's global fn order exactly: a user fn -> {user_fn, its
    // qualified/bare name from the typecheck sig}; the synthetic bodyless `print`
    // -> {builtin,"print"}. The qualified name MUST equal the typecheck sig name
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
        const main_decl = em.nodes[res.fns[eid].decl_node.int()];
        const main_proto = Ast.protoAt(em.tree(), main_decl.lhs.int());
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
    const gid = gf.lower_ids[lower_i];
    const sym = gf.names[gid];
    const is_entry = gid == gf.entry_id;
    // Thread this fn's typecheck sig (program-wide, indexed by global id) so the
    // fn_decl param/return fold uses the ABI-correct GLOBAL type ids — including a
    // CROSS-MODULE qualified `b: rect.Rect`. A bare-name re-resolution would mis-pick
    // the first same-named type in the merged layout table, missing a pub-type
    // layout edit at the importer (cross-module M9 / TOP-RISK-#1 hole).
    const my_sig: ?Fingerprint.Sig = if (gid < gf.sigs.len) gf.sigs[gid] else null;

    // The cross-module callee identity + touched layouts ride in through the
    // program-wide `names`/`sigs`/`layouts` of this fn's `frozen` view, so the
    // fingerprint folds a qualified callee distinctly with NO engine change.
    // tmp_tag = `lower_i`.
    const engine = Engine.initProbe(cache, mode, gf.probe);
    try engine.codegen(gpa, io, target, &frozen, fn_decl, sym, is_entry, my_sig, lower_i, slot);
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
