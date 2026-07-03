//! Type checking over the whole module graph (each module's `Ast.Tree` plus its
//! resolution array), the semantic pass that runs after name resolution succeeded
//! (a name error would otherwise poison every type that depends on it). It is
//! query-fed and multi-phase: a global Pass-A builds the program-wide signature
//! and struct/enum-layout tables (id-ordered for determinism), then a per-function
//! Pass-C (`BodyChecker`, the bodies checked as independent parallel units) infers
//! a `Type` for every expression node and enforces the statement-level rules:
//!
//!   * `name := expr`     — the local takes the initializer's type (() is an
//!                          error: there is nothing to bind).
//!   * `name = expr`      — the value's type must match the variable's type.
//!   * `return expr?`     — the (or () , when bare) type must match the
//!                          function's declared return type.
//!   * `callee(args...)`  — the callee must be a function, and the argument count
//!                          and per-argument types must match its parameters.
//!
//! Expression types: integer/string/bool literals are obvious; an identifier
//! takes its declaration's type (via the resolution); a unary `-`/`!` constrains
//! its operand; binary `+ - * /` are int→int, comparisons/`== !=` yield bool.
//!
//! Errors never cascade: the moment an operand is `invalid` (the poison type) the
//! containing expression silently becomes `invalid` too, so each mistake yields
//! exactly one diagnostic at its origin. Type references (`int`, `bool`) are
//! `identifier` nodes whose text we map straight onto a `Type`.

const std = @import("std");
const Token = @import("ast/Token.zig").Token;
const Ast = @import("ast/Ast.zig");
const Resolve = @import("resolve.zig");
const Resolution = @import("symbols/Resolution.zig").Resolution;
const Sig = @import("symbols/Sig.zig").Sig;
const symbols = @import("symbols/Sym.zig");
const Engine = @import("query/Engine.zig");
const Io = std.Io;

const Typecheck = @This();

/// The type algebra + layout engine (the `Type`/`Kind`/`Layout` value types and
/// the struct/enum layout-cycle state machine) live in their own deep module. The
/// checker drives it via `LayoutEngine.layoutStruct`/`layoutEnum` (id-ordered, from
/// `checkGraph`/`runGraph`) and consumes the laid syms through the re-exported aliases.
const LayoutEngine = @import("layout/Engine.zig");

// Re-export the algebra/layout value types so every downstream importer keeps
// reading `Typecheck.Type`/`.Layout`/`.EnumLayout` etc. unchanged.
pub const Kind = LayoutEngine.Kind;
pub const Type = LayoutEngine.Type;
pub const Layout = LayoutEngine.Layout;
pub const VariantForm = LayoutEngine.VariantForm;
pub const VariantLayout = LayoutEngine.VariantLayout;
pub const EnumLayout = LayoutEngine.EnumLayout;

// Internal aliases for the checker's own scratch tables. These ARE the engine's
// table types (the `Model` aliases them; `BodyChecker` reads `.state`/`.poisoned`/
// `.field_*`/`.variants` through them at fingerprint + exhaustiveness time).
const StructSym = LayoutEngine.StructSym;
const VariantSym = LayoutEngine.VariantSym;
const EnumSym = LayoutEngine.EnumSym;

/// Source spelling of a type reference → `Type`. Anything else is unknown (a
/// struct name, or an error). The unit type `()` is spelled with parens, not an
/// identifier, so it is NOT here (handled in `typeFromNode` via `literal_unit`).
const type_names = std.StaticStringMap(Type).initComptime(.{
    .{ "int", Type.int },
    .{ "bool", Type.bool },
    .{ "str", Type.str },
});

/// Shared type-reference resolution + token/diagnostic helpers, generic over the
/// checker (`*Typecheck` for Pass-A or `*BodyChecker` for Pass-C). Both expose the
/// same cursor (`tree`/`tokens`/`source`/`graph_mod`/`sink`) and table accessors
/// (`graphCtx`/`structSyms`/`enumSyms`/`activeStructMap`/`activeEnumMap`), so this
/// is the single home for the logic AND its diagnostic strings — the two passes
/// can't drift (diagnostic-string drift is otherwise gated only by `check.sh`).
pub const refs = struct {
    const err_unknown_type = "unknown type '{s}'";
    const err_unknown_module = "unknown module '{s}'";
    const err_module_no_type = "module has no type '{s}'";

    pub fn nameText(self: anytype, tok: u32) []const u8 {
        return self.tokens[tok].text(self.source);
    }

    pub fn byteOf(self: anytype, tok: u32) u32 {
        return self.tokens[tok].start;
    }

    /// Human-readable name of a type (a struct/enum's declared name, else its kind tag).
    pub fn typeName(self: anytype, ty: Type) []const u8 {
        if (ty.kind == .@"struct" and ty.struct_id < self.structSyms().len)
            return self.structSyms()[ty.struct_id].name;
        if (ty.kind == .@"enum" and ty.enum_id < self.enumSyms().len)
            return self.enumSyms()[ty.enum_id].name;
        return @tagName(ty.kind);
    }

    /// Map a type-reference node (an `identifier`, a `literal_unit` for `()`, or in
    /// graph mode a qualified `mod.Type` `field_access`) to a `Type`.
    pub fn typeFromNode(self: anytype, type_node: Ast.Index) Type {
        if (type_node == Ast.none) return Type.unit;
        const tn = self.tree.nodes[type_node.int()];
        if (tn.tag == .literal_unit) return Type.unit; // explicit `-> ()` / `p: ()`
        // A qualified cross-module type-ref `mod.Type` parses as a field_access whose
        // receiver binds to a `.module`. Resolve it against the owning module's tables.
        if (tn.tag == .field_access) return refs.typeFromQualified(self, type_node, tn);
        const tok = tn.main_token;
        const name = refs.nameText(self, tok);
        if (type_names.get(name)) |b| return b;
        if (self.activeStructMap().get(name)) |id| return Type.structT(id);
        if (self.activeEnumMap().get(name)) |id| return Type.enumT(id);
        self.sink.emitFmtCode(.T0001, refs.byteOf(self, tok), err_unknown_type, .{name}) catch {};
        return .invalid;
    }

    /// Resolve a qualified `mod.Type` type-reference (a `field_access` in type
    /// position; the receiver binds to a `.module`) to the owning module's GLOBAL
    /// struct/enum id. Graph mode only. Visibility was already enforced by
    /// `resolve_graph`, so a private type here is a defensive `invalid`.
    pub fn typeFromQualified(self: anytype, node_idx: Ast.Index, n: Ast.Node) Type {
        _ = node_idx;
        const g = self.graphCtx();
        const recv = self.tree.nodes[n.lhs.int()];
        if (recv.tag != .identifier) return .invalid;
        const recv_name = refs.nameText(self, recv.main_token);
        const target = g.namespaceOfIn(self.graph_mod, recv_name) orelse {
            self.sink.emitFmtCode(.T0002, refs.byteOf(self, recv.main_token), err_unknown_module, .{recv_name}) catch {};
            return .invalid;
        };
        const member = refs.nameText(self, n.main_token);
        if (g.mods[target].struct_ids.get(member)) |id| return Type.structT(id);
        if (g.mods[target].enum_ids.get(member)) |id| return Type.enumT(id);
        self.sink.emitFmtCode(.T0003, refs.byteOf(self, n.main_token), err_module_no_type, .{member}) catch {};
        return .invalid;
    }
};

/// A reported problem. Same shape as `Parser`/`Resolve` diagnostics so the
/// driver/CLI can render either uniformly (byte offset → line:col).
pub const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;
const DiagnosticSink = @import("diagnostics/Sink.zig");
const codes = @import("diagnostics/codes.zig");

/// One module's parsed + resolved inputs for the graph typecheck.
pub const GraphModuleInput = struct {
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const Resolution,
    /// Import namespace name → imported module id, owned by the caller.
    namespaces: std.StringHashMapUnmanaged(u32),
};

/// A global function descriptor (parallel to the resolver's global fn table).
pub const GraphFnInput = struct {
    /// The fn's decl node (`Ast.none` for the bodyless builtin `print`; identify
    /// builtin-ness by `kind`, never by this sentinel).
    decl_node: Ast.Index,
    /// `.builtin` for the synthetic `print`, else `.user_fn` (from the resolver).
    kind: symbols.SymKind,
    /// Owning module id; meaningless for the homeless `print` builtin.
    module: u32,
    /// Whether this fn is `pub` (drives the pub-signature-coherence check).
    is_pub: bool,
    /// Module-qualified symbol name (used in the coherence diagnostic).
    name: []const u8,
};

/// The whole-graph typecheck output. Caller owns it; free with `GraphResult.deinit`.
/// `node_types` is per-module; `layouts`/`enum_layouts`/`sigs` are PROGRAM-WIDE
/// (global ids), exactly as the lowering stage's Frozen needs.
pub const GraphResult = struct {
    /// One `[]Type` per module (parallel to that module's node array).
    node_types: [][]Type,
    /// Diagnostics tagged with their owning module id via `Diagnostic.scope`.
    diags: []Diagnostic,
    owned_msgs: [][]u8,
    /// One Sig per GLOBAL fn id (parallel to the resolver's global fn table).
    /// `sigs[i].name` is BORROWED from the resolve `fns[i].name` table; its
    /// lifetime is tied to the sibling resolve result (they are torn down
    /// together). NEVER freed through a Sig — `deinit` frees only `sigs[].params`.
    sigs: []Sig,
    /// Program-wide struct table (one Layout per global struct id).
    layouts: []Layout,
    /// Program-wide enum table (one EnumLayout per global enum id).
    enum_layouts: []EnumLayout,

    pub fn deinit(self: *GraphResult, gpa: std.mem.Allocator) void {
        for (self.node_types) |nt| gpa.free(nt);
        gpa.free(self.node_types);
        gpa.free(self.diags);
        for (self.owned_msgs) |m| gpa.free(m);
        gpa.free(self.owned_msgs);
        for (self.sigs) |s| gpa.free(@constCast(s.params));
        gpa.free(self.sigs);
        for (self.layouts) |l| {
            gpa.free(l.name);
            for (l.field_names) |fn_| gpa.free(fn_);
            gpa.free(l.field_names);
            gpa.free(l.field_types);
            gpa.free(l.offsets);
        }
        gpa.free(self.layouts);
        for (self.enum_layouts) |e| {
            gpa.free(e.name);
            for (e.variants) |v| {
                for (v.field_names) |fn_| gpa.free(fn_);
                gpa.free(v.field_names);
                gpa.free(v.field_types);
                gpa.free(v.offsets);
            }
            gpa.free(e.variants);
        }
        gpa.free(self.enum_layouts);
        self.* = undefined;
    }
};

/// A top-level function's signature, decoded once up front so calls can be
/// checked against it (and forward references work).
pub const FnSym = struct {
    /// The fn's decl node (`Ast.none` for the bodyless builtin `print`; identify
    /// builtin-ness by `kind`, never by this sentinel).
    decl_node: Ast.Index,
    /// `.builtin` for the synthetic `print`, else `.user_fn`.
    kind: symbols.SymKind,
    params: []Type,
    ret: Type,
    /// Owning module id (graph mode). 0 in single-file mode. The check loops
    /// switch the active tree/tokens/source to this module before checking.
    mod: u32 = 0,
};

/// Per-construct context, pushed/popped as bodies are entered. A label-
/// addressable stack: `kind` distinguishes a
/// value-yielding `loop`, a `()`-statement `while`/`for`, and a value-yielding
/// labeled bare block. `is_value` is true for `loop` and `labeled_block`. `label`
/// is the construct's label name (or null when unlabeled) and `construct_node` is
/// the inner construct node a `break @L`/`continue @L` matches against. `join`
/// accumulates the merge of all value-break sites (starting at `never`).
const CtxKind = enum { loop, while_for, labeled_block };
pub const LoopCtx = struct {
    kind: CtxKind,
    label: ?[]const u8,
    construct_node: Ast.Index,
    is_value: bool,
    join: Type,
    saw_value_break: bool,
    saw_bare_break: bool,
};

gpa: std.mem.Allocator,
tree: Ast.Tree,
tokens: []const Token,
source: []const u8,
resolutions: []const Resolution,

node_types: []Type,
/// Owns the diagnostic list + message lifetimes. In graph mode the owning module
/// rides in each `Diagnostic.scope` (stamped via `gphSelect` -> `sink.setScope`);
/// single-file leaves every scope `NO_SCOPE`.
sink: DiagnosticSink,
/// Graph mode only: per-module node_types slices. `gphSelect` redirects the
/// active `node_types` to the selected module's slice. Null single-file.
gph_node_types: ?[][]Type = null,

/// Function table, parallel to `Resolve`'s `func` indices: the resolver assigns
/// function indices in source order, and so do we (Pass A below).
fns: std.ArrayList(FnSym),

/// The struct table: one `StructSym` per struct id. The bare-name → global-id
/// map lives per-module in the graph ctx (`activeStructMap`).
structs: std.ArrayList(StructSym),

/// The enum table: one `EnumSym` per enum id. Bare-name → id lives per-module in
/// the graph ctx (`activeEnumMap`).
enums: std.ArrayList(EnumSym),

/// Graph context. Always set in practice: `checkGraph` is the ONE entry and
/// it drives one shared `Typecheck` across the whole module graph (a lone source
/// file is the trivial one-module graph). The `structs`/`enums`/`fns` tables are
/// PROGRAM-WIDE (global ids); each module's bare-name → global-id bindings live
/// in the ctx (`activeStructMap`/`activeEnumMap`, keyed by `graph_mod`). The
/// context resolves a qualified `mod.Type` / `mod.Enum` receiver to the owning
/// module's tables. Pre-collect + layout happen once; only Pass B runs per fn.
graph: *GraphCtx,

/// The active module being type-checked / laid out (graph mode). Single-file
/// leaves it 0. Used to pick the import-namespace table for qualified receivers.
graph_mod: u32 = 0,

/// Graph-mode: one qualified fn name per global fn id (parallel to `t.fns`), set
/// from the `GraphFnInput` table before Pass B so the `signature(fn)` node id
/// folds the IDENTICAL qualified name the codegen callee-sig fold uses (codegen's
/// `signature` node id is `Wyhash("SGNM", SymName.name)`, and the graph SymName.name
/// IS `fns[i].name`). Null single-file (the decl-token name is used instead).
gph_fn_names: ?[]const []const u8 = null,

/// The runtime the per-fn body region (Pass C) dispatches onto. Set in graph
/// mode (carries the `-j N` worker cap from the driver's own pool). The body region
/// fans out per-fn body checks via `Engine.fanOut(io, ...)` whenever a pool is present
/// (`io != null`). `.limited(0)` (`-j1`) drives every unit onto the inline serial path
/// — the byte-identity baseline. `io == null` (single-file internal callers + inline
/// tests) is the only serial trigger; both dispatch modes feed the SAME slots +
/// merge+sort, so the result is byte-identical regardless.
io: ?Io = null,

/// The `-j` jobs knob: the chunk-count basis for the per-fn Pass-C body fan-out
/// (`checkBodies`), 0 => host cpu count. Body checks dispatch ~`ncpu`
/// contiguous chunks (each looping its fns serially) instead of one task per fn, so
/// `-jN` scales instead of drowning in per-task overhead. Only meaningful when
/// `io != null` (the parallel dispatch path); determinism is unchanged (fn-id-ordered
/// merge + one stable sort), so this is a perf lever only.
ncpu: usize = 0,

/// The graph context the orchestrator hands the shared `Typecheck`. It owns the
/// per-module bare-name maps + the import namespaces; `Typecheck` borrows it.
pub const GraphCtx = struct {
    /// One entry per module (index = graph module id). MUTABLE: type registration
    /// fills each module's `struct_ids`/`enum_ids` in place (a put that grows
    /// reallocs the map header, which must be reflected in the ctx, not a copy).
    mods: []ModuleCtx,

    pub const ModuleCtx = struct {
        /// Per-module tree view (selected as the active tree when checking/laying
        /// out a decl owned by this module).
        tree: Ast.Tree,
        tokens: []const Token,
        source: []const u8,
        /// This module's resolution array (parallel to its node array).
        resolutions: []const Resolution,
        /// Bare struct name → GLOBAL struct id (this module's own decls only).
        struct_ids: std.StringHashMapUnmanaged(u32) = .empty,
        /// Bare enum name → GLOBAL enum id.
        enum_ids: std.StringHashMapUnmanaged(u32) = .empty,
        /// Import namespace name → imported module id (graph module id).
        namespaces: std.StringHashMapUnmanaged(u32) = .empty,
    };

    /// Resolve an import namespace receiver name in module `mod` to the imported
    /// module's id, or null if the name is not a namespace there.
    pub fn namespaceOfIn(c: *const GraphCtx, mod: u32, recv_name: []const u8) ?u32 {
        return c.mods[mod].namespaces.get(recv_name);
    }
};

/// The immutable, whole-program model frozen after Pass A: the fn signature
/// table + the laid-out struct/enum tables + their bare-name maps + the graph
/// context. Pass C reads it READ-ONLY through every `BodyChecker`, so per-fn body
/// checking can run against one shared frozen snapshot (the enabler for the
/// parallel fan-out). The slices alias the still-live `Typecheck` ArrayLists,
/// which are not mutated during Pass C. `graph`/`gph_fn_names` are borrowed.
pub const Model = struct {
    fns: []const FnSym,
    structs: []const StructSym,
    enums: []const EnumSym,
    graph: *GraphCtx,
    gph_fn_names: ?[]const []const u8,
};

const BodyChecker = @import("BodyChecker.zig").BodyChecker;

/// Freeze the Pass-A tables into a read-only `Model`. The slices alias the live
/// `Typecheck` ArrayLists; valid for as long as those are not mutated (Pass C).
fn buildModel(t: *Typecheck) Model {
    return .{
        .fns = t.fns.items,
        .structs = t.structs.items,
        .enums = t.enums.items,
        .graph = t.graph,
        .gph_fn_names = t.gph_fn_names,
    };
}

/// Construct a `BodyChecker` for fn `f`, wiring its cursor to the fn's owning
/// module (`graph.mods[f.mod]` + the module's node_types slice). The per-fn
/// scratch starts empty.
fn bodyCheckerFor(t: *const Typecheck, model: *const Model, f: FnSym) BodyChecker {
    var bc: BodyChecker = .{
        .model = model,
        .gpa = t.gpa,
        .tree = t.tree,
        .tokens = t.tokens,
        .source = t.source,
        .resolutions = t.resolutions,
        .node_types = t.node_types,
        .graph_mod = 0,
        .sink = DiagnosticSink.init(t.gpa),
        .gph_fn_names = t.gph_fn_names,
    };
    const mc = &t.graph.mods[f.mod];
    bc.tree = mc.tree;
    bc.tokens = mc.tokens;
    bc.source = mc.source;
    bc.resolutions = mc.resolutions;
    bc.graph_mod = f.mod;
    // Every diagnostic this BodyChecker emits is tagged with the fn's owning module.
    bc.sink.setScope(f.mod);
    if (t.gph_node_types) |nts| bc.node_types = nts[f.mod];
    return bc;
}

/// Switch the active tree/tokens/source/resolutions + bare-name maps to module
/// `mod`. Returns the previous active module so the caller can restore it (layout
/// recursion crosses module boundaries).
fn gphSelect(t: *Typecheck, mod: u32) u32 {
    const prev = t.graph_mod;
    const g = t.graph;
    const mc = &g.mods[mod];
    t.tree = mc.tree;
    t.tokens = mc.tokens;
    t.source = mc.source;
    t.resolutions = mc.resolutions;
    // Stamp every subsequent top-level emit with the active module.
    t.sink.setScope(mod);
    // The active bare-name maps are read via `activeStructMap`/`activeEnumMap`,
    // which dereference ctx.mods[graph_mod] directly (the maps live in the ctx, so
    // a `put` that grows is reflected — copying the map struct into `t` would
    // strand reallocations on a stale header).
    if (t.gph_node_types) |nts| t.node_types = nts[mod];
    t.graph_mod = mod;
    return prev;
}

/// The active bare-name → global-struct-id map: the current module's table.
fn activeStructMap(t: *Typecheck) *std.StringHashMapUnmanaged(u32) {
    return &t.graph.mods[t.graph_mod].struct_ids;
}

/// The active bare-name → global-enum-id map: the current module's table.
fn activeEnumMap(t: *Typecheck) *std.StringHashMapUnmanaged(u32) {
    return &t.graph.mods[t.graph_mod].enum_ids;
}

/// The layout engine's view of this checker: its tables + the per-module accessors
/// the layout recursion needs, wired to the existing methods. The `emit*` thunks
/// forward to `sink.emitFmt` with the SAME literal format strings the layout code
/// used in-line, so the emitted diagnostics stay byte-identical.
fn layoutEnv(t: *Typecheck) LayoutEngine.Env {
    const T = struct {
        fn castGph(ctx: *anyopaque, mod: u32) u32 {
            return gphSelect(@ptrCast(@alignCast(ctx)), mod);
        }
        fn castTree(ctx: *anyopaque) Ast.Tree {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            return tc.tree;
        }
        fn castTypeFromNode(ctx: *anyopaque, n: Ast.Index) Type {
            return typeFromNode(@ptrCast(@alignCast(ctx)), n);
        }
        fn castNameText(ctx: *anyopaque, tok: u32) []const u8 {
            return nameText(@ptrCast(@alignCast(ctx)), tok);
        }
        fn castByteOf(ctx: *anyopaque, tok: u32) u32 {
            return byteOf(@ptrCast(@alignCast(ctx)), tok);
        }
        fn emitRecursive(ctx: *anyopaque, byte: u32, requester: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.emitFmtCode(.T0004, byte, "recursive type '{s}' has infinite size", .{requester});
        }
        fn emitEmptyStruct(ctx: *anyopaque, byte: u32, name: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.emitFmtCode(.T0005, byte, "empty struct '{s}' is not allowed", .{name});
        }
        fn emitEmptyEnum(ctx: *anyopaque, byte: u32, name: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.emitFmtCode(.T0006, byte, "empty enum '{s}' is not allowed", .{name});
        }
        fn emitUnitField(ctx: *anyopaque, byte: u32, field: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.emitFmtCode(.T0007, byte, "field '{s}' cannot have type ()", .{field});
        }
        fn emitUnitPayload(ctx: *anyopaque, byte: u32, variant: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.emitFmtCode(.T0008, byte, "variant '{s}' payload cannot have type ()", .{variant});
        }
    };
    return .{
        .gpa = t.gpa,
        .structs = &t.structs,
        .enums = &t.enums,
        .ctx = t,
        .gphSelect = T.castGph,
        .typeFromNode = T.castTypeFromNode,
        .nameText = T.castNameText,
        .byteOf = T.castByteOf,
        .emitRecursive = T.emitRecursive,
        .emitEmptyStruct = T.emitEmptyStruct,
        .emitEmptyEnum = T.emitEmptyEnum,
        .emitUnitField = T.emitUnitField,
        .emitUnitPayload = T.emitUnitPayload,
        .tree = T.castTree,
    };
}

/// Whole-graph typecheck. Builds ONE program-wide layout table (global
/// struct/enum ids assigned in module-id then decl order — same-named types in
/// different modules are DISTINCT ids), resolves qualified `mod.Type` refs to the
/// owning module's id, checks every fn body cross-module against the resolver's
/// GLOBAL fn table, and enforces pub-signature coherence (a pub fn may not name a
/// non-pub type in its param/return). `mods` is parallel to the module graph;
/// `fns` is parallel to the resolver's global fn table (`print` last, bodyless).
/// Caller owns the returned `GraphResult`.
pub fn checkGraph(
    gpa: std.mem.Allocator,
    ctx: *GraphCtx,
    mods: []const GraphModuleInput,
    fns: []const GraphFnInput,
    entry_mod: u32,
    /// The worker pool the parallel Pass-C body checks fan out onto. `null` forces
    /// SERIAL Pass-C — the single-source path passes null so the per-file pipeline's
    /// own per-file fan-out is the only parallelism (no nested-pool oversubscription),
    /// and a 1-fn graph would never spawn anyway. `-o`/`--emit ir` pass the real `io`.
    /// SERIAL and PARALLEL are byte-identical (the merge is fn-id ordered + stable-
    /// sorted), so this is a perf lever only.
    io: ?Io,
    /// The `-j` jobs knob for the Pass-C body fan-out (0 => host cpus). Ignored when
    /// `io == null` (serial). See `Typecheck.ncpu`.
    ncpu: usize,
) !GraphResult {
    // Per-module node_types (parallel to each module's node array).
    const node_types = try gpa.alloc([]Type, mods.len);
    var nt_built: usize = 0;
    errdefer {
        for (node_types[0..nt_built]) |nt| gpa.free(nt);
        gpa.free(node_types);
    }
    for (mods, 0..) |m, i| {
        const nt = try gpa.alloc(Type, m.tree.nodes.len);
        @memset(nt, .invalid);
        node_types[i] = nt;
        nt_built += 1;
    }

    var t: Typecheck = .{
        .gpa = gpa,
        // Active views start on module 0; gphSelect swaps them per decl.
        .tree = if (mods.len != 0) mods[0].tree else .{ .nodes = &.{}, .extra = &.{} },
        .tokens = if (mods.len != 0) mods[0].tokens else &.{},
        .source = if (mods.len != 0) mods[0].source else &.{},
        .resolutions = if (mods.len != 0) mods[0].resolutions else &.{},
        .node_types = if (mods.len != 0) node_types[0] else &.{},
        .sink = DiagnosticSink.init(gpa),
        .fns = .empty,
        .structs = .empty,
        .enums = .empty,
        .graph = ctx,
        .io = io,
        .ncpu = ncpu,
    };
    defer {
        for (t.fns.items) |f| gpa.free(f.params);
        t.fns.deinit(gpa);
        for (t.enums.items) |e| {
            for (e.variants) |v| {
                gpa.free(v.field_names);
                gpa.free(v.field_types);
                gpa.free(v.offsets);
            }
            gpa.free(e.variants);
        }
        t.enums.deinit(gpa);
        // The bare-name maps live in the ctx (accessed via activeStructMap/
        // activeEnumMap); the CALLER owns and frees them.
        for (t.structs.items) |s| {
            gpa.free(s.field_names);
            gpa.free(s.field_types);
            gpa.free(s.offsets);
        }
        t.structs.deinit(gpa);
    }
    errdefer t.sink.deinit();

    // Point node_types at the active module's slice as we switch modules. The
    // Typecheck writes through t.node_types; redirect it in gphSelect-like fashion
    // by wiring each module's slice here (checkFn/layout set t via the ctx tree but
    // node_types is not part of ctx, so set it alongside graph_mod transitions).
    t.gph_node_types = node_types;

    // The qualified fn names (parallel to the global fn table). Borrowed for the
    // duration of the check (the `fns` table outlives runGraph).
    const fn_names = try gpa.alloc([]const u8, fns.len);
    defer gpa.free(fn_names);
    for (fns, 0..) |gf, i| fn_names[i] = gf.name;
    t.gph_fn_names = fn_names;

    try t.runGraph(mods, fns, entry_mod);

    const sigs = try gpa.alloc(Sig, t.fns.items.len);
    errdefer gpa.free(sigs);
    var sigs_built: usize = 0;
    errdefer for (sigs[0..sigs_built]) |s| gpa.free(@constCast(s.params));
    for (t.fns.items, 0..) |f, i| {
        const name = if (i < fns.len) fns[i].name else "print";
        sigs[i] = .{ .kind = f.kind, .name = name, .params = try gpa.dupe(Type, f.params), .ret = f.ret };
        sigs_built += 1;
    }

    const layouts = try LayoutEngine.snapshotLayouts(gpa, t.structs.items);
    errdefer LayoutEngine.freeLayouts(gpa, layouts);
    const enum_layouts = try LayoutEngine.snapshotEnumLayouts(gpa, t.enums.items);
    errdefer LayoutEngine.freeEnumLayouts(gpa, enum_layouts);

    // Diagnostics already carry their owning module in `scope`, sorted by
    // runGraph. Hand the owned slices to the result.
    const owned = try t.sink.toOwned();

    return GraphResult{
        .node_types = node_types,
        .diags = owned.diags,
        .owned_msgs = owned.owned,
        .sigs = sigs,
        .layouts = layouts,
        .enum_layouts = enum_layouts,
    };
}

/// The graph driver: register all types globally, lay them out, decode all fn
/// sigs, check pub-signature coherence, then check every fn body.
fn runGraph(t: *Typecheck, mods: []const GraphModuleInput, fns: []const GraphFnInput, entry_mod: u32) !void {
    // M1 generics gate. Generic syntax PARSES but has no semantics yet, so a serial
    // pre-scan emits T0013 for every generic declaration / type-application and
    // RETURNS before Phase 0 — no Phase-A/body cascade (a `type_app` in type
    // position would otherwise misfire as T0001 "unknown type '['"). The driver's
    // `tc.diags.len > 0` gate then stops the pipeline before codegen. The scan and
    // its `sink.sort()` are a pure function of source, so `-jN` stays byte-identical.
    if (try t.gateGenerics(mods)) {
        t.sink.sort();
        return;
    }

    // Phase 0: register every module's struct + enum names into ONE global id
    // space, deterministically (module-id order, then decl order). Structs first
    // across ALL modules, then enums, so the id spaces are independent + stable.
    for (mods, 0..) |_, mi| {
        const mod: u32 = @intCast(mi);
        _ = t.gphSelect(mod);
        if (t.tree.nodes.len == 0) continue;
        const prog = t.tree.nodes[Ast.root(t.tree.nodes).int()];
        if (prog.tag != .program) continue;
        try t.registerStructs(Ast.rangeSlice(t.tree, prog.lhs.int()), mod);
    }
    for (mods, 0..) |_, mi| {
        const mod: u32 = @intCast(mi);
        _ = t.gphSelect(mod);
        if (t.tree.nodes.len == 0) continue;
        const prog = t.tree.nodes[Ast.root(t.tree.nodes).int()];
        if (prog.tag != .program) continue;
        try t.registerEnums(Ast.rangeSlice(t.tree, prog.lhs.int()), mod);
    }

    // Phase 0b: lay out every struct then every enum (global id order). Each
    // layoutStruct/layoutEnum switches to its owning module; nested/qualified
    // referents recurse cross-module and restore the active module on return.
    for (0..t.structs.items.len) |id| try LayoutEngine.layoutStruct(t.layoutEnv(), @intCast(id));
    for (0..t.enums.items.len) |id| try LayoutEngine.layoutEnum(t.layoutEnv(), @intCast(id));

    // Phase A: decode every fn signature into the GLOBAL fn table, in the exact
    // order of `fns` (parallel to the resolver's global fn ids), so `.func` ids
    // index this table directly. The synthetic bodyless `print` is one of them.
    for (fns) |gf| {
        if (gf.kind == .builtin) {
            try t.appendPrint();
        } else {
            _ = t.gphSelect(gf.module);
            try t.decodeFnSig(gf.decl_node, gf.module);
        }
    }

    // Phase A2: pub-signature coherence. A `pub` fn that names a NON-pub type in a
    // param/return position is an error (an importer could not name that type).
    try t.checkPubSignatures(fns);

    // Rule 7 (Pass A): the entry-module `main` may only return int or (). Emitted
    // here into the shared diag stream; the final sink.sort() below orders it with
    // every other diagnostic, so it is byte-identical at -j1 and -jN.
    try t.checkMainReturn(entry_mod);

    // Phase B/C: freeze the Pass-A tables into a read-only Model, then check each
    // fn body against it via a per-fn BodyChecker (skip bodyless `print`). The
    // loop index IS the global fn id (parallel to the resolver func ids + the
    // codegen `names`/`sigs`), threaded in for the `body(fid)` node.
    //
    // Pass-A is now frozen: `model` aliases the (no-longer-mutated) tables, every
    // f.ret is final (returnRule=unit-sugar => no fn's return depends on another
    // fn's body), and each BodyChecker writes ONLY its own fn's node_types span +
    // its own local diags. So the per-fn body checks are independent and order-free
    // — the fan-out unit. The merge (concat + stable sort) happens once, serial,
    // after the join, so PARALLEL == SERIAL.
    const model = t.buildModel();
    try t.checkBodies(&model);
}

/// M1 generics gate. Scans every module's nodes in ascending index order and emits
/// T0013 ("generics not yet supported", a borrowed literal) at each `type_app`
/// (its `[`) and each `generic_param` (its name). Returns whether any fired. It is
/// SERIAL and runs before any Pass-C fan-out; module-id order + ascending node
/// index make the emit stream a pure function of source (no hashmap/thread order).
/// The caller calls `sink.sort()` (node-index order != byte-offset order) and
/// early-returns, so the sole other sort site (`checkBodies`) is skipped for a
/// generic program.
fn gateGenerics(t: *Typecheck, mods: []const GraphModuleInput) !bool {
    var fired = false;
    for (mods, 0..) |_, mi| {
        const mod: u32 = @intCast(mi);
        _ = t.gphSelect(mod); // sets the active tree/tokens + the sink emit scope
        for (t.tree.nodes) |n| {
            switch (n.tag) {
                .type_app, .generic_param => {
                    try t.sink.emitCode(.T0013, t.byteOf(n.main_token), "generics not yet supported");
                    fired = true;
                },
                else => {},
            }
        }
    }
    return fired;
}

/// Per-fn body-check result produced by one body-region job. Each holds its own
/// `DiagnosticSink` (already scoped to the fn's module) until the merge transfers it
/// into the shared sink; `node_types` were written directly into the shared
/// per-module arrays (disjoint span, no race).
const BodyResult = struct {
    sink: DiagnosticSink,
    err: ?anyerror = null,
};

/// THE per-fn body region (Pass C): run every fn's body check as an independent
/// unit, then merge the per-fn sinks in fn-id order and STABLE-sort ONCE. This is
/// the SINGLE body code path — there is no serial-vs-parallel fork. Dispatch is the
/// only thing that varies: with a worker pool (`io != null`) the units fan out via
/// `Engine.fanOut`; with no pool (`io == null`: single-file internal callers + inline
/// tests) they run inline on this thread. Both feed the SAME slots and the SAME
/// merge+sort, so the result is byte-identical regardless of dispatch.
///
/// DETERMINISM: units are independent (each writes only its own fn's
/// node_types span + its own local sink); the merge is fn-id ordered and the stable
/// sort breaks (scope, byte_offset) ties by insertion order, reproducing source
/// order exactly — so -j1 and -jN diagnostics are identical.
fn checkBodies(t: *Typecheck, model: *const Model) !void {
    const gpa = t.gpa;
    const n = t.fns.items.len;
    const slots = try gpa.alloc(BodyResult, n);
    defer gpa.free(slots);
    for (slots) |*s| s.* = .{ .sink = DiagnosticSink.init(gpa) };
    // Free every slot's sink on any error path below (merge empties a slot's sink,
    // so a deinit of an already-merged slot is a no-op — no double-free).
    defer for (slots) |*s| s.sink.deinit();

    if (t.io) |io| {
        const Ctx = struct {
            t: *const Typecheck,
            model: *const Model,
            slots: []BodyResult,
            pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(bodyUnit)) {
                return .{ c.t, c.model, @as(u32, @intCast(i)), &c.slots[i] };
            }
        };
        Engine.chunkedFanOut(io, n, t.ncpu, Engine.Chunk.body.threshold, Engine.Chunk.body.chunks_per_cpu, bodyUnit, Ctx{ .t = t, .model = model, .slots = slots });
    } else {
        for (slots, 0..) |*s, i| bodyUnit(t, model, @intCast(i), s);
    }

    for (slots) |s| if (s.err) |e| return e;

    // Merge per-fn sinks in fn-id order, then sort once. `merge` reserves capacity
    // first (infallible appends) and empties each slot, so the trailing `defer`
    // above never double-frees a transferred sink. An OOM in a reserve frees every
    // slot exactly once via that defer.
    for (slots) |*s| try t.sink.merge(&s.sink);
    t.sink.sort();
}

/// One body-region unit: construct a BodyChecker for fn `fid` over the frozen
/// `model`, walk its body, and move its local sink into `out`. Writes only its own
/// fn's node_types span + `out` for shared MUTABLE state, so units are race-free and
/// order-free (safe under `Engine.fanOut` and identical inline).
fn bodyUnit(t: *const Typecheck, model: *const Model, fid: u32, out: *BodyResult) void {
    const f = model.fns[fid];
    if (f.kind == .builtin) return; // the bodyless `print` has no body to walk
    var bc = t.bodyCheckerFor(model, f);
    defer bc.deinit();
    bc.checkBody(fid, f) catch |e| {
        out.err = e;
        return;
    };
    // Transfer the finished sink into the slot; leave bc holding a fresh empty sink
    // so the `defer bc.deinit()` frees nothing it no longer owns.
    out.sink.deinit();
    out.sink = bc.sink;
    bc.sink = DiagnosticSink.init(bc.gpa);
}

/// A `pub` fn must not expose a non-`pub` type: if any param/return type resolves
/// to a struct/enum whose decl is not `pub`, an importer naming the fn could not
/// name the type. Diagnose against the owning module + the offending type-ref.
fn checkPubSignatures(t: *Typecheck, fns: []const GraphFnInput) !void {
    for (fns, 0..) |gf, i| {
        if (gf.kind == .builtin or !gf.is_pub) continue;
        _ = t.gphSelect(gf.module);
        const f = t.fns.items[i];
        const decl = t.tree.nodes[f.decl_node.int()];
        const proto = Ast.protoAt(t.tree, decl.lhs.int());
        for (proto.params, f.params) |param_idx, pty| {
            try t.checkPubType(pty, t.tree.nodes[param_idx.int()].main_token, "function", gf.name);
        }
        if (proto.ret_type != Ast.none)
            try t.checkPubType(f.ret, t.tree.nodes[proto.ret_type.int()].main_token, "function", gf.name);
    }

    // A `pub` struct FIELD or `pub` enum variant PAYLOAD that names a non-pub type
    // leaks it across the boundary exactly as a fn param/return would (an importer
    // can read the field / destructure the variant but cannot name the type): the
    // locked rule "a pub signature naming a type forces that type pub" applies to
    // FIELD types too. Per-type checking makes this transitive: a pub type embedded in
    // another pub type is itself checked.
    for (t.structs.items) |s| {
        if (s.decl_node == Ast.none or !s.pub_export or s.poisoned) continue;
        _ = t.gphSelect(s.mod);
        const field_nodes = Ast.rangeSlice(t.tree, t.tree.nodes[s.decl_node.int()].lhs.int());
        for (s.field_types, 0..) |fty, fi| {
            const at = if (fi < field_nodes.len) t.tree.nodes[field_nodes[fi].int()].main_token else t.tree.nodes[s.decl_node.int()].main_token;
            try t.checkPubType(fty, at, "struct", s.name);
        }
    }
    for (t.enums.items) |e| {
        if (e.decl_node == Ast.none or !e.pub_export or e.poisoned) continue;
        _ = t.gphSelect(e.mod);
        const variant_nodes = Ast.rangeSlice(t.tree, t.tree.nodes[e.decl_node.int()].lhs.int());
        for (e.variants, 0..) |v, vi| {
            const at = if (vi < variant_nodes.len) t.tree.nodes[variant_nodes[vi].int()].main_token else t.tree.nodes[e.decl_node.int()].main_token;
            for (v.field_types) |fty| {
                try t.checkPubType(fty, at, "enum", e.name);
            }
        }
    }
}

/// Emit a coherence error if `ty` is a struct/enum whose declaration is not pub.
/// `owner_kind`/`owner_name` describe the exposing decl ("function" `lib.make`,
/// "struct" `lib.Outer`, "enum" `lib.E`).
fn checkPubType(t: *Typecheck, ty: Type, at_tok: u32, owner_kind: []const u8, owner_name: []const u8) !void {
    const non_pub = switch (ty.kind) {
        .@"struct" => !t.structs.items[ty.struct_id].pub_export,
        .@"enum" => !t.enums.items[ty.enum_id].pub_export,
        else => false,
    };
    if (non_pub)
        try t.sink.emitFmtCode(.T0009, t.byteOf(at_tok), "pub {s} '{s}' exposes non-pub type '{s}'", .{ owner_kind, owner_name, t.typeName(ty) });
}

/// Rule 7: the entry `main` may only yield `int` (the process exit code) or `()`
/// (nothing). Any other return type (bool/str/struct/enum) has no entry-point exit
/// semantics and is rejected here, in the checker, so the diagnostic flows through
/// the same (module, byte_offset)-sorted stream as every other type error (the
/// driver `-o` guard is the codegen-time backstop, kept for defense in depth). A
/// Pass-A check: it reads only the frozen fn signatures, so it never blocks the
/// per-fn Pass-C parallelism. `entry_mod` is the entry module id (0 single-file);
/// the entry is the FIRST fn named `main` in that module, mirroring the driver's
/// own entry selection so the checker and codegen agree on which `main`.
fn checkMainReturn(t: *Typecheck, entry_mod: u32) !void {
    for (t.fns.items) |f| {
        if (f.kind == .builtin or f.mod != entry_mod) continue;
        const tree = t.graph.mods[entry_mod].tree;
        const tokens = t.graph.mods[entry_mod].tokens;
        const source = t.graph.mods[entry_mod].source;
        const main_tok = tree.nodes[f.decl_node.int()].main_token;
        if (!std.mem.eql(u8, tokens[main_tok].text(source), "main")) continue;
        if (f.ret.kind != .int and f.ret.kind != .unit and f.ret.kind != .invalid) {
            // Select the entry module so the sink stamps this diagnostic with the
            // entry module's scope (gphSelect -> sink.setScope).
            _ = t.gphSelect(entry_mod);
            try t.sink.emitCode(.T0010, tokens[main_tok].start, "main must return int or ()");
        }
        return; // only the first `main` is the entry
    }
}

/// Register the struct decls among `decl_nodes` (of the currently-active tree).
/// `mod` is the owning module id (0 single-file). Global ids are assigned in
/// append order; per-module duplicate/shadow diagnostics mirror the single-file
/// rules. The bare name → global id binding goes into the active struct map
/// (`activeStructMap`, the current module's table in the ctx).
fn registerStructs(t: *Typecheck, decl_nodes: []const Ast.Index, mod: u32) !void {
    for (decl_nodes) |decl_idx| {
        const decl = t.tree.nodes[decl_idx.int()];
        if (decl.tag != .struct_decl) continue;
        const name = t.nameText(decl.main_token);
        if (type_names.get(name) != null) {
            try t.sink.emitFmtCode(.T0011, t.byteOf(decl.main_token), "struct '{s}' shadows a builtin type", .{name});
            continue;
        }
        if (t.activeStructMap().get(name) != null) {
            try t.sink.emitFmtCode(.T0012, t.byteOf(decl.main_token), "duplicate struct declaration '{s}'", .{name});
            continue;
        }
        const id: u32 = @intCast(t.structs.items.len);
        try t.structs.append(t.gpa, .{ .decl_node = decl_idx, .name = name, .mod = mod, .pub_export = t.tree.isPub(decl_idx) });
        try t.activeStructMap().put(t.gpa, name, id);
    }
}

/// Register the enum decls among `decl_nodes` (of the currently-active tree).
/// One shared type-name namespace per module: an enum colliding with a builtin,
/// a struct, or another enum (in this module) is rejected.
fn registerEnums(t: *Typecheck, decl_nodes: []const Ast.Index, mod: u32) !void {
    for (decl_nodes) |decl_idx| {
        const decl = t.tree.nodes[decl_idx.int()];
        if (decl.tag != .enum_decl) continue;
        const name = t.nameText(decl.main_token);
        if (type_names.get(name) != null) {
            try t.sink.emitFmt(t.byteOf(decl.main_token), "enum '{s}' shadows a builtin type", .{name});
            continue;
        }
        if (t.activeStructMap().get(name) != null or t.activeEnumMap().get(name) != null) {
            try t.sink.emitFmt(t.byteOf(decl.main_token), "duplicate type declaration '{s}'", .{name});
            continue;
        }
        const id: u32 = @intCast(t.enums.items.len);
        try t.enums.append(t.gpa, .{ .decl_node = decl_idx, .name = name, .mod = mod, .pub_export = t.tree.isPub(decl_idx) });
        try t.activeEnumMap().put(t.gpa, name, id);
    }
}

/// Decode one fn's signature (param + return types) and append a `FnSym` to the
/// global fn table. `fn_idx` is a node in the currently-active tree; `mod` its
/// owning module id. Param/return type-refs resolve via the active maps (and, for
/// a qualified `mod.Type`, via the graph context).
fn decodeFnSig(t: *Typecheck, fn_idx: Ast.Index, mod: u32) !void {
    const decl = t.tree.nodes[fn_idx.int()];
    const proto = Ast.protoAt(t.tree, decl.lhs.int());
    const params = try t.gpa.alloc(Type, proto.params.len);
    for (proto.params, 0..) |param_idx, i| {
        const param = t.tree.nodes[param_idx.int()];
        const pty = t.typeFromNode(param.lhs);
        if (pty.kind == .unit) {
            try t.sink.emitFmt(t.byteOf(param.main_token), "parameter '{s}' cannot have type ()", .{t.nameText(param.main_token)});
            params[i] = .invalid; // poison so call-arg checks don't cascade
        } else {
            params[i] = pty;
        }
    }
    const ret: Type = if (proto.ret_type == Ast.none) Type.unit else t.typeFromNode(proto.ret_type);
    try t.fns.append(t.gpa, .{ .decl_node = fn_idx, .kind = .user_fn, .params = params, .ret = ret, .mod = mod });
}

/// Append the synthetic bodyless `print(str) -> ()` builtin to the fn table.
fn appendPrint(t: *Typecheck) !void {
    const params = try t.gpa.dupe(Type, &.{.str});
    try t.fns.append(t.gpa, .{ .decl_node = Ast.none, .kind = .builtin, .params = params, .ret = .unit });
}

fn typeFromNode(t: *Typecheck, type_node: Ast.Index) Type {
    return refs.typeFromNode(t, type_node);
}

fn typeFromQualified(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) Type {
    return refs.typeFromQualified(t, node_idx, n);
}

fn typeName(t: *const Typecheck, ty: Type) []const u8 {
    return refs.typeName(t, ty);
}

fn nameText(t: *const Typecheck, tok: u32) []const u8 {
    return refs.nameText(t, tok);
}

fn byteOf(t: *const Typecheck, tok: u32) u32 {
    return refs.byteOf(t, tok);
}

fn graphCtx(t: *const Typecheck) *GraphCtx {
    return t.graph;
}
fn structSyms(t: *const Typecheck) []const StructSym {
    return t.structs.items;
}
fn enumSyms(t: *const Typecheck) []const EnumSym {
    return t.enums.items;
}

const testing = std.testing;
const Lexer = @import("lex.zig");
const Parser = @import("parse.zig");
const Graph = @import("driver/Graph.zig");
const ResolveGraph = @import("resolve_graph.zig");
const TypecheckGraph = @import("types_graph.zig");

const Checked = struct {
    tokens: []Token,
    tree: Ast.Tree,
    resolve: ResolveGraph.GraphResult,
    result: GraphResult,
    source: []const u8,

    fn deinit(self: *Checked, gpa: std.mem.Allocator) void {
        self.result.deinit(gpa);
        self.resolve.deinit(gpa);
        gpa.free(self.tokens);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
    }
};

/// Resolve + typecheck a source as the trivial one-module graph (the ONE
/// front-end), returning the whole-graph results WHOLE — exactly the carriers
/// `Driver.pipeline` stores on a `FileResult`. The resolve result owns the fn-name
/// strings the typecheck `sigs[].name` borrow; both are torn down together by
/// `Checked.deinit`. `io = null` forces serial Pass-C (deterministic; a 1-fn graph
/// never spawns anyway). Program-global fields (`result.diags`/`.sigs`) read
/// directly; the node-parallel `result.node_types[0]`/`resolve.resolutions[0]` are
/// the entry module (module 0 IS the file).
fn checkSource(source: []const u8) !Checked {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, source);
    errdefer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }

    var g = try Graph.single(gpa, "main", "", source, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinit(gpa);
    var res = try ResolveGraph.resolveGraph(gpa, &g);
    errdefer res.deinit(gpa);
    const result = try TypecheckGraph.checkGraph(gpa, &g, &res, null, 0);
    return .{ .tokens = tokens, .tree = tree, .resolve = res, .result = result, .source = source };
}

/// Typecheck a source and return the diagnostic count (and free everything).
fn checkDiagCount(source: []const u8) !usize {
    const gpa = testing.allocator;
    var c = try checkSource(source);
    defer c.deinit(gpa);
    return c.result.diags.len;
}

// The pure `Type` algebra (eql/assignable) is owned by — and unit-tested in —
// `layout/Engine.zig` now; the tests below exercise the checker's USE of it.

test "clean program typechecks with zero diagnostics" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn neg(x: int) -> int { return -x }
        \\fn main() {
        \\ x := add(1, 2)
        \\ y := neg(x)
        \\ ok := x == 3
        \\ z := x > y
        \\ return
        \\}
        \\
    ));
}

test "M1 gate: the generics demo reports T0013 with no T0001/body cascade" {
    const gpa = testing.allocator;
    var c = try checkSource("fn id[T](x: T) -> T { x }\nfn main() -> int { return id[int](7) }\n");
    defer c.deinit(gpa);
    try testing.expect(c.result.diags.len > 0);
    // Every diagnostic is the gate — no T0001 "unknown type '['" cascade — and at
    // least one carries the exact substring the examples/check.sh harness greps for.
    var saw_phrase = false;
    for (c.result.diags) |d| {
        try testing.expectEqual(codes.Code.T0013, d.code);
        if (std.mem.indexOf(u8, d.message, "generics not yet supported") != null) saw_phrase = true;
    }
    try testing.expect(saw_phrase);
}

test "M1 gate: a bare type_app in a field type fires; a non-generic program does not" {
    try testing.expect(try checkDiagCount("struct S { v: Box[int] }\nfn main() -> int { return 0 }\n") > 0);
    // No false gate on a plain program (and no spurious T0013).
    try testing.expectEqual(@as(usize, 0), try checkDiagCount("fn main() -> int { return 0 }\n"));
}

// Each value-poison site (function-as-value, bare-struct-as-value, module-as-value)
// routes its `Type.invalid` through `BodyChecker.poison()`, which asserts (Debug/
// ReleaseSafe) that this fn's sink already reported. Running these under `test-bin`
// (Debug) therefore proves the poison co-occurs with a user error — a silent poison
// would trip the assert. We ALSO check the identifier node itself types as `.invalid`
// so the poison actually flows onto the node.

/// The Type of the first identifier node whose text equals `name` in module 0.
fn identTypeOf(c: *const Checked, name: []const u8) Type {
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .identifier and std.mem.eql(u8, c.tokens[n.main_token].text(c.source), name))
            return c.result.node_types[0][i];
    }
    unreachable;
}

test "value-poison: a function used as a value reports and poisons the node" {
    const gpa = testing.allocator;
    var c = try checkSource("fn g() -> int { return 1 }\nfn f() -> int {\n return g\n}\n");
    defer c.deinit(gpa);
    try testing.expect(c.result.diags.len >= 1);
    try testing.expectEqual(Kind.invalid, identTypeOf(&c, "g").kind);
}

test "value-poison: a bare struct name used as a value reports and poisons the node" {
    const gpa = testing.allocator;
    var c = try checkSource("struct P { x: int }\nfn f() -> int {\n q := P\n return q\n}\n");
    defer c.deinit(gpa);
    try testing.expect(c.result.diags.len >= 1);
    // The bare `P` identifier (rhs of `q := P`) is the value-poison site.
    try testing.expectEqual(Kind.invalid, identTypeOf(&c, "P").kind);
}

test "value-poison: a valid program mints no value-poison (poison() unreached)" {
    // If poison() were reached with no reported error its assert would fire; a clean
    // program simply never reaches it. Zero diagnostics confirms it.
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "struct P { x: int }\nfn g() -> int { return 1 }\nfn main() -> int {\n p := P { x: g() }\n return p.x\n}\n",
    ));
}

test "call argument count mismatch" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn add(a: int, b: int) -> int { return a + b }\nfn f() {\n x := add(1)\n return\n}\n",
    ));
}

test "call argument type mismatch (bool to int param)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        "fn add(a: int, b: int) -> int { return a + b }\nfn f() {\n x := add(1, true)\n return\n}\n",
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqualStrings("argument 2: expected int, got bool", c.result.diags[0].message);
}

test "return type mismatch" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n return true\n}\n",
    ));
}

test "bare return in a non-unit function" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n return\n}\n",
    ));
}

test ":= from a () call is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn g() { return }\nfn f() {\n x := g()\n return\n}\n",
    ));
}

test "assignment type mismatch" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n x := 1\n x = true\n return\n}\n",
    ));
}

test "typed local x: T = e binds x to the annotation" {
    const gpa = testing.allocator;
    var c = try checkSource("fn f() {\n x: int = 1\n y := x + 2\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .var_decl and n.rhs != Ast.none) try testing.expectEqual(Type.int, c.result.node_types[0][i]);
    }
}

test "typed local rejects a mismatched initializer" {
    const gpa = testing.allocator;
    var c = try checkSource("fn f() {\n x: int = true\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqualStrings("cannot bind bool to 'x' of type int", c.result.diags[0].message);
}

test "typed local with () annotation is still rejected (cannot bind ())" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn g() { return }\nfn f() {\n x: () = g()\n return\n}\n",
    ));
}

test "rule 7: main returning int is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn main() -> int {\n return 0\n}\n",
    ));
}

test "rule 7: main returning () is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn main() -> () {\n return ()\n}\n",
    ));
}

test "rule 7: main with an omitted (unit-sugar) return is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn main() {\n return\n}\n",
    ));
}

test "rule 7: main returning bool is rejected" {
    const gpa = testing.allocator;
    var c = try checkSource("fn main() -> bool {\n return true\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqualStrings("main must return int or ()", c.result.diags[0].message);
    // The diagnostic points at main's name token (col 4), not the return.
    try testing.expectEqual(@as(u32, 3), c.result.diags[0].byte_offset);
}

test "rule 7: main returning a struct is rejected" {
    const gpa = testing.allocator;
    var c = try checkSource("struct P { x: int }\nfn main() -> P {\n return P { x: 1 }\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqualStrings("main must return int or ()", c.result.diags[0].message);
}

test "rule 7: a non-main fn may return any type" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn flag() -> bool {\n return true\n}\nfn main() -> int {\n return 0\n}\n",
    ));
}

test "typed local flows the annotation into an inferred enum literal" {
    const gpa = testing.allocator;
    var c = try checkSource(
        "enum E { A, B }\nfn f() {\n x: E = .A\n return\n}\n",
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "arithmetic on bool is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n b := true\n y := b + 1\n return\n}\n",
    ));
}

test "comparison yields bool" {
    const gpa = testing.allocator;
    var c = try checkSource("fn f() {\n x := 1 < 2\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The var_decl's bound type is bool.
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .var_decl) try testing.expectEqual(Type.bool, c.result.node_types[0][i]);
    }
}

test "equality type mismatch is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n b := true == 1\n return\n}\n",
    ));
}

test "one bad identifier yields exactly one typecheck diagnostic (poison)" {
    // `bad` is undeclared (a resolve error); typecheck must NOT add more diags
    // for the binary/var_decl that consume its poison type.
    const gpa = testing.allocator;
    var c = try checkSource("fn f() {\n x := bad + 1 + 2\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "error_node types as invalid, emits no diagnostics, and renders (error)" {
    // The fault-tolerant parser will (in a later stage) emit `error_node` at a parse
    // error; here we forge one into an otherwise-clean tree to pin the downstream
    // contract: it is an already-diagnosed poison leaf, so resolve + typecheck add
    // ZERO diagnostics for it and it types as the poison `invalid` (no cascade).
    const gpa = testing.allocator;
    const source = "fn main() {\n 0\n return\n}\n";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, source);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }

    // Locate the sole `literal_number` node (the bare `0` expression statement) and
    // OVERWRITE it in place with an `error_node` leaf, keeping its `main_token` so
    // its byte offset stays valid. lhs/rhs become `none` (it is a leaf).
    var err_idx: ?Ast.Index = null;
    for (tree.nodes, 0..) |*n, i| {
        if (n.tag == .literal_number) {
            n.tag = .error_node;
            n.lhs = Ast.none;
            n.rhs = Ast.none;
            err_idx = Ast.Index.from(@intCast(i));
        }
    }
    const ei = err_idx orelse return error.NoLiteralToPoison;

    // Render must print the `(error)` leaf where the literal used to be.
    var rbuf: [128]u8 = undefined;
    var rw = std.Io.Writer.fixed(&rbuf);
    try Ast.render(&rw, tree, tokens, source);
    try testing.expect(std.mem.indexOf(u8, rw.buffered(), "(error)") != null);

    // Run the real resolve + typecheck over the mutated tree (the same wiring as
    // `checkSource`, just with the injected node).
    var g = try Graph.single(gpa, "main", "", source, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinit(gpa);
    var res = try ResolveGraph.resolveGraph(gpa, &g);
    defer res.deinit(gpa);
    var result = try TypecheckGraph.checkGraph(gpa, &g, &res, null, 0);
    defer result.deinit(gpa);

    // (a) the error_node types as the poison `invalid`.
    try testing.expect(Type.eql(Type.invalid, result.node_types[0][ei.int()]));
    // (b) no-cascade / already-diagnosed: resolve + typecheck emit ZERO diagnostics.
    try testing.expectEqual(@as(usize, 0), result.diags.len);
}

test "unknown type name in a parameter" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f(a: nope) {\n return\n}\n",
    ));
}

test "unknown return type does not cascade (poison)" {
    // `nope` unknown → exactly one diagnostic; `return 1` must not add a spurious
    // 'does not match declared invalid'.
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> nope {\n return 1\n}\n",
    ));
}

test "unknown parameter type does not cascade to call arguments" {
    // `nope` unknown → one diagnostic; calling g(1) must not add 'expected invalid'.
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn g(a: nope) -> int {\n return 0\n}\nfn h() {\n x := g(1)\n return\n}\n",
    ));
}

test "() parameter type is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn k(a: ()) -> int {\n return 0\n}\n",
    ));
}

test "non-unit function that falls off the end is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n x := 1\n}\n",
    ));
}

test "non-unit function ending in return is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n return 1\n}\n",
    ));
}

test "if/else where both arms return is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(n: int) -> int {\n if n < 0 { return 0 } else { return 1 }\n}\n",
    ));
}

test "else-less if does not guarantee a return" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f(n: int) -> int {\n if n < 0 { return 0 }\n}\n",
    ));
}

test "while loop does not guarantee a return" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f(n: int) -> int {\n while n > 0 { return 1 }\n}\n",
    ));
}

test "else-if ladder where all arms return is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(n: int) -> int {\n if n < 0 { return 0 } else if n > 0 { return 1 } else { return 2 }\n}\n",
    ));
}

test "trailing bare block that returns satisfies definite-return" {
    // The `{ return 5 }` is parsed wrapped in an expr_stmt; stmtReturns must unwrap
    // it so the body is seen to return on every path (no fall-off-the-end error).
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n { return 5 }\n}\n",
    ));
}

test "parenthesized diverging if satisfies definite-return" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(n: int) -> int {\n (if n > 0 { return 1 } else { return 2 })\n}\n",
    ));
}

test "value-if with an expr_stmt-wrapped divergent arm merges to the live arm's type" {
    // then-arm is a bare block that diverges; the divergent side is unreachable, so
    // the merge must take the else-arm's int — no 'unit vs int' mismatch.
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(n: int) -> int {\n x := if n > 0 { { return 1 } } else { 2 }\n return x\n}\n",
    ));
}

test "non-bool if condition is rejected with one diagnostic" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n if 1 { return }\n return\n}\n",
    ));
}

test "non-bool while condition is rejected with one diagnostic" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n while 1 { return }\n return\n}\n",
    ));
}

test "&& on int operands is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n b := 1 && 2\n return\n}\n",
    ));
}

test "&& and || on bool operands yield bool" {
    const gpa = testing.allocator;
    var c = try checkSource("fn f() {\n b := (1 < 2) && (3 > 4)\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .var_decl) try testing.expectEqual(Type.bool, c.result.node_types[0][i]);
    }
}

test "string literal types as str" {
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n s := \"hi\"\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The var_decl's bound type is str.
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .var_decl) try testing.expectEqual(Type.str, c.result.node_types[0][i]);
    }
}

test "print of a string literal typechecks clean" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn main() {\n print(\"hi\")\n return\n}\n",
    ));
}

test "print of an int is an argument-type error" {
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n print(42)\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqualStrings("argument 1: expected str, got int", c.result.diags[0].message);
}

test "print with no arguments is an arity error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn main() {\n print()\n return\n}\n",
    ));
}

test "a bare string literal in int context is a type error" {
    // `x := "x"` makes x a str; `x + 1` then fails (operands of '+' must be int).
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n x := \"x\"\n y := x + 1\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
}

test "value-if without else is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n c := 1\n x := if c > 0 { 1 }\n return x\n}\n",
    ));
}

test "value-if with mismatched arm types is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n c := 1\n x := if c > 0 { 1 } else { \"s\" }\n return x\n}\n",
    ));
}

test "value-if with agreeing arms is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n c := 1\n x := if c > 0 { 1 } else { 2 }\n return x\n}\n",
    ));
}

test "value-if where one arm diverges merges to the other arm type" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(n: int) -> int {\n x := if n < 0 { return 0 } else { 1 }\n return x\n}\n",
    ));
}

test "trailing expression satisfies a non-unit function" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n 41 + 1\n}\n",
    ));
}

test "trailing expression of the wrong type is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n \"s\"\n}\n",
    ));
}

test "bare block expression takes its trailing type" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n x := { a := 1\n a + 1 }\n return x\n}\n",
    ));
}

test "unit-returning fn with no -> is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn noop() {\n }\nfn main() -> int {\n noop()\n return 5\n}\n",
    ));
}

test "explicit -> () return type is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn noop() -> () {\n }\nfn main() -> int {\n noop()\n return 0\n}\n",
    ));
}

test "loop value-break agreement is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n i := 0\n loop {\n if i >= 5 { break i * 10 }\n i = i + 1\n }\n}\n",
    ));
}

test "loop value-break disagreement is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n i := 0\n loop {\n if i >= 5 { break i } else { break i > 0 }\n i = i + 1\n }\n}\n",
    ));
}

test "break-less loop is never and satisfies an int fn return" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn ready(n: int) -> bool { return n >= 3 }\nfn f() -> int {\n i := 0\n loop {\n if ready(i) { return 7 }\n i = i + 1\n }\n}\n",
    ));
}

test "value-if with a break-less-loop arm merges to the other arm (never unifies)" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn ready() -> bool { return true }\nfn f() -> int {\n c := 1\n x := if c > 0 { loop { if ready() { return 0 } } } else { 5 }\n return x\n}\n",
    ));
}

test "break with a value in a while loop is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n while true { break 5 }\n return\n}\n",
    ));
}

test "break with a value in a for loop is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n for i in 0..5 { break i }\n return\n}\n",
    ));
}

test "bare break in a while loop is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() {\n while true { break }\n return\n}\n",
    ));
}

test "break outside a loop is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n break\n return\n}\n",
    ));
}

test "continue outside a loop is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n continue\n return\n}\n",
    ));
}

test "for range bound that is not int is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n for i in true..5 { x := i }\n return\n}\n",
    ));
}

test "for body using its int loop variable typechecks clean" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n s := 0\n for i in 0..5 { s = s + i }\n return s\n}\n",
    ));
}

test "print resolves and typechecks at index user_fn_count" {
    // The synthetic `print` must occupy the index right after the user fns in
    // BOTH Resolve and Typecheck so Codegen can lower print(...) as bl print_sym.
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n print(\"hi\")\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Find the call's callee identifier; it must resolve to func == 1 (main is
    // the only user fn, index 0; print is seeded next at index 1).
    var found = false;
    for (c.tree.nodes) |n| {
        if (n.tag == .call) {
            const callee_res = c.resolve.resolutions[0][n.lhs.int()];
            try testing.expect(callee_res == .func);
            try testing.expectEqual(@as(u32, 1), callee_res.func);
            found = true;
        }
    }
    try testing.expect(found);
}

test "labeled bare block value typechecks: trailing + breaks must agree" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn f(a: bool, b: bool) -> int {
        \\ x := @calc {
        \\  if a { break @calc 1 }
        \\  if b { break @calc 2 }
        \\  3
        \\ }
        \\ return x
        \\}
        \\
    ));
}

test "labeled bare block with mismatched break types is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn f(a: bool) -> int {
        \\ x := @calc {
        \\  if a { break @calc 1 }
        \\  true
        \\ }
        \\ return x
        \\}
        \\
    ));
}

test "break a value out of an outer loop typechecks" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn f() -> int {
        \\ @outer loop {
        \\  for j in 0..10 { if j == 4 { break @outer j * 10 } }
        \\ }
        \\}
        \\
    ));
}

test "continue on a labeled bare block is rejected (not a loop)" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn f() {
        \\ @blk { continue @blk }
        \\ return
        \\}
        \\
    ));
}

test "break with a value out of a labeled while is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn f() {
        \\ @w while true { break @w 5 }
        \\ return
        \\}
        \\
    ));
}

test "break with a value out of a labeled for is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn f() {
        \\ @ff for i in 0..3 { break @ff 5 }
        \\ return
        \\}
        \\
    ));
}

test "a break-less labeled loop used as a value is never (merges with caller)" {
    // The outer loop has only `break @outer` from inside, so the inner loop is
    // break-less (never) yet the program typechecks via the outer's value.
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn f() -> int {
        \\ @outer loop {
        \\  loop { break @outer 7 }
        \\ }
        \\}
        \\
    ));
}

test "labeled loop that only breaks to an outer does NOT count as diverging-free" {
    // `@L loop { for j in 0..3 { break @L } }`: the for's bare break binds to the
    // for; the outer loop is exited by `break @L`, so it must NOT be `never`.
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn f() {
        \\ @L loop { for j in 0..3 { break @L } }
        \\ return
        \\}
        \\
    ));
}

test "clean struct construct/read/store/free-fn typechecks with zero diagnostics" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\struct Point { x: int, y: int }
        \\fn area(p: Point) -> int { p.x * p.y }
        \\fn main() -> int {
        \\ p := Point { x: 6, y: 7 }
        \\ p.x = 6
        \\ q := p
        \\ return area(q)
        \\}
        \\
    ));
}

test "field punning typechecks clean" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\struct Point { x: int, y: int }
        \\fn f() -> int {
        \\ x := 1
        \\ y := 2
        \\ p := Point { x, y }
        \\ return p.x + p.y
        \\}
        \\
    ));
}

test "nested field access typechecks clean and records the field type" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Inner { v: int }
        \\struct Outer { i: Inner, w: int }
        \\fn f() -> int {
        \\ o := Outer { i: Inner { v: 1 }, w: 2 }
        \\ return o.i.v + o.w
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "missing field in construction is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int, y: int }\nfn f() -> int { p := P { x: 1 }\n return p.x }\n",
    ));
}

test "unknown field in construction is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P { x: 1, z: 2 }\n return p.x }\n",
    ));
}

test "unknown field in access is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P { x: 1 }\n return p.q }\n",
    ));
}

test "field type mismatch in construction is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P { x: true }\n return p.x }\n",
    ));
}

test "positional construction Point(1) is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P(1)\n return p.x }\n",
    ));
}

test "directly-recursive struct is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct N { n: N }\nfn main() -> int { return 0 }\n",
    ));
}

test "indirectly-recursive struct cycle is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct A { b: B }\nstruct B { a: A }\nfn main() -> int { return 0 }\n",
    ));
}

test "assign-to-field type mismatch is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int }\nfn f() { p := P { x: 1 }\n p.x = true\n return }\n",
    ));
}

test "node_types carry the struct type with the right id, and a 3-int layout is 0/8/16 size 24" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct V3 { a: int, b: int, c: int }
        \\fn f() -> int {
        \\ v := V3 { a: 1, b: 2, c: 3 }
        \\ return v.a
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The struct_init node carries a @"struct" type with struct_id 0.
    var found = false;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .struct_init) {
            try testing.expectEqual(Kind.@"struct", c.result.node_types[0][i].kind);
            try testing.expectEqual(@as(u32, 0), c.result.node_types[0][i].struct_id);
            found = true;
        }
    }
    try testing.expect(found);
    // Layout: offsets 0/8/16, size 24, align 8.
    const l = c.result.layouts[0];
    try testing.expectEqual(@as(usize, 3), l.offsets.len);
    try testing.expectEqual(@as(u32, 0), l.offsets[0]);
    try testing.expectEqual(@as(u32, 8), l.offsets[1]);
    try testing.expectEqual(@as(u32, 16), l.offsets[2]);
    try testing.expectEqual(@as(u32, 24), l.size);
    try testing.expectEqual(@as(u32, 8), l.@"align");
}

test "clean enum: all three variant forms, qualified + inferred, exhaustive match" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\enum Shape { Empty, Circle(int), Rect { w: int, h: int } }
        \\fn area(s: Shape) -> int {
        \\ match s {
        \\  .Circle(r) -> r * r,
        \\  .Rect { w, h } -> w * h,
        \\  .Empty -> 0
        \\ }
        \\}
        \\fn main() -> int {
        \\ c := Shape.Circle(5)
        \\ return area(c) + area(.Empty)
        \\}
        \\
    ));
}

test "enum layout: 3-int struct variant + tag = size 32, payload at 8" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum E { A { p: int, q: int, r: int }, B(int) }
        \\fn f(e: E) -> int { match e { .A { p, q, r } -> p, .B(x) -> x } }
        \\fn main() -> int { return f(E.B(0)) }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const e = c.result.enum_layouts[0];
    try testing.expectEqual(@as(u32, 8), e.tag_size);
    try testing.expectEqual(@as(u32, 8), e.payload_off);
    try testing.expectEqual(@as(u32, 32), e.size); // 8 tag + 24 payload
    try testing.expectEqual(@as(usize, 2), e.variants.len);
}

test "small enum is <=16B (tag + one int)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum E { C(int), N }
        \\fn f(e: E) -> int { match e { .C(r) -> r, .N -> 0 } }
        \\fn main() -> int { return f(E.N) }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(u32, 16), c.result.enum_layouts[0].size);
}

test "non-exhaustive match (missing variant, no _) is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\enum S { A, B }
        \\fn f(s: S) -> int { match s { .A -> 1 } }
        \\fn main() -> int { return f(S.A) }
        \\
    ));
}

test "match covering all variants needs no wildcard" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\enum S { A, B }
        \\fn f(s: S) -> int { match s { .A -> 1, .B -> 2 } }
        \\fn main() -> int { return f(S.A) }
        \\
    ));
}

test "match with a wildcard is exhaustive" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\enum S { A, B, C }
        \\fn f(s: S) -> int { match s { .A -> 1, _ -> 0 } }
        \\fn main() -> int { return f(S.A) }
        \\
    ));
}

test "constructing an unknown variant is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "enum S { A, B }\nfn main() -> int { x := S.Nope\n return 0 }\n",
    ));
}

test "wrong tuple-variant arity is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "enum S { C(int) }\nfn main() -> int { x := S.C(1, 2)\n return 0 }\n",
    ));
}

test "wrong payload type is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "enum S { C(int) }\nfn main() -> int { x := S.C(true)\n return 0 }\n",
    ));
}

test "directly-recursive enum is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "enum R { A(R), B }\nfn main() -> int { return 0 }\n",
    ));
}

test "indirectly-recursive enum/struct cycle is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "enum A { X(B), Y }\nstruct B { a: A }\nfn main() -> int { return 0 }\n",
    ));
}

test "inferred .V with no expected type is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "enum S { A }\nfn main() -> int { x := .A\n return 0 }\n",
    ));
}

test "int match with a wildcard compiles" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(x: int) -> int { match x { 0 -> 1, _ -> 0 } }\nfn main() -> int { return f(1) }\n",
    ));
}

test "int match without a wildcard is non-exhaustive" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f(x: int) -> int { match x { 0 -> 1, 1 -> 2 } }\nfn main() -> int { return f(1) }\n",
    ));
}

test "match as a trailing fn expression with diverging arms satisfies definite-return" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\enum S { A, B }
        \\fn f(s: S) -> int { match s { .A -> { return 1 }, .B -> { return 2 } } }
        \\fn main() -> int { return f(S.A) }
        \\
    ));
}

test "match arm body type mismatch is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\enum S { A, B }
        \\fn f(s: S) -> int { match s { .A -> 1, .B -> true } }
        \\fn main() -> int { return f(S.A) }
        \\
    ));
}

test "inferred .V resolves as a fn argument and a match-arm body" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\enum S { A, B }
        \\fn id(s: S) -> S { match s { .A -> .B, .B -> .A } }
        \\fn main() -> int { x := id(.A)
        \\ return 0 }
        \\
    ));
}

test "an enum node carries an @\"enum\" type with the right id" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum S { A, B }
        \\fn main() -> int { x := S.A
        \\ return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var found = false;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .field_access) {
            // `S.A` (unit variant ref) carries the enum type.
            if (c.result.node_types[0][i].kind == .@"enum") {
                try testing.expectEqual(@as(u32, 0), c.result.node_types[0][i].enum_id);
                found = true;
            }
        }
    }
    try testing.expect(found);
}
