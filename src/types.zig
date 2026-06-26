//! Type checking over a parsed `Ast.Tree` plus its `Resolve.Result`.
//!
//! This is the second semantic pass, run only after name resolution succeeded
//! (a name error would otherwise poison every type that depends on it). It walks
//! the program once, infers a `Type` for every expression node, and checks the
//! statement-level rules:
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
//! `identifier` nodes whose text we map straight onto a `Type`. The result is
//! in-memory only (not a cache phase yet).

const std = @import("std");
const Token = @import("ast/Token.zig").Token;
const Ast = @import("ast/Ast.zig");
const Resolve = @import("resolve.zig");
const Resolution = @import("symbols/Resolution.zig").Resolution;
const Sig = @import("symbols/Sig.zig").Sig;
const symbols = @import("symbols/Sym.zig");
const Dag = @import("query/Dag.zig");

const Typecheck = @This();

/// The type kind. `invalid` is the poison/error type: it absorbs further errors
/// so one mistake produces one diagnostic. `@"struct"` carries a `struct_id`
/// indexing the per-program struct table.
pub const Kind = enum(u8) { invalid, unit, int, bool, str, never, @"struct", @"enum" };

/// A type. A byte-foldable struct (not a tagged union) so it preserves `@memset`,
/// `node_types` triviality, and a stable fingerprint basis. A `@"struct"` kind
/// carries an index into the struct table; a `@"enum"` kind an index into the
/// parallel enum table; all other kinds leave both `no_struct`.
pub const Type = struct {
    kind: Kind,
    struct_id: u32 = no_struct,
    enum_id: u32 = no_struct,

    pub const no_struct: u32 = std.math.maxInt(u32);

    pub const invalid: Type = .{ .kind = .invalid };
    pub const unit: Type = .{ .kind = .unit };
    pub const int: Type = .{ .kind = .int };
    pub const @"bool": Type = .{ .kind = .bool };
    pub const str: Type = .{ .kind = .str };
    pub const never: Type = .{ .kind = .never };

    pub fn structT(id: u32) Type {
        return .{ .kind = .@"struct", .struct_id = id };
    }

    pub fn enumT(id: u32) Type {
        return .{ .kind = .@"enum", .enum_id = id };
    }

    pub fn eql(a: Type, b: Type) bool {
        return a.kind == b.kind and
            (a.kind != .@"struct" or a.struct_id == b.struct_id) and
            (a.kind != .@"enum" or a.enum_id == b.enum_id);
    }

    pub fn isStruct(t: Type) bool {
        return t.kind == .@"struct";
    }

    pub fn isEnum(t: Type) bool {
        return t.kind == .@"enum";
    }
};

/// A resolved struct layout: a self-contained, index-free-ish snapshot threaded
/// into Codegen/Fingerprint. Field types still reference the struct table by id
/// (for nested structs), but offsets/size/align are precomputed here. Owned.
pub const Layout = struct {
    name: []const u8,
    field_names: [][]const u8,
    field_types: []Type,
    offsets: []u32,
    size: u32,
    @"align": u32,
};

/// A variant's form: a unit (no payload), a tuple (positional payload, no field
/// names), or a struct (named payload fields).
pub const VariantForm = enum(u8) { unit, tuple, @"struct" };

/// A resolved enum layout: a value tagged union — an 8-byte tag at offset 0, then
/// payload storage sized to the largest variant's payload at `payload_off`. Each
/// variant carries its payload field types + payload-LOCAL offsets (relative to
/// `payload_off`). Owned (parallel to `Layout`). Threaded read-only into Codegen.
pub const VariantLayout = struct {
    name: []const u8,
    form: VariantForm,
    field_names: [][]const u8,
    field_types: []Type,
    /// Payload-local offsets (add `payload_off` for the absolute byte offset).
    offsets: []u32,
};
pub const EnumLayout = struct {
    name: []const u8,
    variants: []VariantLayout,
    tag_size: u32,
    payload_off: u32,
    size: u32,
    @"align": u32,
};

/// Source spelling of a type reference → `Type`. Anything else is unknown (a
/// struct name, or an error). The unit type `()` is spelled with parens, not an
/// identifier, so it is NOT here (handled in `typeFromNode` via `literal_unit`).
const type_names = std.StaticStringMap(Type).initComptime(.{
    .{ "int", Type.int },
    .{ "bool", Type.bool },
    .{ "str", Type.str },
});

/// A reported problem. Same shape as `Parser`/`Resolve` diagnostics so the
/// driver/CLI can render either uniformly (byte offset → line:col).
pub const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;

/// The pass output. Owned by the caller; free with `deinit`.
pub const Result = struct {
    /// One inferred type per AST node (indexed by node index). Only expression
    /// nodes carry a meaningful value; others stay `.invalid`.
    node_types: []Type,
    /// Type diagnostics, in discovery order.
    diags: []Diagnostic,
    /// Heap-allocated diagnostic messages (the data-bearing ones); owned so they
    /// can be freed. Static-literal messages are not in here.
    owned_msgs: [][]u8,
    /// One signature per function index (source order; the synthetic `print` at
    /// index user_fn_count). The codegen fingerprint folds a callee's sig so
    /// a signature change invalidates its callers. `params` are owned.
    sigs: []Sig,
    /// The struct table: one `Layout` per struct id, in declaration order.
    /// Threaded read-only into Codegen/Fingerprint. Owned.
    layouts: []Layout,
    /// The enum table: one `EnumLayout` per enum id, in declaration order. Owned.
    enum_layouts: []EnumLayout,

    pub fn deinit(self: *Result, gpa: std.mem.Allocator) void {
        gpa.free(self.node_types);
        gpa.free(self.diags);
        for (self.owned_msgs) |m| gpa.free(m);
        gpa.free(self.owned_msgs);
        for (self.sigs) |s| gpa.free(s.params);
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

// ---- M14 graph typecheck (program-wide layout + cross-module check) --------

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
/// `decl_node == Ast.none` is the synthetic bodyless `print`.
pub const GraphFnInput = struct {
    decl_node: Ast.Index,
    /// Owning module id; ignored when `decl_node == Ast.none`.
    module: u32,
    /// Whether this fn is `pub` (drives the pub-signature-coherence check).
    is_pub: bool,
    /// Module-qualified symbol name (used in the coherence diagnostic).
    name: []const u8,
};

/// The whole-graph typecheck output. Caller owns it; free with `deinitGraph`.
/// `node_types` is per-module; `layouts`/`enum_layouts`/`sigs` are PROGRAM-WIDE
/// (global ids), exactly as the lowering stage's Frozen needs.
pub const GraphResult = struct {
    /// One `[]Type` per module (parallel to that module's node array).
    node_types: [][]Type,
    /// Diagnostics tagged with their owning module id.
    diags: []GraphDiagnostic,
    owned_msgs: [][]u8,
    /// One Sig per GLOBAL fn id (parallel to the resolver's global fn table).
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

/// A typecheck diagnostic that knows which module's source it points into.
pub const GraphDiagnostic = struct {
    module: u32,
    byte_offset: u32,
    message: []const u8,
};

/// A top-level function's signature, decoded once up front so calls can be
/// checked against it (and forward references work).
const FnSym = struct {
    decl_node: Ast.Index,
    params: []Type,
    ret: Type,
    /// Owning module id (graph mode). 0 in single-file mode. The check loops
    /// switch the active tree/tokens/source to this module before checking.
    mod: u32 = 0,
};

/// A struct's resolved symbol: its decl node, name, and (after layout) per-field
/// names/types/offsets plus aggregate size/align. `state` guards the layout
/// recursion so a directly- or indirectly-recursive struct is caught once.
const LayoutState = enum { unseen, laying, done };
const StructSym = struct {
    decl_node: Ast.Index,
    name: []const u8,
    field_names: [][]const u8 = &.{},
    field_types: []Type = &.{},
    offsets: []u32 = &.{},
    size: u32 = 0,
    @"align": u32 = 1,
    state: LayoutState = .unseen,
    poisoned: bool = false,
    /// Owning module id (graph mode); 0 single-file. Layout switches to it.
    mod: u32 = 0,
    /// Whether the struct decl is `pub` (graph mode; pub-signature coherence).
    pub_export: bool = false,
};

/// One variant in the scratch enum table (during layout). `field_names`/`name`
/// are BORROWED source slices; `field_types`/`offsets` are owned arrays.
const VariantSym = struct {
    name: []const u8,
    form: VariantForm,
    field_names: [][]const u8 = &.{},
    field_types: []Type = &.{},
    offsets: []u32 = &.{},
    payload_size: u32 = 0,
    payload_align: u32 = 1,
};

/// An enum's resolved symbol. The tag is a fixed 8 bytes at offset 0; the payload
/// is sized to the largest variant and laid at `payload_off`. `state` guards the
/// layout recursion so a recursive enum is caught once.
const EnumSym = struct {
    decl_node: Ast.Index,
    name: []const u8,
    variants: []VariantSym = &.{},
    tag_size: u32 = 8,
    payload_off: u32 = 8,
    size: u32 = 0,
    @"align": u32 = 8,
    state: LayoutState = .unseen,
    poisoned: bool = false,
    /// Owning module id (graph mode); 0 single-file. Layout switches to it.
    mod: u32 = 0,
    /// Whether the enum decl is `pub` (graph mode; pub-signature coherence).
    pub_export: bool = false,
};

/// Natural size/align of a scalar/str type (struct sizes come from the table).
fn scalarSize(kind: Kind) u32 {
    return switch (kind) {
        .int, .bool => 8,
        .str => 16,
        else => 0,
    };
}
fn scalarAlign(kind: Kind) u32 {
    return switch (kind) {
        .int, .bool, .str => 8,
        else => 1,
    };
}
fn roundUp(n: u32, a: u32) u32 {
    if (a == 0) return n;
    return (n + a - 1) / a * a;
}

/// Per-construct context, pushed/popped as bodies are entered. A label-
/// addressable stack: `kind` distinguishes a
/// value-yielding `loop`, a `()`-statement `while`/`for`, and a value-yielding
/// labeled bare block. `is_value` is true for `loop` and `labeled_block`. `label`
/// is the construct's label name (or null when unlabeled) and `construct_node` is
/// the inner construct node a `break @L`/`continue @L` matches against. `join`
/// accumulates the merge of all value-break sites (starting at `never`).
const CtxKind = enum { loop, while_for, labeled_block };
const LoopCtx = struct {
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
diags: std.ArrayList(Diagnostic),
owned_msgs: std.ArrayList([]u8),
/// Graph mode only: the owning module id for each entry in `diags` (parallel).
/// Lets the orchestrator render each cross-module diagnostic against the right
/// source. Empty in single-file mode.
diag_mods: std.ArrayList(u32) = .empty,
/// Graph mode only: per-module node_types slices. `gphSelect` redirects the
/// active `node_types` to the selected module's slice. Null single-file.
gph_node_types: ?[][]Type = null,

/// Function table, parallel to `Resolve`'s `func` indices: the resolver assigns
/// function indices in source order, and so do we (Pass A below).
fns: std.ArrayList(FnSym),

/// Per-function: a local's type indexed by its slot. Slots are function-wide and
/// assigned densely from 0, so a flat array indexed by slot is exact.
slot_types: std.ArrayList(Type),
cur_ret: Type,
loop_stack: std.ArrayList(LoopCtx),

/// The struct table: one `StructSym` per struct id, plus a name→id map.
structs: std.ArrayList(StructSym),
struct_map: std.StringHashMapUnmanaged(u32),

/// The enum table: one `EnumSym` per enum id, plus a name→id map.
enums: std.ArrayList(EnumSym),
enum_map: std.StringHashMapUnmanaged(u32),

/// One-shot expected type for an inferred `.V` construction (a typed sink:
/// fn arg/return, assign target, or match-arm body). Saved/restored around the
/// node it flows into; consumed ONLY by an inferred `enum_init_*` (lhs == none).
expected: ?Type = null,

/// M14 graph context. `null` for the single-file `check` path (everything below
/// is local). When set (the `types_graph` orchestrator drives one shared
/// `Typecheck` across the whole module graph), the `structs`/`enums`/`fns`
/// tables are PROGRAM-WIDE (global ids), and `struct_map`/`enum_map` hold the
/// CURRENT module's bare-name → global-id bindings (swapped per module). The
/// context resolves a qualified `mod.Type` / `mod.Enum` receiver to the owning
/// module's tables. Pre-collect + layout happen once; only Pass B runs per fn.
graph: ?*GraphCtx = null,

/// The active module being type-checked / laid out (graph mode). Single-file
/// leaves it 0. Used to pick the import-namespace table for qualified receivers.
graph_mod: u32 = 0,

/// M16 — the OBSERVATIONAL dependency-recording sink. When non-null, the fine-
/// grained typecheck query projections (`signature`/`body`/`type_of`/`layout`/
/// `resolve_name`) record their nodes + edges into this shared per-build `*Dag`.
/// BORROWED; the driver owns the value on the `--dump-dag` stack frame. When
/// `null` (every default `-o`/`run`/`--emit`/check-table build) EVERY projection
/// is a no-op (`recDep` early-returns) — the result tables are read unchanged, so
/// `sigs`/`node_types`/`layouts`/`enum_layouts` are byte-identical and the fast
/// path stays VERBATIM. The recorded structure is observational: invalidation is
/// still whole-graph content-fingerprint this milestone.
dag: ?*Dag = null,

/// Graph-mode: one qualified fn name per global fn id (parallel to `t.fns`), set
/// from the `GraphFnInput` table before Pass B so the `signature(fn)` node id
/// folds the IDENTICAL qualified name the codegen callee-sig fold uses (codegen's
/// `signature` node id is `Wyhash("SGNM", SymName.name)`, and the graph SymName.name
/// IS `fns[i].name`). Null single-file (the decl-token name is used instead).
gph_fn_names: ?[]const []const u8 = null,

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
    fn namespaceOfIn(c: *const GraphCtx, mod: u32, recv_name: []const u8) ?u32 {
        return c.mods[mod].namespaces.get(recv_name);
    }
};

/// Switch the active tree/tokens/source/resolutions + bare-name maps to module
/// `mod` (graph mode). Returns the previous active module so the caller can
/// restore it (layout recursion crosses module boundaries). No-op single-file.
fn gphSelect(t: *Typecheck, mod: u32) u32 {
    const prev = t.graph_mod;
    const g = t.graph orelse return prev;
    const mc = &g.mods[mod];
    t.tree = mc.tree;
    t.tokens = mc.tokens;
    t.source = mc.source;
    t.resolutions = mc.resolutions;
    // The active bare-name maps are read via `activeStructMap`/`activeEnumMap`,
    // which dereference ctx.mods[graph_mod] directly (the maps live in the ctx, so
    // a `put` that grows is reflected — copying the map struct into `t` would
    // strand reallocations on a stale header).
    if (t.gph_node_types) |nts| t.node_types = nts[mod];
    t.graph_mod = mod;
    return prev;
}

/// The active bare-name → global-struct-id map: the current module's table in
/// graph mode, else the single-file `struct_map`.
fn activeStructMap(t: *Typecheck) *std.StringHashMapUnmanaged(u32) {
    if (t.graph) |g| return &g.mods[t.graph_mod].struct_ids;
    return &t.struct_map;
}

/// The active bare-name → global-enum-id map (per active module in graph mode).
fn activeEnumMap(t: *Typecheck) *std.StringHashMapUnmanaged(u32) {
    if (t.graph) |g| return &g.mods[t.graph_mod].enum_ids;
    return &t.enum_map;
}

/// Typecheck a resolved tree. Caller owns the returned `Result`.
pub fn check(
    gpa: std.mem.Allocator,
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const Resolution,
) !Result {
    const node_types = try gpa.alloc(Type, tree.nodes.len);
    @memset(node_types, .invalid);

    var t: Typecheck = .{
        .gpa = gpa,
        .tree = tree,
        .tokens = tokens,
        .source = source,
        .resolutions = resolutions,
        .node_types = node_types,
        .diags = .empty,
        .owned_msgs = .empty,
        .fns = .empty,
        .slot_types = .empty,
        .cur_ret = .unit,
        .loop_stack = .empty,
        .structs = .empty,
        .struct_map = .empty,
        .enums = .empty,
        .enum_map = .empty,
    };
    defer {
        for (t.fns.items) |f| gpa.free(f.params);
        t.fns.deinit(gpa);
        t.slot_types.deinit(gpa);
        t.loop_stack.deinit(gpa);
        for (t.enums.items) |e| {
            for (e.variants) |v| {
                gpa.free(v.field_names);
                gpa.free(v.field_types);
                gpa.free(v.offsets);
            }
            gpa.free(e.variants);
        }
        t.enums.deinit(gpa);
        t.enum_map.deinit(gpa);
        for (t.structs.items) |s| {
            // field_names entries are BORROWED source slices (from nameText); only
            // the arrays are owned here. The Result snapshot dupes them separately.
            gpa.free(s.field_names);
            gpa.free(s.field_types);
            gpa.free(s.offsets);
        }
        t.structs.deinit(gpa);
        t.struct_map.deinit(gpa);
    }
    errdefer {
        gpa.free(node_types);
        t.diags.deinit(gpa);
        for (t.owned_msgs.items) |m| gpa.free(m);
        t.owned_msgs.deinit(gpa);
    }

    try t.run();

    // Snapshot every fn's signature for the codegen fingerprint. `t.fns` is
    // freed by the `defer` above, so dupe each params slice into owned memory.
    const sigs = try gpa.alloc(Sig, t.fns.items.len);
    errdefer gpa.free(sigs);
    var sigs_built: usize = 0;
    errdefer for (sigs[0..sigs_built]) |s| gpa.free(@constCast(s.params));
    for (t.fns.items, 0..) |f, i| {
        // Identity (kind/name) so a caller's fingerprint folds the bound symbol,
        // not just its sig: a bodyless entry is the synthetic `print` builtin; a
        // bodied one is a user fn named by its decl token. The driver's walkCalls
        // re-derives this from the resolved SymName table, but keeping the sig
        // self-consistent here avoids a misleading half-filled struct.
        const kind: symbols.SymKind = if (f.decl_node == Ast.none) .builtin else .user_fn;
        const name = if (f.decl_node == Ast.none) "print" else t.nameText(t.tree.nodes[f.decl_node].main_token);
        sigs[i] = .{ .kind = kind, .name = name, .params = try gpa.dupe(Type, f.params), .ret = f.ret };
        sigs_built += 1;
    }

    // Snapshot the struct table into owned `Layout`s (the scratch `t.structs` is
    // freed by the `defer` above). Incremental-free errdefer on partial failure.
    const layouts = try gpa.alloc(Layout, t.structs.items.len);
    errdefer gpa.free(layouts);
    var layouts_built: usize = 0;
    errdefer for (layouts[0..layouts_built]) |l| {
        gpa.free(l.name);
        for (l.field_names) |fn_| gpa.free(fn_);
        gpa.free(l.field_names);
        gpa.free(l.field_types);
        gpa.free(l.offsets);
    };
    for (t.structs.items, 0..) |s, i| {
        const fnames = try gpa.alloc([]const u8, s.field_names.len);
        var dn: usize = 0;
        errdefer {
            for (fnames[0..dn]) |x| gpa.free(x);
            gpa.free(fnames);
        }
        for (s.field_names, 0..) |nm, j| {
            fnames[j] = try gpa.dupe(u8, nm);
            dn += 1;
        }
        layouts[i] = .{
            .name = try gpa.dupe(u8, s.name),
            .field_names = @ptrCast(fnames),
            .field_types = try gpa.dupe(Type, s.field_types),
            .offsets = try gpa.dupe(u32, s.offsets),
            .size = s.size,
            .@"align" = s.@"align",
        };
        layouts_built += 1;
    }
    // If the enum snapshot below fails, the layouts slice (built above) must be
    // deep-freed too — the per-element errdefer above only covers a partial
    // layouts build, not a fully-built `layouts` array.
    errdefer for (layouts) |l| {
        gpa.free(l.name);
        for (l.field_names) |fn_| gpa.free(fn_);
        gpa.free(l.field_names);
        gpa.free(l.field_types);
        gpa.free(l.offsets);
    };

    // Snapshot the enum table into owned `EnumLayout`s (parallel to layouts).
    const enum_layouts = try gpa.alloc(EnumLayout, t.enums.items.len);
    errdefer gpa.free(enum_layouts);
    var enums_built: usize = 0;
    errdefer for (enum_layouts[0..enums_built]) |e| {
        gpa.free(e.name);
        for (e.variants) |v| {
            for (v.field_names) |fn_| gpa.free(fn_);
            gpa.free(v.field_names);
            gpa.free(v.field_types);
            gpa.free(v.offsets);
        }
        gpa.free(e.variants);
    };
    for (t.enums.items, 0..) |e, i| {
        const variants = try gpa.alloc(VariantLayout, e.variants.len);
        var vbuilt: usize = 0;
        errdefer {
            for (variants[0..vbuilt]) |v| {
                for (v.field_names) |x| gpa.free(x);
                gpa.free(v.field_names);
                gpa.free(v.field_types);
                gpa.free(v.offsets);
            }
            gpa.free(variants);
        }
        for (e.variants, 0..) |v, j| {
            const fnames = try gpa.alloc([]const u8, v.field_names.len);
            var dn: usize = 0;
            errdefer {
                for (fnames[0..dn]) |x| gpa.free(x);
                gpa.free(fnames);
            }
            for (v.field_names, 0..) |nm, k| {
                fnames[k] = try gpa.dupe(u8, nm);
                dn += 1;
            }
            variants[j] = .{
                .name = v.name,
                .form = v.form,
                .field_names = @ptrCast(fnames),
                .field_types = try gpa.dupe(Type, v.field_types),
                .offsets = try gpa.dupe(u32, v.offsets),
            };
            vbuilt += 1;
        }
        enum_layouts[i] = .{
            .name = try gpa.dupe(u8, e.name),
            .variants = variants,
            .tag_size = e.tag_size,
            .payload_off = e.payload_off,
            .size = e.size,
            .@"align" = e.@"align",
        };
        enums_built += 1;
    }

    return Result{
        .node_types = node_types,
        .diags = try t.diags.toOwnedSlice(gpa),
        .owned_msgs = try t.owned_msgs.toOwnedSlice(gpa),
        .sigs = sigs,
        .layouts = layouts,
        .enum_layouts = enum_layouts,
    };
}

/// Whole-graph typecheck (M14). Builds ONE program-wide layout table (global
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
    dag: ?*Dag,
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
        .diags = .empty,
        .owned_msgs = .empty,
        .fns = .empty,
        .slot_types = .empty,
        .cur_ret = .unit,
        .loop_stack = .empty,
        .structs = .empty,
        .struct_map = .empty, // unused in graph mode (per-module maps live in ctx)
        .enums = .empty,
        .enum_map = .empty,
        .graph = ctx,
        .dag = dag,
    };
    defer {
        for (t.fns.items) |f| gpa.free(f.params);
        t.fns.deinit(gpa);
        t.slot_types.deinit(gpa);
        t.loop_stack.deinit(gpa);
        for (t.enums.items) |e| {
            for (e.variants) |v| {
                gpa.free(v.field_names);
                gpa.free(v.field_types);
                gpa.free(v.offsets);
            }
            gpa.free(e.variants);
        }
        t.enums.deinit(gpa);
        // In graph mode the bare-name maps live in the ctx (accessed via
        // activeStructMap/activeEnumMap); `t.struct_map`/`t.enum_map` stay the
        // empty init maps (own nothing) — deinit is a safe no-op. The CALLER owns
        // and frees the ctx maps.
        t.struct_map.deinit(gpa);
        t.enum_map.deinit(gpa);
        for (t.structs.items) |s| {
            gpa.free(s.field_names);
            gpa.free(s.field_types);
            gpa.free(s.offsets);
        }
        t.structs.deinit(gpa);
        t.diag_mods.deinit(gpa);
    }
    errdefer {
        t.diags.deinit(gpa);
        for (t.owned_msgs.items) |m| gpa.free(m);
        t.owned_msgs.deinit(gpa);
    }

    // Point node_types at the active module's slice as we switch modules. The
    // Typecheck writes through t.node_types; redirect it in gphSelect-like fashion
    // by wiring each module's slice here (checkFn/layout set t via the ctx tree but
    // node_types is not part of ctx, so set it alongside graph_mod transitions).
    t.gph_node_types = node_types;

    // M16: the qualified fn names (parallel to the global fn table) so a
    // `signature(fn)` node id folds the SAME name as the codegen callee-sig fold.
    // Borrowed for the duration of the check (the `fns` table outlives runGraph).
    const fn_names = try gpa.alloc([]const u8, fns.len);
    defer gpa.free(fn_names);
    for (fns, 0..) |gf, i| fn_names[i] = gf.name;
    t.gph_fn_names = fn_names;

    try t.runGraph(mods, fns);

    // ---- snapshot: sigs (qualified names from `fns`) ----
    const sigs = try gpa.alloc(Sig, t.fns.items.len);
    errdefer gpa.free(sigs);
    var sigs_built: usize = 0;
    errdefer for (sigs[0..sigs_built]) |s| gpa.free(@constCast(s.params));
    for (t.fns.items, 0..) |f, i| {
        const kind: symbols.SymKind = if (f.decl_node == Ast.none) .builtin else .user_fn;
        const name = if (i < fns.len) fns[i].name else "print";
        sigs[i] = .{ .kind = kind, .name = name, .params = try gpa.dupe(Type, f.params), .ret = f.ret };
        sigs_built += 1;
    }

    // ---- snapshot: layouts + enum_layouts (program-wide) ----
    const layouts = try snapshotLayouts(gpa, t.structs.items);
    errdefer freeLayouts(gpa, layouts);
    const enum_layouts = try snapshotEnumLayouts(gpa, t.enums.items);
    errdefer freeEnumLayouts(gpa, enum_layouts);

    // ---- snapshot: diagnostics tagged with their owning module ----
    const diags = try gpa.alloc(GraphDiagnostic, t.diags.items.len);
    errdefer gpa.free(diags);
    for (t.diags.items, 0..) |d, i| {
        diags[i] = .{
            .module = if (i < t.diag_mods.items.len) t.diag_mods.items[i] else 0,
            .byte_offset = d.byte_offset,
            .message = d.message,
        };
    }
    // Free the diags ArrayList backing buffer (messages are kept alive via
    // owned_msgs, transferred to the result below). Reset to empty so the top
    // errdefer's `t.diags.deinit` stays a safe no-op if a later step fails.
    t.diags.deinit(gpa);
    t.diags = .empty;

    return GraphResult{
        .node_types = node_types,
        .diags = diags,
        .owned_msgs = try t.owned_msgs.toOwnedSlice(gpa),
        .sigs = sigs,
        .layouts = layouts,
        .enum_layouts = enum_layouts,
    };
}

/// The graph driver: register all types globally, lay them out, decode all fn
/// sigs, check pub-signature coherence, then check every fn body.
fn runGraph(t: *Typecheck, mods: []const GraphModuleInput, fns: []const GraphFnInput) !void {
    // Phase 0: register every module's struct + enum names into ONE global id
    // space, deterministically (module-id order, then decl order). Structs first
    // across ALL modules, then enums, so the id spaces are independent + stable.
    for (mods, 0..) |_, mi| {
        const mod: u32 = @intCast(mi);
        _ = t.gphSelect(mod);
        if (t.tree.nodes.len == 0) continue;
        const prog = t.tree.nodes[Ast.root(t.tree.nodes)];
        if (prog.tag != .program) continue;
        try t.registerStructs(Ast.rangeSlice(t.tree, prog.lhs), mod);
    }
    for (mods, 0..) |_, mi| {
        const mod: u32 = @intCast(mi);
        _ = t.gphSelect(mod);
        if (t.tree.nodes.len == 0) continue;
        const prog = t.tree.nodes[Ast.root(t.tree.nodes)];
        if (prog.tag != .program) continue;
        try t.registerEnums(Ast.rangeSlice(t.tree, prog.lhs), mod);
    }

    // Phase 0b: lay out every struct then every enum (global id order). Each
    // layoutStruct/layoutEnum switches to its owning module; nested/qualified
    // referents recurse cross-module and restore the active module on return.
    for (0..t.structs.items.len) |id| try t.layoutStruct(@intCast(id));
    for (0..t.enums.items.len) |id| try t.layoutEnum(@intCast(id));

    // Phase A: decode every fn signature into the GLOBAL fn table, in the exact
    // order of `fns` (parallel to the resolver's global fn ids), so `.func` ids
    // index this table directly. The synthetic bodyless `print` is one of them.
    for (fns) |gf| {
        if (gf.decl_node == Ast.none) {
            try t.appendPrint();
        } else {
            _ = t.gphSelect(gf.module);
            try t.decodeFnSig(gf.decl_node, gf.module);
        }
    }

    // Phase A2: pub-signature coherence. A `pub` fn that names a NON-pub type in a
    // param/return position is an error (an importer could not name that type).
    try t.checkPubSignatures(fns);

    // Phase B: check each fn body in its owning module (skip bodyless `print`).
    // The loop index IS the global fn id (parallel to the resolver func ids + the
    // codegen `names`/`sigs`), threaded into `checkFn` for the `body(fid)` node.
    for (t.fns.items, 0..) |f, i| {
        if (f.decl_node == Ast.none) continue;
        try t.checkFn(@intCast(i), f);
    }
}

/// A `pub` fn must not expose a non-`pub` type: if any param/return type resolves
/// to a struct/enum whose decl is not `pub`, an importer naming the fn could not
/// name the type. Diagnose against the owning module + the offending type-ref.
fn checkPubSignatures(t: *Typecheck, fns: []const GraphFnInput) !void {
    for (fns, 0..) |gf, i| {
        if (gf.decl_node == Ast.none or !gf.is_pub) continue;
        _ = t.gphSelect(gf.module);
        // M16: pub-coherence reads this fn's SIGNATURE (param/ret types) — a
        // signature(fn) dependency. Enter the fn's body node as the parent so the
        // recorded edge is body(fn)->signature(fn) (same node as codegen/typeOfCall).
        var dag_prev: ?Dag.NodeKey = null;
        var entered = false;
        if (t.dag != null) {
            dag_prev = Dag.Active.enter(.{ .kind = .body, .id = @intCast(i) });
            entered = true;
            t.recordSignature(@intCast(i));
        }
        defer if (entered) Dag.Active.leave(dag_prev);
        const f = t.fns.items[i];
        const decl = t.tree.nodes[f.decl_node];
        const proto = Ast.protoAt(t.tree, decl.lhs);
        for (proto.params, f.params) |param_idx, pty| {
            try t.checkPubType(pty, t.tree.nodes[param_idx].main_token, "function", gf.name);
        }
        if (proto.ret_type != Ast.none)
            try t.checkPubType(f.ret, t.tree.nodes[proto.ret_type].main_token, "function", gf.name);
    }

    // A `pub` struct FIELD or `pub` enum variant PAYLOAD that names a non-pub type
    // leaks it across the boundary exactly as a fn param/return would (an importer
    // can read the field / destructure the variant but cannot name the type) — locked
    // design item #4: "a pub signature naming a type forces that type pub" applies to
    // FIELD types too. Per-type checking makes this transitive: a pub type embedded in
    // another pub type is itself checked.
    for (t.structs.items) |s| {
        if (s.decl_node == Ast.none or !s.pub_export or s.poisoned) continue;
        _ = t.gphSelect(s.mod);
        const field_nodes = Ast.rangeSlice(t.tree, t.tree.nodes[s.decl_node].lhs);
        for (s.field_types, 0..) |fty, fi| {
            const at = if (fi < field_nodes.len) t.tree.nodes[field_nodes[fi]].main_token else t.tree.nodes[s.decl_node].main_token;
            try t.checkPubType(fty, at, "struct", s.name);
        }
    }
    for (t.enums.items) |e| {
        if (e.decl_node == Ast.none or !e.pub_export or e.poisoned) continue;
        _ = t.gphSelect(e.mod);
        const variant_nodes = Ast.rangeSlice(t.tree, t.tree.nodes[e.decl_node].lhs);
        for (e.variants, 0..) |v, vi| {
            const at = if (vi < variant_nodes.len) t.tree.nodes[variant_nodes[vi]].main_token else t.tree.nodes[e.decl_node].main_token;
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
    if (t.graph == null) return;
    const non_pub = switch (ty.kind) {
        .@"struct" => !t.structs.items[ty.struct_id].pub_export,
        .@"enum" => !t.enums.items[ty.enum_id].pub_export,
        else => false,
    };
    if (non_pub)
        try t.emitFmt(t.byteOf(at_tok), "pub {s} '{s}' exposes non-pub type '{s}'", .{ owner_kind, owner_name, t.typeName(ty) });
}

// ---- shared snapshot helpers (used by check + checkGraph) ------------------

fn snapshotLayouts(gpa: std.mem.Allocator, structs: []const StructSym) ![]Layout {
    const layouts = try gpa.alloc(Layout, structs.len);
    var built: usize = 0;
    errdefer {
        freeLayouts(gpa, layouts[0..built]);
        gpa.free(layouts);
    }
    for (structs, 0..) |s, i| {
        const fnames = try gpa.alloc([]const u8, s.field_names.len);
        var dn: usize = 0;
        errdefer {
            for (fnames[0..dn]) |x| gpa.free(x);
            gpa.free(fnames);
        }
        for (s.field_names, 0..) |nm, j| {
            fnames[j] = try gpa.dupe(u8, nm);
            dn += 1;
        }
        layouts[i] = .{
            .name = try gpa.dupe(u8, s.name),
            .field_names = @ptrCast(fnames),
            .field_types = try gpa.dupe(Type, s.field_types),
            .offsets = try gpa.dupe(u32, s.offsets),
            .size = s.size,
            .@"align" = s.@"align",
        };
        built += 1;
    }
    return layouts;
}

fn freeLayouts(gpa: std.mem.Allocator, layouts: []const Layout) void {
    for (layouts) |l| {
        gpa.free(l.name);
        for (l.field_names) |fn_| gpa.free(fn_);
        gpa.free(l.field_names);
        gpa.free(l.field_types);
        gpa.free(l.offsets);
    }
    gpa.free(layouts);
}

fn snapshotEnumLayouts(gpa: std.mem.Allocator, enums: []const EnumSym) ![]EnumLayout {
    const enum_layouts = try gpa.alloc(EnumLayout, enums.len);
    var built: usize = 0;
    errdefer {
        freeEnumLayouts(gpa, enum_layouts[0..built]);
        gpa.free(enum_layouts);
    }
    for (enums, 0..) |e, i| {
        const variants = try gpa.alloc(VariantLayout, e.variants.len);
        var vbuilt: usize = 0;
        errdefer {
            for (variants[0..vbuilt]) |v| {
                for (v.field_names) |x| gpa.free(x);
                gpa.free(v.field_names);
                gpa.free(v.field_types);
                gpa.free(v.offsets);
            }
            gpa.free(variants);
        }
        for (e.variants, 0..) |v, j| {
            const fnames = try gpa.alloc([]const u8, v.field_names.len);
            var dn: usize = 0;
            errdefer {
                for (fnames[0..dn]) |x| gpa.free(x);
                gpa.free(fnames);
            }
            for (v.field_names, 0..) |nm, k| {
                fnames[k] = try gpa.dupe(u8, nm);
                dn += 1;
            }
            variants[j] = .{
                .name = v.name,
                .form = v.form,
                .field_names = @ptrCast(fnames),
                .field_types = try gpa.dupe(Type, v.field_types),
                .offsets = try gpa.dupe(u32, v.offsets),
            };
            vbuilt += 1;
        }
        enum_layouts[i] = .{
            .name = try gpa.dupe(u8, e.name),
            .variants = variants,
            .tag_size = e.tag_size,
            .payload_off = e.payload_off,
            .size = e.size,
            .@"align" = e.@"align",
        };
        built += 1;
    }
    return enum_layouts;
}

fn freeEnumLayouts(gpa: std.mem.Allocator, enum_layouts: []const EnumLayout) void {
    for (enum_layouts) |e| {
        gpa.free(e.name);
        for (e.variants) |v| {
            for (v.field_names) |fn_| gpa.free(fn_);
            gpa.free(v.field_names);
            gpa.free(v.field_types);
            gpa.free(v.offsets);
        }
        gpa.free(e.variants);
    }
    gpa.free(enum_layouts);
}

fn run(t: *Typecheck) !void {
    if (t.tree.nodes.len == 0) return;
    const prog = t.tree.nodes[Ast.root(t.tree.nodes)];
    if (prog.tag != .program) return; // defensive

    const decl_nodes = Ast.rangeSlice(t.tree, prog.lhs);

    try t.registerStructs(decl_nodes, 0);
    try t.registerEnums(decl_nodes, 0);

    // Pass A0b: lay out each struct (visiting-guard catches recursive cycles).
    // Runs AFTER enum registration so a struct field of an enum type resolves.
    for (0..t.structs.items.len) |id| {
        try t.layoutStruct(@intCast(id));
    }
    // Pass A0d: lay out each enum (guard catches recursive cycles).
    for (0..t.enums.items.len) |id| {
        try t.layoutEnum(@intCast(id));
    }

    // Pass A: decode every function signature (forward refs resolve fine since
    // calls look the signature up by index, not by walk order). Indices here
    // match `Resolve`'s `func` indices (both number fns in source order).
    for (decl_nodes) |fn_idx| {
        const decl = t.tree.nodes[fn_idx];
        if (decl.tag != .fn_decl) continue;
        try t.decodeFnSig(fn_idx, 0);
    }

    // Synthetic `print(str) -> ()` builtin. Appended AFTER the user-fn loop so
    // its index == user_fn_count, matching Resolve's seeding order (which assigns
    // print the same index). `decl_node = Ast.none` flags it as bodyless so
    // checkFn skips it (Pass B). Codegen emits a hand-written body at this sym.
    try t.appendPrint();

    // Pass B: check each function body (skip the synthetic, bodyless builtins).
    // The loop index IS the global fn id (parallel to the resolver func ids), used
    // for the `body(fid)` DAG node.
    for (t.fns.items, 0..) |f, i| {
        if (f.decl_node == Ast.none) continue;
        try t.checkFn(@intCast(i), f);
    }
}

/// Register the struct decls among `decl_nodes` (of the currently-active tree).
/// `mod` is the owning module id (0 single-file). Global ids are assigned in
/// append order; per-module duplicate/shadow diagnostics mirror the single-file
/// rules. The bare name → global id binding goes into the active `struct_map`.
fn registerStructs(t: *Typecheck, decl_nodes: []const Ast.Index, mod: u32) !void {
    for (decl_nodes) |decl_idx| {
        const decl = t.tree.nodes[decl_idx];
        if (decl.tag != .struct_decl) continue;
        const name = t.nameText(decl.main_token);
        if (type_names.get(name) != null) {
            try t.emitFmt(t.byteOf(decl.main_token), "struct '{s}' shadows a builtin type", .{name});
            continue;
        }
        if (t.activeStructMap().get(name) != null) {
            try t.emitFmt(t.byteOf(decl.main_token), "duplicate struct declaration '{s}'", .{name});
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
        const decl = t.tree.nodes[decl_idx];
        if (decl.tag != .enum_decl) continue;
        const name = t.nameText(decl.main_token);
        if (type_names.get(name) != null) {
            try t.emitFmt(t.byteOf(decl.main_token), "enum '{s}' shadows a builtin type", .{name});
            continue;
        }
        if (t.activeStructMap().get(name) != null or t.activeEnumMap().get(name) != null) {
            try t.emitFmt(t.byteOf(decl.main_token), "duplicate type declaration '{s}'", .{name});
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
    const decl = t.tree.nodes[fn_idx];
    const proto = Ast.protoAt(t.tree, decl.lhs);
    const params = try t.gpa.alloc(Type, proto.params.len);
    for (proto.params, 0..) |param_idx, i| {
        const param = t.tree.nodes[param_idx];
        const pty = t.typeFromNode(param.lhs);
        if (pty.kind == .unit) {
            try t.emitFmt(t.byteOf(param.main_token), "parameter '{s}' cannot have type ()", .{t.nameText(param.main_token)});
            params[i] = .invalid; // poison so call-arg checks don't cascade
        } else {
            params[i] = pty;
        }
    }
    const ret: Type = if (proto.ret_type == Ast.none) Type.unit else t.typeFromNode(proto.ret_type);
    try t.fns.append(t.gpa, .{ .decl_node = fn_idx, .params = params, .ret = ret, .mod = mod });
}

/// Append the synthetic bodyless `print(str) -> ()` builtin to the fn table.
fn appendPrint(t: *Typecheck) !void {
    const params = try t.gpa.dupe(Type, &.{.str});
    try t.fns.append(t.gpa, .{ .decl_node = Ast.none, .params = params, .ret = .unit });
}

fn checkFn(t: *Typecheck, fid: u32, f: FnSym) !void {
    // Graph mode: check this fn body in its owning module's tree/resolutions.
    const prev = t.gphSelect(f.mod);
    defer _ = t.gphSelect(prev);

    // M16: enter this fn's `body(fid)` node as the Active query parent for the
    // duration of the body check, so every nested signature/type_of/layout/
    // resolve_name read inside records `body(fid) -> callee`. The body fp is a
    // body-content fold DISTINCT from the sig fold (so a body edit flips body fp
    // but leaves the fn's signature fp STABLE — the firewall). No-op + zero
    // overhead when `dag == null` (the verbatim default path).
    var dag_prev: ?Dag.NodeKey = null;
    var entered = false;
    if (t.dag) |d| {
        const node: Dag.NodeKey = .{ .kind = .body, .id = fid };
        // Pass B runs each body at the top of the active stack: the parent is the
        // saved-before value (null at Pass B top level), so `body(fid)` is a ROOT
        // node here (its fp is recorded, no spurious self-edge).
        const parent = Dag.Active.get();
        dag_prev = Dag.Active.enter(node);
        entered = true;
        // Body content fp: a body-content fold DISTINCT from the sig fold; purely
        // observational (the authoritative codegen fingerprint is the cache key).
        d.recordEdge(t.gpa, parent, node, t.bodyFp(fid, f));
    }
    defer if (entered) Dag.Active.leave(dag_prev);

    const decl = t.tree.nodes[f.decl_node];
    const proto = Ast.protoAt(t.tree, decl.lhs);

    // Rebuild the per-function slot→type table. Parameters get slots 0..N first
    // (the resolver declares them first), then `:=` locals as we encounter them.
    t.slot_types.clearRetainingCapacity();
    for (f.params) |pty| try t.slot_types.append(t.gpa, pty);
    t.cur_ret = f.ret;

    // M16: the fn's OWN declared param/return aggregates are a real codegen layout
    // dependency (Walks.walkTouchedSig folds each proto param + the return type).
    // recordSignature records this only on signature(fid), and only at caller
    // call-sites — so an UNCALLED pass-through fn `fn id(p:Point)->Point { p }` whose
    // body never constructs/accesses the aggregate has a body(fid) fp blind to the
    // aggregate's layout (an ABI-boundary layout edit would early-cutoff it as
    // unchanged under M17 = miscompile). Record body(fid)->layout for each param +
    // the return under the entered body(fid) Active node. No-op for scalars / when
    // dag == null.
    if (t.dag != null) {
        const body_node: Dag.NodeKey = .{ .kind = .body, .id = fid };
        for (f.params) |pty| t.recordLayoutOf(body_node, pty);
        t.recordLayoutOf(body_node, f.ret);
    }

    // A function body is a block; a non-unit fn wants its trailing expression to
    // supply the value. A unit fn checks its body in statement context.
    const want_value = (f.ret.kind != .unit and f.ret.kind != .invalid);
    // The fn return type is the expected type for the body's trailing value
    // expression (so a trailing inferred `.V` resolves to the return enum).
    const body_ty = try t.typeOfBlockExpected(decl.rhs, want_value, if (want_value) f.ret else null);

    // A non-unit function must produce a value on every path: either all paths
    // structurally return, OR the body's trailing expression has the declared
    // type (no explicit `return` needed). An else-less `if` or a `while`
    // never guarantees a return, and a trailing non-expression item is unit.
    if (want_value and !blockReturns(t, decl.rhs)) {
        if (body_ty.kind == .invalid) {
            // poison already reported elsewhere; no extra diagnostic
        } else if (Type.eql(body_ty, f.ret) or body_ty.kind == .never) {
            // trailing expression supplies the value — OK (or it is a `never`
            // expression, e.g. a break-less `loop`, that can never fall through).
        } else {
            try t.emitFmt(t.byteOf(decl.main_token), "function '{s}' must return {s} but may fall off the end", .{ t.nameText(decl.main_token), t.typeName(f.ret) });
        }
    }
    _ = proto;
}

/// A body-content fingerprint for `body(fid)`, DISTINCT from `signature(fid)`'s
/// sig-only fold: it folds the body block's SOURCE SPAN (every statement's bytes),
/// so any body edit flips it while a pure signature edit (param/ret type, with the
/// body text unchanged) does NOT. Observational only — used so the `--dump-dag`
/// firewall demo shows a body fp that is stable under a signature-only change and a
/// signature fp that is stable under a body-only change.
fn bodyFp(t: *const Typecheck, fid: u32, f: FnSym) u64 {
    var h = std.hash.Wyhash.init(0x42_4f_44_59); // "BODY"
    var ib: [4]u8 = undefined;
    std.mem.writeInt(u32, &ib, fid, .little);
    h.update(&ib);
    const decl = t.tree.nodes[f.decl_node];
    if (decl.rhs != Ast.none) {
        const body = t.tree.nodes[decl.rhs];
        // Fold the body block's FULL source span `{ ... }`: scan tokens from the
        // body's `{` (its main_token), matching braces, to the closing `}`, and
        // fold that exact source slice. This captures the whole body text (so any
        // body edit flips the fp) yet excludes the proto (param/ret) tokens, which
        // sit BEFORE the `{` — the firewall basis (a sig-only edit leaves it
        // stable). Falls back to the `{` byte alone on a malformed token range.
        const open = body.main_token;
        if (open < t.tokens.len and t.tokens[open].tag == .l_brace) {
            var depth: i32 = 0;
            var ti: usize = open;
            var close: usize = open;
            while (ti < t.tokens.len) : (ti += 1) {
                switch (t.tokens[ti].tag) {
                    .l_brace => depth += 1,
                    .r_brace => {
                        depth -= 1;
                        if (depth == 0) {
                            close = ti;
                            break;
                        }
                    },
                    else => {},
                }
            }
            h.update(t.source[t.tokens[open].start..t.tokens[close].end]);
        }
    }
    return h.final();
}

/// Structural definite-return: a block returns on every path iff its last
/// statement does (earlier statements can't satisfy the requirement, since
/// anything after a guaranteed return would be unreachable).
fn blockReturns(t: *const Typecheck, block_idx: Ast.Index) bool {
    const stmts = Ast.rangeSlice(t.tree, t.tree.nodes[block_idx].lhs);
    if (stmts.len == 0) return false;
    return stmtReturns(t, stmts[stmts.len - 1]);
}

/// Whether a single statement guarantees a return on every path through it.
fn stmtReturns(t: *const Typecheck, stmt_idx: Ast.Index) bool {
    const stmt = t.tree.nodes[stmt_idx];
    return switch (stmt.tag) {
        .return_stmt => true,
        // A statement-position block/if is parsed wrapped in an `expr_stmt`; unwrap
        // it so a trailing diverging bare block (`{ return 5 }`) or parenthesized
        // value-if satisfies definite-return and a divergent arm merges correctly.
        .expr_stmt => stmtReturns(t, stmt.lhs),
        .block => blockReturns(t, stmt_idx),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(t.tree, stmt.rhs);
            // An else-less `if` can be skipped, so it never guarantees a return.
            if (h.else_node == Ast.none) break :blk false;
            const then_ok = blockReturns(t, h.then_block);
            const else_ok = if (t.tree.nodes[h.else_node].tag == .if_stmt)
                stmtReturns(t, h.else_node)
            else
                blockReturns(t, h.else_node);
            break :blk then_ok and else_ok;
        },
        // Conservative: a `while` may never execute, so it can't guarantee a
        // return (`while true { return }` is rejected — acceptable for now).
        .while_stmt => false,
        // A `loop` diverges (returns/never-falls-through) iff it has NO `break`
        // targeting it: the only ways out are `return` or an outer construct.
        .loop_expr => loopDiverges(t, stmt_idx),
        // A labeled wrapper is transparent for definite-return: it returns iff its
        // inner construct does (a labeled bare block via its trailing stmt).
        .labeled => stmtReturns(t, t.tree.nodes[stmt_idx].lhs),
        .for_stmt, .break_stmt, .continue_stmt => false,
        else => false,
    };
}

/// A `loop` diverges (control never falls past it) iff no `break` targets it.
/// `target` is the loop's own node — a break "targets THIS loop" iff it is a
/// BARE break encountered at this loop's nesting level OR a labeled break whose
/// resolved `.label` target == `target`. For the bare case the scan descends
/// `if`/`block`/`expr_stmt` but NOT a nested loop (a bare break there binds to
/// the inner loop); for the labeled case it MUST descend nested loops/labeled
/// wrappers to find a `break @target` buried inside.
fn loopDiverges(t: *const Typecheck, loop_idx: Ast.Index) bool {
    return !blockHasBreak(t, t.tree.nodes[loop_idx].lhs, loop_idx);
}

fn blockHasBreak(t: *const Typecheck, block_idx: Ast.Index, target: Ast.Index) bool {
    for (Ast.rangeSlice(t.tree, t.tree.nodes[block_idx].lhs)) |s| {
        if (stmtHasBreak(t, s, target)) return true;
    }
    return false;
}

/// Whether `stmt` contains a `break` that targets the loop node `target`.
fn stmtHasBreak(t: *const Typecheck, stmt_idx: Ast.Index, target: Ast.Index) bool {
    const stmt = t.tree.nodes[stmt_idx];
    return switch (stmt.tag) {
        // A bare break (no label) binds to the innermost loop — counts only when
        // `target` IS the innermost loop, i.e. the bare break is found before any
        // nested loop swallows it (the nested-loop arms below stop the descent for
        // bare breaks). A labeled break counts iff its resolved target matches.
        .break_stmt => if (t.resolutions[stmt_idx] == .label)
            t.resolutions[stmt_idx].label == target
        else
            true,
        .expr_stmt => stmtHasBreak(t, stmt.lhs, target),
        .block => blockHasBreak(t, stmt_idx, target),
        // A labeled wrapper is transparent: descend its inner construct (a
        // `break @target` may live inside a nested labeled loop).
        .labeled => stmtHasBreak(t, stmt.lhs, target),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(t.tree, stmt.rhs);
            if (blockHasBreak(t, h.then_block, target)) break :blk true;
            if (h.else_node == Ast.none) break :blk false;
            break :blk if (t.tree.nodes[h.else_node].tag == .if_stmt)
                stmtHasBreak(t, h.else_node, target)
            else
                blockHasBreak(t, h.else_node, target);
        },
        // A nested loop/for/while swallows BARE breaks, but a `break @target`
        // buried inside it still targets `target` — so descend its body and only
        // count labeled breaks that name `target`.
        .loop_expr => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].lhs, target),
        .while_stmt => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].rhs, target),
        .for_stmt => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].lhs, target),
        else => false,
    };
}

/// Scan a nested loop's body for a labeled `break @target` (bare breaks here bind
/// to the nested loop, so they do NOT count for `target`).
fn nestedHasLabeledBreak(t: *const Typecheck, block_idx: Ast.Index, target: Ast.Index) bool {
    for (Ast.rangeSlice(t.tree, t.tree.nodes[block_idx].lhs)) |s| {
        if (stmtHasLabeledBreak(t, s, target)) return true;
    }
    return false;
}

fn stmtHasLabeledBreak(t: *const Typecheck, stmt_idx: Ast.Index, target: Ast.Index) bool {
    const stmt = t.tree.nodes[stmt_idx];
    return switch (stmt.tag) {
        .break_stmt => t.resolutions[stmt_idx] == .label and t.resolutions[stmt_idx].label == target,
        .expr_stmt => stmtHasLabeledBreak(t, stmt.lhs, target),
        .block => nestedHasLabeledBreak(t, stmt_idx, target),
        .labeled => stmtHasLabeledBreak(t, stmt.lhs, target),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(t.tree, stmt.rhs);
            if (nestedHasLabeledBreak(t, h.then_block, target)) break :blk true;
            if (h.else_node == Ast.none) break :blk false;
            break :blk if (t.tree.nodes[h.else_node].tag == .if_stmt)
                stmtHasLabeledBreak(t, h.else_node, target)
            else
                nestedHasLabeledBreak(t, h.else_node, target);
        },
        .loop_expr => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].lhs, target),
        .while_stmt => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].rhs, target),
        .for_stmt => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].lhs, target),
        else => false,
    };
}

/// Does this block's final statement DIVERGE — i.e. control never falls past it,
/// so when used as a value-merge arm the arm is `never`? Distinct from
/// `blockReturns` ("guarantees a function return"): a `break`/`continue` diverts
/// control out of the innermost loop (genuinely divergent for the merge) but does
/// NOT guarantee a function return. The merge consumer needs divergence; the
/// definite-return check needs return-guarantee. Do not conflate them.
fn blockDiverges(t: *const Typecheck, block_idx: Ast.Index) bool {
    const stmts = Ast.rangeSlice(t.tree, t.tree.nodes[block_idx].lhs);
    if (stmts.len == 0) return false;
    return stmtDiverges(t, stmts[stmts.len - 1]);
}

fn stmtDiverges(t: *const Typecheck, stmt_idx: Ast.Index) bool {
    const stmt = t.tree.nodes[stmt_idx];
    return switch (stmt.tag) {
        .return_stmt, .break_stmt, .continue_stmt => true,
        .expr_stmt => stmtDiverges(t, stmt.lhs),
        .block => blockDiverges(t, stmt_idx),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(t.tree, stmt.rhs);
            if (h.else_node == Ast.none) break :blk false;
            const then_ok = blockDiverges(t, h.then_block);
            const else_ok = if (t.tree.nodes[h.else_node].tag == .if_stmt)
                stmtDiverges(t, h.else_node)
            else
                blockDiverges(t, h.else_node);
            break :blk then_ok and else_ok;
        },
        .while_stmt => false,
        .loop_expr => loopDiverges(t, stmt_idx),
        .labeled => labeledDiverges(t, stmt_idx),
        .for_stmt => false,
        // A `match` diverges iff it is exhaustive AND every arm body diverges.
        .match_expr => matchDiverges(t, stmt_idx),
        else => false,
    };
}

/// A `match` diverges (control never falls past it) iff it is exhaustive (every
/// variant covered, or a `_` arm) AND every arm body diverges. Needed so a match
/// as a fn's trailing expression satisfies definite-return.
fn matchDiverges(t: *const Typecheck, node_idx: Ast.Index) bool {
    const n = t.tree.nodes[node_idx];
    const st = t.node_types[n.lhs];
    // Conservative for int/bool scrutinees: `return false` (loses only a
    // definite-return optimization, never miscompiles). Only enums get the
    // variant-coverage analysis here.
    if (!st.isEnum()) return false;
    const arms = Ast.rangeSlice(t.tree, n.rhs);
    if (arms.len == 0) return false;
    var has_wildcard = false;
    const e = t.enums.items[st.enum_id];
    var seen = [_]bool{false} ** 64; // enum variant count is small
    for (arms) |arm_idx| {
        const arm = t.tree.nodes[arm_idx];
        const h = Ast.armHeaderAt(t.tree, arm.rhs);
        if (!stmtDiverges(t, h.body)) return false;
        if (h.guard != Ast.none) continue; // a guard can fail → no coverage
        const pat = t.tree.nodes[arm.lhs];
        if (pat.tag == .pattern_wildcard) {
            has_wildcard = true;
        } else if (pat.tag == .pattern_variant) {
            const vname = t.nameText(pat.main_token);
            for (e.variants, 0..) |v, i| {
                if (i < seen.len and std.mem.eql(u8, v.name, vname) and t.variantPayloadIrrefutable(arm.lhs, v)) seen[i] = true;
            }
        }
    }
    if (has_wildcard) return true;
    if (e.variants.len > seen.len) return false;
    for (e.variants, 0..) |_, i| if (!seen[i]) return false;
    return true;
}

/// A `labeled` wrapper diverges (control never falls past it) when its inner
/// construct does. For a labeled loop this is `loopDiverges` (break-less). For a
/// labeled bare block it is: the block's trailing stmt diverges AND no `break`
/// targets the block (a targeting break would exit normally, past the block).
fn labeledDiverges(t: *const Typecheck, idx: Ast.Index) bool {
    const inner_idx = t.tree.nodes[idx].lhs;
    const inner = t.tree.nodes[inner_idx];
    return switch (inner.tag) {
        .loop_expr => loopDiverges(t, inner_idx),
        .while_stmt, .for_stmt => false,
        .block => blockDiverges(t, inner_idx) and !blockHasBreak(t, inner_idx, inner_idx),
        else => false,
    };
}

/// Type a block. Every non-final statement is checked for effect. The block's
/// VALUE is the final item's type IFF that item is an expression carrier and a
/// value is wanted; otherwise the block is `()`. When `!want_value`, a trailing
/// if/block is checked in STATEMENT context (an else-less trailing `if` stays a
/// `()` statement, never "value-if requires else"). Records the type on the node.
/// `checkBlock` with a one-shot expected type for its trailing value expression.
fn typeOfBlockExpected(t: *Typecheck, block_idx: Ast.Index, want_value: bool, exp: ?Type) error{OutOfMemory}!Type {
    const save = t.expected;
    t.expected = exp;
    defer t.expected = save;
    return t.checkBlock(block_idx, want_value);
}

fn checkBlock(t: *Typecheck, block_idx: Ast.Index, want_value: bool) error{OutOfMemory}!Type {
    const stmts = Ast.rangeSlice(t.tree, t.tree.nodes[block_idx].lhs);
    if (stmts.len == 0) {
        t.node_types[block_idx] = .unit;
        return .unit;
    }
    // The expected type (a one-shot for an inferred `.V`) applies ONLY to this
    // block's trailing value expression, never the non-final statements. Capture
    // it, clear it for the non-final walk, then restore for the trailing expr.
    const block_expected = t.expected;
    t.expected = null;
    for (stmts[0 .. stmts.len - 1]) |s| try t.checkStmt(s); // effect only
    const last = stmts[stmts.len - 1];
    const last_n = t.tree.nodes[last];
    var bt: Type = .unit;
    if (last_n.tag == .expr_stmt) {
        bt = try t.typeOfExpected(last_n.lhs, if (want_value) block_expected else null);
        t.node_types[last] = bt; // the expr_stmt carries the value type
    } else if (want_value and (last_n.tag == .if_stmt or last_n.tag == .block)) {
        bt = try t.typeOfExpected(last, block_expected); // value context: validates + types + memoizes
    } else {
        try t.checkStmt(last); // statement context (incl. trailing else-less if)
    }
    t.node_types[block_idx] = bt;
    return bt;
}

fn checkStmt(t: *Typecheck, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const stmt = t.tree.nodes[stmt_idx];
    switch (stmt.tag) {
        .var_decl => {
            const ty = try t.typeOf(stmt.lhs);
            t.node_types[stmt_idx] = ty;
            // Record the new local's type at its slot. The resolver bound the
            // var_decl node to a `.local` slot; append/extend slot_types to fit.
            if (t.resolutions[stmt_idx] == .local) {
                const slot = t.resolutions[stmt_idx].local;
                try t.setSlot(slot, ty);
            }
            if (ty.kind == .unit) {
                try t.emitFmt(t.byteOf(stmt.main_token), "cannot bind () to '{s}'", .{t.nameText(stmt.main_token)});
            }
        },
        .assign => {
            const target = t.tree.nodes[stmt.lhs];
            // Compute the place type FIRST so it can flow into the rhs as the
            // expected type (an inferred `.V` assigned to a known-typed place).
            const lhs: Type = switch (target.tag) {
                .identifier => if (t.resolutions[stmt.lhs] == .local)
                    t.slotType(t.resolutions[stmt.lhs].local)
                else
                    .invalid,
                .field_access => try t.typeOf(stmt.lhs),
                else => .invalid,
            };
            t.node_types[stmt.lhs] = lhs;
            const rhs = try t.typeOfExpected(stmt.rhs, if (lhs.kind == .invalid) null else lhs);
            if (lhs.kind != .invalid and rhs.kind != .invalid and !Type.eql(lhs, rhs)) {
                try t.emitFmt(t.byteOf(target.main_token), "cannot assign {s} to variable of type {s}", .{ t.typeName(rhs), t.typeName(lhs) });
            }
        },
        .return_stmt => {
            const ty: Type = if (stmt.lhs == Ast.none) Type.unit else try t.typeOfExpected(stmt.lhs, if (t.cur_ret.kind == .invalid) null else t.cur_ret);
            // Skip when either side is poison so an unknown return type (already
            // reported) doesn't trigger a spurious second diagnostic here. A
            // `never`-typed operand (e.g. `return x` where x binds a break-less
            // loop) is also fine: control never actually reaches the return, so
            // `never` unifies with any declared type — mirrors the trailing-expr
            // body check above.
            if (ty.kind != .invalid and ty.kind != .never and t.cur_ret.kind != .invalid and !Type.eql(ty, t.cur_ret)) {
                try t.emitFmt(t.byteOf(stmt.main_token), "return type {s} does not match declared {s}", .{ t.typeName(ty), t.typeName(t.cur_ret) });
            }
        },
        .expr_stmt => _ = try t.typeOf(stmt.lhs),
        .block => _ = try t.checkBlock(stmt_idx, false),
        .if_stmt => {
            const ct = try t.typeOf(stmt.lhs);
            if (ct.kind != .invalid and ct.kind != .bool)
                try t.emit(t.byteOf(t.tree.nodes[stmt.lhs].main_token), "if condition must be bool");
            const h = Ast.ifHeaderAt(t.tree, stmt.rhs);
            _ = try t.checkBlock(h.then_block, false);
            if (h.else_node != Ast.none) {
                if (t.tree.nodes[h.else_node].tag == .if_stmt)
                    try t.checkStmt(h.else_node)
                else
                    _ = try t.checkBlock(h.else_node, false);
            }
        },
        .while_stmt => try t.checkWhile(stmt_idx, null),
        .for_stmt => try t.checkFor(stmt_idx, null),
        .labeled => _ = try t.checkLabeled(stmt_idx, false),
        .break_stmt => {
            const ctx = t.targetCtx(stmt_idx) orelse {
                // No matching context: a bare break with an empty stack ("outside a
                // loop"); a labeled break is reported by resolve as undefined.
                if (t.resolutions[stmt_idx] != .label)
                    try t.emit(t.byteOf(stmt.main_token), "break outside of a loop");
                if (stmt.lhs != Ast.none) _ = try t.typeOf(stmt.lhs);
                return;
            };
            if (stmt.lhs == Ast.none) {
                ctx.saw_bare_break = true;
                if (ctx.is_value) ctx.join = try t.merge(stmt.main_token, ctx.join, Type.unit);
            } else {
                const vt = try t.typeOf(stmt.lhs);
                if (!ctx.is_value) {
                    if (vt.kind != .invalid and vt.kind != .unit)
                        try t.emit(t.byteOf(stmt.main_token), "cannot break with a value out of a while/for loop");
                } else {
                    ctx.saw_value_break = true;
                    ctx.join = try t.merge(stmt.main_token, ctx.join, vt);
                }
            }
        },
        .continue_stmt => {
            const ctx = t.targetCtx(stmt_idx) orelse {
                if (t.resolutions[stmt_idx] != .label)
                    try t.emit(t.byteOf(stmt.main_token), "continue outside of a loop");
                return;
            };
            // `continue` is meaningful only on a loop; a labeled bare block is not.
            if (ctx.kind == .labeled_block)
                try t.emit(t.byteOf(stmt.main_token), "cannot continue a labeled block (not a loop)");
        },
        else => _ = try t.typeOf(stmt_idx),
    }
}

/// Select the context a `break`/`continue` targets. A labeled break/continue
/// (resolved to `.label`) scans the stack top-down for the matching construct
/// node (the NAMED target, not the innermost). A bare break/continue scans
/// top-down for the innermost LOOP (skipping labeled bare blocks).
fn targetCtx(t: *Typecheck, stmt_idx: Ast.Index) ?*LoopCtx {
    const items = t.loop_stack.items;
    if (t.resolutions[stmt_idx] == .label) {
        const target = t.resolutions[stmt_idx].label;
        var i = items.len;
        while (i > 0) {
            i -= 1;
            if (items[i].construct_node == target) return &items[i];
        }
        return null;
    }
    var i = items.len;
    while (i > 0) {
        i -= 1;
        if (items[i].kind != .labeled_block) return &items[i];
    }
    return null;
}

/// Check a `while` statement; `label` is its label name (or null). A `()` loop.
fn checkWhile(t: *Typecheck, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
    const stmt = t.tree.nodes[stmt_idx];
    const ct = try t.typeOf(stmt.lhs);
    if (ct.kind != .invalid and ct.kind != .bool)
        try t.emit(t.byteOf(t.tree.nodes[stmt.lhs].main_token), "while condition must be bool");
    try t.loop_stack.append(t.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
    _ = try t.checkBlock(stmt.rhs, false);
    _ = t.loop_stack.pop();
}

/// Check a `for` statement; `label` is its label name (or null). A `()` loop.
fn checkFor(t: *Typecheck, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
    const stmt = t.tree.nodes[stmt_idx];
    const h = Ast.forHeaderAt(t.tree, stmt.rhs);
    const lo = try t.typeOf(h.lo);
    const hi = try t.typeOf(h.hi);
    if (lo.kind != .invalid and lo.kind != .int)
        try t.emit(t.byteOf(t.tree.nodes[h.lo].main_token), "for range bounds must be int");
    if (hi.kind != .invalid and hi.kind != .int)
        try t.emit(t.byteOf(t.tree.nodes[h.hi].main_token), "for range bounds must be int");
    if (t.resolutions[stmt_idx] == .local) try t.setSlot(t.resolutions[stmt_idx].local, Type.int);
    try t.loop_stack.append(t.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
    _ = try t.checkBlock(stmt.lhs, false);
    _ = t.loop_stack.pop();
}

/// Check a `labeled` wrapper. The inner construct is typed through its label-
/// aware variant; the wrapper's type is the inner's type (the loop value, `()`
/// for while/for, or the labeled-block value). `want_value` flows to the block
/// case so a statement-context labeled bare block stays `()`-discarded.
fn checkLabeled(t: *Typecheck, idx: Ast.Index, want_value: bool) error{OutOfMemory}!Type {
    const n = t.tree.nodes[idx];
    const label = t.nameText(n.main_token);
    const inner = t.tree.nodes[n.lhs];
    const ty: Type = switch (inner.tag) {
        .block => try t.typeOfLabeledBlock(n.lhs, label, want_value),
        .loop_expr => try t.typeOfLoop(n.lhs, inner, label),
        .while_stmt => blk: {
            try t.checkWhile(n.lhs, label);
            break :blk Type.unit;
        },
        .for_stmt => blk: {
            try t.checkFor(n.lhs, label);
            break :blk Type.unit;
        },
        else => Type.invalid,
    };
    t.node_types[idx] = ty;
    return ty;
}

/// Type a labeled BARE BLOCK used as a value: its value is the merge of its
/// trailing-expression value and every `break @label <expr>` targeting it
/// (`never` if it can only diverge). Lowers like a value-loop minus the back-edge.
fn typeOfLabeledBlock(t: *Typecheck, block_idx: Ast.Index, label: []const u8, want_value: bool) error{OutOfMemory}!Type {
    try t.loop_stack.append(t.gpa, .{ .kind = .labeled_block, .label = label, .construct_node = block_idx, .is_value = true, .join = .never, .saw_value_break = false, .saw_bare_break = false });
    const fall = try t.checkBlock(block_idx, want_value);
    const ctx = t.loop_stack.pop().?;
    // The trailing-expr value is unreachable iff the block's last statement
    // diverges; in that case the block's value comes entirely from its breaks.
    const ft: Type = if (blockDiverges(t, block_idx)) Type.never else fall;
    return t.merge(t.tree.nodes[block_idx].main_token, ft, ctx.join);
}

/// Type a node with a one-shot expected type (a typed sink). Consumed only by an
/// inferred `.V` construction; saved/restored so it never leaks past this node.
fn typeOfExpected(t: *Typecheck, node_idx: Ast.Index, exp: ?Type) error{OutOfMemory}!Type {
    const save = t.expected;
    t.expected = exp;
    defer t.expected = save;
    return t.typeOf(node_idx);
}

/// Infer (and memoize) the type of an expression node.
fn typeOf(t: *Typecheck, node_idx: Ast.Index) error{OutOfMemory}!Type {
    if (node_idx == Ast.none) return .invalid;
    const n = t.tree.nodes[node_idx];
    const ty: Type = switch (n.tag) {
        .literal_number => Type.int,
        .literal_bool => Type.@"bool",
        .literal_string => Type.str,
        .identifier => switch (t.resolutions[node_idx]) {
            .local => |slot| t.slotType(slot),
            .func => blk: {
                try t.emitFmt(t.byteOf(n.main_token), "function '{s}' is not a value", .{t.nameText(n.main_token)});
                break :blk Type.invalid;
            },
            .unresolved => blk: {
                // Resolve quietly skips a struct-named identifier (it expects this
                // to be a struct type-name in a position it doesn't bind, or a
                // positional `Point(...)` callee it diagnoses elsewhere). A BARE
                // struct name used as a value (`q := P`, `P.x`) reaches here with
                // no diagnostic — report it so it never escapes to codegen.
                if (t.activeStructMap().get(t.nameText(n.main_token)) != null)
                    try t.emitFmt(t.byteOf(n.main_token), "type '{s}' is not a value", .{t.nameText(n.main_token)});
                break :blk Type.invalid;
            },
            .label => Type.invalid, // never on an identifier node (break/continue only)
            .module => blk: {
                // A bare imported-namespace name used as a value (`x := mod`):
                // a module is not a value. (A `mod.member` access never reaches
                // here — the receiver is consumed by typeOfFieldAccess/Call.)
                try t.emitFmt(t.byteOf(n.main_token), "module '{s}' is not a value", .{t.nameText(n.main_token)});
                break :blk Type.invalid;
            },
        },
        .unary => blk: {
            const operand = try t.typeOf(n.lhs);
            if (operand.kind == .invalid) break :blk Type.invalid;
            const op = t.tokens[n.main_token].tag;
            switch (op) {
                .minus => {
                    if (operand.kind == .int) break :blk Type.int;
                    try t.emit(t.byteOf(n.main_token), "operand of '-' must be int");
                },
                .bang => {
                    if (operand.kind == .bool) break :blk Type.@"bool";
                    try t.emit(t.byteOf(n.main_token), "operand of '!' must be bool");
                },
                else => {},
            }
            break :blk Type.invalid;
        },
        .binary => blk: {
            const lt = try t.typeOf(n.lhs);
            const rt = try t.typeOf(n.rhs);
            if (lt.kind == .invalid or rt.kind == .invalid) break :blk Type.invalid;
            const op = t.tokens[n.main_token].tag;
            const op_text = t.tokens[n.main_token].text(t.source);
            switch (op) {
                .plus, .minus, .star, .slash => {
                    if (lt.kind == .int and rt.kind == .int) break :blk Type.int;
                    try t.emitFmt(t.byteOf(n.main_token), "operands of '{s}' must be int", .{op_text});
                },
                .lt, .lt_eq, .gt, .gt_eq => {
                    if (lt.kind == .int and rt.kind == .int) break :blk Type.@"bool";
                    try t.emitFmt(t.byteOf(n.main_token), "operands of '{s}' must be int", .{op_text});
                },
                .eq_eq, .bang_eq => {
                    if (Type.eql(lt, rt) and (lt.kind == .int or lt.kind == .bool)) break :blk Type.@"bool";
                    if (Type.eql(lt, rt) and lt.kind == .str) {
                        // Same type, but str comparison isn't supported — say so,
                        // rather than the misleading "must have the same type".
                        try t.emitFmt(t.byteOf(n.main_token), "'{s}' on str is unsupported", .{op_text});
                    } else {
                        try t.emitFmt(t.byteOf(n.main_token), "operands of '{s}' must have the same type", .{op_text});
                    }
                },
                .amp_amp, .pipe_pipe => {
                    if (lt.kind == .bool and rt.kind == .bool) break :blk Type.@"bool";
                    try t.emitFmt(t.byteOf(n.main_token), "operands of '{s}' must be bool", .{op_text});
                },
                else => {},
            }
            break :blk Type.invalid;
        },
        .call => try t.typeOfCall(node_idx, n),
        .struct_init => try t.typeOfStructInit(node_idx, n),
        .field_access => try t.typeOfFieldAccess(node_idx, n),
        .enum_init_unit, .enum_init_tuple, .enum_init_struct => try t.typeOfEnumInit(node_idx, n),
        .match_expr => return t.typeOfMatch(node_idx, n), // sets node_types itself
        .literal_unit => Type.unit,
        .block => try t.checkBlock(node_idx, true),
        .if_stmt => try t.typeOfIf(node_idx, n),
        .loop_expr => return t.typeOfLoop(node_idx, n, null), // sets node_types itself
        .labeled => return t.checkLabeled(node_idx, true), // sets node_types itself
        else => Type.invalid,
    };
    t.node_types[node_idx] = ty;
    return ty;
}

/// Type a `Name { field: value, ... }` construction. The type name must be a
/// known struct; every declared field must be supplied exactly once; each
/// value's type must match the declared field type. Result is the struct type.
fn typeOfStructInit(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    // A qualified struct-variant `N.V { ... }` would arrive as enum_init_struct
    // (the parser upgrades a `field_access {`), not here — so struct_init's lhs is
    // always a plain type-name identifier; no enum routing needed.
    _ = node_idx;
    const name = t.nameText(t.tree.nodes[n.lhs].main_token);
    const id = t.activeStructMap().get(name) orelse {
        for (Ast.rangeSlice(t.tree, n.rhs)) |fi| _ = try t.typeOf(t.tree.nodes[fi].lhs);
        try t.emitFmt(t.byteOf(t.tree.nodes[n.lhs].main_token), "unknown struct type '{s}'", .{name});
        return .invalid;
    };
    const sym = t.structs.items[id];
    t.recordBodyLayout(Type.structT(id)); // body -> layout(struct): construction reads field offsets (M17)
    const inits = Ast.rangeSlice(t.tree, n.rhs);

    // Track which declared fields are supplied (for missing/duplicate checks).
    var seen = try t.gpa.alloc(bool, sym.field_names.len);
    defer t.gpa.free(seen);
    @memset(seen, false);

    for (inits) |fi_idx| {
        const fi = t.tree.nodes[fi_idx];
        const fname = t.nameText(fi.main_token);
        const vt = try t.typeOf(fi.lhs);
        // Find the declared field by name.
        var found: ?usize = null;
        for (sym.field_names, 0..) |dn, j| {
            if (std.mem.eql(u8, dn, fname)) {
                found = j;
                break;
            }
        }
        if (found) |j| {
            if (seen[j]) {
                try t.emitFmt(t.byteOf(fi.main_token), "duplicate field '{s}' in '{s}'", .{ fname, name });
            }
            seen[j] = true;
            const fty = sym.field_types[j];
            if (vt.kind != .invalid and fty.kind != .invalid and !Type.eql(vt, fty)) {
                try t.emitFmt(t.byteOf(fi.main_token), "field '{s}': expected {s}, got {s}", .{ fname, t.typeName(fty), t.typeName(vt) });
            }
        } else {
            try t.emitFmt(t.byteOf(fi.main_token), "unknown field '{s}' in '{s}'", .{ fname, name });
        }
    }
    for (sym.field_names, 0..) |dn, j| {
        if (!seen[j]) try t.emitFmt(t.byteOf(n.main_token), "missing field '{s}' in '{s}'", .{ dn, name });
    }
    return Type.structT(id);
}

/// In graph mode, resolve a `mod.Enum` node (an inner `field_access` whose
/// receiver binds to a `.module`) to the owning module's GLOBAL enum id, or null
/// if it is not a qualified cross-module enum reference.
fn qualifiedEnumId(t: *Typecheck, node_idx: Ast.Index) ?u32 {
    const g = t.graph orelse return null;
    const n = t.tree.nodes[node_idx];
    if (n.tag != .field_access) return null;
    const recv = t.tree.nodes[n.lhs];
    if (recv.tag != .identifier) return null;
    if (t.resolutions[n.lhs] != .module) return null;
    const recv_name = t.nameText(recv.main_token);
    const target = g.namespaceOfIn(t.graph_mod, recv_name) orelse return null;
    const member = t.nameText(n.main_token);
    return g.mods[target].enum_ids.get(member);
}

/// Type a `recv.field` access. The receiver must be a struct; the field must
/// exist. Yields the field type. Poison receivers stay silent (already reported).
fn typeOfFieldAccess(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    // A qualified UNIT-variant reference `N.V`: the receiver is an enum type-name
    // identifier (resolved quietly to .unresolved). Treat it as construction.
    const recv = t.tree.nodes[n.lhs];
    if (recv.tag == .identifier and t.activeEnumMap().get(t.nameText(recv.main_token)) != null) {
        return t.typeOfEnumInitQualified(node_idx, .unit, n.lhs, n.main_token, Ast.none);
    }
    // A 3-level cross-module unit-variant `mod.Enum.Variant`: the receiver of THIS
    // field_access is the inner `mod.Enum` field_access (graph mode). Resolve the
    // inner to a global enum id and treat this node as a unit-variant construction.
    if (t.qualifiedEnumId(n.lhs)) |enum_id| {
        const ty = try t.checkVariant(enum_id, n.main_token, .unit, Ast.none);
        t.node_types[node_idx] = ty;
        return ty;
    }
    const base = try t.typeOf(n.lhs);
    if (base.kind == .invalid) return .invalid;
    if (!base.isStruct()) {
        try t.emitFmt(t.byteOf(n.main_token), "cannot access field '{s}' of non-struct type {s}", .{ t.nameText(n.main_token), t.typeName(base) });
        return .invalid;
    }
    const sym = t.structs.items[base.struct_id];
    t.recordBodyLayout(base); // body -> layout(struct): field access reads the field offset/type (M17)
    const fname = t.nameText(n.main_token);
    for (sym.field_names, 0..) |dn, j| {
        if (std.mem.eql(u8, dn, fname)) return sym.field_types[j];
    }
    try t.emitFmt(t.byteOf(n.main_token), "no field '{s}' in struct '{s}'", .{ fname, sym.name });
    return .invalid;
}

/// The form a construction NODE supplies (independent of the declared variant).
const InitForm = enum { unit, tuple, @"struct" };

/// Type an inferred-or-qualified `enum_init_*` node. Resolves the enum id (from a
/// qualified type-name lhs, else the expected type), the variant by name, checks
/// the supplied payload against the variant's declared form/arity/types, and
/// yields the enum type.
fn typeOfEnumInit(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    _ = node_idx;
    const node_form: InitForm = switch (n.tag) {
        .enum_init_unit => .unit,
        .enum_init_tuple => .tuple,
        .enum_init_struct => .@"struct",
        else => unreachable,
    };
    const args: Ast.Index = if (n.tag == .enum_init_unit) Ast.none else n.rhs;
    // Resolve the enum id: qualified (lhs is the type-name identifier) or inferred
    // (the one-shot expected type must be an enum).
    var enum_id: u32 = undefined;
    if (n.lhs != Ast.none) {
        const tname = t.nameText(t.tree.nodes[n.lhs].main_token);
        enum_id = t.activeEnumMap().get(tname) orelse {
            try t.typeArgsForEffect(node_form, args);
            try t.emitFmt(t.byteOf(t.tree.nodes[n.lhs].main_token), "'{s}' is not an enum type", .{tname});
            return .invalid;
        };
    } else {
        const exp = t.expected orelse {
            try t.typeArgsForEffect(node_form, args);
            try t.emitFmt(t.byteOf(n.main_token), "cannot infer the enum type for '.{s}' here", .{t.nameText(n.main_token)});
            return .invalid;
        };
        if (!exp.isEnum()) {
            try t.typeArgsForEffect(node_form, args);
            if (exp.kind != .invalid)
                try t.emitFmt(t.byteOf(n.main_token), "'.{s}' expects an enum type, but {s} was expected here", .{ t.nameText(n.main_token), t.typeName(exp) });
            return .invalid;
        }
        enum_id = exp.enum_id;
    }
    return t.checkVariant(enum_id, n.main_token, node_form, args);
}

/// Type a QUALIFIED variant construction reinterpreted from `field_access`/`call`
/// (`N.V`, `N.V(args)`). `type_node` is the enum type-name identifier; `vtok` the
/// variant-name token; `args` the arg/field-init range (or `none` for unit).
fn typeOfEnumInitQualified(t: *Typecheck, node_idx: Ast.Index, node_form: InitForm, type_node: Ast.Index, vtok: u32, args: Ast.Index) error{OutOfMemory}!Type {
    const tname = t.nameText(t.tree.nodes[type_node].main_token);
    const enum_id = t.activeEnumMap().get(tname) orelse return .invalid; // caller checked
    const ty = try t.checkVariant(enum_id, vtok, node_form, args);
    t.node_types[node_idx] = ty;
    return ty;
}

/// Type a construction's payload args for effect only (so their own errors still
/// surface) when the enum/variant could not be resolved.
fn typeArgsForEffect(t: *Typecheck, node_form: InitForm, args: Ast.Index) error{OutOfMemory}!void {
    if (args == Ast.none) return;
    if (node_form == .@"struct") {
        for (Ast.rangeSlice(t.tree, args)) |fi| _ = try t.typeOf(t.tree.nodes[fi].lhs);
    } else {
        for (Ast.rangeSlice(t.tree, args)) |a| _ = try t.typeOf(a);
    }
}

/// Check a variant construction against its declared variant: find the variant by
/// name, check the node's form matches the variant's form, and check payload
/// arity + per-element types (propagating the expected element type so a nested
/// inferred `.V` in a payload resolves). Yields the enum type.
fn checkVariant(t: *Typecheck, enum_id: u32, vtok: u32, node_form: InitForm, args: Ast.Index) error{OutOfMemory}!Type {
    const e = t.enums.items[enum_id];
    t.recordBodyLayout(Type.enumT(enum_id)); // body -> layout(enum): construction reads the tag/payload layout (M17)
    const vname = t.nameText(vtok);
    var vi: ?usize = null;
    for (e.variants, 0..) |v, i| {
        if (std.mem.eql(u8, v.name, vname)) {
            vi = i;
            break;
        }
    }
    const variant = if (vi) |i| e.variants[i] else {
        try t.typeArgsForEffect(node_form, args);
        try t.emitFmt(t.byteOf(vtok), "enum '{s}' has no variant '{s}'", .{ e.name, vname });
        return .invalid;
    };
    const want_form: InitForm = switch (variant.form) {
        .unit => .unit,
        .tuple => .tuple,
        .@"struct" => .@"struct",
    };
    if (node_form != want_form) {
        try t.typeArgsForEffect(node_form, args);
        try t.emitFmt(t.byteOf(vtok), "variant '{s}.{s}' is constructed with the wrong form", .{ e.name, vname });
        return Type.enumT(enum_id);
    }
    switch (variant.form) {
        .unit => {},
        .tuple => {
            const elems = if (args == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(t.tree, args);
            if (elems.len != variant.field_types.len) {
                for (elems) |a| _ = try t.typeOf(a);
                try t.emitFmt(t.byteOf(vtok), "variant '{s}.{s}' expects {d} value(s), got {d}", .{ e.name, vname, variant.field_types.len, elems.len });
                return Type.enumT(enum_id);
            }
            for (elems, variant.field_types) |a, fty| {
                const at = try t.typeOfExpected(a, fty);
                if (at.kind != .invalid and fty.kind != .invalid and !Type.eql(at, fty))
                    try t.emitFmt(t.byteOf(t.tree.nodes[a].main_token), "variant '{s}.{s}': expected {s}, got {s}", .{ e.name, vname, t.typeName(fty), t.typeName(at) });
            }
        },
        .@"struct" => {
            const inits = if (args == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(t.tree, args);
            var seen = try t.gpa.alloc(bool, variant.field_names.len);
            defer t.gpa.free(seen);
            @memset(seen, false);
            for (inits) |fi_idx| {
                const fi = t.tree.nodes[fi_idx];
                const fname = t.nameText(fi.main_token);
                var found: ?usize = null;
                for (variant.field_names, 0..) |dn, j| {
                    if (std.mem.eql(u8, dn, fname)) {
                        found = j;
                        break;
                    }
                }
                if (found) |j| {
                    if (seen[j]) try t.emitFmt(t.byteOf(fi.main_token), "duplicate field '{s}' in '{s}.{s}'", .{ fname, e.name, vname });
                    seen[j] = true;
                    const fty = variant.field_types[j];
                    const vt = try t.typeOfExpected(fi.lhs, fty);
                    if (vt.kind != .invalid and fty.kind != .invalid and !Type.eql(vt, fty))
                        try t.emitFmt(t.byteOf(fi.main_token), "field '{s}': expected {s}, got {s}", .{ fname, t.typeName(fty), t.typeName(vt) });
                } else {
                    _ = try t.typeOf(fi.lhs);
                    try t.emitFmt(t.byteOf(fi.main_token), "unknown field '{s}' in '{s}.{s}'", .{ fname, e.name, vname });
                }
            }
            for (variant.field_names, 0..) |dn, j| {
                if (!seen[j]) try t.emitFmt(t.byteOf(vtok), "missing field '{s}' in '{s}.{s}'", .{ dn, e.name, vname });
            }
        },
    }
    return Type.enumT(enum_id);
}

/// Coverage state for a match, by scrutinee kind. Enum: a per-variant seen bitmap.
/// Bool: which of true/false a literal arm has covered. Int: nothing (an infinite
/// domain — exhaustiveness only via `_`).
const Cov = union(enum) {
    @"enum": []bool,
    @"bool": *BoolCov,
    int,
};
const BoolCov = struct { t: bool = false, f: bool = false };

/// Type a `match scrut { pat [if g] -> body, ... }` expression. The scrutinee may
/// be an enum, int, or bool. Each arm's pattern is checked + binds its payload; an
/// optional guard is typed as bool; all arm bodies merge to one type; the match
/// must be exhaustive (enum: every variant or `_`; bool: true+false or `_`; int:
/// `_` required). A guarded arm does NOT count toward coverage.
fn typeOfMatch(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    const st = try t.typeOf(n.lhs);
    const arms = Ast.rangeSlice(t.tree, n.rhs);
    if (st.kind == .invalid) {
        // Poison-absorb: still walk arms (bodies may have their own errors) but
        // don't emit a scrutinee or exhaustiveness error.
        for (arms) |arm_idx| _ = try t.typeOf(Ast.armHeaderAt(t.tree, t.tree.nodes[arm_idx].rhs).body);
        t.node_types[node_idx] = .invalid;
        return .invalid;
    }
    if (st.kind != .@"enum" and st.kind != .int and st.kind != .bool) {
        for (arms) |arm_idx| _ = try t.typeOf(Ast.armHeaderAt(t.tree, t.tree.nodes[arm_idx].rhs).body);
        try t.emitFmt(t.byteOf(n.main_token), "match scrutinee must be an enum, int, or bool, got {s}", .{t.typeName(st)});
        t.node_types[node_idx] = .invalid;
        return .invalid;
    }

    // body -> layout(enum): codegen reads the scrutinee's variant tags + payload
    // offsets, a dependency of THIS body (M17). No-op for int/bool scrutinees.
    t.recordBodyLayout(st);

    var seen: []bool = &.{};
    var bool_cov: BoolCov = .{};
    var cov: Cov = switch (st.kind) {
        .@"enum" => blk: {
            seen = try t.gpa.alloc(bool, t.enums.items[st.enum_id].variants.len);
            @memset(seen, false);
            break :blk .{ .@"enum" = seen };
        },
        .bool => .{ .@"bool" = &bool_cov },
        else => .int,
    };
    defer if (st.kind == .@"enum") t.gpa.free(seen);

    var has_wildcard = false;
    var result: Type = Type.never;
    for (arms) |arm_idx| {
        const arm = t.tree.nodes[arm_idx];
        const h = Ast.armHeaderAt(t.tree, arm.rhs);
        const guarded = h.guard != Ast.none;
        try t.checkPattern(arm.lhs, st, &cov, &has_wildcard, !guarded);
        if (guarded) {
            const gt = try t.typeOf(h.guard);
            if (gt.kind != .invalid and gt.kind != .bool)
                try t.emitFmt(t.byteOf(t.tree.nodes[h.guard].main_token), "match guard must be bool, got {s}", .{t.typeName(gt)});
        }
        const body_ty0 = try t.typeOfExpected(h.body, t.expected);
        const body_ty: Type = if (t.armDiverges(h.body)) Type.never else body_ty0;
        result = try t.merge(n.main_token, result, body_ty);
    }
    if (!has_wildcard) switch (cov) {
        .@"enum" => |sv| {
            const e = t.enums.items[st.enum_id];
            for (e.variants, 0..) |v, i| {
                if (!sv[i]) try t.emitFmt(t.byteOf(n.main_token), "non-exhaustive match: missing variant '{s}'", .{v.name});
            }
        },
        .bool => |bc| if (!(bc.t and bc.f))
            try t.emitFmt(t.byteOf(n.main_token), "non-exhaustive match: bool requires both true and false (or '_')", .{}),
        .int => try t.emitFmt(t.byteOf(n.main_token), "non-exhaustive match: int match requires '_'", .{}),
    };
    t.node_types[node_idx] = result;
    return result;
}

/// Whether `pat` matches EVERY value of type `ty` (covers the whole position): a
/// wildcard, a bind-whole, a SINGLE-variant enum's variant whose payload is fully
/// covered, or an or-pattern that jointly covers `ty`. A literal is refutable, and
/// — crucially — a variant pattern is REFUTABLE against a MULTI-variant enum (`.V`
/// does not match `.W`). Type-aware on purpose: a type-less version wrongly treats
/// `.Out(.V(x))` as total and accepts a non-exhaustive match (runtime fall-through).
fn irrefutable(t: *const Typecheck, pat_idx: Ast.Index, ty: Type) bool {
    if (ty.kind == .invalid) return true; // poison already reported; don't cascade a spurious miss
    const pat = t.tree.nodes[pat_idx];
    return switch (pat.tag) {
        .pattern_wildcard => true,
        .pattern_binding => pat.rhs == Ast.none or t.irrefutable(pat.rhs, ty),
        .pattern_literal => false,
        .pattern_variant => blk: {
            // Total only when the enum has exactly ONE variant (the tag test cannot
            // fail) AND that variant's payload is fully covered. Against a
            // multi-variant enum a single `.V` is refutable.
            if (ty.kind != .@"enum") break :blk false;
            const e = t.enums.items[ty.enum_id];
            if (e.variants.len != 1) break :blk false;
            break :blk t.variantPayloadIrrefutable(pat_idx, e.variants[0]);
        },
        .pattern_or => t.orCoversType(pat_idx, ty),
        else => false,
    };
}

/// Whether a variant pattern `.V(payload)`'s payload sub-patterns are each
/// irrefutable against their field types — i.e. the arm fully handles variant V.
/// `.V(_)` / `.V(x)` cover V; `.V(0)` and `.V(.W(x))` over a multi-variant inner
/// enum do NOT. Omitted struct fields are implicitly wildcards (irrefutable).
fn variantPayloadIrrefutable(t: *const Typecheck, pat_idx: Ast.Index, variant: VariantSym) bool {
    const pat = t.tree.nodes[pat_idx];
    const binders = if (pat.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(t.tree, pat.rhs);
    switch (variant.form) {
        .unit => return binders.len == 0,
        .tuple => {
            if (binders.len != variant.field_types.len) return false;
            for (binders, variant.field_types) |b, fty| if (!t.irrefutable(b, fty)) return false;
            return true;
        },
        .@"struct" => {
            for (binders) |b_idx| {
                const b = t.tree.nodes[b_idx];
                const src = if (b.lhs != Ast.none) t.nameText(t.tree.nodes[b.lhs].main_token) else t.nameText(b.main_token);
                var fty: Type = .invalid;
                for (variant.field_names, 0..) |dn, j| if (std.mem.eql(u8, dn, src)) {
                    fty = variant.field_types[j];
                    break;
                };
                if (!t.irrefutable(b_idx, fty)) return false;
            }
            return true;
        },
    }
}

/// Whether an or-pattern jointly covers every value of `ty`: a single irrefutable
/// alternative covers it, or (for an enum) the alternatives' covered variants union
/// to all of them. Conservatively false otherwise — sound: under-claiming coverage
/// only rejects a genuinely-exhaustive match, never accepts a non-exhaustive one.
fn orCoversType(t: *const Typecheck, or_idx: Ast.Index, ty: Type) bool {
    const alts = Ast.rangeSlice(t.tree, t.tree.nodes[or_idx].lhs);
    for (alts) |a| if (t.irrefutable(a, ty)) return true;
    if (ty.kind == .@"enum") {
        const e = t.enums.items[ty.enum_id];
        var seen = [_]bool{false} ** 64;
        if (e.variants.len > seen.len) return false;
        for (alts) |a| {
            const ap = t.tree.nodes[a];
            if (ap.tag != .pattern_variant) continue;
            const vname = t.nameText(ap.main_token);
            for (e.variants, 0..) |v, i| {
                if (std.mem.eql(u8, v.name, vname) and t.variantPayloadIrrefutable(a, v)) seen[i] = true;
            }
        }
        for (e.variants, 0..) |_, i| if (!seen[i]) return false;
        return true;
    }
    return false;
}

/// Whether an arm body (an expression node) diverges (control never falls past
/// it), so its value side is unreachable in the merge. A bare expr body never
/// diverges by itself; a block/if/loop/match/return body might.
fn armDiverges(t: *const Typecheck, node_idx: Ast.Index) bool {
    return stmtDiverges(t, node_idx);
}

/// Check a match pattern against the `expected` scrutinee/field type. `count_cov`
/// is true only at a top-level (non-nested) position of an UNGUARDED arm — then an
/// irrefutable wildcard/bind sets `has_wildcard`, a bool literal records its case,
/// and a fully-irrefutable variant marks `seen[variant]`. Nested positions pass
/// `count_cov=false` (a nested literal must not claim coverage). Payload bindings
/// are typed BY VALUE onto their `.local` slots (set by Resolve).
fn checkPattern(t: *Typecheck, pat_idx: Ast.Index, expected: Type, cov: *Cov, has_wildcard: *bool, count_cov: bool) error{OutOfMemory}!void {
    const pat = t.tree.nodes[pat_idx];
    switch (pat.tag) {
        .pattern_wildcard => if (count_cov) {
            has_wildcard.* = true;
        },
        .pattern_binding => {
            // Record the type this binding matched AGAINST on its own node. The
            // binding's slot is SHARED across or-pattern alternatives (Resolve), so
            // the slot type is overwritten and can't reveal a `.A(x) | .B(x)` type
            // divergence; the per-node matched type can (read by `collectBindings`).
            t.node_types[pat_idx] = expected;
            if (pat.rhs == Ast.none) {
                // Bind-whole: type by value; a top-level bare binding is irrefutable.
                if (t.resolutions[pat_idx] == .local) try t.setSlot(t.resolutions[pat_idx].local, expected);
                if (count_cov) has_wildcard.* = true;
            } else {
                try t.checkPattern(pat.rhs, expected, cov, has_wildcard, count_cov);
            }
        },
        .pattern_literal => {
            const lt: Type = if (t.tokens[pat.main_token].tag == .number) Type.int else Type.@"bool";
            if (expected.kind != .invalid and !Type.eql(lt, expected))
                try t.emitFmt(t.byteOf(pat.main_token), "literal pattern type {s} does not match scrutinee {s}", .{ t.typeName(lt), t.typeName(expected) });
            // A bool literal records its case toward coverage; int never covers.
            if (count_cov) switch (cov.*) {
                .bool => |bc| {
                    if (lt.kind == .bool) {
                        if (std.mem.eql(u8, t.nameText(pat.main_token), "true")) bc.t = true else bc.f = true;
                    }
                },
                else => {},
            };
        },
        .pattern_or => {
            for (Ast.rangeSlice(t.tree, pat.lhs)) |a| try t.checkPattern(a, expected, cov, has_wildcard, count_cov);
            try t.checkOrBindings(pat_idx);
        },
        .pattern_variant => try t.checkVariantPattern(pat_idx, expected, cov, has_wildcard, count_cov),
        else => {},
    }
}

/// Check a `.V` / `N.V` variant pattern. Requires an enum scrutinee; marks the
/// variant covered only when `count_cov` AND the whole pattern is irrefutable
/// (so `.C(0)` does NOT cover `.C`). Recurses into payload sub-patterns with the
/// field type and `count_cov=false`.
fn checkVariantPattern(t: *Typecheck, pat_idx: Ast.Index, expected: Type, cov: *Cov, has_wildcard: *bool, count_cov: bool) error{OutOfMemory}!void {
    const pat = t.tree.nodes[pat_idx];
    if (expected.kind != .@"enum") {
        if (expected.kind != .invalid)
            try t.emitFmt(t.byteOf(pat.main_token), "variant pattern on a non-enum scrutinee {s}", .{t.typeName(expected)});
        return;
    }
    const enum_id = expected.enum_id;
    const e = t.enums.items[enum_id];
    t.recordBodyLayout(expected); // body -> layout(enum): pattern reads variant tags/payloads, incl. NESTED enums (M17)
    // A qualified `N.V` pattern: the type-name must name the scrutinee enum.
    if (pat.lhs != Ast.none) {
        const tname = t.nameText(t.tree.nodes[pat.lhs].main_token);
        if (t.activeEnumMap().get(tname)) |qid| {
            if (qid != enum_id)
                try t.emitFmt(t.byteOf(t.tree.nodes[pat.lhs].main_token), "pattern enum '{s}' does not match scrutinee '{s}'", .{ tname, e.name });
        } else {
            try t.emitFmt(t.byteOf(t.tree.nodes[pat.lhs].main_token), "'{s}' is not an enum type", .{tname});
        }
    }
    const vname = t.nameText(pat.main_token);
    var vi: ?usize = null;
    for (e.variants, 0..) |v, i| {
        if (std.mem.eql(u8, v.name, vname)) {
            vi = i;
            break;
        }
    }
    const variant = if (vi) |i| blk: {
        // Cover variant i only when this arm counts AND variant i's payload is fully
        // matched (`.V(_)`/`.V(x)` cover it; `.V(0)` or `.V(.W(x))` over a multi-variant
        // inner enum do not — caught by the type-aware payload check).
        if (count_cov and t.variantPayloadIrrefutable(pat_idx, e.variants[i])) cov.@"enum"[i] = true;
        break :blk e.variants[i];
    } else {
        try t.emitFmt(t.byteOf(pat.main_token), "enum '{s}' has no variant '{s}'", .{ e.name, vname });
        return;
    };
    const binders = if (pat.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(t.tree, pat.rhs);
    switch (variant.form) {
        .unit => {
            if (binders.len != 0)
                try t.emitFmt(t.byteOf(pat.main_token), "unit variant '{s}.{s}' binds no payload", .{ e.name, vname });
        },
        .tuple => {
            if (binders.len != variant.field_types.len) {
                try t.emitFmt(t.byteOf(pat.main_token), "variant '{s}.{s}' binds {d} value(s), got {d}", .{ e.name, vname, variant.field_types.len, binders.len });
                return;
            }
            for (binders, variant.field_types) |b_idx, fty| {
                try t.checkPattern(b_idx, fty, cov, has_wildcard, false);
            }
        },
        .@"struct" => {
            for (binders) |b_idx| {
                const b = t.tree.nodes[b_idx];
                // A struct binding's SOURCE field name is the rename source (lhs),
                // or the bound name itself when punning.
                const src_name = if (b.lhs != Ast.none) t.nameText(t.tree.nodes[b.lhs].main_token) else t.nameText(b.main_token);
                var fty: Type = .invalid;
                var found = false;
                for (variant.field_names, 0..) |dn, j| {
                    if (std.mem.eql(u8, dn, src_name)) {
                        fty = variant.field_types[j];
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    try t.emitFmt(t.byteOf(b.main_token), "no field '{s}' in '{s}.{s}'", .{ src_name, e.name, vname });
                }
                // The carrier IS a pattern_binding: if it has a sub-pattern, match
                // the field against it; else bind the whole field by value.
                try t.checkPattern(b_idx, fty, cov, has_wildcard, false);
            }
        },
    }
}

/// Verify all alternatives of an or-pattern bind the SAME set of names with the
/// SAME types. Collects each alt's `{name -> Type}` (from the binding nodes'
/// resolved slot types) and compares against the first.
fn checkOrBindings(t: *Typecheck, or_idx: Ast.Index) error{OutOfMemory}!void {
    const alts = Ast.rangeSlice(t.tree, t.tree.nodes[or_idx].lhs);
    if (alts.len < 2) return;
    var first_map: std.StringHashMapUnmanaged(Type) = .empty;
    defer first_map.deinit(t.gpa);
    try t.collectBindings(alts[0], &first_map);
    var ok = true;
    for (alts[1..]) |alt| {
        var m: std.StringHashMapUnmanaged(Type) = .empty;
        defer m.deinit(t.gpa);
        try t.collectBindings(alt, &m);
        if (m.count() != first_map.count()) {
            ok = false;
        } else {
            var it = m.iterator();
            while (it.next()) |entry| {
                const want = first_map.get(entry.key_ptr.*) orelse {
                    ok = false;
                    break;
                };
                if (!Type.eql(want, entry.value_ptr.*)) {
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) break;
    }
    if (!ok)
        try t.emitFmt(t.byteOf(t.tree.nodes[or_idx].main_token), "or-pattern alternatives must bind the same names and types", .{});
}

/// Walk a pattern's binding leaves into `out` as `name -> slot type`.
fn collectBindings(t: *Typecheck, pat_idx: Ast.Index, out: *std.StringHashMapUnmanaged(Type)) error{OutOfMemory}!void {
    const pat = t.tree.nodes[pat_idx];
    switch (pat.tag) {
        .pattern_binding => {
            const name = t.nameText(pat.main_token);
            // The PER-NODE matched type (set in checkPattern), NOT the shared slot
            // type — so two alternatives binding the same name at different field
            // types are seen as different and rejected.
            const ty: Type = t.node_types[pat_idx];
            try out.put(t.gpa, name, ty);
            if (pat.rhs != Ast.none) try t.collectBindings(pat.rhs, out);
        },
        .pattern_variant => if (pat.rhs != Ast.none)
            for (Ast.rangeSlice(t.tree, pat.rhs)) |c| try t.collectBindings(c, out),
        .pattern_or => for (Ast.rangeSlice(t.tree, pat.lhs)) |a| try t.collectBindings(a, out),
        else => {},
    }
}

/// Type an `if`/`else` used as a VALUE expression. Both arms must agree:
/// the merge is the one phi/join. An else-less value-if is a type error
/// ("value-if requires else"). A fully-diverging arm (returns on all paths) is
/// compatible with the other arm's concrete type (its value side is unreachable).
fn typeOfIf(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    _ = node_idx;
    const ct = try t.typeOf(n.lhs);
    if (ct.kind != .invalid and ct.kind != .bool)
        try t.emit(t.byteOf(t.tree.nodes[n.lhs].main_token), "if condition must be bool");
    const h = Ast.ifHeaderAt(t.tree, n.rhs);
    if (h.else_node == Ast.none) {
        try t.emit(t.byteOf(n.main_token), "value-if requires else");
        _ = try t.checkBlock(h.then_block, false); // validate the arm anyway
        return .invalid;
    }
    const then_ty0 = try t.checkBlock(h.then_block, true);
    const then_ty: Type = if (blockDiverges(t, h.then_block)) Type.never else then_ty0;
    var else_ty: Type = undefined;
    if (t.tree.nodes[h.else_node].tag == .if_stmt) {
        const e0 = try t.typeOfIf(h.else_node, t.tree.nodes[h.else_node]); // else-if ladder
        else_ty = if (stmtDiverges(t, h.else_node)) Type.never else e0;
    } else {
        const e0 = try t.checkBlock(h.else_node, true);
        else_ty = if (blockDiverges(t, h.else_node)) Type.never else e0;
    }
    return t.merge(n.main_token, then_ty, else_ty);
}

/// Type a `loop` used as a value expression. Its body is checked in statement
/// context; each value-break fills the loop's `join`. A loop with NO break at
/// all is `never` (it can only exit via `return` or an outer construct).
fn typeOfLoop(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node, label: ?[]const u8) error{OutOfMemory}!Type {
    try t.loop_stack.append(t.gpa, .{ .kind = .loop, .label = label, .construct_node = node_idx, .is_value = true, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
    _ = try t.checkBlock(n.lhs, false); // body in statement ctx; breaks fill join
    const ctx = t.loop_stack.pop().?;
    const ty: Type = if (!ctx.saw_value_break and !ctx.saw_bare_break) Type.never else ctx.join;
    t.node_types[node_idx] = ty;
    return ty;
}

/// Merge two value types into a single join. `never` is the bottom type: it
/// unifies with anything (a `never` expression never produces a value), so
/// `never + T = T` and `never + never = never`. No implicit coercion otherwise:
/// a mismatch is a type error.
fn merge(t: *Typecheck, at: u32, a: Type, b: Type) error{OutOfMemory}!Type {
    if (a.kind == .invalid or b.kind == .invalid) return .invalid; // poison absorbs
    if (a.kind == .never) return b; // covers never+never → never
    if (b.kind == .never) return a;
    if (Type.eql(a, b)) return a; // agreement → one phi type
    try t.emitFmt(t.byteOf(at), "branches yield different types ({s} vs {s})", .{ t.typeName(a), t.typeName(b) });
    return .invalid; // mismatch, no coercion
}

// ===========================================================================
// M16 — fine-grained typecheck QUERY PROJECTIONS (observational dependency DAG).
//
// All of these are PURE READ projections over the already-built tables
// (`t.fns`/`t.structs`/`t.enums`/`t.node_types`/`t.resolutions`): the whole-graph
// pass still computes every value exactly once, and these helpers merely READ the
// table that holds the answer and record a `body(caller) -> <callee node>` edge
// into the per-build `*Dag`. When `t.dag == null` (every default build) each one
// early-returns a no-op, so the result is byte-identical and the path is verbatim.
//
//   * signature(fn): a CALLER's body depends on `signature(callee)`, NEVER on
//     body(callee) — the firewall. fp = `Dag.sigFingerprint` (sig-only fold). The
//     node id matches codegen's `Wyhash("SGNM", name)` so a body(caller)->signature
//     and a codegen(caller)->signature edge land on the SAME signature node.
//   * type_of(def): coarse — recorded at the call boundary only; fp folds the Type.
//   * layout(type): mirrors Walks.structLayoutBytes/enumLayoutBytes EXACTLY, reusing
//     the .laying/.done guard so a recursive aggregate emits a sentinel, never loops.
//   * resolve_name(name): coarse firewall; resolve is NOT engine-routed, so the edge
//     is recorded directly from the typecheck read site. fp folds the Resolution.
//
// AUDIT — every CROSS-ITEM read inside a fn body is routed through a projection:
//   * the callee SIGNATURE read `t.fns.items[callee]` (typeOfCall + checkPubSignatures)
//     -> recordSignature (body|sig -> signature edge);
//   * the callee NAME read `t.resolutions[n.lhs]` (typeOfCall) -> recordResolveName
//     (body -> resolve_name edge);
//   * the arg/return aggregate layout read (`t.structs`/`t.enums`) -> recordLayoutOf
//     (sig|type_of -> layout edge);
//   * the call-boundary inferred types -> recordTypeOf (body -> type_of edge).
// INTRA-body reads (a `.local` slot type, a label) stay direct — they are not a
// cross-query dependency and resolve is deliberately COARSE (one node per consumed
// callee name, deduped). All four helpers no-op when `dag == null`, so the DEFAULT
// build path is verbatim/byte-identical and these edges exist ONLY under `--dump-dag`.
// ===========================================================================

/// The stable `signature(fn)` node id for global fn `fid`, folding the SAME name
/// the codegen callee-sig fold uses (graph: the qualified `gph_fn_names[fid]`;
/// single-file: the decl-token spelling, with the synthetic bodyless fn = "print").
/// Must mirror `Engine.codegen`'s `Wyhash(0x53_47_4e_4d, csig.name)` so the typecheck
/// body->signature edge and the codegen->signature edge share one node.
fn sigNodeName(t: *const Typecheck, fid: u32) []const u8 {
    if (t.gph_fn_names) |names| {
        if (fid < names.len) return names[fid];
        return "print";
    }
    const f = t.fns.items[fid];
    if (f.decl_node == Ast.none) return "print";
    return t.nameText(t.tree.nodes[f.decl_node].main_token);
}

fn sigNodeId(t: *const Typecheck, fid: u32) u64 {
    return std.hash.Wyhash.hash(0x53_47_4e_4d, t.sigNodeName(fid)); // "SGNM"
}

/// Record `body(caller) -> signature(fid)` with the sig-only fingerprint. The
/// CALLER's `body` node is the Active parent (entered in `checkFn`). The sig is
/// rebuilt from the SAME `t.fns.items[fid]` table the whole-graph pass already
/// filled (Pass A) — a pure read, no recompute. No-op when `dag == null`.
fn recordSignature(t: *Typecheck, fid: u32) void {
    const d = t.dag orelse return;
    if (fid >= t.fns.items.len) return;
    const f = t.fns.items[fid];
    const sig: Sig = .{
        .kind = if (f.decl_node == Ast.none) .builtin else .user_fn,
        .name = t.sigNodeName(fid),
        .params = f.params,
        .ret = f.ret,
    };
    const node: Dag.NodeKey = .{ .kind = .signature, .id = t.sigNodeId(fid) };
    d.recordEdge(t.gpa, Dag.Active.get(), node, Dag.sigFingerprint(sig));
    // The signature's aggregate param/return types are layout dependencies of the
    // signature node (a layout edit to a named aggregate flows to its sig).
    for (f.params) |p| t.recordLayoutOf(node, p);
    t.recordLayoutOf(node, f.ret);
}

/// Record `body(caller) -> resolve_name(name)` for a consumed resolution. Coarse:
/// one node per resolved symbol, deduped by `recordEdge`. fp folds the Resolution
/// shape (kind tag + bound id). No-op when `dag == null`.
fn recordResolveName(t: *Typecheck, res: Resolution) void {
    const d = t.dag orelse return;
    var h = std.hash.Wyhash.init(0x52_4e_4d_45); // "RNME"
    const tag: u8 = @intFromEnum(std.meta.activeTag(res));
    h.update(&[_]u8{tag});
    const bound: u32 = switch (res) {
        .local => |l| l,
        .func => |fi| fi,
        else => 0,
    };
    var ib: [4]u8 = undefined;
    std.mem.writeInt(u32, &ib, bound, .little);
    h.update(&ib);
    const fp = h.final();
    const node: Dag.NodeKey = .{ .kind = .resolve_name, .id = fp };
    d.recordEdge(t.gpa, Dag.Active.get(), node, fp);
}

/// Record `body(caller) -> type_of(node_idx)` for a typed def/expr at a cross-item
/// boundary (the call result + each arg). Coarse — recorded ONLY at the call
/// boundary, not per leaf. fp folds the Type. No-op when `dag == null`.
fn recordTypeOf(t: *Typecheck, node_idx: Ast.Index, ty: Type) void {
    const d = t.dag orelse return;
    var h = std.hash.Wyhash.init(0x54_4f_46_5f); // "TOF_"
    h.update(&[_]u8{@intFromEnum(ty.kind)});
    var ib: [8]u8 = undefined;
    std.mem.writeInt(u32, ib[0..4], ty.struct_id, .little);
    std.mem.writeInt(u32, ib[4..8], ty.enum_id, .little);
    h.update(&ib);
    // node_idx is a MODULE-LOCAL Ast.Index; in graph mode two modules can share an
    // index, collapsing distinct boundary types to one type_of node (last-writer-wins
    // fp masks a real type change => M17 miscompile). Fold the current module id into
    // the node id (mirroring how `body` uses the global fid), so single-file mode
    // (one module) is unchanged but cross-module nodes stay distinct.
    const node: Dag.NodeKey = .{ .kind = .type_of, .id = (@as(u64, t.graph_mod) << 32) | node_idx };
    d.recordEdge(t.gpa, Dag.Active.get(), node, h.final());
}

/// Record `parent -> layout(type)` when `ty` names an aggregate (struct/enum). fp
/// mirrors Walks.structLayoutBytes/enumLayoutBytes EXACTLY over the laid-out
/// `t.structs`/`t.enums` snapshots, so the recorded layout fp flips iff the codegen
/// layout bytes flip. No-op when `dag == null` or `ty` is a scalar.
fn recordLayoutOf(t: *Typecheck, parent: Dag.NodeKey, ty: Type) void {
    const d = t.dag orelse return;
    switch (ty.kind) {
        .@"struct" => {
            const node: Dag.NodeKey = .{ .kind = .layout, .id = ty.struct_id };
            d.recordEdge(t.gpa, parent, node, t.structLayoutFp(ty.struct_id));
        },
        .@"enum" => {
            // struct ids and enum ids are SEPARATE 0-based sequences; without a kind
            // tag struct(N) and enum(N) collapse to one layout#N node and recordEdge's
            // unconditional fp upsert lets the last writer mask the other's layout edit
            // (invisible change => M17 miscompile). Reserve the high bit for enums.
            const node: Dag.NodeKey = .{ .kind = .layout, .id = @as(u64, ty.enum_id) | (@as(u64, 1) << 63) };
            d.recordEdge(t.gpa, parent, node, t.enumLayoutFp(ty.enum_id));
        },
        else => {},
    }
}

/// Record `body(current) -> layout(ty)` for an aggregate READ inside a function
/// body (struct construct / field access, enum construct / match / variant
/// pattern). The body is the Active node. Without this, a layout-only edit to an
/// aggregate used ONLY in a body (not named in that body's signature) is invisible
/// to the fine-grained DAG — a stale body/codegen result under M17 (the M16
/// signature firewall covers signatures only). No-op when `dag == null` (verbatim).
fn recordBodyLayout(t: *Typecheck, ty: Type) void {
    if (t.dag == null) return;
    const parent = Dag.Active.get() orelse return;
    t.recordLayoutOf(parent, ty);
}

/// Fold struct `id`'s layout into a u64, mirroring `Walks.structLayoutBytes`
/// byte-for-byte (name\0 + per-field name\0+kind+offset, recursing nested structs,
/// + size + align). Folds ONLY a `.done` snapshot; a `.laying`/poisoned child emits
/// a sentinel and does NOT recurse (the recursive-struct guard — KNOWN BUG a).
fn structLayoutFp(t: *const Typecheck, id: u32) u64 {
    var h = std.hash.Wyhash.init(0x4c_41_59_53); // "LAYS"
    t.foldStructLayout(&h, id);
    return h.final();
}
fn foldStructLayout(t: *const Typecheck, h: *std.hash.Wyhash, id: u32) void {
    const s = t.structs.items[id];
    if (s.state != .done or s.poisoned) {
        h.update("<rec>"); // sentinel: a recursive/poisoned referent (matches guard)
        return;
    }
    h.update(s.name);
    h.update(&[_]u8{0});
    for (s.field_names, s.field_types, s.offsets) |fn_, fty, off| {
        h.update(fn_);
        h.update(&[_]u8{0});
        h.update(&[_]u8{@intFromEnum(fty.kind)});
        var ob: [4]u8 = undefined;
        std.mem.writeInt(u32, &ob, off, .little);
        h.update(&ob);
        if (fty.kind == .@"struct") t.foldStructLayout(h, fty.struct_id);
    }
    var sz: [8]u8 = undefined;
    std.mem.writeInt(u32, sz[0..4], s.size, .little);
    std.mem.writeInt(u32, sz[4..8], s.@"align", .little);
    h.update(&sz);
}

/// Fold enum `id`'s layout into a u64, mirroring `Walks.enumLayoutBytes`
/// byte-for-byte. Reuses the `.done`/sentinel guard so a recursive enum never loops.
fn enumLayoutFp(t: *const Typecheck, id: u32) u64 {
    var h = std.hash.Wyhash.init(0x4c_41_59_45); // "LAYE"
    t.foldEnumLayout(&h, id);
    return h.final();
}
fn foldEnumLayout(t: *const Typecheck, h: *std.hash.Wyhash, id: u32) void {
    const e = t.enums.items[id];
    if (e.state != .done or e.poisoned) {
        h.update("<rec>");
        return;
    }
    h.update(e.name);
    h.update(&[_]u8{0});
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], e.tag_size, .little);
    std.mem.writeInt(u32, hdr[4..8], e.payload_off, .little);
    h.update(&hdr);
    for (e.variants) |v| {
        h.update(v.name);
        h.update(&[_]u8{0});
        h.update(&[_]u8{@intFromEnum(v.form)});
        if (v.form == .@"struct") {
            for (v.field_names, v.field_types, v.offsets) |fn_, fty, off| {
                h.update(fn_);
                h.update(&[_]u8{0});
                h.update(&[_]u8{@intFromEnum(fty.kind)});
                var ob: [4]u8 = undefined;
                std.mem.writeInt(u32, &ob, off, .little);
                h.update(&ob);
                if (fty.kind == .@"struct") t.foldStructLayout(h, fty.struct_id);
                if (fty.kind == .@"enum") t.foldEnumLayout(h, fty.enum_id);
            }
        } else {
            for (v.field_types, v.offsets) |fty, off| {
                h.update(&[_]u8{@intFromEnum(fty.kind)});
                var ob: [4]u8 = undefined;
                std.mem.writeInt(u32, &ob, off, .little);
                h.update(&ob);
                if (fty.kind == .@"struct") t.foldStructLayout(h, fty.struct_id);
                if (fty.kind == .@"enum") t.foldEnumLayout(h, fty.enum_id);
            }
        }
    }
    var sz: [8]u8 = undefined;
    std.mem.writeInt(u32, sz[0..4], e.size, .little);
    std.mem.writeInt(u32, sz[4..8], e.@"align", .little);
    h.update(&sz);
}

fn typeOfCall(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    // A qualified tuple-variant construction `N.V(args)` arrives as a `.call`
    // whose callee is a `field_access` over an enum type-name identifier. Route
    // it to the enum-init checker (treating `n` as a tuple construction).
    const callee = t.tree.nodes[n.lhs];
    if (callee.tag == .field_access) {
        const recv = t.tree.nodes[callee.lhs];
        if (recv.tag == .identifier and t.activeEnumMap().get(t.nameText(recv.main_token)) != null) {
            return t.typeOfEnumInitQualified(node_idx, .tuple, callee.lhs, callee.main_token, n.rhs);
        }
        // A cross-module tuple-variant `mod.Enum.Variant(args)` (graph mode): the
        // callee field_access's receiver is the inner `mod.Enum`.
        if (t.qualifiedEnumId(callee.lhs)) |enum_id| {
            const ty = try t.checkVariant(enum_id, callee.main_token, .tuple, n.rhs);
            t.node_types[node_idx] = ty;
            return ty;
        }
    }
    const callee_res = t.resolutions[n.lhs];
    // M16: a caller's body consumes the callee NAME resolution (cross-item read).
    t.recordResolveName(callee_res);
    if (callee_res != .func) {
        // Type the args anyway so their own errors surface, then poison.
        for (Ast.rangeSlice(t.tree, n.rhs)) |arg| _ = try t.typeOf(arg);
        if (callee_res == .local) {
            try t.emit(t.byteOf(n.main_token), "called value is not a function");
        } else if (t.tree.nodes[n.lhs].tag == .identifier) {
            // A struct-named callee `Point(1,2)` is positional construction, which
            // we reject — point at named construction instead.
            const cname = t.nameText(t.tree.nodes[n.lhs].main_token);
            if (t.activeStructMap().get(cname) != null)
                try t.emitFmt(t.byteOf(n.main_token), "use named construction '{s} {{ ... }}', not '{s}(...)'", .{ cname, cname });
        } else if (callee_res == .unresolved and t.tree.nodes[n.lhs].tag == .field_access) {
            // A qualified call `recv.member(...)` whose callee stayed `.unresolved`:
            // resolve neither bound it to a fn nor reported it (e.g. `recv` is a
            // top-level fn shadowing an import namespace, so the field-access value
            // path is taken and left unresolved). Emit a clean diagnostic at the
            // member token instead of silently poisoning — otherwise the call is
            // dropped and `-o` later crashes in codegen with no user error.
            const fa = t.tree.nodes[n.lhs];
            const member = t.nameText(fa.main_token);
            try t.emitFmt(t.byteOf(fa.main_token), "cannot resolve member '{s}' to a callable function", .{member});
        }
        // Any remaining `.unresolved` was already reported by resolve.
        return .invalid;
    }
    // M16 FIREWALL: this caller's body reads the callee's SIGNATURE (params/ret),
    // NEVER its body. Route the cross-item `t.fns.items[callee]` read through the
    // `signature(callee)` query so a body(caller)->signature(callee) edge is recorded
    // (fp = sig-only fold). A body-only edit to the callee leaves this fp STABLE; a
    // signature edit FLIPS it — the firewall, proven on a real program.
    t.recordSignature(callee_res.func);
    const f = t.fns.items[callee_res.func];
    const args = Ast.rangeSlice(t.tree, n.rhs);
    if (args.len != f.params.len) {
        for (args) |arg| _ = try t.typeOf(arg);
        try t.emitFmt(t.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ f.params.len, args.len });
        return f.ret;
    }
    for (args, f.params, 0..) |arg, pty, i| {
        const at = try t.typeOfExpected(arg, if (pty.kind == .invalid) null else pty);
        // M16: the arg's inferred type is a cross-boundary `type_of` read.
        t.recordTypeOf(arg, at);
        // Skip when either side is poison (e.g. a parameter whose type failed to
        // resolve) so we don't cascade a spurious "expected invalid" message.
        if (at.kind != .invalid and pty.kind != .invalid and !Type.eql(at, pty)) {
            try t.emitFmt(t.byteOf(t.tree.nodes[arg].main_token), "argument {d}: expected {s}, got {s}", .{ i + 1, t.typeName(pty), t.typeName(at) });
        }
    }
    // M16: the call result type (the callee's return) is the boundary `type_of`.
    t.recordTypeOf(node_idx, f.ret);
    return f.ret;
}

/// Map a type-reference node (an `identifier`, a `literal_unit` for `()`, or in
/// graph mode a qualified `mod.Type` `field_access`) to a `Type`.
fn typeFromNode(t: *Typecheck, type_node: Ast.Index) Type {
    if (type_node == Ast.none) return Type.unit;
    const tn = t.tree.nodes[type_node];
    if (tn.tag == .literal_unit) return Type.unit; // explicit `-> ()` / `p: ()`
    // A qualified cross-module type-ref `mod.Type` parses as a field_access whose
    // receiver binds to a `.module`. Resolve it against the owning module's tables.
    if (tn.tag == .field_access) return t.typeFromQualified(type_node, tn);
    const tok = tn.main_token;
    const name = t.nameText(tok);
    if (type_names.get(name)) |b| return b;
    if (t.activeStructMap().get(name)) |id| return Type.structT(id);
    if (t.activeEnumMap().get(name)) |id| return Type.enumT(id);
    t.emitFmt(t.byteOf(tok), "unknown type '{s}'", .{name}) catch {};
    return .invalid;
}

/// Resolve a qualified `mod.Type` type-reference (a `field_access` in type
/// position; the receiver binds to a `.module`) to the owning module's GLOBAL
/// struct/enum id. Graph mode only. Visibility was already enforced by
/// `resolve_graph`, so a private type here is a defensive `invalid`.
fn typeFromQualified(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) Type {
    _ = node_idx;
    const g = t.graph orelse {
        t.emitFmt(t.byteOf(n.main_token), "qualified type is not valid here", .{}) catch {};
        return .invalid;
    };
    const recv = t.tree.nodes[n.lhs];
    if (recv.tag != .identifier) return .invalid;
    const recv_name = t.nameText(recv.main_token);
    const target = g.namespaceOfIn(t.graph_mod, recv_name) orelse {
        t.emitFmt(t.byteOf(recv.main_token), "unknown module '{s}'", .{recv_name}) catch {};
        return .invalid;
    };
    const member = t.nameText(n.main_token);
    if (g.mods[target].struct_ids.get(member)) |id| return Type.structT(id);
    if (g.mods[target].enum_ids.get(member)) |id| return Type.enumT(id);
    t.emitFmt(t.byteOf(n.main_token), "module has no type '{s}'", .{member}) catch {};
    return .invalid;
}

/// Lay out struct `id`: field offsets in declaration order with natural
/// alignment, size = aligned total, align = max field align. A `laying` field
/// of struct type means a cycle (direct or indirect) → infinite size, poisoned.
fn layoutStruct(t: *Typecheck, id: u32) error{OutOfMemory}!void {
    if (t.structs.items[id].state == .done) return;
    t.structs.items[id].state = .laying;

    // Graph mode: lay this struct out in ITS owning module's tree (a nested/qualified
    // field type may have switched the active module). Restore on the way out.
    const prev = t.gphSelect(t.structs.items[id].mod);
    defer _ = t.gphSelect(prev);

    const decl = t.tree.nodes[t.structs.items[id].decl_node];
    const field_nodes = Ast.rangeSlice(t.tree, decl.lhs);
    const n = field_nodes.len;

    // An empty struct lays out to size 0 — a zero-size aggregate is an ABI/codegen
    // hazard (no eightbytes, a degenerate sret). Reject it with a clean diagnostic.
    var empty_poison = false;
    if (n == 0) {
        try t.emitFmt(t.byteOf(decl.main_token), "empty struct '{s}' is not allowed", .{t.structs.items[id].name});
        empty_poison = true;
    }

    const names = try t.gpa.alloc([]const u8, n);
    errdefer t.gpa.free(names);
    const types = try t.gpa.alloc(Type, n);
    errdefer t.gpa.free(types);
    const offsets = try t.gpa.alloc(u32, n);
    errdefer t.gpa.free(offsets);

    var running: u32 = 0;
    var max_align: u32 = 1;
    var poisoned = false;
    for (field_nodes, 0..) |field_idx, i| {
        const field = t.tree.nodes[field_idx];
        names[i] = t.nameText(field.main_token);
        const fty = t.typeFromNode(field.lhs);
        types[i] = fty;
        var fsize: u32 = 0;
        var falign: u32 = 1;
        if (fty.kind == .unit) {
            try t.emitFmt(t.byteOf(field.main_token), "field '{s}' cannot have type ()", .{names[i]});
            poisoned = true;
        } else if (fty.kind != .invalid) {
            const sz = try t.layoutReferent(fty, t.byteOf(decl.main_token), t.structs.items[id].name, &poisoned);
            fsize = sz.size;
            falign = sz.@"align";
        }
        const off = roundUp(running, falign);
        offsets[i] = off;
        running = off + fsize;
        if (falign > max_align) max_align = falign;
    }

    t.structs.items[id].field_names = names;
    t.structs.items[id].field_types = types;
    t.structs.items[id].offsets = offsets;
    t.structs.items[id].@"align" = max_align;
    t.structs.items[id].size = if (poisoned or empty_poison) 0 else roundUp(running, max_align);
    t.structs.items[id].poisoned = poisoned or empty_poison;
    t.structs.items[id].state = .done;
}

/// Size/align of a field/payload type, laying out a nested struct/enum on demand.
/// A `laying` referent means a cycle (direct or indirect): set `*requester_poison`
/// and emit a recursion diagnostic at `at` naming `requester`. A scalar/str uses
/// the natural sizes; `invalid`/`unit` size to 0 (the caller diagnoses `unit`).
fn layoutReferent(t: *Typecheck, ty: Type, at: u32, requester: []const u8, requester_poison: *bool) error{OutOfMemory}!struct { size: u32, @"align": u32 } {
    switch (ty.kind) {
        .@"struct" => {
            if (t.structs.items[ty.struct_id].state == .laying) {
                try t.emitFmt(at, "recursive type '{s}' has infinite size", .{requester});
                requester_poison.* = true;
                return .{ .size = 0, .@"align" = 1 };
            }
            try t.layoutStruct(ty.struct_id);
            return .{ .size = t.structs.items[ty.struct_id].size, .@"align" = t.structs.items[ty.struct_id].@"align" };
        },
        .@"enum" => {
            if (t.enums.items[ty.enum_id].state == .laying) {
                try t.emitFmt(at, "recursive type '{s}' has infinite size", .{requester});
                requester_poison.* = true;
                return .{ .size = 0, .@"align" = 1 };
            }
            try t.layoutEnum(ty.enum_id);
            return .{ .size = t.enums.items[ty.enum_id].size, .@"align" = t.enums.items[ty.enum_id].@"align" };
        },
        else => return .{ .size = scalarSize(ty.kind), .@"align" = scalarAlign(ty.kind) },
    }
}

/// Lay out enum `id`: an 8-byte tag at offset 0, then payload storage sized to the
/// largest variant's payload at `payload_off`. Each variant's payload fields get
/// payload-LOCAL offsets. A `laying` referent (direct/indirect cycle) poisons it.
fn layoutEnum(t: *Typecheck, id: u32) error{OutOfMemory}!void {
    if (t.enums.items[id].state == .done) return;
    t.enums.items[id].state = .laying;

    // Graph mode: lay this enum out in ITS owning module's tree. Restore on exit.
    const prev = t.gphSelect(t.enums.items[id].mod);
    defer _ = t.gphSelect(prev);

    const decl = t.tree.nodes[t.enums.items[id].decl_node];
    const variant_nodes = Ast.rangeSlice(t.tree, decl.lhs);
    const nv = variant_nodes.len;

    var poisoned = false;
    if (nv == 0) {
        try t.emitFmt(t.byteOf(decl.main_token), "empty enum '{s}' is not allowed", .{t.enums.items[id].name});
        poisoned = true;
    }

    const variants = try t.gpa.alloc(VariantSym, nv);
    errdefer t.gpa.free(variants);
    var vbuilt: usize = 0;
    errdefer for (variants[0..vbuilt]) |v| {
        t.gpa.free(v.field_names);
        t.gpa.free(v.field_types);
        t.gpa.free(v.offsets);
    };

    var max_payload_size: u32 = 0;
    var max_payload_align: u32 = 1;
    for (variant_nodes, 0..) |vnode_idx, vi| {
        const vnode = t.tree.nodes[vnode_idx];
        const vname = t.nameText(vnode.main_token);
        var form: VariantForm = .unit;
        var payload_nodes: []const Ast.Index = &.{};
        var is_struct_form = false;
        switch (vnode.tag) {
            .enum_variant_unit => {},
            .enum_variant_tuple => {
                form = .tuple;
                payload_nodes = Ast.rangeSlice(t.tree, vnode.lhs);
            },
            .enum_variant_struct => {
                form = .@"struct";
                is_struct_form = true;
                payload_nodes = Ast.rangeSlice(t.tree, vnode.lhs);
            },
            else => {},
        }
        const np = payload_nodes.len;
        const fnames = try t.gpa.alloc([]const u8, if (is_struct_form) np else 0);
        errdefer t.gpa.free(fnames);
        const ftypes = try t.gpa.alloc(Type, np);
        errdefer t.gpa.free(ftypes);
        const foffs = try t.gpa.alloc(u32, np);
        errdefer t.gpa.free(foffs);

        var running: u32 = 0;
        var palign: u32 = 1;
        for (payload_nodes, 0..) |pnode_idx, pi| {
            // A tuple payload node is a type-ref; a struct payload node is a `param`
            // (name + type-ref in lhs).
            const pty: Type = if (is_struct_form) blk: {
                const pnode = t.tree.nodes[pnode_idx];
                fnames[pi] = t.nameText(pnode.main_token);
                break :blk t.typeFromNode(pnode.lhs);
            } else t.typeFromNode(pnode_idx);
            ftypes[pi] = pty;
            var psize: u32 = 0;
            var pa: u32 = 1;
            if (pty.kind == .unit) {
                try t.emitFmt(t.byteOf(vnode.main_token), "variant '{s}' payload cannot have type ()", .{vname});
                poisoned = true;
            } else if (pty.kind != .invalid) {
                const sz = try t.layoutReferent(pty, t.byteOf(decl.main_token), t.enums.items[id].name, &poisoned);
                psize = sz.size;
                pa = sz.@"align";
            }
            const off = roundUp(running, pa);
            foffs[pi] = off;
            running = off + psize;
            if (pa > palign) palign = pa;
        }
        const payload_size = roundUp(running, palign);
        variants[vi] = .{
            .name = vname,
            .form = form,
            .field_names = fnames,
            .field_types = ftypes,
            .offsets = foffs,
            .payload_size = payload_size,
            .payload_align = palign,
        };
        vbuilt += 1;
        if (payload_size > max_payload_size) max_payload_size = payload_size;
        if (palign > max_payload_align) max_payload_align = palign;
    }

    const tag_size: u32 = 8;
    const payload_off = roundUp(tag_size, max_payload_align);
    const aln = @max(@as(u32, 8), max_payload_align);
    t.enums.items[id].variants = variants;
    t.enums.items[id].tag_size = tag_size;
    t.enums.items[id].payload_off = payload_off;
    t.enums.items[id].@"align" = aln;
    t.enums.items[id].size = if (poisoned) 0 else roundUp(payload_off + max_payload_size, aln);
    t.enums.items[id].poisoned = poisoned;
    t.enums.items[id].state = .done;
}

/// Human-readable name of a type (a struct's declared name, else its kind tag).
fn typeName(t: *const Typecheck, ty: Type) []const u8 {
    if (ty.kind == .@"struct" and ty.struct_id < t.structs.items.len)
        return t.structs.items[ty.struct_id].name;
    if (ty.kind == .@"enum" and ty.enum_id < t.enums.items.len)
        return t.enums.items[ty.enum_id].name;
    return @tagName(ty.kind);
}

/// The type at a local slot, or `invalid` if out of range (shouldn't happen).
fn slotType(t: *const Typecheck, slot: u32) Type {
    return if (slot < t.slot_types.items.len) t.slot_types.items[slot] else .invalid;
}

/// Record the type of a local slot, growing the table to fit (slots are dense
/// and monotonically increasing within a function).
fn setSlot(t: *Typecheck, slot: u32, ty: Type) !void {
    while (t.slot_types.items.len <= slot) try t.slot_types.append(t.gpa, .invalid);
    t.slot_types.items[slot] = ty;
}

fn nameText(t: *const Typecheck, tok: u32) []const u8 {
    return t.tokens[tok].text(t.source);
}

fn byteOf(t: *const Typecheck, tok: u32) u32 {
    return t.tokens[tok].start;
}

/// Record a static-literal diagnostic.
fn emit(t: *Typecheck, byte_offset: u32, message: []const u8) !void {
    try t.diags.append(t.gpa, .{ .byte_offset = byte_offset, .message = message });
    if (t.graph != null) try t.diag_mods.append(t.gpa, t.graph_mod);
}

/// Format a data-bearing message, own the buffer, and record a diagnostic.
fn emitFmt(t: *Typecheck, byte_offset: u32, comptime fmt: []const u8, args: anytype) !void {
    const msg = try std.fmt.allocPrint(t.gpa, fmt, args);
    try t.owned_msgs.append(t.gpa, msg);
    try t.diags.append(t.gpa, .{ .byte_offset = byte_offset, .message = msg });
    if (t.graph != null) try t.diag_mods.append(t.gpa, t.graph_mod);
}

const testing = std.testing;
const Lexer = @import("lex.zig");
const Parser = @import("parse.zig");

const Checked = struct {
    tokens: []Token,
    tree: Ast.Tree,
    resolve: Resolve.Result,
    result: Result,
    source: []const u8,

    fn deinit(self: *Checked, gpa: std.mem.Allocator) void {
        self.result.deinit(gpa);
        self.resolve.deinit(gpa);
        gpa.free(self.tokens);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
    }
};

fn checkSource(source: []const u8) !Checked {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    var diag: ?Parser.Diagnostic = null;
    const tree = (try Parser.parse(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
    errdefer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    var res = try Resolve.resolve(gpa, tree, tokens, source);
    errdefer res.deinit(gpa);
    const result = try check(gpa, tree, tokens, source, res.resolutions);
    return .{ .tokens = tokens, .tree = tree, .resolve = res, .result = result, .source = source };
}

/// Typecheck a source and return the diagnostic count (and free everything).
fn checkDiagCount(source: []const u8) !usize {
    const gpa = testing.allocator;
    var c = try checkSource(source);
    defer c.deinit(gpa);
    return c.result.diags.len;
}

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
        if (n.tag == .var_decl) try testing.expectEqual(Type.bool, c.result.node_types[i]);
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
        if (n.tag == .var_decl) try testing.expectEqual(Type.bool, c.result.node_types[i]);
    }
}

test "string literal types as str" {
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n s := \"hi\"\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The var_decl's bound type is str.
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .var_decl) try testing.expectEqual(Type.str, c.result.node_types[i]);
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
            const callee_res = c.resolve.resolutions[n.lhs];
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
            try testing.expectEqual(Kind.@"struct", c.result.node_types[i].kind);
            try testing.expectEqual(@as(u32, 0), c.result.node_types[i].struct_id);
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

// enums + match (M10)

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

test "M11 int match with a wildcard compiles" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(x: int) -> int { match x { 0 -> 1, _ -> 0 } }\nfn main() -> int { return f(1) }\n",
    ));
}

test "M11 int match without a wildcard is non-exhaustive" {
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
            if (c.result.node_types[i].kind == .@"enum") {
                try testing.expectEqual(@as(u32, 0), c.result.node_types[i].enum_id);
                found = true;
            }
        }
    }
    try testing.expect(found);
}
