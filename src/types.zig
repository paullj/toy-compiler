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
const refs = struct {
    const err_unknown_type = "unknown type '{s}'";
    const err_unknown_module = "unknown module '{s}'";
    const err_module_no_type = "module has no type '{s}'";

    fn nameText(self: anytype, tok: u32) []const u8 {
        return self.tokens[tok].text(self.source);
    }

    fn byteOf(self: anytype, tok: u32) u32 {
        return self.tokens[tok].start;
    }

    /// Human-readable name of a type (a struct/enum's declared name, else its kind tag).
    fn typeName(self: anytype, ty: Type) []const u8 {
        if (ty.kind == .@"struct" and ty.struct_id < self.structSyms().len)
            return self.structSyms()[ty.struct_id].name;
        if (ty.kind == .@"enum" and ty.enum_id < self.enumSyms().len)
            return self.enumSyms()[ty.enum_id].name;
        return @tagName(ty.kind);
    }

    /// Map a type-reference node (an `identifier`, a `literal_unit` for `()`, or in
    /// graph mode a qualified `mod.Type` `field_access`) to a `Type`.
    fn typeFromNode(self: anytype, type_node: Ast.Index) Type {
        if (type_node == Ast.none) return Type.unit;
        const tn = self.tree.nodes[type_node];
        if (tn.tag == .literal_unit) return Type.unit; // explicit `-> ()` / `p: ()`
        // A qualified cross-module type-ref `mod.Type` parses as a field_access whose
        // receiver binds to a `.module`. Resolve it against the owning module's tables.
        if (tn.tag == .field_access) return refs.typeFromQualified(self, type_node, tn);
        const tok = tn.main_token;
        const name = refs.nameText(self, tok);
        if (type_names.get(name)) |b| return b;
        if (self.activeStructMap().get(name)) |id| return Type.structT(id);
        if (self.activeEnumMap().get(name)) |id| return Type.enumT(id);
        self.sink.emitFmt(refs.byteOf(self, tok), err_unknown_type, .{name}) catch {};
        return .invalid;
    }

    /// Resolve a qualified `mod.Type` type-reference (a `field_access` in type
    /// position; the receiver binds to a `.module`) to the owning module's GLOBAL
    /// struct/enum id. Graph mode only. Visibility was already enforced by
    /// `resolve_graph`, so a private type here is a defensive `invalid`.
    fn typeFromQualified(self: anytype, node_idx: Ast.Index, n: Ast.Node) Type {
        _ = node_idx;
        const g = self.graphCtx();
        const recv = self.tree.nodes[n.lhs];
        if (recv.tag != .identifier) return .invalid;
        const recv_name = refs.nameText(self, recv.main_token);
        const target = g.namespaceOfIn(self.graph_mod, recv_name) orelse {
            self.sink.emitFmt(refs.byteOf(self, recv.main_token), err_unknown_module, .{recv_name}) catch {};
            return .invalid;
        };
        const member = refs.nameText(self, n.main_token);
        if (g.mods[target].struct_ids.get(member)) |id| return Type.structT(id);
        if (g.mods[target].enum_ids.get(member)) |id| return Type.enumT(id);
        self.sink.emitFmt(refs.byteOf(self, n.main_token), err_module_no_type, .{member}) catch {};
        return .invalid;
    }
};

/// A reported problem. Same shape as `Parser`/`Resolve` diagnostics so the
/// driver/CLI can render either uniformly (byte offset → line:col).
pub const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;
const DiagnosticSink = @import("diagnostics/Sink.zig");

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
const FnSym = struct {
    decl_node: Ast.Index,
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

/// The struct table: one `StructSym` per struct id, plus a name→id map.
structs: std.ArrayList(StructSym),
struct_map: std.StringHashMapUnmanaged(u32),

/// The enum table: one `EnumSym` per enum id, plus a name→id map.
enums: std.ArrayList(EnumSym),
enum_map: std.StringHashMapUnmanaged(u32),

/// M14 graph context. Always set in practice: `checkGraph` is the ONE entry and
/// it drives one shared `Typecheck` across the whole module graph (a lone source
/// file is the trivial one-module graph). The `structs`/`enums`/`fns` tables are
/// PROGRAM-WIDE (global ids), and `struct_map`/`enum_map` hold the CURRENT
/// module's bare-name → global-id bindings (swapped per module). The context
/// resolves a qualified `mod.Type` / `mod.Enum` receiver to the owning module's
/// tables. Pre-collect + layout happen once; only Pass B runs per fn.
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

/// S4 — the runtime the per-fn body region (Pass C) dispatches onto. Set in graph
/// mode (carries the `-j N` worker cap from the driver's own pool). The body region
/// fans out per-fn body checks via `Engine.fanOut(io, ...)` whenever a pool is present
/// (`io != null`). `.limited(0)` (`-j1`) drives every unit onto the inline serial path
/// — the byte-identity baseline. `io == null` (single-file internal callers + inline
/// tests) is the only serial trigger; both dispatch modes feed the SAME slots +
/// merge+sort, so the result is byte-identical regardless.
io: ?Io = null,

/// The `-j` jobs knob: the chunk-count basis for the per-fn Pass-C body fan-out
/// (`checkBodies`), 0 => host cpu count. PERF P1 — body checks dispatch ~`ncpu`
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
    fn namespaceOfIn(c: *const GraphCtx, mod: u32, recv_name: []const u8) ?u32 {
        return c.mods[mod].namespaces.get(recv_name);
    }
};

/// The immutable, whole-program model frozen after Pass A: the fn signature
/// table + the laid-out struct/enum tables + their bare-name maps + the graph
/// context. Pass C reads it READ-ONLY through every `BodyChecker`, so per-fn body
/// checking can run against one shared frozen snapshot (the enabler for S4's
/// parallel fan-out). The slices alias the still-live `Typecheck` ArrayLists,
/// which are not mutated during Pass C. `graph`/`gph_fn_names` are borrowed.
const Model = struct {
    fns: []const FnSym,
    structs: []const StructSym,
    enums: []const EnumSym,
    /// Single-file bare-name maps (graph mode reads the per-module maps in `graph`
    /// via `graph_mod`; these stay the empty init maps then). Borrowed pointers.
    struct_map: *const std.StringHashMapUnmanaged(u32),
    enum_map: *const std.StringHashMapUnmanaged(u32),
    graph: *GraphCtx,
    gph_fn_names: ?[]const []const u8,
};

/// Per-function checking context over a FROZEN `*const Model`. Holds the per-fn
/// scratch (slot_types/cur_ret/loop_stack/expected) and the cursor (tree/tokens/
/// source/resolutions/node_types/graph_mod) — all set once at construction from
/// the fn's owning module, never swapped (gphSelect's save/restore is gone on this
/// path). Diagnostics go to a LOCAL `sink` (scoped to the fn's module); the per-fn
/// body region (`checkBodies`) merges it into the shared sink in fn-id order AFTER
/// the body walk, so parallel Pass C never touches shared mutable diag state.
/// `node_types` aliases the program-wide array but each BodyChecker writes ONLY its
/// own fn's node span (disjoint by construction).
const BodyChecker = struct {
    model: *const Model,
    gpa: std.mem.Allocator,

    // Cursor — set at construction from the fn's owning module; never swapped.
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const Resolution,
    node_types: []Type,
    graph_mod: u32,

    // Per-fn scratch.
    slot_types: std.ArrayList(Type) = .empty,
    cur_ret: Type = .unit,
    loop_stack: std.ArrayList(LoopCtx) = .empty,
    expected: ?Type = null,

    // Local diagnostic sink (merged into the shared result by the Pass-C driver).
    // Its scope is set once in `bodyCheckerFor` to the fn's owning module (graph)
    // or left NO_SCOPE (single-file).
    sink: DiagnosticSink,

    gph_fn_names: ?[]const []const u8 = null,

    fn deinit(bc: *BodyChecker) void {
        bc.slot_types.deinit(bc.gpa);
        bc.loop_stack.deinit(bc.gpa);
        bc.sink.deinit();
    }

    // ---- BodyChecker methods (the per-fn body-walk relations) --------------

    /// Walk this fn's body: rebuild the slot table, type the body block against the
    /// declared return type, and enforce definite-return. The cursor + scratch live
    /// on `bc`; the immutable fn/struct/enum tables are read through `bc.model`.
    fn checkBody(bc: *BodyChecker, fid: u32, f: FnSym) !void {
        _ = fid;
        const decl = bc.tree.nodes[f.decl_node];
        const proto = Ast.protoAt(bc.tree, decl.lhs);

        // Rebuild the per-function slot→type table. Parameters get slots 0..N first
        // (the resolver declares them first), then `:=` locals as we encounter them.
        bc.slot_types.clearRetainingCapacity();
        for (f.params) |pty| try bc.slot_types.append(bc.gpa, pty);
        bc.cur_ret = f.ret;

        const want_value = (f.ret.kind != .unit and f.ret.kind != .invalid);
        const body_ty = try bc.typeOfBlockExpected(decl.rhs, want_value, if (want_value) f.ret else null);

        // A non-unit function must produce a value on every path: either all paths
        // structurally return, OR the body's trailing expression has the declared
        // type. `assignable` folds poison/never/eql exactly as before.
        if (want_value and !bc.blockReturns(decl.rhs) and !Type.assignable(f.ret, body_ty)) {
            try bc.sink.emitFmt(bc.byteOf(decl.main_token), "function '{s}' must return {s} but may fall off the end", .{ bc.nameText(decl.main_token), bc.typeName(f.ret) });
        }
        _ = proto;
    }

    fn blockReturns(bc: *const BodyChecker, block_idx: Ast.Index) bool {
        const stmts = Ast.rangeSlice(bc.tree, bc.tree.nodes[block_idx].lhs);
        if (stmts.len == 0) return false;
        return bc.stmtReturns(stmts[stmts.len - 1]);
    }

    fn stmtReturns(bc: *const BodyChecker, stmt_idx: Ast.Index) bool {
        const stmt = bc.tree.nodes[stmt_idx];
        return switch (stmt.tag) {
            .return_stmt => true,
            // A statement-position block/if is parsed wrapped in an `expr_stmt`; unwrap
            // it so a trailing diverging bare block (`{ return 5 }`) or parenthesized
            // value-if satisfies definite-return and a divergent arm merges correctly.
            .expr_stmt => bc.stmtReturns(stmt.lhs),
            .block => bc.blockReturns(stmt_idx),
            .if_stmt => blk: {
                const h = Ast.ifHeaderAt(bc.tree, stmt.rhs);
                // An else-less `if` can be skipped, so it never guarantees a return.
                if (h.else_node == Ast.none) break :blk false;
                const then_ok = bc.blockReturns(h.then_block);
                const else_ok = if (bc.tree.nodes[h.else_node].tag == .if_stmt)
                    bc.stmtReturns(h.else_node)
                else
                    bc.blockReturns(h.else_node);
                break :blk then_ok and else_ok;
            },
            // Conservative: a `while` may never execute, so it can't guarantee a
            // return (`while true { return }` is rejected — acceptable for now).
            .while_stmt => false,
            // A `loop` diverges (returns/never-falls-through) iff it has NO `break`
            // targeting it: the only ways out are `return` or an outer construct.
            .loop_expr => bc.loopDiverges(stmt_idx),
            // A labeled wrapper is transparent for definite-return: it returns iff its
            // inner construct does (a labeled bare block via its trailing stmt).
            .labeled => bc.stmtReturns(bc.tree.nodes[stmt_idx].lhs),
            .for_stmt, .break_stmt, .continue_stmt => false,
            else => false,
        };
    }

    fn loopDiverges(bc: *const BodyChecker, loop_idx: Ast.Index) bool {
        return !bc.blockHasBreak(bc.tree.nodes[loop_idx].lhs, loop_idx);
    }

    fn blockHasBreak(bc: *const BodyChecker, block_idx: Ast.Index, target: Ast.Index) bool {
        for (Ast.rangeSlice(bc.tree, bc.tree.nodes[block_idx].lhs)) |s| {
            if (bc.stmtHasBreak(s, target)) return true;
        }
        return false;
    }

    fn stmtHasBreak(bc: *const BodyChecker, stmt_idx: Ast.Index, target: Ast.Index) bool {
        const stmt = bc.tree.nodes[stmt_idx];
        return switch (stmt.tag) {
            // A bare break (no label) binds to the innermost loop — counts only when
            // `target` IS the innermost loop, i.e. the bare break is found before any
            // nested loop swallows it (the nested-loop arms below stop the descent for
            // bare breaks). A labeled break counts iff its resolved target matches.
            .break_stmt => if (bc.resolutions[stmt_idx] == .label)
                bc.resolutions[stmt_idx].label == target
            else
                true,
            .expr_stmt => bc.stmtHasBreak(stmt.lhs, target),
            .block => bc.blockHasBreak(stmt_idx, target),
            // A labeled wrapper is transparent: descend its inner construct (a
            // `break @target` may live inside a nested labeled loop).
            .labeled => bc.stmtHasBreak(stmt.lhs, target),
            .if_stmt => blk: {
                const h = Ast.ifHeaderAt(bc.tree, stmt.rhs);
                if (bc.blockHasBreak(h.then_block, target)) break :blk true;
                if (h.else_node == Ast.none) break :blk false;
                break :blk if (bc.tree.nodes[h.else_node].tag == .if_stmt)
                    bc.stmtHasBreak(h.else_node, target)
                else
                    bc.blockHasBreak(h.else_node, target);
            },
            // A nested loop/for/while swallows BARE breaks, but a `break @target`
            // buried inside it still targets `target` — so descend its body and only
            // count labeled breaks that name `target`.
            .loop_expr => bc.nestedHasLabeledBreak(bc.tree.nodes[stmt_idx].lhs, target),
            .while_stmt => bc.nestedHasLabeledBreak(bc.tree.nodes[stmt_idx].rhs, target),
            .for_stmt => bc.nestedHasLabeledBreak(bc.tree.nodes[stmt_idx].lhs, target),
            else => false,
        };
    }

    fn nestedHasLabeledBreak(bc: *const BodyChecker, block_idx: Ast.Index, target: Ast.Index) bool {
        for (Ast.rangeSlice(bc.tree, bc.tree.nodes[block_idx].lhs)) |s| {
            if (bc.stmtHasLabeledBreak(s, target)) return true;
        }
        return false;
    }

    fn stmtHasLabeledBreak(bc: *const BodyChecker, stmt_idx: Ast.Index, target: Ast.Index) bool {
        const stmt = bc.tree.nodes[stmt_idx];
        return switch (stmt.tag) {
            .break_stmt => bc.resolutions[stmt_idx] == .label and bc.resolutions[stmt_idx].label == target,
            .expr_stmt => bc.stmtHasLabeledBreak(stmt.lhs, target),
            .block => bc.nestedHasLabeledBreak(stmt_idx, target),
            .labeled => bc.stmtHasLabeledBreak(stmt.lhs, target),
            .if_stmt => blk: {
                const h = Ast.ifHeaderAt(bc.tree, stmt.rhs);
                if (bc.nestedHasLabeledBreak(h.then_block, target)) break :blk true;
                if (h.else_node == Ast.none) break :blk false;
                break :blk if (bc.tree.nodes[h.else_node].tag == .if_stmt)
                    bc.stmtHasLabeledBreak(h.else_node, target)
                else
                    bc.nestedHasLabeledBreak(h.else_node, target);
            },
            .loop_expr => bc.nestedHasLabeledBreak(bc.tree.nodes[stmt_idx].lhs, target),
            .while_stmt => bc.nestedHasLabeledBreak(bc.tree.nodes[stmt_idx].rhs, target),
            .for_stmt => bc.nestedHasLabeledBreak(bc.tree.nodes[stmt_idx].lhs, target),
            else => false,
        };
    }

    fn blockDiverges(bc: *const BodyChecker, block_idx: Ast.Index) bool {
        const stmts = Ast.rangeSlice(bc.tree, bc.tree.nodes[block_idx].lhs);
        if (stmts.len == 0) return false;
        return bc.stmtDiverges(stmts[stmts.len - 1]);
    }

    fn stmtDiverges(bc: *const BodyChecker, stmt_idx: Ast.Index) bool {
        const stmt = bc.tree.nodes[stmt_idx];
        return switch (stmt.tag) {
            .return_stmt, .break_stmt, .continue_stmt => true,
            .expr_stmt => bc.stmtDiverges(stmt.lhs),
            .block => bc.blockDiverges(stmt_idx),
            .if_stmt => blk: {
                const h = Ast.ifHeaderAt(bc.tree, stmt.rhs);
                if (h.else_node == Ast.none) break :blk false;
                const then_ok = bc.blockDiverges(h.then_block);
                const else_ok = if (bc.tree.nodes[h.else_node].tag == .if_stmt)
                    bc.stmtDiverges(h.else_node)
                else
                    bc.blockDiverges(h.else_node);
                break :blk then_ok and else_ok;
            },
            .while_stmt => false,
            .loop_expr => bc.loopDiverges(stmt_idx),
            .labeled => bc.labeledDiverges(stmt_idx),
            .for_stmt => false,
            // A `match` diverges iff it is exhaustive AND every arm body diverges.
            .match_expr => bc.matchDiverges(stmt_idx),
            else => false,
        };
    }

    fn matchDiverges(bc: *const BodyChecker, node_idx: Ast.Index) bool {
        const n = bc.tree.nodes[node_idx];
        const st = bc.node_types[n.lhs];
        // Conservative for int/bool scrutinees: `return false` (loses only a
        // definite-return optimization, never miscompiles). Only enums get the
        // variant-coverage analysis here.
        if (!st.isEnum()) return false;
        const arms = Ast.rangeSlice(bc.tree, n.rhs);
        if (arms.len == 0) return false;
        var has_wildcard = false;
        const e = bc.model.enums[st.enum_id];
        var seen = [_]bool{false} ** 64; // enum variant count is small
        for (arms) |arm_idx| {
            const arm = bc.tree.nodes[arm_idx];
            const h = Ast.armHeaderAt(bc.tree, arm.rhs);
            if (!bc.stmtDiverges(h.body)) return false;
            if (h.guard != Ast.none) continue; // a guard can fail → no coverage
            const pat = bc.tree.nodes[arm.lhs];
            if (pat.tag == .pattern_wildcard) {
                has_wildcard = true;
            } else if (pat.tag == .pattern_variant) {
                const vname = bc.nameText(pat.main_token);
                for (e.variants, 0..) |v, i| {
                    if (i < seen.len and std.mem.eql(u8, v.name, vname) and bc.variantPayloadIrrefutable(arm.lhs, v)) seen[i] = true;
                }
            }
        }
        if (has_wildcard) return true;
        if (e.variants.len > seen.len) return false;
        for (e.variants, 0..) |_, i| if (!seen[i]) return false;
        return true;
    }

    fn labeledDiverges(bc: *const BodyChecker, idx: Ast.Index) bool {
        const inner_idx = bc.tree.nodes[idx].lhs;
        const inner = bc.tree.nodes[inner_idx];
        return switch (inner.tag) {
            .loop_expr => bc.loopDiverges(inner_idx),
            .while_stmt, .for_stmt => false,
            .block => bc.blockDiverges(inner_idx) and !bc.blockHasBreak(inner_idx, inner_idx),
            else => false,
        };
    }

    fn typeOfBlockExpected(bc: *BodyChecker, block_idx: Ast.Index, want_value: bool, exp: ?Type) error{OutOfMemory}!Type {
        const save = bc.expected;
        bc.expected = exp;
        defer bc.expected = save;
        return bc.checkBlock(block_idx, want_value);
    }

    fn checkBlock(bc: *BodyChecker, block_idx: Ast.Index, want_value: bool) error{OutOfMemory}!Type {
        const stmts = Ast.rangeSlice(bc.tree, bc.tree.nodes[block_idx].lhs);
        if (stmts.len == 0) {
            bc.node_types[block_idx] = .unit;
            return .unit;
        }
        // The expected type (a one-shot for an inferred `.V`) applies ONLY to this
        // block's trailing value expression, never the non-final statements. Capture
        // it, clear it for the non-final walk, then restore for the trailing expr.
        const block_expected = bc.expected;
        bc.expected = null;
        for (stmts[0 .. stmts.len - 1]) |s| try bc.checkStmt(s); // effect only
        const last = stmts[stmts.len - 1];
        const last_n = bc.tree.nodes[last];
        var bt: Type = .unit;
        if (last_n.tag == .expr_stmt) {
            bt = try bc.typeOfExpected(last_n.lhs, if (want_value) block_expected else null);
            bc.node_types[last] = bt; // the expr_stmt carries the value type
        } else if (want_value and (last_n.tag == .if_stmt or last_n.tag == .block)) {
            bt = try bc.typeOfExpected(last, block_expected); // value context: validates + types + memoizes
        } else {
            try bc.checkStmt(last); // statement context (incl. trailing else-less if)
        }
        bc.node_types[block_idx] = bt;
        return bt;
    }

    fn checkStmt(bc: *BodyChecker, stmt_idx: Ast.Index) error{OutOfMemory}!void {
        const stmt = bc.tree.nodes[stmt_idx];
        switch (stmt.tag) {
            .var_decl => {
                // `x: T = e` (rhs is the type ref) binds x:T and checks the
                // initializer against it; `x := e` (rhs none) infers x from e.
                const ty: Type = if (stmt.rhs != Ast.none) blk: {
                    const declared = bc.typeFromNode(stmt.rhs);
                    const got = try bc.typeOfExpected(stmt.lhs, declared);
                    if (!Type.assignable(declared, got)) {
                        try bc.sink.emitFmt(bc.byteOf(stmt.main_token), "cannot bind {s} to '{s}' of type {s}", .{ bc.typeName(got), bc.nameText(stmt.main_token), bc.typeName(declared) });
                    }
                    break :blk declared;
                } else try bc.typeOf(stmt.lhs);
                bc.node_types[stmt_idx] = ty;
                // Record the new local's type at its slot. The resolver bound the
                // var_decl node to a `.local` slot; append/extend slot_types to fit.
                if (bc.resolutions[stmt_idx] == .local) {
                    const slot = bc.resolutions[stmt_idx].local;
                    try bc.setSlot(slot, ty);
                }
                if (ty.kind == .unit) {
                    try bc.sink.emitFmt(bc.byteOf(stmt.main_token), "cannot bind () to '{s}'", .{bc.nameText(stmt.main_token)});
                }
            },
            .assign => {
                const target = bc.tree.nodes[stmt.lhs];
                // Compute the place type FIRST so it can flow into the rhs as the
                // expected type (an inferred `.V` assigned to a known-typed place).
                const lhs: Type = switch (target.tag) {
                    .identifier => if (bc.resolutions[stmt.lhs] == .local)
                        bc.slotType(bc.resolutions[stmt.lhs].local)
                    else
                        .invalid,
                    .field_access => try bc.typeOf(stmt.lhs),
                    else => .invalid,
                };
                bc.node_types[stmt.lhs] = lhs;
                const rhs = try bc.typeOfExpected(stmt.rhs, if (lhs.kind == .invalid) null else lhs);
                if (!Type.assignable(lhs, rhs)) {
                    try bc.sink.emitFmt(bc.byteOf(target.main_token), "cannot assign {s} to variable of type {s}", .{ bc.typeName(rhs), bc.typeName(lhs) });
                }
            },
            .return_stmt => {
                const ty: Type = if (stmt.lhs == Ast.none) Type.unit else try bc.typeOfExpected(stmt.lhs, if (bc.cur_ret.kind == .invalid) null else bc.cur_ret);
                // Skip when either side is poison so an unknown return type (already
                // reported) doesn't trigger a spurious second diagnostic here. A
                // `never`-typed operand (e.g. `return x` where x binds a break-less
                // loop) is also fine: control never actually reaches the return, so
                // `never` unifies with any declared type — mirrors the trailing-expr
                // body check above.
                if (!Type.assignable(bc.cur_ret, ty)) {
                    try bc.sink.emitFmt(bc.byteOf(stmt.main_token), "return type {s} does not match declared {s}", .{ bc.typeName(ty), bc.typeName(bc.cur_ret) });
                }
            },
            .expr_stmt => _ = try bc.typeOf(stmt.lhs),
            .block => _ = try bc.checkBlock(stmt_idx, false),
            .if_stmt => {
                const ct = try bc.typeOf(stmt.lhs);
                if (ct.kind != .invalid and ct.kind != .bool)
                    try bc.sink.emit(bc.byteOf(bc.tree.nodes[stmt.lhs].main_token), "if condition must be bool");
                const h = Ast.ifHeaderAt(bc.tree, stmt.rhs);
                _ = try bc.checkBlock(h.then_block, false);
                if (h.else_node != Ast.none) {
                    if (bc.tree.nodes[h.else_node].tag == .if_stmt)
                        try bc.checkStmt(h.else_node)
                    else
                        _ = try bc.checkBlock(h.else_node, false);
                }
            },
            .while_stmt => try bc.checkWhile(stmt_idx, null),
            .for_stmt => try bc.checkFor(stmt_idx, null),
            .labeled => _ = try bc.checkLabeled(stmt_idx, false),
            .break_stmt => {
                const ctx = bc.targetCtx(stmt_idx) orelse {
                    // No matching context: a bare break with an empty stack ("outside a
                    // loop"); a labeled break is reported by resolve as undefined.
                    if (bc.resolutions[stmt_idx] != .label)
                        try bc.sink.emit(bc.byteOf(stmt.main_token), "break outside of a loop");
                    if (stmt.lhs != Ast.none) _ = try bc.typeOf(stmt.lhs);
                    return;
                };
                if (stmt.lhs == Ast.none) {
                    ctx.saw_bare_break = true;
                    if (ctx.is_value) ctx.join = try bc.merge(stmt.main_token, ctx.join, Type.unit);
                } else {
                    const vt = try bc.typeOf(stmt.lhs);
                    if (!ctx.is_value) {
                        if (vt.kind != .invalid and vt.kind != .unit)
                            try bc.sink.emit(bc.byteOf(stmt.main_token), "cannot break with a value out of a while/for loop");
                    } else {
                        ctx.saw_value_break = true;
                        ctx.join = try bc.merge(stmt.main_token, ctx.join, vt);
                    }
                }
            },
            .continue_stmt => {
                const ctx = bc.targetCtx(stmt_idx) orelse {
                    if (bc.resolutions[stmt_idx] != .label)
                        try bc.sink.emit(bc.byteOf(stmt.main_token), "continue outside of a loop");
                    return;
                };
                // `continue` is meaningful only on a loop; a labeled bare block is not.
                if (ctx.kind == .labeled_block)
                    try bc.sink.emit(bc.byteOf(stmt.main_token), "cannot continue a labeled block (not a loop)");
            },
            else => _ = try bc.typeOf(stmt_idx),
        }
    }

    fn targetCtx(bc: *BodyChecker, stmt_idx: Ast.Index) ?*LoopCtx {
        const items = bc.loop_stack.items;
        if (bc.resolutions[stmt_idx] == .label) {
            const target = bc.resolutions[stmt_idx].label;
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

    fn checkWhile(bc: *BodyChecker, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
        const stmt = bc.tree.nodes[stmt_idx];
        const ct = try bc.typeOf(stmt.lhs);
        if (ct.kind != .invalid and ct.kind != .bool)
            try bc.sink.emit(bc.byteOf(bc.tree.nodes[stmt.lhs].main_token), "while condition must be bool");
        try bc.loop_stack.append(bc.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
        _ = try bc.checkBlock(stmt.rhs, false);
        _ = bc.loop_stack.pop();
    }

    fn checkFor(bc: *BodyChecker, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
        const stmt = bc.tree.nodes[stmt_idx];
        const h = Ast.forHeaderAt(bc.tree, stmt.rhs);
        const lo = try bc.typeOf(h.lo);
        const hi = try bc.typeOf(h.hi);
        if (lo.kind != .invalid and lo.kind != .int)
            try bc.sink.emit(bc.byteOf(bc.tree.nodes[h.lo].main_token), "for range bounds must be int");
        if (hi.kind != .invalid and hi.kind != .int)
            try bc.sink.emit(bc.byteOf(bc.tree.nodes[h.hi].main_token), "for range bounds must be int");
        if (bc.resolutions[stmt_idx] == .local) try bc.setSlot(bc.resolutions[stmt_idx].local, Type.int);
        try bc.loop_stack.append(bc.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
        _ = try bc.checkBlock(stmt.lhs, false);
        _ = bc.loop_stack.pop();
    }

    fn checkLabeled(bc: *BodyChecker, idx: Ast.Index, want_value: bool) error{OutOfMemory}!Type {
        const n = bc.tree.nodes[idx];
        const label = bc.nameText(n.main_token);
        const inner = bc.tree.nodes[n.lhs];
        const ty: Type = switch (inner.tag) {
            .block => try bc.typeOfLabeledBlock(n.lhs, label, want_value),
            .loop_expr => try bc.typeOfLoop(n.lhs, inner, label),
            .while_stmt => blk: {
                try bc.checkWhile(n.lhs, label);
                break :blk Type.unit;
            },
            .for_stmt => blk: {
                try bc.checkFor(n.lhs, label);
                break :blk Type.unit;
            },
            else => Type.invalid,
        };
        bc.node_types[idx] = ty;
        return ty;
    }

    fn typeOfLabeledBlock(bc: *BodyChecker, block_idx: Ast.Index, label: []const u8, want_value: bool) error{OutOfMemory}!Type {
        try bc.loop_stack.append(bc.gpa, .{ .kind = .labeled_block, .label = label, .construct_node = block_idx, .is_value = true, .join = .never, .saw_value_break = false, .saw_bare_break = false });
        const fall = try bc.checkBlock(block_idx, want_value);
        const ctx = bc.loop_stack.pop().?;
        // The trailing-expr value is unreachable iff the block's last statement
        // diverges; in that case the block's value comes entirely from its breaks.
        const ft: Type = if (bc.blockDiverges(block_idx)) Type.never else fall;
        return bc.merge(bc.tree.nodes[block_idx].main_token, ft, ctx.join);
    }

    fn typeOfExpected(bc: *BodyChecker, node_idx: Ast.Index, exp: ?Type) error{OutOfMemory}!Type {
        const save = bc.expected;
        bc.expected = exp;
        defer bc.expected = save;
        return bc.typeOf(node_idx);
    }

    fn typeOf(bc: *BodyChecker, node_idx: Ast.Index) error{OutOfMemory}!Type {
        if (node_idx == Ast.none) return .invalid;
        const n = bc.tree.nodes[node_idx];
        const ty: Type = switch (n.tag) {
            .literal_number => Type.int,
            .literal_bool => Type.@"bool",
            .literal_string => Type.str,
            .identifier => switch (bc.resolutions[node_idx]) {
                .local => |slot| bc.slotType(slot),
                .func => blk: {
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "function '{s}' is not a value", .{bc.nameText(n.main_token)});
                    break :blk Type.invalid;
                },
                .unresolved => blk: {
                    // Resolve quietly skips a struct-named identifier (it expects this
                    // to be a struct type-name in a position it doesn't bind, or a
                    // positional `Point(...)` callee it diagnoses elsewhere). A BARE
                    // struct name used as a value (`q := P`, `P.x`) reaches here with
                    // no diagnostic — report it so it never escapes to codegen.
                    if (bc.activeStructMap().get(bc.nameText(n.main_token)) != null)
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "type '{s}' is not a value", .{bc.nameText(n.main_token)});
                    break :blk Type.invalid;
                },
                .label => Type.invalid, // never on an identifier node (break/continue only)
                .module => blk: {
                    // A bare imported-namespace name used as a value (`x := mod`):
                    // a module is not a value. (A `mod.member` access never reaches
                    // here — the receiver is consumed by typeOfFieldAccess/Call.)
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "module '{s}' is not a value", .{bc.nameText(n.main_token)});
                    break :blk Type.invalid;
                },
            },
            .unary => blk: {
                const operand = try bc.typeOf(n.lhs);
                if (operand.kind == .invalid) break :blk Type.invalid;
                const op = bc.tokens[n.main_token].tag;
                switch (op) {
                    .minus => {
                        if (operand.kind == .int) break :blk Type.int;
                        try bc.sink.emit(bc.byteOf(n.main_token), "operand of '-' must be int");
                    },
                    .bang => {
                        if (operand.kind == .bool) break :blk Type.@"bool";
                        try bc.sink.emit(bc.byteOf(n.main_token), "operand of '!' must be bool");
                    },
                    else => {},
                }
                break :blk Type.invalid;
            },
            .binary => blk: {
                const lt = try bc.typeOf(n.lhs);
                const rt = try bc.typeOf(n.rhs);
                if (lt.kind == .invalid or rt.kind == .invalid) break :blk Type.invalid;
                const op = bc.tokens[n.main_token].tag;
                const op_text = bc.tokens[n.main_token].text(bc.source);
                switch (op) {
                    .plus, .minus, .star, .slash => {
                        if (lt.kind == .int and rt.kind == .int) break :blk Type.int;
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must be int", .{op_text});
                    },
                    .lt, .lt_eq, .gt, .gt_eq => {
                        if (lt.kind == .int and rt.kind == .int) break :blk Type.@"bool";
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must be int", .{op_text});
                    },
                    .eq_eq, .bang_eq => {
                        if (Type.eql(lt, rt) and (lt.kind == .int or lt.kind == .bool)) break :blk Type.@"bool";
                        if (Type.eql(lt, rt) and lt.kind == .str) {
                            // Same type, but str comparison isn't supported — say so,
                            // rather than the misleading "must have the same type".
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "'{s}' on str is unsupported", .{op_text});
                        } else {
                            try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must have the same type", .{op_text});
                        }
                    },
                    .amp_amp, .pipe_pipe => {
                        if (lt.kind == .bool and rt.kind == .bool) break :blk Type.@"bool";
                        try bc.sink.emitFmt(bc.byteOf(n.main_token), "operands of '{s}' must be bool", .{op_text});
                    },
                    else => {},
                }
                break :blk Type.invalid;
            },
            .call => try bc.typeOfCall(node_idx, n),
            .struct_init => try bc.typeOfStructInit(node_idx, n),
            .field_access => try bc.typeOfFieldAccess(node_idx, n),
            .enum_init_unit, .enum_init_tuple, .enum_init_struct => try bc.typeOfEnumInit(node_idx, n),
            .match_expr => return bc.typeOfMatch(node_idx, n), // sets node_types itself
            .literal_unit => Type.unit,
            .block => try bc.checkBlock(node_idx, true),
            .if_stmt => try bc.typeOfIf(node_idx, n),
            .loop_expr => return bc.typeOfLoop(node_idx, n, null), // sets node_types itself
            .labeled => return bc.checkLabeled(node_idx, true), // sets node_types itself
            else => Type.invalid,
        };
        bc.node_types[node_idx] = ty;
        return ty;
    }

    fn typeOfStructInit(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        // A qualified struct-variant `N.V { ... }` would arrive as enum_init_struct
        // (the parser upgrades a `field_access {`), not here — so struct_init's lhs is
        // always a plain type-name identifier; no enum routing needed.
        _ = node_idx;
        const name = bc.nameText(bc.tree.nodes[n.lhs].main_token);
        const id = bc.activeStructMap().get(name) orelse {
            for (Ast.rangeSlice(bc.tree, n.rhs)) |fi| _ = try bc.typeOf(bc.tree.nodes[fi].lhs);
            try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[n.lhs].main_token), "unknown struct type '{s}'", .{name});
            return .invalid;
        };
        const sym = bc.model.structs[id];
        const inits = Ast.rangeSlice(bc.tree, n.rhs);
    
        // Track which declared fields are supplied (for missing/duplicate checks).
        var seen = try bc.gpa.alloc(bool, sym.field_names.len);
        defer bc.gpa.free(seen);
        @memset(seen, false);
    
        for (inits) |fi_idx| {
            const fi = bc.tree.nodes[fi_idx];
            const fname = bc.nameText(fi.main_token);
            const vt = try bc.typeOf(fi.lhs);
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
                    try bc.sink.emitFmt(bc.byteOf(fi.main_token), "duplicate field '{s}' in '{s}'", .{ fname, name });
                }
                seen[j] = true;
                const fty = sym.field_types[j];
                if (!Type.assignable(fty, vt)) {
                    try bc.sink.emitFmt(bc.byteOf(fi.main_token), "field '{s}': expected {s}, got {s}", .{ fname, bc.typeName(fty), bc.typeName(vt) });
                }
            } else {
                try bc.sink.emitFmt(bc.byteOf(fi.main_token), "unknown field '{s}' in '{s}'", .{ fname, name });
            }
        }
        for (sym.field_names, 0..) |dn, j| {
            if (!seen[j]) try bc.sink.emitFmt(bc.byteOf(n.main_token), "missing field '{s}' in '{s}'", .{ dn, name });
        }
        return Type.structT(id);
    }

    fn qualifiedEnumId(bc: *BodyChecker, node_idx: Ast.Index) ?u32 {
        const g = bc.model.graph;
        const n = bc.tree.nodes[node_idx];
        if (n.tag != .field_access) return null;
        const recv = bc.tree.nodes[n.lhs];
        if (recv.tag != .identifier) return null;
        if (bc.resolutions[n.lhs] != .module) return null;
        const recv_name = bc.nameText(recv.main_token);
        const target = g.namespaceOfIn(bc.graph_mod, recv_name) orelse return null;
        const member = bc.nameText(n.main_token);
        return g.mods[target].enum_ids.get(member);
    }

    fn typeOfFieldAccess(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        // A qualified UNIT-variant reference `N.V`: the receiver is an enum type-name
        // identifier (resolved quietly to .unresolved). Treat it as construction.
        const recv = bc.tree.nodes[n.lhs];
        if (recv.tag == .identifier and bc.activeEnumMap().get(bc.nameText(recv.main_token)) != null) {
            return bc.typeOfEnumInitQualified(node_idx, .unit, n.lhs, n.main_token, Ast.none);
        }
        // A 3-level cross-module unit-variant `mod.Enum.Variant`: the receiver of THIS
        // field_access is the inner `mod.Enum` field_access (graph mode). Resolve the
        // inner to a global enum id and treat this node as a unit-variant construction.
        if (bc.qualifiedEnumId(n.lhs)) |enum_id| {
            const ty = try bc.checkVariant(enum_id, n.main_token, .unit, Ast.none);
            bc.node_types[node_idx] = ty;
            return ty;
        }
        const base = try bc.typeOf(n.lhs);
        if (base.kind == .invalid) return .invalid;
        if (!base.isStruct()) {
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "cannot access field '{s}' of non-struct type {s}", .{ bc.nameText(n.main_token), bc.typeName(base) });
            return .invalid;
        }
        const sym = bc.model.structs[base.struct_id];
        const fname = bc.nameText(n.main_token);
        for (sym.field_names, 0..) |dn, j| {
            if (std.mem.eql(u8, dn, fname)) return sym.field_types[j];
        }
        try bc.sink.emitFmt(bc.byteOf(n.main_token), "no field '{s}' in struct '{s}'", .{ fname, sym.name });
        return .invalid;
    }

    fn typeOfEnumInit(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
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
            const tname = bc.nameText(bc.tree.nodes[n.lhs].main_token);
            enum_id = bc.activeEnumMap().get(tname) orelse {
                try bc.typeArgsForEffect(node_form, args);
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[n.lhs].main_token), "'{s}' is not an enum type", .{tname});
                return .invalid;
            };
        } else {
            const exp = bc.expected orelse {
                try bc.typeArgsForEffect(node_form, args);
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "cannot infer the enum type for '.{s}' here", .{bc.nameText(n.main_token)});
                return .invalid;
            };
            if (!exp.isEnum()) {
                try bc.typeArgsForEffect(node_form, args);
                if (exp.kind != .invalid)
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "'.{s}' expects an enum type, but {s} was expected here", .{ bc.nameText(n.main_token), bc.typeName(exp) });
                return .invalid;
            }
            enum_id = exp.enum_id;
        }
        return bc.checkVariant(enum_id, n.main_token, node_form, args);
    }

    fn typeOfEnumInitQualified(bc: *BodyChecker, node_idx: Ast.Index, node_form: InitForm, type_node: Ast.Index, vtok: u32, args: Ast.Index) error{OutOfMemory}!Type {
        const tname = bc.nameText(bc.tree.nodes[type_node].main_token);
        const enum_id = bc.activeEnumMap().get(tname) orelse return .invalid; // caller checked
        const ty = try bc.checkVariant(enum_id, vtok, node_form, args);
        bc.node_types[node_idx] = ty;
        return ty;
    }

    fn typeArgsForEffect(bc: *BodyChecker, node_form: InitForm, args: Ast.Index) error{OutOfMemory}!void {
        if (args == Ast.none) return;
        if (node_form == .@"struct") {
            for (Ast.rangeSlice(bc.tree, args)) |fi| _ = try bc.typeOf(bc.tree.nodes[fi].lhs);
        } else {
            for (Ast.rangeSlice(bc.tree, args)) |a| _ = try bc.typeOf(a);
        }
    }

    fn checkVariant(bc: *BodyChecker, enum_id: u32, vtok: u32, node_form: InitForm, args: Ast.Index) error{OutOfMemory}!Type {
        const e = bc.model.enums[enum_id];
        const vname = bc.nameText(vtok);
        var vi: ?usize = null;
        for (e.variants, 0..) |v, i| {
            if (std.mem.eql(u8, v.name, vname)) {
                vi = i;
                break;
            }
        }
        const variant = if (vi) |i| e.variants[i] else {
            try bc.typeArgsForEffect(node_form, args);
            try bc.sink.emitFmt(bc.byteOf(vtok), "enum '{s}' has no variant '{s}'", .{ e.name, vname });
            return .invalid;
        };
        const want_form: InitForm = switch (variant.form) {
            .unit => .unit,
            .tuple => .tuple,
            .@"struct" => .@"struct",
        };
        if (node_form != want_form) {
            try bc.typeArgsForEffect(node_form, args);
            try bc.sink.emitFmt(bc.byteOf(vtok), "variant '{s}.{s}' is constructed with the wrong form", .{ e.name, vname });
            return Type.enumT(enum_id);
        }
        switch (variant.form) {
            .unit => {},
            .tuple => {
                const elems = if (args == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, args);
                if (elems.len != variant.field_types.len) {
                    for (elems) |a| _ = try bc.typeOf(a);
                    try bc.sink.emitFmt(bc.byteOf(vtok), "variant '{s}.{s}' expects {d} value(s), got {d}", .{ e.name, vname, variant.field_types.len, elems.len });
                    return Type.enumT(enum_id);
                }
                for (elems, variant.field_types) |a, fty| {
                    const at = try bc.typeOfExpected(a, fty);
                    if (!Type.assignable(fty, at))
                        try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[a].main_token), "variant '{s}.{s}': expected {s}, got {s}", .{ e.name, vname, bc.typeName(fty), bc.typeName(at) });
                }
            },
            .@"struct" => {
                const inits = if (args == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, args);
                var seen = try bc.gpa.alloc(bool, variant.field_names.len);
                defer bc.gpa.free(seen);
                @memset(seen, false);
                for (inits) |fi_idx| {
                    const fi = bc.tree.nodes[fi_idx];
                    const fname = bc.nameText(fi.main_token);
                    var found: ?usize = null;
                    for (variant.field_names, 0..) |dn, j| {
                        if (std.mem.eql(u8, dn, fname)) {
                            found = j;
                            break;
                        }
                    }
                    if (found) |j| {
                        if (seen[j]) try bc.sink.emitFmt(bc.byteOf(fi.main_token), "duplicate field '{s}' in '{s}.{s}'", .{ fname, e.name, vname });
                        seen[j] = true;
                        const fty = variant.field_types[j];
                        const vt = try bc.typeOfExpected(fi.lhs, fty);
                        if (!Type.assignable(fty, vt))
                            try bc.sink.emitFmt(bc.byteOf(fi.main_token), "field '{s}': expected {s}, got {s}", .{ fname, bc.typeName(fty), bc.typeName(vt) });
                    } else {
                        _ = try bc.typeOf(fi.lhs);
                        try bc.sink.emitFmt(bc.byteOf(fi.main_token), "unknown field '{s}' in '{s}.{s}'", .{ fname, e.name, vname });
                    }
                }
                for (variant.field_names, 0..) |dn, j| {
                    if (!seen[j]) try bc.sink.emitFmt(bc.byteOf(vtok), "missing field '{s}' in '{s}.{s}'", .{ dn, e.name, vname });
                }
            },
        }
        return Type.enumT(enum_id);
    }

    fn typeOfMatch(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        const st = try bc.typeOf(n.lhs);
        const arms = Ast.rangeSlice(bc.tree, n.rhs);
        if (st.kind == .invalid) {
            // Poison-absorb: still walk arms (bodies may have their own errors) but
            // don't emit a scrutinee or exhaustiveness error.
            for (arms) |arm_idx| _ = try bc.typeOf(Ast.armHeaderAt(bc.tree, bc.tree.nodes[arm_idx].rhs).body);
            bc.node_types[node_idx] = .invalid;
            return .invalid;
        }
        if (st.kind != .@"enum" and st.kind != .int and st.kind != .bool) {
            for (arms) |arm_idx| _ = try bc.typeOf(Ast.armHeaderAt(bc.tree, bc.tree.nodes[arm_idx].rhs).body);
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "match scrutinee must be an enum, int, or bool, got {s}", .{bc.typeName(st)});
            bc.node_types[node_idx] = .invalid;
            return .invalid;
        }

        var seen: []bool = &.{};
        var bool_cov: BoolCov = .{};
        var cov: Cov = switch (st.kind) {
            .@"enum" => blk: {
                seen = try bc.gpa.alloc(bool, bc.model.enums[st.enum_id].variants.len);
                @memset(seen, false);
                break :blk .{ .@"enum" = seen };
            },
            .bool => .{ .@"bool" = &bool_cov },
            else => .int,
        };
        defer if (st.kind == .@"enum") bc.gpa.free(seen);
    
        var has_wildcard = false;
        var result: Type = Type.never;
        for (arms) |arm_idx| {
            const arm = bc.tree.nodes[arm_idx];
            const h = Ast.armHeaderAt(bc.tree, arm.rhs);
            const guarded = h.guard != Ast.none;
            try bc.checkPattern(arm.lhs, st, &cov, &has_wildcard, !guarded);
            if (guarded) {
                const gt = try bc.typeOf(h.guard);
                if (gt.kind != .invalid and gt.kind != .bool)
                    try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[h.guard].main_token), "match guard must be bool, got {s}", .{bc.typeName(gt)});
            }
            const body_ty0 = try bc.typeOfExpected(h.body, bc.expected);
            const body_ty: Type = if (bc.armDiverges(h.body)) Type.never else body_ty0;
            result = try bc.merge(n.main_token, result, body_ty);
        }
        if (!has_wildcard) switch (cov) {
            .@"enum" => |sv| {
                const e = bc.model.enums[st.enum_id];
                for (e.variants, 0..) |v, i| {
                    if (!sv[i]) try bc.sink.emitFmt(bc.byteOf(n.main_token), "non-exhaustive match: missing variant '{s}'", .{v.name});
                }
            },
            .bool => |bcov| if (!(bcov.t and bcov.f))
                try bc.sink.emitFmt(bc.byteOf(n.main_token), "non-exhaustive match: bool requires both true and false (or '_')", .{}),
            .int => try bc.sink.emitFmt(bc.byteOf(n.main_token), "non-exhaustive match: int match requires '_'", .{}),
        };
        bc.node_types[node_idx] = result;
        return result;
    }

    fn irrefutable(bc: *const BodyChecker, pat_idx: Ast.Index, ty: Type) bool {
        if (ty.kind == .invalid) return true; // poison already reported; don't cascade a spurious miss
        const pat = bc.tree.nodes[pat_idx];
        return switch (pat.tag) {
            .pattern_wildcard => true,
            .pattern_binding => pat.rhs == Ast.none or bc.irrefutable(pat.rhs, ty),
            .pattern_literal => false,
            .pattern_variant => blk: {
                // Total only when the enum has exactly ONE variant (the tag test cannot
                // fail) AND that variant's payload is fully covered. Against a
                // multi-variant enum a single `.V` is refutable.
                if (ty.kind != .@"enum") break :blk false;
                const e = bc.model.enums[ty.enum_id];
                if (e.variants.len != 1) break :blk false;
                break :blk bc.variantPayloadIrrefutable(pat_idx, e.variants[0]);
            },
            .pattern_or => bc.orCoversType(pat_idx, ty),
            else => false,
        };
    }

    fn variantPayloadIrrefutable(bc: *const BodyChecker, pat_idx: Ast.Index, variant: VariantSym) bool {
        const pat = bc.tree.nodes[pat_idx];
        const binders = if (pat.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, pat.rhs);
        switch (variant.form) {
            .unit => return binders.len == 0,
            .tuple => {
                if (binders.len != variant.field_types.len) return false;
                for (binders, variant.field_types) |b, fty| if (!bc.irrefutable(b, fty)) return false;
                return true;
            },
            .@"struct" => {
                for (binders) |b_idx| {
                    const b = bc.tree.nodes[b_idx];
                    const src = if (b.lhs != Ast.none) bc.nameText(bc.tree.nodes[b.lhs].main_token) else bc.nameText(b.main_token);
                    var fty: Type = .invalid;
                    for (variant.field_names, 0..) |dn, j| if (std.mem.eql(u8, dn, src)) {
                        fty = variant.field_types[j];
                        break;
                    };
                    if (!bc.irrefutable(b_idx, fty)) return false;
                }
                return true;
            },
        }
    }

    fn orCoversType(bc: *const BodyChecker, or_idx: Ast.Index, ty: Type) bool {
        const alts = Ast.rangeSlice(bc.tree, bc.tree.nodes[or_idx].lhs);
        for (alts) |a| if (bc.irrefutable(a, ty)) return true;
        if (ty.kind == .@"enum") {
            const e = bc.model.enums[ty.enum_id];
            var seen = [_]bool{false} ** 64;
            if (e.variants.len > seen.len) return false;
            for (alts) |a| {
                const ap = bc.tree.nodes[a];
                if (ap.tag != .pattern_variant) continue;
                const vname = bc.nameText(ap.main_token);
                for (e.variants, 0..) |v, i| {
                    if (std.mem.eql(u8, v.name, vname) and bc.variantPayloadIrrefutable(a, v)) seen[i] = true;
                }
            }
            for (e.variants, 0..) |_, i| if (!seen[i]) return false;
            return true;
        }
        return false;
    }

    fn armDiverges(bc: *const BodyChecker, node_idx: Ast.Index) bool {
        return bc.stmtDiverges(node_idx);
    }

    fn checkPattern(bc: *BodyChecker, pat_idx: Ast.Index, expected: Type, cov: *Cov, has_wildcard: *bool, count_cov: bool) error{OutOfMemory}!void {
        const pat = bc.tree.nodes[pat_idx];
        switch (pat.tag) {
            .pattern_wildcard => if (count_cov) {
                has_wildcard.* = true;
            },
            .pattern_binding => {
                // Record the type this binding matched AGAINST on its own node. The
                // binding's slot is SHARED across or-pattern alternatives (Resolve), so
                // the slot type is overwritten and can't reveal a `.A(x) | .B(x)` type
                // divergence; the per-node matched type can (read by `collectBindings`).
                bc.node_types[pat_idx] = expected;
                if (pat.rhs == Ast.none) {
                    // Bind-whole: type by value; a top-level bare binding is irrefutable.
                    if (bc.resolutions[pat_idx] == .local) try bc.setSlot(bc.resolutions[pat_idx].local, expected);
                    if (count_cov) has_wildcard.* = true;
                } else {
                    try bc.checkPattern(pat.rhs, expected, cov, has_wildcard, count_cov);
                }
            },
            .pattern_literal => {
                const lt: Type = if (bc.tokens[pat.main_token].tag == .number) Type.int else Type.@"bool";
                if (expected.kind != .invalid and !Type.eql(lt, expected))
                    try bc.sink.emitFmt(bc.byteOf(pat.main_token), "literal pattern type {s} does not match scrutinee {s}", .{ bc.typeName(lt), bc.typeName(expected) });
                // A bool literal records its case toward coverage; int never covers.
                if (count_cov) switch (cov.*) {
                    .bool => |bcov| {
                        if (lt.kind == .bool) {
                            if (std.mem.eql(u8, bc.nameText(pat.main_token), "true")) bcov.t = true else bcov.f = true;
                        }
                    },
                    else => {},
                };
            },
            .pattern_or => {
                for (Ast.rangeSlice(bc.tree, pat.lhs)) |a| try bc.checkPattern(a, expected, cov, has_wildcard, count_cov);
                try bc.checkOrBindings(pat_idx);
            },
            .pattern_variant => try bc.checkVariantPattern(pat_idx, expected, cov, has_wildcard, count_cov),
            else => {},
        }
    }

    fn checkVariantPattern(bc: *BodyChecker, pat_idx: Ast.Index, expected: Type, cov: *Cov, has_wildcard: *bool, count_cov: bool) error{OutOfMemory}!void {
        const pat = bc.tree.nodes[pat_idx];
        if (expected.kind != .@"enum") {
            if (expected.kind != .invalid)
                try bc.sink.emitFmt(bc.byteOf(pat.main_token), "variant pattern on a non-enum scrutinee {s}", .{bc.typeName(expected)});
            return;
        }
        const enum_id = expected.enum_id;
        const e = bc.model.enums[enum_id];
        // A qualified `N.V` pattern: the type-name must name the scrutinee enum.
        if (pat.lhs != Ast.none) {
            const tname = bc.nameText(bc.tree.nodes[pat.lhs].main_token);
            if (bc.activeEnumMap().get(tname)) |qid| {
                if (qid != enum_id)
                    try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[pat.lhs].main_token), "pattern enum '{s}' does not match scrutinee '{s}'", .{ tname, e.name });
            } else {
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[pat.lhs].main_token), "'{s}' is not an enum type", .{tname});
            }
        }
        const vname = bc.nameText(pat.main_token);
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
            if (count_cov and bc.variantPayloadIrrefutable(pat_idx, e.variants[i])) cov.@"enum"[i] = true;
            break :blk e.variants[i];
        } else {
            try bc.sink.emitFmt(bc.byteOf(pat.main_token), "enum '{s}' has no variant '{s}'", .{ e.name, vname });
            return;
        };
        const binders = if (pat.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(bc.tree, pat.rhs);
        switch (variant.form) {
            .unit => {
                if (binders.len != 0)
                    try bc.sink.emitFmt(bc.byteOf(pat.main_token), "unit variant '{s}.{s}' binds no payload", .{ e.name, vname });
            },
            .tuple => {
                if (binders.len != variant.field_types.len) {
                    try bc.sink.emitFmt(bc.byteOf(pat.main_token), "variant '{s}.{s}' binds {d} value(s), got {d}", .{ e.name, vname, variant.field_types.len, binders.len });
                    return;
                }
                for (binders, variant.field_types) |b_idx, fty| {
                    try bc.checkPattern(b_idx, fty, cov, has_wildcard, false);
                }
            },
            .@"struct" => {
                for (binders) |b_idx| {
                    const b = bc.tree.nodes[b_idx];
                    // A struct binding's SOURCE field name is the rename source (lhs),
                    // or the bound name itself when punning.
                    const src_name = if (b.lhs != Ast.none) bc.nameText(bc.tree.nodes[b.lhs].main_token) else bc.nameText(b.main_token);
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
                        try bc.sink.emitFmt(bc.byteOf(b.main_token), "no field '{s}' in '{s}.{s}'", .{ src_name, e.name, vname });
                    }
                    // The carrier IS a pattern_binding: if it has a sub-pattern, match
                    // the field against it; else bind the whole field by value.
                    try bc.checkPattern(b_idx, fty, cov, has_wildcard, false);
                }
            },
        }
    }

    fn checkOrBindings(bc: *BodyChecker, or_idx: Ast.Index) error{OutOfMemory}!void {
        const alts = Ast.rangeSlice(bc.tree, bc.tree.nodes[or_idx].lhs);
        if (alts.len < 2) return;
        var first_map: std.StringHashMapUnmanaged(Type) = .empty;
        defer first_map.deinit(bc.gpa);
        try bc.collectBindings(alts[0], &first_map);
        var ok = true;
        for (alts[1..]) |alt| {
            var m: std.StringHashMapUnmanaged(Type) = .empty;
            defer m.deinit(bc.gpa);
            try bc.collectBindings(alt, &m);
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
            try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[or_idx].main_token), "or-pattern alternatives must bind the same names and types", .{});
    }

    fn collectBindings(bc: *BodyChecker, pat_idx: Ast.Index, out: *std.StringHashMapUnmanaged(Type)) error{OutOfMemory}!void {
        const pat = bc.tree.nodes[pat_idx];
        switch (pat.tag) {
            .pattern_binding => {
                const name = bc.nameText(pat.main_token);
                // The PER-NODE matched type (set in checkPattern), NOT the shared slot
                // type — so two alternatives binding the same name at different field
                // types are seen as different and rejected.
                const ty: Type = bc.node_types[pat_idx];
                try out.put(bc.gpa, name, ty);
                if (pat.rhs != Ast.none) try bc.collectBindings(pat.rhs, out);
            },
            .pattern_variant => if (pat.rhs != Ast.none)
                for (Ast.rangeSlice(bc.tree, pat.rhs)) |c| try bc.collectBindings(c, out),
            .pattern_or => for (Ast.rangeSlice(bc.tree, pat.lhs)) |a| try bc.collectBindings(a, out),
            else => {},
        }
    }

    fn typeOfIf(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        _ = node_idx;
        const ct = try bc.typeOf(n.lhs);
        if (ct.kind != .invalid and ct.kind != .bool)
            try bc.sink.emit(bc.byteOf(bc.tree.nodes[n.lhs].main_token), "if condition must be bool");
        const h = Ast.ifHeaderAt(bc.tree, n.rhs);
        if (h.else_node == Ast.none) {
            try bc.sink.emit(bc.byteOf(n.main_token), "value-if requires else");
            _ = try bc.checkBlock(h.then_block, false); // validate the arm anyway
            return .invalid;
        }
        const then_ty0 = try bc.checkBlock(h.then_block, true);
        const then_ty: Type = if (bc.blockDiverges(h.then_block)) Type.never else then_ty0;
        var else_ty: Type = undefined;
        if (bc.tree.nodes[h.else_node].tag == .if_stmt) {
            const e0 = try bc.typeOfIf(h.else_node, bc.tree.nodes[h.else_node]); // else-if ladder
            else_ty = if (bc.stmtDiverges(h.else_node)) Type.never else e0;
        } else {
            const e0 = try bc.checkBlock(h.else_node, true);
            else_ty = if (bc.blockDiverges(h.else_node)) Type.never else e0;
        }
        return bc.merge(n.main_token, then_ty, else_ty);
    }

    fn typeOfLoop(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node, label: ?[]const u8) error{OutOfMemory}!Type {
        try bc.loop_stack.append(bc.gpa, .{ .kind = .loop, .label = label, .construct_node = node_idx, .is_value = true, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
        _ = try bc.checkBlock(n.lhs, false); // body in statement ctx; breaks fill join
        const ctx = bc.loop_stack.pop().?;
        const ty: Type = if (!ctx.saw_value_break and !ctx.saw_bare_break) Type.never else ctx.join;
        bc.node_types[node_idx] = ty;
        return ty;
    }

    fn merge(bc: *BodyChecker, at: u32, a: Type, b: Type) error{OutOfMemory}!Type {
        if (a.kind == .invalid or b.kind == .invalid) return .invalid; // poison absorbs
        if (a.kind == .never) return b; // covers never+never → never
        if (b.kind == .never) return a;
        if (Type.eql(a, b)) return a; // agreement → one phi type
        try bc.sink.emitFmt(bc.byteOf(at), "branches yield different types ({s} vs {s})", .{ bc.typeName(a), bc.typeName(b) });
        return .invalid; // mismatch, no coercion
    }

    fn typeOfCall(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
        // A qualified tuple-variant construction `N.V(args)` arrives as a `.call`
        // whose callee is a `field_access` over an enum type-name identifier. Route
        // it to the enum-init checker (treating `n` as a tuple construction).
        const callee = bc.tree.nodes[n.lhs];
        if (callee.tag == .field_access) {
            const recv = bc.tree.nodes[callee.lhs];
            if (recv.tag == .identifier and bc.activeEnumMap().get(bc.nameText(recv.main_token)) != null) {
                return bc.typeOfEnumInitQualified(node_idx, .tuple, callee.lhs, callee.main_token, n.rhs);
            }
            // A cross-module tuple-variant `mod.Enum.Variant(args)` (graph mode): the
            // callee field_access's receiver is the inner `mod.Enum`.
            if (bc.qualifiedEnumId(callee.lhs)) |enum_id| {
                const ty = try bc.checkVariant(enum_id, callee.main_token, .tuple, n.rhs);
                bc.node_types[node_idx] = ty;
                return ty;
            }
        }
        const callee_res = bc.resolutions[n.lhs];
        if (callee_res != .func) {
            // Type the args anyway so their own errors surface, then poison.
            for (Ast.rangeSlice(bc.tree, n.rhs)) |arg| _ = try bc.typeOf(arg);
            if (callee_res == .local) {
                try bc.sink.emit(bc.byteOf(n.main_token), "called value is not a function");
            } else if (bc.tree.nodes[n.lhs].tag == .identifier) {
                // A struct-named callee `Point(1,2)` is positional construction, which
                // we reject — point at named construction instead.
                const cname = bc.nameText(bc.tree.nodes[n.lhs].main_token);
                if (bc.activeStructMap().get(cname) != null)
                    try bc.sink.emitFmt(bc.byteOf(n.main_token), "use named construction '{s} {{ ... }}', not '{s}(...)'", .{ cname, cname });
            } else if (callee_res == .unresolved and bc.tree.nodes[n.lhs].tag == .field_access) {
                // A qualified call `recv.member(...)` whose callee stayed `.unresolved`:
                // resolve neither bound it to a fn nor reported it (e.g. `recv` is a
                // top-level fn shadowing an import namespace, so the field-access value
                // path is taken and left unresolved). Emit a clean diagnostic at the
                // member token instead of silently poisoning — otherwise the call is
                // dropped and `-o` later crashes in codegen with no user error.
                const fa = bc.tree.nodes[n.lhs];
                const member = bc.nameText(fa.main_token);
                try bc.sink.emitFmt(bc.byteOf(fa.main_token), "cannot resolve member '{s}' to a callable function", .{member});
            }
            // Any remaining `.unresolved` was already reported by resolve.
            return .invalid;
        }
        const f = bc.model.fns[callee_res.func];
        const args = Ast.rangeSlice(bc.tree, n.rhs);
        if (args.len != f.params.len) {
            for (args) |arg| _ = try bc.typeOf(arg);
            try bc.sink.emitFmt(bc.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ f.params.len, args.len });
            return f.ret;
        }
        for (args, f.params, 0..) |arg, pty, i| {
            const at = try bc.typeOfExpected(arg, if (pty.kind == .invalid) null else pty);
            if (!Type.assignable(pty, at)) {
                try bc.sink.emitFmt(bc.byteOf(bc.tree.nodes[arg].main_token), "argument {d}: expected {s}, got {s}", .{ i + 1, bc.typeName(pty), bc.typeName(at) });
            }
        }
        return f.ret;
    }

    fn typeName(bc: *const BodyChecker, ty: Type) []const u8 {
        return refs.typeName(bc, ty);
    }

    fn graphCtx(bc: *const BodyChecker) *GraphCtx {
        return bc.model.graph;
    }
    fn structSyms(bc: *const BodyChecker) []const StructSym {
        return bc.model.structs;
    }
    fn enumSyms(bc: *const BodyChecker) []const EnumSym {
        return bc.model.enums;
    }

    fn slotType(bc: *const BodyChecker, slot: u32) Type {
        return if (slot < bc.slot_types.items.len) bc.slot_types.items[slot] else .invalid;
    }

    fn setSlot(bc: *BodyChecker, slot: u32, ty: Type) !void {
        while (bc.slot_types.items.len <= slot) try bc.slot_types.append(bc.gpa, .invalid);
        bc.slot_types.items[slot] = ty;
    }

    fn nameText(bc: *const BodyChecker, tok: u32) []const u8 {
        return refs.nameText(bc, tok);
    }

    fn byteOf(bc: *const BodyChecker, tok: u32) u32 {
        return refs.byteOf(bc, tok);
    }

    fn typeFromNode(bc: *BodyChecker, type_node: Ast.Index) Type {
        return refs.typeFromNode(bc, type_node);
    }

    fn typeFromQualified(bc: *BodyChecker, node_idx: Ast.Index, n: Ast.Node) Type {
        return refs.typeFromQualified(bc, node_idx, n);
    }

    fn activeStructMap(bc: *const BodyChecker) *const std.StringHashMapUnmanaged(u32) {
        return &bc.model.graph.mods[bc.graph_mod].struct_ids;
    }

    fn activeEnumMap(bc: *const BodyChecker) *const std.StringHashMapUnmanaged(u32) {
        return &bc.model.graph.mods[bc.graph_mod].enum_ids;
    }
};

/// Freeze the Pass-A tables into a read-only `Model`. The slices alias the live
/// `Typecheck` ArrayLists; valid for as long as those are not mutated (Pass C).
fn buildModel(t: *Typecheck) Model {
    return .{
        .fns = t.fns.items,
        .structs = t.structs.items,
        .enums = t.enums.items,
        .struct_map = &t.struct_map,
        .enum_map = &t.enum_map,
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
            try tc.sink.emitFmt(byte, "recursive type '{s}' has infinite size", .{requester});
        }
        fn emitEmptyStruct(ctx: *anyopaque, byte: u32, name: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.emitFmt(byte, "empty struct '{s}' is not allowed", .{name});
        }
        fn emitEmptyEnum(ctx: *anyopaque, byte: u32, name: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.emitFmt(byte, "empty enum '{s}' is not allowed", .{name});
        }
        fn emitUnitField(ctx: *anyopaque, byte: u32, field: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.emitFmt(byte, "field '{s}' cannot have type ()", .{field});
        }
        fn emitUnitPayload(ctx: *anyopaque, byte: u32, variant: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.emitFmt(byte, "variant '{s}' payload cannot have type ()", .{variant});
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
        .struct_map = .empty, // unused in graph mode (per-module maps live in ctx)
        .enums = .empty,
        .enum_map = .empty,
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
    const layouts = try LayoutEngine.snapshotLayouts(gpa, t.structs.items);
    errdefer LayoutEngine.freeLayouts(gpa, layouts);
    const enum_layouts = try LayoutEngine.snapshotEnumLayouts(gpa, t.enums.items);
    errdefer LayoutEngine.freeEnumLayouts(gpa, enum_layouts);

    // ---- snapshot: diagnostics (each already carries its owning module in
    // `scope`, sorted by runGraph). Hand the owned slices to the result. ----
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
    for (0..t.structs.items.len) |id| try LayoutEngine.layoutStruct(t.layoutEnv(), @intCast(id));
    for (0..t.enums.items.len) |id| try LayoutEngine.layoutEnum(t.layoutEnv(), @intCast(id));

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
    // — the S4 fan-out unit. The merge (concat + stable sort) happens once, serial,
    // after the join, so PARALLEL == SERIAL.
    const model = t.buildModel();
    try t.checkBodies(&model);
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
/// DETERMINISM ([C11]): units are independent (each writes only its own fn's
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
    if (f.decl_node == Ast.none) return;
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
        if (gf.decl_node == Ast.none or !gf.is_pub) continue;
        _ = t.gphSelect(gf.module);
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
    const non_pub = switch (ty.kind) {
        .@"struct" => !t.structs.items[ty.struct_id].pub_export,
        .@"enum" => !t.enums.items[ty.enum_id].pub_export,
        else => false,
    };
    if (non_pub)
        try t.sink.emitFmt(t.byteOf(at_tok), "pub {s} '{s}' exposes non-pub type '{s}'", .{ owner_kind, owner_name, t.typeName(ty) });
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
        if (f.decl_node == Ast.none or f.mod != entry_mod) continue;
        const tree = t.graph.mods[entry_mod].tree;
        const tokens = t.graph.mods[entry_mod].tokens;
        const source = t.graph.mods[entry_mod].source;
        const main_tok = tree.nodes[f.decl_node].main_token;
        if (!std.mem.eql(u8, tokens[main_tok].text(source), "main")) continue;
        if (f.ret.kind != .int and f.ret.kind != .unit and f.ret.kind != .invalid) {
            // Select the entry module so the sink stamps this diagnostic with the
            // entry module's scope (gphSelect -> sink.setScope).
            _ = t.gphSelect(entry_mod);
            try t.sink.emit(tokens[main_tok].start, "main must return int or ()");
        }
        return; // only the first `main` is the entry
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
            try t.sink.emitFmt(t.byteOf(decl.main_token), "struct '{s}' shadows a builtin type", .{name});
            continue;
        }
        if (t.activeStructMap().get(name) != null) {
            try t.sink.emitFmt(t.byteOf(decl.main_token), "duplicate struct declaration '{s}'", .{name});
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
    const decl = t.tree.nodes[fn_idx];
    const proto = Ast.protoAt(t.tree, decl.lhs);
    const params = try t.gpa.alloc(Type, proto.params.len);
    for (proto.params, 0..) |param_idx, i| {
        const param = t.tree.nodes[param_idx];
        const pty = t.typeFromNode(param.lhs);
        if (pty.kind == .unit) {
            try t.sink.emitFmt(t.byteOf(param.main_token), "parameter '{s}' cannot have type ()", .{t.nameText(param.main_token)});
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





/// The form a construction NODE supplies (independent of the declared variant).
const InitForm = enum { unit, tuple, @"struct" };





/// Coverage state for a match, by scrutinee kind. Enum: a per-variant seen bitmap.
/// Bool: which of true/false a literal arm has covered. Int: nothing (an infinite
/// domain — exhaustiveness only via `_`).
const Cov = union(enum) {
    @"enum": []bool,
    @"bool": *BoolCov,
    int,
};
const BoolCov = struct { t: bool = false, f: bool = false };




















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
    var diag: ?Parser.Diagnostic = null;
    const tree = (try Parser.parse(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
    errdefer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }

    var g = try Graph.single(gpa, "main", "", source, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinitSingle(gpa);
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
            const callee_res = c.resolve.resolutions[0][n.lhs];
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
            if (c.result.node_types[0][i].kind == .@"enum") {
                try testing.expectEqual(@as(u32, 0), c.result.node_types[0][i].enum_id);
                found = true;
            }
        }
    }
    try testing.expect(found);
}
