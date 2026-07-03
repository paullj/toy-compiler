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
const Mono = @import("symbols/Mono.zig");
const Composite = @import("symbols/Composite.zig");
const Infer = @import("symbols/Infer.zig");
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
        // A composite `App` (`Box[int]` / `Either[int,bool]`) renders as its generic
        // ctor's name (`Box` / `Either`) — its args are not spelled here (no alloc in
        // this borrowing accessor), but that beats leaking the internal kind tag "app"
        // into a user diagnostic. The ctor id space is struct-vs-enum disambiguated by
        // `ctor_is_enum` (M6), so render an enum-App off `enumSyms`, a struct-App off
        // `structSyms`.
        if (ty.kind == .app) {
            const e = self.composite.at(ty.appIdx());
            if (e.ctor_is_enum) {
                if (e.ctor < self.enumSyms().len) return self.enumSyms()[e.ctor].name;
            } else {
                if (e.ctor < self.structSyms().len) return self.structSyms()[e.ctor].name;
            }
        }
        return @tagName(ty.kind);
    }

    /// Map a type-reference node (an `identifier`, a `literal_unit` for `()`, or in
    /// graph mode a qualified `mod.Type` `field_access`) to a `Type`.
    pub fn typeFromNode(self: anytype, type_node: Ast.Index) Type {
        if (type_node == Ast.none) return Type.unit;
        const tn = self.tree.nodes[type_node.int()];
        if (tn.tag == .literal_unit) return Type.unit; // explicit `-> ()` / `p: ()`
        // A type application `Box[int]` (in type position or a struct-construction lhs)
        // resolves to a composite `App` (M4): the base is a generic struct ctor, each
        // arg is recursively resolved (subst-aware), interned to one composite index.
        if (tn.tag == .type_app) return refs.typeFromTypeApp(self, tn);
        // A qualified cross-module type-ref `mod.Type` parses as a field_access whose
        // receiver binds to a `.module`. Resolve it against the owning module's tables.
        if (tn.tag == .field_access) return refs.typeFromQualified(self, type_node, tn);
        const tok = tn.main_token;
        const name = refs.nameText(self, tok);
        if (type_names.get(name)) |b| return b;
        if (self.activeStructMap().get(name)) |id| {
            // A generic struct named WITHOUT type args (`x: Box`) is not a value type —
            // it needs its args (M4). Diagnose rather than mis-resolve it to `structT`.
            if (id < self.structSyms().len and self.structSyms()[id].is_generic) {
                self.sink.emitFmtCode(.T0001, refs.byteOf(self, tok), "generic type '{s}' requires type arguments, e.g. {s}[int]", .{ name, name }) catch {};
                return .invalid;
            }
            return Type.structT(id);
        }
        if (self.activeEnumMap().get(name)) |id| {
            // A generic enum named WITHOUT type args (`x: Either`) is not a value type —
            // it needs its args (M6). Diagnose rather than mis-resolve it to `enumT`
            // (mirror the generic-struct gate above).
            if (id < self.enumSyms().len and self.enumSyms()[id].is_generic) {
                self.sink.emitFmtCode(.T0001, refs.byteOf(self, tok), "generic type '{s}' requires type arguments, e.g. {s}[int]", .{ name, name }) catch {};
                return .invalid;
            }
            return Type.enumT(id);
        }
        // Inside an inherent method, `Self` names the receiver type (M8). Resolved via
        // the checker's `selfType` hook (mirroring `genericParamType`); null outside a
        // method => the normal unknown-type path, so non-method decoding is unchanged.
        if (std.mem.eql(u8, name, "Self")) {
            if (self.selfType()) |ty| return ty;
        }
        // A generic-parameter name resolves to a `type_var` (template decode) or a
        // concrete type (per-instance re-check via the checker's substitution). Empty
        // context => null => the normal unknown-type path, byte-identical otherwise.
        if (self.genericParamType(name)) |ty| return ty;
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

    /// Resolve a type application `Box[int]` / `Either[int,bool]` (a `type_app` node) to
    /// a composite `App` (M4/M6). The base must resolve to a GENERIC STRUCT or GENERIC
    /// ENUM ctor; each arg is recursively resolved via `typeFromNode` (so it grounds to a
    /// concrete type inside an instance re-check / a non-generic body, and to a
    /// `type_var` while decoding a template's field/payload patterns), then the
    /// `(ctor, is_enum, args)` tuple is interned to one composite index. A non-generic
    /// base or an arity mismatch is a clean type error.
    pub fn typeFromTypeApp(self: anytype, tn: Ast.Node) Type {
        const base = self.tree.nodes[tn.lhs.int()];
        var ctor_id: u32 = undefined;
        var is_enum = false;
        if (base.tag == .identifier) {
            const bname = refs.nameText(self, base.main_token);
            if (self.activeStructMap().get(bname)) |id| {
                ctor_id = id;
            } else if (self.activeEnumMap().get(bname)) |id| {
                ctor_id = id;
                is_enum = true;
            } else {
                self.sink.emitFmtCode(.T0001, refs.byteOf(self, base.main_token), err_unknown_type, .{bname}) catch {};
                return .invalid;
            }
        } else if (base.tag == .field_access) {
            const q = refs.typeFromQualified(self, tn.lhs, base);
            switch (q.kind) {
                .@"struct" => ctor_id = q.struct_id,
                .@"enum" => {
                    ctor_id = q.enum_id;
                    is_enum = true;
                },
                else => return q, // invalid: typeFromQualified already emitted
            }
        } else return .invalid;

        const gen_name = if (is_enum) self.enumSyms()[ctor_id].name else self.structSyms()[ctor_id].name;
        const gen_params = if (is_enum) self.enumSyms()[ctor_id].generic_params else self.structSyms()[ctor_id].generic_params;
        const gen_is_generic = if (is_enum) self.enumSyms()[ctor_id].is_generic else self.structSyms()[ctor_id].is_generic;
        if (!gen_is_generic) {
            self.sink.emitFmtCode(.T0001, refs.byteOf(self, base.main_token), "'{s}' is not generic; drop the type arguments", .{gen_name}) catch {};
            return .invalid;
        }
        const arg_nodes = Ast.rangeSlice(self.tree, tn.rhs.int());
        if (arg_nodes.len != gen_params.len) {
            self.sink.emitFmtCode(.T0001, refs.byteOf(self, tn.main_token), "'{s}' expects {d} type argument(s), got {d}", .{ gen_name, gen_params.len, arg_nodes.len }) catch {};
            return .invalid;
        }
        var buf: [8]Type = undefined;
        const args: []Type = if (arg_nodes.len <= buf.len) buf[0..arg_nodes.len] else (self.gpa.alloc(Type, arg_nodes.len) catch return .invalid);
        defer if (arg_nodes.len > buf.len) self.gpa.free(args);
        for (arg_nodes, 0..) |an, i| args[i] = refs.typeFromNode(self, an);
        const idx = self.internApp(ctor_id, args, is_enum) catch return .invalid;
        return Type.app(idx);
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
    /// For an inherent method (M8): the receiver type-ref node in the owning
    /// module's tree, decoded by `decodeFnSig` to build the method table + resolve
    /// `Self`. `Ast.none` for an ordinary fn / builtin.
    recv_type: Ast.Index = Ast.none,
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
    /// Monomorphized generic instances (M2), in canonical order. Empty for a
    /// program with no reachable generic instances. Owned.
    instances: []Mono.Instance = &.{},
    /// The program-wide inherent-method table (M8). Each entry's `name` is BORROWED
    /// from the sibling tree/source (never freed here); only the slice is owned.
    methods: []Method = &.{},

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
        for (self.instances) |*inst| {
            gpa.free(@constCast(inst.args));
            gpa.free(inst.node_types);
            gpa.free(@constCast(inst.params));
            gpa.free(@constCast(inst.name));
        }
        gpa.free(self.instances);
        // Method `name`s are borrowed from source (like `Sig.name`) — free only the slice.
        gpa.free(self.methods);
        self.* = undefined;
    }
};

/// One program-wide inherent-method table entry (M8): the receiver `Type`, the
/// SOURCE method name, and the global fn id the method desugared to. Built serially
/// in Pass A (`decodeFnSig`), frozen onto the `Model` before the parallel body pass,
/// and read read-only by `BodyChecker` dispatch, the fingerprint fold, and `lower`.
pub const Method = struct {
    recv: Type,
    name: []const u8,
    fn_id: u32,
    /// `mut self` (M9): the receiver is passed BY ADDRESS and mutated in place, so a
    /// call dispatch must enforce the receiver is a mutable place and `lower` must
    /// pass its address instead of a by-value copy. Type-checking is unaffected —
    /// the receiver type in the `Sig` stays the struct/enum (by-value-logical).
    mut_self: bool = false,
    /// A method on a GENERIC type instance (M10). Two entry flavours share this table:
    ///
    /// - TEMPLATE entry (`recv_generic == true`, `instance == null`): registered in
    ///   Pass A for `impl Box[T] { .. }`. `recv` is a check-time `App` whose index
    ///   points into the composite intern table — VALID only while checking (Pass A /
    ///   Pass C / the mono `scanCalls`); it must NEVER be dereferenced post-typecheck.
    ///   Dispatch keys off `recv_ctor`/`recv_is_enum` via `findGenericMethod`, which
    ///   compares ints only (no composite deref). `Type.eql(App, structT)` is a pure
    ///   12-byte compare that returns false on the kind mismatch, so `findMethod` (used
    ///   post-typecheck) never selects — nor dereferences — a template entry.
    /// - REIFIED-DISPATCH entry (`recv_generic == false`, `instance != null`): appended
    ///   in the mono tail for each reachable `(instance, method)`. `recv` is the reified
    ///   concrete `structT`/`enumT` and `instance` indexes `GraphResult.instances`, so
    ///   post-typecheck consumers (`lower`, `CallVisitor`) resolve the call to the
    ///   instance's mangled symbol via `findMethod` on the reified receiver.
    recv_ctor: u32 = 0,
    recv_is_enum: bool = false,
    recv_generic: bool = false,
    instance: ?u32 = null,
};

/// Look up an inherent method by receiver type + source name. A linear scan over
/// `Type.eql` (pure content comparison of kind+id) + name equality — no hashmap /
/// thread order, so every consumer selects the SAME method at any `-jN`. A generic
/// TEMPLATE entry (`recv` is a check-time `App`) is never selected by a concrete
/// `structT`/`enumT` recv (kind mismatch), so this is safe post-typecheck (when the
/// composite the template `App` indexes is freed); use `findGenericMethod` for the
/// check-time generic-receiver dispatch instead.
pub fn findMethod(methods: []const Method, recv: Type, name: []const u8) ?Method {
    for (methods) |mth| {
        if (Type.eql(mth.recv, recv) and std.mem.eql(u8, mth.name, name)) return mth;
    }
    return null;
}

/// Look up a generic-type method TEMPLATE entry by its receiver type-constructor
/// (M10). Keys off `recv_ctor`/`recv_is_enum` (plain ints, never a composite deref)
/// so it is safe while the composite is alive (Pass A / Pass C / the mono tail) and
/// deterministic at any `-jN`. Returns the first matching template entry; a concrete
/// receiver's `(ctor, is_enum)` uniquely picks the owning `impl`'s method set (one
/// `impl Box[T]` per type until coherence, M11).
pub fn findGenericMethod(methods: []const Method, ctor: u32, is_enum: bool, name: []const u8) ?Method {
    for (methods) |mth| {
        if (mth.recv_generic and mth.recv_ctor == ctor and mth.recv_is_enum == is_enum and std.mem.eql(u8, mth.name, name)) return mth;
    }
    return null;
}

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
    /// Ordered generic-parameter NAMES for a generic template `fn f[T,U](..)`
    /// (borrowed source slices; the outer array is owned by `t.fns`). Empty for a
    /// non-generic fn. `params`/`ret` of a template carry `Type.typeVar(ord)` where
    /// `ord` indexes this list; the mono tail substitutes them to concrete types.
    generic_params: []const []const u8 = &.{},
    /// For an inherent method (M8): the RESOLVED receiver `Type` (params[0] is the
    /// synthesized `self`). Cached here (a 12-byte POD) rather than re-resolved from
    /// the decl, so the body checker can map `Self` without re-running `typeFromNode`
    /// (which would double-emit). `.invalid` ⟺ this fn is not a method.
    self_type: Type = .invalid,

    pub fn isGeneric(f: FnSym) bool {
        return f.generic_params.len > 0;
    }
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

/// Transient: the generic-param names of the fn CURRENTLY being decoded in
/// `decodeFnSig`, so `refs.typeFromNode` maps a matching type-ref name to a
/// `Type.typeVar(ord)`. Set/cleared around each `decodeFnSig`; empty otherwise
/// (so non-generic decoding is byte-identical).
cur_generic_params: []const []const u8 = &.{},

/// The program-wide inherent-method table (M8), built SERIALLY in Phase A
/// (`decodeFnSig`) in global fn-id order, frozen onto the `Model` before the
/// parallel body pass. Transferred into `GraphResult.methods` by `checkGraph`.
methods: std.ArrayList(Method) = .empty,

/// The receiver `Type` of the method currently being decoded/checked, so a `Self`
/// type-ref resolves to it (via `refs.typeFromNode`'s `selfType` hook). Set around
/// each method's `decodeFnSig` (Pass A); null otherwise (non-method decoding is
/// byte-identical). The body pass sets its own copy on the `BodyChecker`.
cur_self_type: ?Type = null,

/// Monomorphization instances discovered by the serial mono tail (M2). Transferred
/// whole into `GraphResult.instances` by `checkGraph`; the leftover (on an error
/// path) is freed by `checkGraph`'s defer.
mono: std.ArrayList(Mono.Instance) = .empty,

/// The composite (`App`) intern table (M4). Heap-allocated in `checkGraph` (stable
/// address across the run, so every `BodyChecker` can borrow it), freed at teardown.
/// `App` types are interned here during checking and REIFIED away before the layout
/// snapshot, so nothing in `GraphResult` references it.
composite: *Composite = undefined,

/// Maps a composite (`App`) index -> the concrete `Type` it reified to in the
/// monomorphization tail (M4/M6): a `structT(fresh_struct_id)` for a struct-App, an
/// `enumT(fresh_enum_id)` for an enum-App. Memoizes the `reifyAppTo` dispatcher so each
/// ground `App` reifies to exactly one concrete type, and so `rewriteApp` can restore
/// the decided type without re-dispatching. Deinit'd at teardown.
reify_map: std.AutoHashMapUnmanaged(u32, Type) = .empty,

/// The current `reifyAppTo` recursion depth (M6 termination guard). An unbounded
/// generic-enum whose variant payload re-applies itself with a strictly-growing arg
/// (`enum L[T] { cons(T, L[Box[T]]) }`) recurses forever through
/// `reifyAppToEnum -> substReify -> reifyAppTo`; each level is a fresh distinct enum id
/// (args are grounded to `structT`, so `Composite.appDepth` stays 1 and CANNOT catch
/// it). This true-nesting counter latches T0017 and bails past `max_instantiation_depth`
/// — covering the same latent hazard for `reifyAppToStruct` for free.
reify_depth: u32 = 0,

/// OWNS the mangled names minted for reified generic-struct instances (M4). A reified
/// `StructSym.name` is a view into one of these; `snapshotLayouts` dupes it, so these
/// are freed wholesale at teardown (the struct teardown frees `field_*`/`offsets` but
/// NOT `name`, which for a source struct is a borrowed source slice).
reified_names: std.ArrayList([]u8) = .empty,

/// One-shot latch for the T0017 instantiation-depth diagnostic (M4): the guard fires
/// at the FIRST too-deep instantiation in the (serial) worklist and then goes quiet,
/// so an unbounded `f[T] -> f[Box[T]]` yields exactly one diagnostic at `-j1`/`-jN`.
mono_depth_capped: bool = false,

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
    /// The program-wide inherent-method table (M8), frozen from Pass A. Read-only
    /// during the parallel body pass; drives `BodyChecker` method dispatch.
    methods: []const Method,
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
        .methods = t.methods.items,
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
        .composite = t.composite,
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
    // A method (M8): put its receiver type in scope so a `Self` type-ref in a body
    // annotation resolves to it. `.invalid` ⟺ not a method (leave the hook null).
    if (f.self_type.kind != .invalid) bc.cur_self_type = f.self_type;
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
        fn reifyApp(ctx: *anyopaque, app_idx: u32) error{OutOfMemory}!Type {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            return tc.reifyAppTo(app_idx);
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
        .reifyApp = T.reifyApp,
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

    // The composite (`App`) intern table (M4). Heap-allocated so its address is stable
    // across the whole run (every `BodyChecker` borrows `*Composite`); freed below.
    const composite = try gpa.create(Composite);
    composite.* = .{};

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
        .composite = composite,
    };
    defer {
        composite.deinit(gpa);
        gpa.destroy(composite);
        t.reify_map.deinit(gpa);
        for (t.reified_names.items) |nm| gpa.free(nm);
        t.reified_names.deinit(gpa);
        for (t.fns.items) |f| {
            gpa.free(f.params);
            if (f.generic_params.len > 0) gpa.free(@constCast(f.generic_params));
        }
        t.fns.deinit(gpa);
        // Any instances not transferred into the result (an error path) are freed
        // here; the success path empties `t.mono` via `toOwnedSlice` first.
        for (t.mono.items) |*inst| {
            gpa.free(@constCast(inst.args));
            gpa.free(inst.node_types);
            gpa.free(@constCast(inst.params));
        }
        t.mono.deinit(gpa);
        // The method table's backing array (entries' names are borrowed source
        // slices). On success `toOwnedSlice` empties it, so this is a no-op there.
        t.methods.deinit(gpa);
        for (t.enums.items) |e| {
            for (e.variants) |v| {
                gpa.free(v.field_names);
                gpa.free(v.field_types);
                gpa.free(v.offsets);
            }
            gpa.free(e.variants);
            // A generic enum template's `generic_params` array is owned (M6); a reified /
            // non-generic enum's default `&.{}` frees to a no-op.
            if (e.generic_params.len > 0) gpa.free(@constCast(e.generic_params));
        }
        t.enums.deinit(gpa);
        // The bare-name maps live in the ctx (accessed via activeStructMap/
        // activeEnumMap); the CALLER owns and frees them.
        for (t.structs.items) |s| {
            gpa.free(s.field_names);
            gpa.free(s.field_types);
            gpa.free(s.offsets);
            // A generic template's `generic_params` array is owned (M4); a reified /
            // non-generic struct's default `&.{}` frees to a no-op.
            if (s.generic_params.len > 0) gpa.free(@constCast(s.generic_params));
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

    // Transfer the monomorphization instances (M2) out of the live table into the
    // result before the layout snapshot. `toOwnedSlice` empties `t.mono` so the
    // teardown defer no longer sees them; an errdefer frees them (incl. their minted
    // names) if a later snapshot fails.
    const instances = try t.mono.toOwnedSlice(gpa);
    errdefer {
        for (instances) |*inst| {
            gpa.free(@constCast(inst.args));
            gpa.free(inst.node_types);
            gpa.free(@constCast(inst.params));
            gpa.free(@constCast(inst.name));
        }
        gpa.free(instances);
    }

    // Transfer the method table (M8) out of the live list before the teardown defer
    // sees it. `toOwnedSlice` empties `t.methods`; the entries' names stay borrowed.
    const methods = try t.methods.toOwnedSlice(gpa);
    errdefer gpa.free(methods);

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
        .instances = instances,
        .methods = methods,
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

    // Phase 0a (M4): decode each generic struct TEMPLATE's field types as PATTERNS
    // (`v: T` -> `type_var(0)`; `b: Box[T]` -> `App(Box, [type_var 0])`) into its
    // `field_names`/`field_types`, with the template's generic params in scope so a
    // `T` ref decodes to a `type_var`. The mono tail's `substReify` grounds these
    // patterns per instance. Marked `.done` so nothing accidentally lays out a
    // template's type-var fields (Phase 0b skips it anyway).
    try t.decodeTemplateFields();
    try t.decodeTemplateVariants();

    // Phase 0b: lay out every NON-generic struct then every enum (global id order).
    // A generic template is skipped (its type-var fields/payloads have no ABI); only its
    // reified concrete instances (minted in the mono tail) get a layout. Each
    // layoutStruct/layoutEnum switches to its owning module; nested/qualified
    // referents recurse cross-module and restore the active module on return.
    for (0..t.structs.items.len) |id| {
        if (t.structs.items[id].is_generic) continue;
        try LayoutEngine.layoutStruct(t.layoutEnv(), @intCast(id));
    }
    for (0..t.enums.items.len) |id| {
        if (t.enums.items[id].is_generic) continue;
        try LayoutEngine.layoutEnum(t.layoutEnv(), @intCast(id));
    }

    // Phase A: decode every fn signature into the GLOBAL fn table, in the exact
    // order of `fns` (parallel to the resolver's global fn ids), so `.func` ids
    // index this table directly. The synthetic bodyless `print` is one of them.
    for (fns) |gf| {
        if (gf.kind == .builtin) {
            try t.appendPrint();
        } else {
            _ = t.gphSelect(gf.module);
            try t.decodeFnSig(gf.decl_node, gf.module, gf.recv_type);
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

    // Monomorphization tail (M2): a SERIAL pass on the LIVE tables, after the
    // per-fn Pass-C fan-out has joined and BEFORE `checkGraph` snapshots/frees them.
    // It discovers every reachable `(template, concrete-args)` instance to a
    // fixpoint, re-checks each instance body into its own `node_types`, and mints a
    // canonical mangled name — a pure function of source, so `-jN` stays identical.
    try t.monomorphize(&model);
}

/// A worklist entry: a `(template gid, concrete args)` to instantiate. `args` is
/// OWNED for the duration of the fixpoint (freed with the worklist).
const Pending = struct { gid: u32, args: []Type };

/// A generous ceiling on the number of monomorphized instances. UNREACHABLE through
/// M3 (type-args must already be concrete — there is no `App`, so the instance set is
/// a finite closure of the source's explicit AND bare-inferred call sites); the M4
/// depth cap below fires first, so this stays the belt-and-suspenders backstop.
const mono_instance_cap: usize = 10_000;

/// The M4 termination guard (T0017): the max `App`-nesting depth of a type-arg before
/// an instantiation is rejected. An unbounded `f[T]` transitively instantiating
/// `f[Box[T]]` forms `App`s of strictly-growing depth (`Box[Box[..[int]..]]`); this
/// cap makes the serial worklist reject it deterministically (never hang/OOM) while a
/// legitimately deep-but-finite generic program (nesting well under 64) still compiles.
const max_instantiation_depth: u32 = 64;

/// True when `ty` is a concrete value type usable as a monomorphization type-arg.
/// M4 admits a ground `App` (a generic-struct instance like `Box[int]` used as a
/// type-arg): it is reified to a concrete `struct_id` in the mono tail. An `App`
/// reaching a type-arg slot in a CHECKED body is always ground (templates are never
/// body-checked; an instance re-check grounds its `type_var`s via `bc.subst`). Still
/// excludes `type_var`, `unit`, and poison.
fn isConcreteValue(ty: Type) bool {
    return switch (ty.kind) {
        .int, .bool, .str, .@"struct", .@"enum", .app => true,
        else => false,
    };
}

/// Substitute a template type through a concrete arg tuple (M4): a `type_var(ord)`
/// becomes `args[ord]`; an `App(ctor, [pat..])` recursively substitutes each arg and
/// re-interns (so a template param spelled `Box[T]` grounds to `Box[int]`); anything
/// else passes through unchanged.
fn substType(t: *Typecheck, ty: Type, args: []const Type) Type {
    if (ty.isTypeVar()) {
        const ord = ty.typeVarOrd();
        return if (ord < args.len) args[ord] else Type.invalid;
    }
    if (ty.isApp()) {
        const e = t.composite.at(ty.appIdx());
        var buf: [8]Type = undefined;
        const sub: []Type = if (e.args.len <= buf.len) buf[0..e.args.len] else (t.gpa.alloc(Type, e.args.len) catch return Type.invalid);
        defer if (e.args.len > buf.len) t.gpa.free(sub);
        for (e.args, 0..) |a, i| sub[i] = t.substType(a, args);
        const idx = t.internApp(e.ctor, sub, e.ctor_is_enum) catch return Type.invalid;
        return Type.app(idx);
    }
    return ty;
}

/// The serial monomorphization tail. Seeds a worklist from every generic call site
/// in the non-generic fn bodies (fn-id order, ascending node order), processes to a
/// fixpoint (each instance re-check may discover nested generic calls), then
/// canonically sorts the instances and mints their mangled names.
fn monomorphize(t: *Typecheck, model: *const Model) !void {
    var worklist: std.ArrayList(Pending) = .empty;
    defer {
        for (worklist.items) |p| t.gpa.free(p.args);
        worklist.deinit(t.gpa);
    }
    // Dedup on the canonical `(gid, arg-bytes)` key so a repeated call site is
    // instantiated once. Keys are owned (freed at the end).
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer {
        var it = seen.keyIterator();
        while (it.next()) |k| t.gpa.free(k.*);
        seen.deinit(t.gpa);
    }

    // Seed: scan every non-generic user fn's body (in fn-id order) for call-position
    // generic calls, reading the Pass-C-populated per-module node_types.
    const nts = t.gph_node_types orelse return; // graph mode always sets it
    for (model.fns) |f| {
        if (f.kind == .builtin or f.isGeneric()) continue;
        try t.scanCalls(model, f.mod, nts[f.mod], &worklist, &seen);
    }

    var capped = false;
    var idx: usize = 0;
    while (idx < worklist.items.len) : (idx += 1) {
        if (t.mono.items.len >= mono_instance_cap) {
            if (!capped) {
                const f = model.fns[worklist.items[idx].gid];
                _ = t.gphSelect(f.mod);
                const decl = t.tree.nodes[f.decl_node.int()];
                try t.sink.emitCode(.T0014, t.byteOf(decl.main_token), "monomorphization instance limit exceeded");
                capped = true;
            }
            break;
        }
        const p = worklist.items[idx];
        const inst = try t.recheck(model, p.gid, p.args);
        try t.mono.append(t.gpa, inst);
        // Discover nested generic calls in the instance body (its own node_types).
        try t.scanCalls(model, inst.mod, inst.node_types, &worklist, &seen);
    }

    // M4 reification: mint a fresh concrete `struct_id` for every reachable ground
    // `App` (in all node_types + instance sigs + non-generic fn sigs) and REWRITE
    // every `.app` to that `structT`. Serial, BEFORE the sort/mangle (so instance
    // args are plain `structT` for `Mono.mangle`) and BEFORE the layout snapshot (so
    // codegen/fingerprint/cache see only concrete structs). Ids are assigned in an
    // index-INDEPENDENT structural-key order, so they are a pure function of source.
    try t.reifyApps(nts);

    // Canonical order (template gid, then arg bytes) drives BOTH the mangled-name
    // assignment and the downstream codegen enumeration — never discovery order.
    std.mem.sort(Mono.Instance, t.mono.items, {}, Mono.lessThan);
    for (t.mono.items) |*inst| {
        const tname = if (t.gph_fn_names) |fns| fns[inst.template_gid] else t.nameText(t.tree.nodes[model.fns[inst.template_gid].decl_node.int()].main_token);
        inst.name = try Mono.mangle(t.gpa, tname, inst.args);
    }
    // A mangled name is a pure function of (template name, arg tuple); two distinct
    // instances therefore never collide (Debug/ReleaseSafe guard).
    if (std.debug.runtime_safety) {
        for (t.mono.items, 0..) |a, i| {
            for (t.mono.items[i + 1 ..]) |b| std.debug.assert(!std.mem.eql(u8, a.name, b.name));
        }
    }

    // M10: append a REIFIED-DISPATCH `Method` entry for each reachable
    // (generic-type instance, method). This MUST run at the very end — after the
    // fixpoint, reify, sort, and naming — because the Pass-C `Model` borrows
    // `t.methods.items` (captured before Pass C) and `scanCalls` read it throughout
    // the fixpoint; appending here (once nothing reads `model.methods` again) is the
    // only realloc-safe point. Snapshot the generic TEMPLATE entries first (a value
    // copy of small PODs), so the append's ArrayList realloc can't dangle the
    // iterator. Iterate the already-canonically-sorted `t.mono`, so the reified
    // entries are a pure function of source; each entry's `recv` is the reified
    // concrete `structT`/`enumT` (the instance's substituted-then-reified self,
    // `inst.params[0]`), which is what `findMethod` keys off in lower / `CallVisitor`.
    var templates: std.ArrayList(Method) = .empty;
    defer templates.deinit(t.gpa);
    for (t.methods.items) |mth| {
        if (mth.recv_generic) try templates.append(t.gpa, mth);
    }
    if (templates.items.len > 0) {
        for (t.mono.items, 0..) |inst, i| {
            if (inst.params.len == 0) continue;
            for (templates.items) |tmpl| {
                if (tmpl.fn_id != inst.template_gid) continue;
                try t.methods.append(t.gpa, .{
                    .recv = inst.params[0],
                    .name = tmpl.name,
                    .fn_id = tmpl.fn_id,
                    .mut_self = tmpl.mut_self,
                    .recv_generic = false,
                    .instance = @intCast(i),
                });
                break;
            }
        }
    }

    // The mono tail may have emitted instance-body diagnostics (and T0014); re-sort
    // the shared stream so the final (scope, byte_offset) order is deterministic.
    t.sink.sort();
}

/// Scan module `mod`'s nodes for call-position generic calls, reading `node_types`
/// for the concrete type-args, and enqueue each new `(gid, args)`. Uniform for the
/// base seed (module node_types) and an instance re-check (the instance's own
/// node_types — other fns' nodes stay `.invalid` there, so only THIS body's calls
/// are seen).
fn scanCalls(t: *Typecheck, model: *const Model, mod: u32, node_types: []const Type, worklist: *std.ArrayList(Pending), seen: *std.StringHashMapUnmanaged(void)) !void {
    const mc = &t.graph.mods[mod];
    const tree = mc.tree;
    const resolutions = mc.resolutions;
    for (tree.nodes) |n| {
        if (n.tag != .call or n.lhs == Ast.none) continue;
        const callee = tree.nodes[n.lhs.int()];
        if (callee.tag == .type_app) {
            // Explicit-args `id[int](..)` (M2): the type-args are the type-app's
            // arg nodes, read straight from node_types.
            const bres = resolutions[callee.lhs.int()];
            if (bres != .func) continue;
            const gid = bres.func;
            const f = model.fns[gid];
            if (!f.isGeneric()) continue;
            const targ_nodes = Ast.rangeSlice(tree, callee.rhs.int());
            if (targ_nodes.len != f.generic_params.len) continue;
            const args = try t.gpa.alloc(Type, targ_nodes.len);
            defer t.gpa.free(args);
            var ok = true;
            for (targ_nodes, 0..) |tn, k| {
                const ty = node_types[tn.int()];
                if (!isConcreteValue(ty)) {
                    ok = false;
                    break;
                }
                args[k] = ty;
            }
            if (!ok) continue;
            try t.enqueueInstance(gid, args, worklist, seen, mc.tokens[n.main_token].start, mod);
        } else if (callee.tag == .identifier) {
            // Bare inferred `id(7)` (M3): re-run the SHARED matcher over the value-arg
            // node_types so discovery selects the exact same instance Pass C created.
            // The never/invalid skip + the `isConcreteValue` gate are identical to Pass
            // C's, so the `(gid, args)` tuple — hence the `Mono.Instance` — agrees.
            const bres = resolutions[n.lhs.int()];
            if (bres != .func) continue;
            const gid = bres.func;
            const f = model.fns[gid];
            if (!f.isGeneric()) continue;
            const value_args = Ast.rangeSlice(tree, n.rhs.int());
            if (value_args.len != f.params.len) continue; // Pass C already erred arity
            const arg_types = try t.gpa.alloc(Type, value_args.len);
            defer t.gpa.free(arg_types);
            for (value_args, 0..) |va, k| arg_types[k] = node_types[va.int()];
            const n_gp: u32 = @intCast(f.generic_params.len);
            const out = try t.gpa.alloc(Type, n_gp);
            defer t.gpa.free(out);
            const bnd = try t.gpa.alloc(bool, n_gp);
            defer t.gpa.free(bnd);
            const fp = try t.gpa.alloc(usize, n_gp);
            defer t.gpa.free(fp);
            switch (Infer.match(n_gp, f.params, arg_types, out, bnd, fp)) {
                .ok => {},
                else => continue, // conflict/unbound: Pass C reported it; mint nothing
            }
            var conc = true;
            for (out) |ta| {
                if (!isConcreteValue(ta)) {
                    conc = false;
                    break;
                }
            }
            if (!conc) continue;
            try t.enqueueInstance(gid, out, worklist, seen, mc.tokens[n.main_token].start, mod);
        } else if (callee.tag == .field_access) {
            // A method call on a generic-type instance `b.get()` (M10): the receiver
            // types to an `App` (reify runs later in this tail), whose ctor selects the
            // `impl Box[T]` method TEMPLATE. Bind the impl's type-params by matching the
            // template's `Self` pattern-args against the receiver App's concrete args —
            // the SAME `Infer.match` Pass C runs — then enqueue `(method_gid, targs)`.
            // (A method's OWN `[U]` generics are not inferred here; only the impl's
            // params bind, so `n_gp` == the impl's param count — deferred.)
            const recv = node_types[callee.lhs.int()];
            if (!recv.isApp()) continue;
            const re = t.composite.at(recv.appIdx());
            const member = mc.tokens[callee.main_token].text(mc.source);
            const m = findGenericMethod(model.methods, re.ctor, re.ctor_is_enum, member) orelse continue;
            const mf = model.fns[m.fn_id];
            if (!mf.self_type.isApp()) continue;
            const pat = t.composite.at(mf.self_type.appIdx()).args;
            if (pat.len != re.args.len) continue; // ctor fixes the arity; defensive
            const n_gp: u32 = @intCast(mf.generic_params.len);
            const out = try t.gpa.alloc(Type, n_gp);
            defer t.gpa.free(out);
            const bnd = try t.gpa.alloc(bool, n_gp);
            defer t.gpa.free(bnd);
            const fp = try t.gpa.alloc(usize, n_gp);
            defer t.gpa.free(fp);
            switch (Infer.match(n_gp, pat, re.args, out, bnd, fp)) {
                .ok => {},
                else => continue, // unbound/conflict: Pass C reported it; mint nothing
            }
            var conc = true;
            for (out) |ta| {
                if (!isConcreteValue(ta)) {
                    conc = false;
                    break;
                }
            }
            if (!conc) continue;
            try t.enqueueInstance(m.fn_id, out, worklist, seen, mc.tokens[n.main_token].start, mod);
        }
    }
}

/// Enqueue `(gid, args)` for instantiation, deduped on the canonical key. `args` is
/// BORROWED (the caller keeps its scratch); a fresh owned copy is stored on the
/// worklist. Shared by the explicit and inferred `scanCalls` branches so their dedup
/// path is byte-identical. `at_byte`/`mod` anchor the T0017 termination diagnostic.
fn enqueueInstance(t: *Typecheck, gid: u32, args: []const Type, worklist: *std.ArrayList(Pending), seen: *std.StringHashMapUnmanaged(void), at_byte: u32, mod: u32) !void {
    // M4 termination guard: reject an instantiation whose type-args nest generic
    // applications beyond the depth cap — the shape an unbounded `f[T] -> f[Box[T]]`
    // recursion produces. The worklist is serial, so the FIRST breach in canonical
    // processing order fires identically at `-j1`/`-jN`; the latch keeps it to one
    // diagnostic. Skipping the enqueue truncates the otherwise-infinite chain.
    var maxd: u32 = 0;
    for (args) |a| {
        const d = t.composite.appDepth(a);
        if (d > maxd) maxd = d;
    }
    if (maxd > max_instantiation_depth) {
        if (!t.mono_depth_capped) {
            _ = t.gphSelect(mod);
            try t.sink.emitCode(.T0017, at_byte, "instantiation too deep: generic type nesting exceeds the depth limit");
            t.mono_depth_capped = true;
        }
        return;
    }
    var keybuf: std.ArrayList(u8) = .empty;
    defer keybuf.deinit(t.gpa);
    try Mono.writeKey(t.gpa, &keybuf, gid, args);
    const gop = try seen.getOrPut(t.gpa, keybuf.items);
    if (gop.found_existing) return;
    gop.key_ptr.* = try t.gpa.dupe(u8, keybuf.items); // own the stored key
    const owned = try t.gpa.dupe(Type, args);
    errdefer t.gpa.free(owned);
    try worklist.append(t.gpa, .{ .gid = gid, .args = owned });
}

/// Re-check one generic instance: substitute the template's params/ret to concrete
/// types, allocate a FRESH per-instance `node_types`, and walk the template body
/// with a substitution-seeded `BodyChecker` so every node types concretely. Returns
/// an owning `Instance` (name filled after the canonical sort).
fn recheck(t: *Typecheck, model: *const Model, gid: u32, args: []const Type) !Mono.Instance {
    const f = model.fns[gid];

    const params = try t.gpa.alloc(Type, f.params.len);
    errdefer t.gpa.free(params);
    for (f.params, 0..) |p, i| params[i] = t.substType(p, args);
    const ret = t.substType(f.ret, args);

    const node_count = t.graph.mods[f.mod].tree.nodes.len;
    const inst_nt = try t.gpa.alloc(Type, node_count);
    errdefer t.gpa.free(inst_nt);
    @memset(inst_nt, .invalid);

    const args_owned = try t.gpa.dupe(Type, args);
    errdefer t.gpa.free(args_owned);

    // The BodyChecker seeds slot_types from `f.params` and `cur_ret` from `f.ret`;
    // hand it the SUBSTITUTED sig so the body types against concrete param/ret. Its
    // `subst` map resolves any generic-param type-ref inside the body to concrete,
    // and `typeOfCall` writes concrete type-arg node_types for nested generic calls.
    var fsub = f;
    fsub.params = params;
    fsub.ret = ret;

    var bc = t.bodyCheckerFor(model, fsub);
    defer bc.deinit();
    bc.node_types = inst_nt;
    bc.subst = .{ .names = f.generic_params, .types = args };
    try bc.checkBody(gid, fsub);
    // Merge the instance's diagnostics into the shared stream (sorted at the end).
    try t.sink.merge(&bc.sink);

    return .{
        .template_gid = gid,
        .args = args_owned,
        .node_types = inst_nt,
        .params = params,
        .ret = ret,
        .name = undefined, // minted in canonical order after the sort
        .mod = f.mod,
        .decl_node = f.decl_node,
    };
}

/// One entry in the reification sort (M4): a composite `App` index + its precomputed
/// nesting depth + its INDEX-INDEPENDENT structural key. Sorting by `(depth, key)`
/// makes reified-`struct_id` assignment a pure function of source (inner `App`s get
/// lower ids than outer; independent `App`s ordered by structural key), so the ids —
/// hence any `s<id>` mangling downstream — are byte-identical at `-j1`/`-jN`.
const ReifyItem = struct { idx: u32, depth: u32, key: []const u8 };

fn reifyItemLess(_: void, a: ReifyItem, b: ReifyItem) bool {
    if (a.depth != b.depth) return a.depth < b.depth;
    return std.mem.lessThan(u8, a.key, b.key);
}

/// Recursively add `ty`'s composite index (and its nested `App` args) to `set` if it
/// is an `App`. Dedup on the index so each ground `App` is reified once.
fn collectApp(t: *Typecheck, ty: Type, set: *std.AutoArrayHashMapUnmanaged(u32, void)) !void {
    if (!ty.isApp()) return;
    const gop = try set.getOrPut(t.gpa, ty.appIdx());
    if (gop.found_existing) return;
    const e = t.composite.at(ty.appIdx());
    for (e.args) |a| try t.collectApp(a, set);
}

/// If `ty` is an `App`, rewrite it in place to the concrete type it reified to
/// (`structT` for a struct-App, `enumT` for an enum-App). The stored `Type` was decided
/// once at reification, so this never has to re-dispatch on `ctor_is_enum`.
fn rewriteApp(t: *Typecheck, ty: *Type) void {
    if (ty.isApp()) {
        if (t.reify_map.get(ty.appIdx())) |rt| ty.* = rt;
    }
}

/// Reify a composite `App` to the concrete `Type` it stands for (M4/M6) — the single
/// dispatcher + memo + termination guard. Memoizes on the composite index so each
/// ground `App` reifies to exactly one concrete type; dispatches to `reifyAppToStruct`
/// (-> `structT`) or `reifyAppToEnum` (-> `enumT`) on the `ctor_is_enum` discriminator.
/// The `reify_depth` counter latches T0017 past `max_instantiation_depth` — an unbounded
/// generic-enum type-growth chain (`enum L[T] { cons(T, L[Box[T]]) }`) recurses through
/// `reifyAppToEnum -> substReify -> reifyAppTo` with a strictly-growing arg, and each
/// level is a fresh distinct id whose args are grounded (so `Composite.appDepth` stays
/// 1 and cannot catch it). Bailing with `.invalid` (UNmemoized, so a re-entry re-bails)
/// truncates the chain deterministically. The memo check runs BEFORE the depth
/// increment, so already-reified inners cost nothing and only true nesting accumulates.
fn reifyAppTo(t: *Typecheck, app_idx: u32) error{OutOfMemory}!Type {
    if (t.reify_map.get(app_idx)) |rt| return rt;
    t.reify_depth += 1;
    defer t.reify_depth -= 1;
    if (t.reify_depth > max_instantiation_depth) {
        if (!t.mono_depth_capped) {
            const e = t.composite.at(app_idx);
            const mod = if (e.ctor_is_enum) t.enums.items[e.ctor].mod else t.structs.items[e.ctor].mod;
            _ = t.gphSelect(mod);
            const decl_node = if (e.ctor_is_enum) t.enums.items[e.ctor].decl_node else t.structs.items[e.ctor].decl_node;
            try t.sink.emitCode(.T0017, t.byteOf(t.tree.nodes[decl_node.int()].main_token), "instantiation too deep: generic type nesting exceeds the depth limit");
            t.mono_depth_capped = true;
        }
        return Type.invalid; // NOT memoized: a re-entry re-bails + re-latches (quietly)
    }
    const e = t.composite.at(app_idx);
    if (e.ctor_is_enum) return t.reifyAppToEnum(app_idx);
    return Type.structT(try t.reifyAppToStruct(app_idx));
}

/// Reify a struct-`App` (`Box[int]`) to a fresh concrete `struct_id` (M4), memoized by
/// the `reifyAppTo` dispatcher (which owns the memo check + depth guard). Grounds the
/// `App`'s own args (a nested `App` arg -> its reified concrete type via `reifyAppTo`,
/// bottom-up, so a mixed nest `Box[Either[int,bool]]` grounds its enum arg to `enumT`),
/// mints a fresh struct with the template's field names + the field patterns
/// substituted through the grounded args (`substReify`), and lays it out via
/// `layoutReified` on the live tables. The minted name (`Box$int`) is a pure function
/// of the template name + the concrete args.
fn reifyAppToStruct(t: *Typecheck, app_idx: u32) error{OutOfMemory}!u32 {
    const e = t.composite.at(app_idx);
    var abuf: [8]Type = undefined;
    const cargs: []Type = if (e.args.len <= abuf.len) abuf[0..e.args.len] else try t.gpa.alloc(Type, e.args.len);
    defer if (e.args.len > abuf.len) t.gpa.free(cargs);
    for (e.args, 0..) |a, i| cargs[i] = if (a.isApp()) try t.reifyAppTo(a.appIdx()) else a;

    const tmpl = t.structs.items[e.ctor]; // value copy; its field slices are stable
    const sid: u32 = @intCast(t.structs.items.len);
    try t.reify_map.put(t.gpa, app_idx, Type.structT(sid));

    const name = try Mono.mangle(t.gpa, tmpl.name, cargs);
    try t.reified_names.append(t.gpa, name); // owns the minted name (freed at teardown)
    // Claim `sid` with a placeholder BEFORE `substReify` runs: a generic-struct-typed
    // field (`struct Wrapper[T] { b: Box[T] }`) makes `substReify` recurse into
    // `reifyAppToStruct` for the field's `App`, which mints from `structs.items.len`.
    // Appending here first advances the length past `sid`, so that inner struct lands
    // at `sid+1` instead of stealing `sid`. The `.laying` guard still catches a
    // genuinely self-referential generic (the memo returns this `sid` on re-entry).
    try t.structs.append(t.gpa, .{
        .decl_node = tmpl.decl_node,
        .name = name,
        .mod = tmpl.mod,
        .is_generic = false,
    });
    const fnames = try t.gpa.dupe([]const u8, tmpl.field_names);
    errdefer t.gpa.free(fnames);
    const ftypes = try t.gpa.alloc(Type, tmpl.field_types.len);
    errdefer t.gpa.free(ftypes);
    for (tmpl.field_types, 0..) |ft, j| ftypes[j] = try t.substReify(ft, cargs);
    t.structs.items[sid].field_names = fnames;
    t.structs.items[sid].field_types = ftypes;
    try LayoutEngine.layoutReified(t.layoutEnv(), sid);
    return sid;
}

/// Reify an enum-`App` (`Either[int,bool]`) to a fresh concrete `enum_id` (M6),
/// memoized by the `reifyAppTo` dispatcher. Mirrors `reifyAppToStruct`: grounds the
/// `App`'s own args (via `reifyAppTo`), copies the template EnumSym, claims a fresh
/// `enum_id` via a placeholder append (so a nested reify lands past it), substitutes
/// each variant's payload patterns through the grounded args (`substReify`), and lays
/// out the concrete variants via `layoutReifiedEnum` on the live tables BEFORE the enum
/// snapshot, so lower/codegen/Abi see only a plain tagged union. The minted name
/// (`Either$int$bool`) is a pure function of the template name + the concrete args.
fn reifyAppToEnum(t: *Typecheck, app_idx: u32) error{OutOfMemory}!Type {
    const e = t.composite.at(app_idx);
    var abuf: [8]Type = undefined;
    const cargs: []Type = if (e.args.len <= abuf.len) abuf[0..e.args.len] else try t.gpa.alloc(Type, e.args.len);
    defer if (e.args.len > abuf.len) t.gpa.free(cargs);
    for (e.args, 0..) |a, i| cargs[i] = if (a.isApp()) try t.reifyAppTo(a.appIdx()) else a;

    const tmpl = t.enums.items[e.ctor]; // value copy; its variant slices are stable
    const eid: u32 = @intCast(t.enums.items.len);
    try t.reify_map.put(t.gpa, app_idx, Type.enumT(eid));

    const name = try Mono.mangle(t.gpa, tmpl.name, cargs);
    try t.reified_names.append(t.gpa, name); // owns the minted name (freed at teardown)
    // Claim `eid` with a placeholder BEFORE substituting variants: a variant payload
    // that re-applies a generic (`w(Box[T])` / `cons(T, L[Box[T]])`) makes `substReify`
    // recurse + append, which must land PAST `eid` rather than steal it (mirrors
    // reifyAppToStruct's placeholder discipline).
    try t.enums.append(t.gpa, .{
        .decl_node = tmpl.decl_node,
        .name = name,
        .mod = tmpl.mod,
        .is_generic = false,
    });
    const src_variants = tmpl.variants;
    const variants = try t.gpa.alloc(VariantSym, src_variants.len);
    errdefer t.gpa.free(variants);
    var vbuilt: usize = 0;
    errdefer for (variants[0..vbuilt]) |v| {
        t.gpa.free(v.field_names);
        t.gpa.free(v.field_types);
    };
    for (src_variants, 0..) |sv, vi| {
        const fnames = try t.gpa.dupe([]const u8, sv.field_names);
        errdefer t.gpa.free(fnames);
        const ftypes = try t.gpa.alloc(Type, sv.field_types.len);
        errdefer t.gpa.free(ftypes);
        for (sv.field_types, 0..) |ft, k| ftypes[k] = try t.substReify(ft, cargs);
        variants[vi] = .{ .name = sv.name, .form = sv.form, .field_names = fnames, .field_types = ftypes };
        vbuilt += 1;
    }
    t.enums.items[eid].variants = variants;
    try LayoutEngine.layoutReifiedEnum(t.layoutEnv(), eid);
    return Type.enumT(eid);
}

/// Substitute a template field-type PATTERN through a reified instance's concrete args
/// (M4), always yielding a CONCRETE type: a `type_var(ord)` becomes `cargs[ord]`; a
/// nested `App(c, [pat..])` (a field like `b: Box[T]`) grounds its args, re-interns,
/// and reifies to `structT` (bottom-up, so `layoutReified` only ever sees `structT`);
/// anything else passes through.
fn substReify(t: *Typecheck, ty: Type, cargs: []const Type) error{OutOfMemory}!Type {
    if (ty.isTypeVar()) {
        const ord = ty.typeVarOrd();
        return if (ord < cargs.len) cargs[ord] else Type.invalid;
    }
    if (ty.isApp()) {
        const e = t.composite.at(ty.appIdx());
        var sbuf: [8]Type = undefined;
        const sub: []Type = if (e.args.len <= sbuf.len) sbuf[0..e.args.len] else try t.gpa.alloc(Type, e.args.len);
        defer if (e.args.len > sbuf.len) t.gpa.free(sub);
        for (e.args, 0..) |a, i| sub[i] = try t.substReify(a, cargs);
        const new_idx = try t.internApp(e.ctor, sub, e.ctor_is_enum);
        // Dispatch struct-vs-enum on the interned discriminator: a `Box[T]` field grounds
        // to `structT`, an `Either[T,U]`/`L[Box[T]]` payload to `enumT` (M6). The
        // `reifyAppTo` depth guard makes an unbounded enum type-growth chain terminate.
        return t.reifyAppTo(new_idx);
    }
    return ty;
}

/// Reify every reachable ground `App` to a concrete `structT`/`enumT` and rewrite all
/// `.app` occurrences to it (M4/M6). See the call site in `monomorphize` for why this
/// runs where it does. `nts` is the per-module node_types (`t.gph_node_types`).
fn reifyApps(t: *Typecheck, nts: [][]Type) !void {
    // (1) Collect every reachable ground `App` index (+ nested) from the slot sets an
    // `App` can reach lower/codegen/the snapshot through: all module node_types, every
    // non-generic fn's sig (lowered directly), and every instance's node_types + sig.
    var to_reify: std.AutoArrayHashMapUnmanaged(u32, void) = .empty;
    defer to_reify.deinit(t.gpa);
    for (nts) |mnt| for (mnt) |ty| try t.collectApp(ty, &to_reify);
    for (t.fns.items) |f| {
        if (f.isGeneric()) continue; // a template sig carries type_var patterns, never lowered
        for (f.params) |p| try t.collectApp(p, &to_reify);
        try t.collectApp(f.ret, &to_reify);
    }
    // A NON-generic struct with a concrete generic-struct field (`struct S { b:
    // Box[int] }`) carries a ground `App` in its `field_types`; reify + rewrite it so
    // the snapshot/fingerprint/lower see a plain `structT`. A generic TEMPLATE's fields
    // are type_var patterns (never ground, never lowered), so skip it; a reified
    // struct's fields are already `structT`, so `collectApp` is a no-op there.
    for (t.structs.items) |s| {
        if (s.is_generic) continue;
        for (s.field_types) |ft| try t.collectApp(ft, &to_reify);
    }
    // A non-generic enum with a concrete generic-aggregate payload (`enum E {
    // v(Box[int]) }` / `enum E { v(Either[int,bool]) }`) carries a ground `App` in a
    // variant's `field_types`; collect it so the same reify+rewrite erases it before the
    // enum snapshot/fingerprint. A generic ENUM TEMPLATE (M6) carries `type_var`/
    // App-over-type_var PATTERNS in its variant payloads (never ground, never lowered) —
    // skip it, exactly as generic structs are skipped above; a reified enum's payloads
    // are already concrete (grounded by `substReify`), so `collectApp` is a no-op there.
    for (t.enums.items) |en| {
        if (en.is_generic) continue;
        for (en.variants) |v| for (v.field_types) |ft| try t.collectApp(ft, &to_reify);
    }
    for (t.mono.items) |inst| {
        for (inst.node_types) |ty| try t.collectApp(ty, &to_reify);
        for (inst.args) |a| try t.collectApp(a, &to_reify);
        for (inst.params) |p| try t.collectApp(p, &to_reify);
        try t.collectApp(inst.ret, &to_reify);
    }

    // (2) Reify in (depth, structural-key) order so ids are a pure function of source.
    // Apps reified on-demand during Phase 0b (concrete generic-aggregate fields) are
    // already memoized; the `reifyAppTo` dispatcher returns their existing ids here.
    if (to_reify.count() > 0) {
        const items = try t.gpa.alloc(ReifyItem, to_reify.count());
        defer {
            for (items) |it| t.gpa.free(@constCast(it.key));
            t.gpa.free(items);
        }
        for (to_reify.keys(), 0..) |ai, i| {
            var kb: std.ArrayList(u8) = .empty;
            errdefer kb.deinit(t.gpa);
            try t.composite.writeStructuralKey(t.gpa, Type.app(ai), &kb);
            items[i] = .{ .idx = ai, .depth = t.composite.appDepth(Type.app(ai)), .key = try kb.toOwnedSlice(t.gpa) };
        }
        std.mem.sort(ReifyItem, items, {}, reifyItemLess);
        for (items) |it| _ = try t.reifyAppTo(it.idx);
    }

    // (3) Rewrite every `.app` -> its reified `structT`/`enumT` across the SAME slot set,
    // so no `App` survives into the snapshot / instance table / fn sigs. A field's
    // reified type has the same size as the `App` it laid out from, so offsets are stable.
    for (nts) |mnt| for (mnt) |*ty| t.rewriteApp(ty);
    for (t.fns.items) |*f| {
        if (f.isGeneric()) continue;
        for (f.params) |*p| t.rewriteApp(p);
        t.rewriteApp(&f.ret);
    }
    for (t.structs.items) |s| {
        if (s.is_generic) continue;
        for (s.field_types) |*ft| t.rewriteApp(ft);
    }
    for (t.enums.items) |en| {
        if (en.is_generic) continue; // a template's payload patterns are never lowered
        for (en.variants) |v| for (v.field_types) |*ft| t.rewriteApp(ft);
    }
    for (t.mono.items) |*inst| {
        for (inst.node_types) |*ty| t.rewriteApp(ty);
        for (@constCast(inst.args)) |*a| t.rewriteApp(a);
        for (@constCast(inst.params)) |*p| t.rewriteApp(p);
        t.rewriteApp(&inst.ret);
    }
}

/// The generics gate, now INERT (M6). Generic FUNCTIONS (M2), generic STRUCTS (M4),
/// and generic ENUMS (M6) — plus `type_app` in both type and call position — are all
/// supported and flow through the full pipeline; there is no longer any generic surface
/// to short-circuit here. Kept as a no-op (rather than deleting the call site) for
/// minimal churn and as the seam for any future pre-Phase-0 gate; always returns false.
fn gateGenerics(t: *Typecheck, mods: []const GraphModuleInput) !bool {
    _ = t;
    _ = mods;
    return false;
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
    // A generic TEMPLATE is checked only through its concrete instances (the mono
    // tail re-checks each instance body with a substitution). Checking the template
    // body directly would type its `type_var`-typed params/locals, which have no ABI
    // — so its node_types stay `.invalid`, never lowered.
    if (f.isGeneric()) return;
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
        // A generic template `struct Box[T] { .. }` carries its generic-param run in
        // `decl.rhs` (M4). Collect the ordered param NAMES so template field-type refs
        // decode to `type_var`s (Phase 0a) and `typeFromTypeApp` can arity-check; mark
        // it `is_generic` so Phase 0b SKIPS laying out its type-var fields.
        var is_generic = false;
        var gparams: []const []const u8 = &.{};
        if (decl.rhs != Ast.none) {
            is_generic = true;
            const gp_nodes = Ast.rangeSlice(t.tree, decl.rhs.int());
            const names = try t.gpa.alloc([]const u8, gp_nodes.len);
            for (gp_nodes, 0..) |gp, i| names[i] = t.nameText(t.tree.nodes[gp.int()].main_token);
            gparams = names;
        }
        const id: u32 = @intCast(t.structs.items.len);
        try t.structs.append(t.gpa, .{ .decl_node = decl_idx, .name = name, .mod = mod, .pub_export = t.tree.isPub(decl_idx), .is_generic = is_generic, .generic_params = gparams });
        try t.activeStructMap().put(t.gpa, name, id);
    }
}

/// Phase 0a (M4): decode each generic struct TEMPLATE's field types as PATTERNS into
/// its `field_names`/`field_types`, with the template's generic params in scope so a
/// `T`-spelled ref decodes to `type_var(ord)` and a `Box[T]` field to
/// `App(Box, [type_var 0])`. These patterns are the input to `substReify`, which
/// grounds them per reified instance. The template itself is never a value type, so
/// it is marked `.done` (size 0) and Phase 0b skips laying it out.
fn decodeTemplateFields(t: *Typecheck) !void {
    for (0..t.structs.items.len) |id| {
        if (!t.structs.items[id].is_generic) continue;
        _ = t.gphSelect(t.structs.items[id].mod);
        t.cur_generic_params = t.structs.items[id].generic_params;
        defer t.cur_generic_params = &.{};
        const decl = t.tree.nodes[t.structs.items[id].decl_node.int()];
        const field_nodes = Ast.rangeSlice(t.tree, decl.lhs.int());
        const names = try t.gpa.alloc([]const u8, field_nodes.len);
        errdefer t.gpa.free(names);
        const ftypes = try t.gpa.alloc(Type, field_nodes.len);
        errdefer t.gpa.free(ftypes);
        for (field_nodes, 0..) |fidx, i| {
            const field = t.tree.nodes[fidx.int()];
            names[i] = t.nameText(field.main_token);
            ftypes[i] = t.typeFromNode(field.lhs); // subst-aware: `T` -> type_var, `Box[T]` -> App
        }
        t.structs.items[id].field_names = names;
        t.structs.items[id].field_types = ftypes;
        t.structs.items[id].size = 0;
        t.structs.items[id].@"align" = 1;
        t.structs.items[id].state = .done;
    }
}

/// Phase 0a (M6): decode each generic ENUM TEMPLATE's variants into `VariantSym`s whose
/// payload `field_types` are PATTERNS (`left(L)` -> `type_var(0)`; `w(Box[T])` ->
/// `App(Box,[type_var 0])`), with the template's generic params in scope so an
/// `L`-spelled ref decodes to `type_var(ord)`. These patterns are the input to
/// `substReify`, which grounds them per reified instance in `reifyAppToEnum`. The
/// template is never a value type, so it is marked `.done` (size 0) and Phase 0b skips
/// laying it out. Mirrors `decodeTemplateFields` + the variant decode of `layoutEnum`.
fn decodeTemplateVariants(t: *Typecheck) !void {
    for (0..t.enums.items.len) |id| {
        if (!t.enums.items[id].is_generic) continue;
        _ = t.gphSelect(t.enums.items[id].mod);
        t.cur_generic_params = t.enums.items[id].generic_params;
        defer t.cur_generic_params = &.{};
        const decl = t.tree.nodes[t.enums.items[id].decl_node.int()];
        const variant_nodes = Ast.rangeSlice(t.tree, decl.lhs.int());
        const variants = try t.gpa.alloc(VariantSym, variant_nodes.len);
        errdefer t.gpa.free(variants);
        var vbuilt: usize = 0;
        errdefer for (variants[0..vbuilt]) |v| {
            t.gpa.free(v.field_names);
            t.gpa.free(v.field_types);
        };
        for (variant_nodes, 0..) |vnode_idx, vi| {
            const vnode = t.tree.nodes[vnode_idx.int()];
            const vname = t.nameText(vnode.main_token);
            var form: VariantForm = .unit;
            var payload_nodes: []const Ast.Index = &.{};
            var is_struct_form = false;
            switch (vnode.tag) {
                .enum_variant_unit => {},
                .enum_variant_tuple => {
                    form = .tuple;
                    payload_nodes = Ast.rangeSlice(t.tree, vnode.lhs.int());
                },
                .enum_variant_struct => {
                    form = .@"struct";
                    is_struct_form = true;
                    payload_nodes = Ast.rangeSlice(t.tree, vnode.lhs.int());
                },
                else => {},
            }
            const np = payload_nodes.len;
            const fnames = try t.gpa.alloc([]const u8, if (is_struct_form) np else 0);
            errdefer t.gpa.free(fnames);
            const ftypes = try t.gpa.alloc(Type, np);
            errdefer t.gpa.free(ftypes);
            for (payload_nodes, 0..) |pnode_idx, pi| {
                // A tuple payload node is a type-ref; a struct payload node is a `param`
                // (name + type-ref in lhs). `typeFromNode` is subst-aware: `L` ->
                // type_var, `Box[T]` -> App-over-type_var.
                if (is_struct_form) {
                    const pnode = t.tree.nodes[pnode_idx.int()];
                    fnames[pi] = t.nameText(pnode.main_token);
                    ftypes[pi] = t.typeFromNode(pnode.lhs);
                } else {
                    ftypes[pi] = t.typeFromNode(pnode_idx);
                }
            }
            variants[vi] = .{ .name = vname, .form = form, .field_names = fnames, .field_types = ftypes };
            vbuilt += 1;
        }
        t.enums.items[id].variants = variants;
        t.enums.items[id].size = 0;
        t.enums.items[id].@"align" = 8;
        t.enums.items[id].state = .done;
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
        // A generic template `enum Either[L,R] { .. }` carries its generic-param run in
        // `decl.rhs` (M6, mirror registerStructs). Collect the ordered param NAMES so
        // variant-payload type-refs decode to `type_var`s (Phase 0a) and
        // `typeFromTypeApp` can arity-check; mark it `is_generic` so Phase 0b SKIPS
        // laying out its type-var payloads.
        var is_generic = false;
        var gparams: []const []const u8 = &.{};
        if (decl.rhs != Ast.none) {
            is_generic = true;
            const gp_nodes = Ast.rangeSlice(t.tree, decl.rhs.int());
            const names = try t.gpa.alloc([]const u8, gp_nodes.len);
            for (gp_nodes, 0..) |gp, i| names[i] = t.nameText(t.tree.nodes[gp.int()].main_token);
            gparams = names;
        }
        const id: u32 = @intCast(t.enums.items.len);
        try t.enums.append(t.gpa, .{ .decl_node = decl_idx, .name = name, .mod = mod, .pub_export = t.tree.isPub(decl_idx), .is_generic = is_generic, .generic_params = gparams });
        try t.activeEnumMap().put(t.gpa, name, id);
    }
}

/// Decode one fn's signature (param + return types) and append a `FnSym` to the
/// global fn table. `fn_idx` is a node in the currently-active tree; `mod` its
/// owning module id. Param/return type-refs resolve via the active maps (and, for
/// a qualified `mod.Type`, via the graph context).
fn decodeFnSig(t: *Typecheck, fn_idx: Ast.Index, mod: u32, recv_type: Ast.Index) !void {
    const gid: u32 = @intCast(t.fns.items.len);
    const decl = t.tree.nodes[fn_idx.int()];
    const proto = Ast.protoAt(t.tree, decl.lhs.int());

    // A generic template: collect its ordered param NAMES so a param/ret type-ref
    // spelled as one decodes (via `genericParamType`) to a `Type.typeVar(ord)`. The
    // outer array is owned by `t.fns`; the name slices are borrowed from source.
    var gnames: [][]const u8 = &.{};
    errdefer if (gnames.len > 0) t.gpa.free(gnames);
    if (proto.generic_params.len > 0) {
        gnames = try t.gpa.alloc([]const u8, proto.generic_params.len);
        for (proto.generic_params, 0..) |gp, i| gnames[i] = t.nameText(t.tree.nodes[gp.int()].main_token);
    }
    t.cur_generic_params = gnames;
    defer t.cur_generic_params = &.{};

    // Inherent method (M8): resolve the receiver type and put it in scope so the
    // synthesized `self` param and any `Self` type-ref in the signature decode to it.
    var self_ty: Type = .invalid;
    if (recv_type != Ast.none) {
        self_ty = t.typeFromNode(recv_type);
        t.cur_self_type = self_ty;
    }
    defer t.cur_self_type = null;

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
    try t.fns.append(t.gpa, .{ .decl_node = fn_idx, .kind = .user_fn, .params = params, .ret = ret, .mod = mod, .generic_params = gnames, .self_type = self_ty });
    // Register the method into the program-wide table (SERIAL, fn-id order). The name
    // is BORROWED from source (like `Sig.name`); the table is frozen before Pass C.
    if (recv_type != Ast.none) {
        // A generic receiver `impl Box[T]` (M10): `self_ty` is a check-time `App`.
        // Record its ctor so `findGenericMethod` can dispatch off a concrete receiver
        // `App`'s ctor without dereferencing the (post-typecheck-freed) composite. A
        // fully-GROUND receiver App (no `type_var` arg) would be a concrete-type
        // inherent impl (`impl Box[int]`), which is coherence territory (M11) — reject
        // it cleanly rather than mint an uninstantiable template.
        var recv_ctor: u32 = 0;
        var recv_is_enum = false;
        var recv_generic = false;
        if (self_ty.isApp()) {
            const e = t.composite.at(self_ty.appIdx());
            recv_ctor = e.ctor;
            recv_is_enum = e.ctor_is_enum;
            recv_generic = true;
            var any_var = false;
            for (e.args) |a| if (a.isTypeVar()) {
                any_var = true;
            };
            if (!any_var) {
                try t.sink.emit(t.byteOf(decl.main_token), "an inherent 'impl' on a concrete type instance is not yet supported (use 'impl Box[T]')");
                return;
            }
        }
        try t.methods.append(t.gpa, .{
            .recv = self_ty,
            .name = t.nameText(decl.main_token),
            .fn_id = gid,
            .mut_self = proto.params.len > 0 and Ast.isMutParam(t.tree, t.tokens, proto.params[0]),
            .recv_ctor = recv_ctor,
            .recv_is_enum = recv_is_enum,
            .recv_generic = recv_generic,
        });
    }
}

/// Append the synthetic bodyless `print(str) -> ()` builtin to the fn table.
fn appendPrint(t: *Typecheck) !void {
    const params = try t.gpa.dupe(Type, &.{.str});
    try t.fns.append(t.gpa, .{ .decl_node = Ast.none, .kind = .builtin, .params = params, .ret = .unit });
}

fn typeFromNode(t: *Typecheck, type_node: Ast.Index) Type {
    return refs.typeFromNode(t, type_node);
}

/// Map a type-ref NAME to a `type_var` when it is a generic parameter of the fn
/// currently being decoded (`cur_generic_params`), else null. Read by
/// `refs.typeFromNode` so a generic template's param/ret type-refs decode to
/// ordinal-carrying type-vars. Empty context (any non-template decode) => null.
pub fn genericParamType(t: *const Typecheck, name: []const u8) ?Type {
    for (t.cur_generic_params, 0..) |gp, i| {
        if (std.mem.eql(u8, gp, name)) return Type.typeVar(@intCast(i));
    }
    return null;
}

/// The receiver type when decoding a method signature (M8), so a `Self` type-ref
/// resolves to it via `refs.typeFromNode`. Null outside a method (byte-identical).
pub fn selfType(t: *const Typecheck) ?Type {
    return t.cur_self_type;
}

/// Intern a composite `App(ctor, args)` to its table index (M4). The `refs`
/// type-application resolver calls this via the shared `anytype` cursor; `BodyChecker`
/// exposes the sibling.
pub fn internApp(t: *Typecheck, ctor: u32, args: []const Type, ctor_is_enum: bool) !u32 {
    return t.composite.intern(t.gpa, ctor, args, ctor_is_enum);
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

test "M2: the generic-fn demo typechecks clean and monomorphizes one instance per type-arg" {
    const gpa = testing.allocator;
    // The M1 gate is NARROWED in M2: a generic FN + an explicit-args CALL now compile.
    var c = try checkSource("fn id[T](x: T) -> T { x }\nstruct P { x: int, y: int }\nfn main() -> int {\n a := id[int](7)\n p := id[P](P{ x: 20, y: 15 })\n return a + p.x + p.y\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Exactly two instances (id$int, id$P) in canonical order (scalar kind < struct
    // kind), each with a distinct mangled name and a fully-concrete substituted sig.
    try testing.expectEqual(@as(usize, 2), c.result.instances.len);
    // The template name is module-qualified (`main.id`) for cross-module uniqueness.
    try testing.expectEqualStrings("main.id$int", c.result.instances[0].name);
    try testing.expect(std.mem.startsWith(u8, c.result.instances[1].name, "main.id$s"));
    try testing.expectEqual(Kind.int, c.result.instances[0].ret.kind);
    try testing.expectEqual(Kind.@"struct", c.result.instances[1].ret.kind);
    // The substituted params/ret are concrete — no `type_var` survives the mono tail.
    for (c.result.instances) |inst| {
        try testing.expect(!inst.ret.isTypeVar());
        for (inst.params) |p| try testing.expect(!p.isTypeVar());
    }
}

test "M2: repeated call sites of one (template,args) monomorphize to ONE instance (dedup)" {
    const gpa = testing.allocator;
    var c = try checkSource("fn id[T](x: T) -> T { x }\nfn main() -> int {\n a := id[int](1)\n b := id[int](2)\n c := id[int](3)\n return a + b + c\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqualStrings("main.id$int", c.result.instances[0].name);
}

test "M6: an uninstantiated generic enum (and struct) is clean and reifies NOTHING" {
    const gpa = testing.allocator;
    // A generic enum decl is NO LONGER gated (M6 un-gates it); UNinstantiated it is
    // clean and emits ZERO reified enums (never a value type). The template `Opt`
    // occupies enum id 0 with `type_var` payload patterns, never laid out / lowered.
    var e = try checkSource("enum Opt[T] { some(T), none }\nfn main() -> int { return 0 }\n");
    defer e.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), e.result.diags.len);
    // Only the template `Opt` exists — no reified `Opt$..` enum was minted.
    var reified: usize = 0;
    for (e.result.enum_layouts) |el| {
        if (std.mem.indexOfScalar(u8, el.name, '$') != null) reified += 1;
    }
    try testing.expectEqual(@as(usize, 0), reified);
    // The un-gated generic STRUCT case stays clean too (M4 regression).
    try testing.expectEqual(@as(usize, 0), try checkDiagCount("struct Box[T] { v: T }\nfn main() -> int { return 0 }\n"));
}

test "M8: an inherent method typechecks clean; the call types to the method return; one method-table entry" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int, y: int }
        \\impl P { fn sum(self) -> int { self.x + self.y } }
        \\fn main() -> int {
        \\ p := P{ x: 20, y: 22 }
        \\ return p.sum()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Exactly one method (P, "sum") in the program-wide table.
    try testing.expectEqual(@as(usize, 1), c.result.methods.len);
    try testing.expectEqualStrings("sum", c.result.methods[0].name);
    try testing.expectEqual(Kind.@"struct", c.result.methods[0].recv.kind);
    // The `p.sum()` call node types to the method's return (int).
    const nts = c.result.node_types[0];
    var found = false;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .call) {
            try testing.expectEqual(Kind.int, nts[i].kind);
            found = true;
        }
    }
    try testing.expect(found);
}

test "M8: a call to a missing method emits exactly one T0018 naming the receiver + method" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\fn main() -> int {
        \\ p := P{ x: 1 }
        \\ return p.nope()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0018, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "nope") != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "P") != null);
}

test "M8: `Self` in a method signature resolves to the receiver type" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P { fn me(self) -> Self { self } }
        \\fn main() -> int {
        \\ p := P{ x: 7 }
        \\ q := p.me()
        \\ return q.x
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The method's return Sig is the receiver struct type (Self resolved to P).
    try testing.expectEqual(@as(usize, 1), c.result.methods.len);
    const gid = c.result.methods[0].fn_id;
    try testing.expectEqual(Kind.@"struct", c.result.sigs[gid].ret.kind);
}

test "M9: decodeFnSig sets Method.mut_self for `mut self` but not plain `self`; the Sig stays by-value" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int, y: int }
        \\impl P {
        \\ fn bump(mut self, d: int) { self.x = self.x + d }
        \\ fn get(self) -> int { self.x }
        \\}
        \\fn main() -> int {
        \\ p := P{ x: 40, y: 0 }
        \\ p.bump(2)
        \\ return p.get()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 2), c.result.methods.len);
    for (c.result.methods) |m| {
        if (std.mem.eql(u8, m.name, "bump")) {
            try testing.expect(m.mut_self);
            // ABI is a lowering concern: the Sig's params[0] stays the struct type.
            try testing.expectEqual(Kind.@"struct", c.result.sigs[m.fn_id].params[0].kind);
        } else if (std.mem.eql(u8, m.name, "get")) {
            try testing.expect(!m.mut_self);
            try testing.expectEqual(Kind.@"struct", c.result.sigs[m.fn_id].params[0].kind);
        } else return error.TestUnexpectedResult;
    }
}

test "M9: a mut-self method on a temporary emits T0019; on a local / a field of a local it does not" {
    const gpa = testing.allocator;
    // On a temporary (the fresh construction): rejected.
    {
        var c = try checkSource(
            \\struct P { x: int, y: int }
            \\impl P { fn bump(mut self, d: int) { self.x = self.x + d } }
            \\fn main() -> int {
            \\ P{ x: 40, y: 0 }.bump(2)
            \\ return 0
            \\}
            \\
        );
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(codes.Code.T0019, c.result.diags[0].code);
    }
    // On a local, and on a field of a local: accepted (no T0019).
    {
        var c = try checkSource(
            \\struct P { x: int, y: int }
            \\struct Q { inner: P }
            \\impl P { fn bump(mut self, d: int) { self.x = self.x + d } }
            \\fn main() -> int {
            \\ p := P{ x: 40, y: 0 }
            \\ p.bump(1)
            \\ q := Q{ inner: P{ x: 1, y: 0 } }
            \\ q.inner.bump(1)
            \\ return p.x + q.inner.x
            \\}
            \\
        );
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    }
}

test "M10: a generic-type method typechecks clean; one template + one reified entry; call types to T" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\impl Box[T] { fn get(self) -> T { self.v } }
        \\fn main() -> int {
        \\ b := Box[int]{ v: 42 }
        \\ return b.get()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);

    // Exactly one generic TEMPLATE entry (recv is a check-time App) and exactly one
    // REIFIED-DISPATCH entry (recv a concrete struct, carrying the mono instance idx).
    var templates: usize = 0;
    var reified: usize = 0;
    for (c.result.methods) |m| {
        if (m.recv_generic) {
            templates += 1;
            try testing.expectEqualStrings("get", m.name);
        } else if (m.instance != null) {
            reified += 1;
            try testing.expectEqual(Kind.@"struct", m.recv.kind);
        }
    }
    try testing.expectEqual(@as(usize, 1), templates);
    try testing.expectEqual(@as(usize, 1), reified);

    // One monomorphized instance: `<path>.Box.get$int`, returning int.
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expect(std.mem.indexOf(u8, c.result.instances[0].name, "get$int") != null);
    try testing.expectEqual(Kind.int, c.result.instances[0].ret.kind);
    // No `.app`/`type_var` survives into the reified instance sig.
    try testing.expect(!c.result.instances[0].ret.isTypeVar());
    try testing.expect(c.result.instances[0].ret.kind != .app);

    // The `b.get()` call node types to int (T substituted).
    const nts = c.result.node_types[0];
    var found = false;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .call) {
            try testing.expectEqual(Kind.int, nts[i].kind);
            found = true;
        }
    }
    try testing.expect(found);
}

test "M10: an UNCALLED generic-type method emits zero instances and zero reified-dispatch entries" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\impl Box[T] { fn get(self) -> T { self.v } }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.instances.len);
    var templates: usize = 0;
    var reified: usize = 0;
    for (c.result.methods) |m| {
        if (m.recv_generic) {
            templates += 1;
        } else if (m.instance != null) {
            reified += 1;
        }
    }
    try testing.expectEqual(@as(usize, 1), templates);
    try testing.expectEqual(@as(usize, 0), reified);
}

test "M10: Box[int] and Box[Point] .get() lower to two DISTINCT instances; an uncalled sibling method emits none" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Point { x: int, y: int }
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\ fn get(self) -> T { self.v }
        \\ fn same(self) -> Box[T] { self }
        \\}
        \\fn main() -> int {
        \\ bi := Box[int]{ v: 40 }
        \\ bp := Box[Point]{ v: Point{ x: 2, y: 9 } }
        \\ p := bp.get()
        \\ return bi.get() + p.x
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Exactly two `get` instances (int + Point); `same` is never called → none.
    try testing.expectEqual(@as(usize, 2), c.result.instances.len);
    try testing.expect(!std.mem.eql(u8, c.result.instances[0].name, c.result.instances[1].name));
    for (c.result.instances) |inst| {
        try testing.expect(std.mem.indexOf(u8, inst.name, "same") == null);
        try testing.expect(std.mem.indexOf(u8, inst.name, "get$") != null);
    }
}

test "M10: a mut-self generic-type method carries mut_self on the template AND the reified entry" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\ fn get(self) -> T { self.v }
        \\ fn set(mut self, x: T) { self.v = x }
        \\}
        \\fn main() -> int {
        \\ b := Box[int]{ v: 0 }
        \\ b.set(42)
        \\ return b.get()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    for (c.result.methods) |m| {
        if (std.mem.eql(u8, m.name, "set")) try testing.expect(m.mut_self);
        if (std.mem.eql(u8, m.name, "get")) try testing.expect(!m.mut_self);
    }
    // One instance each for get and set on Box[int].
    try testing.expectEqual(@as(usize, 2), c.result.instances.len);
}

test "M10: calling a missing method on a generic-type instance emits T0018" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\impl Box[T] { fn get(self) -> T { self.v } }
        \\fn main() -> int {
        \\ b := Box[int]{ v: 1 }
        \\ return b.nope()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0018, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "nope") != null);
}

test "M10: an inherent impl on a CONCRETE type instance (impl Box[int]) is rejected, not monomorphized" {
    // Coherence territory (M11): `impl Box[int]` has a fully-ground receiver App (no
    // type_var), so it is rejected at decode with a clear diagnostic and mints no
    // method — never a silent poison.
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\impl Box[int] { fn get(self) -> int { self.v } }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expect(c.result.diags.len >= 1);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "concrete type instance") != null);
    // No generic-template method entry was registered for the rejected impl.
    for (c.result.methods) |m| try testing.expect(!m.recv_generic);
}

test "M4: Box[int] monomorphizes to a reified 1-int concrete struct (size 8)" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Box[T] { v: T }\nfn main() -> int {\n b := Box[int]{ v: 41 }\n c := Box[int]{ v: 1 }\n return b.v + c.v\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // No generic FN instances (Box is a struct, not a fn).
    try testing.expectEqual(@as(usize, 0), c.result.instances.len);
    // Exactly one reified concrete struct `Box$int` (both `Box[int]` share it), size 8,
    // one int field at offset 0. The template `Box` is laid out to size 0 (never a value).
    var reified: ?Layout = null;
    for (c.result.layouts) |l| {
        if (std.mem.eql(u8, l.name, "Box$int")) reified = l;
    }
    try testing.expect(reified != null);
    try testing.expectEqual(@as(u32, 8), reified.?.size);
    try testing.expectEqual(@as(usize, 1), reified.?.field_types.len);
    try testing.expectEqual(Kind.int, reified.?.field_types[0].kind);
    try testing.expectEqual(@as(u32, 0), reified.?.offsets[0]);
    // No `.app` survives into any node_types slot (rewritten to structT before snapshot).
    for (c.result.node_types) |mnt| for (mnt) |ty| try testing.expect(ty.kind != .app and ty.kind != .type_var);
}

test "M4: Box[int] and Box[bool] reify to TWO distinct concrete layouts" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Box[T] { v: T }\nfn use(x: bool) -> int { return 0 }\nfn main() -> int {\n a := Box[int]{ v: 7 }\n b := Box[bool]{ v: true }\n return a.v + use(b.v)\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var saw_int = false;
    var saw_bool = false;
    for (c.result.layouts) |l| {
        if (std.mem.eql(u8, l.name, "Box$int")) saw_int = true;
        if (std.mem.eql(u8, l.name, "Box$bool")) saw_bool = true;
    }
    try testing.expect(saw_int and saw_bool);
}

test "M4: unbounded f[T]->f[Box[T]] is rejected with T0017 and TERMINATES (no hang)" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Box[T] { v: T }\nfn go[T](x: T) { go[Box[T]](Box[Box[T]]{ v: x }) }\nfn main() { go[int](0) }\n");
    defer c.deinit(gpa);
    // Exactly one T0017 (the one-shot latch), and the checker RETURNED (this test
    // completing at all is the no-hang proof).
    var t0017: usize = 0;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0017) t0017 += 1;
    }
    try testing.expectEqual(@as(usize, 1), t0017);
}

test "M4: a non-generic struct with a concrete generic-struct field compiles + reifies" {
    const gpa = testing.allocator;
    // `S` is non-generic but embeds `Box[int]`; the field `App` is reified to a
    // `structT` and rewritten away, and construction/access typecheck against the
    // same interned `App` the value expression produces (no App-vs-structT mismatch).
    var c = try checkSource("struct Box[T] { v: T }\nstruct S { b: Box[int], n: int }\nfn main() -> int {\n s := S{ b: Box[int]{ v: 40 }, n: 2 }\n return s.b.v + s.n\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // S's `b` field is a concrete struct (the reified Box$int) after rewrite — no `App`
    // survives in a REACHABLE (value-typed) layout. (The uninstantiated template `Box`
    // keeps its `type_var` field pattern in the snapshot, but is never a value type, so
    // it is never touched/folded/lowered.)
    var s_layout: ?Layout = null;
    for (c.result.layouts) |l| {
        if (std.mem.eql(u8, l.name, "S")) s_layout = l;
    }
    try testing.expect(s_layout != null);
    for (s_layout.?.field_types) |ft| try testing.expect(ft.kind != .app and ft.kind != .type_var);
    try testing.expectEqual(Kind.@"struct", s_layout.?.field_types[0].kind);
}

test "M4: a non-generic enum with a concrete generic-struct payload reifies (no App survives the enum snapshot)" {
    const gpa = testing.allocator;
    // Regression (check-vs-build differential): `enum E { some(Box[int]) }` carries a
    // ground `App` in its variant payload. Before the fix, layoutEnum sized the `.app`
    // as a scalar (size 0) and left the raw `App` in the variant field_types, so `check`
    // reported CLEAN while codegen (build) choked on the surviving `App`. The App must be
    // reified to the concrete `Box$int` struct + rewritten away before the enum snapshot.
    var c = try checkSource("struct Box[T] { v: T }\nenum E { none, some(Box[int]) }\nfn get(e: E) -> int {\n match e {\n .some(b) -> b.v,\n _ -> 0\n }\n}\nfn main() -> int {\n return get(E.some(Box[int]{ v: 42 }))\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The reified Box$int exists and is size 8.
    var box_id: ?usize = null;
    for (c.result.layouts, 0..) |l, i| {
        if (std.mem.eql(u8, l.name, "Box$int")) box_id = i;
    }
    try testing.expect(box_id != null);
    try testing.expectEqual(@as(u32, 8), c.result.layouts[box_id.?].size);
    // No `.app` survives ANY enum variant payload; `E.some`'s payload is the reified
    // Box$int (a plain `structT` pointing at the reified id), not an `App`.
    var saw_some_box = false;
    for (c.result.enum_layouts) |el| {
        if (!std.mem.eql(u8, el.name, "E")) continue;
        for (el.variants) |v| {
            for (v.field_types) |ft| try testing.expect(ft.kind != .app and ft.kind != .type_var);
            if (std.mem.eql(u8, v.name, "some")) {
                try testing.expectEqual(Kind.@"struct", v.field_types[0].kind);
                try testing.expectEqual(@as(u32, @intCast(box_id.?)), v.field_types[0].struct_id);
                saw_some_box = true;
            }
        }
    }
    try testing.expect(saw_some_box);
}

test "M4: a legitimately deep-but-finite generic-struct nest still compiles" {
    const gpa = testing.allocator;
    // Box[Box[Box[int]]] is finite (depth 3, well under the cap) — no T0017.
    var c = try checkSource("struct Box[T] { v: T }\nfn main() -> int {\n b := Box[Box[Box[int]]]{ v: Box[Box[int]]{ v: Box[int]{ v: 42 } } }\n return b.v.v.v\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Three reified structs: Box$int, Box$s<..>, Box$s<..> (nested), all size 8.
    var reified_count: usize = 0;
    for (c.result.layouts) |l| {
        if (std.mem.startsWith(u8, l.name, "Box$")) reified_count += 1;
    }
    try testing.expectEqual(@as(usize, 3), reified_count);
}

test "M4: a generic struct with a generic-struct field reifies correctly with the wrapper declared FIRST" {
    const gpa = testing.allocator;
    // Regression: `reifyAppToStruct` must claim its `struct_id` slot BEFORE `substReify`
    // recurses into the field's `App`. With `Wrapper` declared before `Box`, `Wrapper`'s
    // ctor sorts first, so reifying `Wrapper[int]` recurses into `Box[int]` while the
    // wrapper slot is unfilled — the inner struct must NOT steal the wrapper's id.
    var c = try checkSource("struct Wrapper[T] { b: Box[T] }\nstruct Box[T] { v: T }\nfn main() -> int {\n w := Wrapper[int]{ b: Box[int]{ v: 41 } }\n return w.b.v\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Wrapper$int and Box$int are DISTINCT reified structs, both laid out (size > 0).
    var wid: ?usize = null;
    var bid: ?usize = null;
    for (c.result.layouts, 0..) |l, i| {
        if (std.mem.eql(u8, l.name, "Wrapper$int")) wid = i;
        if (std.mem.eql(u8, l.name, "Box$int")) bid = i;
    }
    try testing.expect(wid != null and bid != null);
    try testing.expect(wid.? != bid.?);
    // Box$int is one int, size 8; Wrapper$int is one Box$int, size 8.
    try testing.expectEqual(@as(u32, 8), c.result.layouts[bid.?].size);
    try testing.expectEqual(@as(u32, 8), c.result.layouts[wid.?].size);
    // Wrapper$int's single field is the reified Box$int (not itself, not a bogus id).
    try testing.expectEqual(@as(usize, 1), c.result.layouts[wid.?].field_types.len);
    try testing.expectEqual(Kind.@"struct", c.result.layouts[wid.?].field_types[0].kind);
    try testing.expectEqual(@as(u32, @intCast(bid.?)), c.result.layouts[wid.?].field_types[0].struct_id);
    for (c.result.node_types) |mnt| for (mnt) |ty| try testing.expect(ty.kind != .app and ty.kind != .type_var);
}

test "M5: Box{v:1} infers Box[int] and dedups with explicit Box[int] to ONE reified struct" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Box[T] { v: T }\nfn main() -> int {\n b := Box[int]{ v: 41 }\n c := Box{ v: 1 }\n return b.v + c.v\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Box is a struct, not a fn — no generic-FN instances.
    try testing.expectEqual(@as(usize, 0), c.result.instances.len);
    // The inferred `Box{v:1}` and explicit `Box[int]{..}` share ONE reified layout.
    var box_int_count: usize = 0;
    var reified: ?Layout = null;
    for (c.result.layouts) |l| {
        if (std.mem.eql(u8, l.name, "Box$int")) {
            box_int_count += 1;
            reified = l;
        }
    }
    try testing.expectEqual(@as(usize, 1), box_int_count);
    try testing.expectEqual(@as(u32, 8), reified.?.size);
    try testing.expectEqual(@as(usize, 1), reified.?.field_types.len);
    try testing.expectEqual(Kind.int, reified.?.field_types[0].kind);
    try testing.expectEqual(@as(u32, 0), reified.?.offsets[0]);
    // No `.app`/`.type_var` survives any node_types slot (the M4 invariant).
    for (c.result.node_types) |mnt| for (mnt) |ty| try testing.expect(ty.kind != .app and ty.kind != .type_var);
}

test "M5: an inferred Box{v:true} reifies a distinct Box$bool layout" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Box[T] { v: T }\nfn use(x: bool) -> int { return 0 }\nfn main() -> int {\n c := Box{ v: true }\n return use(c.v)\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var saw_bool = false;
    for (c.result.layouts) |l| {
        if (std.mem.eql(u8, l.name, "Box$bool")) saw_bool = true;
    }
    try testing.expect(saw_bool);
}

test "M5: conflicting inferred field types report exactly one T0015" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Pair[T] { a: T, b: T }\nfn main() -> int {\n p := Pair{ a: 1, b: true }\n return 0\n}\n");
    defer c.deinit(gpa);
    var n15: usize = 0;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0015) n15 += 1;
    }
    try testing.expectEqual(@as(usize, 1), n15);
}

test "M5: a phantom (uninferable) struct type-param routes to T0016" {
    const gpa = testing.allocator;
    var c = try checkSource("struct P[T] { x: int }\nfn main() -> int {\n p := P{ x: 1 }\n return 0\n}\n");
    defer c.deinit(gpa);
    var saw16 = false;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0016) saw16 = true;
    }
    try testing.expect(saw16);
}

test "M5: missing-field-with-inference still infers T then reports the missing field" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Pair[T] { a: T, b: T }\nfn main() -> int {\n p := Pair{ a: 1 }\n return 0\n}\n");
    defer c.deinit(gpa);
    // Inference succeeds (T=int from `a`), so this is neither a conflict nor uninferable;
    // the missing `b` is reported by the shared field-check tail.
    for (c.result.diags) |d| {
        try testing.expect(d.code != codes.Code.T0015);
        try testing.expect(d.code != codes.Code.T0016);
    }
    var saw_missing = false;
    for (c.result.diags) |d| {
        if (std.mem.indexOf(u8, d.message, "missing field 'b'") != null) saw_missing = true;
    }
    try testing.expect(saw_missing);
}

test "M5: a unit-typed inferred field value is gated with T0013 before internApp" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Box[T] { v: T }\nfn nop() {}\nfn main() -> int {\n c := Box{ v: nop() }\n return 0\n}\n");
    defer c.deinit(gpa);
    var saw13 = false;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0013) saw13 = true;
    }
    try testing.expect(saw13);
}

test "M6: the Either e2e typechecks clean and reifies exactly one concrete enum" {
    const gpa = testing.allocator;
    var c = try checkSource("enum Either[L,R] { left(L), right(R) }\nfn main() -> int {\n e := Either[int,bool].left(42)\n return match e { .left(n) -> n, .right(_) -> 0 }\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Exactly one reified concrete enum `Either$int$bool`, with an 8-byte tag and the
    // payload sized to the largest variant (int|bool => 8): total 16.
    var reified: ?EnumLayout = null;
    var reified_count: usize = 0;
    for (c.result.enum_layouts) |el| {
        if (std.mem.indexOfScalar(u8, el.name, '$') != null) {
            reified_count += 1;
            reified = el;
        }
    }
    try testing.expectEqual(@as(usize, 1), reified_count);
    try testing.expectEqualStrings("Either$int$bool", reified.?.name);
    try testing.expectEqual(@as(u32, 8), reified.?.tag_size);
    try testing.expectEqual(@as(u32, 8), reified.?.payload_off);
    try testing.expectEqual(@as(u32, 16), reified.?.size);
    // The reified variants carry CONCRETE payload types (L->int, R->bool) — no `.app`
    // / `.type_var` survives into the enum-layout snapshot.
    for (reified.?.variants) |v| {
        for (v.field_types) |ft| try testing.expect(ft.kind != .app and ft.kind != .type_var);
    }
    var saw_left_int = false;
    for (reified.?.variants) |v| {
        if (std.mem.eql(u8, v.name, "left")) {
            try testing.expectEqual(Kind.int, v.field_types[0].kind);
            saw_left_int = true;
        }
    }
    try testing.expect(saw_left_int);
    // No `.app` / `.type_var` survives ANY node_types slot (the reify invariant that
    // the lower assert at lower.zig:459 guards).
    for (c.result.node_types) |mnt| for (mnt) |ty| try testing.expect(ty.kind != .app and ty.kind != .type_var);
}

test "M6: a reified enum's payload offsets/size match a hand-written non-generic twin" {
    const gpa = testing.allocator;
    // `Pair[int,int]` (reg-pair payload, size 16) vs `Wrap[str]` (str payload: a 16-byte
    // aggregate => size 24, the indirect boundary). Compare each reified layout against a
    // hand-written non-generic enum with the same variants.
    var c = try checkSource(
        \\enum Pair[A,B] { both(A, B), none }
        \\enum W[T] { wrap(T), empty }
        \\enum PairC { both(int, int), none }
        \\enum WC { wrap(str), empty }
        \\fn usePair(p: Pair[int,int]) -> int { return 0 }
        \\fn useW(w: W[str]) -> int { return 0 }
        \\fn usePairC(p: PairC) -> int { return 0 }
        \\fn useWC(w: WC) -> int { return 0 }
        \\fn main() -> int {
        \\ a := usePair(Pair[int,int].both(1, 2))
        \\ b := useW(W[str].wrap("hi"))
        \\ c := usePairC(PairC.both(1, 2))
        \\ d := useWC(WC.wrap("hi"))
        \\ return a + b + c + d
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var pair_g: ?EnumLayout = null;
    var pair_c: ?EnumLayout = null;
    var w_g: ?EnumLayout = null;
    var w_c: ?EnumLayout = null;
    for (c.result.enum_layouts) |el| {
        if (std.mem.eql(u8, el.name, "Pair$int$int")) pair_g = el;
        if (std.mem.eql(u8, el.name, "PairC")) pair_c = el;
        if (std.mem.eql(u8, el.name, "W$str")) w_g = el;
        if (std.mem.eql(u8, el.name, "WC")) w_c = el;
    }
    try testing.expect(pair_g != null and pair_c != null and w_g != null and w_c != null);
    // Reified `Pair$int$int` == hand-written `PairC` (reg-pair payload).
    try testing.expectEqual(pair_c.?.size, pair_g.?.size);
    try testing.expectEqual(pair_c.?.payload_off, pair_g.?.payload_off);
    try testing.expectEqual(pair_c.?.@"align", pair_g.?.@"align");
    // Reified `W$str` == hand-written `WC` (str aggregate payload, the indirect boundary).
    try testing.expectEqual(w_c.?.size, w_g.?.size);
    try testing.expectEqual(w_c.?.payload_off, w_g.?.payload_off);
}

test "M6: Opt.none with no target reports exactly one T0016 (deterministic)" {
    const gpa = testing.allocator;
    var c = try checkSource("enum Opt[T] { some(T), none }\nfn main() -> int {\n x := Opt.none\n return 0\n}\n");
    defer c.deinit(gpa);
    var n16: usize = 0;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0016) n16 += 1;
    }
    try testing.expectEqual(@as(usize, 1), n16);
    // No reified enum was minted (construction failed before reification).
    for (c.result.enum_layouts) |el| try testing.expect(std.mem.indexOfScalar(u8, el.name, '$') == null);
}

test "M6: a construction payload-type mismatch reports one clean error (proves L->int subst)" {
    const gpa = testing.allocator;
    // `Either[int,bool].left(true)`: `left`'s payload pattern `L` substitutes to `int`,
    // so passing a `bool` is a mismatch — proving the substitution actually happened.
    var c = try checkSource("enum Either[L,R] { left(L), right(R) }\nfn main() -> int {\n e := Either[int,bool].left(true)\n return match e { .left(n) -> n, .right(_) -> 0 }\n}\n");
    defer c.deinit(gpa);
    var saw = false;
    for (c.result.diags) |d| {
        if (std.mem.indexOf(u8, d.message, "expected int, got bool") != null) saw = true;
    }
    try testing.expect(saw);
}

test "M6: an M5-style inferred construction Wrap.w(5) dedups with explicit Wrap[int].w(5)" {
    const gpa = testing.allocator;
    var c = try checkSource("enum Wrap[T] { w(T) }\nfn use(x: Wrap[int]) -> int { return 0 }\nfn main() -> int {\n a := use(Wrap[int].w(5))\n b := use(Wrap.w(6))\n return a + b\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The inferred `Wrap.w(6)` and explicit `Wrap[int].w(5)` are the SAME interned App,
    // so they reify to ONE `Wrap$int` enum.
    var count: usize = 0;
    for (c.result.enum_layouts) |el| {
        if (std.mem.eql(u8, el.name, "Wrap$int")) count += 1;
    }
    try testing.expectEqual(@as(usize, 1), count);
}

test "M6: a non-generic struct field of a generic-enum instance reifies + rewrites away" {
    const gpa = testing.allocator;
    // `struct S { o: Opt[int] }` embeds a concrete generic-enum instance; the field `App`
    // is reified to the concrete `Opt$int` enum and rewritten to a plain `enumT`.
    var c = try checkSource("enum Opt[T] { some(T), none }\nstruct S { o: Opt[int] }\nfn main() -> int {\n s := S{ o: Opt[int].some(7) }\n return match s.o { .some(n) -> n, .none -> 0 }\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // S's `o` field is a plain concrete enum (the reified Opt$int) after rewrite — no App.
    var s_layout: ?Layout = null;
    for (c.result.layouts) |l| {
        if (std.mem.eql(u8, l.name, "S")) s_layout = l;
    }
    try testing.expect(s_layout != null);
    try testing.expectEqual(Kind.@"enum", s_layout.?.field_types[0].kind);
    for (s_layout.?.field_types) |ft| try testing.expect(ft.kind != .app and ft.kind != .type_var);
}

test "M6: unbounded generic-enum type-growth is rejected with T0017 and TERMINATES" {
    const gpa = testing.allocator;
    // `enum L[T] { cons(T, L[Box[T]]), nil }` used as a ground type grows the payload
    // App forever (L[int] -> L[Box[int]] -> L[Box[Box[int]]] -> ...). Each level is a
    // fresh distinct enum id whose args are grounded (appDepth stays 1), so ONLY the
    // reify-recursion-depth counter can catch it. This test COMPLETING is the no-hang
    // proof; exactly one T0017 (the one-shot latch).
    var c = try checkSource("struct Box[T] { v: T }\nenum L[T] { cons(T, L[Box[T]]), nil }\nfn f(x: L[int]) -> int { return 0 }\nfn main() -> int { return 0 }\n");
    defer c.deinit(gpa);
    var t0017: usize = 0;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0017) t0017 += 1;
    }
    try testing.expectEqual(@as(usize, 1), t0017);
}

test "M3: a bare (no-explicit-args) generic call infers its type-arg from the argument" {
    const gpa = testing.allocator;
    var c = try checkSource("fn id[T](x: T) -> T { x }\nfn main() -> int { return id(7) }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // `id(7)` infers T=int and monomorphizes to exactly one instance, same as id[int].
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqualStrings("main.id$int", c.result.instances[0].name);
    try testing.expect(!c.result.instances[0].ret.isTypeVar());
}

test "M3: nested bare inference (snd(true, id(42))) infers all type-args; args are concrete" {
    const gpa = testing.allocator;
    var c = try checkSource("fn snd[T,U](a: T, b: U) -> U { b }\nfn id[T](x: T) -> T { x }\nfn main() -> int { return snd(true, id(42)) }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Two instances: id$int (from the nested bare call) and snd$bool$int.
    try testing.expectEqual(@as(usize, 2), c.result.instances.len);
    for (c.result.instances) |inst| {
        try testing.expect(!inst.ret.isTypeVar());
        for (inst.params) |p| try testing.expect(!p.isTypeVar());
    }
    // The snd instance's substituted sig is (bool, int) -> int (no surviving type_var).
    var saw_snd = false;
    for (c.result.instances) |inst| {
        if (std.mem.startsWith(u8, inst.name, "main.snd$")) {
            saw_snd = true;
            try testing.expectEqualStrings("main.snd$bool$int", inst.name);
            try testing.expectEqual(@as(usize, 2), inst.params.len);
            try testing.expectEqual(Kind.bool, inst.params[0].kind);
            try testing.expectEqual(Kind.int, inst.params[1].kind);
            try testing.expectEqual(Kind.int, inst.ret.kind);
        }
    }
    try testing.expect(saw_snd);
}

test "M3: an inferred call and its explicit form DEDUP to ONE instance (byte-for-byte parity)" {
    const gpa = testing.allocator;
    // Both `snd(true, id(42))` (inferred bool,int) and `snd[bool,int](true, 42)`
    // resolve to the SAME (gid, args) tuple, so they collapse to ONE `snd$bool$int`
    // instance — the operational proof that inference selects the explicit instance.
    var c = try checkSource("fn snd[T,U](a: T, b: U) -> U { b }\nfn id[T](x: T) -> T { x }\nfn main() -> int {\n x := snd(true, id(42))\n y := snd[bool,int](true, 42)\n return x + y\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // id$int + one shared snd$bool$int (NOT two).
    try testing.expectEqual(@as(usize, 2), c.result.instances.len);
    var snd_count: usize = 0;
    for (c.result.instances) |inst| {
        if (std.mem.startsWith(u8, inst.name, "main.snd$")) snd_count += 1;
    }
    try testing.expectEqual(@as(usize, 1), snd_count);
}

test "M3: a diverging (never) argument still infers the var from another concrete arg" {
    const gpa = testing.allocator;
    // same[T](a:T,b:T): never at 0 binds nothing; int at 1 binds T=int. Check-only
    // (a break-less loop cannot run to an exit code).
    var c = try checkSource("fn same[T](a: T, b: T) -> T { b }\nfn main() -> int { return same(loop {}, 42) }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqualStrings("main.same$int", c.result.instances[0].name);
}

test "M3: a conflicting bare call reports T0015 naming BOTH argument spans; mints no instance" {
    const gpa = testing.allocator;
    var c = try checkSource("fn same[T](a: T, b: T) -> T { a }\nfn main() -> int { return same(1, true) }\n");
    defer c.deinit(gpa);
    var saw: ?DiagnosticSink.Diagnostic = null;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0015) saw = d;
    }
    try testing.expect(saw != null);
    // Both spans named: the related (earlier) span is set (not the NO_RELATED sentinel).
    try testing.expect(saw.?.related != DiagnosticSink.NO_RELATED);
    // A conflicting call is rejected, so it mints no instance.
    try testing.expectEqual(@as(usize, 0), c.result.instances.len);
}

test "M3: a return-only generic call reports T0016 (explicit args required)" {
    const gpa = testing.allocator;
    var c = try checkSource("fn ro[T]() -> T { loop {} }\nfn main() -> int {\n ro()\n return 0\n}\n");
    defer c.deinit(gpa);
    var saw = false;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0016) saw = true;
    }
    try testing.expect(saw);
    try testing.expectEqual(@as(usize, 0), c.result.instances.len);
}

test "M3: explicit type args still override inference and satisfy an otherwise-uninferable call" {
    const gpa = testing.allocator;
    // The return-only `ro[T]` is uninferable bare, but explicit `ro[int]()` compiles.
    var c = try checkSource("fn ro[T]() -> T { loop {} }\nfn main() -> int { return ro[int]() }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqualStrings("main.ro$int", c.result.instances[0].name);
}

test "M2: an uncalled generic fn mints ZERO instances (free in the binary)" {
    const gpa = testing.allocator;
    var c = try checkSource("fn id[T](x: T) -> T { x }\nfn main() -> int { return 0 }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.instances.len);
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
